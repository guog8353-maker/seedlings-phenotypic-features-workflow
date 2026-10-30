"""消融分析核心：单变量回加消融（add-back ablation）。

用途：检验被丢弃（drop 角色）或未被共识选中的变量中，是否存在能提升
当前最优模型性能的被忽略协变量。

设计要点：
- Base 数据为「共识特征数据」（最小特征集）。
- 每个候选变量单独加回 base 特征，重训选中模型并比较 ΔAUC。
- 评估协议沿用各模型原协议：基础模型 5-fold OOF；Transformer/AutoML 用
  相同 random_state 的 80/20 留出。
- 显著性：DeLong 配对检验 + Benjamini-Hochberg FDR 校正。
- 集成模型不支持消融（其输入为基模型 OOF，需重训全部基模型才有意义），
  在 GUI 中以禁用形式呈现。
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder

from core.base_models import build_model


# --------------------------------------------------------------------------- #
# DeLong 检验：比较同一数据上两条 ROC 曲线的 AUC
# 参考: DeLong et al. (1988), 使用 Sun & Xu 的快速算法
# --------------------------------------------------------------------------- #

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x, kind="mergesort")
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        # midrank = (i + j - 1) / 2 + 1  (1-based)
        T[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted_transposed: np.ndarray, label_1_count: int):
    m = label_1_count
    n = preds_sorted_transposed.shape[1] - m
    k = preds_sorted_transposed.shape[0]
    if m == 0 or n == 0:
        raise ValueError("DeLong 需要同时存在正负样本")

    pos = preds_sorted_transposed[:, :m]
    neg = preds_sorted_transposed[:, m:]
    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(preds_sorted_transposed[r, :])

    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2.0) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_auc_test(y_true: np.ndarray, proba1: np.ndarray, proba2: np.ndarray) -> Tuple[float, float, float]:
    """DeLong 配对检验：比较 proba1 与 proba2 的 AUC 差异。

    Returns:
        (auc1, auc2, p_value)
    """
    y_true = np.asarray(y_true).astype(int)
    proba1 = np.asarray(proba1, dtype=float)
    proba2 = np.asarray(proba2, dtype=float)
    # 正样本排在前面
    order = np.argsort(-y_true, kind="mergesort")
    label_1_count = int(y_true.sum())
    preds = np.vstack([proba1, proba2])[:, order]
    try:
        aucs, delongcov = _fast_delong(preds, label_1_count)
    except Exception:
        return roc_auc_score(y_true, proba1), roc_auc_score(y_true, proba2), np.nan

    var = delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]
    if var <= 0 or not np.isfinite(var):
        # 退化情形：差值方差为 0。AUC 相同 -> p=1；AUC 恒定不同 -> p≈0
        if abs(aucs[0] - aucs[1]) < 1e-12:
            return float(aucs[0]), float(aucs[1]), 1.0
        return float(aucs[0]), float(aucs[1]), 0.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(p)


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR 校正，返回 q 值。"""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    qvals = np.full(n, np.nan)
    valid = np.isfinite(pvals)
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return qvals
    p = pvals[idx]
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(len(p))
    out[order] = q
    qvals[idx] = out
    return qvals


# --------------------------------------------------------------------------- #
# 候选变量收集与预处理
# --------------------------------------------------------------------------- #

NON_CONSENSUS = "非共识"
DROP = "drop"


def collect_candidate_variables(exp: dict) -> pd.DataFrame:
    """收集候选消融变量：drop 角色变量 + 共识表中 consensus=0 的变量。

    Returns:
        DataFrame[variable, source, in_preprocessed]
    """
    config = exp.get("config") or {}
    roles = config.get("roles", {})
    fs_config = config.get("feature_selection") or {}

    records = []
    seen = set()

    # 共识=0 的变量（在预处理 CSV 中）
    consensus_path = fs_config.get("consensus_table_path")
    if consensus_path and Path(consensus_path).exists():
        try:
            table = pd.read_csv(consensus_path)
            if "feature" in table.columns and "consensus" in table.columns:
                for feat in table.loc[table["consensus"] == 0, "feature"].astype(str):
                    if feat not in seen:
                        records.append({"variable": feat, "source": NON_CONSENSUS, "in_preprocessed": True})
                        seen.add(feat)
        except Exception:
            pass

    # drop 角色变量（只在原始数据中）
    for col, role in roles.items():
        if role == "drop" and col not in seen:
            records.append({"variable": col, "source": DROP, "in_preprocessed": False})
            seen.add(col)

    return pd.DataFrame(records, columns=["variable", "source", "in_preprocessed"])


def load_preprocessed_df(exp: dict) -> pd.DataFrame:
    config = exp.get("config") or {}
    prep = config.get("preprocessing") or {}
    path = prep.get("processed_path")
    if not path or not Path(path).exists():
        raise ValueError("未找到预处理数据，请先完成预处理")
    return pd.read_csv(path)


def load_base_xy(exp: dict) -> Tuple[pd.DataFrame, pd.Series, str]:
    """加载 base（共识特征）数据。"""
    config = exp.get("config") or {}
    roles = config.get("roles", {})
    fs_config = config.get("feature_selection") or {}

    df = load_preprocessed_df(exp)
    y_col = [c for c, r in roles.items() if r == "outcome"]
    if not y_col:
        raise ValueError("未设置 outcome 角色")
    y_col = y_col[0]

    if not fs_config.get("consensus_table_path"):
        raise ValueError("未找到共识特征表，请先完成特征选择")
    table = pd.read_csv(fs_config["consensus_table_path"])
    feature_cols = table.loc[table["consensus"] == 1, "feature"].astype(str).tolist()
    if not feature_cols:
        raise ValueError("共识特征为空")
    X = df[feature_cols]
    y = df[y_col]
    return X, y, y_col


def _infer_var_type(s: pd.Series) -> str:
    """推断 drop 变量类型：continuous / categorical。"""
    if pd.api.types.is_numeric_dtype(s):
        return "continuous" if s.nunique(dropna=True) > 10 else "categorical"
    return "categorical"


def _preprocess_drop_column(s: pd.Series, scaling: str) -> pd.Series:
    """按原管线策略预处理单个 drop 变量。"""
    vtype = _infer_var_type(s)
    out = s.copy()
    if vtype == "continuous":
        out = pd.to_numeric(out, errors="coerce")
        if out.notna().sum() == 0:
            raise ValueError("全为缺失或非数值")
        out = out.fillna(out.median())
        # 与原管线一致的缩放
        if scaling == "standard":
            mu, sd = out.mean(), out.std()
            if sd > 0:
                out = (out - mu) / sd
        elif scaling == "minmax":
            rng = out.max() - out.min()
            if rng > 0:
                out = (out - out.min()) / rng
        elif scaling == "robust":
            q1, q3 = out.quantile(0.25), out.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                out = (out - out.median()) / iqr
        # scaling == "none": 不处理
        return out.astype(float)
    else:
        out = out.astype(str).replace({"nan": np.nan, "None": np.nan})
        mode = out.mode(dropna=True)
        fill = mode.iloc[0] if len(mode) > 0 else "missing"
        out = out.fillna(fill)
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        vals = enc.fit_transform(out.to_frame()).ravel()
        return pd.Series(vals, index=s.index, dtype=float)


def prepare_candidates(exp: dict, variables: List[str], cache: bool = True) -> pd.DataFrame:
    """准备候选变量矩阵（与 base 数据按行对齐）。

    - 非共识变量：直接从预处理 CSV 取列；
    - drop 变量：从原始数据读取并按原策略预处理，结果缓存到
      outputs/exp_{id}/ablation/candidates_preprocessed.csv。
    """
    config = exp.get("config") or {}
    prep_config = config.get("preprocessing") or {}
    scaling = prep_config.get("scaling", "standard")

    cand_df_meta = collect_candidate_variables(exp)
    meta_map = dict(zip(cand_df_meta["variable"], cand_df_meta["source"]))

    result = pd.DataFrame()
    prep_df = None

    # 非共识：直接从预处理 CSV 取
    non_consensus_vars = [v for v in variables if meta_map.get(v) == NON_CONSENSUS]
    if non_consensus_vars:
        prep_df = load_preprocessed_df(exp)
        for v in non_consensus_vars:
            if v not in prep_df.columns:
                warnings.warn(f"非共识变量 {v} 不在预处理数据中，已跳过")
                continue
            result[v] = pd.to_numeric(prep_df[v], errors="coerce")

    # drop 变量：原始数据 + 预处理（带缓存）
    drop_vars = [v for v in variables if meta_map.get(v) == DROP]
    if drop_vars:
        output_dir = Path(exp.get("output_dir") or "outputs") / f"exp_{exp['id']}" / "ablation"
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_path = output_dir / "candidates_preprocessed.csv"

        cached = None
        if cache and cache_path.exists():
            try:
                cached = pd.read_csv(cache_path)
            except Exception:
                cached = None

        if cached is not None and all(v in cached.columns for v in drop_vars):
            for v in drop_vars:
                result[v] = cached[v].values
        else:
            data_path = exp.get("data_path")
            if not data_path or not Path(data_path).exists():
                raise ValueError(f"原始数据不存在：{data_path}")
            raw = pd.read_csv(data_path) if str(data_path).lower().endswith(".csv") else None
            if raw is None:
                from core.data_manager import DataManager
                raw = DataManager().load(data_path)

            processed = {}
            for v in drop_vars:
                if v not in raw.columns:
                    warnings.warn(f"drop 变量 {v} 不在原始数据中，已跳过")
                    continue
                try:
                    processed[v] = _preprocess_drop_column(raw[v], scaling)
                except Exception as e:
                    warnings.warn(f"drop 变量 {v} 预处理失败：{e}，已跳过")

            if processed:
                new_cache = pd.DataFrame(processed)
                if cache:
                    # 与已有缓存按列合并，避免换变量组合时反复重算/互相覆盖
                    try:
                        old_cache = pd.read_csv(cache_path)
                        for c in old_cache.columns:
                            if c not in new_cache.columns:
                                new_cache[c] = old_cache[c]
                    except Exception:
                        pass
                    new_cache.to_csv(cache_path, index=False)
                for v, s in processed.items():
                    result[v] = s.values

    if result.empty:
        raise ValueError("没有可用的候选变量（可能全部被跳过）")
    return result


# --------------------------------------------------------------------------- #
# 模型重训（各模块原协议）
# --------------------------------------------------------------------------- #

def _get_model_dir(exp: dict) -> Path:
    config = exp.get("config") or {}
    modeling = config.get("modeling") or {}
    return Path(modeling.get("output_dir", ""))


def _load_base_predictions(exp: dict, name: str, source: str):
    """加载模型的基线预测（用于 AUC_base 与 DeLong 配对）。

    Returns:
        (y_true, base_proba) or (None, None)
    """
    config = exp.get("config") or {}
    if source == "Base":
        model_dir = _get_model_dir(exp)
        f = model_dir / f"{name}_oof.csv"
    elif source == "Transformer":
        tr = config.get("transformer") or {}
        f = Path(tr.get("result_path", "")).parent / f"{name}_val.csv"
    else:  # AutoML
        au = config.get("automl") or {}
        f = Path(au.get("result_path", "")).parent / f"{name}_val.csv"

    if not f.exists():
        return None, None
    df = pd.read_csv(f)
    proba_col = "oof_proba" if "oof_proba" in df.columns else "val_proba"
    if proba_col not in df.columns or "y_true" not in df.columns:
        return None, None
    return df["y_true"].values, df[proba_col].values


def _retrain_base_model(name: str, X: pd.DataFrame, y: pd.Series, exp: dict,
                        cv_folds: int, use_optuna: bool, n_trials: int) -> np.ndarray:
    """基础模型：5-fold OOF（同 random_state）。"""
    model_dir = _get_model_dir(exp)
    pkl_path = model_dir / f"{name}.pkl"
    if not pkl_path.exists():
        raise ValueError(f"模型文件不存在：{pkl_path}")
    with open(pkl_path, "rb") as f:
        saved_model = pickle.load(f)

    if use_optuna:
        from core.optimizer import OptunaOptimizer
        optimizer = OptunaOptimizer(X, y, name, cv_folds=cv_folds, random_state=42, n_jobs=1)
        best = optimizer.optimize(n_trials=n_trials)
        model = build_model(name, best, random_state=42)
    else:
        model = clone(saved_model)  # 沿用原最优超参数

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    from sklearn.model_selection import cross_val_predict
    try:
        oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    except Exception:
        oof_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=1)
        oof = oof_pred.astype(float)
    return oof


def _retrain_transformer(name: str, X: pd.DataFrame, y: pd.Series, exp: dict,
                         use_optuna: bool, n_trials: int) -> Tuple[np.ndarray, np.ndarray]:
    """Transformer：相同 80/20 留出。"""
    config = exp.get("config") or {}
    tr = config.get("transformer") or {}
    arch_path = Path(tr.get("result_path", "")).parent / f"{name}_architecture.json"
    if not arch_path.exists():
        raise ValueError(f"架构文件不存在：{arch_path}")
    with open(arch_path, "r", encoding="utf-8") as f:
        params = json.load(f).get("params", {})

    from core.transformers import TransformerTrainer
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    if use_optuna:
        from core.transformer_optimizer import TransformerOptimizer
        optimizer = TransformerOptimizer(name, X_train, y_train, random_state=42)
        best = optimizer.optimize(n_trials=n_trials)
        params = {**params, **best}

    trainer = TransformerTrainer(name, params=params)
    trainer.fit(X_train, y_train)
    proba = trainer.predict_proba(X_val)
    return y_val.values, proba


def _retrain_automl(framework: str, X: pd.DataFrame, y: pd.Series, exp: dict) -> Tuple[np.ndarray, np.ndarray]:
    """AutoML：相同 80/20 留出，相同时间预算。"""
    config = exp.get("config") or {}
    au = config.get("automl") or {}
    time_limit = int(au.get("time_limit", 300))
    use_gpu = bool(au.get("use_gpu", False))

    from core.automl import AutoMLRunner
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    # 每次重训使用带时间戳的唯一目录，避免 AutoGluon 对非空 predictor 路径告警/产物残留
    from datetime import datetime
    output_dir = (Path(exp.get("output_dir") or "outputs") / f"exp_{exp['id']}" / "ablation"
                  / f"tmp_{framework}_{datetime.now():%Y%m%d_%H%M%S_%f}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 各框架接受的参数不同：output_dir 仅 AutoGluon 支持
    params = {"time_limit": time_limit, "use_gpu": use_gpu}
    if framework == "AutoGluon":
        params["output_dir"] = str(output_dir)
    runner = AutoMLRunner(framework, params=params)
    runner.fit(X_train, y_train)
    proba = runner.predict_proba(X_val)
    return y_val.values, proba


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def run_ablation(
    exp: dict,
    models: List[Dict[str, str]],
    variables: List[str],
    hyperparam_strategy: str = "reuse",
    use_delong: bool = True,
    cv_folds: int = 5,
    n_trials: int = 20,
    pred_dir: Optional[Path] = None,
    skip_pairs: Optional[set] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """执行单变量回加消融。

    Args:
        exp: 实验字典（含 config）。
        models: [{"name": ..., "source": "Base"|"Transformer"|"AutoML"}]
        variables: 候选变量名列表。
        hyperparam_strategy: "reuse"（沿用原最优超参）或 "optuna"（重新调参）。
        use_delong: 是否做 DeLong 检验。
        cv_folds: 基础模型 CV 折数（应与原训练一致，默认 5）。
        n_trials: optuna 策略下的 trial 数。
        progress_callback: (current, total, message) 进度回调。

    Returns:
        长表 DataFrame：variable, var_source, model, source, auc_base, auc_new,
        delta_auc, delong_p, n_eval, protocol, status, error
    """
    X_base, y, _ = load_base_xy(exp)
    cand = prepare_candidates(exp, variables, cache=True)
    cand = cand[[v for v in variables if v in cand.columns]]

    if len(cand) != len(X_base):
        raise ValueError(
            f"候选变量样本数({len(cand)})与 base 数据({len(X_base)})不一致，"
            "可能是预处理时使用了 drop 缺失策略导致行数变化，请检查后重试。"
        )

    # 元信息
    cand_meta = collect_candidate_variables(exp)
    src_map = dict(zip(cand_meta["variable"], cand_meta["source"]))

    use_optuna = hyperparam_strategy == "optuna"
    skip_pairs = skip_pairs or set()
    records = []
    total = len(cand.columns) * len(models)
    done = 0

    for var in cand.columns:
        var_series = cand[var]
        if var_series.nunique(dropna=True) < 2:
            warnings.warn(f"变量 {var} 方差为 0，已跳过")
            done += len(models)
            continue
        X_new = X_base.copy()
        X_new[var] = var_series.values

        for m in models:
            done += 1
            name, source = m["name"], m["source"]
            if (var, name) in skip_pairs:
                continue  # 增量运行：跳过已成功组合
            if progress_callback:
                progress_callback(done, total, f"{var} × {name}")
            rec = {
                "variable": var, "var_source": src_map.get(var, ""),
                "model": name, "source": source,
                "auc_base": np.nan, "auc_new": np.nan, "delta_auc": np.nan,
                "delong_p": np.nan, "n_eval": len(y), "protocol": "",
                "status": "ok", "error": "",
            }
            try:
                y_base, base_proba = _load_base_predictions(exp, name, source)
                if base_proba is None:
                    raise ValueError("缺少基线预测文件，无法配对比较")

                if source == "Base":
                    new_proba = _retrain_base_model(name, X_new, y, exp, cv_folds, use_optuna, n_trials)
                    y_eval = y.values
                    rec["protocol"] = f"{cv_folds}-fold OOF"
                elif source == "Transformer":
                    y_eval, new_proba = _retrain_transformer(name, X_new, y, exp, use_optuna, n_trials)
                    rec["protocol"] = "80/20 holdout"
                else:
                    y_eval, new_proba = _retrain_automl(name, X_new, y, exp)
                    rec["protocol"] = "80/20 holdout"

                # 对齐：AutoML/Transformer 基线预测的样本数应等于留出集大小
                if len(base_proba) != len(new_proba):
                    # 基础模型 OOF 是全样本；留出协议下也可能因子采样不一致
                    if source != "Base" and len(base_proba) != len(y_eval):
                        raise ValueError(f"基线与留出样本数不一致({len(base_proba)} vs {len(y_eval)})")
                    if source == "Base":
                        raise ValueError(f"基线 OOF 与数据样本数不一致({len(base_proba)} vs {len(new_proba)})")

                if source == "Base":
                    y_eval = y_base
                auc_base = roc_auc_score(y_eval, base_proba)
                auc_new = roc_auc_score(y_eval, new_proba)
                rec["auc_base"] = round(float(auc_base), 4)
                rec["auc_new"] = round(float(auc_new), 4)
                rec["delta_auc"] = round(float(auc_new - auc_base), 4)
                rec["n_eval"] = len(y_eval)

                if use_delong:
                    _, _, p = delong_auc_test(y_eval, base_proba, new_proba)
                    rec["delong_p"] = p

                # 保存预测值，供 ROC 曲线绘制
                if pred_dir is not None:
                    pred_dir = Path(pred_dir)
                    pred_dir.mkdir(parents=True, exist_ok=True)
                    safe_var = "".join(ch if ch.isalnum() or ch in "_-." else "_" for ch in var)
                    safe_name = "".join(ch if ch.isalnum() or ch in "_-." else "_" for ch in name)
                    pd.DataFrame({
                        "y_true": y_eval,
                        "proba_base": base_proba,
                        "proba_new": new_proba,
                    }).to_csv(pred_dir / f"{source}_{safe_name}__{safe_var}.csv", index=False)
            except Exception as e:
                rec["status"] = "failed"
                rec["error"] = str(e)
                warnings.warn(f"消融失败 [{var} × {name}]：{e}")
            records.append(rec)

    if records:
        result = pd.DataFrame(records)
    else:
        # 空结果也必须带固定列，否则下游按列筛选（status/variable/model）会 KeyError
        result = pd.DataFrame(columns=[
            "variable", "var_source", "model", "source",
            "auc_base", "auc_new", "delta_auc", "delong_p",
            "n_eval", "protocol", "status", "error",
        ])
    if use_delong and len(result) > 0:
        result["q_value"] = bh_fdr(result["delong_p"].values)
    else:
        result["q_value"] = np.nan
    return result


def summarize_variables(result: pd.DataFrame, delta_thresh: float = 0.005,
                        q_thresh: float = 0.05, min_models: int = 2) -> pd.DataFrame:
    """按变量汇总消融结果，生成榜单。

    Returns:
        DataFrame: variable, var_source, mean_delta, max_delta, n_models,
        n_improved, n_significant, min_q, recommended
    """
    ok = result[result["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    def _agg(g):
        n_models = len(g)
        n_improved = int((g["delta_auc"] > 0).sum())
        n_sig = int((g["q_value"] < q_thresh).sum()) if g["q_value"].notna().any() else 0
        mean_delta = float(g["delta_auc"].mean())
        min_q = float(g["q_value"].min()) if g["q_value"].notna().any() else np.nan
        recommended = (mean_delta > delta_thresh) and (n_sig >= min_models)
        return pd.Series({
            "var_source": g["var_source"].iloc[0],
            "mean_delta": round(mean_delta, 4),
            "max_delta": round(float(g["delta_auc"].max()), 4),
            "n_models": n_models,
            "n_improved": n_improved,
            "n_significant": n_sig,
            "min_q": round(min_q, 4) if np.isfinite(min_q) else np.nan,
            "recommended": recommended,
        })

    summary = ok.groupby("variable").apply(_agg, include_groups=False).reset_index()
    summary = summary.sort_values("mean_delta", ascending=False).reset_index(drop=True)
    return summary


def make_delta_matrix(result: pd.DataFrame) -> pd.DataFrame:
    """生成「变量 × 模型」的 ΔAUC 矩阵。"""
    ok = result[result["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    ok["model_src"] = ok["source"] + ":" + ok["model"]
    mat = ok.pivot_table(index="variable", columns="model_src", values="delta_auc", aggfunc="first")
    mat = mat.loc[mat.mean(axis=1).sort_values(ascending=False).index]
    return mat
