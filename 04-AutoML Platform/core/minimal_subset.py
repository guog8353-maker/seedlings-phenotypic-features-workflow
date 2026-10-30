"""最小协变量子集分析：SHAP 引导的向后剔除。

对每个选中模型，按 SHAP 贡献从低到高逐步删除协变量，直到性能出现
统计学显著下降（DeLong p < 阈值），取前一个子集作为该模型的最小子集。
随后在最小子集上重算 SHAP，得到子集内贡献度排名。

集成模型不支持（其输入为基模型 OOF，协变量级剔除需重训整条基模型链）。
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from core.ablation import (
    delong_auc_test,
    load_base_xy,
    _load_base_predictions,
)


# --------------------------------------------------------------------------- #
# SHAP 排名来源
# --------------------------------------------------------------------------- #

def _shap_dir(exp: dict) -> Path:
    output_dir = Path(exp.get("output_dir") or "outputs")
    return output_dir / f"exp_{exp['id']}" / "shap"


def get_shap_ranking(exp: dict, name: str, source: str, X_cols: List[str]) -> Tuple[Optional[List[str]], str]:
    """获取模型的协变量贡献排名（按贡献从高到低）。

    优先使用该模型自己的 SHAP 排名；缺失时回退到多模型平均排名；
    再缺失返回 (None, "fallback")，由调用方用模型自带 importance 或原顺序。

    Returns:
        (ordered_features 或 None, ranking_source)
    """
    shap_dir = _shap_dir(exp)

    # 1. 该模型自己的 SHAP summary
    f = shap_dir / f"{name}_shap_summary.csv"
    if f.exists():
        try:
            df = pd.read_csv(f)
            if "feature" in df.columns and "mean_abs_shap" in df.columns:
                feats = df.sort_values("mean_abs_shap", ascending=False)["feature"].astype(str).tolist()
                feats = [x for x in feats if x in X_cols]
                # 覆盖率过半即可使用（完整覆盖优先）；避免 SHAP 在超集特征上
                # 计算时静默回退到平均排名
                if len(feats) == len(X_cols):
                    return feats, "model_shap"
                if len(feats) >= max(1, int(len(X_cols) * 0.5)):
                    # 覆盖率过半：SHAP 缺失的特征视为低优先级，补到排名末尾
                    feats = feats + [c for c in X_cols if c not in feats]
                    return feats, "model_shap_partial"
        except Exception:
            pass

    # 2. 多模型平均排名
    f2 = shap_dir / "shap_ranking_all_models.csv"
    if f2.exists():
        try:
            df = pd.read_csv(f2)
            if "feature" in df.columns:
                rank_col = "overall_rank" if "overall_rank" in df.columns else df.columns[1]
                feats = df.sort_values(rank_col)["feature"].astype(str).tolist()
                feats = [x for x in feats if x in X_cols]
                feats += [c for c in X_cols if c not in feats]
                if len(feats) == len(X_cols):
                    return feats, "avg_shap"
        except Exception:
            pass

    return None, "fallback"


def _fallback_ranking(model, X: pd.DataFrame, y: pd.Series, source: str) -> Tuple[List[str], str]:
    """回退排名：基础树模型用 feature_importances_，否则按原顺序。"""
    if source == "Base" and model is not None and hasattr(model, "feature_importances_"):
        try:
            imp = np.asarray(model.feature_importances_)
            order = np.argsort(-imp)
            return [X.columns[i] for i in order], "model_importance"
        except Exception:
            pass
    return list(X.columns), "original_order"


# --------------------------------------------------------------------------- #
# 按原协议重训（返回拟合好的模型，供子集 SHAP 重算）
# --------------------------------------------------------------------------- #

def _model_dir(exp: dict) -> Path:
    config = exp.get("config") or {}
    return Path((config.get("modeling") or {}).get("output_dir", ""))


def _retrain(exp: dict, name: str, source: str, X: pd.DataFrame, y: pd.Series,
             cv_folds: int) -> Tuple[object, np.ndarray, np.ndarray]:
    """按原协议重训模型。

    Returns:
        (fitted_model, y_eval, proba)
        - Base: fitted=全数据拟合模型, y_eval=全样本, proba=OOF
        - Transformer/AutoML: fitted=训练集拟合模型, y_eval=留出集, proba=留出预测
    """
    if source == "Base":
        pkl_path = _model_dir(exp) / f"{name}.pkl"
        if not pkl_path.exists():
            raise ValueError(f"模型文件不存在：{pkl_path}")
        with open(pkl_path, "rb") as f:
            saved = pickle.load(f)
        model = clone(saved)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        try:
            proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        except Exception:
            proba = cross_val_predict(model, X, y, cv=cv, n_jobs=1).astype(float)
        fitted = model.fit(X, y)
        return fitted, y.values, proba

    if source == "Transformer":
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
        trainer = TransformerTrainer(name, params=params)
        trainer.fit(X_train, y_train)
        proba = trainer.predict_proba(X_val)
        return trainer, y_val.values, proba

    # AutoML
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
    output_dir = (Path(exp.get("output_dir") or "outputs") / f"exp_{exp['id']}" / "minimal_subset"
                  / f"tmp_{name}_{datetime.now():%Y%m%d_%H%M%S_%f}")
    output_dir.mkdir(parents=True, exist_ok=True)
    params = {"time_limit": time_limit, "use_gpu": use_gpu}
    if name == "AutoGluon":
        params["output_dir"] = str(output_dir)
    runner = AutoMLRunner(name, params=params)
    runner.fit(X_train, y_train)
    proba = runner.predict_proba(X_val)
    return runner, y_val.values, proba


# --------------------------------------------------------------------------- #
# 子集 SHAP / 重要性重算
# --------------------------------------------------------------------------- #

def compute_permutation_ranking(model, X: pd.DataFrame, y: np.ndarray,
                                 n_repeats: int = 5, random_state: int = 42) -> pd.DataFrame:
    """用排列重要性（整表预测、次数有界）计算特征贡献排名。

    用于 H2O 等 KernelExplainer-SHAP 极慢（每次 predict 需 H2OFrame 转换）的模型。
    返回与 compute_subset_shap 相同格式（mean_abs_shap 列存 AUC 下降幅度）。
    """
    rng = np.random.RandomState(random_state)
    y = np.asarray(y)
    base_auc = roc_auc_score(y, model.predict_proba(X))
    records = []
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            drops.append(base_auc - roc_auc_score(y, model.predict_proba(Xp)))
        records.append({"feature": col, "mean_abs_shap": float(np.mean(drops))})
    df = pd.DataFrame(records).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def compute_subset_shap(fitted_model, X_subset: pd.DataFrame, y_eval: np.ndarray = None,
                        sample_size: int = 200, name: str = "", source: str = "") -> pd.DataFrame:
    """在最小子集上重算贡献度排名。

    - H2O：KernelExplainer-SHAP 极慢（每次 predict 需 H2OFrame 转换，数千次调用），
      改用排列重要性（整表预测、次数有界）。
    - 其余模型：用 SHAP（TreeExplainer 快，sklearn 非树模型 KernelExplainer 可接受）。
    """
    if source == "AutoML" and name == "H2O":
        if y_eval is None:
            raise ValueError("H2O 排列重要性需要 y_eval")
        return compute_permutation_ranking(fitted_model, X_subset, y_eval)

    from core.interpretability import compute_shap_values
    shap_values, X_sample = compute_shap_values(fitted_model, X_subset, sample_size=sample_size)
    abs_shap = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": abs_shap})
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


# --------------------------------------------------------------------------- #
# 向后剔除主流程（单模型）
# --------------------------------------------------------------------------- #

def backward_eliminate(
    exp: dict,
    name: str,
    source: str,
    X_base: pd.DataFrame,
    y: pd.Series,
    p_thresh: float = 0.05,
    cv_folds: int = 5,
    resample_shap: bool = True,
    shap_sample_size: int = 200,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict:
    """对单个模型执行 SHAP 引导的向后剔除。

    Returns:
        dict: {
            name, source, auc_full, ordered_features, ranking_source,
            trajectory: DataFrame[k, n_features, features, auc, delong_p],
            subset_features: List[str], auc_subset, delta_auc, delong_p_final,
            strictly_better: bool, subset_shap: DataFrame or None,
        }
    """
    X_cols = list(X_base.columns)
    n = len(X_cols)
    if n < 2:
        raise ValueError("特征数不足，无法做向后剔除")

    # 全特征基线预测（已保存文件）
    y_full, full_proba = _load_base_predictions(exp, name, source)
    if full_proba is None:
        raise ValueError("缺少全特征基线预测文件，无法配对比较")
    auc_full = roc_auc_score(y_full, full_proba)

    # SHAP 排名（决定剔除顺序）
    ordered, rank_src = get_shap_ranking(exp, name, source, X_cols)

    def _log(msg):
        if progress_callback:
            progress_callback(msg)

    # 轨迹记录
    traj_records = []
    subset_features = X_cols
    auc_subset = auc_full
    delong_p_final = 1.0
    minimal_found = False

    for k in range(n, 0, -1):
        if ordered is None:
            # 回退：先拟合一次全特征模型取 importance
            fitted0, _, _ = _retrain(exp, name, source, X_base, y, cv_folds)
            ordered, rank_src = _fallback_ranking(fitted0, X_base, y, source)

        feats_k = ordered[:k]

        if k == n:
            # 全特征：直接用已保存的基线预测，避免重复训练
            _log(f"{name}：k={k}/{n}（全特征，使用基线）")
            auc_k = auc_full
            p = 1.0
        else:
            X_k = X_base[feats_k]
            _log(f"{name}：k={k}/{n}（{len(feats_k)} 特征）")
            fitted, y_eval, proba = _retrain(exp, name, source, X_k, y, cv_folds)
            y_eval_cmp = y_full
            if len(proba) != len(full_proba):
                raise ValueError(f"预测样本数与基线不一致({len(proba)} vs {len(full_proba)})")
            auc_k = roc_auc_score(y_eval_cmp, proba)
            _, _, p = delong_auc_test(y_eval_cmp, full_proba, proba)

        traj_records.append({
            "k": k, "n_features": k, "features": "|".join(feats_k),
            "auc": round(float(auc_k), 4), "delong_p": p,
        })

        if p is None or (isinstance(p, float) and np.isnan(p)):
            # DeLong 检验失败（无法比较）：保守起见停止剔除，保留当前子集
            _log(f"{name}：k={k} 时 DeLong 检验失败，停止剔除并保留当前子集")
            minimal_found = True
            break
        if p < p_thresh:
            # 显著变差：最小子集为前一个（k+1）
            minimal_found = True
            break
        else:
            subset_features = feats_k
            auc_subset = auc_k
            delong_p_final = p

    trajectory = pd.DataFrame(traj_records)

    # 若全程不显著，最小子集为 1 个特征
    if not minimal_found and len(traj_records) > 0:
        last = traj_records[-1]
        subset_features = last["features"].split("|")
        auc_subset = last["auc"]
        delong_p_final = last["delong_p"]

    # 在最小子集上重新拟合 + 重算贡献度排名
    subset_shap = None
    if resample_shap and subset_features:
        try:
            X_sub = X_base[subset_features]
            method = "排列重要性" if (source == "AutoML" and name == "H2O") else "SHAP"
            _log(f"{name}：在最小子集({len(subset_features)} 特征)上重算 {method}")
            fitted_sub, y_eval_sub, _ = _retrain(exp, name, source, X_sub, y, cv_folds)
            subset_shap = compute_subset_shap(
                fitted_sub, X_sub, y_eval=y_eval_sub,
                sample_size=shap_sample_size, name=name, source=source,
            )
        except Exception as e:
            warnings.warn(f"{name} 子集贡献度重算失败：{e}")

    delta_auc = float(auc_subset - auc_full)
    return {
        "name": name,
        "source": source,
        "auc_full": round(float(auc_full), 4),
        "ordered_features": ordered,
        "ranking_source": rank_src,
        "trajectory": trajectory,
        "subset_features": subset_features,
        "n_subset": len(subset_features),
        "auc_subset": round(float(auc_subset), 4),
        "delta_auc": round(delta_auc, 4),
        "delong_p_final": delong_p_final,
        "strictly_better": delta_auc > 0,
        "subset_shap": subset_shap,
    }


def run_minimal_subset(
    exp: dict,
    models: List[Dict[str, str]],
    p_thresh: float = 0.05,
    cv_folds: int = 5,
    resample_shap: bool = True,
    shap_sample_size: int = 200,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[Dict]:
    """对多个模型执行最小子集分析。

    Returns:
        list of backward_eliminate 的返回 dict（含失败记录）
    """
    X_base, y, _ = load_base_xy(exp)
    results = []
    total = len(models)
    for i, m in enumerate(models, 1):
        name, source = m["name"], m["source"]

        def _cb(msg, _i=i, _t=total, _n=name):
            if progress_callback:
                progress_callback(_i, _t, msg)

        try:
            res = backward_eliminate(
                exp, name, source, X_base, y,
                p_thresh=p_thresh, cv_folds=cv_folds,
                resample_shap=resample_shap, shap_sample_size=shap_sample_size,
                progress_callback=_cb,
            )
            res["status"] = "ok"
            res["error"] = ""
        except Exception as e:
            warnings.warn(f"最小子集分析失败 [{name}]：{e}")
            res = {
                "name": name, "source": source, "status": "failed", "error": str(e),
                "trajectory": pd.DataFrame(), "subset_features": [], "n_subset": 0,
                "auc_full": np.nan, "auc_subset": np.nan, "delta_auc": np.nan,
                "delong_p_final": np.nan, "strictly_better": False, "subset_shap": None,
                "ordered_features": None, "ranking_source": "",
            }
        results.append(res)
        if progress_callback:
            progress_callback(i, total, f"{name} 完成")
    return results


def make_summary_table(results: List[Dict]) -> pd.DataFrame:
    """生成全部模型的对比汇总表。"""
    records = []
    for r in results:
        records.append({
            "model": r["name"],
            "source": r["source"],
            "auc_full": r.get("auc_full"),
            "n_full": len(r.get("ordered_features") or []),
            "n_subset": r.get("n_subset", 0),
            "auc_subset": r.get("auc_subset"),
            "delta_auc": r.get("delta_auc"),
            "delong_p": r.get("delong_p_final"),
            "strictly_better": r.get("strictly_better", False),
            "ranking_source": r.get("ranking_source", ""),
            "status": r.get("status", ""),
            "error": r.get("error", ""),
        })
    df = pd.DataFrame(records)
    if "auc_full" in df.columns:
        df = df.sort_values("auc_full", ascending=False).reset_index(drop=True)
    return df
