"""SHAP 分析引擎：为所有模型计算 SHAP、生成排名表与图片，保存到 exp/shap/。

主要输出（均在 outputs/exp_{id}/shap/）：
- {model_name}_shap_summary.csv        每个特征在该模型下的 mean |SHAP|
- shap_ranking_all_models.csv          跨所有模型聚合的特征重要性排名（仅真实特征）
- shap_ranking_features.csv            shap_ranking_all_models.csv 的别名
- shap_ranking_ensemble_contributions.csv  集成模型基模型贡献排名
- shap_feature_model_matrix.csv        每个模型每个真实特征的详细排名
- shap_feature_model_matrix_full.csv   全部模型×全部特征的完整矩阵（含来源标记）
- shap_model_sources.csv               每个 summary 对应的模型来源（Base/Ensemble/Transformer/AutoML）
- figure_shap_summary_{model}.png      单模型 SHAP summary/beeswarm
- figure_shap_bar_{model}.png          单模型 SHAP bar
- figure_top_{n}_shap_ranking.png      Top-N 模型聚合 SHAP 百分比图
- figure_shap_distribution_{feature}.png  指定指标的 SHAP 值分布（可选）
"""

import json
import multiprocessing as mp
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.interpretability import compute_shap_values
from core.model_trainer import compute_classification_metrics


def _load_shap_model(model_path: Path, source: str, name: str):
    """加载模型供 SHAP 使用。

    支持：
    - .pkl 文件（Base / Ensemble / Transformer / FLAML）
    - H2O 原生模型文件（automl/H2O_model/ 下的单文件）
    - AutoGluon 原生模型目录（automl/AutoGluon_model/）
    """
    if source == "AutoML" and model_path.is_dir():
        if name == "H2O":
            return _load_h2o_model(model_path)
        if name == "AutoGluon":
            return _load_autogluon_model(model_path)
        raise ValueError(f"不支持的 AutoML 目录类型：{model_path}")

    with open(model_path, "rb") as f:
        return pickle.load(f)


def _load_h2o_model(model_dir: Path):
    """从 H2O_model/ 目录加载 H2O 模型并包装为 predict_proba 接口。

    注意：H2O StackedEnsemble 每次 predict_proba 调用都涉及 H2OFrame 转换，
    与 shap.KernelExplainer 配合时会极慢（单次调用可达数十秒），因此目前跳过。
    """
    import h2o
    from core.automl import H2OAutoMLWrapper

    h2o.no_progress()
    h2o.init()
    model_files = [p for p in model_dir.iterdir() if p.is_file()]
    if not model_files:
        raise FileNotFoundError(f"H2O 模型目录为空：{model_dir}")
    # H2O save_model 生成的是单个模型文件
    model_file = model_files[0]
    h2o_model = h2o.load_model(str(model_file))

    # StackedEnsemble 与 KernelExplainer 组合极慢，直接跳过并给出明确原因
    model_id = getattr(h2o_model, "model_id", "")
    if "StackedEnsemble" in str(model_id):
        raise RuntimeError(
            "H2O StackedEnsemble 的 SHAP 计算在当前 KernelExplainer 实现下性能不可接受 "
            "（单次 predict 调用约 20-30 秒），已跳过。建议改用 FLAML/AutoGluon 做 SHAP 解释。"
        )

    wrapper = H2OAutoMLWrapper()
    wrapper.model_ = h2o_model
    return wrapper


def _load_autogluon_model(model_dir: Path):
    """从 AutoGluon_model/ 目录加载 AutoGluon predictor 并包装为 predict_proba 接口。"""
    from autogluon.tabular import TabularPredictor
    from core.automl import AutoGluonWrapper

    predictor = TabularPredictor.load(str(model_dir))
    wrapper = AutoGluonWrapper(output_dir=str(model_dir))
    wrapper.model_ = predictor
    return wrapper


# --------------------------------------------------------------------------- #
# H2O 专用快速 SHAP
# --------------------------------------------------------------------------- #

def _h2o_as_data_frame(h2o_frame) -> pd.DataFrame:
    """H2OFrame -> pandas；若装了 polars+pyarrow 则走多线程转换，更快。"""
    try:
        return h2o_frame.as_data_frame(use_multi_thread=True)
    except Exception:
        return h2o_frame.as_data_frame()


def _h2o_contributions_shap(model, X: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """用 H2O 原生 predict_contributions 计算 SHAP 值（Java 端计算，快）。

    - 树型模型（GBM/DRF/XGBoost）：TreeSHAP，直接 predict_contributions(frame)。
    - GLM：需传背景框架，自动用样本子集作为 background_frame 重试。
    - StackedEnsemble：不支持，抛错（由调用方回退排列重要性）。

    Returns:
        (shap_values: n_samples x n_features, X_sample)
    """
    import h2o
    h2o_model = model.model_ if hasattr(model, "model_") else model
    h2o_df = h2o.H2OFrame(X)
    try:
        contrib = h2o_model.predict_contributions(h2o_df)
    except Exception as e:
        # GLM 等需要背景框架
        if "background frame" in str(e) or "background_frame" in str(e):
            bg = X.sample(min(50, len(X)), random_state=42)
            h2o_bg = h2o.H2OFrame(bg)
            contrib = h2o_model.predict_contributions(h2o_df, background_frame=h2o_bg)
        else:
            raise
    contrib_df = _h2o_as_data_frame(contrib)
    feature_cols = [c for c in contrib_df.columns if c != "BiasTerm"]
    if not feature_cols:
        raise ValueError("predict_contributions 返回为空")
    shap_values = contrib_df[feature_cols].values
    X_sample = X[feature_cols].copy()
    return shap_values, X_sample


def _h2o_permutation_shap(model, X: pd.DataFrame, y, n_repeats: int = 5) -> pd.DataFrame:
    """用排列重要性（整表预测、次数有界）近似 H2O 特征贡献。

    用于 StackedEnsemble 等不支持 predict_contributions 的 H2O 模型。
    返回 DataFrame[feature, mean_abs_shap, rank]（mean_abs_shap 列为 AUC 下降幅度）。
    """
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    base_auc = roc_auc_score(y, model.predict_proba(X))
    records = []
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = np.random.RandomState(42).permutation(Xp[col].values)
            drops.append(base_auc - roc_auc_score(y, model.predict_proba(Xp)))
        records.append({"feature": col, "mean_abs_shap": float(np.mean(drops))})
    df = pd.DataFrame(records).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def _autogluon_feature_importance(model, X: pd.DataFrame, y,
                                  sample_size: int = 200, random_state: int = 42):
    """用 AutoGluon 原生 feature_importance（置换重要性）替代逐点 SHAP。

    AutoGluon 的 predict 是整条集成流水线（特征工程 + 多层 stacking），
    逐点 SHAP 需调用数千次、小时级；feature_importance 整表置换一次完成，
    快几十倍。返回 (abs_shap 对齐 X.columns 的 ndarray, X_sample)。
    """
    predictor = model.model_ if hasattr(model, "model_") else model
    # 置换重要性按整表计算，比逐点 SHAP 便宜，可多给样本（上限 5000 行）
    n = min(len(X), max(sample_size * 10, 1000), 5000)
    X_sample = X.sample(n, random_state=random_state) if len(X) > n else X
    y_arr = np.asarray(y)
    data = X_sample.copy()
    data["target"] = y_arr[X_sample.index]
    fi = predictor.feature_importance(data, silent=True)
    imp = fi["importance"] if "importance" in fi.columns else fi.iloc[:, 0]
    abs_shap = imp.reindex(X.columns).fillna(0.0).values.astype(float)
    return abs_shap, X_sample


def _permutation_importance_shap(predict_fn, X: pd.DataFrame, y,
                                 n_repeats: int = 5, random_state: int = 42) -> pd.DataFrame:
    """通用置换重要性：逐特征打乱，度量 AUC 下降幅度。

    用于 PFN 家族（TabPFN/TabICL）等每次预测都是全量推理的模型——
    逐点 SHAP 需数千次全量推理（小时级），置换重要性整表有界（分钟级）。
    predict_fn 接收 DataFrame，返回一维正类概率。
    """
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    base_auc = roc_auc_score(y, predict_fn(X))
    records = []
    for col in X.columns:
        drops = []
        for r in range(n_repeats):
            Xp = X.copy()
            Xp[col] = np.random.RandomState(random_state + r).permutation(Xp[col].values)
            drops.append(base_auc - roc_auc_score(y, predict_fn(Xp)))
        records.append({"feature": col, "mean_abs_shap": float(np.mean(drops))})
    df = pd.DataFrame(records).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def _build_ensemble_meta_features(model_path: Path, config: dict) -> Optional[pd.DataFrame]:
    """为集成模型构造元特征输入（各基模型的 OOF 预测概率）。"""
    modeling = config.get("modeling") or {}
    model_dir = Path(modeling.get("output_dir", "outputs/exp_1/models"))
    if not model_dir.exists():
        return None
    oof_files = sorted(model_dir.glob("*_oof.csv"))
    if len(oof_files) < 2:
        return None

    oof_data = {}
    y_true = None
    for f in oof_files:
        model_name = f.stem.replace("_oof", "")
        df = pd.read_csv(f)
        oof_data[model_name] = df["oof_proba"].values
        if y_true is None:
            y_true = df["y_true"].values

    meta = pd.DataFrame(oof_data)
    return meta


def _model_auc_from_oof(model_path: Path, model_dir: Path) -> float:
    """从对应 _oof.csv 读取 AUC，用于排序模型。"""
    name = model_path.stem
    oof_path = model_dir / f"{name}_oof.csv"
    if not oof_path.exists():
        return 0.0
    df = pd.read_csv(oof_path)
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(df["y_true"].values, df["oof_proba"].values)
    except Exception:
        return 0.0


def collect_model_files(exp_dir: Path, config: dict) -> List[Path]:
    """收集基础/集成/Transformer/AutoML 各模块保存的 .pkl 模型文件路径。"""
    files = []
    dirs = []

    modeling = config.get("modeling") or {}
    if modeling.get("output_dir"):
        dirs.append(Path(modeling["output_dir"]))

    ensemble = config.get("ensemble") or {}
    if ensemble.get("result_path"):
        dirs.append(Path(ensemble["result_path"]).parent)

    transformer = config.get("transformer") or {}
    if transformer.get("result_path"):
        dirs.append(Path(transformer["result_path"]).parent)

    automl = config.get("automl") or {}
    if automl.get("result_path"):
        dirs.append(Path(automl["result_path"]).parent)

    for d in dirs:
        if d.exists():
            files.extend(sorted(d.glob("*.pkl")))
    return files


# 模型来源与目录的映射顺序（子目录在前，优先匹配更具体的来源）
_SOURCE_DIR_ORDER: List[Tuple[str, str]] = [
    ("AutoML", "automl"),
    ("Transformer", "transformer"),
    ("Ensemble", "ensemble"),
    ("Base", "models"),
]


def _automl_native_name(path: Path) -> Optional[str]:
    """判断路径是否为 AutoML 原生模型目录，并返回模型名（H2O/AutoGluon）。"""
    if not path.is_dir():
        return None
    stem = path.name
    if stem == "H2O_model" or stem.endswith("_H2O_model"):
        return "H2O"
    if stem == "AutoGluon_model" or stem.endswith("_AutoGluon_model"):
        return "AutoGluon"
    return None


def collect_model_files_with_source(exp_dir: Path, config: dict) -> List[Dict[str, any]]:
    """收集所有可加载模型，并标记其来源模块。

    返回：
        [{"path": Path, "source": "Base"|"Ensemble"|"Transformer"|"AutoML", "name": str}, ...]

    说明：
    - 普通模型以 .pkl 文件形式存在；
    - H2O / AutoGluon 模型以其原生目录形式存在（H2O_model/、AutoGluon_model/）。
    """
    entries = []
    seen_paths = set()

    dirs = []
    modeling = config.get("modeling") or {}
    if modeling.get("output_dir"):
        dirs.append(("Base", Path(modeling["output_dir"])))

    ensemble = config.get("ensemble") or {}
    if ensemble.get("result_path"):
        dirs.append(("Ensemble", Path(ensemble["result_path"]).parent))

    transformer = config.get("transformer") or {}
    if transformer.get("result_path"):
        dirs.append(("Transformer", Path(transformer["result_path"]).parent))

    automl = config.get("automl") or {}
    if automl.get("result_path"):
        dirs.append(("AutoML", Path(automl["result_path"]).parent))

    # 按照更具体的子目录优先的顺序收集，避免同一个文件被重复标记
    for source, d in dirs:
        if not d.exists():
            continue
        # .pkl 模型
        for p in sorted(d.glob("*.pkl")):
            rp = p.resolve()
            if rp in seen_paths:
                continue
            seen_paths.add(rp)
            entries.append({"path": p, "source": source, "name": p.stem})

        # AutoML 原生模型目录（H2O / AutoGluon）
        if source == "AutoML":
            for p in sorted(d.iterdir()):
                if not p.is_dir():
                    continue
                native_name = _automl_native_name(p)
                if native_name is None:
                    continue
                rp = p.resolve()
                if rp in seen_paths:
                    continue
                seen_paths.add(rp)
                entries.append({"path": p, "source": source, "name": native_name})

    # 兜底：按目录名自动识别 exp_dir 下的模型
    if not entries and exp_dir.exists():
        for source, dir_name in _SOURCE_DIR_ORDER:
            d = exp_dir / dir_name
            if not d.exists():
                continue
            for p in sorted(d.glob("*.pkl")):
                rp = p.resolve()
                if rp in seen_paths:
                    continue
                seen_paths.add(rp)
                entries.append({"path": p, "source": source, "name": p.stem})
            if source == "AutoML":
                for p in sorted(d.iterdir()):
                    if not p.is_dir():
                        continue
                    native_name = _automl_native_name(p)
                    if native_name is None:
                        continue
                    rp = p.resolve()
                    if rp in seen_paths:
                        continue
                    seen_paths.add(rp)
                    entries.append({"path": p, "source": source, "name": native_name})

    return entries


def _save_model_sources(shap_dir: Path, sources: Dict[str, str]):
    """保存模型名称到来源模块的映射。"""
    records = [{"model": m, "source": s} for m, s in sorted(sources.items())]
    pd.DataFrame(records).to_csv(shap_dir / "shap_model_sources.csv", index=False)


def _load_model_sources(shap_dir: Path) -> Dict[str, str]:
    """读取模型名称到来源模块的映射。"""
    path = shap_dir / "shap_model_sources.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["model"].astype(str), df["source"].astype(str)))


def compute_single_model_shap(
    model_path: Path,
    X: pd.DataFrame,
    output_dir: Path,
    sample_size: int = 200,
    compute_figures: bool = True,
    config: dict = None,
    source: str = "Base",
    name: str = None,
    h2o_mode: str = "native",
    y=None,
) -> Optional[Dict]:
    """计算单个模型的 SHAP，保存 summary CSV 与图片。返回摘要字典或 None（失败/跳过）。

    h2o_mode（仅对 H2O 模型生效）：
        - "off"：跳过
        - "native"：H2O 原生 predict_contributions（TreeSHAP，快；StackedEnsemble 回退排列重要性）
        - "permutation"：排列重要性（快，近似）
        - "kernel"：原始 KernelExplainer（极慢）
    """
    model_name = name or model_path.stem
    try:
        model = _load_shap_model(model_path, source, model_name)

        # 判断模型类型并准备输入
        if source == "Ensemble":
            X_input = _build_ensemble_meta_features(model_path, config or {})
            if X_input is None:
                raise ValueError("无法构造集成模型的元特征输入（缺少基模型 OOF 预测）")
        else:
            X_input = X

        is_h2o = source == "AutoML" and model_name == "H2O"
        shap_values = None
        if is_h2o:
            if h2o_mode == "off":
                return None
            X_sample = X_input.sample(min(sample_size, len(X_input)), random_state=42) if len(X_input) > sample_size else X_input
            if h2o_mode == "kernel":
                shap_values, X_sample = compute_shap_values(model, X_input, sample_size=sample_size)
                abs_shap = np.abs(shap_values).mean(axis=0)
            elif h2o_mode == "permutation":
                if y is None:
                    raise ValueError("排列重要性需要 y")
                perm = _h2o_permutation_shap(model, X_sample, np.asarray(y)[X_sample.index] if hasattr(X_sample, "index") and len(np.asarray(y)) == len(X_input) else y)
                abs_shap = perm.set_index("feature")["mean_abs_shap"].reindex(X_sample.columns).values
                shap_values = None
            else:  # native
                try:
                    shap_values, X_sample = _h2o_contributions_shap(model, X_sample)
                    abs_shap = np.abs(shap_values).mean(axis=0)
                except Exception as e:
                    warnings.warn(f"H2O predict_contributions 失败（{e}），回退排列重要性")
                    if y is None:
                        raise ValueError("StackedEnsemble 不支持原生 SHAP，且排列重要性需要 y")
                    perm = _h2o_permutation_shap(model, X_sample, np.asarray(y)[X_sample.index] if hasattr(X_sample, "index") and len(np.asarray(y)) == len(X_input) else y)
                    abs_shap = perm.set_index("feature")["mean_abs_shap"].reindex(X_sample.columns).values
                    shap_values = None
        else:
            # AutoGluon：predict 是整条 stacking 流水线，逐点 SHAP 需数千次调用、
            # 小时级；改用 AutoGluon 原生 feature_importance（整表置换，快几十倍）
            is_autogluon = source == "AutoML" and model_name == "AutoGluon"
            # PFN 家族（TabPFN/TabICL）：每次预测都是全量上下文推理，
            # 逐点 SHAP 小时级；降级为整表置换重要性（分钟级）
            is_pfn = type(model).__name__ in ("TabPFNv2Wrapper", "TabICLWrapper")
            if is_pfn:
                if y is None:
                    raise ValueError("PFN 模型的置换重要性需要 y")
                n_imp = min(len(X_input), 2000)
                X_sample = (X_input.sample(n_imp, random_state=42)
                            if len(X_input) > n_imp else X_input)
                from core.interpretability import _ModelProbaWrapper
                wrapper = _ModelProbaWrapper(model, list(X_input.columns))
                perm = _permutation_importance_shap(
                    wrapper.predict, X_sample, np.asarray(y)[X_sample.index]
                )
                abs_shap = perm.set_index("feature")["mean_abs_shap"] \
                    .reindex(X_sample.columns).values
                shap_values = None
            elif is_autogluon:
                if y is None:
                    raise ValueError("AutoGluon 原生特征重要性需要 y")
                abs_shap, X_sample = _autogluon_feature_importance(
                    model, X_input, y, sample_size=sample_size
                )
                shap_values = None
            else:
                shap_values, X_sample = compute_shap_values(model, X_input, sample_size=sample_size)
                abs_shap = np.abs(shap_values).mean(axis=0)

        summary = pd.DataFrame({
            "feature": X_sample.columns,
            "mean_abs_shap": abs_shap,
        })
        summary["rank"] = summary["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
        summary = summary.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / f"{model_name}_shap_summary.csv"
        summary.to_csv(summary_path, index=False)

        if compute_figures:
            from core.interpretability import plot_shap_summary, plot_shap_bar
            if shap_values is not None:
                try:
                    plot_shap_summary(shap_values, X_sample,
                                      output_path=str(output_dir / f"figure_shap_summary_{model_name}.png"))
                    plt.close()
                except Exception as e:
                    warnings.warn(f"SHAP summary 图失败 [{model_name}]: {e}")
                try:
                    plot_shap_bar(shap_values, X_sample,
                                  output_path=str(output_dir / f"figure_shap_bar_{model_name}.png"))
                    plt.close()
                except Exception as e:
                    warnings.warn(f"SHAP bar 图失败 [{model_name}]: {e}")
            else:
                # permutation 模式：无 SHAP 矩阵，画重要性条形图
                try:
                    plot_df = summary.sort_values("mean_abs_shap")
                    plt.figure(figsize=(7, max(3, len(plot_df) * 0.4)))
                    plt.barh(plot_df["feature"], plot_df["mean_abs_shap"], color="#2ca02c")
                    plt.xlabel("Importance (AUC drop)")
                    plt.title(f"{model_name}: 特征重要性（排列重要性）")
                    plt.tight_layout()
                    plt.savefig(str(output_dir / f"figure_shap_bar_{model_name}.png"), dpi=300, bbox_inches="tight")
                    plt.close()
                except Exception as e:
                    warnings.warn(f"H2O 重要性条形图失败 [{model_name}]: {e}")

        return {
            "model": model_name,
            "source": source,
            "model_path": str(model_path),
            "summary_path": str(summary_path),
            "n_samples": len(X_sample),
            "n_features": X_sample.shape[1],
        }
    except Exception as e:
        warnings.warn(f"SHAP 计算失败 [{model_name}]: {e}")
        return None


def _model_auc(model_name: str, search_dirs: List[Path]) -> float:
    """在多个可能的结果目录中搜索该模型的 AUC。"""
    for d in search_dirs:
        for suffix in ["_oof", "_val"]:
            oof = d / f"{model_name}{suffix}.csv"
            if not oof.exists():
                continue
            try:
                from sklearn.metrics import roc_auc_score
                df_oof = pd.read_csv(oof)
                proba_col = "oof_proba" if "oof" in suffix else "val_proba"
                return roc_auc_score(df_oof["y_true"].values, df_oof[proba_col].values)
            except Exception:
                pass
    return 0.0


def _get_top_n_models(shap_dir: Path, model_dir: Path, config: dict = None, n: int = 3) -> List[str]:
    """按 AUC 排序返回 Top-N 模型名。"""
    summary_files = sorted(shap_dir.glob("*_shap_summary.csv"))
    if not summary_files:
        return []

    auc_search_dirs = [model_dir]
    if config:
        for key in ["ensemble", "transformer", "automl"]:
            cfg = config.get(key) or {}
            rp = cfg.get("result_path")
            if rp:
                auc_search_dirs.append(Path(rp).parent)

    all_models = []
    for sf in summary_files:
        model_name = sf.stem.replace("_shap_summary", "")
        auc = _model_auc(model_name, auc_search_dirs)
        all_models.append({"model": model_name, "auc": auc})

    all_models_df = pd.DataFrame(all_models).sort_values("auc", ascending=False).reset_index(drop=True)
    return all_models_df.head(n)["model"].tolist()


def aggregate_shap_ranking(
    shap_dir: Path,
    model_dir: Path,
    output_dir: Path,
    top_n: int = 3,
    indicator: Optional[str] = None,
    config: dict = None,
    feature_columns: List[str] = None,
) -> Dict[str, str]:
    """读取所有单模型 SHAP summary，生成聚合排名表与 Top-N 图。

    说明：
    - 真实特征（来自 feature_columns）生成 shap_ranking_all_models.csv / shap_ranking_features.csv
    - Ensemble 模型的“特征”是基模型名，单独生成 shap_ranking_ensemble_contributions.csv
    """
    summary_files = sorted(shap_dir.glob("*_shap_summary.csv"))
    if not summary_files:
        raise ValueError("没有可用的单模型 SHAP summary")

    feature_columns = list(feature_columns or [])
    sources = _load_model_sources(shap_dir)

    # 收集可能存放 *_oof.csv / *_val.csv 的所有目录
    auc_search_dirs = [model_dir]
    if config:
        for key in ["ensemble", "transformer", "automl"]:
            cfg = config.get(key) or {}
            rp = cfg.get("result_path")
            if rp:
                auc_search_dirs.append(Path(rp).parent)

    all_models = []
    feature_totals = {}
    feature_model_count = {}
    ensemble_contrib_totals = {}
    ensemble_contrib_count = {}
    rank_records = []
    ensemble_records = []
    full_records = []

    for sf in summary_files:
        model_name = sf.stem.replace("_shap_summary", "")
        source = sources.get(model_name, "Base")
        df = pd.read_csv(sf)
        auc = _model_auc(model_name, auc_search_dirs)
        all_models.append({"model": model_name, "source": source, "auc": auc})

        is_ensemble = (source == "Ensemble")
        # 兜底：没有来源映射或旧数据时，根据特征是否多数为真实特征判断
        if not is_ensemble and feature_columns:
            n_real = df["feature"].isin(feature_columns).sum()
            if n_real < len(df) * 0.5:
                is_ensemble = True

        for _, row in df.iterrows():
            feat = row["feature"]
            mean_abs = float(row["mean_abs_shap"])
            rank_in_model = int(row["rank"])

            # 完整矩阵：记录所有模型×所有特征
            full_records.append({
                "model": model_name,
                "source": source,
                "feature": feat,
                "mean_abs_shap": mean_abs,
                "rank_in_model": rank_in_model,
                "auc": auc,
            })

            if is_ensemble:
                # 集成模型的“特征”是基模型名
                ensemble_contrib_totals[feat] = ensemble_contrib_totals.get(feat, 0.0) + mean_abs
                ensemble_contrib_count[feat] = ensemble_contrib_count.get(feat, 0) + 1
                ensemble_records.append({
                    "model": model_name,
                    "base_model": feat,
                    "mean_abs_shap": mean_abs,
                    "rank_in_model": rank_in_model,
                    "auc": auc,
                })
            else:
                # 非集成模型只聚合真实特征；如果提供了 feature_columns 则做强制过滤
                if feature_columns and feat not in feature_columns:
                    continue
                feature_totals[feat] = feature_totals.get(feat, 0.0) + mean_abs
                feature_model_count[feat] = feature_model_count.get(feat, 0) + 1
                rank_records.append({
                    "model": model_name,
                    "feature": feat,
                    "mean_abs_shap": mean_abs,
                    "rank_in_model": rank_in_model,
                    "auc": auc,
                })

    # 真实特征聚合排名表
    if feature_totals:
        ranking = pd.DataFrame({
            "feature": list(feature_totals.keys()),
            "total_mean_abs_shap": list(feature_totals.values()),
            "n_models": [feature_model_count[f] for f in feature_totals.keys()],
        })
        ranking["overall_rank"] = ranking["total_mean_abs_shap"].rank(ascending=False, method="min").astype(int)

        rank_df = pd.DataFrame(rank_records)
        mean_rank = rank_df.groupby("feature")["rank_in_model"].mean().reset_index()
        mean_rank.rename(columns={"rank_in_model": "mean_rank_across_models"}, inplace=True)
        ranking = ranking.merge(mean_rank, on="feature", how="left")
        ranking = ranking.sort_values("overall_rank").reset_index(drop=True)
    else:
        ranking = pd.DataFrame(columns=["feature", "total_mean_abs_shap", "n_models", "overall_rank", "mean_rank_across_models"])

    ranking_path = output_dir / "shap_ranking_all_models.csv"
    ranking.to_csv(ranking_path, index=False)
    # 保留旧命名作为兼容别名
    ranking.to_csv(output_dir / "shap_ranking_features.csv", index=False)

    # 集成模型基模型贡献排名表
    ensemble_ranking_path = None
    if ensemble_contrib_totals:
        ensemble_ranking = pd.DataFrame({
            "base_model": list(ensemble_contrib_totals.keys()),
            "total_mean_abs_shap": list(ensemble_contrib_totals.values()),
            "n_ensemble_models": [ensemble_contrib_count[f] for f in ensemble_contrib_totals.keys()],
        })
        ensemble_ranking["overall_rank"] = ensemble_ranking["total_mean_abs_shap"].rank(ascending=False, method="min").astype(int)
        ensemble_records_df = pd.DataFrame(ensemble_records)
        mean_ens_rank = ensemble_records_df.groupby("base_model")["rank_in_model"].mean().reset_index()
        mean_ens_rank.rename(columns={"rank_in_model": "mean_rank_across_ensembles"}, inplace=True)
        ensemble_ranking = ensemble_ranking.merge(mean_ens_rank, on="base_model", how="left")
        ensemble_ranking = ensemble_ranking.sort_values("overall_rank").reset_index(drop=True)
        ensemble_ranking_path = output_dir / "shap_ranking_ensemble_contributions.csv"
        ensemble_ranking.to_csv(ensemble_ranking_path, index=False)

    # 保存每个模型每个真实特征的详细 rank 表
    rank_df = pd.DataFrame(rank_records)
    rank_df_path = output_dir / "shap_feature_model_matrix.csv"
    if not rank_df.empty:
        rank_df.to_csv(rank_df_path, index=False)

    # 保存完整矩阵
    full_df = pd.DataFrame(full_records)
    full_df_path = output_dir / "shap_feature_model_matrix_full.csv"
    if not full_df.empty:
        full_df.to_csv(full_df_path, index=False)

    paths = {
        "ranking": str(ranking_path),
        "feature_matrix": str(rank_df_path) if not rank_df.empty else None,
        "feature_matrix_full": str(full_df_path) if not full_df.empty else None,
    }
    if ensemble_ranking_path:
        paths["ensemble_contributions"] = str(ensemble_ranking_path)

    # 特征 SHAP 聚合排名图
    try:
        paths["feature_ranking_plot"] = plot_shap_feature_ranking(
            ranking_path, output_dir / "figure_shap_feature_ranking.png"
        )
    except Exception as e:
        warnings.warn(f"特征 SHAP 聚合排名图失败：{e}")

    # Top-N 聚合 SHAP 百分比图（单一 top_n，保持兼容）
    top_models = _get_top_n_models(shap_dir, model_dir, config=config, n=top_n)
    if top_models:
        paths["top_n_shap_pct"] = _plot_top_n_shap_pct(
            shap_dir, top_models, output_dir / f"figure_top_{top_n}_shap_ranking.png",
            feature_columns=feature_columns,
        )

    # 指定指标分析（仅针对真实特征）
    if indicator and indicator in feature_columns:
        paths["indicator_distribution"] = analyze_feature_shap(
            shap_dir, model_dir, indicator, output_dir, feature_columns=feature_columns
        )

    return paths


def _plot_top_n_shap_pct(shap_dir: Path, top_models: List[str], output_path: Path,
                          feature_columns: List[str] = None) -> str:
    """绘制 Top-N 模型聚合 SHAP 百分比水平条形图（仅使用真实特征）。"""
    feature_columns = feature_columns or []
    agg = {}
    for model in top_models:
        sf = shap_dir / f"{model}_shap_summary.csv"
        if not sf.exists():
            continue
        df = pd.read_csv(sf)
        for _, row in df.iterrows():
            feat = row["feature"]
            if feature_columns and feat not in feature_columns:
                continue
            agg[feat] = agg.get(feat, 0.0) + row["mean_abs_shap"]

    if not agg:
        raise ValueError("Top-N 模型无 SHAP 数据")

    df = pd.DataFrame({"feature": list(agg.keys()), "mean_abs_shap": list(agg.values())})
    df["pct"] = df["mean_abs_shap"] / df["mean_abs_shap"].sum() * 100
    df = df.sort_values("pct", ascending=True).reset_index(drop=True)

    n = len(df)
    deep = np.array([13, 59, 102]) / 255.0
    light = np.array([144, 202, 249]) / 255.0
    colors = [deep * (1 - i / max(n - 1, 1)) + light * (i / max(n - 1, 1)) for i in range(n)]

    fig, ax = plt.subplots(figsize=(8, max(5, n * 0.42)))
    y = np.arange(n)
    bars = ax.barh(y, df["pct"].values, color=colors, edgecolor="#333333", height=0.55, linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"].values, fontsize=10)
    ax.set_xlim([0, max(35, df["pct"].max() * 1.2)])
    ax.set_xlabel("Percentage of total SHAP importance (%)", fontsize=12)
    ax.set_title(f"Aggregated SHAP Importance across Top {len(top_models)} Models", fontweight="bold", fontsize=14, pad=12)
    ax.set_axisbelow(True)
    ax.grid(True, axis="x", linestyle=":", alpha=0.3, color="#888888")

    for bar, pct in zip(bars, df["pct"].values):
        ax.text(pct + 0.4, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%",
                va="center", ha="left", fontsize=10, fontweight="bold", color="#333333")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(output_path)


def plot_shap_feature_ranking(
    ranking_path: Path,
    output_path: Path,
    top_n: Optional[int] = None,
) -> str:
    """绘制特征 SHAP 聚合排名图。

    以 overall_rank 为纵轴排序，展示：
    - 水平条：total_mean_abs_shap
    - 条上标注：mean_rank_across_models
    - 颜色：按 mean_rank_across_models 渐变（越小越重要，颜色越深）
    """
    ranking = pd.read_csv(ranking_path)
    if ranking.empty:
        raise ValueError("聚合排名表为空")

    required = {"feature", "total_mean_abs_shap", "overall_rank", "mean_rank_across_models"}
    missing = required - set(ranking.columns)
    if missing:
        raise ValueError(f"聚合排名表缺少列：{missing}")

    # 按 overall_rank 升序排列（排名越靠前越重要）
    ranking = ranking.sort_values("overall_rank").reset_index(drop=True)

    if top_n:
        ranking = ranking.head(top_n)

    n = len(ranking)
    # 颜色：mean_rank_across_models 越小（排名越靠前），颜色越深
    ranks = ranking["mean_rank_across_models"].values
    rank_min, rank_max = ranks.min(), ranks.max()
    norm = (ranks - rank_min) / (rank_max - rank_min + 1e-9)
    deep = np.array([13, 59, 102]) / 255.0
    light = np.array([144, 202, 249]) / 255.0
    colors = [deep * (1 - r) + light * r for r in norm]

    fig, ax = plt.subplots(figsize=(9, max(5, n * 0.55)))
    y = np.arange(n)
    bars = ax.barh(
        y, ranking["total_mean_abs_shap"].values,
        color=colors, edgecolor="#333333", height=0.55, linewidth=0.6
    )

    ax.set_yticks(y)
    ax.set_yticklabels(ranking["feature"].values, fontsize=10)
    ax.invert_yaxis()  # overall_rank 1 在顶部
    ax.set_xlabel("Total mean |SHAP| across models", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(
        f"Aggregated SHAP Feature Ranking (n={n})",
        fontweight="bold", fontsize=14, pad=12
    )
    ax.set_axisbelow(True)
    ax.grid(True, axis="x", linestyle=":", alpha=0.3, color="#888888")

    # 标注 mean_rank_across_models 与 overall_rank
    for i, (bar, mean_rank, overall_rank, total) in enumerate(
        zip(bars, ranking["mean_rank_across_models"], ranking["overall_rank"], ranking["total_mean_abs_shap"])
    ):
        label = f"R{int(overall_rank)}  μR={mean_rank:.1f}"
        ax.text(
            total + total * 0.01,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center", ha="left", fontsize=9, fontweight="bold", color="#333333"
        )

    # 添加颜色条图例
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("rank_cmap", [deep, light])
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=rank_min, vmax=rank_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.02, shrink=0.5)
    cbar.set_label("Mean rank across models", fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(output_path)


def analyze_feature_shap(
    shap_dir: Path,
    model_dir: Path,
    feature: str,
    output_dir: Path,
    feature_columns: List[str] = None,
) -> str:
    """对指定真实特征做跨模型 SHAP 分析，返回分布图路径。

    纳入所有成功计算 SHAP 的模型：
    - 真实特征模型（Base/Transformer/AutoML）直接读取该特征的 SHAP。
    - 集成模型以基模型为输入，因此不存在该真实特征的 SHAP，会被自然排除。
    结果 CSV 中通过 source 列区分模型来源，保证模型总数与成功生成 summary 的模型一致。
    """
    feature_columns = list(feature_columns or [])
    sources = _load_model_sources(shap_dir)
    summary_files = sorted(shap_dir.glob("*_shap_summary.csv"))
    records = []
    for sf in summary_files:
        model_name = sf.stem.replace("_shap_summary", "")
        source = sources.get(model_name, "Base")
        df = pd.read_csv(sf)
        row = df[df["feature"] == feature]
        if row.empty:
            continue
        records.append({
            "model": model_name,
            "source": source,
            "mean_abs_shap": float(row["mean_abs_shap"].iloc[0]),
            "rank_in_model": int(row["rank"].iloc[0]),
        })

    if not records:
        raise ValueError(f"指标 {feature} 在 SHAP summary 中不存在")

    analysis_df = pd.DataFrame(records).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    analysis_df["overall_rank"] = analysis_df["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
    analysis_df["rank_in_source"] = analysis_df.groupby("source")["mean_abs_shap"].rank(ascending=False, method="min").astype(int)

    analysis_path = output_dir / f"shap_analysis_{feature}.csv"
    analysis_df.to_csv(analysis_path, index=False)

    # 分布图
    fig, ax = plt.subplots(figsize=(10, 6))
    values = analysis_df["mean_abs_shap"].values
    ax.hist(values, bins=max(10, min(30, len(values))), color="#377eb8", edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
    median_v = np.median(values)
    ax.axvline(median_v, color="orange", linestyle="-", linewidth=2, label=f"Median = {median_v:.4f}")
    ax.set_xlabel(f"Mean |SHAP| for {feature}", fontsize=12)
    ax.set_ylabel("Number of models", fontsize=12)
    ax.set_title(f"SHAP Distribution: {feature} ({len(analysis_df)} models)", fontweight="bold", fontsize=14)
    ax.legend()
    ax.grid(True, axis="y", linestyle=":", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig_path = output_dir / f"figure_shap_distribution_{feature}.png"
    fig.savefig(fig_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return str(fig_path)


# ---------------------------------------------------------------------------
# SHAP 模型级并行：worker 上下文与入口（顶层函数保证可 pickle，
# 数据/配置经 initializer 一次性传入，避免逐模型序列化）
# ---------------------------------------------------------------------------

_SHAP_CTX: Dict[str, Any] = {}


def _shap_init_context(X_values, columns, config, y_values, sample_size, h2o_mode, shap_dir):
    _SHAP_CTX.update(X=X_values, cols=columns, config=config, y=y_values,
                     sample_size=sample_size, h2o_mode=h2o_mode, shap_dir=shap_dir)


def _shap_worker(entry_info):
    """在子进程中计算单个模型的 SHAP，返回 (entry_info, 摘要dict或None, 失败原因或None)。

    compute_single_model_shap 内部会吞掉异常并返回 None，这里用 catch_warnings
    把失败原因带回主进程，避免 worker 里的警告丢失。
    """
    import warnings as _w

    import pandas as pd

    X = pd.DataFrame(_SHAP_CTX["X"], columns=_SHAP_CTX["cols"])
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        res = compute_single_model_shap(
            Path(entry_info["path"]), X, Path(_SHAP_CTX["shap_dir"]),
            sample_size=_SHAP_CTX["sample_size"], compute_figures=True,
            config=_SHAP_CTX["config"], source=entry_info["source"],
            name=entry_info["name"], h2o_mode=_SHAP_CTX["h2o_mode"],
            y=_SHAP_CTX["y"],
        )
    err = None
    if res is None and caught:
        err = "; ".join(str(x.message) for x in caught)[:500]
    return entry_info, res, err


def run_full_shap_analysis(
    model_dir: Path,
    X: pd.DataFrame,
    exp_dir: Path,
    sample_size: int = 200,
    top_n: int = 3,
    top_n_start: int = None,
    top_n_end: int = None,
    indicator: Optional[str] = None,
    progress_callback=None,
    config: dict = None,
    h2o_enabled: bool = True,
    h2o_mode: str = "native",
    y=None,
    max_parallel: int = 1,
    skip_existing: bool = True,
) -> Dict[str, any]:
    """入口函数：为所有模块（Base/Ensemble/Transformer/AutoML）中可加载的模型计算 SHAP。

    注意：Ensemble 的输入为基础模型预测概率，因此其 SHAP 值解释的是各基模型贡献；
    Transformer/AutoML 的输入为原始特征，解释的是原始特征贡献。

    top_n：单一值时仅生成该值的 Top-N 图（向后兼容）。
    top_n_start / top_n_end：同时提供时生成从 start 到 end 每一档的 Top-N 图。
    """
    shap_dir = exp_dir / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    if config:
        model_entries = collect_model_files_with_source(exp_dir, config)
    else:
        model_entries = [{"path": p, "source": "Base", "name": p.stem}
                         for p in sorted(model_dir.glob("*.pkl"))]

    if not model_entries:
        raise ValueError("未找到任何模型文件")

    results = []
    failures = []
    sources = {}
    runnable = []
    for entry in model_entries:
        source = entry["source"]
        model_name = entry.get("name") or entry["path"].stem
        sources[model_name] = source
        # H2O 未启用则直接跳过
        if source == "AutoML" and model_name == "H2O" and not h2o_enabled:
            continue
        # 增量：已有 summary 的模型直接复用，不重算
        if skip_existing:
            summary_path = shap_dir / f"{model_name}_shap_summary.csv"
            if summary_path.exists():
                results.append({
                    "model": model_name, "source": source,
                    "model_path": str(entry["path"]),
                    "summary_path": str(summary_path),
                })
                continue
        runnable.append({"path": str(entry["path"]), "source": source, "name": model_name})

    n_skipped = len(results)
    if n_skipped and progress_callback:
        progress_callback(0, len(runnable), f"增量：跳过已有 SHAP 结果的 {n_skipped} 个模型")

    def _run_serial(todo):
        for i, entry in enumerate(todo):
            if progress_callback:
                progress_callback(i + 1, len(todo), entry["name"])
            # Transformer 等深度学习模型允许失败跳过
            res = compute_single_model_shap(
                Path(entry["path"]), X, shap_dir, sample_size=sample_size,
                config=config, source=entry["source"], name=entry["name"],
                h2o_mode=h2o_mode, y=y,
            )
            (results if res else failures).append(res if res else entry["name"])

    if max_parallel > 1 and len(runnable) > 1:
        done_names = set()
        try:
            mp.freeze_support()
            with ProcessPoolExecutor(
                max_workers=max_parallel,
                initializer=_shap_init_context,
                initargs=(X.values, list(X.columns), config,
                          None if y is None else np.asarray(y),
                          sample_size, h2o_mode, str(shap_dir)),
            ) as executor:
                futures = {executor.submit(_shap_worker, e): e for e in runnable}
                for i, fut in enumerate(as_completed(futures)):
                    entry = futures[fut]
                    if progress_callback:
                        progress_callback(i + 1, len(runnable), entry["name"])
                    try:
                        _, res, err = fut.result()
                    except Exception as e:
                        # worker 进程级崩溃（如 BrokenProcessPool）：该模型记失败，
                        # 不回退串行，避免同样的崩溃发生在主进程（GUI）里
                        res, err = None, f"worker 进程异常：{e}"
                    done_names.add(entry["name"])
                    if res:
                        results.append(res)
                    else:
                        failures.append(entry["name"])
                        warnings.warn(f"SHAP 计算失败 [{entry['name']}]: {err or '未知原因'}")
        except Exception as e:
            # 进程池本身故障：串行补跑未完成的模型
            warnings.warn(f"SHAP 模型级并行不可用（{e}），回退串行")
            _run_serial([e for e in runnable if e["name"] not in done_names])
    else:
        _run_serial(runnable)

    if not results:
        raise ValueError("没有成功计算任何模型的 SHAP 值")

    # 保存模型来源映射，供后续聚合与分析使用
    _save_model_sources(shap_dir, sources)

    # 确定 Top-N 范围
    if top_n_start is None or top_n_end is None:
        top_n_start = top_n_end = top_n
    top_n_start = max(1, top_n_start)
    top_n_end = max(top_n_start, top_n_end)

    meta = {
        "n_models_total": len(model_entries),
        "n_models_success": len(results),
        "n_models_failed": len(failures),
        "failures": failures,
        "sample_size": sample_size,
        "top_n": top_n,
        "top_n_start": top_n_start,
        "top_n_end": top_n_end,
        "indicator": indicator,
        "models": [r["model"] for r in results],
        "sources": {r["model"]: r["source"] for r in results},
    }
    with open(shap_dir / "shap_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 生成聚合排名表（以 top_n_end 作为默认 Top-N 图）
    agg_paths = aggregate_shap_ranking(
        shap_dir, model_dir, shap_dir, top_n=top_n_end, indicator=indicator,
        config=config, feature_columns=list(X.columns)
    )

    # 若指定范围，则补充生成每一档的 Top-N 图
    top_n_plot_paths = []
    if top_n_end > top_n_start:
        for n in range(top_n_start, top_n_end):
            top_models = _get_top_n_models(shap_dir, model_dir, config=config, n=n)
            if top_models:
                plot_path = _plot_top_n_shap_pct(
                    shap_dir, top_models,
                    shap_dir / f"figure_top_{n}_shap_ranking.png",
                    feature_columns=list(X.columns),
                )
                top_n_plot_paths.append(plot_path)
    if top_n_plot_paths:
        agg_paths.setdefault("top_n_shap_pct", [])
        if isinstance(agg_paths["top_n_shap_pct"], str):
            agg_paths["top_n_shap_pct"] = [agg_paths["top_n_shap_pct"]]
        agg_paths["top_n_shap_pct"].extend(top_n_plot_paths)

    return {
        "shap_dir": str(shap_dir),
        "meta": meta,
        "single_model_results": results,
        "aggregate_paths": agg_paths,
    }
