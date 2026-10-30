"""报告可视化图片生成器：ROC、SHAP、Calibration、Confusion Matrix、Envelope、Facet、Ranking、Dot Matrix。"""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.metrics import auc, roc_curve

from core.evaluator import plot_roc_curves, plot_calibration_curve, plot_confusion_matrix
from core.interpretability import compute_shap_values, plot_shap_summary, plot_shap_bar
from core.plot_style import get_palette


@dataclass
class ModelResult:
    name: str
    oof_proba: np.ndarray
    oof_pred: np.ndarray
    y_true: np.ndarray


@dataclass
class ModelPrediction:
    """跨层统一的模型预测结果。"""
    name: str
    layer: str  # Base / Ensemble / Transformer / AutoML
    y_true: np.ndarray
    proba: np.ndarray
    auc: float
    path: Optional[str] = None


LAYER_ORDER = ["Base", "Ensemble", "AutoML", "Transformer"]


def _save_csv(df: pd.DataFrame, output_path: Path) -> str:
    """将 DataFrame 保存为 CSV，作为图片的配套原始数据。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return str(output_path)


def _layer_colors(palette_name: str = "default") -> dict:
    return get_palette(palette_name).get("layer", {})


def _method_colors(palette_name: str = "default") -> dict:
    return get_palette(palette_name).get("method", {})


def load_model_results(model_dir: Path) -> List[ModelResult]:
    """从模型输出目录加载 OOF 结果。"""
    results = []
    for f in sorted(model_dir.glob("*_oof.csv")):
        df = pd.read_csv(f)
        model_name = f.stem.replace("_oof", "")
        results.append(ModelResult(
            name=model_name,
            oof_proba=df["oof_proba"].values,
            oof_pred=df["oof_pred"].values,
            y_true=df["y_true"].values,
        ))
    return results


def _compute_auc(y_true, proba):
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, proba)
    except Exception:
        return 0.0


def collect_all_predictions(exp_dir: Path, config: Dict) -> List[ModelPrediction]:
    """收集所有层（Base/Ensemble/Transformer/AutoML）的预测结果。"""
    predictions: List[ModelPrediction] = []

    # Base models
    modeling_config = config.get("modeling") or {}
    if modeling_config.get("output_dir"):
        model_dir = Path(modeling_config["output_dir"])
        for f in sorted(model_dir.glob("*_oof.csv")):
            df = pd.read_csv(f)
            name = f.stem.replace("_oof", "")
            predictions.append(ModelPrediction(
                name=name,
                layer="Base",
                y_true=df["y_true"].values,
                proba=df["oof_proba"].values,
                auc=_compute_auc(df["y_true"].values, df["oof_proba"].values),
                path=str(f),
            ))

    # Ensemble
    ensemble_config = config.get("ensemble") or {}
    if ensemble_config.get("result_path"):
        ensemble_dir = Path(ensemble_config["result_path"]).parent
        for f in sorted(ensemble_dir.glob("*_oof.csv")):
            df = pd.read_csv(f)
            name = f.stem.replace("_oof", "")
            predictions.append(ModelPrediction(
                name=name,
                layer="Ensemble",
                y_true=df["y_true"].values,
                proba=df["oof_proba"].values,
                auc=_compute_auc(df["y_true"].values, df["oof_proba"].values),
                path=str(f),
            ))

    # Transformer
    transformer_config = config.get("transformer") or {}
    if transformer_config.get("result_path"):
        transformer_dir = Path(transformer_config["result_path"]).parent
        for f in sorted(transformer_dir.glob("*_val.csv")):
            df = pd.read_csv(f)
            name = f.stem.replace("_val", "")
            predictions.append(ModelPrediction(
                name=name,
                layer="Transformer",
                y_true=df["y_true"].values,
                proba=df["val_proba"].values,
                auc=_compute_auc(df["y_true"].values, df["val_proba"].values),
                path=str(f),
            ))

    # AutoML
    automl_config = config.get("automl") or {}
    if automl_config.get("result_path"):
        automl_dir = Path(automl_config["result_path"]).parent
        for f in sorted(automl_dir.glob("*_val.csv")):
            df = pd.read_csv(f)
            name = f.stem.replace("_val", "")
            predictions.append(ModelPrediction(
                name=name,
                layer="AutoML",
                y_true=df["y_true"].values,
                proba=df["val_proba"].values,
                auc=_compute_auc(df["y_true"].values, df["val_proba"].values),
                path=str(f),
            ))

    return sorted(predictions, key=lambda x: x.auc, reverse=True)


def _set_publication_style():
    plt.rcParams["font.family"] = "sans-serif"
    # 优先尝试支持中文的字体，再回退到 Arial / DejaVu
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS",
        "Arial", "DejaVu Sans",
    ]
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.edgecolor"] = "black"
    plt.rcParams["legend.fancybox"] = False


def _compute_roc_curve(y_true, proba):
    fpr, tpr, _ = roc_curve(y_true, proba)
    return fpr, tpr


def _interpolate_roc(fpr, tpr, n_grid=200):
    """将 ROC 曲线插值到统一网格，返回 (fpr_grid, tpr_interp)。"""
    fpr_grid = np.linspace(0, 1, n_grid)
    max_tpr = {}
    for fp, tp in zip(fpr, tpr):
        if fp not in max_tpr or tp > max_tpr[fp]:
            max_tpr[fp] = tp
    unique_fpr = sorted(max_tpr.keys())
    unique_tpr = [max_tpr[fp] for fp in unique_fpr]
    if unique_fpr[0] > 0:
        unique_fpr = [0] + unique_fpr
        unique_tpr = [0] + unique_tpr
    if unique_fpr[-1] < 1:
        unique_fpr = unique_fpr + [1]
        unique_tpr = unique_tpr + [1]
    f = interp1d(unique_fpr, unique_tpr, kind="linear", bounds_error=False, fill_value=(0, 1))
    return fpr_grid, f(fpr_grid)


def _compute_envelope(fpr_list, tpr_list, n_grid=200):
    """计算 envelope 的 median / Q25 / Q75 / min / max。"""
    fpr_grid = np.linspace(0, 1, n_grid)
    interpolated = []
    for fpr, tpr in zip(fpr_list, tpr_list):
        _, tpr_interp = _interpolate_roc(fpr, tpr, n_grid)
        interpolated.append(tpr_interp)
    tpr_stack = np.vstack(interpolated)
    return (
        fpr_grid,
        np.median(tpr_stack, axis=0),
        np.percentile(tpr_stack, 25, axis=0),
        np.percentile(tpr_stack, 75, axis=0),
        np.min(tpr_stack, axis=0),
        np.max(tpr_stack, axis=0),
    )


def plot_roc_envelope(predictions: List[ModelPrediction], output_path: str, title: str = "ROC Envelope",
                      palette: str = "default"):
    """绘制 ROC Envelope：每层用不同颜色，最佳模型实线突出，其余模型用 envelope 区间描述。

    返回：
        (png_path, csv_path) —— CSV 包含每层的包络数据（median/Q25/Q75/min/max）以及每层最佳模型的 ROC 点。
    """
    _set_publication_style()
    layer_colors = _layer_colors(palette)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot([0, 1], [0, 1], color="#999999", lw=1.2, linestyle="solid", label="Random (AUC = 0.500)")

    df = pd.DataFrame([{"layer": p.layer, "model": p.name, "auc": p.auc} for p in predictions])

    csv_records = []
    best_records = []
    for layer in LAYER_ORDER:
        layer_preds = [p for p in predictions if p.layer == layer]
        if not layer_preds:
            continue
        layer_preds = sorted(layer_preds, key=lambda x: x.auc, reverse=True)
        color = layer_colors.get(layer, "#333333")

        fpr_list, tpr_list = [], []
        for p in layer_preds:
            fpr, tpr = _compute_roc_curve(p.y_true, p.proba)
            fpr_list.append(fpr)
            tpr_list.append(tpr)

        fpr_grid, tpr_med, tpr_q25, tpr_q75, tpr_min, tpr_max = _compute_envelope(fpr_list, tpr_list)
        # 区间：min-max（最浅） + Q25-Q75（较深）
        ax.fill_between(fpr_grid, tpr_min, tpr_max, color=color, alpha=0.10)
        ax.fill_between(fpr_grid, tpr_q25, tpr_q75, color=color, alpha=0.28)
        ax.plot(fpr_grid, tpr_med, color=color, lw=1.5, linestyle="dashed", alpha=0.7,
                label=f"{layer} median")

        for fp, tm, tq25, tq75, tmn, tmx in zip(fpr_grid, tpr_med, tpr_q25, tpr_q75, tpr_min, tpr_max):
            csv_records.append({
                "layer": layer, "fpr": fp, "tpr_median": tm,
                "tpr_q25": tq25, "tpr_q75": tq75, "tpr_min": tmn, "tpr_max": tmx,
            })

        best = layer_preds[0]
        best_fpr, best_tpr = _compute_roc_curve(best.y_true, best.proba)
        ax.plot(best_fpr, best_tpr, color=color, lw=3.2, linestyle="solid",
                label=f"{layer} BEST: {best.name} (AUC={best.auc:.4f})")
        for fp, tp in zip(best_fpr, best_tpr):
            best_records.append({
                "layer": layer, "model": best.name, "auc": best.auc, "fpr": fp, "tpr": tp,
            })

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("1 − Specificity (False Positive Rate)", fontweight="bold")
    ax.set_ylabel("Sensitivity (True Positive Rate)", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 保存配套 CSV
    csv_path = Path(output_path).with_suffix(".csv")
    envelope_df = pd.DataFrame(csv_records)
    best_df = pd.DataFrame(best_records)
    combined = {"envelope": envelope_df, "best_models": best_df}
    # 由于是多表，保存到同一文件不同 sheet 不方便用 csv，故拆成两个 csv
    base = Path(output_path).with_suffix("")
    _save_csv(envelope_df, f"{base}_envelope.csv")
    _save_csv(best_df, f"{base}_best_models.csv")
    return output_path, str(base) + "_envelope.csv;" + str(base) + "_best_models.csv"


def plot_roc_facet(predictions: List[ModelPrediction], output_path: str, title: str = "ROC Facet by Layer",
                   palette: str = "default"):
    """按层分面绘制 ROC 曲线；每子图展示该层内所有模型，最佳模型加粗高亮。

    返回：
        (png_path, csv_path) —— CSV 包含每个模型在每个子图中的 ROC 点与 AUC。
    """
    _set_publication_style()
    layer_colors = _layer_colors(palette)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    csv_records = []
    for ax, layer in zip(axes, LAYER_ORDER):
        layer_preds = sorted([p for p in predictions if p.layer == layer], key=lambda x: x.auc, reverse=True)
        if not layer_preds:
            ax.text(0.5, 0.5, f"No {layer} models\n(run module to generate predictions)",
                    ha="center", va="center", fontsize=12)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            continue

        color = layer_colors.get(layer, "#333333")
        ax.plot([0, 1], [0, 1], color="#999999", lw=1.0, linestyle="solid", alpha=0.7)
        n_models = len(layer_preds)

        for idx, p in enumerate(layer_preds):
            fpr, tpr = _compute_roc_curve(p.y_true, p.proba)
            is_best = (idx == 0)
            if is_best:
                lw, alpha, zorder = 3.2, 1.0, 10
                label = f"* {p.name} ({p.auc:.4f})"
            else:
                if n_models > 15:
                    lw, alpha, zorder = 0.7, 0.22, 1
                elif n_models > 8:
                    lw, alpha, zorder = 1.0, 0.35, 1
                else:
                    lw, alpha, zorder = 1.4, 0.55, 1
                label = f"{p.name} ({p.auc:.4f})"
            ax.plot(fpr, tpr, color=color, lw=lw, linestyle="solid", alpha=alpha, zorder=zorder, label=label)
            for fp, tp in zip(fpr, tpr):
                csv_records.append({
                    "layer": layer, "model": p.name, "auc": p.auc,
                    "is_best": int(is_best), "fpr": fp, "tpr": tp,
                })

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("1 − Specificity", fontweight="bold")
        ax.set_ylabel("Sensitivity", fontweight="bold")
        ax.set_title(f"{layer} ({n_models} models)", fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 5:
            ax.legend(handles[:5], labels[:5], loc="lower right", framealpha=0.95, title="Top 5 models")
        else:
            ax.legend(handles, labels, loc="lower right", framealpha=0.95)

    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96], pad=2.5)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    csv_path = Path(output_path).with_suffix(".csv")
    _save_csv(pd.DataFrame(csv_records), csv_path)
    return output_path, str(csv_path)


def plot_auc_ranking(predictions: List[ModelPrediction], output_path: str, title: str = "AUC Ranking",
                     palette: str = "default"):
    """绘制所有模型的 AUC 排名水平条形图。

    返回：
        (png_path, csv_path) —— CSV 包含模型名、所属层、AUC。
    """
    _set_publication_style()
    layer_colors = _layer_colors(palette)
    df = pd.DataFrame([{"model": p.name, "layer": p.layer, "auc": p.auc} for p in predictions])
    df = df.sort_values("auc", ascending=True)
    n_models = len(df)
    fig_height = max(8, n_models * 0.28)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_pos = np.arange(n_models)
    colors = [layer_colors.get(layer, "#999999") for layer in df["layer"]]
    bars = ax.barh(y_pos, df["auc"].values, color=colors, edgecolor="black", linewidth=0.5)

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["auc"] + 0.005, i, f"{row['auc']:.4f}", va="center", ha="left", fontsize=8, color="#333333")

    labels = [f"{row['model']} [{row['layer']}]" for _, row in df.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("AUC", fontweight="bold", fontsize=13)
    ax.set_title(title, fontweight="bold", fontsize=15)
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_xlim([0.45, 1.0])

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=layer_colors["Base"], edgecolor="black", label="Base"),
        Patch(facecolor=layer_colors["Ensemble"], edgecolor="black", label="Ensemble"),
        Patch(facecolor=layer_colors["AutoML"], edgecolor="black", label="AutoML"),
        Patch(facecolor=layer_colors["Transformer"], edgecolor="black", label="Transformer"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.95)
    plt.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    csv_path = Path(output_path).with_suffix(".csv")
    _save_csv(df[["model", "layer", "auc"]], csv_path)
    return output_path, str(csv_path)


def plot_feature_selection_dot_matrix(consensus_table_path: str, output_path: str,
                                      title: str = "Feature Selection Dot Matrix",
                                      min_selected: int = 1,
                                      palette: str = "default"):
    """绘制特征选择方法点矩阵（类似 allcause_method_dot_matrix_clean）。

    返回：
        (png_path, csv_path) —— CSV 为绘图所使用的已排序共识表子集。
    """
    _set_publication_style()
    method_colors = _method_colors(palette)
    df = pd.read_csv(consensus_table_path)
    method_cols = [c for c in df.columns if c not in {"feature", "n_methods_selected", "selected_methods", "consensus"}]
    if not method_cols:
        raise ValueError("共识表中没有方法选择列，无法绘制点矩阵")

    if "n_methods_selected" not in df.columns:
        df["n_methods_selected"] = df[method_cols].sum(axis=1)

    plot_df = df[df["n_methods_selected"] >= min_selected].copy()
    plot_df = plot_df.sort_values("n_methods_selected", ascending=False).reset_index(drop=True)
    n_features = len(plot_df)
    n_methods = len(method_cols)

    fig_height = max(5, n_features * 0.42)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    for i, (_, row) in enumerate(plot_df.iterrows()):
        for j, method in enumerate(method_cols):
            selected = row[method]
            color = method_colors.get(method, "#333333")
            if selected == 1:
                ax.scatter(j, i, s=200, color=color, edgecolor="black", linewidth=0.8, zorder=3)
            else:
                ax.scatter(j, i, s=80, facecolor="none", edgecolor="#cccccc", linewidth=0.8, zorder=2)

    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels(plot_df["feature"].values, fontsize=10)
    ax.set_ylim([-0.5, n_features - 0.5])
    ax.invert_yaxis()

    ax.set_xticks(np.arange(n_methods))
    ax.set_xticklabels(method_cols, rotation=45, ha="right", fontsize=11)
    ax.set_xlim([-0.5, n_methods - 0.5])
    ax.set_xlabel("Feature Selection Methods", fontweight="bold", fontsize=12)

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(np.arange(n_features))
    ax2.set_yticklabels(plot_df["n_methods_selected"].astype(str).values, fontsize=10)
    ax2.set_ylabel("N selected", fontweight="bold", fontsize=12)

    ax.set_title(title, fontweight="bold", fontsize=14, pad=15)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    csv_path = Path(output_path).with_suffix(".csv")
    _save_csv(plot_df, csv_path)
    return output_path, str(csv_path)


def plot_feature_selection_consensus_dot_matrix(
    consensus_table_path: str,
    output_path: str,
    threshold: int = 4,
    title: str = "Feature Selection Consensus Dot Matrix",
    palette: str = "default",
):
    """绘制共识阈值截断后的特征选择点矩阵。

    只保留 n_methods_selected >= threshold 的特征，并在标题中标注阈值与保留数量。

    返回：
        (png_path, csv_path) —— CSV 为截断后保留的共识表。
    """
    _set_publication_style()
    method_colors = _method_colors(palette)
    df = pd.read_csv(consensus_table_path)
    method_cols = [c for c in df.columns if c not in {"feature", "n_methods_selected", "selected_methods", "consensus"}]
    if not method_cols:
        raise ValueError("共识表中没有方法选择列，无法绘制点矩阵")

    if "n_methods_selected" not in df.columns:
        df["n_methods_selected"] = df[method_cols].sum(axis=1)

    # 应用共识阈值截断
    kept = df[df["n_methods_selected"] >= threshold].copy()
    kept = kept.sort_values("n_methods_selected", ascending=False).reset_index(drop=True)
    n_features = len(kept)
    n_methods = len(method_cols)
    n_total = len(df)

    if n_features == 0:
        raise ValueError(f"阈值 {threshold} 过高，没有特征被保留")

    fig_height = max(5, n_features * 0.42)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    for i, (_, row) in enumerate(kept.iterrows()):
        for j, method in enumerate(method_cols):
            selected = row[method]
            color = method_colors.get(method, "#333333")
            if selected == 1:
                ax.scatter(j, i, s=200, color=color, edgecolor="black", linewidth=0.8, zorder=3)
            else:
                ax.scatter(j, i, s=80, facecolor="none", edgecolor="#cccccc", linewidth=0.8, zorder=2)

    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels(kept["feature"].values, fontsize=10)
    ax.set_ylim([-0.5, n_features - 0.5])
    ax.invert_yaxis()

    ax.set_xticks(np.arange(n_methods))
    ax.set_xticklabels(method_cols, rotation=45, ha="right", fontsize=11)
    ax.set_xlim([-0.5, n_methods - 0.5])
    ax.set_xlabel("Feature Selection Methods", fontweight="bold", fontsize=12)

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(np.arange(n_features))
    ax2.set_yticklabels(kept["n_methods_selected"].astype(str).values, fontsize=10)
    ax2.set_ylabel("N selected", fontweight="bold", fontsize=12)

    full_title = f"{title} (k≥{threshold}, {n_features}/{n_total} kept)"
    ax.set_title(full_title, fontweight="bold", fontsize=14, pad=15)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    csv_path = Path(output_path).with_suffix(".csv")
    _save_csv(kept, csv_path)
    return output_path, str(csv_path)


def plot_shap_pct_all_models(model_dir: Path, X: pd.DataFrame, output_path: str,
                             top_n_models: int = 5, sample_size: int = 200,
                             indicator: Optional[str] = None,
                             indicator_output_path: Optional[str] = None,
                             palette: str = "default"):
    """聚合多个 Top 模型的 SHAP 重要性百分比，并可选绘制指定指标的 SHAP 分布。

    返回：
        (png_path, csv_path) —— CSV 包含各特征的聚合 mean_abs_shap 与百分比；
        若绘制指标分布，则同时保存该指标的长格式 SHAP 值 CSV。
    """
    _set_publication_style()
    palette_cfg = get_palette(palette)
    results = load_model_results(model_dir)
    if not results:
        raise ValueError("没有模型结果可供计算 SHAP")

    results = sorted(results, key=lambda r: _compute_auc(r.y_true, r.oof_proba), reverse=True)[:top_n_models]

    agg_shap = {}
    indicator_shap_by_model: Dict[str, List[float]] = {}
    for r in results:
        model_path = model_dir / f"{r.name}.pkl"
        if not model_path.exists():
            continue
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            shap_values, X_sample = compute_shap_values(model, X, sample_size=sample_size)
            abs_shap = np.abs(shap_values).mean(axis=0)
            for j, col in enumerate(X_sample.columns):
                agg_shap[col] = agg_shap.get(col, 0.0) + abs_shap[j]
            if indicator and indicator in X_sample.columns:
                idx = list(X_sample.columns).index(indicator)
                indicator_shap_by_model[r.name] = shap_values[:, idx].tolist()
        except Exception as e:
            print(f"SHAP 计算失败 [{r.name}]: {e}")
            continue

    if not agg_shap:
        raise ValueError("没有成功计算任何 SHAP 值")

    df = pd.DataFrame({"feature": list(agg_shap.keys()), "mean_abs_shap": list(agg_shap.values())})
    df["pct"] = df["mean_abs_shap"] / df["mean_abs_shap"].sum() * 100
    df = df.sort_values("pct", ascending=True).reset_index(drop=True)

    n = len(df)
    from core.plot_style import make_gradient_colors
    colors = make_gradient_colors(
        n, primary=palette_cfg["primary"], secondary=palette_cfg["secondary"]
    )

    fig, ax = plt.subplots(figsize=(8, max(5, n * 0.42)))
    y = np.arange(n)
    bars = ax.barh(y, df["pct"].values, color=colors, edgecolor="#333333", height=0.55, linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"].values, fontsize=10)
    ax.set_xlim([0, max(35, df["pct"].max() * 1.2)])
    ax.set_xlabel("Percentage of total SHAP importance (%)", fontsize=12)
    ax.set_title(f"Aggregated SHAP Importance across Top {len(results)} Models", fontweight="bold", fontsize=14, pad=12)
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

    # 保存 SHAP 百分比 CSV
    csv_path = Path(output_path).with_suffix(".csv")
    _save_csv(df[["feature", "mean_abs_shap", "pct"]], csv_path)

    # 指标 SHAP 分布图：按模型分组的箱线图，便于比较方差与方向
    if indicator and indicator_shap_by_model and indicator_output_path:
        # 按平均 |SHAP| 降序排列模型
        model_order = sorted(
            indicator_shap_by_model.keys(),
            key=lambda m: np.mean(np.abs(indicator_shap_by_model[m])),
            reverse=True,
        )
        data_for_plot = [indicator_shap_by_model[m] for m in model_order]
        mean_abs = [np.mean(np.abs(indicator_shap_by_model[m])) for m in model_order]

        fig, ax = plt.subplots(figsize=(10, max(5, len(model_order) * 0.45)))
        bp = ax.boxplot(
            data_for_plot,
            vert=False,
            patch_artist=True,
            widths=0.6,
            showmeans=True,
            meanline=True,
            medianprops={"color": "black", "linewidth": 1.5},
            meanprops={"color": "red", "linestyle": "--", "linewidth": 1.5},
            boxprops={"facecolor": palette_cfg["secondary"], "edgecolor": "black", "alpha": 0.7},
            whiskerprops={"color": "#555555"},
            capprops={"color": "#555555"},
        )

        # 在每个箱线旁标注 mean |SHAP|
        for i, (m, ma) in enumerate(zip(model_order, mean_abs)):
            ax.text(
                ax.get_xlim()[1] * 1.01, i + 1,
                f"mean|SHAP|={ma:.3f}",
                va="center", ha="left", fontsize=9, color="#333333"
            )

        ax.set_yticks(np.arange(1, len(model_order) + 1))
        ax.set_yticklabels(model_order, fontsize=10)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.set_xlabel(f"SHAP value for {indicator}", fontsize=12)
        ax.set_ylabel("Model", fontsize=12)
        ax.set_title(
            f"SHAP Distribution of {indicator} across Top {len(model_order)} Models\n"
            "(box=IQR, black line=median, red dashed line=mean)",
            fontweight="bold", fontsize=13, pad=12
        )
        ax.grid(True, axis="x", linestyle=":", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # 调整 x 范围给标注留空
        x_min, x_max = ax.get_xlim()
        ax.set_xlim([x_min, x_max * 1.25])

        plt.tight_layout()
        fig.savefig(indicator_output_path, dpi=600, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        # 保存指标 SHAP 长格式 CSV
        indicator_csv_path = Path(indicator_output_path).with_suffix(".csv")
        long_records = []
        for m, vals in indicator_shap_by_model.items():
            for v in vals:
                long_records.append({"model": m, "indicator": indicator, "shap_value": v})
        _save_csv(pd.DataFrame(long_records), indicator_csv_path)

    return output_path, str(csv_path)


def generate_report_figures(
    model_dir: Path,
    X: pd.DataFrame,
    output_dir: Path,
    shap_sample_size: int = 200,
) -> Dict[str, str]:
    """生成报告所需图片，返回 {caption: path}。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    results = load_model_results(model_dir)
    if not results:
        return paths

    y_true = results[0].y_true

    # 1. 多模型 ROC 曲线
    roc_path = output_dir / "roc_curves.png"
    try:
        plot_roc_curves(results, y_true=y_true, output_path=str(roc_path))
        plt.close()
        paths["ROC Curves"] = str(roc_path)
    except Exception as e:
        print(f"ROC 曲线生成失败：{e}")

    # 2. 最佳模型 SHAP
    best = max(results, key=lambda r: np.mean(r.oof_proba[y_true == 1]) - np.mean(r.oof_proba[y_true == 0]))
    model_path = model_dir / f"{best.name}.pkl"
    if model_path.exists():
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            shap_values, X_sample = compute_shap_values(model, X, sample_size=shap_sample_size)

            shap_summary_path = output_dir / f"shap_summary_{best.name}.png"
            plot_shap_summary(shap_values, X_sample, output_path=str(shap_summary_path))
            plt.close()
            paths[f"SHAP Summary ({best.name})"] = str(shap_summary_path)

            shap_bar_path = output_dir / f"shap_bar_{best.name}.png"
            plot_shap_bar(shap_values, X_sample, output_path=str(shap_bar_path))
            plt.close()
            paths[f"SHAP Bar ({best.name})"] = str(shap_bar_path)
        except Exception as e:
            print(f"SHAP 生成失败：{e}")

    # 3. 最佳模型 Calibration
    calib_path = output_dir / f"calibration_{best.name}.png"
    try:
        plot_calibration_curve(y_true, best.oof_proba, n_bins=10, output_path=str(calib_path))
        plt.close()
        paths[f"Calibration ({best.name})"] = str(calib_path)
    except Exception as e:
        print(f"Calibration 生成失败：{e}")

    # 4. 最佳模型 Confusion Matrix
    cm_path = output_dir / f"confusion_matrix_{best.name}.png"
    try:
        plot_confusion_matrix(y_true, best.oof_pred, output_path=str(cm_path))
        plt.close()
        paths[f"Confusion Matrix ({best.name})"] = str(cm_path)
    except Exception as e:
        print(f"Confusion Matrix 生成失败：{e}")

    return paths


def generate_experiment_figures(
    exp_dir: Path,
    config: Dict,
    X: pd.DataFrame,
    output_dir: Path,
    shap_indicator: Optional[str] = None,
    palette: str = "default",
    top_n: int = 3,
) -> Dict[str, Dict[str, str]]:
    """生成实验级综合可视化图片（跨层），并同步输出配套原始数据 CSV。

    返回：
        {图注: {"png": png路径, "csv": csv路径}}。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    predictions = collect_all_predictions(exp_dir, config)
    if not predictions:
        return paths

    experiment_name = config.get("name", "Experiment")

    def _add(caption: str, png: Path, csv: str):
        paths[caption] = {"png": str(png), "csv": csv}

    # 1. ROC Envelope
    try:
        path = output_dir / "figure_roc_envelope.png"
        _, csv = plot_roc_envelope(predictions, str(path), title=f"{experiment_name} — ROC Envelope", palette=palette)
        _add("ROC Envelope", path, csv)
    except Exception as e:
        print(f"ROC Envelope 生成失败：{e}")

    # 2. ROC Facet
    try:
        path = output_dir / "figure_roc_facet.png"
        _, csv = plot_roc_facet(predictions, str(path), title=f"{experiment_name} — ROC by Layer", palette=palette)
        _add("ROC Facet", path, csv)
    except Exception as e:
        print(f"ROC Facet 生成失败：{e}")

    # 3. AUC Ranking
    try:
        path = output_dir / "figure_auc_ranking.png"
        _, csv = plot_auc_ranking(predictions, str(path), title=f"{experiment_name} — AUC Ranking", palette=palette)
        _add("AUC Ranking", path, csv)
    except Exception as e:
        print(f"AUC Ranking 生成失败：{e}")

    # 4. Feature Selection Dot Matrix
    fs_config = config.get("feature_selection") or {}
    consensus_path = fs_config.get("consensus_table_path")
    threshold = fs_config.get("consensus_threshold", 4)
    if consensus_path and Path(consensus_path).exists():
        try:
            path = output_dir / "figure_feature_selection_dot_matrix.png"
            _, csv = plot_feature_selection_dot_matrix(str(consensus_path), str(path),
                                              title=f"{experiment_name} — Feature Selection Dot Matrix",
                                              palette=palette)
            _add("Feature Selection Dot Matrix", path, csv)
        except Exception as e:
            print(f"Feature Selection Dot Matrix 生成失败：{e}")

        # 4b. 共识阈值截断后的点矩阵
        try:
            path = output_dir / "figure_feature_selection_consensus_dot_matrix.png"
            _, csv = plot_feature_selection_consensus_dot_matrix(
                str(consensus_path), str(path),
                threshold=int(threshold),
                title=f"{experiment_name} — Consensus Feature Selection Dot Matrix",
                palette=palette,
            )
            _add("Feature Selection Consensus Dot Matrix", path, csv)
        except Exception as e:
            print(f"Feature Selection Consensus Dot Matrix 生成失败：{e}")

    # 5. SHAP Pct + Indicator Distribution
    modeling_config = config.get("modeling") or {}
    if modeling_config.get("output_dir"):
        try:
            shap_path = output_dir / "figure_shap_pct.png"
            indicator_path = output_dir / f"figure_shap_distribution_{shap_indicator}.png" if shap_indicator else None
            _, csv = plot_shap_pct_all_models(
                Path(modeling_config["output_dir"]), X, str(shap_path),
                top_n_models=top_n, sample_size=100,
                indicator=shap_indicator,
                indicator_output_path=str(indicator_path) if indicator_path else None,
                palette=palette,
            )
            _add("SHAP Percentage", shap_path, csv)
            if indicator_path:
                # 指标分布 CSV 与图同名
                _add(f"SHAP Distribution ({shap_indicator})", indicator_path,
                     str(indicator_path.with_suffix(".csv")))
        except Exception as e:
            print(f"SHAP Pct 生成失败：{e}")

    return paths
