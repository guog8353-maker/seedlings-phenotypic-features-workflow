"""评估与可视化辅助：ROC、PR、Calibration、DCA、Confusion Matrix。"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    auc,
)
from sklearn.calibration import calibration_curve


def plot_roc_curves(results: List, y_true=None, output_path: str = None, figsize=(8, 8)):
    """绘制多条 ROC 曲线。

    Args:
        results: ModelResult 列表，每个对象需包含 name、oof_proba。
        y_true: 真实标签。若 results[0] 包含 y_true 属性也可自动获取。
        output_path: 保存路径。
    """
    plt.figure(figsize=figsize)
    if y_true is None and results and hasattr(results[0], "y_true"):
        y_true = results[0].y_true

    if y_true is None:
        raise ValueError("plot_roc_curves 需要 y_true 参数或 results 中包含 y_true")

    for r in results:
        fpr, tpr, _ = roc_curve(y_true, r.oof_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{r.name} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return plt.gcf()


def plot_roc_curve(y_true, y_proba, label: str = "Model", output_path: str = None, figsize=(6, 6)):
    from sklearn.metrics import auc
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return plt.gcf()


def plot_pr_curve(y_true, y_proba, label: str = "Model", output_path: str = None, figsize=(6, 6)):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, lw=2, label=label)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return plt.gcf()


def plot_calibration_curve(y_true, y_proba, n_bins: int = 10, output_path: str = None, figsize=(6, 6)):
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    prob_true = np.asarray(prob_true)
    prob_pred = np.asarray(prob_pred)
    plt.figure(figsize=figsize)
    plt.plot(prob_pred, prob_true, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend(loc="lower right")
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return plt.gcf()


def plot_confusion_matrix(y_true, y_pred, output_path: str = None, figsize=(5, 4)):
    import seaborn as sns
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return plt.gcf()


def compute_dca(y_true, y_proba, thresholds: np.ndarray = None) -> pd.DataFrame:
    """计算 Decision Curve Analysis 数据。"""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    prevalence = y_true.mean()

    records = []
    for p_t in thresholds:
        pred = (y_proba >= p_t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        net_benefit = (tp / n) - (fp / n) * (p_t / (1 - p_t))
        treat_all = prevalence - (1 - prevalence) * (p_t / (1 - p_t))
        records.append({
            "threshold": p_t,
            "net_benefit": net_benefit,
            "treat_all": treat_all,
            "treat_none": 0.0,
        })
    return pd.DataFrame(records)


def plot_dca(y_true, y_proba, output_path: str = None, figsize=(8, 6)):
    dca = compute_dca(y_true, y_proba)
    plt.figure(figsize=figsize)
    plt.plot(dca["threshold"], dca["net_benefit"], lw=2, label="Model")
    plt.plot(dca["threshold"], dca["treat_all"], "k--", lw=1, label="Treat all")
    plt.plot(dca["threshold"], dca["treat_none"], "k:", lw=1, label="Treat none")
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis")
    plt.legend(loc="upper right")
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return plt.gcf()
