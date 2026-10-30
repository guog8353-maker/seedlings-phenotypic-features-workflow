"""可解释性模块：SHAP、Permutation Importance、PDP、ALE。"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


class _ModelProbaWrapper:
    """把需要 DataFrame 输入的模型包装成可接受 numpy 数组的形式。"""

    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names

    def _to_frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=self.feature_names)

    def _extract_proba(self, proba):
        if isinstance(proba, list):
            proba = proba[1] if len(proba) > 1 else proba[0]
        if isinstance(proba, pd.Series):
            proba = proba.values
        proba = np.asarray(proba)
        if proba.ndim == 2:
            return proba
        # 一维概率 -> 返回两列 [1-p, p]
        return np.column_stack([1 - proba, proba])

    def predict(self, X):
        proba = self.predict_proba(X)
        return proba[:, 1]

    def predict_proba(self, X):
        X_df = self._to_frame(X)
        try:
            proba = self.model.predict_proba(X_df)
        except Exception:
            # 特征名不匹配时（例如历史模型用原始名训练、而当前 X 为清洗后名字），
            # 改用 numpy 输入：sklearn 对 numpy 输入仅发出警告而不报错，
            # 且清洗仅重命名不改变列顺序，位置对齐仍然正确。
            X_arr = X_df.values if isinstance(X_df, pd.DataFrame) else np.asarray(X_df)
            try:
                proba = self.model.predict_proba(X_arr)
            except Exception:
                # 某些模型没有 predict_proba，尝试 predict
                pred = self.model.predict(X_arr)
                pred = np.asarray(pred)
                if pred.ndim == 1:
                    return np.column_stack([1 - pred, pred])
                return pred
        return self._extract_proba(proba)


_TREE_MODULE_HINTS = ("xgboost", "lightgbm", "catboost")
_TREE_SKLEARN_CLASSES = {
    "RandomForestClassifier", "ExtraTreesClassifier", "GradientBoostingClassifier",
    "DecisionTreeClassifier", "HistGradientBoostingClassifier", "AdaBoostClassifier",
}


def _is_tree_model(model) -> bool:
    """判断模型是否可用 shap.TreeExplainer（避免对非树模型无谓地 try/except）。"""
    cls = type(model)
    module = cls.__module__.lower()
    if any(h in module for h in _TREE_MODULE_HINTS):
        # DART booster 的树含 dropout 结构，TreeExplainer 不支持（会直接崩溃），
        # 路由到 Permutation 路径
        try:
            if str(model.get_params().get("booster", "")).lower() == "dart":
                return False
        except Exception:
            pass
        return True
    return cls.__name__ in _TREE_SKLEARN_CLASSES


def compute_shap_values(model, X: pd.DataFrame, feature_names: List[str] = None,
                        sample_size: int = 200, n_background: int = 50) -> np.ndarray:
    """自动选择 SHAP Explainer 并计算 SHAP 值（统一返回 shape: n_samples x n_features）。

    解释器路由：
    - 树模型 → TreeExplainer（精确、秒级）；
    - 非树模型 → PermutationExplainer（model-agnostic，评估次数 2M+1，
      比 KernelExplainer 的 2M+2048 快约 30 倍，且估计通常更稳）；
    - 上述失败 → KernelExplainer 兜底。
    """
    X_sample = X.sample(min(sample_size, len(X)), random_state=42) if len(X) > sample_size else X
    feature_names = feature_names or list(X_sample.columns)

    shap_values = None

    # 1) 树模型：TreeExplainer
    if _is_tree_model(model):
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        except Exception:
            shap_values = None

    # 2) 非树模型：PermutationExplainer（1 维概率输出，values 形状为 n×M）
    if shap_values is None:
        try:
            wrapper = _ModelProbaWrapper(model, feature_names)
            background = shap.sample(X, min(n_background, len(X)), random_state=42)
            explainer = shap.explainers.Permutation(
                wrapper.predict, shap.maskers.Independent(background), seed=42
            )
            shap_values = np.asarray(explainer(X_sample).values)
        except Exception:
            shap_values = None

    # 3) 兜底：KernelExplainer
    if shap_values is None:
        try:
            wrapper = _ModelProbaWrapper(model, feature_names)
            explainer = shap.KernelExplainer(wrapper.predict_proba, shap.sample(X, 50))
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        except Exception as e:
            raise RuntimeError(f"SHAP 计算失败：{e}")

    # 统一处理多类别输出格式
    if isinstance(shap_values, list):
        # list of arrays per class -> 取正类
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    # TreeExplainer 可能返回 (n_samples, n_features, n_classes) 数组
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # 取正类 (最后一类)
        shap_values = shap_values[:, :, -1]

    return shap_values, X_sample


def plot_shap_summary(shap_values, X_sample: pd.DataFrame, output_path: str = None,
                      figsize=(10, 8)):
    plt.figure(figsize=figsize)
    shap.summary_plot(shap_values, X_sample, show=False)
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return plt.gcf()


def plot_shap_bar(shap_values, X_sample: pd.DataFrame, output_path: str = None,
                  figsize=(10, 8)):
    plt.figure(figsize=figsize)
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return plt.gcf()


def compute_permutation_importance(model, X: pd.DataFrame, y: pd.Series,
                                   n_repeats: int = 10, scoring: str = "roc_auc",
                                   random_state: int = 42) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        scoring=scoring,
        random_state=random_state,
        n_jobs=-1,
    )
    df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })
    return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def plot_permutation_importance(importance_df: pd.DataFrame, output_path: str = None,
                                figsize=(10, 8), top_n: int = 20):
    df = importance_df.head(top_n).sort_values("importance_mean")
    plt.figure(figsize=figsize)
    plt.barh(df["feature"], df["importance_mean"], xerr=df["importance_std"])
    plt.xlabel("Permutation Importance")
    plt.title("Permutation Feature Importance")
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return plt.gcf()


def compute_pdp(model, X: pd.DataFrame, feature: str, grid_resolution: int = 20):
    """计算部分依赖图（PDP）数据。"""
    values = np.linspace(X[feature].quantile(0.05), X[feature].quantile(0.95), grid_resolution)
    predictions = []
    for v in values:
        X_temp = X.copy()
        X_temp[feature] = v
        pred = model.predict_proba(X_temp)[:, 1]
        predictions.append(pred.mean())
    return pd.DataFrame({"value": values, "partial_dependence": predictions})


def plot_pdp(pdp_df: pd.DataFrame, feature: str, output_path: str = None,
             figsize=(7, 5)):
    plt.figure(figsize=figsize)
    plt.plot(pdp_df["value"], pdp_df["partial_dependence"], lw=2)
    plt.xlabel(feature)
    plt.ylabel("Partial Dependence")
    plt.title(f"PDP: {feature}")
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return plt.gcf()


def compute_ale(model, X: pd.DataFrame, feature: str, grid_resolution: int = 20):
    """计算累积局部效应（ALE）近似值。"""
    values = np.linspace(X[feature].quantile(0.05), X[feature].quantile(0.95), grid_resolution)
    ale_values = []
    for i in range(len(values) - 1):
        lower, upper = values[i], values[i + 1]
        mask = (X[feature] >= lower) & (X[feature] < upper)
        if mask.sum() == 0:
            ale_values.append(0)
            continue
        X_lower = X.loc[mask].copy()
        X_upper = X.loc[mask].copy()
        X_lower[feature] = lower
        X_upper[feature] = upper
        pred_diff = model.predict_proba(X_upper)[:, 1] - model.predict_proba(X_lower)[:, 1]
        ale_values.append(pred_diff.mean())
    ale_cum = np.cumsum([0] + ale_values)
    ale_cum = ale_cum - ale_cum.mean()
    return pd.DataFrame({"value": values, "ale": ale_cum})


def plot_ale(ale_df: pd.DataFrame, feature: str, output_path: str = None,
             figsize=(7, 5)):
    plt.figure(figsize=figsize)
    plt.plot(ale_df["value"], ale_df["ale"], lw=2)
    plt.axhline(0, color="k", linestyle="--", lw=1)
    plt.xlabel(feature)
    plt.ylabel("ALE")
    plt.title(f"ALE: {feature}")
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return plt.gcf()
