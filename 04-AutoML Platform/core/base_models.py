"""基础模型库：32 个模型，支持默认参数与 Optuna 搜索空间。

命名尽量与 `experiments3` 保持一致，便于对照复现。
"""

from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.linear_model import (
    LogisticRegression, SGDClassifier, RidgeClassifier, Perceptron,
)
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, BaggingClassifier, HistGradientBoostingClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from core.feature_utils import sanitize_feature_names


# ---- 多线程模型名单与 GPU 开关 ----
# 这些模型内部本身使用多核（或 GPU），CV 折级并行会与它们争抢线程，
# ModelTrainer / OptunaOptimizer 对它们的交叉验证使用折串行；
# 其余单线程模型则使用折级并行。
THREADED_MODELS = {
    "XGBoost", "XGBoost-DART", "LightGBM", "LightGBM-DART", "catboost",
    "HistGradientBoosting", "RandomForest", "ExtraTrees", "Treebag",
    "Bagging", "KNN", "KKNN",
}

_USE_GPU = False
_GPU_OK: Dict[str, bool] = {}


def set_gpu_enabled(flag: bool) -> None:
    """全局 GPU 开关：开启后 XGBoost/LightGBM/CatBoost 优先用 GPU 训练。"""
    global _USE_GPU
    _USE_GPU = bool(flag)


def gpu_enabled() -> bool:
    return _USE_GPU


def _gpu_usable(lib: str) -> bool:
    """用迷你数据实测对应库的 GPU 是否可用（结果按库缓存，失败自动回退 CPU）。"""
    if lib in _GPU_OK:
        return _GPU_OK[lib]
    ok = False
    try:
        rng = np.random.RandomState(0)
        Xs = rng.rand(64, 8)
        ys = rng.randint(0, 2, 64)
        if lib == "xgboost":
            from xgboost import XGBClassifier
            XGBClassifier(n_estimators=2, tree_method="hist", device="cuda").fit(Xs, ys)
            ok = True
        elif lib == "lightgbm":
            if SanitizedLGBMClassifier is not None:
                SanitizedLGBMClassifier(n_estimators=2, device="gpu", verbose=-1).fit(Xs, ys)
                ok = True
        elif lib == "catboost":
            from catboost import CatBoostClassifier
            CatBoostClassifier(iterations=2, verbose=False, task_type="GPU").fit(Xs, ys)
            ok = True
    except Exception:
        ok = False
    _GPU_OK[lib] = ok
    return ok


# ---- LightGBM 包装：内部自动清洗特征名中的 JSON 特殊字符 ----
# 子类仍是 LGBMClassifier，因此 SHAP TreeExplainer 可正常识别并走快速路径；
# 定义在模块顶层，保证 pickle 可正常反序列化。
try:
    from lightgbm import LGBMClassifier

    class SanitizedLGBMClassifier(LGBMClassifier):
        """在 fit/predict 前自动将特征名中的空格、逗号等替换为下划线的 LGBMClassifier。"""

        @staticmethod
        def _sanitize(X):
            if hasattr(X, "columns"):
                X = X.copy()
                X.columns = sanitize_feature_names(X.columns)
            return X

        def fit(self, X, y, **kwargs):
            return super().fit(self._sanitize(X), y, **kwargs)

        def predict(self, X, **kwargs):
            return super().predict(self._sanitize(X), **kwargs)

        def predict_proba(self, X, **kwargs):
            return super().predict_proba(self._sanitize(X), **kwargs)

except ImportError:  # lightgbm 未安装
    SanitizedLGBMClassifier = None


class ModelSpec:
    """模型规格：名称、构建函数、默认参数、Optuna 搜索空间。"""

    def __init__(
        self,
        name: str,
        builder: Callable,
        default_params: Dict[str, Any],
        optuna_space: Optional[Callable] = None,
        supports_proba: bool = True,
    ):
        self.name = name
        self.builder = builder
        self.default_params = default_params
        self.optuna_space = optuna_space
        self.supports_proba = supports_proba

    def build(self, params: Dict = None, random_state: int = 42):
        if params is None:
            params = {}
        final_params = {**self.default_params, **params}
        # 注入 random_state（如支持）
        if "random_state" in self.builder.__code__.co_varnames or hasattr(self.builder, '__code__') and "random_state" in self.builder.__code__.co_varnames:
            pass
        return self.builder(random_state=random_state, **final_params)


def _set_random_state(params: dict, random_state: int) -> dict:
    p = params.copy()
    p["random_state"] = random_state
    return p


def _parse_hidden_layer_sizes(value):
    """把 Optuna 返回的字符串形式隐藏层尺寸解析为 MLPClassifier 可接受的元组。"""
    if isinstance(value, str):
        return tuple(int(x) for x in value.split(","))
    return value


# ------------------------ 32 个模型 builder ------------------------

def build_GBM(random_state=42, **kwargs):
    return GradientBoostingClassifier(**_set_random_state(kwargs, random_state))


def build_PLR(random_state=42, **kwargs):
    return LogisticRegression(max_iter=2000, solver="lbfgs", **kwargs)


def build_KNN(random_state=42, **kwargs):
    kwargs.setdefault("n_jobs", -1)
    return KNeighborsClassifier(**kwargs)


def build_XGBoost(random_state=42, **kwargs):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("XGBoost 未安装")
    params = {"use_label_encoder": False, "eval_metric": "logloss"}
    params.update(kwargs)
    if _USE_GPU and _gpu_usable("xgboost"):
        params.setdefault("tree_method", "hist")
        params.setdefault("device", "cuda")
    return XGBClassifier(**_set_random_state(params, random_state))


def build_RPART(random_state=42, **kwargs):
    return DecisionTreeClassifier(**_set_random_state(kwargs, random_state))


def build_NB(random_state=42, **kwargs):
    return GaussianNB(**kwargs)


def _oversample_to_balance(X, y, random_state=42):
    """随机过采样少数类到与多数类数量一致（缓解 MLP 在不平衡数据下塌缩）。"""
    rng = np.random.RandomState(random_state)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    # 重置索引，保证布尔索引时 X 与 y 对齐（CV 切分后索引可能不一致）
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2 or counts.min() == counts.max():
        return X, y
    minc = classes[np.argmin(counts)]
    X_min = X[y == minc]
    n_dup = int(counts.max() - counts.min())
    idx = rng.choice(len(X_min), size=n_dup, replace=True)
    X_res = pd.concat([X, X_min.iloc[idx]], ignore_index=True)
    y_res = pd.concat([y, pd.Series([minc] * n_dup)], ignore_index=True)
    return X_res, y_res


class AverageMLP(ClassifierMixin, BaseEstimator):
    """平均 MLP：训练 5 个不同初始化的 MLP 并平均预测概率。

    fit 时对少数类随机过采样，缓解不平衡数据下输出全低概率（塌缩）的问题。
    """

    def __init__(self, random_state=42, **params):
        self.random_state = random_state
        self.params = params
        self.models = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        # 少数类过采样，避免 MLP 在极度不平衡下全猜多数类
        X_res, y_res = _oversample_to_balance(X, y, self.random_state)
        self.models = []
        for i in range(5):
            m = MLPClassifier(
                max_iter=1000,
                early_stopping=True,
                random_state=self.random_state + i,
                **self.params,
            )
            m.fit(X_res, y_res)
            self.models.append(m)
        return self

    def predict_proba(self, X):
        probs = np.mean([m.predict_proba(X) for m in self.models], axis=0)
        return probs

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build_avNNET(random_state=42, **kwargs):
    if "hidden_layer_sizes" in kwargs:
        kwargs["hidden_layer_sizes"] = _parse_hidden_layer_sizes(kwargs["hidden_layer_sizes"])
    return AverageMLP(random_state=random_state, **kwargs)


def build_LDA(random_state=42, **kwargs):
    return LinearDiscriminantAnalysis(**kwargs)


class PLSClassifier(ClassifierMixin, BaseEstimator):
    """PLS 回归 + 阈值分类。"""

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pls = PLSRegression(n_components=n_components)
        self.threshold = 0.5

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.pls.fit(X, y)
        y_pred = self.pls.predict(X).ravel()
        # 简单阈值
        self.threshold = np.median(y_pred)
        return self

    def predict_proba(self, X):
        p = self.pls.predict(X).ravel()
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.pls.predict(X).ravel() >= self.threshold).astype(int)]


def build_PLS(random_state=42, **kwargs):
    return PLSClassifier(**kwargs)


def build_svmRADIAl(random_state=42, **kwargs):
    return SVC(kernel="rbf", probability=True, **kwargs)


def build_svmLinear(random_state=42, **kwargs):
    """线性 SVM：使用 LinearSVC + CalibratedClassifierCV 替代 SVC(probability=True)，
    避免大 C 值 + Platt 标定在样本量大/不平衡数据上耗时过长甚至卡死。"""
    C = kwargs.pop("C", 1.0)
    base = LinearSVC(
        C=C,
        max_iter=10000,
        dual="auto",
        random_state=random_state,
    )
    return CalibratedClassifierCV(base, cv=3, method="sigmoid")


def build_svmPoly(random_state=42, **kwargs):
    return SVC(kernel="poly", probability=True, **kwargs)


def build_Glmnet(random_state=42, **kwargs):
    params = {"l1_ratio": 0.5}
    params.update(kwargs)
    return LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        max_iter=5000,
        **_set_random_state(params, random_state),
    )


def build_KKNN(random_state=42, **kwargs):
    # 改用标准 KNeighborsClassifier：原 RadiusNeighborsClassifier(outlier_label=0)
    # 会把半径内无邻居的样本强制判为负类，在高维不平衡数据上绝大多数样本被判负（塌缩）。
    kwargs.pop("outlier_label", None)
    kwargs.setdefault("n_jobs", -1)
    return KNeighborsClassifier(**kwargs)


def build_C5_0(random_state=42, **kwargs):
    return DecisionTreeClassifier(criterion="entropy", **_set_random_state(kwargs, random_state))


class GAMWrapper(ClassifierMixin, BaseEstimator):
    """pygam LogisticGAM 包装器。"""

    def __init__(self, n_splines=10, lam=0.6):
        self.n_splines = n_splines
        self.lam = lam
        self.gam = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        try:
            from pygam import LogisticGAM, s
        except ImportError:
            raise ImportError("pygam 未安装")
        terms = s(0, n_splines=self.n_splines, lam=self.lam)
        for i in range(1, X.shape[1]):
            terms += s(i, n_splines=self.n_splines, lam=self.lam)
        self.gam = LogisticGAM(terms).fit(X, y)
        return self

    def predict_proba(self, X):
        p = self.gam.predict_proba(X)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.gam.predict(X) > 0.5).astype(int)]


def build_EARTH(random_state=42, **kwargs):
    return GAMWrapper(**kwargs)


def build_Treebag(random_state=42, **kwargs):
    kwargs.setdefault("n_jobs", -1)
    return BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        **_set_random_state(kwargs, random_state),
    )


def build_QDA(random_state=42, **kwargs):
    return QuadraticDiscriminantAnalysis(**kwargs)


class MDAC(ClassifierMixin, BaseEstimator):
    """高斯混合判别分析的简化实现。"""

    def __init__(self, n_components=2, random_state=42, **params):
        self.n_components = n_components
        self.random_state = random_state
        self.params = params
        self.models = {}

    def fit(self, X, y):
        classes = np.unique(y)
        for c in classes:
            gm = GaussianMixture(
                n_components=self.n_components,
                **_set_random_state(self.params, self.random_state),
            )
            gm.fit(X[y == c])
            self.models[c] = gm
        self.classes_ = classes
        return self

    def predict_proba(self, X):
        scores = np.array([self.models[c].score_samples(X) for c in self.classes_]).T
        scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return scores / scores.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build_MDA(random_state=42, **kwargs):
    return MDAC(random_state=random_state, **kwargs)


class FDAC(ClassifierMixin, BaseEstimator):
    """柔性判别分析：先多项式扩展再 LDA。"""

    def __init__(self, degree=2):
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.lda = LinearDiscriminantAnalysis()

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.lda.fit(self.poly.fit_transform(X), y)
        return self

    def predict_proba(self, X):
        return self.lda.predict_proba(self.poly.transform(X))

    def predict(self, X):
        return self.lda.predict(self.poly.transform(X))


def build_FDA(random_state=42, **kwargs):
    return FDAC(**kwargs)


def build_RidgeClassifier(random_state=42, **kwargs):
    # 用 CalibratedClassifierCV 包装：原 RidgeClassifier 无 predict_proba，
    # 在不平衡数据上硬预测全塌缩为多数类（AUC=0.5）；校准后可输出真实概率。
    base = RidgeClassifier(**_set_random_state(kwargs, random_state))
    return CalibratedClassifierCV(base, cv=3)


def build_ExtraTrees(random_state=42, **kwargs):
    kwargs.setdefault("n_jobs", -1)
    return ExtraTreesClassifier(**_set_random_state(kwargs, random_state))


def build_HistGradientBoosting(random_state=42, **kwargs):
    return HistGradientBoostingClassifier(**_set_random_state(kwargs, random_state))


def build_XGBoost_DART(random_state=42, **kwargs):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("XGBoost 未安装")
    params = {"booster": "dart", "use_label_encoder": False, "eval_metric": "logloss"}
    params.update(kwargs)
    if _USE_GPU and _gpu_usable("xgboost"):
        params.setdefault("tree_method", "hist")
        params.setdefault("device", "cuda")
    return XGBClassifier(**_set_random_state(params, random_state))


def build_MLP(random_state=42, **kwargs):
    if "hidden_layer_sizes" in kwargs:
        kwargs["hidden_layer_sizes"] = _parse_hidden_layer_sizes(kwargs["hidden_layer_sizes"])
    return MLPClassifier(max_iter=1000, early_stopping=True, **_set_random_state(kwargs, random_state))


def build_ElasticNet(random_state=42, **kwargs):
    return SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        max_iter=2000,
        **_set_random_state(kwargs, random_state),
    )


def build_catboost(random_state=42, **kwargs):
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        raise ImportError("CatBoost 未安装")
    params = {"verbose": False, "loss_function": "Logloss"}
    params.update(kwargs)
    # 实测（2 万行×124 列，RTX 4060）：CatBoost GPU 比 CPU 慢约 4 倍，不走 GPU。
    return CatBoostClassifier(**_set_random_state(params, random_state))


def build_Bagging(random_state=42, **kwargs):
    kwargs.setdefault("n_jobs", -1)
    return BaggingClassifier(
        estimator=LogisticRegression(max_iter=2000),
        **_set_random_state(kwargs, random_state),
    )


def build_AdaBoost(random_state=42, **kwargs):
    return AdaBoostClassifier(**_set_random_state(kwargs, random_state))


def build_LightGBM(random_state=42, **kwargs):
    if SanitizedLGBMClassifier is None:
        raise ImportError("LightGBM 未安装")
    if _USE_GPU and _gpu_usable("lightgbm"):
        kwargs.setdefault("device", "gpu")
    return SanitizedLGBMClassifier(verbose=-1, **_set_random_state(kwargs, random_state))


def build_LightGBM_DART(random_state=42, **kwargs):
    if SanitizedLGBMClassifier is None:
        raise ImportError("LightGBM 未安装")
    params = {"boosting_type": "dart", "verbose": -1}
    params.update(kwargs)
    if _USE_GPU and _gpu_usable("lightgbm"):
        params.setdefault("device", "gpu")
    return SanitizedLGBMClassifier(**_set_random_state(params, random_state))


def build_RandomForest(random_state=42, **kwargs):
    kwargs.setdefault("n_jobs", -1)
    return RandomForestClassifier(**_set_random_state(kwargs, random_state))


# ------------------------ Optuna 搜索空间 ------------------------

def optuna_GBM(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }


def optuna_PLR(trial):
    return {"C": trial.suggest_float("C", 1e-4, 10.0, log=True)}


def optuna_KNN(trial):
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 30),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
    }


def optuna_XGBoost(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def optuna_RPART(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
    }


def optuna_svmRADIAl(trial):
    return {
        "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
    }


def optuna_svmLinear(trial):
    return {"C": trial.suggest_float("C", 1e-4, 100.0, log=True)}


def optuna_svmPoly(trial):
    return {
        "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
        "degree": trial.suggest_int("degree", 2, 5),
    }


def optuna_Glmnet(trial):
    return {
        "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
    }


def optuna_C5_0(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
    }


def optuna_Treebag(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 10, 200),
        "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
    }


def optuna_RidgeClassifier(trial):
    return {"alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True)}


def optuna_ExtraTrees(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
    }


def optuna_HistGradientBoosting(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "max_iter": trial.suggest_int("max_iter", 50, 500),
    }


def optuna_MLP(trial):
    arch_str = trial.suggest_categorical(
        "hidden_layer_sizes",
        ["64", "128", "64,32", "128,64", "256,128"],
    )
    hidden_layer_sizes = tuple(int(x) for x in arch_str.split(","))
    return {
        "hidden_layer_sizes": hidden_layer_sizes,
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
    }


def optuna_ElasticNet(trial):
    return {
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
    }


def optuna_catboost(trial):
    return {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
    }


def optuna_Bagging(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 10, 200),
        "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
    }


def optuna_AdaBoost(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
    }


def optuna_LightGBM(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 10, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def optuna_RandomForest(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }


# ------------------------ 注册表 ------------------------

MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "GBM": ModelSpec("GBM", build_GBM, {}, optuna_GBM),
    "PLR": ModelSpec("PLR", build_PLR, {}, optuna_PLR),
    "KNN": ModelSpec("KNN", build_KNN, {}, optuna_KNN),
    "XGBoost": ModelSpec("XGBoost", build_XGBoost, {}, optuna_XGBoost),
    "RPART": ModelSpec("RPART", build_RPART, {}, optuna_RPART),
    "NB": ModelSpec("NB", build_NB, {}, None),
    "avNNET": ModelSpec("avNNET", build_avNNET, {}, optuna_MLP),
    "LDA": ModelSpec("LDA", build_LDA, {}, None),
    "PLS": ModelSpec("PLS", build_PLS, {}, None),
    "svmRADIAl": ModelSpec("svmRADIAl", build_svmRADIAl, {}, optuna_svmRADIAl),
    "svmLinear": ModelSpec("svmLinear", build_svmLinear, {}, optuna_svmLinear),
    "svmPoly": ModelSpec("svmPoly", build_svmPoly, {}, optuna_svmPoly),
    "Glmnet": ModelSpec("Glmnet", build_Glmnet, {}, optuna_Glmnet),
    "KKNN": ModelSpec("KKNN", build_KKNN, {}, None),
    "C5.0": ModelSpec("C5.0", build_C5_0, {}, optuna_C5_0),
    "EARTH": ModelSpec("EARTH", build_EARTH, {}, None),
    "Treebag": ModelSpec("Treebag", build_Treebag, {}, optuna_Treebag),
    "QDA": ModelSpec("QDA", build_QDA, {}, None),
    "MDA": ModelSpec("MDA", build_MDA, {}, None),
    "FDA": ModelSpec("FDA", build_FDA, {}, None),
    "RidgeClassifier": ModelSpec("RidgeClassifier", build_RidgeClassifier, {}, optuna_RidgeClassifier),
    "ExtraTrees": ModelSpec("ExtraTrees", build_ExtraTrees, {}, optuna_ExtraTrees),
    "HistGradientBoosting": ModelSpec("HistGradientBoosting", build_HistGradientBoosting, {}, optuna_HistGradientBoosting),
    "XGBoost-DART": ModelSpec("XGBoost-DART", build_XGBoost_DART, {}, optuna_XGBoost),
    "MLP": ModelSpec("MLP", build_MLP, {}, optuna_MLP),
    "ElasticNet": ModelSpec("ElasticNet", build_ElasticNet, {}, optuna_ElasticNet),
    "catboost": ModelSpec("catboost", build_catboost, {}, optuna_catboost),
    "Bagging": ModelSpec("Bagging", build_Bagging, {}, optuna_Bagging),
    "AdaBoost": ModelSpec("AdaBoost", build_AdaBoost, {}, optuna_AdaBoost),
    "LightGBM": ModelSpec("LightGBM", build_LightGBM, {}, optuna_LightGBM),
    "LightGBM-DART": ModelSpec("LightGBM-DART", build_LightGBM_DART, {}, optuna_LightGBM),
    "RandomForest": ModelSpec("RandomForest", build_RandomForest, {}, optuna_RandomForest),
}


def get_model_names() -> list:
    return list(MODEL_REGISTRY.keys())


def build_model(name: str, params: dict = None, random_state: int = 42):
    spec = MODEL_REGISTRY[name]
    return spec.build(params, random_state)
