"""预处理引擎：缺失值、异常值、编码、缩放、不平衡处理。

设计原则：
- 默认使用 Label / Ordinal Encoding，禁用 one-hot，保持 SHAP 解释一致性。
- 所有步骤封装为 sklearn Pipeline，可持久化。
- 支持复杂抽样设计的 weight/strata/psu 列透传。
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder
from sklearn.feature_selection import VarianceThreshold


class IdentityPassthrough(BaseEstimator, TransformerMixin):
    """透传列，不做任何变换（用于复杂抽样设计的 weight/strata/psu 等）。"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features


class TypeCaster(BaseEstimator, TransformerMixin):
    """将输入统一转为字符串类型，避免 OrdinalEncoder 因混合类型报错。"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=object)

    def get_feature_names_out(self, input_features=None):
        return input_features


class NumericTypeCaster(BaseEstimator, TransformerMixin):
    """将连续型列强制转换为数值类型。

    若列中存在无法转换为数值的字符串（如 '≥30'），则抛出明确错误，
    提示用户应将这些列标记为分类变量。
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            # 先尝试直接转换
            converted = pd.to_numeric(X[col], errors="coerce")
            n_invalid = converted.isna().sum() - X[col].isna().sum()
            if n_invalid > 0:
                # 找出原始非缺失但转换失败的样本值
                invalid_mask = X[col].notna() & converted.isna()
                invalid_samples = X.loc[invalid_mask, col].dropna().unique()[:5].tolist()
                raise ValueError(
                    f"连续型列 '{col}' 包含无法转换为数值的内容（例如：{invalid_samples}）。"
                    f"该列实际为分类变量，请在数据选项卡中将其角色改为「分类」或「binary」，"
                    f"而不是「连续」。"
                )
            X[col] = converted
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features


class OutlierClipper(BaseEstimator, TransformerMixin):
    """使用 IQR 或百分位截断异常值。"""

    def __init__(self, method: str = "iqr", lower: float = 0.01, upper: float = 0.99):
        self.method = method
        self.lower = lower
        self.upper = upper
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        if self.method == "iqr":
            q1 = X.quantile(0.25)
            q3 = X.quantile(0.75)
            iqr = q3 - q1
            self.lower_bounds_ = q1 - 1.5 * iqr
            self.upper_bounds_ = q3 + 1.5 * iqr
        elif self.method == "winsorize":
            self.lower_bounds_ = X.quantile(self.lower)
            self.upper_bounds_ = X.quantile(self.upper)
        else:
            raise ValueError(f"不支持的异常值处理方法：{self.method}")
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = X[col].clip(lower=self.lower_bounds_[col], upper=self.upper_bounds_[col])
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features


class Preprocessor:
    def __init__(
        self,
        numeric_columns: List[str] = None,
        categorical_columns: List[str] = None,
        passthrough_columns: List[str] = None,
        missing_strategy: str = "median",
        outlier_method: str = "iqr",
        encoding: str = "ordinal",
        scaling: str = "standard",
        variance_threshold: float = 0.0,
    ):
        self.numeric_columns = numeric_columns or []
        self.categorical_columns = categorical_columns or []
        self.passthrough_columns = passthrough_columns or []
        self.missing_strategy = missing_strategy
        self.outlier_method = outlier_method
        self.encoding = encoding
        self.scaling = scaling
        self.variance_threshold = variance_threshold

        self.pipeline_: Optional[Pipeline] = None
        self.feature_names_out_: List[str] = []

    def _build_numeric_pipeline(self) -> Pipeline:
        steps = [("type_caster", NumericTypeCaster())]
        if self.missing_strategy == "median":
            steps.append(("imputer", SimpleImputer(strategy="median")))
        elif self.missing_strategy == "mean":
            steps.append(("imputer", SimpleImputer(strategy="mean")))
        elif self.missing_strategy == "knn":
            steps.append(("imputer", KNNImputer(n_neighbors=5)))
        elif self.missing_strategy == "drop":
            steps.append(("imputer", SimpleImputer(strategy="median")))  # 先填充，避免删除
        else:
            raise ValueError(f"不支持的缺失值策略：{self.missing_strategy}")

        if self.outlier_method in {"iqr", "winsorize"}:
            steps.append(("outlier", OutlierClipper(method=self.outlier_method)))

        if self.scaling == "standard":
            steps.append(("scaler", StandardScaler()))
        elif self.scaling == "minmax":
            steps.append(("scaler", MinMaxScaler()))
        elif self.scaling == "robust":
            steps.append(("scaler", RobustScaler()))

        return Pipeline(steps)

    def _build_categorical_pipeline(self) -> Pipeline:
        steps = []
        if self.encoding in {"label", "ordinal"}:
            steps.append(("caster", TypeCaster()))
            steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
            steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
        elif self.encoding == "target":
            # target encoding 需要在拟合时传入 y，暂用 ordinal 占位
            steps.append(("caster", TypeCaster()))
            steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
            steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
        else:
            raise ValueError(f"不支持的编码方式：{self.encoding}（禁用 one-hot）")
        return Pipeline(steps)

    def build_pipeline(self) -> Pipeline:
        transformers = []
        if self.numeric_columns:
            transformers.append(("num", self._build_numeric_pipeline(), self.numeric_columns))
        if self.categorical_columns:
            transformers.append(("cat", self._build_categorical_pipeline(), self.categorical_columns))
        if self.passthrough_columns:
            transformers.append(("pass", IdentityPassthrough(), self.passthrough_columns))

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

        steps = [("preprocessor", preprocessor)]
        if self.variance_threshold and self.variance_threshold > 0:
            steps.append(("variance", VarianceThreshold(threshold=self.variance_threshold)))

        self.pipeline_ = Pipeline(steps)
        return self.pipeline_

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self.build_pipeline()
        X_transformed = self.pipeline_.fit_transform(X, y)

        # 重建列名
        names = []
        if self.numeric_columns:
            names.extend(self.numeric_columns)
        if self.categorical_columns:
            names.extend(self.categorical_columns)
        if self.passthrough_columns:
            names.extend(self.passthrough_columns)

        if hasattr(self.pipeline_, "get_feature_names_out"):
            try:
                names = self.pipeline_.get_feature_names_out(names)
            except Exception:
                pass

        self.feature_names_out_ = list(names)
        return pd.DataFrame(X_transformed, columns=self.feature_names_out_, index=X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline_ is None:
            raise ValueError("请先调用 fit_transform")
        X_transformed = self.pipeline_.transform(X)
        return pd.DataFrame(X_transformed, columns=self.feature_names_out_, index=X.index)

    def get_feature_names_out(self) -> List[str]:
        return self.feature_names_out_


def apply_imbalance_handling(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "none",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """处理类别不平衡。

    支持：smote, adasyn, undersample, class_weight（仅返回权重，不采样）
    """
    if method == "none":
        return X, y
    if method == "smote":
        from imblearn.over_sampling import SMOTE
        sampler = SMOTE(random_state=random_state)
        X_res, y_res = sampler.fit_resample(X, y)
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)
    if method == "adasyn":
        from imblearn.over_sampling import ADASYN
        sampler = ADASYN(random_state=random_state)
        X_res, y_res = sampler.fit_resample(X, y)
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)
    if method == "undersample":
        from imblearn.under_sampling import RandomUnderSampler
        sampler = RandomUnderSampler(random_state=random_state)
        X_res, y_res = sampler.fit_resample(X, y)
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)
    if method == "class_weight":
        return X, y  # 权重在模型层处理
    raise ValueError(f"不支持的不平衡处理方法：{method}")


def compute_class_weight(y: pd.Series, method: str = "balanced") -> Dict:
    """计算类别权重。"""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight(class_weight=method, classes=classes, y=y)
    return {cls: w for cls, w in zip(classes, weights)}
