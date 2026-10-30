"""Phase 2 测试：预处理 + 特征选择。"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.preprocessor import Preprocessor, apply_imbalance_handling
from core.feature_selector import FeatureSelectionRunner


def make_synthetic_data(n=200, n_features=10):
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(n, n_features),
        columns=[f"f{i}" for i in range(n_features)],
    )
    # 加入一个分类变量
    X["cat"] = np.random.choice(["A", "B", "C"], size=n)
    # 加入缺失值
    X.loc[np.random.choice(n, 10, replace=False), "f0"] = np.nan
    y = pd.Series((X["f0"] + X["f1"] - X["f2"] > 0).astype(int), name="outcome")
    return X, y


def test_preprocessor():
    X, y = make_synthetic_data()
    prep = Preprocessor(
        numeric_columns=[c for c in X.columns if c != "cat"],
        categorical_columns=["cat"],
        missing_strategy="median",
        encoding="ordinal",
        scaling="standard",
    )
    X_trans = prep.fit_transform(X)
    assert X_trans.isna().sum().sum() == 0
    assert "cat" in X_trans.columns
    print(f"Preprocessor OK: {X.shape} -> {X_trans.shape}")


def test_feature_selection():
    X, y = make_synthetic_data()
    # 编码分类变量
    X["cat"] = X["cat"].astype("category").cat.codes
    runner = FeatureSelectionRunner(
        methods=["LASSO", "MutualInformation", "RFE"],
        consensus_threshold=2,
    )
    table = runner.run(X, y)
    print(f"Feature selection OK: {table.shape[0]} features, consensus features={table['consensus'].sum()}")
    print(table.head())


if __name__ == "__main__":
    test_preprocessor()
    test_feature_selection()
