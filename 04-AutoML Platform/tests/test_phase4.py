"""Phase 4 测试：集成学习 + Transformer + AutoML + 生存分析。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.datasets import make_classification
from core.ensemble import EnsembleRunner
from core.transformers import TransformerTrainer
from core.survival import SurvivalRunner, compute_c_index


def make_data(n=300, n_features=15):
    X, y = make_classification(
        n_samples=n,
        n_features=n_features,
        n_informative=5,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series(y, name="outcome")
    return X, y


def make_survival_data(n=200, n_features=10):
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(n, n_features), columns=[f"f{i}" for i in range(n_features)])
    hazard = np.exp(X["f0"] + X["f1"])
    time = np.random.exponential(1.0 / hazard)
    event = np.random.binomial(1, 0.8, size=n)
    return X, pd.Series(time, name="time"), pd.Series(event, name="event")


def test_ensemble():
    X, y = make_data()
    np.random.seed(42)
    meta = pd.DataFrame({
        "m1": np.random.rand(len(y)),
        "m2": np.random.rand(len(y)),
        "m3": np.random.rand(len(y)),
    })
    runner = EnsembleRunner("SimpleAverage")
    runner.fit(meta, y)
    proba = runner.predict_proba(meta)
    print(f"SimpleAverage ensemble OK: {proba[:5]}")


def test_transformer():
    X, y = make_data(n=200, n_features=15)
    # 将部分特征转为整型，模拟分类变量
    for c in ["f0", "f1"]:
        X[c] = (X[c] > 0).astype(int)
    trainer = TransformerTrainer("FT-Transformer", params={"num_epochs": 5, "dim": 16, "depth": 1, "heads": 2})
    trainer.fit(X, y)
    proba = trainer.predict_proba(X)
    print(f"FT-Transformer OK: AUC shape {proba.shape}")


def test_survival():
    X, time, event = make_survival_data()
    runner = SurvivalRunner("CoxPH", params={"penalizer": 0.1})
    runner.fit(X, time, event)
    metrics = runner.evaluate(X, time, event)
    print(f"CoxPH OK: C-index={metrics['c_index']}")


if __name__ == "__main__":
    test_ensemble()
    test_transformer()
    test_survival()
