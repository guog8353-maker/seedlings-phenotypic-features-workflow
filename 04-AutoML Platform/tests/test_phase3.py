"""Phase 3 测试：基础模型 + Optuna。"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.datasets import make_classification
from core.model_trainer import ModelTrainer


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


def test_train_models():
    X, y = make_data()
    trainer = ModelTrainer(X, y, cv_folds=3, output_dir=str(PROJECT_ROOT / "outputs/models"))
    leaderboard = trainer.train_multiple(
        names=["PLR", "RandomForest", "LightGBM"],
        use_optuna=False,
    )
    print("Leaderboard:")
    print(leaderboard)
    assert not leaderboard.empty


def test_optuna():
    X, y = make_data()
    trainer = ModelTrainer(X, y, cv_folds=3, output_dir=str(PROJECT_ROOT / "outputs/models"))
    result = trainer.train_model("PLR", use_optuna=True, n_trials=5)
    print(f"Optuna PLR AUC: {result.metrics['auc']}")


if __name__ == "__main__":
    test_train_models()
    test_optuna()
