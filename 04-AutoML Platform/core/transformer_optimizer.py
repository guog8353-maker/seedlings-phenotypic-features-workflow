"""Transformer 超参数 Optuna 优化器。"""

import warnings
from typing import Dict

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from core.transformers import TransformerTrainer


class TransformerOptimizer:
    """对表格 Transformer 进行 Optuna 超参搜索（使用简化 CV）。"""

    def __init__(
        self,
        name: str,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 3,
        random_state: int = 42,
    ):
        self.name = name
        self.X = X
        self.y = y
        self.cv = cv
        self.random_state = random_state

    def optimize(self, n_trials: int = 10) -> Dict:
        def objective(trial):
            params = {
                "num_epochs": trial.suggest_int("num_epochs", 10, 100),
                "dim": trial.suggest_categorical("dim", [16, 32, 64]),
                "depth": trial.suggest_int("depth", 1, 4),
                "heads": trial.suggest_categorical("heads", [2, 4, 8]),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            }
            return self._cv_score(params)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return dict(study.best_params)

    def _cv_score(self, params: Dict) -> float:
        cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        scores = []
        for train_idx, val_idx in cv.split(self.X, self.y):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            try:
                trainer = TransformerTrainer(self.name, params=params, random_state=self.random_state)
                trainer.fit(X_train, y_train)
                proba = trainer.predict_proba(X_val)
                scores.append(roc_auc_score(y_val, proba))
            except Exception as e:
                warnings.warn(f"Transformer CV fold 失败：{e}")
                scores.append(0.0)
        return float(np.mean(scores))
