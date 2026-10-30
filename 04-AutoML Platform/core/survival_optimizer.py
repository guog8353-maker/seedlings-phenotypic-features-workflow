"""生存分析超参数 Optuna 优化器。"""

import warnings
from typing import Dict

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold

from core.survival import SurvivalRunner, compute_c_index


class SurvivalOptimizer:
    """对生存模型进行 Optuna 超参搜索。"""

    def __init__(
        self,
        name: str,
        X: pd.DataFrame,
        time: pd.Series,
        event: pd.Series,
        weights: pd.Series = None,
        cv: int = 3,
        random_state: int = 42,
    ):
        self.name = name
        self.X = X
        self.time = time
        self.event = event
        self.weights = weights
        self.cv = cv
        self.random_state = random_state

    def optimize(self, n_trials: int = 10) -> Dict:
        if self.name in {"CoxPH", "WeightedCoxPH"}:
            return self._optimize_cox(n_trials)
        if self.name == "DeepSurv":
            return self._optimize_deepsurv(n_trials)
        raise ValueError(f"{self.name} 不支持 Optuna 调参")

    def _optimize_cox(self, n_trials):
        def objective(trial):
            penalizer = trial.suggest_float("penalizer", 1e-4, 10.0, log=True)
            runner = SurvivalRunner(self.name, params={"penalizer": penalizer})
            return self._cv_score(runner)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return dict(study.best_params)

    def _optimize_deepsurv(self, n_trials):
        def objective(trial):
            hidden_dims = trial.suggest_categorical("hidden_dims", [(32,), (64,), (64, 32), (128, 64)])
            dropout = trial.suggest_float("dropout", 0.0, 0.5)
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            num_epochs = trial.suggest_int("num_epochs", 20, 150)
            batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
            params = {
                "hidden_dims": hidden_dims,
                "dropout": dropout,
                "lr": lr,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
            }
            runner = SurvivalRunner(self.name, params=params)
            return self._cv_score(runner)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return dict(study.best_params)

    def _cv_score(self, runner: SurvivalRunner) -> float:
        cv = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        scores = []
        for train_idx, val_idx in cv.split(self.X):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            t_train, t_val = self.time.iloc[train_idx], self.time.iloc[val_idx]
            e_train, e_val = self.event.iloc[train_idx], self.event.iloc[val_idx]
            try:
                runner.fit(X_train, t_train, e_train, self.weights)
                metrics = runner.evaluate(X_val, t_val, e_val)
                scores.append(metrics["c_index"])
            except Exception as e:
                warnings.warn(f"生存分析 CV fold 失败：{e}")
                scores.append(0.0)
        return float(np.mean(scores))
