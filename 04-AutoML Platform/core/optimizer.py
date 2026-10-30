"""Optuna 优化器：为每个基础模型提供超参数搜索。"""

import warnings
from typing import Dict

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score

from core.base_models import MODEL_REGISTRY, build_model, THREADED_MODELS


class OptunaOptimizer:
    def __init__(
        self,
        X,
        y,
        model_name: str,
        cv_folds: int = 5,
        scoring: str = "roc_auc",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.X = X
        self.y = y
        self.model_name = model_name
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs

        spec = MODEL_REGISTRY[model_name]
        if spec.optuna_space is None:
            raise ValueError(f"{model_name} 未定义 Optuna 搜索空间")
        self.spec = spec

    def objective(self, trial: optuna.Trial) -> float:
        params = self.spec.optuna_space(trial)
        model = build_model(self.model_name, params, random_state=self.random_state)
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        # 多线程模型折串行（模型内已满核），单线程模型折级并行
        cv_n_jobs = 1 if self.model_name in THREADED_MODELS else self.n_jobs
        try:
            scores = cross_val_score(
                model, self.X, self.y,
                cv=cv,
                scoring=self.scoring,
                n_jobs=cv_n_jobs,
            )
            return float(np.mean(scores))
        except Exception as e:
            warnings.warn(f"Trial 失败：{e}")
            return 0.0

    def optimize(self, n_trials: int = 30, direction: str = "maximize", show_progress: bool = False) -> Dict:
        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(
            self.objective,
            n_trials=n_trials,
            show_progress_bar=show_progress,
        )
        return dict(study.best_params)
