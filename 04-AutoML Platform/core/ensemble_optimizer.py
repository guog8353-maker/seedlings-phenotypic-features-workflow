"""集成学习超参数 Optuna 优化器。"""

import warnings
from typing import Dict, List

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from core.ensemble import (
    EnsembleRunner, ENSEMBLE_REGISTRY,
    GASelectionEnsemble, StackingEnsemble, CaruanaEnsembleSelection,
    CaruanaOptimizedEnsemble, SuperLearnerEnsemble,
)


META_LEARNER_OPTIONS = {
    "LR": LogisticRegression(max_iter=2000),
    "GBM": GradientBoostingClassifier(random_state=42),
    "RF": RandomForestClassifier(random_state=42, n_jobs=-1),
    "MLP": MLPClassifier(max_iter=1000, early_stopping=True, random_state=42),
}


def _compute_auc(y_true, proba):
    try:
        return roc_auc_score(y_true, proba)
    except Exception:
        return 0.0


class EnsembleOptimizer:
    """对支持超参数的集成方法进行 Optuna 搜索。"""

    def __init__(
        self,
        oof_df: pd.DataFrame,
        y_true: np.ndarray,
        cv: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.oof_df = oof_df
        self.y_true = y_true
        self.cv = cv
        self.random_state = random_state
        # trial 级并行数：-1 → 物理核数；trial 并行时内层模型降为单核防超订
        self.n_jobs = n_jobs
        # 内层模型并行度：-1 → 用满核。
        # 注意：Optuna trial 级并行（study.optimize n_jobs）实测为负优化
        # （线程级并行 + TPE 重复建议），因此 trial 保持串行，
        # 并行度全部放在 trial 内部的模型/CV 上。
        self._inner_n_jobs = -1 if n_jobs == -1 else max(1, n_jobs)

    def optimize(self, method: str, n_trials: int = 20) -> Dict:
        if method == "GASelection":
            return self._optimize_ga(n_trials)
        if method == "Stacking":
            return self._optimize_stacking(n_trials)
        if method == "CaruanaEnsembleSelection":
            return self._optimize_caruana(n_trials)
        if method == "CaruanaOptimized":
            return self._optimize_caruana_optimized(n_trials)
        if method == "SuperLearner":
            return self._optimize_super_learner(n_trials)
        raise ValueError(f"{method} 不支持 Optuna 调参或无超参数")

    def _optimize_ga(self, n_trials):
        def objective(trial):
            pop = trial.suggest_int("population_size", 10, 100)
            gen = trial.suggest_int("generations", 10, 80)
            ens = GASelectionEnsemble(population_size=pop, generations=gen, random_state=self.random_state)
            ens.fit(self.oof_df.values, self.y_true)
            proba = ens.predict_proba(self.oof_df.values)
            return _compute_auc(self.y_true, proba)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return dict(study.best_params)

    def _optimize_stacking(self, n_trials):
        def objective(trial):
            meta_choices = trial.suggest_categorical("meta_learner", list(META_LEARNER_OPTIONS.keys()))
            cv = trial.suggest_int("cv", 3, 7)
            ens = StackingEnsemble(meta_learners=[meta_choices], cv=cv,
                                   random_state=self.random_state,
                                   n_jobs=self._inner_n_jobs)
            ens.fit(self.oof_df.values, self.y_true)
            proba = ens.oof_proba_ if hasattr(ens, "oof_proba_") else ens.predict_proba(self.oof_df.values)
            return _compute_auc(self.y_true, proba)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = study.best_params
        return {
            "meta_learners": [best["meta_learner"]],
            "cv": best["cv"],
        }

    def _optimize_caruana(self, n_trials):
        def objective(trial):
            max_models = trial.suggest_int("max_models", 2, self.oof_df.shape[1])
            ens = CaruanaEnsembleSelection(max_models=max_models)
            ens.fit(self.oof_df.values, self.y_true)
            proba = ens.predict_proba(self.oof_df.values)
            return _compute_auc(self.y_true, proba)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return dict(study.best_params)

    def _optimize_caruana_optimized(self, n_trials):
        def objective(trial):
            max_models = trial.suggest_int("max_models", 2, self.oof_df.shape[1])
            ens = CaruanaOptimizedEnsemble(max_models=max_models)
            ens.fit(self.oof_df.values, self.y_true)
            proba = ens.predict_proba(self.oof_df.values)
            return _compute_auc(self.y_true, proba)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return dict(study.best_params)

    def _optimize_super_learner(self, n_trials):
        def objective(trial):
            meta_name = trial.suggest_categorical("meta_learner", list(META_LEARNER_OPTIONS.keys()))
            cv = trial.suggest_int("cv", 3, 7)
            ens = SuperLearnerEnsemble(meta_learner=META_LEARNER_OPTIONS[meta_name],
                                       cv=cv, random_state=self.random_state,
                                       n_jobs=self._inner_n_jobs)
            ens.fit(self.oof_df.values, self.y_true)
            proba = ens.oof_proba_ if hasattr(ens, "oof_proba_") else ens.predict_proba(self.oof_df.values)
            return _compute_auc(self.y_true, proba)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = study.best_params
        return {
            "meta_learner": best["meta_learner"],  # 保存字符串，避免 JSON 序列化失败
            "cv": best["cv"],
        }
