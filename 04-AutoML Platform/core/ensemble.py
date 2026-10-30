"""集成学习模块：9 种方法（含 v2.1 新增的 DES-KNORA 动态集成选择）。

输入为基础模型的 OOF 预测概率（或新数据预测概率）。
"""

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict


class EnsembleMethod:
    name: str = ""

    def __init__(self, name: str):
        self.name = name
        self.weights_: Optional[np.ndarray] = None

    def fit(self, X_meta: np.ndarray, y: np.ndarray, model_names: List[str] = None):
        """X_meta: (n_samples, n_models) 各基模型预测概率。"""
        pass

    def predict_proba(self, X_meta: np.ndarray) -> np.ndarray:
        pass

    def predict(self, X_meta: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X_meta) >= 0.5).astype(int)


class SimpleAverageEnsemble(EnsembleMethod):
    """Simple Average：等权平均概率。"""

    def __init__(self):
        super().__init__("SimpleAverage")

    def fit(self, X_meta, y, model_names=None):
        return self

    def predict_proba(self, X_meta):
        return np.mean(X_meta, axis=1)


class WeightedAverageEnsemble(EnsembleMethod):
    """Weighted Average：按各模型 OOF AUC 加权。"""

    def __init__(self):
        super().__init__("WeightedAverage")

    def fit(self, X_meta, y, model_names=None):
        from sklearn.metrics import roc_auc_score
        aucs = np.array([roc_auc_score(y, X_meta[:, i]) for i in range(X_meta.shape[1])])
        # 避免负数，最小 0.5
        aucs = np.clip(aucs, 0.5, 1.0)
        self.weights_ = aucs / aucs.sum()
        return self

    def predict_proba(self, X_meta):
        return np.average(X_meta, axis=1, weights=self.weights_)


class GASelectionEnsemble(EnsembleMethod):
    """GA Selection：遗传算法优化模型权重。"""

    def __init__(self, population_size=50, generations=30, random_state=42):
        super().__init__("GASelection")
        self.population_size = population_size
        self.generations = generations
        self.random_state = random_state

    def fit(self, X_meta, y, model_names=None):
        try:
            from deap import base, creator, tools, algorithms
        except ImportError:
            raise ImportError("GA 集成需要 DEAP，请执行：pip install deap")

        n_models = X_meta.shape[1]

        if "FitnessMaxEns" not in dir(creator):
            creator.create("FitnessMaxEns", base.Fitness, weights=(1.0,))
        if "IndividualEns" not in dir(creator):
            creator.create("IndividualEns", list, fitness=creator.FitnessMaxEns)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", np.random.random)
        toolbox.register("individual", tools.initRepeat, creator.IndividualEns, toolbox.attr_float, n=n_models)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        from sklearn.metrics import roc_auc_score

        def eval_weights(weights):
            w = np.abs(np.array(weights))
            w = w / w.sum()
            proba = np.average(X_meta, axis=1, weights=w)
            return (roc_auc_score(y, proba),)

        toolbox.register("evaluate", eval_weights)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # 注册边界算子
        toolbox.register("map", map)

        np.random.seed(self.random_state)
        pop = toolbox.population(n=self.population_size)
        algorithms.eaSimple(
            pop, toolbox,
            cxpb=0.5,
            mutpb=0.2,
            ngen=self.generations,
            verbose=False,
        )
        best = tools.selBest(pop, k=1)[0]
        w = np.abs(np.array(best))
        self.weights_ = w / w.sum()
        return self

    def predict_proba(self, X_meta):
        return np.average(X_meta, axis=1, weights=self.weights_)


class StackingEnsemble(EnsembleMethod):
    """Stacking：用多个元学习器进行二次学习。"""

    META_LEARNERS = {
        "LR": LogisticRegression(max_iter=2000),
        "GBM": GradientBoostingClassifier(random_state=42),
        "RF": RandomForestClassifier(random_state=42, n_jobs=-1),
        "MLP": MLPClassifier(max_iter=1000, early_stopping=True, random_state=42),
    }

    def __init__(self, meta_learners: List[str] = None, cv=5, random_state=42, n_jobs=-1):
        super().__init__("Stacking")
        self.meta_learners = meta_learners or ["LR"]
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models_ = {}

    def fit(self, X_meta, y, model_names=None):
        cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        for name in self.meta_learners:
            if name not in self.META_LEARNERS:
                warnings.warn(f"未知元学习器 {name}，跳过")
                continue
            # clone 模块级共享的估计器实例，避免并发/重复 fit 时互相污染；
            # n_jobs=1（如 Optuna trial 并行场景）时把内层模型也降为单核防超订
            base = clone(self.META_LEARNERS[name])
            if self.n_jobs == 1:
                try:
                    base.set_params(n_jobs=1)
                except Exception:
                    pass
            # OOF 预测作为下一层输入
            oof_meta = cross_val_predict(base, X_meta, y, cv=cv, method="predict_proba",
                                         n_jobs=self.n_jobs)[:, 1]
            base.fit(X_meta, y)
            self.models_[name] = {"model": base, "oof": oof_meta}

        # 选择 OOF AUC 最好的元学习器作为最终模型
        from sklearn.metrics import roc_auc_score
        best_name = max(self.models_, key=lambda n: roc_auc_score(y, self.models_[n]["oof"]))
        self.final_model_ = self.models_[best_name]["model"]
        self.oof_proba_ = self.models_[best_name]["oof"]
        return self

    def predict_proba(self, X_meta):
        return self.final_model_.predict_proba(X_meta)[:, 1]


class CaruanaEnsembleSelection(EnsembleMethod):
    """Caruana Ensemble Selection：贪心向前选择模型。"""

    def __init__(self, metric="auc", max_models=None):
        super().__init__("CaruanaEnsembleSelection")
        self.metric = metric
        self.max_models = max_models

    def fit(self, X_meta, y, model_names=None):
        from sklearn.metrics import roc_auc_score
        n_models = X_meta.shape[1]
        max_models = self.max_models or n_models
        selected = []
        best_score = -np.inf

        for _ in range(max_models):
            best_candidate = None
            best_candidate_score = best_score
            for i in range(n_models):
                if i in selected:
                    continue
                current = selected + [i]
                proba = np.mean(X_meta[:, current], axis=1)
                score = roc_auc_score(y, proba)
                if score > best_candidate_score:
                    best_candidate_score = score
                    best_candidate = i
            if best_candidate is not None and best_candidate_score > best_score:
                selected.append(best_candidate)
                best_score = best_candidate_score
            else:
                break

        self.selected_indices_ = selected
        self.weights_ = np.ones(n_models) / len(selected) if selected else np.ones(n_models) / n_models
        for i in range(n_models):
            if i not in selected:
                self.weights_[i] = 0.0
        return self

    def predict_proba(self, X_meta):
        return np.average(X_meta, axis=1, weights=self.weights_)


class CaruanaOptimizedEnsemble(EnsembleMethod):
    """Caruana Optimized：在 Caruana 选择后用 Logistic 回归优化权重。"""

    def __init__(self, max_models=None):
        super().__init__("CaruanaOptimized")
        self.max_models = max_models

    def fit(self, X_meta, y, model_names=None):
        selector = CaruanaEnsembleSelection(max_models=self.max_models)
        selector.fit(X_meta, y)
        selected = selector.selected_indices_

        if not selected:
            self.coef_ = np.ones(X_meta.shape[1]) / X_meta.shape[1]
            return self

        X_sel = X_meta[:, selected]
        model = LogisticRegression(max_iter=2000)
        model.fit(X_sel, y)
        self.selected_indices_ = selected
        self.model_ = model
        return self

    def predict_proba(self, X_meta):
        X_sel = X_meta[:, self.selected_indices_]
        return self.model_.predict_proba(X_sel)[:, 1]


class SuperLearnerEnsemble(EnsembleMethod):
    """Super Learner：交叉验证拟合元学习器的标准实现。"""

    def __init__(self, meta_learner=None, cv=5, random_state=42, n_jobs=-1):
        super().__init__("SuperLearner")
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        if meta_learner is None:
            self.meta_learner = LogisticRegression(max_iter=2000)
        elif isinstance(meta_learner, str):
            from core.ensemble_optimizer import META_LEARNER_OPTIONS
            self.meta_learner = META_LEARNER_OPTIONS.get(meta_learner, LogisticRegression(max_iter=2000))
        else:
            self.meta_learner = meta_learner

    def fit(self, X_meta, y, model_names=None):
        cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        # clone 避免污染调用方传入的估计器实例（Optuna 共享实例场景）
        self.meta_learner = clone(self.meta_learner)
        if self.n_jobs == 1:
            try:
                self.meta_learner.set_params(n_jobs=1)
            except Exception:
                pass
        self.oof_meta_ = cross_val_predict(
            self.meta_learner, X_meta, y,
            cv=cv, method="predict_proba", n_jobs=self.n_jobs,
        )[:, 1]
        self.oof_proba_ = self.oof_meta_
        self.meta_learner.fit(X_meta, y)
        return self

    def predict_proba(self, X_meta):
        return self.meta_learner.predict_proba(X_meta)[:, 1]


class HardVotingEnsemble(EnsembleMethod):
    """Hard Voting：基于类别标签投票。"""

    def __init__(self):
        super().__init__("HardVoting")

    def fit(self, X_meta, y, model_names=None):
        return self

    def predict(self, X_meta):
        preds = (X_meta >= 0.5).astype(int)
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=preds)

    def predict_proba(self, X_meta):
        # 硬投票的概率用投票比例近似
        preds = (X_meta >= 0.5).astype(int)
        pos_votes = preds.sum(axis=1) / preds.shape[1]
        return pos_votes


ENSEMBLE_REGISTRY = {
    "SimpleAverage": SimpleAverageEnsemble,
    "WeightedAverage": WeightedAverageEnsemble,
    "GASelection": GASelectionEnsemble,
    "Stacking": StackingEnsemble,
    "CaruanaEnsembleSelection": CaruanaEnsembleSelection,
    "CaruanaOptimized": CaruanaOptimizedEnsemble,
    "SuperLearner": SuperLearnerEnsemble,
    "HardVoting": HardVotingEnsemble,
}


class EnsembleRunner:
    def __init__(self, method: str, method_params: dict = None):
        if method not in ENSEMBLE_REGISTRY:
            raise ValueError(f"未知集成方法：{method}")
        self.method = method
        self.method_params = method_params or {}
        self.ensemble_: Optional[EnsembleMethod] = None

    def fit(self, X_meta: pd.DataFrame, y: pd.Series):
        cls = ENSEMBLE_REGISTRY[self.method]
        self.ensemble_ = cls(**self.method_params)
        self.ensemble_.fit(X_meta.values, y.values, model_names=list(X_meta.columns))
        return self

    def predict_proba(self, X_meta: pd.DataFrame) -> np.ndarray:
        return self.ensemble_.predict_proba(X_meta.values)

    def predict(self, X_meta: pd.DataFrame) -> np.ndarray:
        return self.ensemble_.predict(X_meta.values)


# v2.1 新增：动态集成选择（KNORA-E/U）。放在模块底部注册，
# 因为 core.des_ensemble 需要引用本模块的 EnsembleMethod（避免循环导入）。
# des_ensemble 先被导入时此 import 会失败，由其模块底部完成注册兜底。
try:
    from core.des_ensemble import DES_REGISTRY

    ENSEMBLE_REGISTRY.update(DES_REGISTRY)
except ImportError:
    pass
