"""特征选择模块：封装 7 种方法，输出共识特征与可视化数据。"""

import multiprocessing as mp
import os
import random
import re
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV, SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def _resolve_n_jobs(n_jobs: int) -> int:
    """将 -1 解析为物理核数（P/E 混合架构上比逻辑核数更稳），其余值截断。"""
    try:
        import psutil
        cpus = psutil.cpu_count(logical=False) or (os.cpu_count() or 1)
    except Exception:
        cpus = os.cpu_count() or 1
    if n_jobs == -1:
        return cpus
    return max(1, min(n_jobs, cpus))


def _limit_blas_threads() -> None:
    """将当前进程的 BLAS 线程数限制为 1，防止多进程并行时线程超订。"""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1, "blas")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# GA / stepAIC 多进程 worker 上下文与评估函数（必须是模块顶层函数才能被 pickle，
# 数据通过 initializer 在 worker 启动时一次性传入，避免逐任务序列化大矩阵）
# ---------------------------------------------------------------------------

_GA_CTX: Dict[str, Any] = {}


def _ga_init_context(X_values, y_values, feature_names, random_state) -> None:
    _limit_blas_threads()
    _GA_CTX.update(X=X_values, y=y_values, names=feature_names, rs=random_state)


def _ga_eval_individual(individual):
    from sklearn.model_selection import cross_val_score

    mask = np.flatnonzero(individual)
    if mask.size == 0:
        return (0.0,)
    clf = RandomForestClassifier(n_estimators=50, random_state=_GA_CTX["rs"], n_jobs=1)
    scores = cross_val_score(
        clf, _GA_CTX["X"][:, mask], _GA_CTX["y"], cv=3, scoring="roc_auc", n_jobs=1
    )
    return (float(np.mean(scores)),)


_STEPAIC_CTX: Dict[str, Any] = {}


def _stepaic_init_context(X_values, y_values, col_to_idx) -> None:
    _limit_blas_threads()
    _STEPAIC_CTX.update(X=X_values, y=y_values, idx=col_to_idx)


def _stepaic_eval_candidate(args):
    """对单个候选列拟合 Logit，返回 (列名, 该列 p 值)。列布局：const + selected + col。"""
    col, selected = args
    import statsmodels.api as sm

    ctx = _STEPAIC_CTX
    cols_idx = [ctx["idx"][c] for c in selected] + [ctx["idx"][col]]
    Xs = ctx["X"][:, cols_idx]
    try:
        model = sm.Logit(ctx["y"], sm.add_constant(Xs)).fit(disp=0)
        return col, float(model.pvalues[-1])
    except Exception:
        return col, 1.0


_LASSO_CTX: Dict[str, Any] = {}


def _lasso_init_context(X_values, y_values, folds, max_iter, random_state) -> None:
    _limit_blas_threads()
    _LASSO_CTX.update(X=X_values, y=y_values, folds=folds,
                      max_iter=max_iter, rs=random_state)


def _lasso_eval_c_fold(args):
    """拟合单个 (C, 折) 的 L1 逻辑回归，返回 (C, 折序号, 验证折 ROC AUC)。

    与 LogisticRegressionCV(penalty="l1", solver="saga", scoring="roc_auc")
    的网格逐一对应，只是把 C×折 的串行扫描展开为并行任务。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    C, fold_i = args
    ctx = _LASSO_CTX
    tr, te = ctx["folds"][fold_i]
    model = LogisticRegression(
        penalty="l1", solver="saga", C=C,
        max_iter=ctx["max_iter"], random_state=ctx["rs"],
    )
    model.fit(ctx["X"][tr], ctx["y"][tr])
    proba = model.predict_proba(ctx["X"][te])[:, 1]
    return C, fold_i, float(roc_auc_score(ctx["y"][te], proba))


class BaseFeatureSelector(ABC):
    name: str = ""

    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        self.random_state = random_state
        # n_jobs=-1 表示使用全部核心，原样传给各估计器/进程池
        self.n_jobs = n_jobs
        self.selected_features_: List[str] = []
        self.scores_: Optional[pd.Series] = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseFeatureSelector":
        pass

    def get_selected_features(self) -> List[str]:
        return self.selected_features_

    def get_scores(self) -> Optional[pd.Series]:
        return self.scores_


class LassoSelector(BaseFeatureSelector):
    name = "LASSO"

    def __init__(
        self,
        C_min: float = 1e-4,
        C_max: float = 1e4,
        n_Cs: int = 20,
        cv: int = 5,
        max_iter: int = 1000,
        coef_threshold: float = 1e-8,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        super().__init__(random_state, n_jobs)
        self.C_min = C_min
        self.C_max = C_max
        self.n_Cs = n_Cs
        self.cv = cv
        self.max_iter = max_iter
        self.coef_threshold = coef_threshold

    def fit(self, X: pd.DataFrame, y: pd.Series):
        from sklearn.model_selection import StratifiedKFold

        # 使用 ROC AUC 而非默认 accuracy 进行交叉验证，避免类别不平衡时
        # 模型退化为全零系数（始终预测多数类）。
        # C 值越小正则化越强，筛选越严格；通过 C_min/C_max/n_Cs 控制正则化路径范围。
        Cs = np.logspace(np.log10(self.C_min), np.log10(self.C_max), self.n_Cs)

        # 并行路径：LogisticRegressionCV 的 n_jobs 只并行 CV 折（5 个），
        # 这里把 C×折 的全部独立拟合铺到进程池，核数利用从 ~5 核提升到满核。
        # 拟合集合与原实现完全相同（同样的 Cs、同样的分层折），仅调度方式不同。
        best_C = None
        n_proc = _resolve_n_jobs(self.n_jobs)
        if n_proc > 1:
            executor = None
            try:
                mp.freeze_support()
                skf = StratifiedKFold(n_splits=self.cv)
                folds = [(tr, te) for tr, te in skf.split(X.values, np.asarray(y))]
                executor = ProcessPoolExecutor(
                    max_workers=n_proc,
                    initializer=_lasso_init_context,
                    initargs=(X.values, np.asarray(y), folds,
                              self.max_iter, self.random_state),
                )
                tasks = [(C, i) for C in Cs for i in range(self.cv)]
                # 探测首个任务，确认 worker 可用
                results = [executor.submit(_lasso_eval_c_fold, tasks[0]).result()]
                results += list(executor.map(_lasso_eval_c_fold, tasks[1:]))
                # 每个 C 取各折 AUC 均值；并列时取较大 C（与 LogisticRegressionCV 一致）
                mean_auc = {C: float(np.mean([a for c2, _, a in results if c2 == C]))
                            for C in Cs}
                best_C, best_auc = None, -np.inf
                for C in sorted(Cs, reverse=True):
                    if mean_auc[C] > best_auc:
                        best_C, best_auc = C, mean_auc[C]
            except Exception:
                best_C = None
            finally:
                if executor is not None:
                    executor.shutdown(wait=True)

        if best_C is not None:
            model = LogisticRegression(
                penalty="l1", solver="saga", C=best_C,
                max_iter=self.max_iter, random_state=self.random_state,
            )
            model.fit(X.values, np.asarray(y))
            coef = np.mean(np.abs(model.coef_), axis=0)
        else:
            # 回退：进程池不可用时走原始 LogisticRegressionCV（仅折级并行）
            model = LogisticRegressionCV(
                penalty="l1",
                solver="saga",
                Cs=Cs,
                cv=self.cv,
                scoring="roc_auc",
                max_iter=self.max_iter,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            model.fit(X, y)
            coef = np.mean(np.abs(model.coef_), axis=0)

        self.scores_ = pd.Series(coef, index=X.columns)
        self.selected_features_ = X.columns[coef > self.coef_threshold].tolist()
        return self


class MutualInformationSelector(BaseFeatureSelector):
    name = "MutualInformation"

    def __init__(self, k: int = 20, random_state: int = 42, n_jobs: int = -1):
        super().__init__(random_state, n_jobs)
        self.k = k

    def fit(self, X: pd.DataFrame, y: pd.Series):
        k = min(self.k, X.shape[1])
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        selector.fit(X, y)
        self.scores_ = pd.Series(selector.scores_, index=X.columns)
        self.selected_features_ = X.columns[selector.get_support()].tolist()
        return self


class RFESelector(BaseFeatureSelector):
    name = "RFE"

    def __init__(self, step: int = 1, cv: int = 5, scoring: str = "roc_auc",
                 random_state: int = 42, n_jobs: int = -1):
        super().__init__(random_state, n_jobs)
        self.step = step
        self.cv = cv
        self.scoring = scoring

    def fit(self, X: pd.DataFrame, y: pd.Series):
        estimator = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        selector = RFECV(
            estimator=estimator,
            step=self.step,
            cv=self.cv,
            scoring=self.scoring,
            # 折内随机森林已用满全部核心，外层 CV 保持串行，避免线程超订
            n_jobs=1,
        )
        selector.fit(X, y)
        self.scores_ = pd.Series(selector.ranking_, index=X.columns)  # ranking 越小越好
        self.selected_features_ = X.columns[selector.support_].tolist()
        return self


class BorutaSelector(BaseFeatureSelector):
    name = "Boruta"

    def __init__(self, max_iter: int = 100, random_state: int = 42, n_jobs: int = -1):
        super().__init__(random_state, n_jobs)
        self.max_iter = max_iter

    def fit(self, X: pd.DataFrame, y: pd.Series):
        try:
            from boruta import BorutaPy
        except ImportError:
            raise ImportError("Boruta 特征选择需要 boruta 包，请执行：pip install BorutaPy")

        estimator = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        selector = BorutaPy(
            estimator=estimator,
            n_estimators="auto",
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        selector.fit(X.values, y.values)
        support = selector.support_
        self.scores_ = pd.Series(selector.ranking_, index=X.columns)
        self.selected_features_ = X.columns[support].tolist()
        return self


class StepAICSelector(BaseFeatureSelector):
    name = "stepAIC"

    def __init__(self, direction: str = "forward", threshold_in: float = 0.05,
                 threshold_out: float = 0.10, max_iter: int = 100,
                 random_state: int = 42, n_jobs: int = -1):
        super().__init__(random_state, n_jobs)
        self.direction = direction
        self.threshold_in = threshold_in
        self.threshold_out = threshold_out
        self.max_iter = max_iter

    def fit(self, X: pd.DataFrame, y: pd.Series):
        import statsmodels.api as sm

        features = list(X.columns)
        selected = []

        if self.direction == "forward":
            # 多进程并行评估每步的候选列：数据经 initializer 一次性传入，
            # 子进程不可用时回退为串行循环。
            executor = None
            n_proc = _resolve_n_jobs(self.n_jobs)
            if n_proc > 1 and len(features) > 1:
                try:
                    mp.freeze_support()
                    executor = ProcessPoolExecutor(
                        max_workers=min(n_proc, len(features)),
                        initializer=_stepaic_init_context,
                        initargs=(X.values, np.asarray(y),
                                  {c: i for i, c in enumerate(features)}),
                    )
                    # 探测：确认 worker 能正常 pickle 并拟合，失败则回退串行
                    executor.submit(_stepaic_eval_candidate, (features[0], ())).result()
                except Exception:
                    if executor is not None:
                        executor.shutdown(wait=False, cancel_futures=True)
                        executor = None

            try:
                current_score = None
                for _ in range(min(self.max_iter, len(features))):
                    remaining = [c for c in features if c not in selected]
                    if executor is not None:
                        candidate_scores = dict(executor.map(
                            _stepaic_eval_candidate,
                            [(c, tuple(selected)) for c in remaining],
                        ))
                    else:
                        candidate_scores = {}
                        for col in remaining:
                            cols = selected + [col]
                            try:
                                model = sm.Logit(y, sm.add_constant(X[cols])).fit(disp=0)
                                candidate_scores[col] = model.pvalues.get(col, 1.0)
                            except Exception:
                                candidate_scores[col] = 1.0

                    if not candidate_scores:
                        break
                    best_col = min(candidate_scores, key=candidate_scores.get)
                    if candidate_scores[best_col] < self.threshold_in:
                        selected.append(best_col)
                    else:
                        break
            finally:
                if executor is not None:
                    executor.shutdown(wait=True)

        self.selected_features_ = selected
        self.scores_ = pd.Series({c: 0.0 if c in selected else 1.0 for c in X.columns})
        return self


class SCADSelector(BaseFeatureSelector):
    name = "SCAD"

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        coef_threshold: float = 1e-8,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        super().__init__(random_state, n_jobs)
        # C 为 L1 正则化强度的倒数：C 越小，惩罚越强，入选特征越少。
        self.C = C
        self.max_iter = max_iter
        self.coef_threshold = coef_threshold

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # SCAD 惩罚的近似代理：使用 LogisticRegression L1。
        # 通过 C 直接控制正则化强度，并可通过 coef_threshold 进一步过滤微弱系数。
        model = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        model.fit(X, y)
        coef = np.mean(np.abs(model.coef_), axis=0)
        self.scores_ = pd.Series(coef, index=X.columns)
        self.selected_features_ = X.columns[coef > self.coef_threshold].tolist()
        return self


class GASelector(BaseFeatureSelector):
    name = "GA"

    def __init__(self, population_size: int = 50, generations: int = 30,
                 cx_prob: float = 0.5, mut_prob: float = 0.2,
                 random_state: int = 42, n_jobs: int = -1):
        super().__init__(random_state, n_jobs)
        self.population_size = population_size
        self.generations = generations
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob

    def fit(self, X: pd.DataFrame, y: pd.Series):
        try:
            from deap import base, creator, tools, algorithms
        except ImportError:
            raise ImportError("GA 特征选择需要 DEAP，请执行：pip install deap")

        from sklearn.model_selection import cross_val_score

        n_features = X.shape[1]
        feature_names = list(X.columns)

        # 避免重复注册 creator
        if "FitnessMax" not in dir(creator):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if "Individual" not in dir(creator):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", np.random.randint, 0, 2)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def eval_individual(individual):
            selected = [feature_names[i] for i, v in enumerate(individual) if v == 1]
            if len(selected) == 0:
                return (0.0,)
            clf = RandomForestClassifier(n_estimators=50, random_state=self.random_state, n_jobs=1)
            scores = cross_val_score(clf, X[selected], y, cv=3, scoring="roc_auc", n_jobs=1)
            return (float(np.mean(scores)),)

        toolbox.register("evaluate", eval_individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        np.random.seed(self.random_state)
        random.seed(self.random_state)
        pop = toolbox.population(n=self.population_size)

        # 多进程并行评估种群：数据经 initializer 一次性传入各 worker，
        # 避免逐个体序列化大矩阵；子进程不可用时回退为串行评估。
        # 注意：pool.map 会对 evaluate 函数做 pickle，因此并行时必须换成
        # 模块顶层的 _ga_eval_individual（局部函数无法 pickle）。
        pool = None
        n_proc = _resolve_n_jobs(self.n_jobs)
        if n_proc > 1:
            try:
                mp.freeze_support()
                pool = mp.Pool(
                    processes=n_proc,
                    initializer=_ga_init_context,
                    initargs=(X.values, np.asarray(y), feature_names, self.random_state),
                )
                # 探测：确认 worker 能正常 pickle 并评估，失败则回退串行
                pool.map(_ga_eval_individual, [pop[0]])
                toolbox.register("map", pool.map)
                toolbox.register("evaluate", _ga_eval_individual)
            except Exception:
                if pool is not None:
                    pool.terminate()
                    pool = None
        try:
            algorithms.eaSimple(
                pop, toolbox,
                cxpb=self.cx_prob,
                mutpb=self.mut_prob,
                ngen=self.generations,
                verbose=False,
            )
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        best = tools.selBest(pop, k=1)[0]
        selected = [feature_names[i] for i, v in enumerate(best) if v == 1]
        self.selected_features_ = selected
        self.scores_ = pd.Series({c: 1.0 if c in selected else 0.0 for c in X.columns})
        return self


SELECTOR_REGISTRY = {
    "LASSO": LassoSelector,
    "Boruta": BorutaSelector,
    "RFE": RFESelector,
    "MutualInformation": MutualInformationSelector,
    "stepAIC": StepAICSelector,
    "SCAD": SCADSelector,
    "GA": GASelector,
}


class FeatureSelectionRunner:
    """统一入口：运行多种特征选择方法并计算共识特征。"""

    def __init__(self, methods: List[str] = None, consensus_threshold: int = 3,
                 random_state: int = 42, n_jobs: int = -1,
                 method_params: Dict[str, Dict] = None):
        if methods is None:
            methods = list(SELECTOR_REGISTRY.keys())
        self.methods = methods
        self.consensus_threshold = consensus_threshold
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.method_params = method_params or {}
        self.results: Dict[str, List[str]] = {}
        self.selectors: Dict[str, BaseFeatureSelector] = {}

    def run(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        import time

        from loguru import logger

        self.results = {}
        self.selectors = {}

        for method_name in self.methods:
            if method_name not in SELECTOR_REGISTRY:
                warnings.warn(f"未知特征选择方法：{method_name}，已跳过")
                continue
            selector_cls = SELECTOR_REGISTRY[method_name]
            t0 = time.perf_counter()
            logger.info(f"特征选择 [{method_name}] 开始（{len(self.methods)} 个方法之一）")
            try:
                params = self._normalize_params(method_name, self.method_params.get(method_name, {}))
                selector = selector_cls(
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    **params,
                )
                selector.fit(X, y)
                self.results[method_name] = selector.get_selected_features()
                self.selectors[method_name] = selector
                logger.info(
                    f"特征选择 [{method_name}] 完成：耗时 {time.perf_counter() - t0:.1f}s，"
                    f"选中 {len(self.results[method_name])} 个特征"
                )
            except Exception as e:
                logger.warning(
                    f"特征选择 [{method_name}] 失败（耗时 {time.perf_counter() - t0:.1f}s）：{e}"
                )
                self.results[method_name] = []

        return self.build_consensus_table(X.columns)

    def _normalize_params(self, method_name: str, params: dict) -> dict:
        """兼容旧版参数名，并过滤掉选择器不支持的参数。"""
        params = dict(params)
        if method_name == "LASSO":
            # 旧版使用 Cs（int）表示采样点数；新版使用 C_min/C_max/n_Cs
            if "Cs" in params and "n_Cs" not in params:
                params["n_Cs"] = int(params.pop("Cs"))
        elif method_name == "SCAD":
            # 旧版使用 alpha（逆正则化强度）；新版使用 C
            if "alpha" in params and "C" not in params:
                alpha = float(params.pop("alpha"))
                params["C"] = 1.0 / alpha if alpha > 0 else 1.0

        # 过滤出选择器支持的参数
        selector_cls = SELECTOR_REGISTRY[method_name]
        import inspect
        sig = inspect.signature(selector_cls.__init__)
        valid_keys = set(sig.parameters.keys()) - {"self", "random_state", "n_jobs"}
        return {k: v for k, v in params.items() if k in valid_keys}

    def build_consensus_table(self, all_features) -> pd.DataFrame:
        records = []
        for feat in all_features:
            selected_by = [m for m, selected in self.results.items() if feat in selected]
            records.append({
                "feature": feat,
                **{m: int(feat in selected) for m, selected in self.results.items()},
                "n_methods_selected": len(selected_by),
                "selected_methods": ",".join(selected_by),
            })
        df = pd.DataFrame(records)
        df = df.sort_values("n_methods_selected", ascending=False).reset_index(drop=True)
        df["consensus"] = (df["n_methods_selected"] >= self.consensus_threshold).astype(int)
        return df

    def get_consensus_features(self) -> List[str]:
        table = self.build_consensus_table([])
        if table.empty:
            return []
        return table.loc[table["consensus"] == 1, "feature"].tolist()
