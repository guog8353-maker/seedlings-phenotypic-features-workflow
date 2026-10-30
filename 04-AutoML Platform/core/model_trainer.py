"""模型训练器：支持 32 个基础模型的交叉验证、OOF 预测、Optuna 调参与全数据最终模型。"""

import multiprocessing as mp
import os
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, log_loss, brier_score_loss,
    cohen_kappa_score, confusion_matrix,
)

from core import base_models
from core.base_models import MODEL_REGISTRY, build_model, THREADED_MODELS


@dataclass
class ModelResult:
    name: str
    metrics: Dict[str, float]
    oof_proba: np.ndarray
    oof_pred: np.ndarray
    model_path: Optional[str] = None
    oof_path: Optional[str] = None


# ---------------------------------------------------------------------------
# 模型级并行：worker 上下文与入口（必须是模块顶层函数才能被 pickle，
# 数据通过 initializer 在 worker 启动时一次性传入，避免逐模型序列化大矩阵）
# ---------------------------------------------------------------------------

_TRAIN_CTX: Dict[str, Any] = {}


def _physical_cores() -> int:
    try:
        import psutil
        return psutil.cpu_count(logical=False) or (os.cpu_count() or 1)
    except Exception:
        return os.cpu_count() or 1


def _trainer_init_context(X_values, y_values, columns, thread_quota) -> None:
    try:
        from threadpoolctl import threadpool_limits
        # 限制 worker 内 BLAS/OpenMP 线程，防止并发模型之间线程超订
        threadpool_limits(thread_quota, "blas_openmp")
    except Exception:
        pass
    _TRAIN_CTX.update(X=X_values, y=y_values, cols=columns, quota=thread_quota)


def _train_model_worker(args):
    """在子进程中训练单个模型，返回 ModelResult（含 numpy 数组，可 pickle）。"""
    (name, params, use_optuna, n_trials, fit_final,
     cv_folds, random_state, output_dir, class_weight, use_gpu) = args
    quota = _TRAIN_CTX["quota"]
    X = pd.DataFrame(_TRAIN_CTX["X"], columns=_TRAIN_CTX["cols"])
    y = pd.Series(_TRAIN_CTX["y"])
    trainer = ModelTrainer(
        X, y,
        cv_folds=cv_folds,
        random_state=random_state,
        output_dir=output_dir,
        n_jobs=quota,
        class_weight=class_weight,
        use_gpu=use_gpu,
    )
    # 内部多线程的模型固定到配额核数（参数名各异；HistGradientBoosting 无
    # n_jobs 参数，其 OpenMP 线程已由 worker 初始化的 threadpool_limits 限制）
    if name in THREADED_MODELS and quota:
        core_param = {"catboost": "thread_count",
                      "HistGradientBoosting": None}.get(name, "n_jobs")
        if core_param:
            params = {**(params or {}), core_param: quota}
    try:
        return trainer.train_model(
            name, params=params, use_optuna=use_optuna,
            n_trials=n_trials, fit_final=fit_final,
        )
    finally:
        # cross_val_predict 留下的 joblib 可复用 loky 进程池默认空闲 300s 才退出，
        # 会拖住本 worker 的进程退出；显式关闭让 worker 即完即走。
        try:
            from joblib.externals.loky import get_reusable_executor
            get_reusable_executor().shutdown(wait=True)
        except Exception:
            pass


def find_optimal_threshold(y_true, proba, criterion: str = "youden") -> float:
    """在验证/OOF 预测上自动寻找最优分类阈值。

    criterion:
        - "youden"：Youden's J = 敏感度 + 特异度 - 1（默认，兼顾查全与查准）；
        - "f1"：使 F1 最大的阈值；
        - 0~1 的浮点数：直接作为自定义阈值。
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    # 自定义数值阈值
    try:
        thr = float(criterion)
        if 0.0 < thr < 1.0:
            return thr
    except (TypeError, ValueError):
        pass

    from sklearn.metrics import roc_curve, precision_recall_curve
    if criterion == "f1":
        precision, recall, thresholds = precision_recall_curve(y_true, proba)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        if len(thresholds) == 0:
            return 0.5
        return float(thresholds[int(np.argmax(f1[:-1] if len(f1) > 1 else f1))]) if len(thresholds) else 0.5
    # 默认 youden
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    if len(thresholds) == 0:
        return 0.5
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


def compute_metrics_optimal_threshold(y_true, proba, criterion: str = "youden") -> Dict[str, float]:
    """用最优阈值（默认 Youden's J）计算分类指标，并附带所用阈值。"""
    proba = np.asarray(proba, dtype=float)
    thr = find_optimal_threshold(y_true, proba, criterion)
    pred = (proba >= thr).astype(int)
    metrics = compute_classification_metrics(y_true, proba, pred)
    metrics["threshold"] = round(thr, 4)
    return metrics


def compute_classification_metrics(y_true, y_proba, y_pred) -> Dict[str, float]:
    """计算二分类评估指标。"""
    eps = 1e-8
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    metrics = {
        "auc": roc_auc_score(y_true, y_proba),
        "auprc": average_precision_score(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "ppv": precision_score(y_true, y_pred, zero_division=0),
        "npv": precision_score(1 - y_true, 1 - y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "logloss": log_loss(y_true, np.clip(y_proba, eps, 1 - eps)),
        "brier": brier_score_loss(y_true, y_proba),
    }
    return {k: round(v, 4) for k, v in metrics.items()}


class ModelTrainer:
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        random_state: int = 42,
        output_dir: str = None,
        n_jobs: int = -1,
        class_weight: bool = False,
        use_gpu: bool = False,
    ):
        self.X = X
        self.y = y
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/models")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # n_jobs=-1 表示使用全部核心，原样传给交叉验证与估计器
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self.use_gpu = use_gpu
        self.results: List[ModelResult] = []

    def _cv_n_jobs(self, name: str) -> int:
        """折级并行度：内部已多线程/用 GPU 的模型走折串行（避免线程超订），
        单线程模型（GBM、MLP、SVM 等）走折级并行。"""
        return 1 if name in THREADED_MODELS else self.n_jobs

    def _apply_class_weight(self, model):
        """给支持类别权重的模型设置 balanced 权重（XGBoost 用 scale_pos_weight）。"""
        if not self.class_weight:
            return model
        # 优先 class_weight="balanced"
        try:
            model.set_params(class_weight="balanced")
            return model
        except Exception:
            pass
        # XGBoost：scale_pos_weight = 负类/正类
        try:
            n_pos = max(int((self.y == 1).sum()), 1)
            n_neg = int((self.y == 0).sum())
            model.set_params(scale_pos_weight=n_neg / n_pos)
        except Exception:
            pass
        return model

    def train_model(
        self,
        name: str,
        params: Dict = None,
        use_optuna: bool = False,
        n_trials: int = 30,
        fit_final: bool = True,
    ) -> ModelResult:
        from core.optimizer import OptunaOptimizer

        spec = MODEL_REGISTRY[name]
        params = params or {}
        base_models.set_gpu_enabled(self.use_gpu)

        # Optuna 调参
        if use_optuna and spec.optuna_space is not None:
            optimizer = OptunaOptimizer(
                X=self.X,
                y=self.y,
                model_name=name,
                cv_folds=self.cv_folds,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            best_params = optimizer.optimize(n_trials=n_trials)
            params = {**params, **best_params}

        model = self._apply_class_weight(build_model(name, params, random_state=self.random_state))

        # 交叉验证 OOF 预测
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        cv_n_jobs = self._cv_n_jobs(name)
        try:
            oof_proba = cross_val_predict(
                model, self.X, self.y,
                cv=cv,
                method="predict_proba",
                n_jobs=cv_n_jobs,
            )[:, 1]
        except Exception as e:
            warnings.warn(f"{name} predict_proba 失败：{e}，改用 predict")
            oof_pred = cross_val_predict(model, self.X, self.y, cv=cv, n_jobs=cv_n_jobs)
            oof_proba = oof_pred.astype(float)

        # 用最优阈值（Youden's J）计算指标，并用同一阈值生成 OOF 预测标签
        metrics = compute_metrics_optimal_threshold(self.y, oof_proba, "youden")
        oof_pred = (oof_proba >= metrics["threshold"]).astype(int)

        # 全数据最终模型
        if fit_final:
            model.fit(self.X, self.y)
            model_path = self.output_dir / f"{name}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        else:
            model_path = None

        oof_df = pd.DataFrame({
            "y_true": self.y.values,
            "oof_proba": oof_proba,
            "oof_pred": oof_pred,
        })
        oof_path = self.output_dir / f"{name}_oof.csv"
        oof_df.to_csv(oof_path, index=False)

        result = ModelResult(
            name=name,
            metrics=metrics,
            oof_proba=oof_proba,
            oof_pred=oof_pred,
            model_path=str(model_path) if model_path else None,
            oof_path=str(oof_path),
        )
        self.results.append(result)
        return result

    def train_multiple(
        self,
        names: List[str],
        use_optuna: bool = False,
        n_trials: int = 30,
        fit_final: bool = True,
        max_parallel: int = None,
    ) -> pd.DataFrame:
        """训练多个模型。max_parallel>1 时用进程池并发训练多个模型，
        每个模型分配到 quota = 物理核数/max_parallel 个核心；
        进程池不可用时自动回退串行。"""
        import time

        from loguru import logger

        base_models.set_gpu_enabled(self.use_gpu)
        cores = _physical_cores()
        if max_parallel is None:
            max_parallel = max(1, min(4, cores // 10))
        max_parallel = max(1, min(int(max_parallel), len(names)))

        def _run_serial(todo: List[str]):
            for name in todo:
                t0 = time.perf_counter()
                try:
                    self.train_model(name, use_optuna=use_optuna,
                                     n_trials=n_trials, fit_final=fit_final)
                    r = self.results[-1]
                    logger.info(f"模型 [{name}] 完成：耗时 {time.perf_counter()-t0:.1f}s，"
                                f"AUC={r.metrics.get('auc', float('nan')):.4f}")
                except Exception as e:
                    logger.warning(f"训练 {name} 失败：{e}")

        if max_parallel <= 1:
            _run_serial(names)
            return self.get_leaderboard()

        quota = max(1, cores // max_parallel)
        logger.info(f"模型级并行：{max_parallel} 个模型并发，每个限 {quota} 核")
        done_names = set()
        try:
            mp.freeze_support()
            with ProcessPoolExecutor(
                max_workers=max_parallel,
                initializer=_trainer_init_context,
                initargs=(self.X.values, np.asarray(self.y),
                          list(self.X.columns), quota),
            ) as executor:
                futures = {
                    executor.submit(_train_model_worker, (
                        name, None, use_optuna, n_trials, fit_final,
                        self.cv_folds, self.random_state, str(self.output_dir),
                        self.class_weight, self.use_gpu,
                    )): name for name in names
                }
                for fut in as_completed(futures):
                    name = futures[fut]
                    try:
                        result = fut.result()
                        self.results.append(result)
                        done_names.add(name)
                        logger.info(f"模型 [{name}] 完成：AUC="
                                    f"{result.metrics.get('auc', float('nan')):.4f}")
                    except Exception as e:
                        done_names.add(name)  # 失败不再重试，与串行行为一致
                        logger.warning(f"训练 {name} 失败：{e}")
        except Exception as e:
            # 进程池本身故障（如 pickle 失败）：串行补跑未完成的模型
            logger.warning(f"模型级并行不可用（{e}），回退串行")
            _run_serial([n for n in names if n not in done_names])

        return self.get_leaderboard()

    def get_leaderboard(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        records = [{"model": r.name, **r.metrics} for r in self.results]
        df = pd.DataFrame(records)
        if "auc" in df.columns:
            df = df.sort_values("auc", ascending=False).reset_index(drop=True)
        return df
