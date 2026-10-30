"""AutoML 模块：H2O、AutoGluon、FLAML 统一封装。"""

import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def is_gpu_available() -> bool:
    """检测系统是否有可用的 NVIDIA GPU（不依赖 PyTorch）。"""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return result.returncode == 0 and result.stdout.decode().strip().startswith("GPU")
    except Exception:
        return False


class BaseAutoMLWrapper:
    name: str = ""

    def __init__(self, time_limit: int = 600, random_state: int = 42, use_gpu: bool = False):
        self.time_limit = time_limit
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.model_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)


class H2OAutoMLWrapper(BaseAutoMLWrapper):
    name = "H2O AutoML"

    def __init__(self, time_limit: int = 600, random_state: int = 42, use_gpu: bool = False):
        super().__init__(time_limit, random_state, use_gpu)

    def fit(self, X, y):
        try:
            import h2o
            from h2o.automl import H2OAutoML
        except ImportError:
            raise ImportError("H2O 未安装，请执行：pip install h2o")

        h2o.init()
        df = X.copy()
        df["target"] = y.values
        h2o_df = h2o.H2OFrame(df)
        h2o_df["target"] = h2o_df["target"].asfactor()

        aml = H2OAutoML(
            max_runtime_secs=self.time_limit,
            seed=self.random_state,
            sort_metric="AUC",
        )
        aml.train(x=list(X.columns), y="target", training_frame=h2o_df)
        self.model_ = aml.leader
        return self

    def predict_proba(self, X):
        import h2o
        h2o_df = h2o.H2OFrame(X)
        preds = self.model_.predict(h2o_df)
        return preds["p1"].as_data_frame().values.flatten()


class AutoGluonWrapper(BaseAutoMLWrapper):
    name = "AutoGluon"

    def __init__(self, time_limit: int = 600, random_state: int = 42,
                 output_dir: str = "outputs/autogluon", use_gpu: bool = False,
                 dynamic_stacking: bool = False, eval_metric: str = "roc_auc",
                 memory_budget_gb: float = 0):
        super().__init__(time_limit, random_state, use_gpu)
        self.output_dir = Path(output_dir)
        # dynamic_stacking 很耗时，常导致在 time_limit 内无法完成 full fit；
        # 默认关闭以保证在预算内跑完；如需更优堆叠，请加大 time_limit 并设为 True。
        self.dynamic_stacking = dynamic_stacking
        # 评估指标：默认 accuracy 在不平衡数据上会让 AutoGluon 学到"全判多数类"的
        # 退化模型（>90% 样本概率为 0，AUC 极低）。二分类默认 roc_auc 优化排序能力。
        self.eval_metric = eval_metric
        # 内存预算（GB，0=不限，按现状跑）；仅在 use_gpu=True 时生效
        self.memory_budget_gb = float(memory_budget_gb or 0)
        self.tier_used_ = None

    def fit(self, X, y):
        try:
            from autogluon.tabular import TabularDataset, TabularPredictor
        except ImportError:
            raise ImportError("AutoGluon 未安装，请执行：pip install autogluon")

        df = X.copy()
        df["target"] = y.values
        train_data = TabularDataset(df)
        predictor = TabularPredictor(
            label="target",
            path=str(self.output_dir),
            problem_type="binary",
            eval_metric=self.eval_metric,
        )

        presets = "good_quality"
        fit_kwargs = {}
        hyperparameters = "default"
        if self.use_gpu:
            gpu_params = {
                "GBM": [{"device": "gpu", "extra_trees": False, "n_estimators": 10000, "learning_rate": 0.05}],
                "CAT": {"task_type": "GPU", "iterations": 10000, "learning_rate": 0.05},
                "XGB": {"tree_method": "gpu_hist", "n_estimators": 10000, "learning_rate": 0.05},
            }
            # 按内存预算选档（0=不限 → L1 满配，即原行为）
            tier = "L1_GPU_FULL" if self.memory_budget_gb <= 0 else select_ag_tier(
                len(df), X.shape[1], self.memory_budget_gb)
            self.tier_used_ = tier
            from loguru import logger
            if tier == "L1_GPU_FULL":
                hyperparameters = gpu_params
            elif tier == "L2_GPU_LIGHT":
                hyperparameters = gpu_params
                fit_kwargs.update(num_bag_folds=0, num_stack_levels=0)
            elif tier == "L3_GPU_MIN":
                hyperparameters = {k: v for k, v in gpu_params.items() if k in ("GBM", "CAT")}
                presets = "medium_quality"
                fit_kwargs.update(num_bag_folds=0, num_stack_levels=0)
            else:  # L4_CPU：预算内任何 GPU 档都超 → 回退 CPU（不启用 GPU 参数）
                hyperparameters = "default"
            est = estimate_ag_memory_gb(len(df), X.shape[1], tier)
            logger.info(f"AutoGluon 内存档位：{tier}（预算 {self.memory_budget_gb or '不限'}GB，"
                        f"预估峰值 ≈{est:.1f}GB）")

        self.model_ = predictor.fit(
            train_data=train_data,
            time_limit=self.time_limit,
            presets=presets,
            hyperparameters=hyperparameters,
            dynamic_stacking=self.dynamic_stacking,
            **fit_kwargs,
        )
        return self

    def predict_proba(self, X):
        preds = self.model_.predict_proba(X)
        return preds[1].values


from core.feature_utils import sanitize_feature_names


class FLAMLWrapper(BaseAutoMLWrapper):
    name = "FLAML"

    def __init__(self, time_limit: int = 600, random_state: int = 42, use_gpu: bool = False):
        super().__init__(time_limit, random_state, use_gpu)
        self._feature_name_map: dict = {}

    def fit(self, X, y):
        try:
            from flaml import AutoML
        except ImportError:
            raise ImportError("FLAML 未安装，请执行：pip install flaml")

        X = X.copy()
        sanitized = sanitize_feature_names(X.columns)
        self._feature_name_map = dict(zip(sanitized, X.columns))
        X.columns = sanitized

        automl = AutoML()
        fit_kwargs = {
            "X_train": X,
            "y_train": y,
            "task": "classification",
            "time_budget": self.time_limit,
            "metric": "roc_auc",
            "seed": self.random_state,
        }
        # 注意：FLAML 2.x 的 AutoML.fit() 不接受 gpu_id 参数（会被透传给 LightGBM 报错）；
        # 且 Windows 上 LightGBM 官方 wheel 仅 CPU 版，FLAML 无法使用 GPU，故不再传 gpu_id。
        if self.use_gpu:
            warnings.warn("FLAML 在 Windows 上仅支持 CPU（LightGBM 官方 wheel 无 GPU），已按 CPU 运行")
        automl.fit(**fit_kwargs)
        self.model_ = automl
        return self

    def predict_proba(self, X):
        X = X.copy()
        X.columns = sanitize_feature_names(X.columns)
        return self.model_.predict_proba(X)[:, 1]


AUTOML_REGISTRY = {
    "H2O": H2OAutoMLWrapper,
    "AutoGluon": AutoGluonWrapper,
    "FLAML": FLAMLWrapper,
}


# ---------------------------------------------------------------------------
# AutoGluon 内存预算：估算公式 + 4 档配置阶梯（v2.1 新增）
#
# 内存大头是结构性的：bag 折数 × GPU 模型类型数 × 每份 CUDA 上下文，
# 与模型池无关。阶梯只在 bagging/stacking/并发上降配，不动模型池。
# 常数经本机实测校准（L1 实测峰值 23.19GB，CPU 实测 6.26GB）。
# ---------------------------------------------------------------------------

AG_BASE_GB = 4.0      # AutoGluon 本体 + Python 进程基线
AG_CTX_GB = 1.0       # 每份 (bag折 × GPU模型类型) CUDA 上下文
AG_SAFETY = 1.2       # 估算安全余量（实测噪声 ±30~50%）
AG_CPU_PEAK_GB = 6.5  # CPU 模式实测峰值（全类型 default）
AG_GPU_TYPES = ("GBM", "CAT", "XGB")

# 阶梯：L1 满配 / L2 轻配(无bagging) / L3 极简(无bagging+仅两类GPU) / L4 CPU 保底
AG_TIER_LADDER = ["L1_GPU_FULL", "L2_GPU_LIGHT", "L3_GPU_MIN"]


def estimate_ag_memory_gb(n_rows: int, n_cols: int, tier: str) -> float:
    """估算 AutoGluon 指定档位的峰值内存（GB，含安全余量）。"""
    data_gb = n_rows * n_cols * 4 / 1e9  # float32 数据体量
    if tier == "L1_GPU_FULL":
        folds, types = 5, 3
    elif tier == "L2_GPU_LIGHT":
        folds, types = 1, 3
    elif tier == "L3_GPU_MIN":
        folds, types = 1, 2
    else:  # L4_CPU
        return AG_CPU_PEAK_GB
    return (AG_BASE_GB + folds * types * AG_CTX_GB + data_gb * folds * 2) * AG_SAFETY


def select_ag_tier(n_rows: int, n_cols: int, budget_gb: float) -> str:
    """按内存预算选择阶梯档位：返回预算内最高档；都超预算则回退 CPU。"""
    for tier in AG_TIER_LADDER:
        if estimate_ag_memory_gb(n_rows, n_cols, tier) <= budget_gb:
            return tier
    return "L4_CPU"


class AutoMLRunner:
    def __init__(self, name: str, params: dict = None):
        if name not in AUTOML_REGISTRY:
            raise ValueError(f"未知 AutoML：{name}")
        self.name = name
        self.params = params or {}
        self.wrapper_: Optional[BaseAutoMLWrapper] = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        cls = AUTOML_REGISTRY[self.name]
        self.wrapper_ = cls(**self.params)
        self.wrapper_.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.wrapper_.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.wrapper_.predict(X)
