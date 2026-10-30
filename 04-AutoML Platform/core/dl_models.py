"""新一代表格深度学习模型包装层（v2.1 新增）。

统一把 pytabkit（TabM/RealMLP/RealTabR/XRFM）与表格基础模型
（TabPFN v2 / TabICL）包装为与 TransformerTrainer 兼容的接口：
    cls(random_state=..., **params) -> .fit(X, y) -> .predict_proba(X)[:, 1]

GUI 传入的 Transformer 通用超参（num_epochs/dim/depth/heads/dropout/lr/
weight_decay/batch_size）会被有选择地映射，不适用的静默忽略——这些模型
使用作者调好的默认参数（D/TD 后缀即 tuned defaults），通常无需手调。
"""

import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _auto_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _patch_faiss_gpu_shim() -> None:
    """faiss-cpu 环境下，用 torch 在 GPU 上实现 pytabkit TabR 需要的
    faiss GPU 索引接口（GpuIndexFlatL2 等）。

    pytabkit 仅将 faiss 用于最近邻检索（L2 topk），torch.cdist+topk
    在 GPU 上等价且更快（pytabkit 源码注释亦说明小数据集 naive 更快）。
    幂等：已存在 GPU faiss（如 Linux faiss-gpu）时不做任何事。
    """
    import faiss
    import torch

    if hasattr(faiss, "GpuIndexFlatConfig"):
        return

    class GpuIndexFlatConfig:  # noqa: D401 - 仅模拟 faiss 的配置对象
        def __init__(self):
            self.device = 0

    class StandardGpuResources:  # noqa: D401 - 占位
        pass

    class GpuIndexFlatL2:
        def __init__(self, _res, d, _cfg=None):
            self.d = d
            self._k = None

        def reset(self):
            self._k = None

        def add(self, k):
            k = k.detach()
            self._k = k if self._k is None else torch.cat([self._k, k], dim=0)

        def search(self, q, n):
            n = min(n, len(self._k))
            # faiss 返回平方 L2；cdist**2 保持相同语义（且 topk 排序一致）
            dists = torch.cdist(q, self._k) ** 2
            return torch.topk(dists, n, largest=False)

    faiss.GpuIndexFlatConfig = GpuIndexFlatConfig
    faiss.StandardGpuResources = StandardGpuResources
    faiss.GpuIndexFlatL2 = GpuIndexFlatL2


class _BaseDLWrapper:
    """公共基类：透传 fit/predict_proba，忽略不适用的 GUI 超参。"""

    name = ""
    # 标记：非本项目的 torch 原生模型，transformer_tab 保存 .pt 时应跳过
    is_external_wrapper = True

    def __init__(self, random_state: int = 42, device: str = None,
                 num_epochs: int = None, **ignored):
        self.random_state = random_state
        self.device = device or _auto_device()
        self.num_epochs = num_epochs
        if ignored:
            # dim/depth/heads/dropout/lr/weight_decay/batch_size 等不适用
            pass
        self.model_ = None

    def _build(self):
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model_ = self._build()
        self.model_.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.model_.predict_proba(X)
        return np.asarray(proba)


class TabMWrapper(_BaseDLWrapper):
    """TabM（ICLR 2025）：参数高效集成的 MLP，性能对标 GBDT。"""

    name = "TabM"

    def _build(self):
        from pytabkit import TabM_D_Classifier
        params: Dict[str, Any] = {"device": self.device, "random_state": self.random_state}
        if self.num_epochs:
            params["n_epochs"] = self.num_epochs
        return TabM_D_Classifier(**params)


class RealMLPWrapper(_BaseDLWrapper):
    """RealMLP（Holzmüller 2024）：精心调参的 MLP，免调参强基线。"""

    name = "RealMLP"

    def _build(self):
        from pytabkit import RealMLP_TD_Classifier
        params: Dict[str, Any] = {"device": self.device, "random_state": self.random_state}
        if self.num_epochs:
            params["n_epochs"] = self.num_epochs
        return RealMLP_TD_Classifier(**params)


class RealTabRWrapper(_BaseDLWrapper):
    """RealTabR（TabR 系，ICLR 2024）：检索增强表格深度学习。

    faiss 检索的 GPU 路径需要 faiss-gpu（Windows pip 无此轮子），
    因此在 faiss-cpu 环境下先用 torch shim 补齐 GPU 索引接口，
    使模型可正常在 CUDA 上训练（比 CPU 快一个数量级以上）。
    """

    name = "RealTabR"

    def _build(self):
        _patch_faiss_gpu_shim()
        from pytabkit import RealTabR_D_Classifier
        params: Dict[str, Any] = {"device": self.device, "random_state": self.random_state}
        if self.num_epochs:
            params["n_epochs"] = self.num_epochs
        return RealTabR_D_Classifier(**params)


class XRFMWrapper(_BaseDLWrapper):
    """xRFM（2025）：极限随机特征机，检索/核方法路线的新 SOTA 候选。"""

    name = "XRFM"

    def _build(self):
        from pytabkit import XRFM_D_Classifier
        params: Dict[str, Any] = {"device": self.device, "random_state": self.random_state}
        return XRFM_D_Classifier(**params)


class TabPFNv2Wrapper(_BaseDLWrapper):
    """TabPFN（Nature 2025，当前使用 v3 权重）：表格基础模型，免训练 in-context 预测。

    官方限制 ≤10000 行 / 500 特征；超出时自动随机子采样训练集
    （random_state 固定，可复现）。权重优先级：本地 v3（models_ckpt/）
    > 自动下载 v3（需授权）> v2 检查点（免授权回退）。
    """

    name = "TabPFN"
    MAX_ROWS = 10000

    # 本地权重优先：v3（用户手动下载）> 自动下载 v3（需授权）> v2（免授权回退）
    _LOCAL_V3 = "tabpfn-v3-classifier-v3_20260417_binary.ckpt"

    def _build(self):
        from pathlib import Path

        from tabpfn import TabPFNClassifier
        local = Path(__file__).resolve().parent.parent / "models_ckpt" / self._LOCAL_V3
        if local.exists():
            return TabPFNClassifier(device=self.device, model_path=str(local))
        try:
            return TabPFNClassifier(device=self.device, model_path="auto")
        except Exception:
            # v3 门控未授权或网络不通时回退免授权的 v2 检查点
            return TabPFNClassifier(device=self.device,
                                    model_path="tabpfn-v2-classifier.ckpt")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if len(X) > self.MAX_ROWS:
            warnings.warn(
                f"TabPFN 限制 {self.MAX_ROWS} 行，已从 {len(X)} 行随机子采样"
            )
            idx = np.random.RandomState(self.random_state).choice(
                len(X), self.MAX_ROWS, replace=False
            )
            X = X.iloc[idx]
            y = y.iloc[idx]
        return super().fit(X, y)


class TabICLWrapper(_BaseDLWrapper):
    """TabICL（2025）：面向大数据的表格基础模型（in-context learning）。

    权重默认从 HuggingFace 自动下载；若网络不通，可手动下载放到
    项目根 models_ckpt/ 目录（优先使用本地副本）。
    """

    name = "TabICL"

    def _build(self):
        from pathlib import Path

        from tabicl import TabICLClassifier
        local = (Path(__file__).resolve().parent.parent
                 / "models_ckpt" / "tabicl-classifier-v2-20260212.ckpt")
        if local.exists():
            return TabICLClassifier(device=self.device, model_path=str(local))
        return TabICLClassifier(device=self.device)


# 注册到 Transformer 选项卡的模型池（键名即界面显示名）
DL_WRAPPER_REGISTRY = {
    "TabM": TabMWrapper,
    "RealMLP": RealMLPWrapper,
    "RealTabR": RealTabRWrapper,
    "XRFM": XRFMWrapper,
    "TabPFN": TabPFNv2Wrapper,
    "TabICL": TabICLWrapper,
}
