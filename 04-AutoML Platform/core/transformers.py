"""Transformer 模块：FT-Transformer、TabTransformer、SAINT。

使用 PyTorch 实现简化版表格 Transformer，支持分类与连续特征。
"""

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


def get_transformer_device_info() -> str:
    """返回当前 PyTorch 可用的计算设备信息。"""
    if torch.cuda.is_available():
        return f"CUDA 可用：{torch.cuda.get_device_name(0)}"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "MPS 可用（Apple Silicon）"
    return "CPU 模式（未检测到可用的 GPU）"


class BaseTabularTransformer(BaseEstimator, ClassifierMixin):
    """所有表格 Transformer 的基类。"""

    def __init__(
        self,
        num_epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        dropout: float = 0.1,
        dim: int = 32,
        depth: int = 4,
        heads: int = 8,
        random_state: int = 42,
        device: str = None,
        class_weight: bool = True,
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.class_weight = class_weight
        self.le_ = LabelEncoder()
        self.is_fitted_ = False

    def _prepare_data(self, X: pd.DataFrame, y: pd.Series = None, fit: bool = False):
        # 分类列与连续列
        self.cat_cols_ = [c for c in X.columns if pd.api.types.is_integer_dtype(X[c]) and X[c].nunique() < 50]
        self.num_cols_ = [c for c in X.columns if c not in self.cat_cols_]

        X_num = X[self.num_cols_].astype(float).values
        X_cat = X[self.cat_cols_].astype(int).values if self.cat_cols_ else np.zeros((len(X), 0))

        if fit:
            self.num_dim_ = X_num.shape[1]
            self.cat_dims_ = [int(X[c].max()) + 1 for c in self.cat_cols_] if self.cat_cols_ else []

        if y is not None:
            if fit:
                y_enc = self.le_.fit_transform(y)
            else:
                y_enc = self.le_.transform(y)
            y_tensor = torch.LongTensor(y_enc)
        else:
            y_tensor = None

        return torch.FloatTensor(X_num), torch.LongTensor(X_cat), y_tensor

    def _train_epoch(self, model, loader, optimizer, criterion):
        model.train()
        total_loss = 0
        for xb_num, xb_cat, yb in loader:
            xb_num, xb_cat, yb = xb_num.to(self.device), xb_cat.to(self.device), yb.to(self.device)
            optimizer.zero_grad()
            out = model(xb_num, xb_cat)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        torch.manual_seed(self.random_state)
        X_num, X_cat, y_tensor = self._prepare_data(X, y, fit=True)

        dataset = TensorDataset(X_num, X_cat, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_ = self._build_model()
        self.model_.to(self.device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # 类别加权损失：缓解不平衡数据下模型"全猜多数类"（sensitivity/f1/ppv=0）
        if self.class_weight:
            classes, counts = np.unique(y_tensor.cpu().numpy(), return_counts=True)
            # 权重 = N / (n_classes * count_c)，少数类权重更高
            w = torch.tensor(
                len(y_tensor) / (len(classes) * counts.astype(float)),
                dtype=torch.float, device=self.device,
            )
            criterion = nn.CrossEntropyLoss(weight=w)
        else:
            criterion = nn.CrossEntropyLoss()

        for epoch in range(self.num_epochs):
            self._train_epoch(self.model_, loader, optimizer, criterion)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame):
        if not self.is_fitted_:
            raise ValueError("模型未拟合")
        X_num, X_cat, _ = self._prepare_data(X, fit=False)
        self.model_.eval()
        probs = []
        with torch.no_grad():
            # 分批前向：SAINT 的样本间注意力把 batch 当序列长度，
            # 显存随批内样本数平方增长；整集一次前向会 OOM
            # （1.3~1.7 万行的验证集约需 170GB），按训练 batch_size 分块。
            bs = max(int(self.batch_size), 1)
            for i in range(0, len(X_num), bs):
                out = self.model_(X_num[i:i + bs].to(self.device),
                                  X_cat[i:i + bs].to(self.device))
                probs.append(torch.softmax(out, dim=1).cpu().numpy())
        return np.concatenate(probs, axis=0)

    def predict(self, X: pd.DataFrame):
        return self.le_.inverse_transform(np.argmax(self.predict_proba(X), axis=1))

    def _build_model(self):
        raise NotImplementedError


# ------------------------ FT-Transformer ------------------------

class FTTransformer(nn.Module):
    """FT-Transformer：连续特征线性投影 + 分类特征 Embedding + Transformer Encoder。"""

    def __init__(self, num_dim, cat_dims, dim, depth, heads, dropout, num_classes):
        super().__init__()
        self.num_embeddings = nn.ModuleList([nn.Linear(1, dim) for _ in range(num_dim)])
        self.cat_embeddings = nn.ModuleList([nn.Embedding(d, dim) for d in cat_dims])

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x_num, x_cat):
        bsz = x_num.size(0)
        tokens = []
        for i, emb in enumerate(self.num_embeddings):
            tokens.append(emb(x_num[:, i:i + 1]))
        for i, emb in enumerate(self.cat_embeddings):
            tokens.append(emb(x_cat[:, i]))
        x = torch.stack(tokens, dim=1)
        cls = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        return self.mlp(x[:, 0, :])


class FTTransformerClassifier(BaseTabularTransformer):
    name = "FT-Transformer"

    def _build_model(self):
        return FTTransformer(
            num_dim=self.num_dim_,
            cat_dims=self.cat_dims_,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            dropout=self.dropout,
            num_classes=len(self.le_.classes_),
        )


# ------------------------ TabTransformer ------------------------

class TabTransformer(nn.Module):
    """TabTransformer：仅对分类特征应用 Transformer，连续特征直接 MLP。"""

    def __init__(self, num_dim, cat_dims, dim, depth, heads, dropout, num_classes):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([nn.Embedding(d, dim) for d in cat_dims])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.num_mlp = nn.Sequential(
            nn.Linear(num_dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        total_dim = dim + (len(cat_dims) * dim if cat_dims else 0)
        self.head = nn.Sequential(
            nn.Linear(total_dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x_num, x_cat):
        cat_tokens = []
        for i, emb in enumerate(self.cat_embeddings):
            cat_tokens.append(emb(x_cat[:, i]))
        if cat_tokens:
            cat_x = torch.stack(cat_tokens, dim=1)
            cat_x = self.transformer(cat_x)
            cat_x = cat_x.reshape(cat_x.size(0), -1)
        else:
            cat_x = torch.zeros(x_num.size(0), 0).to(x_num.device)
        num_x = self.num_mlp(x_num)
        x = torch.cat([num_x, cat_x], dim=1)
        return self.head(x)


class TabTransformerClassifier(BaseTabularTransformer):
    name = "TabTransformer"

    def _build_model(self):
        return TabTransformer(
            num_dim=self.num_dim_,
            cat_dims=self.cat_dims_,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            dropout=self.dropout,
            num_classes=len(self.le_.classes_),
        )


# ------------------------ SAINT ------------------------

class SAINT(nn.Module):
    """SAINT：特征自注意力 + 样本间注意力（intersample attention）。

    真正的 SAINT 与 FT-Transformer 的关键区别在于多了一个
    **样本间注意力块**：先对每个样本的特征做自注意力（feature-wise），
    再对每个特征在 batch 内不同样本之间做注意力（sample-wise）。
    """

    def __init__(self, num_dim, cat_dims, dim, depth, heads, dropout, num_classes):
        super().__init__()
        self.num_embeddings = nn.ModuleList([nn.Linear(1, dim) for _ in range(num_dim)])
        self.cat_embeddings = nn.ModuleList([nn.Embedding(d, dim) for d in cat_dims])
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # 1) 特征自注意力（与 FT-Transformer 类似，对每个样本的特征序列做注意力）
        self.self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=dim * 4,
                dropout=dropout, batch_first=True,
            ),
            num_layers=depth,
        )
        # 2) 样本间注意力（intersample：对每个特征在 batch 样本维度上做注意力）
        self.intersample_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=dim * 4,
                dropout=dropout, batch_first=True,
            ),
            num_layers=1,  # 样本间注意力通常 1 层即可，避免过深导致过拟合/过慢
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x_num, x_cat):
        bsz = x_num.size(0)
        tokens = []
        for i, emb in enumerate(self.num_embeddings):
            tokens.append(emb(x_num[:, i:i + 1]))
        for i, emb in enumerate(self.cat_embeddings):
            tokens.append(emb(x_cat[:, i]))
        x = torch.stack(tokens, dim=1)  # (bsz, n_feat, dim)
        cls = self.cls_token.expand(bsz, -1, -1)  # (bsz, 1, dim)
        x = torch.cat([cls, x], dim=1)  # (bsz, n_feat+1, dim)

        # 1) 特征自注意力：对每个样本，特征之间相互注意
        x = self.self_attn(x)  # (bsz, n_feat+1, dim)

        # 2) 样本间注意力：把 (bsz, n_feat+1, dim) 换成 (n_feat+1, bsz, dim)，
        #    让每个特征在 batch 内不同样本之间做注意力
        x = x.permute(1, 0, 2)  # (n_feat+1, bsz, dim)
        x = self.intersample_attn(x)
        x = x.permute(1, 0, 2)  # (bsz, n_feat+1, dim)

        return self.mlp(x[:, 0, :])


class SAINTClassifier(BaseTabularTransformer):
    name = "SAINT"

    def _build_model(self):
        return SAINT(
            num_dim=self.num_dim_,
            cat_dims=self.cat_dims_,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            dropout=self.dropout,
            num_classes=len(self.le_.classes_),
        )


class sTFT(nn.Module):
    """sTFT（simplified Feature Tokenizer Transformer）：
    简化的 FT-Transformer 变体，使用所有特征 token 的均值池化进行分类。
    """

    def __init__(self, num_dim, cat_dims, dim, depth, heads, dropout, num_classes):
        super().__init__()
        self.num_embeddings = nn.ModuleList([nn.Linear(1, dim) for _ in range(num_dim)])
        self.cat_embeddings = nn.ModuleList([nn.Embedding(d, dim) for d in cat_dims])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x_num, x_cat):
        tokens = []
        for i, emb in enumerate(self.num_embeddings):
            tokens.append(emb(x_num[:, i:i + 1]).unsqueeze(1))
        for i, emb in enumerate(self.cat_embeddings):
            tokens.append(emb(x_cat[:, i]).unsqueeze(1))
        x = torch.cat(tokens, dim=1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.head(x)


class sTFTClassifier(BaseTabularTransformer):
    name = "sTFT"

    def _build_model(self):
        return sTFT(
            num_dim=self.num_dim_,
            cat_dims=self.cat_dims_,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            dropout=self.dropout,
            num_classes=len(self.le_.classes_),
        )


TRANSFORMER_REGISTRY = {
    "FT-Transformer": FTTransformerClassifier,
    "TabTransformer": TabTransformerClassifier,
    "SAINT": SAINTClassifier,
    "sTFT": sTFTClassifier,
}

# v2.1 新增：新一代表格深度学习模型（pytabkit）与表格基础模型（TabPFN/TabICL）。
# 包装类在 core/dl_models.py，依赖惰性导入，未安装对应包时不影响本模块加载。
from core.dl_models import DL_WRAPPER_REGISTRY

TRANSFORMER_REGISTRY.update(DL_WRAPPER_REGISTRY)


def train_greedy_soup(cls, params: dict, X_train, y_train, X_val, y_val,
                      n_soups: int = 3, random_state: int = 42):
    """Greedy Soup（v2.1 新增）：训练 n_soups 个不同随机种子的实例，
    按验证集 AUC 从高到低贪心尝试加入"权重平均汤"，AUC 不降才纳入。

    仅适用于本项目的 torch 原生模型（BaseTabularTransformer 子类）。
    权重平均是模型集成/正则化的零成本手段（Wortsman et al., 2022）。

    返回 (soup_model, details)：soup_model 已加载平均权重并可直接 predict_proba。
    """
    from sklearn.metrics import roc_auc_score

    def _build_from_sd(model_cls, sd):
        probe = model_cls(random_state=random_state, **params)
        probe._prepare_data(X_train, y_train, fit=True)
        probe.model_ = probe._build_model()
        probe.model_.load_state_dict(sd)
        probe.model_.to(probe.device)
        probe.is_fitted_ = True
        return probe

    candidates = []
    for i in range(n_soups):
        m = cls(random_state=random_state + i, **params)
        m.fit(X_train, y_train)
        auc = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
        candidates.append({
            "seed": random_state + i,
            "auc": auc,
            "sd": {k: v.detach().cpu().clone()
                   for k, v in m.model_.state_dict().items()},
        })
    candidates.sort(key=lambda c: c["auc"], reverse=True)

    soup_sd = candidates[0]["sd"]
    soup_members = [candidates[0]["seed"]]
    best_auc = candidates[0]["auc"]
    for cand in candidates[1:]:
        n = len(soup_members)
        new_sd = {k: (soup_sd[k].float() * n + cand["sd"][k].float()) / (n + 1)
                  for k in soup_sd}
        probe = _build_from_sd(cls, new_sd)
        auc = roc_auc_score(y_val, probe.predict_proba(X_val)[:, 1])
        if auc >= best_auc:
            soup_sd = new_sd
            soup_members.append(cand["seed"])
            best_auc = auc

    final = _build_from_sd(cls, soup_sd)
    details = {
        "soup_members": soup_members,
        "val_auc": round(float(best_auc), 4),
        "candidates": [{"seed": c["seed"], "val_auc": round(float(c["auc"]), 4)}
                       for c in candidates],
    }
    return final, details


class TransformerTrainer:
    """统一入口：训练表格 Transformer 并输出 OOF 预测。"""

    def __init__(self, name: str, params: dict = None, cv_folds: int = 5,
                 random_state: int = 42):
        if name not in TRANSFORMER_REGISTRY:
            raise ValueError(f"未知 Transformer：{name}")
        self.name = name
        self.params = params or {}
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.model_: Optional[BaseTabularTransformer] = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        cls = TRANSFORMER_REGISTRY[self.name]
        self.model_ = cls(random_state=self.random_state, **self.params)
        self.model_.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X)
