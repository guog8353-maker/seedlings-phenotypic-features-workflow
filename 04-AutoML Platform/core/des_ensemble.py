"""动态集成选择（DES，v2.1 新增）。

KNORA-E / KNORA-U 的自研实现：对每个样本，在其**原始特征空间**的 k 近邻上
评估各基模型的"局部胜任力"，只让在邻域内表现好的模型参与该样本的预测。

- KNORA-E：要求模型在 k 个邻居上全部预测正确；无满足者时退化为 KNORA-U；
- KNORA-U：要求模型在 k 个邻居上至少一个预测正确；再无则使用全部模型。

说明：
- 能力评估基于各基模型的 OOF 概率（out-of-fold，无泄漏），输出亦为 OOF 平均；
- 邻域计算需要原始特征矩阵 X_full（由 ensemble_tab 在运行时注入）；
- DESlib 0.3.7 与 sklearn>=1.7 不兼容（_validate_data 已移除），故自研，
  实现为全向量化（无逐样本 Python 循环），5 万行秒级完成。
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from core.ensemble import EnsembleMethod


class KNORAEnsemble(EnsembleMethod):
    """KNORA-E/U 动态集成选择。"""

    def __init__(self, k: int = 7, algorithm: str = "KNORA-E", cv: int = 5,
                 random_state: int = 42, X_full=None):
        super().__init__("DES-KNORA")
        self.k = int(k)
        self.algorithm = algorithm
        self.cv = int(cv)
        self.random_state = random_state
        self.X_full = X_full

    # ------------------------------------------------------------------
    def _select_predict(self, X_meta_q: np.ndarray, X_full_q: np.ndarray,
                        X_ref: np.ndarray, correct_ref: np.ndarray) -> np.ndarray:
        """对查询样本动态选择基模型并输出平均概率（向量化）。

        X_meta_q: (n, M) 查询样本各基模型 OOF 概率；
        X_full_q: (n, p) 查询样本原始特征（用于找邻居）；
        X_ref:    (n_ref, p) 参考集原始特征；
        correct_ref: (n_ref, M) 各基模型在参考集上是否预测正确。
        """
        k = max(1, min(self.k, len(X_ref)))
        nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        nn.fit(X_ref)
        _, idx = nn.kneighbors(X_full_q)

        c = correct_ref[idx]            # (n, k, M) 布尔
        sel_all = c.all(axis=1)         # (n, M)：邻域全对
        sel_any = c.any(axis=1)         # (n, M)：邻域至少一个对

        if self.algorithm.upper().endswith("E"):
            mask = sel_all
            empty = ~mask.any(axis=1)
            mask[empty] = sel_any[empty]   # KNORA-E 找不到全能模型 → 退化 KNORA-U
        else:
            mask = sel_any
        empty = ~mask.any(axis=1)
        mask[empty] = True                 # 兜底：用全部模型

        return (X_meta_q * mask).sum(axis=1) / mask.sum(axis=1)

    # ------------------------------------------------------------------
    def fit(self, X_meta: np.ndarray, y: np.ndarray, model_names: List[str] = None):
        if self.X_full is None:
            raise ValueError(
                "DES-KNORA 需要原始特征矩阵 X_full 用于邻域计算"
                "（请在集成选项卡中保证可加载建模数据）"
            )
        X_meta = np.asarray(X_meta)
        X_full = np.asarray(self.X_full)
        y = np.asarray(y)
        if len(X_full) != len(X_meta):
            raise ValueError(
                f"X_full 行数({len(X_full)})与 OOF 行数({len(X_meta)})不一致"
            )

        correct_full = (X_meta >= 0.5) == y[:, None]

        # 用 CV 生成诚实的 OOF 预测（与 Stacking 的 oof_proba_ 语义一致）
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True,
                              random_state=self.random_state)
        oof = np.zeros(len(y))
        for tr, te in skf.split(X_full, y):
            oof[te] = self._select_predict(
                X_meta[te], X_full[te], X_full[tr], correct_full[tr]
            )
        self.oof_proba_ = oof

        # 保存全量参考集，供 predict_proba 使用
        self._ref_X = X_full
        self._ref_correct = correct_full
        return self

    def predict_proba(self, X_meta: np.ndarray, X_full=None) -> np.ndarray:
        X_ref = self._ref_X if X_full is None else np.asarray(X_full)
        return self._select_predict(
            np.asarray(X_meta), X_ref, self._ref_X, self._ref_correct
        )


DES_REGISTRY = {
    "DES-KNORA": KNORAEnsemble,
}

# 注册进 core.ensemble 的方法池（双路径兜底：无论本模块还是
# core.ensemble 先被导入，DES-KNORA 最终都会出现在注册表中）。
from core.ensemble import ENSEMBLE_REGISTRY

ENSEMBLE_REGISTRY.update(DES_REGISTRY)
