"""数据管理模块：导入、预览、变量类型标注、训练集划分。"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".sas7bdat", ".sav", ".dta"}

VARIABLE_ROLES = [
    "unknown",
    "id",
    "outcome",
    "time",          # survival time
    "weight",        # sampling weight
    "strata",        # strata
    "psu",           # PSU
    "continuous",
    "binary",
    "ordinal",
    "nominal",
    "drop",
]


class DataManager:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.file_path: Optional[str] = None
        self.roles: Dict[str, str] = {}
        self.sample_weights: Optional[pd.Series] = None

    @staticmethod
    def _read_csv_with_encoding(path: str) -> pd.DataFrame:
        for enc in ["utf-8", "gbk", "gb2312", "gb18030", "latin1"]:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        # fallback
        return pd.read_csv(path, encoding="utf-8", errors="replace")

    def load(self, path: str) -> pd.DataFrame:
        path_obj = Path(path)
        ext = path_obj.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件格式：{ext}，支持：{SUPPORTED_EXTENSIONS}")

        if ext == ".csv":
            df = self._read_csv_with_encoding(path)
        elif ext in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        elif ext == ".sas7bdat":
            df = pd.read_sas(path)
        elif ext == ".sav":
            df = pd.read_spss(path)
        elif ext == ".dta":
            df = pd.read_stata(path)
        else:
            raise ValueError(f"未识别的扩展名：{ext}")

        self.df = df
        self.file_path = path
        # 默认所有变量为 unknown
        self.roles = {c: "unknown" for c in df.columns}
        # 自动识别常见 ID / weight / strata / psu / time 列
        self._auto_detect_roles()
        return df.copy()

    def _auto_detect_roles(self):
        """基于列名启发式识别抽样设计变量与常见变量。"""
        if self.df is None:
            return
        for col in self.df.columns:
            name = col.lower()
            if name in {"wtmec2yr", "mec_wt", "sample_weight"}:
                self.roles[col] = "weight"
            elif name in {"sdmvstra", "strata"}:
                self.roles[col] = "strata"
            elif name in {"sdmvpsu", "psu"}:
                self.roles[col] = "psu"
            elif name in {"time", "permth_int", "follow_up", "survival_time"}:
                self.roles[col] = "time"
            elif name in {"id", "seqn", "participant_id"}:
                self.roles[col] = "id"

    def get_preview(self, n_rows: int = 100) -> pd.DataFrame:
        if self.df is None:
            return pd.DataFrame()
        return self.df.head(n_rows)

    def get_summary(self) -> pd.DataFrame:
        if self.df is None:
            return pd.DataFrame()
        summary = []
        for col in self.df.columns:
            s = self.df[col]
            summary.append({
                "variable": col,
                "dtype": str(s.dtype),
                "n_missing": int(s.isna().sum()),
                "missing_pct": round(s.isna().mean() * 100, 2),
                "n_unique": s.nunique(dropna=True),
                "role": self.roles.get(col, "unknown"),
            })
        return pd.DataFrame(summary)

    def set_role(self, column: str, role: str):
        if role not in VARIABLE_ROLES:
            raise ValueError(f"未知角色：{role}，可选：{VARIABLE_ROLES}")
        self.roles[column] = role

    def get_columns_by_role(self, roles: List[str]) -> List[str]:
        return [c for c, r in self.roles.items() if r in roles]

    def infer_types(self) -> Dict[str, str]:
        """基于变量角色与数据特征推断每列的数据类型。"""
        if self.df is None:
            return {}
        types = {}
        for col in self.df.columns:
            role = self.roles.get(col, "unknown")
            if role in {"continuous", "time", "weight"}:
                types[col] = "numeric"
            elif role in {"binary", "ordinal", "nominal"}:
                types[col] = "categorical"
            elif role == "outcome":
                n_uniq = self.df[col].nunique(dropna=True)
                types[col] = "binary" if n_uniq <= 2 else "continuous"
            else:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    types[col] = "numeric"
                else:
                    types[col] = "categorical"
        return types

    def get_feature_columns(self, include_time: bool = False) -> List[str]:
        """返回用于建模的特征列（排除 id/outcome/time/weight/strata/psu/drop）。"""
        exclude = {"id", "outcome", "weight", "strata", "psu", "drop"}
        if not include_time:
            exclude.add("time")
        return [c for c, r in self.roles.items() if r not in exclude]

    def get_sampling_design_cols(self) -> Dict[str, Optional[str]]:
        """返回抽样设计列名（复杂抽样调查，如已标注）。"""
        by_role = self.get_columns_by_role
        return {
            "weight": by_role(["weight"])[0] if by_role(["weight"]) else None,
            "strata": by_role(["strata"])[0] if by_role(["strata"]) else None,
            "psu": by_role(["psu"])[0] if by_role(["psu"]) else None,
        }

    def train_test_split(self, test_size: float = 0.2, stratify: bool = True,
                         random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        from sklearn.model_selection import train_test_split
        if self.df is None:
            raise ValueError("未加载数据")
        y_col = self.get_columns_by_role(["outcome"])
        y = self.df[y_col[0]] if y_col else None
        strat = y if stratify and y is not None and y.nunique() <= 10 else None
        train, test = train_test_split(
            self.df, test_size=test_size, stratify=strat, random_state=random_state
        )
        return train.reset_index(drop=True), test.reset_index(drop=True)
