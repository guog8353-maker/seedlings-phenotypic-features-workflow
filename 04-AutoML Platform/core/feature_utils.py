"""特征名工具函数。"""

import re


def sanitize_feature_names(columns) -> list:
    """将列名中的 JSON 特殊字符替换为下划线。

    LightGBM/FLAML 不接受 feature name 中含 JSON 特殊字符（空格、逗号、引号、
    括号、冒号等）。此函数用于在训练前清洗列名，并保证重名时追加序号。

    示例：
        "Age, years" -> "Age_years"
        "Creatinine, mg/dL" -> "Creatinine_mg_dL"
        "Baseline TyG-FI29" -> "Baseline_TyG-FI29"
    """

    def _clean(name: str) -> str:
        # 仅保留字母、数字、下划线、点、连字符，其余替换为下划线
        s = re.sub(r"[^a-zA-Z0-9_\-.]+", "_", str(name))
        s = s.strip("_")
        return s if s else "feature"

    seen = {}
    new_cols = []
    for c in columns:
        base = _clean(c)
        new_c = base
        i = 1
        while new_c in seen:
            new_c = f"{base}_{i}"
            i += 1
        seen[new_c] = c
        new_cols.append(new_c)
    return new_cols
