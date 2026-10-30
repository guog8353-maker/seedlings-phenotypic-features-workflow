"""绘图样式与配色管理。

提供集中管理的调色盘，支持：
- 四类模型层（Base / Ensemble / AutoML / Transformer）主色
- 特征选择方法配色
- 单色系渐变（用于 SHAP 排名等）
- 用户自定义主/副色调扩展接口
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# 内置调色盘的 key（自定义调色盘以 custom_ 前缀区分）
_BUILTIN_KEYS = {"default", "colorblind", "warm", "cool", "nature", "science", "lancet"}


DEFAULT_LAYER_COLORS = {
    "Base": "#1f77b4",
    "Ensemble": "#ff7f0e",
    "AutoML": "#2ca02c",
    "Transformer": "#d62728",
}

DEFAULT_METHOD_COLORS = {
    "LASSO": "#e41a1c",
    "Boruta": "#377eb8",
    "RFE": "#4daf4a",
    "MutualInformation": "#984ea3",
    "stepAIC": "#ff7f00",
    "SCAD": "#f781bf",
    "GA": "#a65628",
}


PALETTES = {
    "default": {
        "name": "默认",
        "layer": DEFAULT_LAYER_COLORS,
        "method": DEFAULT_METHOD_COLORS,
        "primary": "#1f77b4",
        "secondary": "#aec7e8",
        "cmap": "Blues",
    },
    "colorblind": {
        "name": "色盲友好",
        "layer": {
            "Base": "#0173B2",
            "Ensemble": "#DE8F05",
            "AutoML": "#029E73",
            "Transformer": "#D55E00",
        },
        "method": {
            "LASSO": "#CC78BC",
            "Boruta": "#0173B2",
            "RFE": "#029E73",
            "MutualInformation": "#CA9161",
            "stepAIC": "#FBAFE4",
            "SCAD": "#949494",
            "GA": "#56B4E9",
        },
        "primary": "#0173B2",
        "secondary": "#8FD4FF",
        "cmap": "cividis",
    },
    "warm": {
        "name": "暖色调",
        "layer": {
            "Base": "#E41A1C",
            "Ensemble": "#FF7F00",
            "AutoML": "#F781BF",
            "Transformer": "#A65628",
        },
        "method": {
            "LASSO": "#E41A1C",
            "Boruta": "#FF7F00",
            "RFE": "#F781BF",
            "MutualInformation": "#A65628",
            "stepAIC": "#984EA3",
            "SCAD": "#DD1C77",
            "GA": "#E41A1C",
        },
        "primary": "#E41A1C",
        "secondary": "#FDBB84",
        "cmap": "Reds",
    },
    "cool": {
        "name": "冷色调",
        "layer": {
            "Base": "#377EB8",
            "Ensemble": "#4DAF4A",
            "AutoML": "#984EA3",
            "Transformer": "#41B6C4",
        },
        "method": {
            "LASSO": "#377EB8",
            "Boruta": "#41B6C4",
            "RFE": "#4DAF4A",
            "MutualInformation": "#984EA3",
            "stepAIC": "#225EA8",
            "SCAD": "#41B6C4",
            "GA": "#0C2C84",
        },
        "primary": "#377EB8",
        "secondary": "#BDD7E7",
        "cmap": "Blues",
    },
    "nature": {
        "name": "Nature 科研",
        "layer": {
            "Base": "#0F4C81",
            "Ensemble": "#E69F00",
            "AutoML": "#009E73",
            "Transformer": "#CC79A7",
        },
        "method": {
            "LASSO": "#0F4C81",
            "Boruta": "#56B4E9",
            "RFE": "#009E73",
            "MutualInformation": "#E69F00",
            "stepAIC": "#0072B2",
            "SCAD": "#CC79A7",
            "GA": "#D55E00",
        },
        "primary": "#0F4C81",
        "secondary": "#A6C8E0",
        "cmap": "YlGnBu",
    },
    "science": {
        "name": "Science 科研",
        "layer": {
            "Base": "#1F77B4",
            "Ensemble": "#FF7F0E",
            "AutoML": "#2CA02C",
            "Transformer": "#D62728",
        },
        "method": {
            "LASSO": "#1F77B4",
            "Boruta": "#FF7F0E",
            "RFE": "#2CA02C",
            "MutualInformation": "#9467BD",
            "stepAIC": "#8C564B",
            "SCAD": "#E377C2",
            "GA": "#7F7F7F",
        },
        "primary": "#1F77B4",
        "secondary": "#AEC7E8",
        "cmap": "viridis",
    },
    "lancet": {
        "name": "期刊经典",
        "layer": {
            "Base": "#00468B",
            "Ensemble": "#ED0000",
            "AutoML": "#42B540",
            "Transformer": "#994F9C",
        },
        "method": {
            "LASSO": "#00468B",
            "Boruta": "#ED0000",
            "RFE": "#42B540",
            "MutualInformation": "#994F9C",
            "stepAIC": "#0099B4",
            "SCAD": "#FDAE61",
            "GA": "#F46D43",
        },
        "primary": "#00468B",
        "secondary": "#B3D1E6",
        "cmap": "RdYlBu",
    },
}


def get_palette(name: str) -> Dict:
    """按名称获取调色盘，未知名称返回 default。"""
    return PALETTES.get(name, PALETTES["default"])


# --------------------------------------------------------------------------- #
# 自定义调色盘的持久化（保存 / 加载 / 删除）
# --------------------------------------------------------------------------- #

def get_custom_palettes() -> Dict[str, Dict]:
    """返回所有自定义调色盘（非内置）。"""
    return {k: v for k, v in PALETTES.items() if k not in _BUILTIN_KEYS}


def save_custom_palettes(filepath) -> None:
    """把自定义调色盘持久化到 JSON 文件。"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(get_custom_palettes(), f, ensure_ascii=False, indent=2)


def load_custom_palettes(filepath) -> int:
    """从 JSON 文件加载自定义调色盘到 PALETTES，返回加载数量。"""
    filepath = Path(filepath)
    if not filepath.exists():
        return 0
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            custom = json.load(f)
        n = 0
        for k, v in custom.items():
            if k not in _BUILTIN_KEYS:
                PALETTES[k] = v
                n += 1
        return n
    except Exception:
        return 0


def delete_custom_palette(key: str, filepath=None) -> bool:
    """删除一个自定义调色盘（内置不可删），并同步持久化文件。"""
    if key in _BUILTIN_KEYS or key not in PALETTES:
        return False
    del PALETTES[key]
    if filepath is not None:
        save_custom_palettes(filepath)
    return True


def list_palettes() -> List[Tuple[str, str]]:
    """返回 [(key, 中文名), ...]。"""
    return [(k, v["name"]) for k, v in PALETTES.items()]


def make_gradient_colors(n: int, primary: str = "#1f77b4", secondary: str = "#aec7e8") -> List[Tuple[float, float, float, float]]:
    """在 primary 与 secondary 之间生成 n 个颜色的渐变（RGBA）。"""
    from matplotlib.colors import to_rgba
    c1 = np.array(to_rgba(primary))
    c2 = np.array(to_rgba(secondary))
    return [tuple(c1 * (1 - i / max(n - 1, 1)) + c2 * (i / max(n - 1, 1))) for i in range(n)]


def _hex_to_rgb(hex_color: str) -> np.ndarray:
    """将 #RRGGBB 转为 [R, G, B]（0-255）。"""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"无效颜色格式：{hex_color}")
    return np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)])


def _rgb_to_hex(rgb: np.ndarray) -> str:
    """将 [R, G, B]（0-255）转为 #RRGGBB。"""
    return "#" + "".join(f"{int(c):02x}" for c in rgb)


def _shift_hue(hex_color: str, angle: float) -> str:
    """在 HSV 空间对颜色进行色相偏移（角度制），返回 HEX。"""
    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
    rgb = _hex_to_rgb(hex_color) / 255.0
    hsv = rgb_to_hsv(rgb)
    hsv[0] = (hsv[0] + angle / 360.0) % 1.0
    new_rgb = hsv_to_rgb(hsv)
    return _rgb_to_hex(np.clip(new_rgb * 255, 0, 255).astype(int))


def _blend(primary: str, secondary: str, alpha: float) -> str:
    """按 alpha 混合主副色。"""
    c1 = _hex_to_rgb(primary)
    c2 = _hex_to_rgb(secondary)
    return _rgb_to_hex(np.clip(c1 * (1 - alpha) + c2 * alpha, 0, 255).astype(int))


def make_custom_palette(primary: str, secondary: str) -> Dict:
    """基于用户指定的主/副色生成一个自定义调色盘。

    主色映射 Base，副色映射 Ensemble；AutoML 与 Transformer 由主色做色相偏移得到，
    使四类层颜色既区分明显又保持视觉协调。
    """
    return {
        "name": f"自定义({primary}/{secondary})",
        "layer": {
            "Base": primary,
            "Ensemble": secondary,
            "AutoML": _shift_hue(primary, 120),
            "Transformer": _shift_hue(primary, 240),
        },
        "method": {
            "LASSO": primary,
            "Boruta": secondary,
            "RFE": _shift_hue(primary, 90),
            "MutualInformation": _shift_hue(primary, 180),
            "stepAIC": _blend(primary, secondary, 0.5),
            "SCAD": _shift_hue(primary, 270),
            "GA": _blend(primary, secondary, 0.25),
        },
        "primary": primary,
        "secondary": secondary,
        "cmap": "Blues",
    }
