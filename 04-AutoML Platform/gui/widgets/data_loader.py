"""根据数据源选择加载 X, y（或生存分析用的 X, time, event, weights）。"""

from pathlib import Path

import pandas as pd


def load_modeling_data(experiment, use_feature_selected: bool = True,
                       use_minimal_subset: bool = False):
    """加载建模数据。

    use_minimal_subset=True 时优先使用最小子集分析回写的特征集
    （config["minimal_features"]["features"]），缺失时回退共识特征。

    Returns:
        X, y, data_info(dict)
    """
    config = experiment.get("config") or {}
    prep_config = config.get("preprocessing")
    if not prep_config or not prep_config.get("processed_path"):
        raise ValueError("请先完成预处理")

    processed_path = prep_config["processed_path"]
    df = pd.read_csv(processed_path)
    roles = config.get("roles", {})
    y_col = [c for c, r in roles.items() if r == "outcome"]
    if not y_col:
        raise ValueError("未设置 outcome 角色")
    y_col = y_col[0]

    minimal = config.get("minimal_features") or {}
    minimal_feats = [c for c in (minimal.get("features") or []) if c in df.columns]
    if use_minimal_subset and minimal_feats:
        feature_cols = minimal_feats
        data_source = (f"最小子集特征数据（{len(feature_cols)} 个特征，"
                       f"来自 {minimal.get('model', '未知模型')}）")
    else:
        fs_config = config.get("feature_selection")
        if use_feature_selected and fs_config and fs_config.get("consensus_table_path"):
            fs_table = pd.read_csv(fs_config["consensus_table_path"])
            feature_cols = fs_table.loc[fs_table["consensus"] == 1, "feature"].tolist()
            data_source = f"共识特征数据（{len(feature_cols)} 个特征）"
        else:
            feature_cols = [c for c in df.columns if c != y_col]
            data_source = "预处理完整数据"

    # 保持原始特征名：LightGBM/FLAML 内部会自行清洗 JSON 特殊字符，
    # 其他模型统一使用原始名，保证已训练模型与 SHAP 的特征名一致。
    X = df[feature_cols]
    y = df[y_col]

    info = {
        "processed_path": processed_path,
        "data_source": data_source,
        "n_features": len(feature_cols),
        "n_samples": len(df),
    }
    return X, y, info


def load_survival_data(experiment, use_feature_selected: bool = True):
    """加载生存分析数据。

    Returns:
        X, time, event, weights, data_info(dict)
    """
    config = experiment.get("config") or {}
    prep_config = config.get("preprocessing")
    if not prep_config or not prep_config.get("processed_path"):
        raise ValueError("请先完成预处理")

    processed_path = prep_config["processed_path"]
    df = pd.read_csv(processed_path)
    roles = config.get("roles", {})
    time_col = [c for c, r in roles.items() if r == "time"]
    event_col = [c for c, r in roles.items() if r == "outcome"]
    weight_col = [c for c, r in roles.items() if r == "weight"]

    if not time_col or not event_col:
        raise ValueError("请设置 time 和 outcome 角色")

    if use_feature_selected:
        fs_config = config.get("feature_selection")
        if fs_config and fs_config.get("consensus_table_path"):
            fs_table = pd.read_csv(fs_config["consensus_table_path"])
            feature_cols = fs_table.loc[fs_table["consensus"] == 1, "feature"].tolist()
            data_source = f"共识特征数据（{len(feature_cols)} 个特征）"
        else:
            feature_cols = [c for c, r in roles.items() if r in {"continuous", "binary", "ordinal", "nominal"}]
            data_source = "预处理完整数据（按角色筛选）"
    else:
        feature_cols = [c for c, r in roles.items() if r in {"continuous", "binary", "ordinal", "nominal"}]
        data_source = "预处理完整数据（按角色筛选）"

    # 与 load_modeling_data 相同，保持原始特征名
    X = df[feature_cols]
    time = df[time_col[0]]
    event = df[event_col[0]]
    weights = df[weight_col[0]] if weight_col else None

    info = {
        "processed_path": processed_path,
        "data_source": data_source,
        "n_features": len(feature_cols),
        "n_samples": len(df),
    }
    return X, time, event, weights, info
