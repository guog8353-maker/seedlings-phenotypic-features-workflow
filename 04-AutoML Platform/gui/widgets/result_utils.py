"""结果管理工具：删除 / 归档各模块的输出目录。"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


# 下游结果目录 → 对应 config 键（None 表示无独立 config 键，仅靠目录）
DOWNSTREAM_ALL = {
    "models": "modeling",
    "ensemble": "ensemble",
    "transformer": "transformer",
    "automl": "automl",
    "shap": None,
    "ablation": "ablation",
    "minimal_subset": "minimal_subset",
    "survival": "survival",
    "reports": None,
}
# 依赖基础模型的下游（模型重跑时归档这些）
DOWNSTREAM_OF_MODEL = ["ensemble", "shap", "ablation", "minimal_subset", "reports"]


def archive_downstream_results(db, exp_id: int, trigger: str, logger=None) -> List[str]:
    """上游变动时，把下游结果目录归档到 exp_{id}/archive/{时间戳}_{trigger}/ 并清空对应 config。

    旧结果全部保留在 archive 中（不删除），下游选项卡随后显示"无结果/需重跑"。

    Args:
        db: ExperimentDB 实例。
        exp_id: 实验 ID。
        trigger: "data" | "preprocessing" | "feature_selection"（归档全部下游）
                 或 "modeling"（仅归档依赖基础模型的下游）。
        logger: 可选日志记录器。

    Returns:
        已归档的目录名列表。
    """
    exp = db.get_experiment(exp_id)
    if not exp:
        return []
    config = exp.get("config") or {}
    exp_dir = Path(exp.get("output_dir") or "outputs") / f"exp_{exp_id}"
    if not exp_dir.exists():
        return []

    if trigger in ("data", "preprocessing", "feature_selection"):
        dir_names = list(DOWNSTREAM_ALL.keys())
    elif trigger == "modeling":
        dir_names = list(DOWNSTREAM_OF_MODEL)
    else:
        return []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = exp_dir / "archive" / f"{timestamp}_{trigger}"
    archived: List[str] = []
    for name in dir_names:
        d = exp_dir / name
        if d.exists():
            try:
                archive_root.mkdir(parents=True, exist_ok=True)
                shutil.move(str(d), str(archive_root / name))
                archived.append(name)
            except Exception as e:
                if logger:
                    logger.warning(f"归档 {name} 失败：{e}")
        # 清空对应 config 键
        key = DOWNSTREAM_ALL.get(name)
        if key and key in config:
            del config[key]
    # 一并清理派生配置
    for extra_key in ("minimal_features",):
        if extra_key in config:
            del config[extra_key]

    if archived:
        db.update_experiment(exp_id, {"config": config})
        if logger:
            logger.info(f"上游变动({trigger})：已归档 {len(archived)} 个下游结果到 {archive_root}（{', '.join(archived)}）")
    return archived


def is_result_stale(result_path, upstream_paths: List) -> Tuple[bool, Optional[str]]:
    """检测结果文件是否比任一上游文件旧（即上游已更新、结果可能已过期）。

    Args:
        result_path: 结果文件路径。
        upstream_paths: 上游文件路径列表（如模型 leaderboard、SHAP meta 等）。

    Returns:
        (is_stale, stale_reason)
    """
    rp = Path(result_path) if result_path else None
    if rp is None or not rp.exists():
        return False, None
    try:
        result_mtime = rp.stat().st_mtime
    except Exception:
        return False, None
    for up in upstream_paths:
        if not up:
            continue
        up = Path(up)
        if up.exists() and up.stat().st_mtime > result_mtime:
            return True, up.name
    return False, None


def delete_module_results(
    result_dir: Path,
    table_widget=None,
    status_label=None,
    logger=None,
) -> Tuple[bool, str]:
    """删除模块结果目录，并可选择清空结果表格与状态标签。

    Args:
        result_dir: 模块结果目录（如 outputs/exp_1/models）。
        table_widget: PyQt 结果表格，删除成功后清空。
        status_label: PyQt 状态标签，删除成功后更新文本。
        logger: 可选日志记录器。

    Returns:
        (success, message)
    """
    if not result_dir.exists():
        return False, f"结果目录不存在：{result_dir}"
    try:
        shutil.rmtree(result_dir)
        if table_widget is not None:
            table_widget.setRowCount(0)
            table_widget.setColumnCount(0)
        if status_label is not None:
            status_label.setText("结果已删除")
        msg = f"已删除结果目录：{result_dir}"
        if logger:
            logger.info(msg)
        return True, msg
    except Exception as e:
        msg = f"删除结果目录失败：{e}"
        if logger:
            logger.error(msg)
        return False, msg


def archive_module_results(
    result_dir: Path,
    archive_dir: Optional[Path] = None,
    logger=None,
) -> Tuple[bool, str]:
    """将模块结果目录移动到 archive/ 下并附加时间戳。

    Args:
        result_dir: 模块结果目录。
        archive_dir: 归档根目录，默认为 result_dir.parent / "archive"。
        logger: 可选日志记录器。

    Returns:
        (success, message)
    """
    if not result_dir.exists():
        return False, f"结果目录不存在：{result_dir}"
    if archive_dir is None:
        archive_dir = result_dir.parent / "archive"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = archive_dir / f"{result_dir.name}_{timestamp}"
    try:
        archive_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(result_dir), str(dest))
        msg = f"已归档至：{dest}"
        if logger:
            logger.info(msg)
        return True, msg
    except Exception as e:
        msg = f"归档失败：{e}"
        if logger:
            logger.error(msg)
        return False, msg


# ---------------------------------------------------------------------------
# 增量训练：已有结果读取（供 transformer / ensemble / automl 选项卡跳过已完成项）
# ---------------------------------------------------------------------------

def load_successful_results(result_path, key_col: str,
                            artifact_dir=None, artifact_suffix: str = None,
                            proba_col: str = None):
    """加载已有的成功结果行（用于增量训练时跳过）。

    优先从结果 CSV 读取；CSV 中缺失、但产物目录里存在预测文件
    （``{名称}{artifact_suffix}``，含 y_true 与 proba 列）的条目，就地重建指标行。
    含 error 或 auc<=0 的失败行不视为已有结果（允许重跑）。

    Args:
        result_path: 结果 CSV 路径（Path）。
        key_col: 条目标识列名（如 "model" / "method" / "framework"）。
        artifact_dir: 产物目录（Path），为 None 时只做 CSV 读取。
        artifact_suffix: 预测文件后缀（如 "_val.csv" / "_oof.csv"）。
        proba_col: 预测概率列名（如 "val_proba" / "oof_proba"）。

    Returns:
        pd.DataFrame，每行一个已成功条目；无结果时为空 DataFrame。
    """
    import pandas as pd
    from core.model_trainer import compute_metrics_optimal_threshold

    rows = {}
    result_path = Path(result_path)
    if result_path.exists():
        try:
            df = pd.read_csv(result_path)
            if key_col in df.columns:
                for _, r in df.iterrows():
                    err = r.get("error") if "error" in df.columns else None
                    if err is not None and pd.notna(err) and str(err).strip():
                        continue
                    if "auc" in df.columns and not (r.get("auc", 0) > 0):
                        continue
                    rows[str(r[key_col])] = r.to_dict()
        except Exception:
            pass

    if artifact_dir is not None and artifact_suffix and proba_col:
        for f in sorted(Path(artifact_dir).glob(f"*{artifact_suffix}")):
            name = f.name[: -len(artifact_suffix)]
            if name in rows:
                continue
            try:
                d = pd.read_csv(f)
                metrics = compute_metrics_optimal_threshold(d["y_true"], d[proba_col], "youden")
                rows[name] = {key_col: name, **metrics}
            except Exception:
                continue

    return pd.DataFrame(list(rows.values())) if rows else pd.DataFrame()


def merge_result_rows(existing, new_records, key_col: str):
    """合并已有结果与新记录：新记录覆盖同名条目，其余旧条目保留，按 auc 降序。"""
    import pandas as pd

    new_df = pd.DataFrame(new_records)
    if existing is None or existing.empty:
        result = new_df
    elif new_df.empty:
        result = existing
    else:
        new_keys = set(new_df[key_col].astype(str))
        old_keep = existing[~existing[key_col].astype(str).isin(new_keys)]
        result = pd.concat([old_keep, new_df], ignore_index=True)
    if "auc" in result.columns and not result.empty:
        result = result.sort_values("auc", ascending=False).reset_index(drop=True)
    return result
