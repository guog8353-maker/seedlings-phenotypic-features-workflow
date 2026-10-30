"""Command-line batch feature extraction for one labeled crop dataset.

The input directory is scanned recursively.  It may be a single grid_*
folder, a parent containing several grid_* folders, or a flat image folder.
Each image becomes one output row; labels come from immediate 0/1 parents.
"""

import argparse
import multiprocessing
import os
import re
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.batch_features import META_COLUMNS, write_features_meta
from src.features_v2 import extract_crop_features
from src.features_v2.geometry import canopy_mask
from src.features_v2.spatial import spatial_features_from_coverage
from src.folder_loader import scan_folder
from src.image_io import imread_unicode

ALL_GROUPS = ["geometry", "texture", "color", "vegetation", "spatial"]
FORMATS = ["csv", "parquet", "sqlite"]


def _worker(task):
    """Compute crop-local features in a spawned/forked worker process."""
    idx, path, groups = task
    img = imread_unicode(path)
    if img is None or img.size == 0:
        return idx, {}, np.nan
    if img.ndim == 3 and img.shape[2] == 4:
        import cv2
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    local_groups = [group for group in groups if group != "spatial"]
    feats = extract_crop_features(img, local_groups) if local_groups else {}
    cc = np.nan
    if "spatial" in groups:
        mask = canopy_mask(img)
        cc = float(np.count_nonzero(mask)) / float(mask.size)
    return idx, feats, cc


def _fill_spatial(items, rows, cc_list):
    """Aggregate coverage per (original_image, grid_strategy) parent."""
    parents = {}
    for i, item in enumerate(items):
        parents.setdefault((item.original_image, item.grid_strategy), []).append(i)

    for (original, grid), indices in parents.items():
        coords = [(items[i].row, items[i].col) for i in indices]
        if any(r is None or c is None for r, c in coords) or len(indices) < 4:
            continue
        max_row = max(r for r, _ in coords)
        max_col = max(c for _, c in coords)
        coverage = np.full((max_row + 1, max_col + 1), np.nan)
        for i in indices:
            coverage[items[i].row, items[i].col] = cc_list[i]
        spatial = spatial_features_from_coverage(coverage)
        for i in indices:
            rows[i].update(spatial)


def process_dataset(folder, groups, executor, chunksize):
    """Read every crop, compute local features, then backfill spatial features."""
    items = scan_folder(str(folder))
    total = len(items)
    if total == 0:
        raise ValueError(f"输入目录中没有支持的图片：{folder}")

    print(f"发现 {total} 个子块：{folder}", flush=True)
    tasks = [(i, str(item.path), groups) for i, item in enumerate(items)]
    features = [None] * total
    coverage_values = [np.nan] * total

    started = time.time()
    report_every = max(1, min(1000, total // 20 or 1))
    for done, (idx, feats, cc) in enumerate(
        executor.map(_worker, tasks, chunksize=chunksize), 1
    ):
        features[idx] = feats
        coverage_values[idx] = cc
        if done % report_every == 0 or done == total:
            rate = done / max(time.time() - started, 1e-9)
            print(f"进度 {done}/{total}（{rate:.1f} 块/秒）", flush=True)

    rows = []
    for i, item in enumerate(items):
        row = {
            "image_path": str(item.path),
            "original_image": item.original_image,
            "row": item.row if item.row is not None else "",
            "col": item.col if item.col is not None else "",
            "grid_strategy": item.grid_strategy,
            "label": item.label,
        }
        if features[i]:
            row.update(features[i])
        rows.append(row)

    if "spatial" in groups:
        _fill_spatial(items, rows, coverage_values)
    return rows


def _sqlite_table_name(name):
    table = re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_") or "features"
    if table[0].isdigit():
        table = "dataset_" + table
    return table


def write_outputs(name, rows, output_dir, groups, formats):
    """Write the requested CSV, Parquet and/or SQLite representations."""
    import pandas as pd

    df = pd.DataFrame(rows)
    ordered = META_COLUMNS + [c for c in df.columns if c not in META_COLUMNS]
    df = df[ordered]
    output_dir.mkdir(parents=True, exist_ok=True)

    if "csv" in formats:
        path = output_dir / f"{name}_features.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"CSV：{path}", flush=True)

    if "parquet" in formats:
        path = output_dir / f"{name}_features.parquet"
        try:
            df.to_parquet(path, index=False)
        except ImportError as exc:
            raise RuntimeError(
                "Parquet 输出需要 pyarrow；请执行 "
                "python -m pip install -r requirements-batch.txt，"
                "或使用 --formats csv sqlite。"
            ) from exc
        print(f"Parquet：{path}", flush=True)

    if "sqlite" in formats:
        path = output_dir / "features.sqlite"
        table = _sqlite_table_name(name)
        with sqlite3.connect(path) as conn:
            df.to_sql(table, conn, if_exists="replace", index=False)
        print(f"SQLite：{path}（表 {table}）", flush=True)

    meta_path = output_dir / "features_meta.json"
    write_features_meta(str(meta_path), groups)
    print(f"特征说明：{meta_path}", flush=True)
    print(f"完成：{len(df)} 行 × {len(df.columns)} 列", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="递归读取子块图片，批量生成特征 CSV、Parquet 和 SQLite。"
    )
    parser.add_argument("--input", required=True, help="子块数据集目录")
    parser.add_argument("--output", required=True, help="结果输出目录")
    parser.add_argument(
        "--name", help="结果文件前缀和 SQLite 表名；默认使用输入目录名"
    )
    parser.add_argument(
        "--groups", nargs="+", choices=ALL_GROUPS, default=ALL_GROUPS,
        help="参与计算的特征组，默认全部",
    )
    parser.add_argument(
        "--formats", nargs="+", choices=FORMATS, default=FORMATS,
        help="输出格式，默认 csv parquet sqlite",
    )
    parser.add_argument(
        "--workers", type=int, default=min(16, os.cpu_count() or 4),
        help="并行进程数，默认 min(16, CPU 逻辑核心数)",
    )
    parser.add_argument(
        "--chunksize", type=int, default=128,
        help="每个并行任务批包含的图片数，默认 128",
    )
    return parser.parse_args()


def main():
    multiprocessing.freeze_support()
    args = parse_args()
    if args.workers < 1 or args.chunksize < 1:
        raise ValueError("--workers 和 --chunksize 必须大于 0")

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")
    name = (args.name or input_dir.name).strip()
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("--name 必须是非空的单段文件名前缀")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = process_dataset(input_dir, args.groups, executor, args.chunksize)
    write_outputs(name, rows, output_dir, args.groups, set(args.formats))


if __name__ == "__main__":
    main()
