"""Batch runner for the three multi-task tasks: 病斑、徒长、黄化.

Only grid_* directories are scanned, avoiding duplicate flat 0/1 exports.
All local paths are supplied through command-line arguments.
"""

import argparse
import multiprocessing
import os
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

GROUPS = ["geometry", "texture", "color", "vegetation", "spatial"]
FORMATS = ["csv", "parquet", "sqlite"]
DATASETS = {
    "病斑": ("病斑数据", "病斑特征", "bingban"),
    "徒长": ("徒长数据", "徒长特征", "tuchang"),
    "黄化": ("黄化数据", "黄化特征", "huanghua"),
}
DEFAULT_GRIDS = ["grid_36x18", "grid_41x20", "grid_48x28"]


def _worker(task):
    idx, path = task
    img = imread_unicode(path)
    if img is None or img.size == 0:
        return idx, {}, np.nan
    if img.ndim == 3 and img.shape[2] == 4:
        import cv2
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    feats = extract_crop_features(img, GROUPS[:-1])
    mask = canopy_mask(img)
    cc = float(np.count_nonzero(mask)) / float(mask.size)
    return idx, feats, cc


def _fill_spatial(items, rows, coverage_values):
    parents = {}
    for i, item in enumerate(items):
        parents.setdefault((item.original_image, item.grid_strategy), []).append(i)
    for (original, grid), indices in parents.items():
        coords = [(items[i].row, items[i].col) for i in indices]
        if any(r is None or c is None for r, c in coords) or len(indices) < 4:
            continue
        coverage = np.full(
            (max(r for r, _ in coords) + 1, max(c for _, c in coords) + 1),
            np.nan,
        )
        for i in indices:
            coverage[items[i].row, items[i].col] = coverage_values[i]
        spatial = spatial_features_from_coverage(coverage)
        for i in indices:
            rows[i].update(spatial)


def process_task(task_name, folder, grids, executor, chunksize):
    items = []
    for grid in grids:
        grid_dir = folder / grid
        if grid_dir.is_dir():
            items.extend(scan_folder(str(grid_dir)))
    total = len(items)
    if total == 0:
        print(f"[{task_name}] 未找到图片，跳过：{folder}", flush=True)
        return []

    print(f"[{task_name}] {total} 个子块", flush=True)
    tasks = [(i, str(item.path)) for i, item in enumerate(items)]
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
            print(f"[{task_name}] {done}/{total}（{rate:.1f} 块/秒）", flush=True)

    rows = []
    for i, item in enumerate(items):
        row = {
            "task": task_name,
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
    _fill_spatial(items, rows, coverage_values)
    return rows


def write_outputs(stem, table, rows, output_dir, formats):
    import pandas as pd

    df = pd.DataFrame(rows)
    ordered = ["task"] + META_COLUMNS + [
        c for c in df.columns if c not in {"task", *META_COLUMNS}
    ]
    df = df[ordered]

    if "csv" in formats:
        path = output_dir / f"{stem}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"CSV：{path}", flush=True)
    if "parquet" in formats:
        path = output_dir / f"{stem}.parquet"
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
        path = output_dir / "features_multi_task.sqlite"
        with sqlite3.connect(path) as conn:
            df.to_sql(table, conn, if_exists="replace", index=False)
        print(f"SQLite：{path}（表 {table}）", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="批量计算病斑、徒长、黄化三个 multi-task 数据集的特征。"
    )
    parser.add_argument(
        "--base-dir", required=True,
        help="包含病斑数据、徒长数据、黄化数据子目录的根目录",
    )
    parser.add_argument(
        "--output-dir", help="输出目录，默认 <base-dir>/Features"
    )
    parser.add_argument(
        "--tasks", nargs="+", choices=list(DATASETS), default=list(DATASETS),
        help="要处理的任务，默认全部",
    )
    parser.add_argument(
        "--grids", nargs="+", default=DEFAULT_GRIDS,
        help="要扫描的 grid_* 目录名，默认三种分块策略",
    )
    parser.add_argument(
        "--formats", nargs="+", choices=FORMATS, default=FORMATS,
        help="输出格式，默认 csv parquet sqlite",
    )
    parser.add_argument(
        "--workers", type=int, default=min(16, os.cpu_count() or 4),
        help="并行进程数",
    )
    parser.add_argument("--chunksize", type=int, default=128, help="并行任务批大小")
    return parser.parse_args()


def main():
    multiprocessing.freeze_support()
    args = parse_args()
    if args.workers < 1 or args.chunksize < 1:
        raise ValueError("--workers 和 --chunksize 必须大于 0")
    base = Path(args.base_dir).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"根目录不存在：{base}")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else base / "Features"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for task in args.tasks:
            folder_name, stem, table = DATASETS[task]
            rows = process_task(
                task, base / folder_name, args.grids, executor, args.chunksize
            )
            if rows:
                write_outputs(stem, table, rows, output_dir, set(args.formats))

    meta_path = output_dir / "features_meta.json"
    write_features_meta(str(meta_path), GROUPS)
    print(f"全部指定任务处理结束；特征说明：{meta_path}", flush=True)


if __name__ == "__main__":
    main()
