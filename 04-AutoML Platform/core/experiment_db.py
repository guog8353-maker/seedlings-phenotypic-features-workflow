"""SQLite 实验管理：存储实验元数据、配置、结果路径。"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class ExperimentDB:
    def __init__(self, db_path: str = None):
        if db_path is None:
            # 相对路径（工作目录为项目根，app.py 启动时已 chdir），便于开源/迁移
            db_path = "outputs/experiments.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    status TEXT DEFAULT 'created',
                    config TEXT,
                    data_path TEXT,
                    output_dir TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    step TEXT,
                    model_name TEXT,
                    metrics TEXT,
                    artifact_path TEXT,
                    created_at TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            """)

    def create_experiment(self, name: str, description: str = "", config: dict = None,
                          data_path: str = "", output_dir: str = "") -> int:
        now = datetime.now().isoformat()
        config_json = json.dumps(config, ensure_ascii=False) if config else "{}"
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO experiments (name, description, status, config, data_path, output_dir, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (name, description, 'created', config_json, data_path, output_dir, now, now),
            )
            return cur.lastrowid

    def update_experiment(self, experiment_id: int, updates: dict):
        now = datetime.now().isoformat()
        updates['updated_at'] = now
        if 'config' in updates and isinstance(updates['config'], dict):
            updates['config'] = json.dumps(updates['config'], ensure_ascii=False)
        set_clause = ", ".join([f"{k}=?" for k in updates.keys()])
        values = list(updates.values()) + [experiment_id]
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"UPDATE experiments SET {set_clause} WHERE id=?", values)

    def get_experiment(self, experiment_id: int) -> Optional[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
            if row:
                d = dict(row)
                d['config'] = json.loads(d.get('config') or '{}')
                return d
            return None

    def list_experiments(self) -> List[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM experiments ORDER BY updated_at DESC").fetchall()
            experiments = []
            for row in rows:
                d = dict(row)
                d['config'] = json.loads(d.get('config') or '{}')
                experiments.append(d)
            return experiments

    def delete_experiment(self, experiment_id: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM results WHERE experiment_id=?", (experiment_id,))
            conn.execute("DELETE FROM experiments WHERE id=?", (experiment_id,))

    def add_result(self, experiment_id: int, step: str, model_name: str,
                   metrics: dict, artifact_path: str = ""):
        now = datetime.now().isoformat()
        metrics_json = json.dumps(metrics, ensure_ascii=False)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO results (experiment_id, step, model_name, metrics, artifact_path, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (experiment_id, step, model_name, metrics_json, artifact_path, now),
            )

    def get_results(self, experiment_id: int) -> List[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM results WHERE experiment_id=? ORDER BY created_at DESC",
                (experiment_id,),
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d['metrics'] = json.loads(d.get('metrics') or '{}')
                results.append(d)
            return results
