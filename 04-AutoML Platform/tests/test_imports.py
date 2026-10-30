"""Phase 1 冒烟测试：验证核心模块可导入、数据库可创建、DataManager 可工作。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PyQt6.QtWidgets import QApplication
from core.experiment_db import ExperimentDB
from core.data_manager import DataManager


def test_app():
    app = QApplication([])
    print("QApplication created OK")

    db = ExperimentDB(str(PROJECT_ROOT / "outputs" / "test_experiments.db"))
    import time
    exp_id = db.create_experiment(name=f"smoke_test_{int(time.time())}", config={"test": True})
    exp = db.get_experiment(exp_id)
    assert "smoke_test" in exp['name']
    print(f"ExperimentDB OK (id={exp_id})")

    dm = DataManager()
    df = dm.load(str(PROJECT_ROOT / "examples" / "demo.csv")) if (PROJECT_ROOT / "examples" / "demo.csv").exists() else None
    if df is not None:
        print(f"DataManager load OK, shape={df.shape}")
    else:
        print("DataManager instantiated OK (no demo data)")

    print("All smoke tests passed.")


if __name__ == "__main__":
    test_app()
