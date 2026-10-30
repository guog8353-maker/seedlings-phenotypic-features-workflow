"""Phase 5 测试：可解释性与报告。"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from core.interpretability import compute_permutation_importance, plot_permutation_importance
from core.report_generator import ReportGenerator


def make_data(n=200, n_features=10):
    X, y = make_classification(n_samples=n, n_features=n_features, n_informative=5,
                               n_redundant=2, random_state=42)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)]), pd.Series(y, name="outcome")


def test_permutation_importance():
    X, y = make_data()
    model = RandomForestClassifier(random_state=42).fit(X, y)
    imp = compute_permutation_importance(model, X, y, n_repeats=3)
    print("Permutation importance OK")
    print(imp.head())


def test_report():
    X, y = make_data()
    leaderboard = pd.DataFrame({
        "model": ["RF", "LR"],
        "auc": [0.9, 0.85],
    })
    gen = ReportGenerator("test_exp", str(PROJECT_ROOT / "outputs/reports"))
    gen.build_from_results(
        config={"cv_folds": 5, "encoding": "label"},
        leaderboard=leaderboard,
        figure_paths=[],
    )
    md_path = gen.save_markdown()
    print(f"Report saved: {md_path}")


if __name__ == "__main__":
    test_permutation_importance()
    test_report()
