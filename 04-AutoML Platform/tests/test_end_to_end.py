"""端到端测试：数据加载 → 预处理 → 特征选择 → 模型训练 → 集成 → 报告。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.datasets import make_classification
from core.data_manager import DataManager
from core.preprocessor import Preprocessor, apply_imbalance_handling
from core.feature_selector import FeatureSelectionRunner
from core.model_trainer import ModelTrainer
from core.ensemble import EnsembleRunner
from core.report_generator import ReportGenerator


def make_data(n=500, n_features=20):
    X, y = make_classification(
        n_samples=n,
        n_features=n_features,
        n_informative=8,
        n_redundant=4,
        n_classes=2,
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    # 加入一个分类变量
    X["cat"] = np.random.choice(["A", "B", "C"], size=n)
    y = pd.Series(y, name="outcome")
    return X, y


def main():
    print("=" * 50)
    print("AutoML-Platform 端到端测试")
    print("=" * 50)

    # 1. 数据加载
    X, y = make_data()
    dm = DataManager()
    dm.df = X.copy()
    dm.df["outcome"] = y
    dm.roles = {c: "continuous" for c in X.columns}
    dm.roles["cat"] = "nominal"
    dm.roles["outcome"] = "outcome"
    print(f"[1/6] 数据加载完成：{dm.df.shape}")

    # 2. 预处理
    feature_cols = [c for c in dm.df.columns if c != "outcome"]
    numeric_cols = [c for c in feature_cols if dm.roles[c] == "continuous"]
    categorical_cols = [c for c in feature_cols if dm.roles[c] == "nominal"]

    prep = Preprocessor(
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        missing_strategy="median",
        encoding="ordinal",
        scaling="standard",
    )
    X_processed = prep.fit_transform(dm.df[feature_cols])
    y_processed = dm.df["outcome"]
    print(f"[2/6] 预处理完成：{X_processed.shape}")

    # 3. 特征选择
    runner = FeatureSelectionRunner(
        methods=["LASSO", "MutualInformation", "RFE"],
        consensus_threshold=2,
    )
    consensus_table = runner.run(X_processed, y_processed)
    selected_features = consensus_table.loc[consensus_table["consensus"] == 1, "feature"].tolist()
    print(f"[3/6] 特征选择完成：共识特征 {len(selected_features)} 个")

    if selected_features:
        X_model = X_processed[selected_features]
    else:
        X_model = X_processed

    # 4. 模型训练
    trainer = ModelTrainer(X_model, y_processed, cv_folds=3,
                           output_dir=str(PROJECT_ROOT / "outputs/e2e/models"))
    leaderboard = trainer.train_multiple(
        names=["PLR", "RandomForest", "LightGBM"],
        use_optuna=False,
    )
    print("[4/6] 模型训练完成")
    print(leaderboard[["model", "auc", "auprc", "f1"]])

    # 5. 集成学习
    if len(trainer.results) >= 2:
        oof_df = pd.DataFrame({r.name: r.oof_proba for r in trainer.results})
        ensemble = EnsembleRunner("WeightedAverage")
        ensemble.fit(oof_df, y_processed)
        ensemble_proba = ensemble.predict_proba(oof_df)
        from sklearn.metrics import roc_auc_score
        ensemble_auc = roc_auc_score(y_processed, ensemble_proba)
        print(f"[5/6] 集成学习完成：WeightedAverage AUC={ensemble_auc:.4f}")
    else:
        print("[5/6] 集成学习跳过：模型数量不足")

    # 6. 报告
    gen = ReportGenerator("e2e_test", str(PROJECT_ROOT / "outputs/e2e/reports"))
    gen.build_from_results(
        config={"cv_folds": 3, "encoding": "ordinal", "feature_selection": ["LASSO", "MI", "RFE"]},
        leaderboard=leaderboard,
        figure_paths=[],
    )
    report_path = gen.save_markdown()
    print(f"[6/6] 报告生成完成：{report_path}")

    print("=" * 50)
    print("端到端测试通过")
    print("=" * 50)


if __name__ == "__main__":
    main()
