# AutoML-Platform

**A semi-automated machine-learning workflow desktop platform for tabular (structured) data** — a PyQt6 GUI that covers the entire journey from raw data to a publication-ready report: data loading, preprocessing, feature selection, model training, ensembling, deep tabular models, AutoML, survival analysis, interpretability, visualization, and report export.

[中文文档](README-CN.md)

---

## Why AutoML-Platform?

AutoML-Platform combines the speed of automation with full, hands-on control: every stage of the workflow is a dedicated GUI tab with sensible defaults, and every knob — missing-value strategy, feature-selection consensus, Optuna search space, ensemble method, memory budget — is visible and adjustable. Experiments are persisted in a local SQLite database, so you can archive, revisit, and incrementally extend any run.

## Key Features

- **End-to-end tabular ML workflow** — data → preprocessing → feature selection → modeling → ensemble → interpretation → report, all inside one desktop app.
- **30+ base models** — from logistic regression and SVMs to XGBoost / LightGBM / CatBoost, with **Optuna** hyperparameter search and **incremental training** (add new models to a finished experiment without re-running everything).
- **7-method consensus feature selection** — LASSO, Boruta, RFE, mutual information, stepAIC, SCAD, and genetic algorithm (GA), combined into a consensus feature set.
- **8 ensemble strategies** — simple/weighted averaging, hard voting, GA selection, stacking, Caruana ensemble selection (classic + optimized), and SuperLearner.
- **Deep tabular models on GPU** — FT-Transformer, TabTransformer, SAINT, sTFT, TabM, RealMLP, RealTabR, XRFM, TabPFN v3, TabICL.
- **AutoML backends** — AutoGluon (with an explicit **memory budget** tier selector), FLAML, and H2O AutoML.
- **Survival analysis** — CoxPH, weighted Cox, and DeepSurv.
- **GPU acceleration** — PyTorch, XGBoost, and CatBoost train on CUDA automatically when available.
- **SHAP interpretability** — cross-model SHAP summaries, dependence plots, and indicator-level comparison.
- **Experiment management** — every run is recorded in a local SQLite database; results can be archived, reloaded, or deleted from the GUI.
- **Report export** — one-click Markdown / Word reports with customizable sections.
- **Complex survey designs** — sampling-weight / strata / PSU columns can be tagged and passed through the whole pipeline for weighted analysis.

## GUI Tour (16 Tabs)

| Tab | What it does |
|---|---|
| **首页 (Home)** | Create / open / delete experiments; each experiment is an isolated workspace with its own data, settings, and results. |
| **数据 (Data)** | Load CSV / Excel / SAS / SPSS / Stata files; tag column roles: `id`, `outcome`, `time` (survival), `weight`, `strata`, `psu`, `drop`, or a feature type. |
| **预处理 (Preprocessing)** | Missing-value imputation, outlier handling, categorical encoding (label/ordinal by default — keeps SHAP consistent), scaling, and class-imbalance handling. All steps are sklearn Pipelines and persistable. |
| **特征选择 (Feature Selection)** | Run any subset of the 7 selectors in parallel; inspect per-method rankings and the consensus set. |
| **模型 (Models)** | Train any of the 32 base models with cross-validation; optional Optuna tuning per model; incremental training to add more models later. |
| **集成 (Ensemble)** | Combine trained models via 8 ensemble methods; OOF-based meta-learning. |
| **Transformer** | Train deep tabular models (FT-Transformer, TabTransformer, SAINT, sTFT, TabM, RealMLP, RealTabR, XRFM, TabPFN v3, TabICL) on GPU. |
| **AutoML** | Launch AutoGluon / FLAML / H2O with time limits and an AutoGluon memory budget (model tier is auto-selected to fit your RAM/VRAM). |
| **生存分析 (Survival)** | CoxPH, weighted Cox, DeepSurv; C-index evaluation. |
| **消融分析 (Ablation)** | Leave-one-out / leave-one-in ablation over models or feature groups to quantify each component's contribution. |
| **最小子集 (Minimal Subset)** | Search for the smallest feature subset that preserves performance. |
| **工作流 (Workflow)** | View and re-run the experiment as an ordered pipeline of steps. |
| **结果 (Results)** | Leaderboard across all layers (base / ensemble / AutoML / Transformer), plus SHAP analysis. |
| **可视化 (Visualization)** | ROC / PR / calibration curves, confusion matrices, SHAP plots; multiple built-in color palettes (including a journal-classic palette) plus custom palettes. |
| **报告 (Report)** | Pick sections and export a Markdown or Word report. |
| **日志 (Log)** | Live application log for the session. |

## Installation

### Prerequisites

- **Python 3.9+** (reference environments: 3.11 CPU / 3.12 GPU)
- **Windows 10/11**
- Optional but recommended: **NVIDIA GPU + CUDA** for Transformer / XGBoost / CatBoost acceleration
- Optional: **Java 8+** (only needed for the H2O AutoML backend)

### Option A — conda (recommended)

GPU environment (RTX-series, CUDA 12.8 wheels; see `docs/gpu_environment_setup.md`):

```bash
conda env create -f environment_gpu.yml
conda activate automl-platform-gpu
```

CPU-only environment:

```bash
conda env create -f environment.yml
conda activate automl-platform
```

### Option B — pip

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt          # CPU
# or, for GPU builds:
pip install -r requirements-gpu.txt
```

Verify a GPU install:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Quick Start

1. **Launch the app**

   - Windows: double-click `run.bat` (it tries the conda envs `automl-platform-gpu` → `automl-platform` in order, then falls back to `python` on PATH), or
   - any platform: `python app.py`

2. **Create an experiment** on the Home tab (name + optional description).

3. **Load data** on the Data tab. A reproducible synthetic demo (random seed 42; no real subjects) ships at `examples/demo_data.csv`. Tag the `outcome` column (and `time` for survival tasks); everything else is auto-detected and adjustable.

4. **Walk the workflow left to right**: Preprocessing → Feature Selection → Models → Ensemble / Transformer / AutoML → Results → Visualization → Report.

5. **Export a report** from the Report tab (Markdown or Word).

### Model weights (TabPFN v3 / TabICL)

TabPFN v3 and TabICL weights **download automatically from HuggingFace** on first use. You can also place the checkpoints manually in a `models_ckpt/` folder at the project root — local copies take priority.

### H2O note

The H2O backend starts a local Java cluster; install a JRE/JDK (8+) and make sure `java` is on PATH if you plan to use it. H2O does not support GPU on Windows — give it a modest `time_limit` instead.

## Project Structure

```
AutoML-Platform/
├── app.py                  # Application entry point
├── run.bat                 # Windows launcher (conda env auto-detect)
├── config/                 # YAML configs (defaults, model params, AutoML, ensemble, …)
├── core/                   # All ML logic: data, preprocessing, FS, models, ensembles,
│                           #   transformers, AutoML, survival, SHAP, reports, experiment DB
├── gui/                    # PyQt6 tabs and reusable widgets
├── utils/                  # Logging
├── tests/                  # End-to-end / import / phase tests
├── examples/demo_data.csv  # Tiny demo dataset
└── docs/                   # GPU environment setup notes
```

## Running Tests

```bash
python -m pytest tests/          # or run individual files, e.g. python tests/test_imports.py
```

## Documentation

- `docs/gpu_environment_setup.md` — verified GPU stack (torch cu128, XGBoost, CatBoost), install steps, and troubleshooting.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 AutoML-Platform contributors.
