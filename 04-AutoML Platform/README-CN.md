# AutoML-Platform

**面向表格（结构化）数据的半自动化机器学习工作流桌面平台** —— 基于 PyQt6 的图形界面，覆盖从原始数据到可交付报告的完整流程：数据加载、预处理、特征选择、模型训练、集成学习、深度表格模型、AutoML、生存分析、可解释性、可视化与报告导出。

[English README](README.md)

---

## 为什么选择 AutoML-Platform？

AutoML-Platform 把自动化效率与全程可控结合在一起：工作流的每个环节都有独立选项卡和合理默认值，每个参数——缺失值策略、特征选择共识规则、Optuna 搜索空间、集成方式、内存预算——都看得见、调得动。实验持久化在本地 SQLite 数据库中，随时归档、回看、增量续跑。

## 核心特性

- **端到端表格机器学习工作流** —— 数据 → 预处理 → 特征选择 → 建模 → 集成 → 解释 → 报告，全部在一个桌面应用内完成。
- **30+ 基础模型** —— 从逻辑回归、SVM 到 XGBoost / LightGBM / CatBoost，支持 **Optuna** 超参寻优与**增量训练**（实验完成后追加新模型，无需全部重跑）。
- **7 法共识特征选择** —— LASSO、Boruta、RFE、互信息、stepAIC、SCAD、遗传算法（GA），汇总为共识特征集。
- **8 种集成策略** —— 简单平均、加权平均、硬投票、GA 选择、Stacking、Caruana 集成选择（经典版 + 优化版）、SuperLearner。
- **GPU 深度表格模型** —— FT-Transformer、TabTransformer、SAINT、sTFT、TabM、RealMLP、RealTabR、XRFM、TabPFN v3、TabICL。
- **AutoML 后端** —— AutoGluon（带显式**内存预算**分档）、FLAML、H2O AutoML。
- **生存分析** —— CoxPH、加权 Cox、DeepSurv。
- **GPU 加速** —— PyTorch、XGBoost、CatBoost 检测到 CUDA 后自动上卡训练。
- **SHAP 可解释性** —— 跨模型 SHAP 汇总、依赖图、指标级对比。
- **实验管理** —— 每次运行都记录在本地 SQLite 数据库；结果可在界面内归档、重新加载或删除。
- **报告导出** —— 一键生成 Markdown / Word 报告，章节可自选。
- **复杂抽样设计支持** —— 可标注 weight / strata / psu 列并在全流程透传，用于加权分析。

## 界面导览（16 个选项卡）

| 选项卡 | 功能 |
|---|---|
| **首页** | 新建 / 打开 / 删除实验；每个实验是独立工作区，拥有自己的数据、设置与结果。 |
| **数据** | 加载 CSV / Excel / SAS / SPSS / Stata 文件；标注列角色：`id`、`outcome`、`time`（生存时间）、`weight`、`strata`、`psu`、`drop` 或特征类型。 |
| **预处理** | 缺失值插补、异常值处理、类别编码（默认 Label/Ordinal，保持 SHAP 一致性）、缩放、类不平衡处理。所有步骤封装为 sklearn Pipeline，可持久化。 |
| **特征选择** | 并行运行 7 种选择器的任意子集；查看各方法排名与共识特征集。 |
| **模型** | 32 个基础模型任选，交叉验证训练；每个模型可选 Optuna 调参；支持事后增量训练。 |
| **集成** | 用 8 种集成方法组合已训练模型；基于 OOF 的元学习。 |
| **Transformer** | 在 GPU 上训练深度表格模型（FT-Transformer、TabTransformer、SAINT、sTFT、TabM、RealMLP、RealTabR、XRFM、TabPFN v3、TabICL）。 |
| **AutoML** | 启动 AutoGluon / FLAML / H2O，支持时间限制与 AutoGluon 内存预算（按内存/显存自动选择模型档位）。 |
| **生存分析** | CoxPH、加权 Cox、DeepSurv；C-index 评估。 |
| **消融分析** | 对模型或特征组做留一 / 留单消融，量化各组件贡献。 |
| **最小子集** | 搜索保持性能的最小特征子集。 |
| **工作流** | 以有序步骤管道形式查看并重跑实验。 |
| **结果** | 跨层级（基础 / 集成 / AutoML / Transformer）排行榜，附 SHAP 分析。 |
| **可视化** | ROC / PR / 校准曲线、混淆矩阵、SHAP 图；内置多套调色板（含期刊经典配色），支持自定义。 |
| **报告** | 勾选章节，导出 Markdown 或 Word 报告。 |
| **日志** | 当前会话的实时应用日志。 |

## 安装

### 前置要求

- **Python 3.9+**（参考环境：CPU 3.11 / GPU 3.12）
- **Windows 10/11**
- 可选（推荐）：**NVIDIA GPU + CUDA**，用于 Transformer / XGBoost / CatBoost 加速
- 可选：**Java 8+**（仅 H2O AutoML 后端需要）

### 方式 A —— conda（推荐）

GPU 环境（RTX 系列，CUDA 12.8 wheel；详见 `docs/gpu_environment_setup.md`）：

```bash
conda env create -f environment_gpu.yml
conda activate automl-platform-gpu
```

纯 CPU 环境：

```bash
conda env create -f environment.yml
conda activate automl-platform
```

### 方式 B —— pip

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt          # CPU
# 或 GPU 版本：
pip install -r requirements-gpu.txt
```

验证 GPU 安装：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 快速开始

1. **启动应用**
   - Windows：双击 `run.bat`（依次尝试 conda 环境 `automl-platform-gpu` → `automl-platform`，最后回退到 PATH 上的 `python`）；或
   - 任意平台：`python app.py`

2. 在**首页**新建实验（名称 + 可选描述）。

3. 在**数据**页加载数据。仓库自带一个小型演示文件 `examples/demo_data.csv`。标注 `outcome` 列（生存任务还需 `time` 列）；其余列自动识别、可手动调整。

4. **从左到右依次走工作流**：预处理 → 特征选择 → 模型 → 集成 / Transformer / AutoML → 结果 → 可视化 → 报告。

5. 在**报告**页导出报告（Markdown 或 Word）。

### 模型权重（TabPFN v3 / TabICL）

TabPFN v3 与 TabICL 的权重在首次使用时会**从 HuggingFace 自动下载**；也可以手动下载后放入项目根目录的 `models_ckpt/` 文件夹，本地副本优先。

### H2O 说明

H2O 后端会启动本地 Java 集群；如需使用请安装 JRE/JDK（8+）并确保 `java` 在 PATH 中。H2O 在 Windows 上不支持 GPU，把 `time_limit` 调小即可。

## 项目结构

```
AutoML-Platform/
├── app.py                  # 应用入口
├── run.bat                 # Windows 启动器（自动探测 conda 环境）
├── config/                 # YAML 配置（默认值、模型参数、AutoML、集成等）
├── core/                   # 全部 ML 逻辑：数据、预处理、特征选择、模型、集成、
│                           #   Transformer、AutoML、生存分析、SHAP、报告、实验数据库
├── gui/                    # PyQt6 选项卡与可复用组件
├── utils/                  # 日志
├── tests/                  # 端到端 / 导入 / 阶段测试
├── examples/demo_data.csv  # 小型演示数据集
└── docs/                   # GPU 环境搭建记录
```

## 运行测试

```bash
python -m pytest tests/          # 或单独运行，如 python tests/test_imports.py
```

## 文档

- `docs/gpu_environment_setup.md` —— 实测可用的 GPU 技术栈（torch cu128、XGBoost、CatBoost）、安装步骤与踩坑记录。

## 许可证

MIT —— 见 [LICENSE](LICENSE)。Copyright (c) 2026 AutoML-Platform contributors。

示例数据说明：`examples/demo_data.csv` 为固定随机种子 42 生成的合成分类数据，不含真实受试者或实验样本信息；目标列为 `outcome`。
