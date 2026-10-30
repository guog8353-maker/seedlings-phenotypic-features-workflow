# Seedlings Phenotypic Features Workflow

秧苗表型特征判别流程

A four-stage workflow for seedling image stitching, annotation, phenotypic feature extraction, and tabular machine learning. The workflow supports research on phenotypic discrimination of physiological seedling wilt and damping-off.

本仓库整合四个可独立运行的 Python 桌面项目，用于秧苗表型特征整理与判别研究。各阶段通过图片、标签表和特征表衔接；具体算法、配置与运行方式见各模块文档。

| 阶段 | 目录与文档 | 输入 | 输出 |
|---|---|---|---|
| 1. 扫描拼接 | [01-tray-scan-stitcher](01-tray-scan-stitcher/README.md) | 扫描视频 | 全景母图与拼接报告 |
| 2. 图像标注 | [02-label-studio](02-label-studio/README.md) | 全景图或秧盘图片 | 校正图、掩膜、网格子块和标签表 |
| 3. 特征提取 | [03-feature-lab](03-feature-lab/README.md) | 标注子块图片 | 数值特征 CSV、特征说明及可选 Parquet/SQLite |
| 4. 建模分析 | [04-AutoML Platform](04-AutoML%20Platform/README-CN.md) | 结构化特征表 | 特征选择、分类模型、评估、解释与报告 |

```text
扫描视频 → 全景母图 → 人工复核标注与分块 → 特征表 → 模型训练与评估
```

## 安装与启动

建议使用 Windows 10/11。前三个模块建议 Python 3.12，采用 PySide6 + OpenCV；第四个模块采用 PyQt6，CPU/GPU 环境配置见其独立文档。各模块使用独立虚拟环境。

以下为 Windows CMD 示例：

```bat
cd 01-tray-scan-stitcher
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python main.py
```

第二、第三个模块替换为对应目录后执行相同步骤。第四个模块进入 `04-AutoML Platform`，按其依赖说明安装后执行 `python app.py`。各模块也提供 Windows 启动脚本。

第三个模块的批处理入口为 `run_feature_extraction.py` 和 `run_feature_extraction_multi_task.py`；可选输出依赖见 `requirements-batch.txt`。

## 数据衔接

- 网格目录采用 `grid_<列>x<行>`，例如 `grid_36x18` 表示 36 列、18 行。
- 标准子块文件名为 `<母图名>_r<行号>_c<列号>.jpg`，行列号从 0 开始。
- 标注模块默认按吐水掩膜面积阈值派生 `0/1` 标签；这些标签的含义由标注规则决定，不能直接当作病害诊断结果。
- 将第三模块生成的特征表导入第四模块，并显式设置目标列、样本标识列和特征列。训练前应排除路径、母图标识等可能泄漏标签的信息，并按母图或实验批次合理划分训练与测试数据。
- 特征公式与实现说明见 [FEATURE_REFERENCE.md](03-feature-lab/docs/FEATURE_REFERENCE.md)。

## 示例与使用范围

`04-AutoML Platform/examples/demo_data.csv` 是固定随机种子 42 生成的合成分类示例，包含 `ID`、三个通用特征和 `outcome`，仅供验证操作流程。

本仓库提供工作流源码。论文原始实验数据、训练权重及已验证的病害诊断模型不随源码提供。自动拼接、预标注和特征分析结果需要结合实验设计与人工复核使用。

## 许可证

第四模块沿用其 [MIT 许可证](04-AutoML%20Platform/LICENSE)。其他模块的原始材料未附许可证，本仓库未为其新增授权条款。
