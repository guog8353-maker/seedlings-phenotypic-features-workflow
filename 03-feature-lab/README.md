# Guttation Feature Lab

## 水稻吐水子块特征库与特征图分析平台

Guttation Feature Lab 有两条相对独立的研究工作流：

1. **子块数值特征库**：递归读取软件二或其他流程生成的子块图片，每张图片计算一行结构化特征，导出 CSV 和特征说明；命令行脚本还支持 Parquet、SQLite 和多进程。
2. **像素级特征图研究**：打开一张全景母图，在原始分辨率上计算 26 类响应图，通过热力叠加、参数滑条和 ROI 观察哪些视觉线索能突出水珠。

软件负责特征工程与可视化，不负责训练分类器，也不会根据图片内容改写 `0/1` 标签。

完整数值公式见 **[数值特征完整计算参考](docs/FEATURE_REFERENCE.md)**。

---

## 1. 安装与启动

建议 Windows 10/11 + Python 3.12。

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

也可双击 `启动软件.cmd`，脚本会优先使用项目 `.venv`，否则使用系统 `PATH` 中的 `pythonw`。

### 依赖分别用在哪里

| 依赖 | 用途 |
|---|---|
| PySide6 | 主窗口、数据表、特征图界面和后台线程。 |
| opencv-python / numpy | 图像读取、色彩转换、滤波、形态学和数组计算。 |
| scikit-image | GLCM、LBP 等纹理计算。 |
| pandas | 命令行批处理输出及保留的旧版批处理模块。 |
| matplotlib / openpyxl | 保留的旧版可视化和 Excel 导出兼容模块；当前主窗口不依赖它们完成 117 维导出。 |

命令行默认还会输出 Parquet，需要额外安装：

```bat
python -m pip install -r requirements-batch.txt
```

如果只要 CSV 或 SQLite，可不安装 `pyarrow`，并在命令中指定 `--formats csv sqlite` 或 `--formats csv`。

CNN 是可选特征图，不包含在默认依赖中。需要时自行安装与平台兼容的 `torch`、`torchvision`；MobileNetV2 默认权重首次使用时可能需要联网下载。

---

## 2. 子块特征库快速上手

1. 点击“浏览文件夹”，选择一个 `grid_列x行` 目录，或包含多个 grid 目录的上级目录。
2. 勾选需要的特征组；做完整特征库时保持五组全选。
3. 点击“开始计算特征”，等待后台进度完成。
4. 在“特征数据表”检查 `label`、母图名、行列号和若干数值列。
5. 点击“导出 CSV + 特征说明”，选择目标 CSV 路径。
6. 软件会在同目录生成 `features_meta.json`。

全部五组启用且空间特征可计算时，典型结果为 **6 个元数据列 + 117 个特征列 = 123 列**。实际列数会随特征组和图片/命名有效性变化，详见特征参考。

---

## 3. 输入目录与文件命名契约

### 3.1 推荐结构

```text
<输入目录>/
└── grid_36x18/
    ├── 0/
    │   ├── tray01_r000_c000.jpg
    │   └── tray01_r000_c002.jpg
    └── 1/
        └── tray01_r000_c001.jpg
```

也可以选择更高层目录，同时递归读取多个 `grid_*`：

```text
<输入目录>/
├── grid_36x18/0|1/...
├── grid_41x20/0|1/...
└── grid_48x28/0|1/...
```

`0|1` 在以上示意图中表示两个独立目录，不是目录名本身。

### 3.2 递归扫描规则

支持 `.jpg`、`.jpeg`、`.png`、`.bmp`、`.tif`、`.tiff`。每个图片对应一个 `CropItem`，再对应一行输出。

| 元数据 | 解析方式 |
|---|---|
| `label` | 只读取**直接父目录**名；父目录必须是 `0` 或 `1`，否则为空。 |
| `original_image` | 文件名匹配 `^(.+?)_r(\d+)_c(\d+)$` 时取前面的 stem；否则取整个文件 stem。 |
| `row` / `col` | 从标准文件名解析整数；非标准名留空。 |
| `grid_strategy` | 在父目录和祖父目录中查找 `grid_<列>x<行>`，保存为 `<列>x<行>`。 |
| `image_path` | 当前扫描到的图片路径字符串。 |

普通平铺图片文件夹也能计算局部特征，但标签、行列、grid 策略可能为空，空间特征通常无法计算。

**避免把输出特征图或无关图片放回输入目录**：递归扫描会把所有支持的图片都当成子块，无法仅凭内容判断哪些是真正的训练子块。

---

## 4. 主窗口：每个组件和选项

### 4.1 ① 子块文件夹

| 组件 | 工作方式 |
|---|---|
| 路径输入框 | 显示待扫描目录，可手动输入或粘贴；开始计算时检查目录是否存在。 |
| 浏览文件夹 | 打开目录选择器，并记住本次运行中最近使用的目录。 |
| Ctrl + O | 与浏览文件夹相同。 |

### 4.2 ② 特征组复选框

| 选项 | 默认 | 计算内容与维度 |
|---|---|---|
| 几何特征 | 勾选 | `CC/GPA/IGR/LGA/GPN`，5 列；使用 ExG + OTSU 冠层掩膜。 |
| 纹理特征 | 勾选 | GLCM 14 列 + LBP 3 列 + Gabor 2 列，共 19 列。 |
| 色彩特征 | 勾选 | BGR/HSV/Lab/YCrCb 通道统计 24 列 + GCV/YR 2 列，共 26 列。 |
| RGB 植被指数 | 勾选 | 21 个指数 × mean/std/range，共 63 列。 |
| 空间分布特征 | 勾选 | `GCCV/MI/CE/LAC`，4 列；按母图和 grid 策略聚合后回填。 |

可以只选一组进行消融。空间组不依赖几何复选框：即使只勾选空间组，也会单独计算各子块 CC 用于重建母图覆盖矩阵。

至少勾选一组才能开始。组名的内部标识依次为 `geometry`、`texture`、`color`、`vegetation`、`spatial`。

### 4.3 ③ 计算

| 组件 | 工作方式 |
|---|---|
| 开始计算特征 | 创建 `FeatureBatchWorker` 后台线程，递归扫描并逐图计算；每 20 张及最后一张更新一次进度。 |
| 进度条 | 显示已处理图片数/扫描总数；不是导出进度。 |
| 状态栏 | 完成后显示总行数、标签 1/0 数量和特征列数。父目录不是 0/1 的行在统计中会被归入“未吐水数量”的剩余项，因此有未知标签时应直接检查 CSV，而不是把剩余项全部视为真实 0。 |

当前主窗口没有“取消计算”按钮；`FeatureBatchWorker.cancel()` 是供代码扩展的接口。关闭程序前也应等待大任务结束，避免强制终止后台线程。

无法解码的单张图片不会中止整批：该图片仍产生元数据行，但特征为空。其他未捕获的批处理异常会在状态栏显示错误摘要。

### 4.4 特征数据表

- 每行一个子块，浮点数显示 4 位小数；这是显示格式，不是 CSV 精度截断。
- 单元格只读，可横向滚动查看特征列。
- 最多显示前 5000 行，**不会截断 CSV**。
- GUI 列名从前 200 行的字段并集生成；CSV 写出则从全部行的字段并集生成。因此，极少数“前 200 行均无有效特征、后面才有效”的数据，GUI 列可能与 CSV 不完全一致。

### 4.5 ④ 导出

| 组件 | 工作方式 |
|---|---|
| 导出 CSV + 特征说明 | 只有完成一次计算且有内存行时可用；默认路径为输入目录下的 `features.csv`，可另选位置。 |
| `features.csv` | UTF-8 文本，6 个元数据列在前，其后是全部行出现的特征列；不是 Excel 文件。 |
| `features_meta.json` | 固定写在所选 CSV 同目录，包含元数据说明、组级说明和本次启用组。 |

当前 `features_meta.json` 是**组级**说明，不会逐项展开全部 117 个字段。逐列定义请使用 `docs/FEATURE_REFERENCE.md`。

重复导出到相同文件名会覆盖 CSV；相同目录的 `features_meta.json` 也会被本次启用组覆盖。

### 4.6 ⑤ 参考大图与标签页

| 组件 | 工作方式 |
|---|---|
| 打开大图到特征图分析 | 选择 JPG/PNG/BMP/TIFF 等图片，保留原始数组用于计算，另建最长边 2048 的显示预览，并自动切换到特征图页。 |
| 特征数据表 | 数值特征库结果。 |
| 特征图分析 | 与数值子块计算独立的像素级研究界面。 |
| Ctrl + Q | 退出程序。 |

---

## 5. 五组数值特征：核心定义

这里给出快速索引，完整公式、单位和实现注意事项见 [FEATURE_REFERENCE.md](docs/FEATURE_REFERENCE.md)。

### 5.1 几何（5）

- `CC`：冠层像素/总像素；
- `GPA`：冠层像素数，单位 px；
- `IGR`：`1−CC`；
- `LGA`：最大背景 8 连通域面积；
- `GPN`：达到最小面积阈值的背景斑块数量。

### 5.2 纹理（19）

- GLCM：32 灰度级、距离 1/2、四个方向，7 属性各输出 mean/std；
- LBP：8 邻域、半径 1、uniform，输出熵/均值/标准差；
- Gabor：4 方向 × 2 频率，8 个平均绝对响应再求 mean/std。

### 5.3 色彩（26）

- `rgb_c1/c2/c3` 实际依次是 B/G/R，因为 OpenCV 输入为 BGR；
- HSV、Lab、YCrCb 各通道输出 mean/std；
- `GCV = std(G)/(mean(G)+ε)`；
- `YR` 为冠层掩膜内黄化像素比例。

### 5.4 植被指数（63）

包括 ExG、TGI、GBDI、MExG、ExR、ExGR、VEG、VDVI、CIVE、NGRDI、RGRI、VARI、MGRVI、EGRBDI、RGBRI、E-NGBDI、ExB、IKAW、COM1、COM2、GBRI。每个指数对有限像素响应输出 mean/std/range。

### 5.5 空间分布（4）

按 `(original_image, grid_strategy)` 将子块 CC 放回 `(row,col)` 矩阵：

- `GCCV`：CC 变异系数；
- `MI`：8 邻域 Moran’s I；
- `CE`：10 bin 覆盖直方图的熵形式；
- `LAC`：简化的空缺度 `(std/mean)²+1`。

同一母图、同一 grid 策略的所有行共享相同空间值。组内图片少于 4 张或存在非标准行列号时，该组不计算。

> **科研复现提醒**：当前 CE 使用 `density=True` 的直方图密度值而未归一化为概率；MI 使用 `np.roll`，边界存在循环邻接。二者与某些理论定义不完全相同，必须阅读特征参考第 8.3 节后再用于论文公式或跨数据集比较。

---

## 6. 特征图分析：每个选项

### 6.1 视图与叠加

| 组件 | 工作方式 |
|---|---|
| 特征选择列表 | 选择后立即在当前原图或 ROI 上同步重算；26 个名称见下表。 |
| 原图视图 | 显示当前分析区域的缩小副本；也是 ROI 框选区域。 |
| 特征图视图 | 将响应限制到 0～1 后应用 JET 色图，再按叠加比例与原图混合。 |
| 叠加透明度 | 0～100，默认 55。0 显示原图，100 只显示热力图，中间值使用 `addWeighted`。只改变显示，不改变响应数据。 |

所有特征图计算都在当前原始分辨率数组上进行；只有显示时缩小。大图上的计算在 GUI 主线程同步执行，调整滑条可能造成界面等待，不等于后台数值特征线程。

### 6.2 参数滑条

| 参数 | 范围 / 默认 | 实际影响 |
|---|---|---|
| 亮度差阈值 | 5～80 / 25 | `LocalBrightness` 的响应缩放；形态候选要求 V 高于局部背景该数值；并间接影响 Fusion。 |
| 高斯 sigma | 3～40 / 12 | `LocalBrightness` 和形态候选的背景高斯 sigma；`LocalContrast` 使用 `int(sigma)` 后强制为奇数的邻域边长；并间接影响融合图。Blob 的 3 组 DoG sigma 是固定值，不由此控制。 |
| Blob 最小面积 | 4～500 / 12 | 仅用于 `Morph_Circularity/Morph_Area/Morph_Aspect` 的轮廓面积过滤；**不用于 `Blob` DoG 响应图**。 |
| 圆度下限 | 20～90 / 45 | 换算为 0.20～0.90 写入 `FeatureParams.min_circularity`，但**当前形态计算没有读取该字段，因此滑条暂时不改变结果**。 |
| 内部 `blob_max_area` | 4000 | 形态轮廓最大面积，界面未暴露；`Morph_Area` 也用它归一化。 |
| 内部 `highlight_v` | 200 | Highlight 的 V 高光起点，界面未暴露。 |

Fusion 的内部权重为亮度、局部对比、Blob、低饱和各 0.25；GuttationEnhance 权重为 Highlight 0.35、LocalContrast 0.25、Blob 0.40。

### 6.3 ROI

| 按钮 | 工作方式 |
|---|---|
| 框选 ROI 后仅分析该区域 | 开启后在左侧原图视图拖出矩形，松开即换算回原图坐标并重算；后续参数和特征切换都只处理该区域。 |
| 清除 ROI（整图分析） | 清除内部 ROI 并恢复整图计算。 |

ROI 必须宽、高都大于 8 个显示像素才会被接受。`feature_report.json` 中 `roi` 保存相对原始整图的坐标；各特征的 `max_position` 则相对于**当前分析区域/ROI 裁剪图**，需要加上 ROI 左上坐标才能回到整图位置。

### 6.4 保存全部特征图 + 报告

- 对当前整图或 ROI 依次计算全部 26 个已注册特征；
- 每张响应图写成 `feature_<小写名称>.png`，数值从 0～1 映射为 0～255 灰度；
- 保存的是原始响应灰度图，不是当前 JET 叠加效果；
- `feature_report.json` 记录当前分析图尺寸、ROI，以及每个成功特征的最大响应、最大位置、平均响应和响应 >0.5 的像素数；
- 如果 CNN 依赖不可用，`compute_feature_map()` 返回 `None`，该特征被跳过，不写空图片或虚假统计；
- 输出目录已存在时，同名文件会被覆盖。

---

## 7. 26 种特征图分别如何工作

| 名称 | 计算与用途 | 主要相关参数 |
|---|---|---|
| `R` / `G` / `B` | 对应颜色通道除以 255，观察颜色响应。 | 无 |
| `H` | OpenCV 色相除以 180 归一化。 | 无 |
| `S` / `V` | 饱和度/亮度除以 255。 | 无 |
| `LAB_L` / `LAB_A` / `LAB_B` | Lab 三通道分别除以 255；OpenCV 8-bit Lab 编码不是直接物理单位。 | 无 |
| `Gray` | BGR 转灰度后除以 255。 | 无 |
| `LocalBrightness` | 灰度减去高斯背景，保留正差值，再按亮度差参数缩放并归一化；类似 top-hat，用于突出局部亮点。 | 亮度差、sigma |
| `LocalContrast` | 邻域灰度标准差，水珠边缘和纹理处响应较高。 | sigma 决定邻域 |
| `Highlight` | V 高于 200 的软响应乘以低饱和权重，用于高光。 | 内部 highlight_v |
| `Edge_Canny` | 固定阈值 50/150 的 Canny 二值边缘。 | 无 |
| `Edge_Sobel` | X/Y Sobel 梯度幅值后归一化。 | 无 |
| `Edge_Laplacian` | Laplacian 响应绝对值后归一化。 | 无 |
| `Blob` | `(2,4)、(4,8)、(8,16)` 三组 Difference-of-Gaussians 正响应的最大值，用于多尺度小亮点。 | 固定尺度 |
| `Morph_Circularity` | 对高亮低饱和候选轮廓计算 `4πA/P²`，把圆度填回轮廓区域。 | 亮度差、sigma、最小/最大面积 |
| `Morph_Area` | 将候选轮廓面积/4000 填回区域后再归一化。 | 同上 |
| `Morph_Aspect` | 对候选外接框计算 `1/(宽/高)` 并填回，越接近方形响应越高。 | 同上 |
| `HOG` | HOG 风格的梯度能量：Sobel 幅值缩成约 16×16 像素块统计，再上采样；不是完整定向直方图描述子。 | 无 |
| `LBP` | 8 邻域、半径 1、uniform LBP 编码图归一化。 | 无 |
| `ORB_Density` | 最多 1500 个 ORB 关键点，在关键点位累加后用 sigma=8 高斯扩散形成密度图。 | 无 |
| `Fusion` | 亮度、局部对比、Blob 和低饱和按各 0.25 加权，形成综合水珠线索响应。 | 亮度差、sigma |
| `GuttationEnhance` | Highlight×0.35 + LocalContrast×0.25 + Blob×0.40 的增强图。 | sigma；Highlight 内部阈值 |
| `CNN` | 可选 MobileNetV2 features：图片缩为 224×224，通道维取均值，再把低分辨率响应上采样。 | torch/torchvision、权重 |

局部邻域图对大图采用 2048 像素 tile + 64 像素重叠边缘计算，再保留中心区域，目的是减少边框伪影和内存压力；纯像素通道图直接整图计算。该机制不等于所有特征都分块，例如 CNN 仍先缩放整图。

---

## 8. 通用命令行批处理

`run_feature_extraction.py` 处理一个输入目录，已去除所有写死的数据路径。

### 8.1 示例

只输出 CSV 和 SQLite：

```bat
python run_feature_extraction.py ^
  --input "D:\data\labeled_crops" ^
  --output "D:\data\feature_results" ^
  --name tray_experiment_01 ^
  --formats csv sqlite
```

全特征、全格式、8 进程：

```bat
python run_feature_extraction.py ^
  --input "D:\data\labeled_crops" ^
  --output "D:\data\feature_results" ^
  --name tray_experiment_01 ^
  --groups geometry texture color vegetation spatial ^
  --formats csv parquet sqlite ^
  --workers 8 --chunksize 128
```

Windows CMD 中 `^` 是续行符；PowerShell 可写成一行，或改用反引号续行。

### 8.2 每个命令行选项

| 选项 | 必填 / 默认 | 工作方式 |
|---|---|---|
| `--input` | 必填 | 递归扫描的子块目录。 |
| `--output` | 必填 | 输出目录，不存在时创建。 |
| `--name` | 输入目录名 | CSV/Parquet 文件前缀及 SQLite 表名来源；必须是单段非空名称。SQLite 表名会把非字母数字下划线替换为 `_`，数字开头时加 `dataset_`。 |
| `--groups` | 五组全部 | 可选择一个或多个 `geometry texture color vegetation spatial`。 |
| `--formats` | `csv parquet sqlite` | 可任意组合。选择 Parquet 时要求 pyarrow。 |
| `--workers` | `min(16, CPU逻辑核心数)` | `ProcessPoolExecutor` 进程数，必须 ≥1。 |
| `--chunksize` | `128` | 每次分派给 worker 的图片任务批大小，必须 ≥1。 |

### 8.3 输出文件

- `<name>_features.csv`：UTF-8 with BOM；
- `<name>_features.parquet`：列式数据；
- `features.sqlite`：每个 name 对应一张表，使用 `if_exists="replace"`；
- `features_meta.json`：本次启用组和组级说明。

相同 name、相同输出目录会覆盖同名 CSV/Parquet，并替换同名 SQLite 表。需要保留多批实验时请使用不同 name 或输出目录。

---

## 9. multi-task 三任务批处理脚本

`run_feature_extraction_multi_task.py` 用于已有目录约定的病斑、徒长、黄化三个数据集。它只扫描指定 `grid_*` 目录，避免把平铺 `0/1` 中与 `grid_36x18` 重复的图片再次计算。

### 9.1 输入约定

```text
<base-dir>/
├── 病斑数据/
│   ├── grid_36x18/...
│   ├── grid_41x20/...
│   └── grid_48x28/...
├── 徒长数据/
└── 黄化数据/
```

### 9.2 示例

```bat
python run_feature_extraction_multi_task.py ^
  --base-dir "D:\data\labeling-multi-task" ^
  --output-dir "D:\data\multi-task-features" ^
  --tasks 病斑 徒长 黄化 ^
  --formats csv sqlite ^
  --workers 8
```

只处理黄化、只输出 CSV：

```bat
python run_feature_extraction_multi_task.py ^
  --base-dir "D:\data\labeling-multi-task" ^
  --tasks 黄化 --formats csv
```

### 9.3 每个命令行选项

| 选项 | 默认 | 工作方式 |
|---|---|---|
| `--base-dir` | 必填 | 包含三个任务数据目录的根目录。 |
| `--output-dir` | `<base-dir>/Features` | 输出位置。 |
| `--tasks` | 病斑、徒长、黄化 | 可选择一个或多个；每个任务使用自己的文件前缀和 SQLite 表。 |
| `--grids` | `grid_36x18 grid_41x20 grid_48x28` | 指定要扫描的 grid 目录名，可增减。 |
| `--formats` | `csv parquet sqlite` | 输出格式组合。 |
| `--workers` | `min(16, CPU逻辑核心数)` | 并行进程数。 |
| `--chunksize` | `128` | 并行任务批大小。 |

输出为 `病斑特征.*`、`徒长特征.*`、`黄化特征.*`，SQLite 文件为 `features_multi_task.sqlite`，表名分别为 `bingban`、`tuchang`、`huanghua`。每行额外包含 `task` 列，因此全特征时通常是 124 列（task + 6 元数据 + 117 特征）。

---

## 10. 源码组件地图

### 当前数值特征和特征图主流程

| 文件 / 类 | 职责 |
|---|---|
| `main.py` | 初始化日志和主窗口。 |
| `src/app.py` | 文件夹选择、五组复选框、后台计算、表格、CSV 导出和参考图加载。 |
| `src/batch_features.py` | 数值特征后台线程、母图空间回填、CSV 和组级元数据。 |
| `src/folder_loader.py` | 递归扫描、标签/母图/行列/grid 解析。 |
| `src/features_v2/__init__.py` | 特征组注册和子块特征调度。 |
| `src/features_v2/geometry.py` | ExG+OTSU 冠层掩膜与 5 个几何特征。 |
| `src/features_v2/texture2.py` | 19 个 GLCM/LBP/Gabor 特征。 |
| `src/features_v2/color2.py` | 24 个通道统计、GCV 和 YR。 |
| `src/features_v2/vegetation.py` | 21 个植被指数及 63 个统计列。 |
| `src/features_v2/spatial.py` | GCCV、MI、CE、LAC。 |
| `src/feature_tab.py` | 特征图列表、滑条、叠加、ROI 和批量保存。 |
| `src/feature_maps.py` | 26 个响应图注册与具体算法、tile 工具、可选 CNN。 |
| `src/image_canvas.py` | 参考大图 Unicode 读取和显示预览；保留网格选择接口。 |
| `src/grid_manager.py` | 网格坐标生成和位置查询。 |
| `src/image_io.py` | Unicode 安全图片读写。 |
| `src/error_reporting.py` | 日志、阶段和异常对话框。 |
| `run_feature_extraction.py` | 通用多进程命令行批处理及 CSV/Parquet/SQLite。 |
| `run_feature_extraction_multi_task.py` | 三任务、多 grid 的专用批处理。 |

### 保留的旧版/兼容模块

- `src/features/`：旧版逐格色彩、纹理、几何提取器，与当前 `features_v2` 的列定义不同；
- `src/batch_processor.py`：旧版按整图 M×N 重新分块、批量写 CSV/XLSX 的线程，当前主窗口不调用；
- `src/visualization/plot_canvas.py`：旧版 Matplotlib 直方图、柱状图、雷达图、热力图等组件，当前特征图页使用 OpenCV/QPixmap；
- `src/utils.py`：旧版图片文件收集工具；
- `src/image_canvas.py` 的网格选中接口和 `src/grid_manager.py` 部分方法主要为旧版可视化保留。

新研究字段应优先加入 `features_v2`，不要同时修改旧版 `features` 后误以为主窗口会使用它。

---

## 11. 日志、性能与故障排查

### 日志

`logs/app.log` 记录运行过程，`logs/errors.log` 记录错误和 traceback。加载参考图、导出 CSV 等阶段有 `StageTracker` 标记。批处理 worker 的部分错误只发回状态栏，排查时应同时查看控制台和日志。

### 没有 123 列

依次检查：

1. 是否五组都勾选或都在 `--groups` 中；
2. 是否有图片解码失败；
3. 文件名是否符合 `_r数字_c数字`；
4. 每个母图+grid 组是否至少 4 张；
5. 是否整个数据集都没有计算出空间 4 列。

### 标签全部为空或空间特征为空

标签只来自直接父目录 `0/1`。空间特征还依赖标准文件名、行列和 grid 分组。普通平铺目录可以算局部特征，但通常不能重建母图空间矩阵。

### Parquet 报错

安装 `requirements-batch.txt`，或改用 `--formats csv sqlite`。GUI 本身不输出 Parquet，因此不受此依赖影响。

### CNN 没有图片

检查 `torch`、`torchvision` 是否可导入，以及 MobileNetV2 默认权重是否能取得。CNN 不可用时会被明确跳过，不影响其他 25 种响应图。

### 大图调整滑条很卡

特征图在 GUI 主线程按原始分辨率同步重算。可先框选 ROI，或减少不必要的整图计算；数值特征后台线程与这是两条不同路径。

### 科研数据切分

同一母图的子块共享相似内容，且空间 4 列会在母图内重复。训练/验证/测试划分建议按 `original_image` 分组，而不是简单逐行随机，避免数据泄漏。
