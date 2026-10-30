# AutoML-Platform GPU 环境（automl-platform-gpu）搭建记录

> 本文档记录 `automl-platform-gpu` 环境的**实际可用配置**（已在本机 RTX 4060 上验证通过）。

## 1. 硬件与驱动

- GPU：NVIDIA GeForce RTX 4060（Ada Lovelace，计算能力 sm_89，8GB 显存）
- 驱动：591.86（最高支持 CUDA 13.1，向下兼容 12.x）
- 无需单独安装系统级 CUDA Toolkit：PyTorch / XGBoost / CatBoost 的 pip wheel 自带 CUDA 运行库。

## 2. 版本选择（实测通过）

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | 3.12.13 | 与生产环境一致 |
| torch | **2.9.1+cu128** | 官方 cu128 wheel，自带 CUDA 12.8 运行库 |
| torchvision / torchaudio | 0.24.1 / 2.9.1（cu128） | 与 torch 版本锁定（表格 Transformer 只用 torch，可选） |
| xgboost | 3.1.3 | wheel 自带 GPU 支持 |
| catboost | 1.2.10 | 自带 GPU 支持，Windows 开箱即用 |
| lightgbm | 4.6.0 | Windows 官方 wheel 仅 CPU |
| autogluon | 1.5.0 | 自动复用 GPU 版 torch |
| h2o | 3.46.0.10 | Java 后端，Windows 不支持 GPU |
| 其余 | 见 `requirements-gpu.txt` | 与生产环境一致 |

## 3. 安装步骤

```bash
conda env create -f environment_gpu.yml
conda activate automl-platform-gpu

# 验证
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

期望输出：`2.9.1+cu128 True NVIDIA GeForce RTX 4060`

## 4. 踩坑记录（重要）

### 4.1 conda 镜像 403
清华 TUNA 的 `anaconda/pkgs/free` 与 `pkgs/main` 当前均返回 403（Anaconda 收紧镜像授权）。
**解决**：将 `~/.condarc` 的 channels 改为仅 `defaults`（官方源 `repo.anaconda.com`）。

### 4.2 scikit-learn 版本冲突
- `scikit-survival 0.27.0` 名义上要求 `scikit-learn>=1.8.0`；
- 但 `autogluon 1.5.0` 会把 `scikit-learn` 钉到 `1.7.2`。
- **结论**：代码中**并未 import `sksurv`**（scikit-survival 仅在 requirements 中声明、未被使用），该冲突为纯元数据警告、无运行时影响，与生产环境状态一致。最终采用 `scikit-learn==1.7.2`。

### 4.3 numpy 版本
torch cu128 带入 numpy 2.4.x，随后 autogluon 将其降到 **2.3.5**，与生产环境兼容。

## 5. 各库 GPU 使用方式

| 库 | GPU | 启用方式 |
|---|---|---|
| PyTorch（Transformer 页） | ✅ | 代码自动检测 cuda，自动上 GPU（提速最明显） |
| XGBoost | ✅ | `XGBClassifier(tree_method="hist", device="cuda")` |
| CatBoost | ✅ | `CatBoostClassifier(task_type="GPU")` |
| AutoGluon | ✅ | 其 torch 系模型自动用 GPU |
| LightGBM | ⚠️ CPU | Windows 官方 wheel 无 GPU；如需可源码编译（不建议） |
| H2O | ❌ | Windows 不支持 GPU，调小 time_limit 即可 |

## 6. 切换与回退

- `run.bat` 已更新：优先 `conda activate automl-platform-gpu`，失败则依次回退 `automl-platform`。
