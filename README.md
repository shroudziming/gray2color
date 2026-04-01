# mygray2color

一个最小可复现的“灰度图 → 彩色图”图像上色项目（PyTorch + U-Net）。

## 功能

- 数据集读取：按文件名配对读取 `gray/` 与 `color/`（取交集）
- 训练：保存 `outputs/unet_last.pt`、`outputs/unet_best.pt`、训练曲线与每个 epoch 的预测网格图
- 评估：输出 Objective / MAE(L1) / SSIM / PSNR，并保存预测效果 `outputs/predictions.png`
- 一键流程：训练 + 评估（见 `run_pipeline.py`）

## 目录约定（数据）

代码默认从仓库根目录向下寻找以下结构（两种都支持）：

- `input/landscape Images/color`
- `input/landscape Images/gray`

或（如果你用脚本下载后保持原始目录名）：

- `input/landscape-image-colorization/landscape Images/color`
- `input/landscape-image-colorization/landscape Images/gray`

注意：

- `color/` 与 `gray/` 下的图片文件名需要一一对应（数据加载时以“共同文件名”配对）
- 仅识别扩展名：`.jpg/.jpeg/.png`

## 数据来源

本项目使用的示例数据集来自 Kaggle：

- Dataset：**Landscape Image Colorization**
- Kaggle ID：`theblackmamba31/landscape-image-colorization`
- 链接：https://www.kaggle.com/datasets/theblackmamba31/landscape-image-colorization

数据下载脚本（使用 `kagglehub`）：

```python
import kagglehub

path = kagglehub.dataset_download("theblackmamba31/landscape-image-colorization")

print("Path to dataset files:", path)
```

请遵守 Kaggle 及该数据集页面中给出的许可协议与使用条款；如用于课程作业/报告，建议在报告与仓库文档中同时保留上述来源信息。

## 安装依赖

最小依赖主要包括：`torch`、`numpy`、`Pillow`、`matplotlib`、`tqdm`。

示例：

```bash
pip install torch numpy pillow matplotlib tqdm
```

（如需 GPU 版 PyTorch，请按你的 CUDA/驱动版本参考 PyTorch 官方安装指引。）

## 运行方式（从仓库根目录执行）

### 1) 一键训练 + 评估

```bash
python -m run_pipeline quick l1
# 或
python -m run_pipeline full l1+ssim
```

- `quick`：更少 epoch/图片数，适合验证环境与流程
- `full`：更接近完整训练（默认 50 epochs，参数见 `run_pipeline.py`）
- 第二个参数是训练/评估目标：`l1`、`ssim`、`l1+ssim`

### 2) 仅训练

```bash
python -m train
```

### 3) 仅评估（需要已有 checkpoint）

```bash
python -m eval
```

## 输出文件

训练与评估的默认输出目录是仓库根目录下的 `outputs/`：

- `outputs/unet_last.pt`：最后一次保存的 checkpoint
- `outputs/unet_best.pt`：测试集 L1 更优时保存的 checkpoint
- `outputs/loss_curve.png`：训练曲线
- `outputs/predictions.png`：评估阶段可视化（GT / Gray / Pred）
- `outputs/predictions_by_epoch/epoch_*.png`：每个 epoch 的可视化网格图

## 参考

- Kaggle Notebook（本项目主要参考来源）：https://www.kaggle.com/code/theblackmamba31/autoencoder-grayscale-to-color-image/notebook
