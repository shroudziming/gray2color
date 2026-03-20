from __future__ import annotations

import torch
import torch.nn.functional as F


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor_to_hwc_uint8(img_chw: torch.Tensor) -> torch.Tensor:
    """Convert CHW float tensor in [0,1] to HWC uint8 for visualization."""
    img = img_chw.permute(1, 2, 0).clamp(0.0, 1.0)  # HWC
    return (img * 255.0).to(torch.uint8)


def ssim_value(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    kernel_size: int = 11,
) -> torch.Tensor:
    """Differentiable SSIM scalar for NCHW images in [0, data_range]."""
    if pred.ndim != 4 or target.ndim != 4:
        raise ValueError("ssim_value expects NCHW tensors")
    if pred.shape != target.shape:
        raise ValueError("pred and target must have same shape")

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    x = pred.float()
    y = target.float()
    pad = kernel_size // 2

    mu_x = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, kernel_size=kernel_size, stride=1, padding=pad)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.avg_pool2d(x * x, kernel_size=kernel_size, stride=1, padding=pad) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(y * y, kernel_size=kernel_size, stride=1, padding=pad) - mu_y_sq
    sigma_xy = F.avg_pool2d(x * y, kernel_size=kernel_size, stride=1, padding=pad) - mu_xy

    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = num / den.clamp_min(1e-12)
    return ssim_map.mean()


def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    kernel_size: int = 11,
) -> torch.Tensor:
    """SSIM loss = 1 - SSIM."""
    return 1.0 - ssim_value(pred, target, data_range=data_range, kernel_size=kernel_size)
