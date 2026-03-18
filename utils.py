from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor_to_hwc_uint8(img_chw: torch.Tensor) -> torch.Tensor:
    """Convert CHW float tensor in [0,1] to HWC uint8 for visualization."""
    img = img_chw.permute(1, 2, 0).clamp(0.0, 1.0)  # HWC
    return (img * 255.0).to(torch.uint8)
