from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from .data import DatasetConfig, LandscapeColorizationDataset, make_splits
from .paths import resolve_landscape_dirs
from .unet import UNet
from .utils import get_device, tensor_to_hwc_uint8

@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """PSNR for images in [0,1]. Returns per-batch scalar."""
    mse = torch.mean((pred - target) ** 2)
    return 10.0 * torch.log10(1.0 / (mse + eps))


@torch.no_grad()
def evaluate(
    root: str | Path = Path("."),
    checkpoint: str | Path = Path("outputs") / "unet_last.pt",
    image_size: int = 160,
    max_images: int | None = None,
    train_count: int = 5500,
    batch_size: int = 50,
    base_channels: int = 64,
    concat_input: bool = True,
    out_dir: str | Path = Path("outputs"),
    n_vis: int = 8,
    vis_shuffle: bool = True,   # 是否打乱用于可视化的样本
    vis_seed: int = 0,          # 换个 seed 就能换一批图
) -> dict:
    root = Path(root)
    checkpoint = Path(checkpoint)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    color_dir, gray_dir = resolve_landscape_dirs(root)
    cfg = DatasetConfig(image_size=image_size, max_images=max_images)
    ds = LandscapeColorizationDataset(color_dir=color_dir, gray_dir=gray_dir, cfg=cfg)

    # 评估用：固定顺序（不打乱）
    _, test_set = make_splits(ds, train_count=train_count)
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = get_device()
    model = UNet(
        in_channels=3,
        out_classes=3,
        base_channels=base_channels,
        concat_input=concat_input,
        output_activation="sigmoid",
    ).to(device)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    l1_fn = nn.L1Loss(reduction="mean")

    total_l1 = 0.0
    total_psnr = 0.0
    count = 0

    # 1) 指标：跑完整个 test_loader
    for x, y in tqdm(test_loader, desc="Evaluating"):
        x = x.to(device)
        y = y.to(device)
        pred = model(x).clamp(0.0, 1.0)

        l1 = l1_fn(pred, y)
        p = psnr(pred, y)

        bs = x.shape[0]
        total_l1 += float(l1.item()) * bs
        total_psnr += float(p.item()) * bs
        count += bs

    avg_l1 = total_l1 / max(count, 1)
    avg_psnr = total_psnr / max(count, 1)

    # 2) 可视化：单独用一个 loader，默认打乱，换 seed 就换图
    gen = torch.Generator()
    gen.manual_seed(vis_seed)

    vis_loader = DataLoader(
        test_set,
        batch_size=min(batch_size, max(1, n_vis)),
        shuffle=vis_shuffle,
        generator=gen if vis_shuffle else None,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    vis_batches = []
    got = 0
    for x, y in vis_loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x).clamp(0.0, 1.0)

        vis_batches.append((x.detach().cpu(), y.detach().cpu(), pred.detach().cpu()))
        got += x.shape[0]
        if got >= n_vis:
            break

    vis_path = out_dir / "predictions.png"
    _save_visualization(vis_batches, vis_path, n_vis=n_vis)

    metrics = {
        "checkpoint": str(checkpoint),
        "device": str(device),
        "test_size": count,
        "mae_l1": avg_l1,
        "psnr": avg_psnr,
        "predictions_png": str(vis_path),
        "vis_shuffle": vis_shuffle,
        "vis_seed": vis_seed,
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def _save_visualization(vis_batches, path: Path, n_vis: int = 8) -> None:
    if not vis_batches:
        return

    xs, ys, ps = [], [], []
    for x, y, p in vis_batches:
        xs.append(x)
        ys.append(y)
        ps.append(p)

    x = torch.cat(xs, dim=0)[:n_vis]
    y = torch.cat(ys, dim=0)[:n_vis]
    p = torch.cat(ps, dim=0)[:n_vis]

    n = x.shape[0]
    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(9, max(2, n * 2)))
    if n == 1:
        axes = [axes]

    for i in range(n):
        axes[i][0].imshow(tensor_to_hwc_uint8(y[i]))
        axes[i][0].set_title("Color (GT)")
        axes[i][0].axis("off")

        axes[i][1].imshow(tensor_to_hwc_uint8(x[i]))
        axes[i][1].set_title("Gray (Input)")
        axes[i][1].axis("off")

        axes[i][2].imshow(tensor_to_hwc_uint8(p[i]))
        axes[i][2].set_title("Predicted")
        axes[i][2].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    m = evaluate()
    print("MAE(L1):", m["mae_l1"], "PSNR:", m["psnr"])
    print("Saved:", m["predictions_png"])


if __name__ == "__main__":
    main()