from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from .data import DatasetConfig, LandscapeColorizationDataset, make_loaders
from .paths import resolve_landscape_dirs
from .unet import UNet
from .utils import get_device, ssim_value, tensor_to_hwc_uint8

def _configure_torch_for_speed(*, enable_tf32: bool, enable_cudnn_benchmark: bool) -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(enable_cudnn_benchmark)
        torch.backends.cuda.matmul.allow_tf32 = bool(enable_tf32)
        torch.backends.cudnn.allow_tf32 = bool(enable_tf32)
    try:
        # PyTorch 2.x: can improve matmul perf on supported GPUs.
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def _save_predictions_grid(x: torch.Tensor, y: torch.Tensor, pred: torch.Tensor, path: Path, *, n_vis: int = 8) -> None:
    """Save an image grid: columns are [GT | Gray(Input) | Pred]."""
    n = min(int(x.shape[0]), int(y.shape[0]), int(pred.shape[0]), int(n_vis))
    if n <= 0:
        return

    x = x[:n]
    y = y[:n]
    pred = pred[:n]

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

        axes[i][2].imshow(tensor_to_hwc_uint8(pred[i]))
        axes[i][2].set_title("Predicted")
        axes[i][2].axis("off")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def _eval_metrics(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    use_amp: bool,
    loss_mode: str,
    l1_weight: float,
    ssim_weight: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_l1 = 0.0
    total_ssim = 0.0
    count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(use_amp) and device.type == "cuda"):
            pred = model(x)
            loss, l1, ssim = _compute_loss(
                pred,
                y,
                loss_mode=loss_mode,
                l1_weight=l1_weight,
                ssim_weight=ssim_weight,
            )
        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        total_l1 += float(l1.item()) * bs
        total_ssim += float(ssim.item()) * bs
        count += bs

    denom = max(count, 1)
    return {
        "loss": total_loss / denom,
        "l1": total_l1 / denom,
        "ssim": total_ssim / denom,
    }


def _compute_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    *,
    loss_mode: str,
    l1_weight: float,
    ssim_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    l1 = F.l1_loss(pred, y)
    ssim = ssim_value(pred, y)
    mode = loss_mode.lower()

    if mode == "l1":
        loss = l1
    elif mode == "ssim":
        loss = 1.0 - ssim
    elif mode in {"l1+ssim", "l1_ssim", "l1ssim"}:
        norm = max(float(l1_weight) + float(ssim_weight), 1e-8)
        loss = (float(l1_weight) * l1 + float(ssim_weight) * (1.0 - ssim)) / norm
    else:
        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

    return loss, l1, ssim


def train(
    root: str | Path = Path("."),
    image_size: int = 160,
    max_images: int | None = None,
    train_count: int = 5500,
    batch_size: int = 50,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int = 2,
    epochs: int = 10,
    lr: float = 1e-3,
    base_channels: int = 64,
    concat_input: bool = True,
    amp: bool | None = None,
    tf32: bool = True,
    cudnn_benchmark: bool = True,
    compile_model: bool = False,
    save_predictions_each_epoch: bool = True,
    predictions_subdir: str = "predictions_by_epoch",
    predictions_n_vis: int = 8,
    out_dir: str | Path = Path("outputs"),
    loss_mode: str = "l1",
    l1_weight: float = 0.8,
    ssim_weight: float = 0.2,
) -> dict:
    root = Path(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    color_dir, gray_dir = resolve_landscape_dirs(root)
    cfg = DatasetConfig(image_size=image_size, max_images=max_images)
    ds = LandscapeColorizationDataset(color_dir=color_dir, gray_dir=gray_dir, cfg=cfg)
    train_loader, test_loader = make_loaders(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        train_count=train_count,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    device = get_device()
    _configure_torch_for_speed(enable_tf32=tf32, enable_cudnn_benchmark=cudnn_benchmark)
    use_amp = (device.type == "cuda") if amp is None else bool(amp)
    model = UNet(
        in_channels=3,
        out_classes=3,
        base_channels=base_channels,
        concat_input=concat_input,
        output_activation="sigmoid",
    ).to(device)

    if compile_model:
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
        except Exception as e:
            print("torch.compile unavailable/failed, continuing without compile:", repr(e))

    mode = loss_mode.lower()
    if mode not in {"l1", "ssim", "l1+ssim", "l1_ssim", "l1ssim"}:
        raise ValueError("loss_mode must be one of: l1 | ssim | l1+ssim")

    optim = Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(device.type, enabled=bool(use_amp) and device.type == "cuda")

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_l1": [],
        "train_ssim": [],
        "test_loss": [],
        "test_l1": [],
        "test_ssim": [],
    }

    best = float("inf")
    best_path = out_dir / "unet_best.pt"
    last_path = out_dir / "unet_last.pt"
    pred_dir = out_dir / predictions_subdir

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_l1 = 0.0
        running_ssim = 0.0
        seen = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(use_amp) and device.type == "cuda"):
                pred = model(x)
                loss, l1, ssim = _compute_loss(
                    pred,
                    y,
                    loss_mode=mode,
                    l1_weight=l1_weight,
                    ssim_weight=ssim_weight,
                )

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            bs = x.shape[0]
            running_loss += float(loss.item()) * bs
            running_l1 += float(l1.item()) * bs
            running_ssim += float(ssim.item()) * bs
            seen += bs
            pbar.set_postfix(train_loss=running_loss / max(seen, 1), train_l1=running_l1 / max(seen, 1))

        train_loss = running_loss / max(seen, 1)
        train_l1 = running_l1 / max(seen, 1)
        train_ssim = running_ssim / max(seen, 1)
        test_metrics = (
            _eval_metrics(
                model,
                test_loader,
                device,
                use_amp=use_amp,
                loss_mode=mode,
                l1_weight=l1_weight,
                ssim_weight=ssim_weight,
            )
            if len(test_loader.dataset)
            else {"loss": float("nan"), "l1": float("nan"), "ssim": float("nan")}
        )
        test_loss = test_metrics["loss"]
        test_l1 = test_metrics["l1"]
        test_ssim = test_metrics["ssim"]

        history["train_loss"].append(float(train_loss))
        history["train_l1"].append(float(train_l1))
        history["train_ssim"].append(float(train_ssim))
        history["test_loss"].append(float(test_loss))
        history["test_l1"].append(float(test_l1))
        history["test_ssim"].append(float(test_ssim))

        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "cfg": asdict(cfg),
                "train_count": train_count,
            },
            last_path,
        )

        # Per-epoch qualitative visualization using current model.
        if save_predictions_each_epoch:
            model.eval()
            with torch.no_grad():
                vis_loader = test_loader if len(test_loader.dataset) else train_loader
                x_vis, y_vis = next(iter(vis_loader))
                x_vis = x_vis.to(device, non_blocking=True)
                y_vis = y_vis.to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=bool(use_amp) and device.type == "cuda",
                ):
                    pred_vis = model(x_vis).clamp(0.0, 1.0)

                out_path = pred_dir / f"epoch_{epoch:03d}.png"
                _save_predictions_grid(
                    x_vis.detach().cpu(),
                    y_vis.detach().cpu(),
                    pred_vis.detach().cpu(),
                    out_path,
                    n_vis=predictions_n_vis,
                )

        if test_loss == test_loss and test_loss < best:  # not NaN
            best = test_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "cfg": asdict(cfg),
                    "train_count": train_count,
                },
                best_path,
            )

    _save_loss_curve(history, out_dir / "loss_curve.png")

    info = {
        "device": str(device),
        "color_dir": str(color_dir),
        "gray_dir": str(gray_dir),
        "dataset_size": len(ds),
        "train_size": len(train_loader.dataset),
        "test_size": len(test_loader.dataset),
        "config": asdict(cfg),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "loss_mode": mode,
        "l1_weight": l1_weight,
        "ssim_weight": ssim_weight,
        "history": history,
        "last_checkpoint": str(last_path),
        "best_checkpoint": str(best_path),
    }

    (out_dir / "history.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def _save_loss_curve(history: dict[str, list[float]], path: Path) -> None:
    train_loss = history.get("train_loss", [])
    test_loss = history.get("test_loss", [])
    train_l1 = history.get("train_l1", [])
    test_l1 = history.get("test_l1", [])
    train_ssim = history.get("train_ssim", [])
    test_ssim = history.get("test_ssim", [])
    if not train_loss:
        return

    xs = list(range(1, len(train_loss) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(xs, train_loss, label="train_loss")
    if any(v == v for v in test_loss):
        axes[0].plot(xs, test_loss, label="test_loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("objective")
    axes[0].set_title("Training Objective")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if train_l1:
        axes[1].plot(xs, train_l1, label="train_l1")
    if any(v == v for v in test_l1):
        axes[1].plot(xs, test_l1, label="test_l1")
    if train_ssim:
        axes[1].plot(xs, train_ssim, label="train_ssim")
    if any(v == v for v in test_ssim):
        axes[1].plot(xs, test_ssim, label="test_ssim")
    axes[1].set_xlabel("epoch")
    axes[1].set_title("L1 / SSIM")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    info = train(epochs=10)
    print("Done. Last checkpoint:", info["last_checkpoint"])


if __name__ == "__main__":
    main()
