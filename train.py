from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from .data import DatasetConfig, LandscapeColorizationDataset, make_loaders
from .paths import resolve_landscape_dirs
from .unet import UNet
from .utils import get_device, tensor_to_hwc_uint8

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
def _eval_l1(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    use_amp: bool,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(use_amp) and device.type == "cuda"):
            pred = model(x)
            loss = loss_fn(pred, y)
        bs = x.shape[0]
        total += float(loss.item()) * bs
        count += bs
    return total / max(count, 1)


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

    loss_fn = nn.L1Loss()
    optim = Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(device.type, enabled=bool(use_amp) and device.type == "cuda")

    history: dict[str, list[float]] = {"train_l1": [], "test_l1": []}

    best = float("inf")
    best_path = out_dir / "unet_best.pt"
    last_path = out_dir / "unet_last.pt"
    pred_dir = out_dir / predictions_subdir

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        seen = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(use_amp) and device.type == "cuda"):
                pred = model(x)
                loss = loss_fn(pred, y)

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            bs = x.shape[0]
            running += float(loss.item()) * bs
            seen += bs
            pbar.set_postfix(train_l1=running / max(seen, 1))

        train_l1 = running / max(seen, 1)
        test_l1 = (
            _eval_l1(model, test_loader, loss_fn, device, use_amp=use_amp)
            if len(test_loader.dataset)
            else float("nan")
        )

        history["train_l1"].append(float(train_l1))
        history["test_l1"].append(float(test_l1))

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

        if test_l1 == test_l1 and test_l1 < best:  # not NaN
            best = test_l1
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
        "history": history,
        "last_checkpoint": str(last_path),
        "best_checkpoint": str(best_path),
    }

    (out_dir / "history.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def _save_loss_curve(history: dict[str, list[float]], path: Path) -> None:
    train_l1 = history.get("train_l1", [])
    test_l1 = history.get("test_l1", [])
    if not train_l1:
        return

    xs = list(range(1, len(train_l1) + 1))
    plt.figure(figsize=(6, 4))
    plt.plot(xs, train_l1, label="train_l1")
    if any(v == v for v in test_l1):
        plt.plot(xs, test_l1, label="test_l1")
    plt.xlabel("epoch")
    plt.ylabel("L1 (MAE)")
    plt.title("Training Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    info = train(epochs=10)
    print("Done. Last checkpoint:", info["last_checkpoint"])


if __name__ == "__main__":
    main()
