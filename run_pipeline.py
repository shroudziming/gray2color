from __future__ import annotations

import sys
from pathlib import Path

from .eval import evaluate
from .paths import resolve_landscape_dirs
from .train import train


def verify_data(root: str | Path = Path(".")) -> None:
    color_dir, gray_dir = resolve_landscape_dirs(root)
    print("Using color dir:", color_dir.resolve())
    print("Using gray dir:", gray_dir.resolve())

    exts = {".jpg", ".jpeg", ".png"}
    n_color = sum(1 for p in color_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)
    n_gray = sum(1 for p in gray_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)
    print("Found:", n_color, "color images;", n_gray, "gray images")


def main():
    root = Path(".")
    verify_data(root)

    mode = (sys.argv[1] if len(sys.argv) > 1 else "full").strip().lower()
    loss_mode = (sys.argv[2] if len(sys.argv) > 2 else "l1").strip().lower()
    if loss_mode not in {"l1", "ssim", "l1+ssim", "l1_ssim", "l1ssim"}:
        raise SystemExit("Usage: python -m mygray2color.run_pipeline [full|quick] [l1|ssim|l1+ssim]")
    l1_weight = 0.8
    ssim_weight = 0.2

    if mode == "quick":
        cfg = {
            "epochs": 1,
            "max_images": 200,
            "train_count": 150,
            "batch_size": 10,
            "base_channels": 32,
            "loss_mode": loss_mode,
            "l1_weight": l1_weight,
            "ssim_weight": ssim_weight,
        }
    elif mode == "full":
        cfg = {
            "epochs": 50,
            "max_images": None,
            "train_count": 5500,
            "batch_size": 50,
            "base_channels": 64,
            "loss_mode": loss_mode,
            "l1_weight": l1_weight,
            "ssim_weight": ssim_weight,
        }
    else:
        raise SystemExit("Usage: python -m mygray2color.run_pipeline [full|quick] [l1|ssim|l1+ssim]")

    print("Running mode:", mode)
    print("Train cfg:", cfg)

    info = train(root=root, out_dir="outputs", **cfg)
    metrics = evaluate(
        root=root,
        checkpoint=info["last_checkpoint"],
        out_dir="outputs",
        max_images=cfg["max_images"],
        train_count=cfg["train_count"],
        batch_size=cfg["batch_size"],
        base_channels=cfg["base_channels"],
        n_vis=8,
        loss_mode=cfg["loss_mode"],
        l1_weight=cfg["l1_weight"],
        ssim_weight=cfg["ssim_weight"],
    )

    print("\nSummary")
    print("- Checkpoint:", metrics["checkpoint"])
    print("- Objective:", metrics["objective"], f"(mode={metrics['loss_mode']})")
    print("- MAE(L1):", metrics["mae_l1"])
    print("- SSIM:", metrics["ssim"])
    print("- PSNR:", metrics["psnr"])
    print("- Predictions grid:", metrics["predictions_png"])


if __name__ == "__main__":
    main()
