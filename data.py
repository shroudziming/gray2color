from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset


def _sorted_alphanumeric(names: Iterable[str]) -> list[str]:
    convert = lambda text: int(text) if text.isdigit() else text.lower()  # noqa: E731

    def alphanum_key(key: str):
        return [convert(c) for c in re.split(r"([0-9]+)", key)]

    return sorted(list(names), key=alphanum_key)


def _list_image_files(folder: Path) -> list[str]:
    exts = {".jpg", ".jpeg", ".png"}
    names = [p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return _sorted_alphanumeric(names)


@dataclass(frozen=True)
class DatasetConfig:
    image_size: int = 160
    max_images: int | None = None


class LandscapeColorizationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Pairs (gray_rgb, color_rgb) as float tensors in [0,1], shape (3,H,W)."""

    def __init__(self, color_dir: Path, gray_dir: Path, cfg: DatasetConfig = DatasetConfig()):
        self.color_dir = Path(color_dir)
        self.gray_dir = Path(gray_dir)
        self.cfg = cfg

        color_files = set(_list_image_files(self.color_dir))
        gray_files = set(_list_image_files(self.gray_dir))
        common = _sorted_alphanumeric(color_files.intersection(gray_files))

        if cfg.max_images is not None:
            common = common[: cfg.max_images]

        if not common:
            raise RuntimeError(
                f"No paired images found. color={self.color_dir} gray={self.gray_dir}"
            )

        self.filenames = common

    def __len__(self) -> int:
        return len(self.filenames)

    def _load_rgb_tensor(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = img.resize((self.cfg.image_size, self.cfg.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3)
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,H,W)
        return t

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.filenames[idx]
        color = self._load_rgb_tensor(self.color_dir / name)
        gray = self._load_rgb_tensor(self.gray_dir / name)
        return gray, color


def make_splits(
    dataset: Dataset,
    train_count: int = 5500,
) -> tuple[Subset, Subset]:
    n = len(dataset)
    train_count = min(train_count, n)
    train_idx = list(range(0, train_count))
    test_idx = list(range(train_count, n))
    return Subset(dataset, train_idx), Subset(dataset, test_idx)


def make_loaders(
    dataset: Dataset,
    batch_size: int = 50,
    num_workers: int = 0,
    train_count: int = 5500,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int = 2,
) -> tuple[DataLoader, DataLoader]:
    train_set, test_set = make_splits(dataset, train_count=train_count)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    loader_kwargs: dict = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, test_loader
