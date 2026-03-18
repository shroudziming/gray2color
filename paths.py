from __future__ import annotations

from pathlib import Path


def resolve_landscape_dirs(root: Path | str = Path(".")) -> tuple[Path, Path]:
    """Resolve dataset directories in this workspace.

    Expected structure (your repo):
      input/landscape Images/color
      input/landscape Images/gray

    Returns:
        (color_dir, gray_dir)
    """
    root = Path(root)
    candidates = [
        root / "input" / "landscape Images",
        root / "input" / "landscape-image-colorization" / "landscape Images",
        root / "./input" / "landscape Images",
        root / "./input" / "landscape-image-colorization" / "landscape Images",
    ]

    tried: list[str] = []
    for base in candidates:
        color_dir = base / "color"
        gray_dir = base / "gray"
        tried.append(str(base))
        if color_dir.exists() and gray_dir.exists():
            return color_dir, gray_dir

    hint = "input/landscape Images/color and input/landscape Images/gray"
    raise FileNotFoundError(
        "Could not find dataset folders. Expected both 'color' and 'gray' under one of:\n"
        + "\n".join(tried)
        + f"\n\nWorkspace hint: your repo seems to have {hint}."
    )
