from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".gif",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
}

OUTPUT_DIR_NAME = "_classified"


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def iter_images(root: Path, output_dir: Path) -> Iterator[Path]:
    """Yield images under root in sorted order, skipping output folders.

    Sorted traversal keeps the order deterministic and walks each directory's
    files together, which is friendlier to spinning disks.
    """
    output_dir = output_dir.resolve()
    for current_root, dirnames, filenames in os.walk(root):
        current_path = Path(current_root)
        dirnames[:] = sorted(
            dirname
            for dirname in dirnames
            if dirname != OUTPUT_DIR_NAME
            and (current_path / dirname).resolve() != output_dir
        )
        for filename in sorted(filenames):
            path = current_path / filename
            if is_image(path):
                yield path
