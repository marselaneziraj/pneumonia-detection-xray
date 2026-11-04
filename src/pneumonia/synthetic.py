"""Synthetic chest-xray-shaped tree for offline tests and the CPU smoke run.

The images are NOT radiographs: two dark ellipses ("lung fields") on a grey body, with
bright blotches inside the lungs for the PNEUMONIA class. File names follow the real
Kaggle patterns so patient-id parsing and grouped splitting are exercised for real.
Nothing produced from this tree is a result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _image(rng, positive: bool, size: int) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size] / size
    img = 0.55 + 0.05 * rng.standard_normal((size, size))
    for cx in (0.32, 0.68):
        lung = ((xx - cx) / 0.15) ** 2 + ((yy - 0.5) / 0.28) ** 2 < 1
        img[lung] -= 0.3
        if positive:
            bx, by = cx + rng.uniform(-0.06, 0.06), 0.5 + rng.uniform(-0.15, 0.15)
            blob = ((xx - bx) / 0.07) ** 2 + ((yy - by) / 0.07) ** 2 < 1
            img[blob & lung] += 0.35
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), mode="L")


def make_synthetic_tree(
    root: str | Path,
    n_per_class: dict | None = None,
    size: int = 96,
    seed: int = 0,
    imgs_per_patient: int = 2,
) -> Path:
    """Write chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg under root (nested like
    some Kaggle versions). n_per_class maps split -> images per class."""
    rng = np.random.default_rng(seed)
    n_per_class = n_per_class or {"train": 4, "val": 2, "test": 4}
    base = Path(root) / "chest_xray" / "chest_xray"
    pid = {"NORMAL": 100, "PNEUMONIA": 1}
    for split, n in n_per_class.items():
        for cls in ("NORMAL", "PNEUMONIA"):
            d = base / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for k in range(n):
                if k % imgs_per_patient == 0:
                    pid[cls] += 1
                if cls == "NORMAL":
                    pre = "NORMAL2-" if pid[cls] % 2 else ""
                    name = f"{pre}IM-{pid[cls]:04d}-{k % imgs_per_patient + 1:04d}.jpeg"
                else:
                    sub = "virus" if pid[cls] % 3 == 0 else "bacteria"
                    name = f"person{pid[cls]}_{sub}_{k + 1}.jpeg"
                _image(rng, cls == "PNEUMONIA", size).save(d / name, quality=95)
    return base
