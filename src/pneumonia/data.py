"""Patient-grouped splitting, leakage guard, manifests, transforms and the torch Dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from .indexing import find_dataset_root

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLITS = ("train", "val", "test")
MANIFEST_COLS = ["path", "class_name", "label", "patient_id", "subtype", "official_split", "split"]


class LeakageError(AssertionError):
    pass


# ----------------------------------------------------------------------------- splitting
def assert_no_patient_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """Raise LeakageError if any patient_id appears in more than one split."""
    names = list(splits)
    problems = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = set(splits[a]["patient_id"]) & set(splits[b]["patient_id"])
            if shared:
                ex = sorted(shared)[:5]
                problems.append(f"{a}/{b}: {len(shared)} shared patients, e.g. {ex}")
    if problems:
        raise LeakageError("Patient leakage between splits: " + "; ".join(problems))


def make_splits(
    index: pd.DataFrame,
    val_fraction: float = 0.2,
    seed: int = 42,
    exclude_test_patients_from_trainval: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Pool the official train+val folders, re-split by patient (stratified by class).

    The official test folder is returned untouched. If a patient id from the test
    folder also occurs in train/val (the dataset does not document whether person
    numbers are namespaced per folder), those train/val images are dropped -- we
    shrink training rather than touch the test set.
    """
    pool = index[index["official_split"].isin(["train", "val"])].reset_index(drop=True)
    test = index[index["official_split"] == "test"].reset_index(drop=True)

    shared = set(pool["patient_id"]) & set(test["patient_id"])
    info = {
        "test_trainval_shared_patient_ids": len(shared),
        "dropped_trainval_images_for_test_overlap": 0,
    }
    if shared and exclude_test_patients_from_trainval:
        mask = pool["patient_id"].isin(shared)
        info["dropped_trainval_images_for_test_overlap"] = int(mask.sum())
        pool = pool[~mask].reset_index(drop=True)

    n_splits = max(2, int(round(1.0 / val_fraction)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf.split(pool, y=pool["label"], groups=pool["patient_id"]))
    out = {
        "train": pool.iloc[train_idx].reset_index(drop=True),
        "val": pool.iloc[val_idx].reset_index(drop=True),
        "test": test,
    }
    for name, df in out.items():
        df["split"] = name
    if exclude_test_patients_from_trainval:
        assert_no_patient_leakage(out)
    else:
        assert_no_patient_leakage({"train": out["train"], "val": out["val"]})
    info.update({"val_fraction": val_fraction, "n_splits": n_splits, "seed": seed})
    return out, info


def split_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in splits.items():
        for cls in ("NORMAL", "PNEUMONIA"):
            sub = df[df["class_name"] == cls]
            rows.append(
                {
                    "split": name,
                    "class_name": cls,
                    "images": len(sub),
                    "patients": sub["patient_id"].nunique(),
                }
            )
        rows.append(
            {
                "split": name,
                "class_name": "ALL",
                "images": len(df),
                "patients": df["patient_id"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def write_manifests(splits: dict[str, pd.DataFrame], out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, df in splits.items():
        p = out_dir / f"split_{name}.csv"
        df[MANIFEST_COLS].sort_values("path").to_csv(p, index=False)
        paths.append(p)
    return paths


def read_manifests(splits_dir: str | Path) -> dict[str, pd.DataFrame]:
    splits_dir = Path(splits_dir)
    out = {}
    for name in SPLITS:
        p = splits_dir / f"split_{name}.csv"
        if not p.exists():
            raise FileNotFoundError(
                f"Split manifest {p} not found. Run `pneumonia prepare --config <cfg>` first."
            )
        out[name] = pd.read_csv(p)
    assert_no_patient_leakage(out)
    return out


# ----------------------------------------------------------------------------- caching
def cache_path(cache_dir: Path, rel: str) -> Path:
    return (cache_dir / rel).with_suffix(".png")


def build_cache(df: pd.DataFrame, root: Path, cache_dir: Path, size: int) -> int:
    """Pre-resize every image to size x size greyscale PNG (lossless). Returns #written.

    The full radiographs are 1-2k px JPEGs; decoding them every epoch dominates the
    run time on a 2-vCPU Colab box. Resizing once gives the same pixels the
    Resize((size, size)) transform would produce.
    """
    n = 0
    for rel in df["path"]:
        dst = cache_path(cache_dir, rel)
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(root / rel) as im:
            im.convert("L").resize((size, size), Image.BILINEAR).save(dst)
        n += 1
    return n


# ----------------------------------------------------------------------------- transforms
def build_transforms(cfg: dict, train: bool):
    from torchvision import transforms as T

    size = int(cfg["data"]["image_size"])
    a = cfg["augment"]
    ops = [T.Resize((size, size), antialias=True)]
    if train and a.get("enabled", True):
        if a.get("hflip_p", 0) > 0:
            ops.append(T.RandomHorizontalFlip(p=float(a["hflip_p"])))
        ops.append(
            T.RandomAffine(
                degrees=float(a.get("rotate_deg", 0)),
                translate=(float(a.get("translate", 0)),) * 2,
                scale=(1 - float(a.get("scale", 0)), 1 + float(a.get("scale", 0))),
                fill=0,
            )
        )
        ops.append(
            T.ColorJitter(
                brightness=float(a.get("brightness", 0)), contrast=float(a.get("contrast", 0))
            )
        )
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


def denormalize(x):
    """CHW normalised tensor -> HW numpy in [0, 1] (first channel; inputs are greyscale)."""
    img = x[0].detach().cpu().numpy() * IMAGENET_STD[0] + IMAGENET_MEAN[0]
    return np.clip(img, 0, 1)


# ----------------------------------------------------------------------------- dataset
class CXRDataset:
    """Greyscale radiograph -> 3-channel tensor so ImageNet weights apply."""

    def __init__(self, df: pd.DataFrame, root: str | Path, transform, cache_dir=None):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def __len__(self) -> int:
        return len(self.df)

    def load_image(self, i: int) -> Image.Image:
        rel = self.df.at[i, "path"]
        if self.cache_dir is not None:
            cp = cache_path(self.cache_dir, rel)
            if cp.exists():
                return Image.open(cp).convert("L").convert("RGB")
        with Image.open(self.root / rel) as im:
            return im.convert("L").convert("RGB")

    def __getitem__(self, i: int):
        import torch

        x = self.transform(self.load_image(i))
        y = torch.tensor(float(self.df.at[i, "label"]))
        return x, y, i


def resolve_root(cfg: dict) -> Path:
    """Dataset root from config; raises DatasetNotFoundError naming the download command."""
    return find_dataset_root(cfg["data"]["data_dir"])


def cache_dir_for(cfg: dict) -> Path | None:
    if not cfg["data"].get("cache_resized", False):
        return None
    return Path(cfg["data"]["data_dir"]) / f"cache_{int(cfg['data']['image_size'])}"


def make_loader(cfg: dict, df: pd.DataFrame, root: Path, train: bool, shuffle: bool | None = None):
    import torch
    from torch.utils.data import DataLoader

    from .utils import seed_worker

    ds = CXRDataset(df, root, build_transforms(cfg, train=train), cache_dir_for(cfg))
    g = torch.Generator()
    g.manual_seed(int(cfg["experiment"]["seed"]))
    nw = int(cfg["train"].get("num_workers", 0))
    return DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=train if shuffle is None else shuffle,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=nw > 0,
    )


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
