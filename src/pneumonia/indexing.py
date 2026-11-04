"""Walk the Kaggle chest-xray tree into a tidy frame and audit it.

Output columns: path (relative to the dataset root), class_name, label, official_split,
patient_id, subtype (bacteria / virus / normal), filename.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd
from PIL import Image

from . import CLASS_NAMES, POSITIVE_CLASS

KAGGLE_SLUG = "paultimothymooney/chest-xray-pneumonia"
DOWNLOAD_CMD = f"kaggle datasets download -d {KAGGLE_SLUG} -p data --unzip"
OFFICIAL_SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpeg", ".jpg", ".png"}

# person8_bacteria_37.jpeg -> person8 ; person1_virus_6.jpeg -> person1 ;
# person1_virus_6_1.jpeg (rare extra suffix in the Kaggle copy) -> person1
_PNEUMONIA_RE = re.compile(r"^(person\d+)_(bacteria|virus)_\d+(?:_\d+)*$", re.IGNORECASE)
# IM-0115-0001.jpeg -> IM-0115 ; NORMAL2-IM-0222-0001.jpeg -> NORMAL2-IM-0222 ;
# IM-0429-0001-0002.jpeg (extra study/series suffix) -> IM-0429
_NORMAL_RE = re.compile(r"^((?:NORMAL2-)?IM-\d+)(?:-\d+)+$", re.IGNORECASE)


class DatasetNotFoundError(FileNotFoundError):
    pass


def parse_filename(filename: str) -> tuple[str, str]:
    """Return (patient_id, subtype) parsed from a file name.

    subtype is 'bacteria', 'virus' or 'normal'. Unparseable names fall back to a
    per-file id prefixed 'UNPARSED-' (so they can never silently merge patients) and
    emit a warning.
    """
    stem = Path(filename).stem
    m = _PNEUMONIA_RE.match(stem)
    if m:
        return m.group(1).lower(), m.group(2).lower()
    m = _NORMAL_RE.match(stem)
    if m:
        return m.group(1).upper(), "normal"
    warnings.warn(f"Could not parse patient id from {filename!r}; using per-file id", stacklevel=2)
    return f"UNPARSED-{stem}", "unknown"


def parse_patient_id(filename: str) -> str:
    return parse_filename(filename)[0]


def _looks_like_root(p: Path) -> bool:
    return all((p / s).is_dir() for s in ("train", "test")) and (p / "train" / "NORMAL").is_dir()


def find_dataset_root(data_dir: str | Path) -> Path:
    """Resolve the real dataset root, tolerating the nested chest_xray/chest_xray layout.

    Picks the candidate directory holding train/{NORMAL,PNEUMONIA} with the most images
    (some Kaggle versions ship a duplicated nested copy and a __MACOSX folder).
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise DatasetNotFoundError(_missing_msg(data_dir))
    candidates = [data_dir] if _looks_like_root(data_dir) else []
    # our own resized cache (data/cache_224/...) mirrors the layout; never pick it as the root
    for p in data_dir.rglob("*"):
        if (
            p.is_dir()
            and "__MACOSX" not in p.parts
            and not any(part.startswith("cache_") for part in p.relative_to(data_dir).parts)
            and _looks_like_root(p)
        ):
            candidates.append(p)
    if not candidates:
        raise DatasetNotFoundError(_missing_msg(data_dir))

    def n_images(root: Path) -> int:
        return sum(1 for f in root.rglob("*") if f.suffix.lower() in IMAGE_EXTS)

    return max(candidates, key=lambda r: (n_images(r), -len(r.parts)))


def _missing_msg(data_dir: Path) -> str:
    return (
        f"Chest X-ray dataset not found under '{data_dir.resolve()}'.\n"
        f"Expected a folder containing train/, val/, test/ each with NORMAL/ and PNEUMONIA/ "
        f"(e.g. {data_dir}/chest_xray/train/NORMAL/*.jpeg).\n"
        f"Download it with:\n    {DOWNLOAD_CMD}\n"
        f"(needs a Kaggle API token in ~/.kaggle/kaggle.json, see README)."
    )


def build_index(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    rows = []
    for split in OFFICIAL_SPLITS:
        for cls in CLASS_NAMES:
            d = root / split / cls
            if not d.is_dir():
                if split == "val":
                    continue
                raise DatasetNotFoundError(f"Missing folder {d}\n{_missing_msg(root)}")
            for f in sorted(d.iterdir()):
                if f.suffix.lower() not in IMAGE_EXTS or f.name.startswith("."):
                    continue
                pid, subtype = parse_filename(f.name)
                rows.append(
                    {
                        "path": f.relative_to(root).as_posix(),
                        "filename": f.name,
                        "class_name": cls,
                        "label": int(cls == POSITIVE_CLASS),
                        "official_split": split,
                        "patient_id": pid,
                        "subtype": subtype,
                    }
                )
    if not rows:
        raise DatasetNotFoundError(_missing_msg(root))
    return pd.DataFrame(rows)


def count_table(df: pd.DataFrame, split_col: str = "official_split") -> pd.DataFrame:
    """Images and distinct patients per split x class, plus the class ratio."""
    g = df.groupby([split_col, "class_name"])
    t = pd.DataFrame({"images": g.size(), "patients": g["patient_id"].nunique()}).unstack(
        "class_name"
    )
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    t = t.fillna(0).astype(int)
    t["images_total"] = t.filter(like="images_").sum(axis=1)
    t["patients_total"] = df.groupby(split_col)["patient_id"].nunique()
    neg = t.get("images_NORMAL", 0)
    pos = t.get("images_PNEUMONIA", 0)
    t["pos_to_neg_ratio"] = (pos / neg.replace(0, float("nan"))).round(2)
    return t


def audit_images(df: pd.DataFrame, root: str | Path) -> pd.DataFrame:
    """Open every file (asserting it decodes) and record its size and mode."""
    root = Path(root)
    recs, bad = [], []
    for rel in df["path"]:
        try:
            with Image.open(root / rel) as im:
                im.verify()
            with Image.open(root / rel) as im:
                w, h = im.size
                mode = im.mode
            recs.append({"path": rel, "width": w, "height": h, "mode": mode})
        except Exception as e:  # noqa: BLE001
            bad.append((rel, repr(e)))
    if bad:
        raise AssertionError(f"{len(bad)} unreadable images, first: {bad[:3]}")
    out = pd.DataFrame(recs)
    out["aspect"] = (out["width"] / out["height"]).round(3)
    return out
