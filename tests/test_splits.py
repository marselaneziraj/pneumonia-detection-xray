import pandas as pd
import pytest

from pneumonia.data import (
    LeakageError,
    assert_no_patient_leakage,
    make_splits,
    read_manifests,
    write_manifests,
)
from pneumonia.indexing import build_index, find_dataset_root


def _df(pids, label=1):
    return pd.DataFrame({"patient_id": pids, "label": label})


def test_guard_raises_on_shared_patient():
    with pytest.raises(LeakageError, match="person8"):
        assert_no_patient_leakage({"train": _df(["person8", "person9"]), "val": _df(["person8"])})


def test_guard_passes_on_disjoint():
    assert_no_patient_leakage({"train": _df(["a", "b"]), "val": _df(["c"]), "test": _df(["d"])})


def _index(n_patients=60, imgs=3):
    rows = []
    for p in range(n_patients):
        lab = int(p % 4 != 0)  # 3:1 PNEUMONIA:NORMAL
        for k in range(imgs):
            rows.append(
                {
                    "path": f"train/x/{p}_{k}.jpeg",
                    "class_name": "PNEUMONIA" if lab else "NORMAL",
                    "label": lab,
                    "official_split": "train",
                    "patient_id": f"p{p}",
                    "subtype": "x",
                }
            )
    for p in range(10):
        rows.append(
            {
                "path": f"test/x/t{p}.jpeg",
                "class_name": "NORMAL",
                "label": p % 2,
                "official_split": "test",
                "patient_id": f"t{p}",
                "subtype": "x",
            }
        )
    return pd.DataFrame(rows)


def test_grouped_split_has_no_leakage_and_keeps_test():
    idx = _index()
    splits, info = make_splits(idx, val_fraction=0.2, seed=1)
    assert set(splits["train"].patient_id).isdisjoint(splits["val"].patient_id)
    assert len(splits["test"]) == 10 and (splits["test"].official_split == "test").all()
    # stratified: val keeps roughly the 3:1 ratio
    assert 0.6 < splits["val"].label.mean() < 0.9
    # every patient's images stay together
    assert splits["val"].groupby("patient_id").size().eq(3).all()


def test_trainval_patients_seen_in_test_are_dropped():
    idx = _index()
    idx.loc[idx.patient_id == "p1", "patient_id"] = "t3"  # collide with a test id
    splits, info = make_splits(idx, val_fraction=0.2, seed=1)
    assert info["test_trainval_shared_patient_ids"] == 1
    assert info["dropped_trainval_images_for_test_overlap"] == 3
    assert_no_patient_leakage(splits)


def test_manifests_roundtrip(synth_tree, tmp_path):
    idx = build_index(find_dataset_root(synth_tree))
    splits, _ = make_splits(idx, val_fraction=0.25, seed=0)
    write_manifests(splits, tmp_path / "splits")
    back = read_manifests(tmp_path / "splits")
    assert sum(len(v) for v in back.values()) == 20
    assert sorted(back["test"].path) == sorted(splits["test"].path)
