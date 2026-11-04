"""End-to-end CPU smoke run on the synthetic tree: prepare -> train -> evaluate -> gradcam."""

import json

import pandas as pd

from pneumonia.cli import main
from pneumonia.synthetic import make_synthetic_tree


def test_end_to_end_smoke(tmp_path, monkeypatch):
    make_synthetic_tree(tmp_path / "data_synthetic", {"train": 16, "val": 4, "test": 6}, size=64)
    import shutil
    from pathlib import Path

    shutil.copytree(Path(__file__).parents[1] / "configs", tmp_path / "configs")
    monkeypatch.chdir(tmp_path)
    assert main(["prepare", "--config", "configs/smoke.yaml"]) == 0
    assert main(["train", "--config", "configs/smoke.yaml", "--limit-train-batches", "2"]) == 0
    hist = pd.read_csv("runs_smoke/smoke_resnet18/history.csv")
    assert {"val_auroc", "val_threshold", "val_recall", "val_specificity"} <= set(hist.columns)
    assert main(["evaluate", "--configs", "configs/smoke.yaml"]) == 0
    res = pd.read_csv("results_smoke/results.csv")
    base = res[(res.model == "always PNEUMONIA") & (res.split == "test")].iloc[0]
    assert base.recall == 1.0 and base.specificity == 0.0
    m = json.load(open("results_smoke/metrics.json"))
    assert m["models"]["smoke_resnet18"]["status"] == "evaluated"
    assert main(["gradcam", "--config", "configs/smoke.yaml", "--per-case", "1"]) == 0
    assert (tmp_path / "results_smoke/figures/gradcam_panel.png").exists()


def test_missing_data_gives_clear_error(tmp_path, monkeypatch, capsys):
    import shutil
    from pathlib import Path

    shutil.copytree(Path(__file__).parents[1] / "configs", tmp_path / "configs")
    monkeypatch.chdir(tmp_path)
    assert main(["prepare", "--config", "configs/densenet121.yaml"]) == 2
    assert "kaggle datasets download" in capsys.readouterr().err
