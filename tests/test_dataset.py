import torch

from pneumonia.config import load_config
from pneumonia.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    CXRDataset,
    build_transforms,
    make_loader,
    make_splits,
)
from pneumonia.indexing import build_index, find_dataset_root


def _cfg(tmp_path, data_dir):
    cfg = load_config("configs/smoke.yaml")
    cfg["data"]["data_dir"] = str(data_dir)
    cfg["data"]["image_size"] = 224
    cfg["data"]["cache_resized"] = False
    cfg["train"]["batch_size"] = 4
    return cfg


def test_dataset_on_synthetic_tree(synth_tree, tmp_path):
    root = find_dataset_root(synth_tree)
    idx = build_index(root)
    assert len(idx) == 20
    cfg = _cfg(tmp_path, synth_tree)
    ds = CXRDataset(idx, root, build_transforms(cfg, train=True))
    x, y, i = ds[0]
    assert x.shape == (3, 224, 224) and x.dtype == torch.float32
    # greyscale replicated to 3 channels: identical once ImageNet normalisation is undone
    raw = x * torch.tensor(IMAGENET_STD)[:, None, None] + torch.tensor(IMAGENET_MEAN)[:, None, None]
    assert torch.allclose(raw[0], raw[1], atol=1e-5) and torch.allclose(raw[1], raw[2], atol=1e-5)
    assert y.item() in (0.0, 1.0)
    splits, _ = make_splits(idx, 0.25, 0)
    xb, yb, ib = next(iter(make_loader(cfg, splits["train"], root, train=True)))
    assert xb.shape[1:] == (3, 224, 224) and yb.shape[0] == xb.shape[0]
