import pytest

from pneumonia.synthetic import make_synthetic_tree


@pytest.fixture
def synth_tree(tmp_path):
    """20-image synthetic tree: train 6+6, val 1+1, test 3+3 (nested chest_xray/chest_xray)."""
    make_synthetic_tree(tmp_path / "data", {"train": 6, "val": 1, "test": 3}, size=64, seed=0)
    return tmp_path / "data"
