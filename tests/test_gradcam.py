import numpy as np
import pytest
import torch

from pneumonia.gradcam import GradCAM, auto_flag, border_mass
from pneumonia.model import build_model


@pytest.mark.parametrize("arch, size", [("resnet18", 224), ("densenet121", 96)])
def test_gradcam_shape_range_and_hook_cleanup(arch, size):
    torch.manual_seed(0)
    model = build_model(arch, pretrained=False)
    x = torch.randn(2, 3, size, size)
    with GradCAM(model, model.features) as cam:
        maps, probs = cam(x)
        assert len(model.features._forward_hooks) == 1
    assert maps.shape == (2, size, size)
    assert np.isfinite(maps).all()
    assert maps.min() >= 0.0 and maps.max() <= 1.0 + 1e-6
    assert probs.shape == (2,) and ((probs > 0) & (probs < 1)).all()
    assert len(model.features._forward_hooks) == 0  # hooks removed on exit


def test_single_logit_head():
    m = build_model("resnet18", pretrained=False)
    assert m(torch.randn(3, 3, 64, 64)).shape == (3,)


def test_border_mass_and_flag():
    cam = np.zeros((100, 100))
    cam[45:55, 45:55] = 1
    assert border_mass(cam) == pytest.approx(0.0)
    assert auto_flag(cam) == "central"
    edge = np.zeros((100, 100))
    edge[0:5, 0:5] = 1
    assert border_mass(edge) == pytest.approx(1.0)
    assert auto_flag(edge) == "off-anatomy?"


def test_empty_map_is_not_called_off_anatomy():
    assert auto_flag(np.zeros((50, 50))) == "empty"
