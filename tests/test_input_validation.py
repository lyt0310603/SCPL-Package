import sys
from pathlib import Path

import pytest
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decoupleflow import DecoupleFlow


def _small_model():
    return nn.Sequential(
        nn.Linear(4, 4),
        nn.Linear(4, 2),
    )


def test_raises_when_device_map_is_none():
    with pytest.raises(ValueError, match="device_map cannot be None"):
        DecoupleFlow(
            custom_model=_small_model(),
            device_map=None,
            loss_fn="CL",
            num_classes=2,
            multi_t=False,
        )


def test_raises_when_layer_count_mismatch_device_map():
    with pytest.raises(ValueError, match="Layers of model don't equal to balance"):
        DecoupleFlow(
            custom_model=_small_model(),
            device_map=[{"device": "cpu", "layers": 1}],
            loss_fn="CL",
            num_classes=2,
            multi_t=False,
        )


def test_raises_when_device_layer_count_non_positive():
    with pytest.raises(ValueError, match="must be > 0"):
        DecoupleFlow(
            custom_model=_small_model(),
            device_map=[{"device": "cpu", "layers": 0}, {"device": "cpu", "layers": 2}],
            loss_fn="CL",
            num_classes=2,
            multi_t=False,
        )


def test_raises_when_device_is_none():
    with pytest.raises(ValueError, match="Device is None"):
        DecoupleFlow(
            custom_model=_small_model(),
            device_map=[{"device": None, "layers": 1}, {"device": "cpu", "layers": 1}],
            loss_fn="CL",
            num_classes=2,
            multi_t=False,
        )


def test_raises_when_deinfo_missing_num_classes():
    with pytest.raises(ValueError, match="DeInfo Loss need pass class nums"):
        DecoupleFlow(
            custom_model=_small_model(),
            device_map=[{"device": "cpu", "layers": 1}, {"device": "cpu", "layers": 1}],
            loss_fn="DeInfo",
            num_classes=None,
            multi_t=False,
        )


def test_raises_when_device_map_type_invalid():
    with pytest.raises(TypeError, match="device_map must be dict or list"):
        DecoupleFlow(
            custom_model=_small_model(),
            device_map="cpu",
            loss_fn="CL",
            num_classes=2,
            multi_t=False,
        )


def test_raises_when_device_map_item_missing_key():
    with pytest.raises(KeyError, match="missing required key: 'layers'"):
        DecoupleFlow(
            custom_model=_small_model(),
            device_map=[{"device": "cpu"}, {"device": "cpu", "layers": 1}],
            loss_fn="CL",
            num_classes=2,
            multi_t=False,
        )


def test_raises_when_transform_funcs_length_mismatch():
    with pytest.raises(ValueError, match="Cannot distribute transform_funcs"):
        DecoupleFlow(
            custom_model=_small_model(),
            device_map=[{"device": "cpu", "layers": 1}, {"device": "cpu", "layers": 1}],
            loss_fn="CL",
            num_classes=2,
            transform_funcs=[lambda x: x],
            multi_t=False,
        )
