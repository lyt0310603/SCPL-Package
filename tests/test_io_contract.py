import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decoupleflow import DecoupleFlow


def _make_model() -> DecoupleFlow:
    backbone = nn.Sequential(
        nn.Linear(8, 16),
        nn.Linear(16, 8),
        nn.Linear(8, 4),
    )
    return DecoupleFlow(
        custom_model=backbone,
        device_map=[
            {"device": "cpu", "layers": 1},
            {"device": "cpu", "layers": 1},
            {"device": "cpu", "layers": 1},
        ],
        loss_fn="CL",
        num_classes=4,
        optimizer_fn=torch.optim.SGD,
        optimizer_param={"lr": 0.01},
        multi_t=False,
    )


def _make_transformer_model() -> DecoupleFlow:
    backbone = nn.Sequential(
        nn.TransformerEncoderLayer(d_model=8, nhead=2, batch_first=True),
        nn.TransformerEncoderLayer(d_model=8, nhead=2, batch_first=True),
    )
    return DecoupleFlow(
        custom_model=backbone,
        device_map=[
            {"device": "cpu", "layers": 1},
            {"device": "cpu", "layers": 1},
        ],
        loss_fn="CL",
        num_classes=4,
        optimizer_fn=torch.optim.SGD,
        optimizer_param={"lr": 0.01},
        multi_t=False,
    )


def test_train_step_accepts_valid_input_shapes():
    model = _make_model()
    x = torch.randn(5, 8)
    y = torch.tensor([0, 1, 2, 3, 1], dtype=torch.long)

    model.train()
    features, total_loss, labels = model.train_step(x, y)

    assert features.shape == (5, 4)
    assert isinstance(total_loss, float)
    assert labels.shape == (5,)


def test_train_step_raises_on_invalid_feature_shape():
    model = _make_model()
    x = torch.randn(5, 7)  # Expected feature dim is 8.
    y = torch.tensor([0, 1, 2, 3, 1], dtype=torch.long)

    model.train()
    with pytest.raises(RuntimeError):
        _ = model.train_step(x, y)


def test_test_step_output_shape_and_device_contract():
    model = _make_model()
    x = torch.randn(6, 8)
    y = torch.tensor([0, 1, 2, 3, 1, 2], dtype=torch.long)

    model.eval()
    output, moved_y = model.test_step(x, y)

    assert output.shape == (6, 4)
    assert moved_y.shape == (6,)
    assert output.device.type == "cpu"
    assert moved_y.device.type == "cpu"


def test_mask_path_runs_and_output_contract_is_correct():
    torch.manual_seed(7)
    model = _make_transformer_model()
    x = torch.randn(4, 5, 8)
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    mask = torch.tensor(
        [
            [False, False, False, False, False],
            [False, False, True, True, True],
            [False, False, False, True, True],
            [False, True, True, True, True],
        ],
        dtype=torch.bool,
    )

    model.eval()
    output, moved_y = model.test_step(x, y, mask=mask)

    assert output.shape == (4, 5, 8)
    assert moved_y.shape == (4,)
    assert output.device.type == "cpu"
    assert moved_y.device.type == "cpu"


def test_all_false_mask_matches_none_mask():
    torch.manual_seed(11)
    model = _make_transformer_model()
    x = torch.randn(3, 4, 8)
    y = torch.tensor([0, 1, 2], dtype=torch.long)
    all_false_mask = torch.zeros(3, 4, dtype=torch.bool)

    model.eval()
    output_without_mask, y_without_mask = model.test_step(x, y, mask=None)
    output_with_mask, y_with_mask = model.test_step(x, y, mask=all_false_mask)

    assert torch.allclose(output_without_mask, output_with_mask), (
        "全 False mask 應與不傳 mask 產生一致輸出。"
    )
    assert torch.equal(y_without_mask, y_with_mask)
