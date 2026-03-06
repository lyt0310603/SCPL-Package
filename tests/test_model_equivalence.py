import copy
import random
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
LEGACY_MODULE_ROOT = SRC_ROOT / "decoupleflow"
for p in (SRC_ROOT, LEGACY_MODULE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from decoupleflow import DecoupleFlow


def _set_seed(seed: int = 2026) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _make_backbone() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.Linear(16, 16),
        nn.Linear(16, 8),
        nn.Linear(8, 4),
    )


@pytest.mark.parametrize(
    "device_map",
    [
        pytest.param(
            [{"device": "cpu", "layers": 2}, {"device": "cpu", "layers": 2}],
            id="split-2-2",
        ),
        pytest.param(
            [{"device": "cpu", "layers": 1}, {"device": "cpu", "layers": 3}],
            id="split-1-3",
        ),
        pytest.param(
            [{"device": "cpu", "layers": 3}, {"device": "cpu", "layers": 1}],
            id="split-3-1",
        ),
        pytest.param(
            [
                {"device": "cpu", "layers": 1},
                {"device": "cpu", "layers": 1},
                {"device": "cpu", "layers": 2},
            ],
            id="split-1-1-2",
        ),
        pytest.param(
            [
                {"device": "cpu", "layers": 2},
                {"device": "cpu", "layers": 1},
                {"device": "cpu", "layers": 1},
            ],
            id="split-2-1-1",
        ),
        pytest.param(
            [
                {"device": "cpu", "layers": 1},
                {"device": "cpu", "layers": 2},
                {"device": "cpu", "layers": 1},
            ],
            id="split-1-2-1",
        ),
        pytest.param(
            [
                {"device": "cpu", "layers": 1},
                {"device": "cpu", "layers": 1},
                {"device": "cpu", "layers": 1},
                {"device": "cpu", "layers": 1},
            ],
            id="split-1-1-1-1",
        ),
    ],
)
def test_split_model_forward_matches_original_model(device_map):
    _set_seed(7)

    original_model = _make_backbone().eval()
    x = torch.randn(6, 8)
    y = torch.zeros(6, dtype=torch.long)

    with torch.no_grad():
        expected = original_model(x)

    split_model = DecoupleFlow(
        custom_model=copy.deepcopy(original_model),
        device_map=device_map,
        loss_fn="CL",
        num_classes=4,
        multi_t=False,
    )
    split_model.eval()

    with torch.no_grad():
        actual, _ = split_model.test_step(x, y)

    assert torch.equal(
        actual, expected
    ), "切分後模型輸出與原始模型不一致。"
