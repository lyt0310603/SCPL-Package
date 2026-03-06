import copy
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decoupleflow import DecoupleFlow


def _set_seed(seed: int = 99) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


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
        optimizer_param={"lr": 0.03},
        multi_t=False,
    )


def test_state_dict_save_load_preserves_inference_output(tmp_path):
    _set_seed(1001)
    x = torch.randn(5, 8)
    y = torch.tensor([0, 1, 2, 3, 1], dtype=torch.long)

    source_model = _make_model()
    source_model.train()
    _ = source_model(x, y)
    source_model.eval()

    with torch.no_grad():
        expected_output, expected_y = source_model.test_step(x, y)

    checkpoint_path = tmp_path / "decoupleflow_state.pt"
    torch.save(copy.deepcopy(source_model.state_dict()), checkpoint_path)

    target_model = _make_model()
    target_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    target_model.eval()

    with torch.no_grad():
        actual_output, actual_y = target_model.test_step(x, y)

    assert torch.equal(actual_output, expected_output), "load_state_dict 後推論輸出不一致。"
    assert torch.equal(actual_y, expected_y), "load_state_dict 後標籤張量不一致。"
