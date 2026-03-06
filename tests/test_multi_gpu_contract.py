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


requires_two_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="需要至少兩張 GPU 才能執行此測試。",
)


def _make_multi_gpu_model() -> DecoupleFlow:
    backbone = nn.Sequential(
        nn.Linear(8, 16),
        nn.Linear(16, 8),
        nn.Linear(8, 4),
    )
    return DecoupleFlow(
        custom_model=backbone,
        device_map=[
            {"device": "cuda:0", "layers": 1},
            {"device": "cuda:1", "layers": 1},
            {"device": "cuda:1", "layers": 1},
        ],
        loss_fn="CL",
        num_classes=4,
        optimizer_fn=torch.optim.SGD,
        optimizer_param={"lr": 0.01},
        multi_t=False,
    )


@requires_two_gpus
def test_last_output_and_label_are_on_same_gpu():
    model = _make_multi_gpu_model()
    model.eval()

    x = torch.randn(5, 8, device="cuda:0")
    y = torch.tensor([0, 1, 2, 3, 1], dtype=torch.long, device="cuda:0")

    output, moved_y = model.test_step(x, y)

    assert output.device.type == "cuda"
    assert moved_y.device.type == "cuda"
    assert output.device == moved_y.device == torch.device("cuda:1")


@requires_two_gpus
def test_blocks_follow_device_map_assignment():
    model = _make_multi_gpu_model()
    expected_devices = ["cuda:0", "cuda:1", "cuda:1"]

    for idx, block in enumerate(model.model):
        block_param_devices = {p.device for p in block.parameters()}
        assert block_param_devices, f"Block {idx} 沒有可檢查的參數。"
        assert block_param_devices == {torch.device(expected_devices[idx])}, (
            f"Block {idx} 裝置分配錯誤，預期 {expected_devices[idx]}，"
            f"實際為 {block_param_devices}。"
        )
