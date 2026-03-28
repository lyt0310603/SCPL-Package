import sys
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decoupleflow import DecoupleFlow
from decoupleflow.BasicBlock import AdaptiveBasicBlock
from decoupleflow.Projector import ProjectorLayer


def test_non_last_blocks_have_projector_and_local_loss_with_independent_optimizers():
    custom_model = nn.Sequential(
        nn.Linear(8, 16),
        nn.Linear(16, 16),
        nn.Linear(16, 4),
    )
    device_map = [
        {"device": "cpu", "layers": 1},
        {"device": "cpu", "layers": 1},
        {"device": "cpu", "layers": 1},
    ]

    model = DecoupleFlow(
        custom_model=custom_model,
        device_map=device_map,
        loss_fn="CL",
        num_classes=4,
        optimizer_fn=torch.optim.SGD,
        optimizer_param={"lr": 0.01},
        multi_t=False,
    )

    num_blocks = len(model.model)
    assert num_blocks == 3

    # Verify per-block projector and loss assignment policy.
    for idx, block in enumerate(model.model):
        assert hasattr(block, "projector_layer"), f"Block {idx} 缺少 projector_layer。"
        assert isinstance(block.projector_layer, ProjectorLayer), (
            f"Block {idx} 的 projector_layer 不是 ProjectorLayer。"
        )
        assert hasattr(block, "loss_layer"), f"Block {idx} 缺少 loss_layer。"

        if idx < num_blocks - 1:
            assert block.loss_layer.loss_function == ("CL" or "DeInfo"), (
                f"Block {idx} 應為局部 CL 或 DeInfo 損失。"
            )
        else:
            assert block.loss_layer.loss_function == "CE", (
                "最後一個 Block 應為字串 CE。"
            )

    # Verify one optimizer per block and no shared parameters between optimizers.
    assert len(model.optimizers) == num_blocks, "每個 block 應有一個對應 optimizer。"
    optimizer_ids = {id(opt.optimizer) for opt in model.optimizers}
    assert len(optimizer_ids) == num_blocks, "optimizer 物件不應共用。"

    per_block_param_id_sets = []
    for idx, opt_wrapper in enumerate(model.optimizers):
        opt_param_ids = {
            id(p)
            for group in opt_wrapper.optimizer.param_groups
            for p in group["params"]
        }
        block_param_ids = {id(p) for p in model.model[idx].parameters()}
        assert opt_param_ids == block_param_ids, (
            f"Block {idx} 的 optimizer 參數不等於該 block 參數。"
        )
        per_block_param_id_sets.append(opt_param_ids)

    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            assert per_block_param_id_sets[i].isdisjoint(per_block_param_id_sets[j]), (
                f"Block {i} 與 Block {j} 的 optimizer 參數有重疊。"
            )


def test_adaptive_mode_all_blocks_have_local_loss_and_extra_head():
    custom_model = nn.Sequential(
        nn.Linear(8, 16),
        nn.Linear(16, 16),
        nn.Linear(16, 4),
    )
    device_map = [
        {"device": "cpu", "layers": 1},
        {"device": "cpu", "layers": 1},
        {"device": "cpu", "layers": 1},
    ]

    model = DecoupleFlow(
        custom_model=custom_model,
        device_map=device_map,
        loss_fn="CL",
        num_classes=4,
        optimizer_fn=torch.optim.SGD,
        optimizer_param={"lr": 0.01},
        multi_t=False,
        is_adaptive=True,
    )

    num_blocks = len(model.model)
    assert num_blocks == 3

    for idx, block in enumerate(model.model):
        assert isinstance(block, AdaptiveBasicBlock), (
            f"Adaptive 模式下 Block {idx} 應為 AdaptiveBasicBlock。"
        )
        assert hasattr(block, "projector_layer"), f"Block {idx} 缺少 projector_layer。"
        assert isinstance(block.projector_layer, ProjectorLayer), (
            f"Block {idx} 的 projector_layer 不是 ProjectorLayer。"
        )
        assert hasattr(block, "loss_layer"), f"Block {idx} 缺少 loss_layer。"
        assert block.loss_layer.loss_function == "CL", (
            f"Adaptive 模式下 Block {idx} 應為局部 CL 損失。"
        )
        assert hasattr(block, "extra_layer"), f"Adaptive 模式下 Block {idx} 缺少 extra_layer。"

    assert len(model.optimizers) == num_blocks, "Adaptive 模式下每個 block 應有一個 optimizer。"
    optimizer_ids = {id(opt.optimizer) for opt in model.optimizers}
    assert len(optimizer_ids) == num_blocks, "Adaptive 模式下 optimizer 物件不應共用。"

    per_block_param_id_sets = []
    for idx, opt_wrapper in enumerate(model.optimizers):
        opt_param_ids = {
            id(p)
            for group in opt_wrapper.optimizer.param_groups
            for p in group["params"]
        }
        block_param_ids = {id(p) for p in model.model[idx].parameters()}
        assert opt_param_ids == block_param_ids, (
            f"Adaptive 模式下 Block {idx} 的 optimizer 參數不等於該 block 參數。"
        )
        per_block_param_id_sets.append(opt_param_ids)

    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            assert per_block_param_id_sets[i].isdisjoint(per_block_param_id_sets[j]), (
                f"Adaptive 模式下 Block {i} 與 Block {j} 的 optimizer 參數有重疊。"
            )
