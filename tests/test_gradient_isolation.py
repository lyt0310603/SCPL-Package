import copy
import random
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
LEGACY_MODULE_ROOT = SRC_ROOT / "decoupleflow"
for p in (SRC_ROOT, LEGACY_MODULE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from decoupleflow import DecoupleFlow


def _set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _make_model() -> DecoupleFlow:
    custom_model = nn.Sequential(
        nn.Linear(4, 4),
        nn.Linear(4, 4),
        nn.Linear(4, 3),
    )
    return DecoupleFlow(
        custom_model=custom_model,
        device_map=[
            {"device": "cpu", "layers": 1},
            {"device": "cpu", "layers": 1},
            {"device": "cpu", "layers": 1},
        ],
        loss_fn="CL",
        num_classes=3,
        optimizer_fn=torch.optim.SGD,
        optimizer_param={"lr": 0.05},
        multi_t=False,
    )


def _make_perturbed_loss():
    def _perturbed_loss(hidden_state: torch.Tensor, true_y: torch.Tensor):
            del true_y
            noise = torch.randn_like(hidden_state)
            # Deliberately replace local objective with a huge random objective.
            loss = (hidden_state * noise).sum() * 1e6
            return loss, hidden_state
    return _perturbed_loss


def _run_and_get_weights(
    model: DecoupleFlow, x: torch.Tensor, y: torch.Tensor, perturb_blocks: Iterable[int]
):
    for idx in perturb_blocks:
        model.model[idx].loss = _make_perturbed_loss()

    model.train()
    _ = model(x, y)
    return [
        [p.detach().clone() for p in model.model[idx].parameters()]
        for idx in range(len(model.model))
    ]


def test_three_blocks_are_mutually_independent_under_local_loss_perturbation():
    _set_seed(42)
    x = torch.randn(4, 4)
    y = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    template = _make_model()
    init_state = copy.deepcopy(template.state_dict())

    _set_seed(42)
    baseline_model = _make_model()
    baseline_model.load_state_dict(init_state)
    baseline_weights = _run_and_get_weights(
        baseline_model, x, y, perturb_blocks=()
    )

    for perturbed_idx in (0, 1, 2):
        _set_seed(42)
        perturbed_model = _make_model()
        perturbed_model.load_state_dict(init_state)
        perturbed_weights = _run_and_get_weights(
            perturbed_model, x, y, perturb_blocks=(perturbed_idx,)
        )

        for block_idx in (0, 1, 2):
            same = all(
                torch.equal(wa, wb)
                for wa, wb in zip(baseline_weights[block_idx], perturbed_weights[block_idx])
            )
            if block_idx == perturbed_idx:
                assert not same, (
                    f"擾動 Block {perturbed_idx} 後該區塊權重未改變，"
                    "請確認 monkey patch 是否生效。"
                )
            else:
                assert same, (
                    f"擾動 Block {perturbed_idx} 不應影響 Block {block_idx}，"
                    "但檢測到權重差異。"
                )
