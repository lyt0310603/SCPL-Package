import copy
import random
import sys
from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decoupleflow import DecoupleFlow


def _set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _make_model(multi_t: bool) -> DecoupleFlow:
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
        multi_t=multi_t,
    )


def _make_perturbed_loss():
    def _perturbed_loss(hidden_state: torch.Tensor, true_y: torch.Tensor):
        del true_y
        noise = torch.randn_like(hidden_state)
        loss = (hidden_state * noise).sum() * 1e6
        return loss, hidden_state

    return _perturbed_loss


def _run_and_get_weights(
    model: DecoupleFlow, x: torch.Tensor, y: torch.Tensor, perturb_blocks: Iterable[int]
) -> List[List[torch.Tensor]]:
    for idx in perturb_blocks:
        model.model[idx].loss_cal = _make_perturbed_loss()

    model.train()
    _ = model(x, y)
    return [
        [p.detach().clone() for p in model.model[idx].parameters()]
        for idx in range(len(model.model))
    ]


def test_multi_t_matches_single_thread_weight_update():
    _set_seed(2026)
    x = torch.randn(4, 4)
    y = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    template = _make_model(multi_t=False)
    init_state = copy.deepcopy(template.state_dict())

    _set_seed(2026)
    single_thread_model = _make_model(multi_t=False)
    single_thread_model.load_state_dict(init_state)
    single_thread_weights = _run_and_get_weights(
        single_thread_model, x, y, perturb_blocks=()
    )

    _set_seed(2026)
    multi_thread_model = _make_model(multi_t=True)
    multi_thread_model.load_state_dict(init_state)
    multi_thread_weights = _run_and_get_weights(
        multi_thread_model, x, y, perturb_blocks=()
    )

    for block_idx in (0, 1, 2):
        for w_st, w_mt in zip(single_thread_weights[block_idx], multi_thread_weights[block_idx]):
            assert torch.equal(w_st, w_mt), (
                f"Block {block_idx} 在 multi_t=True 與 multi_t=False 下更新不一致。"
            )


def test_multi_t_preserves_cross_block_isolation_under_perturbation():
    _set_seed(2027)
    x = torch.randn(4, 4)
    y = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    template = _make_model(multi_t=True)
    init_state = copy.deepcopy(template.state_dict())

    _set_seed(2027)
    baseline_model = _make_model(multi_t=True)
    baseline_model.load_state_dict(init_state)
    baseline_weights = _run_and_get_weights(
        baseline_model, x, y, perturb_blocks=()
    )

    for perturbed_idx in (0, 1, 2):
        _set_seed(2027)
        perturbed_model = _make_model(multi_t=True)
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
                    f"multi_t 模式下擾動 Block {perturbed_idx} 後該區塊權重未改變，"
                    "請確認 monkey patch 是否生效。"
                )
            else:
                assert same, (
                    f"multi_t 模式下擾動 Block {perturbed_idx} 不應影響 Block {block_idx}，"
                    "但檢測到權重差異。"
                )
