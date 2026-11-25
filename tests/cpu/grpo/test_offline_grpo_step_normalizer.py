"""The offline-GRPO denominator spans the whole optimizer step: how its rows split into accumulation micro-batches and
data-parallel ranks leaves every row's gradient unchanged, so each group keeps its ``1/group_size`` share. Each row
trains its own per-token log-probs, so a row's gradient is its weight in the step."""

import datetime
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn
from transformers import Trainer

from halo.data.collators.offline_grpo import OfflineGRPOCPDataCollatorWithPadding, OfflineGRPODataCollatorWithPadding
from halo.trainers.grpo.objective.offline_loss import LOSS_TYPES
from halo.trainers.grpo.offline.trainer import OfflineGRPOTrainer
from halo.trainers.metrics.world_metrics import WorldMetrics
from tests.common.gloo import run_gloo_ranks

# Row r's prompt is the single token ROW_TOKEN + r, which the stub scorers read back to pick the row's log-probs.
ROW_TOKEN = 50
WIDTH = 8
# Group A is one completion (row 0); group B is three (rows 1-3).
GROUP_SIZES = [1, 3, 3, 3]
UNEVEN_LENGTHS = [2, 3, 1, 2]
SPLITS = {
    "one micro-batch": [[0, 1, 2, 3]],
    "two micro-batches": [[0, 1], [2, 3]],
    "four micro-batches": [[0], [1], [2], [3]],
}
PATHS = ("plain", "cp")


def _rows(lengths: list[int]) -> list[dict]:
    return [
        {
            "prompt_input_ids": [ROW_TOKEN + row],
            "completion_input_ids": [60 + position for position in range(length)],
            "group_id": 0 if size == 1 else 1,
            "group_size": size,
            "advantage": 1.0 - 0.5 * row,
        }
        for row, (size, length) in enumerate(zip(GROUP_SIZES, lengths, strict=True))
    ]


def _trainer(path: str, loss_type: str, theta: torch.Tensor, world_size: int = 1) -> OfflineGRPOTrainer:
    trainer = OfflineGRPOTrainer.__new__(OfflineGRPOTrainer)
    trainer.parallelism_config = SimpleNamespace(is_cp_mode=path == "cp", stage_world_size=world_size)
    trainer.cp_config = SimpleNamespace(cp_size=1, cp_rank=0, process_group=None)
    trainer.model = nn.Linear(1, 1)
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer._peft_has_been_casted_to_bf16 = False
    trainer.model_accepts_loss_kwargs = False
    trainer.compute_loss_func = None
    trainer.min_log_prob = None
    trainer.beta = 0.0
    trainer.loss_type = loss_type
    trainer.max_completion_length = WIDTH
    trainer.args = SimpleNamespace(policy_gradient_formulation="reinforce", average_tokens_across_devices=True)
    trainer._sign_metrics = {"train": WorldMetrics(), "eval": WorldMetrics()}
    trainer._get_per_token_logps = lambda model, ids, mask, keep: theta[ids[:, 0] - ROW_TOKEN, :keep]
    trainer._cp_chunked_logps = lambda model, ids, mask, labels: (
        theta[ids[:, 0] - ROW_TOKEN, : ids.size(1) - 1],
        labels[:, 1:],
    )
    return trainer


def _collate(path: str, rows: list[dict]) -> dict:
    if path == "cp":
        return OfflineGRPOCPDataCollatorWithPadding(pad_token_id=0, cp_size=1)(rows)
    return OfflineGRPODataCollatorWithPadding(pad_token_id=0)(rows)


def _step_gradient(path: str, loss_type: str, rows: list[dict], split: list[list[int]], world_size: int = 1):
    """One optimizer step as the train loop runs it: the window's batches drawn through ``get_batch_samples``, each
    micro-batch's loss divided by the window's micro-batch count, as ``training_step`` divides an own-mean loss."""
    theta = torch.zeros(len(rows), WIDTH, requires_grad=True)
    trainer = _trainer(path, loss_type, theta, world_size)
    batches = iter([_collate(path, [rows[index] for index in micro_batch]) for micro_batch in split])
    batch_samples, num_items_in_batch = Trainer.get_batch_samples(trainer, batches, len(split), torch.device("cpu"))
    for batch in batch_samples:
        loss = trainer.compute_loss(trainer.model, batch, num_items_in_batch=num_items_in_batch)
        (loss / len(batch_samples)).backward()
    return theta.grad


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("loss_type", LOSS_TYPES)
def test_row_gradients_do_not_depend_on_the_micro_batch_split(path, loss_type):
    rows = _rows(UNEVEN_LENGTHS)
    whole = _step_gradient(path, loss_type, rows, SPLITS["one micro-batch"])
    assert whole.abs().sum() > 0
    for name in ("two micro-batches", "four micro-batches"):
        split = _step_gradient(path, loss_type, rows, SPLITS[name])
        torch.testing.assert_close(split, whole, rtol=1e-6, atol=1e-7, msg=f"{name}: row weights moved")


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("loss_type", LOSS_TYPES)
@pytest.mark.parametrize("split", SPLITS)
def test_groups_of_different_sizes_count_equally(path, loss_type, split):
    rows = [{**row, "advantage": 1.0} for row in _rows([2, 2, 2, 2])]
    per_row = _step_gradient(path, loss_type, rows, SPLITS[split]).abs().sum(dim=1)
    group_a, group_b = per_row[0], per_row[1:].sum()
    torch.testing.assert_close(group_a / (group_a + group_b), torch.tensor(0.5))


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("loss_type", LOSS_TYPES)
def test_one_micro_batch_step_matches_the_micro_batch_loss_bitwise(path, loss_type):
    """At one micro-batch and one rank the step's denominator is the micro-batch's own."""
    theta = torch.randn(len(GROUP_SIZES), WIDTH, generator=torch.Generator().manual_seed(0))
    trainer = _trainer(path, loss_type, theta)
    batch = _collate(path, _rows(UNEVEN_LENGTHS))
    _, num_items_in_batch = Trainer.get_batch_samples(trainer, iter([batch]), 1, torch.device("cpu"))
    step = trainer.compute_loss(trainer.model, batch, num_items_in_batch=num_items_in_batch)
    micro_batch = trainer.compute_loss(trainer.model, batch)
    assert num_items_in_batch is not None
    assert torch.equal(step, micro_batch), (step, micro_batch)


def _data_parallel_worker(rank: int, path: str, loss_type: str, expected: torch.Tensor) -> None:
    """Rank r holds rows 2r and 2r+1 as two micro-batches; the averaged gradient is the one-rank step's. A pipeline
    replica holding the same rows as one batch divides by the same share."""
    rows = _rows(UNEVEN_LENGTHS)
    world_size = dist.get_world_size()
    gradient = _step_gradient(path, loss_type, rows, [[2 * rank], [2 * rank + 1]], world_size=world_size)
    # FSDP's reduce-scatter averages the replicas' gradients.
    dist.all_reduce(gradient)
    gradient /= world_size
    torch.testing.assert_close(gradient, expected, rtol=1e-6, atol=1e-7)

    if path == "plain":
        trainer = _trainer(path, loss_type, torch.zeros(len(rows), WIDTH), world_size)
        trainer.padding_value = 0
        trainer.args.max_length = 1 + WIDTH
        batch = _collate(path, rows[2 * rank : 2 * rank + 2])
        _, share = Trainer.get_batch_samples(trainer, iter([batch]), 1, torch.device("cpu"))
        assert torch.equal(trainer._pp_step_normalizer(trainer._pp_batch_transform(batch)), share)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("loss_type", LOSS_TYPES)
def test_two_data_parallel_ranks_train_the_one_rank_step(path, loss_type):
    expected = _step_gradient(path, loss_type, _rows(UNEVEN_LENGTHS), SPLITS["one micro-batch"])
    run_gloo_ranks(_data_parallel_worker, 2, path, loss_type, expected, pg_timeout=datetime.timedelta(seconds=60))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
