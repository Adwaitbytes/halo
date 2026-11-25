"""DPO trains a templated completion's tokens and nothing after them: a turn the template closes with
``<terminator>\\n`` (Qwen3.5's ``<|im_end|>\\n``) or with a role marker other than ``eos_token`` (Gemma 4's
``<turn|>\\n``) gets no second terminator, and gpt-oss, whose turns end on ``eos_token``, tokenizes as TRL's own path
does."""

import pytest
from datasets import Dataset
from transformers import GenerationConfig
from trl import DPOConfig, DPOTrainer

from halo.data.pipeline.preferences import render_preference_row
from halo.data.turn_markers import resolve_eos_token_ids
from halo.trainers.preference.dpo import DistributedDPOTrainer
from tests.common.models import GEMMA4_26B_A4B, GPT_OSS_20B_OPENAI, QWEN3_5_9B
from tests.common.tokenizers import load_cached_config, load_cached_tokenizer

ROWS = [
    {
        "prompt": [{"role": "user", "content": question}],
        "chosen": [{"role": "assistant", "content": answer}],
        "rejected": [{"role": "assistant", "content": wrong}],
    }
    for question, answer, wrong in (
        ("What is 2+2?", "4.", "5."),
        ("What is 2+2, written as a digit and not as a word?", "The answer is 4.", "It is 22, by concatenation."),
    )
]
COLUMNS = ["prompt_ids", "chosen_ids", "rejected_ids"]


def _rendered_rows(tokenizer) -> Dataset:
    """The rows the DPO script hands the trainer."""
    return Dataset.from_list([render_preference_row(dict(row), tokenizer) for row in ROWS])


def _policy_configs(model) -> tuple:
    """A loaded policy's config and generation config; Qwen3.5-9B ships no ``generation_config.json``, and these rows
    end on no tool-call stop only a shipped one lists."""
    config = load_cached_config(model)
    return config, GenerationConfig.from_model_config(config)


def _dpo_config(tmp_path, max_length=None) -> DPOConfig:
    return DPOConfig(
        output_dir=str(tmp_path),
        use_cpu=True,
        bf16=False,
        report_to=[],
        dataset_num_proc=1,
        max_length=max_length,
        truncation_mode="keep_start",
    )


def _prepare(model, tokenizer, rows, tmp_path, max_length=None) -> Dataset:
    trainer = object.__new__(DistributedDPOTrainer)
    trainer._tokenizer = tokenizer
    trainer._is_vlm = False
    # What the constructor captures off the policy before TRL prepares the splits.
    trainer._turn_end_configs = _policy_configs(model)
    return trainer._prepare_dataset(rows, tokenizer, _dpo_config(tmp_path, max_length), "train")


@pytest.mark.parametrize("model", [QWEN3_5_9B, GEMMA4_26B_A4B], ids=["qwen3.5", "gemma4"])
def test_a_completion_ends_on_the_templates_own_terminator(model, tmp_path):
    tokenizer = load_cached_tokenizer(model)
    terminators = resolve_eos_token_ids(tokenizer, *_policy_configs(model))
    rows = _rendered_rows(tokenizer)

    prepared = _prepare(model, tokenizer, rows, tmp_path)

    for rendered, tokenized in zip(rows, prepared, strict=True):
        for side in ("chosen", "rejected"):
            ids = tokenized[f"{side}_ids"]
            assert tokenizer.decode(ids) == rendered[side], f"{side} trains tokens the template never rendered"
            assert sum(token_id in terminators for token_id in ids) == 1, f"{side}: {tokenizer.decode(ids)!r}"


def _trl_prepare(tokenizer, rows, tmp_path, max_length=None) -> Dataset:
    trl_stub = object.__new__(DPOTrainer)
    trl_stub._tokenizer = tokenizer
    return DPOTrainer._prepare_dataset(trl_stub, rows, tokenizer, _dpo_config(tmp_path, max_length), "train")


def test_gpt_oss_tokenizes_as_trls_own_path(tmp_path):
    tokenizer = load_cached_tokenizer(GPT_OSS_20B_OPENAI)
    rows = _rendered_rows(tokenizer)

    prepared = _prepare(GPT_OSS_20B_OPENAI, tokenizer, rows, tmp_path)

    assert (
        prepared.select_columns(COLUMNS).to_list()
        == _trl_prepare(tokenizer, rows, tmp_path).select_columns(COLUMNS).to_list()
    )


def test_a_prompt_that_fills_max_length_drops_its_row_as_trl_does(tmp_path):
    """Under ``keep_start`` the collator would cut every completion token of such a row."""
    tokenizer = load_cached_tokenizer(GPT_OSS_20B_OPENAI)
    rows = _rendered_rows(tokenizer)
    max_length = len(_prepare(GPT_OSS_20B_OPENAI, tokenizer, rows, tmp_path)[1]["prompt_ids"])

    prepared = _prepare(GPT_OSS_20B_OPENAI, tokenizer, rows, tmp_path, max_length)

    assert prepared["prompt"] == [rows[0]["prompt"]]
    assert (
        prepared.select_columns(COLUMNS).to_list()
        == _trl_prepare(tokenizer, rows, tmp_path, max_length).select_columns(COLUMNS).to_list()
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
