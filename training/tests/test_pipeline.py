"""No API calls or model downloads: data invariants and a tiny CPU training run."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

import pytest

from training.common import (
    encode_example,
    encode_prompt,
    make_prompt,
    normalize_action,
    normalize_thinking,
    prompt_hash,
    read_jsonl,
)
from training.evaluate import summarize
from training.prepare import prepare
from training.train import check_splits


def write_run(
    directory: Path, seed: int, agent: str, states: list[str], actions: list[str]
) -> None:
    rows = [
        {
            "step": step,
            "state": {"game_turn": step, "terminal": False},
            "original_state": state,
            "action": action,
            "thinking": "SECRET_TEACHER_REASONING",
            "score": "FUTURE_REWARD_MUST_NOT_LEAK",
        }
        for step, (state, action) in enumerate(zip(states, actions))
    ]
    (directory / f"snake_easy_4k_seed{seed}_{agent}.json").write_text(json.dumps(rows))


def prepare_fixture(tmp_path: Path, target_mode: str = "action") -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    for seed in (0, 6, 7):
        write_run(
            raw,
            seed,
            "reactive",
            [f"Current state {seed}", f"Next observed state {seed}"],
            ["U", "R"],
        )
    output = tmp_path / "prepared"
    prepare(raw, output, {6}, {7}, {"snake"}, {"reactive"}, target_mode=target_mode)
    return output


def test_preparation_excludes_reasoning_scores_and_future_states(
    tmp_path: Path,
) -> None:
    output = prepare_fixture(tmp_path)
    rows = read_jsonl(output / "train.jsonl")
    text = (output / "train.jsonl").read_text()
    assert "SECRET_TEACHER_REASONING" not in text
    assert "FUTURE_REWARD_MUST_NOT_LEAK" not in text
    assert rows[0]["prompt"][1]["content"] == "Current state 0"
    assert rows[0]["completion"] == "U"
    assert rows[1]["completion"] == "R"
    for split, seed in (("train", 0), ("validation", 6), ("test", 7)):
        assert {
            row["metadata"]["seed"] for row in read_jsonl(output / f"{split}.jsonl")
        } == {seed}


def test_cot_preparation_keeps_reasoning_only_in_target(tmp_path: Path) -> None:
    output = prepare_fixture(tmp_path, "cot")
    rows = read_jsonl(output / "train.jsonl", "cot")
    assert all(row["reasoning"] == "SECRET_TEACHER_REASONING" for row in rows)
    assert "SECRET_TEACHER_REASONING" not in json.dumps([row["prompt"] for row in rows])
    assert "FUTURE_REWARD_MUST_NOT_LEAK" not in (output / "train.jsonl").read_text()
    assert rows[0]["completion"] == "U"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["reasoning_included"] is True
    assert manifest["target_mode"] == "cot"
    with pytest.raises(ValueError, match="target mode differs"):
        read_jsonl(output / "train.jsonl")


@pytest.mark.parametrize(
    "trace",
    [
        None,
        "",
        "  ",
        "Still thinking...",
        "<think>Still thinking...</think>",
        "<think></think>",
        "<|im_end|>",
    ],
)
def test_unusable_reasoning_is_rejected(trace: Any) -> None:  # noqa: ANN401
    with pytest.raises(ValueError):
        normalize_thinking(trace)


def test_existing_thinking_tags_are_not_nested_and_plan_text_is_preserved() -> None:
    assert (
        normalize_thinking("<think>Reason\r\nmore</think>\nPlan: U")
        == "Reason\nmore\nPlan: U"
    )


def test_cot_placeholders_are_reported_without_guessing_reasoning(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    for seed in (0, 6, 7):
        write_run(
            raw, seed, "reactive", [f"State {seed}", f"Missing {seed}"], ["U", "R"]
        )
        path = raw / f"snake_easy_4k_seed{seed}_reactive.json"
        rows = json.loads(path.read_text())
        rows[1]["thinking"] = "Still thinking..."
        path.write_text(json.dumps(rows))
    output = tmp_path / "prepared"
    report = prepare(raw, output, {6}, {7}, {"snake"}, {"reactive"}, target_mode="cot")
    assert report["removed"]["placeholder_thinking"] == 3
    assert len(read_jsonl(output / "train.jsonl", "cot")) == 1


def test_all_agent_variants_stay_together_and_overlaps_are_purged(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    for seed in (0, 6, 7):
        for agent in ("reactive", "planning", "agile"):
            write_run(raw, seed, agent, ["Shared state", f"Seed {seed}"], ["U", "R"])
    # A contradictory label is retained and reported, never silently relabeled.
    write_run(raw, 0, "planning", ["Shared state", "Seed 0"], ["D", "L"])
    output = tmp_path / "prepared"
    report = prepare(
        raw, output, {6}, {7}, {"snake"}, {"reactive", "planning", "agile"}
    )
    keys = {
        split: {
            prompt_hash(row["prompt"]) for row in read_jsonl(output / f"{split}.jsonl")
        }
        for split in ("train", "validation", "test")
    }
    assert not keys["train"] & keys["validation"]
    assert not keys["train"] & keys["test"]
    assert not keys["validation"] & keys["test"]
    assert report["splits"]["train"]["conflicting_prompts"] == 1
    assert report["removed"]["train_heldout_overlap"] == 3
    assert len(report["files"]) == 9


@pytest.mark.parametrize("action", ["", "UD", "\\boxed{U}", "turn 0: R", "I", None])
def test_invalid_snake_labels_fail(action: Any) -> None:  # noqa: ANN401
    with pytest.raises(ValueError):
        normalize_action("snake", action)


def test_keep_means_continue_in_snake() -> None:
    assert normalize_action("snake", "Keep") == "S"
    assert "does not stop" in make_prompt("snake", "State")[0]["content"]


def test_malformed_labels_are_reported_without_guessing(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    for seed in (0, 6, 7):
        write_run(
            raw,
            seed,
            "agile",
            [f"State {seed}", "Bad label"],
            ["U", "Turn 1: U\nTurn 2: D"],
        )
    out = tmp_path / "prepared"
    report = prepare(raw, out, {6}, {7}, {"snake"}, {"agile"})
    assert report["removed"]["snake_invalid_action"] == 3
    assert len((out / "rejected.jsonl").read_text().splitlines()) == 3
    assert len(read_jsonl(out / "train.jsonl")) == 1
    with pytest.raises(ValueError, match="Invalid snake action"):
        prepare(
            raw,
            tmp_path / "strict",
            {6},
            {7},
            {"snake"},
            {"agile"},
            invalid_action_policy="error",
        )


@pytest.mark.parametrize(
    "text",
    ["<think>leak</think>", "Guidance from a Previous Thinking Model: U", "\\boxed{U}"],
)
def test_reasoning_or_answers_in_original_state_fail(text: str) -> None:
    with pytest.raises(ValueError):
        make_prompt("freeway", text)


def test_misaligned_state_fails(tmp_path: Path) -> None:
    write_run(tmp_path, 0, "reactive", ["State"], ["U"])
    path = next(tmp_path.glob("*.json"))
    rows = json.loads(path.read_text())
    rows[0]["state"]["game_turn"] = 1
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="State/step mismatch"):
        prepare(tmp_path, tmp_path / "out", {6}, {7}, {"snake"}, {"reactive"})


def test_split_overlap_is_rejected(tmp_path: Path) -> None:
    output = prepare_fixture(tmp_path)
    rows = read_jsonl(output / "train.jsonl")
    with pytest.raises(ValueError, match="seeds overlap"):
        check_splits(rows, rows)


@pytest.fixture
def tokenizer() -> PreTrainedTokenizerFast:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    vocab = {
        token: i
        for i, token in enumerate(
            [
                "[UNK]",
                "[PAD]",
                "[EOS]",
                "U",
                "D",
                "L",
                "R",
                "S",
                "I",
                "system",
                "user",
                "assistant",
                "<think>",
                "</think>",
                "State",
                "0",
                "6",
                "7",
            ]
        )
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    result = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    result.chat_template = (
        "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}"
        "{% if add_generation_prompt %}assistant {% if enable_thinking is defined and not enable_thinking %}<think> </think> {% endif %}{% endif %}"
    )
    return result


def test_loss_masks_entire_prompt_and_padding_but_keeps_action_and_eos(
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    from transformers import DataCollatorForSeq2Seq

    row = {"prompt": make_prompt("snake", "State 0"), "completion": "R"}
    encoded = encode_example(tokenizer, row, 4096)
    prefix = encode_prompt(tokenizer, row["prompt"])
    assert encoded["input_ids"][: len(prefix)] == prefix
    assert all(value == -100 for value in encoded["labels"][: len(prefix)])
    assert (
        tokenizer.decode([value for value in encoded["labels"] if value != -100])
        == "R [EOS]"
    )
    short = encode_example(
        tokenizer, {"prompt": make_prompt("freeway", "State"), "completion": "U"}, 4096
    )
    batch = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100)([encoded, short])
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
    assert (batch["labels"] != -100).sum().item() == 4
    with pytest.raises(ValueError, match="was not truncated"):
        encode_example(tokenizer, row, 10)


def test_cot_loss_supervises_trace_delimiters_action_and_eos(
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    from transformers import DataCollatorForSeq2Seq

    row = {
        "prompt": make_prompt("snake", "State 0", "cot"),
        "target_mode": "cot",
        "reasoning": "State 7",
        "completion": "R",
    }
    encoded = encode_example(tokenizer, row, 1024)
    prefix = encode_prompt(tokenizer, row["prompt"], thinking=True)
    assert encoded["input_ids"][: len(prefix)] == prefix
    assert all(label == -100 for label in encoded["labels"][: len(prefix)])
    supervised = [label for label in encoded["labels"] if label != -100]
    assert tokenizer.decode(supervised) == "<think> State 7 </think> R [EOS]"
    assert tokenizer.decode(prefix).endswith("assistant")
    short = encode_example(tokenizer, {**row, "reasoning": "State"}, 1024)
    batch = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100)([encoded, short])
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
    with pytest.raises(ValueError, match="was not truncated"):
        encode_example(tokenizer, row, len(encoded["input_ids"]) - 1)


def test_accuracy_counts_invalid_output_as_wrong() -> None:
    results = summarize(
        [
            {"game": "snake", "difficulty": "easy", "target": "U", "prediction": "U"},
            {"game": "snake", "difficulty": "easy", "target": "L", "prediction": "I"},
            {
                "game": "freeway",
                "difficulty": "hard",
                "target": "S",
                "prediction": "I choose S",
            },
        ]
    )
    assert results["overall"]["exact_action_accuracy"] == pytest.approx(1 / 3)
    assert results["overall"]["invalid_action_rate"] == pytest.approx(2 / 3)
    assert results["macro_game_accuracy"] == 0.25


@pytest.mark.parametrize("checkpoint_restart", [False, True])
@pytest.mark.parametrize("target_mode", ["action", "cot"])
def test_tiny_qwen_lora_train_save_reload_and_evaluate(
    tmp_path: Path,
    tokenizer: PreTrainedTokenizerFast,
    checkpoint_restart: bool,
    target_mode: str,
) -> None:
    import torch
    from safetensors.torch import load_file
    from transformers import Qwen3Config, Qwen3ForCausalLM

    data = prepare_fixture(tmp_path, target_mode)
    base = tmp_path / "tiny-base"
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=1024,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    )
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    adapter = tmp_path / "adapter"
    config = {
        "model": str(base),
        "target_mode": target_mode,
        "max_length": 1024,
        "lora": {"r": 2, "lora_alpha": 4, "target_modules": ["q_proj", "v_proj"]},
        "training": {
            "output_dir": str(adapter),
            "use_cpu": True,
            "max_steps": 2,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "learning_rate": 0.001,
            "report_to": "none",
            "save_strategy": "steps",
            "save_steps": 1,
            "eval_strategy": "steps",
            "eval_steps": 1,
            "load_best_model_at_end": True,
            "seed": 42,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "prediction_loss_only": True,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    repo = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        "-m",
        "training.train",
        "--config",
        str(config_path),
        "--data-dir",
        str(data),
        "--tokenization-report",
        str(tmp_path / "tokenization-report.json"),
    ]
    if checkpoint_restart:
        interrupted = subprocess.run(
            command + ["--checkpoint-after-seconds", "0.001"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert interrupted.returncode == 85, interrupted.stdout + interrupted.stderr
        checkpoint = adapter / "checkpoint-1"
        for name in (
            "adapter_model.safetensors",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "trainer_state.json",
        ):
            assert (checkpoint / name).is_file(), name
        command += ["--resume-from-checkpoint", str(checkpoint)]
    subprocess.run(
        command,
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    weights = load_file(adapter / "adapter_model.safetensors")
    preflight = json.loads((tmp_path / "tokenization-report.json").read_text())
    assert preflight["target_mode"] == target_mode
    assert preflight["splits"]["train"]["examples"] == 2
    assert json.loads((adapter / "trainer_state.json").read_text())["global_step"] == 2
    assert any(
        torch.count_nonzero(value).item() > 0
        for name, value in weights.items()
        if "lora_B" in name
    )
    assert math.isfinite(
        json.loads((adapter / "train_results.json").read_text())["train_loss"]
    )
    if target_mode == "cot":
        from peft import PeftModel

        run = json.loads((adapter / "run.json").read_text())
        assert run["config"]["target_mode"] == "cot"
        assert run["tokenization"]["train"]["supervised_tokens"] > 4
        restored = PeftModel.from_pretrained(
            Qwen3ForCausalLM.from_pretrained(base), adapter
        )
        prompt = read_jsonl(data / "train.jsonl", "cot")[0]["prompt"]
        ids = torch.tensor([encode_prompt(tokenizer, prompt, thinking=True)])
        with torch.inference_mode():
            generated = restored.generate(ids, max_new_tokens=4, do_sample=False)
        assert generated.shape[1] > ids.shape[1]
        return
    evaluation = tmp_path / "evaluation"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "training.evaluate",
            "--adapter",
            str(adapter),
            "--data",
            str(data / "test.jsonl"),
            "--train-data",
            str(data / "train.jsonl"),
            "--output-dir",
            str(evaluation),
            "--cpu",
            "--batch-size",
            "2",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    report = json.loads((evaluation / "metrics.json").read_text())
    assert report["overall"]["examples"] == 2
    assert 0 <= report["overall"]["exact_action_accuracy"] <= 1
    assert report["majority_action_baseline"]["overall"]["exact_action_accuracy"] == 0.5
    if not checkpoint_restart:
        from training.play import ActionPolicy

        sampled_policy = ActionPolicy(
            str(base), "main", adapter, True, 1024, 4, temperature=0.7
        )
        prompt = make_prompt("overcooked", "State 0")
        torch.manual_seed(3)
        first = sampled_policy(prompt)
        first_tokens = sampled_policy.last_generation_tokens
        torch.manual_seed(3)
        assert sampled_policy(prompt) == first
        assert sampled_policy.last_generation_tokens == first_tokens
        assert 1 <= first_tokens <= 4
        pytest.importorskip("pygame")
        gameplay = tmp_path / "gameplay"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "training.play",
                "--adapter",
                str(adapter),
                "--game",
                "snake",
                "--seeds",
                "6",
                "--max-steps",
                "3",
                "--cpu",
                "--output-dir",
                str(gameplay),
                "--save-gif",
            ],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        episode = json.loads((gameplay / "metrics.json").read_text())["episodes"][0]
        assert 1 <= episode["steps"] <= 3
        assert (gameplay / "seed-6/episode.gif").is_file()
