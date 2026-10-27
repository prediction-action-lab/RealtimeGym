# Qwen3-8B action-only supervised fine-tuning

This experiment teaches `Qwen/Qwen3-8B` to predict the action recorded in
`SALT-NLP/RealtimeGym`, given the game rules and the current observed state.
The recorded action is an imitation label, not a guarantee of optimal play.

An example has this shape:

```text
system: Game rules and a request to return one action letter.
user:   The trajectory's original_state, including past action history if present.
target: R
```

The `thinking` field is excluded from both input and target. The current step's
`score`, future states, and agent/budget metadata never enter the model prompt.
Only the action and end-of-response token contribute to the loss; prompt and
padding labels are `-100`. Qwen's chat template is used with
`enable_thinking=False` for both training and evaluation. Its empty thinking
delimiter, if inserted by the template, is part of the masked prompt, not a
teacher reasoning trace.

## Install

For CHTC, use the [container, staging, and HTCondor guide](chtc/README.md).
It includes a real-model smoke job and automatic checkpoint/resume support.

Run all commands below **from the repository root**. This is a separate `uv`
project so the GPU training stack does not change the game runtime dependencies.
Python 3.11 is selected by `training/.python-version`.

```bash
uv sync --project training --locked
```

The default configuration uses BF16 LoRA on the attention and MLP projections,
rank 16, learning rate `1e-4`, three epochs, gradient checkpointing, microbatch
size 1, and gradient accumulation 16. These are starting settings, not tuned
results. Use an A100/H100 with a driver compatible with the installed PyTorch
CUDA build. Actual memory use depends on sequence lengths; the dry run below
checks lengths without loading model weights.

## Prepare the dataset

```bash
uv run --project training python -m training.prepare
```

This downloads the raw trajectory JSON files (roughly 665 MiB) using
`huggingface_hub`, resolves the dataset revision to a commit, and writes:

```text
data/sft/train.jsonl
data/sft/validation.jsonl
data/sft/test.jsonl
data/sft/manifest.json
data/sft/rejected.jsonl
```

The dataset's nested state schemas differ across games, so preparation reads
each raw JSON file directly instead of relying on automatic dataset loading.
`original_state` preserves the description used by the demonstrator, including
Overcooked recipe information that is not all present in its structured state.

By default, seed 6 is validation, seed 7 is test, and the remaining seeds are
training. All games, difficulties, time budgets, and agent paradigms with the
same seed stay together. Exact repeated prompts are removed across splits with
priority test, then validation, then training; whitespace and `t_0`/`t_1`
notation are normalized for that check. Duplicate prompt/action pairs within a
split are also removed. Conflicting labels for the same prompt are retained
and counted in the manifest rather than assigning an invented "best" label.

This prevents exact prompt leakage, but a seed holdout is not a guarantee of
unseen layouts or novel game dynamics. The dataset contains repeated scenarios
and has limited seed diversity. Inspect the per-game counts and conflicts in
`manifest.json` before interpreting results.

Preparation excludes invalid actions and records their source steps and raw
labels in `rejected.jsonl`. Some published `action` fields contain entire plans,
equations, or unknown letters; the pipeline never guesses an action from them.
Use `--invalid-action-policy error` to fail instead. Missing state descriptions,
suspicious reasoning markers in the state text, and state/step misalignment
always cause an error. Terminal observations are skipped. Known action names such as Snake's `Keep` are mapped
to their environment equivalent (`S`, continuing in the current direction).
No reward filtering or choice of a preferred teacher is applied by default.

To use a local snapshot or choose a subset:

```bash
uv run --project training python -m training.prepare \
  --input-dir /path/to/raw-json \
  --games snake \
  --agents reactive agile \
  --output-dir data/snake-sft
```

Use `--revision COMMIT` to download a specific dataset revision, and
`--validation-seeds`/`--test-seeds` to choose different held-out seeds. Output
directories must be empty; choose a new directory to preserve previous runs.

## Validate and train

```bash
# Downloads only tokenizer files; validates masks, lengths, and split overlap.
uv run --project training python -m training.train --dry-run

# Short GPU smoke run before the complete experiment.
uv run --project training python -m training.train \
  --max-steps 5 --output-dir outputs/qwen3-8b-smoke

# Full LoRA fine-tuning.
uv run --project training python -m training.train
```

Configuration is in `training/configs/qwen3-8b.json`, with the Qwen3 model
revision pinned for reproducibility. Increase `max_length` if
preflight rejects a long example; inputs are never silently truncated. Paths
in the configuration and command line are relative to the working directory.
To train on another prepared directory, pass `--data-dir data/snake-sft`.

Validation loss selects the best epoch checkpoint. The final directory
`outputs/qwen3-8b-action-lora` contains the LoRA adapter, tokenizer, metrics,
trainer state, and `run.json` with model commit, configuration, and data hashes.
The base model weights remain separate. No model is uploaded automatically.

Resume a checkpoint with the same configuration and prepared data:

```bash
uv run --project training python -m training.train \
  --resume-from-checkpoint outputs/qwen3-8b-action-lora/checkpoint-N
```

For multiple GPUs on one host:

```bash
uv run --project training torchrun --standalone --nproc_per_node=2 \
  -m training.train
```

This uses distributed data parallelism: each GPU holds a full copy of the base
model. Effective batch size is microbatch size × accumulation steps × GPU
count. Adjust accumulation if you want to keep the effective batch size fixed.

## Evaluate recorded-action prediction

For actual game episodes and GIF replays, see the [gameplay inference guide](chtc/PLAY.md).
For Overcooked with 8 game seeds × 4 sampling seeds per difficulty, see the
[CHTC Overcooked evaluation guide](chtc/OVERCOOKED.md).
`training.evaluate` below measures agreement with dataset labels;
`training.play` runs the policy in the game environments and measures scores.

Evaluate the original model and the adapter on the **same held-out file**:

```bash
uv run --project training python -m training.evaluate \
  --output-dir outputs/eval-base

uv run --project training python -m training.evaluate \
  --adapter outputs/qwen3-8b-action-lora \
  --output-dir outputs/eval-finetuned
```

Use validation data while making training choices (`--data
data/sft/validation.jsonl`); reserve test results for the final comparison.
For a different prepared directory also set `--train-data`, which is used
only to fit a majority-action baseline and check input overlap.

Evaluation generates responses with thinking disabled and deterministic greedy
decoding. It does not constrain outputs to valid actions or extract a letter
from an explanation. After stripping surrounding whitespace, the whole response
must equal the recorded action. Invalid responses count as incorrect.

`metrics.json` reports overall action accuracy, per-game and per-difficulty
accuracy, macro-average accuracy across games, invalid-action rate, confusion
counts, and a training-set majority-action baseline. `predictions.jsonl` keeps
each prediction and its source step for inspection. `--limit N` is available
for debugging and is recorded in the metrics; do not use a small prefix as a
benchmark result. Evaluation currently runs on one GPU.

This measures agreement with demonstrations. Several actions can be reasonable
in a state, and recorded runs can disagree. Improved prediction accuracy does
not establish improved game score; live environment rollouts are a separate
follow-up evaluation.

## Tests

```bash
uv sync --project training --locked --extra gameplay
uv run --project training python -m pytest -c training/pyproject.toml training/tests
uv run --project training ruff check training
uv run --project training ruff format --check training
uv run --project training ty check --project training
```

Tests cover reasoning/reward exclusion, step alignment, grouped splits,
deduplication, conflicting labels, loss/padding masks, overlength rejection,
and invalid predictions. A tiny randomly initialized Qwen3 model exercises
LoRA training, saving, reloading, and evaluation on CPU without downloading
model weights or contacting an API.

See [DATA_AUDIT.md](DATA_AUDIT.md) for the full-dataset preparation and real
Qwen3 tokenizer checks performed for this implementation.

References: [dataset](https://huggingface.co/datasets/SALT-NLP/RealtimeGym),
[Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B),
[Transformers Trainer](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer).
