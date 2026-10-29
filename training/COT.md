# Qwen3-8B reasoning-and-action LoRA

This is a separate adapter trained from the same frozen `Qwen/Qwen3-8B` base
revision as the action-only experiment. It uses the `thinking` field from
`SALT-NLP/RealtimeGym` as a supervised response target alongside the recorded
action. The existing action-only configuration and prepared data remain usable.

```text
system: Game rules; request reasoning inside <think> tags and one final action.
user:   original_state for this step
target: <think>
        recorded thinking text
        </think>

        R
```

The teacher trace is never appended to the input state. Loss covers the entire
reasoning response, its delimiters, the canonical recorded action, and EOS.
Prompt and padding labels stay masked to `-100`. The Qwen template uses
`enable_thinking=True`, so it does not inject the empty thinking block used
by action-only training. Existing `<think>`/`</think>` delimiters in the raw
field are removed before wrapping the remaining text exactly once. Text after
an existing closing tag (for example, the teacher's plan) is retained inside
the new reasoning target. The final action always comes from `action`.

Missing/empty traces and placeholders such as `Still thinking...` are excluded
and counted in the manifest and rejection log. Teacher traces can be partial,
stale, or inconsistent with the executed action, particularly for planning
agents. They are imitation targets, not verified solutions. No reward filtering,
teacher selection, or action extraction from reasoning is performed.

The same seeds, split priority, and prompt/action deduplication are used as
before: seeds 0–5 train, 6 validation, 7 test; exact overlapping input prompts
are removed across splits. The first eligible trace for a repeated prompt and
action is retained, rather than treating every repeated explanation as an
additional example. Removing placeholders means this training subset differs
from the action-only dataset; report both manifests when comparing adapters.

## Prepare and check locally

From the repository root:

```bash
uv run --project training python -m training.prepare \
    --target-mode cot \
    --revision 044d7e301056baa6b88134c6e7080579ca662af6 \
    --output-dir data/sft-cot

uv run --project training python -m training.train \
    --config training/configs/qwen3-8b-cot.json \
    --data-dir data/sft-cot --dry-run \
    --tokenization-report data/sft-cot/tokenization-report.json
```

For an existing raw snapshot, add `--input-dir /path/to/raw-json` to preparation.
Keep the dataset commit in the experiment notes when using a local snapshot.
The dry run loads only the tokenizer; it does not load model weights or run
held-out action prediction. It checks masks, split separation, and full sequence
lengths, and fails rather than silently truncating any reasoning or state.

Configuration: `training/configs/qwen3-8b-cot.json`. It preserves BF16, rank 16,
alpha 32, dropout 0.05, all seven attention/MLP projections, learning rate 1e-4,
three epochs, microbatch 1, accumulation 16, and seeds 42 from the first run.
The context limit is increased from 4096 to 32768 for the longer teacher traces.
Long sequences increase memory use and training time; the GPU smoke run checks
the existing 80 GB GPU request before the full run. Dependencies are unchanged.

## Package for CHTC

```bash
python3 -m training.chtc.bundle \
    --data-dir data/sft-cot \
    --config training/configs/qwen3-8b-cot.json \
    --output-dir outputs/chtc-cot-bundle
```

The selected config is stored as `model-config.json` beside the archive and as
`training/configs/qwen3-8b.json` inside it, the canonical path used by the worker.
The CoT splits are placed at `data/sft` inside the archive. The builder rejects
data/config target-mode mismatches. Each cluster job has an isolated sandbox;
its new adapter is saved at `outputs/adapter`.

Upload this directory into a **new** directory on the CHTC access point. Reuse
the existing `realtimegym-sft.sif` and `qwen3-8b-cache.tar` in the original
staging directory. There is no container rebuild or base-model download for
this experiment. The generated `train.sub` points at the same staging inputs
and names its results `cot-MODE-CLUSTER.PROCESS-results.tar.gz`.

In the new bundle directory on CHTC:

```bash
chmod +x run.sh
condor_submit -dry-run cot-smoke.ad train.sub -append 'mode = smoke'
condor_submit train.sub -append 'mode = smoke'
```

The smoke job uses five optimizer updates, 128 training examples and 16
validation examples selected by descending trace text length. CoT smoke uses
accumulation 1 to test long sequences with five short updates; full training
retains accumulation 16. As before, smoke exits 85 after its first update,
transfers its checkpoint, and resumes to update five. Confirm successful
restart and final exit code 0 before submitting the full run:

```bash
condor_history SMOKE_JOB_ID -limit 1 -af ClusterId ProcId JobStatus ExitCode
condor_submit -dry-run cot-train.ad train.sub -append 'mode = train'
condor_submit train.sub -append 'mode = train'
```

Inspect the preview arguments and use `-append` for overrides. Full training
uses the same 250-update save/evaluation interval and three-hour exit-driven
checkpoint/resume mechanism as the original run. Validation loss selects the
final adapter; no separate held-out action-accuracy evaluation is submitted.

For direct GPU training instead of CHTC:

```bash
uv run --project training python -m training.train \
    --config training/configs/qwen3-8b-cot.json --data-dir data/sft-cot
```

The existing four-token, thinking-disabled gameplay commands are for the
action-only adapter. This adapter needs thinking-enabled prompts, enough output
tokens for its trace, and final-action parsing after `</think>` when used later
for inference. Do not interpret an unfinished reasoning response as an action.

See [COT_DATA_AUDIT.md](COT_DATA_AUDIT.md) for the prepared counts and real-tokenizer checks.
