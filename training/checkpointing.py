"""Cooperative, single-process checkpoints for HTCondor exit-driven restart."""

from __future__ import annotations

import time

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class ExitCheckpointCallback(TrainerCallback):
    """Exit only after Trainer has saved weights, optimizer, RNG, and state."""

    def __init__(self, seconds: float) -> None:
        self.deadline = time.monotonic() + seconds
        self.exit_pending = False

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> TrainerControl:
        if time.monotonic() >= self.deadline and state.global_step < state.max_steps:
            self.exit_pending = True
            control.should_save = True
            # Keep best-checkpoint selection consistent at this extra save.
            control.should_evaluate = args.do_eval
        return control

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        if self.exit_pending:
            print(
                f"Checkpoint complete at step {state.global_step}; exiting 85",
                flush=True,
            )
            raise SystemExit(85)
