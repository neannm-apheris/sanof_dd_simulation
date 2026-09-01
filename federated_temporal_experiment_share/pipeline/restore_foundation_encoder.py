#!/usr/bin/env python3
"""Restore exact foundation encoder weights in a task-specific Chemprop checkpoint.

Chemprop 2.2 requires ``--freeze-encoder`` to be paired with ``--checkpoint``,
while ``--from-foundation`` creates the required new task head.  The runner first
uses a numerically inert one-epoch CLI initialization to create that task-shaped
checkpoint and its training-only scaler.  This helper then restores the message
passing and aggregation weights bit-for-bit before the real CLI head-only fit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from chemprop.models.utils import load_model, load_output_columns, save_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-checkpoint", required=True, type=Path)
    parser.add_argument("--foundation", required=True, type=Path)
    parser.add_argument(
        "--chemeleon-message-passing",
        action="store_true",
        help="Treat --foundation as Chemprop's cached chemeleon_mp.pt state-dict payload.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--init-lr", type=float, default=1e-4)
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--final-lr", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_model = load_model(args.task_checkpoint)
    if args.chemeleon_message_passing:
        payload = torch.load(args.foundation, map_location="cpu", weights_only=True)
        expected_message_passing = payload["state_dict"]
        expected_aggregation: dict[str, torch.Tensor] = {}
    else:
        foundation_model = load_model(args.foundation)
        expected_message_passing = foundation_model.message_passing.state_dict()
        expected_aggregation = foundation_model.agg.state_dict()
    task_model.message_passing.load_state_dict(expected_message_passing, strict=True)
    task_model.agg.load_state_dict(expected_aggregation, strict=True)
    if hasattr(task_model.bn, "reset_parameters"):
        task_model.bn.reset_parameters()
    for name in ("warmup_epochs", "init_lr", "max_lr", "final_lr"):
        value = getattr(args, name)
        setattr(task_model, name, value)
        task_model.hparams[name] = value
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_model(args.output, task_model, load_output_columns(args.task_checkpoint))

    restored = load_model(args.output)
    for name, expected in expected_message_passing.items():
        actual = restored.message_passing.state_dict()[name]
        if not torch.equal(actual, expected):
            raise RuntimeError(f"Encoder tensor was not restored exactly: {name}")
    for name, expected in expected_aggregation.items():
        actual = restored.agg.state_dict()[name]
        if not torch.equal(actual, expected):
            raise RuntimeError(f"Aggregation tensor was not restored exactly: {name}")
    print(f"Exact foundation encoder restored to {args.output}")


if __name__ == "__main__":
    main()
