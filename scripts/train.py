#!/usr/bin/env python3
"""Train a classification model on Unipen or CASIA.

Usage examples::

    python scripts/train.py --dataset unipen_1a --arch cnn3
    python scripts/train.py --dataset unipen_1b --arch blstm2
    python scripts/train.py --dataset casia     --arch cnn1d
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AITE classification model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["unipen_1a", "unipen_1b", "unipen_1c", "casia"],
        help="Dataset to train on.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=["cnn3", "cnn4", "blstm2", "transformer", "cnn1d", "blstm", "transformer_casia"],
        help="Model architecture.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file. Defaults to configs/<dataset>.yaml.",
    )
    parser.add_argument("--save_dir", type=str, default="pretrained/", help="Checkpoint directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count from config.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate from config.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "scripts/train.py is a stub. "
        "Implement training logic using aite.data, aite.models, and aite.reproducibility."
    )


if __name__ == "__main__":
    main()
