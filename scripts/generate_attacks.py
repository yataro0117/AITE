#!/usr/bin/env python3
"""Generate adversarial samples for a dataset and save them to disk.

Usage examples::

    python scripts/generate_attacks.py --dataset unipen_1a --attack aite \\
        --checkpoint pretrained/unipen_1a_cnn3_best.pt --indices 0 5 10

    python scripts/generate_attacks.py --dataset casia --attack fgsm \\
        --checkpoint pretrained/casia_cnn1d_best.pt --split test
"""

from __future__ import annotations

import argparse


SUPPORTED_ATTACKS = [
    "aite",
    "fgsm",
    "bim",
    "pgd",
    "cw_l2",
    "mi_fgsm",
    "ni_fgsm",
    "ti_mi_fgsm_1d",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate adversarial samples using AITE attacks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["unipen_1a", "unipen_1b", "unipen_1c", "casia"],
    )
    parser.add_argument(
        "--attack",
        type=str,
        required=True,
        choices=SUPPORTED_ATTACKS,
        help="Attack algorithm to run.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the source model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file. Defaults to configs/<dataset>.yaml.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
    )
    parser.add_argument(
        "--indices",
        nargs="*",
        type=int,
        default=None,
        help="Sample indices to attack. If omitted, attack all samples in the split.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/adversarial_samples",
        help="Root directory for saved adversarial .npy files.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run attack even if output file already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "scripts/generate_attacks.py is a stub. "
        "Implement generation logic using aite.attacks and aite.data."
    )


if __name__ == "__main__":
    main()
