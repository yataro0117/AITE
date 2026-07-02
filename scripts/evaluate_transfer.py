#!/usr/bin/env python3
"""Evaluate black-box transfer of saved adversarial samples.

Loads adversarial samples saved by ``generate_attacks.py`` and evaluates their
classification accuracy on one or more target model checkpoints.

Usage example::

    python scripts/evaluate_transfer.py \\
        --dataset unipen_1a \\
        --attack aite \\
        --source_checkpoint checkpoints/unipen_1a_cnn3_best.pt \\
        --target_checkpoints checkpoints/unipen_1a_cnn4_best.pt \\
                             checkpoints/unipen_1a_blstm2_best.pt \\
        --adv_dir results/adversarial_samples/unipen_1a/aite
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate adversarial transfer across model architectures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["unipen_1a", "unipen_1b", "unipen_1c", "casia"])
    parser.add_argument("--attack", type=str, required=True,
                        help="Attack name (subdirectory under adv_dir).")
    parser.add_argument("--source_checkpoint", type=str, required=True,
                        help="Checkpoint used to generate the adversarial samples.")
    parser.add_argument("--target_checkpoints", nargs="+", required=True,
                        help="One or more target model checkpoints to evaluate against.")
    parser.add_argument("--adv_dir", type=str, required=True,
                        help="Directory containing sample_XXXXXX.npy files.")
    parser.add_argument("--out_dir", type=str, default="results/transfer_eval",
                        help="Where to write per-model CSV reports.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Cap on number of samples to evaluate (-1 = all).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "scripts/evaluate_transfer.py is a stub. "
        "Implement evaluation logic using aite.data, aite.models, and aite.metrics."
    )


if __name__ == "__main__":
    main()
