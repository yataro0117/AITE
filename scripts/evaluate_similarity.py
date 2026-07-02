#!/usr/bin/env python3
"""Compute perturbation similarity metrics for saved adversarial samples.

Computes L2, L-inf (coordinate space) and SSIM (image space) between original
and adversarial samples.

Usage example::

    python scripts/evaluate_similarity.py \\
        --dataset unipen_1a \\
        --attack aite \\
        --adv_dir results/adversarial_samples/unipen_1a/aite \\
        --out_dir results/similarity_eval/unipen_1a/aite
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute L2/SSIM similarity between original and adversarial samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["unipen_1a", "unipen_1b", "unipen_1c", "casia"])
    parser.add_argument("--attack", type=str, required=True,
                        help="Attack name label used in output filenames.")
    parser.add_argument("--adv_dir", type=str, required=True,
                        help="Directory containing adversarial sample_XXXXXX.npy files.")
    parser.add_argument("--out_dir", type=str, default="results/similarity_eval",
                        help="Output directory for metrics CSV.")
    parser.add_argument("--img_size", type=int, default=64,
                        help="Image side length for trajectory rasterization.")
    parser.add_argument("--line_width", type=float, default=1.0,
                        help="Stroke width for trajectory rasterization.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Parallel workers for metric computation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "scripts/evaluate_similarity.py is a stub. "
        "Implement metric computation using aite.metrics (trajectory, rendering)."
    )


if __name__ == "__main__":
    main()
