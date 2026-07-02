#!/usr/bin/env python3
"""Aggregate evaluation results into paper-ready summary tables.

Reads CSV files produced by ``evaluate_transfer.py`` and
``evaluate_similarity.py`` and writes a combined summary table (CSV + LaTeX).

Usage example::

    python scripts/make_tables.py \\
        --eval_root results/ \\
        --dataset unipen_1a \\
        --out_dir results/tables/unipen_1a
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation CSVs into summary tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--eval_root", type=str, default="results/",
                        help="Root directory containing evaluation output subdirectories.")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["unipen_1a", "unipen_1b", "unipen_1c", "casia", "all"],
                        help="Dataset to aggregate; 'all' processes every dataset.")
    parser.add_argument("--out_dir", type=str, default="results/tables",
                        help="Directory for output CSV and LaTeX files.")
    parser.add_argument("--attacks", nargs="*", default=None,
                        help="Filter to specific attack names. Defaults to all found.")
    parser.add_argument("--no_latex", action="store_true",
                        help="Skip LaTeX table generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "scripts/make_tables.py is a stub. "
        "Implement aggregation logic to read evaluation CSVs and produce summary tables."
    )


if __name__ == "__main__":
    main()
