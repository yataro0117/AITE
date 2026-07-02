"""Adversarial attack implementations.

Submodules
----------
temporal_editing : Pure-numpy point insertion/deletion primitives.
aite             : AITE (Adversarial Iterative Temporal Editing) core attack.
baselines        : Gradient-based baseline attacks (FGSM, BIM, PGD, CW, MI-FGSM, etc.).
"""

from .temporal_editing import insert_midpoint, delete_point
from .aite import run_aite, aite_unipen, aite_casia
from .baselines import fgsm, bim, pgd, cw_l2, mi_fgsm, ni_fgsm, ti_mi_fgsm_1d

__all__ = [
    "insert_midpoint",
    "delete_point",
    "run_aite",
    "aite_unipen",
    "aite_casia",
    "fgsm",
    "bim",
    "pgd",
    "cw_l2",
    "mi_fgsm",
    "ni_fgsm",
    "ti_mi_fgsm_1d",
]
