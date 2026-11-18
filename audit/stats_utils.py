"""Small statistics helpers shared by the audit modules."""
from __future__ import annotations
from math import sqrt
import numpy as np


def wilson_ci(successes: int, total: int, z: float = 1.96):
    """95% Wilson score interval for a proportion (copied from Module 3, cell 'UNCERTAINTY OF SUBGROUP RECALL')."""
    if total == 0:
        return np.nan, np.nan
    p = successes / total
    denom = 1 + z ** 2 / total
    centre = (p + z ** 2 / (2 * total)) / denom
    half = z * sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / denom
    return centre - half, centre + half


def safe_divide(a, b):
    """Module 3's safe_divide: NaN instead of a ZeroDivisionError."""
    return a / b if b != 0 else np.nan
