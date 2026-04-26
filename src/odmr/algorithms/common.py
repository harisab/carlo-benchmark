from __future__ import annotations

import numpy as np

from odmr.benchmark_config import NORMALIZATION_MODES


DEMEAN_MODES = {"demean", "demean_l1", "demean_l2"}
L1_MODES = {"l1", "demean_l1"}
L2_MODES = {"l2", "demean_l2"}


def peak_from_dip(y_dip: np.ndarray) -> np.ndarray:
    """Convert dip-form ODMR data into peak-form."""
    return 1.0 - np.asarray(y_dip, dtype=float)


def lorentzian_peak(
    x: np.ndarray,
    center: float,
    gamma: float,
    height: float,
) -> np.ndarray:
    """Lorentzian peak with gamma interpreted as a width parameter."""
    return height / (1.0 + ((x - center) / gamma) ** 2)


def correlate_same(y: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Simple same-length correlation using NumPy only."""
    full = np.correlate(y, template, mode="full")
    start = (len(full) - len(y)) // 2
    return full[start:start + len(y)]


def process_signal_for_correlation(
    y_peak: np.ndarray,
    normalization_mode: str,
) -> np.ndarray:
    """
    Apply the signal-side processing implied by the benchmark variant.

    Current convention:
    - raw, l1, l2: signal unchanged
    - demean, demean_l1, demean_l2: subtract signal mean
    """
    if normalization_mode not in NORMALIZATION_MODES:
        raise ValueError(f"Unsupported normalization_mode: {normalization_mode}")

    y = np.asarray(y_peak, dtype=float).copy()

    if normalization_mode in DEMEAN_MODES:
        y = y - np.mean(y)

    return y


def process_template_for_correlation(
    template: np.ndarray,
    normalization_mode: str,
) -> np.ndarray:
    """
    Apply the template-side processing implied by the benchmark variant.

    Variants:
    - raw
    - l1
    - l2
    - demean
    - demean_l1
    - demean_l2
    """
    if normalization_mode not in NORMALIZATION_MODES:
        raise ValueError(f"Unsupported normalization_mode: {normalization_mode}")

    t = np.asarray(template, dtype=float).copy()

    if normalization_mode in DEMEAN_MODES:
        t = t - np.mean(t)

    if normalization_mode in L1_MODES:
        denom = float(np.sum(np.abs(t)))
        if denom > 0:
            t = t / denom

    elif normalization_mode in L2_MODES:
        denom = float(np.linalg.norm(t))
        if denom > 0:
            t = t / denom

    return t