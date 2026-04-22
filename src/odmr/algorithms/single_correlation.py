from __future__ import annotations

import numpy as np


def peak_from_dip(y_dip: np.ndarray) -> np.ndarray:
    """Convert dip-form ODMR data into peak-form."""
    return 1.0 - y_dip


def lorentzian_peak(
    x: np.ndarray,
    center: float,
    gamma: float,
    height: float,
) -> np.ndarray:
    """Lorentzian peak with gamma interpreted as HWHM-like width."""
    return height / (1.0 + ((x - center) / gamma) ** 2)


def correlate_same(y: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Simple same-length correlation using NumPy only."""
    full = np.correlate(y, template, mode="full")
    start = (len(full) - len(y)) // 2
    return full[start:start + len(y)]


def run_single_correlation(
    x: np.ndarray,
    y_dip: np.ndarray,
    *,
    min_width: float,
    max_width: float,
    width_step: float,
    template_height: float = 0.15,
    normalize_template: bool = False,
    demean: bool = True,
) -> dict:
    """
    Single-correlation ODMR estimator.

    Strategy:
    - convert dip spectrum to peak-form
    - correlate with one centered Lorentzian template for many widths
    - search best peak independently on left and right halves
    """
    y = peak_from_dip(y_dip).astype(float)
    if demean:
        y = y - np.mean(y)

    x = np.asarray(x, dtype=float)
    mid_idx = len(x) // 2
    xmid = float(np.mean(x))

    left_candidates = []
    right_candidates = []

    for gamma in np.arange(min_width, max_width + 1e-12, width_step):
        gamma = float(gamma)

        template = lorentzian_peak(x, xmid, gamma, template_height)

        if demean:
            template = template - np.mean(template)

        if normalize_template:
            nrm = np.linalg.norm(template)
            if nrm > 0:
                template = template / nrm

        corr = correlate_same(y, template)

        left_corr = corr[:mid_idx]
        right_corr = corr[mid_idx:]

        li = int(np.argmax(left_corr))
        ri = int(np.argmax(right_corr))

        left_candidates.append((gamma, float(left_corr[li]), float(x[li])))
        right_candidates.append((gamma, float(right_corr[ri]), float(x[mid_idx + ri])))

    best_left = max(left_candidates, key=lambda t: t[1])
    best_right = max(right_candidates, key=lambda t: t[1])

    f1_hat = float(min(best_left[2], best_right[2]))
    f2_hat = float(max(best_left[2], best_right[2]))

    return {
        "name": "SingleCorrelation",
        "f1_hat": f1_hat,
        "f2_hat": f2_hat,
        "gamma_left": float(best_left[0]),
        "gamma_right": float(best_right[0]),
        "score": float(best_left[1] + best_right[1]),
    }