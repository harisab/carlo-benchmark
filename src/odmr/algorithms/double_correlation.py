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


def run_double_correlation(
    x: np.ndarray,
    y_dip: np.ndarray,
    *,
    min_width: float,
    max_width: float,
    width_step: float,
    template_height: float = 0.15,
    normalize_template: bool = False,
    demean: bool = True,
    center_step_bins: int = 1,
    restrict_window_mhz: float | None = None,
) -> dict:
    """
    Double-correlation ODMR estimator.

    Strategy:
    - convert dip spectrum to peak-form
    - build left and right Lorentzian template banks
    - jointly score left/right resonance pairs
    - return the best pair
    """
    x = np.asarray(x, dtype=float)
    y = peak_from_dip(y_dip).astype(float)

    if demean:
        y = y - np.mean(y)

    mid_idx = len(x) // 2
    step = max(1, int(center_step_bins))

    left_centers_all = x[: mid_idx + 1 : step]
    right_centers_all = x[mid_idx::step]

    if restrict_window_mhz is not None:
        iL = int(np.argmax(y[:mid_idx]))
        iR = mid_idx + int(np.argmax(y[mid_idx:]))

        guess_L = float(x[iL])
        guess_R = float(x[iR])

        left_centers = left_centers_all[np.abs(left_centers_all - guess_L) <= restrict_window_mhz]
        right_centers = right_centers_all[np.abs(right_centers_all - guess_R) <= restrict_window_mhz]

        if len(left_centers) < 1:
            left_centers = left_centers_all
        if len(right_centers) < 1:
            right_centers = right_centers_all
    else:
        left_centers = left_centers_all
        right_centers = right_centers_all

    best = None  # (score, gamma, xL, xR)

    for gamma in np.arange(min_width, max_width + 1e-12, width_step):
        gamma = float(gamma)

        L = np.stack(
            [lorentzian_peak(x, float(c), gamma, template_height) for c in left_centers],
            axis=0,
        )
        R = np.stack(
            [lorentzian_peak(x, float(c), gamma, template_height) for c in right_centers],
            axis=0,
        )

        if demean:
            L = L - L.mean(axis=1, keepdims=True)
            R = R - R.mean(axis=1, keepdims=True)

        if not normalize_template:
            sL = L @ y
            sR = R @ y

            iL = int(np.argmax(sL))
            iR = int(np.argmax(sR))

            score = float(sL[iL] + sR[iR])
            xL = float(left_centers[iL])
            xR = float(right_centers[iR])

            if best is None or score > best[0]:
                best = (score, gamma, xL, xR)

        else:
            sL = L @ y
            sR = R @ y
            numer = sL[:, None] + sR[None, :]

            LL = np.sum(L * L, axis=1)
            RR = np.sum(R * R, axis=1)
            LR = L @ R.T

            denom = np.sqrt(np.maximum(LL[:, None] + RR[None, :] + 2.0 * LR, 1e-18))
            score_mat = numer / denom

            flat_idx = int(np.argmax(score_mat))
            iL, iR = np.unravel_index(flat_idx, score_mat.shape)

            score = float(score_mat[iL, iR])
            xL = float(left_centers[iL])
            xR = float(right_centers[iR])

            if best is None or score > best[0]:
                best = (score, gamma, xL, xR)

    if best is None:
        raise RuntimeError("Double correlation failed to find a valid solution.")

    score, gamma, xL, xR = best

    return {
        "name": "DoubleCorrelation",
        "f1_hat": float(min(xL, xR)),
        "f2_hat": float(max(xL, xR)),
        "gamma": float(gamma),
        "score": float(score),
    }