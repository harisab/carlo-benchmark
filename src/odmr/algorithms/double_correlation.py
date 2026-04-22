from __future__ import annotations

import numpy as np

from odmr.benchmark_config import BenchmarkConfig, get_search_regions, get_width_candidates, with_overrides


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


def run_double_correlation(
    x: np.ndarray,
    y_dip: np.ndarray,
    *,
    cfg: BenchmarkConfig | None = None,
    min_width: float | None = None,
    max_width: float | None = None,
    width_step: float | None = None,
    template_height: float | None = None,
    normalize_template: bool | None = None,
    demean: bool | None = None,
    center_step_bins: int | None = None,
    restrict_window_mhz: float | None = None,
) -> dict:
    """
    Double-correlation ODMR estimator.

    This keeps backward compatibility with the earlier simple call signature,
    but now also supports a shared BenchmarkConfig.
    """
    if cfg is None:
        cfg = BenchmarkConfig()

    cfg = with_overrides(
        cfg,
        min_width=min_width,
        max_width=max_width,
        width_step=width_step,
        template_height=template_height,
        normalize_template=normalize_template,
        demean=demean,
        center_step_bins=center_step_bins,
        restrict_window_mhz=restrict_window_mhz,
    )

    x = np.asarray(x, dtype=float)
    y = peak_from_dip(y_dip)

    if cfg.demean:
        y = y - np.mean(y)

    regions = get_search_regions(x, y, cfg)
    left_centers = regions["left_centers"]
    right_centers = regions["right_centers"]

    width_candidates = get_width_candidates(cfg)

    best = None  # (score, gamma, xL, xR)

    for gamma in width_candidates:
        gamma = float(gamma)

        L = np.stack(
            [lorentzian_peak(x, float(c), gamma, cfg.template_height) for c in left_centers],
            axis=0,
        )
        R = np.stack(
            [lorentzian_peak(x, float(c), gamma, cfg.template_height) for c in right_centers],
            axis=0,
        )

        if cfg.demean:
            L = L - L.mean(axis=1, keepdims=True)
            R = R - R.mean(axis=1, keepdims=True)

        if not cfg.normalize_template:
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
        "benchmark_variant": f"{'norm' if cfg.normalize_template else 'raw'}_{cfg.width_mode}",
        "f1_hat": float(min(xL, xR)),
        "f2_hat": float(max(xL, xR)),
        "gamma": float(gamma),
        "score": float(score),
        "used_cfg": cfg,
    }