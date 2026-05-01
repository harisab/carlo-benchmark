from __future__ import annotations

import numpy as np

from odmr.algorithms.common import (
    merged_settings,
    candidate_widths,
    lorentzian_peak,
    peak_space,
    process_vector,
    split_left_right_indices,
    template_score,
    two_peak_dip,
)

def _best_side_match(
    x: np.ndarray,
    signal_processed: np.ndarray,
    candidate_indices: np.ndarray,
    widths: np.ndarray,
    template_height: float,
    normalization_mode: str,
) -> tuple[float, float, float]:
    best_score = -np.inf
    best_center = float(x[candidate_indices[0]])
    best_gamma = float(widths[0])

    for idx in candidate_indices:
        center = float(x[idx])
        for gamma in widths:
            template = lorentzian_peak(x, center, float(gamma), template_height)
            template_processed = process_vector(template, normalization_mode)
            score = template_score(signal_processed, template_processed)

            if score > best_score:
                best_score = score
                best_center = center
                best_gamma = float(gamma)

    return best_center, best_gamma, float(best_score)

def run_single_correlation(
    x: np.ndarray,
    y_dip: np.ndarray,
    *,
    settings: dict | None = None,
) -> dict:
    cfg = merged_settings(settings)

    x = np.asarray(x, dtype=float)
    y_dip = np.asarray(y_dip, dtype=float)

    widths = candidate_widths(cfg)
    normalization_mode = str(cfg["normalization_mode"])
    signal_processed = process_vector(peak_space(y_dip), normalization_mode)

    left_indices, right_indices = split_left_right_indices(
        len(x),
        int(cfg["center_step_bins"]),
    )
    if len(left_indices) == 0 or len(right_indices) == 0:
        raise ValueError("Trace is too short to split into left and right halves.")

    f1_hat, gamma_left, score_left = _best_side_match(
        x,
        signal_processed,
        left_indices,
        widths,
        float(cfg["template_height"]),
        normalization_mode,
    )
    f2_hat, gamma_right, score_right = _best_side_match(
        x,
        signal_processed,
        right_indices,
        widths,
        float(cfg["template_height"]),
        normalization_mode,
    )

    best_fit = two_peak_dip(
        x,
        f1_hat,
        f2_hat,
        gamma_left,
        gamma_right,
        float(cfg["template_height"]),
    )

    return {
        "name": "SingleCorrelation",
        "benchmark_variant": str(cfg.get("benchmark_variant", "single_correlation")),
        "f1_hat": float(f1_hat),
        "f2_hat": float(f2_hat),
        "gamma_left": float(gamma_left),
        "gamma_right": float(gamma_right),
        "score": float(score_left + score_right),
        "used_settings": cfg,
        "best_fit": np.asarray(best_fit, dtype=float),
    }