from __future__ import annotations

import numpy as np

from odmr.project_defaults import BENCHMARK_DEFAULTS
from odmr.algorithms.common import (
    candidate_widths,
    lorentzian_peak,
    peak_space,
    process_vector,
    split_left_right_indices,
    template_score,
    two_peak_dip,
)


def _settings(settings: dict | None) -> dict:
    out = dict(BENCHMARK_DEFAULTS)
    if settings is not None:
        out.update(settings)
    return out


def run_double_correlation(
    x: np.ndarray,
    y_dip: np.ndarray,
    *,
    settings: dict | None = None,
) -> dict:
    cfg = _settings(settings)

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

    best_score = -np.inf
    best_f1 = float(x[left_indices[0]])
    best_f2 = float(x[right_indices[0]])
    best_gamma = float(widths[0])

    for gamma in widths:
        for i_left in left_indices:
            f1 = float(x[i_left])
            left_template = lorentzian_peak(
                x,
                f1,
                float(gamma),
                float(cfg["template_height"]),
            )

            for i_right in right_indices:
                f2 = float(x[i_right])
                right_template = lorentzian_peak(
                    x,
                    f2,
                    float(gamma),
                    float(cfg["template_height"]),
                )
                template = left_template + right_template
                template_processed = process_vector(template, normalization_mode)
                score = template_score(signal_processed, template_processed)

                if score > best_score:
                    best_score = score
                    best_f1 = f1
                    best_f2 = f2
                    best_gamma = float(gamma)

    best_fit = two_peak_dip(
        x,
        best_f1,
        best_f2,
        best_gamma,
        best_gamma,
        float(cfg["template_height"]),
    )

    return {
        "name": "DoubleCorrelation",
        "benchmark_variant": str(cfg.get("benchmark_variant", "double_correlation")),
        "f1_hat": float(best_f1),
        "f2_hat": float(best_f2),
        "gamma": float(best_gamma),
        "score": float(best_score),
        "used_settings": cfg,
        "best_fit": np.asarray(best_fit, dtype=float),
    }