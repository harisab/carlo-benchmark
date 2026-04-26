from __future__ import annotations

import numpy as np

from odmr.benchmark_config import (
    BenchmarkConfig,
    get_search_regions,
    get_width_candidates,
    variant_label,
    with_overrides,
)
from odmr.algorithms.common import (
    lorentzian_peak,
    peak_from_dip,
    process_signal_for_correlation,
    process_template_for_correlation,
)


def _legacy_to_normalization_mode(
    normalize_template: bool | None,
    demean: bool | None,
) -> str | None:
    """
    Backward-compatibility bridge from the old boolean API.
    """
    if normalize_template is None and demean is None:
        return None

    nrm = bool(normalize_template) if normalize_template is not None else False
    dm = bool(demean) if demean is not None else False

    if not dm and not nrm:
        return "raw"
    if not dm and nrm:
        return "l2"
    if dm and not nrm:
        return "demean"
    return "demean_l2"


def run_double_correlation(
    x: np.ndarray,
    y_dip: np.ndarray,
    *,
    cfg: BenchmarkConfig | None = None,
    normalization_mode: str | None = None,
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

    Current benchmark variants are controlled through cfg.normalization_mode.

    For correctness across raw / l1 / l2 / demean / demean_l1 / demean_l2,
    the pairwise score is computed from the combined two-peak kernel.
    """
    if cfg is None:
        cfg = BenchmarkConfig()

    if normalization_mode is None:
        normalization_mode = _legacy_to_normalization_mode(normalize_template, demean)

    cfg = with_overrides(
        cfg,
        normalization_mode=normalization_mode,
        min_width=min_width,
        max_width=max_width,
        width_step=width_step,
        template_height=template_height,
        center_step_bins=center_step_bins,
        restrict_window_mhz=restrict_window_mhz,
    )

    x = np.asarray(x, dtype=float)
    y_peak = peak_from_dip(y_dip)
    y_proc = process_signal_for_correlation(y_peak, cfg.normalization_mode)

    regions = get_search_regions(x, y_peak, cfg)
    left_centers = regions["left_centers"]
    right_centers = regions["right_centers"]

    width_candidates = get_width_candidates(cfg)

    best = None  # (score, gamma, xL, xR)

    for gamma in width_candidates:
        gamma = float(gamma)

        left_templates_raw = [
            lorentzian_peak(x, float(c), gamma, cfg.template_height)
            for c in left_centers
        ]
        right_templates_raw = [
            lorentzian_peak(x, float(c), gamma, cfg.template_height)
            for c in right_centers
        ]

        for iL, xL in enumerate(left_centers):
            tL = left_templates_raw[iL]

            for iR, xR in enumerate(right_centers):
                tR = right_templates_raw[iR]

                combined_raw = tL + tR
                combined_proc = process_template_for_correlation(
                    combined_raw,
                    cfg.normalization_mode,
                )

                score = float(np.dot(y_proc, combined_proc))

                if best is None or score > best[0]:
                    best = (score, gamma, float(xL), float(xR))

    if best is None:
        raise RuntimeError("Double correlation failed to find a valid solution.")

    score, gamma, xL, xR = best

    return {
        "name": "DoubleCorrelation",
        "benchmark_variant": variant_label(cfg),
        "f1_hat": float(min(xL, xR)),
        "f2_hat": float(max(xL, xR)),
        "gamma": float(gamma),
        "score": float(score),
        "used_cfg": cfg,
    }