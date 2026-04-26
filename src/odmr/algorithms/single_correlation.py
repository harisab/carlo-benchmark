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
    correlate_same,
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


def run_single_correlation(
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
) -> dict:
    """
    Single-correlation ODMR estimator.

    Current benchmark variants are controlled through cfg.normalization_mode.

    Backward compatibility:
    - old normalize_template / demean booleans still map into the new mode system
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
    )

    x = np.asarray(x, dtype=float)
    y_peak = peak_from_dip(y_dip)
    y_proc = process_signal_for_correlation(y_peak, cfg.normalization_mode)

    regions = get_search_regions(x, y_peak, cfg)
    left_slice = regions["left_slice"]
    right_slice = regions["right_slice"]

    xmid = float(np.mean(x))
    width_candidates = get_width_candidates(cfg)

    left_candidates = []
    right_candidates = []

    for gamma in width_candidates:
        gamma = float(gamma)

        template_raw = lorentzian_peak(x, xmid, gamma, cfg.template_height)
        template_proc = process_template_for_correlation(template_raw, cfg.normalization_mode)

        corr = correlate_same(y_proc, template_proc)

        left_corr = corr[left_slice]
        right_corr = corr[right_slice]

        li_local = int(np.argmax(left_corr))
        ri_local = int(np.argmax(right_corr))

        li = li_local if left_slice.start == 0 else left_slice.start + li_local
        ri = ri_local if right_slice.start == 0 else right_slice.start + ri_local

        left_candidates.append((gamma, float(corr[li]), float(x[li])))
        right_candidates.append((gamma, float(corr[ri]), float(x[ri])))

    best_left = max(left_candidates, key=lambda t: t[1])
    best_right = max(right_candidates, key=lambda t: t[1])

    f1_hat = float(min(best_left[2], best_right[2]))
    f2_hat = float(max(best_left[2], best_right[2]))

    return {
        "name": "SingleCorrelation",
        "benchmark_variant": variant_label(cfg),
        "f1_hat": f1_hat,
        "f2_hat": f2_hat,
        "gamma_left": float(best_left[0]),
        "gamma_right": float(best_right[0]),
        "score": float(best_left[1] + best_right[1]),
        "used_cfg": cfg,
    }