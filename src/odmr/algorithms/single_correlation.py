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


def correlate_same(y: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Simple same-length correlation using NumPy only."""
    full = np.correlate(y, template, mode="full")
    start = (len(full) - len(y)) // 2
    return full[start:start + len(y)]


def run_single_correlation(
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
) -> dict:
    """
    Single-correlation ODMR estimator.

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
    )

    x = np.asarray(x, dtype=float)
    y = peak_from_dip(y_dip)

    if cfg.demean:
        y = y - np.mean(y)

    regions = get_search_regions(x, y, cfg)
    left_slice = regions["left_slice"]
    right_slice = regions["right_slice"]

    xmid = float(np.mean(x))
    width_candidates = get_width_candidates(cfg)

    left_candidates = []
    right_candidates = []

    for gamma in width_candidates:
        gamma = float(gamma)

        template = lorentzian_peak(x, xmid, gamma, cfg.template_height)

        if cfg.demean:
            template = template - np.mean(template)

        if cfg.normalize_template:
            nrm = np.linalg.norm(template)
            if nrm > 0:
                template = template / nrm

        corr = correlate_same(y, template)

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
        "benchmark_variant": f"{'norm' if cfg.normalize_template else 'raw'}_{cfg.width_mode}",
        "f1_hat": f1_hat,
        "f2_hat": f2_hat,
        "gamma_left": float(best_left[0]),
        "gamma_right": float(best_right[0]),
        "score": float(best_left[1] + best_right[1]),
        "used_cfg": cfg,
    }