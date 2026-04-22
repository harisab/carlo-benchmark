from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np


@dataclass(frozen=True)
class BenchmarkConfig:
    """
    Shared benchmark configuration used by multiple algorithms.

    Key benchmark assumptions:
    - require_one_peak_per_side=True means left/right halves are searched separately
    - width_mode='scan' means search across a width grid
    - width_mode='fixed' means use one standard width only
    - normalize_template distinguishes normalized vs raw correlation benchmarks
    """
    min_width: float = 10.0
    max_width: float = 50.0
    width_step: float = 1.0
    standard_width: float = 20.0
    width_mode: str = "scan"  # "scan" or "fixed"

    template_height: float = 0.15
    normalize_template: bool = False
    demean: bool = True

    center_step_bins: int = 1
    restrict_window_mhz: float | None = None

    require_one_peak_per_side: bool = True


def with_overrides(cfg: BenchmarkConfig, **kwargs) -> BenchmarkConfig:
    """
    Return a copy of cfg with only non-None overrides applied.
    """
    clean = {k: v for k, v in kwargs.items() if v is not None}
    if not clean:
        return cfg
    return replace(cfg, **clean)


def get_width_candidates(cfg: BenchmarkConfig) -> np.ndarray:
    """
    Return the width candidates for the chosen benchmark mode.
    """
    if cfg.width_mode == "fixed":
        return np.asarray([float(cfg.standard_width)], dtype=float)

    if cfg.width_mode != "scan":
        raise ValueError(f"Unsupported width_mode: {cfg.width_mode}")

    return np.arange(cfg.min_width, cfg.max_width + 1e-12, cfg.width_step, dtype=float)


def get_search_regions(
    x: np.ndarray,
    y_peak: np.ndarray,
    cfg: BenchmarkConfig,
) -> dict:
    """
    Shared left/right search-region logic for benchmark algorithms.
    """
    x = np.asarray(x, dtype=float)
    y_peak = np.asarray(y_peak, dtype=float)

    mid_idx = len(x) // 2
    step = max(1, int(cfg.center_step_bins))

    if cfg.require_one_peak_per_side:
        left_slice = slice(0, mid_idx)
        right_slice = slice(mid_idx, len(x))

        left_centers_all = x[: mid_idx + 1 : step]
        right_centers_all = x[mid_idx::step]
    else:
        left_slice = slice(0, len(x))
        right_slice = slice(0, len(x))

        left_centers_all = x[::step]
        right_centers_all = x[::step]

    if cfg.restrict_window_mhz is not None and cfg.require_one_peak_per_side:
        iL = int(np.argmax(y_peak[left_slice]))
        iR_local = int(np.argmax(y_peak[right_slice]))
        iR = mid_idx + iR_local

        guess_L = float(x[iL])
        guess_R = float(x[iR])

        left_centers = left_centers_all[np.abs(left_centers_all - guess_L) <= cfg.restrict_window_mhz]
        right_centers = right_centers_all[np.abs(right_centers_all - guess_R) <= cfg.restrict_window_mhz]

        if len(left_centers) < 1:
            left_centers = left_centers_all
        if len(right_centers) < 1:
            right_centers = right_centers_all
    else:
        left_centers = left_centers_all
        right_centers = right_centers_all

    return {
        "mid_idx": mid_idx,
        "left_slice": left_slice,
        "right_slice": right_slice,
        "left_centers": left_centers,
        "right_centers": right_centers,
    }