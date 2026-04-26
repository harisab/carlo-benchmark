from __future__ import annotations

import numpy as np
from lmfit import Model

from odmr.benchmark_config import BenchmarkConfig


def double_lorentzian_dip(
    x: np.ndarray,
    baseline: float,
    amplitude1: float,
    amplitude2: float,
    f1: float,
    f2: float,
    gamma: float,
) -> np.ndarray:
    return (
        baseline
        - amplitude1 / (1.0 + ((x - f1) / gamma) ** 2)
        - amplitude2 / (1.0 + ((x - f2) / gamma) ** 2)
    )


def _initial_guesses(
    x: np.ndarray,
    y_dip: np.ndarray,
    cfg: BenchmarkConfig,
) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y_dip = np.asarray(y_dip, dtype=float)

    y_peak = 1.0 - y_dip
    mid = len(x) // 2

    left_idx = int(np.argmax(y_peak[:mid]))
    right_idx = mid + int(np.argmax(y_peak[mid:]))

    baseline0 = float(np.median(y_dip))
    amp1_0 = float(max(1e-4, baseline0 - y_dip[left_idx]))
    amp2_0 = float(max(1e-4, baseline0 - y_dip[right_idx]))

    gamma0 = float(cfg.standard_width)
    gamma0 = max(float(cfg.min_width), min(float(cfg.max_width), gamma0))

    return {
        "baseline": baseline0,
        "amplitude1": amp1_0,
        "amplitude2": amp2_0,
        "f1": float(x[left_idx]),
        "f2": float(x[right_idx]),
        "gamma": gamma0,
    }


def run_lmfit_double_joint(
    x: np.ndarray,
    y_dip: np.ndarray,
    *,
    cfg: BenchmarkConfig | None = None,
) -> dict:
    """
    Joint two-peak lmfit model over the full trace.
    This is the lmfit analogue of DoubleCorrelation.
    """
    if cfg is None:
        cfg = BenchmarkConfig()

    x = np.asarray(x, dtype=float)
    y_dip = np.asarray(y_dip, dtype=float)

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    mid_x = float(x[len(x) // 2])

    init = _initial_guesses(x, y_dip, cfg)

    model = Model(double_lorentzian_dip)
    params = model.make_params(
        baseline=init["baseline"],
        amplitude1=init["amplitude1"],
        amplitude2=init["amplitude2"],
        f1=init["f1"],
        f2=init["f2"],
        gamma=init["gamma"],
    )

    params["baseline"].set(min=0.0, max=2.0)
    params["amplitude1"].set(min=0.0, max=2.0)
    params["amplitude2"].set(min=0.0, max=2.0)

    if cfg.require_one_peak_per_side:
        params["f1"].set(min=x_min, max=mid_x)
        params["f2"].set(min=mid_x, max=x_max)
    else:
        params["f1"].set(min=x_min, max=x_max)
        params["f2"].set(min=x_min, max=x_max)

    params["gamma"].set(min=float(cfg.min_width), max=float(cfg.max_width))

    result = model.fit(y_dip, params, x=x)
    best = result.best_values

    f1_hat = float(best["f1"])
    f2_hat = float(best["f2"])
    if f1_hat > f2_hat:
        f1_hat, f2_hat = f2_hat, f1_hat

    return {
        "name": "LMFitDoubleJoint",
        "benchmark_variant": "lmfit_double_joint",
        "f1_hat": f1_hat,
        "f2_hat": f2_hat,
        "gamma": float(best["gamma"]),
        "score": -float(result.chisqr),
        "used_cfg": cfg,
        "fit_success": bool(result.success),
        "fit_message": str(result.message),
        "chisqr": float(result.chisqr),
        "best_fit": np.asarray(result.best_fit, dtype=float),
    }