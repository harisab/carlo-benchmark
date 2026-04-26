from __future__ import annotations

import numpy as np
from lmfit import Model

from odmr.benchmark_config import BenchmarkConfig


def single_lorentzian_dip(
    x: np.ndarray,
    baseline: float,
    amplitude: float,
    f0: float,
    gamma: float,
) -> np.ndarray:
    return baseline - amplitude / (1.0 + ((x - f0) / gamma) ** 2)


def _fit_one_side(
    x: np.ndarray,
    y_dip: np.ndarray,
    *,
    gamma_init: float,
    gamma_min: float,
    gamma_max: float,
) -> tuple[object, dict[str, float]]:
    x = np.asarray(x, dtype=float)
    y_dip = np.asarray(y_dip, dtype=float)

    i0 = int(np.argmin(y_dip))
    baseline0 = float(np.median(y_dip))
    amp0 = float(max(1e-4, baseline0 - y_dip[i0]))
    f0_0 = float(x[i0])

    model = Model(single_lorentzian_dip)
    params = model.make_params(
        baseline=baseline0,
        amplitude=amp0,
        f0=f0_0,
        gamma=gamma_init,
    )

    params["baseline"].set(min=0.0, max=2.0)
    params["amplitude"].set(min=0.0, max=2.0)
    params["f0"].set(min=float(np.min(x)), max=float(np.max(x)))
    params["gamma"].set(min=float(gamma_min), max=float(gamma_max))

    result = model.fit(y_dip, params, x=x)

    best = result.best_values
    return result, {
        "baseline": float(best["baseline"]),
        "amplitude": float(best["amplitude"]),
        "f0": float(best["f0"]),
        "gamma": float(best["gamma"]),
    }


def run_lmfit_single_side(
    x: np.ndarray,
    y_dip: np.ndarray,
    *,
    cfg: BenchmarkConfig | None = None,
) -> dict:
    """
    Independent left/right single-Lorentzian fits.
    This is the lmfit analogue of SingleCorrelation.
    """
    if cfg is None:
        cfg = BenchmarkConfig()

    x = np.asarray(x, dtype=float)
    y_dip = np.asarray(y_dip, dtype=float)

    mid = len(x) // 2
    x_left = x[:mid]
    y_left = y_dip[:mid]
    x_right = x[mid:]
    y_right = y_dip[mid:]

    gamma0 = float(cfg.standard_width)
    gamma0 = max(float(cfg.min_width), min(float(cfg.max_width), gamma0))

    left_fit, left_best = _fit_one_side(
        x_left,
        y_left,
        gamma_init=gamma0,
        gamma_min=float(cfg.min_width),
        gamma_max=float(cfg.max_width),
    )
    right_fit, right_best = _fit_one_side(
        x_right,
        y_right,
        gamma_init=gamma0,
        gamma_min=float(cfg.min_width),
        gamma_max=float(cfg.max_width),
    )

    f1_hat = float(left_best["f0"])
    f2_hat = float(right_best["f0"])
    if f1_hat > f2_hat:
        f1_hat, f2_hat = f2_hat, f1_hat

    best_fit = np.empty_like(y_dip, dtype=float)
    best_fit[:mid] = np.asarray(left_fit.best_fit, dtype=float)
    best_fit[mid:] = np.asarray(right_fit.best_fit, dtype=float)

    chisqr_total = float(left_fit.chisqr + right_fit.chisqr)

    return {
        "name": "LMFitSinglePerSide",
        "benchmark_variant": "lmfit_single_side",
        "f1_hat": f1_hat,
        "f2_hat": f2_hat,
        "gamma_left": float(left_best["gamma"]),
        "gamma_right": float(right_best["gamma"]),
        "score": -chisqr_total,
        "used_cfg": cfg,
        "fit_success": bool(left_fit.success and right_fit.success),
        "fit_message": f"L:{left_fit.message} | R:{right_fit.message}",
        "chisqr": chisqr_total,
        "best_fit": best_fit,
    }