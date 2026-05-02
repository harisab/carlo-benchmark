from __future__ import annotations

import math
from typing import Any

import numpy as np

from odmr.algorithms.common import (
    candidate_widths,
    merged_settings,
    peak_space,
    split_left_right_indices,
    two_peak_dip,
)


def _p_lorentz(
    x: np.ndarray,
    center: float,
    gamma: float,
    pmax: float,
) -> np.ndarray:
    eps = 1e-12
    p = pmax / (1.0 + ((x - center) / gamma) ** 2)
    return np.clip(p, eps, 1.0 - eps)


def _candidate_centers(
    x: np.ndarray,
    *,
    center_step_bins: int,
    require_one_peak_per_side: bool,
) -> tuple[np.ndarray, np.ndarray]:
    step = max(1, int(center_step_bins))

    if require_one_peak_per_side:
        mid = len(x) // 2
        left_centers = x[: mid + 1 : step]
        right_centers = x[mid::step]
    else:
        all_centers = x[::step]
        left_centers = all_centers
        right_centers = all_centers

    if len(left_centers) < 1 or len(right_centers) < 1:
        left_indices, right_indices = split_left_right_indices(len(x), step)
        left_centers = x[left_indices]
        right_centers = x[right_indices]

    return np.asarray(left_centers, dtype=float), np.asarray(right_centers, dtype=float)


def _observed_success_counts(
    y_dip: np.ndarray,
    *,
    n_tries: int,
) -> np.ndarray:
    """
    Our dip data is:

        y_dip = (n_tries - (successes_1 + successes_2)) / n_tries

    Therefore:

        successes_total ~= (1 - y_dip) * n_tries

    The total successes can range from 0 to 2*n_tries because two ODMR
    resonances contribute.
    """
    n = int(n_tries)
    k_obs = np.rint(peak_space(y_dip) * n).astype(int)
    return np.clip(k_obs, 0, 2 * n)


def run_double_mle_exact(
    x: np.ndarray,
    y_dip: np.ndarray,
    *,
    settings: dict[str, Any] | None = None,
) -> dict:
    """
    Exact double-MLE.

    Model per bin:

        K = Binomial(n, p1) + Binomial(n, p2)

    The probability P(K=k) is computed by exact convolution:

        P(K=k) = sum_t P(Bin(n,p1)=t) P(Bin(n,p2)=k-t)

    This is accurate but slower.
    """
    cfg = merged_settings(settings)

    x = np.asarray(x, dtype=float)
    y_dip = np.asarray(y_dip, dtype=float)

    n = int(cfg.get("num_tries", cfg.get("n_tries", 20)))
    if n <= 0:
        raise ValueError("num_tries must be positive for DoubleMLE_Exact.")

    k_obs = _observed_success_counts(y_dip, n_tries=n)

    pmax = float(cfg["template_height"])
    pmax = float(np.clip(pmax, 1e-12, 1.0 - 1e-12))

    widths = candidate_widths(cfg)

    left_centers, right_centers = _candidate_centers(
        x,
        center_step_bins=int(cfg["center_step_bins"]),
        require_one_peak_per_side=bool(cfg["require_one_peak_per_side"]),
    )

    comb = np.array([math.comb(n, t) for t in range(n + 1)], dtype=np.float64)

    idx_by_t: list[np.ndarray] = []
    u_by_t: list[np.ndarray] = []

    for t in range(n + 1):
        u = k_obs - t
        idx = np.where((u >= 0) & (u <= n))[0]
        idx_by_t.append(idx)
        u_by_t.append(u[idx])

    t_column = np.arange(n + 1, dtype=np.float64)[:, None]

    def pmf_matrix_for_center(center: float, gamma: float) -> np.ndarray:
        p = _p_lorentz(x, center, gamma, pmax)
        p2d = p[None, :]
        q2d = (1.0 - p)[None, :]
        return comb[:, None] * (p2d**t_column) * (q2d ** (n - t_column))

    best_log_likelihood = -np.inf
    best_gamma = float(widths[0])
    best_f1 = float(left_centers[0])
    best_f2 = float(right_centers[0])

    for gamma in widths:
        gamma = float(gamma)

        left_pmfs = {
            float(center): pmf_matrix_for_center(float(center), gamma)
            for center in left_centers
        }
        right_pmfs = {
            float(center): pmf_matrix_for_center(float(center), gamma)
            for center in right_centers
        }

        for f1, pmf1 in left_pmfs.items():
            for f2, pmf2 in right_pmfs.items():
                probability = np.zeros(len(x), dtype=np.float64)

                for t in range(n + 1):
                    idx = idx_by_t[t]
                    u = u_by_t[t]

                    if len(idx) > 0:
                        probability[idx] += pmf1[t, idx] * pmf2[u, idx]

                log_likelihood = float(np.sum(np.log(probability + 1e-300)))

                if log_likelihood > best_log_likelihood:
                    best_log_likelihood = log_likelihood
                    best_gamma = gamma
                    best_f1 = float(f1)
                    best_f2 = float(f2)

    f1_hat = float(min(best_f1, best_f2))
    f2_hat = float(max(best_f1, best_f2))

    best_fit = two_peak_dip(
        x,
        f1_hat,
        f2_hat,
        best_gamma,
        best_gamma,
        pmax,
    )

    return {
        "name": "DoubleMLE_Exact",
        "benchmark_variant": str(cfg.get("benchmark_variant", "mle_exact")),
        "f1_hat": f1_hat,
        "f2_hat": f2_hat,
        "gamma": float(best_gamma),
        "score": float(best_log_likelihood),
        "used_settings": cfg,
        "best_fit": np.asarray(best_fit, dtype=float),
    }


def run_double_mle_approx(
    x: np.ndarray,
    y_dip: np.ndarray,
    *,
    settings: dict[str, Any] | None = None,
) -> dict:
    """
    Approximate double-MLE.

    Approximation:

        Binomial(n, p1) + Binomial(n, p2)
        approximately Binomial(2n, p_eff)

    where:

        p_eff = (p1 + p2) / 2

    This preserves the expected value:

        E[K] = n(p1 + p2)
    """
    cfg = merged_settings(settings)

    x = np.asarray(x, dtype=float)
    y_dip = np.asarray(y_dip, dtype=float)

    n = int(cfg.get("num_tries", cfg.get("n_tries", 20)))
    if n <= 0:
        raise ValueError("num_tries must be positive for DoubleMLE_Approx.")

    k_obs = _observed_success_counts(y_dip, n_tries=n)

    pmax = float(cfg["template_height"])
    pmax = float(np.clip(pmax, 1e-12, 1.0 - 1e-12))

    widths = candidate_widths(cfg)

    left_centers, right_centers = _candidate_centers(
        x,
        center_step_bins=int(cfg["center_step_bins"]),
        require_one_peak_per_side=bool(cfg["require_one_peak_per_side"]),
    )

    two_n = 2 * n

    log_comb_2n = np.array(
        [
            math.lgamma(two_n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(two_n - k + 1)
            for k in range(two_n + 1)
        ],
        dtype=np.float64,
    )

    best_log_likelihood = -np.inf
    best_gamma = float(widths[0])
    best_f1 = float(left_centers[0])
    best_f2 = float(right_centers[0])

    for gamma in widths:
        gamma = float(gamma)

        left_p = {
            float(center): _p_lorentz(x, float(center), gamma, pmax)
            for center in left_centers
        }
        right_p = {
            float(center): _p_lorentz(x, float(center), gamma, pmax)
            for center in right_centers
        }

        for f1, p1 in left_p.items():
            for f2, p2 in right_p.items():
                p_eff = np.clip(0.5 * (p1 + p2), 1e-12, 1.0 - 1e-12)

                log_likelihood_per_bin = (
                    log_comb_2n[k_obs]
                    + k_obs * np.log(p_eff)
                    + (two_n - k_obs) * np.log(1.0 - p_eff)
                )

                log_likelihood = float(np.sum(log_likelihood_per_bin))

                if log_likelihood > best_log_likelihood:
                    best_log_likelihood = log_likelihood
                    best_gamma = gamma
                    best_f1 = float(f1)
                    best_f2 = float(f2)

    f1_hat = float(min(best_f1, best_f2))
    f2_hat = float(max(best_f1, best_f2))

    best_fit = two_peak_dip(
        x,
        f1_hat,
        f2_hat,
        best_gamma,
        best_gamma,
        pmax,
    )

    return {
        "name": "DoubleMLE_Approx",
        "benchmark_variant": str(cfg.get("benchmark_variant", "mle_approx")),
        "f1_hat": f1_hat,
        "f2_hat": f2_hat,
        "gamma": float(best_gamma),
        "score": float(best_log_likelihood),
        "used_settings": cfg,
        "best_fit": np.asarray(best_fit, dtype=float),
    }