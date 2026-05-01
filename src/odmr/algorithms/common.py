from __future__ import annotations

import numpy as np

from odmr.project_defaults import BENCHMARK_DEFAULTS


def lorentzian_peak(
    x: np.ndarray,
    center: float,
    gamma: float,
    height: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return height / (1.0 + ((x - center) / gamma) ** 2)


def two_peak_dip(
    x: np.ndarray,
    f1: float,
    f2: float,
    gamma_left: float,
    gamma_right: float,
    height: float,
) -> np.ndarray:
    return 1.0 - (
        lorentzian_peak(x, f1, gamma_left, height)
        + lorentzian_peak(x, f2, gamma_right, height)
    )


def peak_space(y_dip: np.ndarray) -> np.ndarray:
    return 1.0 - np.asarray(y_dip, dtype=float)


def split_left_right_indices(n: int, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    mid = int(n // 2)
    step = max(1, int(step))
    left = np.arange(0, mid, step, dtype=int)
    right = np.arange(mid, n, step, dtype=int)
    return left, right


def _settings(settings: dict | None) -> dict:
    out = dict(BENCHMARK_DEFAULTS)
    if settings is not None:
        out.update(settings)
    return out


def candidate_widths(settings: dict | None) -> np.ndarray:
    cfg = _settings(settings)
    width_mode = str(cfg["width_mode"])

    if width_mode == "fixed":
        return np.array([float(cfg["standard_width"])], dtype=float)

    if width_mode == "scan":
        widths = np.arange(
            float(cfg["min_width"]),
            float(cfg["max_width"]) + 0.5 * float(cfg["width_step"]),
            float(cfg["width_step"]),
            dtype=float,
        )
        if len(widths) == 0:
            return np.array([float(cfg["standard_width"])], dtype=float)
        return widths

    raise ValueError(f"Unsupported width_mode: {width_mode}")


def _safe_l1_normalize(v: np.ndarray) -> np.ndarray:
    denom = float(np.sum(np.abs(v)))
    if denom <= 0.0:
        return v.copy()
    return v / denom


def _safe_l2_normalize(v: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(v))
    if denom <= 0.0:
        return v.copy()
    return v / denom


def process_vector(v: np.ndarray, normalization_mode: str) -> np.ndarray:
    v = np.asarray(v, dtype=float)

    if normalization_mode == "raw":
        return v.copy()

    if normalization_mode == "l1":
        return _safe_l1_normalize(v)

    if normalization_mode == "l2":
        return _safe_l2_normalize(v)

    demeaned = v - float(np.mean(v))

    if normalization_mode == "demean":
        return demeaned

    if normalization_mode == "demean_l1":
        return _safe_l1_normalize(demeaned)

    if normalization_mode == "demean_l2":
        return _safe_l2_normalize(demeaned)

    raise ValueError(f"Unsupported normalization_mode: {normalization_mode}")


def template_score(signal_processed: np.ndarray, template_processed: np.ndarray) -> float:
    return float(np.dot(signal_processed, template_processed))