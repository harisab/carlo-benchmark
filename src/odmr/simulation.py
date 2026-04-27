from __future__ import annotations

from typing import Any

import numpy as np

from odmr.project_defaults import SIMULATION_DEFAULTS
from odmr.algorithms.common import lorentzian_peak


def generate_random_odmr_trace(
    num_points: int = int(SIMULATION_DEFAULTS["num_points"]),
    num_tries: int = int(SIMULATION_DEFAULTS["num_tries"]),
    range_start: float = float(SIMULATION_DEFAULTS["range_start"]),
    range_end: float = float(SIMULATION_DEFAULTS["range_end"]),
    center_frequency: float = float(SIMULATION_DEFAULTS["center_frequency"]),
    offset_max: float = float(SIMULATION_DEFAULTS["offset_max"]),
    width_min: int = int(SIMULATION_DEFAULTS["width_min"]),
    width_max: int = int(SIMULATION_DEFAULTS["width_max"]),
    success_probability_at_resonance: float = float(SIMULATION_DEFAULTS["success_probability_at_resonance"]),
    seed: int = int(SIMULATION_DEFAULTS["seed"]),
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if num_points < 3:
        raise ValueError("num_points must be at least 3")
    if num_tries < 1:
        raise ValueError("num_tries must be >= 1")
    if range_end <= range_start:
        raise ValueError("range_end must be greater than range_start")
    if width_min <= 0 or width_max <= 0:
        raise ValueError("width bounds must be positive")
    if width_max < width_min:
        raise ValueError("width_max must be >= width_min")
    if offset_max <= 0:
        raise ValueError("offset_max must be positive")
    if not (0.0 < success_probability_at_resonance < 1.0):
        raise ValueError("success_probability_at_resonance must be between 0 and 1")

    rng = np.random.default_rng(seed)

    x = np.linspace(range_start, range_end, num_points, dtype=float)

    offset = float(rng.uniform(0.0, offset_max))
    gamma = float(rng.integers(width_min, width_max + 1))

    resonance_value1 = float(center_frequency - offset)
    resonance_value2 = float(center_frequency + offset)

    probs = lorentzian_peak(x, resonance_value1, gamma, success_probability_at_resonance)
    probs += lorentzian_peak(x, resonance_value2, gamma, success_probability_at_resonance)
    probs = np.clip(probs, 0.0, 1.0)

    counts = rng.binomial(num_tries, probs)
    y_dip = 1.0 - (counts.astype(float) / float(num_tries))

    truth = {
        "resonance_value1": resonance_value1,
        "resonance_value2": resonance_value2,
        "width": gamma,
        "num_tries": int(num_tries),
        "success_probability_at_resonance": float(success_probability_at_resonance),
        "offset": offset,
        "center_frequency": float(center_frequency),
        "seed": int(seed),
    }

    return x, y_dip, truth