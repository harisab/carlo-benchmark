from __future__ import annotations

import numpy as np


def simulate_single_resonance_counts(
    resonance_value: float,
    range_start: float,
    range_end: float,
    num_points: int,
    num_tries: int,
    success_probability_at_resonance: float,
    width: float,
    rng: np.random.Generator,
) -> np.ndarray:
    freq_axis = np.linspace(range_start, range_end, num_points)
    successful_events_per_bin = np.zeros(num_points, dtype=float)

    for i, value in enumerate(freq_axis):
        success_prob = success_probability_at_resonance / (
            1 + ((value - resonance_value) / width) ** 2
        )
        successes = rng.binomial(num_tries, success_prob)
        successful_events_per_bin[i] = successes

    return successful_events_per_bin


def generate_odmr_trace(
    *,
    num_points: int = 199,
    num_tries: int = 20,
    range_start: float = 3000.0,
    range_end: float = 4000.0,
    center_frequency: float = 3500.0,
    success_probability_at_resonance: float = 0.15,
    width: float = 20.0,
    resonance_value1: float = 3300.0,
    resonance_value2: float = 3700.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    freq_axis = np.linspace(range_start, range_end, num_points)

    counts1 = simulate_single_resonance_counts(
        resonance_value=resonance_value1,
        range_start=range_start,
        range_end=range_end,
        num_points=num_points,
        num_tries=num_tries,
        success_probability_at_resonance=success_probability_at_resonance,
        width=width,
        rng=rng,
    )

    counts2 = simulate_single_resonance_counts(
        resonance_value=resonance_value2,
        range_start=range_start,
        range_end=range_end,
        num_points=num_points,
        num_tries=num_tries,
        success_probability_at_resonance=success_probability_at_resonance,
        width=width,
        rng=rng,
    )

    odmr_data = (num_tries - (counts1 + counts2)) / num_tries

    metadata = {
        "resonance_value1": float(resonance_value1),
        "resonance_value2": float(resonance_value2),
        "width": float(width),
        "num_tries": int(num_tries),
        "num_points": int(num_points),
        "range_start": float(range_start),
        "range_end": float(range_end),
        "center_frequency": float(center_frequency),
        "success_probability_at_resonance": float(success_probability_at_resonance),
    }

    return freq_axis, odmr_data.astype(float), metadata


def generate_random_odmr_trace(
    *,
    num_points: int = 199,
    num_tries: int = 20,
    range_start: float = 3000.0,
    range_end: float = 4000.0,
    center_frequency: float = 3500.0,
    offset_max: float = 450.0,
    width_min: int = 10,
    width_max: int = 50,
    success_probability_at_resonance: float = 0.15,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)

    width = int(rng.integers(width_min, width_max + 1))
    resonance_value1 = float(center_frequency - rng.integers(0, int(offset_max) + 1))
    resonance_value2 = float(center_frequency + rng.integers(0, int(offset_max) + 1))

    return generate_odmr_trace(
        num_points=num_points,
        num_tries=num_tries,
        range_start=range_start,
        range_end=range_end,
        center_frequency=center_frequency,
        success_probability_at_resonance=success_probability_at_resonance,
        width=width,
        resonance_value1=resonance_value1,
        resonance_value2=resonance_value2,
        seed=seed,
    )