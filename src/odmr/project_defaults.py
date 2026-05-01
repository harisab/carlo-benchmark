from __future__ import annotations


# =============================================================================
# Simulation defaults
# =============================================================================

SIMULATION_DEFAULTS = {
    # Faithful to the original legacy generator.
    "num_points": 200,
    "num_tries": 20,
    "range_start": 3000.0,
    "range_end": 4000.0,
    "center_frequency": 3500.0,
    "offset_max": 450.0,
    "width_min": 10,
    "width_max": 50,
    "success_probability_at_resonance": 0.15,
    "seed": 123,
}


# =============================================================================
# Benchmark defaults
# =============================================================================

BENCHMARK_DEFAULTS = {
    "min_width": 10.0,
    "max_width": 50.0,
    "width_step": 1.0,
    "standard_width": 30.0,

    # Manual template-height default.
    # GUI/script use success probability by default unless user overrides.
    "template_height": 1.0,
    "use_success_probability_for_template_height": True,

    "center_step_bins": 1,
    "require_one_peak_per_side": True,

    "paper_ca_k_y": 4,
    "paper_ca_max_iter": 300,
    "paper_ca_random_state": 0,
}


# =============================================================================
# App defaults
# =============================================================================

APP_DEFAULTS = {
    "window_title": "ODMR Algorithm Benchmark GUI",
    "bar_order": "table_order",
    "graph_splitter_sizes": [950, 650],
    "bottom_splitter_sizes": [1600, 400],
}

TRUTH_COLOR = "limegreen"


# =============================================================================
# Explicit benchmark cases
# =============================================================================
#
# Each entry is one GUI row and one runnable benchmark case.
# Script default: run every case.
# GUI default: every case is checked under "Run".
#
# LMFit cases do not use normalization_mode / width_mode.
# Correlation cases do.

BENCHMARK_CASES = (
    {
        "algorithm": "LMFitSinglePerSide",
        "variant": "lmfit_single_side",
        "normalization_mode": None,
        "width_mode": None,
    },
    {
        "algorithm": "LMFitDoubleJoint",
        "variant": "lmfit_double_joint",
        "normalization_mode": None,
        "width_mode": None,
    },

    # SingleCorrelation variants
    {
        "algorithm": "SingleCorrelation",
        "variant": "raw_fixed",
        "normalization_mode": "raw",
        "width_mode": "fixed",
    },
    {
        "algorithm": "SingleCorrelation",
        "variant": "raw_scan",
        "normalization_mode": "raw",
        "width_mode": "scan",
    },
    {
        "algorithm": "SingleCorrelation",
        "variant": "l1_fixed",
        "normalization_mode": "l1",
        "width_mode": "fixed",
    },
    {
        "algorithm": "SingleCorrelation",
        "variant": "l1_scan",
        "normalization_mode": "l1",
        "width_mode": "scan",
    },
    {
        "algorithm": "SingleCorrelation",
        "variant": "l2_fixed",
        "normalization_mode": "l2",
        "width_mode": "fixed",
    },
    {
        "algorithm": "SingleCorrelation",
        "variant": "l2_scan",
        "normalization_mode": "l2",
        "width_mode": "scan",
    },
    {
        "algorithm": "SingleCorrelation",
        "variant": "demean_fixed",
        "normalization_mode": "demean",
        "width_mode": "fixed",
    },
    {
        "algorithm": "SingleCorrelation",
        "variant": "demean_scan",
        "normalization_mode": "demean",
        "width_mode": "scan",
    },
    {
        "algorithm": "SingleCorrelation",
        "variant": "demean_l1_fixed",
        "normalization_mode": "demean_l1",
        "width_mode": "fixed",
    },
    {
        "algorithm": "SingleCorrelation",
        "variant": "demean_l1_scan",
        "normalization_mode": "demean_l1",
        "width_mode": "scan",
    },
    {
        "algorithm": "SingleCorrelation",
        "variant": "demean_l2_fixed",
        "normalization_mode": "demean_l2",
        "width_mode": "fixed",
    },
    {
        "algorithm": "SingleCorrelation",
        "variant": "demean_l2_scan",
        "normalization_mode": "demean_l2",
        "width_mode": "scan",
    },

    # DoubleCorrelation variants
    {
        "algorithm": "DoubleCorrelation",
        "variant": "raw_fixed",
        "normalization_mode": "raw",
        "width_mode": "fixed",
    },
    {
        "algorithm": "DoubleCorrelation",
        "variant": "raw_scan",
        "normalization_mode": "raw",
        "width_mode": "scan",
    },
    {
        "algorithm": "DoubleCorrelation",
        "variant": "l1_fixed",
        "normalization_mode": "l1",
        "width_mode": "fixed",
    },
    {
        "algorithm": "DoubleCorrelation",
        "variant": "l1_scan",
        "normalization_mode": "l1",
        "width_mode": "scan",
    },
    {
        "algorithm": "DoubleCorrelation",
        "variant": "l2_fixed",
        "normalization_mode": "l2",
        "width_mode": "fixed",
    },
    {
        "algorithm": "DoubleCorrelation",
        "variant": "l2_scan",
        "normalization_mode": "l2",
        "width_mode": "scan",
    },
    {
        "algorithm": "DoubleCorrelation",
        "variant": "demean_fixed",
        "normalization_mode": "demean",
        "width_mode": "fixed",
    },
    {
        "algorithm": "DoubleCorrelation",
        "variant": "demean_scan",
        "normalization_mode": "demean",
        "width_mode": "scan",
    },
    {
        "algorithm": "DoubleCorrelation",
        "variant": "demean_l1_fixed",
        "normalization_mode": "demean_l1",
        "width_mode": "fixed",
    },
    {
        "algorithm": "DoubleCorrelation",
        "variant": "demean_l1_scan",
        "normalization_mode": "demean_l1",
        "width_mode": "scan",
    },
    {
        "algorithm": "DoubleCorrelation",
        "variant": "demean_l2_fixed",
        "normalization_mode": "demean_l2",
        "width_mode": "fixed",
    },
    {
        "algorithm": "DoubleCorrelation",
        "variant": "demean_l2_scan",
        "normalization_mode": "demean_l2",
        "width_mode": "scan",
    },
    {
        "algorithm": "PaperCA_Verbatim",
        "variant": "paper_ca_verbatim",
        "normalization_mode": None,
        "width_mode": None,
    },
    {
        "algorithm": "PaperCA_Clean",
        "variant": "paper_ca_clean",
        "normalization_mode": None,
        "width_mode": None,
    },
)


BENCHMARK_ALGORITHM_NAMES = tuple(
    dict.fromkeys(case["algorithm"] for case in BENCHMARK_CASES)
)