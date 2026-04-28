from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable


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

NORMALIZATION_MODES = (
    "raw",
    "l1",
    "l2",
    "demean",
    "demean_l1",
    "demean_l2",
)

WIDTH_MODES = (
    "fixed",
    "scan",
)

BENCHMARK_DEFAULTS = {
    "min_width": 10.0,
    "max_width": 50.0,
    "width_step": 1.0,
    "standard_width": 30.0,
    "template_height": 1.0,
    "use_success_probability_for_template_height": True,
    "center_step_bins": 1,
    "require_one_peak_per_side": True,
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

# Colors are assigned by row/order, not permanently married to algorithms.
PLOT_COLORS = (
    "magenta",
    "orange",
    "gold",
    "cyan",
    "tab:blue",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
)


# =============================================================================
# Concrete config passed to algorithms
# =============================================================================

@dataclass(frozen=True)
class BenchmarkConfig:
    """
    Concrete benchmark configuration passed to algorithms.

    width_mode is always concrete:
        "fixed" or "scan"

    normalization_mode is always concrete:
        raw / l1 / l2 / demean / demean_l1 / demean_l2

    All variant expansion happens in build_row_specs().
    """

    min_width: float = float(BENCHMARK_DEFAULTS["min_width"])
    max_width: float = float(BENCHMARK_DEFAULTS["max_width"])
    width_step: float = float(BENCHMARK_DEFAULTS["width_step"])
    standard_width: float = float(BENCHMARK_DEFAULTS["standard_width"])
    template_height: float = float(BENCHMARK_DEFAULTS["template_height"])

    width_mode: str = WIDTH_MODES[0]
    normalization_mode: str = NORMALIZATION_MODES[0]

    center_step_bins: int = int(BENCHMARK_DEFAULTS["center_step_bins"])
    require_one_peak_per_side: bool = bool(BENCHMARK_DEFAULTS["require_one_peak_per_side"])


# =============================================================================
# Explicit benchmark design
# =============================================================================

BENCHMARK_ALGORITHMS = (
    "LMFitSinglePerSide",
    "LMFitDoubleJoint",
    "SingleCorrelation",
    "DoubleCorrelation",
)


def variant_label(cfg: BenchmarkConfig) -> str:
    return f"{cfg.normalization_mode}_{cfg.width_mode}"


def all_correlation_variants(base_cfg: BenchmarkConfig) -> list[BenchmarkConfig]:
    """
    Build all concrete correlation variants.

    This always expands across:
    - NORMALIZATION_MODES
    - WIDTH_MODES

    No algorithm receives "all".
    """
    variants: list[BenchmarkConfig] = []

    for normalization_mode in NORMALIZATION_MODES:
        for width_mode in WIDTH_MODES:
            variants.append(
                replace(
                    base_cfg,
                    normalization_mode=normalization_mode,
                    width_mode=width_mode,
                )
            )

    return variants


def build_row_specs(
    base_cfg: BenchmarkConfig,
    *,
    algorithm_keys: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Build table rows / benchmark rows.

    By default:
    - truth row
    - LMFitSinglePerSide
    - LMFitDoubleJoint
    - all SingleCorrelation variants
    - all DoubleCorrelation variants

    algorithm_keys can be used by scripts to run only selected algorithms.
    """
    selected = set(algorithm_keys) if algorithm_keys is not None else None

    rows: list[dict[str, Any]] = [{"kind": "truth"}]

    if selected is None or "LMFitSinglePerSide" in selected:
        rows.append(
            {
                "kind": "variant",
                "algorithm": "LMFitSinglePerSide",
                "variant": "lmfit_single_side",
                "cfg": base_cfg,
            }
        )

    if selected is None or "LMFitDoubleJoint" in selected:
        rows.append(
            {
                "kind": "variant",
                "algorithm": "LMFitDoubleJoint",
                "variant": "lmfit_double_joint",
                "cfg": base_cfg,
            }
        )

    if selected is None or "SingleCorrelation" in selected:
        for cfg in all_correlation_variants(base_cfg):
            rows.append(
                {
                    "kind": "variant",
                    "algorithm": "SingleCorrelation",
                    "variant": variant_label(cfg),
                    "cfg": cfg,
                }
            )

    if selected is None or "DoubleCorrelation" in selected:
        for cfg in all_correlation_variants(base_cfg):
            rows.append(
                {
                    "kind": "variant",
                    "algorithm": "DoubleCorrelation",
                    "variant": variant_label(cfg),
                    "cfg": cfg,
                }
            )

    return rows


# =============================================================================
# Row/job helpers
# =============================================================================

def row_key(spec: dict[str, Any]) -> str:
    if spec["kind"] == "truth":
        return "truth"

    return f"variant:{spec['algorithm']}:{spec['variant']}"


def record_key(algorithm: str, variant: str) -> str:
    return f"{algorithm}:{variant}"


def default_row_run_states(row_specs: list[dict[str, Any]]) -> dict[str, bool]:
    """
    Default script behavior:
    - do not run truth row
    - run every algorithm/variant row
    """
    return {
        row_key(spec): spec["kind"] == "variant"
        for spec in row_specs
    }


def build_jobs_from_rows(
    row_specs: list[dict[str, Any]],
    row_run_states: dict[str, bool],
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []

    for spec in row_specs:
        if spec["kind"] != "variant":
            continue

        if row_run_states.get(row_key(spec), False):
            jobs.append(
                {
                    "algorithm": spec["algorithm"],
                    "cfg": spec["cfg"],
                }
            )

    return jobs


def plot_color_for_index(index: int) -> str:
    return PLOT_COLORS[index % len(PLOT_COLORS)]


# =============================================================================
# Algorithm dispatch
# =============================================================================

def run_algorithm_job(job: dict[str, Any], x, y_dip) -> dict:
    """
    Run one benchmark job.

    Imports are local to avoid circular imports because algorithm modules import
    BenchmarkConfig from this file.
    """
    algorithm = job["algorithm"]
    cfg = job["cfg"]

    if algorithm == "LMFitSinglePerSide":
        from odmr.algorithms.lmfit_single_side import run_lmfit_single_side

        return run_lmfit_single_side(x, y_dip, cfg=cfg)

    if algorithm == "LMFitDoubleJoint":
        from odmr.algorithms.lmfit_double import run_lmfit_double_joint

        return run_lmfit_double_joint(x, y_dip, cfg=cfg)

    if algorithm == "SingleCorrelation":
        from odmr.algorithms.single_correlation import run_single_correlation

        return run_single_correlation(x, y_dip, cfg=cfg)

    if algorithm == "DoubleCorrelation":
        from odmr.algorithms.double_correlation import run_double_correlation

        return run_double_correlation(x, y_dip, cfg=cfg)

    raise ValueError(f"Unsupported algorithm: {algorithm}")