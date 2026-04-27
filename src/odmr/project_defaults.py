from __future__ import annotations

from dataclasses import dataclass, replace


SIMULATION_DEFAULTS = {
    "num_points": 199,
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

BENCHMARK_DEFAULTS = {
    "min_width": 10.0,
    "max_width": 50.0,
    "width_step": 1.0,
    "standard_width": 20.0,
    "width_mode": "scan",
    "normalization_mode": "raw",
    "center_step_bins": 1,
    "require_one_peak_per_side": True,
}

APP_DEFAULTS = {
    "window_title": "ODMR Correlation GUI",
    "bar_order": "best_first",
    "graph_splitter_sizes": [950, 650],
    "bottom_splitter_sizes": [1600, 400],
}

NORMALIZATION_MODES = (
    "raw",
    "l1",
    "l2",
    "demean",
    "demean_l1",
    "demean_l2",
)

WIDTH_MODES = (
    "scan",
    "fixed",
)

TRUTH_COLOR = "limegreen"


@dataclass(frozen=True)
class BenchmarkConfig:
    min_width: float = float(BENCHMARK_DEFAULTS["min_width"])
    max_width: float = float(BENCHMARK_DEFAULTS["max_width"])
    width_step: float = float(BENCHMARK_DEFAULTS["width_step"])
    standard_width: float = float(BENCHMARK_DEFAULTS["standard_width"])
    width_mode: str = str(BENCHMARK_DEFAULTS["width_mode"])
    template_height: float = float(SIMULATION_DEFAULTS["success_probability_at_resonance"])
    normalization_mode: str = str(BENCHMARK_DEFAULTS["normalization_mode"])
    center_step_bins: int = int(BENCHMARK_DEFAULTS["center_step_bins"])
    require_one_peak_per_side: bool = bool(BENCHMARK_DEFAULTS["require_one_peak_per_side"])


@dataclass(frozen=True)
class AlgorithmSpec:
    key: str
    display_name: str
    color: str
    family: str  # "standalone" or "correlation_variants"
    standalone_variant_name: str | None = None
    default_run: bool = True
    default_show_center: bool = False
    default_show_wave: bool = False


ALGORITHM_SPECS: list[AlgorithmSpec] = [
    AlgorithmSpec(
        key="LMFitSinglePerSide",
        display_name="LMFitSinglePerSide",
        color="magenta",
        family="standalone",
        standalone_variant_name="lmfit_single_side",
    ),
    AlgorithmSpec(
        key="LMFitDoubleJoint",
        display_name="LMFitDoubleJoint",
        color="orange",
        family="standalone",
        standalone_variant_name="lmfit_double_joint",
    ),
    AlgorithmSpec(
        key="SingleCorrelation",
        display_name="SingleCorrelation",
        color="gold",
        family="correlation_variants",
    ),
    AlgorithmSpec(
        key="DoubleCorrelation",
        display_name="DoubleCorrelation",
        color="cyan",
        family="correlation_variants",
    ),
]


def get_algorithm_specs() -> list[AlgorithmSpec]:
    return list(ALGORITHM_SPECS)


def get_algorithm_spec_map() -> dict[str, AlgorithmSpec]:
    return {spec.key: spec for spec in ALGORITHM_SPECS}


def default_template_height(success_probability_at_resonance: float) -> float:
    return float(success_probability_at_resonance)


def variant_label(cfg: BenchmarkConfig) -> str:
    return f"{cfg.normalization_mode}_{cfg.width_mode}"


def all_correlation_variants(base_cfg: BenchmarkConfig) -> list[BenchmarkConfig]:
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


def build_row_specs(base_cfg: BenchmarkConfig) -> list[dict]:
    rows: list[dict] = [{"kind": "truth"}]

    for spec in ALGORITHM_SPECS:
        if spec.family == "standalone":
            rows.append(
                {
                    "kind": "variant",
                    "algorithm": spec.key,
                    "variant": str(spec.standalone_variant_name),
                    "cfg": base_cfg,
                }
            )
        elif spec.family == "correlation_variants":
            for cfg in all_correlation_variants(base_cfg):
                rows.append(
                    {
                        "kind": "variant",
                        "algorithm": spec.key,
                        "variant": variant_label(cfg),
                        "cfg": cfg,
                    }
                )
        else:
            raise ValueError(f"Unsupported family: {spec.family}")

    return rows


def row_key(spec: dict) -> str:
    if spec["kind"] == "truth":
        return "truth"
    return f"variant:{spec['algorithm']}:{spec['variant']}"


def record_key(algorithm: str, variant: str) -> str:
    return f"{algorithm}:{variant}"


def build_jobs_from_rows(row_specs: list[dict], row_run_states: dict[str, bool]) -> list[dict]:
    jobs: list[dict] = []

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


def run_algorithm_job(job: dict, x, y_dip) -> dict:
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