from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from odmr.benchmark_config import BenchmarkConfig, all_correlation_variants, variant_label
from odmr.algorithms.lmfit_single_side import run_lmfit_single_side
from odmr.algorithms.lmfit_double import run_lmfit_double_joint
from odmr.algorithms.single_correlation import run_single_correlation
from odmr.algorithms.double_correlation import run_double_correlation


@dataclass(frozen=True)
class AlgorithmSpec:
    key: str
    display_name: str
    color: str
    family: str  # "standalone" or "correlation_variants"
    default_run: bool = True
    default_show_center: bool = False
    default_show_wave: bool = False


ALGORITHM_SPECS: list[AlgorithmSpec] = [
    AlgorithmSpec(
        key="LMFitSinglePerSide",
        display_name="LMFitSinglePerSide",
        color="magenta",
        family="standalone",
    ),
    AlgorithmSpec(
        key="LMFitDoubleJoint",
        display_name="LMFitDoubleJoint",
        color="orange",
        family="standalone",
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


def row_key(spec: dict[str, Any]) -> str:
    if spec["kind"] == "truth":
        return "truth"
    return f"variant:{spec['algorithm']}:{spec['variant']}"


def record_key(algorithm: str, variant: str) -> str:
    return f"{algorithm}:{variant}"


def build_variant_rows(base_cfg: BenchmarkConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [{"kind": "truth"}]

    for spec in ALGORITHM_SPECS:
        if spec.family == "standalone":
            variant = "lmfit_single_side" if spec.key == "LMFitSinglePerSide" else "lmfit_double_joint"
            rows.append(
                {
                    "kind": "variant",
                    "algorithm": spec.key,
                    "variant": variant,
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
            raise ValueError(f"Unsupported algorithm family: {spec.family}")

    return rows


def build_jobs_from_rows(
    row_specs: list[dict[str, Any]],
    row_run_states: dict[str, bool],
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for spec in row_specs:
        if spec["kind"] != "variant":
            continue
        key = row_key(spec)
        if row_run_states.get(key, False):
            jobs.append(
                {
                    "algorithm": spec["algorithm"],
                    "cfg": spec["cfg"],
                }
            )
    return jobs


def run_algorithm_job(job: dict[str, Any], x, y_dip) -> dict:
    algorithm = job["algorithm"]
    cfg = job["cfg"]

    if algorithm == "LMFitSinglePerSide":
        return run_lmfit_single_side(x, y_dip, cfg=cfg)
    if algorithm == "LMFitDoubleJoint":
        return run_lmfit_double_joint(x, y_dip, cfg=cfg)
    if algorithm == "SingleCorrelation":
        return run_single_correlation(x, y_dip, cfg=cfg)
    if algorithm == "DoubleCorrelation":
        return run_double_correlation(x, y_dip, cfg=cfg)

    raise ValueError(f"Unsupported algorithm: {algorithm}")