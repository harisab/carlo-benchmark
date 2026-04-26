from __future__ import annotations

from odmr.simulation import generate_random_odmr_trace
from odmr.benchmark_config import BenchmarkConfig
from odmr.benchmark_registry import (
    build_jobs_from_rows,
    build_variant_rows,
    default_template_height,
    run_algorithm_job,
)


def mean_error(result: dict, truth: dict) -> tuple[float, float, float]:
    e1 = abs(result["f1_hat"] - truth["resonance_value1"])
    e2 = abs(result["f2_hat"] - truth["resonance_value2"])
    return e1, e2, 0.5 * (e1 + e2)


def print_result(label: str, result: dict, truth: dict) -> None:
    e1, e2, em = mean_error(result, truth)

    print(label)
    print(f"  variant   = {result['benchmark_variant']}")
    print(f"  f1_hat    = {result['f1_hat']:.3f} MHz")
    print(f"  f2_hat    = {result['f2_hat']:.3f} MHz")

    if "gamma_left" in result:
        print(f"  gamma_L   = {result['gamma_left']:.3f}")
        print(f"  gamma_R   = {result['gamma_right']:.3f}")
    else:
        print(f"  gamma     = {result['gamma']:.3f}")

    print(f"  score     = {result['score']:.6f}")
    print(f"  err_f1    = {e1:.3f} MHz")
    print(f"  err_f2    = {e2:.3f} MHz")
    print(f"  mean_err  = {em:.3f} MHz")
    print()


def main() -> None:
    x, y_dip, truth = generate_random_odmr_trace(seed=123)

    print("Truth")
    print(f"  f1        = {truth['resonance_value1']:.3f} MHz")
    print(f"  f2        = {truth['resonance_value2']:.3f} MHz")
    print(f"  width     = {truth['width']:.3f} MHz")
    print()

    base_cfg = BenchmarkConfig(
        standard_width=20.0,
        template_height=default_template_height(truth["success_probability_at_resonance"]),
        require_one_peak_per_side=True,
    )

    row_specs = build_variant_rows(base_cfg)
    row_run_states = {
        ("truth" if spec["kind"] == "truth" else f"variant:{spec['algorithm']}:{spec['variant']}"): spec["kind"] == "variant"
        for spec in row_specs
    }
    jobs = build_jobs_from_rows(row_specs, row_run_states)

    for job in jobs:
        result = run_algorithm_job(job, x, y_dip)
        print_result(job["algorithm"], result, truth)


if __name__ == "__main__":
    main()