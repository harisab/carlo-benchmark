from __future__ import annotations

from odmr.simulation import generate_random_odmr_trace
from odmr.benchmark_config import BenchmarkConfig, all_correlation_variants
from odmr.algorithms.single_correlation import run_single_correlation
from odmr.algorithms.double_correlation import run_double_correlation


def mean_error(result: dict, truth: dict) -> tuple[float, float, float]:
    e1 = abs(result["f1_hat"] - truth["resonance_value1"])
    e2 = abs(result["f2_hat"] - truth["resonance_value2"])
    return e1, e2, 0.5 * (e1 + e2)


def print_result(label: str, result: dict, truth: dict) -> None:
    e1, e2, em = mean_error(result, truth)

    print(f"{label}")
    print(f"  variant   = {result['benchmark_variant']}")
    print(f"  f1_hat    = {result['f1_hat']:.3f} MHz")
    print(f"  f2_hat    = {result['f2_hat']:.3f} MHz")

    if "gamma_left" in result:
        print(f"  gamma_L   = {result['gamma_left']:.3f}")
        print(f"  gamma_R   = {result['gamma_right']:.3f}")
    else:
        print(f"  gamma     = {result['gamma']:.3f}")

    print(f"  score     = {result['score']:.3f}")
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
        template_height=1.0,
        require_one_peak_per_side=True,
    )

    for cfg in all_correlation_variants(base_cfg):
        result_single = run_single_correlation(x, y_dip, cfg=cfg)
        print_result("SingleCorrelation", result_single, truth)

        result_double = run_double_correlation(x, y_dip, cfg=cfg)
        print_result("DoubleCorrelation", result_double, truth)


if __name__ == "__main__":
    main()