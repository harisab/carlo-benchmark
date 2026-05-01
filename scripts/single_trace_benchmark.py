from __future__ import annotations

import argparse
from typing import Any

from odmr.algorithms.double_correlation import run_double_correlation
from odmr.algorithms.lmfit_double import run_lmfit_double_joint
from odmr.algorithms.lmfit_single_side import run_lmfit_single_side
from odmr.algorithms.single_correlation import run_single_correlation
from odmr.simulation import generate_random_odmr_trace
from odmr.algorithms.paper_ca import run_paper_ca_clean, run_paper_ca_verbatim
from odmr.project_defaults import (
    BENCHMARK_ALGORITHM_NAMES,
    BENCHMARK_CASES,
    BENCHMARK_DEFAULTS,
    SIMULATION_DEFAULTS,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one ODMR benchmark trace.")

    parser.add_argument(
        "--algorithms",
        nargs="*",
        default=None,
        choices=BENCHMARK_ALGORITHM_NAMES,
        help=(
            "Optional algorithm subset. "
            "If omitted, every case in BENCHMARK_CASES is run."
        ),
    )

    parser.add_argument(
        "--template-height-success-prob",
        action=argparse.BooleanOptionalAction,
        default=bool(BENCHMARK_DEFAULTS["use_success_probability_for_template_height"]),
        help=(
            "Use the trace success probability as template height. "
            "Use --no-template-height-success-prob to use BENCHMARK_DEFAULTS['template_height']."
        ),
    )

    parser.add_argument(
        "--require-one-peak-per-side",
        action=argparse.BooleanOptionalAction,
        default=bool(BENCHMARK_DEFAULTS["require_one_peak_per_side"]),
        help=(
            "Constrain algorithms to one left-side and one right-side resonance. "
            "Use --no-require-one-peak-per-side to disable."
        ),
    )

    return parser.parse_args()


def should_run(case: dict[str, Any], selected_algorithms: list[str] | None) -> bool:
    return selected_algorithms is None or case["algorithm"] in selected_algorithms


def settings_for_case(
    case: dict[str, Any],
    *,
    template_height: float,
    require_one_peak_per_side: bool,
) -> dict[str, Any]:
    settings = dict(BENCHMARK_DEFAULTS)
    settings.update(
        {
            "template_height": float(template_height),
            "require_one_peak_per_side": bool(require_one_peak_per_side),
            "normalization_mode": case["normalization_mode"] or "raw",
            "width_mode": case["width_mode"] or "fixed",
            "benchmark_variant": case["variant"],
        }
    )
    return settings


def run_case(case: dict[str, Any], x, y_dip, settings: dict[str, Any]) -> dict:
    algorithm = case["algorithm"]

    if algorithm == "LMFitSinglePerSide":
        return run_lmfit_single_side(x, y_dip, settings=settings)

    if algorithm == "LMFitDoubleJoint":
        return run_lmfit_double_joint(x, y_dip, settings=settings)

    if algorithm == "SingleCorrelation":
        return run_single_correlation(x, y_dip, settings=settings)

    if algorithm == "DoubleCorrelation":
        return run_double_correlation(x, y_dip, settings=settings)
    
    if algorithm == "PaperCA_Verbatim":
        return run_paper_ca_verbatim(x, y_dip, settings=settings)

    if algorithm == "PaperCA_Clean":
        return run_paper_ca_clean(x, y_dip, settings=settings)

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def main() -> None:
    args = parse_args()

    x, y_dip, truth = generate_random_odmr_trace(
        seed=int(SIMULATION_DEFAULTS["seed"])
    )

    if args.template_height_success_prob:
        template_height = float(truth["success_probability_at_resonance"])
    else:
        template_height = float(BENCHMARK_DEFAULTS["template_height"])

    print("Truth")
    print(f"  f1        = {truth['resonance_value1']:.3f} MHz")
    print(f"  f2        = {truth['resonance_value2']:.3f} MHz")
    print(f"  width     = {truth['width']:.3f} MHz")
    print(f"  seed      = {truth['seed']}")
    print()

    print("Benchmark settings")
    print(f"  template_height_success_prob = {args.template_height_success_prob}")
    print(f"  template_height              = {template_height:.6f}")
    print(f"  require_one_peak_per_side    = {args.require_one_peak_per_side}")
    print()

    for case in BENCHMARK_CASES:
        if not should_run(case, args.algorithms):
            continue

        settings = settings_for_case(
            case,
            template_height=template_height,
            require_one_peak_per_side=args.require_one_peak_per_side,
        )
        result = run_case(case, x, y_dip, settings)
        print_result(f"{case['algorithm']} | {case['variant']}", result, truth)


if __name__ == "__main__":
    main()