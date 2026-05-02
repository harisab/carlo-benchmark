from __future__ import annotations

import argparse
import csv
import datetime as _dt
import time
from pathlib import Path
from typing import Any

import numpy as np

from odmr.algorithms.double_correlation import run_double_correlation
from odmr.algorithms.lmfit_double import run_lmfit_double_joint
from odmr.algorithms.lmfit_single_side import run_lmfit_single_side
from odmr.algorithms.paper_ca import run_paper_ca_clean, run_paper_ca_verbatim
from odmr.algorithms.single_correlation import run_single_correlation
from odmr.algorithms.double_mle import run_double_mle_approx, run_double_mle_exact
from odmr.project_defaults import (
    BENCHMARK_ALGORITHM_NAMES,
    BENCHMARK_CASES,
    BENCHMARK_DEFAULTS,
    SIMULATION_DEFAULTS,
)
from odmr.simulation import generate_random_odmr_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run complete multi-trace ODMR benchmark and export summary CSV."
    )

    # Run controls
    parser.add_argument(
        "--num-traces",
        type=int,
        default=100,
        help="Number of simulated traces to benchmark.",
    )

    parser.add_argument(
        "--start-seed",
        type=int,
        default=int(SIMULATION_DEFAULTS["seed"]),
        help="First random seed. Trace i uses start_seed + i.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output summary CSV path. If omitted, a timestamped file is created.",
    )

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
            "Use trace success probability as template height. "
            "Use --no-template-height-success-prob to use BENCHMARK_DEFAULTS['template_height']."
        ),
    )

    parser.add_argument(
        "--require-one-peak-per-side",
        action=argparse.BooleanOptionalAction,
        default=bool(BENCHMARK_DEFAULTS["require_one_peak_per_side"]),
        help=(
            "Constrain applicable algorithms to one left-side and one right-side resonance. "
            "Use --no-require-one-peak-per-side to disable."
        ),
    )

    # Simulation controls
    parser.add_argument(
        "--num-points",
        type=int,
        default=int(SIMULATION_DEFAULTS["num_points"]),
        help="Number of frequency bins.",
    )

    parser.add_argument(
        "--num-tries",
        type=int,
        default=int(SIMULATION_DEFAULTS["num_tries"]),
        help="Number of binomial trials per frequency bin.",
    )

    parser.add_argument(
        "--range-start",
        type=float,
        default=float(SIMULATION_DEFAULTS["range_start"]),
        help="Microwave frequency range start.",
    )

    parser.add_argument(
        "--range-end",
        type=float,
        default=float(SIMULATION_DEFAULTS["range_end"]),
        help="Microwave frequency range end.",
    )

    parser.add_argument(
        "--center-frequency",
        type=float,
        default=float(SIMULATION_DEFAULTS["center_frequency"]),
        help="Center frequency used for random resonance generation.",
    )

    parser.add_argument(
        "--offset-max",
        type=float,
        default=float(SIMULATION_DEFAULTS["offset_max"]),
        help="Maximum random resonance offset from center frequency.",
    )

    parser.add_argument(
        "--width-min",
        type=int,
        default=int(SIMULATION_DEFAULTS["width_min"]),
        help="Minimum simulated Lorentzian width.",
    )

    parser.add_argument(
        "--width-max",
        type=int,
        default=int(SIMULATION_DEFAULTS["width_max"]),
        help="Maximum simulated Lorentzian width.",
    )

    parser.add_argument(
        "--success-probability",
        type=float,
        default=float(SIMULATION_DEFAULTS["success_probability_at_resonance"]),
        help="Success probability / contrast at resonance.",
    )

    return parser.parse_args()


def selected_cases(selected_algorithms: list[str] | None) -> list[dict[str, Any]]:
    if selected_algorithms is None:
        return list(BENCHMARK_CASES)

    return [
        case
        for case in BENCHMARK_CASES
        if case["algorithm"] in selected_algorithms
    ]


def simulation_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "num_points": int(args.num_points),
        "num_tries": int(args.num_tries),
        "range_start": float(args.range_start),
        "range_end": float(args.range_end),
        "center_frequency": float(args.center_frequency),
        "offset_max": float(args.offset_max),
        "width_min": int(args.width_min),
        "width_max": int(args.width_max),
        "success_probability_at_resonance": float(args.success_probability),
    }


def settings_for_case(
    case: dict[str, Any],
    *,
    truth: dict[str, Any],
    template_height_success_prob: bool,
    require_one_peak_per_side: bool,
) -> dict[str, Any]:
    if template_height_success_prob:
        template_height = float(truth["success_probability_at_resonance"])
    else:
        template_height = float(BENCHMARK_DEFAULTS["template_height"])

    settings = dict(BENCHMARK_DEFAULTS)
    settings.update(
        {
            "template_height": template_height,
            "num_tries": int(truth.get("num_tries", SIMULATION_DEFAULTS["num_tries"])),
            "require_one_peak_per_side": bool(require_one_peak_per_side),
            "normalization_mode": case["normalization_mode"] or "raw",
            "width_mode": case["width_mode"] or "fixed",
            "benchmark_variant": case["variant"],
        }
    )
    return settings


def run_case(
    case: dict[str, Any],
    x: np.ndarray,
    y_dip: np.ndarray,
    settings: dict[str, Any],
) -> dict:
    algorithm = case["algorithm"]

    if algorithm == "LMFitSinglePerSide":
        return run_lmfit_single_side(x, y_dip, settings=settings)

    if algorithm == "LMFitDoubleJoint":
        return run_lmfit_double_joint(x, y_dip, settings=settings)
    
    if algorithm == "DoubleMLE_Exact":
        return run_double_mle_exact(x, y_dip, settings=settings)

    if algorithm == "DoubleMLE_Approx":
        return run_double_mle_approx(x, y_dip, settings=settings)

    if algorithm == "PaperCA_Verbatim":
        return run_paper_ca_verbatim(x, y_dip, settings=settings)

    if algorithm == "PaperCA_Clean":
        return run_paper_ca_clean(x, y_dip, settings=settings)

    if algorithm == "SingleCorrelation":
        return run_single_correlation(x, y_dip, settings=settings)

    if algorithm == "DoubleCorrelation":
        return run_double_correlation(x, y_dip, settings=settings)

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def case_key(case: dict[str, Any]) -> str:
    return f"{case['algorithm']}:{case['variant']}"


def mean_error(result: dict, truth: dict) -> tuple[float, float, float]:
    err_f1 = abs(float(result["f1_hat"]) - float(truth["resonance_value1"]))
    err_f2 = abs(float(result["f2_hat"]) - float(truth["resonance_value2"]))
    mean_err = 0.5 * (err_f1 + err_f2)
    return err_f1, err_f2, mean_err


def summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean_err": float("nan"),
            "std_err": float("nan"),
            "median_err": float("nan"),
            "best_err": float("nan"),
            "worst_err": float("nan"),
        }

    arr = np.asarray(values, dtype=float)

    return {
        "mean_err": float(np.mean(arr)),
        # Population std, matching the multi-trace GUI behavior.
        "std_err": float(np.std(arr, ddof=0)) if len(arr) > 1 else 0.0,
        "median_err": float(np.median(arr)),
        "best_err": float(np.min(arr)),
        "worst_err": float(np.max(arr)),
    }


def build_summary_rows(
    cases: list[dict[str, Any]],
    errors_by_key: dict[str, list[float]],
    runtimes_by_key: dict[str, list[float]],
    failures_by_key: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for case in cases:
        key = case_key(case)
        errors = errors_by_key[key]
        runtimes = runtimes_by_key[key]
        stats = summarize_values(errors)

        avg_runtime_ms = float(np.mean(runtimes)) if runtimes else float("nan")

        rows.append(
            {
                "algorithm": case["algorithm"],
                "variant": case["variant"],
                "n": len(errors),
                "failure_count": int(failures_by_key[key]),
                "mean_err": stats["mean_err"],
                "std_err": stats["std_err"],
                "median_err": stats["median_err"],
                "best_err": stats["best_err"],
                "worst_err": stats["worst_err"],
                "avg_runtime_ms": avg_runtime_ms,
            }
        )

    return rows


def output_path_from_args(args: argparse.Namespace) -> Path:
    if args.output:
        path = Path(args.output)
    else:
        stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = Path(f"odmr_complete_benchmark_summary_{stamp}.csv")

    if path.suffix.lower() != ".csv":
        path = path.with_suffix(".csv")

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "algorithm",
        "variant",
        "n",
        "failure_count",
        "mean_err",
        "std_err",
        "median_err",
        "best_err",
        "worst_err",
        "avg_runtime_ms",
    ]

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            np.inf if not np.isfinite(float(r["mean_err"])) else float(r["mean_err"]),
            str(r["algorithm"]),
            str(r["variant"]),
        ),
    )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow(row)


def print_summary(rows: list[dict[str, Any]]) -> None:
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            np.inf if not np.isfinite(float(r["mean_err"])) else float(r["mean_err"]),
            str(r["algorithm"]),
            str(r["variant"]),
        ),
    )

    print()
    print("Summary")
    print("-" * 110)
    print(
        f"{'algorithm':<22} "
        f"{'variant':<20} "
        f"{'n':>6} "
        f"{'fail':>6} "
        f"{'mean':>10} "
        f"{'std':>10} "
        f"{'median':>10} "
        f"{'best':>10} "
        f"{'worst':>10} "
        f"{'avg_ms':>10}"
    )
    print("-" * 110)

    for row in rows_sorted:
        print(
            f"{row['algorithm']:<22} "
            f"{row['variant']:<20} "
            f"{row['n']:>6} "
            f"{row['failure_count']:>6} "
            f"{row['mean_err']:>10.3f} "
            f"{row['std_err']:>10.3f} "
            f"{row['median_err']:>10.3f} "
            f"{row['best_err']:>10.3f} "
            f"{row['worst_err']:>10.3f} "
            f"{row['avg_runtime_ms']:>10.2f}"
        )

    print("-" * 110)


def validate_args(args: argparse.Namespace) -> None:
    if int(args.num_traces) <= 0:
        raise ValueError("--num-traces must be positive.")

    if int(args.num_points) < 3:
        raise ValueError("--num-points must be at least 3.")

    if int(args.num_tries) <= 0:
        raise ValueError("--num-tries must be positive.")

    if float(args.range_end) <= float(args.range_start):
        raise ValueError("--range-end must be greater than --range-start.")

    if int(args.width_max) < int(args.width_min):
        raise ValueError("--width-max must be greater than or equal to --width-min.")

    if not (0.0 < float(args.success_probability) < 1.0):
        raise ValueError("--success-probability must be between 0 and 1.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    cases = selected_cases(args.algorithms)
    if not cases:
        raise ValueError("No benchmark cases selected.")

    simulation_kwargs = simulation_kwargs_from_args(args)
    output_path = output_path_from_args(args)

    errors_by_key: dict[str, list[float]] = {
        case_key(case): []
        for case in cases
    }
    runtimes_by_key: dict[str, list[float]] = {
        case_key(case): []
        for case in cases
    }
    failures_by_key: dict[str, int] = {
        case_key(case): 0
        for case in cases
    }

    total_jobs = int(args.num_traces) * len(cases)
    completed_jobs = 0
    progress_every = max(1, total_jobs // 100)

    start_time = time.perf_counter()

    print("Complete ODMR benchmark")
    print(f"  num_traces                    = {args.num_traces}")
    print(f"  start_seed                    = {args.start_seed}")
    print(f"  selected algorithm families   = {args.algorithms or 'ALL'}")
    print(f"  selected benchmark cases      = {len(cases)}")
    print(f"  template_height_success_prob  = {args.template_height_success_prob}")
    print(f"  require_one_peak_per_side     = {args.require_one_peak_per_side}")
    print(f"  output                        = {output_path}")
    print()

    for trace_idx in range(int(args.num_traces)):
        seed = int(args.start_seed) + trace_idx

        x, y_dip, truth = generate_random_odmr_trace(
            seed=seed,
            **simulation_kwargs,
        )

        for case in cases:
            key = case_key(case)
            settings = settings_for_case(
                case,
                truth=truth,
                template_height_success_prob=bool(args.template_height_success_prob),
                require_one_peak_per_side=bool(args.require_one_peak_per_side),
            )

            t0 = time.perf_counter()

            try:
                result = run_case(case, x, y_dip, settings)
                runtime_ms = 1000.0 * (time.perf_counter() - t0)

                _err_f1, _err_f2, mean_err = mean_error(result, truth)

                errors_by_key[key].append(float(mean_err))
                runtimes_by_key[key].append(float(runtime_ms))

            except Exception as exc:
                failures_by_key[key] += 1
                print(
                    f"[WARNING] case failed | seed={seed} | "
                    f"{case['algorithm']} | {case['variant']} | "
                    f"{type(exc).__name__}: {exc}"
                )

            completed_jobs += 1

            if completed_jobs % progress_every == 0 or completed_jobs == total_jobs:
                elapsed = time.perf_counter() - start_time
                pct = 100.0 * completed_jobs / total_jobs
                print(
                    f"Progress: {completed_jobs}/{total_jobs} "
                    f"({pct:5.1f}%) | elapsed={elapsed:.1f}s"
                )

    rows = build_summary_rows(
        cases,
        errors_by_key,
        runtimes_by_key,
        failures_by_key,
    )

    write_summary_csv(output_path, rows)
    print_summary(rows)

    elapsed = time.perf_counter() - start_time
    print()
    print(f"Saved summary CSV: {output_path}")
    print(f"Total elapsed time: {elapsed:.2f} s")


if __name__ == "__main__":
    main()