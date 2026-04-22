from __future__ import annotations

from odmr.simulation import generate_random_odmr_trace
from odmr.algorithms.double_correlation import run_double_correlation


def main() -> None:
    x, y_dip, meta = generate_random_odmr_trace(seed=123)

    result = run_double_correlation(
        x,
        y_dip,
        min_width=10.0,
        max_width=50.0,
        width_step=1.0,
        template_height=0.15,
        normalize_template=False,
        demean=True,
        center_step_bins=1,
        restrict_window_mhz=None,
    )

    print("Truth")
    print(f"  f1 = {meta['resonance_value1']:.3f} MHz")
    print(f"  f2 = {meta['resonance_value2']:.3f} MHz")
    print(f"  width = {meta['width']:.3f} MHz")

    print("\nEstimate")
    print(f"  f1_hat = {result['f1_hat']:.3f} MHz")
    print(f"  f2_hat = {result['f2_hat']:.3f} MHz")
    print(f"  gamma = {result['gamma']:.3f}")
    print(f"  score = {result['score']:.3f}")

    e1 = abs(result["f1_hat"] - meta["resonance_value1"])
    e2 = abs(result["f2_hat"] - meta["resonance_value2"])
    em = 0.5 * (e1 + e2)

    print("\nError")
    print(f"  err_f1 = {e1:.3f} MHz")
    print(f"  err_f2 = {e2:.3f} MHz")
    print(f"  mean_err = {em:.3f} MHz")


if __name__ == "__main__":
    main()