from __future__ import annotations

from odmr.simulation import generate_random_odmr_trace


def main() -> None:
    freq_axis, odmr_data, metadata = generate_random_odmr_trace(seed=123)

    print("Generated ODMR trace")
    print(f"f1 = {metadata['resonance_value1']:.3f} MHz")
    print(f"f2 = {metadata['resonance_value2']:.3f} MHz")
    print(f"width = {metadata['width']:.3f} MHz")
    print(f"num_points = {len(freq_axis)}")
    print(f"first 5 y values = {odmr_data[:5]}")


if __name__ == "__main__":
    main()