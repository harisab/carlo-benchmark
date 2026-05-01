# ODMR Benchmark Project

This project benchmarks different algorithms for estimating two ODMR resonance locations from simulated ODMR dip spectra.

The project is intentionally kept lightweight and readable. It is not meant to be a large framework. The main goals are:

- generate synthetic ODMR traces
- run several benchmark cases on the same data
- compare predicted resonance frequencies against known truth values
- inspect single-trace behavior visually
- compare algorithms statistically across many simulated traces

---

## Main entry points

### Single-trace GUI

Use the single-trace GUI when you want to inspect one simulated ODMR trace and visually compare algorithm predictions.

```bash
python scripts/run_gui.py
```

The single-trace GUI lets you:

- generate one synthetic ODMR trace
- run selected benchmark cases
- show or hide predicted centerlines
- show or hide fitted/matched waves
- compare estimated `f1`, `f2`, width/gamma, score, and error
- view a mean-error bar chart for the current trace

By default:

- all benchmark rows are selected to run
- only the truth centerlines are visible
- template height follows the simulated success probability

---

### Multi-trace GUI

Use the multi-trace GUI when you want to compare algorithms statistically across many random traces.

```bash
python scripts/run_multi_trace_gui.py
```

The multi-trace GUI lets you:

- choose how many traces to simulate
- choose the start seed
- edit simulation settings
- select which benchmark cases to run
- run without freezing the UI
- cancel a long benchmark run
- view live summary statistics
- view a live mean-error bar chart
- view running mean error for the current top cases

---

### Exporting CSV from the multi-trace GUI

After a multi-trace run finishes, click:

```text
Export summary CSV
```

The exported CSV contains one row per benchmark case with:

```text
algorithm
variant
n
mean_err
std_err
median_err
best_err
worst_err
avg_runtime_ms
```

This is useful for saving benchmark results from 100, 500, 1000, or more simulated traces.

---

### Command-line single-trace benchmark

You can also run one benchmark trace from the command line:

```bash
python scripts/single_trace_benchmark.py
```

Run only selected algorithm families:

```bash
python scripts/single_trace_benchmark.py --algorithms SingleCorrelation DoubleCorrelation
```

Run only the paper clustering algorithms:

```bash
python scripts/single_trace_benchmark.py --algorithms PaperCA_Verbatim PaperCA_Clean
```

Disable using the success probability as the template height:

```bash
python scripts/single_trace_benchmark.py --no-template-height-success-prob
```

Disable the one-peak-per-side constraint:

```bash
python scripts/single_trace_benchmark.py --no-require-one-peak-per-side
```

---

## Project defaults

Most project-wide constants live in:

```text
src/odmr/project_defaults.py
```

Important dictionaries/lists:

```text
SIMULATION_DEFAULTS
BENCHMARK_DEFAULTS
APP_DEFAULTS
BENCHMARK_CASES
```

The benchmark design is intentionally explicit: each entry in `BENCHMARK_CASES` is one GUI row and one runnable benchmark case.

---

## Benchmark cases

Each benchmark case has:

```text
algorithm
variant
normalization_mode
width_mode
```

For LMFit and Paper CA cases, `normalization_mode` and `width_mode` may be `None` because those algorithms do not use those options.

---

### LMFitSinglePerSide

Fits one Lorentzian dip independently on the left half and right half of the trace.

Variant:

```text
lmfit_single_side
```

This is similar in spirit to independently finding one resonance on each side.

---

### LMFitDoubleJoint

Fits a two-Lorentzian dip model jointly across the whole trace.

Variant:

```text
lmfit_double_joint
```

This uses a summed two-dip model and fits both resonance centers together.

---

### PaperCA_Verbatim

Single-trace adaptation of Dylan Stone's paper clustering algorithm.

This version aims to preserve the reference behavior as closely as practical for our in-memory trace format. It keeps the likely width calculation bug from the reference implementation where `width2` is computed from `cluster1` again instead of `cluster2`.

Variant:

```text
paper_ca_verbatim
```

Use this when you want to compare against the paper/reference logic as closely as possible.

---

### PaperCA_Clean

Same clustering idea as `PaperCA_Verbatim`, but with small cleanup changes:

- fixes `width2` to use `cluster2`
- uses a reproducible `random_state`
- keeps the same high-level vertical-then-horizontal clustering approach

Variant:

```text
paper_ca_clean
```

Use this when you want a cleaner implementation of the same idea for fair benchmarking.

---

### SingleCorrelation

Matches one Lorentzian template independently on the left and right sides of the trace.

Variants combine normalization mode and width mode:

```text
raw_fixed
raw_scan
l1_fixed
l1_scan
l2_fixed
l2_scan
demean_fixed
demean_scan
demean_l1_fixed
demean_l1_scan
demean_l2_fixed
demean_l2_scan
```

Meaning:

- `raw`: no normalization
- `l1`: normalize by total absolute weight
- `l2`: normalize by vector energy
- `demean`: subtract mean before scoring
- `demean_l1`: subtract mean, then L1 normalize
- `demean_l2`: subtract mean, then L2 normalize
- `fixed`: use `standard_width`
- `scan`: search over widths from `min_width` to `max_width`

---

### DoubleCorrelation

Searches over paired left/right Lorentzian templates jointly.

It uses the same variant list as `SingleCorrelation`:

```text
raw_fixed
raw_scan
l1_fixed
l1_scan
l2_fixed
l2_scan
demean_fixed
demean_scan
demean_l1_fixed
demean_l1_scan
demean_l2_fixed
demean_l2_scan
```

Conceptually:

- `SingleCorrelation` finds each side independently.
- `DoubleCorrelation` evaluates paired left/right templates together.

---

## Template height

The manual default template height is:

```text
1.0
```

However, by default the GUI and command-line benchmark use the simulated trace success probability as the effective template height.

This default is controlled by:

```python
BENCHMARK_DEFAULTS["use_success_probability_for_template_height"]
```

If enabled, the effective template height becomes:

```python
truth["success_probability_at_resonance"]
```

If disabled, the effective template height becomes:

```python
BENCHMARK_DEFAULTS["template_height"]
```

---

## One-peak-per-side assumption

The benchmark defaults assume one resonance on the left side and one resonance on the right side.

This is controlled by:

```python
BENCHMARK_DEFAULTS["require_one_peak_per_side"]
```

In the GUI, this is exposed as:

```text
require_one_peak_per_side
```

In the command-line script, you can disable it with:

```bash
python scripts/single_trace_benchmark.py --no-require-one-peak-per-side
```

---

## Recommended workflow

### 1. Inspect one trace

```bash
python scripts/run_gui.py
```

Use this to understand what each algorithm is doing visually.

### 2. Run statistical comparison

```bash
python scripts/run_multi_trace_gui.py
```

Use this to compare benchmark cases across many traces.

### 3. Export results

Use the multi-trace GUI's `Export summary CSV` button.

### 4. Commit stable milestones

Example:

```bash
git add .
git commit -m "Add Paper CA benchmarks and multi-trace export"
```

---

## Notes

This project intentionally avoids excessive framework-style abstraction. The benchmark cases are explicit so the code is easy to read and easy to review.

