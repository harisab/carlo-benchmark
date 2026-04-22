# ODMR Benchmark

A small Python project for:

- generating synthetic ODMR spectra
- benchmarking classical peak/dip detection algorithms
- training simple PyTorch models for ODMR prediction

## Main components

- simulation
- correlation / MLE / clustering baselines
- PyTorch training and inference
- optional GUI for local exploration

## Quick start

```bash
pip install -e ".[ml,gui,dev]"
```

## Example scripts
- python scripts/generate_dataset.py
- python scripts/benchmark_algorithms.py
- python scripts/train_pytorch.py
- python scripts/predict_pytorch.py