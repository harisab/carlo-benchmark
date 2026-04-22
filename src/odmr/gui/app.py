from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from odmr.benchmark_config import BenchmarkConfig
from odmr.simulation import generate_random_odmr_trace
from odmr.algorithms.single_correlation import run_single_correlation
from odmr.algorithms.double_correlation import run_double_correlation


def apply_modern_dark(app) -> None:
    app.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(20, 20, 24))
    pal.setColor(QPalette.WindowText, QColor(230, 230, 230))
    pal.setColor(QPalette.Base, QColor(14, 14, 18))
    pal.setColor(QPalette.AlternateBase, QColor(24, 24, 28))
    pal.setColor(QPalette.ToolTipBase, QColor(230, 230, 230))
    pal.setColor(QPalette.ToolTipText, QColor(20, 20, 24))
    pal.setColor(QPalette.Text, QColor(230, 230, 230))
    pal.setColor(QPalette.Button, QColor(30, 30, 36))
    pal.setColor(QPalette.ButtonText, QColor(230, 230, 230))
    pal.setColor(QPalette.Highlight, QColor(72, 118, 255))
    pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(pal)

    app.setStyleSheet(
        """
        QWidget { font-size: 12px; }
        QGroupBox {
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 10px;
            margin-top: 10px;
            padding: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 6px;
            color: rgba(255,255,255,0.90);
        }
        QPushButton {
            padding: 8px 12px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.06);
        }
        QPushButton:hover { background: rgba(255,255,255,0.10); }
        QPushButton:disabled { color: rgba(255,255,255,0.30); }
        QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            padding: 6px;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.10);
            background: rgba(255,255,255,0.04);
        }
        """
    )


class MplCanvas(FigureCanvas):
    def __init__(self, title: str = "") -> None:
        fig = Figure(figsize=(7.0, 5.0), dpi=100)
        self.ax = fig.add_subplot(111)
        if title:
            self.ax.set_title(title)
        super().__init__(fig)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ODMR Correlation GUI")

        self.current_x: np.ndarray | None = None
        self.current_y_dip: np.ndarray | None = None
        self.current_truth: dict[str, Any] | None = None

        self.last_single_result: dict[str, Any] | None = None
        self.last_double_result: dict[str, Any] | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        left = QWidget()
        left_layout = QVBoxLayout(left)

        left_layout.addWidget(self._build_simulation_group())
        left_layout.addWidget(self._build_benchmark_group())
        left_layout.addWidget(self._build_actions_group())
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.lbl_truth = QLabel("Truth: —")
        self.lbl_truth.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.canvas = MplCanvas("ODMR dip trace + correlation estimates")

        self.txt_results = QPlainTextEdit()
        self.txt_results.setReadOnly(True)
        self.txt_results.setPlaceholderText("Generate a trace, then run the selected config or the standard benchmark set.")

        right_layout.addWidget(self.lbl_truth)
        right_layout.addWidget(self.canvas, stretch=1)
        right_layout.addWidget(self.txt_results, stretch=0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([380, 980])

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.addWidget(splitter)

        self.setCentralWidget(root)

    def _build_simulation_group(self) -> QGroupBox:
        g = QGroupBox("Simulation")
        form = QFormLayout(g)

        self.sp_bins = QSpinBox()
        self.sp_bins.setRange(50, 5000)
        self.sp_bins.setValue(199)

        self.sp_tries = QSpinBox()
        self.sp_tries.setRange(1, 1000)
        self.sp_tries.setValue(20)

        self.ds_start = QDoubleSpinBox()
        self.ds_start.setRange(0.0, 1e9)
        self.ds_start.setDecimals(3)
        self.ds_start.setValue(3000.0)

        self.ds_end = QDoubleSpinBox()
        self.ds_end.setRange(0.0, 1e9)
        self.ds_end.setDecimals(3)
        self.ds_end.setValue(4000.0)

        self.ds_center = QDoubleSpinBox()
        self.ds_center.setRange(0.0, 1e9)
        self.ds_center.setDecimals(3)
        self.ds_center.setValue(3500.0)

        self.ds_offset = QDoubleSpinBox()
        self.ds_offset.setRange(0.0, 1e6)
        self.ds_offset.setDecimals(3)
        self.ds_offset.setValue(450.0)

        self.ds_p = QDoubleSpinBox()
        self.ds_p.setRange(1e-6, 0.99)
        self.ds_p.setDecimals(6)
        self.ds_p.setSingleStep(0.01)
        self.ds_p.setValue(0.15)

        self.ds_wmin = QSpinBox()
        self.ds_wmin.setRange(1, 100000)
        self.ds_wmin.setValue(10)

        self.ds_wmax = QSpinBox()
        self.ds_wmax.setRange(1, 100000)
        self.ds_wmax.setValue(50)

        self.sp_seed = QSpinBox()
        self.sp_seed.setRange(0, 2_000_000_000)
        self.sp_seed.setValue(123)

        form.addRow("n_bins", self.sp_bins)
        form.addRow("n_tries", self.sp_tries)
        form.addRow("MW start (MHz)", self.ds_start)
        form.addRow("MW end (MHz)", self.ds_end)
        form.addRow("center_frequency", self.ds_center)
        form.addRow("offset_max", self.ds_offset)
        form.addRow("success_prob", self.ds_p)
        form.addRow("width_min", self.ds_wmin)
        form.addRow("width_max", self.ds_wmax)
        form.addRow("seed", self.sp_seed)

        return g

    def _build_benchmark_group(self) -> QGroupBox:
        g = QGroupBox("Benchmark config")
        form = QFormLayout(g)

        self.cmb_width_mode = QComboBox()
        self.cmb_width_mode.addItems(["scan", "fixed"])
        self.cmb_width_mode.setCurrentText("scan")

        self.ds_standard_width = QDoubleSpinBox()
        self.ds_standard_width.setRange(1e-6, 1e6)
        self.ds_standard_width.setDecimals(3)
        self.ds_standard_width.setValue(20.0)

        self.ds_min_width = QDoubleSpinBox()
        self.ds_min_width.setRange(1e-6, 1e6)
        self.ds_min_width.setDecimals(3)
        self.ds_min_width.setValue(10.0)

        self.ds_max_width = QDoubleSpinBox()
        self.ds_max_width.setRange(1e-6, 1e6)
        self.ds_max_width.setDecimals(3)
        self.ds_max_width.setValue(50.0)

        self.ds_width_step = QDoubleSpinBox()
        self.ds_width_step.setRange(1e-6, 1e6)
        self.ds_width_step.setDecimals(3)
        self.ds_width_step.setValue(1.0)

        self.ds_template_height = QDoubleSpinBox()
        self.ds_template_height.setRange(0.0, 10.0)
        self.ds_template_height.setDecimals(6)
        self.ds_template_height.setValue(0.15)

        self.ck_normalize = QCheckBox("normalize template")
        self.ck_normalize.setChecked(False)

        self.ck_demean = QCheckBox("demean before correlation")
        self.ck_demean.setChecked(True)

        self.ck_require_side = QCheckBox("require one peak per side")
        self.ck_require_side.setChecked(True)

        self.sp_center_step = QSpinBox()
        self.sp_center_step.setRange(1, 100)
        self.sp_center_step.setValue(1)

        self.ds_restrict = QDoubleSpinBox()
        self.ds_restrict.setRange(0.0, 1e6)
        self.ds_restrict.setDecimals(3)
        self.ds_restrict.setValue(0.0)

        form.addRow("width_mode", self.cmb_width_mode)
        form.addRow("standard_width", self.ds_standard_width)
        form.addRow("min_width", self.ds_min_width)
        form.addRow("max_width", self.ds_max_width)
        form.addRow("width_step", self.ds_width_step)
        form.addRow("template_height", self.ds_template_height)
        form.addRow(self.ck_normalize)
        form.addRow(self.ck_demean)
        form.addRow(self.ck_require_side)
        form.addRow("center_step_bins", self.sp_center_step)
        form.addRow("restrict_window_mhz (0=off)", self.ds_restrict)

        return g

    def _build_actions_group(self) -> QGroupBox:
        g = QGroupBox("Actions")
        layout = QVBoxLayout(g)

        self.btn_generate = QPushButton("Generate trace")
        self.btn_run_selected = QPushButton("Run selected config")
        self.btn_run_benchmark_set = QPushButton("Run standard 4-variant set")

        self.btn_generate.clicked.connect(self._on_generate)
        self.btn_run_selected.clicked.connect(self._on_run_selected)
        self.btn_run_benchmark_set.clicked.connect(self._on_run_benchmark_set)

        layout.addWidget(self.btn_generate)
        layout.addWidget(self.btn_run_selected)
        layout.addWidget(self.btn_run_benchmark_set)

        return g

    def _simulation_kwargs(self) -> dict[str, Any]:
        return {
            "num_points": int(self.sp_bins.value()),
            "num_tries": int(self.sp_tries.value()),
            "range_start": float(self.ds_start.value()),
            "range_end": float(self.ds_end.value()),
            "center_frequency": float(self.ds_center.value()),
            "offset_max": float(self.ds_offset.value()),
            "width_min": int(self.ds_wmin.value()),
            "width_max": int(self.ds_wmax.value()),
            "success_probability_at_resonance": float(self.ds_p.value()),
            "seed": int(self.sp_seed.value()),
        }

    def _cfg_from_ui(self) -> BenchmarkConfig:
        restrict = float(self.ds_restrict.value())
        restrict_value = None if restrict <= 0.0 else restrict

        return BenchmarkConfig(
            min_width=float(self.ds_min_width.value()),
            max_width=float(self.ds_max_width.value()),
            width_step=float(self.ds_width_step.value()),
            standard_width=float(self.ds_standard_width.value()),
            width_mode=str(self.cmb_width_mode.currentText()),
            template_height=float(self.ds_template_height.value()),
            normalize_template=bool(self.ck_normalize.isChecked()),
            demean=bool(self.ck_demean.isChecked()),
            center_step_bins=int(self.sp_center_step.value()),
            restrict_window_mhz=restrict_value,
            require_one_peak_per_side=bool(self.ck_require_side.isChecked()),
        )

    def _four_standard_variants(self) -> list[BenchmarkConfig]:
        base = self._cfg_from_ui()

        return [
            replace(base, normalize_template=False, width_mode="scan"),
            replace(base, normalize_template=True, width_mode="scan"),
            replace(base, normalize_template=False, width_mode="fixed"),
            replace(base, normalize_template=True, width_mode="fixed"),
        ]

    def _append_results(self, text: str) -> None:
        self.txt_results.appendPlainText(text)

    def _clear_results(self) -> None:
        self.txt_results.setPlainText("")

    def _format_error_block(self, result: dict[str, Any], truth: dict[str, Any]) -> str:
        e1 = abs(float(result["f1_hat"]) - float(truth["resonance_value1"]))
        e2 = abs(float(result["f2_hat"]) - float(truth["resonance_value2"]))
        em = 0.5 * (e1 + e2)

        lines = [
            f"  variant   = {result['benchmark_variant']}",
            f"  f1_hat    = {result['f1_hat']:.3f} MHz",
            f"  f2_hat    = {result['f2_hat']:.3f} MHz",
        ]

        if "gamma_left" in result:
            lines.append(f"  gamma_L   = {result['gamma_left']:.3f}")
            lines.append(f"  gamma_R   = {result['gamma_right']:.3f}")
        else:
            lines.append(f"  gamma     = {result['gamma']:.3f}")

        lines.append(f"  score     = {result['score']:.3f}")
        lines.append(f"  err_f1    = {e1:.3f} MHz")
        lines.append(f"  err_f2    = {e2:.3f} MHz")
        lines.append(f"  mean_err  = {em:.3f} MHz")
        return "\n".join(lines)

    def _plot_current_trace(self) -> None:
        ax = self.canvas.ax
        ax.clear()
        ax.set_title("ODMR dip trace + correlation estimates")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Normalized count (dip)")

        if self.current_x is None or self.current_y_dip is None or self.current_truth is None:
            self.canvas.draw_idle()
            return

        x = self.current_x
        y = self.current_y_dip
        truth = self.current_truth

        ax.scatter(x, y, s=12, label="data")

        ax.axvline(float(truth["resonance_value1"]), linestyle="--", linewidth=1.5, label="truth f1")
        ax.axvline(float(truth["resonance_value2"]), linestyle="--", linewidth=1.5, label="truth f2")

        if self.last_single_result is not None:
            ax.axvline(float(self.last_single_result["f1_hat"]), linestyle=":", linewidth=1.5, label="single f1")
            ax.axvline(float(self.last_single_result["f2_hat"]), linestyle=":", linewidth=1.5, label="single f2")

        if self.last_double_result is not None:
            ax.axvline(float(self.last_double_result["f1_hat"]), linestyle="-", linewidth=1.5, label="double f1")
            ax.axvline(float(self.last_double_result["f2_hat"]), linestyle="-", linewidth=1.5, label="double f2")

        ax.legend(fontsize=8, loc="best")
        self.canvas.draw_idle()

    @Slot()
    def _on_generate(self) -> None:
        try:
            x, y_dip, truth = generate_random_odmr_trace(**self._simulation_kwargs())
        except Exception as exc:
            QMessageBox.critical(self, "Generation error", f"{type(exc).__name__}: {exc}")
            return

        self.current_x = x
        self.current_y_dip = y_dip
        self.current_truth = truth
        self.last_single_result = None
        self.last_double_result = None

        self.lbl_truth.setText(
            "Truth: "
            f"f1={truth['resonance_value1']:.3f} MHz, "
            f"f2={truth['resonance_value2']:.3f} MHz, "
            f"width={truth['width']:.3f}, "
            f"n_tries={truth['num_tries']}"
        )

        self._clear_results()
        self._append_results("Generated trace.")
        self._append_results(
            f"Truth\n"
            f"  f1        = {truth['resonance_value1']:.3f} MHz\n"
            f"  f2        = {truth['resonance_value2']:.3f} MHz\n"
            f"  width     = {truth['width']:.3f}\n"
        )
        self._plot_current_trace()

    def _run_current_algorithms(self, cfg: BenchmarkConfig) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.current_x is None or self.current_y_dip is None:
            raise RuntimeError("No current trace. Generate a trace first.")

        single = run_single_correlation(self.current_x, self.current_y_dip, cfg=cfg)
        double = run_double_correlation(self.current_x, self.current_y_dip, cfg=cfg)
        return single, double

    @Slot()
    def _on_run_selected(self) -> None:
        if self.current_truth is None:
            QMessageBox.information(self, "No trace", "Generate a trace first.")
            return

        try:
            cfg = self._cfg_from_ui()
            single, double = self._run_current_algorithms(cfg)
        except Exception as exc:
            QMessageBox.critical(self, "Run error", f"{type(exc).__name__}: {exc}")
            return

        self.last_single_result = single
        self.last_double_result = double
        self._plot_current_trace()

        self._clear_results()
        self._append_results("Selected benchmark config")
        self._append_results(
            f"  width_mode={cfg.width_mode}, normalize_template={cfg.normalize_template}, "
            f"require_one_peak_per_side={cfg.require_one_peak_per_side}"
        )
        self._append_results("")

        self._append_results("SingleCorrelation")
        self._append_results(self._format_error_block(single, self.current_truth))
        self._append_results("")

        self._append_results("DoubleCorrelation")
        self._append_results(self._format_error_block(double, self.current_truth))

    @Slot()
    def _on_run_benchmark_set(self) -> None:
        if self.current_truth is None:
            QMessageBox.information(self, "No trace", "Generate a trace first.")
            return

        try:
            variants = self._four_standard_variants()
        except Exception as exc:
            QMessageBox.critical(self, "Config error", f"{type(exc).__name__}: {exc}")
            return

        self._clear_results()
        self._append_results("Standard 4-variant correlation benchmark set")
        self._append_results("")

        for cfg in variants:
            try:
                single, double = self._run_current_algorithms(cfg)
            except Exception as exc:
                QMessageBox.critical(self, "Run error", f"{type(exc).__name__}: {exc}")
                return

            self._append_results(
                f"Variant: width_mode={cfg.width_mode}, normalize_template={cfg.normalize_template}"
            )
            self._append_results("SingleCorrelation")
            self._append_results(self._format_error_block(single, self.current_truth))
            self._append_results("")
            self._append_results("DoubleCorrelation")
            self._append_results(self._format_error_block(double, self.current_truth))
            self._append_results("")