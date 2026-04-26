from __future__ import annotations

import time
from typing import Any

import numpy as np

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from odmr.benchmark_config import BenchmarkConfig
from odmr.benchmark_registry import (
    build_jobs_from_rows,
    build_variant_rows,
    default_template_height,
    get_algorithm_spec_map,
    record_key,
    row_key,
    run_algorithm_job,
)
from odmr.algorithms.common import lorentzian_peak
from odmr.simulation import generate_random_odmr_trace


TRUTH_COLOR = "limegreen"


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
        QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTableWidget {
            padding: 6px;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.10);
            background: rgba(255,255,255,0.04);
        }
        QHeaderView::section {
            background: rgba(255,255,255,0.06);
            padding: 6px;
            border: 1px solid rgba(255,255,255,0.08);
        }
        """
    )


class MplCanvas(FigureCanvas):
    def __init__(self, title: str = "") -> None:
        fig = Figure(figsize=(8.2, 5.5), dpi=100)
        self.ax = fig.add_subplot(111)
        if title:
            self.ax.set_title(title)
        super().__init__(fig)


def _table_item(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item


def template_dip_from_params(
    x: np.ndarray,
    f1: float,
    f2: float,
    gamma: float,
    height: float,
) -> np.ndarray:
    return 1.0 - (
        lorentzian_peak(x, f1, gamma, height) +
        lorentzian_peak(x, f2, gamma, height)
    )


class BenchmarkWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(list)
    cancelled = Signal(list)
    errored = Signal(str)

    def __init__(
        self,
        x: np.ndarray,
        y_dip: np.ndarray,
        truth: dict[str, Any],
        jobs: list[dict[str, Any]],
    ) -> None:
        super().__init__()
        self.x = x
        self.y_dip = y_dip
        self.truth = truth
        self.jobs = jobs
        self._cancel_requested = False

    @Slot()
    def cancel(self) -> None:
        self._cancel_requested = True

    @Slot()
    def run(self) -> None:
        try:
            records: list[dict[str, Any]] = []
            total = max(1, len(self.jobs))
            done = 0

            for job in self.jobs:
                if self._cancel_requested:
                    self.cancelled.emit(records)
                    return

                t0 = time.perf_counter()
                result = run_algorithm_job(job, self.x, self.y_dip)
                runtime_ms = 1000.0 * (time.perf_counter() - t0)

                e1 = abs(float(result["f1_hat"]) - float(self.truth["resonance_value1"]))
                e2 = abs(float(result["f2_hat"]) - float(self.truth["resonance_value2"]))
                em = 0.5 * (e1 + e2)

                if "gamma_left" in result:
                    gamma_repr = f"{result['gamma_left']:.3f} / {result['gamma_right']:.3f}"
                else:
                    gamma_repr = f"{result['gamma']:.3f}"

                records.append(
                    {
                        "algorithm": result["name"],
                        "variant": result["benchmark_variant"],
                        "f1_hat": float(result["f1_hat"]),
                        "f2_hat": float(result["f2_hat"]),
                        "gamma_repr": gamma_repr,
                        "score": float(result["score"]),
                        "err_f1": float(e1),
                        "err_f2": float(e2),
                        "mean_err": float(em),
                        "runtime_ms": float(runtime_ms),
                        "result": result,
                    }
                )

                done += 1
                self.progress.emit(done, total, f"{result['name']} | {result['benchmark_variant']}")

            self.finished.emit(records)

        except Exception as exc:
            self.errored.emit(f"{type(exc).__name__}: {exc}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ODMR Correlation GUI")

        self.current_x: np.ndarray | None = None
        self.current_y_dip: np.ndarray | None = None
        self.current_truth: dict[str, Any] | None = None

        self.records_by_key: dict[str, dict[str, Any]] = {}
        self.all_records: list[dict[str, Any]] = []
        self.algorithm_specs = get_algorithm_spec_map()

        self._worker_thread: QThread | None = None
        self._worker: BenchmarkWorker | None = None

        self.row_specs: list[dict[str, Any]] = []
        self.row_run_states: dict[str, bool] = {}
        self.row_center_states: dict[str, bool] = {}
        self.row_wave_states: dict[str, bool] = {}

        self._build_ui()
        self._rebuild_main_table()

    def _build_ui(self) -> None:
        top_controls = QWidget()
        top_controls_layout = QHBoxLayout(top_controls)

        top_controls_layout.addWidget(self._build_simulation_group())
        top_controls_layout.addWidget(self._build_benchmark_group())
        top_controls_layout.addWidget(self._build_actions_group())
        top_controls_layout.addWidget(self._build_main_table_group(), stretch=1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        self.lbl_truth = QLabel("Truth: —")
        self.lbl_truth.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.canvas = MplCanvas("ODMR dip trace + selected overlays")
        self.canvas_err = MplCanvas("Mean error by variant")

        graphs_splitter = QSplitter(Qt.Horizontal)
        graphs_splitter.addWidget(self.canvas)
        graphs_splitter.addWidget(self.canvas_err)
        graphs_splitter.setSizes([950, 650])

        left_layout.addWidget(self.lbl_truth)
        left_layout.addWidget(graphs_splitter, stretch=1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(QLabel("Status"))
        self.txt_status = QPlainTextEdit()
        self.txt_status.setReadOnly(True)
        right_layout.addWidget(self.txt_status, stretch=1)

        bottom_splitter = QSplitter(Qt.Horizontal)
        bottom_splitter.addWidget(left_panel)
        bottom_splitter.addWidget(right_panel)
        bottom_splitter.setSizes([1600, 400])

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.addWidget(top_controls, stretch=0)
        root_layout.addWidget(bottom_splitter, stretch=1)

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
        self.ds_template_height.setValue(float(self.ds_p.value()))

        self.sp_center_step = QSpinBox()
        self.sp_center_step.setRange(1, 100)
        self.sp_center_step.setValue(1)

        self.cmb_require_side = QComboBox()
        self.cmb_require_side.addItems(["true", "false"])
        self.cmb_require_side.setCurrentText("true")

        form.addRow("standard_width", self.ds_standard_width)
        form.addRow("min_width", self.ds_min_width)
        form.addRow("max_width", self.ds_max_width)
        form.addRow("width_step", self.ds_width_step)
        form.addRow("template_height", self.ds_template_height)
        form.addRow("center_step_bins", self.sp_center_step)
        form.addRow("require_one_peak_per_side", self.cmb_require_side)

        return g

    def _build_actions_group(self) -> QGroupBox:
        g = QGroupBox("Actions")
        layout = QVBoxLayout(g)

        self.btn_generate = QPushButton("Generate trace")
        self.btn_run_selected = QPushButton("Run checked rows")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)

        self.cmb_bar_order = QComboBox()
        self.cmb_bar_order.addItems(["table_order", "best_first"])
        self.cmb_bar_order.setCurrentText("best_first")
        self.cmb_bar_order.currentIndexChanged.connect(lambda _i: self._update_error_chart())

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)

        self.lbl_progress = QLabel("Idle")

        layout.addWidget(self.btn_generate)
        layout.addWidget(self.btn_run_selected)
        layout.addWidget(self.btn_cancel)
        layout.addWidget(QLabel("Bar chart order"))
        layout.addWidget(self.cmb_bar_order)
        layout.addWidget(self.progress)
        layout.addWidget(self.lbl_progress)

        self.btn_generate.clicked.connect(self._on_generate)
        self.btn_run_selected.clicked.connect(self._on_run_selected)
        self.btn_cancel.clicked.connect(self._on_cancel)

        return g

    def _build_main_table_group(self) -> QGroupBox:
        g = QGroupBox("Algorithms / variants")
        layout = QVBoxLayout(g)

        self.tbl_main = QTableWidget(0, 12)
        self.tbl_main.setHorizontalHeaderLabels([
            "Algorithm", "Variant", "Run", "Centerlines", "Wave",
            "f1", "f2", "Gamma", "Score",
            "err_f1", "err_f2", "mean_err"
        ])
        self.tbl_main.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_main.verticalHeader().setVisible(False)
        self.tbl_main.setMinimumWidth(1400)
        self.tbl_main.setMaximumHeight(520)

        layout.addWidget(self.tbl_main)
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

    def _base_cfg(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            min_width=float(self.ds_min_width.value()),
            max_width=float(self.ds_max_width.value()),
            width_step=float(self.ds_width_step.value()),
            standard_width=float(self.ds_standard_width.value()),
            width_mode="scan",
            template_height=float(self.ds_template_height.value()),
            normalization_mode="raw",
            center_step_bins=int(self.sp_center_step.value()),
            require_one_peak_per_side=(self.cmb_require_side.currentText() == "true"),
        )

    def _append_status(self, text: str) -> None:
        self.txt_status.appendPlainText(text)

    def _clear_status(self) -> None:
        self.txt_status.setPlainText("")

    def _rebuild_main_table(self) -> None:
        old_run = dict(self.row_run_states)
        old_center = dict(self.row_center_states)
        old_wave = dict(self.row_wave_states)

        self.row_specs = build_variant_rows(self._base_cfg())

        self.tbl_main.setRowCount(len(self.row_specs))
        self.row_run_states = {}
        self.row_center_states = {}
        self.row_wave_states = {}

        for row, spec in enumerate(self.row_specs):
            key = row_key(spec)

            if spec["kind"] == "truth":
                default_run = False
                default_center = True
                default_wave = False
            else:
                algo_defaults = self.algorithm_specs[spec["algorithm"]]
                default_run = algo_defaults.default_run
                default_center = algo_defaults.default_show_center
                default_wave = algo_defaults.default_show_wave

            self.row_run_states[key] = old_run.get(key, default_run)
            self.row_center_states[key] = old_center.get(key, default_center)
            self.row_wave_states[key] = old_wave.get(key, default_wave)

            self._populate_row(row, spec)

    def _checkbox(self, checked: bool, enabled: bool, slot) -> QCheckBox:
        ck = QCheckBox()
        ck.setChecked(checked)
        ck.setEnabled(enabled)
        ck.stateChanged.connect(slot)
        return ck

    def _populate_row(self, row: int, spec: dict[str, Any]) -> None:
        key = row_key(spec)

        run_enabled = spec["kind"] != "truth"
        run_ck = self._checkbox(self.row_run_states[key], run_enabled, lambda _s, k=key: self._on_run_checkbox_changed(k))
        center_ck = self._checkbox(self.row_center_states[key], True, lambda _s, k=key: self._on_center_checkbox_changed(k))
        wave_ck = self._checkbox(self.row_wave_states[key], True, lambda _s, k=key: self._on_wave_checkbox_changed(k))

        self.tbl_main.setCellWidget(row, 2, run_ck)
        self.tbl_main.setCellWidget(row, 3, center_ck)
        self.tbl_main.setCellWidget(row, 4, wave_ck)

        if spec["kind"] == "truth":
            f1_text = f"{self.current_truth['resonance_value1']:.3f}" if self.current_truth else ""
            f2_text = f"{self.current_truth['resonance_value2']:.3f}" if self.current_truth else ""

            self.tbl_main.setItem(row, 0, _table_item("Truth"))
            self.tbl_main.setItem(row, 1, _table_item("truth"))
            self.tbl_main.setItem(row, 5, _table_item(f1_text))
            self.tbl_main.setItem(row, 6, _table_item(f2_text))
            self.tbl_main.setItem(row, 7, _table_item(self.current_truth and f"{self.current_truth['width']:.3f}" or ""))
            self.tbl_main.setItem(row, 8, _table_item(""))
            self.tbl_main.setItem(row, 9, _table_item(""))
            self.tbl_main.setItem(row, 10, _table_item(""))
            self.tbl_main.setItem(row, 11, _table_item(""))
            return

        algo_name = spec["algorithm"]
        variant = spec["variant"]
        rec = self.records_by_key.get(record_key(algo_name, variant))

        self.tbl_main.setItem(row, 0, _table_item(algo_name))
        self.tbl_main.setItem(row, 1, _table_item(variant))
        self.tbl_main.setItem(row, 5, _table_item(f"{rec['f1_hat']:.3f}" if rec else ""))
        self.tbl_main.setItem(row, 6, _table_item(f"{rec['f2_hat']:.3f}" if rec else ""))
        self.tbl_main.setItem(row, 7, _table_item(rec["gamma_repr"] if rec else ""))
        self.tbl_main.setItem(row, 8, _table_item(f"{rec['score']:.3f}" if rec else ""))
        self.tbl_main.setItem(row, 9, _table_item(f"{rec['err_f1']:.3f}" if rec else ""))
        self.tbl_main.setItem(row, 10, _table_item(f"{rec['err_f2']:.3f}" if rec else ""))
        self.tbl_main.setItem(row, 11, _table_item(f"{rec['mean_err']:.3f}" if rec else ""))

    def _update_records(self, records: list[dict[str, Any]]) -> None:
        self.records_by_key = {record_key(r["algorithm"], r["variant"]): r for r in records}

    def _update_error_chart(self) -> None:
        ax = self.canvas_err.ax
        ax.clear()
        ax.set_title("Mean error by variant")
        ax.set_xlabel("Mean error (MHz)")
        ax.set_ylabel("Variant")

        if not self.all_records:
            self.canvas_err.draw_idle()
            return

        mode = self.cmb_bar_order.currentText()

        if mode == "table_order":
            ordered = []
            for spec in self.row_specs:
                if spec["kind"] != "variant":
                    continue
                rec = self.records_by_key.get(record_key(spec["algorithm"], spec["variant"]))
                if rec is not None:
                    ordered.append(rec)
        else:
            ordered = sorted(self.all_records, key=lambda r: (r["mean_err"], r["algorithm"], r["variant"]))

        labels = [f"{r['algorithm']} | {r['variant']}" for r in ordered]
        values = [r["mean_err"] for r in ordered]

        y = np.arange(len(values))
        ax.barh(y, values)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        self.canvas_err.figure.tight_layout()
        self.canvas_err.draw_idle()

    def _plot_current_trace(self) -> None:
        ax = self.canvas.ax
        ax.clear()
        ax.set_title("ODMR dip trace + selected overlays")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Normalized count (dip)")

        if self.current_x is None or self.current_y_dip is None or self.current_truth is None:
            self.canvas.draw_idle()
            return

        x = self.current_x
        y = self.current_y_dip
        truth = self.current_truth
        contrast = float(truth.get("success_probability_at_resonance", 0.15))

        ax.scatter(x, y, s=12, label="data")

        for spec in self.row_specs:
            key = row_key(spec)
            show_center = self.row_center_states.get(key, False)
            show_wave = self.row_wave_states.get(key, False)

            if not show_center and not show_wave:
                continue

            if spec["kind"] == "truth":
                if show_center:
                    ax.axvline(float(truth["resonance_value1"]), linestyle="--", linewidth=1.8, color=TRUTH_COLOR, label="truth f1")
                    ax.axvline(float(truth["resonance_value2"]), linestyle="--", linewidth=1.8, color=TRUTH_COLOR, label="truth f2")
                if show_wave:
                    y_truth = template_dip_from_params(
                        x,
                        float(truth["resonance_value1"]),
                        float(truth["resonance_value2"]),
                        float(truth["width"]),
                        contrast,
                    )
                    ax.plot(x, y_truth, linewidth=2.0, color=TRUTH_COLOR, label="truth wave")
                continue

            algo_name = spec["algorithm"]
            variant = spec["variant"]
            rec = self.records_by_key.get(record_key(algo_name, variant))
            if rec is None:
                continue

            result = rec["result"]
            color = self.algorithm_specs[algo_name].color

            if show_center:
                ax.axvline(float(result["f1_hat"]), linestyle=":", linewidth=1.8, color=color, label=f"{algo_name} {variant} f1")
                ax.axvline(float(result["f2_hat"]), linestyle=":", linewidth=1.8, color=color, label=f"{algo_name} {variant} f2")

            if show_wave:
                if "best_fit" in result:
                    ax.plot(x, np.asarray(result["best_fit"], dtype=float), linewidth=2.0, color=color, label=f"{algo_name} {variant} wave")
                else:
                    if "gamma_left" in result:
                        gamma = 0.5 * (float(result["gamma_left"]) + float(result["gamma_right"]))
                    else:
                        gamma = float(result["gamma"])

                    y_model = template_dip_from_params(
                        x,
                        float(result["f1_hat"]),
                        float(result["f2_hat"]),
                        gamma,
                        contrast,
                    )
                    ax.plot(x, y_model, linewidth=2.0, color=color, label=f"{algo_name} {variant} wave")

        ax.legend(fontsize=8, loc="best")
        self.canvas.draw_idle()

    def _set_running_ui(self, running: bool) -> None:
        self.btn_generate.setEnabled(not running)
        self.btn_run_selected.setEnabled(not running)
        self.btn_cancel.setEnabled(running)

    def _start_worker(self, jobs: list[dict[str, Any]]) -> None:
        if self.current_x is None or self.current_y_dip is None or self.current_truth is None:
            QMessageBox.information(self, "No trace", "Generate a trace first.")
            return

        if not jobs:
            QMessageBox.information(self, "No rows selected", "Check at least one run row.")
            return

        if self._worker_thread is not None:
            QMessageBox.information(self, "Busy", "A run is already in progress.")
            return

        self._clear_status()
        self.progress.setRange(0, max(1, len(jobs)))
        self.progress.setValue(0)
        self.lbl_progress.setText("Starting...")
        self.records_by_key = {}
        self.all_records = []
        self._rebuild_main_table()
        self._plot_current_trace()
        self._update_error_chart()

        self._worker_thread = QThread()
        self._worker = BenchmarkWorker(
            self.current_x,
            self.current_y_dip,
            self.current_truth,
            jobs,
        )
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.cancelled.connect(self._on_worker_cancelled)
        self._worker.errored.connect(self._on_worker_error)

        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.cancelled.connect(self._worker_thread.quit)
        self._worker.errored.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._on_thread_finished)

        self._set_running_ui(True)
        self._worker_thread.start()

    @Slot(int, int, str)
    def _on_worker_progress(self, done: int, total: int, label: str) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.lbl_progress.setText(f"{done}/{total} | {label}")

    @Slot(list)
    def _on_worker_finished(self, records: list[dict[str, Any]]) -> None:
        self.all_records = records
        self._update_records(records)
        self._rebuild_main_table()
        self._plot_current_trace()
        self._update_error_chart()
        self.lbl_progress.setText("Done")
        self._append_status(f"Completed {len(records)} runs.")

    @Slot(list)
    def _on_worker_cancelled(self, records: list[dict[str, Any]]) -> None:
        self.all_records = records
        self._update_records(records)
        self._rebuild_main_table()
        self._plot_current_trace()
        self._update_error_chart()
        self.lbl_progress.setText("Cancelled")
        self._append_status(f"Cancelled. Partial results kept: {len(records)} runs.")

    @Slot(str)
    def _on_worker_error(self, msg: str) -> None:
        self.lbl_progress.setText("Error")
        QMessageBox.critical(self, "Run error", msg)

    @Slot()
    def _on_thread_finished(self) -> None:
        self._worker = None
        self._worker_thread = None
        self._set_running_ui(False)

    @Slot()
    def _on_generate(self) -> None:
        try:
            used_seed = int(self.sp_seed.value())
            x, y_dip, truth = generate_random_odmr_trace(**self._simulation_kwargs())
        except Exception as exc:
            QMessageBox.critical(self, "Generation error", f"{type(exc).__name__}: {exc}")
            return

        self.current_x = x
        self.current_y_dip = y_dip
        self.current_truth = truth
        self.records_by_key = {}
        self.all_records = []

        self.sp_seed.setValue(min(self.sp_seed.maximum(), used_seed + 1))
        self.ds_template_height.setValue(default_template_height(truth["success_probability_at_resonance"]))

        self.lbl_truth.setText(
            "Truth: "
            f"f1={truth['resonance_value1']:.3f} MHz, "
            f"f2={truth['resonance_value2']:.3f} MHz, "
            f"width={truth['width']:.3f}, "
            f"n_tries={truth['num_tries']} "
            f"(seed={used_seed})"
        )

        self._clear_status()
        self._append_status("Generated trace.")
        self._append_status(
            f"Truth | f1={truth['resonance_value1']:.3f} MHz, "
            f"f2={truth['resonance_value2']:.3f} MHz, width={truth['width']:.3f}, seed={used_seed}"
        )
        self._rebuild_main_table()
        self._plot_current_trace()
        self._update_error_chart()

    @Slot()
    def _on_run_selected(self) -> None:
        self._rebuild_main_table()
        self._start_worker(build_jobs_from_rows(self.row_specs, self.row_run_states))

    @Slot()
    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.lbl_progress.setText("Cancelling...")

    def _find_row_index(self, key: str) -> int | None:
        for idx, spec in enumerate(self.row_specs):
            if row_key(spec) == key:
                return idx
        return None

    def _on_run_checkbox_changed(self, key: str) -> None:
        row = self._find_row_index(key)
        if row is None:
            return
        widget = self.tbl_main.cellWidget(row, 2)
        if isinstance(widget, QCheckBox):
            self.row_run_states[key] = widget.isChecked()

    def _on_center_checkbox_changed(self, key: str) -> None:
        row = self._find_row_index(key)
        if row is None:
            return
        widget = self.tbl_main.cellWidget(row, 3)
        if isinstance(widget, QCheckBox):
            self.row_center_states[key] = widget.isChecked()
        self._plot_current_trace()

    def _on_wave_checkbox_changed(self, key: str) -> None:
        row = self._find_row_index(key)
        if row is None:
            return
        widget = self.tbl_main.cellWidget(row, 4)
        if isinstance(widget, QCheckBox):
            self.row_wave_states[key] = widget.isChecked()
        self._plot_current_trace()

    def closeEvent(self, event) -> None:
        if self._worker is not None:
            self._worker.cancel()

        if self._worker_thread is not None:
            self._worker_thread.quit()
            self._worker_thread.wait(2000)

        super().closeEvent(event)