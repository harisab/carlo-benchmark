from __future__ import annotations

import time
from typing import Any

import numpy as np

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
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

from odmr.algorithms.double_correlation import run_double_correlation
from odmr.algorithms.lmfit_double import run_lmfit_double_joint
from odmr.algorithms.lmfit_single_side import run_lmfit_single_side
from odmr.algorithms.single_correlation import run_single_correlation
from odmr.project_defaults import (
    APP_DEFAULTS,
    BENCHMARK_CASES,
    BENCHMARK_DEFAULTS,
    SIMULATION_DEFAULTS,
)
from odmr.simulation import generate_random_odmr_trace


PLOT_COLORS = (
    "magenta",
    "orange",
    "gold",
    "cyan",
    "tab:blue",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
)


def apply_modern_dark(app) -> None:
    app.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(20, 20, 24))
    pal.setColor(QPalette.WindowText, QColor(230, 230, 230))
    pal.setColor(QPalette.Base, QColor(14, 14, 18))
    pal.setColor(QPalette.AlternateBase, QColor(24, 24, 28))
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
        QSpinBox, QComboBox, QTableWidget {
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
        fig = Figure(figsize=(8.0, 5.0), dpi=100)
        self.ax = fig.add_subplot(111)
        if title:
            self.ax.set_title(title)
        super().__init__(fig)


def _table_item(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item


def _case_key(case: dict[str, Any]) -> str:
    return f"{case['algorithm']}:{case['variant']}"


def _plot_color_for_index(index: int) -> str:
    return PLOT_COLORS[index % len(PLOT_COLORS)]


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")

    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def _run_case(case: dict[str, Any], x: np.ndarray, y_dip: np.ndarray, settings: dict[str, Any]) -> dict:
    algorithm = case["algorithm"]

    if algorithm == "LMFitSinglePerSide":
        return run_lmfit_single_side(x, y_dip, settings=settings)

    if algorithm == "LMFitDoubleJoint":
        return run_lmfit_double_joint(x, y_dip, settings=settings)

    if algorithm == "SingleCorrelation":
        return run_single_correlation(x, y_dip, settings=settings)

    if algorithm == "DoubleCorrelation":
        return run_double_correlation(x, y_dip, settings=settings)

    raise ValueError(f"Unsupported algorithm: {algorithm}")


class MultiTraceWorker(QObject):
    progress = Signal(int, int, str)
    stats_updated = Signal(object)
    finished = Signal(object)
    cancelled = Signal(object)
    errored = Signal(str)

    def __init__(
        self,
        *,
        cases: list[dict[str, Any]],
        num_traces: int,
        start_seed: int,
        template_height_success_prob: bool,
        require_one_peak_per_side: bool,
    ) -> None:
        super().__init__()
        self.cases = cases
        self.num_traces = int(num_traces)
        self.start_seed = int(start_seed)
        self.template_height_success_prob = bool(template_height_success_prob)
        self.require_one_peak_per_side = bool(require_one_peak_per_side)
        self._cancel_requested = False

    @Slot()
    def cancel(self) -> None:
        self._cancel_requested = True

    def _settings_for_case(self, case: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
        if self.template_height_success_prob:
            template_height = float(truth["success_probability_at_resonance"])
        else:
            template_height = float(BENCHMARK_DEFAULTS["template_height"])

        settings = dict(BENCHMARK_DEFAULTS)
        settings.update(
            {
                "template_height": template_height,
                "require_one_peak_per_side": self.require_one_peak_per_side,
                "normalization_mode": case["normalization_mode"] or "raw",
                "width_mode": case["width_mode"] or "fixed",
                "benchmark_variant": case["variant"],
            }
        )
        return settings

    def _snapshot(
        self,
        *,
        trace_done: int,
        errors_by_key: dict[str, list[float]],
        runtimes_by_key: dict[str, list[float]],
    ) -> dict[str, Any]:
        records = []

        for case in self.cases:
            key = _case_key(case)
            errors = errors_by_key.get(key, [])
            runtimes = runtimes_by_key.get(key, [])

            if errors:
                mean_err, std_err = _mean_std(errors)
                median_err = float(np.median(errors))
                best_err = float(np.min(errors))
                worst_err = float(np.max(errors))
            else:
                mean_err = std_err = median_err = best_err = worst_err = float("nan")

            avg_runtime = float(np.mean(runtimes)) if runtimes else float("nan")

            records.append(
                {
                    "key": key,
                    "algorithm": case["algorithm"],
                    "variant": case["variant"],
                    "n": len(errors),
                    "mean_err": mean_err,
                    "std_err": std_err,
                    "median_err": median_err,
                    "best_err": best_err,
                    "worst_err": worst_err,
                    "avg_runtime_ms": avg_runtime,
                }
            )

        return {
            "trace_done": trace_done,
            "records": records,
        }

    @Slot()
    def run(self) -> None:
        try:
            total_jobs = max(1, self.num_traces * len(self.cases))
            completed_jobs = 0

            errors_by_key: dict[str, list[float]] = {_case_key(case): [] for case in self.cases}
            runtimes_by_key: dict[str, list[float]] = {_case_key(case): [] for case in self.cases}

            for trace_idx in range(self.num_traces):
                if self._cancel_requested:
                    snapshot = self._snapshot(
                        trace_done=trace_idx,
                        errors_by_key=errors_by_key,
                        runtimes_by_key=runtimes_by_key,
                    )
                    self.cancelled.emit(snapshot)
                    return

                seed = self.start_seed + trace_idx
                x, y_dip, truth = generate_random_odmr_trace(seed=seed)

                for case in self.cases:
                    if self._cancel_requested:
                        snapshot = self._snapshot(
                            trace_done=trace_idx,
                            errors_by_key=errors_by_key,
                            runtimes_by_key=runtimes_by_key,
                        )
                        self.cancelled.emit(snapshot)
                        return

                    key = _case_key(case)
                    settings = self._settings_for_case(case, truth)

                    t0 = time.perf_counter()
                    result = _run_case(case, x, y_dip, settings)
                    runtime_ms = 1000.0 * (time.perf_counter() - t0)

                    e1 = abs(float(result["f1_hat"]) - float(truth["resonance_value1"]))
                    e2 = abs(float(result["f2_hat"]) - float(truth["resonance_value2"]))
                    mean_err = 0.5 * (e1 + e2)

                    errors_by_key[key].append(float(mean_err))
                    runtimes_by_key[key].append(float(runtime_ms))

                    completed_jobs += 1
                    self.progress.emit(
                        completed_jobs,
                        total_jobs,
                        f"trace {trace_idx + 1}/{self.num_traces} | {case['algorithm']} | {case['variant']}",
                    )

                snapshot = self._snapshot(
                    trace_done=trace_idx + 1,
                    errors_by_key=errors_by_key,
                    runtimes_by_key=runtimes_by_key,
                )
                self.stats_updated.emit(snapshot)

            snapshot = self._snapshot(
                trace_done=self.num_traces,
                errors_by_key=errors_by_key,
                runtimes_by_key=runtimes_by_key,
            )
            self.finished.emit(snapshot)

        except Exception as exc:
            self.errored.emit(f"{type(exc).__name__}: {exc}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("ODMR Multi-Trace Benchmark GUI")

        self.row_specs: list[dict[str, Any]] = []
        self.row_run_states: dict[str, bool] = {}
        self.stats_by_key: dict[str, dict[str, Any]] = {}
        self.history_by_key: dict[str, list[tuple[int, float]]] = {}

        self._worker_thread: QThread | None = None
        self._worker: MultiTraceWorker | None = None

        self._build_ui()
        self._build_case_table()

    def _build_ui(self) -> None:
        top = QWidget()
        top_layout = QHBoxLayout(top)

        top_layout.addWidget(self._build_controls_group())
        top_layout.addWidget(self._build_actions_group())
        top_layout.addWidget(self._build_case_group(), stretch=1)

        self.canvas_bar = MplCanvas("Mean error by benchmark case")
        self.canvas_line = MplCanvas("Running mean error")

        graph_splitter = QSplitter(Qt.Horizontal)
        graph_splitter.addWidget(self.canvas_bar)
        graph_splitter.addWidget(self.canvas_line)
        graph_splitter.setSizes([900, 900])

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.addWidget(top, stretch=0)
        root_layout.addWidget(graph_splitter, stretch=1)

        self.setCentralWidget(root)

    def _build_controls_group(self) -> QGroupBox:
        g = QGroupBox("Benchmark run")
        form = QFormLayout(g)

        self.sp_num_traces = QSpinBox()
        self.sp_num_traces.setRange(1, 100000)
        self.sp_num_traces.setValue(100)

        self.sp_start_seed = QSpinBox()
        self.sp_start_seed.setRange(0, 2_000_000_000)
        self.sp_start_seed.setValue(int(SIMULATION_DEFAULTS["seed"]))

        self.ck_template_height_success_prob = QCheckBox()
        self.ck_template_height_success_prob.setChecked(
            bool(BENCHMARK_DEFAULTS["use_success_probability_for_template_height"])
        )

        self.cmb_require_side = QComboBox()
        self.cmb_require_side.addItems(["true", "false"])
        self.cmb_require_side.setCurrentText(
            "true" if bool(BENCHMARK_DEFAULTS["require_one_peak_per_side"]) else "false"
        )

        form.addRow("num_traces", self.sp_num_traces)
        form.addRow("start_seed", self.sp_start_seed)
        form.addRow("template_height = success_prob", self.ck_template_height_success_prob)
        form.addRow("require_one_peak_per_side", self.cmb_require_side)

        return g

    def _build_actions_group(self) -> QGroupBox:
        g = QGroupBox("Actions")
        layout = QVBoxLayout(g)

        self.btn_run = QPushButton("Run checked cases")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)

        self.lbl_status = QLabel("Idle")

        layout.addWidget(self.btn_run)
        layout.addWidget(self.btn_cancel)
        layout.addWidget(self.progress)
        layout.addWidget(self.lbl_status)

        self.btn_run.clicked.connect(self._on_run)
        self.btn_cancel.clicked.connect(self._on_cancel)

        return g

    def _build_case_group(self) -> QGroupBox:
        g = QGroupBox("Benchmark cases")
        layout = QVBoxLayout(g)

        self.tbl_cases = QTableWidget(0, 10)
        self.tbl_cases.setHorizontalHeaderLabels(
            [
                "Run",
                "Algorithm",
                "Variant",
                "n",
                "mean_err",
                "std",
                "median",
                "best",
                "worst",
                "avg_ms",
            ]
        )
        self.tbl_cases.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_cases.verticalHeader().setVisible(False)
        self.tbl_cases.setMinimumWidth(1200)
        self.tbl_cases.setMaximumHeight(520)

        layout.addWidget(self.tbl_cases)
        return g

    def _build_case_table(self) -> None:
        self.row_specs = []

        for case in BENCHMARK_CASES:
            key = _case_key(case)
            self.row_specs.append(
                {
                    "key": key,
                    "case": case,
                }
            )

        self.tbl_cases.setRowCount(len(self.row_specs))

        for row, spec in enumerate(self.row_specs):
            key = spec["key"]
            case = spec["case"]

            self.row_run_states[key] = True

            ck = QCheckBox()
            ck.setChecked(True)
            ck.stateChanged.connect(lambda _state, k=key: self._on_run_checkbox_changed(k))
            self.tbl_cases.setCellWidget(row, 0, ck)

            self.tbl_cases.setItem(row, 1, _table_item(str(case["algorithm"])))
            self.tbl_cases.setItem(row, 2, _table_item(str(case["variant"])))

            for col in range(3, 10):
                self.tbl_cases.setItem(row, col, _table_item(""))

    def _checked_cases(self) -> list[dict[str, Any]]:
        cases: list[dict[str, Any]] = []

        for spec in self.row_specs:
            key = spec["key"]
            if self.row_run_states.get(key, False):
                cases.append(spec["case"])

        return cases

    def _on_run_checkbox_changed(self, key: str) -> None:
        row = self._find_row_index(key)
        if row is None:
            return

        widget = self.tbl_cases.cellWidget(row, 0)
        if isinstance(widget, QCheckBox):
            self.row_run_states[key] = widget.isChecked()

    def _find_row_index(self, key: str) -> int | None:
        for i, spec in enumerate(self.row_specs):
            if spec["key"] == key:
                return i
        return None

    def _set_running_ui(self, running: bool) -> None:
        self.btn_run.setEnabled(not running)
        self.btn_cancel.setEnabled(running)

    def _on_run(self) -> None:
        cases = self._checked_cases()

        if not cases:
            QMessageBox.information(self, "No cases selected", "Check at least one benchmark case.")
            return

        if self._worker_thread is not None:
            QMessageBox.information(self, "Busy", "A benchmark run is already in progress.")
            return

        self.stats_by_key = {}
        self.history_by_key = {}

        for row in range(self.tbl_cases.rowCount()):
            for col in range(3, 10):
                self.tbl_cases.setItem(row, col, _table_item(""))

        self._update_plots()

        num_traces = int(self.sp_num_traces.value())
        start_seed = int(self.sp_start_seed.value())

        self.progress.setRange(0, max(1, num_traces * len(cases)))
        self.progress.setValue(0)
        self.lbl_status.setText("Starting...")

        self._worker_thread = QThread()
        self._worker = MultiTraceWorker(
            cases=cases,
            num_traces=num_traces,
            start_seed=start_seed,
            template_height_success_prob=self.ck_template_height_success_prob.isChecked(),
            require_one_peak_per_side=(self.cmb_require_side.currentText() == "true"),
        )
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.stats_updated.connect(self._on_stats_updated)
        self._worker.finished.connect(self._on_finished)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.errored.connect(self._on_error)

        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.cancelled.connect(self._worker_thread.quit)
        self._worker.errored.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._on_thread_finished)

        self._set_running_ui(True)
        self._worker_thread.start()

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.lbl_status.setText("Cancelling...")

    @Slot(int, int, str)
    def _on_worker_progress(self, done: int, total: int, label: str) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.lbl_status.setText(f"{done}/{total} | {label}")

    @Slot(object)
    def _on_stats_updated(self, snapshot: dict[str, Any]) -> None:
        self._apply_snapshot(snapshot)

    @Slot(object)
    def _on_finished(self, snapshot: dict[str, Any]) -> None:
        self._apply_snapshot(snapshot)
        self.lbl_status.setText(f"Done | traces={snapshot['trace_done']}")

    @Slot(object)
    def _on_cancelled(self, snapshot: dict[str, Any]) -> None:
        self._apply_snapshot(snapshot)
        self.lbl_status.setText(f"Cancelled | traces={snapshot['trace_done']}")

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self.lbl_status.setText("Error")
        QMessageBox.critical(self, "Benchmark error", msg)

    @Slot()
    def _on_thread_finished(self) -> None:
        self._worker = None
        self._worker_thread = None
        self._set_running_ui(False)

    def _apply_snapshot(self, snapshot: dict[str, Any]) -> None:
        trace_done = int(snapshot["trace_done"])

        for record in snapshot["records"]:
            key = record["key"]
            self.stats_by_key[key] = record

            if record["n"] > 0 and np.isfinite(record["mean_err"]):
                self.history_by_key.setdefault(key, []).append(
                    (trace_done, float(record["mean_err"]))
                )

        self._update_table()
        self._update_plots()

    def _update_table(self) -> None:
        for row, spec in enumerate(self.row_specs):
            key = spec["key"]
            rec = self.stats_by_key.get(key)

            if rec is None:
                continue

            values = [
                str(rec["n"]),
                f"{rec['mean_err']:.3f}",
                f"{rec['std_err']:.3f}",
                f"{rec['median_err']:.3f}",
                f"{rec['best_err']:.3f}",
                f"{rec['worst_err']:.3f}",
                f"{rec['avg_runtime_ms']:.2f}",
            ]

            for offset, value in enumerate(values):
                self.tbl_cases.setItem(row, 3 + offset, _table_item(value))

    def _update_plots(self) -> None:
        self._update_bar_chart()
        self._update_line_chart()

    def _update_bar_chart(self) -> None:
        ax = self.canvas_bar.ax
        ax.clear()
        ax.set_title("Mean error by benchmark case")
        ax.set_xlabel("Mean error (MHz)")
        ax.set_ylabel("Case")

        records = [
            self.stats_by_key[spec["key"]]
            for spec in self.row_specs
            if spec["key"] in self.stats_by_key and self.stats_by_key[spec["key"]]["n"] > 0
        ]

        if not records:
            self.canvas_bar.draw_idle()
            return

        records = sorted(records, key=lambda r: r["mean_err"])

        labels = [f"{r['algorithm']} | {r['variant']}" for r in records]
        values = [r["mean_err"] for r in records]
        colors = [_plot_color_for_index(i) for i in range(len(records))]

        y = np.arange(len(values))
        ax.barh(y, values, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()

        self.canvas_bar.figure.tight_layout()
        self.canvas_bar.draw_idle()

    def _update_line_chart(self) -> None:
        ax = self.canvas_line.ax
        ax.clear()
        ax.set_title("Running mean error: current top cases")
        ax.set_xlabel("Traces completed")
        ax.set_ylabel("Mean error (MHz)")

        records = [
            rec
            for rec in self.stats_by_key.values()
            if rec["n"] > 0 and np.isfinite(rec["mean_err"])
        ]

        if not records:
            self.canvas_line.draw_idle()
            return

        top_records = sorted(records, key=lambda r: r["mean_err"])[:6]

        for i, rec in enumerate(top_records):
            key = rec["key"]
            history = self.history_by_key.get(key, [])
            if not history:
                continue

            xs = [p[0] for p in history]
            ys = [p[1] for p in history]

            ax.plot(
                xs,
                ys,
                label=f"{rec['algorithm']} | {rec['variant']}",
                color=_plot_color_for_index(i),
            )

        ax.legend(fontsize=8, loc="best")

        self.canvas_line.figure.tight_layout()
        self.canvas_line.draw_idle()

    def closeEvent(self, event) -> None:
        if self._worker is not None:
            self._worker.cancel()

        if self._worker_thread is not None:
            self._worker_thread.quit()
            self._worker_thread.wait(2000)

        super().closeEvent(event)