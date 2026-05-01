from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from odmr.gui.multi_trace_app import MainWindow, apply_modern_dark


def main() -> None:
    app = QApplication(sys.argv)
    apply_modern_dark(app)

    win = MainWindow()
    win.resize(1800, 950)
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()