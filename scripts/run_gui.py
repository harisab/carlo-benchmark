from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from odmr.gui import MainWindow, apply_modern_dark


def main() -> None:
    app = QApplication(sys.argv)
    apply_modern_dark(app)

    win = MainWindow()
    win.resize(1950, 900)
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()