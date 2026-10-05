"""PyInstaller-friendly entry point for the main desktop application."""

from deepsearch.qt_app import main

if __name__ == "__main__":
    raise SystemExit(main())
