"""Application entry point for the ASL Sign Language System."""

from __future__ import annotations

from src.config import load_config
from src.gui.app import App
from src.utils.logger import setup_logger


def main() -> None:
    """Load config, configure logging, and launch the Tkinter GUI."""
    config = load_config("config.yaml")
    setup_logger(config)
    app = App(config)
    app.mainloop()


if __name__ == "__main__":
    main()
