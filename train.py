"""Command-line training entry point for SignNet."""

from __future__ import annotations

from src.config import load_config
from src.trainer import Trainer
from src.utils.logger import setup_logger


def main() -> None:
    """Run the configured SignNet training pipeline."""
    config = load_config("config.yaml")
    setup_logger(config)
    Trainer(config).train()


if __name__ == "__main__":
    main()
