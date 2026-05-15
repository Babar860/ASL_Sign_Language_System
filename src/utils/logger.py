"""Logging configuration for the Sign Language Avatar System.

Provides a single ``setup_logger`` function that configures the Python root
logger according to the ``logging`` section of the application config dict.
"""

import logging
import sys
from typing import Any


def setup_logger(config: dict[str, Any]) -> logging.Logger:
    """Configure the Python root logger and return it.

    Reads the ``logging.log_level`` key from *config* (default ``"INFO"``).
    Attaches a ``StreamHandler`` that writes to *stdout* with a timestamped
    format if no handlers are already attached to the root logger, preventing
    duplicate log lines when the function is called more than once.

    Args:
        config: The application configuration dict as returned by
            ``src.config.load_config``. Only the ``logging`` sub-dict is
            inspected; all other keys are ignored.

    Returns:
        The configured root :class:`logging.Logger` instance.

    Example::

        from src.config import load_config
        from src.utils.logger import setup_logger

        cfg = load_config("config.yaml")
        logger = setup_logger(cfg)
        logger.info("Application started.")
    """
    log_cfg: dict[str, Any] = config["logging"]
    level_name: str = str(log_cfg["log_level"]).upper()
    level: int = getattr(logging, level_name, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Only add a handler if none exist to avoid duplicate output.
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    return root_logger
