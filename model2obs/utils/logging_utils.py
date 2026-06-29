"""Shared logging setup utilities for model2obs runtime modules."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

def setup_package_logger(config: Dict[str, Any], parallel: bool) -> Tuple[logging.Logger, str, str]:
    """Configure package logger and return logger, run-log path, and logs folder.

    Args:
        config: Workflow configuration dictionary.
        parallel: Whether workflow execution is parallel.

    Returns:
        Tuple containing:
        - package logger (``model2obs``),
        - resolved run log file path,
        - resolved logs folder path.
    """
    log_file = os.path.expandvars(config["log_file"])
    logs_folder = os.path.dirname(log_file)
    os.makedirs(logs_folder, exist_ok=True)

    package_logger = logging.getLogger("model2obs")
    package_logger.setLevel(logging.DEBUG)
    package_logger.propagate = False
    for handler in package_logger.handlers:
        handler.close()
    package_logger.handlers.clear()

    if parallel:
        fmt = "%(asctime)s | %(threadName)s | %(message)s"
    else:
        fmt = "%(asctime)s | %(message)s"

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            fmt=fmt,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    package_logger.addHandler(file_handler)

    return package_logger, log_file, logs_folder


def emit_info(message: str, logger: Optional[logging.Logger] = None) -> None:
    """Emit high-level progress to screen and log."""
    print(message)
    active_logger = logger or logging.getLogger(__name__)
    active_logger.info(message)


def emit_debug(message: str, logger: Optional[logging.Logger] = None) -> None:
    """Emit detailed diagnostics to log only."""
    active_logger = logger or logging.getLogger(__name__)
    active_logger.debug(message)


def emit_warning(message: str, logger: Optional[logging.Logger] = None) -> None:
    """Emit warning to screen and log."""
    print(f"Warning: {message}")
    active_logger = logger or logging.getLogger(__name__)
    active_logger.warning(message)

class ModuleLogger:
    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    def info(self, message: str) -> None:
        emit_info(message, self._logger)

    def debug(self, message: str) -> None:
        emit_debug(message, self._logger)

    def warning(self, message: str) -> None:
        emit_warning(message, self._logger)

def get_module_logger(name: str) -> ModuleLogger:
    return ModuleLogger(name)
