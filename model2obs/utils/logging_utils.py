"""Shared logging setup utilities for model2obs runtime modules."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple


def resolve_run_log_path(config: Dict[str, Any]) -> str:
    """Resolve run log path from configuration with a logs-folder default."""
    output_folder = config["output_folder"]
    logging_cfg_raw = config.get("logging")
    logging_cfg = logging_cfg_raw if isinstance(logging_cfg_raw, dict) else {}

    run_log_file = logging_cfg.get("run_log_file")
    if run_log_file:
        if os.path.isabs(run_log_file):
            return run_log_file
        return os.path.join(output_folder, run_log_file)

    default_logs_folder = os.path.join(output_folder, "logs")
    return os.path.join(default_logs_folder, "model2obs.log")

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
    run_log_path = resolve_run_log_path(config)
    logs_folder = os.path.dirname(run_log_path)
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

    file_handler = logging.FileHandler(run_log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            fmt=fmt,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    package_logger.addHandler(file_handler)

    return package_logger, run_log_path, logs_folder


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
