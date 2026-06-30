"""Unit tests for logging_utils methods.

Tests cover logger generation and setup, and three different logging levels
(info, debug, warning)

"""

import logging
import os
import pytest
from unittest.mock import MagicMock

# Assuming your code is in a file named logging_utils.py
from model2obs.utils.logging_utils import (
    setup_package_logger,
    emit_info,
    emit_debug,
    emit_warning,
    ModuleLogger,
    get_module_logger
)


class TestSetupPackageLogger:
    """Test suite for setup_package_logger() method."""

    def test_setup_package_logger_creates_dir_and_file(self, tmp_path):
        """Verify log directory and file creation under normal operation."""
        log_file_path = tmp_path / "logs" / "model2obs.log"
        config = {"log_file": str(log_file_path)}

        logger, log_file, logs_folder = setup_package_logger(config, parallel=False)

        # Assert paths match and directory was created
        assert log_file == str(log_file_path)
        assert logs_folder == str(tmp_path / "logs")
        assert os.path.exists(logs_folder)

        # Assert logger configuration
        assert logger.name == "model2obs"
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.FileHandler)


    @pytest.mark.parametrize("parallel_flag, expected_format", [
        (True, "%(asctime)s | %(threadName)s | %(message)s"),
        (False, "%(asctime)s | %(message)s")
    ])
    def test_setup_package_logger_formatting(self, tmp_path, parallel_flag, expected_format):
        """Verify format strings toggle based on the parallel parameter."""
        log_file_path = tmp_path / "model2obs.log"
        config = {"log_file": str(log_file_path)}

        logger, _, _ = setup_package_logger(config, parallel=parallel_flag)
        formatter = logger.handlers[0].formatter

        assert formatter._fmt == expected_format


    def test_setup_package_logger_warns_on_overwrite(self, tmp_path):
        """Verify a Python warning triggers if the log file already exists."""
        log_file_path = tmp_path / "model2obs.log"
        log_file_path.write_text("existing content")
        config = {"log_file": str(log_file_path)}

        # Catch Python warnings
        with pytest.warns(UserWarning, match="exists: overwriting it!"):
            setup_package_logger(config, parallel=False)


class TestEmit:
    """Test suite for emitting methods."""

    def test_emit_info(self, capsys, caplog):
        """Verify info outputs to stdout and captures at INFO level."""
        mock_logger = logging.getLogger("test_info_logger")

        with caplog.at_level(logging.INFO):
            emit_info("Hello Info", logger=mock_logger)

        # Check screen stdout
        captured_stdout = capsys.readouterr().out
        assert captured_stdout == "Hello Info\n"

        # Check log telemetry
        assert len(caplog.records) == 1
        assert caplog.records[0].message == "Hello Info"
        assert caplog.records[0].levelname == "INFO"


    def test_emit_debug(self, capsys, caplog):
        """Verify debug outputs ONLY to logs, leaving stdout blank."""
        mock_logger = logging.getLogger("test_debug_logger")

        with caplog.at_level(logging.DEBUG):
            emit_debug("Hello Debug", logger=mock_logger)

        # Screen stdout should be entirely empty
        captured_stdout = capsys.readouterr().out
        assert captured_stdout == ""

        # Check log telemetry
        assert len(caplog.records) == 1
        assert caplog.records[0].message == "Hello Debug"
        assert caplog.records[0].levelname == "DEBUG"


    def test_emit_warning(self, capsys, caplog):
        """Verify warning prepends string to stdout and logs correctly."""
        mock_logger = logging.getLogger("test_warning_logger")

        with caplog.at_level(logging.WARNING):
            emit_warning("Hello Warning", logger=mock_logger)

        # Check stdout formatting
        captured_stdout = capsys.readouterr().out
        assert captured_stdout == "Warning: Hello Warning\n"

        # Check log telemetry
        assert len(caplog.records) == 1
        assert caplog.records[0].message == "Hello Warning"
        assert caplog.records[0].levelname == "WARNING"


class TestModuleLogger:
    """Test suite for ModuleLogger class."""

    def test_module_logger_routes_correctly(self, caplog, capsys):
        """Verify ModuleLogger class interface proxies methods properly."""
        logger_name = "test_module"
        module_logger = get_module_logger(logger_name)

        assert isinstance(module_logger, ModuleLogger)

        with caplog.at_level(logging.DEBUG):
            module_logger.debug("Debug payload")
            module_logger.info("Info payload")
            module_logger.warning("Warning payload")

        # Check that all 3 reached the logger records correctly
        assert len(caplog.records) == 3
        assert caplog.records[0].name == logger_name
        assert caplog.records[0].levelname == "DEBUG"
        assert caplog.records[1].levelname == "INFO"
        assert caplog.records[2].levelname == "WARNING"
