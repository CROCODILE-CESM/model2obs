"""Shared pytest fixtures for model2obs test suite.

This module provides common fixtures used across all test categories.
"""

import os
from pathlib import Path
from typing import Dict, Any

import pytest
import yaml


# ============================================================================
# Path Fixtures
# ============================================================================

@pytest.fixture
def test_root() -> Path:
    """Provide path to the test directory root.
    
    Returns:
        Path object pointing to tests/ directory
    """
    return Path(__file__).parent


@pytest.fixture
def fixtures_root(test_root: Path) -> Path:
    """Provide path to the fixtures directory.
    
    Args:
        test_root: Path to test directory root
        
    Returns:
        Path object pointing to tests/fixtures/
    """
    return test_root / "fixtures"


@pytest.fixture
def config_files_dir(fixtures_root: Path) -> Path:
    """Provide path to configuration files fixtures directory.
    
    Args:
        fixtures_root: Path to fixtures directory
        
    Returns:
        Path object pointing to tests/fixtures/config_files/
    """
    return fixtures_root / "config_files"


@pytest.fixture
def project_root(test_root: Path) -> Path:
    """Provide path to the project root directory.
    
    Args:
        test_root: Path to test directory root
        
    Returns:
        Path object pointing to model2obs/ root directory
    """
    return test_root.parent


# ============================================================================
# Environment Fixtures
# ============================================================================

@pytest.fixture
def mock_env_vars(monkeypatch) -> Dict[str, str]:
    """Provide mock environment variables for testing.
    
    Sets up common environment variables used in configs (e.g., $WORK).
    Automatically cleans up after test completion.
    
    Args:
        monkeypatch: pytest monkeypatch fixture
        
    Returns:
        Dictionary of mock environment variables
        
    Example:
        def test_with_env(mock_env_vars):
            assert os.environ['WORK'] == '/mock/work'
    """
    env_vars = {
        'WORK': '/mock/work',
        'HOME': '/mock/home',
        'USER': 'mockuser'
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    return env_vars


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def minimal_valid_config(tmp_path: Path) -> Dict[str, Any]:
    """Provide a minimal valid configuration dictionary.
    
    This configuration contains only the required keys with valid values.
    Uses tmp_path for all directory paths to ensure isolation.
    
    Args:
        tmp_path: pytest temporary directory fixture
        
    Returns:
        Dictionary with minimal valid configuration
        
    Note:
        This is an in-memory config dict, not a file.
        Use write_config_file() helper to create YAML files.
    """
    return {
        'model_files_folder': str(tmp_path / 'model_files'),
        'obs_seq_in_folder': str(tmp_path / 'obs_seq_in'),
        'output_folder': str(tmp_path / 'output'),
        'template_file': str(tmp_path / 'template.nc'),
        'static_file': str(tmp_path / 'static.nc'),
        'ocean_geometry': str(tmp_path / 'ocean_geometry.nc'),
        'perfect_model_obs_dir': str(tmp_path / 'dart_work'),
        'parquet_folder': str(tmp_path / 'parquet'),
        'time_window': {
            'days': 1,
            'hours': 0,
            'minutes': 0,
            'seconds': 0,
            'weeks': 0,
            'months': 0,
            'years': 0
        }
    }


@pytest.fixture
def write_config_file(tmp_path: Path):
    """Provide a helper function to write configuration YAML files.
    
    This fixture returns a function that writes config dicts to YAML files.
    Useful for testing file loading and path resolution.
    
    Args:
        tmp_path: pytest temporary directory fixture
        
    Returns:
        Function that writes config dict to YAML file
        
    Example:
        def test_config_loading(write_config_file, minimal_valid_config):
            config_path = write_config_file(minimal_valid_config, "test_config.yaml")
            loaded_config = read_config(config_path)
            assert loaded_config is not None
    """
    def _write_config(config_dict: Dict[str, Any], filename: str = "config.yaml") -> Path:
        """Write configuration dictionary to YAML file.
        
        Args:
            config_dict: Configuration dictionary to write
            filename: Name of the output file
            
        Returns:
            Path to the created config file
        """
        config_path = tmp_path / filename
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f)
        return config_path
    
    return _write_config


# ============================================================================
# File Creation Helpers
# ============================================================================

@pytest.fixture
def create_mock_directory_structure(tmp_path: Path):
    """Provide a helper to create mock directory structures with files.
    
    Returns a function that creates directories and populates them with
    mock files. Useful for testing directory validation functions.
    
    Args:
        tmp_path: pytest temporary directory fixture
        
    Returns:
        Function that creates directory structure
        
    Example:
        def test_directory_validation(create_mock_directory_structure):
            model_dir = create_mock_directory_structure(
                "model_files",
                [("model_001.nc", "mock content"), ("model_002.nc", "mock content")]
            )
            assert len(list(model_dir.glob("*.nc"))) == 2
    """
    def _create_structure(dir_name: str, files: list = None) -> Path:
        """Create directory with mock files.
        
        Args:
            dir_name: Name of directory to create
            files: List of (filename, content) tuples to create in directory
            
        Returns:
            Path to created directory
        """
        dir_path = tmp_path / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        
        if files:
            for filename, content in files:
                file_path = dir_path / filename
                file_path.write_text(content)
        
        return dir_path
    
    return _create_structure


# ============================================================================
# Observation Sequence Fixtures
# ============================================================================

@pytest.fixture
def create_obs_seq_file(tmp_path: Path):
    """Provide a helper to create obs_seq.in files with specified observations.
    
    Returns a function that creates valid obs_seq.in files for testing
    using the pydartdiags-compatible format.
    
    Args:
        tmp_path: pytest temporary directory fixture
        
    Returns:
        Function that creates obs_seq.in files
        
    Example:
        def test_obs_trimming(create_obs_seq_file):
            obs_file = create_obs_seq_file(
                "obs_seq.in",
                [(15.0, 45.0, 10.0, 17.32, 12)]  # lon, lat, depth, value, type
            )
            assert obs_file.exists()
    """
    from tests.fixtures.mock_obs_seq_files import create_obs_seq_in, degrees_to_radians
    
    def _create_file(filename: str, observations: list) -> Path:
        """Create obs_seq.in file with observations.
        
        Args:
            filename: Name for the obs_seq.in file
            observations: List of tuples (lon_deg, lat_deg, depth_m, value, obs_type)
                         Longitude and latitude should be in degrees.
                         
        Returns:
            Path to created obs_seq.in file
        """
        file_path = tmp_path / filename
        
        obs_in_radians = [
            (degrees_to_radians(lon), degrees_to_radians(lat), depth, value, obs_type)
            for lon, lat, depth, value, obs_type in observations
        ]
        
        create_obs_seq_in(file_path, obs_in_radians)
        return file_path
    
    return _create_file


# ============================================================================
# Test Markers
# ============================================================================

def pytest_configure(config):
    """Register custom markers for test categorization.
    
    Markers allow selective test execution (e.g., pytest -m "not slow").
    """
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "mocking: marks tests focused on external dependency mocking"
    )
