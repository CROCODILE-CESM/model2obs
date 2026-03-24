"""Unit tests for model2obs.utils.config module.

This module tests configuration file loading, validation, and processing.
Configuration correctness is CRITICAL as invalid configs cause workflow failures.

Tests are organized by function (loading, validation, path resolution, etc.).
Coverage Target: >95% (CRITICAL module)
"""

import os
from pathlib import Path
from datetime import timedelta

import pytest
import yaml

from model2obs.utils import config


class TestReadConfig:
    """Test suite for read_config() function.
    
    Tests cover:
    - Valid YAML loading
    - Invalid YAML syntax handling
    - Missing file handling
    - Environment variable expansion
    """
    
    def test_read_config_valid_minimal(self, config_files_dir: Path):
        """Test read_config loads a minimal valid configuration file.
        
        Given: A YAML file with all required keys and valid values
        When: read_config() is called with the file path
        Then: Configuration dictionary is returned with all keys present
        """
        config_file = config_files_dir / "config_valid_minimal.yaml"
        
        result = config.read_config(str(config_file))
        
        assert result is not None
        assert isinstance(result, dict)

        fields = [
            "model_files_folder",
            "obs_seq_in_folder",
            "output_folder",
            "template_file",
            "static_file",
            "ocean_geometry",
            "perfect_model_obs_dir",
            "parquet_folder"
        ]
        for f in fields:
            assert f in result
        assert isinstance(result['time_window'], dict)
        assert 'days' in result['time_window']
        assert 'seconds' in result['time_window']
    
    def test_read_config_missing_file(self, tmp_path: Path):
        """Test read_config raises FileNotFoundError when config file does not exist.
        
        Given: A non-existent config file path
        When: read_config() is called
        Then: FileNotFoundError is raised with descriptive message
        """
        nonexistent_file = tmp_path / "does_not_exist.yaml"
        
        with pytest.raises(FileNotFoundError, match="does not exist"):
            config.read_config(str(nonexistent_file))
    
    def test_read_config_invalid_yaml_syntax(self, config_files_dir: Path):
        """Test read_config raises ValueError when YAML syntax is invalid.
        
        Given: A YAML file with invalid syntax (missing colon)
        When: read_config() is called
        Then: ValueError is raised with descriptive message about YAML parsing
        """
        invalid_config = config_files_dir / "config_invalid_syntax.yaml"
        
        with pytest.raises(ValueError, match="Error parsing YAML"):
            config.read_config(str(invalid_config))
    
    def test_read_config_resolves_paths(self, config_files_dir: Path):
        """Test read_config resolves all paths to absolute paths.
        
        Given: A valid config file
        When: read_config() is called
        Then: All path values are converted to absolute paths
        """
        config_file = config_files_dir / "config_valid_minimal.yaml"
        
        result = config.read_config(str(config_file))
        
        fields = [
            "model_files_folder",
            "obs_seq_in_folder",
            "output_folder",
            "template_file",
            "static_file",
            "ocean_geometry",
            "perfect_model_obs_dir",
            "parquet_folder"
        ]

        for f in fields:
            assert os.path.isabs(result[f])
    
    def test_read_config_expands_environment_variables(
        self, config_files_dir: Path, mock_env_vars: dict
    ):
        """Test read_config expands environment variables in paths.
        
        Given: A config file with $WORK and $HOME environment variables
        When: read_config() is called
        Then: Environment variables are expanded to their values
        
        Note: mock_env_vars fixture sets $WORK='/mock/work' and $HOME='/mock/home'
        """
        config_file = config_files_dir / "config_env_vars.yaml"
        
        result = config.read_config(str(config_file))
        
        assert '/mock/work' in result['model_files_folder']
        assert '/mock/home' in result['template_file']
    
    def test_read_config_converts_time_window(self, config_files_dir: Path):
        """Test read_config converts time_window to days and seconds format.
        
        Given: A config file with time_window in various units
        When: read_config() is called
        Then: time_window is converted to {days: int, seconds: int} format
        """
        config_file = config_files_dir / "config_valid_minimal.yaml"
        
        result = config.read_config(str(config_file))
        
        assert 'time_window' in result
        assert 'days' in result['time_window']
        assert 'seconds' in result['time_window']
        assert isinstance(result['time_window']['days'], int)
        assert isinstance(result['time_window']['seconds'], int)


class TestResolvePath:
    """Test suite for resolve_path() function.
    
    Tests cover:
    - Absolute path handling
    - Relative path resolution
    - Environment variable expansion
    - Path normalization
    """
    
    def test_resolve_path_absolute(self):
        """Test resolve_path returns absolute path unchanged.
        
        Given: An absolute path
        When: resolve_path() is called
        Then: The same absolute path is returned (normalized)
        """
        absolute_path = "/tmp/test/model_files"
        
        result = config.resolve_path(absolute_path, relative_to=None)
        
        assert os.path.isabs(result)
        assert result == os.path.normpath(absolute_path)
    
    def test_resolve_path_relative(self, tmp_path: Path):
        """Test resolve_path resolves relative path relative to reference file.
        
        Given: A relative path and a reference file path
        When: resolve_path() is called with relative_to parameter
        Then: Path is resolved relative to reference file's directory
        """
        relative_path = "model_files"
        reference_file = tmp_path / "config.yaml"
        reference_file.touch()  # Create reference file
        
        result = config.resolve_path(relative_path, relative_to=str(reference_file))
        
        assert os.path.isabs(result)
        expected = os.path.normpath(tmp_path / relative_path)
        assert result == expected
    
    def test_resolve_path_expands_env_vars(self, mock_env_vars: dict):
        """Test resolve_path expands environment variables.
        
        Given: A path with environment variable ($WORK)
        When: resolve_path() is called
        Then: Environment variable is expanded
        
        Note: mock_env_vars sets $WORK='/mock/work'
        """
        path_with_env = "$WORK/model_files"
        
        result = config.resolve_path(path_with_env, relative_to=None)
        
        assert '$WORK' not in result
        assert '/mock/work' in result
    
    def test_resolve_path_relative_without_reference(self):
        """Test resolve_path with relative path but no reference (uses CWD).
        
        Given: A relative path and relative_to=None
        When: resolve_path() is called
        Then: Path is resolved relative to current working directory
        """
        relative_path = "model_files"
        
        result = config.resolve_path(relative_path, relative_to=None)
        
        assert os.path.isabs(result)
        assert result.endswith("model_files")


class TestConvertTimeWindow:
    """Test suite for convert_time_window() function.
    
    Tests cover:
    - Valid time window conversion
    - Edge cases (zero values, large values)
    - Conversion to days+seconds format
    """
    
    @pytest.mark.parametrize("time_window_input,expected_days,expected_seconds", [
        # Basic cases
        ({"days": 1, "hours": 0, "minutes": 0, "seconds": 0, "weeks": 0, "months": 0, "years": 0}, 1, 0),
        ({"days": 0, "hours": 24, "minutes": 0, "seconds": 0, "weeks": 0, "months": 0, "years": 0}, 1, 0),
        ({"days": 0, "hours": 12, "minutes": 0, "seconds": 0, "weeks": 0, "months": 0, "years": 0}, 0, 43200),
        # Complex cases
        ({"days": 1, "hours": 12, "minutes": 30, "seconds": 45, "weeks": 0, "months": 0, "years": 0}, 1, 45045),
        ({"days": 0, "hours": 0, "minutes": 0, "seconds": 0, "weeks": 1, "months": 0, "years": 0}, 7, 0),
        # Zero case
        ({"days": 0, "hours": 0, "minutes": 0, "seconds": 0, "weeks": 0, "months": 0, "years": 0}, 0, 0),
    ])
    def test_convert_time_window_valid(
        self, time_window_input: dict, expected_days: int, expected_seconds: int
    ):
        """Test convert_time_window with various valid inputs.
        
        Given: A config with time_window in various units
        When: convert_time_window() is called
        Then: time_window is converted to correct days and seconds
        """
        test_config = {"time_window": time_window_input}
        
        result = config.convert_time_window(test_config)
        
        assert 'time_window' in result
        assert result['time_window']['days'] == expected_days
        assert result['time_window']['seconds'] == expected_seconds
    
    def test_convert_time_window_missing_raises_error(self):
        """Test convert_time_window raises KeyError when time_window is missing.
        
        Given: A config without time_window key
        When: convert_time_window() is called
        Then: KeyError is raised with descriptive message
        """
        test_config = {}
        
        with pytest.raises(KeyError, match="No time window has been specified"):
            config.convert_time_window(test_config)
    
    def test_convert_time_window_partial_units(self):
        """Test convert_time_window handles missing unit keys by defaulting to 0.
        
        Given: A time_window dict with only some units specified
        When: convert_time_window() is called
        Then: Missing units are treated as 0
        """
        test_config = {
            "time_window": {
                "days": 5,
                # hours, minutes, seconds, weeks, months, years omitted
            }
        }
        
        result = config.convert_time_window(test_config)
        
        assert result['time_window']['days'] == 5
        assert result['time_window']['seconds'] == 0


class TestValidateConfigKeys:
    """Test suite for validate_config_keys() function.
    
    Tests cover:
    - Valid configuration passes validation
    - Missing required keys raises error
    """
    
    def test_validate_config_keys_valid(self, minimal_valid_config: dict):
        """Test validate_config_keys passes with all required keys present.
        
        Given: A config dict with all required keys
        When: validate_config_keys() is called
        Then: No exception is raised
        """
        required_keys = [
            'model_files_folder', 'obs_seq_in_folder', 'output_folder',
            'template_file', 'static_file', 'ocean_geometry',
            'perfect_model_obs_dir', 'parquet_folder'
        ]
        
        config.validate_config_keys(minimal_valid_config, required_keys)
    
    def test_validate_config_keys_missing_keys(self):
        """Test validate_config_keys raises KeyError when required keys are missing.
        
        Given: A config dict missing some required keys
        When: validate_config_keys() is called
        Then: KeyError is raised listing the missing keys
        """
        incomplete_config = {
            'model_files_folder': '/tmp/model',
            # Missing other required keys
        }
        required_keys = ['model_files_folder', 'obs_seq_in_folder', 'output_folder']
        
        with pytest.raises(KeyError, match="Required keys missing"):
            config.validate_config_keys(incomplete_config, required_keys)


class TestCheckDirectoryNotEmpty:
    """Test suite for check_directory_not_empty() function.
    
    Tests cover:
    - Valid non-empty directory passes
    - Non-existent directory raises error
    - Empty directory raises error
    - File path (not directory) raises error
    """
    
    def test_check_directory_not_empty_valid(self, create_mock_directory_structure):
        """Test check_directory_not_empty passes with non-empty directory.
        
        Given: A directory containing at least one file
        When: check_directory_not_empty() is called
        Then: No exception is raised
        """
        dir_path = create_mock_directory_structure(
            "model_files",
            [("model_001.nc", "mock content")]
        )
        
        config.check_directory_not_empty(str(dir_path), "test_directory")
    
    def test_check_directory_not_empty_missing_directory(self, tmp_path: Path):
        """Test check_directory_not_empty raises error for non-existent directory.
        
        Given: A path to a non-existent directory
        When: check_directory_not_empty() is called
        Then: NotADirectoryError is raised
        """
        nonexistent_dir = tmp_path / "does_not_exist"
        
        with pytest.raises(NotADirectoryError, match="does not exist"):
            config.check_directory_not_empty(str(nonexistent_dir), "test_directory")
    
    def test_check_directory_not_empty_empty_directory(self, tmp_path: Path):
        """Test check_directory_not_empty raises error for empty directory.
        
        Given: An empty directory
        When: check_directory_not_empty() is called
        Then: ValueError is raised with message about empty directory
        """
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        
        with pytest.raises(ValueError, match="is empty"):
            config.check_directory_not_empty(str(empty_dir), "test_directory")
    
    def test_check_directory_not_empty_file_not_directory(self, tmp_path: Path):
        """Test check_directory_not_empty raises error when path is a file.
        
        Given: A path pointing to a file (not a directory)
        When: check_directory_not_empty() is called
        Then: NotADirectoryError is raised
        """
        file_path = tmp_path / "file.txt"
        file_path.write_text("content")
        
        with pytest.raises(NotADirectoryError, match="not a directory"):
            config.check_directory_not_empty(str(file_path), "test_file")


class TestCheckNcFilesOnly:
    """Test suite for check_nc_files_only() function.
    
    Tests cover:
    - Directory with only .nc files passes
    - Directory with mixed file types raises error
    - Empty directory raises error
    """
    
    def test_check_nc_files_only_valid(self, create_mock_directory_structure):
        """Test check_nc_files_only passes when directory contains only .nc files.
        
        Given: A directory containing only .nc files
        When: check_nc_files_only() is called
        Then: No exception is raised
        """
        dir_path = create_mock_directory_structure(
            "model_files",
            [("model_001.nc", "content"), ("model_002.nc", "content")]
        )
        
        config.check_nc_files_only(str(dir_path), "test_directory")
    
    def test_check_nc_files_only_mixed_files(self, create_mock_directory_structure):
        """Test check_nc_files_only raises error when non-.nc files are present.
        
        Given: A directory containing .nc and non-.nc files
        When: check_nc_files_only() is called
        Then: ValueError is raised listing non-.nc files
        """
        dir_path = create_mock_directory_structure(
            "model_files",
            [("model_001.nc", "content"), ("readme.txt", "content")]
        )
        
        with pytest.raises(ValueError, match="non-.nc files"):
            config.check_nc_files_only(str(dir_path), "test_directory")
    
    def test_check_nc_files_only_empty_directory(self, tmp_path: Path):
        """Test check_nc_files_only raises error for empty directory.
        
        Given: An empty directory
        When: check_nc_files_only() is called
        Then: ValueError is raised indicating no files
        """
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        
        with pytest.raises(ValueError, match="does not contain any files"):
            config.check_nc_files_only(str(empty_dir), "test_directory")


class TestCheckNcFile:
    """Test suite for check_nc_file() function.
    
    Tests cover:
    - Valid .nc file passes
    - Non-existent file raises error
    - File without .nc extension raises error
    """
    
    def test_check_nc_file_valid(self, tmp_path: Path):
        """Test check_nc_file passes for valid .nc file.
        
        Given: A path to an existing .nc file
        When: check_nc_file() is called
        Then: No exception is raised
        """
        nc_file = tmp_path / "model.nc"
        nc_file.write_text("mock netcdf content")
        
        config.check_nc_file(str(nc_file), "test_file")
    
    def test_check_nc_file_missing(self, tmp_path: Path):
        """Test check_nc_file raises error for non-existent file.
        
        Given: A path to a non-existent file
        When: check_nc_file() is called
        Then: FileNotFoundError is raised
        """
        nonexistent_file = tmp_path / "missing.nc"
        
        with pytest.raises(FileNotFoundError, match="does not exist"):
            config.check_nc_file(str(nonexistent_file), "test_file")
    
    def test_check_nc_file_wrong_extension(self, tmp_path: Path):
        """Test check_nc_file raises error for non-.nc file.
        
        Given: An existing file without .nc extension
        When: check_nc_file() is called
        Then: ValueError is raised indicating wrong extension
        """
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("content")
        
        with pytest.raises(ValueError, match="is not a .nc file"):
            config.check_nc_file(str(txt_file), "test_file")


class TestCheckOrCreateFolder:
    """Test suite for check_or_create_folder() function.
    
    Tests cover:
    - Existing empty folder passes
    - Non-existent folder is created
    - Existing non-empty folder raises error
    - Path exists as file (not folder) raises error
    """
    
    def test_check_or_create_folder_existing_empty(self, tmp_path: Path):
        """Test check_or_create_folder passes for existing empty folder.
        
        Given: An existing empty directory
        When: check_or_create_folder() is called
        Then: No exception is raised
        """
        empty_dir = tmp_path / "empty_folder"
        empty_dir.mkdir()
        
        config.check_or_create_folder(str(empty_dir), "test_folder")
    
    def test_check_or_create_folder_creates_new(self, tmp_path: Path):
        """Test check_or_create_folder creates non-existent folder.
        
        Given: A path to a non-existent directory
        When: check_or_create_folder() is called
        Then: Directory is created and no exception is raised
        """
        new_dir = tmp_path / "new_folder"
        assert not new_dir.exists()
        
        config.check_or_create_folder(str(new_dir), "test_folder")
        
        assert new_dir.exists()
        assert new_dir.is_dir()
    
    def test_check_or_create_folder_non_empty_raises_error(
        self, create_mock_directory_structure
    ):
        """Test check_or_create_folder raises error for non-empty folder.
        
        Given: An existing directory containing files
        When: check_or_create_folder() is called
        Then: ValueError is raised indicating folder is not empty
        """
        non_empty_dir = create_mock_directory_structure(
            "non_empty",
            [("file.txt", "content")]
        )
        
        with pytest.raises(ValueError, match="not empty"):
            config.check_or_create_folder(str(non_empty_dir), "test_folder")
    
    def test_check_or_create_folder_path_is_file(self, tmp_path: Path):
        """Test check_or_create_folder raises error when path is a file.
        
        Given: A path that exists as a file (not directory)
        When: check_or_create_folder() is called
        Then: NotADirectoryError is raised
        """
        file_path = tmp_path / "file.txt"
        file_path.write_text("content")
        
        with pytest.raises(NotADirectoryError, match="not a directory"):
            config.check_or_create_folder(str(file_path), "test_folder")


class TestClearFolder:
    """Test suite for clear_folder() function.
    
    Tests cover:
    - Files in folder are deleted
    - Subdirectories are also deleted (full clear, not files-only)
    - Non-existent folder is handled gracefully
    - The folder itself is preserved (recreated empty)
    """
    
    def test_clear_folder_deletes_files(self, create_mock_directory_structure):
        """Test clear_folder deletes all files in directory.
        
        Given: A directory containing multiple files
        When: clear_folder() is called
        Then: All files are deleted, directory remains empty
        """
        dir_path = create_mock_directory_structure(
            "test_folder",
            [("file1.txt", "content1"), ("file2.txt", "content2")]
        )
        assert len(list(dir_path.iterdir())) == 2
        
        config.clear_folder(str(dir_path))
        
        assert dir_path.exists()  # Directory still exists
        assert len(list(dir_path.iterdir())) == 0  # But is now empty

    def test_clear_folder_nonexistent_directory(self, tmp_path: Path):
        """Test clear_folder handles non-existent directory gracefully.
        
        Given: A path to a non-existent directory
        When: clear_folder() is called
        Then: No exception is raised (function returns early)
        """
        nonexistent_dir = tmp_path / "does_not_exist"
        
        config.clear_folder(str(nonexistent_dir))


class TestParseObsDefOceanMod:
    """Test suite for parse_obs_def_ocean_mod() function.
    
    Tests cover:
    - Valid RST file parsing
    - Missing file handling
    - Malformed RST file handling
    - Empty type definitions section
    """
    
    def test_parse_obs_def_ocean_mod_valid(self, fixtures_root: Path):
        """Test parse_obs_def_ocean_mod parses valid RST file.
        
        Given: A valid DART obs_def_ocean_mod.rst file
        When: parse_obs_def_ocean_mod() is called
        Then: Two dictionaries are returned with correct mappings
        """
        rst_file = fixtures_root / "mock_obs_def_ocean_mod.rst"
        
        obs_type_to_qty, qty_to_obs_types = config.parse_obs_def_ocean_mod(str(rst_file))
        
        assert 'FLOAT_TEMPERATURE' in obs_type_to_qty
        assert obs_type_to_qty['FLOAT_TEMPERATURE'] == 'QTY_TEMPERATURE'
        assert obs_type_to_qty['FLOAT_SALINITY'] == 'QTY_SALINITY'
        assert obs_type_to_qty['CTD_TEMPERATURE'] == 'QTY_TEMPERATURE'
        
        assert 'QTY_TEMPERATURE' in qty_to_obs_types
        assert 'FLOAT_TEMPERATURE' in qty_to_obs_types['QTY_TEMPERATURE']
        assert 'CTD_TEMPERATURE' in qty_to_obs_types['QTY_TEMPERATURE']
        assert 'DRIFTER_TEMPERATURE' in qty_to_obs_types['QTY_TEMPERATURE']
        
        assert 'QTY_SALINITY' in qty_to_obs_types
        assert 'FLOAT_SALINITY' in qty_to_obs_types['QTY_SALINITY']
        assert 'CTD_SALINITY' in qty_to_obs_types['QTY_SALINITY']
    
    def test_parse_obs_def_ocean_mod_missing_file(self, tmp_path: Path):
        """Test parse_obs_def_ocean_mod raises error for missing file.
        
        Given: A path to a non-existent RST file
        When: parse_obs_def_ocean_mod() is called
        Then: FileNotFoundError is raised
        """
        nonexistent_file = tmp_path / "missing.rst"
        
        with pytest.raises(FileNotFoundError, match="does not exist"):
            config.parse_obs_def_ocean_mod(str(nonexistent_file))
    
    def test_parse_obs_def_ocean_mod_no_type_definitions(self, tmp_path: Path):
        """Test parse_obs_def_ocean_mod raises error when no definitions found.
        
        Given: An RST file with BEGIN/END markers but no actual definitions
        When: parse_obs_def_ocean_mod() is called
        Then: ValueError is raised indicating no definitions found
        """
        empty_rst = tmp_path / "empty.rst"
        empty_rst.write_text("""
! BEGIN DART PREPROCESS TYPE DEFINITIONS
! END DART PREPROCESS TYPE DEFINITIONS
""")
        
        with pytest.raises(ValueError, match="No observation type definitions found"):
            config.parse_obs_def_ocean_mod(str(empty_rst))
    
    def test_parse_obs_def_ocean_mod_missing_markers(self, tmp_path: Path):
        """Test parse_obs_def_ocean_mod raises error when markers are missing.
        
        Given: An RST file without BEGIN/END DART PREPROCESS markers
        When: parse_obs_def_ocean_mod() is called
        Then: ValueError is raised indicating markers not found
        """
        invalid_rst = tmp_path / "invalid.rst"
        invalid_rst.write_text("Some content without markers")
        
        with pytest.raises(ValueError, match="Could not find type definitions section"):
            config.parse_obs_def_ocean_mod(str(invalid_rst))


class TestValidateAndExpandObsTypes:
    """Test suite for validate_and_expand_obs_types() function.
    
    Tests cover:
    - Valid specific observation types
    - ALL_<FIELD> expansion
    - Invalid observation types
    - Mixed specific and ALL_ types
    """
    
    def test_validate_and_expand_obs_types_specific(self, fixtures_root: Path):
        """Test validate_and_expand_obs_types with specific observation types.
        
        Given: A list of specific observation types
        When: validate_and_expand_obs_types() is called
        Then: The same types are returned (no expansion)
        """
        obs_types_list = ['FLOAT_TEMPERATURE', 'FLOAT_SALINITY']

        obs_type_to_qty = {}
        obs_type_to_qty['FLOAT_TEMPERATURE'] = 'QTY_TEMPERATURE'
        obs_type_to_qty['FLOAT_SALINITY'] = 'QTY_SALINITY'
        qty_to_obs_types = {}
                
        result = config.validate_and_expand_obs_types(
            obs_types_list, (obs_type_to_qty, qty_to_obs_types)
        )
        
        assert 'FLOAT_TEMPERATURE' in result
        assert 'FLOAT_SALINITY' in result
        assert len(result) == 2
    
    def test_validate_and_expand_obs_types_all_temperature(self, fixtures_root: Path):
        """Test validate_and_expand_obs_types with ALL_TEMPERATURE.
        
        Given: A list containing 'ALL_TEMPERATURE'
        When: validate_and_expand_obs_types() is called
        Then: All temperature observation types are returned
        """
        obs_types_list = ['ALL_TEMPERATURE']
        
        obs_type_to_qty = {}
        obs_type_to_qty['FLOAT_TEMPERATURE'] = 'QTY_TEMPERATURE'
        obs_type_to_qty['CTD_TEMPERATURE'] = 'QTY_TEMPERATURE'
        obs_type_to_qty['DRIFTER_TEMPERATURE'] = 'QTY_TEMPERATURE'
        obs_type_to_qty['MOORING_TEMPERATURE'] = 'QTY_TEMPERATURE'
        obs_type_to_qty['GLIDER_TEMPERATURE'] = 'QTY_TEMPERATURE'
        qty_to_obs_types = {}
        qty_to_obs_types['QTY_TEMPERATURE'] = [
            'FLOAT_TEMPERATURE',
            'CTD_TEMPERATURE',
            'DRIFTER_TEMPERATURE',
            'MOORING_TEMPERATURE',
            'GLIDER_TEMPERATURE'            
        ]
        
        result = config.validate_and_expand_obs_types(
            obs_types_list, (obs_type_to_qty, qty_to_obs_types)
        )
        
        assert 'FLOAT_TEMPERATURE' in result
        assert 'CTD_TEMPERATURE' in result
        assert 'DRIFTER_TEMPERATURE' in result
        assert 'MOORING_TEMPERATURE' in result
        assert 'GLIDER_TEMPERATURE' in result
        # Should not include salinity types
        assert 'FLOAT_SALINITY' not in result
    
    def test_validate_and_expand_obs_types_all_salinity(self, fixtures_root: Path):
        """Test validate_and_expand_obs_types with ALL_SALINITY.
        
        Given: A list containing 'ALL_SALINITY'
        When: validate_and_expand_obs_types() is called
        Then: All salinity observation types are returned
        """
        obs_types_list = ['ALL_SALINITY']
        
        obs_type_to_qty = {}
        obs_type_to_qty['FLOAT_SALINITY'] = 'QTY_SALINITY'
        obs_type_to_qty['CTD_SALINITY'] = 'QTY_SALINITY'
        obs_type_to_qty['MOORING_SALINITY'] = 'QTY_SALINITY'
        obs_type_to_qty['GLIDER_SALINITY'] = 'QTY_SALINITY'
        qty_to_obs_types = {}
        qty_to_obs_types['QTY_SALINITY'] = [
            'FLOAT_SALINITY',
            'CTD_SALINITY',
            'MOORING_SALINITY',
            'GLIDER_SALINITY'            
        ]
        
        result = config.validate_and_expand_obs_types(
            obs_types_list, (obs_type_to_qty, qty_to_obs_types)
        )
        
        assert 'FLOAT_SALINITY' in result
        assert 'CTD_SALINITY' in result
        assert 'MOORING_SALINITY' in result
        assert 'GLIDER_SALINITY' in result
        # Should not include temperature types
        assert 'FLOAT_TEMPERATURE' not in result
    
    def test_validate_and_expand_obs_types_mixed(self, fixtures_root: Path):
        """Test validate_and_expand_obs_types with mixed specific and ALL_ types.
        
        Given: A list with both specific types and ALL_<FIELD> patterns
        When: validate_and_expand_obs_types() is called
        Then: Specific types and expanded ALL_ types are all returned
        """
        obs_types_list = ['CTD_TEMPERATURE', 'ALL_SALINITY']
        obs_type_to_qty = {}
        obs_type_to_qty['FLOAT_SALINITY'] = 'QTY_SALINITY'
        obs_type_to_qty['CTD_SALINITY'] = 'QTY_SALINITY'
        obs_type_to_qty['MOORING_SALINITY'] = 'QTY_SALINITY'
        obs_type_to_qty['CTD_TEMPERATURE'] = 'QTY_TEMPERATURE'
        qty_to_obs_types = {}
        qty_to_obs_types['QTY_SALINITY'] = [
            'FLOAT_SALINITY',
            'CTD_SALINITY',
            'MOORING_SALINITY',
        ]
        qty_to_obs_types['QTY_TEMPERATURE'] = [
            'CTD_TEMPERATURE',
        ]
        
        result = config.validate_and_expand_obs_types(
            obs_types_list, (obs_type_to_qty, qty_to_obs_types)
        )
        
        assert 'CTD_TEMPERATURE' in result
        assert 'FLOAT_SALINITY' in result
        assert 'CTD_SALINITY' in result
        assert 'MOORING_SALINITY' in result
    
    def test_validate_and_expand_obs_types_invalid_type(self, fixtures_root: Path):
        """Test validate_and_expand_obs_types raises error for invalid type.
        
        Given: A list containing an invalid observation type
        When: validate_and_expand_obs_types() is called
        Then: ValueError is raised indicating invalid type
        """
        obs_types_list = ['INVALID_TYPE']
        obs_type_to_qty = {}
        qty_to_obs_types = {}
        
        with pytest.raises(ValueError, match="Invalid observation type"):
            config.validate_and_expand_obs_types(
                obs_types_list,
                (obs_type_to_qty, qty_to_obs_types)
            )
    
    def test_validate_and_expand_obs_types_invalid_all(self):
        """Test validate_and_expand_obs_types raises error for invalid ALL_ pattern.

        Given: A list containing 'ALL_DENSITY' and dicts that have no QTY_DENSITY entry
        When: validate_and_expand_obs_types() is called
        Then: ValueError is raised indicating no types found for pattern
        """
        obs_types_list = ['ALL_DENSITY']
        obs_type_to_qty = {'FLOAT_TEMPERATURE': 'QTY_TEMPERATURE'}
        qty_to_obs_types = {'QTY_TEMPERATURE': ['FLOAT_TEMPERATURE']}

        with pytest.raises(ValueError, match="No observation types found for"):
            config.validate_and_expand_obs_types(
                obs_types_list, (obs_type_to_qty, qty_to_obs_types)
            )


class TestValidateConfigKeys:
    """Test suite for validate_config_keys() function.
    
    Tests cover:
    - Valid configuration passes validation
    - Missing required keys raises error
    """
    
    def test_validate_config_keys_valid(self, minimal_valid_config: dict):
        """Test validate_config_keys passes with all required keys present.
        
        Given: A config dict with all required keys
        When: validate_config_keys() is called
        Then: No exception is raised
        """
        required_keys = [
            'model_files_folder', 'obs_seq_in_folder', 'output_folder',
            'template_file', 'static_file', 'ocean_geometry',
            'perfect_model_obs_dir', 'parquet_folder'
        ]
        
        config.validate_config_keys(minimal_valid_config, required_keys)
    
    def test_validate_config_keys_missing_keys(self):
        """Test validate_config_keys raises KeyError when required keys are missing.
        
        Given: A config dict missing some required keys
        When: validate_config_keys() is called
        Then: KeyError is raised listing the missing keys
        """
        incomplete_config = {
            'model_files_folder': '/tmp/model',
            # Missing other required keys
        }
        required_keys = ['model_files_folder', 'obs_seq_in_folder', 'output_folder']
        
        with pytest.raises(KeyError, match="Required keys missing"):
            config.validate_config_keys(incomplete_config, required_keys)


# ============================================================================
# Mark all tests in this module as unit tests
# ============================================================================
pytestmark = pytest.mark.unit


class TestCheckOrCreateFolderEdgeCases:
    """Additional tests for check_or_create_folder error paths."""
    
    def test_check_or_create_folder_mkdir_error(self, tmp_path, monkeypatch):
        """Test check_or_create_folder handles OSError during folder creation."""
        target = tmp_path / "new_folder"
        
        def mock_makedirs(*args, **kwargs):
            raise OSError("Permission denied")
        
        monkeypatch.setattr(os, "makedirs", mock_makedirs)
        
        with pytest.raises(OSError, match="Could not create"):
            config.check_or_create_folder(str(target), "test_folder")


class TestParseObsDefOceanModEdgeCases:
    """Additional tests for parse_obs_def_ocean_mod error paths."""
    
    def test_parse_obs_def_ocean_mod_ioerror(self, tmp_path, monkeypatch):
        """Test parse_obs_def_ocean_mod handles IOError during file reading."""
        rst_file = tmp_path / "obs_def.rst"
        rst_file.write_text("! BEGIN DART PREPROCESS TYPE DEFINITIONS\n! END DART PREPROCESS TYPE DEFINITIONS")
        
        def mock_open(*args, **kwargs):
            raise IOError("Cannot read file")
        
        monkeypatch.setattr("builtins.open", mock_open)
        
        with pytest.raises(IOError, match="Could not read RST file"):
            config.parse_obs_def_ocean_mod(str(rst_file))
    
    def test_parse_obs_def_ocean_mod_wrong_format(self, tmp_path):
        """Test parse_obs_def_ocean_mod raise error if malformed."""
        rst_content = """
! BEGIN DART PREPROCESS TYPE DEFINITIONS
FLOAT_TEMPERATURE, QTY_TEMPERATURE, COMMON_CODE
INVALID_LINE
FLOAT_SALINITY, QTY_SALINITY, COMMON_CODE
! END DART PREPROCESS TYPE DEFINITIONS
"""
        rst_file = tmp_path / "obs_def.rst"
        rst_file.write_text(rst_content)
        
        with pytest.raises(ValueError, match="No observation type definitions found in"):
            config.parse_obs_def_ocean_mod(str(rst_file))
