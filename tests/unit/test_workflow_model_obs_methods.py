"""Unit tests for WorkflowModelObs class methods.

Tests cover initialization, configuration validation, file processing logic,
and namelist manipulation without requiring external subprocess calls.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock, call
import numpy as np
import xarray as xr
import pandas as pd
from datetime import timedelta

from model2obs.workflows.workflow_model_obs import WorkflowModelObs, RunOptions


@pytest.fixture
def base_config(tmp_path):
    """Provide base configuration dictionary with ocean_model for MOM6."""
    return {
        'ocean_model': 'MOM6',
        'model_files_folder': str(tmp_path / 'model'),
        'obs_seq_in_folder': str(tmp_path / 'obs'),
        'output_folder': str(tmp_path / 'output'),
        'template_file': str(tmp_path / 'template.nc'),
        'static_file': str(tmp_path / 'static.nc'),
        'ocean_geometry': str(tmp_path / 'ocean.nc'),
        'perfect_model_obs_dir': str(tmp_path / 'dart'),
        'parquet_folder': str(tmp_path / 'parquet'),
    }


@pytest.fixture
def roms_rutgers_config(tmp_path):
    """Provide base configuration dictionary with ocean_model for ROMS_RUTGERS."""
    return {
        'ocean_model': 'ROMS_RUTGERS',
        'model_files_folder': str(tmp_path / 'model'),
        'obs_seq_in_folder': str(tmp_path / 'obs'),
        'output_folder': str(tmp_path / 'output'),
        'roms_filename': str(tmp_path / 'roms'),
        'perfect_model_obs_dir': str(tmp_path / 'dart'),
        'parquet_folder': str(tmp_path / 'parquet'),
    }


class TestWorkflowModelObsInit:
    """Tests for WorkflowModelObs initialization."""
    
    def test_init_removes_existing_log_file(self, base_config, tmp_path, monkeypatch):
        """Test __init__ removes existing perfect_model_obs.log file."""
        monkeypatch.chdir(tmp_path)
        log_file = tmp_path / "perfect_model_obs.log"
        log_file.write_text("old log content")
        
        workflow = WorkflowModelObs(base_config)
        
        assert not log_file.exists()
        assert workflow.model_obs_df is None
    
    def test_init_creates_namelist_template_path(self, base_config):
        """Test __init__ sets input_nml_template path."""
        workflow = WorkflowModelObs(base_config)
        
        assert workflow.input_nml_template is not None
        assert 'input_template.nml' in str(workflow.input_nml_template)


class TestGetRequiredConfigKeys:
    """Tests for get_required_config_keys method."""
    
    def test_returns_all_required_keys(self, base_config):
        """Test get_required_config_keys returns complete list."""
        workflow = WorkflowModelObs(base_config)
        required_keys = workflow.model_adapter.get_required_config_keys()
        
        assert isinstance(required_keys, list)
        assert len(required_keys) == 8
        assert 'model_files_folder' in required_keys
        assert 'obs_seq_in_folder' in required_keys
        assert 'parquet_folder' in required_keys


class TestRunMethod:
    """Tests for run() method."""
    
    @patch('model2obs.workflows.workflow_model_obs.config_utils.clear_folder')
    @patch.object(WorkflowModelObs, 'merge_model_obs_to_parquet')
    @patch.object(WorkflowModelObs, 'process_files')
    def test_run_with_clear_output(self, mock_process, mock_merge, mock_clear, base_config, tmp_path, capsys):
        """Test run() clears output folders when clear_output=True."""
        base_config['input_nml_bck'] = str(tmp_path / 'bck')
        base_config['trimmed_obs_folder'] = str(tmp_path / 'trimmed')
        
        workflow = WorkflowModelObs(base_config)
        mock_process.return_value = 5
        
        workflow.run(clear_output=True, trim_obs=False)
        
        assert mock_clear.call_count == 4
        captured = capsys.readouterr()
        assert "Clearing all output folders" in captured.out
    
    @patch.object(WorkflowModelObs, 'merge_model_obs_to_parquet')
    @patch.object(WorkflowModelObs, 'process_files')
    def test_run_parquet_only_skips_processing(self, mock_process, mock_merge, base_config, capsys):
        """Test run() with parquet_only=True skips file processing."""
        workflow = WorkflowModelObs(base_config)
        
        workflow.run(parquet_only=True, trim_obs=False)
        
        mock_process.assert_not_called()
        mock_merge.assert_called_once()
        captured = capsys.readouterr()
        assert "Starting files processing" not in captured.out
    
    @patch.object(WorkflowModelObs, 'merge_model_obs_to_parquet')
    @patch.object(WorkflowModelObs, 'process_files')
    def test_run_validates_options_mom6_all_supported(self, mock_process, mock_merge, base_config):
        """Test run() validates run options for MOM6 with all options enabled."""
        workflow = WorkflowModelObs(base_config)
        
        workflow.run(trim_obs=True, no_matching=True, force_obs_time=True, parquet_only=True)
        
        mock_process.assert_not_called()
        mock_merge.assert_called_once()
    
    @patch.object(WorkflowModelObs, 'merge_model_obs_to_parquet')
    @patch.object(WorkflowModelObs, 'process_files')
    def test_run_validates_options_mom6_default(self, mock_process, mock_merge, base_config):
        """Test run() validates run options for MOM6 with default options."""
        workflow = WorkflowModelObs(base_config)
        
        workflow.run(parquet_only=True)
        
        mock_process.assert_not_called()
        mock_merge.assert_called_once()
    
    @patch.object(WorkflowModelObs, 'merge_model_obs_to_parquet')
    def test_run_validates_options_roms_rutgers_unsupported_trim_obs(self, mock_merge, roms_rutgers_config):
        """Test run() raises NotImplementedError for ROMS_RUTGERS with trim_obs=True."""
        workflow = WorkflowModelObs(roms_rutgers_config)
        
        with pytest.raises(NotImplementedError, match="does not support.*observation files trimming"):
            workflow.run(trim_obs=True, parquet_only=True)
    
    @patch.object(WorkflowModelObs, 'merge_model_obs_to_parquet')
    def test_run_validates_options_roms_rutgers_unsupported_no_matching(self, mock_merge, roms_rutgers_config):
        """Test run() raises NotImplementedError for ROMS_RUTGERS with no_matching=True."""
        workflow = WorkflowModelObs(roms_rutgers_config)
        
        with pytest.raises(NotImplementedError, match="does not support.*skipping time matching"):
            workflow.run(trim_obs=False, no_matching=True, parquet_only=True)
    
    @patch.object(WorkflowModelObs, 'merge_model_obs_to_parquet')
    def test_run_validates_options_roms_rutgers_unsupported_force_obs_time(self, mock_merge, roms_rutgers_config):
        """Test run() raises NotImplementedError for ROMS_RUTGERS with force_obs_time=True."""
        workflow = WorkflowModelObs(roms_rutgers_config)
        
        with pytest.raises(NotImplementedError, match="does not support.*observations reference time"):
            workflow.run(trim_obs=False, force_obs_time=True, parquet_only=True)
    
    @patch.object(WorkflowModelObs, 'merge_model_obs_to_parquet')
    @patch.object(WorkflowModelObs, 'process_files')
    def test_run_validates_options_roms_rutgers_all_false(self, mock_process, mock_merge, roms_rutgers_config):
        """Test run() succeeds for ROMS_RUTGERS when all unsupported options are False."""
        workflow = WorkflowModelObs(roms_rutgers_config)
        
        workflow.run(trim_obs=False, no_matching=False, force_obs_time=False, parquet_only=True)
        
        mock_process.assert_not_called()
        mock_merge.assert_called_once()
    
    @patch.object(WorkflowModelObs, 'merge_model_obs_to_parquet')
    def test_run_validates_options_roms_rutgers_multiple_unsupported(self, mock_merge, roms_rutgers_config):
        """Test run() raises NotImplementedError for ROMS_RUTGERS with multiple unsupported options."""
        workflow = WorkflowModelObs(roms_rutgers_config)
        
        with pytest.raises(NotImplementedError):
            workflow.run(trim_obs=True, no_matching=True, parquet_only=True)


class TestProcessFiles:
    """Tests for process_files() method."""
    
    def test_process_files_checks_perfect_model_obs_dir(self, tmp_path):
        """Test process_files checks for perfect_model_obs_dir in config."""
        config = {
            'ocean_model': 'MOM6',
            'model_files_folder': str(tmp_path / 'model'),
            'obs_seq_in_folder': str(tmp_path / 'obs'),
            'output_folder': str(tmp_path / 'output'),
            'template_file': 'template.nc',
            'static_file': 'static.nc',
            'ocean_geometry': 'ocean.nc',
            'perfect_model_obs_dir': str(tmp_path / 'dart'),
            'parquet_folder': str(tmp_path / 'parquet'),
        }
        
        workflow = WorkflowModelObs(config)
        workflow.config['perfect_model_obs_dir'] = None
        
        with pytest.raises(ValueError, match="perfect_model_obs_dir"):
            workflow.process_files()
    
    @patch.object(WorkflowModelObs, '_process_model_obs_pair')
    @patch.object(WorkflowModelObs, '_print_workflow_config')
    @patch.object(WorkflowModelObs, '_initialize_model_namelist')
    @patch('model2obs.workflows.workflow_model_obs.file_utils.get_sorted_files')
    @patch('model2obs.model_adapter.model_adapter_MOM6.ModelAdapterMOM6.validate_paths')
    def test_process_files_no_matching_mode(self, mock_validate_paths, mock_get_files, 
                                           mock_init_nml, mock_print, mock_process_pair, 
                                           tmp_path):
        """Test process_files with no_matching=True processes files in pairs."""
        config = {
            'ocean_model': 'MOM6',
            'model_files_folder': str(tmp_path / 'model'),
            'obs_seq_in_folder': str(tmp_path / 'obs'),
            'output_folder': str(tmp_path / 'output'),
            'template_file': 'template.nc',
            'static_file': 'static.nc',
            'ocean_geometry': 'ocean.nc',
            'perfect_model_obs_dir': str(tmp_path / 'dart'),
            'parquet_folder': str(tmp_path / 'parquet'),
        }
        
        model_files = ['model1.nc', 'model2.nc']
        obs_files = ['obs1.in', 'obs2.in']
        mock_get_files.side_effect = [model_files, obs_files]
        
        workflow = WorkflowModelObs(config)
        workflow._namelist = Mock()
        workflow._namelist.cleanup_namelist_symlink = Mock()
        
        workflow.process_files(no_matching=True, trim_obs=False)
        
        assert mock_process_pair.call_count == 2
        workflow._namelist.cleanup_namelist_symlink.assert_called_once()


class TestPrintWorkflowConfig:
    """Tests for _print_workflow_config() method."""
    
    def test_print_workflow_config_basic(self, tmp_path, capsys):
        """Test _print_workflow_config prints all configuration values."""
        config = {
            'ocean_model': 'MOM6',
            'model_files_folder': '/path/to/model',
            'obs_seq_in_folder': '/path/to/obs',
            'output_folder': '/path/to/output',
            'template_file': 'template.nc',
            'static_file': 'static.nc',
            'ocean_geometry': 'ocean.nc',
            'perfect_model_obs_dir': '/path/to/dart',
            'parquet_folder': '/path/to/parquet',
            'input_nml_bck': '/path/to/bck',
            'tmp_folder': '/path/to/tmp',
        }
        
        workflow = WorkflowModelObs(config)
        workflow._namelist = Mock()
        workflow._namelist.namelist_path = '/path/to/input.nml'
        
        workflow._print_workflow_config(trim_obs=False)
        
        captured = capsys.readouterr()
        assert "Configuration:" in captured.out
        assert "/path/to/model" in captured.out
        assert "/path/to/obs" in captured.out
        assert "/path/to/dart" in captured.out
    
    def test_print_workflow_config_with_trim_obs(self, tmp_path, capsys):
        """Test _print_workflow_config includes trimmed_obs_folder when trim_obs=True."""
        config = {
            'ocean_model': 'MOM6',
            'model_files_folder': str(tmp_path),
            'obs_seq_in_folder': str(tmp_path),
            'output_folder': str(tmp_path),
            'template_file': 'template.nc',
            'static_file': 'static.nc',
            'ocean_geometry': 'ocean.nc',
            'perfect_model_obs_dir': str(tmp_path),
            'parquet_folder': str(tmp_path),
            'trimmed_obs_folder': '/path/to/trimmed',
            'tmp_folder': str(tmp_path),
        }
        
        workflow = WorkflowModelObs(config)
        workflow._namelist = Mock()
        workflow._namelist.namelist_path = 'input.nml'
        
        workflow._print_workflow_config(trim_obs=True)
        
        captured = capsys.readouterr()
        assert "trimmed_obs_folder" in captured.out


class TestValidateWorkflowPaths:
    """Tests for path validation.
    
    Note: Path validation has been moved to ModelAdapter classes.
    See tests/unit/test_adapter.py::TestModelAdapterPathValidation for tests.
    """
    pass


class TestInitializeModelNamelist:
    """Tests for _initialize_model_namelist() method."""
    
    @patch('model2obs.workflows.workflow_model_obs.namelist.Namelist')
    def test_initialize_model_namelist_basic_params(self, mock_namelist_class, tmp_path):
        """Test _initialize_model_namelist sets basic namelist parameters."""
        config = {
            'ocean_model': 'MOM6',
            'model_files_folder': str(tmp_path),
            'obs_seq_in_folder': str(tmp_path),
            'output_folder': str(tmp_path),
            'template_file': 'template.nc',
            'static_file': 'static.nc',
            'ocean_geometry': 'ocean.nc',
            'perfect_model_obs_dir': str(tmp_path),
            'parquet_folder': str(tmp_path),
            'time_window': {'days': 1, 'seconds': 0},
        }
        
        mock_nml = Mock()
        mock_namelist_class.return_value = mock_nml
        
        workflow = WorkflowModelObs(config)
        workflow._initialize_model_namelist()
        
        assert workflow._namelist == mock_nml
        assert mock_nml.update_namelist_param.call_count >= 5
    
    @patch('model2obs.workflows.workflow_model_obs.config_utils.validate_and_expand_obs_types')
    @patch('model2obs.workflows.workflow_model_obs.namelist.Namelist')
    def test_initialize_model_namelist_with_obs_types(self, mock_namelist_class, 
                                                     mock_validate_obs, tmp_path, capsys):
        """Test _initialize_model_namelist processes use_these_obs config."""
        config = {
            'ocean_model': 'MOM6',
            'model_files_folder': str(tmp_path),
            'obs_seq_in_folder': str(tmp_path),
            'output_folder': str(tmp_path),
            'template_file': 'template.nc',
            'static_file': 'static.nc',
            'ocean_geometry': 'ocean.nc',
            'perfect_model_obs_dir': str(tmp_path),
            'parquet_folder': str(tmp_path),
            'time_window': {'days': 1, 'seconds': 0},
            'use_these_obs': ['ALL_TEMPERATURE'],
        }
        
        mock_nml = Mock()
        mock_namelist_class.return_value = mock_nml
        mock_validate_obs.return_value = ['FLOAT_TEMPERATURE', 'DRIFTER_TEMPERATURE']
        
        workflow = WorkflowModelObs(config)
        workflow._initialize_model_namelist()
        
        captured = capsys.readouterr()
        assert "Processing observation types" in captured.out
        mock_validate_obs.assert_called_once()
    
    @patch('model2obs.workflows.workflow_model_obs.config_utils.validate_and_expand_obs_types')
    @patch('model2obs.workflows.workflow_model_obs.namelist.Namelist')
    def test_initialize_model_namelist_obs_types_error_handling(self, mock_namelist_class,
                                                                mock_validate_obs, tmp_path, capsys):
        """Test _initialize_model_namelist handles obs_types validation errors."""
        config = {
            'ocean_model': 'MOM6',
            'model_files_folder': str(tmp_path),
            'obs_seq_in_folder': str(tmp_path),
            'output_folder': str(tmp_path),
            'template_file': 'template.nc',
            'static_file': 'static.nc',
            'ocean_geometry': 'ocean.nc',
            'perfect_model_obs_dir': str(tmp_path),
            'parquet_folder': str(tmp_path),
            'time_window': {'days': 1, 'seconds': 0},
            'use_these_obs': ['INVALID_TYPE'],
        }
        
        mock_nml = Mock()
        mock_namelist_class.return_value = mock_nml
        mock_validate_obs.side_effect = FileNotFoundError("RST file not found")
        
        workflow = WorkflowModelObs(config)
        workflow._initialize_model_namelist()
        
        captured = capsys.readouterr()
        assert "Warning: Could not process observation types" in captured.out
        assert "Continuing with existing obs_kind_nml" in captured.out


class TestProcessModelObsPair:
    """Tests for _process_model_obs_pair() method."""
    
    @patch('subprocess.Popen')
    @patch('model2obs.workflows.workflow_model_obs.file_utils.get_model_time_in_days_seconds')
    @patch('model2obs.workflows.workflow_model_obs.obs_seq_tools.trim_obs_seq_in')
    def test_process_model_obs_pair_with_trimming(self, mock_trim, mock_get_time,
                                                  mock_popen, tmp_path, capsys):
        """Test _process_model_obs_pair with trim_obs=True."""
        config = {
            'ocean_model': 'MOM6',
            'model_files_folder': str(tmp_path),
            'obs_seq_in_folder': str(tmp_path),
            'output_folder': str(tmp_path / 'output'),
            'template_file': 'template.nc',
            'static_file': 'static.nc',
            'ocean_geometry': 'ocean.nc',
            'perfect_model_obs_dir': str(tmp_path / 'dart'),
            'parquet_folder': str(tmp_path / 'parquet'),
            'trimmed_obs_folder': str(tmp_path / 'trimmed'),
            'input_nml_bck': str(tmp_path / 'bck'),
        }
        
        (tmp_path / 'output').mkdir()
        (tmp_path / 'trimmed').mkdir()
        (tmp_path / 'bck').mkdir()
        (tmp_path / 'dart').mkdir()
        
        mock_get_time.return_value = (100, 0)
        mock_process = Mock()
        mock_process.wait.return_value = None
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        workflow = WorkflowModelObs(config)
        workflow._namelist = Mock()
        
        workflow._process_model_obs_pair(
            str(tmp_path / 'model.nc'),
            str(tmp_path / 'obs.in'),
            trim_obs=True,
            counter=0,
            hull_polygon=Mock(),
            hull_points=np.array([[0, 0], [1, 1]]),
            force_obs_time=False
        )
        
        mock_trim.assert_called_once()
        captured = capsys.readouterr()
        assert "Trimming obs_seq file" in captured.out
        assert "Calling perfect_model_obs" in captured.out
    
    @patch('subprocess.Popen')
    @patch('model2obs.workflows.workflow_model_obs.file_utils.get_obs_time_in_days_seconds')
    def test_process_model_obs_pair_force_obs_time(self, mock_get_obs_time, mock_popen, tmp_path):
        """Test _process_model_obs_pair with force_obs_time=True uses obs time."""
        config = {
            'ocean_model': 'MOM6',
            'model_files_folder': str(tmp_path),
            'obs_seq_in_folder': str(tmp_path),
            'output_folder': str(tmp_path / 'output'),
            'template_file': 'template.nc',
            'static_file': 'static.nc',
            'ocean_geometry': 'ocean.nc',
            'perfect_model_obs_dir': str(tmp_path / 'dart'),
            'parquet_folder': str(tmp_path / 'parquet'),
            'input_nml_bck': str(tmp_path / 'bck'),
        }
        
        (tmp_path / 'output').mkdir()
        (tmp_path / 'bck').mkdir()
        (tmp_path / 'dart').mkdir()
        
        mock_get_obs_time.return_value = (200, 43200)
        mock_process = Mock()
        mock_process.wait.return_value = None
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        workflow = WorkflowModelObs(config)
        workflow._namelist = Mock()
        
        workflow._process_model_obs_pair(
            str(tmp_path / 'model.nc'),
            str(tmp_path / 'obs.in'),
            trim_obs=False,
            counter=0,
            hull_polygon=None,
            hull_points=None,
            force_obs_time=True
        )
        
        mock_get_obs_time.assert_called_once()
        workflow._namelist.update_namelist_param.assert_any_call(
            'perfect_model_obs_nml', 'init_time_days', 200, string=False
        )
    
    @patch('subprocess.Popen')
    @patch('model2obs.workflows.workflow_model_obs.file_utils.get_model_time_in_days_seconds')
    def test_process_model_obs_pair_subprocess_error(self, mock_get_time, mock_popen, tmp_path):
        """Test _process_model_obs_pair raises error on subprocess failure."""
        config = {
            'ocean_model': 'MOM6',
            'model_files_folder': str(tmp_path),
            'obs_seq_in_folder': str(tmp_path),
            'output_folder': str(tmp_path / 'output'),
            'template_file': 'template.nc',
            'static_file': 'static.nc',
            'ocean_geometry': 'ocean.nc',
            'perfect_model_obs_dir': str(tmp_path / 'dart'),
            'parquet_folder': str(tmp_path / 'parquet'),
            'input_nml_bck': str(tmp_path / 'bck'),
        }
        
        (tmp_path / 'output').mkdir()
        (tmp_path / 'bck').mkdir()
        (tmp_path / 'dart').mkdir()
        
        mock_get_time.return_value = (100, 0)
        mock_process = Mock()
        mock_process.wait.return_value = None
        mock_process.returncode = 1
        mock_process.stderr = Mock()
        mock_process.stderr.read.return_value = "Subprocess error"
        mock_popen.return_value = mock_process
        
        workflow = WorkflowModelObs(config)
        workflow._namelist = Mock()
        
        with pytest.raises(RuntimeError, match="Error"):
            workflow._process_model_obs_pair(
                str(tmp_path / 'model.nc'),
                str(tmp_path / 'obs.in'),
                trim_obs=False,
                counter=0,
                hull_polygon=None,
                hull_points=None,
                force_obs_time=False
            )


pytestmark = pytest.mark.unit
