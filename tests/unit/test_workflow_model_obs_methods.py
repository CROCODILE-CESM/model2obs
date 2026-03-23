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
    @patch.object(WorkflowModelObs, '_validate_model_file_timestamps')
    @patch.object(WorkflowModelObs, '_print_workflow_config')
    @patch.object(WorkflowModelObs, '_initialize_model_namelist')
    @patch('model2obs.workflows.workflow_model_obs.file_utils.get_sorted_files')
    @patch('model2obs.model_adapter.model_adapter_MOM6.ModelAdapterMOM6.validate_paths')
    def test_process_files_no_matching_mode(self, mock_validate_paths, mock_get_files,
                                           mock_init_nml, mock_print, mock_validate_ts,
                                           mock_process_pair,
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
        mock_validate_ts.assert_called_once_with(model_files)
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
    """Tests for _process_model_obs_pair() method.

    The method is self-contained: it creates its own temporary directory and a
    local Namelist instance via Namelist.from_content().  Tests mock
    Namelist.from_content and subprocess.Popen instead of self._namelist.

    obsq.ObsSequence is mocked via the autouse fixture below because the
    method now reads the obs_seq.in file before running DART and the obs_seq.out
    file after DART to collect statistics for the pair summary log.
    """

    @pytest.fixture(autouse=True)
    def _mock_obs_sequence(self):
        """Mock obsq.ObsSequence for all tests in this class.

        Returns a minimal DataFrame with a time column and a QC column so that
        the statistics-collection code in _process_model_obs_pair runs without
        touching the filesystem.
        """
        mock_df = pd.DataFrame({
            "time": [pd.Timestamp("2019-01-15T12:00:00")],
            "DART_QC_0": [0],
        })
        mock_seq = Mock()
        mock_seq.df = mock_df
        with patch('model2obs.workflows.workflow_model_obs.obsq.ObsSequence',
                   return_value=mock_seq):
            yield mock_seq

    def _make_config(self, tmp_path):
        """Return a minimal config dict with all folders created."""
        for d in ['output', 'trimmed', 'bck', 'dart', 'tmp']:
            (tmp_path / d).mkdir(exist_ok=True)
        return {
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
            'tmp_folder': str(tmp_path / 'tmp'),
        }

    @patch('subprocess.Popen')
    @patch('model2obs.workflows.workflow_model_obs.namelist.Namelist.from_content')
    @patch('model2obs.workflows.workflow_model_obs.file_utils.get_model_time_in_days_seconds')
    @patch('model2obs.workflows.workflow_model_obs.obs_seq_tools.trim_obs_seq_in')
    def test_process_model_obs_pair_with_trimming(self, mock_trim, mock_get_time,
                                                  mock_from_content, mock_popen,
                                                  tmp_path, capsys):
        """Test _process_model_obs_pair with trim_obs=True."""
        config = self._make_config(tmp_path)

        mock_get_time.return_value = (100, 0)
        mock_process = Mock()
        mock_process.wait.return_value = None
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        mock_local_nml = Mock()
        mock_from_content.return_value = mock_local_nml

        workflow = WorkflowModelObs(config)
        workflow._base_nml_content = "&model_nml\n/"

        workflow._process_model_obs_pair(
            str(tmp_path / 'model.nc'),
            str(tmp_path / 'obs.in'),
            trim_obs=True,
            counter=0,
            hull_polygon=Mock(),
            hull_points=np.array([[0, 0], [1, 1]]),
            force_obs_time=False,
        )

        mock_trim.assert_called_once()
        mock_from_content.assert_called_once()
        captured = capsys.readouterr()
        assert "Trimming obs_seq file" in captured.out
        assert "Calling perfect_model_obs" in captured.out

    @patch('subprocess.Popen')
    @patch('model2obs.workflows.workflow_model_obs.namelist.Namelist.from_content')
    @patch('model2obs.workflows.workflow_model_obs.file_utils.get_obs_time_in_days_seconds')
    def test_process_model_obs_pair_force_obs_time(self, mock_get_obs_time, mock_from_content,
                                                   mock_popen, tmp_path):
        """Test _process_model_obs_pair with force_obs_time=True uses obs time."""
        config = self._make_config(tmp_path)

        mock_get_obs_time.return_value = (200, 43200)
        mock_process = Mock()
        mock_process.wait.return_value = None
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        mock_local_nml = Mock()
        mock_from_content.return_value = mock_local_nml

        workflow = WorkflowModelObs(config)
        workflow._base_nml_content = "&model_nml\n/"

        workflow._process_model_obs_pair(
            str(tmp_path / 'model.nc'),
            str(tmp_path / 'obs.in'),
            trim_obs=False,
            counter=0,
            hull_polygon=None,
            hull_points=None,
            force_obs_time=True,
        )

        mock_get_obs_time.assert_called_once()
        mock_local_nml.update_namelist_param.assert_any_call(
            'perfect_model_obs_nml', 'init_time_days', 200, string=False
        )

    @patch('subprocess.Popen')
    @patch('model2obs.workflows.workflow_model_obs.namelist.Namelist.from_content')
    @patch('model2obs.workflows.workflow_model_obs.file_utils.get_model_time_in_days_seconds')
    def test_process_model_obs_pair_subprocess_error(self, mock_get_time, mock_from_content,
                                                     mock_popen, tmp_path):
        """Test _process_model_obs_pair raises RuntimeError on subprocess failure."""
        config = self._make_config(tmp_path)

        mock_get_time.return_value = (100, 0)
        mock_process = Mock()
        mock_process.wait.return_value = None
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        mock_from_content.return_value = Mock()

        workflow = WorkflowModelObs(config)
        workflow._base_nml_content = "&model_nml\n/"

        with pytest.raises(RuntimeError, match="perfect_model_obs failed"):
            workflow._process_model_obs_pair(
                str(tmp_path / 'model.nc'),
                str(tmp_path / 'obs.in'),
                trim_obs=False,
                counter=0,
                hull_polygon=None,
                hull_points=None,
                force_obs_time=False,
            )

    @patch('subprocess.Popen')
    @patch('model2obs.workflows.workflow_model_obs.namelist.Namelist.from_content')
    @patch('model2obs.workflows.workflow_model_obs.file_utils.get_model_time_in_days_seconds')
    def test_process_model_obs_pair_tmpdir_cleaned_on_success(self, mock_get_time,
                                                              mock_from_content,
                                                              mock_popen, tmp_path):
        """Test that the worker tmpdir — including files written inside it — is removed
        after successful execution."""
        config = self._make_config(tmp_path)

        mock_get_time.return_value = (100, 0)
        mock_process = Mock()
        mock_process.wait.return_value = None
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        def from_content_side_effect(content, working_dir):
            Path(working_dir, "input.nml").write_text(content)
            return Mock()

        mock_from_content.side_effect = from_content_side_effect

        workflow = WorkflowModelObs(config)
        workflow._base_nml_content = "&model_nml\n/"

        tmp_folder = tmp_path / 'tmp'

        workflow._process_model_obs_pair(
            str(tmp_path / 'model.nc'),
            str(tmp_path / 'obs.in'),
            trim_obs=False, counter=0,
            hull_polygon=None, hull_points=None,
            force_obs_time=False,
        )

        assert list(tmp_folder.iterdir()) == [], "Worker tmpdir was not cleaned up"

    @patch('subprocess.Popen')
    @patch('model2obs.workflows.workflow_model_obs.namelist.Namelist.from_content')
    @patch('model2obs.workflows.workflow_model_obs.file_utils.get_model_time_in_days_seconds')
    def test_process_model_obs_pair_tmpdir_cleaned_on_failure(self, mock_get_time,
                                                              mock_from_content,
                                                              mock_popen, tmp_path):
        """Test that the worker tmpdir — including files written inside it — is removed
        even when the subprocess fails."""
        config = self._make_config(tmp_path)

        mock_get_time.return_value = (100, 0)
        mock_process = Mock()
        mock_process.wait.return_value = None
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        def from_content_side_effect(content, working_dir):
            Path(working_dir, "input.nml").write_text(content)
            return Mock()

        mock_from_content.side_effect = from_content_side_effect

        workflow = WorkflowModelObs(config)
        workflow._base_nml_content = "&model_nml\n/"

        tmp_folder = tmp_path / 'tmp'

        with pytest.raises(RuntimeError):
            workflow._process_model_obs_pair(
                str(tmp_path / 'model.nc'),
                str(tmp_path / 'obs.in'),
                trim_obs=False, counter=0,
                hull_polygon=None, hull_points=None,
                force_obs_time=False,
            )

        assert list(tmp_folder.iterdir()) == [], "Worker tmpdir was not cleaned up on failure"

    @patch('subprocess.Popen')
    @patch('model2obs.workflows.workflow_model_obs.namelist.Namelist.from_content')
    @patch('model2obs.workflows.workflow_model_obs.file_utils.get_model_time_in_days_seconds')
    def test_process_model_obs_pair_per_pair_log(self, mock_get_time, mock_from_content,
                                                 mock_popen, tmp_path):
        """Test that each pair writes to its own numbered log file.

        Calls _process_model_obs_pair directly to validate the filename format
        (zero-padded counter) for a single pair.  The integration counterpart
        verifies that all N log files are present after a full parallel run.
        """
        config = self._make_config(tmp_path)

        mock_get_time.return_value = (100, 0)
        mock_process = Mock()
        mock_process.wait.return_value = None
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        mock_from_content.return_value = Mock()

        workflow = WorkflowModelObs(config)
        workflow._base_nml_content = "&model_nml\n/"

        workflow._process_model_obs_pair(
            str(tmp_path / 'model.nc'),
            str(tmp_path / 'obs.in'),
            trim_obs=False, counter=7,
            hull_polygon=None, hull_points=None,
            force_obs_time=False,
        )

        log_path = tmp_path / 'output' / 'perfect_model_obs_0007.log'
        assert log_path.exists(), "Per-pair log file was not created"


class TestWritePairSummaryLog:
    """Tests for _write_pair_summary_log() method."""

    def _make_workflow(self, tmp_path):
        (tmp_path / "output").mkdir(exist_ok=True)
        return WorkflowModelObs({
            'ocean_model': 'MOM6',
            'model_files_folder': str(tmp_path),
            'obs_seq_in_folder': str(tmp_path),
            'output_folder': str(tmp_path / 'output'),
            'template_file': 'template.nc',
            'static_file': 'static.nc',
            'ocean_geometry': 'ocean.nc',
            'perfect_model_obs_dir': str(tmp_path),
            'parquet_folder': str(tmp_path),
        })

    def _common_kwargs(self, tmp_path):
        return dict(
            file_number="0003",
            model_in_file=str(tmp_path / "model_tmp.nc"),
            original_model_file=str(tmp_path / "model.nc"),
            log_time_days=152685,
            log_time_seconds=43200,
            time_source="model file",
            obs_in_file_nml=str(tmp_path / "trimmed.in"),
            obs_in_file_orig=str(tmp_path / "obs.in"),
            obs_min_time=pd.Timestamp("2019-01-15T00:00:00"),
            obs_max_time=pd.Timestamp("2019-01-15T23:59:59"),
            obs_submitted_count=150,
            obs_original_count=200,
            dart_exit_code=0,
            obs_output_path=str(tmp_path / "output" / "obs_seq_0003.out"),
            n_success=120,
            n_fail=30,
        )

    def test_log_file_created(self, tmp_path):
        """Log file is written to output_folder/pair_summary_NNNN.log."""
        wf = self._make_workflow(tmp_path)
        wf._write_pair_summary_log(**self._common_kwargs(tmp_path))
        assert (tmp_path / "output" / "pair_summary_0003.log").exists()

    def test_log_contains_model_file(self, tmp_path):
        """Log contains the submitted NC file path."""
        wf = self._make_workflow(tmp_path)
        wf._write_pair_summary_log(**self._common_kwargs(tmp_path))
        content = (tmp_path / "output" / "pair_summary_0003.log").read_text()
        assert "model_tmp.nc" in content

    def test_log_contains_original_model_file_when_different(self, tmp_path):
        """Log records the original NC file when it differs from the submitted one."""
        wf = self._make_workflow(tmp_path)
        wf._write_pair_summary_log(**self._common_kwargs(tmp_path))
        content = (tmp_path / "output" / "pair_summary_0003.log").read_text()
        assert "Original NC file" in content
        assert "model.nc" in content

    def test_log_omits_original_model_file_when_same(self, tmp_path):
        """Log omits the original NC file line when it is the same as the submitted one."""
        wf = self._make_workflow(tmp_path)
        kwargs = self._common_kwargs(tmp_path)
        kwargs["original_model_file"] = kwargs["model_in_file"]  # same file
        wf._write_pair_summary_log(**kwargs)
        content = (tmp_path / "output" / "pair_summary_0003.log").read_text()
        assert "Original NC file" not in content

    def test_log_contains_time_used(self, tmp_path):
        """Log contains a human-readable time and its source label."""
        wf = self._make_workflow(tmp_path)
        wf._write_pair_summary_log(**self._common_kwargs(tmp_path))
        content = (tmp_path / "output" / "pair_summary_0003.log").read_text()
        assert "Time used" in content
        assert "(from model file)" in content

    def test_log_contains_obs_counts(self, tmp_path):
        """Log records both submitted count and original pre-trim count."""
        wf = self._make_workflow(tmp_path)
        wf._write_pair_summary_log(**self._common_kwargs(tmp_path))
        content = (tmp_path / "output" / "pair_summary_0003.log").read_text()
        assert "150" in content   # submitted
        assert "200" in content   # original before trimming

    def test_log_contains_interpolation_counts(self, tmp_path):
        """Log records successful and failed interpolation counts with percentages."""
        wf = self._make_workflow(tmp_path)
        wf._write_pair_summary_log(**self._common_kwargs(tmp_path))
        content = (tmp_path / "output" / "pair_summary_0003.log").read_text()
        assert "120" in content    # n_success
        assert "30" in content     # n_fail
        assert "80.0%" in content

    def test_log_dart_success_label(self, tmp_path):
        """Log writes 'success' when DART exit code is 0."""
        wf = self._make_workflow(tmp_path)
        wf._write_pair_summary_log(**self._common_kwargs(tmp_path))
        content = (tmp_path / "output" / "pair_summary_0003.log").read_text()
        assert "success" in content

    def test_log_dart_failed_label(self, tmp_path):
        """Log writes 'FAILED' when DART exit code is non-zero."""
        wf = self._make_workflow(tmp_path)
        kwargs = self._common_kwargs(tmp_path)
        kwargs["dart_exit_code"] = 1
        wf._write_pair_summary_log(**kwargs)
        content = (tmp_path / "output" / "pair_summary_0003.log").read_text()
        assert "FAILED" in content

    def test_log_missing_qc_shows_fallback(self, tmp_path):
        """Log writes a fallback message when interpolation counts are None."""
        wf = self._make_workflow(tmp_path)
        kwargs = self._common_kwargs(tmp_path)
        kwargs["n_success"] = None
        kwargs["n_fail"] = None
        wf._write_pair_summary_log(**kwargs)
        content = (tmp_path / "output" / "pair_summary_0003.log").read_text()
        assert "not found or QC column absent" in content

    def test_log_write_failure_is_non_fatal(self, tmp_path, capsys):
        """Log write failure prints a warning but does not raise."""
        wf = self._make_workflow(tmp_path)
        kwargs = self._common_kwargs(tmp_path)
        # Point output_folder to a non-existent path to force write failure
        wf.config["output_folder"] = str(tmp_path / "nonexistent_dir")
        wf._write_pair_summary_log(**kwargs)  # must not raise
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_log_written_on_dart_failure(self, tmp_path):
        """Summary log is present even when DART exit code is non-zero."""
        wf = self._make_workflow(tmp_path)
        kwargs = self._common_kwargs(tmp_path)
        kwargs["dart_exit_code"] = 1
        kwargs["n_success"] = None
        kwargs["n_fail"] = None
        wf._write_pair_summary_log(**kwargs)
        assert (tmp_path / "output" / "pair_summary_0003.log").exists()


class TestValidateModelFileTimestamps:
    """Tests for _validate_model_file_timestamps() method."""

    def _make_workflow(self, tmp_path):
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
        }
        return WorkflowModelObs(config)

    def test_unique_timestamps_pass(self, tmp_path):
        """Test that files with non-overlapping timestamps pass validation."""
        import xarray as xr
        files = []
        for i in range(3):
            path = str(tmp_path / f"model_{i}.nc")
            ds = xr.Dataset(coords={"time": pd.date_range(f"2020-01-0{i+1}", periods=1)})
            ds.to_netcdf(path)
            files.append(path)

        workflow = self._make_workflow(tmp_path)
        # Should not raise
        workflow._validate_model_file_timestamps(files)

    def test_duplicate_timestamps_raise(self, tmp_path):
        """Test that duplicate timestamps across files raise ValueError."""
        import xarray as xr
        files = []
        for i in range(2):
            path = str(tmp_path / f"model_{i}.nc")
            ds = xr.Dataset(coords={"time": pd.date_range("2020-01-01", periods=1)})
            ds.to_netcdf(path)
            files.append(path)

        workflow = self._make_workflow(tmp_path)
        with pytest.raises(ValueError, match="Duplicate timestamp"):
            workflow._validate_model_file_timestamps(files)

    def test_empty_file_list_passes(self, tmp_path):
        """Test that an empty file list passes validation without error."""
        workflow = self._make_workflow(tmp_path)
        workflow._validate_model_file_timestamps([])  # should not raise


class TestProcessModelFileWorker:
    """Tests for _process_model_file_worker() method."""

    def _make_config(self, tmp_path):
        for d in ['output', 'bck', 'dart', 'tmp']:
            (tmp_path / d).mkdir(exist_ok=True)
        return {
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
            'tmp_folder': str(tmp_path / 'tmp'),
            'time_window': {'days': 1, 'seconds': 0},
        }

    @patch.object(WorkflowModelObs, '_process_model_obs_pair')
    def test_worker_calls_process_pair_on_match(self, mock_process_pair, tmp_path):
        """Test _process_model_file_worker calls _process_model_obs_pair on match."""
        import xarray as xr
        config = self._make_config(tmp_path)

        model_file = str(tmp_path / 'model.nc')
        ds = xr.Dataset(coords={"time": pd.date_range("2020-01-01", periods=1)})
        ds.to_netcdf(model_file)

        obs_file = str(tmp_path / 'obs.in')

        # Build a mock obs sequence whose time window falls within the model time
        mock_obs = Mock()
        ts = pd.Timestamp("2020-01-01")
        mock_obs.df.time.min.return_value = ts - timedelta(hours=1)
        mock_obs.df.time.max.return_value = ts + timedelta(hours=1)

        workflow = WorkflowModelObs(config)
        workflow._base_nml_content = "&model_nml\n/"

        with patch('model2obs.workflows.workflow_model_obs.obsq.ObsSequence', return_value=mock_obs):
            count = workflow._process_model_file_worker(
                model_file, [obs_file], base_counter=5,
                trim_obs=False, hull_polygon=None, hull_points=None,
                force_obs_time=False,
            )

        assert count == 1
        mock_process_pair.assert_called_once()
        # Counter must be base_counter + local_match_index (5 + 0 = 5)
        call_args = mock_process_pair.call_args
        assert call_args[0][3] == 5

    @patch.object(WorkflowModelObs, '_process_model_obs_pair')
    def test_worker_returns_zero_on_no_match(self, mock_process_pair, tmp_path):
        """Test _process_model_file_worker returns 0 when no obs match."""
        import xarray as xr
        config = self._make_config(tmp_path)

        model_file = str(tmp_path / 'model.nc')
        ds = xr.Dataset(coords={"time": pd.date_range("2020-01-01", periods=1)})
        ds.to_netcdf(model_file)

        obs_file = str(tmp_path / 'obs.in')

        # Obs time far outside the model time window
        mock_obs = Mock()
        mock_obs.df.time.min.return_value = pd.Timestamp("2021-01-01")
        mock_obs.df.time.max.return_value = pd.Timestamp("2021-01-02")

        workflow = WorkflowModelObs(config)
        workflow._base_nml_content = "&model_nml\n/"

        with patch('model2obs.workflows.workflow_model_obs.obsq.ObsSequence', return_value=mock_obs):
            count = workflow._process_model_file_worker(
                model_file, [obs_file], base_counter=0,
                trim_obs=False, hull_polygon=None, hull_points=None,
                force_obs_time=False,
            )

        assert count == 0
        mock_process_pair.assert_not_called()

    @patch.object(WorkflowModelObs, '_process_model_obs_pair')
    def test_worker_shared_used_obs_list_prevents_rematch(self, mock_process_pair, tmp_path):
        """Test that an already-matched obs file is skipped when used_obs_in_files is shared."""
        import xarray as xr
        config = self._make_config(tmp_path)

        model_file = str(tmp_path / 'model.nc')
        ds = xr.Dataset(coords={"time": pd.date_range("2020-01-01", periods=1)})
        ds.to_netcdf(model_file)

        obs_file = str(tmp_path / 'obs.in')

        mock_obs = Mock()
        ts = pd.Timestamp("2020-01-01")
        mock_obs.df.time.min.return_value = ts - timedelta(hours=1)
        mock_obs.df.time.max.return_value = ts + timedelta(hours=1)

        workflow = WorkflowModelObs(config)
        workflow._base_nml_content = "&model_nml\n/"

        # Pre-populate the shared list with the obs file — simulates it being
        # matched by a previous model file in _process_with_time_matching.
        shared_used = [obs_file]

        with patch('model2obs.workflows.workflow_model_obs.obsq.ObsSequence', return_value=mock_obs):
            count = workflow._process_model_file_worker(
                model_file, [obs_file], base_counter=0,
                trim_obs=False, hull_polygon=None, hull_points=None,
                force_obs_time=False,
                used_obs_in_files=shared_used,
            )

        assert count == 0
        mock_process_pair.assert_not_called()

    @patch('subprocess.run')
    @patch.object(WorkflowModelObs, '_process_model_obs_pair')
    def test_worker_multi_snapshot_slices_with_ncks(
        self, mock_process_pair, mock_run, tmp_path
    ):
        """Test that a model file with multiple snapshots triggers an ncks slice."""
        import xarray as xr
        config = self._make_config(tmp_path)

        model_file = str(tmp_path / 'model.nc')
        ds = xr.Dataset(coords={"time": pd.date_range("2020-01-01", periods=2)})
        ds.to_netcdf(model_file)

        obs_file = str(tmp_path / 'obs.in')

        def ncks_side_effect(args, **kwargs):
            # Simulate ncks: create the output slice file so os.remove succeeds
            Path(args[-1]).touch()
            return Mock(returncode=0)

        mock_run.side_effect = ncks_side_effect

        mock_obs = Mock()
        ts = pd.Timestamp("2020-01-01")
        mock_obs.df.time.min.return_value = ts - timedelta(hours=1)
        mock_obs.df.time.max.return_value = ts + timedelta(hours=1)

        workflow = WorkflowModelObs(config)
        workflow._base_nml_content = "&model_nml\n/"

        with patch('model2obs.workflows.workflow_model_obs.obsq.ObsSequence', return_value=mock_obs):
            count = workflow._process_model_file_worker(
                model_file, [obs_file], base_counter=0,
                trim_obs=False, hull_polygon=None, hull_points=None,
                force_obs_time=False,
            )

        assert count == 1
        mock_process_pair.assert_called_once()
        # ncks should have been called to slice out snapshot 0
        ncks_call_args = mock_run.call_args[0][0]
        assert ncks_call_args[0] == "ncks"
        assert model_file in ncks_call_args
        # Temp slice file must not be left behind after successful processing
        tmp_files = list(Path(config['tmp_folder']).iterdir())
        assert tmp_files == [], f"Temp files not cleaned up: {tmp_files}"


class TestParallelDispatch:
    """Tests for parallel=True dispatch in process_files()."""

    def _base_workflow(self, tmp_path):
        for d in ['model', 'obs', 'output', 'bck', 'dart', 'tmp', 'parquet']:
            (tmp_path / d).mkdir(exist_ok=True)
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
            'input_nml_bck': str(tmp_path / 'bck'),
            'tmp_folder': str(tmp_path / 'tmp'),
            'time_window': {'days': 1, 'seconds': 0},
        }
        return WorkflowModelObs(config), config

    @patch.object(WorkflowModelObs, '_process_model_obs_pair')
    @patch.object(WorkflowModelObs, '_validate_model_file_timestamps')
    @patch.object(WorkflowModelObs, '_print_workflow_config')
    @patch.object(WorkflowModelObs, '_initialize_model_namelist')
    @patch('model2obs.workflows.workflow_model_obs.file_utils.get_model_time_in_days_seconds',
           return_value=(100, 0))
    @patch('model2obs.workflows.workflow_model_obs.file_utils.get_sorted_files')
    @patch('model2obs.model_adapter.model_adapter_MOM6.ModelAdapterMOM6.validate_paths')
    def test_parallel_no_matching_dispatches_all_pairs(
        self, mock_validate_paths, mock_get_files, mock_get_time, mock_init_nml,
        mock_print, mock_validate_ts, mock_process_pair, tmp_path
    ):
        """Test parallel=True with no_matching dispatches all pairs via ThreadPoolExecutor."""
        workflow, _ = self._base_workflow(tmp_path)
        workflow._namelist = Mock()

        model_files = ['model1.nc', 'model2.nc', 'model3.nc']
        obs_files = ['obs1.in', 'obs2.in', 'obs3.in']
        mock_get_files.side_effect = [model_files, obs_files]

        workflow.process_files(no_matching=True, trim_obs=False, parallel=True)

        assert mock_process_pair.call_count == 3

    @patch.object(WorkflowModelObs, '_process_model_file_worker')
    @patch.object(WorkflowModelObs, '_validate_model_file_timestamps')
    @patch.object(WorkflowModelObs, '_print_workflow_config')
    @patch.object(WorkflowModelObs, '_initialize_model_namelist')
    @patch('model2obs.workflows.workflow_model_obs.file_utils.get_sorted_files')
    @patch('model2obs.model_adapter.model_adapter_MOM6.ModelAdapterMOM6.validate_paths')
    def test_parallel_time_matching_dispatches_workers(
        self, mock_validate_paths, mock_get_files, mock_init_nml,
        mock_print, mock_validate_ts, mock_file_worker, tmp_path
    ):
        """Test parallel=True with time_matching dispatches _process_model_file_worker."""
        import xarray as xr

        workflow, config = self._base_workflow(tmp_path)
        workflow._namelist = Mock()

        # Create real model files so the pre-scan can open them
        model_files = []
        for i in range(2):
            path = str(tmp_path / 'model' / f'model_{i}.nc')
            ds = xr.Dataset(coords={"time": pd.date_range(f"2020-01-0{i+1}", periods=1)})
            ds.to_netcdf(path)
            model_files.append(path)

        obs_files = ['obs1.in', 'obs2.in']
        mock_get_files.side_effect = [model_files, obs_files]
        mock_file_worker.return_value = 0

        workflow.process_files(no_matching=False, trim_obs=False, parallel=True)

        assert mock_file_worker.call_count == 2

    @patch.object(WorkflowModelObs, '_process_model_obs_pair')
    @patch.object(WorkflowModelObs, '_validate_model_file_timestamps')
    @patch.object(WorkflowModelObs, '_print_workflow_config')
    @patch.object(WorkflowModelObs, '_initialize_model_namelist')
    @patch('model2obs.workflows.workflow_model_obs.file_utils.get_model_time_in_days_seconds',
           return_value=(100, 0))
    @patch('model2obs.workflows.workflow_model_obs.file_utils.get_sorted_files')
    @patch('model2obs.model_adapter.model_adapter_MOM6.ModelAdapterMOM6.validate_paths')
    def test_parallel_worker_exception_propagates(
        self, mock_validate_paths, mock_get_files, mock_get_time, mock_init_nml,
        mock_print, mock_validate_ts, mock_process_pair, tmp_path
    ):
        """Test that any exception raised inside a worker re-raises in the main thread.

        Uses a mocked _process_model_obs_pair so the test focuses solely on the
        ThreadPoolExecutor dispatch mechanism, independent of what causes the failure.
        The integration counterpart tests the specific subprocess-failure path.
        """
        workflow, _ = self._base_workflow(tmp_path)
        workflow._namelist = Mock()

        model_files = ['model1.nc', 'model2.nc']
        obs_files = ['obs1.in', 'obs2.in']
        mock_get_files.side_effect = [model_files, obs_files]
        mock_process_pair.side_effect = RuntimeError("Worker failure")

        with pytest.raises(RuntimeError, match="Worker failure"):
            workflow.process_files(no_matching=True, trim_obs=False, parallel=True)

    @patch.object(WorkflowModelObs, '_process_model_obs_pair')
    @patch.object(WorkflowModelObs, '_validate_model_file_timestamps')
    @patch.object(WorkflowModelObs, '_print_workflow_config')
    @patch.object(WorkflowModelObs, '_initialize_model_namelist')
    @patch('model2obs.workflows.workflow_model_obs.file_utils.get_sorted_files')
    @patch('model2obs.model_adapter.model_adapter_MOM6.ModelAdapterMOM6.validate_paths')
    def test_validate_timestamps_called_for_serial_path(
        self, mock_validate_paths, mock_get_files, mock_init_nml,
        mock_print, mock_validate_ts, mock_process_pair, tmp_path
    ):
        """Test _validate_model_file_timestamps is called even in serial mode."""
        workflow, _ = self._base_workflow(tmp_path)
        workflow._namelist = Mock()

        model_files = ['model1.nc']
        obs_files = ['obs1.in']
        mock_get_files.side_effect = [model_files, obs_files]

        workflow.process_files(no_matching=True, trim_obs=False, parallel=False)

        mock_validate_ts.assert_called_once_with(model_files)

    @patch.object(WorkflowModelObs, '_process_model_obs_pair')
    @patch.object(WorkflowModelObs, '_validate_model_file_timestamps')
    @patch.object(WorkflowModelObs, '_print_workflow_config')
    @patch.object(WorkflowModelObs, '_initialize_model_namelist')
    @patch('model2obs.workflows.workflow_model_obs.file_utils.get_sorted_files')
    @patch('model2obs.model_adapter.model_adapter_MOM6.ModelAdapterMOM6.validate_paths')
    @patch('model2obs.workflows.workflow_model_obs.model_tools.get_model_boundaries',
           return_value=('polygon', 'points'))
    def test_trim_obs_calls_get_model_boundaries(
        self, mock_boundaries, mock_validate_paths, mock_get_files, mock_init_nml,
        mock_print, mock_validate_ts, mock_process_pair, tmp_path
    ):
        """Test that trim_obs=True calls get_model_boundaries before processing."""
        workflow, _ = self._base_workflow(tmp_path)
        workflow._namelist = Mock()

        model_files = ['model1.nc']
        obs_files = ['obs1.in']
        mock_get_files.side_effect = [model_files, obs_files]

        workflow.process_files(no_matching=True, trim_obs=True, parallel=False)

        mock_boundaries.assert_called_once_with(workflow.config['ocean_geometry'])
        call_args = mock_process_pair.call_args
        assert call_args[0][4] == 'polygon'
        assert call_args[0][5] == 'points'


pytestmark = pytest.mark.unit

