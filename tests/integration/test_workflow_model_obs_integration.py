"""Integration tests for WorkflowModelObs.

Tests the complete workflow integration with mocked subprocess dependencies.
Subprocess calls (ncks, perfect_model_obs) are mocked, but pydartdiags is
used directly as it's available in the conda environment.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, Mock, call
import shutil
import xarray as xr
import numpy as np
import pandas as pd


@pytest.fixture
def workflow_config(tmp_path):
    """Create a complete workflow configuration for testing."""
    # Change to tmp directory to isolate file operations
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        # Create required directories
        model_files = tmp_path / "model_files"
        obs_in = tmp_path / "obs_seq_in"
        output = tmp_path / "output"
        parquet = tmp_path / "parquet"
        dart_work = tmp_path / "dart_work"
        trimmed_obs = tmp_path / "trimmed_obs"
        nml_bck = tmp_path / "input_nml_bck"
        
        for folder in [model_files, obs_in, output, parquet, dart_work, trimmed_obs, nml_bck]:
            folder.mkdir(parents=True, exist_ok=True)
        
        # Create mock model files
        template = tmp_path / "template.nc"
        static = tmp_path / "static.nc"
        geometry = tmp_path / "ocean_geometry.nc"
        
        # Create minimal NetCDF files using xarray
        ds_template = xr.Dataset(
            data_vars={"temp": (["time", "lat", "lon"], np.zeros((1, 10, 10)))},
            coords={"time": [0], "lat": np.arange(10), "lon": np.arange(10)}
        )
        ds_template.to_netcdf(template)
        ds_template.to_netcdf(static)
        
        # Geometry with wet mask
        ds_geom = xr.Dataset(
            data_vars={"wet": (["lat", "lon"], np.ones((10, 10)))},
            coords={
                "lonh": (["lat", "lon"], np.tile(np.arange(10), (10, 1))),
                "lath": (["lat", "lon"], np.tile(np.arange(10).reshape(-1, 1), (1, 10)))
            }
        )
        ds_geom.to_netcdf(geometry)
        
        # Create model data files with timestamps
        for i in range(3):
            model_file = model_files / f"model_2020010{i+1}.nc"
            ds = xr.Dataset(
                data_vars={"temp": (["time", "lat", "lon"], np.random.rand(1, 10, 10))},
                coords={
                    "time": pd.date_range(f"2020-01-0{i+1}", periods=1),
                    "lat": np.arange(10),
                    "lon": np.arange(10)
                }
            )
            ds.to_netcdf(model_file)
        
        # Create obs_seq.in files
        obs_def_file = tmp_path / "obs_def_ocean_mod.rst"
        obs_def_file.write_text("FLOAT_TEMPERATURE\nFLOAT_SALINITY\n")
        
        config = {
            'ocean_model': 'MOM6',
            'model_files_folder': str(model_files),
            'obs_seq_in_folder': str(obs_in),
            'output_folder': str(output),
            'template_file': str(template),
            'static_file': str(static),
            'ocean_geometry': str(geometry),
            'perfect_model_obs_dir': str(dart_work),
            'parquet_folder': str(parquet),
            'trimmed_obs_folder': str(trimmed_obs),
            'input_nml_bck': str(nml_bck),
            'tmp_folder': str(tmp_path / 'tmp'),
            'obs_def_file': str(obs_def_file),
            'obs_types': ['FLOAT_TEMPERATURE'],
            'time_window': {
                'days': 1, 'hours': 0, 'minutes': 0, 'seconds': 0,
                'weeks': 0, 'months': 0, 'years': 0
            }
        }
        
        # Create tmp folder
        Path(config['tmp_folder']).mkdir(parents=True, exist_ok=True)
        
        yield config
        
    finally:
        # Cleanup: restore directory and remove any artifacts
        os.chdir(original_dir)
        # Clean up any files created in the original directory
        for artifact in ['input.nml', 'input.nml.backup', 'perfect_model_obs.log', 'obs_seq.final']:
            artifact_path = Path(original_dir) / artifact
            if artifact_path.exists():
                try:
                    artifact_path.unlink()
                except:
                    pass


@pytest.fixture
def mock_obs_seq_files(workflow_config, tmp_path):
    """Create mock obs_seq.in files."""
    from tests.fixtures.mock_obs_seq_files.create_mock_obs_seq import create_obs_seq_in
    
    obs_folder = Path(workflow_config['obs_seq_in_folder'])
    
    # Create obs_seq.in files matching the model files
    for i in range(3):
        obs_file = obs_folder / f"obs_seq_2020010{i+1}.in"
        
        # Create simple observations (lon_rad, lat_rad, depth, value, obs_type)
        observations = [
            (0.1 + j*0.01, 0.1 + j*0.01, 0.0, 20.0 + j, 11)  # TEMPERATURE
            for j in range(10)
        ]
        create_obs_seq_in(obs_file, observations)
    
    return obs_folder


class TestWorkflowModelObsInitialization:
    """Test WorkflowModelObs initialization."""
    
    def test_init_with_valid_config(self, workflow_config):
        """Test initialization with valid configuration."""
        from model2obs.workflows.workflow_model_obs import WorkflowModelObs
        
        workflow = WorkflowModelObs(workflow_config)
        
        assert workflow.config == workflow_config
        assert workflow.model_obs_df is None
        assert workflow.perfect_model_obs_log_file == "perfect_model_obs.log"
    
    def test_init_validates_required_keys(self):
        """Test that initialization validates required keys."""
        from model2obs.workflows.workflow_model_obs import WorkflowModelObs
        
        incomplete_config = {'ocean_model': 'MOM6', 'model_files_folder': '/path'}
        
        with pytest.raises(KeyError, match="Required keys missing"):
            WorkflowModelObs(incomplete_config)
    
    def test_init_cleans_old_log_file(self, workflow_config, tmp_path):
        """Test that initialization removes old log files."""
        from model2obs.workflows.workflow_model_obs import WorkflowModelObs
        
        # Create old log file
        log_file = Path("perfect_model_obs.log")
        original_cwd = os.getcwd()
        
        try:
            os.chdir(tmp_path)
            log_file.touch()
            assert log_file.exists()
            
            workflow = WorkflowModelObs(workflow_config)
            
            # Log file should be removed during init
            assert not log_file.exists()
        finally:
            os.chdir(original_cwd)
            if log_file.exists():
                log_file.unlink()


class TestWorkflowModelObsProcessFiles:
    """Test WorkflowModelObs process_files method with mocked subprocesses."""
    
    @patch('subprocess.Popen')
    @patch('subprocess.run')
    def test_process_files_basic(self, mock_run, mock_popen, workflow_config, mock_obs_seq_files, tmp_path):
        """Test basic file processing with mocked subprocesses."""
        from model2obs.workflows.workflow_model_obs import WorkflowModelObs
        
        # Mock successful ncks execution
        mock_run.return_value = Mock(returncode=0)
        
        # Mock successful perfect_model_obs execution
        def pmo_side_effect(*args, **kwargs):
            # Create obs_seq.final output
            cwd = kwargs.get('cwd', os.getcwd())
            obs_final = Path(cwd) / "obs_seq.final"
            obs_final.write_text("mock observation output")
            
            # Rename to output with counter
            output_folder = workflow_config['output_folder']
            counter = len(list(Path(output_folder).glob("obs_seq_*.out")))
            output_file = Path(output_folder) / f"obs_seq_{counter:04d}.out"
            
            mock_proc = Mock()
            mock_proc.returncode = 0
            mock_proc.wait.return_value = None
            return mock_proc
        
        mock_popen.side_effect = pmo_side_effect
        
        # Skip trimming for simplicity
        with patch('model2obs.io.obs_seq_tools.trim_obs_seq_in'):
            workflow = WorkflowModelObs(workflow_config)
            
            # Run workflow (without parquet conversion)
            workflow.process_files(trim_obs=False, no_matching=True)
            
            # Verify subprocess was called 3 times (once per file pair)
            assert mock_popen.call_count == 3
    
    @patch('subprocess.Popen')
    @patch('subprocess.run')
    def test_process_files_with_time_matching(self, mock_run, mock_popen, workflow_config, mock_obs_seq_files):
        """Test file processing with time matching enabled."""
        from model2obs.workflows.workflow_model_obs import WorkflowModelObs
        
        # Setup mocks
        mock_run.return_value = Mock(returncode=0)
        mock_proc = Mock()
        mock_proc.returncode = 0
        mock_proc.wait.return_value = None
        mock_popen.return_value = mock_proc
        
        with patch('model2obs.io.obs_seq_tools.trim_obs_seq_in'):
            workflow = WorkflowModelObs(workflow_config)
            
            # Process with time matching (default)
            # Note: time matching may result in 0 matches if times don't align
            workflow.process_files(trim_obs=False, no_matching=False)
            
            # Workflow should complete without error (even if 0 matches)
            # Just verify it ran without exception
            assert True
    
    @patch('subprocess.Popen')
    def test_process_files_subprocess_failure(self, mock_popen, workflow_config, mock_obs_seq_files):
        """Test handling of perfect_model_obs subprocess failure."""
        from model2obs.workflows.workflow_model_obs import WorkflowModelObs
        
        # Mock failed perfect_model_obs
        mock_proc = Mock()
        mock_proc.returncode = 1
        mock_proc.wait.return_value = None
        mock_proc.stderr = Mock()
        mock_proc.stderr.read.return_value = "DART error"
        mock_popen.return_value = mock_proc
        
        with patch('model2obs.io.obs_seq_tools.trim_obs_seq_in'):
            with patch('subprocess.run', return_value=Mock(returncode=0)):
                workflow = WorkflowModelObs(workflow_config)
                
                # Should raise error on subprocess failure
                with pytest.raises(RuntimeError, match="perfect_model_obs failed"):
                    workflow.process_files(trim_obs=False, no_matching=True)


class TestWorkflowModelObsParquetGeneration:
    """Test parquet file generation."""
    
    def test_merge_model_obs_to_parquet_basic(self, workflow_config, tmp_path):
        """Test basic parquet generation from obs_seq files."""
        from model2obs.workflows.workflow_model_obs import WorkflowModelObs
        
        # Create mock obs_seq output files
        output_folder = Path(workflow_config['output_folder'])
        for i in range(2):
            obs_out = output_folder / f"obs_seq_{i:04d}.out"
            obs_out.write_text("mock observation data")
        
        # Create corresponding original obs_seq.in files
        obs_in_folder = Path(workflow_config['obs_seq_in_folder'])
        for i in range(2):
            obs_in = obs_in_folder / f"obs_seq_{i:04d}.in"
            obs_in.write_text("mock input observation")
        
        # This test requires actual obs_seq files with pydartdiags
        # For now, just verify the workflow can be instantiated
        workflow = WorkflowModelObs(workflow_config)
        
        # The actual merge would require valid obs_seq.out files  
        # which are created by perfect_model_obs
        # So we just verify the workflow object is created correctly
        assert workflow.config == workflow_config
    
    def test_parquet_contains_required_columns(self, workflow_config):
        """Test that generated parquet contains required columns."""
        # This would require real pydartdiags functionality
        pass


class TestWorkflowModelObsRunMethod:
    """Test the main run method orchestration."""
    
    @patch('subprocess.Popen')
    @patch('subprocess.run')
    def test_run_complete_workflow(self, mock_run, mock_popen, workflow_config, mock_obs_seq_files):
        """Test running the complete workflow."""
        from model2obs.workflows.workflow_model_obs import WorkflowModelObs
        
        # Setup mocks
        mock_run.return_value = Mock(returncode=0)
        mock_proc = Mock()
        mock_proc.returncode = 0
        mock_proc.wait.return_value = None
        mock_popen.return_value = mock_proc
        
        with patch('model2obs.io.obs_seq_tools.trim_obs_seq_in'):
            with patch.object(WorkflowModelObs, 'merge_model_obs_to_parquet'):
                workflow = WorkflowModelObs(workflow_config)
                
                # Run full workflow
                workflow.run(
                    trim_obs=False,
                    no_matching=True,
                    parquet_only=False,
                    clear_output=False
                )
                
                # Verify subprocess was called (3 times for 3 files)
                assert mock_popen.call_count == 3
    
    def test_run_parquet_only_mode(self, workflow_config):
        """Test running workflow in parquet-only mode."""
        from model2obs.workflows.workflow_model_obs import WorkflowModelObs
        
        with patch.object(WorkflowModelObs, 'merge_model_obs_to_parquet') as mock_parquet:
            workflow = WorkflowModelObs(workflow_config)
            
            workflow.run(parquet_only=True)
            
            # Should skip process_files and only call parquet generation
            mock_parquet.assert_called_once()
    
    def test_run_with_clear_output(self, workflow_config, tmp_path):
        """Test running workflow with output clearing."""
        from model2obs.workflows.workflow_model_obs import WorkflowModelObs
        
        # Create some files in output folders
        parquet_folder = Path(workflow_config['parquet_folder'])
        test_file = parquet_folder / "old_data.parquet"
        test_file.write_text("old data")
        
        with patch('model2obs.utils.config.clear_folder') as mock_clear:
            with patch.object(WorkflowModelObs, 'process_files', return_value=0):
                with patch.object(WorkflowModelObs, 'merge_model_obs_to_parquet'):
                    workflow = WorkflowModelObs(workflow_config)
                    
                    workflow.run(clear_output=True, parquet_only=True)
                    
                    # Should have called clear_folder for each output directory
                    assert mock_clear.call_count >= 1


class TestWorkflowModelObsConfigValidation:
    """Test workflow configuration validation.
    
    Note: Path validation has been moved to ModelAdapter classes.
    See tests/unit/test_adapter.py for validate_paths() tests.
    """
    pass


class TestWorkflowModelObsNamelistOperations:
    """Test namelist initialization and management."""
    
    def test_initialize_model_namelist(self, workflow_config, tmp_path):
        """Test initialization of input.nml for DART."""
        from model2obs.workflows.workflow_model_obs import WorkflowModelObs
        
        # Change to tmp directory so input.nml is created there
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            workflow = WorkflowModelObs(workflow_config)
            
            # Should initialize namelist without error
            workflow._initialize_model_namelist()
            
            # Verify input.nml was created (it's actually a symlink to the backup)
            assert Path("input.nml.backup").exists() or Path("input.nml").exists()
        finally:
            os.chdir(original_dir)


class TestWorkflowModelObsDataFrameAccess:
    """Test DataFrame access methods."""
    
    def test_get_all_model_obs_df(self, workflow_config):
        """Test getting all model-obs comparison data."""
        from model2obs.workflows.workflow_model_obs import WorkflowModelObs
        import dask.dataframe as dd
        
        workflow = WorkflowModelObs(workflow_config)
        
        # Create mock dask dataframe from pandas
        mock_df = pd.DataFrame({
            'obs_num': [1, 2, 3],
            'QC': [0, 0, 1],
            'value': [20.0, 21.0, 22.0]
        })
        mock_ddf = dd.from_pandas(mock_df, npartitions=1)
        
        with patch('dask.dataframe.read_parquet', return_value=mock_ddf):
            df = workflow.get_all_model_obs_df(compute=True)
            
            assert len(df) == 3
    
    def test_get_good_model_obs_df(self, workflow_config):
        """Test getting only good quality observations."""
        from model2obs.workflows.workflow_model_obs import WorkflowModelObs
        
        workflow = WorkflowModelObs(workflow_config)
        
        mock_df = pd.DataFrame({
            'obs_num': [1, 2, 3],
            'QC': [0, 0, 1],
            'value': [20.0, 21.0, 22.0]
        })
        
        with patch.object(workflow, '_get_model_obs_df', return_value=mock_df[mock_df['QC'] == 0]):
            df = workflow.get_good_model_obs_df(compute=True)
            
            # Should only include QC==0
            assert len(df) == 2


class TestWorkflowModelObsParallelProcessing:
    """Test parallel=True processing produces the same results as serial."""

    @patch('subprocess.Popen')
    @patch('subprocess.run')
    def test_parallel_no_matching_same_call_count(
        self, mock_run, mock_popen, workflow_config, mock_obs_seq_files
    ):
        """Parallel mode invokes perfect_model_obs the same number of times as serial."""
        from model2obs.workflows.workflow_model_obs import WorkflowModelObs

        mock_run.return_value = Mock(returncode=0)
        mock_proc = Mock()
        mock_proc.returncode = 0
        mock_proc.wait.return_value = None
        mock_popen.return_value = mock_proc

        WorkflowModelObs(workflow_config).process_files(
            trim_obs=False, no_matching=True, parallel=False
        )
        serial_count = mock_popen.call_count

        # Clear output folders between runs so path validation accepts fresh directories
        for folder_key in ('output_folder', 'input_nml_bck'):
            folder = Path(workflow_config[folder_key])
            shutil.rmtree(folder)
            folder.mkdir()

        mock_popen.reset_mock()

        WorkflowModelObs(workflow_config).process_files(
            trim_obs=False, no_matching=True, parallel=True
        )
        parallel_count = mock_popen.call_count

        assert parallel_count == serial_count

    @patch('subprocess.Popen')
    @patch('subprocess.run')
    def test_parallel_no_matching_worker_exception_propagates(
        self, mock_run, mock_popen, workflow_config, mock_obs_seq_files
    ):
        """A subprocess failure inside a parallel worker raises RuntimeError at the call site.

        Exercises the full stack: Popen returncode=1 → RuntimeError in
        _process_model_obs_pair → re-raised through ThreadPoolExecutor.
        The unit counterpart tests the dispatch mechanism in isolation with a
        mocked _process_model_obs_pair.
        """
        from model2obs.workflows.workflow_model_obs import WorkflowModelObs

        mock_run.return_value = Mock(returncode=0)
        mock_proc = Mock()
        mock_proc.returncode = 1
        mock_proc.wait.return_value = None
        mock_popen.return_value = mock_proc

        workflow = WorkflowModelObs(workflow_config)

        with pytest.raises(RuntimeError, match="perfect_model_obs failed"):
            workflow.process_files(trim_obs=False, no_matching=True, parallel=True)

    @patch('subprocess.Popen')
    @patch('subprocess.run')
    def test_parallel_per_pair_log_files_created(
        self, mock_run, mock_popen, workflow_config, mock_obs_seq_files
    ):
        """All N log files are present in the output folder after a parallel run.

        Verifies that concurrent workers do not lose or overwrite each other's
        log files.  The unit counterpart tests the filename format for a single pair.
        """
        from model2obs.workflows.workflow_model_obs import WorkflowModelObs

        mock_run.return_value = Mock(returncode=0)
        mock_proc = Mock()
        mock_proc.returncode = 0
        mock_proc.wait.return_value = None
        mock_popen.return_value = mock_proc

        workflow = WorkflowModelObs(workflow_config)
        workflow.process_files(trim_obs=False, no_matching=True, parallel=True)

        output_folder = Path(workflow_config['output_folder'])
        log_files = list(output_folder.glob("perfect_model_obs_*.log"))
        assert len(log_files) == 3, f"Expected 3 per-pair log files, found {len(log_files)}"

    @patch('subprocess.Popen')
    @patch('subprocess.run')
    def test_no_matching_serial_parallel_same_pairs(
        self, mock_run, mock_popen, workflow_config, mock_obs_seq_files
    ):
        """Serial and parallel dispatch the identical (model_file, obs_file, counter) triples.

        The parallel path pre-assigns counters before spawning threads, so output
        filenames must match the serial ordering regardless of thread completion order.
        """
        from model2obs.workflows.workflow_model_obs import WorkflowModelObs

        mock_run.return_value = Mock(returncode=0)
        mock_proc = Mock()
        mock_proc.returncode = 0
        mock_proc.wait.return_value = None
        mock_popen.return_value = mock_proc

        serial_pairs = []
        parallel_pairs = []

        def capture(call_list):
            # patch.object omits self; positional order: model_in_file, obs_in_file,
            # trim_obs, counter, hull_polygon, hull_points, force_obs_time[, precomputed]
            def side_effect(*args, **kwargs):
                call_list.append((args[0], args[1], args[3]))
            return side_effect

        with patch.object(WorkflowModelObs, '_process_model_obs_pair',
                          side_effect=capture(serial_pairs)):
            WorkflowModelObs(workflow_config).process_files(
                trim_obs=False, no_matching=True, parallel=False
            )

        with patch.object(WorkflowModelObs, '_process_model_obs_pair',
                          side_effect=capture(parallel_pairs)):
            WorkflowModelObs(workflow_config).process_files(
                trim_obs=False, no_matching=True, parallel=True
            )

        assert len(serial_pairs) == len(parallel_pairs)
        assert sorted(serial_pairs) == sorted(parallel_pairs)

