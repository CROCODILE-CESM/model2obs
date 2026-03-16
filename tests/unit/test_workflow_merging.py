"""Unit tests for workflow_model_obs data merging methods.

Tests cover hash generation, DataFrame merging, diagnostic calculations,
and parquet generation for model-observation comparison workflows.
"""

from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import warnings

import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd

from model2obs.workflows import workflow_model_obs


class TestMergePairToParquet:
    """Test suite for _merge_pair_to_parquet() method."""
    
    @pytest.fixture
    def workflow(self, tmp_path):
        """Create a WorkflowModelObs instance with minimal config."""
        config = {
            'ocean_model': "mom6",
            'model_files_folder': str(tmp_path / "model"),
            'obs_seq_in_folder': str(tmp_path / "obs"),
            'output_folder': str(tmp_path / "output"),
            'template_file': str(tmp_path / "template.nc"),
            'static_file': str(tmp_path / "static.nc"),
            'ocean_geometry': str(tmp_path / "geometry.nc"),
            'perfect_model_obs_dir': str(tmp_path / "perfect"),
            'parquet_folder': str(tmp_path / "parquet"),
            'input_nml_bck': str(tmp_path / "nml_backup"),
            'trimmed_obs_folder': str(tmp_path / "trimmed"),
        }
        return workflow_model_obs.WorkflowModelObs(config)
    
    @pytest.fixture
    def mock_obs_dataframes(self):
        """Create mock dataframes for testing merge operations."""
        # Perfect model output
        model_df = pd.DataFrame({
            'obs_num': [1, 2, 3, 4],
            'longitude': [10.0, 11.0, 12.0, 13.0],
            'latitude': [40.0, 41.0, 42.0, 43.0],
            'vertical': [0.0, 10.0, 20.0, 30.0],
            'time': pd.to_datetime(['2020-01-01'] * 4),
            'type': ['TEMP'] * 4,
            'days': [0, 0, 0, 0],
            'seconds': [0, 3600, 7200, 10800],
            'truth': [20.0, 21.0, 22.0, 23.0],  # Model interpolated values
            'truth_QC': [0, 0, 0, 1],  # QC flags (1 = failed)
            'obs_err_var': [0.01, 0.01, 0.01, 0.01]
        })
        
        # Original observations
        obs_df = pd.DataFrame({
            'obs_num': [1, 2, 3, 4],
            'longitude': [10.0, 11.0, 12.0, 13.0],
            'latitude': [40.0, 41.0, 42.0, 43.0],
            'vertical': [0.0, 10.0, 20.0, 30.0],
            'time': pd.to_datetime(['2020-01-01'] * 4),
            'type': ['TEMP'] * 4,
            'days': [0, 0, 0, 0],
            'seconds': [0, 3600, 7200, 10800],
            'observation': [20.1, 21.2, 21.9, 23.5],  # Observed values
            'obs_err_var': [0.01, 0.01, 0.01, 0.01]
        })
        
        return {'model': model_df, 'obs': obs_df}
    
    def test_hash_generation(self, workflow, mock_obs_dataframes):
        """Test hash generation for observation matching."""
        obs_df = mock_obs_dataframes['obs'].copy()
        
        concat = obs_df[['obs_num', 'seconds', 'days']].astype(str).agg('-'.join, axis=1)
        obs_df['hash'] = pd.util.hash_pandas_object(concat, index=False).astype('int64')
        
        assert 'hash' in obs_df.columns
        assert obs_df['hash'].dtype == np.int64
        assert len(obs_df['hash'].unique()) == len(obs_df)
    
    def test_hash_consistency(self, workflow, mock_obs_dataframes):
        """Test that identical rows generate identical hashes."""
        obs_df = mock_obs_dataframes['obs'].copy()
        
        concat1 = obs_df[['obs_num', 'seconds', 'days']].astype(str).agg('-'.join, axis=1)
        hash1 = pd.util.hash_pandas_object(concat1, index=False).astype('int64')
        
        concat2 = obs_df[['obs_num', 'seconds', 'days']].astype(str).agg('-'.join, axis=1)
        hash2 = pd.util.hash_pandas_object(concat2, index=False).astype('int64')
        
        np.testing.assert_array_equal(hash1, hash2)
    
    @patch('model2obs.workflows.workflow_model_obs.obsq.ObsSequence')
    def test_merge_pair_basic(self, mock_obsq, workflow, mock_obs_dataframes, tmp_path):
        """Test basic merge of observation pairs."""
        model_df = mock_obs_dataframes['model'].copy()
        obs_df = mock_obs_dataframes['obs'].copy()
        
        mock_model = Mock()
        mock_model.df = model_df
        mock_obs = Mock()
        mock_obs.df = obs_df
        
        mock_obsq.side_effect = [mock_model, mock_obs]
        
        parquet_path = tmp_path / "parquet"
        parquet_path.mkdir()
        
        workflow._merge_pair_to_parquet(
            "perf_obs.out",
            "obs_seq.in",
            str(parquet_path)
        )
        
        parquet_files = list(parquet_path.glob("*.parquet"))
        assert len(parquet_files) > 0
    
    @patch('model2obs.workflows.workflow_model_obs.obsq.ObsSequence')
    def test_diagnostic_columns_calculated(self, mock_obsq, workflow, mock_obs_dataframes, tmp_path):
        """Test that all diagnostic columns are calculated correctly."""
        model_df = mock_obs_dataframes['model'].copy()
        obs_df = mock_obs_dataframes['obs'].copy()
        
        mock_model = Mock()
        mock_model.df = model_df
        mock_obs = Mock()
        mock_obs.df = obs_df
        
        mock_obsq.side_effect = [mock_model, mock_obs]
        
        parquet_path = tmp_path / "parquet"
        parquet_path.mkdir()
        
        workflow._merge_pair_to_parquet(
            "perf_obs.out",
            "obs_seq.in",
            str(parquet_path)
        )
        
        result = dd.read_parquet(str(parquet_path)).compute()
        
        assert 'difference' in result.columns
        assert 'abs_difference' in result.columns
        assert 'squared_difference' in result.columns
        assert 'normalized_difference' in result.columns
        assert 'log_likelihood' in result.columns
    
    @patch('model2obs.workflows.workflow_model_obs.obsq.ObsSequence')
    def test_diagnostic_difference_calculation(self, mock_obsq, workflow, mock_obs_dataframes, tmp_path):
        """Test difference = obs - model calculation."""
        model_df = mock_obs_dataframes['model'].copy()
        obs_df = mock_obs_dataframes['obs'].copy()
        
        mock_model = Mock()
        mock_model.df = model_df
        mock_obs = Mock()
        mock_obs.df = obs_df
        
        mock_obsq.side_effect = [mock_model, mock_obs]
        
        parquet_path = tmp_path / "parquet"
        parquet_path.mkdir()
        
        workflow._merge_pair_to_parquet(
            "perf_obs.out",
            "obs_seq.in",
            str(parquet_path)
        )
        
        result = dd.read_parquet(str(parquet_path)).compute()
        
        expected_diff = result['obs'] - result['interpolated_model']
        np.testing.assert_allclose(result['difference'], expected_diff, rtol=1e-10)
    
    @patch('model2obs.workflows.workflow_model_obs.obsq.ObsSequence')
    def test_diagnostic_normalized_difference(self, mock_obsq, workflow, mock_obs_dataframes, tmp_path):
        """Test normalized_difference = difference / sqrt(obs_err_var)."""
        model_df = mock_obs_dataframes['model'].copy()
        obs_df = mock_obs_dataframes['obs'].copy()
        
        mock_model = Mock()
        mock_model.df = model_df
        mock_obs = Mock()
        mock_obs.df = obs_df
        
        mock_obsq.side_effect = [mock_model, mock_obs]
        
        parquet_path = tmp_path / "parquet"
        parquet_path.mkdir()
        
        workflow._merge_pair_to_parquet(
            "perf_obs.out",
            "obs_seq.in",
            str(parquet_path)
        )
        
        result = dd.read_parquet(str(parquet_path)).compute()
        
        expected_norm = result['difference'] / np.sqrt(result['obs_err_var'])
        np.testing.assert_allclose(result['normalized_difference'], expected_norm, rtol=1e-10)
    
    @patch('model2obs.workflows.workflow_model_obs.obsq.ObsSequence')
    def test_diagnostic_log_likelihood(self, mock_obsq, workflow, mock_obs_dataframes, tmp_path):
        """Test log_likelihood calculation."""
        model_df = mock_obs_dataframes['model'].copy()
        obs_df = mock_obs_dataframes['obs'].copy()
        
        mock_model = Mock()
        mock_model.df = model_df
        mock_obs = Mock()
        mock_obs.df = obs_df
        
        mock_obsq.side_effect = [mock_model, mock_obs]
        
        parquet_path = tmp_path / "parquet"
        parquet_path.mkdir()
        
        workflow._merge_pair_to_parquet(
            "perf_obs.out",
            "obs_seq.in",
            str(parquet_path)
        )
        
        result = dd.read_parquet(str(parquet_path)).compute()
        
        expected_ll = -0.5 * (
            result['difference'] ** 2 / result['obs_err_var'] +
            np.log(2 * np.pi * result['obs_err_var'])
        )
        np.testing.assert_allclose(result['log_likelihood'], expected_ll, rtol=1e-10)
    
    @patch('model2obs.workflows.workflow_model_obs.obsq.ObsSequence')
    def test_column_ordering(self, mock_obsq, workflow, mock_obs_dataframes, tmp_path):
        """Test that columns are ordered correctly in output."""
        model_df = mock_obs_dataframes['model'].copy()
        obs_df = mock_obs_dataframes['obs'].copy()
        
        mock_model = Mock()
        mock_model.df = model_df
        mock_obs = Mock()
        mock_obs.df = obs_df
        
        mock_obsq.side_effect = [mock_model, mock_obs]
        
        parquet_path = tmp_path / "parquet"
        parquet_path.mkdir()
        
        workflow._merge_pair_to_parquet(
            "perf_obs.out",
            "obs_seq.in",
            str(parquet_path)
        )
        
        result = dd.read_parquet(str(parquet_path)).compute()
        
        expected_first_cols = [
            'time', 'longitude', 'latitude', 'vertical', 'type',
            'interpolated_model', 'obs', 'obs_err_var'
        ]
        actual_first_cols = result.columns[:len(expected_first_cols)].tolist()
        assert actual_first_cols == expected_first_cols
    
    @patch('model2obs.workflows.workflow_model_obs.obsq.ObsSequence')
    def test_multiple_observation_columns_raises_error(self, mock_obsq, workflow, tmp_path):
        """Test that multiple observation columns raise ValueError."""
        model_df = pd.DataFrame({
            'obs_num': [1],
            'longitude': [10.0],
            'observation': [20.0],
            'TEMP_observation': [20.1],  # Duplicate observation column
            'truth': [20.0],
            'truth_QC': [0]
        })
        
        mock_model = Mock()
        mock_model.df = model_df
        mock_obs = Mock()
        mock_obs.df = model_df.copy()
        
        mock_obsq.side_effect = [mock_model, mock_obs]
        
        parquet_path = tmp_path / "parquet"
        parquet_path.mkdir()
        
        with pytest.raises(ValueError, match="More than one observation columns found"):
            workflow._merge_pair_to_parquet(
                "perf_obs.out",
                "obs_seq.in",
                str(parquet_path)
            )
    
    @patch('model2obs.workflows.workflow_model_obs.obsq.ObsSequence')
    def test_multiple_qc_columns_raises_error(self, mock_obsq, workflow, tmp_path):
        """Test that multiple QC columns raise ValueError."""
        obs_df = pd.DataFrame({
            'obs_num': [1],
            'observation': [20.0],
        })
        
        model_df = pd.DataFrame({
            'obs_num': [1],
            'truth': [20.0],
            'truth_QC': [0],
            'model_QC': [0],  # Duplicate QC column
        })
        
        mock_model = Mock()
        mock_model.df = model_df
        mock_obs = Mock()
        mock_obs.df = obs_df
        
        mock_obsq.side_effect = [mock_model, mock_obs]
        
        parquet_path = tmp_path / "parquet"
        parquet_path.mkdir()
        
        with pytest.raises(ValueError, match="More than one QC column found"):
            workflow._merge_pair_to_parquet(
                "perf_obs.out",
                "obs_seq.in",
                str(parquet_path)
            )
    
    @patch('model2obs.workflows.workflow_model_obs.obsq.ObsSequence')
    def test_reference_column_mismatch_raises_error(self, mock_obsq, workflow, tmp_path):
        """Test that mismatched reference columns raise ValueError."""
        model_df = pd.DataFrame({
            'obs_num': [1, 2],
            'longitude': [10.0, 11.0],
            'latitude': [40.0, 41.0],
            'vertical': [0.0, 10.0],
            'time': pd.to_datetime(['2020-01-01', '2020-01-01']),
            'type': ['TEMP', 'TEMP'],
            'days': [0, 0],
            'seconds': [0, 3600],
            'truth': [20.0, 21.0],
            'truth_QC': [0, 0],
            'obs_err_var': [0.01, 0.01]
        })
        
        obs_df = model_df.copy()
        obs_df['longitude'] = [10.0, 11.5]  # Different longitude for 2nd obs
        obs_df['observation'] = [20.1, 21.2]
        
        mock_model = Mock()
        mock_model.df = model_df
        mock_obs = Mock()
        mock_obs.df = obs_df
        
        mock_obsq.side_effect = [mock_model, mock_obs]
        
        parquet_path = tmp_path / "parquet"
        parquet_path.mkdir()
        
        with pytest.raises(ValueError, match="not identical"):
            workflow._merge_pair_to_parquet(
                "perf_obs.out",
                "obs_seq.in",
                str(parquet_path)
            )
    
    @patch('model2obs.workflows.workflow_model_obs.obsq.ObsSequence')
    def test_sorting_by_time_position_depth(self, mock_obsq, workflow, mock_obs_dataframes, tmp_path):
        """Test that results are sorted by time, position, depth."""
        model_df = mock_obs_dataframes['model'].copy()
        obs_df = mock_obs_dataframes['obs'].copy()
        
        model_df['time'] = pd.to_datetime(['2020-01-02', '2020-01-01', '2020-01-03', '2020-01-01'])
        model_df['longitude'] = [11.0, 10.0, 12.0, 10.5]
        obs_df['time'] = model_df['time'].copy()
        obs_df['longitude'] = model_df['longitude'].copy()
        
        mock_model = Mock()
        mock_model.df = model_df
        mock_obs = Mock()
        mock_obs.df = obs_df
        
        mock_obsq.side_effect = [mock_model, mock_obs]
        
        parquet_path = tmp_path / "parquet"
        parquet_path.mkdir()
        
        workflow._merge_pair_to_parquet(
            "perf_obs.out",
            "obs_seq.in",
            str(parquet_path)
        )
        
        result = dd.read_parquet(str(parquet_path)).compute()
        
        assert result['time'].is_monotonic_increasing


class TestMergeModelObsToParquet:
    """Test suite for merge_model_obs_to_parquet() method."""
    
    @pytest.fixture
    def workflow_with_files(self, tmp_path):
        """Create workflow with mock file structure."""
        output_folder = tmp_path / "output"
        output_folder.mkdir()
        parquet_folder = tmp_path / "parquet"
        parquet_folder.mkdir()
        obs_folder = tmp_path / "obs"
        obs_folder.mkdir()
        
        (output_folder / "obs_seq_001.out").write_text("mock")
        (output_folder / "obs_seq_002.out").write_text("mock")
        (obs_folder / "obs_seq_001.in").write_text("mock")
        (obs_folder / "obs_seq_002.in").write_text("mock")
        
        config = {
            'ocean_model': 'mom6',
            'model_files_folder': str(tmp_path / "model"),
            'obs_seq_in_folder': str(obs_folder),
            'output_folder': str(output_folder),
            'template_file': str(tmp_path / "template.nc"),
            'static_file': str(tmp_path / "static.nc"),
            'ocean_geometry': str(tmp_path / "geometry.nc"),
            'perfect_model_obs_dir': str(tmp_path / "perfect"),
            'parquet_folder': str(parquet_folder),
            'input_nml_bck': str(tmp_path / "nml_backup"),
            'trimmed_obs_folder': str(tmp_path / "trimmed"),
        }
        return workflow_model_obs.WorkflowModelObs(config)
    
    @patch('model2obs.workflows.workflow_model_obs.WorkflowModelObs._merge_pair_to_parquet')
    @patch('model2obs.workflows.workflow_model_obs.WorkflowModelObs._set_model_obs_df')
    @patch('model2obs.workflows.workflow_model_obs.WorkflowModelObs.get_all_model_obs_df')
    @patch('model2obs.workflows.workflow_model_obs.WorkflowModelObs.get_good_model_obs_df')
    @patch('model2obs.workflows.workflow_model_obs.WorkflowModelObs.get_failed_model_obs_df')
    @patch('model2obs.workflows.workflow_model_obs.dd.read_parquet')
    def test_merge_creates_parquet_files(self, mock_read, mock_failed, mock_good, mock_all,
                                        mock_set_df, mock_merge, workflow_with_files, capsys):
        """Test that merge_model_obs_to_parquet creates parquet files."""
        mock_ddf = Mock()
        mock_ddf.repartition.return_value = mock_ddf
        mock_read.return_value = mock_ddf
        
        mock_all.return_value = pd.DataFrame({'a': range(10)})
        mock_good.return_value = pd.DataFrame({'a': range(8)})
        mock_failed.return_value = pd.DataFrame({'a': range(2)})
        
        workflow_with_files.merge_model_obs_to_parquet(trim_obs=False)
        
        assert mock_merge.call_count == 2
        captured = capsys.readouterr()
        assert "Total number of obs" in captured.out
        assert "Succesfull interpolations" in captured.out
        assert "Failed interpolations" in captured.out
    
    @patch('model2obs.workflows.workflow_model_obs.WorkflowModelObs._merge_pair_to_parquet')
    @patch('model2obs.workflows.workflow_model_obs.WorkflowModelObs._set_model_obs_df')
    @patch('model2obs.workflows.workflow_model_obs.WorkflowModelObs.get_all_model_obs_df')
    @patch('model2obs.workflows.workflow_model_obs.WorkflowModelObs.get_good_model_obs_df')
    @patch('model2obs.workflows.workflow_model_obs.WorkflowModelObs.get_failed_model_obs_df')
    @patch('model2obs.workflows.workflow_model_obs.dd.read_parquet')
    def test_merge_uses_trimmed_obs_folder(self, mock_read, mock_failed, mock_good, mock_all,
                                          mock_set_df, mock_merge, tmp_path):
        """Test that trim_obs=True uses trimmed_obs_folder."""
        trimmed_folder = tmp_path / "trimmed"
        trimmed_folder.mkdir()
        (trimmed_folder / "obs_seq_001.in").write_text("mock")
        
        output_folder = tmp_path / "output"
        output_folder.mkdir()
        (output_folder / "obs_seq_001.out").write_text("mock")
        
        config = {
            'ocean_model': 'mom6',
            'model_files_folder': str(tmp_path / "model"),
            'obs_seq_in_folder': str(tmp_path / "obs"),
            'trimmed_obs_folder': str(trimmed_folder),
            'output_folder': str(output_folder),
            'template_file': str(tmp_path / "template.nc"),
            'static_file': str(tmp_path / "static.nc"),
            'ocean_geometry': str(tmp_path / "geometry.nc"),
            'perfect_model_obs_dir': str(tmp_path / "perfect"),
            'parquet_folder': str(tmp_path / "parquet"),
            'input_nml_bck': str(tmp_path / "nml_backup"),
        }
        workflow = workflow_model_obs.WorkflowModelObs(config)
        
        mock_ddf = Mock()
        mock_ddf.repartition.return_value = mock_ddf
        mock_read.return_value = mock_ddf
        
        mock_all.return_value = pd.DataFrame({'a': range(10)})
        mock_good.return_value = pd.DataFrame({'a': range(8)})
        mock_failed.return_value = pd.DataFrame({'a': range(2)})
        
        workflow.merge_model_obs_to_parquet(trim_obs=True)
        
        call_args = mock_merge.call_args_list[0][0]
        assert "trimmed" in call_args[1]
    
    @patch('model2obs.workflows.workflow_model_obs.WorkflowModelObs._merge_pair_to_parquet')
    @patch('model2obs.workflows.workflow_model_obs.WorkflowModelObs.get_all_model_obs_df')
    @patch('model2obs.workflows.workflow_model_obs.WorkflowModelObs.get_good_model_obs_df')
    @patch('model2obs.workflows.workflow_model_obs.WorkflowModelObs.get_failed_model_obs_df')
    @patch('model2obs.workflows.workflow_model_obs.WorkflowModelObs._set_model_obs_df')
    @patch('model2obs.workflows.workflow_model_obs.dd.read_parquet')
    def test_merge_prints_statistics(self, mock_read, mock_set_df, mock_failed, mock_good, mock_all, 
                                     mock_merge, workflow_with_files, capsys):
        """Test that merge prints observation statistics."""
        mock_ddf = Mock()
        mock_ddf.repartition.return_value = mock_ddf
        mock_read.return_value = mock_ddf
        
        mock_all.return_value = pd.DataFrame({'a': range(100)})
        mock_good.return_value = pd.DataFrame({'a': range(80)})
        mock_failed.return_value = pd.DataFrame({'a': range(20)})
        
        workflow_with_files.merge_model_obs_to_parquet(trim_obs=False)
        
        captured = capsys.readouterr()
        assert "Total number of observations" in captured.out
        assert "Succesfull interpolations" in captured.out
        assert "Failed interpolations" in captured.out
    
    @patch('model2obs.workflows.workflow_model_obs.WorkflowModelObs._merge_pair_to_parquet')
    @patch('model2obs.workflows.workflow_model_obs.WorkflowModelObs._set_model_obs_df')
    @patch('model2obs.workflows.workflow_model_obs.WorkflowModelObs.get_all_model_obs_df')
    @patch('model2obs.workflows.workflow_model_obs.WorkflowModelObs.get_good_model_obs_df')
    @patch('model2obs.workflows.workflow_model_obs.WorkflowModelObs.get_failed_model_obs_df')
    @patch('model2obs.workflows.workflow_model_obs.dd.read_parquet')
    def test_merge_cleans_up_tmp_folder(self, mock_read, mock_failed, mock_good, mock_all,
                                       mock_set_df, mock_merge, workflow_with_files, tmp_path):
        """Test that temporary parquet folder is cleaned up."""
        mock_ddf = Mock()
        mock_ddf.repartition.return_value = mock_ddf
        mock_read.return_value = mock_ddf
        
        mock_all.return_value = pd.DataFrame({'a': range(10)})
        mock_good.return_value = pd.DataFrame({'a': range(8)})
        mock_failed.return_value = pd.DataFrame({'a': range(2)})
        
        workflow_with_files.merge_model_obs_to_parquet(trim_obs=False)
        
        tmp_folder = Path(workflow_with_files.config['parquet_folder']) / "tmp"
        assert not tmp_folder.exists()
    
    @patch('model2obs.workflows.workflow_model_obs.WorkflowModelObs._merge_pair_to_parquet')
    @patch('model2obs.workflows.workflow_model_obs.WorkflowModelObs._set_model_obs_df')
    @patch('model2obs.workflows.workflow_model_obs.WorkflowModelObs.get_all_model_obs_df')
    @patch('model2obs.workflows.workflow_model_obs.WorkflowModelObs.get_good_model_obs_df')
    @patch('model2obs.workflows.workflow_model_obs.WorkflowModelObs.get_failed_model_obs_df')
    @patch('model2obs.workflows.workflow_model_obs.dd.read_parquet')
    def test_merge_repartitions_output(self, mock_read, mock_failed, mock_good, mock_all,
                                       mock_set_df, mock_merge, workflow_with_files):
        """Test that output is repartitioned to 300MB chunks."""
        mock_ddf = Mock()
        mock_ddf.repartition.return_value = mock_ddf
        mock_read.return_value = mock_ddf
        
        mock_all.return_value = pd.DataFrame({'a': range(10)})
        mock_good.return_value = pd.DataFrame({'a': range(8)})
        mock_failed.return_value = pd.DataFrame({'a': range(2)})
        
        workflow_with_files.merge_model_obs_to_parquet(trim_obs=False)
        
        mock_ddf.repartition.assert_called_once_with(partition_size="300MB")
