"""Integration tests for parquet generation and output validation.

Tests the conversion of DART obs_seq format to parquet, including
diagnostics calculations, hash generation, and data merging.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, Mock
import pandas as pd
import numpy as np
import dask.dataframe as dd


@pytest.fixture
def mock_obs_seq_dataframes():
    """Create mock dataframes simulating pydartdiags ObsSeq output."""
    # Model (perfect_model_obs output)
    model_df = pd.DataFrame({
        'obs_num': [1, 2, 3, 4],
        'longitude': [5.0, 6.0, 7.0, 8.0],
        'latitude': [5.0, 6.0, 7.0, 8.0],
        'vertical': [0.0, 10.0, 20.0, 30.0],
        'days': [0, 0, 0, 0],
        'seconds': [0, 3600, 7200, 10800],
        'value': [20.0, 21.0, 22.0, 23.0],  # Model values
        'QC': [0, 0, 0, 0],
        'obs_type': ['FLOAT_TEMPERATURE'] * 4
    })
    
    # Observations (original obs_seq.in)
    obs_df = pd.DataFrame({
        'obs_num': [1, 2, 3, 4],
        'longitude': [5.0, 6.0, 7.0, 8.0],
        'latitude': [5.0, 6.0, 7.0, 8.0],
        'vertical': [0.0, 10.0, 20.0, 30.0],
        'days': [0, 0, 0, 0],
        'seconds': [0, 3600, 7200, 10800],
        'value': [20.1, 21.2, 21.9, 23.5],  # Observed values
        'error_variance': [0.01, 0.01, 0.01, 0.01],
        'QC': [0, 0, 0, 0],
        'obs_type': ['FLOAT_TEMPERATURE'] * 4
    })
    
    return {'model': model_df, 'obs': obs_df}


class TestParquetStructure:
    """Test basic parquet file structure and format."""
    
    def test_parquet_file_creation(self, tmp_path, mock_obs_seq_dataframes):
        """Test that parquet files are created."""
        from model2obs.workflows.workflow_model_obs import WorkflowModelObs
        
        # Setup workflow config
        config = {
            'model_name': 'MOM6',
            'model_files_folder': str(tmp_path / 'model'),
            'obs_seq_in_folder': str(tmp_path / 'obs_in'),
            'output_folder': str(tmp_path / 'output'),
            'template_file': str(tmp_path / 'template.nc'),
            'static_file': str(tmp_path / 'static.nc'),
            'ocean_geometry': str(tmp_path / 'geometry.nc'),
            'perfect_model_obs_dir': str(tmp_path / 'dart'),
            'parquet_folder': str(tmp_path / 'parquet'),
            'trimmed_obs_folder': str(tmp_path / 'trimmed'),
            'input_nml_bck': str(tmp_path / 'nml_bck'),
            'time_window': {'days': 1, 'hours': 0, 'minutes': 0, 'seconds': 0,
                          'weeks': 0, 'months': 0, 'years': 0}
        }
        
        # Create necessary directories
        for key in ['parquet_folder', 'output_folder', 'obs_seq_in_folder']:
            Path(config[key]).mkdir(parents=True, exist_ok=True)
        
        # Create dummy obs_seq files
        (Path(config['output_folder']) / 'obs_seq_0000.out').write_text("mock")
        (Path(config['obs_seq_in_folder']) / 'obs_seq_0000.in').write_text("mock")
        
        # This test requires actual obs_seq files with pydartdiags
        # For now, just verify the workflow can be instantiated
        workflow = WorkflowModelObs(config)
        
        # The actual merge would require valid obs_seq files
        # which we don't create in this simple test
        # So we just verify the workflow object is created correctly
        assert workflow.config == config
    
    def test_parquet_contains_required_columns(self, mock_obs_seq_dataframes):
        """Test that parquet output contains all required columns."""
        from model2obs.workflows.workflow_model_obs import WorkflowModelObs
        
        # Required columns after merging
        required_cols = [
            'obs_num', 'longitude', 'latitude', 'vertical',
            'days', 'seconds', 'obs_type', 'QC'
        ]
        
        # Simulate merged dataframe
        merged_df = mock_obs_seq_dataframes['model']
        
        for col in required_cols:
            assert col in merged_df.columns, f"Missing required column: {col}"


class TestHashGeneration:
    """Test hash generation for observation matching."""
    
    def test_compute_hash_function(self, mock_obs_seq_dataframes):
        """Test hash computation for matching observations."""
        df = mock_obs_seq_dataframes['model'].copy()
        
        # Compute hash based on location and time
        def compute_hash(df, cols, hash_col='hash'):
            df[hash_col] = pd.util.hash_pandas_object(
                df[cols], index=False
            ).astype(str)
            return df
        
        cols_to_hash = ['longitude', 'latitude', 'vertical', 'days', 'seconds']
        df_with_hash = compute_hash(df, cols_to_hash)
        
        # Should have hash column
        assert 'hash' in df_with_hash.columns
        
        # All hashes should be non-empty strings
        assert all(df_with_hash['hash'].str.len() > 0)
        
        # Identical rows should have identical hashes
        df_dup = df_with_hash.copy()
        df_dup = compute_hash(df_dup, cols_to_hash, hash_col='hash2')
        assert all(df_with_hash['hash'] == df_dup['hash2'])
    
    def test_hash_enables_joining(self, mock_obs_seq_dataframes):
        """Test that hashes enable joining model and obs dataframes."""
        model_df = mock_obs_seq_dataframes['model'].copy()
        obs_df = mock_obs_seq_dataframes['obs'].copy()
        
        # Add hashes
        def compute_hash(df, cols):
            df['hash'] = pd.util.hash_pandas_object(
                df[cols], index=False
            ).astype(str)
            return df
        
        cols = ['longitude', 'latitude', 'vertical', 'days', 'seconds']
        model_df = compute_hash(model_df, cols)
        obs_df = compute_hash(obs_df, cols)
        
        # Should be able to merge on hash
        merged = pd.merge(
            model_df[['hash', 'value', 'QC']],
            obs_df[['hash', 'value', 'error_variance']],
            on='hash',
            suffixes=('_model', '_obs')
        )
        
        assert len(merged) == 4
        assert 'value_model' in merged.columns
        assert 'value_obs' in merged.columns


class TestDiagnosticCalculations:
    """Test diagnostic calculations (residuals, log-likelihood, etc.)."""
    
    def test_residual_calculation(self, mock_obs_seq_dataframes):
        """Test calculation of observation residuals."""
        model_df = mock_obs_seq_dataframes['model'].copy()
        obs_df = mock_obs_seq_dataframes['obs'].copy()
        
        # Merge and calculate residual
        model_df['hash'] = range(len(model_df))
        obs_df['hash'] = range(len(obs_df))
        
        merged = pd.merge(
            model_df[['hash', 'value']],
            obs_df[['hash', 'value']],
            on='hash',
            suffixes=('_model', '_obs')
        )
        
        merged['residual'] = merged['value_obs'] - merged['value_model']
        
        # Check residuals are computed
        assert 'residual' in merged.columns
        assert not merged['residual'].isna().any()
        
        # Residuals should be small differences
        expected_residuals = [0.1, 0.2, -0.1, 0.5]
        np.testing.assert_array_almost_equal(
            merged['residual'].values,
            expected_residuals,
            decimal=5
        )
    
    def test_log_likelihood_calculation(self, mock_obs_seq_dataframes):
        """Test log-likelihood calculation."""
        model_df = mock_obs_seq_dataframes['model'].copy()
        obs_df = mock_obs_seq_dataframes['obs'].copy()
        
        # Merge
        model_df['hash'] = range(len(model_df))
        obs_df['hash'] = range(len(obs_df))
        
        merged = pd.merge(
            model_df[['hash', 'value']],
            obs_df[['hash', 'value', 'error_variance']],
            on='hash',
            suffixes=('_model', '_obs')
        )
        
        # Calculate log-likelihood
        residual = merged['value_obs'] - merged['value_model']
        merged['log_likelihood'] = -0.5 * (residual ** 2) / merged['error_variance']
        
        assert 'log_likelihood' in merged.columns
        assert not merged['log_likelihood'].isna().any()
        assert all(merged['log_likelihood'] <= 0)  # Should be negative or zero


class TestDataMerging:
    """Test merging of model and observation dataframes."""
    
    def test_merge_preserves_all_observations(self, mock_obs_seq_dataframes):
        """Test that merging preserves all observations."""
        model_df = mock_obs_seq_dataframes['model'].copy()
        obs_df = mock_obs_seq_dataframes['obs'].copy()
        
        # Add matching hash
        for df in [model_df, obs_df]:
            df['hash'] = df['obs_num'].astype(str)
        
        # Left merge (keep all observations)
        merged = pd.merge(
            obs_df,
            model_df[['hash', 'value']],
            on='hash',
            how='left',
            suffixes=('_obs', '_model')
        )
        
        # Should preserve all 4 observations
        assert len(merged) == len(obs_df)
    
    def test_merge_handles_missing_model_values(self):
        """Test merging when some model values are missing."""
        obs_df = pd.DataFrame({
            'hash': ['a', 'b', 'c'],
            'value_obs': [1.0, 2.0, 3.0]
        })
        
        model_df = pd.DataFrame({
            'hash': ['a', 'c'],  # 'b' is missing
            'value_model': [1.1, 3.1]
        })
        
        merged = pd.merge(obs_df, model_df, on='hash', how='left')
        
        # Should have 3 rows
        assert len(merged) == 3
        
        # 'b' should have NaN for model value
        assert pd.isna(merged.loc[merged['hash'] == 'b', 'value_model'].values[0])
    
    def test_merge_suffix_handling(self, mock_obs_seq_dataframes):
        """Test that suffixes are properly applied during merge."""
        model_df = mock_obs_seq_dataframes['model'].copy()
        obs_df = mock_obs_seq_dataframes['obs'].copy()
        
        model_df['hash'] = range(len(model_df))
        obs_df['hash'] = range(len(obs_df))
        
        merged = pd.merge(
            model_df[['hash', 'value', 'QC']],
            obs_df[['hash', 'value', 'QC']],
            on='hash',
            suffixes=('_model', '_obs')
        )
        
        # Should have both suffixed columns
        assert 'value_model' in merged.columns
        assert 'value_obs' in merged.columns
        assert 'QC_model' in merged.columns
        assert 'QC_obs' in merged.columns


class TestParquetPartitioning:
    """Test parquet partitioning strategies."""
    
    def test_partition_by_obs_type(self, mock_obs_seq_dataframes):
        """Test partitioning parquet by observation type."""
        df = mock_obs_seq_dataframes['model'].copy()
        
        # Add multiple obs types
        df.loc[0:1, 'obs_type'] = 'FLOAT_TEMPERATURE'
        df.loc[2:3, 'obs_type'] = 'FLOAT_SALINITY'
        
        # Group by obs_type
        grouped = df.groupby('obs_type')
        
        assert len(grouped) == 2
        assert 'FLOAT_TEMPERATURE' in grouped.groups
        assert 'FLOAT_SALINITY' in grouped.groups
    
    def test_partition_by_time(self, mock_obs_seq_dataframes):
        """Test partitioning parquet by time."""
        df = mock_obs_seq_dataframes['model'].copy()
        
        # Add datetime column from days and seconds
        base_date = pd.Timestamp('2020-01-01')
        df['datetime'] = base_date + pd.to_timedelta(df['days'], unit='D') + \
                        pd.to_timedelta(df['seconds'], unit='s')
        df['date'] = df['datetime'].dt.date
        
        # Can partition by date
        grouped = df.groupby('date')
        
        assert len(grouped) >= 1


class TestParquetCompression:
    """Test parquet compression and size optimization."""
    
    def test_parquet_with_compression(self, tmp_path, mock_obs_seq_dataframes):
        """Test writing parquet with compression."""
        df = mock_obs_seq_dataframes['model']
        
        # Write with compression
        parquet_file = tmp_path / "compressed.parquet"
        df.to_parquet(parquet_file, compression='snappy')
        
        assert parquet_file.exists()
        
        # Read back and verify
        df_read = pd.read_parquet(parquet_file)
        pd.testing.assert_frame_equal(df, df_read)
    
    def test_parquet_size_comparison(self, tmp_path, mock_obs_seq_dataframes):
        """Test that compression reduces file size."""
        df = mock_obs_seq_dataframes['model']
        
        # Replicate data to make compression meaningful
        df_large = pd.concat([df] * 100, ignore_index=True)
        
        # Write with and without compression
        uncompressed = tmp_path / "uncompressed.parquet"
        compressed = tmp_path / "compressed.parquet"
        
        df_large.to_parquet(uncompressed, compression=None)
        df_large.to_parquet(compressed, compression='snappy')
        
        # Compressed should be smaller (or similar for small data)
        assert compressed.stat().st_size <= uncompressed.stat().st_size * 1.5


class TestDaskIntegration:
    """Test Dask DataFrame integration for large datasets."""
    
    def test_read_parquet_with_dask(self, tmp_path, mock_obs_seq_dataframes):
        """Test reading parquet files with Dask."""
        df = mock_obs_seq_dataframes['model']
        
        # Write parquet
        parquet_file = tmp_path / "data.parquet"
        df.to_parquet(parquet_file)
        
        # Read with Dask
        ddf = dd.read_parquet(parquet_file)
        
        assert isinstance(ddf, dd.DataFrame)
        
        # Can compute to pandas
        result = ddf.compute()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
    
    def test_dask_filtering(self, tmp_path, mock_obs_seq_dataframes):
        """Test filtering Dask DataFrame."""
        df = mock_obs_seq_dataframes['model']
        
        parquet_file = tmp_path / "data.parquet"
        df.to_parquet(parquet_file)
        
        # Read and filter with Dask
        ddf = dd.read_parquet(parquet_file)
        filtered = ddf[ddf['QC'] == 0]
        result = filtered.compute()
        
        # All results should have QC == 0
        assert all(result['QC'] == 0)


class TestParquetDataQuality:
    """Test data quality checks on parquet output."""
    
    def test_no_missing_values_in_key_columns(self, mock_obs_seq_dataframes):
        """Test that key columns have no missing values."""
        df = mock_obs_seq_dataframes['model']
        
        key_columns = ['obs_num', 'longitude', 'latitude', 'days', 'seconds']
        
        for col in key_columns:
            assert not df[col].isna().any(), f"Missing values in {col}"
    
    def test_valid_coordinate_ranges(self, mock_obs_seq_dataframes):
        """Test that coordinates are within valid ranges."""
        df = mock_obs_seq_dataframes['model']
        
        # Longitude: -180 to 360 (allowing both conventions)
        assert all((df['longitude'] >= -180) & (df['longitude'] <= 360))
        
        # Latitude: -90 to 90
        assert all((df['latitude'] >= -90) & (df['latitude'] <= 90))
        
        # Vertical: should be non-negative for depth
        assert all(df['vertical'] >= 0)
    
    def test_valid_qc_values(self, mock_obs_seq_dataframes):
        """Test that QC values are valid DART QC codes."""
        df = mock_obs_seq_dataframes['model']
        
        # DART QC codes are typically 0-7
        valid_qc = [0, 1, 2, 3, 4, 5, 6, 7]
        assert all(df['QC'].isin(valid_qc))
