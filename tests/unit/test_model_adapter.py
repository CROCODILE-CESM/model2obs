"""Unit tests for ModelAdapter classes and registry.

Tests the model adapter architecture including:
- Adapter registry and factory function (create_model_adapter)
- MOM6-specific adapter behavior and unit conversions
- ROMS_RUTGERS-specific adapter behavior and unit conversions  
- Dataset opening with context managers
- Run options validation and capabilities checking
- Configuration key requirements per model
- parse_dart_obs_type() dispatcher behavior

The tests use fixtures to create temporary netCDF files and mock
dependencies where appropriate.
"""

import os
import shutil
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from pathlib import Path
from shapely.geometry import Polygon
from typing import Any, Dict, List
from unittest.mock import patch, mock_open
import xarray as xr

from model2obs.model_adapter.model_adapter import ModelAdapter, ModelAdapterCapabilities
from model2obs.model_adapter.model_adapter_MOM6 import ModelAdapterMOM6
from model2obs.model_adapter.model_adapter_ROMS_Rutgers import ModelAdapterROMSRutgers
from model2obs.model_adapter.model_adapter_CICE import ModelAdapterCICE
from model2obs.model_adapter.registry import create_model_adapter
from model2obs.workflows.workflow_model_obs import RunOptions


class TestModelAdapterRegistry:
    """Test model adapter registry and factory function."""

    def test_create_mom6_adapter_lowercase(self):
        """Test creating MOM6 adapter with lowercase name."""
        adapter = create_model_adapter("mom6")
        assert isinstance(adapter, ModelAdapterMOM6)
        assert adapter.time_varname == "time"

    def test_create_mom6_adapter_uppercase(self):
        """Test creating MOM6 adapter with uppercase name."""
        adapter = create_model_adapter("MOM6")
        assert isinstance(adapter, ModelAdapterMOM6)

    def test_create_roms_rutgers_adapter_lowercase(self):
        """Test creating ROMS_Rutgers adapter with lowercase name."""
        adapter = create_model_adapter("roms_rutgers")
        assert isinstance(adapter, ModelAdapterROMSRutgers)
        assert adapter.time_varname == "ocean_time"

    def test_create_roms_rutgers_adapter_uppercase(self):
        """Test creating ROMS_Rutgers adapter with uppercase name."""
        adapter = create_model_adapter("ROMS_Rutgers")
        assert isinstance(adapter, ModelAdapterROMSRutgers)

    def test_create_adapter_invalid_model(self):
        """Test creating adapter with invalid model name raises error."""
        with pytest.raises(ValueError, match="Unknown model_name"):
            create_model_adapter("invalid_model")

    def test_create_adapter_none_raises_error(self):
        """Test creating adapter with None raises error."""
        with pytest.raises(ValueError, match="model_name is required"):
            create_model_adapter(None)

    def test_create_adapter_whitespace_handling(self):
        """Test creating adapter handles whitespace in model name."""
        adapter = create_model_adapter("  MOM6  ")
        assert isinstance(adapter, ModelAdapterMOM6)

    def test_create_cice_adapter_raises_value_error(self):
        """Test that CICE is registered but raises ValueError on instantiation.

        Given: The model name 'cice' (or 'CICE')
        When: create_model_adapter() is called
        Then: ValueError is raised because ModelAdapterCICE.__init__ is incomplete
        """
        with pytest.raises(ValueError, match="time_varname not defined for CICE yet"):
            create_model_adapter("cice")

        with pytest.raises(ValueError, match="time_varname not defined for CICE yet"):
            create_model_adapter("CICE")


@pytest.fixture
def create_tmp_MOM6_nc(tmp_path):
    """Create temporary netcdf file to read from disk"""

    original_dir = os.getcwd()
    os.chdir(tmp_path)
    # Create minimal NetCDF files using xarray
    ds_template = xr.Dataset(
        data_vars={"temp": (["time", "lat", "lon"], np.zeros((1, 10, 10)))},
        coords={"time": [0], "lat": np.arange(10), "lon": np.arange(10)}
    )
    model_nc = tmp_path / "model.nc"
    ds_template.to_netcdf(model_nc)

@pytest.fixture
def create_tmp_ROMS_Rutgers_nc(tmp_path):
    """Create temporary netcdf file to read from disk"""

    original_dir = os.getcwd()
    os.chdir(tmp_path)
    # Create minimal NetCDF files using xarray
    ds_template = xr.Dataset(
        data_vars={"temp": (["ocean_time", "lat", "lon"], np.zeros((1, 10, 10)))},
        coords={"ocean_time": [0], "lat": np.arange(10), "lon": np.arange(10)}
    )
    model_nc = tmp_path / "model.nc"
    ds_template.to_netcdf(model_nc)
    
class TestModelAdapterMOM6:
    """Test ModelAdapterMOM6 methods"""

    def test_init(self):
        """Test constructor"""

        model_adapter = create_model_adapter("mom6")

        assert isinstance(model_adapter.time_varname, str)
        assert model_adapter.time_varname == "time"

    def test_get_required_config_keys(self):
        """Test get_required_config_keys returns complete list."""

        target_keys = [
            'model_files_folder', 
            'obs_seq_in_folder', 
            'output_folder',
            'template_file', 
            'static_file', 
            'ocean_geometry',
            'perfect_model_obs_dir', 
            'parquet_folder'
        ]

        model_adapter = create_model_adapter("mom6")
        required_keys = model_adapter.get_required_config_keys()
        
        assert isinstance(required_keys, list)
        assert all(isinstance(item, str) for item in required_keys)
        assert required_keys == target_keys

    def test_get_common_model_keys(self):
        """Test get_required_config_keys returns complete list."""

        target_keys = [
            'template_file',
            'static_file',
            'ocean_geometry',
            'use_pseudo_depth',
            'model_state_variables',
            'layer_name'
        ]

        model_adapter = create_model_adapter("mom6")
        common_model_keys = model_adapter.get_common_model_keys()
        
        assert isinstance(common_model_keys, list)
        assert all(isinstance(item, str) for item in common_model_keys)
        assert common_model_keys == target_keys

    def test_open_dataset_ctx(self, create_tmp_MOM6_nc, tmp_path):
        """Test open_dataset_ctx updates calendar and time varname"""
        model_adapter = create_model_adapter("mom6")

        model_nc = tmp_path / "model.nc"
        with model_adapter.open_dataset_ctx(model_nc) as ds:
            assert "time" in ds.coords
            assert ds[model_adapter.time_varname].attrs.get("calendar") == "proleptic_gregorian"

    def test_convert_units(self):
        """Test convert_units does not modify MOM6 data."""
        obs_types = [
            "BOTTLE_SALINITY",
            "ARGO_SALINITY",
            "SALINITY",
            "TEMPERATURE",
            "ARGO_TEMPERATURE",
            "SALTY"
        ]

        obs_values = np.array([30, 33, 35.1, 14, 16.7, 49])
        interpolated = obs_values + 0.023
        mock_df = pd.DataFrame({
            'interpolated_model': interpolated,
            'obs': obs_values,
            'type': obs_types
        })

        model_adapter = create_model_adapter("mom6")
        df = model_adapter.convert_units(mock_df)

        assert_frame_equal(mock_df, df)

    def test_convert_units_with_missing_values(self):
        """Test convert_units handles NA values correctly."""
        mock_df = pd.DataFrame({
            'interpolated_model': [20.0, np.nan, 22.0],
            'obs': [20.1, 21.0, np.nan],
            'type': ['SALINITY', 'TEMPERATURE', 'SALINITY']
        })

        model_adapter = create_model_adapter("mom6")
        df = model_adapter.convert_units(mock_df)

        assert pd.isna(df.loc[1, 'interpolated_model'])
        assert pd.isna(df.loc[2, 'obs'])
        assert df.loc[0, 'obs'] == 20.1

    def test_convert_units_preserves_dtypes(self):
        """Test convert_units preserves column data types."""
        mock_df = pd.DataFrame({
            'interpolated_model': np.array([20.0, 21.0], dtype=np.float64),
            'obs': np.array([20.1, 21.1], dtype=np.float64),
            'type': pd.Series(['SALINITY', 'TEMPERATURE'], dtype='object')
        })

        model_adapter = create_model_adapter("mom6")
        df = model_adapter.convert_units(mock_df)

        assert df['interpolated_model'].dtype == np.float64
        assert df['obs'].dtype == np.float64
        assert df['type'].dtype == object

    def test_convert_units_empty_dataframe(self):
        """Test convert_units handles empty dataframe."""
        mock_df = pd.DataFrame({
            'interpolated_model': pd.Series([], dtype=np.float64),
            'obs': pd.Series([], dtype=np.float64),
            'type': pd.Series([], dtype=object)
        })

        model_adapter = create_model_adapter("mom6")
        df = model_adapter.convert_units(mock_df)

        assert len(df) == 0
        assert list(df.columns) == ['interpolated_model', 'obs', 'type']

    def test_rename_time_varname(self):
        """Test rename_time_varname renames time coordinate."""
        ds = xr.Dataset(
            data_vars={"temp": (["time", "lat"], np.zeros((2, 3)))},
            coords={"time": [0, 1], "lat": [1, 2, 3]}
        )

        model_adapter = create_model_adapter("mom6")
        renamed_ds = model_adapter.rename_time_varname(ds)

        assert "time" in renamed_ds.coords
        assert model_adapter.time_varname not in renamed_ds.coords or model_adapter.time_varname == "time"

    def test_open_dataset_ctx_file_closure(self, create_tmp_MOM6_nc, tmp_path):
        """Test open_dataset_ctx closes file after use."""
        model_adapter = create_model_adapter("mom6")
        model_nc = tmp_path / "model.nc"

        with model_adapter.open_dataset_ctx(model_nc) as ds:
            assert ds is not None

        # Dataset should be closed after context exit
        # xarray doesn't expose a direct "is_closed" attribute,
        # but we verify no exception is raised

    def test_open_dataset_ctx_nonexistent_file(self, tmp_path):
        """Test open_dataset_ctx raises error for nonexistent file."""
        model_adapter = create_model_adapter("mom6")
        nonexistent = tmp_path / "nonexistent.nc"

        with pytest.raises(FileNotFoundError):
            with model_adapter.open_dataset_ctx(nonexistent) as ds:
                pass

    def test_open_dataset_ctx_no_time_variable(self, tmp_path: Path):
        """Test open_dataset_ctx succeeds on a static geometry file with no time variable.

        Given: A NetCDF file with spatial variables only (no time dimension)
        When:  open_dataset_ctx is used to open it
        Then:  The dataset is yielded without error and spatial variables are accessible
        """
        geometry_nc = tmp_path / "ocean_geometry.nc"
        lonh = np.linspace(10.0, 15.0, 4)
        lath = np.linspace(40.0, 45.0, 4)
        ds_in = xr.Dataset({
            'wet': (['lath', 'lonh'], np.ones((len(lath), len(lonh)), dtype=int)),
            'lonh': lonh,
            'lath': lath,
        })
        ds_in.to_netcdf(geometry_nc)

        adapter = create_model_adapter("mom6")
        with adapter.open_dataset_ctx(str(geometry_nc)) as ds:
            assert "time" not in ds
            assert "lonh" in ds
            assert "lath" in ds
            assert "wet" in ds

    def test_validate_run_options_all_true(self):
        """Test that all run options are validated"""
        model_adapter = create_model_adapter("mom6")

        run_opts = RunOptions(
            trim_obs = True,
            no_matching = True,
            force_obs_time = True
        )
        model_adapter.validate_run_options(run_opts)

    def test_validate_run_options_some_false(self):
        """Test that some run options are valid if false"""
        model_adapter = create_model_adapter("mom6")

        run_opts = RunOptions(
            trim_obs = True,
            no_matching = False,
            force_obs_time = False
        )
        model_adapter.validate_run_options(run_opts)


class TestModelAdapterPathValidation:
    """Test path validation in ModelAdapter and subclasses."""

    def test_mom6_validate_paths_success(self, tmp_path):
        """Test MOM6 validate_paths with all valid paths."""
        model_adapter = create_model_adapter("mom6")
        
        model_folder = tmp_path / "model_files"
        model_folder.mkdir()
        (model_folder / "model_01.nc").touch()
        
        obs_folder = tmp_path / "obs"
        obs_folder.mkdir()
        (obs_folder / "obs_01.in").touch()
        
        output_folder = tmp_path / "output"
        tmp_folder = tmp_path / "tmp"
        parquet_folder = tmp_path / "parquet"
        
        template_nc = tmp_path / "template.nc"
        template_nc.touch()
        static_nc = tmp_path / "static.nc"
        static_nc.touch()
        geometry_nc = tmp_path / "ocean_geometry.nc"
        geometry_nc.touch()
        
        config = {
            'model_files_folder': str(model_folder),
            'obs_seq_in_folder': str(obs_folder),
            'output_folder': str(output_folder),
            'tmp_folder': str(tmp_folder),
            'parquet_folder': str(parquet_folder),
            'template_file': str(template_nc),
            'static_file': str(static_nc),
            'ocean_geometry': str(geometry_nc)
        }
        
        run_opts = RunOptions(trim_obs=False, no_matching=False, force_obs_time=False)
        
        model_adapter.validate_paths(config, run_opts)
        
        assert Path(output_folder).exists()
        assert Path(tmp_folder).exists()
        assert Path(parquet_folder).exists()

    def test_mom6_validate_paths_creates_output_folders(self, tmp_path):
        """Test MOM6 validate_paths creates missing output folders."""
        model_adapter = create_model_adapter("mom6")
        
        model_folder = tmp_path / "model_files"
        model_folder.mkdir()
        (model_folder / "model_01.nc").touch()
        
        obs_folder = tmp_path / "obs"
        obs_folder.mkdir()
        (obs_folder / "obs_01.in").touch()
        
        template_nc = tmp_path / "template.nc"
        template_nc.touch()
        static_nc = tmp_path / "static.nc"
        static_nc.touch()
        geometry_nc = tmp_path / "ocean_geometry.nc"
        geometry_nc.touch()
        
        config = {
            'model_files_folder': str(model_folder),
            'obs_seq_in_folder': str(obs_folder),
            'output_folder': str(tmp_path / "output"),
            'tmp_folder': str(tmp_path / "tmp"),
            'parquet_folder': str(tmp_path / "parquet"),
            'template_file': str(template_nc),
            'static_file': str(static_nc),
            'ocean_geometry': str(geometry_nc)
        }
        
        run_opts = RunOptions(trim_obs=False, no_matching=False, force_obs_time=False)
        
        model_adapter.validate_paths(config, run_opts)
        
        assert Path(config['output_folder']).exists()
        assert Path(config['tmp_folder']).exists()
        assert Path(config['parquet_folder']).exists()

    def test_mom6_validate_paths_with_trim_obs(self, tmp_path):
        """Test MOM6 validate_paths creates trimmed_obs_folder when trim_obs=True."""
        model_adapter = create_model_adapter("mom6")
        
        model_folder = tmp_path / "model_files"
        model_folder.mkdir()
        (model_folder / "model_01.nc").touch()
        
        obs_folder = tmp_path / "obs"
        obs_folder.mkdir()
        (obs_folder / "obs_01.in").touch()
        
        template_nc = tmp_path / "template.nc"
        template_nc.touch()
        static_nc = tmp_path / "static.nc"
        static_nc.touch()
        geometry_nc = tmp_path / "ocean_geometry.nc"
        geometry_nc.touch()
        
        config = {
            'model_files_folder': str(model_folder),
            'obs_seq_in_folder': str(obs_folder),
            'output_folder': str(tmp_path / "output"),
            'tmp_folder': str(tmp_path / "tmp"),
            'parquet_folder': str(tmp_path / "parquet"),
            'template_file': str(template_nc),
            'static_file': str(static_nc),
            'ocean_geometry': str(geometry_nc)
        }
        
        run_opts = RunOptions(trim_obs=True, no_matching=False, force_obs_time=False)
        
        model_adapter.validate_paths(config, run_opts)
        
        assert 'trimmed_obs_folder' in config
        assert Path(config['trimmed_obs_folder']).exists()

    def test_mom6_validate_paths_missing_model_folder(self, tmp_path):
        """Test MOM6 validate_paths raises error for missing model folder."""
        model_adapter = create_model_adapter("mom6")
        
        obs_folder = tmp_path / "obs"
        obs_folder.mkdir()
        (obs_folder / "obs_01.in").touch()
        
        config = {
            'model_files_folder': str(tmp_path / "nonexistent"),
            'obs_seq_in_folder': str(obs_folder),
            'output_folder': str(tmp_path / "output"),
            'tmp_folder': str(tmp_path / "tmp"),
            'parquet_folder': str(tmp_path / "parquet"),
            'template_file': str(tmp_path / "template.nc"),
            'static_file': str(tmp_path / "static.nc"),
            'ocean_geometry': str(tmp_path / "ocean_geometry.nc")
        }
        
        run_opts = RunOptions(trim_obs=False, no_matching=False, force_obs_time=False)
        
        with pytest.raises(NotADirectoryError):
            model_adapter.validate_paths(config, run_opts)

    def test_mom6_validate_paths_empty_model_folder(self, tmp_path):
        """Test MOM6 validate_paths raises error for empty model folder."""
        model_adapter = create_model_adapter("mom6")
        
        model_folder = tmp_path / "model_files"
        model_folder.mkdir()
        
        obs_folder = tmp_path / "obs"
        obs_folder.mkdir()
        (obs_folder / "obs_01.in").touch()
        
        config = {
            'model_files_folder': str(model_folder),
            'obs_seq_in_folder': str(obs_folder),
            'output_folder': str(tmp_path / "output"),
            'tmp_folder': str(tmp_path / "tmp"),
            'parquet_folder': str(tmp_path / "parquet"),
            'template_file': str(tmp_path / "template.nc"),
            'static_file': str(tmp_path / "static.nc"),
            'ocean_geometry': str(tmp_path / "ocean_geometry.nc")
        }
        
        run_opts = RunOptions(trim_obs=False, no_matching=False, force_obs_time=False)
        
        with pytest.raises(ValueError, match="empty"):
            model_adapter.validate_paths(config, run_opts)

    def test_mom6_validate_paths_non_nc_files_in_model_folder(self, tmp_path):
        """Test MOM6 validate_paths raises error for non-.nc files in model folder."""
        model_adapter = create_model_adapter("mom6")
        
        model_folder = tmp_path / "model_files"
        model_folder.mkdir()
        (model_folder / "model_01.txt").touch()
        
        obs_folder = tmp_path / "obs"
        obs_folder.mkdir()
        (obs_folder / "obs_01.in").touch()
        
        config = {
            'model_files_folder': str(model_folder),
            'obs_seq_in_folder': str(obs_folder),
            'output_folder': str(tmp_path / "output"),
            'tmp_folder': str(tmp_path / "tmp"),
            'parquet_folder': str(tmp_path / "parquet"),
            'template_file': str(tmp_path / "template.nc"),
            'static_file': str(tmp_path / "static.nc"),
            'ocean_geometry': str(tmp_path / "ocean_geometry.nc")
        }
        
        run_opts = RunOptions(trim_obs=False, no_matching=False, force_obs_time=False)
        
        with pytest.raises(ValueError, match="non-.nc files"):
            model_adapter.validate_paths(config, run_opts)

    def test_mom6_validate_paths_missing_template_file(self, tmp_path):
        """Test MOM6 validate_paths raises error for missing template file."""
        model_adapter = create_model_adapter("mom6")
        
        model_folder = tmp_path / "model_files"
        model_folder.mkdir()
        (model_folder / "model_01.nc").touch()
        
        obs_folder = tmp_path / "obs"
        obs_folder.mkdir()
        (obs_folder / "obs_01.in").touch()
        
        static_nc = tmp_path / "static.nc"
        static_nc.touch()
        geometry_nc = tmp_path / "ocean_geometry.nc"
        geometry_nc.touch()
        
        config = {
            'model_files_folder': str(model_folder),
            'obs_seq_in_folder': str(obs_folder),
            'output_folder': str(tmp_path / "output"),
            'tmp_folder': str(tmp_path / "tmp"),
            'parquet_folder': str(tmp_path / "parquet"),
            'template_file': str(tmp_path / "missing_template.nc"),
            'static_file': str(static_nc),
            'ocean_geometry': str(geometry_nc)
        }
        
        run_opts = RunOptions(trim_obs=False, no_matching=False, force_obs_time=False)
        
        with pytest.raises(FileNotFoundError, match="template_file"):
            model_adapter.validate_paths(config, run_opts)

    def test_mom6_validate_paths_non_nc_template_file(self, tmp_path):
        """Test MOM6 validate_paths raises error for non-.nc template file."""
        model_adapter = create_model_adapter("mom6")
        
        model_folder = tmp_path / "model_files"
        model_folder.mkdir()
        (model_folder / "model_01.nc").touch()
        
        obs_folder = tmp_path / "obs"
        obs_folder.mkdir()
        (obs_folder / "obs_01.in").touch()
        
        template_txt = tmp_path / "template.txt"
        template_txt.touch()
        
        config = {
            'model_files_folder': str(model_folder),
            'obs_seq_in_folder': str(obs_folder),
            'output_folder': str(tmp_path / "output"),
            'tmp_folder': str(tmp_path / "tmp"),
            'parquet_folder': str(tmp_path / "parquet"),
            'template_file': str(template_txt),
            'static_file': str(tmp_path / "static.nc"),
            'ocean_geometry': str(tmp_path / "ocean_geometry.nc")
        }
        
        run_opts = RunOptions(trim_obs=False, no_matching=False, force_obs_time=False)
        
        with pytest.raises(ValueError, match="not a .nc file"):
            model_adapter.validate_paths(config, run_opts)

    def _make_valid_mom6_base_config(self, tmp_path) -> dict:
        """Create a minimal valid MOM6 config with all required paths on disk."""
        model_folder = tmp_path / "model_files"
        model_folder.mkdir()
        (model_folder / "model_01.nc").touch()

        obs_folder = tmp_path / "obs"
        obs_folder.mkdir()
        (obs_folder / "obs_01.in").touch()

        (tmp_path / "template.nc").touch()
        (tmp_path / "static.nc").touch()
        (tmp_path / "ocean_geometry.nc").touch()

        return {
            'model_files_folder': str(model_folder),
            'obs_seq_in_folder': str(obs_folder),
            'output_folder': str(tmp_path / "output"),
            'tmp_folder': str(tmp_path / "tmp"),
            'parquet_folder': str(tmp_path / "parquet"),
            'template_file': str(tmp_path / "template.nc"),
            'static_file': str(tmp_path / "static.nc"),
            'ocean_geometry': str(tmp_path / "ocean_geometry.nc"),
        }

    def test_mom6_validate_paths_vertical_interp_pseudo_depth(self, tmp_path):
        """Test MOM6 validate_paths passes when use_pseudo_depth is True.

        Given: A valid MOM6 config with use_pseudo_depth set to True
        When: validate_paths() is called
        Then: Validation succeeds without error
        """
        model_adapter = create_model_adapter("mom6")
        config = self._make_valid_mom6_base_config(tmp_path)
        config['use_pseudo_depth'] = True
        run_opts = RunOptions(trim_obs=False, no_matching=False, force_obs_time=False)

        model_adapter.validate_paths(config, run_opts)  # should not raise

    def test_mom6_validate_paths_vertical_interp_layer_thickness(self, tmp_path):
        """Test MOM6 validate_paths passes when QTY_LAYER_THICKNESS is in state.

        Given: A valid MOM6 config with h mapped to QTY_LAYER_THICKNESS
              and use_pseudo_depth absent (or False)
        When: validate_paths() is called
        Then: Validation succeeds without error
        """
        model_adapter = create_model_adapter("mom6")
        config = self._make_valid_mom6_base_config(tmp_path)
        config['use_pseudo_depth'] = False
        config['model_state_variables'] = {
            'so': 'QTY_SALINITY',
            'thetao': 'QTY_POTENTIAL_TEMPERATURE',
            'h': 'QTY_LAYER_THICKNESS',
        }
        run_opts = RunOptions(trim_obs=False, no_matching=False, force_obs_time=False)

        model_adapter.validate_paths(config, run_opts)  # should not raise

    def test_mom6_validate_paths_vertical_interp_missing(self, tmp_path):
        """Test MOM6 validate_paths raises ValueError when neither pseudo-depth
        nor QTY_LAYER_THICKNESS is configured.

        Given: A valid MOM6 config without use_pseudo_depth and without
               QTY_LAYER_THICKNESS in model_state_variables
        When: validate_paths() is called
        Then: ValueError is raised pointing to DART error 1013
        """
        model_adapter = create_model_adapter("mom6")
        config = self._make_valid_mom6_base_config(tmp_path)
        config['use_pseudo_depth'] = False
        config['model_state_variables'] = {
            'so': 'QTY_SALINITY',
            'thetao': 'QTY_POTENTIAL_TEMPERATURE',
        }
        run_opts = RunOptions(trim_obs=False, no_matching=False, force_obs_time=False)

        with pytest.raises(ValueError, match="1013"):
            model_adapter.validate_paths(config, run_opts)

    def test_roms_validate_paths_success(self, tmp_path):
        """Test ROMS validate_paths with all valid paths."""
        model_adapter = create_model_adapter("roms_rutgers")
        
        model_folder = tmp_path / "model_files"
        model_folder.mkdir()
        (model_folder / "roms_avg_01.nc").touch()
        
        obs_folder = tmp_path / "obs"
        obs_folder.mkdir()
        (obs_folder / "obs_01.in").touch()
        
        roms_nc = tmp_path / "roms_grid.nc"
        roms_nc.touch()
        
        config = {
            'model_files_folder': str(model_folder),
            'obs_seq_in_folder': str(obs_folder),
            'output_folder': str(tmp_path / "output"),
            'tmp_folder': str(tmp_path / "tmp"),
            'parquet_folder': str(tmp_path / "parquet"),
            'roms_filename': str(roms_nc)
        }
        
        run_opts = RunOptions(trim_obs=False, no_matching=False, force_obs_time=False)
        
        model_adapter.validate_paths(config, run_opts)
        
        assert Path(config['output_folder']).exists()

    def test_roms_validate_paths_missing_roms_file(self, tmp_path):
        """Test ROMS validate_paths raises error for missing roms_filename."""
        model_adapter = create_model_adapter("roms_rutgers")
        
        model_folder = tmp_path / "model_files"
        model_folder.mkdir()
        (model_folder / "roms_avg_01.nc").touch()
        
        obs_folder = tmp_path / "obs"
        obs_folder.mkdir()
        (obs_folder / "obs_01.in").touch()
        
        config = {
            'model_files_folder': str(model_folder),
            'obs_seq_in_folder': str(obs_folder),
            'output_folder': str(tmp_path / "output"),
            'tmp_folder': str(tmp_path / "tmp"),
            'parquet_folder': str(tmp_path / "parquet"),
            'roms_filename': str(tmp_path / "missing_roms.nc")
        }
        
        run_opts = RunOptions(trim_obs=False, no_matching=False, force_obs_time=False)
        
        with pytest.raises(FileNotFoundError, match="roms_filename"):
            model_adapter.validate_paths(config, run_opts)

    def test_base_adapter_sets_default_trimmed_obs_folder(self, tmp_path):
        """Test base adapter sets default trimmed_obs_folder."""
        model_adapter = create_model_adapter("mom6")
        
        model_folder = tmp_path / "model_files"
        model_folder.mkdir()
        (model_folder / "model_01.nc").touch()
        
        obs_folder = tmp_path / "obs"
        obs_folder.mkdir()
        (obs_folder / "obs_01.in").touch()
        
        template_nc = tmp_path / "template.nc"
        template_nc.touch()
        static_nc = tmp_path / "static.nc"
        static_nc.touch()
        geometry_nc = tmp_path / "ocean_geometry.nc"
        geometry_nc.touch()
        
        config = {
            'model_files_folder': str(model_folder),
            'obs_seq_in_folder': str(obs_folder),
            'output_folder': str(tmp_path / "output"),
            'tmp_folder': str(tmp_path / "tmp"),
            'parquet_folder': str(tmp_path / "parquet"),
            'template_file': str(template_nc),
            'static_file': str(static_nc),
            'ocean_geometry': str(geometry_nc)
        }
        
        run_opts = RunOptions(trim_obs=True, no_matching=False, force_obs_time=False)
        
        model_adapter.validate_paths(config, run_opts)
        
        assert config['trimmed_obs_folder'] == 'trimmed_obs_seq'

    def test_base_adapter_sets_default_input_nml_bck(self, tmp_path):
        """Test base adapter sets default input_nml_bck."""
        model_adapter = create_model_adapter("mom6")
        
        model_folder = tmp_path / "model_files"
        model_folder.mkdir()
        (model_folder / "model_01.nc").touch()
        
        obs_folder = tmp_path / "obs"
        obs_folder.mkdir()
        (obs_folder / "obs_01.in").touch()
        
        template_nc = tmp_path / "template.nc"
        template_nc.touch()
        static_nc = tmp_path / "static.nc"
        static_nc.touch()
        geometry_nc = tmp_path / "ocean_geometry.nc"
        geometry_nc.touch()
        
        config = {
            'model_files_folder': str(model_folder),
            'obs_seq_in_folder': str(obs_folder),
            'output_folder': str(tmp_path / "output"),
            'tmp_folder': str(tmp_path / "tmp"),
            'parquet_folder': str(tmp_path / "parquet"),
            'template_file': str(template_nc),
            'static_file': str(static_nc),
            'ocean_geometry': str(geometry_nc)
        }
        
        run_opts = RunOptions(trim_obs=False, no_matching=False, force_obs_time=False)
        
        model_adapter.validate_paths(config, run_opts)
        
        assert config['input_nml_bck'] == 'input.nml.backup'

    def test_cice_validate_paths_valid_cice_file(self, tmp_path):
        """Test CICE validate_paths succeeds when cice_filename is a valid .nc file.

        Given: A ModelAdapterCICE instance and a config with a real .nc file
        When: validate_paths() is called
        Then: No exception is raised
        """
        adapter = object.__new__(ModelAdapterCICE)
        adapter.model_name = "CICE"
        adapter.time_varname = None
        adapter.capabilities = ModelAdapterCICE.capabilities

        model_folder = tmp_path / "model_files"
        model_folder.mkdir()
        (model_folder / "model_01.nc").touch()

        obs_folder = tmp_path / "obs"
        obs_folder.mkdir()
        (obs_folder / "obs_01.in").touch()

        cice_file = tmp_path / "cice_output.nc"
        cice_file.touch()

        cfg = {
            'model_files_folder': str(model_folder),
            'obs_seq_in_folder': str(obs_folder),
            'output_folder': str(tmp_path / "output"),
            'tmp_folder': str(tmp_path / "tmp"),
            'parquet_folder': str(tmp_path / "parquet"),
            'cice_filename': str(cice_file),
        }
        run_opts = RunOptions(trim_obs=False, no_matching=False, force_obs_time=False)

        adapter.validate_paths(cfg, run_opts)

    def test_cice_validate_paths_missing_cice_file(self, tmp_path):
        """Test CICE validate_paths raises FileNotFoundError for missing cice_filename.

        Given: A ModelAdapterCICE instance and a config pointing to a non-existent file
        When: validate_paths() is called
        Then: FileNotFoundError is raised
        """
        adapter = object.__new__(ModelAdapterCICE)
        adapter.model_name = "CICE"
        adapter.time_varname = None
        adapter.capabilities = ModelAdapterCICE.capabilities

        model_folder = tmp_path / "model_files"
        model_folder.mkdir()
        (model_folder / "model_01.nc").touch()

        obs_folder = tmp_path / "obs"
        obs_folder.mkdir()
        (obs_folder / "obs_01.in").touch()

        cfg = {
            'model_files_folder': str(model_folder),
            'obs_seq_in_folder': str(obs_folder),
            'output_folder': str(tmp_path / "output"),
            'tmp_folder': str(tmp_path / "tmp"),
            'parquet_folder': str(tmp_path / "parquet"),
            'cice_filename': str(tmp_path / "nonexistent.nc"),
        }
        run_opts = RunOptions(trim_obs=False, no_matching=False, force_obs_time=False)

        with pytest.raises(FileNotFoundError, match="cice_filename"):
            adapter.validate_paths(cfg, run_opts)

    def test_cice_validate_paths_non_nc_cice_file(self, tmp_path):
        """Test CICE validate_paths raises ValueError when cice_filename is not .nc.

        Given: A ModelAdapterCICE instance and a config with a non-.nc cice_filename
        When: validate_paths() is called
        Then: ValueError is raised
        """
        adapter = object.__new__(ModelAdapterCICE)
        adapter.model_name = "CICE"
        adapter.time_varname = None
        adapter.capabilities = ModelAdapterCICE.capabilities

        model_folder = tmp_path / "model_files"
        model_folder.mkdir()
        (model_folder / "model_01.nc").touch()

        obs_folder = tmp_path / "obs"
        obs_folder.mkdir()
        (obs_folder / "obs_01.in").touch()

        wrong_file = tmp_path / "cice_output.txt"
        wrong_file.touch()

        cfg = {
            'model_files_folder': str(model_folder),
            'obs_seq_in_folder': str(obs_folder),
            'output_folder': str(tmp_path / "output"),
            'tmp_folder': str(tmp_path / "tmp"),
            'parquet_folder': str(tmp_path / "parquet"),
            'cice_filename': str(wrong_file),
        }
        run_opts = RunOptions(trim_obs=False, no_matching=False, force_obs_time=False)

        with pytest.raises(ValueError, match="cice_filename"):
            adapter.validate_paths(cfg, run_opts)


class TestModelAdapterROMSRutgers:
    """Test ModelAdapterROMSRutgers methods"""

    def test_init(self):
        """Test constructor"""

        model_adapter = create_model_adapter("roms_rutgers")

        assert isinstance(model_adapter.time_varname, str)
        assert model_adapter.time_varname == "ocean_time"

    def test_get_required_config_keys(self):
        """Test get_required_config_keys returns complete list."""

        target_keys = [
            'model_files_folder', 
            'obs_seq_in_folder', 
            'output_folder',
            'roms_filename',
            'perfect_model_obs_dir', 
            'parquet_folder'
        ]

        model_adapter = create_model_adapter("roms_rutgers")
        required_keys = model_adapter.get_required_config_keys()
        
        assert isinstance(required_keys, list)
        assert all(isinstance(item, str) for item in required_keys)
        assert required_keys == target_keys

    def test_get_common_model_keys(self):
        """Test get_required_config_keys returns complete list."""

        target_keys = [
            'roms_filename',
            'variables',
            'debug'
        ]

        model_adapter = create_model_adapter("roms_rutgers")
        common_model_keys = model_adapter.get_common_model_keys()
        
        assert isinstance(common_model_keys, list)
        assert all(isinstance(item, str) for item in common_model_keys)
        assert common_model_keys == target_keys

    def test_open_dataset_ctx(self, create_tmp_ROMS_Rutgers_nc, tmp_path):
        """Test open_dataset_ctx updates calendar and time varname"""
        model_adapter = create_model_adapter("roms_rutgers")

        model_nc = tmp_path / "model.nc"
        with model_adapter.open_dataset_ctx(model_nc) as ds:
            assert "time" in ds.coords

    def test_convert_units(self):
        """Test convert_units converts ROMS_Rutgers salinity from PSU/1000 to PSU."""
        obs_types = [
            "BOTTLE_SALINITY",
            "ARGO_SALINITY",
            "SALINITY",
            "TEMPERATURE",
            "ARGO_TEMPERATURE",
            "SALTY"
        ]

        obs_values = np.array([30*1e-3, 33*1e-3, 35.1*1e-3, 14, 16.7, 49])
        interpolated = obs_values + 0.023
        mock_df = pd.DataFrame({
            'interpolated_model': interpolated,
            'obs': obs_values,
            'type': obs_types
        })

        target_values = np.array([30, 33, 35.1, 14, 16.7, 49])

        target_df = pd.DataFrame({
            'interpolated_model': interpolated,
            'obs': target_values,
            'type': obs_types
        })
         
        model_adapter = create_model_adapter("roms_rutgers")
        df = model_adapter.convert_units(mock_df)

        assert_frame_equal(df, target_df)

    def test_convert_units_with_missing_values(self):
        """Test convert_units handles NA values in salinity."""
        mock_df = pd.DataFrame({
            'interpolated_model': [0.020, np.nan, 0.022],
            'obs': [0.0201, 0.021, np.nan],
            'type': ['SALINITY', 'TEMPERATURE', 'SALINITY']
        })

        model_adapter = create_model_adapter("roms_rutgers")
        df = model_adapter.convert_units(mock_df)

        assert pd.isna(df.loc[1, 'interpolated_model'])
        assert pd.isna(df.loc[2, 'obs'])
        assert np.isclose(df.loc[0, 'obs'], 20.1)

    def test_convert_units_preserves_dtypes(self):
        """Test convert_units preserves column data types."""
        mock_df = pd.DataFrame({
            'interpolated_model': np.array([0.020, 0.021], dtype=np.float64),
            'obs': np.array([0.0201, 0.0211], dtype=np.float64),
            'type': pd.Series(['SALINITY', 'TEMPERATURE'], dtype='object')
        })

        model_adapter = create_model_adapter("roms_rutgers")
        df = model_adapter.convert_units(mock_df)

        assert df['interpolated_model'].dtype == np.float64
        assert df['obs'].dtype == np.float64
        assert df['type'].dtype == object

    def test_convert_units_only_salinity_affected(self):
        """Test convert_units only modifies salinity observations."""
        mock_df = pd.DataFrame({
            'interpolated_model': [0.020, 15.0, 0.022],
            'obs': [0.0201, 15.5, 0.0221],
            'type': ['SALINITY', 'TEMPERATURE', 'ARGO_SALINITY']
        })

        model_adapter = create_model_adapter("roms_rutgers")
        df = model_adapter.convert_units(mock_df)

        assert np.isclose(df.loc[0, 'obs'], 20.1)
        assert df.loc[1, 'obs'] == 15.5
        assert np.isclose(df.loc[2, 'obs'], 22.1)

    def test_convert_units_empty_dataframe(self):
        """Test convert_units handles empty dataframe."""
        mock_df = pd.DataFrame({
            'interpolated_model': pd.Series([], dtype=np.float64),
            'obs': pd.Series([], dtype=np.float64),
            'type': pd.Series([], dtype=object)
        })

        model_adapter = create_model_adapter("roms_rutgers")
        df = model_adapter.convert_units(mock_df)

        assert len(df) == 0
        assert list(df.columns) == ['interpolated_model', 'obs', 'type']

    def test_rename_time_varname(self):
        """Test rename_time_varname renames ocean_time to time."""
        ds = xr.Dataset(
            data_vars={"temp": (["ocean_time", "lat"], np.zeros((2, 3)))},
            coords={"ocean_time": [0, 1], "lat": [1, 2, 3]}
        )

        model_adapter = create_model_adapter("roms_rutgers")
        renamed_ds = model_adapter.rename_time_varname(ds)

        assert "time" in renamed_ds.coords
        assert "ocean_time" not in renamed_ds.coords

    def test_open_dataset_ctx_file_closure(self, create_tmp_ROMS_Rutgers_nc, tmp_path):
        """Test open_dataset_ctx closes file after use."""
        model_adapter = create_model_adapter("roms_rutgers")
        model_nc = tmp_path / "model.nc"

        with model_adapter.open_dataset_ctx(model_nc) as ds:
            assert ds is not None

    def test_open_dataset_ctx_nonexistent_file(self, tmp_path):
        """Test open_dataset_ctx raises error for nonexistent file."""
        model_adapter = create_model_adapter("roms_rutgers")
        nonexistent = tmp_path / "nonexistent.nc"

        with pytest.raises(FileNotFoundError):
            with model_adapter.open_dataset_ctx(nonexistent) as ds:
                pass

    def test_open_dataset_ctx_no_time_variable(self, tmp_path: Path):
        """Test open_dataset_ctx succeeds on a static file with no time variable.

        Given: A NetCDF file with spatial variables only (no time dimension)
        When:  open_dataset_ctx is used to open it
        Then:  The dataset is yielded without error and spatial variables are accessible
        """
        geometry_nc = tmp_path / "roms_grid.nc"
        ds_in = xr.Dataset({
            'h': (['eta_rho', 'xi_rho'], np.ones((4, 4), dtype=float)),
        })
        ds_in.to_netcdf(geometry_nc)

        adapter = create_model_adapter("roms_rutgers")
        with adapter.open_dataset_ctx(str(geometry_nc)) as ds:
            assert "ocean_time" not in ds
            assert "time" not in ds
            assert "h" in ds

    def test_validate_run_options_all_true(self):
        """Test that not all run options are supported"""
        model_adapter = create_model_adapter("roms_rutgers")

        run_opts = RunOptions(
            trim_obs = True,
            no_matching = True,
            force_obs_time = True
        )

        with pytest.raises(NotImplementedError):
            model_adapter.validate_run_options(run_opts)

    def test_validate_run_options_all_false(self):
        """Test that if all run options are false, they are valid"""
        model_adapter = create_model_adapter("roms_rutgers")

        run_opts = RunOptions(
            trim_obs = False,
            no_matching = False,
            force_obs_time = False
        )
        model_adapter.validate_run_options(run_opts)


class TestModelAdapterCICE:
    """Test suite for ModelAdapterCICE.

    Because ModelAdapterCICE.__init__ is not yet complete (raises ValueError),
    tests for individual methods use object.__new__ to bypass __init__ and
    set required attributes manually. Only test_init_raises_value_error
    calls __init__ directly.
    """

    @staticmethod
    def _make_cice_adapter() -> ModelAdapterCICE:
        """Return a ModelAdapterCICE instance with __init__ bypassed."""
        adapter = object.__new__(ModelAdapterCICE)
        adapter.model_name = "CICE"
        adapter.time_varname = None
        return adapter

    def test_init_raises_value_error(self):
        """Test ModelAdapterCICE() raises ValueError (incomplete implementation).

        Given: Direct instantiation of ModelAdapterCICE
        When: __init__ is called
        Then: ValueError is raised with message about time_varname
        """
        with pytest.raises(ValueError, match="time_varname not defined for CICE yet"):
            ModelAdapterCICE()

    def test_capabilities_trim_obs_false(self):
        """Test CICE adapter does not support trim_obs."""
        assert ModelAdapterCICE.capabilities.supports_trim_obs is False

    def test_capabilities_no_matching_true(self):
        """Test CICE adapter supports no_matching."""
        assert ModelAdapterCICE.capabilities.supports_no_matching is True

    def test_capabilities_force_obs_time_true(self):
        """Test CICE adapter supports force_obs_time."""
        assert ModelAdapterCICE.capabilities.supports_force_obs_time is True

    def test_get_required_config_keys(self):
        """Test get_required_config_keys returns all CICE-specific keys.

        Given: A ModelAdapterCICE instance (bypassing __init__)
        When: get_required_config_keys() is called
        Then: Returns a list containing the six required keys including cice_filename
        """
        adapter = self._make_cice_adapter()
        keys = adapter.get_required_config_keys()

        assert 'model_files_folder' in keys
        assert 'obs_seq_in_folder' in keys
        assert 'output_folder' in keys
        assert 'cice_filename' in keys
        assert 'perfect_model_obs_dir' in keys
        assert 'parquet_folder' in keys

    def test_get_common_model_keys(self):
        """Test get_common_model_keys returns CICE-specific common keys.

        Given: A ModelAdapterCICE instance (bypassing __init__)
        When: get_common_model_keys() is called
        Then: Returns a list containing cice_filename and variables
        """
        adapter = self._make_cice_adapter()
        keys = adapter.get_common_model_keys()

        assert 'cice_filename' in keys
        assert 'variables' in keys

    def test_open_dataset_ctx_raises_value_error(self):
        """Test open_dataset_ctx raises ValueError (not yet implemented).

        Given: A ModelAdapterCICE instance (bypassing __init__)
        When: open_dataset_ctx() is entered
        Then: ValueError is raised
        """
        adapter = self._make_cice_adapter()
        with pytest.raises(ValueError, match="not implemented for CICE yet"):
            with adapter.open_dataset_ctx("some_path.nc"):
                pass

    def test_convert_units_raises_value_error(self):
        """Test convert_units raises ValueError (not yet implemented).

        Given: A ModelAdapterCICE instance (bypassing __init__)
        When: convert_units() is called
        Then: ValueError is raised
        """
        import pandas as pd
        adapter = self._make_cice_adapter()
        with pytest.raises(ValueError, match="not implemented for CICE yet"):
            adapter.convert_units(pd.DataFrame())


class TestGetModelTimeInDaysSeconds:
    """Test suite for get_model_time_in_days_seconds() function."""
    
    def test_get_model_time_in_days_seconds_valid_file(self, tmp_path: Path):
        """Test get_model_time_in_days_seconds with valid model file.
        
        Given: A NetCDF file with a single time dimension
        When: get_model_time_in_days_seconds() is called
        Then: Correct days and seconds are returned
        """

        model_file = tmp_path / "model.nc"
        
        ds = xr.Dataset({
            'temp': (['time', 'x'], [[20.0, 21.0, 22.0]]),
        }, coords={
            'time': [np.datetime64('2020-06-15T12:00:00')],
            'x': [0, 1, 2]
        })
        ds.to_netcdf(model_file)

        adapter = create_model_adapter("mom6")
        days, seconds = adapter.get_model_time_in_days_seconds(str(model_file))
        
        assert isinstance(days, (int, np.integer))
        assert isinstance(seconds, (int, np.integer))
        assert days > 150000
        assert seconds == 12 * 3600
    
    def test_get_model_time_in_days_seconds_multiple_times_raises_error(self, tmp_path: Path):
        """Test get_model_time_in_days_seconds raises error for multiple time steps.
        
        Given: A NetCDF file with multiple time steps
        When: get_model_time_in_days_seconds() is called
        Then: ValueError is raised
        """
        model_file = tmp_path / "model_multi.nc"
        
        ds = xr.Dataset({
            'temp': (['time', 'x'], [[20.0, 21.0], [22.0, 23.0]]),
        }, coords={
            'time': [np.datetime64('2020-06-15T12:00:00'), 
                     np.datetime64('2020-06-16T12:00:00')],
            'x': [0, 1]
        })
        ds.to_netcdf(model_file)
        
        adapter = create_model_adapter("mom6")
        with pytest.raises(ValueError, match="multiple time steps"):
            adapter.get_model_time_in_days_seconds(str(model_file))
    
    def test_get_model_time_in_days_seconds_missing_file(self):
        """Test get_model_time_in_days_seconds raises error for missing file.
        
        Given: A path to a nonexistent file
        When: get_model_time_in_days_seconds() is called
        Then: FileNotFoundError is raised
        """
        adapter = create_model_adapter("mom6")
        with pytest.raises(FileNotFoundError):
            adapter.get_model_time_in_days_seconds("/nonexistent/model.nc")


class TestParseDartObsType:
    """Tests for ModelAdapter.parse_dart_obs_type().

    Covers:
    - Delegation to config_utils.parse_obs_def_model_mod for ocean and sea-ice models
    - NotImplementedError for unsupported model types
    - End-to-end parsing with real RST and f90 fixture files
    """

    _MOCK_PARSE_TARGET = (
        'model2obs.model_adapter.model_adapter.config_utils.parse_obs_def_model_mod'
    )
    _MOCK_RESULT = (
        {'FLOAT_TEMPERATURE': 'QTY_TEMPERATURE'},
        {'QTY_TEMPERATURE': ['FLOAT_TEMPERATURE']},
    )

    def test_parse_dart_obs_type_mom6_delegates_to_config_utils(self):
        """Test parse_dart_obs_type calls parse_obs_def_model_mod for MOM6.

        Given: A MOM6 adapter (is_ocean=True)
        When: parse_dart_obs_type() is called
        Then: config_utils.parse_obs_def_model_mod is called with the RST path
              and its return value is returned unchanged
        """
        adapter = ModelAdapterMOM6()
        with patch(self._MOCK_PARSE_TARGET, return_value=self._MOCK_RESULT) as mock_parser:
            result = adapter.parse_dart_obs_type('/path/to/')
            mock_parser.assert_called_once_with('/path/to/obs_def_ocean_mod.rst')
            assert result == self._MOCK_RESULT

    def test_parse_dart_obs_type_roms_delegates_to_config_utils(self):
        """Test parse_dart_obs_type calls parse_obs_def_model_mod for ROMS.

        Given: A ROMS_Rutgers adapter (is_ocean=True)
        When: parse_dart_obs_type() is called
        Then: config_utils.parse_obs_def_model_mod is called with the RST path
        """
        adapter = ModelAdapterROMSRutgers()
        with patch(self._MOCK_PARSE_TARGET, return_value=self._MOCK_RESULT) as mock_parser:
            result = adapter.parse_dart_obs_type('/path/to/')
            mock_parser.assert_called_once_with('/path/to/obs_def_ocean_mod.rst')
            assert result == self._MOCK_RESULT

    def test_parse_dart_obs_type_non_ocean_non_seaice_raises_not_implemented(self):
        """Test parse_dart_obs_type raises NotImplementedError for an adapter with no model type.

        Given: An adapter whose capabilities have is_ocean=False and is_sea_ice=False
        When: parse_dart_obs_type() is called
        Then: NotImplementedError is raised
        """
        class _NeutralAdapter(ModelAdapterMOM6):
            capabilities = ModelAdapterCapabilities(
                supports_trim_obs=True,
                supports_no_matching=True,
                supports_force_obs_time=True,
                is_ocean=False,
                is_sea_ice=False,
            )

        adapter = object.__new__(_NeutralAdapter)
        with pytest.raises(NotImplementedError):
            adapter.parse_dart_obs_type('/path/to/')

    def test_parse_dart_obs_type_with_valid_rst_file(self, fixtures_root: Path, tmp_path: Path):
        """Test parse_dart_obs_type returns correct dicts for an ocean adapter.

        Given: A MOM6 adapter and a directory containing obs_def_ocean_mod.rst
        When: parse_dart_obs_type() is called with the directory path
        Then: Returns non-empty obs_type_to_qty and qty_to_obs_types dicts
        """
        shutil.copy(
            fixtures_root / "mock_obs_def_ocean_mod.rst",
            tmp_path / "obs_def_ocean_mod.rst"
        )
        adapter = ModelAdapterMOM6()
        obs_type_to_qty, qty_to_obs_types = adapter.parse_dart_obs_type(str(tmp_path))

        assert isinstance(obs_type_to_qty, dict)
        assert isinstance(qty_to_obs_types, dict)
        assert len(obs_type_to_qty) > 0
        assert all(v.startswith('QTY_') for v in obs_type_to_qty.values())

    def test_parse_dart_obs_type_missing_rst_file(self):
        """Test parse_dart_obs_type raises FileNotFoundError for a missing RST file.

        Given: A MOM6 adapter and a nonexistent RST file path
        When: parse_dart_obs_type() is called
        Then: FileNotFoundError is raised
        """
        adapter = ModelAdapterMOM6()
        with pytest.raises(FileNotFoundError):
            adapter.parse_dart_obs_type('/nonexistent/obs_def_ocean_mod.rst')

    def test_parse_dart_obs_type_cice_delegates_to_config_utils(self):
        """Test parse_dart_obs_type calls parse_obs_def_model_mod for a CICE adapter.

        Given: An adapter with is_ocean=False and is_sea_ice=True
        When: parse_dart_obs_type() is called with a directory path
        Then: config_utils.parse_obs_def_model_mod is called with obs_def_cice_mod.f90
              appended to the directory, and its return value is returned unchanged
        """
        adapter = ModelAdapterCICE()
        with patch(self._MOCK_PARSE_TARGET, return_value=self._MOCK_RESULT) as mock_parser:
            result = adapter.parse_dart_obs_type('/path/to/')
            mock_parser.assert_called_once_with('/path/to/obs_def_cice_mod.f90')
            assert result == self._MOCK_RESULT

    def test_parse_dart_obs_type_with_valid_f90_file(self, fixtures_root: Path, tmp_path: Path):
        """Test parse_dart_obs_type returns correct dicts for a sea-ice adapter.

        Given: An adapter with is_sea_ice=True and a directory containing
               obs_def_cice_mod.f90 copied from the CICE fixture
        When: parse_dart_obs_type() is called with the directory path
        Then: Returns non-empty dicts with QTY_SEAICE_* values
        """
        shutil.copy(
            fixtures_root / "mock_obs_def_cice_mod.f90",
            tmp_path / "obs_def_cice_mod.f90"
        )
        adapter = ModelAdapterCICE()
        obs_type_to_qty, qty_to_obs_types = adapter.parse_dart_obs_type(str(tmp_path))

        assert isinstance(obs_type_to_qty, dict)
        assert isinstance(qty_to_obs_types, dict)
        assert len(obs_type_to_qty) > 0
        assert all(v.startswith('QTY_') for v in obs_type_to_qty.values())
        assert any('SEAICE' in v for v in obs_type_to_qty.values())


class TestGetModelBoundaries:
    """Test suite for get_model_boundaries() function."""
    
    def test_get_model_boundaries_basic_grid(self, tmp_path: Path):
        """Test get_model_boundaries with a simple rectangular grid.
        
        Given: A model file with a simple rectangular grid
        When: get_model_boundaries() is called
        Then: Returns a Polygon and hull points array
        """
        model_file = tmp_path / "model_grid.nc"
        
        lonh = np.array([10.0, 11.0, 12.0])
        lath = np.array([40.0, 41.0, 42.0])
        wet = np.ones((len(lath), len(lonh)), dtype=int)

        ds = xr.Dataset({
            'wet': (['lath', 'lonh'], wet),
            'lonh': lonh,
            'lath': lath,
        })
        ds.to_netcdf(model_file)

        adapter = create_model_adapter("mom6")
        hull_polygon, hull_points = adapter.get_model_boundaries(str(model_file))

        assert isinstance(hull_polygon, Polygon)
        assert isinstance(hull_points, np.ndarray)
        assert hull_polygon.is_valid
        assert len(hull_points) >= 3
    
    def test_get_model_boundaries_negative_longitude(self, tmp_path: Path):
        """Test get_model_boundaries converts negative longitudes to 0-360.
        
        Given: A model file with negative longitudes
        When: get_model_boundaries() is called
        Then: Hull points use 0-360 longitude convention
        """
        model_file = tmp_path / "model_neg_lon.nc"
        
        lonh = np.array([-10.0, -5.0, 0.0, 5.0])
        lath = np.array([40.0, 41.0, 42.0])
        wet = np.ones((len(lath), len(lonh)), dtype=int)

        ds = xr.Dataset({
            'wet': (['lath', 'lonh'], wet),
            'lonh': lonh,
            'lath': lath,
        })
        ds.to_netcdf(model_file)
        
        adapter = create_model_adapter("mom6")
        hull_polygon, hull_points = adapter.get_model_boundaries(str(model_file))
        
        assert np.all(hull_points[:, 0] >= 0)
        assert np.all(hull_points[:, 0] <= 360)
    
    def test_get_model_boundaries_with_dry_points(self, tmp_path: Path):
        """Test get_model_boundaries excludes dry grid points.
        
        Given: A model file with some dry points (wet=0)
        When: get_model_boundaries() is called
        Then: Only wet points are included in convex hull
        """
        model_file = tmp_path / "model_dry.nc"
        
        lonh = np.array([10.0, 11.0, 12.0, 13.0])
        lath = np.array([40.0, 41.0, 42.0, 43.0])
        wet = np.ones((len(lath), len(lonh)), dtype=int)
        wet[0, 0] = 0
        wet[0, -1] = 0
        wet[-1, 0] = 0

        ds = xr.Dataset({
            'wet': (['lath', 'lonh'], wet),
            'lonh': lonh,
            'lath': lath,
        })
        ds.to_netcdf(model_file)
        
        adapter = create_model_adapter("mom6")
        hull_polygon, hull_points = adapter.get_model_boundaries(str(model_file))
        
        assert hull_polygon.is_valid
        assert len(hull_points) >= 3
        num_wet = np.sum(wet == 1)
        assert len(hull_points) <= num_wet
    
    def test_get_model_boundaries_irregular_domain(self, tmp_path: Path):
        """Test get_model_boundaries with irregular domain shape.
        
        Given: A model file with L-shaped wet domain
        When: get_model_boundaries() is called
        Then: Convex hull encompasses the irregular shape
        """
        model_file = tmp_path / "model_irregular.nc"
        
        lonh = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        lath = np.array([40.0, 41.0, 42.0, 43.0, 44.0])
        wet = np.zeros((len(lath), len(lonh)), dtype=int)

        wet[0:3, 0:2] = 1
        wet[0:5, 2] = 1

        ds = xr.Dataset({
            'wet': (['lath', 'lonh'], wet),
            'lonh': lonh,
            'lath': lath,
        })
        ds.to_netcdf(model_file)
        
        adapter = create_model_adapter("mom6")
        hull_polygon, hull_points = adapter.get_model_boundaries(str(model_file))
        
        assert hull_polygon.is_valid
        assert hull_polygon.area > 0
    
    def test_get_model_boundaries_insufficient_points_raises_error(self, tmp_path: Path):
        """Test get_model_boundaries raises error with insufficient valid points.
        
        Given: A model file with fewer than 3 wet points
        When: get_model_boundaries() is called
        Then: ValueError is raised
        """
        model_file = tmp_path / "model_few_points.nc"
        
        lonh = np.array([10.0, 11.0])
        lath = np.array([40.0, 41.0])
        wet = np.zeros((len(lath), len(lonh)), dtype=int)
        wet[0, 0] = 1
        wet[1, 1] = 1

        ds = xr.Dataset({
            'wet': (['lath', 'lonh'], wet),
            'lonh': lonh,
            'lath': lath,
        })
        ds.to_netcdf(model_file)
        
        adapter = create_model_adapter("mom6")
        with pytest.raises(ValueError, match="Not enough valid points"):
            adapter.get_model_boundaries(str(model_file))
    
    def test_get_model_boundaries_missing_file(self):
        """Test get_model_boundaries raises error for missing file.
        
        Given: A path to a nonexistent file
        When: get_model_boundaries() is called
        Then: FileNotFoundError is raised
        """
        adapter = create_model_adapter("mom6")
        with pytest.raises(FileNotFoundError):
            adapter.get_model_boundaries("/nonexistent/model.nc")
    
    def test_get_model_boundaries_missing_required_variables(self, tmp_path: Path):
        """Test get_model_boundaries raises error when required variables missing.
        
        Given: A NetCDF file without lonh/lath/wet variables
        When: get_model_boundaries() is called
        Then: KeyError or similar error is raised
        """
        model_file = tmp_path / "model_no_vars.nc"
        
        ds = xr.Dataset({
            'temperature': (['x', 'y'], np.random.rand(3, 3)),
            'time': [np.datetime64('2020-06-15T12:00:00')]
        })
        ds.to_netcdf(model_file)
        
        adapter = create_model_adapter("mom6")
        with pytest.raises(KeyError):
            adapter.get_model_boundaries(str(model_file))
    
    def test_get_model_boundaries_all_dry_points(self, tmp_path: Path):
        """Test get_model_boundaries raises error when all points are dry.
        
        Given: A model file with all wet=0
        When: get_model_boundaries() is called
        Then: ValueError is raised
        """
        model_file = tmp_path / "model_all_dry.nc"
        
        lonh = np.array([10.0, 11.0, 12.0])
        lath = np.array([40.0, 41.0, 42.0])
        wet = np.zeros((len(lath), len(lonh)), dtype=int)

        ds = xr.Dataset({
            'wet': (['lath', 'lonh'], wet),
            'lonh': lonh,
            'lath': lath,
        })
        ds.to_netcdf(model_file)
        
        adapter = create_model_adapter("mom6")
        with pytest.raises(ValueError, match="Not enough valid points"):
            adapter.get_model_boundaries(str(model_file))
    
    def test_get_model_boundaries_output_format(self, tmp_path: Path):
        """Test get_model_boundaries output format and types.
        
        Given: A valid model file
        When: get_model_boundaries() is called
        Then: Returns tuple of (Polygon, ndarray) with correct shapes
        """
        model_file = tmp_path / "model_format.nc"
        
        lonh = np.array([10.0, 11.0, 12.0, 13.0])
        lath = np.array([40.0, 41.0, 42.0])
        wet = np.ones((len(lath), len(lonh)), dtype=int)

        ds = xr.Dataset({
            'wet': (['lath', 'lonh'], wet),
            'lonh': lonh,
            'lath': lath,
        })
        ds.to_netcdf(model_file)
        
        adapter = create_model_adapter("mom6")
        hull_polygon, hull_points = adapter.get_model_boundaries(str(model_file))
        
        assert isinstance(hull_polygon, Polygon)
        assert isinstance(hull_points, np.ndarray)
        assert hull_points.ndim == 2
        assert hull_points.shape[1] == 2
        assert len(list(hull_polygon.exterior.coords)) > 0
    
    def test_get_model_boundaries_boundary_coordinates(self, tmp_path: Path):
        """Test get_model_boundaries returns correct boundary coordinates.
        
        Given: A model file with known grid extent
        When: get_model_boundaries() is called
        Then: Hull points encompass expected lon/lat ranges
        """
        model_file = tmp_path / "model_bounds.nc"
        
        lon_min, lon_max = 10.0, 15.0
        lat_min, lat_max = 40.0, 45.0
        
        lonh = np.linspace(lon_min, lon_max, 6)
        lath = np.linspace(lat_min, lat_max, 6)
        wet = np.ones((len(lath), len(lonh)), dtype=int)

        ds = xr.Dataset({
            'wet': (['lath', 'lonh'], wet),
            'lonh': lonh,
            'lath': lath,
        })
        ds.to_netcdf(model_file)
        
        adapter = create_model_adapter("mom6")
        hull_polygon, hull_points = adapter.get_model_boundaries(str(model_file))
        
        hull_lon_min = hull_points[:, 0].min()
        hull_lon_max = hull_points[:, 0].max()
        hull_lat_min = hull_points[:, 1].min()
        hull_lat_max = hull_points[:, 1].max()
        
        assert hull_lon_min >= lon_min
        assert hull_lon_max <= lon_max
        assert hull_lat_min >= lat_min
        assert hull_lat_max <= lat_max

    def test_get_model_boundaries_geometry_file_no_time(self, tmp_path: Path):
        """Test get_model_boundaries works on a static geometry file with no time variable.

        The ocean_geometry file (used in production) contains only spatial variables —
        no time dimension.  Previously get_model_boundaries used a plain xr.open_dataset;
        the refactored version must not rely on the MOM6 open_dataset_ctx which tries to
        access self.time_varname ('time') and raises KeyError on timeless files.

        Given: A geometry-like NetCDF with lonh, lath, wet but NO time variable
        When: get_model_boundaries() is called
        Then: Returns a valid Polygon and hull points without raising KeyError
        """
        model_file = tmp_path / "ocean_geometry.nc"

        lonh = np.linspace(10.0, 15.0, 5)
        lath = np.linspace(40.0, 45.0, 5)
        wet = np.ones((len(lath), len(lonh)), dtype=int)

        ds = xr.Dataset({
            'wet': (['lath', 'lonh'], wet),
            'lonh': lonh,
            'lath': lath,
        })
        ds.to_netcdf(model_file)

        adapter = create_model_adapter("mom6")
        hull_polygon, hull_points = adapter.get_model_boundaries(str(model_file))

        assert isinstance(hull_polygon, Polygon)
        assert isinstance(hull_points, np.ndarray)
        assert hull_polygon.is_valid
