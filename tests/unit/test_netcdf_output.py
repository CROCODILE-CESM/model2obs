"""Unit tests for model2obs.io.netcdf_output.write_interpolated_to_netcdf."""

import warnings
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from model2obs.io.netcdf_output import write_interpolated_to_netcdf

_T0 = pd.Timestamp("2020-06-15 12:00:00")

# Default tolerances used in tests that do not exercise tolerance-merging.
_DEFAULT_TOLERANCES = {"longitude": 1e-2, "latitude": 1e-2, "depth": 1e-1}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ddf() -> dd.DataFrame:
    """Return a Dask DataFrame with 3 valid observations.

    Each row has a distinct longitude, latitude, and depth so the output
    grid contains exactly 3 non-NaN cells.  All rows share the same
    timestamp (2020-06-15 12:00:00).
    """
    df = pd.DataFrame({
        "interpolated_model": [1.5, 2.3, 0.8],
        "longitude": [10.0, 20.0, 30.0],
        "latitude": [40.0, 50.0, 60.0],
        "vertical": [0.0, 10.0, 50.0],
        "time": pd.to_datetime([_T0, _T0, _T0]).astype("datetime64[s]"),
        "interpolated_model_QC": [0, 4, 1018],
    })
    return dd.from_pandas(df, npartitions=1)


@pytest.fixture
def two_lat_ddf() -> dd.DataFrame:
    """Return a Dask DataFrame with two rows whose latitudes are close.

    Latitudes 30.003 and 30.005 both round to 30.0 under a tolerance of
    1e-2, so the output grid should have a single latitude coordinate.
    The rows share the same time, depth, and longitude.
    """
    df = pd.DataFrame({
        "interpolated_model": [1.0, 2.0],
        "longitude": [15.0, 15.0],
        "latitude": [30.003, 30.005],
        "vertical": [5.0, 5.0],
        "time": pd.to_datetime([_T0, _T0]).astype("datetime64[s]"),
        "interpolated_model_QC": [0, 0],
    })
    return dd.from_pandas(df, npartitions=1)


@pytest.fixture
def sparse_grid_ddf() -> dd.DataFrame:
    """Return a Dask DataFrame that produces a sparse 4-D grid.

    Two observations share the same time and depth but sit at different
    (lat, lon) corners of a 2x2 horizontal grid.  The two unfilled cells
    should contain NaN in the output.
    """
    df = pd.DataFrame({
        "interpolated_model": [5.0, 7.0],
        "longitude": [10.0, 20.0],
        "latitude": [40.0, 50.0],
        "vertical": [0.0, 0.0],
        "time": pd.to_datetime([_T0, _T0]).astype("datetime64[s]"),
        "interpolated_model_QC": [0, 0],
    })
    return dd.from_pandas(df, npartitions=1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasicWriting:
    """Tests verifying that the function produces a readable NetCDF file."""

    def test_file_is_created(self, sample_ddf: dd.DataFrame, tmp_path: Path):
        """File exists on disk after a successful call."""
        out = tmp_path / "out.nc"
        write_interpolated_to_netcdf(sample_ddf, str(out), _DEFAULT_TOLERANCES)
        assert out.exists()

    def test_file_is_openable_with_xarray(
        self, sample_ddf: dd.DataFrame, tmp_path: Path
    ):
        """Written file can be opened by xarray without errors."""
        out = tmp_path / "out.nc"
        write_interpolated_to_netcdf(sample_ddf, str(out), _DEFAULT_TOLERANCES)
        ds = xr.open_dataset(str(out))
        ds.close()


class TestDimensions:
    """Tests for coordinate dimensions in the output dataset."""

    def test_output_has_four_dimensions(
        self, sample_ddf: dd.DataFrame, tmp_path: Path
    ):
        """Output dataset contains time, depth, latitude, and longitude dims."""
        out = tmp_path / "out.nc"
        write_interpolated_to_netcdf(sample_ddf, str(out), _DEFAULT_TOLERANCES)
        with xr.open_dataset(str(out)) as ds:
            assert set(ds.dims) == {"time", "depth", "latitude", "longitude"}

    def test_depth_dimension_not_vertical(
        self, sample_ddf: dd.DataFrame, tmp_path: Path
    ):
        """The 'vertical' column is renamed to 'depth' in the output."""
        out = tmp_path / "out.nc"
        write_interpolated_to_netcdf(sample_ddf, str(out), _DEFAULT_TOLERANCES)
        with xr.open_dataset(str(out)) as ds:
            assert "depth" in ds.dims
            assert "vertical" not in ds.dims


class TestDataVariables:
    """Tests for the data variables in the output dataset."""

    def test_interpolated_model_variable_present(
        self, sample_ddf: dd.DataFrame, tmp_path: Path
    ):
        """'interpolated_model' variable exists with all four dimensions."""
        out = tmp_path / "out.nc"
        write_interpolated_to_netcdf(sample_ddf, str(out), _DEFAULT_TOLERANCES)
        with xr.open_dataset(str(out)) as ds:
            assert "interpolated_model" in ds
            assert set(ds["interpolated_model"].dims) == {
                "time", "depth", "latitude", "longitude"
            }

    def test_qc_flag_variable_present(
        self, sample_ddf: dd.DataFrame, tmp_path: Path
    ):
        """'qc_flag' variable exists with all four dimensions."""
        out = tmp_path / "out.nc"
        write_interpolated_to_netcdf(sample_ddf, str(out), _DEFAULT_TOLERANCES)
        with xr.open_dataset(str(out)) as ds:
            assert "qc_flag" in ds
            assert set(ds["qc_flag"].dims) == {
                "time", "depth", "latitude", "longitude"
            }


class TestCoordinateTolerance:
    """Tests for coordinate rounding / tolerance-based merging."""

    def test_close_latitudes_merged_to_single_value(
        self, two_lat_ddf: dd.DataFrame, tmp_path: Path
    ):
        """Two latitudes within tolerance round to one coordinate value."""
        out = tmp_path / "out.nc"
        write_interpolated_to_netcdf(
            two_lat_ddf, str(out), tolerances={"latitude": 1e-2}
        )
        with xr.open_dataset(str(out)) as ds:
            assert ds.sizes["latitude"] == 1


class TestFillValue:
    """Tests for NaN fill behaviour at unobserved grid points."""

    def test_missing_grid_point_is_nan(
        self, sparse_grid_ddf: dd.DataFrame, tmp_path: Path
    ):
        """Grid cells with no observation contain NaN in interpolated_model."""
        out = tmp_path / "out.nc"
        write_interpolated_to_netcdf(sparse_grid_ddf, str(out), _DEFAULT_TOLERANCES)
        with xr.open_dataset(str(out)) as ds:
            values = ds["interpolated_model"].values
            nan_count = np.sum(np.isnan(values))
            assert nan_count > 0


class TestCFCompliance:
    """Tests for CF-1.8 metadata on coordinates and variables."""

    def test_time_values_roundtrip(
        self, sample_ddf: dd.DataFrame, tmp_path: Path
    ):
        """Timestamps survive a write/read roundtrip."""
        out = tmp_path / "out.nc"
        write_interpolated_to_netcdf(sample_ddf, str(out), _DEFAULT_TOLERANCES)
        with xr.open_dataset(str(out)) as ds:
            decoded = pd.Timestamp(ds["time"].values[0])
            assert decoded == _T0

    def test_depth_cf_attributes(
        self, sample_ddf: dd.DataFrame, tmp_path: Path
    ):
        """Depth coordinate has standard_name, units, and positive='down'."""
        out = tmp_path / "out.nc"
        write_interpolated_to_netcdf(sample_ddf, str(out), _DEFAULT_TOLERANCES)
        with xr.open_dataset(str(out)) as ds:
            attrs = ds["depth"].attrs
            assert attrs.get("standard_name") == "depth"
            assert attrs.get("units") == "meters"
            assert attrs.get("positive") == "down"


class TestQCFlagAttributes:
    """Tests for qc_flag variable metadata."""

    def test_qc_flag_lacks_range_and_flag_metadata(
        self, sample_ddf: dd.DataFrame, tmp_path: Path
    ):
        """qc_flag has no valid_range, flag_values, or flag_meanings attrs."""
        out = tmp_path / "out.nc"
        write_interpolated_to_netcdf(sample_ddf, str(out), _DEFAULT_TOLERANCES)
        with xr.open_dataset(str(out)) as ds:
            attrs = ds["qc_flag"].attrs
            assert "valid_range" not in attrs
            assert "flag_values" not in attrs
            assert "flag_meanings" not in attrs


class TestEdgeCases:
    """Tests for empty DataFrames and missing column validation."""

    def test_empty_dataframe_writes_valid_netcdf(self, tmp_path: Path):
        """An empty Dask DataFrame produces a valid (empty) NetCDF file."""
        df = pd.DataFrame(columns=[
            "interpolated_model", "longitude", "latitude",
            "vertical", "time", "interpolated_model_QC",
        ])
        ddf = dd.from_pandas(df, npartitions=1)
        out = tmp_path / "empty.nc"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            write_interpolated_to_netcdf(ddf, str(out), _DEFAULT_TOLERANCES)
        assert out.exists()
        ds = xr.open_dataset(str(out))
        ds.close()

    def test_missing_required_columns_raises_value_error(self, tmp_path: Path):
        """ValueError is raised when the DataFrame is missing required columns."""
        df = pd.DataFrame({
            "interpolated_model": [1.0],
            "longitude": [10.0],
        })
        ddf = dd.from_pandas(df, npartitions=1)
        out = tmp_path / "bad.nc"
        with pytest.raises(ValueError, match="missing required columns"):
            write_interpolated_to_netcdf(ddf, str(out), _DEFAULT_TOLERANCES)
