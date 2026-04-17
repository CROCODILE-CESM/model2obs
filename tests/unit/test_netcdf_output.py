"""Unit tests for model2obs.io.netcdf_output.write_interpolated_to_netcdf."""

import warnings
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from model2obs.io.netcdf_output import (
    _has_unique_latlon_pairs,
    write_interpolated_to_netcdf,
)

_T0 = pd.Timestamp("2020-06-15 12:00:00")

# Default tolerances used in tests that do not exercise tolerance-merging.
_DEFAULT_TOLERANCES = {"longitude": 1e-2, "latitude": 1e-2, "depth": 1e-1}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ddf() -> dd.DataFrame:
    """Return a Dask DataFrame with 3 valid observations of type TEMPERATURE.

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
        "type": ["TEMPERATURE", "TEMPERATURE", "TEMPERATURE"],
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
        "type": ["TEMPERATURE", "TEMPERATURE"],
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
        "type": ["TEMPERATURE", "TEMPERATURE"],
    })
    return dd.from_pandas(df, npartitions=1)


@pytest.fixture
def grid_ddf() -> dd.DataFrame:
    """Return a Dask DataFrame whose lat-lon pairs are NOT bijective.

    Latitude 40.0 appears with two different longitudes (10.0 and 20.0),
    so the output must use grid mode (4-D).  The unfilled grid cell
    (lat=50, lon=10) contains NaN.
    """
    df = pd.DataFrame({
        "interpolated_model": [1.0, 2.0, 3.0],
        "longitude": [10.0, 20.0, 10.0],
        "latitude": [40.0, 40.0, 50.0],
        "vertical": [0.0, 0.0, 0.0],
        "time": pd.to_datetime([_T0, _T0, _T0]).astype("datetime64[s]"),
        "interpolated_model_QC": [0, 0, 0],
        "type": ["TEMPERATURE", "TEMPERATURE", "TEMPERATURE"],
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
        self, grid_ddf: dd.DataFrame, tmp_path: Path
    ):
        """Grid-mode output contains time, depth, latitude, and longitude dims."""
        out = tmp_path / "out.nc"
        write_interpolated_to_netcdf(grid_ddf, str(out), _DEFAULT_TOLERANCES)
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
        self, grid_ddf: dd.DataFrame, tmp_path: Path
    ):
        """'interpolated_TEMPERATURE' variable exists with all four dimensions in grid mode."""
        out = tmp_path / "out.nc"
        write_interpolated_to_netcdf(grid_ddf, str(out), _DEFAULT_TOLERANCES)
        with xr.open_dataset(str(out)) as ds:
            assert "interpolated_TEMPERATURE" in ds
            assert set(ds["interpolated_TEMPERATURE"].dims) == {
                "time", "depth", "latitude", "longitude"
            }

    def test_qc_flag_variable_present(
        self, grid_ddf: dd.DataFrame, tmp_path: Path
    ):
        """'qc_flag_TEMPERATURE' variable exists with all four dimensions in grid mode."""
        out = tmp_path / "out.nc"
        write_interpolated_to_netcdf(grid_ddf, str(out), _DEFAULT_TOLERANCES)
        with xr.open_dataset(str(out)) as ds:
            assert "qc_flag_TEMPERATURE" in ds
            assert set(ds["qc_flag_TEMPERATURE"].dims) == {
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
        self, grid_ddf: dd.DataFrame, tmp_path: Path
    ):
        """Grid cells with no observation contain NaN in interpolated_TEMPERATURE."""
        out = tmp_path / "out.nc"
        write_interpolated_to_netcdf(grid_ddf, str(out), _DEFAULT_TOLERANCES)
        with xr.open_dataset(str(out)) as ds:
            values = ds["interpolated_TEMPERATURE"].values
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
        """qc_flag_TEMPERATURE has no valid_range, flag_values, or flag_meanings attrs."""
        out = tmp_path / "out.nc"
        write_interpolated_to_netcdf(sample_ddf, str(out), _DEFAULT_TOLERANCES)
        with xr.open_dataset(str(out)) as ds:
            attrs = ds["qc_flag_TEMPERATURE"].attrs
            assert "valid_range" not in attrs
            assert "flag_values" not in attrs
            assert "flag_meanings" not in attrs


class TestEdgeCases:
    """Tests for empty DataFrames and missing column validation."""

    def test_empty_dataframe_writes_valid_netcdf(self, tmp_path: Path):
        """An empty Dask DataFrame produces a valid (empty) NetCDF file."""
        df = pd.DataFrame(columns=[
            "interpolated_model", "longitude", "latitude",
            "vertical", "time", "interpolated_model_QC", "type",
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


# ---------------------------------------------------------------------------
# New tests for the bijection check helper
# ---------------------------------------------------------------------------

class TestHasUniqueLatLonPairs:
    """Unit tests for the _has_unique_latlon_pairs helper."""

    def test_bijective_pairs_return_true(self):
        """Distinct one-to-one lat-lon pairs are detected as bijective."""
        lat = np.array([40.0, 50.0, 60.0])
        lon = np.array([10.0, 20.0, 30.0])
        assert _has_unique_latlon_pairs(lat, lon) is True

    def test_same_lat_two_lons_returns_false(self):
        """A latitude paired with two different longitudes is not bijective."""
        lat = np.array([40.0, 40.0, 50.0])
        lon = np.array([10.0, 20.0, 30.0])
        assert _has_unique_latlon_pairs(lat, lon) is False

    def test_same_lon_two_lats_returns_false(self):
        """A longitude paired with two different latitudes is not bijective."""
        lat = np.array([40.0, 50.0, 60.0])
        lon = np.array([10.0, 10.0, 30.0])
        assert _has_unique_latlon_pairs(lat, lon) is False

    def test_single_observation_returns_true(self):
        """A single (lat, lon) pair is trivially bijective."""
        assert _has_unique_latlon_pairs(np.array([40.0]), np.array([10.0])) is True

    def test_duplicate_identical_pairs_return_true(self):
        """Repeated identical pairs deduplicate to a bijective set."""
        lat = np.array([40.0, 40.0, 50.0, 50.0])
        lon = np.array([10.0, 10.0, 20.0, 20.0])
        assert _has_unique_latlon_pairs(lat, lon) is True


# ---------------------------------------------------------------------------
# New tests for transect mode (bijective lat-lon)
# ---------------------------------------------------------------------------

class TestTransectMode:
    """Tests for transect-mode output (bijective lat-lon pairs)."""

    def test_transect_mode_has_three_dimensions(
        self, sample_ddf: dd.DataFrame, tmp_path: Path
    ):
        """Bijective input produces a dataset with time, depth, latitude dims only."""
        out = tmp_path / "out.nc"
        write_interpolated_to_netcdf(sample_ddf, str(out), _DEFAULT_TOLERANCES)
        with xr.open_dataset(str(out)) as ds:
            assert set(ds.dims) == {"time", "depth", "latitude"}

    def test_transect_mode_longitude_is_coordinate_not_dim(
        self, sample_ddf: dd.DataFrame, tmp_path: Path
    ):
        """Longitude is present as a coordinate but not as a dimension."""
        out = tmp_path / "out.nc"
        write_interpolated_to_netcdf(sample_ddf, str(out), _DEFAULT_TOLERANCES)
        with xr.open_dataset(str(out)) as ds:
            assert "longitude" in ds.coords
            assert "longitude" not in ds.dims

    def test_transect_mode_no_spurious_nans(
        self, tmp_path: Path
    ):
        """Transect mode adds no NaN cells from a lon dimension cross-product.

        We use a fixture where every (time, depth) combination has exactly one
        observation per latitude, so the 3-D grid (time × depth × latitude) is
        fully dense — no NaN at all.  This confirms that longitude does not
        inflate the dataset with empty cells.
        """
        df = pd.DataFrame({
            "interpolated_model": [1.5, 2.3, 0.8],
            "longitude": [10.0, 20.0, 30.0],
            "latitude": [40.0, 50.0, 60.0],
            "vertical": [0.0, 0.0, 0.0],   # same depth for all
            "time": pd.to_datetime([_T0, _T0, _T0]).astype("datetime64[s]"),
            "interpolated_model_QC": [0, 4, 1018],
            "type": ["TEMPERATURE", "TEMPERATURE", "TEMPERATURE"],
        })
        ddf = dd.from_pandas(df, npartitions=1)
        out = tmp_path / "out.nc"
        write_interpolated_to_netcdf(ddf, str(out), _DEFAULT_TOLERANCES)
        with xr.open_dataset(str(out)) as ds:
            assert not np.any(np.isnan(ds["interpolated_TEMPERATURE"].values))

    def test_transect_mode_longitude_values_correct(
        self, sample_ddf: dd.DataFrame, tmp_path: Path
    ):
        """Longitude coordinate values match the original input data."""
        out = tmp_path / "out.nc"
        write_interpolated_to_netcdf(sample_ddf, str(out), _DEFAULT_TOLERANCES)
        with xr.open_dataset(str(out)) as ds:
            np.testing.assert_allclose(
                sorted(ds["longitude"].values),
                sorted([10.0, 20.0, 30.0]),
                atol=1e-2,
            )

    def test_transect_mode_coordinate_structure_attr(
        self, sample_ddf: dd.DataFrame, tmp_path: Path
    ):
        """Global attribute coordinate_structure is set to 'transect'."""
        out = tmp_path / "out.nc"
        write_interpolated_to_netcdf(sample_ddf, str(out), _DEFAULT_TOLERANCES)
        with xr.open_dataset(str(out)) as ds:
            assert ds.attrs.get("coordinate_structure") == "transect"

    def test_grid_mode_coordinate_structure_attr(
        self, grid_ddf: dd.DataFrame, tmp_path: Path
    ):
        """Global attribute coordinate_structure is set to 'grid' for non-bijective input."""
        out = tmp_path / "out.nc"
        write_interpolated_to_netcdf(grid_ddf, str(out), _DEFAULT_TOLERANCES)
        with xr.open_dataset(str(out)) as ds:
            assert ds.attrs.get("coordinate_structure") == "grid"


# ---------------------------------------------------------------------------
# Tests for per-obs-type variable generation
# ---------------------------------------------------------------------------

class TestPerTypeVariables:
    """Tests for per-observation-type variable naming in the output dataset."""

    @pytest.fixture
    def multi_type_transect_ddf(self) -> dd.DataFrame:
        """Return a bijective DataFrame with two observation types.

        TEMPERATURE at lat 40/50 and SALINITY at lat 40/50 (same locations).
        Both types share the same transect layout.
        """
        df = pd.DataFrame({
            "interpolated_model": [1.5, 2.3, 35.0, 35.5],
            "longitude": [10.0, 20.0, 10.0, 20.0],
            "latitude": [40.0, 50.0, 40.0, 50.0],
            "vertical": [0.0, 0.0, 0.0, 0.0],
            "time": pd.to_datetime([_T0, _T0, _T0, _T0]).astype("datetime64[s]"),
            "interpolated_model_QC": [0, 0, 0, 0],
            "type": ["TEMPERATURE", "TEMPERATURE", "SALINITY", "SALINITY"],
        })
        return dd.from_pandas(df, npartitions=1)

    @pytest.fixture
    def sparse_type_transect_ddf(self) -> dd.DataFrame:
        """Return a bijective DataFrame where the two types cover different latitudes.

        TEMPERATURE is observed at lat=40 only; SALINITY at lat=50 only.
        Both share the same longitude mapping (40↔10, 50↔20).
        On the shared 2-latitude grid, each type will have one NaN cell.
        """
        df = pd.DataFrame({
            "interpolated_model": [1.5, 35.5],
            "longitude": [10.0, 20.0],
            "latitude": [40.0, 50.0],
            "vertical": [0.0, 0.0],
            "time": pd.to_datetime([_T0, _T0]).astype("datetime64[s]"),
            "interpolated_model_QC": [0, 0],
            "type": ["TEMPERATURE", "SALINITY"],
        })
        return dd.from_pandas(df, npartitions=1)

    @pytest.fixture
    def multi_type_grid_ddf(self) -> dd.DataFrame:
        """Return a non-bijective DataFrame with two observation types.

        Lat 40 maps to both lon 10 and lon 20, forcing grid mode.
        TEMPERATURE covers (lat=40,lon=10) and (lat=40,lon=20).
        SALINITY covers (lat=40,lon=10) and (lat=50,lon=10).
        """
        df = pd.DataFrame({
            "interpolated_model": [1.0, 2.0, 35.0, 36.0],
            "longitude": [10.0, 20.0, 10.0, 10.0],
            "latitude": [40.0, 40.0, 40.0, 50.0],
            "vertical": [0.0, 0.0, 0.0, 0.0],
            "time": pd.to_datetime([_T0, _T0, _T0, _T0]).astype("datetime64[s]"),
            "interpolated_model_QC": [0, 0, 0, 0],
            "type": ["TEMPERATURE", "TEMPERATURE", "SALINITY", "SALINITY"],
        })
        return dd.from_pandas(df, npartitions=1)

    def test_multi_type_variables_present_transect(
        self, multi_type_transect_ddf: dd.DataFrame, tmp_path: Path
    ):
        """Both interpolated_TEMPERATURE and interpolated_SALINITY are present."""
        out = tmp_path / "out.nc"
        write_interpolated_to_netcdf(
            multi_type_transect_ddf, str(out), _DEFAULT_TOLERANCES
        )
        with xr.open_dataset(str(out)) as ds:
            assert "interpolated_TEMPERATURE" in ds
            assert "interpolated_SALINITY" in ds
            assert "qc_flag_TEMPERATURE" in ds
            assert "qc_flag_SALINITY" in ds

    def test_no_generic_interpolated_model_variable(
        self, multi_type_transect_ddf: dd.DataFrame, tmp_path: Path
    ):
        """The old 'interpolated_model' variable is no longer generated."""
        out = tmp_path / "out.nc"
        write_interpolated_to_netcdf(
            multi_type_transect_ddf, str(out), _DEFAULT_TOLERANCES
        )
        with xr.open_dataset(str(out)) as ds:
            assert "interpolated_model" not in ds
            assert "qc_flag" not in ds

    def test_multi_type_variables_present_grid(
        self, multi_type_grid_ddf: dd.DataFrame, tmp_path: Path
    ):
        """Both types generate variables in grid mode."""
        out = tmp_path / "out.nc"
        write_interpolated_to_netcdf(
            multi_type_grid_ddf, str(out), _DEFAULT_TOLERANCES
        )
        with xr.open_dataset(str(out)) as ds:
            assert "interpolated_TEMPERATURE" in ds
            assert "interpolated_SALINITY" in ds
            assert set(ds["interpolated_TEMPERATURE"].dims) == {
                "time", "depth", "latitude", "longitude"
            }

    def test_per_type_values_correct(
        self, multi_type_transect_ddf: dd.DataFrame, tmp_path: Path
    ):
        """Each type variable contains only its own observations."""
        out = tmp_path / "out.nc"
        write_interpolated_to_netcdf(
            multi_type_transect_ddf, str(out), _DEFAULT_TOLERANCES
        )
        with xr.open_dataset(str(out)) as ds:
            temp_vals = sorted(ds["interpolated_TEMPERATURE"].values.ravel().tolist())
            sal_vals = sorted(ds["interpolated_SALINITY"].values.ravel().tolist())
            np.testing.assert_allclose(temp_vals, [1.5, 2.3], atol=1e-3)
            np.testing.assert_allclose(sal_vals, [35.0, 35.5], atol=1e-3)

    def test_sparse_type_produces_nans_on_shared_grid(
        self, sparse_type_transect_ddf: dd.DataFrame, tmp_path: Path
    ):
        """A type covering only part of the grid gets NaN at the missing locations."""
        out = tmp_path / "out.nc"
        write_interpolated_to_netcdf(
            sparse_type_transect_ddf, str(out), _DEFAULT_TOLERANCES
        )
        with xr.open_dataset(str(out)) as ds:
            # TEMPERATURE only at lat=40 → lat=50 cell is NaN
            temp_vals = ds["interpolated_TEMPERATURE"].values.ravel()
            assert np.sum(np.isnan(temp_vals)) == 1
            # SALINITY only at lat=50 → lat=40 cell is NaN
            sal_vals = ds["interpolated_SALINITY"].values.ravel()
            assert np.sum(np.isnan(sal_vals)) == 1

    def test_single_type_uses_type_name(
        self, sample_ddf: dd.DataFrame, tmp_path: Path
    ):
        """A single-type DataFrame produces 'interpolated_TEMPERATURE', not 'interpolated_model'."""
        out = tmp_path / "out.nc"
        write_interpolated_to_netcdf(sample_ddf, str(out), _DEFAULT_TOLERANCES)
        with xr.open_dataset(str(out)) as ds:
            assert "interpolated_TEMPERATURE" in ds
            assert "interpolated_model" not in ds
