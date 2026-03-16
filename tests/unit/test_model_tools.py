"""Unit tests for model2obs.io.model_tools module.

Tests cover model grid boundary extraction and convex hull computation.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from shapely.geometry import Polygon

from model2obs.io import model_tools


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
        
        hull_polygon, hull_points = model_tools.get_model_boundaries(str(model_file))
        
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
        
        hull_polygon, hull_points = model_tools.get_model_boundaries(str(model_file))
        
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
        
        hull_polygon, hull_points = model_tools.get_model_boundaries(str(model_file))
        
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
        
        hull_polygon, hull_points = model_tools.get_model_boundaries(str(model_file))
        
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
        
        with pytest.raises(ValueError, match="Not enough valid points"):
            model_tools.get_model_boundaries(str(model_file))
    
    def test_get_model_boundaries_missing_file(self):
        """Test get_model_boundaries raises error for missing file.
        
        Given: A path to a nonexistent file
        When: get_model_boundaries() is called
        Then: FileNotFoundError is raised
        """
        with pytest.raises(FileNotFoundError):
            model_tools.get_model_boundaries("/nonexistent/model.nc")
    
    def test_get_model_boundaries_missing_required_variables(self, tmp_path: Path):
        """Test get_model_boundaries raises error when required variables missing.
        
        Given: A NetCDF file without lonh/lath/wet variables
        When: get_model_boundaries() is called
        Then: KeyError or similar error is raised
        """
        model_file = tmp_path / "model_no_vars.nc"
        
        ds = xr.Dataset({
            'temperature': (['x', 'y'], np.random.rand(3, 3)),
        })
        ds.to_netcdf(model_file)
        
        with pytest.raises(KeyError):
            model_tools.get_model_boundaries(str(model_file))
    
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
        
        with pytest.raises(ValueError, match="Not enough valid points"):
            model_tools.get_model_boundaries(str(model_file))
    
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
        
        hull_polygon, hull_points = model_tools.get_model_boundaries(str(model_file))
        
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
        
        hull_polygon, hull_points = model_tools.get_model_boundaries(str(model_file))
        
        hull_lon_min = hull_points[:, 0].min()
        hull_lon_max = hull_points[:, 0].max()
        hull_lat_min = hull_points[:, 1].min()
        hull_lat_max = hull_points[:, 1].max()
        
        assert hull_lon_min >= lon_min
        assert hull_lon_max <= lon_max
        assert hull_lat_min >= lat_min
        assert hull_lat_max <= lat_max
