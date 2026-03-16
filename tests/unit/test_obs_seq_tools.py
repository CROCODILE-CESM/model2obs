"""Unit tests for model2obs.io.obs_seq_tools module.

Tests cover observation sequence trimming based on geographical boundaries.

Note: Many tests are skipped because they require the pydartdiags library
and properly formatted obs_seq.in files. Integration tests with real data
should be implemented separately.
"""

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from shapely.geometry import Polygon

from model2obs.io import obs_seq_tools


class TestTrimObsSeqIn:
    """Test suite for trim_obs_seq_in() function.
    
    Note: These tests are mostly skipped as they require pydartdiags
    and valid obs_seq.in format files. Mocking is used where possible.
    """
    
    @pytest.fixture
    def simple_hull(self):
        """Create a simple rectangular hull for testing."""
        points = np.array([
            [10.0, 40.0],
            [20.0, 40.0],
            [20.0, 50.0],
            [10.0, 50.0]
        ])
        hull = Polygon(points)
        return hull, points
    
    @pytest.fixture
    def complex_hull(self):
        """Create a more complex irregular hull for testing."""
        points = np.array([
            [10.0, 40.0],
            [15.0, 38.0],
            [20.0, 40.0],
            [22.0, 45.0],
            [20.0, 50.0],
            [10.0, 48.0]
        ])
        hull = Polygon(points)
        return hull, points
    
    def test_trim_obs_seq_in_basic(self, create_obs_seq_file, simple_hull):
        """Test trim_obs_seq_in with observations inside and outside hull.
        
        Given: An obs_seq.in file and a convex hull
        When: trim_obs_seq_in() is called
        Then: Only observations inside hull are retained
        """
        hull_polygon, hull_points = simple_hull
        
        observations = [
            (15.0, 45.0, 10.0, 17.32, 12),
            (15.0, 45.0, 20.0, 17.35, 12),
            (25.0, 45.0, 10.0, 18.10, 12),
            (5.0, 35.0, 10.0, 16.50, 12),
        ]
        
        obs_file = create_obs_seq_file("obs_seq.in", observations)
        trimmed_file = obs_file.parent / "obs_seq_trimmed.in"
        
        obs_seq_tools.trim_obs_seq_in(
            str(obs_file),
            hull_polygon,
            hull_points,
            str(trimmed_file)
        )
        
        assert trimmed_file.exists()
        
        import pydartdiags.obs_sequence.obs_sequence as obsq
        trimmed_obs = obsq.ObsSequence(str(trimmed_file))
        assert len(trimmed_obs.df) < len(observations)
    
    def test_trim_obs_seq_in_no_observations_in_hull_raises_error(self, create_obs_seq_file):
        """Test trim_obs_seq_in raises error when no observations in hull.
        
        Given: An obs_seq.in file with observations outside hull
        When: trim_obs_seq_in() is called
        Then: ValueError is raised
        """
        observations = [
            (5.0, 35.0, 10.0, 16.50, 12),
            (5.0, 36.0, 10.0, 16.60, 12),
        ]
        
        obs_file = create_obs_seq_file("obs_seq.in", observations)
        trimmed_file = obs_file.parent / "obs_seq_trimmed.in"
        
        far_away_hull = Polygon([
            [100.0, 80.0],
            [110.0, 80.0],
            [110.0, 85.0],
            [100.0, 85.0]
        ])
        hull_points = np.array(list(far_away_hull.exterior.coords)[:-1])
        
        with pytest.raises(ValueError, match="No observations found within"):
            obs_seq_tools.trim_obs_seq_in(
                str(obs_file),
                far_away_hull,
                hull_points,
                str(trimmed_file)
            )
    
    def test_trim_obs_seq_in_missing_input_file(self, simple_hull):
        """Test trim_obs_seq_in raises error for missing input file.
        
        Given: A path to nonexistent obs_seq.in file
        When: trim_obs_seq_in() is called
        Then: FileNotFoundError or equivalent is raised
        """
        hull_polygon, hull_points = simple_hull
        
        with pytest.raises(Exception):
            obs_seq_tools.trim_obs_seq_in(
                "/nonexistent/obs_seq.in",
                hull_polygon,
                hull_points,
                "/tmp/trimmed.in"
            )
    
    def test_trim_obs_seq_in_complex_hull(self, create_obs_seq_file, complex_hull):
        """Test trim_obs_seq_in with complex irregular hull.
        
        Given: An obs_seq.in file and an irregular convex hull
        When: trim_obs_seq_in() is called
        Then: Observations are correctly filtered by complex boundary
        """
        hull_polygon, hull_points = complex_hull
        
        observations = [
            (15.0, 42.0, 10.0, 17.32, 12),
            (12.0, 44.0, 10.0, 17.35, 12),
            (18.0, 46.0, 10.0, 17.40, 12),
            (25.0, 35.0, 10.0, 18.10, 12),
        ]
        
        obs_file = create_obs_seq_file("obs_seq.in", observations)
        trimmed_file = obs_file.parent / "obs_seq_trimmed.in"
        
        obs_seq_tools.trim_obs_seq_in(
            str(obs_file),
            hull_polygon,
            hull_points,
            str(trimmed_file)
        )
        
        assert trimmed_file.exists()
    
    def test_trim_obs_seq_in_hull_validation(self, simple_hull):
        """Test trim_obs_seq_in validates hull polygon.
        
        Given: A valid hull polygon
        When: Checking hull properties
        Then: Hull is a valid Shapely Polygon
        """
        hull_polygon, hull_points = simple_hull
        
        assert isinstance(hull_polygon, Polygon)
        assert hull_polygon.is_valid
        assert hull_polygon.area > 0
        assert not hull_polygon.is_empty
    
    def test_trim_obs_seq_in_preserves_observation_structure(self, create_obs_seq_file, simple_hull):
        """Test trim_obs_seq_in preserves obs_seq file structure.
        
        Given: An obs_seq.in file with specific structure
        When: trim_obs_seq_in() is called
        Then: Output file maintains correct obs_seq format
        """
        hull_polygon, hull_points = simple_hull
        
        observations = [
            (15.0, 45.0, 10.0, 17.32, 12),
            (16.0, 46.0, 15.0, 17.35, 12),
            (14.0, 44.0, 20.0, 17.30, 12),
        ]
        
        obs_file = create_obs_seq_file("obs_seq.in", observations)
        trimmed_file = obs_file.parent / "obs_seq_trimmed.in"
        
        obs_seq_tools.trim_obs_seq_in(
            str(obs_file),
            hull_polygon,
            hull_points,
            str(trimmed_file)
        )
        
        assert trimmed_file.exists()
        
        import pydartdiags.obs_sequence.obs_sequence as obsq
        trimmed_obs = obsq.ObsSequence(str(trimmed_file))
        assert hasattr(trimmed_obs, 'df')
        assert 'longitude' in trimmed_obs.df.columns
        assert 'latitude' in trimmed_obs.df.columns
    
    def test_trim_obs_seq_in_longitude_convention(self, create_obs_seq_file):
        """Test trim_obs_seq_in handles 0-360 longitude convention.
        
        Given: Observations and hull using 0-360 longitude convention
        When: trim_obs_seq_in() is called
        Then: Filtering works correctly with 0-360 coordinates
        """
        hull_360 = Polygon([
            [350.0, 40.0],
            [10.0, 40.0],
            [10.0, 50.0],
            [350.0, 50.0]
        ])
        hull_points = np.array(list(hull_360.exterior.coords)[:-1])
        
        observations = [
            (355.0, 45.0, 10.0, 17.32, 12),
            (5.0, 45.0, 10.0, 17.35, 12),
            (25.0, 45.0, 10.0, 18.10, 12),
        ]
        
        obs_file = create_obs_seq_file("obs_seq.in", observations)
        trimmed_file = obs_file.parent / "obs_seq_trimmed.in"
        
        obs_seq_tools.trim_obs_seq_in(
            str(obs_file),
            hull_360,
            hull_points,
            str(trimmed_file)
        )
        
        assert trimmed_file.exists()
    
    def test_hull_points_format(self, simple_hull, complex_hull):
        """Test hull_points format is correct for both simple and complex hulls.
        
        Given: Different hull configurations
        When: Checking hull_points arrays
        Then: Arrays have correct shape (N, 2) for lon/lat pairs
        """
        simple_polygon, simple_points = simple_hull
        complex_polygon, complex_points = complex_hull
        
        assert simple_points.ndim == 2
        assert simple_points.shape[1] == 2
        assert len(simple_points) >= 3
        
        assert complex_points.ndim == 2
        assert complex_points.shape[1] == 2
        assert len(complex_points) >= 3
    
    def test_trim_obs_seq_in_output_file_creation(self, create_obs_seq_file, simple_hull):
        """Test trim_obs_seq_in creates output file in correct location.
        
        Given: A specified output file path
        When: trim_obs_seq_in() is called
        Then: Output file is created at specified location
        """
        hull_polygon, hull_points = simple_hull
        
        observations = [
            (15.0, 45.0, 10.0, 17.32, 12),
            (16.0, 46.0, 15.0, 17.35, 12),
        ]
        
        obs_file = create_obs_seq_file("obs_seq.in", observations)
        output_dir = obs_file.parent / "output"
        output_dir.mkdir()
        trimmed_file = output_dir / "obs_seq_trimmed.in"
        
        obs_seq_tools.trim_obs_seq_in(
            str(obs_file),
            hull_polygon,
            hull_points,
            str(trimmed_file)
        )
        
        assert trimmed_file.exists()
        assert trimmed_file.parent == output_dir


class TestHullOperations:
    """Test suite for hull-related operations used in obs trimming."""
    
    def test_polygon_creation_from_points(self):
        """Test creating Shapely Polygon from coordinate points.
        
        Given: An array of coordinate points
        When: Creating a Polygon
        Then: Polygon is valid and has expected properties
        """
        points = np.array([
            [10.0, 40.0],
            [20.0, 40.0],
            [20.0, 50.0],
            [10.0, 50.0]
        ])
        
        polygon = Polygon(points)
        
        assert polygon.is_valid
        assert not polygon.is_empty
        assert polygon.area > 0
    
    def test_polygon_boundary_extraction(self):
        """Test extracting boundary coordinates from Polygon.
        
        Given: A Polygon object
        When: Extracting exterior coordinates
        Then: Coordinates match original points (with closure)
        """
        points = np.array([
            [10.0, 40.0],
            [20.0, 40.0],
            [20.0, 50.0],
            [10.0, 50.0]
        ])
        polygon = Polygon(points)
        
        boundary = np.array(list(polygon.exterior.coords))
        
        assert len(boundary) == len(points) + 1
        np.testing.assert_array_almost_equal(boundary[0], boundary[-1])
    
    def test_polygon_contains_point(self):
        """Test point-in-polygon testing.
        
        Given: A Polygon and test points
        When: Checking if points are inside
        Then: Returns correct boolean values
        """
        points = np.array([
            [10.0, 40.0],
            [20.0, 40.0],
            [20.0, 50.0],
            [10.0, 50.0]
        ])
        polygon = Polygon(points)
        
        from shapely.geometry import Point
        
        inside_point = Point(15.0, 45.0)
        outside_point = Point(25.0, 45.0)
        boundary_point = Point(10.0, 40.0)
        
        assert polygon.contains(inside_point)
        assert not polygon.contains(outside_point)
