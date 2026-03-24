"""Unit tests for model2obs.io.file_utils module.

Tests cover file discovery, timestamp conversions, and time extraction from model/obs files.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from model2obs.io import file_utils


class TestGetSortedFiles:
    """Test suite for get_sorted_files() function."""
    
    def test_get_sorted_files_basic(self, tmp_path: Path):
        """Test get_sorted_files returns sorted file list.
        
        Given: A directory with multiple files
        When: get_sorted_files() is called
        Then: Files are returned in sorted order
        """
        test_dir = tmp_path / "files"
        test_dir.mkdir()
        
        files = ["file_c.txt", "file_a.txt", "file_b.txt"]
        for fname in files:
            (test_dir / fname).touch()
        
        result = file_utils.get_sorted_files(str(test_dir))
        
        assert len(result) == 3
        assert Path(result[0]).name == "file_a.txt"
        assert Path(result[1]).name == "file_b.txt"
        assert Path(result[2]).name == "file_c.txt"
    
    def test_get_sorted_files_with_pattern(self, tmp_path: Path):
        """Test get_sorted_files filters by pattern.
        
        Given: A directory with mixed file types
        When: get_sorted_files() is called with a pattern
        Then: Only matching files are returned
        """
        test_dir = tmp_path / "mixed"
        test_dir.mkdir()
        
        (test_dir / "data1.nc").touch()
        (test_dir / "data2.nc").touch()
        (test_dir / "readme.txt").touch()
        (test_dir / "script.py").touch()
        
        result = file_utils.get_sorted_files(str(test_dir), "*.nc")
        
        assert len(result) == 2
        assert all(f.endswith('.nc') for f in result)
    
    def test_get_sorted_files_empty_directory(self, tmp_path: Path):
        """Test get_sorted_files returns empty list for empty directory.
        
        Given: An empty directory
        When: get_sorted_files() is called
        Then: Empty list is returned
        """
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        result = file_utils.get_sorted_files(str(empty_dir))
        
        assert result == []
    
    def test_get_sorted_files_excludes_directories(self, tmp_path: Path):
        """Test get_sorted_files excludes subdirectories.
        
        Given: A directory containing files and subdirectories
        When: get_sorted_files() is called
        Then: Only files are returned, not directories
        """
        test_dir = tmp_path / "mixed_content"
        test_dir.mkdir()
        
        (test_dir / "file1.txt").touch()
        (test_dir / "file2.txt").touch()
        (test_dir / "subdir").mkdir()
        
        result = file_utils.get_sorted_files(str(test_dir))
        
        assert len(result) == 2
        assert all(Path(f).is_file() for f in result)
    
    def test_get_sorted_files_nonexistent_directory(self):
        """Test get_sorted_files returns empty list for nonexistent directory.
        
        Given: A nonexistent directory path
        When: get_sorted_files() is called
        Then: Empty list is returned
        """
        result = file_utils.get_sorted_files("/nonexistent/path")
        
        assert result == []


class TestTimestampToDaysSeconds:
    """Test suite for timestamp_to_days_seconds() function."""
    
    def test_timestamp_to_days_seconds_reference_date(self):
        """Test timestamp_to_days_seconds with reference date.
        
        Given: A timestamp at 1601-01-01 00:00:00
        When: timestamp_to_days_seconds() is called
        Then: Returns (0, 0)
        """
        timestamp = np.datetime64('1601-01-01T00:00:00')
        
        days, seconds = file_utils.timestamp_to_days_seconds(timestamp)
        
        assert days == 0
        assert seconds == 0
    
    def test_timestamp_to_days_seconds_one_day_later(self):
        """Test timestamp_to_days_seconds one day after reference.
        
        Given: A timestamp at 1601-01-02 00:00:00
        When: timestamp_to_days_seconds() is called
        Then: Returns (1, 0)
        """
        timestamp = np.datetime64('1601-01-02T00:00:00')
        
        days, seconds = file_utils.timestamp_to_days_seconds(timestamp)
        
        assert days == 1
        assert seconds == 0
    
    def test_timestamp_to_days_seconds_with_time(self):
        """Test timestamp_to_days_seconds with hours, minutes, seconds.
        
        Given: A timestamp at 1601-01-01 12:30:45
        When: timestamp_to_days_seconds() is called
        Then: Returns (0, 45045) where 45045 = 12*3600 + 30*60 + 45
        """
        timestamp = np.datetime64('1601-01-01T12:30:45')
        
        days, seconds = file_utils.timestamp_to_days_seconds(timestamp)
        
        assert days == 0
        assert seconds == 12 * 3600 + 30 * 60 + 45
    
    def test_timestamp_to_days_seconds_modern_date(self):
        """Test timestamp_to_days_seconds with a modern date.
        
        Given: A timestamp at 2020-01-01 00:00:00
        When: timestamp_to_days_seconds() is called
        Then: Returns correct days count (419 years * 365.25 days)
        """
        timestamp = np.datetime64('2020-01-01T00:00:00')
        
        days, seconds = file_utils.timestamp_to_days_seconds(timestamp)
        
        assert days > 150000
        assert seconds == 0
    
    def test_timestamp_to_days_seconds_leap_year_handling(self):
        """Test timestamp_to_days_seconds handles leap years.
        
        Given: Timestamps spanning a leap year
        When: timestamp_to_days_seconds() is called
        Then: Day counts differ by 366 for leap year
        """
        ts1 = np.datetime64('2020-01-01T00:00:00')
        ts2 = np.datetime64('2021-01-01T00:00:00')
        
        days1, _ = file_utils.timestamp_to_days_seconds(ts1)
        days2, _ = file_utils.timestamp_to_days_seconds(ts2)
        
        assert days2 - days1 == 366


class TestGetObsTimeInDaysSeconds:
    """Test suite for get_obs_time_in_days_seconds() function.
    
    Note: These tests require pydartdiags.obs_sequence module.
    They will be skipped if the module is not available.
    """
    
    def test_get_obs_time_in_days_seconds_valid_file(self, create_obs_seq_file):
        """Test get_obs_time_in_days_seconds with valid obs_seq.in file.
        
        Given: A valid obs_seq.in file with observations
        When: get_obs_time_in_days_seconds() is called
        Then: Midpoint time in days and seconds is returned
        """
        observations = [
            (15.0, 45.0, 10.0, 17.32, 12),
            (16.0, 46.0, 15.0, 17.35, 12),
        ]
        
        obs_file = create_obs_seq_file("obs_seq.in", observations)
        
        days, seconds = file_utils.get_obs_time_in_days_seconds(str(obs_file))
        
        assert isinstance(days, (int, np.integer))
        assert isinstance(seconds, (int, np.integer))
        assert 0 <= seconds < 86400
    
    def test_get_obs_time_in_days_seconds_missing_file(self):
        """Test get_obs_time_in_days_seconds raises error for missing file.
        
        Given: A path to a nonexistent obs_seq.in file
        When: get_obs_time_in_days_seconds() is called
        Then: FileNotFoundError or equivalent is raised
        """
        with pytest.raises(Exception):
            file_utils.get_obs_time_in_days_seconds("/nonexistent/obs_seq.in")
