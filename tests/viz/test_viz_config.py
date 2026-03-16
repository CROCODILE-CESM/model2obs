"""Tests for visualization configuration classes."""

import pytest
from model2obs.viz.viz_config import ProfileConfig, MapConfig


class TestProfileConfig:
    """Test ProfileConfig initialization and defaults."""
    
    def test_default_initialization(self):
        """Test ProfileConfig with default values."""
        config = ProfileConfig()
        
        assert config.colormap == 'viridis'
        assert config.figure_size == (10, 8)
        assert config.marker_size == 5
        assert config.marker_alpha == 1
        assert config.invert_yaxis is True
        assert config.grid is True
        assert config.initial_x is None
        assert config.initial_y is None
    
    def test_custom_initialization(self):
        """Test ProfileConfig with custom values."""
        config = ProfileConfig(
            colormap='plasma',
            figure_size=(12, 10),
            marker_size=10,
            marker_alpha=0.5,
            invert_yaxis=False,
            grid=False,
            initial_x='temperature',
            initial_y='depth'
        )
        
        assert config.colormap == 'plasma'
        assert config.figure_size == (12, 10)
        assert config.marker_size == 10
        assert config.marker_alpha == 0.5
        assert config.invert_yaxis is False
        assert config.grid is False
        assert config.initial_x == 'temperature'
        assert config.initial_y == 'depth'
    
    def test_partial_custom_initialization(self):
        """Test ProfileConfig with some custom values."""
        config = ProfileConfig(colormap='coolwarm', marker_size=15)
        
        assert config.colormap == 'coolwarm'
        assert config.marker_size == 15
        # Verify defaults for unspecified params
        assert config.figure_size == (10, 8)
        assert config.marker_alpha == 1
        assert config.invert_yaxis is True
    
    def test_figure_size_tuple(self):
        """Test ProfileConfig with various figure sizes."""
        config1 = ProfileConfig(figure_size=(8, 6))
        config2 = ProfileConfig(figure_size=(15, 12))
        
        assert config1.figure_size == (8, 6)
        assert config2.figure_size == (15, 12)
    
    def test_alpha_values(self):
        """Test ProfileConfig with various alpha values."""
        config1 = ProfileConfig(marker_alpha=0.0)
        config2 = ProfileConfig(marker_alpha=0.5)
        config3 = ProfileConfig(marker_alpha=1.0)
        
        assert config1.marker_alpha == 0.0
        assert config2.marker_alpha == 0.5
        assert config3.marker_alpha == 1.0


class TestMapConfig:
    """Test MapConfig initialization and defaults."""
    
    def test_default_initialization(self):
        """Test MapConfig with default values."""
        config = MapConfig()
        
        assert config.colormap == 'cividis'
        assert config.map_extent is None
        assert config.vrange is None
        assert config.padding == 5.0
        assert config.figure_size == (8, 6)
        assert config.scatter_size == 100
        assert config.scatter_alpha == 0.7
        assert config.default_window_hours == 24
        assert 'time' in config.disallowed_plotvars
        assert 'longitude' in config.disallowed_plotvars
        assert 'latitude' in config.disallowed_plotvars
    
    def test_custom_initialization(self):
        """Test MapConfig with custom values."""
        config = MapConfig(
            colormap='viridis',
            map_extent=(-180, 180, -90, 90),
            vertical_range=(0, 1000),
            padding=10.0,
            figure_size=(12, 8),
            scatter_size=50,
            scatter_alpha=0.5,
            default_window_hours=48,
            disallowed_plotvars=['time', 'id']
        )
        
        assert config.colormap == 'viridis'
        assert config.map_extent == (-180, 180, -90, 90)
        assert config.vrange is not None
        assert config.vrange['min'] == 0
        assert config.vrange['max'] == 1000
        assert config.padding == 10.0
        assert config.figure_size == (12, 8)
        assert config.scatter_size == 50
        assert config.scatter_alpha == 0.5
        assert config.default_window_hours == 48
        assert config.disallowed_plotvars == ['time', 'id']
    
    def test_vertical_range_conversion(self):
        """Test vertical_range conversion to vrange dict."""
        config1 = MapConfig(vertical_range=(0, 100))
        config2 = MapConfig(vertical_range=(-50, 50))
        config3 = MapConfig()
        
        assert config1.vrange == {'min': 0, 'max': 100}
        assert config2.vrange == {'min': -50, 'max': 50}
        assert config3.vrange is None
    
    def test_map_extent_tuple(self):
        """Test MapConfig with various map extents."""
        config1 = MapConfig(map_extent=(-170, -120, 40, 60))
        config2 = MapConfig(map_extent=(0, 360, -80, 80))
        
        assert config1.map_extent == (-170, -120, 40, 60)
        assert config2.map_extent == (0, 360, -80, 80)
    
    def test_default_disallowed_plotvars(self):
        """Test default disallowed plot variables."""
        config = MapConfig()
        
        expected_defaults = ["time", "type", "longitude", "latitude", "vertical"]
        assert config.disallowed_plotvars == expected_defaults
    
    def test_custom_disallowed_plotvars(self):
        """Test custom disallowed plot variables."""
        custom_disallowed = ["var1", "var2", "var3"]
        config = MapConfig(disallowed_plotvars=custom_disallowed)
        
        assert config.disallowed_plotvars == custom_disallowed
    
    def test_padding_values(self):
        """Test MapConfig with various padding values."""
        config1 = MapConfig(padding=0.0)
        config2 = MapConfig(padding=2.5)
        config3 = MapConfig(padding=20.0)
        
        assert config1.padding == 0.0
        assert config2.padding == 2.5
        assert config3.padding == 20.0
    
    def test_scatter_parameters(self):
        """Test scatter plot parameters."""
        config = MapConfig(scatter_size=200, scatter_alpha=0.3)
        
        assert config.scatter_size == 200
        assert config.scatter_alpha == 0.3
    
    def test_default_window_hours(self):
        """Test default_window_hours parameter."""
        config1 = MapConfig()
        config2 = MapConfig(default_window_hours=72)
        config3 = MapConfig(default_window_hours=None)
        
        assert config1.default_window_hours == 24
        assert config2.default_window_hours == 72
        assert config3.default_window_hours == 24


class TestConfigEdgeCases:
    """Test edge cases for configuration classes."""
    
    def test_profile_config_zero_marker_size(self):
        """Test ProfileConfig with zero marker size."""
        config = ProfileConfig(marker_size=0)
        assert config.marker_size == 0
    
    def test_profile_config_large_figure(self):
        """Test ProfileConfig with very large figure size."""
        config = ProfileConfig(figure_size=(100, 100))
        assert config.figure_size == (100, 100)
    
    def test_map_config_empty_disallowed(self):
        """Test MapConfig with empty disallowed list."""
        config = MapConfig(disallowed_plotvars=[])
        assert config.disallowed_plotvars == []
    
    def test_map_config_negative_padding(self):
        """Test MapConfig with negative padding."""
        config = MapConfig(padding=-5.0)
        assert config.padding == -5.0
    
    def test_map_config_vertical_range_same_values(self):
        """Test MapConfig with same min/max vertical range."""
        config = MapConfig(vertical_range=(100, 100))
        assert config.vrange['min'] == 100
        assert config.vrange['max'] == 100
    
    def test_map_config_inverted_vertical_range(self):
        """Test MapConfig with inverted vertical range."""
        config = MapConfig(vertical_range=(1000, 0))
        assert config.vrange['min'] == 1000
        assert config.vrange['max'] == 0


class TestConfigTypeHandling:
    """Test type handling in configuration classes."""
    
    def test_profile_config_float_marker_size(self):
        """Test ProfileConfig accepts float for marker_size."""
        config = ProfileConfig(marker_size=7.5)
        assert config.marker_size == 7.5
    
    def test_profile_config_string_initial_axes(self):
        """Test ProfileConfig with string axis names."""
        config = ProfileConfig(initial_x='temperature', initial_y='pressure')
        assert config.initial_x == 'temperature'
        assert config.initial_y == 'pressure'
    
    def test_map_config_tuple_map_extent(self):
        """Test MapConfig map_extent is stored as tuple."""
        extent = (-180, 180, -90, 90)
        config = MapConfig(map_extent=extent)
        assert config.map_extent == extent
        assert isinstance(config.map_extent, tuple)
    
    def test_map_config_list_disallowed_plotvars(self):
        """Test MapConfig disallowed_plotvars as list."""
        disallowed = ['var1', 'var2']
        config = MapConfig(disallowed_plotvars=disallowed)
        assert config.disallowed_plotvars == disallowed
        assert isinstance(config.disallowed_plotvars, list)
