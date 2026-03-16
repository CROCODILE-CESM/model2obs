"""Tests for InteractiveWidgetMap initialization and state management."""

import pytest
import pandas as pd
import dask.dataframe as dd
from datetime import timedelta
from unittest.mock import Mock, patch

from crococamp.viz.interactive_widget_map import InteractiveWidgetMap
from crococamp.viz.viz_config import MapConfig


class TestMapWidgetInitialization:
    """Test InteractiveWidgetMap initialization."""
    
    def test_init_with_pandas_dataframe(self):
        """Test initialization with pandas DataFrame."""
        df = pd.DataFrame({
            'longitude': [-170.0, -160.0, -150.0],
            'latitude': [40.0, 45.0, 50.0],
            'vertical': [-10, -20, -30],
            'obs': [20.0, 21.0, 22.0]
        })
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df)
            
            assert isinstance(widget.df, pd.DataFrame)
            assert widget.config is not None
            assert isinstance(widget.config, MapConfig)
    
    def test_init_with_dask_dataframe(self):
        """Test initialization with Dask DataFrame."""
        df = pd.DataFrame({
            'longitude': [-170.0, -160.0],
            'latitude': [40.0, 45.0],
            'obs': [20.0, 21.0]
        })
        ddf = dd.from_pandas(df, npartitions=1)
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(ddf)
            
            assert isinstance(widget.df, dd.DataFrame)
    
    def test_init_with_custom_config(self):
        """Test initialization with custom MapConfig."""
        df = pd.DataFrame({
            'longitude': [-170.0],
            'latitude': [40.0]
        })
        config = MapConfig(colormap='viridis', scatter_size=200)
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df, config=config)
            
            assert widget.config is config
            assert widget.config.colormap == 'viridis'
            assert widget.config.scatter_size == 200


class TestMapWidgetStateInitialization:
    """Test _initialize_state method."""
    
    def test_initialize_state_sets_defaults(self):
        """Test that _initialize_state sets default values."""
        df = pd.DataFrame({
            'longitude': [-170.0, -160.0],
            'latitude': [40.0, 45.0],
            'vertical': [-10, -20]
        })
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df)
            widget._initialize_state()
            
            assert widget.filtered_df is None
            assert widget.min_time is None
            assert widget.max_time is None
            assert widget.total_hours is None
            assert widget.plot_var is None
            assert widget.plot_title is None


class TestMapWidgetExtentCalculation:
    """Test map extent calculation."""
    
    def test_calculate_extent_with_config_extent(self):
        """Test that config extent is used when provided."""
        df = pd.DataFrame({
            'longitude': [-170.0, -160.0],
            'latitude': [40.0, 45.0]
        })
        config_extent = (-180, -140, 30, 60)
        config = MapConfig(map_extent=config_extent)
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df, config=config)
            widget._calculate_map_extent()
            
            assert widget.map_extent == config_extent
    
    def test_calculate_extent_auto_with_padding(self):
        """Test auto-calculation of extent with padding."""
        df = pd.DataFrame({
            'longitude': [-170.0, -160.0],
            'latitude': [40.0, 45.0]
        })
        config = MapConfig(padding=5.0)
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df, config=config)
            widget._calculate_map_extent()
            
            # Check that extent was calculated
            assert widget.map_extent is not None
            assert len(widget.map_extent) == 4
            
            # Check that padding was applied
            lon_min, lon_max, lat_min, lat_max = widget.map_extent
            assert lon_min < -170.0
            assert lon_max > -160.0
            assert lat_min < 40.0
            assert lat_max > 45.0
    
    def test_calculate_extent_clamps_to_limits(self):
        """Test that extent is clamped to valid ranges."""
        df = pd.DataFrame({
            'longitude': [-179.0, 179.0],
            'latitude': [-89.0, 89.0]
        })
        config = MapConfig(padding=20.0)
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df, config=config)
            widget._calculate_map_extent()
            
            lon_min, lon_max, lat_min, lat_max = widget.map_extent
            
            # Should be clamped to -180/180 and -90/90
            assert lon_min >= -180
            assert lon_max <= 180
            assert lat_min >= -90
            assert lat_max <= 90


class TestMapWidgetVerticalLimits:
    """Test vertical coordinate limits calculation."""
    
    def test_calculate_vertical_limits_with_config(self):
        """Test vertical limits from config."""
        df = pd.DataFrame({
            'vertical': [-10, -20, -30]
        })
        config = MapConfig(vertical_range=(0, 100))
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df, config=config)
            
            assert widget.config.vrange is not None
            assert widget.config.vrange['min'] == 0
            assert widget.config.vrange['max'] == 100


class TestMapWidgetConfiguration:
    """Test configuration handling in MapWidget."""
    
    def test_default_config_applied(self):
        """Test that default config is created when none provided."""
        df = pd.DataFrame({
            'longitude': [-170.0],
            'latitude': [40.0]
        })
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df)
            
            assert isinstance(widget.config, MapConfig)
            assert widget.config.colormap == 'cividis'
            assert widget.config.default_window_hours == 24
    
    def test_custom_scatter_parameters(self):
        """Test widget with custom scatter parameters."""
        df = pd.DataFrame({
            'longitude': [-170.0],
            'latitude': [40.0]
        })
        config = MapConfig(scatter_size=150, scatter_alpha=0.5)
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df, config=config)
            
            assert widget.config.scatter_size == 150
            assert widget.config.scatter_alpha == 0.5
    
    def test_custom_padding(self):
        """Test widget with custom padding."""
        df = pd.DataFrame({
            'longitude': [-170.0],
            'latitude': [40.0]
        })
        config = MapConfig(padding=10.0)
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df, config=config)
            
            assert widget.config.padding == 10.0
    
    def test_disallowed_plotvars(self):
        """Test disallowed plot variables configuration."""
        df = pd.DataFrame({
            'longitude': [-170.0],
            'latitude': [40.0],
            'obs': [20.0]
        })
        config = MapConfig()
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df, config=config)
            
            assert 'time' in widget.config.disallowed_plotvars
            assert 'longitude' in widget.config.disallowed_plotvars
            assert 'latitude' in widget.config.disallowed_plotvars


class TestMapWidgetDataFrameHandling:
    """Test DataFrame handling in MapWidget."""
    
    def test_empty_dataframe(self):
        """Test initialization with empty DataFrame."""
        df = pd.DataFrame()
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df)
            
            assert len(widget.df) == 0
    
    def test_single_row_dataframe(self):
        """Test initialization with single row DataFrame."""
        df = pd.DataFrame({
            'longitude': [-170.0],
            'latitude': [40.0],
            'obs': [20.0]
        })
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df)
            
            assert len(widget.df) == 1
    
    def test_large_dataframe(self):
        """Test initialization with large DataFrame."""
        df = pd.DataFrame({
            'longitude': [-170.0] * 10000,
            'latitude': [40.0] * 10000,
            'obs': range(10000)
        })
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df)
            
            assert len(widget.df) == 10000
    
    def test_dataframe_with_missing_values(self):
        """Test initialization with DataFrame containing NaN."""
        df = pd.DataFrame({
            'longitude': [-170.0, None, -150.0],
            'latitude': [40.0, 45.0, None],
            'obs': [20.0, None, 22.0]
        })
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df)
            
            assert widget.df['longitude'].isna().sum() == 1
            assert widget.df['latitude'].isna().sum() == 1
            assert widget.df['obs'].isna().sum() == 1


class TestMapWidgetLongitudeHandling:
    """Test longitude coordinate handling."""
    
    def test_longitude_range_0_360(self):
        """Test handling of 0-360 longitude range."""
        df = pd.DataFrame({
            'longitude': [0.0, 180.0, 359.0],
            'latitude': [0.0, 0.0, 0.0]
        })
        config = MapConfig(padding=5.0)
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df, config=config)
            widget._calculate_map_extent()
            
            # Extent should be calculated
            assert widget.map_extent is not None
    
    def test_longitude_range_minus180_180(self):
        """Test handling of -180 to 180 longitude range."""
        df = pd.DataFrame({
            'longitude': [-180.0, 0.0, 180.0],
            'latitude': [0.0, 0.0, 0.0]
        })
        config = MapConfig(padding=5.0)
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df, config=config)
            widget._calculate_map_extent()
            
            assert widget.map_extent is not None


class TestMapWidgetEdgeCases:
    """Test edge cases for MapWidget."""
    
    def test_zero_padding(self):
        """Test widget with zero padding."""
        df = pd.DataFrame({
            'longitude': [-170.0, -160.0],
            'latitude': [40.0, 45.0]
        })
        config = MapConfig(padding=0.0)
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df, config=config)
            widget._calculate_map_extent()
            
            # Should still calculate extent, just without padding
            assert widget.map_extent is not None
    
    def test_custom_figure_size(self):
        """Test widget with custom figure size."""
        df = pd.DataFrame({
            'longitude': [-170.0],
            'latitude': [40.0]
        })
        config = MapConfig(figure_size=(15, 10))
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df, config=config)
            
            assert widget.config.figure_size == (15, 10)
    
    def test_empty_disallowed_plotvars(self):
        """Test widget with empty disallowed plotvars list."""
        df = pd.DataFrame({
            'longitude': [-170.0],
            'latitude': [40.0]
        })
        config = MapConfig(disallowed_plotvars=[])
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df, config=config)
            
            assert widget.config.disallowed_plotvars == []


class TestMapWidgetCreation:
    """Test widget creation methods."""
    
    def test_create_widgets_with_type_column(self):
        """Test widget creation when type column exists."""
        df = pd.DataFrame({
            'longitude': [-170.0, -160.0],
            'latitude': [40.0, 45.0],
            'vertical': [-10, -20],
            'time': pd.date_range('2020-01-01', periods=2, freq='h'),
            'type': ['FLOAT_TEMPERATURE', 'FLOAT_SALINITY'],
            'difference': [0.5, -0.3]
        })
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df)
            widget._initialize_state()
            widget._calculate_map_extent()
            widget._calculate_vertical_limits()
            widget._create_widgets()
            
            # Check that widgets were created
            assert widget.output is not None
            assert hasattr(widget, 'type_dropdown')
            assert hasattr(widget, 'refvar_dropdown')
            assert hasattr(widget, 'window_slider')
            assert hasattr(widget, 'center_slider')
            assert hasattr(widget, 'colorbar_slider')
            assert hasattr(widget, 'vrange_slider')
    
    def test_type_dropdown_default_value(self):
        """Test type dropdown has correct default value."""
        df = pd.DataFrame({
            'longitude': [-170.0, -160.0],
            'latitude': [40.0, 45.0],
            'vertical': [-10, -20],
            'time': pd.date_range('2020-01-01', periods=2, freq='h'),
            'type': ['FLOAT_TEMPERATURE', 'FLOAT_SALINITY'],
            'obs': [20.0, 35.0]
        })
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df)
            widget._initialize_state()
            widget._calculate_map_extent()
            widget._calculate_vertical_limits()
            widget._create_widgets()
            
            # Should default to FLOAT_TEMPERATURE if available
            assert widget.type_dropdown.value == 'FLOAT_TEMPERATURE'
    
    def test_refvar_dropdown_excludes_disallowed(self):
        """Test refvar dropdown excludes disallowed plot variables."""
        df = pd.DataFrame({
            'longitude': [-170.0],
            'latitude': [40.0],
            'vertical': [-10],
            'time': [pd.Timestamp('2020-01-01')],
            'type': ['FLOAT_TEMPERATURE'],
            'obs': [20.0],
            'difference': [0.5]
        })
        config = MapConfig(disallowed_plotvars=['time', 'longitude', 'latitude'])
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df, config=config)
            widget._initialize_state()
            widget._calculate_map_extent()
            widget._calculate_vertical_limits()
            widget._create_widgets()
            
            # Check that disallowed vars are not in options
            options = widget.refvar_dropdown.options
            assert 'time' not in options
            assert 'longitude' not in options
            assert 'latitude' not in options
            assert 'obs' in options
            assert 'difference' in options


class TestMapWidgetCallbacks:
    """Test widget callback functionality."""
    
    def test_setup_callbacks(self):
        """Test that callbacks are set up correctly."""
        df = pd.DataFrame({
            'longitude': [-170.0],
            'latitude': [40.0],
            'vertical': [-10],
            'time': [pd.Timestamp('2020-01-01')],
            'type': ['FLOAT_TEMPERATURE'],
            'obs': [20.0]
        })
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df)
            widget._initialize_state()
            widget._calculate_map_extent()
            widget._calculate_vertical_limits()
            widget._create_widgets()
            
            # Mock the observe method to track calls
            widget.type_dropdown.observe = Mock()
            widget.refvar_dropdown.observe = Mock()
            widget.window_slider.observe = Mock()
            widget.center_slider.observe = Mock()
            widget.colorbar_slider.observe = Mock()
            widget.min_cb.observe = Mock()
            widget.max_cb.observe = Mock()
            widget.vrange_slider.observe = Mock()
            
            widget._setup_callbacks()
            
            # Verify observe was called for each widget
            assert widget.type_dropdown.observe.called
            assert widget.refvar_dropdown.observe.called
            assert widget.window_slider.observe.called
            assert widget.center_slider.observe.called
            assert widget.colorbar_slider.observe.called
            assert widget.min_cb.observe.called
            assert widget.max_cb.observe.called
            assert widget.vrange_slider.observe.called


class TestMapWidgetInitializeWidget:
    """Test _initialize_widget method."""
    
    @patch('matplotlib.pyplot.close')
    def test_initialize_widget_with_time_column(self, mock_close):
        """Test initialize_widget when time column exists."""
        df = pd.DataFrame({
            'longitude': [-170.0, -160.0],
            'latitude': [40.0, 45.0],
            'time': pd.date_range('2020-01-01', periods=2, freq='h'),
            'type': ['FLOAT_TEMPERATURE', 'FLOAT_TEMPERATURE'],
            'vertical': [-10, -20],
            'obs': [20.0, 21.0]
        })
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df)
            widget._initialize_state()
            widget._create_widgets()
            widget._initialize_widget()
            
            # Should have calculated time range
            assert widget.min_time is not None
            assert widget.max_time is not None
            assert widget.total_hours is not None
    
    @patch('matplotlib.pyplot.close')
    def test_initialize_widget_calculates_vertical_limits(self, mock_close):
        """Test that initialize_widget calculates vertical limits."""
        df = pd.DataFrame({
            'longitude': [-170.0],
            'latitude': [40.0],
            'time': [pd.Timestamp('2020-01-01')],
            'type': ['FLOAT_TEMPERATURE'],
            'vertical': [-10],
            'obs': [20.0]
        })
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df)
            widget._initialize_state()
            widget._create_widgets()
            widget._initialize_widget()
            
            # Should have calculated vertical limits
            assert hasattr(widget, 'vrange')
            assert widget.vrange is not None


class TestMapWidgetPlotting:
    """Test plotting methods."""
    
    @patch('matplotlib.pyplot.close')
    def test_plot_method_exists(self, mock_close):
        """Test that _plot method exists and can be called."""
        df = pd.DataFrame({
            'longitude': [-170.0, -160.0],
            'latitude': [40.0, 45.0],
            'time': pd.date_range('2020-01-01', periods=2, freq='h'),
            'type': ['FLOAT_TEMPERATURE', 'FLOAT_TEMPERATURE'],
            'difference': [0.5, -0.3],
            'vertical': [-10, -20]
        })
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df)
            
            # Should have _plot method
            assert hasattr(widget, '_plot')
            assert callable(widget._plot)


class TestMapWidgetFiltering:
    """Test data filtering functionality."""
    
    def test_update_filtered_df_method_exists(self):
        """Test that filtering method exists."""
        df = pd.DataFrame({
            'longitude': [-170.0, -160.0, -150.0],
            'latitude': [40.0, 45.0, 50.0],
            'type': ['FLOAT_TEMPERATURE', 'FLOAT_SALINITY', 'FLOAT_TEMPERATURE'],
            'vertical': [-10, -20, -30],
            'time': pd.date_range('2020-01-01', periods=3, freq='h'),
            'obs': [20.0, 35.0, 21.0]
        })
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df)
            
            # Should have filtering method
            assert hasattr(widget, '_update_filtered_df')
            assert callable(widget._update_filtered_df)


class TestMapWidgetTimeHandling:
    """Test time-related functionality."""
    
    def test_time_window_calculation(self):
        """Test time window calculation."""
        df = pd.DataFrame({
            'longitude': [-170.0] * 10,
            'latitude': [40.0] * 10,
            'time': pd.date_range('2020-01-01', periods=10, freq='h'),
            'type': ['FLOAT_TEMPERATURE'] * 10,
            'vertical': [-10] * 10,
            'obs': range(10)
        })
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df)
            widget._initialize_state()
            widget._calculate_map_extent()
            widget._calculate_vertical_limits()
            widget._create_widgets()
            
            # Check window slider was created with correct max
            assert hasattr(widget, 'window_slider')
            assert widget.window_slider.max == widget.config.default_window_hours
    
    def test_parse_window_hours(self):
        """Test parsing window text to timedelta."""
        df = pd.DataFrame({
            'longitude': [-170.0],
            'latitude': [40.0],
            'vertical': [-10],
            'time': [pd.Timestamp('2020-01-01')],
            'type': ['FLOAT_TEMPERATURE']
        })
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df)
            
            # Test parsing hours
            result = widget.parse_window('24 hours')
            assert result == timedelta(hours=24)
    
    def test_parse_window_days(self):
        """Test parsing window text for days."""
        df = pd.DataFrame({
            'longitude': [-170.0],
            'latitude': [40.0],
            'vertical': [-10],
            'time': [pd.Timestamp('2020-01-01')],
            'type': ['FLOAT_TEMPERATURE']
        })
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df)
            
            # Test parsing days
            result = widget.parse_window('7 days')
            assert result == timedelta(days=7)
    
    def test_get_window_timedelta(self):
        """Test getting window timedelta from slider."""
        df = pd.DataFrame({
            'longitude': [-170.0],
            'latitude': [40.0],
            'vertical': [-10],
            'time': [pd.Timestamp('2020-01-01')],
            'type': ['FLOAT_TEMPERATURE'],
            'difference': [0.5]
        })
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df)
            widget._initialize_state()
            widget._calculate_map_extent()
            widget._calculate_vertical_limits()
            widget._create_widgets()
            
            # Should return timedelta based on slider value
            result = widget._get_window_timedelta()
            assert isinstance(result, timedelta)
            assert result.total_seconds() == widget.window_slider.value * 3600


class TestMapWidgetEventSimulation:
    """Test map widget event callbacks with simulation."""
    
    def test_type_change_updates_filtered_data(self):
        """Test that changing type updates filtered DataFrame."""
        df = pd.DataFrame({
            'longitude': [-170.0, -160.0, -150.0],
            'latitude': [40.0, 45.0, 50.0],
            'vertical': [-10, -20, -30],
            'time': pd.date_range('2020-01-01', periods=3, freq='h'),
            'type': ['FLOAT_TEMPERATURE', 'FLOAT_SALINITY', 'FLOAT_TEMPERATURE'],
            'obs': [20.0, 35.0, 21.0]
        })
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df)
            widget._initialize_state()
            widget._calculate_map_extent()
            widget._calculate_vertical_limits()
            widget._create_widgets()
            
            # Update to FLOAT_TEMPERATURE
            widget._update_filtered_df('FLOAT_TEMPERATURE')
            temp_len = len(widget._compute_if_needed(widget.filtered_df))
            assert temp_len == 2
            
            # Update to FLOAT_SALINITY
            widget._update_filtered_df('FLOAT_SALINITY')
            sal_len = len(widget._compute_if_needed(widget.filtered_df))
            assert sal_len == 1
    
    def test_refvar_change_updates_plot_variable(self):
        """Test that changing reference variable updates plot_var."""
        df = pd.DataFrame({
            'longitude': [-170.0],
            'latitude': [40.0],
            'vertical': [-10],
            'time': [pd.Timestamp('2020-01-01')],
            'type': ['FLOAT_TEMPERATURE'],
            'obs': [20.0],
            'difference': [0.5],
            'squared_difference': [0.25]
        })
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df)
            widget._initialize_state()
            
            # Update to difference
            widget._update_refvar('difference')
            assert widget.plot_var == 'difference'
            assert widget.plot_title == 'difference'
            
            # Update to squared_difference
            widget._update_refvar('squared_difference')
            assert widget.plot_var == 'squared_difference'
            assert widget.plot_title == 'squared_difference'
    
    def test_window_change_updates_center_slider(self):
        """Test that changing window updates center slider options."""
        df = pd.DataFrame({
            'longitude': [-170.0] * 24,
            'latitude': [40.0] * 24,
            'vertical': [-10] * 24,
            'time': pd.date_range('2020-01-01', periods=24, freq='h'),
            'type': ['FLOAT_TEMPERATURE'] * 24,
            'obs': range(24)
        })
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df)
            widget._initialize_state()
            widget._calculate_map_extent()
            widget._calculate_vertical_limits()
            widget._create_widgets()
            widget._update_filtered_df('FLOAT_TEMPERATURE')
            
            # Test with 6-hour window
            window_td = timedelta(hours=6)
            widget._update_center_slider(window_td)
            
            # Verify slider was updated
            assert len(widget.center_slider.options) > 0
            
            # Test with 12-hour window
            window_td = timedelta(hours=12)
            widget._update_center_slider(window_td)
            
            # Verify slider was updated with different options
            assert len(widget.center_slider.options) > 0


class TestMapWidgetIntegration:
    """Integration tests for map widget full workflow."""
    
    @patch('matplotlib.pyplot.close')
    def test_full_initialization_workflow(self, mock_close):
        """Test complete widget initialization workflow."""
        df = pd.DataFrame({
            'longitude': [-170.0, -160.0, -150.0],
            'latitude': [40.0, 45.0, 50.0],
            'vertical': [-10, -20, -30],
            'time': pd.date_range('2020-01-01', periods=3, freq='h'),
            'type': ['FLOAT_TEMPERATURE'] * 3,
            'difference': [0.5, -0.3, 0.1]
        })
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df)
            widget._initialize_state()
            widget._calculate_map_extent()
            widget._calculate_vertical_limits()
            widget._create_widgets()
            widget._setup_callbacks()
            
            # Manually set what _initialize_widget would set
            widget._update_refvar(widget.refvar_dropdown.value)
            widget._update_filtered_df(widget.type_dropdown.value)
            
            # Verify complete initialization
            assert widget.filtered_df is not None
            assert widget.min_time is not None
            assert widget.max_time is not None
            assert widget.plot_var is not None
            assert widget.map_extent is not None
            assert widget.vrange is not None
    
    def test_colorbar_slider_update_logic(self):
        """Test colorbar slider constraint handling."""
        df = pd.DataFrame({
            'longitude': [-170.0, -160.0],
            'latitude': [40.0, 45.0],
            'vertical': [-10, -20],
            'time': pd.date_range('2020-01-01', periods=2, freq='h'),
            'type': ['FLOAT_TEMPERATURE'] * 2,
            'difference': [0.5, 2.5]
        })
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df)
            widget._initialize_state()
            widget._calculate_map_extent()
            widget._calculate_vertical_limits()
            widget._create_widgets()
            widget._update_filtered_df('FLOAT_TEMPERATURE')
            widget._update_refvar('difference')
            
            # Update center slider first to have valid options
            widget._update_center_slider(timedelta(hours=1))
            
            # Set center slider to a valid value from its options
            if len(widget.center_slider.options) > 0:
                widget.center_slider.value = widget.center_slider.options[0]
                
                # Update colorbar slider
                widget._update_colorbar_slider()
                
                # Verify colorbar was updated
                assert widget.colorbar_slider.value[0] <= widget.colorbar_slider.value[1]
                assert widget.colorbar_slider.min <= widget.colorbar_slider.value[0]
                assert widget.colorbar_slider.max >= widget.colorbar_slider.value[1]
    
    def test_vertical_range_filtering(self):
        """Test that vertical range filtering works."""
        df = pd.DataFrame({
            'longitude': [-170.0] * 5,
            'latitude': [40.0] * 5,
            'vertical': [0, -10, -20, -30, -40],
            'time': pd.date_range('2020-01-01', periods=5, freq='h'),
            'type': ['FLOAT_TEMPERATURE'] * 5,
            'obs': [20.0, 20.5, 21.0, 21.5, 22.0]
        })
        
        with patch.object(InteractiveWidgetMap, '_setup_widget_workflow'):
            widget = InteractiveWidgetMap(df)
            widget._initialize_state()
            widget._calculate_map_extent()
            widget._calculate_vertical_limits()
            
            # Check vertical limits were calculated
            assert widget.vertical_limits['min'] == -40
            assert widget.vertical_limits['max'] == 0
            
            # Check vrange is set
            assert widget.vrange is not None
            assert 'min' in widget.vrange
            assert 'max' in widget.vrange
