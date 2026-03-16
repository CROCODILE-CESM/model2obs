"""Tests for InteractiveWidgetProfile initialization and state management."""

import pytest
import pandas as pd
import dask.dataframe as dd
from unittest.mock import Mock, patch, MagicMock

from model2obs.viz.interactive_widget_profile import InteractiveWidgetProfile
from model2obs.viz.viz_config import ProfileConfig


class TestProfileWidgetInitialization:
    """Test InteractiveWidgetProfile initialization."""
    
    def test_init_with_pandas_dataframe(self):
        """Test initialization with pandas DataFrame."""
        df = pd.DataFrame({
            'obs': [20.0, 21.0, 22.0],
            'vertical': [-10, -20, -30],
            'type': ['FLOAT_TEMPERATURE'] * 3
        })
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df)
            
            assert isinstance(widget.df, pd.DataFrame)
            assert widget.config is not None
            assert isinstance(widget.config, ProfileConfig)
    
    def test_init_with_dask_dataframe(self):
        """Test initialization with Dask DataFrame."""
        df = pd.DataFrame({
            'obs': [20.0, 21.0],
            'vertical': [-10, -20]
        })
        ddf = dd.from_pandas(df, npartitions=1)
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(ddf)
            
            assert isinstance(widget.df, dd.DataFrame)
    
    def test_init_with_custom_config(self):
        """Test initialization with custom ProfileConfig."""
        df = pd.DataFrame({'obs': [1, 2], 'vertical': [-1, -2]})
        config = ProfileConfig(colormap='plasma', marker_size=10)
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df, config=config)
            
            assert widget.config is config
            assert widget.config.colormap == 'plasma'
            assert widget.config.marker_size == 10
    
    def test_init_with_explicit_axes(self):
        """Test initialization with explicit x and y axes."""
        df = pd.DataFrame({
            'temperature': [20.0, 21.0],
            'depth': [-10, -20],
            'obs': [19.5, 20.5]
        })
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df, x='temperature', y='depth')
            
            assert widget.x_column == 'temperature'
            assert widget.y_column == 'depth'
    
    def test_init_with_config_initial_axes(self):
        """Test initialization with axes from config."""
        df = pd.DataFrame({
            'temp': [20.0, 21.0],
            'press': [100, 200]
        })
        config = ProfileConfig(initial_x='temp', initial_y='press')
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df, config=config)
            
            assert widget.x_column == 'temp'
            assert widget.y_column == 'press'
    
    def test_init_parameter_overrides_config(self):
        """Test that init parameters override config values."""
        df = pd.DataFrame({
            'var1': [1, 2],
            'var2': [3, 4],
            'var3': [5, 6]
        })
        config = ProfileConfig(initial_x='var2', initial_y='var3')
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df, x='var1', config=config)
            
            # Parameter should override config
            assert widget.x_column == 'var1'
            # Config value used for y since not specified in params
            assert widget.y_column == 'var3'


class TestProfileWidgetStateInitialization:
    """Test _initialize_state method."""
    
    def test_initialize_state_sets_defaults(self):
        """Test that _initialize_state sets default values."""
        df = pd.DataFrame({'obs': [1, 2], 'vertical': [-1, -2]})
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df)
            widget._initialize_state()
            
            assert widget.filtered_df is None
            assert widget.plot_title == ""
    
    def test_set_default_axes_with_obs_column(self):
        """Test default axes when 'obs' column exists."""
        df = pd.DataFrame({
            'obs': [20.0, 21.0],
            'vertical': [-10, -20],
            'other': [1, 2]
        })
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df)
            widget._set_default_axes()
            
            assert widget.x_column == 'obs'
    
    def test_set_default_axes_with_vertical_column(self):
        """Test default axes when 'vertical' column exists."""
        df = pd.DataFrame({
            'temperature': [20.0, 21.0],
            'vertical': [-10, -20],
            'salinity': [35, 35.5]
        })
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df)
            widget._set_default_axes()
            
            assert widget.y_column == 'vertical'
    
    def test_set_default_axes_without_obs_vertical(self):
        """Test default axes when neither 'obs' nor 'vertical' exist."""
        df = pd.DataFrame({
            'time': pd.date_range('2020-01-01', periods=3),
            'var1': [1, 2, 3],
            'var2': [4, 5, 6]
        })
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df)
            widget._set_default_axes()
            
            # Should pick first non-time column for x
            assert widget.x_column == 'var1'
            # Should pick second non-time column for y
            assert widget.y_column == 'var2'
    
    def test_set_default_axes_with_only_time(self):
        """Test default axes with only time column."""
        df = pd.DataFrame({
            'time': pd.date_range('2020-01-01', periods=3)
        })
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df)
            widget._set_default_axes()
            
            # Should default to time if that's all there is
            assert widget.x_column == 'time'
            assert widget.y_column is None


class TestProfileWidgetConfiguration:
    """Test configuration handling in ProfileWidget."""
    
    def test_default_config_applied(self):
        """Test that default config is created when none provided."""
        df = pd.DataFrame({'obs': [1], 'vertical': [-1]})
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df)
            
            assert isinstance(widget.config, ProfileConfig)
            assert widget.config.colormap == 'viridis'
    
    def test_custom_colormap(self):
        """Test widget with custom colormap."""
        df = pd.DataFrame({'obs': [1], 'vertical': [-1]})
        config = ProfileConfig(colormap='coolwarm')
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df, config=config)
            
            assert widget.config.colormap == 'coolwarm'
    
    def test_custom_figure_size(self):
        """Test widget with custom figure size."""
        df = pd.DataFrame({'obs': [1], 'vertical': [-1]})
        config = ProfileConfig(figure_size=(15, 12))
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df, config=config)
            
            assert widget.config.figure_size == (15, 12)
    
    def test_invert_yaxis_setting(self):
        """Test invert_yaxis configuration."""
        df = pd.DataFrame({'obs': [1], 'vertical': [-1]})
        config = ProfileConfig(invert_yaxis=False)
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df, config=config)
            
            assert widget.config.invert_yaxis is False


class TestProfileWidgetDataFrameHandling:
    """Test DataFrame handling in ProfileWidget."""
    
    def test_empty_dataframe(self):
        """Test initialization with empty DataFrame."""
        df = pd.DataFrame()
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df)
            
            assert len(widget.df) == 0
    
    def test_single_row_dataframe(self):
        """Test initialization with single row DataFrame."""
        df = pd.DataFrame({'obs': [20.0], 'vertical': [-10]})
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df)
            
            assert len(widget.df) == 1
    
    def test_large_dataframe(self):
        """Test initialization with large DataFrame."""
        df = pd.DataFrame({
            'obs': range(10000),
            'vertical': range(-10000, 0)
        })
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df)
            
            assert len(widget.df) == 10000
    
    def test_dataframe_with_missing_values(self):
        """Test initialization with DataFrame containing NaN."""
        df = pd.DataFrame({
            'obs': [20.0, None, 22.0],
            'vertical': [-10, -20, None]
        })
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df)
            
            assert widget.df['obs'].isna().sum() == 1
            assert widget.df['vertical'].isna().sum() == 1


class TestProfileWidgetEdgeCases:
    """Test edge cases for ProfileWidget."""
    
    def test_none_x_and_y(self):
        """Test initialization with no x and y specified."""
        df = pd.DataFrame({
            'var1': [1, 2, 3],
            'var2': [4, 5, 6]
        })
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df, x=None, y=None)
            widget._set_default_axes()
            
            # Should fall back to defaults after calling _set_default_axes
            assert widget.x_column == 'var1'
            assert widget.y_column == 'var2'
    
    def test_nonexistent_column_names(self):
        """Test that widget stores nonexistent column names."""
        df = pd.DataFrame({'col1': [1], 'col2': [2]})
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df, x='nonexistent', y='alsononexistent')
            
            # Widget should store the names even if they don't exist
            assert widget.x_column == 'nonexistent'
            assert widget.y_column == 'alsononexistent'
    
    def test_same_x_and_y_column(self):
        """Test initialization with same column for x and y."""
        df = pd.DataFrame({'value': [1, 2, 3]})
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df, x='value', y='value')
            
            assert widget.x_column == 'value'
            assert widget.y_column == 'value'


class TestProfileWidgetCreation:
    """Test widget creation methods."""
    
    def test_create_widgets_with_type_column(self):
        """Test widget creation when type column exists."""
        df = pd.DataFrame({
            'obs': [20.0, 21.0],
            'vertical': [-10, -20],
            'type': ['FLOAT_TEMPERATURE', 'FLOAT_SALINITY']
        })
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df)
            widget._initialize_state()
            widget._set_default_axes()
            widget._create_widgets()
            
            # Check that widgets were created
            assert widget.output is not None
            assert hasattr(widget, 'x_dropdown')
            assert hasattr(widget, 'y_dropdown')
            assert hasattr(widget, 'type_selector')
    
    def test_create_widgets_without_type_column(self):
        """Test widget creation when type column doesn't exist."""
        df = pd.DataFrame({
            'obs': [20.0, 21.0],
            'vertical': [-10, -20]
        })
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df)
            widget._initialize_state()
            widget._set_default_axes()
            widget._create_widgets()
            
            # Should create dummy type selector
            assert hasattr(widget, 'type_selector')
    
    def test_x_dropdown_excludes_time(self):
        """Test x-axis dropdown excludes time column."""
        df = pd.DataFrame({
            'time': pd.date_range('2020-01-01', periods=3),
            'obs': [20.0, 21.0, 22.0],
            'vertical': [-10, -20, -30]
        })
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df)
            widget._initialize_state()
            widget._set_default_axes()
            widget._create_widgets()
            
            # Check that time is not in x dropdown options
            options = widget.x_dropdown.options
            assert 'time' not in options
            assert 'obs' in options
            assert 'vertical' in options
    
    def test_type_selector_default_value(self):
        """Test type selector has correct default value."""
        df = pd.DataFrame({
            'obs': [20.0, 21.0],
            'vertical': [-10, -20],
            'type': ['FLOAT_TEMPERATURE', 'FLOAT_SALINITY']
        })
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df)
            widget._initialize_state()
            widget._set_default_axes()
            widget._create_widgets()
            
            # Should default to FLOAT_TEMPERATURE if available
            assert 'FLOAT_TEMPERATURE' in widget.type_selector.value


class TestProfileWidgetCallbacks:
    """Test widget callback functionality."""
    
    def test_setup_callbacks(self):
        """Test that callbacks are set up correctly."""
        df = pd.DataFrame({
            'obs': [20.0],
            'vertical': [-10],
            'type': ['FLOAT_TEMPERATURE']
        })
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df)
            widget._initialize_state()
            widget._set_default_axes()
            widget._create_widgets()
            
            # Mock the observe method to track calls
            widget.x_dropdown.observe = Mock()
            widget.y_dropdown.observe = Mock()
            widget.type_selector.observe = Mock()
            
            widget._setup_callbacks()
            
            # Verify observe was called for each widget
            assert widget.x_dropdown.observe.called
            assert widget.y_dropdown.observe.called
            assert widget.type_selector.observe.called


class TestProfileWidgetUpdateFilteredDf:
    """Test _update_filtered_df method."""
    
    def test_update_filtered_df_with_types(self):
        """Test filtering dataframe by types."""
        df = pd.DataFrame({
            'obs': [20.0, 35.0, 21.0],
            'vertical': [-10, -20, -30],
            'type': ['FLOAT_TEMPERATURE', 'FLOAT_SALINITY', 'FLOAT_TEMPERATURE']
        })
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df)
            widget._initialize_state()
            widget._set_default_axes()
            widget._create_widgets()
            
            # Filter by FLOAT_TEMPERATURE only
            widget._update_filtered_df(['FLOAT_TEMPERATURE'])
            
            # Should have filtered dataframe
            assert widget.filtered_df is not None
            assert len(widget._compute_if_needed(widget.filtered_df)) == 2
    
    def test_update_filtered_df_without_type_column(self):
        """Test filtering when no type column exists."""
        df = pd.DataFrame({
            'obs': [20.0, 21.0],
            'vertical': [-10, -20]
        })
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df)
            widget._initialize_state()
            widget._set_default_axes()
            widget._create_widgets()
            
            # Should use full dataframe
            widget._update_filtered_df([])
            
            assert len(widget._compute_if_needed(widget.filtered_df)) == 2
    
    def test_update_filtered_df_empty_selection(self):
        """Test filtering with empty type selection."""
        df = pd.DataFrame({
            'obs': [20.0, 21.0],
            'vertical': [-10, -20],
            'type': ['FLOAT_TEMPERATURE', 'FLOAT_SALINITY']
        })
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df)
            widget._initialize_state()
            widget._set_default_axes()
            widget._create_widgets()
            
            # Empty selection should use full dataframe
            widget._update_filtered_df([])
            
            assert len(widget._compute_if_needed(widget.filtered_df)) == 2
    
    def test_update_filtered_df_multiple_types(self):
        """Test filtering with multiple types selected."""
        df = pd.DataFrame({
            'obs': [20.0, 35.0, 21.0, 5.0],
            'vertical': [-10, -20, -30, -40],
            'type': ['FLOAT_TEMPERATURE', 'FLOAT_SALINITY', 'FLOAT_TEMPERATURE', 'DRIFTER_U_CURRENT']
        })
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df)
            widget._initialize_state()
            widget._set_default_axes()
            widget._create_widgets()
            
            # Filter by two types
            widget._update_filtered_df(['FLOAT_TEMPERATURE', 'FLOAT_SALINITY'])
            
            assert len(widget._compute_if_needed(widget.filtered_df)) == 3
    
    def test_update_plot_title_single_type(self):
        """Test plot title update with single type."""
        df = pd.DataFrame({
            'obs': [20.0],
            'vertical': [-10],
            'type': ['FLOAT_TEMPERATURE']
        })
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df)
            widget._initialize_state()
            widget._set_default_axes()
            widget._create_widgets()
            
            widget._update_filtered_df(['FLOAT_TEMPERATURE'])
            
            assert 'FLOAT_TEMPERATURE' in widget.plot_title
    
    def test_update_plot_title_multiple_types(self):
        """Test plot title update with multiple types."""
        df = pd.DataFrame({
            'obs': [20.0, 35.0],
            'vertical': [-10, -20],
            'type': ['FLOAT_TEMPERATURE', 'FLOAT_SALINITY']
        })
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df)
            widget._initialize_state()
            widget._set_default_axes()
            widget._create_widgets()
            
            widget._update_filtered_df(['FLOAT_TEMPERATURE', 'FLOAT_SALINITY'])
            
            assert 'FLOAT_TEMPERATURE' in widget.plot_title
            assert 'FLOAT_SALINITY' in widget.plot_title
    
    def test_update_plot_title_many_types(self):
        """Test plot title when more than 3 types selected."""
        df = pd.DataFrame({
            'obs': [1, 2, 3, 4, 5],
            'vertical': [-1, -2, -3, -4, -5],
            'type': ['TYPE1', 'TYPE2', 'TYPE3', 'TYPE4', 'TYPE5']
        })
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df)
            widget._initialize_state()
            widget._set_default_axes()
            widget._create_widgets()
            
            widget._update_filtered_df(['TYPE1', 'TYPE2', 'TYPE3', 'TYPE4'])
            
            # Should show count or abbreviated list
            assert 'TYPE1' in widget.plot_title


class TestProfileWidgetPlotting:
    """Test plotting methods."""
    
    @patch('matplotlib.pyplot.close')
    def test_plot_method_exists(self, mock_close):
        """Test that _plot method exists and can be called."""
        df = pd.DataFrame({
            'obs': [20.0, 21.0],
            'vertical': [-10, -20],
            'type': ['FLOAT_TEMPERATURE', 'FLOAT_TEMPERATURE']
        })
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df)
            
            # Should have _plot method
            assert hasattr(widget, '_plot')
            assert callable(widget._plot)


class TestProfileWidgetInitializeWidget:
    """Test _initialize_widget method."""
    
    @patch('matplotlib.pyplot.close')
    def test_initialize_widget_updates_filtered_df(self, mock_close):
        """Test that initialize_widget updates filtered dataframe."""
        df = pd.DataFrame({
            'obs': [20.0, 35.0],
            'vertical': [-10, -20],
            'type': ['FLOAT_TEMPERATURE', 'FLOAT_SALINITY']
        })
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df)
            widget._initialize_state()
            widget._set_default_axes()
            widget._create_widgets()
            widget._initialize_widget()
            
            # Should have initialized filtered_df
            assert widget.filtered_df is not None


class TestProfileWidgetAxisHandling:
    """Test axis selection and updates."""
    
    def test_on_axis_change_callback_exists(self):
        """Test that axis change callback exists."""
        df = pd.DataFrame({
            'obs': [20.0],
            'vertical': [-10]
        })
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df)
            
            # Should have _on_axis_change method
            assert hasattr(widget, '_on_axis_change')
    
    def test_on_type_change_callback_exists(self):
        """Test that type change callback exists."""
        df = pd.DataFrame({
            'obs': [20.0],
            'vertical': [-10],
            'type': ['FLOAT_TEMPERATURE']
        })
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df)
            
            # Should have _on_type_change method
            assert hasattr(widget, '_on_type_change')


class TestProfileWidgetEventSimulation:
    """Test widget event callbacks with simulation."""
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_axis_change_triggers_replot(self, mock_figure, mock_show):
        """Test that changing axis triggers plot update."""
        df = pd.DataFrame({
            'obs': [20.0, 21.0, 22.0],
            'vertical': [-10, -20, -30],
            'salinity': [35.0, 35.1, 35.2],
            'type': ['FLOAT_TEMPERATURE'] * 3
        })
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df)
            widget._initialize_state()
            widget._set_default_axes()
            widget._create_widgets()
            widget._update_filtered_df(['FLOAT_TEMPERATURE'])
            
            # Set initial state
            widget.x_column = 'obs'
            widget.y_column = 'vertical'
            
            # Change axis via dropdown
            widget.x_dropdown.value = 'salinity'
            widget.x_column = 'salinity'
            
            # Trigger callback
            widget._on_axis_change({'new': 'salinity'})
            
            # Verify plot was called
            assert mock_figure.called
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_type_change_updates_filtered_data(self, mock_figure, mock_show):
        """Test that changing type selection updates filtered DataFrame."""
        df = pd.DataFrame({
            'obs': [20.0, 35.0, 21.0],
            'vertical': [-10, -20, -30],
            'type': ['FLOAT_TEMPERATURE', 'FLOAT_SALINITY', 'FLOAT_TEMPERATURE']
        })
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df)
            widget._initialize_state()
            widget._set_default_axes()
            widget._create_widgets()
            
            # Initial selection
            widget._update_filtered_df(['FLOAT_TEMPERATURE'])
            initial_len = len(widget._compute_if_needed(widget.filtered_df))
            assert initial_len == 2
            
            # Change type selection
            widget._update_filtered_df(['FLOAT_SALINITY'])
            new_len = len(widget._compute_if_needed(widget.filtered_df))
            assert new_len == 1
    
    def test_multiple_type_selection(self):
        """Test widget with multiple types selected."""
        df = pd.DataFrame({
            'obs': [20.0, 35.0, 21.0, 5.0],
            'vertical': [-10, -20, -30, -40],
            'type': ['FLOAT_TEMPERATURE', 'FLOAT_SALINITY', 
                     'FLOAT_TEMPERATURE', 'DRIFTER_U_CURRENT']
        })
        
        with patch.object(InteractiveWidgetProfile, '_setup_widget_workflow'):
            widget = InteractiveWidgetProfile(df)
            widget._initialize_state()
            widget._set_default_axes()
            widget._create_widgets()
            
            # Select multiple types
            selected = ['FLOAT_TEMPERATURE', 'FLOAT_SALINITY']
            widget._update_filtered_df(selected)
            
            # Should have filtered to 3 rows
            assert len(widget._compute_if_needed(widget.filtered_df)) == 3
            
            # Plot title should reflect multiple types
            assert 'FLOAT_TEMPERATURE' in widget.plot_title
            assert 'FLOAT_SALINITY' in widget.plot_title
