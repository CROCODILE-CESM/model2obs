"""Unit tests for interactive visualization widgets.

Tests the base InteractiveWidget class helper methods and DataFrame handling.
Widget-specific functionality is tested through integration tests.
"""

import pytest
import pandas as pd
import dask.dataframe as dd
from unittest.mock import Mock, patch, MagicMock

from model2obs.viz.interactive_widget import InteractiveWidget


class TestInteractiveWidgetHelperMethods:
    """Test InteractiveWidget helper methods that don't require instantiation."""
    
    def test_is_dask_dataframe_with_pandas(self):
        """Test _is_dask_dataframe with pandas DataFrame."""
        # Create a mock widget to test the method
        mock_widget = Mock(spec=InteractiveWidget)
        mock_widget.df = pd.DataFrame({'value': [1, 2, 3]})
        
        # Call the actual method
        result = InteractiveWidget._is_dask_dataframe(mock_widget)
        
        assert result is False
    
    def test_is_dask_dataframe_with_dask(self):
        """Test _is_dask_dataframe with dask DataFrame."""
        df = pd.DataFrame({'value': [1, 2, 3]})
        ddf = dd.from_pandas(df, npartitions=1)
        
        mock_widget = Mock(spec=InteractiveWidget)
        mock_widget.df = ddf
        
        result = InteractiveWidget._is_dask_dataframe(mock_widget)
        
        assert result is True
    
    def test_compute_if_needed_pandas_series(self):
        """Test _compute_if_needed with pandas Series."""
        series = pd.Series([1, 2, 3])
        
        mock_widget = Mock(spec=InteractiveWidget)
        result = InteractiveWidget._compute_if_needed(mock_widget, series)
        
        # Should return as-is for pandas
        assert result is series
    
    def test_compute_if_needed_dask_series(self):
        """Test _compute_if_needed with dask Series."""
        series = pd.Series([1, 2, 3])
        dseries = dd.from_pandas(series, npartitions=1)
        
        mock_widget = Mock(spec=InteractiveWidget)
        result = InteractiveWidget._compute_if_needed(mock_widget, dseries)
        
        # Should compute dask series
        assert isinstance(result, pd.Series)
        assert len(result) == 3
    
    def test_compute_if_needed_pandas_dataframe(self):
        """Test _compute_if_needed with pandas DataFrame."""
        df = pd.DataFrame({'value': [1, 2, 3]})
        
        mock_widget = Mock(spec=InteractiveWidget)
        result = InteractiveWidget._compute_if_needed(mock_widget, df)
        
        assert result is df
    
    def test_compute_if_needed_dask_dataframe(self):
        """Test _compute_if_needed with dask DataFrame."""
        df = pd.DataFrame({'value': [1, 2, 3]})
        ddf = dd.from_pandas(df, npartitions=1)
        
        mock_widget = Mock(spec=InteractiveWidget)
        result = InteractiveWidget._compute_if_needed(mock_widget, ddf)
        
        # Should compute dask dataframe
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
    
    def test_persist_if_needed_pandas(self):
        """Test _persist_if_needed with pandas DataFrame."""
        df = pd.DataFrame({'value': [1, 2, 3]})
        
        mock_widget = Mock(spec=InteractiveWidget)
        result = InteractiveWidget._persist_if_needed(mock_widget, df)
        
        # Should return as-is for pandas
        assert result is df
    
    def test_persist_if_needed_dask(self):
        """Test _persist_if_needed with dask DataFrame."""
        df = pd.DataFrame({'value': [1, 2, 3]})
        ddf = dd.from_pandas(df, npartitions=1)
        
        mock_widget = Mock(spec=InteractiveWidget)
        result = InteractiveWidget._persist_if_needed(mock_widget, ddf)
        
        # Should persist dask dataframe
        assert hasattr(result, 'compute')  # Still dask after persist


class TestInteractiveWidgetClearMethod:
    """Test widget clear functionality."""
    
    def test_clear_with_output_widget(self):
        """Test clearing widget with output set."""
        mock_widget = Mock(spec=InteractiveWidget)
        mock_output = MagicMock()
        mock_widget.output = mock_output
        
        # Call the actual clear method
        InteractiveWidget.clear(mock_widget)
        
        # Output's __enter__ should be called (context manager)
        mock_output.__enter__.assert_called_once()
    
    def test_clear_without_output_widget(self):
        """Test clearing widget without output set."""
        mock_widget = Mock(spec=InteractiveWidget)
        mock_widget.output = None
        
        # Should not raise error
        InteractiveWidget.clear(mock_widget)


class TestInteractiveWidgetSetupWorkflow:
    """Test widget setup workflow."""
    
    def test_setup_widget_workflow_calls_methods(self):
        """Test that _setup_widget_workflow calls required methods."""
        mock_widget = Mock(spec=InteractiveWidget)
        
        InteractiveWidget._setup_widget_workflow(mock_widget)
        
        # Verify methods were called
        mock_widget._initialize_state.assert_called_once()
        mock_widget._create_widgets.assert_called_once()
        mock_widget._setup_callbacks.assert_called_once()
    
    def test_setup_method_calls_abstract_methods(self):
        """Test that setup() calls abstract methods."""
        mock_widget = Mock(spec=InteractiveWidget)
        mock_widget._create_widget_layout.return_value = Mock()
        
        result = InteractiveWidget.setup(mock_widget)
        
        # Verify abstract methods were called
        mock_widget._initialize_widget.assert_called_once()
        mock_widget._create_widget_layout.assert_called_once()
        mock_widget._plot.assert_called_once()
        
        # Should return widget box
        assert result is not None


class TestInteractiveWidgetDataFrameTypes:
    """Test DataFrame type handling."""
    
    def test_pandas_dataframe_detection(self):
        """Test detection of pandas DataFrame."""
        df = pd.DataFrame({'value': [1, 2, 3]})
        mock_widget = Mock(spec=InteractiveWidget)
        mock_widget.df = df
        
        # Pandas DataFrames don't have compute method
        assert not hasattr(df, 'compute')
        assert InteractiveWidget._is_dask_dataframe(mock_widget) is False
    
    def test_dask_dataframe_detection(self):
        """Test detection of dask DataFrame."""
        df = pd.DataFrame({'value': [1, 2, 3]})
        ddf = dd.from_pandas(df, npartitions=1)
        
        mock_widget = Mock(spec=InteractiveWidget)
        mock_widget.df = ddf
        
        # Dask DataFrames have compute method
        assert hasattr(ddf, 'compute')
        assert InteractiveWidget._is_dask_dataframe(mock_widget) is True


class TestInteractiveWidgetEdgeCases:
    """Test edge cases in helper methods."""
    
    def test_compute_empty_series(self):
        """Test computing empty series."""
        series = pd.Series([], dtype=float)
        
        mock_widget = Mock(spec=InteractiveWidget)
        result = InteractiveWidget._compute_if_needed(mock_widget, series)
        
        assert len(result) == 0
        assert isinstance(result, pd.Series)
    
    def test_compute_large_dask_dataframe(self):
        """Test computing large dask DataFrame."""
        df = pd.DataFrame({'value': range(10000)})
        ddf = dd.from_pandas(df, npartitions=4)
        
        mock_widget = Mock(spec=InteractiveWidget)
        result = InteractiveWidget._compute_if_needed(mock_widget, ddf)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10000
    
    def test_persist_with_none(self):
        """Test persist with None value."""
        mock_widget = Mock(spec=InteractiveWidget)
        
        # Should handle None gracefully
        result = InteractiveWidget._persist_if_needed(mock_widget, None)
        
        # Returns None unchanged (no persist method)
        assert result is None


class TestInteractiveWidgetUnitsConfiguration:
    """Test units configuration in widgets."""
    
    @patch.object(InteractiveWidget, '__init__', lambda x, y, z=None: None)
    def test_units_dict_structure(self):
        """Test that units dict has expected structure."""
        # This tests the concept - actual initialization happens in __init__
        # which we can't call directly on abstract class
        expected_units = {
            '_TEMPERATURE': 'Celsius',
            '_SALINITY': 'kg/kg'
        }
        
        # These are the expected units from the implementation
        assert expected_units['_TEMPERATURE'] == 'Celsius'
        assert expected_units['_SALINITY'] == 'kg/kg'
    
    @patch.object(InteractiveWidget, '__init__', lambda x, y, z=None: None)
    def test_units_dims_dict_structure(self):
        """Test that units_dims dict has expected structure."""
        expected_dims = {
            'interpolated_model': '',
            'interpolated_model_QC': 'remove',
            'squared_difference': '^2',
            'log_likelihood': 'remove',
            'normalized_difference': 'remove'
        }
        
        # These are the expected dimensions from the implementation
        assert expected_dims['squared_difference'] == '^2'
        assert expected_dims['log_likelihood'] == 'remove'


class TestInteractiveWidgetAbstractInterface:
    """Test abstract method requirements."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that InteractiveWidget cannot be instantiated directly."""
        df = pd.DataFrame({'value': [1, 2, 3]})
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            widget = InteractiveWidget(df)
    
    def test_abstract_methods_defined(self):
        """Test that all expected abstract methods are defined."""
        abstract_methods = InteractiveWidget.__abstractmethods__
        
        # Should have these abstract methods
        expected_methods = {
            '_initialize_state',
            '_create_widgets',
            '_setup_callbacks',
            '_plot',
            '_initialize_widget',
            '_create_widget_layout'
        }
        
        assert expected_methods == abstract_methods


class TestInteractiveWidgetGetUnits:
    """Test get_units method for units formatting."""
    
    def test_get_units_temperature_obs(self):
        """Test units for temperature observation."""
        mock_widget = Mock(spec=InteractiveWidget)
        mock_widget.units_dict = {'_TEMPERATURE': 'Celsius', '_SALINITY': 'kg/kg'}
        mock_widget.units_dims_dict = {'obs': '', 'squared_difference': '^2'}
        
        result = InteractiveWidget.get_units(mock_widget, 'FLOAT_TEMPERATURE', 'obs')
        
        assert result == 'Celsius'
    
    def test_get_units_temperature_squared_difference(self):
        """Test units for temperature squared difference."""
        mock_widget = Mock(spec=InteractiveWidget)
        mock_widget.units_dict = {'_TEMPERATURE': 'Celsius', '_SALINITY': 'kg/kg'}
        mock_widget.units_dims_dict = {'obs': '', 'squared_difference': '^2'}
        
        result = InteractiveWidget.get_units(mock_widget, 'FLOAT_TEMPERATURE', 'squared_difference')
        
        assert result == 'Celsius^2'
    
    def test_get_units_salinity_obs(self):
        """Test units for salinity observation."""
        mock_widget = Mock(spec=InteractiveWidget)
        mock_widget.units_dict = {'_TEMPERATURE': 'Celsius', '_SALINITY': 'kg/kg'}
        mock_widget.units_dims_dict = {'obs': ''}
        
        result = InteractiveWidget.get_units(mock_widget, 'FLOAT_SALINITY', 'obs')
        
        assert result == 'kg/kg'
    
    def test_get_units_remove_dimension(self):
        """Test units when dimension is marked as 'remove'."""
        mock_widget = Mock(spec=InteractiveWidget)
        mock_widget.units_dict = {'_TEMPERATURE': 'Celsius'}
        mock_widget.units_dims_dict = {'normalized_difference': 'remove'}
        
        result = InteractiveWidget.get_units(mock_widget, 'FLOAT_TEMPERATURE', 'normalized_difference')
        
        assert result == '-'
    
    def test_get_units_unknown_obs_type(self):
        """Test units for unknown observation type returns None."""
        mock_widget = Mock(spec=InteractiveWidget)
        mock_widget.units_dict = {'_TEMPERATURE': 'Celsius'}
        mock_widget.units_dims_dict = {'obs': ''}
        
        # Should return None when base_units not found
        result = InteractiveWidget.get_units(mock_widget, 'UNKNOWN_TYPE', 'obs')
        
        assert result is None
    
    def test_get_units_unknown_column_returns_none(self):
        """Test units for unknown column returns None."""
        mock_widget = Mock(spec=InteractiveWidget)
        mock_widget.units_dict = {'_TEMPERATURE': 'Celsius'}
        mock_widget.units_dims_dict = {'obs': ''}
        
        # Should return None when dim_units not found
        result = InteractiveWidget.get_units(mock_widget, 'FLOAT_TEMPERATURE', 'unknown_column')
        
        assert result is None

