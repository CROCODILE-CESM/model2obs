"""Base class for interactive visualization widgets."""

from abc import ABC, abstractmethod
from typing import Union

import dask.dataframe as dd
import ipywidgets as widgets
import pandas as pd
from IPython.display import display, clear_output


class InteractiveWidget(ABC):
    """Abstract base class for interactive visualization widgets.

    This class provides common functionality for visualization widgets including:
    - Pandas/Dask DataFrame handling
    - Widget lifecycle management
    - Abstract methods for widget-specific implementation
    """

    def __init__(self, dataframe: Union[pd.DataFrame, dd.DataFrame], viz_config=None):
        """Initialize the interactive widget.

        Args:
            dataframe: Input dataframe (pandas or dask) containing data
            config: Configuration instance for customization (optional)
        """
        self.df = dataframe
        self.config = viz_config
        self.output = None
        self.units_dict = {}
        self.units_dict['_TEMPERATURE'] = 'Celsius'
        self.units_dict['_SALINITY'] = 'kg/kg'
        self.units_dims_dict = {}
        self.units_dims_dict['interpolated_model'] = ''
        self.units_dims_dict['interpolated_model_QC'] = 'remove'
        self.units_dims_dict['obs'] = ''
        self.units_dims_dict['obs_err_var'] = ''
        self.units_dims_dict['difference'] = ''
        self.units_dims_dict['abs_difference'] = ''
        self.units_dims_dict['squared_difference'] = '^2'
        self.units_dims_dict['normalized_difference'] = 'remove'
        self.units_dims_dict['log_likelihood'] = 'remove'

    def _setup_widget_workflow(self):
        """Execute the standard widget setup workflow."""
        self._initialize_state()
        self._create_widgets()
        self._setup_callbacks()

    @abstractmethod
    def _initialize_state(self) -> None:
        """Initialize widget-specific state variables."""

    @abstractmethod
    def _create_widgets(self) -> None:
        """Create all UI widgets specific to this visualization type."""

    @abstractmethod
    def _setup_callbacks(self) -> None:
        """Set up widget observers for interactive updates."""

    @abstractmethod
    def _plot(self) -> None:
        """Create the visualization plot."""

    def _is_dask_dataframe(self) -> bool:
        """Check if the dataframe is a dask DataFrame."""
        return hasattr(self.df, 'compute')

    def _compute_if_needed(
        self, series_or_df: Union[pd.Series, pd.DataFrame, dd.Series, dd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        """Compute dask series/dataframe if needed, otherwise return as-is."""
        if hasattr(series_or_df, 'compute'):
            return series_or_df.compute()
        return series_or_df

    def _persist_if_needed(
        self, df: Union[pd.DataFrame, dd.DataFrame]
    ) -> Union[pd.DataFrame, dd.DataFrame]:
        """Persist dask dataframe if needed, otherwise return as-is."""
        if hasattr(df, 'persist'):
            return df.persist()
        return df

    def clear(self) -> None:
        """Clear the visualization output.

        This method clears the current visualization so users can call setup()
        again with different parameters without having both visualizations displayed.
        """
        if self.output:
            with self.output:
                clear_output(wait=True)

    def setup(self) -> widgets.Widget:
        """Initialize the widget with default selections and display.

        Returns:
            The widget box containing all controls and output
        """
        # Initialize with defaults
        self._initialize_widget()

        # Create and display widget layout
        widget_box = self._create_widget_layout()

        # Initial plot
        self._plot()

        return widget_box

    @abstractmethod
    def _initialize_widget(self) -> None:
        """Initialize widget state for display."""

    @abstractmethod
    def _create_widget_layout(self) -> widgets.Widget:
        """Create the widget layout for display."""

    def get_units(self,obs_type,col_name):
        """Returns the units of the selected variable.
        
        Args:
            obs_type: Observation type string (e.g., 'FLOAT_TEMPERATURE')
            col_name: Column name (e.g., 'obs', 'squared_difference')
            
        Returns:
            str: Formatted units string, or None if units cannot be determined
        """
        base_units = None
        dim_units = None
        units = None
        
        # Find base units from observation type
        for key in self.units_dict:
            if obs_type.endswith(key):
                base_units = self.units_dict[key]
                break
        if base_units is None:
            print(f"Warning: no unit found for {obs_type} in {self.units_dict}.")
            return None
        
        # Find dimension modifier from column name
        for key in self.units_dims_dict:
            if col_name == key:
                dim_units = self.units_dims_dict[key]
                break
        
        # Construct units string
        if dim_units == 'remove':
            units = '-'
        elif dim_units is None:
            print(f"Warning: no unit dimension found for {col_name} in {self.units_dims_dict}.")
            return None
        else:
            units = base_units + dim_units

        return units
