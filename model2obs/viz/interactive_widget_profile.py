"""Interactive 2D profile widget for model-observation comparison visualization."""

from typing import Optional, Union, List

import dask.dataframe as dd
import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output

from .interactive_widget import InteractiveWidget
from .viz_config import ProfileConfig


class InteractiveWidgetProfile(InteractiveWidget):
    """Interactive 2D profile widget for visualizing model-observation comparisons.

    This widget provides an interactive interface for creating 2D scatter plots
    with customizable x and y axes, and type filtering. Supports both dask and
    pandas DataFrames.
    """

    def __init__(
        self,
        dataframe: Union[pd.DataFrame, dd.DataFrame],
        x: Optional[str] = None,
        y: Optional[str] = None,
        config: Optional[ProfileConfig] = None
    ) -> None:
        """Initialize the interactive profile widget.

        Args:
            dataframe: Input dataframe (pandas or dask) containing observation data
            x: Column name for x-axis (defaults to 'obs' if available, 
               else first numeric column)
            y: Column name for y-axis (defaults to 'vertical' if available, 
               else second numeric column)
            config: ProfileConfig instance for customization (optional)
        """
        self.config = config or ProfileConfig()

        # Set initial axes from parameters or config
        self.x_column = x or self.config.initial_x
        self.y_column = y or self.config.initial_y

        super().__init__(dataframe, self.config)
        self._setup_widget_workflow()

    def _initialize_state(self) -> None:
        """Initialize widget-specific state variables."""
        # Internal state
        self.filtered_df = None
        self.plot_title = ""

        # Initialize default axes if not provided
        self._set_default_axes()

    def _set_default_axes(self) -> None:
        """Set default axes if not provided by user."""
        columns = self.df.columns.tolist()

        # Set x-axis default
        if self.x_column is None:
            if 'obs' in columns:
                self.x_column = 'obs'
            else:
                # Find first column that's not time
                for col in columns:
                    if col != 'time':
                        self.x_column = col
                        break
                if self.x_column is None and columns:
                    self.x_column = columns[0]

        # Set y-axis default
        if self.y_column is None:
            if 'vertical' in columns:
                self.y_column = 'vertical'
            else:
                # Find second column that's not time and different from x
                for col in columns:
                    if col not in ('time', self.x_column):
                        self.y_column = col
                        break
                if self.y_column is None and len(columns) > 1:
                    next_col = columns[1] if columns[1] != self.x_column else columns[0]
                    self.y_column = next_col

    def _create_widgets(self) -> None:
        """Create all UI widgets."""
        # Output widget for plot display
        self.output = widgets.Output()

        # X-axis dropdown
        axis_options = [col for col in self.df.columns.tolist() if col != 'time']
        self.x_dropdown = widgets.Dropdown(
            options=axis_options,
            value=self.x_column if self.x_column in axis_options else axis_options[0],
            description="X-axis:",
            style={'description_width': 'initial'}
        )

        # Y-axis dropdown
        self.y_dropdown = widgets.Dropdown(
            options=axis_options,
            value=self.y_column if self.y_column in axis_options else (
                axis_options[1] if len(axis_options) > 1 else axis_options[0]
            ),
            description="Y-axis:",
            style={'description_width': 'initial'}
        )

        # Type selection widget (multi-select)
        if 'type' in self.df.columns:
            type_options = self._compute_if_needed(
                self.df["type"].drop_duplicates()
            ).sort_values().tolist()
            default_types = (["FLOAT_TEMPERATURE"] if "FLOAT_TEMPERATURE" in type_options
                             else [type_options[0]])

            self.type_selector = widgets.SelectMultiple(
                options=type_options,
                value=default_types,
                description="Types:",
                style={'description_width': 'initial'},
                rows=min(8, len(type_options))  # Limit height
            )
        else:
            # Create dummy widget if no type column
            self.type_selector = widgets.HTML(value="<i>No 'type' column found</i>")

    def _setup_callbacks(self) -> None:
        """Set up widget observers."""
        self.x_dropdown.observe(self._on_axis_change, names='value')
        self.y_dropdown.observe(self._on_axis_change, names='value')
        if hasattr(self.type_selector, 'observe'):
            self.type_selector.observe(self._on_type_change, names='value')

    def _update_filtered_df(self, selected_types: List[str]) -> None:
        """Update filtered dataframe based on selected types."""
        if 'type' not in self.df.columns or not selected_types:
            self.filtered_df = self.df
        else:
            # Filter by selected types
            mask = self.df['type'].isin(selected_types)
            self.filtered_df = self._persist_if_needed(self.df[mask])

        # Update plot title
        if selected_types and len(selected_types) <= 3:
            type_str = ", ".join(selected_types)
        elif selected_types and len(selected_types) > 3:
            type_str = f"{selected_types[0]} + {len(selected_types)-1} others"
        else:
            type_str = "All data"

        x_axis = self.x_dropdown.value
        y_axis = self.y_dropdown.value
        self.plot_title = f"{y_axis} vs {x_axis} ({type_str})"

    def _plot(self) -> None:
        """Create the 2D scatter plot."""
        with self.output:
            clear_output(wait=True)

            if self.filtered_df is None:
                print("No data selected.")
                return

            # Get selected types for filtering
            if hasattr(self.type_selector, 'value'):
                selected_types = list(self.type_selector.value)
            else:
                selected_types = []

            # Get the data for plotting
            x_col = self.x_dropdown.value
            y_col = self.y_dropdown.value

            # Create the plot
            plt.figure(figsize=self.config.figure_size)
            ax = plt.gca()

            for obs_type in selected_types:

                self._update_filtered_df([obs_type])

                plot_df = self._compute_if_needed(
                    self.filtered_df[[x_col, y_col]]
                )

                if plot_df.empty:
                    print("No valid data points for selected axes and types.")
                    return

                ax.plot(
                    plot_df[x_col],
                    plot_df[y_col],
                    marker='o',
                    linestyle='None',
                    label=obs_type,
                    markersize = self.config.marker_size,
                    alpha = self.config.marker_alpha
                )

            ax.legend()

            # Configure axes
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)

            # Invert y-axis if configured (typical for depth profiles)
            if self.config.invert_yaxis:
                ax.invert_yaxis()

            # Add grid if configured
            if self.config.grid:
                ax.grid(True, alpha=0.3)

            # Set title
            ax.set_title(self.plot_title, fontsize=14, pad=15)

            # Add data count to plot
            bbox_props = {'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8}
            ax.text(0.02, 0.98, f'n = {len(plot_df):,} points',
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=bbox_props)

            plt.tight_layout()
            plt.show()

    # Callback methods
    def _on_axis_change(self, change):
        """Callback for axis dropdown change events."""
        # Suppress unused argument warning - required by ipywidgets interface
        _ = change
        self._plot()

    def _on_type_change(self, change):
        """Callback for type selector change events."""
        # Suppress unused argument warning - required by ipywidgets interface
        _ = change
        self._plot()

    def _initialize_widget(self) -> None:
        """Initialize widget state for display."""
        # Get initial selected types
        if hasattr(self.type_selector, 'value'):
            selected_types = list(self.type_selector.value)
        else:
            selected_types = []

        self._update_filtered_df(selected_types)

    def _create_widget_layout(self) -> widgets.Widget:
        """Create the widget layout for display."""
        # Create widget layout
        controls = [self.x_dropdown, self.y_dropdown]
        if hasattr(self.type_selector, 'observe'):  # Only add if it's a real selector
            controls.append(self.type_selector)

        widget_box = widgets.VBox([
            *controls,
            self.output
        ])

        return widget_box

    # Legacy method names for backward compatibility
    def plot_profile(self) -> None:
        """Legacy method name for _plot()."""
        self._plot()
