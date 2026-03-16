"""Interactive map widget for model-observation comparison visualization."""

import re
from datetime import timedelta
from typing import Any, Optional, Tuple, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dask.dataframe as dd
import ipywidgets as widgets
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from IPython.display import display, clear_output

from .interactive_widget import InteractiveWidget
from .viz_config import MapConfig


class InteractiveWidgetMap(InteractiveWidget):
    """Interactive map widget for visualizing model-observation comparisons.
    
    This widget provides an interactive interface for exploring temporal and spatial
    patterns in model-observation differences with support for both dask and pandas DataFrames.
    """
    
    def __init__(self, dataframe: Union[pd.DataFrame, dd.DataFrame], config: Optional[MapConfig] = None) -> None:
        """Initialize the interactive map widget.
        
        Args:
            dataframe: Input dataframe (pandas or dask) containing observation data
            config: MapConfig instance for customization (optional)
        """
        self.config = config or MapConfig()
        super().__init__(dataframe, self.config)
        self._setup_widget_workflow()

    def _initialize_state(self) -> None:
        """Initialize widget-specific state variables."""
        # Internal state
        self.filtered_df = None
        self.min_time = None
        self.max_time = None
        self.total_hours = None
        self.plot_var = None
        self.map_extent = None
        self.plot_title = None
        self.vrange = None
        
        # Initialize map-specific calculations
        self._calculate_vertical_limits()
        self._calculate_map_extent()

    def _calculate_map_extent(self) -> None:
        """Calculate map extent from data if not provided in config."""
        if self.config.map_extent is not None:
            self.map_extent = self.config.map_extent
            return
            
        # Auto-calculate extent with padding
        padding = self.config.padding
        lon_min = (self._compute_if_needed(self.df['longitude'].min()) % 180 - 180) - padding
        if lon_min < -170:
            lon_min = -180
        lon_max = (self._compute_if_needed(self.df['longitude'].max()) % 180 - 180) + padding
        if lon_max > 170:
            lon_max = 180
        lat_min = self._compute_if_needed(self.df['latitude'].min()) - padding
        if lat_min < -80:
            lat_min = -90
        lat_max = self._compute_if_needed(self.df['latitude'].max()) + padding
        if lat_max > 80:
            lat_max = 90
            
        self.map_extent = (lon_min, lon_max, lat_min, lat_max)
        
    def _calculate_vertical_limits(self) -> None:
        """Calculate limits for vertical coordinate."""

        # Auto-calculate extent with padding
        vert_min = (self._compute_if_needed(self.df['vertical'].min()))
        vert_max = (self._compute_if_needed(self.df['vertical'].max()))

        self.vertical_limits = {}
        self.vertical_limits["min"] = vert_min
        self.vertical_limits["max"] = vert_max

        if self.config.vrange is None:
            self.vrange = self.vertical_limits
        else:
            self.vrange = self.config.vrange

    def _create_widgets(self) -> None:
        """Create all UI widgets."""
        # Output widget for plot display
        self.output = widgets.Output()
        
        # Select available observation types
        type_options = self._compute_if_needed(self.df["type"].drop_duplicates()).sort_values().tolist()
        self.type_dropdown = widgets.Dropdown(
            options=type_options,
            value="FLOAT_TEMPERATURE" if "FLOAT_TEMPERATURE" in type_options else type_options[0],
            description="Observation type:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='85%')
        )
        
        # Select plotted variable
        refvar_options = [val for val in self.df.columns.to_list() 
                         if val not in self.config.disallowed_plotvars]
        self.refvar_dropdown = widgets.Dropdown(
            options=refvar_options,
            value="difference" if "difference" in refvar_options else refvar_options[0],
            description="Plotted variable",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='85%')
        )
        
        # Window sliders for selecting the time window
        self.window_slider = widgets.IntSlider(
            value=self.config.default_window_hours,
            min=1,
            max=self.config.default_window_hours,
            step=1,
            description='Window (hrs):',
            style={'description_width': 'initial'},
            continuous_update=False,
            layout=widgets.Layout(width='85%')
        )

        # Center time slider initialization (dummy value to avoid TraitError)
        dummy_time = pd.Timestamp('2000-01-01 00:00:00')
        self.center_slider = widgets.SelectionSlider(
            options=[dummy_time],
            value=dummy_time,
            description='Window centered on:',
            style={'description_width': 'initial'},
            continuous_update=False,
            layout=widgets.Layout(width='85%')
        )
        
        # Colorbar slider for map color limits
        self.colorbar_slider = widgets.FloatRangeSlider(
            value=[0, 1],
            min=-1e10,
            max=1e10,
            step=0.01,
            description='Colorbar limits:',
            style={'description_width': 'initial'},
            continuous_update=False,
            layout=widgets.Layout(width='85%')
        )
        self.min_cb = widgets.FloatText(
            value=self.colorbar_slider.value[0],
            description='Colorbar min:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='210px'),
            step=0.001,
        )
        self.max_cb = widgets.FloatText(
            value=self.colorbar_slider.value[1],
            description='Colorbar max:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='210px'),
            step=0.001
        )
        
        # Vertical coordinate slider for selecting the depth range to plot
        self.vrange_slider = widgets.FloatRangeSlider(
            value=[self.vrange['min'], self.vrange['max']],
            min=self.vertical_limits['min'],
            max=self.vertical_limits['max'],
            step=0.1,
            description='Depth range [m]:',
            style={'description_width': 'initial'},
            continuous_update=False,
            layout=widgets.Layout(width='85%')
        )

    def _setup_callbacks(self) -> None:
        """Set up widget observers."""
        self.refvar_dropdown.observe(self._on_refvar_change, names='value')
        self.type_dropdown.observe(self._on_type_change, names='value')
        self.window_slider.observe(self._on_window_change, names='value')
        self.center_slider.observe(self._on_center_change, names='value')
        self.colorbar_slider.observe(self._on_colorbar_change, names='value')
        self.min_cb.observe(self._on_min_cb_change, names='value')
        self.max_cb.observe(self._on_max_cb_change, names='value')
        self.vrange_slider.observe(self._on_vrange_change, names='value')
        
    def parse_window(self, text: str) -> timedelta:
        """Parse a human-readable window string to a timedelta."""
        text = text.strip().lower()
        if not text:
            return None
        patterns = [
            (r'(\d+)\s*weeks?', 'weeks'),
            (r'(\d+)\s*days?', 'days'),
            (r'(\d+)\s*hours?', 'hours'),
            (r'(\d+)\s*minutes?', 'minutes'),
        ]
        kwargs = {}
        for pat, unit in patterns:
            m = re.search(pat, text)
            if m:
                kwargs[unit] = int(m.group(1))
        if kwargs:
            return timedelta(**kwargs)
        try:
            return pd.to_timedelta(text)
        except Exception:
            return None
            
    def _get_window_timedelta(self):
        """Get current window timedelta from text or slider."""
        return timedelta(hours=self.window_slider.value)

    def _update_refvar(self, selected_var):
        """Update global variable for the plotted variable."""
        self.plot_var = selected_var
        self.plot_title = f'{self.plot_var}'

    def _update_filtered_df(self, selected_type):
        """Update filtered dataframe and its time metadata for a selected type."""
        self.filtered_df = self._persist_if_needed(
            self.df[self.df["type"] == selected_type]
        )
        self.min_time = self._compute_if_needed(self.filtered_df['time'].min())
        self.max_time = self._compute_if_needed(self.filtered_df['time'].max())
        self.total_hours = int((self.max_time - self.min_time).total_seconds() // 3600)

        # update vertical coordinate data
        vert_min = (self._compute_if_needed(self.df['vertical'].min()))
        vert_max = (self._compute_if_needed(self.df['vertical'].max()))

        self.vertical_limits = {}
        self.vertical_limits["min"] = vert_min
        self.vertical_limits["max"] = vert_max
        if self.vrange is None:
            self.vrange = self.vertical_limits
        
    def _update_center_slider(self, window_td):
        """Update center time slider options based on currently filtered data and window."""
        dummy_time = pd.Timestamp('2000-01-01 00:00:00')
        if self.min_time is None or self.max_time is None:
            self.center_slider.options = [dummy_time]
            self.center_slider.value = dummy_time
            return
        half_window = window_td / 2
        center_min = self.min_time + half_window
        center_max = self.max_time - half_window
        if center_min > center_max:
            self.center_slider.options = [self.min_time]
            self.center_slider.value = self.min_time
            return
        options = pd.date_range(center_min, center_max, freq='1h')
        self.center_slider.options = options
        if self.center_slider.value not in options:
            self.center_slider.value = options[0] if len(options) > 0 else self.min_time
            
    def _update_colorbar_slider(self):
        """Update the colorbar slider limits and step according to the filtered data
        and selected variable. Uses the current time window.
        """
        t0 = self.center_slider.value - self._get_window_timedelta() / 2
        t1 = self.center_slider.value + self._get_window_timedelta() / 2
        df_win = self.filtered_df[
            (self.filtered_df['time'] >= t0) &
            (self.filtered_df['time'] <= t1)
        ]
        col_min = self._compute_if_needed(df_win[self.plot_var].min())
        col_max = self._compute_if_needed(df_win[self.plot_var].max())
        step = (col_max - col_min) / 100. if (col_max - col_min) > 0 else 0.01

        # Update slider properties in the correct order to avoid constraint violations
        # First set value to be within the old range but prepare for new range
        old_min = self.colorbar_slider.min
        old_max = self.colorbar_slider.max
        new_min = float(col_min)
        new_max = float(col_max)

        # If new range is entirely above old range, raise max first
        if new_min >= old_max:
            self.colorbar_slider.max = new_max
            self.colorbar_slider.min = new_min
        # If new range is entirely below old range, lower min first
        elif new_max <= old_min:
            self.colorbar_slider.min = new_min
            self.colorbar_slider.max = new_max
        # Otherwise, expand the range first, then narrow it
        else:
            self.colorbar_slider.min = min(old_min, new_min)
            self.colorbar_slider.max = max(old_max, new_max)
            self.colorbar_slider.min = new_min
            self.colorbar_slider.max = new_max

        self.colorbar_slider.step = step
        self.colorbar_slider.value = [new_min, new_max]

    def _plot(self) -> None:
        """Create the map plot with current settings."""
        center = self.center_slider.value if hasattr(self, 'center_slider') else None
        window_td = self._get_window_timedelta() if hasattr(self, 'window_slider') else timedelta(hours=24)
        self.plot_map(center, window_td)
        
    def plot_map(self, center: pd.Timestamp, window_td: timedelta) -> None:
        """Plot the reference map showing mean values of plot_var for each location (lat, lon)
        within the selected time window.
        """
        with self.output:
            clear_output(wait=True)
            if self.filtered_df is None or center is None or window_td is None:
                print("No data selected.")
                return
            t0 = center - window_td / 2
            t1 = center + window_td / 2
            df_win = self.filtered_df[
                (self.filtered_df['time'] >= t0) &
                (self.filtered_df['time'] <= t1) &
                (self.filtered_df['vertical'] >= self.vrange['min']) &
                (self.filtered_df['vertical'] <= self.vrange['max'])
            ]
            df_win = self._compute_if_needed(
                df_win[ ['latitude', 'longitude','vertical',self.plot_var] ]
            )

            fig = plt.figure(figsize=self.config.figure_size)
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
            ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            
            if not df_win.empty:
                vmin, vmax = self.colorbar_slider.value
                df_win = df_win.sort_values(by='vertical', ascending=False)
                scatter = ax.scatter(
                    df_win['longitude'],
                    df_win['latitude'],
                    s=self.config.scatter_size,
                    alpha=self.config.scatter_alpha,
                    c=df_win[self.plot_var],
                    vmin=vmin,
                    vmax=vmax,
                    cmap=self.config.colormap,
                    label=f'{self.plot_var} (n={len(df_win):,})',
                    marker='o',
                    edgecolors='none',
                    transform=ccrs.PlateCarree()
                )

                cbar = fig.colorbar(scatter, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
                cbar.set_label(f'{self.plot_var} [{self.get_units(self.type_dropdown.value,self.plot_var)}]')

                ax.set_extent(self.map_extent, crs=ccrs.PlateCarree())

            else:
                ax.set_global()
                ax.text(
                    0.5,
                    0.5,
                    'No data in selected window',
                    ha='center',
                    va='center',
                    fontsize=20,
                    transform=ax.transAxes
                )
            gl = ax.gridlines(
                draw_labels=True,
                linewidth=1,
                color='gray',
                alpha=0.5,
                linestyle='--'
            )
            gl.top_labels = False
            gl.right_labels = False

            wtd_days = window_td.days
            wtd_hours = window_td.seconds // 3600
            plt.title(
                f'{self.plot_title}\n({len(df_win):,} points)\nTime window: {wtd_days} days, {wtd_hours} hours\nCentered on: {self.center_slider.value}',
                fontsize=16,
                pad=20
            )
            plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
            plt.tight_layout()
            plt.show()

    def _initialize_widget(self) -> None:
        """Initialize widget state for display."""
        # Initial setup
        self._update_refvar(self.refvar_dropdown.value)
        self._update_filtered_df(self.type_dropdown.value)
        self.window_slider.max = max(self.total_hours, 1)
        self._update_center_slider(self._get_window_timedelta())
        self._update_colorbar_slider()

    def _create_widget_layout(self) -> widgets.Widget:
        """Create the widget layout for display."""
        # Create and display widget layout
        widgets_list = [
            self.refvar_dropdown,
            self.type_dropdown,
            self.window_slider,
            self.center_slider,
            self.colorbar_slider,
            self.vrange_slider,
        ]
        grid_box = widgets.GridBox(
            widgets_list,
            layout=widgets.Layout(grid_template_columns="repeat(2, 500px)")
        )
        cb_text_box = widgets.HBox(
            [self.min_cb, self.max_cb,],
            layout=widgets.Layout(width='500px')
        )

        widgets_box = widgets.VBox(
            [grid_box, cb_text_box, self.output]
        )

        return widgets_box

    # Callback methods
    def _on_type_change(self, change):
        """Callback for type dropdown change event."""
        self._update_filtered_df(change['new'])
        self.window_slider.max = max(self.total_hours, 1)
        self.window_slider.value = min(self.window_slider.value, self.window_slider.max)
        self._update_center_slider(self._get_window_timedelta())
        self._update_colorbar_slider()
        self.plot_map(self.center_slider.value, self._get_window_timedelta())

    def _on_refvar_change(self, change):
        """Callback for reference variable dropdown change event."""
        self._update_refvar(change['new'])
        self.window_slider.max = max(self.total_hours, 1)
        self.window_slider.value = min(self.window_slider.value, self.window_slider.max)
        self._update_center_slider(self._get_window_timedelta())
        self._update_colorbar_slider()
        self.plot_map(self.center_slider.value, self._get_window_timedelta())

    def _on_window_change(self, change):
        """Callback for window slider/text change event."""
        window_td = self._get_window_timedelta()
        self._update_center_slider(window_td)
        self._update_colorbar_slider()
        self.plot_map(self.center_slider.value, window_td)

    def _on_center_change(self, change):
        """Callback for center slider change event."""
        self._update_colorbar_slider()
        self.plot_map(self.center_slider.value, self._get_window_timedelta())

    def _on_colorbar_change(self, change):
        """Callback for colorbar slider change event."""
        min_val, max_val = change['new']
        if self.min_cb.value != min_val:
            self.min_cb.value = min_val
        if self.max_cb.value != max_val:
            self.max_cb.value = max_val
        self.plot_map(self.center_slider.value, self._get_window_timedelta())

    def _on_min_cb_change(self, change):
        """Update slider when min_cb changes."""
        self.colorbar_slider.value = (change['new'], self.colorbar_slider.value[1])
        self.plot_map(self.center_slider.value, self._get_window_timedelta())

    def _on_max_cb_change(self, change):
        """Update slider when max_cb changes."""
        self.colorbar_slider.value = (self.colorbar_slider.value[0], change['new'])
        self.plot_map(self.center_slider.value, self._get_window_timedelta())
        
    def _on_vrange_change(self, change):
        """Callback for vertical coordinate slider change event."""
        window_td = self._get_window_timedelta()
        self.vrange['min'], self.vrange['max'] = self.vrange_slider.value
        self._update_colorbar_slider()
        self.plot_map(self.center_slider.value, window_td)
