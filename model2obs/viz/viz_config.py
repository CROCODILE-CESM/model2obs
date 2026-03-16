"""Configuration classes for visualization defaults."""

from typing import List, Optional, Tuple


class ProfileConfig:
    """Configuration for interactive profile visualization.

    Provides default settings for the InteractiveProfileWidget with options for customization.
    All parameters can be overridden during initialization.
    """

    def __init__(
        self,
        colormap: str = 'viridis',
        figure_size: Tuple[int, int] = (10, 8),
        marker_size: int = 5,
        marker_alpha: float = 1,
        invert_yaxis: bool = True,
        grid: bool = True,
        initial_x: Optional[str] = None,
        initial_y: Optional[str] = None
    ) -> None:
        """Initialize profile configuration.

        Args:
            colormap: Matplotlib colormap name for colored plots
            figure_size: Figure size as (width, height) in inches
            marker_size: Size of scatter plot markers
            marker_alpha: Alpha transparency of markers
            invert_yaxis: Whether to invert the y-axis (typical for depth profiles)
            grid: Whether to show grid lines
            initial_x: Default x-axis column name
            initial_y: Default y-axis column name
        """
        self.colormap = colormap
        self.figure_size = figure_size
        self.marker_size = marker_size
        self.marker_alpha = marker_alpha
        self.invert_yaxis = invert_yaxis
        self.grid = grid
        self.initial_x = initial_x
        self.initial_y = initial_y


class MapConfig:
    """Configuration for interactive map visualization.

    Provides default settings for the InteractiveMapWidget with options for customization.
    All parameters can be overridden during initialization.
    """

    def __init__(
        self,
        colormap: str = 'cividis',
        map_extent: Optional[Tuple[float, float, float, float]] = None,
        vertical_range: Optional[Tuple[float, float]] = None,
        padding: float = 5.0,
        figure_size: Tuple[int, int] = (8, 6),
        scatter_size: int = 100,
        scatter_alpha: float = 0.7,
        default_window_hours: Optional[int] = None,
        disallowed_plotvars: Optional[List[str]] = None
    ) -> None:
        """Initialize map configuration.

        Args:
            colormap: Matplotlib colormap name
            plot_title: Base title for the plot
            map_extent: Map extent as (lon_min, lon_max, lat_min, lat_max) or None for auto
            padding: Padding in degrees for auto extent calculation
            figure_size: Figure size as (width, height) in inches
            scatter_size: Size of scatter points
            scatter_alpha: Alpha transparency of scatter points
            default_window_hours: Default time window in hours (defaults to 4 weeks)
            disallowed_plotvars: Variables to exclude from plot variable dropdown
        """
        self.colormap = colormap
        self.map_extent = map_extent
        self.vrange = None
        if vertical_range is not None:
            self.vrange = {}
            self.vrange['min'] = vertical_range[0]
            self.vrange['max'] = vertical_range[1]

        self.padding = padding
        self.figure_size = figure_size
        self.scatter_size = scatter_size
        self.scatter_alpha = scatter_alpha
        self.default_window_hours = default_window_hours or (24)  # 24 hours

        if disallowed_plotvars is None:
            self.disallowed_plotvars = ["time", "type", "longitude", "latitude", "vertical"]
        else:
            self.disallowed_plotvars = disallowed_plotvars

