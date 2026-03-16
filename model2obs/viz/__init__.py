"""Visualization tools for CrocoCamp data analysis.

This module provides interactive visualization widgets for analyzing
model-observation comparisons with support for both dask and pandas DataFrames.
"""

from .interactive_widget import InteractiveWidget
from .viz_config import MapConfig, ProfileConfig
from .interactive_widget_map import InteractiveWidgetMap
from .interactive_widget_profile import InteractiveWidgetProfile

__all__ = ['InteractiveWidget', 'InteractiveWidgetMap', 'InteractiveWidgetProfile', 'MapConfig', 'ProfileConfig']
