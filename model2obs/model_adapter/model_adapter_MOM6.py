"""ModelAdapter class to normalize MOM6 model input."""

from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import xarray as xr

from . import ModelAdapter, ModelAdapterCapabilities
from ..utils import config as config_utils

class ModelAdapterMOM6(ModelAdapter):
    """Base class for all model normalizations

    Provides common functionality for model input normalization.
    """
    capabilities = ModelAdapterCapabilities(
        supports_trim_obs = True,
        supports_no_matching = True,
        supports_force_obs_time = True,
        is_ocean = True
    )

    def __init__(self) -> None:

        # Assign ocean model name
        self.model_name = "MOM6"
        # Assign time_variable_name
        self.time_varname = "time"
        return

    def get_required_config_keys(self) -> List[str]:
        """Return list of required configuration keys.
        
        Returns:
            List of required configuration key names
        """
    
        return [
            'model_files_folder', 
            'obs_seq_in_folder', 
            'output_folder',
            'template_file', 
            'static_file', 
            'ocean_geometry',
            'perfect_model_obs_dir', 
            'parquet_folder'
        ]
    
    def get_common_model_keys(self) -> List[str]:
        """Return list of keys that are common to all input.nml files for this
        model
        
        Returns:
            List of common key

        """

        return [
            'template_file',
            'static_file',
            'ocean_geometry',
            'use_pseudo_depth',
            'model_state_variables',
            'layer_name'
        ]

    def validate_paths(self, config, run_opts) -> None:
        """Validate paths provided in config file."""

        super().validate_paths(config, run_opts)

        # MOM6 specific paths
        print("  Validating .nc files for model_nml...")
        config_utils.check_nc_file(config['template_file'], "template_file")
        config_utils.check_nc_file(config['static_file'], "static_file")
        config_utils.check_nc_file(config['ocean_geometry'], "ocean_geometry")

        # Ensure DART can perform vertical interpolation: either use_pseudo_depth
        # must be True, or the layer thickness variable must be in the state.
        # Only enforced when model_state_variables is explicitly configured.
        print("  Validating MOM6 vertical interpolation settings...")
        state_vars = config.get('model_state_variables')
        if state_vars is not None:
            use_pseudo_depth = config.get('use_pseudo_depth', False)
            has_thickness = 'QTY_LAYER_THICKNESS' in state_vars.values()
            if not use_pseudo_depth and not has_thickness:
                raise ValueError(
                    "MOM6 vertical interpolation is not configured. Either set "
                    "'use_pseudo_depth: true' in the config, or add a variable "
                    "mapped to 'QTY_LAYER_THICKNESS' in 'model_state_variables' "
                    "(e.g. 'h: QTY_LAYER_THICKNESS'). Without one of these, DART "
                    "will fail silently with error 1013 (THICKNESS_NOT_IN_STATE)."
                )

        return

    @contextmanager
    def open_dataset_ctx(self, path: str) -> Iterator[xr.Dataset]:
        """Open a MOM6 dataset, applying time fixes only when a time variable is present.

        For time-varying model output files, the MOM6 calendar attribute is fixed and the
        time variable is renamed to the canonical ``"time"`` name. For static files such
        as ``ocean_geometry.nc`` that carry no time dimension, the dataset is returned as-is.

        Args:
            path: Path to the netCDF file.

        Yields:
            xr.Dataset ready for use; time variable renamed to ``"time"`` when present.
        """
        
        ds = xr.open_dataset(
            path,
            decode_times=False
        )

        try:
            # Fix calendar as xarray does not read it consistently with ncviews.
            # Static geometry files have no time variable, so skip time processing.
            if self.time_varname in ds:
                ds[self.time_varname].attrs['calendar'] = 'proleptic_gregorian'
                ds = xr.decode_cf(ds, decode_timedelta=True)
                ds = self.rename_time_varname(ds)
            yield ds
        finally:
            ds.close()

    def convert_units(self, df) -> pd.DataFrame:
        """Convert observation or model units to match workflow
        
        Args:
            df: DataFrame with columns including 'type' and 'obs'

        Returns:
            df: Converted dataframe

        """

        # MOM6 is in PSU
        # DART's obs_seq are in PSU/1000
        # DART's pmo for MOM6 converts units
        # In the future DART might move to PSU
    
        return df

    def get_model_boundaries(self, geometry_file: str, margin: float = 0.0) -> Tuple[Polygon, np.ndarray]:
        """Get geographical boundaries from model input file using convex hull."""

        with self.open_dataset_ctx(geometry_file) as ds:
            # Extract geographical coordinates from the dataset
            xh = ds['lonh'].values
            yh = ds['lath'].values

            # Build grid and stack coordinates for convex hull calculation
            xh_mesh, yh_mesh = np.meshgrid(xh, yh)
            xh_flat = xh_mesh.flatten()
            yh_flat = yh_mesh.flatten()

            # Remove points of rectangular grid where model was not run
            # (e.g. Pacific when modeling Atlantic)
            # Assuming 'wet' variable indicates valid points
            ref_var = ds['wet'].values  # Shape: (len(yh), len(xh))
            valid_data = ref_var==1
            valid_data = valid_data.flatten()
            xh_flat = xh_flat[valid_data]
            yh_flat = yh_flat[valid_data]

            # Convert longitude to 0-360 convention and stack points for polygon
            xh_flat_360 = np.where(xh_flat < 0, xh_flat + 360, xh_flat)
            points = np.column_stack((xh_flat_360, yh_flat))
            if len(points) < 3:
                raise ValueError("Not enough valid points to create convex hull")

            # Calculate convex hull
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]

            # Create shapely polygon for point-in-polygon testing
            hull_polygon = Polygon(hull_points)

            # Get bounding box for reference
            lon_min, lat_min = hull_points.min(axis=0)
            lon_max, lat_max = hull_points.max(axis=0)

            print(f"    Model grid convex hull bounding box (lon, lat): "
                  f"[{lon_min:.2f}, {lon_max:.2f}], [{lat_min:.2f}, {lat_max:.2f}]")

            return hull_polygon, hull_points
