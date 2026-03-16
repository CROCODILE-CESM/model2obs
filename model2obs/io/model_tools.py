"""Model grid and geospatial operations for CrocoCamp workflows."""

import numpy as np
import xarray as xr
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from typing import Tuple


def get_model_boundaries(model_file: str, margin: float = 0.0) -> Tuple[Polygon, np.ndarray]:
    """Get geographical boundaries from model input file using convex hull."""

    with xr.open_dataset(model_file) as ds:
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
