"""
NetCDF output module for interpolated model-observation comparison data.

This module provides functionality to convert interpolated model values from
Dask DataFrames (stored in Parquet) to CF-compliant NetCDF4 files with a
4D gridded structure (time, depth, latitude, longitude).
"""

import warnings
from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr


def write_interpolated_to_netcdf(
    ddf,
    output_path: str,
    tolerances: Dict[str, float],
) -> None:
    """
    Write interpolated model-observation data to a CF-compliant NetCDF file.

    Converts a Dask DataFrame containing interpolated model values and their
    coordinates to a 4D gridded NetCDF structure (time × depth × latitude ×
    longitude).  Applies coordinate tolerance to merge nearby locations and
    handles sparse data with NaN fill values.

    Time is stored as datetime objects; xarray handles CF-compliant encoding
    automatically.

    Parameters
    ----------
    ddf : dask.dataframe.DataFrame or pandas.DataFrame
        DataFrame with columns:

        - ``interpolated_model``: float, model value at observation location
        - ``longitude``: float, degrees East
        - ``latitude``: float, degrees North
        - ``vertical``: float, depth/vertical coordinate (renamed to ``depth``)
        - ``time``: datetime-like, calendar time of the observation
        - ``interpolated_model_QC``: int, DART QC flag

    output_path : str
        Path to the output NetCDF file.
    tolerances : dict of str to float
        Coordinate merging tolerances with keys:

        - ``'longitude'``: degrees (e.g., ``1e-2``)
        - ``'latitude'``: degrees (e.g., ``1e-2``)
        - ``'depth'``: metres (e.g., ``1e-1``)

    Raises
    ------
    ValueError
        If required columns are missing from the DataFrame.
    IOError
        If the NetCDF file cannot be written.

    Notes
    -----
    - The ``vertical`` column is renamed to ``depth`` with units in metres.
    - Missing grid points are filled with NaN (``_FillValue``).
    - Compression (zlib level 4) is applied to all variables.
    - Output follows CF conventions with proper attributes.
    """
    try:
        import dask.dataframe as dd
    except ImportError as exc:
        raise ImportError(
            "Dask is required for NetCDF output. "
            "Install with: pip install dask[dataframe]"
        ) from exc

    if not isinstance(ddf, (dd.DataFrame, pd.DataFrame)):
        raise TypeError(f"Expected Dask or Pandas DataFrame, got {type(ddf)}")

    required_cols = [
        'interpolated_model', 'longitude', 'latitude',
        'vertical', 'time', 'interpolated_model_QC',
    ]
    missing_cols = [col for col in required_cols if col not in ddf.columns]
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")

    df = ddf.compute() if isinstance(ddf, dd.DataFrame) else ddf.copy()

    if len(df) == 0:
        warnings.warn(
            f"DataFrame is empty, creating NetCDF with no data: {output_path}",
            UserWarning
        )
        xr.Dataset().to_netcdf(output_path)
        return

    df = df.rename(columns={'vertical': 'depth'})

    tol_lon = tolerances.get('longitude', 1e-2)
    tol_lat = tolerances.get('latitude', 1e-2)
    tol_depth = tolerances.get('depth', 1e-1)

    df['longitude_rounded'] = np.round(df['longitude'] / tol_lon) * tol_lon
    df['latitude_rounded'] = np.round(df['latitude'] / tol_lat) * tol_lat
    df['depth_rounded'] = np.round(df['depth'] / tol_depth) * tol_depth

    df = df.set_index(['time', 'depth_rounded', 'latitude_rounded', 'longitude_rounded'])

    interpolated = df.groupby(level=[0, 1, 2, 3])['interpolated_model'].mean()
    qc_flag = df.groupby(level=[0, 1, 2, 3])['interpolated_model_QC'].max()

    ds_interp = interpolated.to_xarray().unstack(fill_value=np.nan)
    ds_qc = qc_flag.to_xarray().unstack(fill_value=-999)

    ds_interp = ds_interp.rename({
        'depth_rounded': 'depth',
        'latitude_rounded': 'latitude',
        'longitude_rounded': 'longitude',
    })
    ds_qc = ds_qc.rename({
        'depth_rounded': 'depth',
        'latitude_rounded': 'latitude',
        'longitude_rounded': 'longitude',
    })

    ds = xr.Dataset({'interpolated_model': ds_interp, 'qc_flag': ds_qc})

    ds['time'].attrs = {
        'standard_name': 'time',
        'long_name': 'Time',
        'axis': 'T',
    }
    ds['depth'].attrs = {
        'units': 'meters',
        'standard_name': 'depth',
        'long_name': 'Depth below sea surface',
        'positive': 'down',
        'axis': 'Z',
    }
    ds['latitude'].attrs = {
        'units': 'degrees_north',
        'standard_name': 'latitude',
        'long_name': 'Latitude',
        'axis': 'Y',
    }
    ds['longitude'].attrs = {
        'units': 'degrees_east',
        'standard_name': 'longitude',
        'long_name': 'Longitude',
        'axis': 'X',
    }
    ds['interpolated_model'].attrs = {
        'long_name': 'Interpolated model value at observation location',
        'coordinates': 'time depth latitude longitude',
    }
    ds['qc_flag'].attrs = {
        'units': '1',
        'long_name': 'DART quality control flag',
        'coordinates': 'time depth latitude longitude',
    }
    ds.attrs = {
        'title': 'Interpolated model values at observation locations',
        'source': 'CrocoCamp model-observation comparison workflow',
        'Conventions': 'CF-1.8',
        'coordinate_tolerances': (
            f"longitude={tol_lon}, latitude={tol_lat}, depth={tol_depth}"
        ),
        'history': 'Created by CrocoCamp write_interpolated_to_netcdf',
        'comment': (
            'Sparse 4D grid with interpolated model values. '
            'Missing grid points filled with NaN.'
        ),
    }

    encoding = {
        'interpolated_model': {
            'zlib': True, 'complevel': 4, '_FillValue': np.nan, 'dtype': 'float32',
        },
        'qc_flag': {
            'zlib': True, 'complevel': 4, '_FillValue': -999, 'dtype': 'int32',
        },
        'depth': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
        'latitude': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
        'longitude': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
    }

    try:
        ds.to_netcdf(output_path, format='NETCDF4', encoding=encoding)
    except Exception as exc:
        raise IOError(f"Failed to write NetCDF file {output_path}: {exc}") from exc
