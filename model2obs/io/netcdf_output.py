"""
NetCDF output module for interpolated model-observation comparison data.

This module provides functionality to convert interpolated model values from
Dask DataFrames (stored in Parquet) to CF-compliant NetCDF4 files.

Two output structures are supported, selected automatically:

- **Transect mode** – used when the (latitude, longitude) pairs form a
  bijection (each unique latitude maps to exactly one longitude and vice
  versa).  The output has three dimensions (time × depth × latitude) and
  longitude is stored as a non-dimension coordinate ``longitude(latitude)``.
  This eliminates the spurious NaN cells that would arise from a Cartesian
  lat × lon cross-product.

- **Grid mode** – used otherwise.  The output has four dimensions
  (time × depth × latitude × longitude), matching the previous behaviour.

In both modes, a separate data variable is generated for each unique
observation type found in the ``type`` column of the input DataFrame
(e.g. ``interpolated_TEMPERATURE``, ``qc_flag_TEMPERATURE``).  All
per-type variables share the same coordinate grid, which is derived from
the union of all observation types.
"""

import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import xarray as xr


def _has_unique_latlon_pairs(
    lat_arr: "np.ndarray",
    lon_arr: "np.ndarray",
) -> bool:
    """Return ``True`` if the (latitude, longitude) pairs form a bijection.

    A bijection exists when every unique latitude value maps to exactly one
    unique longitude value **and** every unique longitude value maps to
    exactly one unique latitude value.

    Parameters
    ----------
    lat_arr:
        1-D array of (rounded) latitude values.
    lon_arr:
        1-D array of (rounded) longitude values with the same length.

    Returns
    -------
    bool
        ``True`` when the pairs are bijective; ``False`` otherwise.
    """
    unique_pairs = set(zip(lat_arr, lon_arr))
    lats = [p[0] for p in unique_pairs]
    lons = [p[1] for p in unique_pairs]
    # Each lat and each lon must appear exactly once across all unique pairs.
    return len(lats) == len(set(lats)) and len(lons) == len(set(lons))


def write_interpolated_to_netcdf(
    ddf,
    output_path: str,
    tolerances: Dict[str, float],
) -> None:
    """
    Write interpolated model-observation data to a CF-compliant NetCDF file.

    Converts a Dask DataFrame containing interpolated model values and their
    coordinates to a gridded NetCDF structure.  Applies coordinate tolerance
    to merge nearby locations.

    The output structure is selected automatically:

    - **Transect mode** – when the rounded (latitude, longitude) pairs form a
      bijection (each latitude ↔ exactly one longitude).  Dimensions are
      time × depth × latitude; longitude is stored as a non-dimension
      coordinate ``longitude(latitude)``.  This avoids the NaN overhead of a
      Cartesian lat × lon cross-product.
    - **Grid mode** – otherwise.  Dimensions are time × depth × latitude ×
      longitude (4-D), matching the original behaviour.

    In both modes a separate pair of variables is written for each unique
    observation type found in the ``type`` column, e.g.
    ``interpolated_TEMPERATURE`` / ``qc_flag_TEMPERATURE`` and
    ``interpolated_U_CURRENT_COMPONENT`` / ``qc_flag_U_CURRENT_COMPONENT``.
    All per-type variables share the same coordinate grid (derived from the
    union of all observation types).

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
        - ``type``: str, DART observation kind (e.g. ``"TEMPERATURE"``)

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
    - Global attribute ``coordinate_structure`` records which mode was used
      (``"transect"`` or ``"grid"``).
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
        'vertical', 'time', 'interpolated_model_QC', 'type',
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

    obs_types: List[str] = sorted(df['type'].unique())

    use_transect = _has_unique_latlon_pairs(
        df['latitude_rounded'].values,
        df['longitude_rounded'].values,
    )

    if use_transect:
        ds = _build_transect_dataset(df, obs_types)
        coord_structure = 'transect'
    else:
        ds = _build_grid_dataset(df, obs_types)
        coord_structure = 'grid'

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
    lon_attrs = {
        'units': 'degrees_east',
        'standard_name': 'longitude',
        'long_name': 'Longitude',
    }
    if not use_transect:
        # longitude is a full dimension only in grid mode
        lon_attrs['axis'] = 'X'
    ds['longitude'].attrs = lon_attrs

    for var in ds.data_vars:
        if var.startswith('interpolated_'):
            ds[var].attrs = {
                'long_name': (
                    f'Interpolated model value at observation location '
                    f'({var[len("interpolated_"):]})'
                ),
                'coordinates': 'time depth latitude longitude',
            }
        elif var.startswith('qc_flag_'):
            ds[var].attrs = {
                'units': '',
                'long_name': f'DART quality control flag ({var[len("qc_flag_"):]})',
                'coordinates': 'time depth latitude longitude',
            }

    if use_transect:
        comment = (
            'Transect-structured dataset: each latitude maps to exactly one '
            'longitude.  Longitude is stored as a non-dimension coordinate '
            'longitude(latitude).  No NaN fill from lat×lon cross-product.'
        )
    else:
        comment = (
            'Sparse 4D grid with interpolated model values. '
            'Missing grid points filled with NaN.'
        )

    ds.attrs = {
        'title': 'Interpolated model values at observation locations',
        'source': 'CrocoCamp model-observation comparison workflow',
        'Conventions': 'CF-1.8',
        'coordinate_tolerances': (
            f"longitude={tol_lon}, latitude={tol_lat}, depth={tol_depth}"
        ),
        'coordinate_structure': coord_structure,
        'history': 'Created by CrocoCamp write_interpolated_to_netcdf',
        'comment': comment,
    }

    encoding: Dict[str, dict] = {
        'depth': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
        'latitude': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
        'longitude': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
    }
    for var in ds.data_vars:
        if var.startswith('interpolated_'):
            encoding[var] = {
                'zlib': True, 'complevel': 4, '_FillValue': np.nan, 'dtype': 'float32',
            }
        elif var.startswith('qc_flag_'):
            encoding[var] = {
                'zlib': True, 'complevel': 4, '_FillValue': -999, 'dtype': 'int32',
            }

    try:
        ds.to_netcdf(output_path, format='NETCDF4', encoding=encoding)
    except Exception as exc:
        raise IOError(f"Failed to write NetCDF file {output_path}: {exc}") from exc


def _build_transect_dataset(df: pd.DataFrame, types: List[str]) -> xr.Dataset:
    """Build a 3-D transect-mode xarray Dataset with per-type variables.

    Dimensions are ``(time, depth, latitude)``.  Longitude is added as a
    non-dimension coordinate ``longitude(latitude)`` using the bijective
    lat → lon mapping present in *df*.

    One ``interpolated_{TYPE}`` and one ``qc_flag_{TYPE}`` variable is
    generated for each entry in *types*.  All variables share the same
    coordinate grid, which is derived from the union of all observation
    types in *df*.

    Parameters
    ----------
    df:
        DataFrame with columns ``time``, ``depth_rounded``,
        ``latitude_rounded``, ``longitude_rounded``, ``type``,
        ``interpolated_model``, and ``interpolated_model_QC``.
        The rounded columns must already satisfy the bijection property.
    types:
        Sorted list of unique observation type strings (values of the
        ``type`` column).

    Returns
    -------
    xr.Dataset
    """
    # Build shared coordinate template (union of all observations)
    all_grouped = df.groupby(['time', 'depth_rounded', 'latitude_rounded'])
    template = (
        all_grouped['interpolated_model']
        .mean()
        .to_xarray()
        .rename({'depth_rounded': 'depth', 'latitude_rounded': 'latitude'})
    )

    # Build the lat → lon mapping (bijective by construction for all types)
    lat_lon = (
        df[['latitude_rounded', 'longitude_rounded']]
        .drop_duplicates()
        .set_index('latitude_rounded')['longitude_rounded']
        .sort_index()
    )

    data_vars = {}
    for type_name in types:
        sub = df[df['type'] == type_name]
        grouped = sub.groupby(['time', 'depth_rounded', 'latitude_rounded'])
        da_interp = (
            grouped['interpolated_model']
            .mean()
            .to_xarray()
            .rename({'depth_rounded': 'depth', 'latitude_rounded': 'latitude'})
            .reindex_like(template, fill_value=np.nan)
        )
        da_qc = (
            grouped['interpolated_model_QC']
            .max()
            .to_xarray()
            .rename({'depth_rounded': 'depth', 'latitude_rounded': 'latitude'})
            .reindex_like(template, fill_value=-999)
            .astype('int32')
        )
        data_vars[f'interpolated_{type_name}'] = da_interp
        data_vars[f'qc_flag_{type_name}'] = da_qc

    ds = xr.Dataset(data_vars)

    lat_vals = ds['latitude'].values
    lon_vals = lat_lon.reindex(lat_vals).values
    ds = ds.assign_coords(
        longitude=xr.DataArray(lon_vals, coords=[lat_vals], dims=['latitude'])
    )

    return ds


def _build_grid_dataset(df: pd.DataFrame, types: List[str]) -> xr.Dataset:
    """Build a 4-D grid-mode xarray Dataset with per-type variables.

    Dimensions are ``(time, depth, latitude, longitude)``.  Unobserved
    grid points are filled with NaN / -999.

    One ``interpolated_{TYPE}`` and one ``qc_flag_{TYPE}`` variable is
    generated for each entry in *types*.  All variables share the same
    coordinate grid, which is derived from the union of all observation
    types in *df*.

    Parameters
    ----------
    df:
        DataFrame with columns ``time``, ``depth_rounded``,
        ``latitude_rounded``, ``longitude_rounded``, ``type``,
        ``interpolated_model``, and ``interpolated_model_QC``.
    types:
        Sorted list of unique observation type strings (values of the
        ``type`` column).

    Returns
    -------
    xr.Dataset
    """
    _index_cols = ['time', 'depth_rounded', 'latitude_rounded', 'longitude_rounded']
    _rename_map = {
        'depth_rounded': 'depth',
        'latitude_rounded': 'latitude',
        'longitude_rounded': 'longitude',
    }

    # Build shared coordinate template (union of all observations)
    all_indexed = df.set_index(_index_cols)
    all_interp = all_indexed.groupby(level=[0, 1, 2, 3])['interpolated_model'].mean()
    template = all_interp.to_xarray().unstack(fill_value=np.nan).rename(_rename_map)

    data_vars = {}
    for type_name in types:
        sub = df[df['type'] == type_name].set_index(_index_cols)
        da_interp = (
            sub.groupby(level=[0, 1, 2, 3])['interpolated_model']
            .mean()
            .to_xarray()
            .unstack(fill_value=np.nan)
            .rename(_rename_map)
            .reindex_like(template, fill_value=np.nan)
        )
        da_qc = (
            sub.groupby(level=[0, 1, 2, 3])['interpolated_model_QC']
            .max()
            .to_xarray()
            .unstack(fill_value=-999)
            .rename(_rename_map)
            .reindex_like(template, fill_value=-999)
            .astype('int32')
        )
        data_vars[f'interpolated_{type_name}'] = da_interp
        data_vars[f'qc_flag_{type_name}'] = da_qc

    return xr.Dataset(data_vars)
