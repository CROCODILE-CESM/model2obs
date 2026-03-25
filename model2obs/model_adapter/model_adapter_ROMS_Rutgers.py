"""ModelAdapter class to normalize ROMS Rutgers model input."""

from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any, Dict, List
import pandas as pd
import xarray as xr

from . import ModelAdapter, ModelAdapterCapabilities
from ..utils import config as config_utils

class ModelAdapterROMSRutgers(ModelAdapter):
    """Base class for all model normalizations

    Provides common functionality for model input normalization.
    """

    capabilities = ModelAdapterCapabilities(
        supports_trim_obs = False,
        supports_no_matching = False,
        supports_force_obs_time = False
    )

    def __init__(self) -> None:

        # Assign ocean model name
        self.ocean_model = "ROMS_Rutgers"
        # Toggle ocean model output
        self.is_ocean = True
        # Assign time_varname_name
        self.time_varname = "ocean_time"
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
            'roms_filename',
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
            'roms_filename',
            'variables',
            'debug'
        ]


    def validate_paths(self, config, run_opts) -> None:
        """Validate paths provided in config file."""

        super().validate_paths(config, run_opts)

        # ROMS specific validation
        print("  Validating roms model file...")
        config_utils.check_nc_file(config['roms_filename'], "roms_filename")

        return

    @contextmanager
    def open_dataset_ctx(self, path: str) -> Iterator[xr.Dataset]:
        """Open a ROMS dataset, applying time fixes only when a time variable is present.

        For time-varying model output files, the time variable is decoded and renamed to
        the canonical ``"time"`` name. For static files that carry no time dimension,
        the dataset is returned as-is.

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

        # ROMS is in PSU
        # DART's obs_seq are in PSU/1000
        # DART's pmo for ROMS Rutgers does not convert units
        # In the future DART might move to PSU
        condition = df["type"].str.contains("SALINITY")
        df["obs"] = df["obs"].mask(condition, df["obs"] * 1000)
    
        return df

