"""ModelAdapter class to normalize MOM6 model input."""

from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any, Dict, List
import pandas as pd
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
        supports_force_obs_time = True
    )

    def __init__(self) -> None:

        # Assign ocean model name
        self.ocean_model = "MOM6"
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
        
        return

    @contextmanager
    def open_dataset_ctx(self, path: str) -> Iterator[xr.Dataset]:
        """Open a MOM6 dataset with proper time decoding and calendar handling.

        Args:
            path: Path to MOM6 netCDF file

        Yields:
            xr.Dataset with properly decoded times and renamed time variable
        """
        
        ds = xr.open_dataset(
            path,
            decode_times=False
        )

        try:
            # Fix calendar as xarray does not read it consistently with ncviews
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

