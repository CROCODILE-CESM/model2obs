"""ModelAdapter class to normalize CICE model input."""

from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import xarray as xr
import warnings

from . import ModelAdapter, ModelAdapterCapabilities
from ..utils import config as config_utils

class ModelAdapterCICE(ModelAdapter):
    """Base class for all model normalizations

    Provides common functionality for model input normalization.
    """
    capabilities = ModelAdapterCapabilities(
        supports_trim_obs = False,
        supports_no_matching = True,
        supports_force_obs_time = True,
        is_sea_ice = True
    )

    def __init__(self) -> None:

        # Assign ocean model name
        self.model_name = "CICE"
        # Assign time_variable_name
        self.time_varname = None
        warnings.warn(
            "time_varname not defined for CICE yet"
        )
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
            'variables',
        ]

    def validate_paths(self, config, run_opts) -> None:
        """Validate paths provided in config file."""

        super().validate_paths(config, run_opts)

        # CICE specific validation
        print("  Validating CICE model file...")
        config_utils.check_nc_file(config['cice_filename'], "cice_filename")

        return

    @contextmanager
    def open_dataset_ctx(self, path: str) -> Iterator[xr.Dataset]:
        """Open a CICE dataset, applying time fixes only when a time variable is present.

        For time-varying model output files, the CICE calendar attribute is fixed and the
        time variable is renamed to the canonical ``"time"`` name. For static files such
        as ``ocean_geometry.nc`` that carry no time dimension, the dataset is returned as-is.

        Args:
            path: Path to the netCDF file.

        Yields:
            xr.Dataset ready for use; time variable renamed to ``"time"`` when present.
        """
        
        raise ValueError(
            "not implemented for CICE yet"
        )

    def convert_units(self, df) -> pd.DataFrame:
        """Convert observation or model units to match workflow
        
        Args:
            df: DataFrame with columns including 'type' and 'obs'

        Returns:
            df: Converted dataframe

        """

        raise ValueError(
            "not implemented for CICE yet"
        )
