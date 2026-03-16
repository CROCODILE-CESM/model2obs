"""File handling utilities for CrocoCamp workflows."""

import glob
import os
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import xarray as xr


def get_sorted_files(directory: str, pattern: str = "*") -> List[str]:
    """Get sorted list of files in directory matching pattern."""
    file_pattern = os.path.join(directory, pattern)
    files = glob.glob(file_pattern)
    files = [f for f in files if os.path.isfile(f)]
    return sorted(files)


def timestamp_to_days_seconds(timestamp: np.datetime64) -> Tuple[int, int]:
    """Convert YYYYMMDD HH:MM:SS timestamp to number of days, number of
    seconds since 1601-01-01

    Arguments:
    timestamp: timestamp in numpy datetime64 format

    Returns:
    days (int): number of days since 1601-01-01
    seconds (int): number of seconds since (1601-01-01 + days)
    """

    timestamp = timestamp.astype('datetime64[s]').astype(datetime)
    reference_date = datetime(1601, 1, 1)
    time_difference = timestamp - reference_date
    days = time_difference.days
    seconds_in_day = time_difference.seconds

    return days, seconds_in_day


def get_model_time_in_days_seconds(model_in_file: str, time_var: str) -> Tuple[int, int]:
    """Get model time in days and seconds from model input file."""

    with xr.open_dataset(model_in_file, decode_timedelta=True) as model_ds:
        model_time = model_ds[time_var].values
    model_time = np.atleast_1d(model_time)
    if len(model_time) > 1:
        raise ValueError(f"Model input file {model_in_file} contains multiple time steps, expected single time step.")
    return timestamp_to_days_seconds(model_time[0])


def get_obs_time_in_days_seconds(obs_in_file: str) -> Tuple[int, int]:
    """Get obs_seq.in time in days and seconds from obs input file."""
    import pydartdiags.obs_sequence.obs_sequence as obsq

    obs_in_df = obsq.ObsSequence(obs_in_file)
    t1 = obs_in_df.df.time.min()
    t2 = obs_in_df.df.time.max()
    tmid = pd.Timestamp((t1.value + t2.value) // 2)

    return timestamp_to_days_seconds(np.datetime64(tmid))
