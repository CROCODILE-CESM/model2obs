#!/usr/bin/env sh
export DART_ROOT_PATH="${DART_ROOT_PATH:-/glade/u/home/emilanese/work/DART-11.21.2-Casper/}"

# Define conda environment name (if not defined already)
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-model2obs}"

# Root of the shared, read-only workshop data on GLADE campaign storage
export NCAR_SHARED_DATA_ROOT="${NCAR_SHARED_DATA_ROOT:-/glade/campaign/cgd/oce/projects/CROCODILE/workshops/2026}"

#### DO NOT MODIFY BELOW THIS LINE ####
# Set up paths
export CROCOLAKE_OBS_CONV_PATH=${DART_ROOT_PATH%/}/observations/obs_converters/CrocoLake/
export PYTHONPATH="$CROCOLAKE_OBS_CONV_PATH:\$PYTHONPATH"

export MODEL2OBS_PATH=$(dirname "$PWD")/

# CrocoLake is read-only and stays on campaign storage: it is never copied.
# Derived unconditionally, so a stale value exported by an activated model2obs
# conda env cannot override it: use NCAR_SHARED_DATA_ROOT to relocate the data.
export CROCOLAKE_PATH="${NCAR_SHARED_DATA_ROOT%/}/CrocoLake/"

# Tutorial data is read and written, so it lives in the user's clone
export TUTORIAL_DATA_PATH=$(dirname "$PWD")/tutorial_data/
