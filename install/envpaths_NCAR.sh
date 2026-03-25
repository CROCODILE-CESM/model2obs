#!/usr/bin/env sh
export DART_ROOT_PATH="/glade/u/home/emilanese/work/DART-11.21.2-Casper/"

# Define conda environment name
export CONDA_ENV_NAME="model2obs"

#### DO NOT MODIFY BELOW THIS LINE ####
# Set up paths
export CROCOLAKE_OBS_CONV_PATH=${DART_ROOT_PATH%/}/observations/obs_converters/CrocoLake/
export PYTHONPATH="$CROCOLAKE_OBS_CONV_PATH:\$PYTHONPATH"

export MODEL2OBS_PATH=$(dirname "$PWD")/

export CROCOLAKE_PATH=$(dirname "$PWD")/CrocoLake/
export TUTORIAL_DATA_PATH=$(dirname "$PWD")/tutorial_data/
