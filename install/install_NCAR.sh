#!/usr/bin/env bash

set -euo pipefail

# flag to execute downloads for tutorials
TUTORIAL=0
for arg in "$@"; do
    if [[ "$arg" == "--tutorial" ]]; then
        TUTORIAL=1
    fi
done

## create conda environment with name set in envpaths_NCAR.sh
source ./envpaths_NCAR.sh
mamba env create --name "$CONDA_ENV_NAME" -f ../environment.yml -y
CONDA_ENV_PATH=$(conda env list | awk -v env="$CONDA_ENV_NAME" '$1 == env { print $NF }')

## scripts to run when environment is activated
mkdir -p $CONDA_ENV_PATH/etc/conda/activate.d

# load nco tools
echo 'module load nco' > $CONDA_ENV_PATH/etc/conda/activate.d/load_modules.sh
chmod +x $CONDA_ENV_PATH/etc/conda/activate.d/load_modules.sh

# load environmental paths
CONDA_SCRIPTS_PATH=$CONDA_ENV_PATH/etc/conda/activate.d/

# copy resolve environmental variables to conda's activate path
cat > "${CONDA_SCRIPTS_PATH}envpaths.sh" << EOF
export DART_ROOT_PATH="$DART_ROOT_PATH"
export CONDA_ENV_NAME="$CONDA_ENV_NAME"
export CROCOLAKE_OBS_CONV_PATH="$CROCOLAKE_OBS_CONV_PATH"
export PYTHONPATH="$PYTHONPATH"
export MODEL2OBS_PATH="$MODEL2OBS_PATH"
export CROCOLAKE_PATH="$CROCOLAKE_PATH"
export TUTORIAL_DATA_PATH="$TUTORIAL_DATA_PATH"
EOF

echo "source \"${CONDA_SCRIPTS_PATH}envpaths_NCAR.sh\"" > $CONDA_ENV_PATH/etc/conda/activate.d/load_paths.sh
chmod +x $CONDA_ENV_PATH/etc/conda/activate.d/load_paths.sh

if [[ "$TUTORIAL" -eq 1 ]]; then
    conda run -n "$CONDA_ENV_NAME" --no-capture-output ./tutorials_download.sh
fi
