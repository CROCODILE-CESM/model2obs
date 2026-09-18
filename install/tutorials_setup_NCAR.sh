#!/usr/bin/env bash

# Set up tutorial data on NCAR HPC from the shared campaign storage.
# CrocoLake is read directly from campaign and never copied; the tutorial data
# is copied into the clone because the tutorials write into it. There is no
# download fallback: on NCAR the campaign data is required.

set -euo pipefail

shared_tutorial_data_path="${NCAR_SHARED_DATA_ROOT%/}/model2obs"

fail() {
    printf '%s\n' "$@" >&2
    exit 1
}

for var in NCAR_SHARED_DATA_ROOT CROCOLAKE_PATH TUTORIAL_DATA_PATH; do
    [ -n "${!var:-}" ] || fail "Error: $var is not set: source envpaths_NCAR.sh first."
done

[ -d "$CROCOLAKE_PATH" ] && [ -r "$CROCOLAKE_PATH" ] || fail \
    "Error: CrocoLake is not readable at" \
    "         $CROCOLAKE_PATH" \
    "Check that you have access to GLADE campaign storage (group 'ncar')," \
    "or set NCAR_SHARED_DATA_ROOT to a location that holds CrocoLake/."

[ -d "$shared_tutorial_data_path" ] && [ -r "$shared_tutorial_data_path" ] || fail \
    "Error: shared tutorial data is not readable at" \
    "         $shared_tutorial_data_path" \
    "Check that you have access to GLADE campaign storage (group 'ncar')," \
    "or set NCAR_SHARED_DATA_ROOT to a location that holds model2obs/."

printf '%s\n' "Reading CrocoLake from $CROCOLAKE_PATH (not copied)."
printf '%s\n' "Copying tutorial data from $shared_tutorial_data_path..."
mkdir -p "$TUTORIAL_DATA_PATH"
cp -a "$shared_tutorial_data_path/." "$TUTORIAL_DATA_PATH/"
printf '%s\n' "Tutorial data setup complete."
