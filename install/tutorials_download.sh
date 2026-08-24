#!/usr/bin/env bash

set -euo pipefail

NCAR_SHARED_DATA_ROOT="${NCAR_SHARED_DATA_ROOT:-/glade/work/emilanese/workshop_2026_data}"

is_ncar_hpc_host() {
    hostname_value=$(hostname -s 2>/dev/null || hostname)
    hostname_value=$(printf '%s' "$hostname_value" | tr '[:upper:]' '[:lower:]')
    case "$hostname_value" in
        dec*|derecho*|crlogin*|crht*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

copy_shared_tutorial_data() {
    shared_crocolake_path="$NCAR_SHARED_DATA_ROOT/CrocoLake"
    shared_tutorial_data_path="$NCAR_SHARED_DATA_ROOT/tutorial_data"

    if [ ! -d "$shared_crocolake_path" ] || [ ! -d "$shared_tutorial_data_path" ]; then
        printf '%s\n' \
            "Warning: NCAR shared tutorial data is unavailable or incomplete at" \
            "         $NCAR_SHARED_DATA_ROOT; falling back to web downloads." >&2
        return 1
    fi

    printf '%s\n' "Using NCAR shared tutorial data from $NCAR_SHARED_DATA_ROOT..."
    mkdir -p "$CROCOLAKE_PATH" "$TUTORIAL_DATA_PATH"
    cp -a "$shared_crocolake_path/." "$CROCOLAKE_PATH/"
    cp -a "$shared_tutorial_data_path/." "$TUTORIAL_DATA_PATH/"
    printf '%s\n' "Local tutorial data setup complete."
    return 0
}

download_tutorial_data_from_web() {
    echo "Downloading datasets for tutorial notebooks..."
    echo "    Downloading CrocoLake - PHY..."
    download_crocolake -t PHY -d CrocoLake --destination "$CROCOLAKE_PATH"
    echo "    downloaded."
    echo "    Downloading tutorials data (this might take a few hours because of zenodo's slow servers)..."
    download_tutorials_data --destination "$TUTORIAL_DATA_PATH"
    echo "    downloaded."
    echo "done."
}

if is_ncar_hpc_host; then
    copy_shared_tutorial_data || download_tutorial_data_from_web
else
    download_tutorial_data_from_web
fi
