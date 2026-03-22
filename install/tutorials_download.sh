#!/usr/bin/env bash

set -euo pipefail

echo "Downloading datasets for tutorial notebooks..."
echo "    Downloading CrocoLake - PHY..."
download_crocolake -t PHY -d CrocoLake --destination ../CrocoLake/
echo "    downloaded."
echo "    Downloading tutorials data (this might take a few hours because of zenodo's slow servers)..."
download_tutorials_data --destination ../2025_Crocodile_tutorial/
echo "    downloaded."
echo "done."
