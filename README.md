# model2obs

**model2obs** is a Python toolset for comparing ocean model outputs and observation datasets. It streamlines workflows for interpolating model data into the observation space, producing tabular data in Parquet format ready for analysis and interactive visualization.

**New in April 2026:** model2obs now supports optional **NetCDF output** for interpolated model values (see [NetCDF Output](#netcdf-output)), and `preview_namelist()` to inspect the generated `input.nml` before running.

**New in March 2026:** model2obs (v0.5.1) now supports **parallel processing** of model output files! See the parallel version of Tutorial 1 for how to use it. It also supports the latest DART v11.21.2, including update scripts to install model2obs on NCAR's Casper HPC. A bug where the wrong calendar was used when converting model time to days, seconds, has also been fixed.   

**New in December 2025:** model2obs (v0.3.0) now supports **ROMS (Regional Ocean Modeling System)** in addition to MOM6, with a flexible model adapter architecture that enables easy extension to other ocean models. The new architecture abstracts model-specific operations (file I/O, unit conversions, configuration requirements) into dedicated adapters, making the codebase more maintainable and extensible.

## Summary

- [Features](#features)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
    - [Observation Types Configuration](#observation-types-configuration)
    - [Time window](#time-window)
- [Architecture](#architecture)
    - [Key Classes and Functions](#key-classes-and-functions)
- [Usage](#usage)
    - [Programmatic Usage (Class-based API, e.g for Jupyter notebooks)](#programmatic-usage-class-based-api-eg-for-jupyter-notebooks)
    - [Previewing the input.nml before running](#previewing-the-inputnml-before-running)
    - [Command Line Interface](#command-line-interface)
- [How to Cite](#how-to-cite)
      
## Features

Current:
- Batch processing of model and observation files
- Generation of diagnostic and comparison files in Parquet format
- Robust YAML configuration with model-specific validation
- Designed for extensibility and reproducibility
- **Ocean models supported: MOM6, ROMS** (via model adapter architecture)
- Ocean observation format supported: DART obs_seq.in format
- Model adapter system for easy extension to new ocean models
- `preview_namelist()` to inspect the generated `input.nml` before running

Future:
- Refined temporal and spatial resampling tools:
    - Customizable time windows independent of model and obs_seq.in files aggregation period.
    - Automated regridding when comparing models to gridded products (e.g. GLORYS)
- Larger support for ROMS:
    - Download of demo data
    - Support for more run-time options (e.g. observations trimming)

## Installation

### Prerequisites and Dependencies

DART (Data Assimilation Research Testbed) is required to run the `perfect_model_obs` executable, which interpolates MOM6 ocean model output onto the observation space provided in obs_seq.in format. In the context of this workshop, DART is already pre-compiled both on Derecho and Casper. If you are interested in the installation on other operating systems or more detailed information, see the [DART documentation](https://docs.dart.ucar.edu/).

### Installation Steps

#### 1. Clone model2obs
```bash
git clone https://github.com/CROCODILE-CESM/model2obs.git
cd model2obs/install
```

#### 2. Configure Environment Paths

Copy the template file and edit it to set your DART installation path and conda environment name:

```bash
cp envpaths.sh.template envpaths.sh
```

Edit `envpaths.sh` to set:
- `DART_ROOT_PATH`: Path to your DART installation (e.g., `/path/to/DART/`)
- `CONDA_ENV_NAME`: Name for your conda environment (e.g., `model2obs`)

**Note for NCAR HPC Users:** You can use the pre-configured `envpaths_NCAR.sh` and `install_NCAR.sh` which are already set up with NCAR-specific paths.

#### 3. Run Installation Script

Create the conda environment and configure paths:

```bash
./install.sh
```

To also download tutorial datasets from Zenodo:
```bash
./install.sh --tutorial
```

#### 4. Activate the Environment

```bash
conda activate model2obs  # or your chosen environment name
```

The installation script will:
- Create a conda environment from `environment.yml`
- Configure the environment to load DART paths automatically when activated
- Set up Python paths for CrocoLake observation converters
- Register a Jupyter kernel for the environment
- Optionally download tutorial datasets (with `--tutorial` flag)

## Getting Started

The best way to learn model2obs is through the hands-on tutorials in the `tutorials/` folder. Demo data is available for download during installation for MOM6 workflows. The Jupyter notebooks guide you through:

**MOM6 - Tutorial 1** (`tutorial1_MOM6-CL-comparison.ipynb`): 
- Setting up a basic model-observation comparison workflow
- Using MOM6 ocean model output and CrocoLake observations
- Visualizing results with the interactive map widget

**MOM6 - Tutorial 2** (`tutorial2_MOM6-CL-comparison-float.ipynb`):
- Generating custom observation files from CrocoLake 
- Analyzing single Argo float profiles
- Using the interactive profile widget for vertical profile comparisons
- Passing custom configurations to profile and map widgets

**ROMS - Tutorial 1** (`ROMS_tutorial_1-CL-comparison.ipynb`):
- Setting up a basic model-observation comparison workflow
- Generating custom observation files from CrocoLake 
- Using ROMS Rutgers ocean model output and CrocoLake observations
- Using the interactive profile and interactive map widgets


These tutorials demonstrate:
- Loading and configuring workflows with `WorkflowModelObs`
- Running the complete processing pipeline
- Exploring results including diagnostic values such as:
  - `residual` (obs - model)
  - `abs_residual` (absolute residual)
  - `normalized_residual` (residual normalized by observation error)
  - `squared_residual` (squared residual)
  - `log_likelihood` (log-likelihood of model-observation fit)

## Configuration

Edit the provided `configs/config_template.yaml` to set your input, output, and model/obs paths. The template file contains all necessary configuration options with detailed comments.

**Important:** You must specify the `ocean_model` field in your configuration file to select the appropriate model adapter:
```yaml
ocean_model: MOM6  # or ROMS
```

Different ocean models may require different configuration keys. For example:
- **MOM6** requires: `model_files_folder`, `perfect_model_obs_dir`, and standard MOM6 grid files
- **ROMS** requires: `roms_filename`, `layer_name`, `model_state_variables`, and ROMS-specific grid files

**Note for NCAR HPC Users:** The paths in the provided configuration files and some paths used in the tutorial notebooks are pre-configured for resources available on NCAR's High Performance Computing systems, including:
- CrocoLake observation dataset paths
- DART tools paths 
- Pre-compiled `perfect_model_obs` executable locations
These paths need to be adjusted if running on other systems.
Note that DART needs to be compiled separately on Derecho and Casper, so we provide two pre-compiled paths for the workshop:
- Derecho: `/glade/u/home/emilanese/work/CROCODILE-DART-fork/models/MOM6/work`
- Casper: `/glade/u/home/emilanese/work/DART-Casper/models/MOM6/work`

### Observation Types Configuration

model2obs supports automatic configuration of [observation types for DART]([url](https://github.com/NCAR/DART/blob/main/observations/forward_operators/obs_def_ocean_mod.rst)) assimilation through the `use_these_obs` field in your configuration file.

#### Using specific observation types

You need to specify at least one observation type to use, and that needs to be in your observation sequence files header.

Specify your desired observation types in the `use_these_obs` field in `config.yaml`, for example:

```yaml
# Basic observation types
use_these_obs:
  - FLOAT_TEMPERATURE
  - FLOAT_SALINITY
  - CTD_TEMPERATURE
  - CTD_SALINITY
```

This would interpolate MOM6 model results to all observations marked with the same verbatim type in the provided observation sequence files.

#### ALL_<FIELD> Syntax

You can use the `ALL_<FIELD>` syntax to automatically include all observation types for a specific quantity, for example:

```yaml
use_these_obs:
  - ALL_TEMPERATURE    # Includes all temperature-related obs types
  - ALL_SALINITY       # Includes all salinity-related obs types  
```

In this case, model2obs builds a dictionary of all the supported temperature and salinity observation types in DART, and will look for all of those types in the observation sequence files you provided.

### Time window

The `time_window` field in the configuration file determines the temporal range over which model and observation sequence files are matched. For example, for a model file that contains daily averages for the date 2025-10-01 12:00 and observation sequence file that contains observations between 2025-09-30 10:00 and 2025-10-01 23:00, the time window must be 52 hours or larger to interpolate the model onto the observations in the observation file. The time window is centered on the model date, so 26 hours on each side would span the time between 2025-09-30 10:00 and 2025-10-02 14:00, including all the observations. Note that it's an 'all or nothing' approach: if only one observation is not inside the time window, the whole file is skipped and no observation is used.

In pseudo code:
```
tw                 # provided time window
tm                 # model date
to_1               # minimum observation time in observation sequence file
to_2               # minimum observation time in observation sequence file
tm_1 = tm - tw/2   # lower bound for matching
tm_2 = tm + tw/2   # higher bound for matching
if (tm_1 <= to_1 <= tm_2) and (tm_1 <= to_2 <= tm_2):
    interpolate the model onto this observation sequence file
else:
    skip and check next observation sequence file (if any)
```

### NetCDF Output

#### Enabling NetCDF Output

Set the `interpolate_only` flag to `true` in your configuration file to enable NetCDF generation:

```yaml
interpolate_only: true
netcdf_output_folder: "netcdf_output"  # optional, defaults to "netcdf_output"
```

When enabled, one NetCDF file is created per model-observation pair (parallel to the Parquet workflow): `model-obs-0000.nc`, `model-obs-0001.nc`, etc.

#### NetCDF File Structure

Each NetCDF file uses a 4D gridded structure following CF conventions:

**Dimensions:**
- `time`: Observation times (seconds since 1601-01-01 00:00:00)
- `depth`: Observation depths in meters below sea surface (positive down)
- `latitude`: Observation latitudes (degrees North)
- `longitude`: Observation longitudes (degrees East)

**Data Variables:**
- `interpolated_model_<OBS_KIND>(time, depth, latitude, longitude)`: Interpolated model value of OBS_KIND at each observation location
- `qc_flag_<OBS_KIND>(time, depth, latitude, longitude)`: DART quality control flag for each interpolation of OBS_KIND

**Sparse Grid:** Not all grid points contain observations. Missing points are filled with `NaN` (for `interpolated_model`) or `-999` (for `qc_flag`).

**CF Compliance:** Proper `units`, `standard_name`, and coordinate attributes follow CF-1.8 conventions.

#### Coordinate Tolerance (Optional)

To reduce file size and dimension cardinality, nearby observation locations can be merged using coordinate tolerances:

```yaml
netcdf_coord_tolerance:
  longitude: 0.01  # degrees (default: 1e-2)
  latitude: 0.01   # degrees (default: 1e-2)
  depth: 0.1       # meters (default: 1e-1)
```

For example, with `latitude: 0.01`, observations at lat=30.003° and lat=30.005° are both assigned to lat=30.00°, reducing the number of unique latitude values in the NetCDF grid.

#### Complete Configuration Example

```yaml
# Required: Enable NetCDF output
interpolate_only: true

# Optional: Output folder (defaults to "netcdf_output")
netcdf_output_folder: "my_netcdf_results"

# Optional: Coordinate tolerances (defaults shown)
netcdf_coord_tolerance:
  longitude: 1.0e-2  # 0.01 degrees (~1 km at equator)
  latitude: 1.0e-2   # 0.01 degrees (~1 km)
  depth: 1.0e-1      # 0.1 meters
```

#### Notes

- NetCDF output is **supplementary** to the Parquet format. The Parquet files remain the primary output for downstream analysis.
- Set `interpolate_only: false` (or omit the parameter) to disable NetCDF generation.

## Architecture

The toolkit is organized into logical modules:

- **`utils/`** - Configuration and namelist file utilities
- **`io/`** - File handling and observation sequence processing
- **`workflows/`** - High-level workflow orchestration
- **`model_adapter/`** - Model-specific adapters for MOM6, ROMS, and future models
- **`cli/`** - Command-line interfaces
- **`viz/`** - Interactive visualization widgets for data analysis

### Key Classes and Functions

#### Workflow Classes

**`WorkflowModelObs`** - Main workflow class for model-observation comparisons
- `from_config_file(config_file)` - Create workflow from YAML configuration file
- `run()` - Execute complete workflow 
- `process_files()` - Process model and observation files
- `merge_model_obs_to_parquet()` - Convert results to parquet format
- `preview_namelist(filename=None)` - Inspect the generated `input.nml` (print to screen, or save to file)
- `get_config(key)` - Get configuration value
- `set_config(key, value)` - Set configuration value
- `print_config()` - Print current configuration

#### Model Adapter Classes

**`ModelAdapter`** - Abstract base class for model-specific operations
- Defines interface for handling different ocean model formats
- Model-specific subclasses: `ModelAdapterMOM6`, `ModelAdapterROMS`

**Key adapter features:**
- Automatic model detection from `ocean_model` configuration field
- Model-specific file I/O with correct time decoding and calendar handling
- Unit conversion (e.g., salinity units differ between MOM6 and ROMS)
- Validation of compatible workflow run options per model
- Configuration key requirements specific to each model

#### Visualization Classes

Both widgets below support both pandas and dask DataFrames.

**`InteractiveWidgetMap`** - Interactive map widget for spatial data visualization
- Constructor: `InteractiveWidgetMap(dataframe, config=None)`
- `setup()` - Initialize and display the interactive map widget
- Provides dropdowns for selecting plot variables, observation types, and time filtering

**`MapConfig`** - Configuration class for map widget customization
- Parameters: `colormap`, `figure_size`, `scatter_size`, `map_extent`, etc.

**`InteractiveWidgetProfile`** - Interactive profile widget for vertical profile analysis  
- Constructor: `InteractiveWidgetProfile(dataframe, x='obs', y='vertical', config=None)`
- `setup()` - Initialize and display the interactive profile widget
- Supports custom x and y axis selections for profile analysis
- Ideal for analyzing Argo float or CTD profile comparisons

**`ProfileConfig`** - Configuration class for profile widget customization  
- Parameters: `figure_size`, `marker_size`, `invert_yaxis`, etc.

## Usage

### Programmatic Usage (Class-based API, e.g for Jupyter notebooks)

For Python scripts and Jupyter notebooks, use the class-based API:

```python
from model2obs.workflows import WorkflowModelObs

# Load workflow from configuration file
workflow = WorkflowModelObs.from_config_file("config.yaml")

# Or create workflow with config dictionary directly in code
# Example for MOM6:
config_mom6 = {
    'ocean_model': 'MOM6',
    'model_files_folder': '/path/to/model/files',
    'obs_seq_in_folder': '/path/to/obs_seq_in/files', 
    'output_folder': '/path/to/output',
    'template_file': '/path/to/template.nc',
    'static_file': '/path/to/static.nc',
    'ocean_geometry': '/path/to/geometry.nc',
    'perfect_model_obs_dir': '/path/to/perfect_model_obs',
    'parquet_folder': '/path/to/parquet'
}
workflow = WorkflowModelObs(config_mom6)

# Example for ROMS:
config_roms = {
    'ocean_model': 'ROMS',
    'roms_filename': '/path/to/roms_avg.nc',
    'obs_seq_in_folder': '/path/to/obs_seq_in/files',
    'output_folder': '/path/to/output',
    'template_file': '/path/to/template.nc',
    'static_file': '/path/to/static.nc',
    'ocean_geometry': '/path/to/geometry.nc',
    'layer_name': 's_rho',
    'model_state_variables': {'temp': 'QTY_POTENTIAL_TEMPERATURE', 'salt': 'QTY_SALINITY'},
    'parquet_folder': '/path/to/parquet'
}
workflow = WorkflowModelObs(config_roms)

# Run the complete workflow
files_processed = workflow.run(trim_obs=True, no_matching=False)

# Or run specific steps
files_processed = workflow.process_files(trim_obs=True)
workflow.merge_model_obs_to_parquet(trim_obs=True)
```

You can also override configuration values:

```python
# Override config values when creating workflow
workflow = WorkflowModelObs.from_config_file(
    "config.yaml", 
    output_folder="/custom/output/path",
    trim_obs=True
)

# Or modify configuration after creation
workflow.set_config("parquet_folder", "/custom/parquet/path")

# Access configuration values
output_folder = workflow.get_config("output_folder") 
workflow.print_config()  # Print all current configuration

# Get required configuration keys for validation
required_keys = workflow.get_required_config_keys()

workflow.run()
```

### Previewing the input.nml before running

`preview_namelist()` lets you inspect the `input.nml` that would be passed to `perfect_model_obs`
without executing the full workflow. This is useful for debugging configurations and verifying
that model-specific keys are populated correctly.

```python
from model2obs.workflows import WorkflowModelObs

workflow = WorkflowModelObs.from_config_file("config.yaml")

# Print the namelist to screen
workflow.preview_namelist()

# Or save it to a file for further inspection
workflow.preview_namelist("my_input.nml")
```

The method lazily initialises the namelist on first call, so it is safe to call before `run()`.

### Command Line Interface

Process model-observation pairs using the main CLI:

```bash
# Basic usage
perfect-model-obs -c config.yaml

# With observation trimming to model grid boundaries
perfect-model-obs -c config.yaml --trim

# Skip time matching (assumes 1:1 file correspondence)
perfect-model-obs -c config.yaml --no-matching

# Convert existing outputs to parquet only
perfect-model-obs -c config.yaml --parquet-only
```

### Tests

#### Run main tests

To test your installation of model2obs, run:

```bash
pytest tests/
```

Note that this executes all tests except the thorough sanity test that compares
the parquet output of the serial and parallel workflows when these are called on
the same real model output (stored as one file for the serial workflow and two
files for the parallel workflow). The thorough test is exclude from the regular
tests because it is both time and resource consuming (it requires an HPC setup)
and depends on the tutorial data (which are 60+ GB unzipped). The equivalence
between the two workflows is also tested by `pytest tests/` but not on real
data: while that should be enough, the thorough test is provided in case
something goes south at some point in the development and/or if there is the
suspicion that the general tests are not testing the workflows properly.

#### Run the extra test

The end-to-end thorough sanity test verifies that serial and parallel workflow runs
produce identical parquet output.  

**Prerequisites:**

1. Set the required environment variables:
   ```bash
   export TUTORIAL_DATA_PATH=/path/to/tutorial/data
   export DART_ROOT_PATH=/path/to/DART
   ```

2. Download the tutorial datasets (if not already present):
   ```bash
   download_tutorials_data --destination $TUTORIAL_DATA_PATH
   ```

3. Ensure DART is compiled for MOM6 (`$DART_ROOT_PATH/models/MOM6/work/perfect_model_obs` must exist).

**Run the sanity test:**

```bash
# Recommended: show live workflow output as the test runs
pytest tests/sanity/ -s -v

# Minimal output (just pass/fail)
pytest tests/sanity/
```

The `-s` flag disables output capture so workflow progress prints to the terminal;
`-v` shows the full test name and result.

The test runs the serial workflow from `tutorials/config_tutorial_1.yaml` and the
parallel workflow from `tutorials/config_tutorial_1_parallel.yaml`, then asserts
that every row in both parquet outputs is identical (order-independent).

**Failure diagnostics:**

If the parity assertion fails, three diagnostic steps run automatically and print
structured information to stdout (visible when using `-s`):

1. **obs_seq.out comparison** – Each `obs_seq_NNNN.out` file is loaded and compared
   pair-by-pair.  Results are reported as `MATCH`, `MISMATCH (N rows differ)`,
   `EXTRA` (only in one workflow), or `MISSING`.

2. **Model input equivalence** – The serial single-file `in_mom6/` dataset is
   compared with the concatenated `in_mom6_par/` multi-file dataset using xarray.
   Any differing variables are listed.

3. **Failure statistics** – Rows that differ between the two parquet tables are
   summarised by observation type, day, and QC code.

> **Note:** The parquet output does not record a thread-number column, so
> per-thread attribution of differing rows is not available from the parquet alone.
> Per-file logs in the parallel `output_folder` provide additional context.

## How to Cite

If you use model2obs in your research, please cite it as:

Milanese, E. (2025). model2obs. Zenodo. https://doi.org/10.5281/zenodo.17336480

For reproducibility and traceability, it is recommended that in your work you also specify the version you used (e.g. v0.2.0); each version also has a unique doi in Zenodo.

### BibTeX Entry

```bibtex
@software{model2obs,
  author       = {Milanese, Enrico},
  title        = {model2obs},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17336480},
  url          = {https://doi.org/10.5281/zenodo.17336480}
}
```

---

**For any questions, please open an issue.**
