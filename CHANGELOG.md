# model2obs Changelog

All notable changes to this project will be documented in this file.

## [0.5.1] - 2026-03-25
- Add support to DART v11.21.2: MOM6 model_state_variables in input.nml now contains 5 values
- Update paths in install files for NCAR install to link to new pre-installed DART v11.21.2 on Casper
- Add reader method to ModelAdapter to dispatch correct parser for DART observation types depending on component (allows for future inclusion of CICE, for example)
- Add supports for parallel processing of model output files
- model_tools methods have been moved to ModelAdapter
	
## [0.5.0] - 2026-03-24
- Add supports for parallel processing of model output files
- Add tutorial 1 parallel version
- Update zenodo link to new tutorial data
- Fix bug where utility method to convert model time to (days, seconds) was reading wrong calendar for MOM6 model output
- Fix some install paths that were not automated
- Add log file per model-obs files processing pair

## [0.4.0] - 2026-03-16
- CrocoCamp is renamed to model2obs
- Tests for interactive widgets are added

## [0.3.0] - 2025-12-22
- Add basic support for ROMS Rutgers model output
- New ModelAdapter class and subclasses handle model-dependent operations (paths and keys validation, xarray dataset opening, units conversion) 

## [0.2.0] - 2025-11-13
- CrocoCamp now runs on any machine (not on NCAR's HPC only)
- `install.sh --tutorial` downloads the data necessary to run the tutorials locally
- Dedicated installer for use on NCAR machines is provided in `install_NCAR.sh`

## [0.1.1] - 2025-10-16
- Bugfix: fixes bug that sometimes prevented the interactive map from being
  updated when changing the plotted variable

## [0.1.0] - 2025-10-10
### Added
- Initial release for Crocodile 2025 workshop at NSF NCAR, 2025-10-13 to 2025-10-17
- Basic functionality: 
  - model-obs comparison with MOM6 models and any obs sequence format observation file
  - interactive maps and plots to visualize model-obs performance
- Tested on NCAR HPC machines Derecho and Casper
- Requires existing installation of DART on the machine

