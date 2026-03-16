"""Configuration utilities for CrocoCamp workflows."""

from datetime import timedelta
import os
import re
from typing import Any, Dict, List, Tuple
import yaml


def resolve_path(path: str, relative_to: str = None) -> str:
    """Resolve path to absolute, using relative_to location as base for relative paths."""
    path = os.path.expandvars(path)
    if os.path.isabs(path):
        return os.path.normpath(path)
    if relative_to is None:
        print("Path is not absolute but no base for relative paths was provided, using './'")
        relative_to = "./"
    relative_dir = os.path.dirname(os.path.abspath(relative_to))
    return os.path.normpath(os.path.abspath(os.path.join(relative_dir, path)))

def resolve_config_paths(config: Dict[str,any], relative_to: str = None) -> Dict[str,any]:
    """Resolve paths in config settings."""
    for key in config:
        if isinstance(config[key], str) and key not in ["layer_name","ocean_model"]:
            config[key] = resolve_path(config[key], relative_to)
    return config

def read_config(config_file: str) -> Dict[str, Any]:
    """Read configuration from YAML file."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file '{config_file}' does not exist")

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        config = resolve_config_paths(config, config_file)
        config = convert_time_window(config)
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}") from e

def validate_config_keys(config: Dict[str, Any], required_keys: List[str]) -> None:
    """Validate that all required keys are present in config."""
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Required keys missing from config: {missing_keys}")

def convert_time_window(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate time window for perfect_model_obs"""

    # Assuming 7 in week, 30 days in month, 365 in year
    days_in_week = 7
    days_in_month = 30
    days_in_year = 365

    keys = ["years", "months", "weeks", "days", "hours", "minutes", "seconds"]
    time_window_dict = config.get("time_window", None)
    if time_window_dict is None:
        raise KeyError("No time window has been specified.")

    for key in keys:
        if key not in time_window_dict:
            time_window_dict[key] = 0

    years = time_window_dict["years"]
    months = time_window_dict["months"]
    weeks = time_window_dict["weeks"]
    days = time_window_dict["days"]
    hours = time_window_dict["hours"]
    minutes = time_window_dict["minutes"]
    seconds = time_window_dict["seconds"]

    time_window = timedelta(
        days=days+weeks*days_in_week+months*days_in_month+years*days_in_year,
        hours=hours,
        minutes=minutes,
        seconds=seconds
    )

    config["time_window"] = {}
    config["time_window"]["days"] = time_window.days
    config["time_window"]["seconds"] = time_window.seconds

    return config

def check_directory_not_empty(dir_path: str, name: str) -> None:
    """Check if directory exists and is not empty."""
    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"{name} '{dir_path}' does not exist or is not a directory")

    if not os.listdir(dir_path):
        raise ValueError(f"{name} '{dir_path}' is empty")

def check_nc_files_only(dir_path: str, name: str) -> None:
    """Check if directory contains only .nc files."""
    all_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

    if not all_files:
        raise ValueError(f"{name} '{dir_path}' does not contain any files")

    nc_files = [f for f in all_files if f.endswith('.nc')]

    if len(nc_files) != len(all_files):
        non_nc_files = [f for f in all_files if not f.endswith('.nc')]
        raise ValueError(f"{name} '{dir_path}' contains non-.nc files: {non_nc_files}")

    if not nc_files:
        raise ValueError(f"{name} '{dir_path}' does not contain any .nc files")

def check_nc_file(file_path: str, name: str) -> None:
    """Check if file exists and has .nc extension."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"{name} '{file_path}' does not exist")

    if not file_path.endswith('.nc'):
        raise ValueError(f"{name} '{file_path}' is not a .nc file")

def check_or_create_folder(output_folder: str, name: str) -> None:
    """Check if folder exists, if not, create it."""
    if os.path.exists(output_folder):
        if not os.path.isdir(output_folder):
            raise NotADirectoryError(f"{name} '{output_folder}' exists but is not a directory")
        if os.listdir(output_folder):
            raise ValueError(f"{name} '{output_folder}' exists but is not empty")
    else:
        try:
            os.makedirs(output_folder, exist_ok=True)
        except OSError as e:
            raise OSError(f"Could not create {name} '{output_folder}': {e}") from e

def clear_folder(folder_path: str) -> None:
    """Clear content at folder_path."""
    import shutil  # pylint: disable=import-outside-toplevel

    if not os.path.isdir(folder_path): return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
                print(f'  Deleted file: {file_path}')
        except OSError as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def parse_obs_def_ocean_mod(rst_file_path: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Parse obs_def_ocean_mod.rst file to extract observation type definitions.

    Args:
        rst_file_path: Path to the obs_def_ocean_mod.rst file

    Returns:
        Tuple containing:
        - obs_type_to_qty: Dictionary mapping observation types to their QTY values
        - qty_to_obs_types: Dictionary mapping QTY values to lists of observation types

    Raises:
        FileNotFoundError: If the RST file doesn't exist
        ValueError: If the type definitions section isn't found or is malformed
    """
    if not os.path.isfile(rst_file_path):
        raise FileNotFoundError(f"RST file '{rst_file_path}' does not exist")

    obs_type_to_qty = {}
    qty_to_obs_types = {}

    try:
        with open(rst_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except IOError as e:
        raise IOError(f"Could not read RST file '{rst_file_path}': {e}") from e

    # Find the section between BEGIN and END markers
    begin_pattern = r'! BEGIN DART PREPROCESS TYPE DEFINITIONS'
    end_pattern = r'! END DART PREPROCESS TYPE DEFINITIONS'

    begin_match = re.search(begin_pattern, content)
    end_match = re.search(end_pattern, content)

    if not begin_match or not end_match:
        raise ValueError(f"Could not find type definitions section in '{rst_file_path}'")

    # Extract the section content
    section_content = content[begin_match.end():end_match.start()]

    # Parse each line in the section
    for line in section_content.split('\n'):
        line = line.strip()
        if not line or not line.startswith('!'):
            continue

        # Remove the leading '!' and whitespace
        line = line[1:].strip()
        if not line:
            continue

        # Parse the line format: OBS_TYPE, QTY_TYPE, [COMMON_CODE]
        parts = [part.strip() for part in line.split(',')]
        if len(parts) < 2:
            continue

        obs_type = parts[0].strip()
        qty_type = parts[1].strip()

        if obs_type and qty_type:
            obs_type_to_qty[obs_type] = qty_type

            if qty_type not in qty_to_obs_types:
                qty_to_obs_types[qty_type] = []
            qty_to_obs_types[qty_type].append(obs_type)

    if not obs_type_to_qty:
        raise ValueError(f"No observation type definitions found in '{rst_file_path}'")

    return obs_type_to_qty, qty_to_obs_types

def validate_and_expand_obs_types(obs_types_list: List[str], rst_file_path: str) -> List[str]:
    """Validate and expand observation types list.

    Args:
        obs_types_list: List of observation types from config, may include ALL_<FIELD> entries
        rst_file_path: Path to the obs_def_ocean_mod.rst file

    Returns:
        Expanded list of valid observation types

    Raises:
        ValueError: If any observation type is invalid or expansion fails
    """
    obs_type_to_qty, qty_to_obs_types = parse_obs_def_ocean_mod(rst_file_path)

    expanded_types = set()

    for obs_type in obs_types_list:
        if obs_type.startswith('ALL_'):
            # Extract the field name and create the QTY pattern
            field_name = obs_type[4:]  # Remove 'ALL_' prefix
            qty_pattern = f'QTY_{field_name}'

            if qty_pattern in qty_to_obs_types:
                expanded_types.update(qty_to_obs_types[qty_pattern])
            else:
                raise ValueError(f"No observation types found for '{obs_type}' "
                               f"(looking for {qty_pattern})")
        else:
            # Regular observation type - validate it exists
            if obs_type not in obs_type_to_qty:
                raise ValueError(f"Invalid observation type '{obs_type}'. "
                               f"Must be one of: {list(obs_type_to_qty.keys())}")
            expanded_types.add(obs_type)

    return sorted(list(expanded_types))
