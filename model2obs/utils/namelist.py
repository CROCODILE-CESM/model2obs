"""Namelist file utilities for CrocoCamp workflows."""

import os
import re
import shutil
import tempfile
from typing import Any, Dict, List, Union

class Namelist():
    """Class to handle file operations related to perfect_model_obs input.nml
    namelist file.

    It includes methods to read, write, and update parameters, as well as
    generating necessary symlink for perfect_model_obs to execute correctly.
    
    The update_namelist_param method supports both single-value parameters and
    multi-line block parameters (from dict or list inputs).
    """

    def __init__(self, namelist_path: str) -> None:
        """Initialize Namelist with path to namelist file.

        Arguments:
        namelist_path: Path to the namelist file
        """

        print("    Setting up symlink for input.nml...")
        self.namelist_path = namelist_path
        self.symlink_dest = os.path.join(os.getcwd(), "input.nml")

        # Create backup and read namelist
        shutil.copy2(self.namelist_path, "input.nml.backup")
        print("    Created backup: input.nml.backup")

        self.content = self.read_namelist()

    def read_namelist(self) -> str:
        """Read namelist file and return as string."""
        if not os.path.isfile(self.namelist_path):
            raise FileNotFoundError(f"Namelist file '{self.namelist_path}' does not exist")

        try:
            with open(self.namelist_path, 'r') as f:
                return f.read()
        except IOError as e:
            raise IOError(f"Could not read namelist file '{self.namelist_path}': {e}")

    def write_namelist(self, namelist_path : str = None, content : str = None) -> None:
        """Write content to namelist file."""
        if namelist_path is None:
            namelist_path = self.namelist_path
        if content is None:
            content = self.content

        try:
            with open(namelist_path, 'w') as f:
                f.write(content)
        except IOError as e:
            raise IOError(f"Could not write namelist file '{namelist_path}': {e}")

    def symlink_to_namelist(self, namelist_path : str = None) -> None:
        """Create a symbolic link to a namelist file."""
        if namelist_path is None:
            if not os.path.isfile(self.namelist_path):
                raise FileNotFoundError(f"Source namelist file '{self.namelist_path}' does not exist")
            else:
                namelist_path = self.namelist_path

        try:

            if self.symlink_dest == namelist_path:
                raise ValueError("Source and destination for symlink are the same.")
            if os.path.islink(self.symlink_dest):
                os.remove(self.symlink_dest)
                print(f"          Symlink '{self.symlink_dest}' removed.")
            elif os.path.exists(self.symlink_dest):
                raise ValueError(f"'{self.symlink_dest}' exists and is not a symlink. Not removing nor continuing execution.")
            os.symlink(namelist_path, self.symlink_dest)
            print(f"          Symlink {self.symlink_dest} -> '{namelist_path}' created.")
        except OSError as e:
            raise OSError(f"Could not create symlink from "
                         f"'{namelist_path}' to '{self.symlink_dest}': {e}")

    def cleanup_namelist_symlink(self) -> None:
        """Remove the symbolic link to the namelist file."""
        try:
            if os.path.islink(self.symlink_dest):
                os.remove(self.symlink_dest)
                print(f"          Symlink '{self.symlink_dest}' removed.")
            elif os.path.exists(self.symlink_dest):
                print(f"          '{self.symlink_dest}' exists but is not a symlink. Not removing.")
            else:
                print(f"          No symlink '{self.symlink_dest}' found to remove.")
        except OSError as e:
            raise OSError(f"Could not remove symlink '{self.symlink_dest}': {e}")

    def _format_namelist_block_param(self, param: str, value: Union[Dict, List], dict_format: str = 'triplet') -> List[str]:
        """Format a dict or list value as multi-line namelist block.
        
        Arguments:
        param: Parameter name
        value: Dictionary or list to format as multi-line block
        dict_format: Format for dict values - 'triplet' adds 'UPDATE' as third element,
                    'pair' uses just key-value pairs
        
        Returns:
        List of formatted lines for the block
        """
        lines = []
        
        if isinstance(value, dict):
            items = list(value.items())
            if items:
                # First line with parameter name  
                key, val = items[0]
                if dict_format == 'triplet':
                    # Format as triplets: 'key', 'value', 'UPDATE'
                    padded_val = f"{val}".ljust(25)  # Match the padding from original
                    first_line = f"   {param.ljust(27)}= '{key} ', '{padded_val}', 'NA', 'NA', 'UPDATE',"
                else:
                    # Format as key-value pairs: 'key', 'value'
                    first_line = f"   {param.ljust(27)}= '{key}', '{val}',"
                lines.append(first_line)
                
                # Subsequent lines aligned with the first quote after '='
                # The alignment should be: 3 spaces + param width + "= " + alignment to first quote
                indent = " " * (3 + 27 + 2)  # 3 + param_width + len("= ")
                for key, val in items[1:]:
                    if dict_format == 'triplet':
                        padded_val = f"{val}".ljust(25)
                        line = f"{indent}'{key} ', '{padded_val}', 'NA', 'NA', 'UPDATE',"
                    else:
                        line = f"{indent}'{key}', '{val}',"
                    lines.append(line)
                    
        elif isinstance(value, list):
            # Format as simple list of values
            if value:
                # First line with parameter name
                first_val = value[0]
                first_line = f"   {param.ljust(27)}= '{first_val}'"
                lines.append(first_line)
                
                # Subsequent lines aligned with the first quote
                indent = " " * (3 + 27 + 2)  # Same alignment as dict case
                for val in value[1:]:
                    line = f"{indent}'{val}'"
                    lines.append(line)
                    
        return lines

    def update_namelist_param(self, section: str, param: str, value: Union[str, int, float, bool, Dict, List], string: bool = True, dict_format: str = 'triplet') -> None:
        """Update a parameter in a namelist section.

        Supports both single-value parameters and multi-line block parameters.
        
        Arguments:
        section: Namelist section (without initial '&')
        param: Parameter name to update
        value: New value for the parameter. Can be:
            - Scalar (str, int, float, bool) for single-line parameters
            - Dict for multi-line blocks formatted as triplets (key, value, 'UPDATE') or pairs
            - List for multi-line blocks formatted as simple values
        string: Whether scalar values should be quoted (True) or not (False)
                (default: True). Ignored for dict/list values.
        dict_format: Format for dict values - 'triplet' (default) adds 'UPDATE' as third element,
                    'pair' uses just key-value pairs
        
        Examples:
        # Single-line parameter
        update_namelist_param('model_nml', 'assimilation_period_days', 5, string=False)
        
        # Multi-line block from dict (formatted as triplets) - for model_state_variables
        update_namelist_param('model_nml', 'model_state_variables', 
                             {'so': 'QTY_SALINITY', 'thetao': 'QTY_POTENTIAL_TEMPERATURE'})
        
        # Multi-line block from dict (formatted as pairs) - for other dict parameters
        update_namelist_param('some_nml', 'some_param', 
                             {'key1': 'val1', 'key2': 'val2'}, dict_format='pair')
        
        # Multi-line block from list
        update_namelist_param('obs_kind_nml', 'assimilate_these_obs_types',
                             ['FLOAT_SALINITY', 'FLOAT_TEMPERATURE'])
        """

        # Detect if this is a multi-line block parameter
        is_block_param = isinstance(value, (dict, list))
        
        section_pattern = f'&{section}'
        lines = self.content.split('\n')
        in_section = False
        updated = False
        section_end_idx = None

        # Find the section and handle parameter update/insertion
        for j, line in enumerate(lines):
            if line.strip().startswith(section_pattern):
                in_section = True
                continue

            if in_section and line.strip().startswith('&') and not line.strip().startswith(section_pattern):
                in_section = False
                continue
                
            if in_section and line.strip() == '/':
                section_end_idx = j
                break

            if in_section:
                param_pattern = rf'^\s*{re.escape(param)}\s*='
                if re.match(param_pattern, line):
                    if is_block_param:
                        # Handle multi-line block replacement
                        updated = self._replace_block_parameter(lines, j, param, value, dict_format)
                    else:
                        # Handle single-line parameter replacement
                        if string:
                            lines[j] = f'   {param.ljust(27)}= "{value}",'
                        else:
                            lines[j] = f'   {param.ljust(27)}= {value},'
                        updated = True
                    break

        # If parameter not found, insert it
        if not updated:
            if section_end_idx is not None:
                if is_block_param:
                    # Insert multi-line block before section terminator (/)
                    new_lines = self._format_namelist_block_param(param, value, dict_format)
                    lines[section_end_idx:section_end_idx] = new_lines
                else:
                    # Insert single-line parameter
                    if string:
                        new_line = f'   {param.ljust(27)}= "{value}",'
                    else:
                        new_line = f'   {param.ljust(27)}= {value},'
                    lines.insert(section_end_idx, new_line)
                updated = True
            else:
                raise ValueError(f"Section '&{section}' not found or malformed")
        
        if not updated:
            raise ValueError(f"Parameter '{param}' not found in section '&{section}' and could not be inserted")

        self.content = '\n'.join(lines)

    def _replace_block_parameter(self, lines: List[str], start_idx: int, param: str, value: Union[Dict, List], dict_format: str = 'triplet') -> bool:
        """Replace an existing multi-line block parameter.
        
        Correctly handles blocks of different sizes - the new block can have more,
        fewer, or the same number of lines as the old block.
        
        Arguments:
        lines: List of lines from the namelist content
        start_idx: Index of the line where the parameter starts
        param: Parameter name
        value: New value for the parameter
        dict_format: Format for dict values ('triplet' or 'pair')
        
        Returns:
        True if replacement was successful
        """
        # Find the end of the current block
        end_idx = start_idx
        param_pattern = rf'^\s*{re.escape(param)}\s*='
        
        # Find all continuation lines for this parameter
        for i in range(start_idx + 1, len(lines)):
            line = lines[i].strip()
            # Stop if we hit another parameter, section end, or new section
            if (line.startswith('/') or 
                line.startswith('&') or 
                (line and not line.startswith(('\'', '"')) and '=' in line and not re.match(param_pattern, lines[i]))):
                break
            # If this looks like a continuation line (indented, starts with quote, or is empty)
            if (lines[i].startswith((' ', '\t')) or 
                line.startswith(('\'', '"')) or 
                not line):
                end_idx = i
            else:
                break
                
        # Generate new block lines
        new_lines = self._format_namelist_block_param(param, value, dict_format)
        
        # Replace the old block with the new one (intentionally replaces existing lines)
        # Python slice assignment automatically handles blocks of different sizes:
        # - If new block is shorter: removes extra old lines
        # - If new block is longer: extends the list with additional lines
        # - If same size: replaces line-by-line
        lines[start_idx:end_idx + 1] = new_lines
        
        return True
