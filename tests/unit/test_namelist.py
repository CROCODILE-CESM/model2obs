"""Unit tests for model2obs.utils.namelist module.

This module tests Fortran namelist file operations for DART integration.
Namelist correctness is CRITICAL as incorrect generation causes silent DART failures.

Tests are organized by method (read, write, symlink, update, format).
Coverage Target: >95% (CRITICAL module)
"""

import os
from pathlib import Path
from typing import Dict, List

import pytest

from model2obs.utils.namelist import Namelist


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def namelist_template_file(fixtures_root: Path) -> Path:
    """Provide path to template namelist file.
    
    Args:
        fixtures_root: Path to fixtures directory
        
    Returns:
        Path to input_template.nml fixture
    """
    return fixtures_root / "namelist_files" / "input_template.nml"


@pytest.fixture
def create_test_namelist(tmp_path: Path):
    """Create a test namelist file for testing.
    
    Returns function that creates namelist file with given content.
    
    Args:
        tmp_path: pytest temporary directory fixture
        
    Returns:
        Function to create namelist file
    """
    def _create_namelist(content: str, filename: str = "test.nml") -> Path:
        """Create namelist file with content.
        
        Args:
            content: Namelist file content
            filename: Name of file to create
            
        Returns:
            Path to created file
        """
        nml_file = tmp_path / filename
        nml_file.write_text(content)
        return nml_file
    
    return _create_namelist


# ============================================================================
# Tests
# ============================================================================

class TestNamelistInit:
    """Test suite for Namelist.__init__() method.
    
    Tests cover:
    - Successful initialization with valid namelist file
    - Backup file creation
    - Content reading
    - Symlink destination setup
    """
    
    def test_init_success(self, namelist_template_file: Path, tmp_path: Path):
        """Test Namelist initialization with valid file.
        
        Given: A valid namelist file
        When: Namelist() is initialized
        Then: Object is created, backup made, content loaded
        """
        original_cwd = os.getcwd()
        os.chdir(tmp_path)  # Change to tmp to avoid polluting real directory
        
        try:
            nml = Namelist(str(namelist_template_file))
            
            assert nml.namelist_path == str(namelist_template_file)
            assert nml.symlink_dest == os.path.join(os.getcwd(), "input.nml")
            assert nml.content is not None
            assert len(nml.content) > 0
            assert "&model_nml" in nml.content
            
            # Check backup was created
            assert os.path.exists("input.nml.backup")
            
        finally:
            os.chdir(original_cwd)
    
    def test_init_missing_file(self, tmp_path: Path):
        """Test Namelist initialization with non-existent file.
        
        Given: A path to non-existent namelist file
        When: Namelist() is initialized
        Then: FileNotFoundError is raised during read_namelist call
        
        Note: The backup copy attempt happens before read_namelist,
              so this may raise IOError from shutil.copy2 first.
        """
        nonexistent_file = tmp_path / "missing.nml"
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            with pytest.raises((FileNotFoundError, IOError)):
                Namelist(str(nonexistent_file))
        finally:
            os.chdir(original_cwd)


class TestReadNamelist:
    """Test suite for read_namelist() method.
    
    Tests cover:
    - Reading valid namelist file
    - Handling missing file
    - Handling read errors
    """
    
    def test_read_namelist_valid(self, create_test_namelist, tmp_path: Path):
        """Test read_namelist reads valid file content.
        
        Given: A valid namelist file
        When: read_namelist() is called
        Then: File content is returned as string
        """
        content = "&test_nml\n   param = 1,\n/"
        nml_file = create_test_namelist(content)
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            nml = Namelist(str(nml_file))
            
            result = nml.read_namelist()
            
            assert result == content
            assert "&test_nml" in result
        finally:
            os.chdir(original_cwd)


class TestWriteNamelist:
    """Test suite for write_namelist() method.
    
    Tests cover:
    - Writing to default path (self.namelist_path)
    - Writing to custom path
    - Writing custom content
    - Handling write errors
    """
    
    def test_write_namelist_default(self, create_test_namelist, tmp_path: Path):
        """Test write_namelist writes to default path.
        
        Given: A Namelist object with content
        When: write_namelist() is called without arguments
        Then: Content is written to self.namelist_path
        """
        original_content = "&test_nml\n   param = 1,\n/"
        nml_file = create_test_namelist(original_content)
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            nml = Namelist(str(nml_file))
            nml.content = "&test_nml\n   param = 2,\n/"
            
            nml.write_namelist()
            
            with open(nml_file, 'r') as f:
                written_content = f.read()
            assert written_content == nml.content
            assert "param = 2" in written_content
        finally:
            os.chdir(original_cwd)
    
    def test_write_namelist_custom_path(self, create_test_namelist, tmp_path: Path):
        """Test write_namelist writes to custom path.
        
        Given: A Namelist object
        When: write_namelist(custom_path) is called
        Then: Content is written to custom path
        """
        original_content = "&test_nml\n   param = 1,\n/"
        nml_file = create_test_namelist(original_content)
        custom_file = tmp_path / "custom.nml"
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            nml = Namelist(str(nml_file))
            
            nml.write_namelist(str(custom_file))
            
            assert custom_file.exists()
            with open(custom_file, 'r') as f:
                written_content = f.read()
            assert written_content == nml.content
        finally:
            os.chdir(original_cwd)
    
    def test_write_namelist_custom_content(self, create_test_namelist, tmp_path: Path):
        """Test write_namelist writes custom content.
        
        Given: A Namelist object
        When: write_namelist(content=custom_content) is called
        Then: Custom content is written (not self.content)
        """
        original_content = "&test_nml\n   param = 1,\n/"
        nml_file = create_test_namelist(original_content)
        custom_content = "&test_nml\n   param = 99,\n/"
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            nml = Namelist(str(nml_file))
            
            nml.write_namelist(content=custom_content)
            
            with open(nml_file, 'r') as f:
                written_content = f.read()
            assert written_content == custom_content
            assert "param = 99" in written_content
        finally:
            os.chdir(original_cwd)


class TestSymlinkToNamelist:
    """Test suite for symlink_to_namelist() method.
    
    Tests cover:
    - Creating symlink to default path
    - Creating symlink to custom path
    - Replacing existing symlink
    - Error when destination is regular file
    - Error when source and dest are same
    """
    
    def test_symlink_to_namelist_default(self, create_test_namelist, tmp_path: Path):
        """Test symlink_to_namelist creates symlink to default path.
        
        Given: A Namelist object
        When: symlink_to_namelist() is called without arguments
        Then: Symlink is created pointing to self.namelist_path
        """
        content = "&test_nml\n   param = 1,\n/"
        nml_file = create_test_namelist(content)
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            nml = Namelist(str(nml_file))
            
            nml.symlink_to_namelist()
            
            assert os.path.islink("input.nml")
            assert os.readlink("input.nml") == str(nml_file)
        finally:
            # Cleanup
            if os.path.islink("input.nml"):
                os.remove("input.nml")
            os.chdir(original_cwd)
    
    def test_symlink_to_namelist_custom_path(self, create_test_namelist, tmp_path: Path):
        """Test symlink_to_namelist creates symlink to custom path.
        
        Given: A Namelist object and a custom namelist file
        When: symlink_to_namelist(custom_path) is called
        Then: Symlink is created pointing to custom path
        """
        content = "&test_nml\n   param = 1,\n/"
        nml_file = create_test_namelist(content)
        custom_file = create_test_namelist(content, "custom.nml")
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            nml = Namelist(str(nml_file))
            
            nml.symlink_to_namelist(str(custom_file))
            
            assert os.path.islink("input.nml")
            assert os.readlink("input.nml") == str(custom_file)
        finally:
            if os.path.islink("input.nml"):
                os.remove("input.nml")
            os.chdir(original_cwd)
    
    def test_symlink_to_namelist_replace_existing(self, create_test_namelist, tmp_path: Path):
        """Test symlink_to_namelist replaces existing symlink.
        
        Given: An existing symlink at destination
        When: symlink_to_namelist() is called
        Then: Old symlink is removed and new one created
        """
        content = "&test_nml\n   param = 1,\n/"
        nml_file = create_test_namelist(content)
        other_file = create_test_namelist(content, "other.nml")
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            nml = Namelist(str(nml_file))
            # Create initial symlink
            os.symlink(str(other_file), "input.nml")
            assert os.readlink("input.nml") == str(other_file)
            
            nml.symlink_to_namelist()
            
            assert os.path.islink("input.nml")
            assert os.readlink("input.nml") == str(nml_file)
        finally:
            if os.path.islink("input.nml"):
                os.remove("input.nml")
            os.chdir(original_cwd)
    
    def test_symlink_to_namelist_dest_is_file(self, create_test_namelist, tmp_path: Path):
        """Test symlink_to_namelist raises error when dest is regular file.
        
        Given: A regular file exists at symlink destination
        When: symlink_to_namelist() is called
        Then: ValueError is raised indicating file exists
        """
        content = "&test_nml\n   param = 1,\n/"
        nml_file = create_test_namelist(content)
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            nml = Namelist(str(nml_file))
            # Create regular file at destination
            with open("input.nml", 'w') as f:
                f.write("existing file")
            
            with pytest.raises(ValueError, match="exists and is not a symlink"):
                nml.symlink_to_namelist()
        finally:
            if os.path.exists("input.nml"):
                os.remove("input.nml")
            os.chdir(original_cwd)
    
    def test_symlink_to_namelist_source_equals_dest(self, tmp_path: Path):
        """Test symlink_to_namelist raises error when source equals dest.
        
        Given: A namelist file at input.nml location
        When: symlink_to_namelist() is called
        Then: ValueError is raised (file exists and is not symlink)
        
        Note: This tests the case where the namelist is already at the
              symlink destination, which is caught by the "exists and is
              not a symlink" check.
        """
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Create file at symlink destination location
            with open("input.nml", 'w') as f:
                f.write("&test_nml\n   param = 1,\n/")
            
            nml = Namelist("input.nml")
            
            with pytest.raises(ValueError, match="exists and is not a symlink"):
                nml.symlink_to_namelist()
        finally:
            if os.path.exists("input.nml"):
                os.remove("input.nml")
            if os.path.exists("input.nml.backup"):
                os.remove("input.nml.backup")
            os.chdir(original_cwd)


class TestCleanupNamelistSymlink:
    """Test suite for cleanup_namelist_symlink() method.
    
    Tests cover:
    - Removing existing symlink
    - Handling non-existent symlink gracefully
    - Not removing regular files
    """
    
    def test_cleanup_symlink_exists(self, create_test_namelist, tmp_path: Path):
        """Test cleanup_namelist_symlink removes existing symlink.
        
        Given: A symlink exists at destination
        When: cleanup_namelist_symlink() is called
        Then: Symlink is removed
        """
        content = "&test_nml\n   param = 1,\n/"
        nml_file = create_test_namelist(content)
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            nml = Namelist(str(nml_file))
            nml.symlink_to_namelist()
            assert os.path.islink("input.nml")
            
            nml.cleanup_namelist_symlink()
            
            assert not os.path.exists("input.nml")
        finally:
            os.chdir(original_cwd)
    
    def test_cleanup_symlink_not_exists(self, create_test_namelist, tmp_path: Path):
        """Test cleanup_namelist_symlink handles missing symlink gracefully.
        
        Given: No symlink exists at destination
        When: cleanup_namelist_symlink() is called
        Then: No error is raised (graceful handling)
        """
        content = "&test_nml\n   param = 1,\n/"
        nml_file = create_test_namelist(content)
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            nml = Namelist(str(nml_file))
            # Don't create symlink
            
            nml.cleanup_namelist_symlink()
        finally:
            os.chdir(original_cwd)
    
    def test_cleanup_symlink_is_file(self, create_test_namelist, tmp_path: Path):
        """Test cleanup_namelist_symlink doesn't remove regular files.
        
        Given: A regular file exists at symlink destination
        When: cleanup_namelist_symlink() is called
        Then: File is not removed (message printed)
        """
        content = "&test_nml\n   param = 1,\n/"
        nml_file = create_test_namelist(content)
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            nml = Namelist(str(nml_file))
            # Create regular file at destination
            with open("input.nml", 'w') as f:
                f.write("regular file")
            
            nml.cleanup_namelist_symlink()
            
            assert os.path.exists("input.nml")
            assert not os.path.islink("input.nml")
        finally:
            if os.path.exists("input.nml") and not os.path.islink("input.nml"):
                os.remove("input.nml")
            os.chdir(original_cwd)


class TestUpdateNamelistParam:
    """Test suite for update_namelist_param() method.
    
    Tests cover:
    - Updating single-value string parameters
    - Updating single-value numeric parameters
    - Updating multi-line dict parameters (triplet format)
    - Updating multi-line list parameters
    - Inserting new parameters
    - Replacing existing block parameters
    """
    
    def test_update_param_single_string(self, create_test_namelist, tmp_path: Path):
        """Test update_namelist_param updates single string parameter.
        
        Given: A namelist with a string parameter
        When: update_namelist_param() is called with new string value
        Then: Parameter is updated in content
        """
        content = """&test_nml
   param1                      = "old_value",
/"""
        nml_file = create_test_namelist(content)
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            nml = Namelist(str(nml_file))
            
            nml.update_namelist_param("test_nml", "param1", "new_value", string=True)
            
            assert '"new_value"' in nml.content
            assert "old_value" not in nml.content
        finally:
            os.chdir(original_cwd)
    
    def test_update_param_single_numeric(self, create_test_namelist, tmp_path: Path):
        """Test update_namelist_param updates single numeric parameter.
        
        Given: A namelist with a numeric parameter
        When: update_namelist_param() is called with new number
        Then: Parameter is updated without quotes
        """
        content = """&test_nml
   param1                      = 1,
/"""
        nml_file = create_test_namelist(content)
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            nml = Namelist(str(nml_file))
            
            nml.update_namelist_param("test_nml", "param1", 42, string=False)
            
            assert "= 42," in nml.content
            assert "param1" in nml.content
        finally:
            os.chdir(original_cwd)

    def test_update_param_bool_true(self, create_test_namelist, tmp_path: Path):
        """Test update_namelist_param writes Python True as Fortran .true.

        Given: A namelist parameter
        When: update_namelist_param() is called with Python True
        Then: Parameter is written as '.true.' (not 'True' or '1')
        """
        content = """&test_nml
   use_pseudo_depth            = .false.,
/"""
        nml_file = create_test_namelist(content)
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            nml = Namelist(str(nml_file))
            nml.update_namelist_param("test_nml", "use_pseudo_depth", True)
            assert "= .true.," in nml.content
            assert "True" not in nml.content
        finally:
            os.chdir(original_cwd)

    def test_update_param_bool_false(self, create_test_namelist, tmp_path: Path):
        """Test update_namelist_param writes Python False as Fortran .false.

        Given: A namelist parameter
        When: update_namelist_param() is called with Python False
        Then: Parameter is written as '.false.' (not 'False' or '0')
        """
        content = """&test_nml
   use_pseudo_depth            = .true.,
/"""
        nml_file = create_test_namelist(content)
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            nml = Namelist(str(nml_file))
            nml.update_namelist_param("test_nml", "use_pseudo_depth", False)
            assert "= .false.," in nml.content
            assert "False" not in nml.content
        finally:
            os.chdir(original_cwd)
    
    def test_update_param_dict_triplet(self, create_test_namelist, tmp_path: Path):
        """Test update_namelist_param updates dict as 3-field triplet block (MOM6 format).
        
        Given: A namelist with a block parameter
        When: update_namelist_param() is called with dict and dict_format='triplet'
        Then: Multi-line 3-field block is created without 'NA' clamp placeholders
        """
        content = """&test_nml
   param1                      = "value1",
/"""
        nml_file = create_test_namelist(content)
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            nml = Namelist(str(nml_file))
            
            nml.update_namelist_param(
                "test_nml", 
                "model_state_variables",
                {"so": "QTY_SALINITY", "thetao": "QTY_POTENTIAL_TEMPERATURE"},
                dict_format='triplet'
            )
            
            assert "model_state_variables" in nml.content
            assert "so" in nml.content
            assert "QTY_SALINITY" in nml.content
            assert "UPDATE" in nml.content
            assert "'NA'" not in nml.content
        finally:
            os.chdir(original_cwd)

    def test_update_param_dict_quintuplet(self, create_test_namelist, tmp_path: Path):
        """Test update_namelist_param updates dict as 5-field quintuplet block (ROMS format).
        
        Given: A namelist with a block parameter
        When: update_namelist_param() is called with dict and dict_format='quintuplet'
        Then: Multi-line 5-field block is created with 'NA' clamp placeholders
        """
        content = """&test_nml
   param1                      = "value1",
/"""
        nml_file = create_test_namelist(content)
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            nml = Namelist(str(nml_file))
            
            nml.update_namelist_param(
                "test_nml",
                "variables",
                {"salt": "QTY_SALINITY", "temp": "QTY_TEMPERATURE"},
                dict_format='quintuplet'
            )
            
            assert "variables" in nml.content
            assert "salt" in nml.content
            assert "QTY_SALINITY" in nml.content
            assert "UPDATE" in nml.content
            assert "'NA'" in nml.content
        finally:
            os.chdir(original_cwd)
    
    def test_update_param_list(self, create_test_namelist, tmp_path: Path):
        """Test update_namelist_param updates list as multi-line block.
        
        Given: A namelist
        When: update_namelist_param() is called with list
        Then: Multi-line list block is created
        """
        content = """&test_nml
   param1                      = "value1",
/"""
        nml_file = create_test_namelist(content)
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            nml = Namelist(str(nml_file))
            
            nml.update_namelist_param(
                "test_nml",
                "obs_types",
                ["FLOAT_TEMPERATURE", "FLOAT_SALINITY"]
            )
            
            assert "obs_types" in nml.content
            assert "FLOAT_TEMPERATURE" in nml.content
            assert "FLOAT_SALINITY" in nml.content
        finally:
            os.chdir(original_cwd)
    
    def test_update_param_insert_new(self, create_test_namelist, tmp_path: Path):
        """Test update_namelist_param inserts new parameter if missing.
        
        Given: A namelist without a specific parameter
        When: update_namelist_param() is called for that parameter
        Then: Parameter is inserted before section terminator (/)
        """
        content = """&test_nml
   param1                      = "value1",
/"""
        nml_file = create_test_namelist(content)
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            nml = Namelist(str(nml_file))
            
            nml.update_namelist_param("test_nml", "new_param", "new_value", string=True)
            
            assert "new_param" in nml.content
            assert '"new_value"' in nml.content
            # Should be before the /
            lines = nml.content.split('\n')
            new_param_idx = next(i for i, line in enumerate(lines) if "new_param" in line)
            terminator_idx = next(i for i, line in enumerate(lines) if line.strip() == '/')
            assert new_param_idx < terminator_idx
        finally:
            os.chdir(original_cwd)

    def test_update_param_float_small_uses_fixed_point(self, create_test_namelist, tmp_path: Path):
        """Test update_namelist_param formats small floats as fixed-point, not scientific notation.

        Given: A namelist with a float parameter
        When: update_namelist_param() is called with a small float like 0.00002 and string=False
        Then: The value is written as '0.00002' not '2e-05'
        """
        content = """&model_nml
   model_perturbation_amplitude = 0.000200,
/"""
        nml_file = create_test_namelist(content)
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            nml = Namelist(str(nml_file))
            nml.update_namelist_param("model_nml", "model_perturbation_amplitude", 0.00002,
                                      string=False)
            assert "e-" not in nml.content
            assert "0.00002" in nml.content
        finally:
            os.chdir(original_cwd)

    def test_update_param_float_zero_uses_six_decimals(self, create_test_namelist, tmp_path: Path):
        """Test update_namelist_param formats 0.0 with six decimal places.

        Given: A namelist with a float parameter
        When: update_namelist_param() is called with 0.0 and string=False
        Then: The value is written as '0.000000' (6-decimal fallback)
        """
        content = """&model_nml
   some_float                   = 1.0,
/"""
        nml_file = create_test_namelist(content)
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            nml = Namelist(str(nml_file))
            nml.update_namelist_param("model_nml", "some_float", 0.0, string=False)
            assert "0.000000" in nml.content
        finally:
            os.chdir(original_cwd)

    def test_update_param_float_large_uses_six_decimals(self, create_test_namelist, tmp_path: Path):
        """Test update_namelist_param formats large floats with six decimal places.

        Given: A namelist with a float parameter
        When: update_namelist_param() is called with 100.0 (positive exponent) and string=False
        Then: The value is written as '100.000000', not scientific notation
        """
        content = """&model_nml
   some_float                   = 1.0,
/"""
        nml_file = create_test_namelist(content)
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            nml = Namelist(str(nml_file))
            nml.update_namelist_param("model_nml", "some_float", 100.0, string=False)
            assert "100.000000" in nml.content
            assert "e+" not in nml.content
        finally:
            os.chdir(original_cwd)

    def test_update_param_float_very_small_precision(self, create_test_namelist, tmp_path: Path):
        """Test update_namelist_param uses enough decimals for very small floats.

        Given: A namelist with a float parameter
        When: update_namelist_param() is called with 1e-10 and string=False
        Then: The value is written with 10 decimal places so no precision is lost
        """
        content = """&model_nml
   some_float                   = 1.0,
/"""
        nml_file = create_test_namelist(content)
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            nml = Namelist(str(nml_file))
            nml.update_namelist_param("model_nml", "some_float", 1e-10, string=False)
            assert "0.0000000001" in nml.content
            assert "e-" not in nml.content
        finally:
            os.chdir(original_cwd)

    def test_update_param_float_inserts_new_param_fixed_point(self, create_test_namelist, tmp_path: Path):
        """Test update_namelist_param inserts a new float param as fixed-point.

        Given: A namelist that does not yet contain a float parameter
        When: update_namelist_param() is called to insert it with string=False
        Then: The new parameter is written in fixed-point notation, not scientific
        """
        content = """&model_nml
   existing_param               = 1,
/"""
        nml_file = create_test_namelist(content)
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            nml = Namelist(str(nml_file))
            nml.update_namelist_param("model_nml", "new_float_param", 0.00002, string=False)
            assert "new_float_param" in nml.content
            assert "e-" not in nml.content
            assert "0.00002" in nml.content
        finally:
            os.chdir(original_cwd)

    def test_update_param_int_unaffected_by_float_formatting(self, create_test_namelist, tmp_path: Path):
        """Test update_namelist_param does not apply float formatting to integers.

        Given: A namelist with an integer parameter
        When: update_namelist_param() is called with an int value and string=False
        Then: The value is written without a decimal point
        """
        content = """&model_nml
   debug                        = 0,
/"""
        nml_file = create_test_namelist(content)
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            nml = Namelist(str(nml_file))
            nml.update_namelist_param("model_nml", "debug", 1, string=False)
            assert "= 1," in nml.content
            assert "1." not in nml.content
        finally:
            os.chdir(original_cwd)


# ============================================================================
# Mark all tests in this module as unit tests
# ============================================================================
pytestmark = pytest.mark.unit
