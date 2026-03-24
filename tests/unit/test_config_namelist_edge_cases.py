"""Additional tests for config and namelist edge cases to increase coverage.

Tests focus on error paths and edge cases that are difficult to trigger
in normal usage but important for robustness.
"""

import os
from pathlib import Path
import pytest

from model2obs.utils import config
from model2obs.utils.namelist import Namelist


class TestConfigEdgeCases:
    """Additional config tests for missing coverage lines."""
    
    def test_check_nc_files_only_no_nc_files(self, tmp_path):
        """Test check_nc_files_only raises ValueError when no .nc files present."""
        folder = tmp_path / "empty_nc"
        folder.mkdir()
        (folder / "data.txt").write_text("not a netcdf")
        
        with pytest.raises(ValueError, match="contains non-.nc files"):
            config.check_nc_files_only(str(folder), "test_folder")
    
    def test_check_or_create_folder_mkdir_error(self, tmp_path, monkeypatch):
        """Test check_or_create_folder handles OSError during mkdir."""
        target = tmp_path / "new_folder"
        
        original_makedirs = os.makedirs
        
        def mock_makedirs(path, *args, **kwargs):
            if str(path) == str(target):
                raise OSError("Permission denied")
            return original_makedirs(path, *args, **kwargs)
        
        monkeypatch.setattr(os, "makedirs", mock_makedirs)
        
        with pytest.raises(OSError, match="Could not create"):
            config.check_or_create_folder(str(target), "test_folder")
    

class TestNamelistEdgeCases:
    """Additional namelist tests for missing coverage lines."""
    
    def test_read_namelist_missing_file(self, tmp_path):
        """Test read_namelist raises FileNotFoundError for missing file."""
        nml_path = tmp_path / "missing.nml"
        
        with pytest.raises(FileNotFoundError):
            nml = Namelist(str(nml_path))
    
    def test_read_namelist_ioerror(self, tmp_path, monkeypatch):
        """Test read_namelist handles IOError during file reading."""
        nml_path = tmp_path / "test.nml"
        nml_path.write_text("&test\n/")
        
        original_open = open
        call_count = [0]
        
        def mock_open(path, *args, **kwargs):
            call_count[0] += 1
            if 'test.nml' in str(path) and 'r' in str(args) and call_count[0] > 1:
                raise IOError("Cannot read file")
            return original_open(path, *args, **kwargs)
        
        monkeypatch.setattr("builtins.open", mock_open)
        
        with pytest.raises(IOError, match="Could not read namelist file"):
            nml = Namelist(str(nml_path))
            nml.read_namelist()
    
    def test_write_namelist_ioerror(self, tmp_path, monkeypatch):
        """Test write_namelist handles IOError during file writing."""
        nml_path = tmp_path / "test.nml"
        nml_path.write_text("&test\n/")
        nml = Namelist(str(nml_path))
        
        original_open = open
        
        def mock_open(path, mode='r', *args, **kwargs):
            if 'w' in mode and 'test.nml' in str(path):
                raise IOError("Cannot write file")
            return original_open(path, mode, *args, **kwargs)
        
        monkeypatch.setattr("builtins.open", mock_open)
        
        with pytest.raises(IOError, match="Could not write namelist file"):
            nml.write_namelist()
    
    def test_symlink_to_namelist_same_source_dest(self, tmp_path, monkeypatch):
        """Test symlink_to_namelist raises ValueError when source equals dest."""
        nml_path = tmp_path / "test.nml"
        nml_path.write_text("&test\n/")

        monkeypatch.chdir(tmp_path)
        nml = Namelist(str(nml_path))

        # Force symlink_dest to equal the namelist path to trigger the same-path check
        nml.symlink_dest = str(nml_path)

        with pytest.raises(ValueError, match="Source and destination.*are the same"):
            nml.symlink_to_namelist()
    
    def test_format_block_param_dict_pair_format(self, tmp_path):
        """Test _format_namelist_block_param with pair format."""
        nml_path = tmp_path / "test.nml"
        nml_path.write_text("&test_nml\n/")
        nml = Namelist(str(nml_path))
        
        test_dict = {'key1': 'val1', 'key2': 'val2'}
        lines = nml._format_namelist_block_param('test_param', test_dict, dict_format='pair')
        
        assert len(lines) == 2
        assert "'key1'" in lines[0]
        assert "'val1'" in lines[0]
        assert 'UPDATE' not in lines[0]
    
    def test_format_block_param_list(self, tmp_path):
        """Test _format_namelist_block_param with list values."""
        nml_path = tmp_path / "test.nml"
        nml_path.write_text("&test_nml\n/")
        nml = Namelist(str(nml_path))
        
        test_list = ['val1', 'val2', 'val3']
        lines = nml._format_namelist_block_param('test_param', test_list)
        
        assert len(lines) == 3
        assert "'val1'" in lines[0]
    
    def test_replace_block_parameter_multi_line(self, tmp_path):
        """Test _replace_block_parameter replaces multi-line blocks."""
        nml_content = """&test_section
   block_param = 'key1', 'val1', 'UPDATE',
                 'key2', 'val2', 'UPDATE',
   other_param = 1
/"""
        nml_path = tmp_path / "test.nml"
        nml_path.write_text(nml_content)
        nml = Namelist(str(nml_path))

        lines = nml.content.split('\n')
        new_dict = {'newkey': 'newval'}
        result = nml._replace_block_parameter(lines, 1, 'block_param', new_dict, 'triplet')
        
        assert result is True
        assert "'newkey '" in '\n'.join(lines)


pytestmark = pytest.mark.unit
