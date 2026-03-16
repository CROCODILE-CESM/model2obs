"""Unit tests for CLI perfect_model_obs interface.

Tests the command-line interface for perfect model observation processing,
including argument parsing, config loading, and workflow execution.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
from io import StringIO

from model2obs.cli import perfect_model_obs


class TestArgumentParsing:
    """Test command-line argument parsing."""
    
    def test_parse_config_argument(self, tmp_path):
        """Test parsing with config file argument."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("model_files_folder: /path\n")
        
        test_args = ['perfect_model_obs.py', '-c', str(config_file)]
        
        with patch('sys.argv', test_args):
            with patch.object(perfect_model_obs.WorkflowModelObs, 'from_config_file') as mock_from_config:
                mock_instance = Mock()
                mock_instance.get_config.return_value = '/some/path'
                mock_instance.run.return_value = 3
                mock_from_config.return_value = mock_instance
                
                perfect_model_obs.main()
                
                # Verify from_config_file was called with correct file
                mock_from_config.assert_called_once_with(str(config_file))
    
    def test_parse_trim_flag(self, tmp_path):
        """Test parsing --trim flag."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model_files_folder: /path\n")
        
        test_args = ['perfect_model_obs.py', '-c', str(config_file), '--trim']
        
        with patch('sys.argv', test_args):
            with patch.object(perfect_model_obs.WorkflowModelObs, 'from_config_file') as mock_from_config:
                mock_instance = Mock()
                mock_instance.get_config.return_value = '/some/path'
                mock_instance.run.return_value = 3
                mock_from_config.return_value = mock_instance
                
                perfect_model_obs.main()
                
                # Verify run was called with trim_obs=True
                mock_instance.run.assert_called_once()
                call_kwargs = mock_instance.run.call_args[1]
                assert call_kwargs['trim_obs'] is True
    
    def test_parse_no_matching_flag(self, tmp_path):
        """Test parsing --no-matching flag."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model_files_folder: /path\n")
        
        test_args = ['perfect_model_obs.py', '-c', str(config_file), '--no-matching']
        
        with patch('sys.argv', test_args):
            with patch.object(perfect_model_obs.WorkflowModelObs, 'from_config_file') as mock_from_config:
                mock_instance = Mock()
                mock_instance.get_config.return_value = '/some/path'
                mock_instance.run.return_value = 3
                mock_from_config.return_value = mock_instance
                
                perfect_model_obs.main()
                
                call_kwargs = mock_instance.run.call_args[1]
                assert call_kwargs['no_matching'] is True
    
    def test_parse_force_obs_time_flag(self, tmp_path):
        """Test parsing --force-obs-time flag."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model_files_folder: /path\n")
        
        test_args = ['perfect_model_obs.py', '-c', str(config_file), '--force-obs-time']
        
        with patch('sys.argv', test_args):
            with patch.object(perfect_model_obs.WorkflowModelObs, 'from_config_file') as mock_from_config:
                mock_instance = Mock()
                mock_instance.get_config.return_value = '/some/path'
                mock_instance.run.return_value = 3
                mock_from_config.return_value = mock_instance
                
                perfect_model_obs.main()
                
                call_kwargs = mock_instance.run.call_args[1]
                assert call_kwargs['force_obs_time'] is True
    
    def test_parse_parquet_only_flag(self, tmp_path):
        """Test parsing --parquet-only flag."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model_files_folder: /path\n")
        
        test_args = ['perfect_model_obs.py', '-c', str(config_file), '--parquet-only']
        
        with patch('sys.argv', test_args):
            with patch.object(perfect_model_obs.WorkflowModelObs, 'from_config_file') as mock_from_config:
                mock_instance = Mock()
                mock_instance.get_config.return_value = '/some/path'
                mock_instance.run.return_value = 3
                mock_from_config.return_value = mock_instance
                
                perfect_model_obs.main()
                
                call_kwargs = mock_instance.run.call_args[1]
                assert call_kwargs['parquet_only'] is True
    
    def test_parse_clear_output_flag(self, tmp_path):
        """Test parsing --clear-output flag."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model_files_folder: /path\n")
        
        test_args = ['perfect_model_obs.py', '-c', str(config_file), '--clear-output']
        
        with patch('sys.argv', test_args):
            with patch.object(perfect_model_obs.WorkflowModelObs, 'from_config_file') as mock_from_config:
                mock_instance = Mock()
                mock_instance.get_config.return_value = '/some/path'
                mock_instance.run.return_value = 3
                mock_from_config.return_value = mock_instance
                
                perfect_model_obs.main()
                
                call_kwargs = mock_instance.run.call_args[1]
                assert call_kwargs['clear_output'] is True
    
    def test_parse_multiple_flags(self, tmp_path):
        """Test parsing multiple flags together."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model_files_folder: /path\n")
        
        test_args = [
            'perfect_model_obs.py', 
            '-c', str(config_file),
            '--trim',
            '--no-matching',
            '--clear-output'
        ]
        
        with patch('sys.argv', test_args):
            with patch.object(perfect_model_obs.WorkflowModelObs, 'from_config_file') as mock_from_config:
                mock_instance = Mock()
                mock_instance.get_config.return_value = '/some/path'
                mock_instance.run.return_value = 3
                mock_from_config.return_value = mock_instance
                
                perfect_model_obs.main()
                
                call_kwargs = mock_instance.run.call_args[1]
                assert call_kwargs['trim_obs'] is True
                assert call_kwargs['no_matching'] is True
                assert call_kwargs['clear_output'] is True


class TestCLIWorkflowExecution:
    """Test CLI workflow execution."""
    
    def test_successful_workflow_execution(self, tmp_path):
        """Test successful execution of complete workflow."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model_files_folder: /path\n")
        
        test_args = ['perfect_model_obs.py', '-c', str(config_file)]
        
        with patch('sys.argv', test_args):
            with patch.object(perfect_model_obs.WorkflowModelObs, 'from_config_file') as mock_from_config:
                mock_instance = Mock()
                mock_instance.get_config.return_value = '/some/path'
                mock_instance.run.return_value = 3
                mock_from_config.return_value = mock_instance
                
                perfect_model_obs.main()
                
                # Verify workflow was created from config file
                mock_from_config.assert_called_once_with(str(config_file))
                
                # Verify run was called
                mock_instance.run.assert_called_once()
    
    def test_missing_config_file(self):
        """Test error handling when config file is missing."""
        test_args = ['perfect_model_obs.py', '-c', '/nonexistent/config.yaml']
        
        with patch('sys.argv', test_args):
            with patch.object(perfect_model_obs.WorkflowModelObs, 'from_config_file', side_effect=FileNotFoundError):
                with pytest.raises(FileNotFoundError):
                    perfect_model_obs.main()
    
    def test_invalid_config_file(self, tmp_path):
        """Test error handling with invalid config."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("invalid: yaml: content:")
        
        test_args = ['perfect_model_obs.py', '-c', str(config_file)]
        
        with patch('sys.argv', test_args):
            with patch.object(perfect_model_obs.WorkflowModelObs, 'from_config_file', side_effect=ValueError("Invalid YAML")):
                with pytest.raises(ValueError, match="Invalid YAML"):
                    perfect_model_obs.main()
    
    def test_workflow_execution_error(self, tmp_path):
        """Test error handling during workflow execution."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model_files_folder: /path\n")
        
        test_args = ['perfect_model_obs.py', '-c', str(config_file)]
        
        with patch('sys.argv', test_args):
            with patch.object(perfect_model_obs.WorkflowModelObs, 'from_config_file') as mock_from_config:
                mock_instance = Mock()
                mock_instance.get_config.return_value = '/some/path'
                mock_instance.run.side_effect = RuntimeError("Workflow failed")
                mock_from_config.return_value = mock_instance
                
                with pytest.raises(RuntimeError, match="Workflow failed"):
                    perfect_model_obs.main()


class TestCLIHelpOutput:
    """Test CLI help and usage information."""
    
    def test_help_flag(self):
        """Test that --help flag displays help."""
        test_args = ['perfect_model_obs.py', '--help']
        
        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                perfect_model_obs.main()
            
            # --help exits with code 0
            assert exc_info.value.code == 0
    
    def test_help_contains_description(self, capsys):
        """Test that help output contains script description."""
        test_args = ['perfect_model_obs.py', '--help']
        
        with patch('sys.argv', test_args):
            try:
                perfect_model_obs.main()
            except SystemExit:
                pass
        
        captured = capsys.readouterr()
        assert 'perfect_model_obs' in captured.out.lower() or 'perfect_model_obs' in captured.err.lower()


class TestCLIConfigHandling:
    """Test configuration file handling in CLI."""
    
    def test_default_config_path(self):
        """Test that default config path is used when not specified."""
        with patch('sys.argv', ['perfect_model_obs.py']):
            with patch.object(perfect_model_obs.WorkflowModelObs, 'from_config_file', side_effect=FileNotFoundError) as mock_from_config:
                with pytest.raises(FileNotFoundError):
                    perfect_model_obs.main()
                
                # Should try to read from default path
                mock_from_config.assert_called_once()
                assert 'config.yaml' in str(mock_from_config.call_args[0][0])
    
    def test_custom_config_path(self, tmp_path):
        """Test using custom config path."""
        config_file = tmp_path / "custom_config.yaml"
        config_file.write_text("model_files_folder: /path\n")
        
        test_args = ['perfect_model_obs.py', '--config', str(config_file)]
        
        with patch('sys.argv', test_args):
            with patch.object(perfect_model_obs.WorkflowModelObs, 'from_config_file') as mock_from_config:
                mock_instance = Mock()
                mock_instance.get_config.return_value = '/some/path'
                mock_instance.run.return_value = 3
                mock_from_config.return_value = mock_instance
                
                perfect_model_obs.main()
                
                mock_from_config.assert_called_once_with(str(config_file))


class TestCLIEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_short_and_long_flags_equivalent(self, tmp_path):
        """Test that short and long flag versions work identically."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model_files_folder: /path\n")
        
        # Test -c and -t
        test_args_short = ['perfect_model_obs.py', '-c', str(config_file), '-t']
        # Test --config and --trim
        test_args_long = ['perfect_model_obs.py', '--config', str(config_file), '--trim']
        
        results = []
        for test_args in [test_args_short, test_args_long]:
            with patch('sys.argv', test_args):
                with patch.object(perfect_model_obs.WorkflowModelObs, 'from_config_file') as mock_from_config:
                    mock_instance = Mock()
                    mock_instance.get_config.return_value = '/some/path'
                    mock_instance.run.return_value = 3
                    mock_from_config.return_value = mock_instance
                    
                    perfect_model_obs.main()
                    
                    call_kwargs = mock_instance.run.call_args[1]
                    results.append(call_kwargs['trim_obs'])
        
        # Both should have trim_obs=True
        assert results[0] == results[1] == True
