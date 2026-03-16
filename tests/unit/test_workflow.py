"""Unit tests for base Workflow class.

Tests the abstract base Workflow class functionality including configuration
loading, validation, and workflow lifecycle management.
"""

import pytest
from typing import Any, Dict, List
from unittest.mock import patch, mock_open, MagicMock

from model2obs.workflows.workflow import Workflow


class ConcreteWorkflow(Workflow):
    """Concrete implementation of Workflow for testing."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
    
    def run(self) -> Any:
        """Execute the workflow."""
        return "workflow_executed"


@pytest.fixture
def mock_model_adapter():
    """Create a mock model adapter with configurable required keys."""
    adapter = MagicMock()
    adapter.get_required_config_keys.return_value = ['key1', 'key2']
    return adapter


class TestWorkflowInit:
    """Test Workflow initialization."""
    
    def test_init_with_valid_config_real_adapter(self):
        """Integration test: initialization with real MOM6 adapter."""
        config = {
            'ocean_model': 'MOM6',
            'model_files_folder': './model/',
            'obs_seq_in_folder': './obs',
            'trimmed_obs_folder': './trimmed_obs',
            'output_folder': './output_folder', 
            'template_file': './template.nc',
            'static_file': './static.nc',
            'ocean_geometry': './geometry.nc',
            'perfect_model_obs_dir': './perfect/',
            'parquet_folder': './parquet',
            'input_nml_bck': '/nml_backup',
        }

        workflow = ConcreteWorkflow(config)
        
        assert workflow.config == config
        assert workflow.model_adapter is not None
    
    def test_init_with_valid_config_mocked(self, mock_model_adapter):
        """Test initialization stores configuration correctly."""
        config = {'key1': 'value1', 'key2': 'value2', 'ocean_model': 'MOM6'}
        
        with patch('model2obs.workflows.workflow.create_model_adapter', return_value=mock_model_adapter):
            workflow = ConcreteWorkflow(config)
            
            assert workflow.config == config
            assert workflow.model_adapter == mock_model_adapter
    
    def test_init_validates_required_keys(self, mock_model_adapter):
        """Test that initialization validates required keys."""
        config = {
            'key1': 'value1',
            'ocean_model': 'MOM6'
        }
        
        with patch('model2obs.workflows.workflow.create_model_adapter', return_value=mock_model_adapter):
            with pytest.raises(KeyError, match="Required keys missing from config"):
                ConcreteWorkflow(config)
    
    def test_init_validates_ocean_model(self):
        """Test that initialization validates ocean_model key."""
        config = {
            'key1': 'value1',
        }
        
        with pytest.raises(ValueError, match="ocean_model is required"):
            ConcreteWorkflow(config)
    
    def test_init_with_empty_config(self):
        """Test initialization when no keys are provided."""
        config = {}
        with pytest.raises(ValueError, match="ocean_model is required"):
            ConcreteWorkflow(config)


class TestWorkflowFromConfigFile:
    """Test Workflow.from_config_file class method."""
    
    def test_from_config_file_valid_real_adapter(self, tmp_path):
        """Integration test: loading workflow from YAML with real adapter."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
ocean_model: MOM6
model_files_folder: ./model/
obs_seq_in_folder: ./obs
output_folder: ./output_folder 
template_file: ./template.nc
static_file: ./static.nc
ocean_geometry: ./geometry.nc
perfect_model_obs_dir: ./perfect/
parquet_folder: ./parquet
time_window:
  days: 999
  hours: 0 
""")
        
        target_paths = {
            'model_files_folder': 'model',
            'obs_seq_in_folder': 'obs',
            'output_folder': 'output_folder', 
            'template_file': 'template.nc',
            'static_file': 'static.nc',
            'ocean_geometry': 'geometry.nc',
            'perfect_model_obs_dir': 'perfect',
            'parquet_folder': 'parquet',
        }

        target_time = {"days": 999, "seconds": 0}

        workflow = ConcreteWorkflow.from_config_file(str(config_file))

        for k in workflow.config:
            if k not in ["ocean_model", "time_window"]:
                assert workflow.get_config(k) == str(tmp_path)+"/"+target_paths[k]
        assert workflow.get_config("ocean_model") == "MOM6"
        assert workflow.get_config("time_window") == target_time
    
    def test_from_config_file_validates_required_keys(self, tmp_path, mock_model_adapter):
        """Test that loading from file validates required keys."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
ocean_model: MOM6
key1: value1
time_window:
  days: 1
""")
        
        with patch('model2obs.workflows.workflow.create_model_adapter', return_value=mock_model_adapter):
            with pytest.raises(KeyError, match="Required keys missing from config"):
                ConcreteWorkflow.from_config_file(str(config_file))
    
    def test_from_config_file_with_kwargs_override(self, tmp_path, mock_model_adapter):
        """Test that kwargs override config file values."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
ocean_model: MOM6
key1: value1
key2: value2
parquet_folder: original
time_window:
  days: 1
""")
        
        with patch('model2obs.workflows.workflow.create_model_adapter', return_value=mock_model_adapter):
            workflow = ConcreteWorkflow.from_config_file(
                str(config_file),
                parquet_folder='overridden'
            )
            
            assert workflow.get_config('parquet_folder') == 'overridden'
            assert workflow.get_config('key1') == str(tmp_path) + '/value1'
    
    def test_from_config_file_with_new_kwargs(self, tmp_path, mock_model_adapter):
        """Test that kwargs can add new configuration keys."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
ocean_model: MOM6
key1: value1
key2: value2
time_window:
  days: 1
""")
        
        with patch('model2obs.workflows.workflow.create_model_adapter', return_value=mock_model_adapter):
            workflow = ConcreteWorkflow.from_config_file(
                str(config_file),
                new_key='new_value'
            )
            
            assert workflow.get_config('new_key') == 'new_value'
            assert workflow.get_config('key1') == str(tmp_path) + '/value1'
    
    def test_from_config_file_missing(self):
        """Test error when config file does not exist."""
        with pytest.raises(FileNotFoundError):
            ConcreteWorkflow.from_config_file('/nonexistent/config.yaml')
    
    def test_from_config_file_invalid_yaml(self, tmp_path):
        """Test error with invalid YAML syntax."""
        config_file = tmp_path / "bad_config.yaml"
        config_file.write_text("""
key1: value1
key2: [unclosed list
""")
        
        with pytest.raises(Exception):
            ConcreteWorkflow.from_config_file(str(config_file))


class TestWorkflowConfigMethods:
    """Test Workflow configuration accessor methods."""
    
    def test_get_config_existing_key(self, mock_model_adapter):
        """Test getting existing configuration value."""
        config = {'key1': 'value1', 'key2': 'value2', 'ocean_model': 'MOM6'}
        
        with patch('model2obs.workflows.workflow.create_model_adapter', return_value=mock_model_adapter):
            workflow = ConcreteWorkflow(config)
            assert workflow.get_config('key1') == 'value1'
    
    def test_get_config_missing_key_with_default(self, mock_model_adapter):
        """Test getting missing key returns default value."""
        config = {'key1': 'value1', 'key2': 'value2', 'ocean_model': 'MOM6'}
        
        with patch('model2obs.workflows.workflow.create_model_adapter', return_value=mock_model_adapter):
            workflow = ConcreteWorkflow(config)
            assert workflow.get_config('nonexistent', 'default') == 'default'
    
    def test_get_config_missing_key_no_default(self, mock_model_adapter):
        """Test getting missing key without default returns None."""
        config = {'key1': 'value1', 'key2': 'value2', 'ocean_model': 'MOM6'}
        
        with patch('model2obs.workflows.workflow.create_model_adapter', return_value=mock_model_adapter):
            workflow = ConcreteWorkflow(config)
            assert workflow.get_config('nonexistent') is None
    
    def test_set_config_new_key(self, mock_model_adapter):
        """Test setting new configuration key."""
        config = {'key1': 'value1', 'key2': 'value2', 'ocean_model': 'MOM6'}
        
        with patch('model2obs.workflows.workflow.create_model_adapter', return_value=mock_model_adapter):
            workflow = ConcreteWorkflow(config)
            workflow.set_config('key3', 'value3')
            
            assert workflow.get_config('key3') == 'value3'
            assert workflow.config['key3'] == 'value3'
    
    def test_set_config_existing_key(self, mock_model_adapter):
        """Test overwriting existing configuration key."""
        config = {'key1': 'value1', 'key2': 'value2', 'ocean_model': 'MOM6'}
        
        with patch('model2obs.workflows.workflow.create_model_adapter', return_value=mock_model_adapter):
            workflow = ConcreteWorkflow(config)
            workflow.set_config('key1', 'new_value')
            
            assert workflow.get_config('key1') == 'new_value'
    
    def test_set_config_various_types(self, mock_model_adapter):
        """Test setting configuration values of various types."""
        config = {'key1': 'value1', 'key2': 'value2', 'ocean_model': 'MOM6'}
        
        with patch('model2obs.workflows.workflow.create_model_adapter', return_value=mock_model_adapter):
            workflow = ConcreteWorkflow(config)
            workflow.set_config('int_val', 42)
            workflow.set_config('list_val', [1, 2, 3])
            workflow.set_config('dict_val', {'nested': 'value'})
            workflow.set_config('bool_val', True)
            
            assert workflow.get_config('int_val') == 42
            assert workflow.get_config('list_val') == [1, 2, 3]
            assert workflow.get_config('dict_val') == {'nested': 'value'}
            assert workflow.get_config('bool_val') is True


class TestWorkflowRun:
    """Test Workflow run method."""
    
    def test_run_executes(self, mock_model_adapter):
        """Test that run method executes."""
        config = {'key1': 'value1', 'key2': 'value2', 'ocean_model': 'MOM6'}
        
        with patch('model2obs.workflows.workflow.create_model_adapter', return_value=mock_model_adapter):
            workflow = ConcreteWorkflow(config)
            result = workflow.run()
            
            assert result == "workflow_executed"


class TestWorkflowAbstractMethods:
    """Test that abstract methods must be implemented."""
    
    def test_missing_run_method(self, mock_model_adapter):
        """Test that subclass must implement run."""
        class IncompleteWorkflow(Workflow):
            pass
        
        config = {'ocean_model': 'MOM6'}
        
        with patch('model2obs.workflows.workflow.create_model_adapter', return_value=mock_model_adapter):
            with pytest.raises(TypeError, match="Can't instantiate abstract class"):
                IncompleteWorkflow(config)


class TestWorkflowPrintConfig:
    """Test Workflow print_config method."""
    
    def test_print_config_output(self, capsys, mock_model_adapter):
        """Test that print_config displays configuration."""
        config = {'key1': 'value1', 'key2': 'value2', 'ocean_model': 'MOM6'}
        
        with patch('model2obs.workflows.workflow.create_model_adapter', return_value=mock_model_adapter):
            workflow = ConcreteWorkflow(config)
            workflow.print_config()
            
            captured = capsys.readouterr()
            assert 'Configuration:' in captured.out
            assert 'key1: value1' in captured.out
            assert 'key2: value2' in captured.out
    
    def test_print_config_empty(self, capsys, mock_model_adapter):
        """Test print_config with empty configuration."""
        mock_model_adapter.get_required_config_keys.return_value = []
        config = {'ocean_model': 'MOM6'}
        
        with patch('model2obs.workflows.workflow.create_model_adapter', return_value=mock_model_adapter):
            workflow = ConcreteWorkflow(config)
            workflow.print_config()
            
            captured = capsys.readouterr()
            assert 'Configuration:' in captured.out


class TestWorkflowValidation:
    """Test Workflow configuration validation."""
    
    def test_validate_config_called_on_init(self, mock_model_adapter):
        """Test that _validate_config is called during initialization."""
        config = {'key1': 'value1', 'ocean_model': 'MOM6'}
        
        with patch('model2obs.workflows.workflow.create_model_adapter', return_value=mock_model_adapter):
            with pytest.raises(KeyError, match="Required keys missing from config"):
                ConcreteWorkflow(config)
    
    def test_validate_config_with_all_required_keys(self, mock_model_adapter):
        """Test validation passes with all required keys present."""
        config = {'key1': 'value1', 'key2': 'value2', 'extra': 'allowed', 'ocean_model': 'MOM6'}
        
        with patch('model2obs.workflows.workflow.create_model_adapter', return_value=mock_model_adapter):
            workflow = ConcreteWorkflow(config)
            assert workflow.config == config
    
    def test_validate_config_multiple_missing_keys(self, mock_model_adapter):
        """Test validation error message with multiple missing keys."""
        config = {'key3': 'value3', 'ocean_model': 'MOM6'}
        
        with patch('model2obs.workflows.workflow.create_model_adapter', return_value=mock_model_adapter):
            with pytest.raises(KeyError) as excinfo:
                ConcreteWorkflow(config)
            
            assert 'key1' in str(excinfo.value)
            assert 'key2' in str(excinfo.value)
