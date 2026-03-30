"""Model adapter registry and factory for model2obs.

This module provides the adapter registry and factory function for creating
model-specific adapter instances. The registry maps ocean model names (case-insensitive)
to their corresponding adapter classes.

Supported models:
    - MOM6: Modular Ocean Model version 6
    - ROMS_Rutgers: Regional Ocean Modeling System, Rutgers flavour

Usage:
    from model2obs.model_adapter.registry import create_model_adapter
    
    # Create adapter from model name
    adapter = create_model_adapter("MOM6")  # Returns ModelAdapterMOM6 instance
    adapter = create_model_adapter("ROMS_Rutgers")  # Returns ModelAdapterROMS_Rutgers instance
    
    # Case-insensitive and whitespace-tolerant
    adapter = create_model_adapter("  mom6  ")  # Also works

To add a new model:
    1. Create a new adapter class inheriting from ModelAdapter
    2. Add it to the _ADAPTERS registry dictionary below
    3. Implement all required abstract methods

See Also:
    - ModelAdapter: Abstract base class for all adapters
    - ModelAdapterMOM6: MOM6-specific implementation
    - ModelAdapterROMS_Rutgers: ROMS-specific implementation
"""
from __future__ import annotations

from typing import Type

from .model_adapter import ModelAdapter
from .model_adapter_MOM6 import ModelAdapterMOM6
from .model_adapter_ROMS_Rutgers import ModelAdapterROMSRutgers

_ADAPTERS: dict[str, Type[ModelAdapter]] = {
    "mom6": ModelAdapterMOM6,
    "roms_rutgers": ModelAdapterROMSRutgers,
    "cice": ModelAdapterCICE,
}

def create_model_adapter(model_name: str, **kwargs) -> ModelAdapter:
    """Create a model adapter instance for the specified ocean model.
    
    Factory function that creates and returns the appropriate ModelAdapter
    subclass based on the model_name string. Model names are case-insensitive
    and whitespace is automatically stripped.
    
    Args:
        model_name: Name of the model (e.g., 'MOM6', 'ROMS_Rutgers', 'CICE'). 
                    Case-insensitive. Cannot be None.
        **kwargs: Additional keyword arguments passed to adapter constructor
                 (currently unused but reserved for future extensibility).
    
    Returns:
        ModelAdapter subclass instance appropriate for the specified model.
    
    Raises:
        ValueError: If model_name is None or if the model name is not
                   registered in the adapter registry.
    
    Examples:
        >>> # Basic usage
        >>> adapter = create_model_adapter("MOM6")
        >>> isinstance(adapter, ModelAdapterMOM6)
        True
        >>> adapter.model_name
        'MOM6'
        
        >>> # Case-insensitive
        >>> adapter = create_model_adapter("roms_rutgers")
        >>> isinstance(adapter, ModelAdapterROMS)
        True
        
        >>> # Whitespace handling
        >>> adapter = create_model_adapter("  MOM6  ")
        >>> isinstance(adapter, ModelAdapterMOM6)
        True
        
        >>> # Invalid model raises error
        >>> create_model_adapter("invalid")
        Traceback (most recent call last):
            ...
        ValueError: Unknown model_name='invalid'. Allowed values: mom6, roms.
    """
    allowed = ", ".join(sorted(_ADAPTERS.keys()))
    if model_name is None:
        raise ValueError("model_name is required (allowed values: {allowed}).")

    key = model_name.strip().lower()
    try:
        adapter_cls = _ADAPTERS[key]
    except KeyError as e:
        raise ValueError(
            f"Unknown model_name={model_name!r}. Allowed values: {allowed}."
        ) from e

    return adapter_cls(**kwargs)
