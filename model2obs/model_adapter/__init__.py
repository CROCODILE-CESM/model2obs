"""ModelAdapter modules for CrocoCamp."""

from .model_adapter import ModelAdapter
from .model_adapter import ModelAdapterCapabilities
from .model_adapter_MOM6 import ModelAdapterMOM6
from .model_adapter_ROMS_Rutgers import ModelAdapterROMSRutgers

__all__ = [
    "ModelAdapter",
    "ModelAdapterMOM6",
    "ModelAdapterROMSRutgers",
    "ModelAdapterCapabilities"
]
