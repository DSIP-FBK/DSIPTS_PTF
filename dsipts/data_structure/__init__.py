"""Data structure module for time series forecasting."""

# D1 Layer imports
from .d1_layers import BaseD1Layer, MultiSourceTSDataSet

# D2 Layer imports
from .d2_layers import EncoderDecoder
from .d2_layers.utils import custom_collate_fn

# Backward compatibility aliases
# Since the old files are removed, we provide aliases to the new classes
LegacyMultiSourceTSDataSet = MultiSourceTSDataSet  # D1 layer compatibility
TSDataModule = EncoderDecoder  # D2 layer compatibility

__all__ = [
    # New structure
    "BaseD1Layer",
    "MultiSourceTSDataSet",
    "EncoderDecoder",
    "custom_collate_fn",
    # Legacy compatibility
    "LegacyMultiSourceTSDataSet",
    "TSDataModule",
]
