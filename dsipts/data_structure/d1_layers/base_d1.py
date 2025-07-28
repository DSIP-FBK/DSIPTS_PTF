"""Base class for D1 layer implementations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class BaseD1Layer(ABC):
    """
    Abstract base class for D1 layer implementations.

    This class defines the mandatory interface that all D1 layers must implement
    to ensure compatibility with D2 layers and the rest of the forecasting stack.
    """

    def __init__(self):
        """Initialize the base D1 layer."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dict containing the sample data with keys:
            - 'x': Input features (past data)
            - 'y': Target values (future data)
            - 'group_id': Group identifier
            - 'past_time': Time indices for past data
            - 'future_time': Time indices for future data
            - Additional metadata as needed
        """
        pass

    @property
    @abstractmethod
    def group_cols(self) -> Union[str, List[str]]:
        """Get the group columns."""
        pass

    @property
    @abstractmethod
    def target_cols(self) -> List[str]:
        """Get the target columns."""
        pass

    @property
    @abstractmethod
    def feature_cols(self) -> List[str]:
        """Get the feature columns."""
        pass

    @property
    @abstractmethod
    def cat_cols(self) -> Optional[List[str]]:
        """Get the categorical columns."""
        pass

    @property
    @abstractmethod
    def known_cols(self) -> Optional[List[str]]:
        """Get the known future columns."""
        pass

    @property
    @abstractmethod
    def unknown_cols(self) -> Optional[List[str]]:
        """Get the unknown future columns."""
        pass


class BaseD1LayerWithDefaults(BaseD1Layer):
    """
    Base D1 layer with default property implementations.

    Subclasses can inherit from this class and set the private attributes
    (_group_cols, _target_cols, etc.) to get default property behavior.
    """

    def __init__(self):
        """Initialize the base D1 layer with defaults."""
        super().__init__()
        # Initialize default attributes - subclasses should override these
        self._group_cols = []
        self._target_cols = []
        self._feature_cols = []
        self._cat_cols = []
        self._known_cols = []
        self._unknown_cols = []

    @property
    def group_cols(self) -> Union[str, List[str]]:
        """Get the group columns."""
        return self._group_cols

    @property
    def target_cols(self) -> List[str]:
        """Get the target columns."""
        return self._target_cols

    @property
    def feature_cols(self) -> List[str]:
        """Get the feature columns."""
        return self._feature_cols

    @property
    def cat_cols(self) -> Optional[List[str]]:
        """Get the categorical columns."""
        return self._cat_cols

    @property
    def known_cols(self) -> Optional[List[str]]:
        """Get the known future columns."""
        return self._known_cols

    @property
    def unknown_cols(self) -> Optional[List[str]]:
        """Get the unknown future columns."""
        return self._unknown_cols
