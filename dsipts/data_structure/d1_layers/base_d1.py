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
    def past_cols(self) -> Optional[List[str]]:
        """Get the columns available in past sequence."""
        return self._past_cols

    @property
    def future_cols(self) -> Optional[List[str]]:
        """Get the columns available in future sequence."""
        return self._future_cols
