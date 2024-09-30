from typing import Generic, TypeVar
from abc import ABC, abstractmethod

import torch

T, U = TypeVar('T'), TypeVar('U')

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar('T')
U = TypeVar('U')

class Metric(ABC, Generic[T, U]):
    """
    Abstract base class for defining metrics.

    This class serves as a base class for defining metrics in machine learning tasks.
    Subclasses should implement the `__call__` method to calculate the metric value.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """

    def __init__(self):
        ...

    @abstractmethod
    def __call__(self, y_pred: T, y_true: T) -> U:
        """
        Calculate the metric value.

        This method should be implemented by subclasses to calculate the metric value
        based on the predicted values `y_pred` and the true values `y_true`.

        Parameters:
        -----------
        y_pred : T
            The predicted values.
        y_true : T
            The true values.

        Returns:
        --------
        U
            The calculated metric value.
        """
        pass

class Loss(ABC, Generic[T, U]):
    """
    Abstract base class for defining loss functions.

    This class serves as a base class for defining loss functions in machine learning tasks.
    Subclasses should implement the `__call__` method to calculate the loss value.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """

    def __init__(self):
        ...        

    @abstractmethod
    def __call__(self, y_pred:T, y_true: T) -> U:
        """
        Calculate the loss value.

        This method should be implemented by subclasses to calculate the loss value
        based on the predicted values `y_pred` and the true values `y_true`.

        Parameters:
        -----------
        y_pred : torch.Tensor
            The predicted values.
        y_true : torch.Tensor
            The true values.

        Returns:
        --------
        torch.Tensor
            The calculated loss value.
        """
        pass
    
    @abstractmethod
    def __str__(self):
        pass