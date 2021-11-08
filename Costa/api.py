from __future__ import annotations

from abc import abstractmethod, abstractproperty, abstractclassmethod, ABC
from pathlib import Path

from typing import Dict, Union, List

import numpy as np


Parameters = Dict[str, Union[float, List[float]]]
Vector = np.ndarray
Matrix = np.ndarray


class PhysicsModel(ABC):

    @abstractproperty
    def ndof(self) -> int:
        """Return the number of degrees of freedom."""

    @abstractmethod
    def dirichlet_dofs(self) -> Vector:
        """Return a list of Dirichlet DoF IDs (1-indexed)."""

    @abstractmethod
    def initial_condition(self, params: Parameters) -> Vector:
        """Return the configured initial condition for a set of parameters."""

    @abstractmethod
    def predict(self, params: Parameters, uprev: Vector) -> Vector:
        """Make an uncorrected prediction of the next timestep given the
        previous timestep.  This is nothing more than a standard discrete
        timestep method.

        :param params: Dictionary of parameters.
            By convention the timestep is named 'dt'.
        :param uprev: Previous timestep.  May be ignored by a stationary solver.
        :return: Prediction of next timestep.
        """

    @abstractmethod
    def residual(self, params: Parameters, uprev: Vector, unext: Vector) -> Vector:
        """Calculate the residual Au - b given the assumed solution unext.

        :param params: Dictionary of parameters.
            By convention the timestep is named 'dt'.
        :param uprev: Previous timestep.  May be ignored by a stationary solver.
        :param unext: The purported exact or experimental solution.
        :return: The residual Au - b."""

    @abstractmethod
    def correct(self, params: Parameters, uprev: Vector, sigma: Vector) -> Vector:
        """Calculate a corrected prediction of the next timestep given
        the previous timestep and a right-hand side perturbation.

        :param params: Dictionary of parameters.
            By convention the timestep is named 'dt'.
        :param uprev: Previous timestep.  May be ignored by a stationary solver.
        :param sigma: Right-hand-side perturbation. If equal to zero, this method
            should be equivalent to predict(params, uprev).
        :return: Corrected prediction of next timestep.
        """



class DataModel(ABC):

    @abstractclassmethod
    def from_file(cls, filename: Union[str, Path]) -> DataModel:
        """Load a data model from a file."""

    @abstractmethod
    def __call__(self, params: Parameters, upred: Vector) -> Vector:
        """Calculate a right-hand side perturbation to use for a corrected
        prediction of the next timestep given the uncorrected prediction.

        :param params: Dictionary of parameters.
            By convention the timestep is named 'dt'.
        :param upred: Uncorrected prediction of next timestep (return value of
            PhysicsModel.predict).
        :return: Right-hand-side perturbation for use in PhysicModel.correct.
        """

    @abstractmethod
    def save(self, filename: Union[str, Path]):
        """Save a data model to a file."""


class DataTrainer(ABC):

    @abstractmethod
    def append(self, x: Matrix, y: Matrix):
        """Add training data."""

    @abstractmethod
    def train(self) -> DataModel:
        """Train the model."""
