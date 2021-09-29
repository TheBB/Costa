from pathlib import Path

from typing import List, Union

from keras.models import load_model
import numpy as np

from .api import PhysicsModel, DataModel


class Keras(DataModel):
    """A data-driven model backed by a Keras neural network."""

    def __init__(self, filename: Union[str, Path]):
        model = load_model(filename)
        assert model
        self.model = model

    def __call__(self, params, upred: np.ndarray) -> np.ndarray:
        return self.model.predict(np.array(upred).reshape(1, -1)).reshape(-1)


class Omniscient(DataModel):
    """A data-driven model intended for testing.  Given a sequence of intended
    (and arbitrary) solutions, and access to the physics-based model, this DDM
    will provide 'perfect' corrections.

    This assumes that the DDM will be called to make predictions exactly once
    per timestep.
    """

    def __init__(self, pbm: PhysicsModel, solutions: List[np.ndarray]):
        self.pbm = pbm
        self.dirichlet_dofs = np.array(pbm.dirichlet_dofs(), dtype=int) - 1
        self.solutions = list(solutions)

    def __call__(self, params, upred: np.ndarray) -> np.ndarray:
        uprev = self.solutions.pop(0)
        unext = self.solutions[0]
        sigma = self.pbm.residual(params, uprev, unext)
        return np.array(sigma)
