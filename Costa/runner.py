from functools import cached_property
from typing import Iterable

import numpy as np

from . import util
from .api import PhysicsModel, DataModel


class Timestepper:

    pbm: PhysicsModel
    ddm: DataModel
    dirichlet_dofs: np.ndarray
    ndofs: int

    def __init__(self, pbm: PhysicsModel, ddm: DataModel):
        self.pbm = pbm
        self.ddm = ddm
        self.dirichlet_dofs = np.array(pbm.dirichlet_dofs(), dtype=int) - 1
        self.ndofs = pbm.ndof

    @cached_property
    def internal_mask(self) -> np.ndarray:
        """Return a mask (boolean vector) where internal (non-Dirichlet) degrees
        of freedom are True and external (Dirichlet) degrees of freedom are
        False.
        """
        return util.make_mask(self.ndofs, self.dirichlet_dofs)

    def to_external(self, vector: np.ndarray) -> np.ndarray:
        """Given a vector of internal values, return a full vector of internal
        and external values, where the external degrees of freedom are set to
        zero.
        """
        return util.to_external(vector, mask=self.internal_mask)

    def to_internal(self, vector: np.ndarray) -> np.ndarray:
        """Given a vector of external and internal values, return only the
        internal ones.
        """
        return util.to_internal(vector, mask=self.internal_mask)

    def solve(self, initial: np.ndarray, dt: float, nsteps: int) -> Iterable[np.ndarray]:
        """Solve the corrected source-term problem for a given number of
        timesteps.

        :param initial: Initial condition
        :param dt: Time step
        :param nsteps: Number of timesteps to make
        :return: Iterator of solution vectors
        """
        yield initial

        params = [dt]
        for _ in range(nsteps):
            predicted = self.pbm.predict(params, initial)
            residual = self.to_external(self.ddm(params, predicted))
            initial = self.pbm.correct(params, initial, residual)
            yield initial
