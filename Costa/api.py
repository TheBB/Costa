from abc import abstractmethod, ABC


class PhysicsModel(ABC):

    @abstractmethod
    def dirichlet_dofs(self):
        """Return a list of Dirichlet DoF IDs (1-indexed)."""
        ...

    @abstractmethod
    def predict(self, params, uprev):
        """Make an uncorrected prediction of the next timestep given the
        previous timestep.  This is nothing more than a standard discrete
        timestep method.

        :param params: List of parameters.  By convention the first parameter is
            the timestep.
        :param uprev: Previous timestep.  May be ignored by a stationary solver.
        :return: Prediction of next timestep.
        """
        ...

    @abstractmethod
    def residual(self, params, uprev, unext):
        """Calculate the residual b - Au given the assumed solution unext.

        :param params: List of parameters.  By convention the first parameter is
            the timestep.
        :param uprev: Previous timestep.  May be ignored by a stationary solver.
        :param unext: The purported exact or experimental solution.
        :return: The residual b - Au."""
        ...

    @abstractmethod
    def correct(self, params, uprev, sigma):
        """Calculate a corrected prediction of the next timestep given
        the previous timestep and a right-hand side perturbation.

        :param params: List of parameters.  By convention the first parameter is
            the timestep.
        :param uprev: Previous timestep.  May be ignored by a stationary solver.
        :param sigma: Right-hand-side perturbation. If equal to zero, this method
            should be equivalent to predict(params, uprev).
        :return: Corrected prediction of next timestep.
        """
        ...



class DataModel(ABC):

    @abstractmethod
    def __call__(self, params, upred):
        """Calculate a right-hand side perturbation to use for a corrected
        prediction of the next timestep given the uncorrected prediction.

        :param params: List of parameters.  By convention the first parameter is
            the timestep.
        :param upred: Uncorrected prediction of next timestep (return value of
            PhysicsModel.predict) - external vector.
        :return: Right-hand-side perturbation for use in PhysicModel.correct
            (internal vector).
        """
        ...
