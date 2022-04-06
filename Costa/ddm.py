from __future__ import annotations

from pathlib import Path

from typing import List, Union, Optional

from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

from .api import PhysicsModel, DataModel, DataTrainer


class KerasTrainer(DataTrainer):

    pbm: PhysicsModel
    x: List[np.ndarray]
    y: List[np.ndarray]
    model: Optional[Model] = None

    raw_store: Optional[Path] = None

    def __init__(self, pbm: PhysicsModel, model: Optional[Model] = None, raw_store: Optional[Path] = None):
        self.pbm = pbm
        self.model = model
        self.prev, self.next = [], []
        self.x, self.y = [], []
        self.raw_store = raw_store

    def store(self):
        if self.raw_store:
            np.savez(self.raw_store, x=self.x, y=self.y, prev=self.prev, next=self.next)

    def append(self, params, uprev: np.ndarray, unext: np.ndarray):
        self.prev.append(uprev)
        self.next.append(unext)
        x = np.array(self.pbm.predict(params, uprev)).flatten()
        y = np.array(self.pbm.residual(params, uprev, unext)).flatten()
        self.x.append(x)
        self.y.append(y)

    def ensure_model(self, nlayers: int = 4, layer_size_factor: float = 4):
        if self.model is not None:
            return
        inshape, outshape = self.x[0].shape[0], self.y[0].shape[0]
        midshape = int(layer_size_factor * inshape)

        model = Sequential()
        model.add(Dense(inshape))
        for _ in range(nlayers):
            model.add(Dense(midshape))
            model.add(LeakyReLU(0.01))
        model.add(Dense(outshape))

        self.model = model

    def train(self, learing_rate: float = 1e-5, **kwargs) -> Keras:
        self.ensure_model()
        assert self.model is not None

        optimizer = Adam(learning_rate=learing_rate)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])

        monitor = EarlyStopping(patience=20)
        self.model.fit(
            np.array(self.x),
            np.array(self.y),
            batch_size=32,
            epochs=100,
            validation_split=0.1,
            callbacks=[monitor],
        )
        return Keras(self.model)


class Keras(DataModel):
    """A data-driven model backed by a Keras neural network."""

    model: Model

    @classmethod
    def from_file(cls, filename: Union[str, Path]) -> Keras:
        model = load_model(filename, compile=False)
        assert model
        return cls(model)

    def __init__(self, model: Model):
        self.model = model

    def __call__(self, params, upred: np.ndarray) -> np.ndarray:
        return self.model(np.array(upred).reshape(1, -1), training=False).numpy().reshape(-1)

    def save(self, filename: Union[str, Path]):
        self.model.save(filename)


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
