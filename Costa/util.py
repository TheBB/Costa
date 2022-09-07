from typing import Dict, List, Union
import numpy as np
import rich

from . import api


message_styles = {
    'device': 'yellow',
    'pbm': 'blue',
    'ddm': 'red',
    'client': 'green',
}


class Logger:

    log_type: str = '?'
    log_sender: str

    def __init__(self, log_sender: str):
        self.log_sender = log_sender

    def log(self, *args):
        style = message_styles.get(self.log_type, 'green')
        rich.print(f'[{style}][{self.log_sender}][/{style}]', *args)



def make_mask(ndofs, dofs):
    mask = np.ones((ndofs,), dtype=bool)
    mask[dofs] = False
    return mask


def to_external(internal, dofs=None, mask=None):
    if mask is None:
        return to_external(internal, mask=make_mask(len(internal) + len(dofs), dofs))
    retval = np.zeros(mask.shape)
    retval[mask] = internal
    return retval


def to_internal(external, dofs=None, mask=None):
    if mask is None:
        return to_internal(external, mask=make_mask(len(external), dofs))
    if not isinstance(external, np.ndarray):
        external = np.array(external)
    return external[mask]


class Flattener:

    fieldnames: List[str]

    def __init__(self, *fieldnames: str):
        self.fieldnames = list(fieldnames)

    def deflatten(self, data: Union[List, np.ndarray]):
        if isinstance(data, list):
            data = np.array(data)
        data = data.reshape(-1, len(self.fieldnames))
        return {fieldname: data[:, i] for i, fieldname in enumerate(self.fieldnames)}

    def flatten(self, data: Dict[str, np.ndarray]):
        return np.array([data[fieldname] for fieldname in self.fieldnames]).T.flatten()


class DdmFlattener(api.DataModel, Flattener):

    ddm: api.DataModel

    def __init__(self, ddm: api.DataModel, *fieldnames: str):
        self.ddm = ddm
        Flattener.__init__(self, *fieldnames)

    def __call__(self, mu, upred):
        return self.deflatten(self.ddm(mu, self.flatten(upred)))


class PbmFlattener(api.PhysicsModel, Flattener):

    pbm: api.PhysicsModel

    def __init__(self, pbm: api.PhysicsModel, *fieldnames: str):
        self.pbm = pbm
        Flattener.__init__(self, *fieldnames)

    @property
    def ndof(self):
        return self.pbm.ndof

    def dirichlet_dofs(self):
        return self.pbm.dirichlet_dofs()

    def initial_condition(self, mu):
        return self.deflatten(self.pbm.initial_condition(mu))

    def predict(self, mu, uprev):
        r = self.pbm.predict(mu, self.flatten(uprev))
        return self.deflatten(r)

    def residual(self, mu, uprev, unext):
        r = self.pbm.residual(mu, self.flatten(uprev), self.flatten(unext))
        return self.deflatten(r)

    def correct(self, mu, uprev, sigma):
        r = self.pbm.correct(mu, self.flatten(uprev), self.flatten(sigma))
        return self.deflatten(r)

    def qi(self, mu, u, name):
        return self.pbm.qi(mu, self.flatten(u), name)
