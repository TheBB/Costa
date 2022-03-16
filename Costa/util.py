import numpy as np
import rich


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
