import numpy as np


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
