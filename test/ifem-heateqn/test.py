import numpy as np

import IFEM_CoSTA as Ifem
from Costa.runner import Timestepper
from Costa.ddm import Omniscient
from Costa import util


nsteps = 10
dt = 1e-1

# Initialize our physics-based model
pbm = Ifem.HeatEquation('square.xinp', verbose=False)

# Create a set of arbitrary and random solutions
i_ndofs = pbm.ndof - len(pbm.dirichlet_dofs())
ddofs = np.array(pbm.dirichlet_dofs(), dtype=int) - 1
solutions = [
    util.to_external(np.random.rand(i_ndofs), dofs=ddofs)
    for _ in range(nsteps + 1)
]

# The omniscient DDM generates perfect corrections
ddm = Omniscient(pbm, solutions)

# Create a corrected timestepper using IFEM and our DDM, then run it
runner = Timestepper(pbm, ddm)
sols = list(runner.solve(solutions[0], dt, nsteps))

# Check for matching solutions
assert len(sols) == nsteps + 1
for sol, ref in zip(sols, solutions):
    assert np.linalg.norm(sol - ref) < 1e-12
