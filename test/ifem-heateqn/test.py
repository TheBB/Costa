import numpy as np

import IFEM_CoSTA as Ifem
from Costa.runner import Timestepper
from Costa.ddm import Omniscient


nsteps = 10
dt = 1e-1

# Initialize our physics-based model
pbm = Ifem.HeatEquation('square.xinp')

# Create a set of arbitrary and random solutions
i_ndofs = pbm.ndof - len(pbm.dirichlet_dofs())
solutions = [np.random.rand(i_ndofs) for _ in range(nsteps + 1)]

# The omniscient DDM generates perfect corrections
ddm = Omniscient(pbm, solutions)

# Create a corrected timestepper using IFEM and our DDM, then run it
runner = Timestepper(pbm, ddm)
initial = runner.to_external(solutions[0])
sols = list(runner.solve(initial, dt, nsteps))

# Check for matching solutions
for sol, ref in zip(sols, solutions):
    assert np.linalg.norm(sol - runner.to_external(ref)) < 1e-12
