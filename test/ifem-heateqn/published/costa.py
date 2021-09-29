import sys

from tqdm import tqdm

import IFEM_CoSTA as Ifem
from Costa.ddm import Keras
from Costa.runner import Timestepper

example_name = sys.argv[1]
alphas = [-.5, .7, 1.5, 2.5]

for alpha in alphas:
    pbm = Ifem.HeatEquation(f'{example_name}/pbm.xinp', verbose=False)
    ddm = Keras(f'{example_name}/ham.h5')
    runner = Timestepper(pbm, ddm)
    init = pbm.anasol({'t': 0.0, 'dt': 0.001, 'ALPHA': alpha})['primary']

    for sol in tqdm(runner.solve(init, dt=0.001, nsteps=5000, ALPHA=alpha)):
        pass
