import click
import os
from pathlib import Path
import time

import numpy as np

from Costa.iot import PhysicalDevice
import IFEM_CoSTA as Ifem


Problem = click.Path(file_okay=False, path_type=Path)
cstr = os.getenv('COSTA_PHYSICAL_CSTR')
source_alphas = [.1, .2, .3, .4, .5, .6, .8, .9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 1.8, 1.8, 2.0]


@click.command()
@click.option('--timesteps', '-t', type=int, default=5000)
@click.option('--final', '-f', type=float, default=5.0)
@click.argument('problem', type=Problem)
def main(problem: Path, timesteps: int, final: float):
    pbm = Ifem.HeatEquation(str(problem / 'pbm.xinp'), verbose=False)
    with PhysicalDevice(cstr) as device:
        for alpha in source_alphas:
            device.emit_clean()
            for step in range(timesteps + 1):
                t = step * final / timesteps
                params = {'ALPHA': alpha, 't': t, 'dt': final / timesteps}
                state = np.array(pbm.anasol(params)['primary'])
                device.emit_state(params, state)
                time.sleep(1.0)


if __name__ == '__main__':
    main()
