from os import path
from pathlib import Path

import click
import IFEM_CoSTA as Ifem
import numpy as np
from tqdm import tqdm

from Costa.ddm import KerasTrainer, Keras
from Costa.runner import Timestepper


Problem = click.Path(file_okay=False, path_type=Path)

source_alphas = [.1, .2, .3, .4, .5, .6, .8, .9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 1.8, 1.8, 2.0]
test_alphas = [-.5, .7, 1.5, 2.5]


@click.group()
def main():
    pass


@main.command()
@click.option('--timesteps', '-t', type=int, default=5000)
@click.option('--final', '-f', type=float, default=5.0)
@click.argument('problem', type=Problem)
def train(problem: Path, timesteps: int, final: float):
    pbm = Ifem.HeatEquation(str(problem / 'pbm.xinp'), verbose=False)
    trainer = KerasTrainer(pbm)

    for alpha in tqdm(source_alphas):
        mu = {'ALPHA': alpha, 't': 0.0, 'dt': final/timesteps}
        uprev = pbm.anasol(mu)['primary']
        for _ in tqdm(range(timesteps), leave=False):
            mu['t'] += mu['dt']
            unext = pbm.anasol(mu)['primary']
            trainer.append(mu, uprev, unext)
            uprev = unext

    ddm = trainer.train()
    ddm.save(problem / 'ddm.h5')


@main.command()
@click.option('--timesteps', '-t', type=int, default=5000)
@click.option('--final', '-f', type=float, default=5.0)
@click.argument('problem', type=Problem)
def test(problem: Path, timesteps: int, final: float):
    pbm = Ifem.HeatEquation(str(problem / 'pbm.xinp'), verbose=False)
    ddm = Keras.from_file(problem / 'ddm.h5')
    dt = final / timesteps

    timestepper = Timestepper(pbm, ddm)
    for alpha in tqdm(test_alphas):
        mu = {'ALPHA': alpha}
        initial = pbm.anasol({**mu, 't': 0.0, 'dt': dt})['primary']
        for _ in tqdm(timestepper.solve(initial, dt, timesteps, ALPHA=alpha)):
            pass


if __name__ == '__main__':
    main()
