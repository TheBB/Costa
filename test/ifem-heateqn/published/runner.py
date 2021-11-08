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


def _make_training_data(pbm, mu, nsteps: int, final: float):
    mu = {**mu, 'dt': final/nsteps, 't': 0.0}
    uprev = pbm.anasol(mu)['primary']
    x = np.empty((nsteps, pbm.ndof))
    y = np.empty((nsteps, pbm.ndof))

    for i in tqdm(range(nsteps), leave=False):
        mu['t'] += mu['dt']
        unext = pbm.anasol(mu)['primary']
        x[i] = pbm.predict(mu, uprev)
        y[i] = pbm.residual(mu, uprev, unext)
        uprev = unext

    return x, y


@click.group()
def main():
    pass


@main.command()
@click.option('--timesteps', '-t', type=int, default=5000)
@click.option('--final', '-f', type=float, default=5.0)
@click.argument('problem', type=Problem)
def make_training_data(problem: Path, timesteps: int, final: float):
    it = zip([source_alphas, test_alphas], ['source', 'test'])
    for alphas, name in it:
        print(f"Creating {problem}/{name} dataset")
        (problem / name).mkdir(exist_ok=True)

        xs, ys = [], []
        for alpha in tqdm(alphas):
            pbm = Ifem.HeatEquation(str(problem / 'pbm.xinp'), verbose=False)
            mu = {'ALPHA': alpha}
            x, y = _make_training_data(pbm, mu, timesteps, final)
            xs.append(x)
            ys.append(y)

        np.save(problem / name / 'x.npy', np.vstack(xs))
        np.save(problem / name / 'y.npy', np.vstack(ys))


@main.command()
@click.argument('problem', type=Problem)
def train(problem: Path):
    x = np.load(problem / 'source' / 'x.npy')
    y = np.load(problem / 'source' / 'y.npy')

    trainer = KerasTrainer()
    trainer.append(x, y)
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

