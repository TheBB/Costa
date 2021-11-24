import click
import os
from pathlib import Path

from Costa.ddm import KerasTrainer
from Costa.iot import DdmTrainer, PbmClient


Problem = click.Path(file_okay=False, path_type=Path)
cstr = os.getenv('COSTA_DDM_CSTR')
rstr = os.getenv('COSTA_RSTR')
hstr = os.getenv('COSTA_HSTR')
sstr = os.getenv('COSTA_SSTR')
container = os.getenv('COSTA_CONTAINER')


@click.command()
@click.option('--interval', '-i', type=int, default=20)
@click.argument('problem', type=Problem)
def main(problem: Path, interval: int):
    kwargs = {
        'retrain_frequency': interval,
        'filename': problem / 'ddm.h5',
    }
    pbm = PbmClient(rstr, 'TestPbm', sstr=sstr, container=container)
    assert pbm.ping_remote()
    trainer = KerasTrainer(pbm)
    with DdmTrainer(trainer, hstr, cstr, sstr, container, **kwargs) as trainer_client:
        trainer_client.listen()


if __name__ == '__main__':
    main()
