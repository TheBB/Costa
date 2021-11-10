import click
import os
from pathlib import Path

from Costa.ddm import KerasTrainer
from Costa.iot import DdmTrainer, PbmClient


Problem = click.Path(file_okay=False, path_type=Path)
cstr = os.getenv('COSTA_DDM_CSTR')
rstr = os.getenv('COSTA_RSTR')
hstr = os.getenv('COSTA_HSTR')


@click.command()
def main():
    pbm = PbmClient(rstr, 'TestPbm')
    assert pbm.ping_remote()
    trainer = KerasTrainer(pbm)
    with DdmTrainer(trainer, hstr, cstr, retrain_frequency=20) as trainer_client:
        trainer_client.listen()


if __name__ == '__main__':
    main()
