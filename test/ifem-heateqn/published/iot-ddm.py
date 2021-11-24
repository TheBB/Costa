import click
import os
from pathlib import Path

from Costa.ddm import Keras
from Costa.iot import DdmServer


Problem = click.Path(file_okay=False, path_type=Path)
cstr = os.getenv('COSTA_DDM_CSTR')
sstr = os.getenv('COSTA_SSTR')


@click.command()
@click.argument('problem', type=Problem)
def main(problem):
    ddm = Keras.from_file(problem / 'ddm.h5')
    with DdmServer(cstr, ddm, sstr=sstr) as server:
        server.wait()


if __name__ == '__main__':
    main()
