import click
import os
from pathlib import Path

from Costa.iot import PbmServer
import IFEM_CoSTA as Ifem


Problem = click.Path(file_okay=False, path_type=Path)
cstr = os.getenv('COSTA_PBM_CSTR')


@click.command()
@click.argument('problem', type=Problem)
def main(problem):
    pbm = Ifem.HeatEquation(str(problem / 'pbm.xinp'), verbose=False)
    with PbmServer(cstr, pbm) as server:
        server.wait()


if __name__ == '__main__':
    main()
