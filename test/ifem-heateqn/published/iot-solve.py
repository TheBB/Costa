import os

from tqdm import tqdm

from Costa.iot import PbmClient, DdmClient
from Costa.runner import Timestepper


rstr = os.getenv('COSTA_RSTR')
sstr = os.getenv('COSTA_SSTR')
container = os.getenv('COSTA_CONTAINER')

pbm = PbmClient(rstr, 'TestPbm', sstr=sstr, container=container)
assert pbm.ping_remote()

ddm = DdmClient(rstr, 'TestDdm', sstr=sstr, container=container)
assert ddm.ping_remote()

timestepper = Timestepper(pbm, ddm)

mu = {'ALPHA': -0.5}
initial = pbm.initial_condition({'ALPHA': -0.5, 't': 0.0, 'dt': 0.001})

for _ in tqdm(timestepper.solve(initial, 0.001, 50, ALPHA=-0.5)):
    pass
