import os

from tqdm import tqdm

from Costa.iot import PbmClient, DdmClient
from Costa.runner import Timestepper


rstr = os.getenv('COSTA_RSTR')

pbm = PbmClient(rstr, 'TestPbm')
assert pbm.ping_remote()

ddm = DdmClient(rstr, 'TestDdm')
assert ddm.ping_remote()

timestepper = Timestepper(pbm, ddm)

mu = {'ALPHA': -0.5}
initial = pbm.initial_condition({'ALPHA': -0.5, 't': 0.0, 'dt': 0.001})

for _ in tqdm(timestepper.solve(initial, 0.001, 50, ALPHA=-0.5)):
    pass
