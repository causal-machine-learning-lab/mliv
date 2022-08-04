from mliv.dataset.demand import gen_data
from mliv.utils import CausalDataset
gen_data()
data = CausalDataset('./Data/Demand/0.5_1.0_0.0_10000/1/')

from mliv.inference import Vanilla2SLS
from mliv.inference import Poly2SLS
from mliv.inference import NN2SLS
from mliv.inference import OneSIV
from mliv.inference import KernelIV
from mliv.inference import DualIV
from mliv.inference import DFL
from mliv.inference import AGMM
from mliv.inference import DeepGMM
from mliv.inference import DFIV
try:
    from mliv.inference import DeepIV
except:
    pass

for mod in [OneSIV,KernelIV,DualIV,DFL,AGMM,DeepGMM,DFIV,Vanilla2SLS,Poly2SLS,NN2SLS]:

    try:
        model = mod()
        model.config['num'] = 100
        model.config['epochs'] = 10
        model.fit(data)

        print(mod)
    except:
        print('Error: ...')

try:
    model = DeepIV()
    model.config['num'] = 100
    model.config['epochs'] = 10
    model.fit(data)

    print(mod)
except:
    print(f'Error: ...{mod}...')
