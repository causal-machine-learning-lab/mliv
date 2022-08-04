from .trainer import AGMM

example = '''
from mliv.inference import AGMM

model = AGMM()
model.fit(data)
ITE = model.predict(data.train)
ATE,_ = model.ATE(data.train)
'''