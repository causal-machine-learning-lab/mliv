from .trainer import DeepGMM

example = '''
from mliv.inference import DeepGMM

model = DeepGMM()
model.fit(data)
ITE = model.predict(data.train)
ATE,_ = model.ATE(data.train)
'''