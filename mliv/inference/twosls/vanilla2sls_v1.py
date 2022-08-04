from sklearn.linear_model import LinearRegression
import numpy as np
from mliv.utils import set_seed

example = '''
from mliv.inference import Vanilla2SLS

model = Vanilla2SLS()
model.fit(data)
ITE = model.predict(data.train)
ATE,_ = model.ATE(data.train)
'''

class Vanilla2SLS(object):
    def __init__(self) -> None:
        self.config = {
                    'methodName': 'Vanilla2SLS',
                    'seed': 2022
                    }

    def set_Configuration(self, config):
        self.config = config

    def fit(self, data, exp=-1, config=None):
        if config is None:
            config = self.config

        set_seed(config['seed'])
        data.numpy()

        print('Run {}-th experiment for {}. '.format(exp, config['methodName']))

        stage_1 = LinearRegression()
        stage_1.fit(np.concatenate([data.train.z, data.train.x], axis=1), data.train.t)
        t_hat = stage_1.predict(np.concatenate([data.train.z, data.train.x], axis=1))

        stage_2 = LinearRegression()
        stage_2.fit(np.concatenate([t_hat, data.train.x], axis=1), data.train.y)

        self.data = data
        self.stage_1 = stage_1
        self.stage_2 = stage_2

        print('End. ' + '-'*20)

        def estimation(data):
            return stage_2.predict(np.concatenate([data.t-data.t, data.x], axis=1)), stage_2.predict(np.concatenate([data.t, data.x], axis=1))

        self.estimation = estimation

    def predict(self, data=None, t=None, x=None):
        if data is None:
            data = self.data.test

        if x is None:
            x = data.x

        if t is None:
            t = data.t

        return self.stage_2.predict(np.concatenate([t, x], axis=1))

    def ITE(self, data=None, t=None, x=None):
        if data is None:
            data = self.data.test

        if x is None:
            x = data.x

        if t is None:
            t = data.t

        ITE_0 = self.stage_2.predict(np.concatenate([t-t, x], axis=1))
        ITE_1 = self.stage_2.predict(np.concatenate([t-t+1, x], axis=1))
        ITE_t = self.stage_2.predict(np.concatenate([t, x], axis=1))

        return ITE_0,ITE_1,ITE_t

    def ATE(self, data=None, t=None, x=None):
        ITE_0,ITE_1,ITE_t = self.ITE(data,t,x)

        return np.mean(ITE_1-ITE_0), np.mean(ITE_t-ITE_0)