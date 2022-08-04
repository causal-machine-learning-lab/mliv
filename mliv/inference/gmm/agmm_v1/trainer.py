import torch
from torch import nn
import numpy as np
from .net import AGMM_Net
from mliv.utils import set_seed, cat

example = '''
from mliv.inference import AGMM

model = AGMM()
model.fit(data)
ITE = model.predict(data.train)
ATE,_ = model.ATE(data.train)
'''

class AGMM(object):
    def __init__(self) -> None:
        self.config = {
                    'methodName': 'AGMM',
                    'dropout': 0.1,
                    'n_hidden': 100,
                    'g_features': 100,
                    'learner_lr': 1e-4,
                    'adversary_lr': 1e-4,
                    'learner_l2': 1e-3,
                    'adversary_l2': 1e-4,
                    'adversary_norm_reg': 1e-3,
                    'epochs': 100,
                    'batch_size': 100,
                    'sigma': 2.0,
                    'n_centers': 100,
                    'device': 'cuda:0',
                    'mode': 'final',
                    'resultDir': './Results/tmp/',
                    'seed': 2022,   
                    }

    def set_Configuration(self, config):
        self.config = config

    def fit(self, data, exp=-1, config=None):
        if config is None:
            config = self.config

        device = config['device']
        p = config['dropout']
        n_hidden = config['n_hidden']
        g_features = config['g_features']
        learner_lr = config['learner_lr']
        adversary_lr = config['adversary_lr']
        learner_l2 = config['learner_l2']
        adversary_l2 = config['adversary_l2']
        adversary_norm_reg = config['adversary_norm_reg']
        epochs = config['epochs']
        bs = config['batch_size']
        sigma = config['sigma'] / g_features
        n_centers = config['n_centers']
        resultDir = config['resultDir']
        self.mode = config['mode']

        set_seed(config['seed'])
        data.numpy()

        self.z_dim = data.train.z.shape[1]
        self.x_dim = data.train.x.shape[1]
        self.t_dim = data.train.t.shape[1]

        learner = nn.Sequential(nn.Dropout(p=p), nn.Linear(self.t_dim+self.x_dim, n_hidden), nn.LeakyReLU(),
                                nn.Dropout(p=p), nn.Linear(n_hidden, n_hidden), nn.ReLU(),
                                nn.Dropout(p=p), nn.Linear(n_hidden, 1))

        adversary_fn = nn.Sequential(nn.Dropout(p=p), nn.Linear(self.z_dim+self.x_dim, n_hidden), nn.LeakyReLU(),
                                nn.Dropout(p=p), nn.Linear(n_hidden, n_hidden), nn.ReLU(),
                                nn.Dropout(p=p), nn.Linear(n_hidden, 1))


        Z_train, T_train, Y_train, G_train = map(lambda x: torch.Tensor(x).to(device), (np.concatenate([data.train.z, data.train.x],1), np.concatenate([data.train.t, data.train.x],1), data.train.y, data.train.g))
        Z_val, T_val, Y_val, G_val = map(lambda x: torch.Tensor(x).to(device), (np.concatenate([data.valid.z, data.valid.x],1), np.concatenate([data.valid.t, data.valid.x],1), data.valid.y, data.valid.g))
        T_test_tens = torch.Tensor(np.concatenate([data.test.t, data.test.x],1)).to(device)
        G_test_tens = torch.Tensor(data.test.g).to(device)

        print('Run {}-th experiment for {}. '.format(exp, config['methodName']))

        agmm = AGMM_Net(learner, adversary_fn).fit(Z_train, T_train, Y_train, Z_val, T_val, Y_val, T_test_tens, G_val, 
                                                learner_lr=learner_lr, adversary_lr=adversary_lr,
                                                learner_l2=learner_l2, adversary_l2=adversary_l2,
                                                n_epochs=epochs, bs=bs, 
                                                results_dir=resultDir, device=device, verbose=0)
        
        print('End. ' + '-'*20)

        def estimation(data):
            input0 = torch.Tensor(np.concatenate([data.t-data.t, data.x],1)).to(device)
            point0 = agmm.predict(input0, model=self.mode)

            inputt = torch.Tensor(np.concatenate([data.t, data.x],1)).to(device)
            pointt = agmm.predict(inputt, model=self.mode)

            return point0, pointt

        self.estimation = estimation
        self.device = device
        self.agmm = agmm

    def predict(self, data=None, t=None, x=None):
        if data is None:
            data = self.data.test

        if x is None:
            x = data.x

        if t is None:
            t = data.t

        pred  = self.agmm.predict(torch.Tensor(np.concatenate([t, x],1)).to(self.device), model=self.mode)

        return pred

    def ITE(self, data=None, t=None, x=None):
        if data is None:
            data = self.data.test

        if x is None:
            x = data.x

        if t is None:
            t = data.t

        ITE_0 = self.agmm.predict(torch.Tensor(np.concatenate([t-t, x],1)).to(self.device), model=self.mode)
        ITE_1 = self.agmm.predict(torch.Tensor(np.concatenate([t-t+1, x],1)).to(self.device), model=self.mode)
        ITE_t = self.agmm.predict(torch.Tensor(np.concatenate([t, x],1)).to(self.device), model=self.mode)

        return ITE_0,ITE_1,ITE_t

    def ATE(self, data=None, t=None, x=None):
        ITE_0,ITE_1,ITE_t = self.ITE(data,t,x)

        return np.mean(ITE_1-ITE_0), np.mean(ITE_t-ITE_0)
