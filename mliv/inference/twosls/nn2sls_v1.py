import torch
from torch import nn
from mliv.utils import set_seed, cat
from torch.utils.data import DataLoader
import numpy as np

example = '''
from mliv.inference import NN2SLS

model = NN2SLS()
model.fit(data)
ITE = model.predict(data.train)
ATE,_ = model.ATE(data.train)
'''

class NN2SLS(object):
    def __init__(self) -> None:
        self.config = {
                    'methodName': 'NN2SLS',
                    'device': 'cuda:0',
                    'instrumental_weight_decay': 0.0,
                    'covariate_weight_decay': 0.0,
                    'learning_rate': 0.005,
                    'verbose':1,
                    'show_per_epoch':5,
                    'lam2':0.1,
                    'epochs':100,
                    'batch_size':1000,
                    'seed': 2022
                    }

    def set_Configuration(self, config):
        self.config = config

    def fit(self, data, exp=-1, config=None):
        if config is None:
            config = self.config

        self.z_dim = data.train.z.shape[1]
        self.x_dim = data.train.x.shape[1]
        self.t_dim = data.train.t.shape[1]

        self.device = config['device']
        self.instrumental_weight_decay = config['instrumental_weight_decay']
        self.covariate_weight_decay = config['covariate_weight_decay']
        self.learning_rate = config['learning_rate']

        self.verbose = config['verbose']
        self.show_per_epoch = config['show_per_epoch']
        self.lam2 = config['lam2']
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']

        self.build_net()

        set_seed(config['seed'])
        data.tensor()
        data.to(self.device)
        self.data = data

        print('Run {}-th experiment for {}. '.format(exp, config['methodName']))

        self.train()

        print('End. ' + '-'*20)

    def build_net(self):
        self.instrumental_net = nn.Sequential(nn.Linear(self.z_dim+self.x_dim, 1280),
                                      nn.ReLU(),
                                      nn.Linear(1280, 320),
                                      nn.BatchNorm1d(320),
                                      nn.ReLU(),
                                      nn.Linear(320, 32),
                                      nn.ReLU(),
                                      nn.Linear(32, 1))

        self.covariate_net = nn.Sequential(nn.Linear(self.x_dim+self.t_dim, 1280),
                                      nn.ReLU(),
                                      nn.Linear(1280, 320),
                                      nn.BatchNorm1d(320),
                                      nn.ReLU(),
                                      nn.Linear(320, 32),
                                      nn.ReLU(),
                                      nn.Linear(32, 1))

        self.instrumental_net.to(self.device)
        self.covariate_net.to(self.device)

        self.instrumental_opt = torch.optim.Adam(self.instrumental_net.parameters(),lr=self.learning_rate,weight_decay=self.instrumental_weight_decay)
        self.covariate_opt = torch.optim.Adam(self.covariate_net.parameters(),lr=self.learning_rate,weight_decay=self.covariate_weight_decay)

        self.loss_fn4t = torch.nn.MSELoss()
        self.loss_fn4y = torch.nn.MSELoss()

    def train(self, verbose=None, show_per_epoch=None):
        if verbose is None or show_per_epoch is None:
            verbose, show_per_epoch = self.verbose, self.show_per_epoch

        self.lam2 *= self.data.train.length

        for exp in range(self.epochs):
            self.instrumental_update(self.data.train, verbose)

            if verbose >= 1 and (exp % show_per_epoch == 0 or exp == self.epochs - 1):
                print(type(self.data.train.z))
                train_t_hat = self.instrumental_net(cat([self.data.train.x,self.data.train.z])).detach()
                valid_t_hat = self.instrumental_net(cat([self.data.valid.x,self.data.valid.z])).detach()
                
                loss_train = self.loss_fn4t(train_t_hat, self.data.train.t)
                loss_valid = self.loss_fn4t(valid_t_hat, self.data.valid.t)

                print("Epoch {} ended: train - {:.4f}, valid - {:.4f}.".format(exp, loss_train, loss_valid))


        for exp in range(self.epochs):
            self.covariate_update(self.data.train, verbose)

            if verbose >= 1 and (exp % show_per_epoch == 0 or exp == self.epochs - 1):
                eval_train = self.evaluate(self.data.train)
                eval_valid = self.evaluate(self.data.valid)
                eval_test  = self.evaluate(self.data.test)

                print(f"Epoch {exp} ended:")
                print(f"Train: {eval_train}. ")
                print(f"Valid: {eval_valid}. ")
                print(f"Test : {eval_test}. ")

    def get_loader(self, data=None):
        if data is None:
            data = self.train
        loader = DataLoader(data, batch_size=self.batch_size)
        return loader

    def instrumental_update(self, data, verbose):
        loader = self.get_loader(data)
        self.instrumental_net.train(True)

        for idx, inputs in enumerate(loader):
            x = inputs['x'].to(self.device)
            t = inputs['t'].to(self.device)
            z = inputs['z'].to(self.device)

            t_hat = self.instrumental_net(cat([x,z]))

            loss = self.loss_fn4t(t_hat, t)

            self.instrumental_opt.zero_grad()
            loss.backward()
            self.instrumental_opt.step()

            if verbose >= 2:
                print('Batch {} - loss: {:.4f}'.format(idx, loss))
        
        self.instrumental_net.train(False)

    def covariate_update(self, data, verbose):
        loader = self.get_loader(data)
        self.covariate_net.train(True)

        for idx, inputs in enumerate(loader):
            x = inputs['x'].to(self.device)
            z = inputs['z'].to(self.device)
            y = inputs['y'].to(self.device)

            t_hat = self.instrumental_net(cat([x,z]))
            y_hat = self.covariate_net(cat([x,t_hat]))

            loss = self.loss_fn4y(y_hat, y)

            self.covariate_opt.zero_grad()
            loss.backward()
            self.covariate_opt.step()

            if verbose >= 2:
                print('Batch {} - loss: {:.4f}'.format(idx, loss))

        self.covariate_net.train(False)

    def predict(self, data=None, t=None, x=None):
        if data is None:
            data = self.data.test

        if x is None:
            x = data.x

        if t is None:
            t = data.t

        return self.covariate_net(cat([x,t])).detach().cpu().numpy()

    def ITE(self, data=None, t=None, x=None):
        if data is None:
            data = self.data.test

        if x is None:
            x = data.x

        if t is None:
            t = data.t

        ITE_0 = self.covariate_net(cat([x,t-t])).detach().cpu().numpy()
        ITE_1 = self.covariate_net(cat([x,t-t+1])).detach().cpu().numpy()
        ITE_t = self.covariate_net(cat([x,t])).detach().cpu().numpy()

        return ITE_0,ITE_1,ITE_t

    def ATE(self, data=None, t=None, x=None):
        ITE_0,ITE_1,ITE_t = self.ITE(data,t,x)

        return np.mean(ITE_1-ITE_0), np.mean(ITE_t-ITE_0)

    def estimation(self, data):
        self.covariate_net.train(False)

        y0_hat = self.covariate_net(cat([data.x,data.t-data.t]))
        yt_hat = self.covariate_net(cat([data.x,data.t]))

        return y0_hat, yt_hat

    def evaluate(self, data):
        y0_hat, yt_hat = self.estimation(data)

        loss_y = self.loss_fn4y(yt_hat, data.y)

        eval_str = 'loss_y: {:.4f}'.format(loss_y)
        return eval_str