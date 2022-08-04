from typing import Dict, Any, Optional, List
import torch
from torch import nn
import numpy as np

from .model import DeepGMMModel
from .dataclass import TrainDataSet, TrainDataSetTorch, TestDataSetTorch, TestDataSet
from mliv.utils import set_seed, cat

example = '''
from mliv.inference import DeepGMM

model = DeepGMM()
model.fit(data)
ITE = model.predict(data.train)
ATE,_ = model.ATE(data.train)
'''

def build_net_for_demand(z_dim, x_dim, t_dim):
    response_net = nn.Sequential(nn.Linear(z_dim + x_dim, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32),
                                 nn.Tanh(),
                                 nn.Linear(32, 1))

    dual_net = nn.Sequential(nn.Linear(t_dim + x_dim, 128),
                             nn.ReLU(),
                             nn.Linear(128, 64),
                             nn.ReLU(),
                             nn.Linear(64, 1))

    return response_net, dual_net

class DeepGMMTrainer(object):

    def __init__(self, data_list: List, net_list: List, train_params: Dict[str, Any],
                 device: str = 'cpu'):
        self.data_list = data_list
        self.device = device if torch.cuda.is_available() else 'cpu'

        # configure training params
        self.dual_iter: int = train_params["dual_iter"]
        self.primal_iter: int = train_params["primal_iter"]
        self.epochs: int = train_params["epochs"]

        # build networks
        networks = net_list
        self.primal_net: nn.Module = networks[0]
        self.dual_net: nn.Module = networks[1]
        self.primal_weight_decay = train_params["primal_weight_decay"]
        self.dual_weight_decay = train_params["dual_weight_decay"]

        self.primal_net.to(self.device)
        self.dual_net.to(self.device)

        self.primal_opt = torch.optim.Adam(self.primal_net.parameters(),
                                           weight_decay=self.primal_weight_decay,
                                           lr=0.0005, betas=(0.5, 0.9))
        self.dual_opt = torch.optim.Adam(self.dual_net.parameters(),
                                         weight_decay=self.dual_weight_decay,
                                         lr=0.0025, betas=(0.5, 0.9))

        # build monitor
        self.monitor = None

    def train(self, rand_seed: int = 42, verbose: int = 0, epoch_show: int = 20) -> float:
        """

        Parameters
        ----------
        rand_seed: int
            random seed
        verbose : int
            Determine the level of logging
        Returns
        -------
        oos_result : float
            The performance of model evaluated by oos
        """
        train_data = self.data_list[0]
        test_data = self.data_list[2]
        if train_data.covariate is not None:
            train_data = TrainDataSet(treatment=np.concatenate([train_data.treatment, train_data.covariate], axis=1),
                                      structural=train_data.structural,
                                      covariate=None,
                                      instrumental=train_data.instrumental,
                                      outcome=train_data.outcome)
            test_data = TestDataSet(treatment=np.concatenate([test_data.treatment, test_data.covariate], axis=1),
                                     covariate=None,
                                     structural=test_data.structural)

        train_data_t = TrainDataSetTorch.from_numpy(train_data)
        test_data_t = TestDataSetTorch.from_numpy(test_data)

        train_data_t = train_data_t.to(self.device)
        test_data_t = test_data_t.to(self.device)

        for t in range(self.epochs):
            self.dual_update(train_data_t, verbose)
            self.primal_update(train_data_t, verbose)
            if t % epoch_show == 0 or t == self.epochs - 1:
                print(f"Epoch {t} ended")
                if verbose >= 1:
                    print(f"Epoch {t} ended")
                    mdl = DeepGMMModel(self.primal_net, self.dual_net)
                    print(f"test error {mdl.evaluate_t(test_data_t).data.item()}")

        mdl = DeepGMMModel(self.primal_net, self.dual_net)
        oos_loss: float = mdl.evaluate_t(test_data_t).data.item()
        print(f"test_loss:{oos_loss}")
        return oos_loss

    def dual_update(self, train_data_t: TrainDataSetTorch, verbose: int):
        self.dual_net.train(True)
        self.primal_net.train(False)
        with torch.no_grad():
            epsilon = train_data_t.outcome - self.primal_net(train_data_t.treatment)
        for t in range(self.dual_iter):
            self.dual_opt.zero_grad()
            moment = torch.mean(self.dual_net(train_data_t.instrumental) * epsilon)
            reg = 0.25 * torch.mean((self.dual_net(train_data_t.instrumental) * epsilon) ** 2)
            loss = -moment + reg
            if verbose >= 2:
                print(f"dual loss:{loss.data.item()}")
            loss.backward()
            self.dual_opt.step()

    def primal_update(self, train_data_t: TrainDataSetTorch, verbose: int):
        self.dual_net.train(False)
        self.primal_net.train(True)
        with torch.no_grad():
            dual = self.dual_net(train_data_t.instrumental)
        for t in range(self.primal_iter):
            self.primal_opt.zero_grad()
            epsilon = train_data_t.outcome - self.primal_net(train_data_t.treatment)
            loss = torch.mean(dual * epsilon)
            if verbose >= 2:
                print(f"primal loss:{loss.data.item()}")
            loss.backward()
            self.primal_opt.step()

class DeepGMM(object):
    def __init__(self) -> None:
        self.config = {
                    'methodName': 'DeepGMM',
                    'resultDir': './Results/tmp/',
                    "primal_iter": 1, 
                    "dual_iter": 5, 
                    "epochs": 300, 
                    "primal_weight_decay": 0.0, 
                    "dual_weight_decay": 0.0,
                    'device': 'cuda:0',
                    'verbose': 1, 
                    'epoch_show': 50,
                    'seed': 2022,   
                    }

    def set_Configuration(self, config):
        self.config = config

    def fit(self, data, exp=-1, config=None):
        if config is None:
            config = self.config

        set_seed(config['seed'])
        data.numpy()

        self.z_dim = data.train.z.shape[1]
        self.x_dim = data.train.x.shape[1]
        self.t_dim = data.train.t.shape[1]

        response_net, dual_net = build_net_for_demand(self.z_dim,self.x_dim,self.t_dim)
        net_list = [response_net, dual_net]

        train_data = TrainDataSet(treatment=np.concatenate([data.train.t, data.train.x],1),
                                    instrumental=np.concatenate([data.train.z, data.train.x],1),
                                    covariate=None,
                                    outcome=data.train.y,
                                    structural=data.train.z)

        val_data = TrainDataSet(treatment=np.concatenate([data.valid.t, data.valid.x],1),
                                instrumental=np.concatenate([data.valid.z, data.valid.x],1),
                                covariate=None,
                                outcome=data.valid.y,
                                structural=data.valid.z)

        test_data = TestDataSet(treatment=np.concatenate([data.test.t, data.test.x],1),
                                instrumental=np.concatenate([data.test.z, data.test.x],1),
                                covariate=None,
                                outcome=None,
                                structural=data.test.z)

        data_list = [train_data, val_data, test_data]

        train_config = {"primal_iter": config['primal_iter'], 
                        "dual_iter": config['dual_iter'], 
                        "epochs": config['epochs'], 
                        "primal_weight_decay": config['primal_weight_decay'], 
                        "dual_weight_decay": config['dual_weight_decay'], 
                        }
        device = config['device']

        print('Run {}-th experiment for {}. '.format(exp, config['methodName']))

        trainer = DeepGMMTrainer(data_list, net_list, train_config, device)
        test_loss = trainer.train(rand_seed=config['seed'], verbose=config['verbose'], epoch_show=config['epoch_show'])

        def estimation(data):
            input0 = torch.Tensor(np.concatenate([data.t-data.t, data.x],1)).to(self.device)
            point0 = response_net(input0).detach().cpu().data().numpy()

            inputt = torch.Tensor(np.concatenate([data.t, data.x],1)).to(self.device)
            pointt = response_net(inputt).detach().cpu().data().numpy()

            return point0, pointt

        print('End. ' + '-'*20)

        self.estimation = estimation
        self.response_net = response_net
        self.dual_net = dual_net
        self.device = device


    def predict(self, data=None, t=None, x=None):
        if data is None:
            data = self.data.test

        if x is None:
            x = data.x

        if t is None:
            t = data.t

        with torch.no_grad():
            input = torch.Tensor(np.concatenate([t,x],1)).to(self.device)
            pred = self.response_net(input).detach().cpu().numpy()

        return pred

    def ITE(self, data=None, t=None, x=None):
        if data is None:
            data = self.data.test

        if x is None:
            x = data.x

        if t is None:
            t = data.t

        input_0 = torch.Tensor(np.concatenate([t-t,x],1)).to(self.device)
        input_1 = torch.Tensor(np.concatenate([t-t+1,x],1)).to(self.device)
        input_t = torch.Tensor(np.concatenate([t,x],1)).to(self.device)

        ITE_0 = self.response_net(input_0).detach().cpu().numpy()
        ITE_1 = self.response_net(input_1).detach().cpu().numpy()
        ITE_t = self.response_net(input_t).detach().cpu().numpy()

        return ITE_0,ITE_1,ITE_t

    def ATE(self, data=None, t=None, x=None):
        ITE_0,ITE_1,ITE_t = self.ITE(data,t,x)

        return np.mean(ITE_1-ITE_0), np.mean(ITE_t-ITE_0)