import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader
from mliv.utils import set_seed, cat

example = '''
from mliv.inference import OneSIV

model = OneSIV()
model.fit(data)
ITE = model.predict(data.train)
ATE,_ = model.ATE(data.train)
'''

class Networks(nn.Module):
    def __init__(self, z_dim, x_dim, t_dim, dropout):
        super(Networks, self).__init__()

        t_input_dim, y_input_dim = z_dim+x_dim, t_dim+x_dim

        self.t_net = nn.Sequential(nn.Linear(t_input_dim, 1280),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(1280, 320),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(320, 32),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(32, t_dim))

        self.y_net = nn.Sequential(nn.Linear(y_input_dim, 1280),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(1280, 320),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(320, 32),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(32, 1))
        
    def forward(self, z, x):   
        pred_t = self.t_net(cat([z,x]))
        yt_input = torch.cat((pred_t,x), 1)
        pred_yt = self.y_net(yt_input)
        
        return pred_t, pred_yt

class OneSIV(object):
    def __init__(self) -> None:
        self.config = {
                    'methodName': 'OneSIV',
                    'device': 'cuda:0',
                    'learning_rate': 0.005,
                    'dropout': 0.5,
                    'beta1': 0.9,
                    'beta2': 0.999,
                    'eps': 1e-8,
                    'w1': 0.0017,
                    'w2': 1.0,
                    'epochs': 30,
                    'verbose': 1,
                    'show_per_epoch': 10,
                    'batch_size':1000,
                    'seed': 2022,   
                    }

    def set_Configuration(self, config):
        self.config = config

    def get_loader(self, data=None):
        if data is None:
            data = self.train
        loader = DataLoader(data, batch_size=self.batch_size)
        return loader

    def fit(self, data, exp=-1, config=None):
        if config is None:
            config = self.config

        self.z_dim = data.train.z.shape[1]
        self.x_dim = data.train.x.shape[1]
        self.t_dim = data.train.t.shape[1]
        
        self.device = config['device']
        self.batch_size = config['batch_size']

        set_seed(config['seed'])
        data.tensor()
        data.to(self.device)
        self.data = data

        OneSIV_dict = {
            'z_dim':self.z_dim, 
            'x_dim':self.x_dim, 
            't_dim':self.t_dim, 
            'dropout':config['dropout'],
        }

        net = Networks(**OneSIV_dict)
        net.to(self.device)

        optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate'], betas=(config['beta1'], config['beta2']),eps=config['eps'])
        t_loss = torch.nn.MSELoss()
        y_loss = torch.nn.MSELoss()

        print('Run {}-th experiment for {}. '.format(exp, config['methodName']))

        train_loader = self.get_loader(data.train)

        def estimation(data):
            net.eval()
            return net.y_net(cat([data.t-data.t, data.x])), net.y_net(cat([data.t, data.x]))

        for epoch in range(config['epochs']):
            net.train()

            for idx, inputs in enumerate(train_loader):
                z = inputs['z'].to(self.device)
                x = inputs['x'].to(self.device)
                t = inputs['t'].to(self.device)
                y = inputs['y'].to(self.device)

                pred_t, pred_y = net(z,x)
                loss = config['w1'] * y_loss(pred_y, y) + config['w2'] * t_loss(pred_t, t)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            net.eval()
            if (config['verbose'] >= 1 and epoch % config['show_per_epoch'] == 0 ) or epoch == config['epochs']-1:
                _, pred_test_y = estimation(data.test)
                print(f'Epoch {epoch}: {y_loss(pred_test_y, data.test.y)}. ')

        print('End. ' + '-'*20)

        self.estimation = estimation
        self.y_net = net.y_net
        self.t_net = net.t_net

    def predict(self, data=None, t=None, x=None):
        if data is None:
            data = self.data.test

        if x is None:
            x = data.x

        if t is None:
            t = data.t

        return self.y_net(cat([t,x])).detach().cpu().numpy()

    def ITE(self, data=None, t=None, x=None):
        if data is None:
            data = self.data.test

        if x is None:
            x = data.x

        if t is None:
            t = data.t

        ITE_0 = self.y_net(cat([t-t,x])).detach().cpu().numpy()
        ITE_1 = self.y_net(cat([t-t+1,x])).detach().cpu().numpy()
        ITE_t = self.y_net(cat([t,x])).detach().cpu().numpy()

        return ITE_0,ITE_1,ITE_t

    def ATE(self, data=None, t=None, x=None):
        ITE_0,ITE_1,ITE_t = self.ITE(data,t,x)

        return np.mean(ITE_1-ITE_0), np.mean(ITE_t-ITE_0)

