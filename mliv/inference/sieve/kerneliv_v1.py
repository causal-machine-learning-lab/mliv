from mliv.utils import set_seed, cat
from typing import NamedTuple, Dict, Any, Optional, List
import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split

example = '''
from mliv.inference import KernelIV

model = KernelIV()
model.fit(data)
ITE = model.predict(data.train)
ATE,_ = model.ATE(data.train)
'''

##############################################   data_class.py   #######################
class TrainDataSet(NamedTuple):
    treatment: np.ndarray
    instrumental: np.ndarray
    covariate: Optional[np.ndarray]
    outcome: np.ndarray
    structural: np.ndarray

class TestDataSet(NamedTuple):
    treatment: np.ndarray
    covariate: Optional[np.ndarray]
    structural: np.ndarray
    instrumental: Optional[np.ndarray]
    outcome: Optional[np.ndarray]

class TrainDataSetTorch(NamedTuple):
    treatment: torch.Tensor
    instrumental: torch.Tensor
    covariate: torch.Tensor
    outcome: torch.Tensor
    structural: torch.Tensor

    @classmethod
    def from_numpy(cls, train_data: TrainDataSet):
        covariate = None
        if train_data.covariate is not None:
            covariate = torch.tensor(train_data.covariate, dtype=torch.float32)
        return TrainDataSetTorch(treatment=torch.tensor(train_data.treatment, dtype=torch.float32),
                                 instrumental=torch.tensor(train_data.instrumental, dtype=torch.float32),
                                 covariate=covariate,
                                 outcome=torch.tensor(train_data.outcome, dtype=torch.float32),
                                 structural=torch.tensor(train_data.structural, dtype=torch.float32))

    def to_gpu(self):
        covariate = None
        if self.covariate is not None:
            covariate = self.covariate.cuda()
        return TrainDataSetTorch(treatment=self.treatment.cuda(),
                                 instrumental=self.instrumental.cuda(),
                                 covariate=covariate,
                                 outcome=self.outcome.cuda(),
                                 structural=self.structural.cuda())


class TestDataSetTorch(NamedTuple):
    treatment: torch.Tensor
    instrumental: torch.Tensor
    covariate: torch.Tensor
    outcome: torch.Tensor
    structural: torch.Tensor

    @classmethod
    def from_numpy(cls, test_data: TestDataSet):
        covariate = None
        instrumental = None
        outcome = None
        if test_data.covariate is not None:
            covariate = torch.tensor(test_data.covariate, dtype=torch.float32)
        if test_data.instrumental is not None:
            instrumental = torch.tensor(test_data.instrumental, dtype=torch.float32)
        if test_data.outcome is not None:
            outcome = torch.tensor(test_data.outcome, dtype=torch.float32)
        return TestDataSetTorch(treatment=torch.tensor(test_data.treatment, dtype=torch.float32),
                                covariate=covariate,
                                instrumental=instrumental,
                                outcome=outcome,
                                structural=torch.tensor(test_data.structural, dtype=torch.float32))
    def to_gpu(self):
        covariate = None
        instrumental = None
        outcome = None
        if self.covariate is not None:
            covariate = self.covariate.cuda()
        if self.instrumental is not None:
            instrumental = self.instrumental.cuda()
        if self.outcome is not None:
            outcome = self.outcome.cuda()
        return TestDataSetTorch(treatment=self.treatment.cuda(),
                                covariate=covariate,
                                instrumental=instrumental,
                                outcome=outcome,
                                structural=self.structural.cuda())

#################################  model.py    ############################
class KernelIVModel:

    def __init__(self, X_train: np.ndarray, alpha: np.ndarray, sigma: float):
        """

        Parameters
        ----------
        X_train: np.ndarray[n_stage1, dim_treatment]
            data for treatment
        alpha:  np.ndarray[n_stage1*n_stage2 ,dim_outcome]
            final weight for prediction
        sigma: gauss parameter
        """
        self.X_train = X_train
        self.alpha = alpha
        self.sigma = sigma

    @staticmethod
    def cal_gauss(XA, XB, sigma: float = 1):
        """
        Returns gaussian kernel matrix
        Parameters
        ----------
        XA : np.ndarray[n_data1, n_dim]
        XB : np.ndarray[n_data2, n_dim]
        sigma : float

        Returns
        -------
        mat: np.ndarray[n_data1, n_data2]
        """
        dist_mat = cdist(XA, XB, "sqeuclidean")
        return np.exp(-dist_mat / sigma)

    def predict(self, treatment: np.ndarray, covariate: np.ndarray):
        X = np.array(treatment, copy=True)
        if covariate is not None:
            X = np.concatenate([X, covariate], axis=1)
        Kx = self.cal_gauss(X, self.X_train, self.sigma)
        return np.dot(Kx, self.alpha)

    def evaluate(self, test_data: TestDataSet):
        pred = self.predict(test_data.treatment, test_data.covariate)
        return np.mean((test_data.structural - pred)**2)

############## trainer.py ##############
def get_median(X) -> float:
    dist_mat = cdist(X, X, "sqeuclidean")
    res: float = np.median(dist_mat)
    return res


class KernelIVTrainer:

    def __init__(self, data_list: List, train_params: Dict[str, Any]):
        self.data_list = data_list

        self.lambda1 = train_params["lam1"]
        self.lambda2 = train_params["lam2"]
        self.split_ratio = train_params["split_ratio"]

    def split_train_data(self, train_data: TrainDataSet):
        n_data = train_data[0].shape[0]
        idx_train_1st, idx_train_2nd = train_test_split(np.arange(n_data), train_size=self.split_ratio)

        def get_data(data, idx):
            return data[idx] if data is not None else None

        train_1st_data = TrainDataSet(*[get_data(data, idx_train_1st) for data in train_data])
        train_2nd_data = TrainDataSet(*[get_data(data, idx_train_2nd) for data in train_data])
        return train_1st_data, train_2nd_data

    def train(self, rand_seed: int = 42, verbose: int = 0) -> float:
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
        train_1st_data, train_2nd_data = self.split_train_data(train_data)

        # get stage1 data
        X1 = train_1st_data.treatment
        if train_1st_data.covariate is not None:
            X1 = np.concatenate([X1, train_1st_data.covariate], axis=-1)
        Z1 = train_1st_data.instrumental
        Y1 = train_1st_data.outcome
        N = X1.shape[0]

        # get stage2 data
        X2 = train_2nd_data.treatment
        if train_2nd_data.covariate is not None:
            X2 = np.concatenate([X2, train_2nd_data.covariate], axis=-1)
        Z2 = train_2nd_data.instrumental
        Y2 = train_2nd_data.outcome
        M = X2.shape[0]

        if verbose > 0:
            print("start stage1")

        sigmaX = get_median(X1)
        sigmaZ = get_median(Z1)
        KX1X1 = KernelIVModel.cal_gauss(X1, X1, sigmaX)
        KZ1Z1 = KernelIVModel.cal_gauss(Z1, Z1, sigmaZ)
        KZ1Z2 = KernelIVModel.cal_gauss(Z1, Z2, sigmaZ)
        KX1X2 = KernelIVModel.cal_gauss(X1, X2, sigmaX)

        if isinstance(self.lambda1, list):
            self.lambda1 = 10 ** np.linspace(self.lambda1[0], self.lambda1[1], 50)
            gamma = self.stage1_tuning(KX1X1, KX1X2, KZ1Z1, KZ1Z2)
        else:
            gamma = np.linalg.solve(KZ1Z1 + N * self.lambda1 * np.eye(N), KZ1Z2)
        W = KX1X1.dot(gamma)
        if verbose > 0:
            print("end stage1")
            print("start stage2")

        if isinstance(self.lambda2, list):
            self.lambda2 = 10 ** np.linspace(self.lambda2[0], self.lambda2[1], 50)
            alpha = self.stage2_tuning(W, KX1X1, Y1, Y2)
        else:
            alpha = np.linalg.solve(W.dot(W.T) + M * self.lambda2 * KX1X1, W.dot(Y2))

        if verbose > 0:
            print("end stage2")

        mdl = KernelIVModel(X1, alpha, sigmaX)
        train_loss = mdl.evaluate(train_data)

        test_loss = mdl.evaluate(test_data)
        if verbose > 0:
            print(f"test_loss:{test_loss}")

        return train_loss, test_loss, mdl

    def stage1_tuning(self, KX1X1, KX1X2, KZ1Z1, KZ1Z2):
        N = KX1X1.shape[0]
        gamma_list = [np.linalg.solve(KZ1Z1 + N * lam1 * np.eye(N), KZ1Z2) for lam1 in self.lambda1]
        score = [np.trace(gamma.T.dot(KX1X1.dot(gamma)) - 2 * KX1X2.T.dot(gamma)) for gamma in gamma_list]
        self.lambda1 = self.lambda1[np.argmin(score)]
        return gamma_list[np.argmin(score)]

    def stage2_tuning(self, W, KX1X1, Y1, Y2):
        M = W.shape[1]
        b = W.dot(Y2)
        A = W.dot(W.T)
        alpha_list = [np.linalg.solve(A + M * lam2 * KX1X1, b) for lam2 in self.lambda2]
        score = [np.linalg.norm(Y1 - KX1X1.dot(alpha)) for alpha in alpha_list]
        self.lambda2 = self.lambda2[np.argmin(score)]
        return alpha_list[np.argmin(score)]

class KernelIV(object):
    def __init__(self) -> None:
        self.config = {
                    'methodName': 'KernelIV',
                    'num': -1,
                    'lam1': [-2, -10],
                    'lam2': [-2, -10],
                    'split_ratio': 0.5,
                    'verbose': 1,
                    'seed': 2022,   
                    }

    def set_Configuration(self, config):
        self.config = config

    def fit(self, data, exp=-1, config=None):
        if config is None:
            config = self.config

        set_seed(config['seed'])
        data.numpy()

        num = config['num'] 
        num = num if num > 0 else data.train.length

        train_config = {'lam1': config['lam1'],
                        'lam2': config['lam2'],
                        'split_ratio': config['split_ratio']}
        verbose = config['verbose']

        train_data = TrainDataSet(treatment=data.train.t[:num],
                                    instrumental=cat([data.train.z, data.train.x])[:num],
                                    covariate=data.train.x[:num],
                                    outcome=data.train.y[:num],
                                    structural=data.train.g[:num])
        val_data = TrainDataSet(treatment=data.valid.t,
                                    instrumental=cat([data.valid.z, data.valid.x]),
                                    covariate=data.valid.x,
                                    outcome=data.valid.y,
                                    structural=data.valid.g)
        test_data = TestDataSet(treatment=data.test.t,
                                    instrumental=cat([data.test.z, data.test.x]),
                                    covariate=data.test.x,
                                    outcome=data.test.y,
                                    structural=data.test.g)
        data_list = [train_data, val_data, test_data]

        print('Run {}-th experiment for {}. '.format(exp, config['methodName']))

        trainer = KernelIVTrainer(data_list, train_config)
        train_loss, test_loss, mdl = trainer.train(rand_seed=42, verbose=verbose)

        print('End. ' + '-'*20)

        def estimation(data):
            return mdl.predict(data.t-data.t, data.x), mdl.predict(data.t, data.x)

        self.mdl = mdl
        self.estimation = estimation

    def predict(self, data=None, t=None, x=None):
        if data is None:
            data = self.data.test

        if x is None:
            x = data.x

        if t is None:
            t = data.t

        return self.mdl.predict(t,x)

    def ITE(self, data=None, t=None, x=None):
        if data is None:
            data = self.data.test

        if x is None:
            x = data.x

        if t is None:
            t = data.t

        ITE_0 = self.mdl.predict(t-t,x)
        ITE_1 = self.mdl.predict(t-t+1,x)
        ITE_t = self.mdl.predict(t,x)

        return ITE_0,ITE_1,ITE_t

    def ATE(self, data=None, t=None, x=None):
        ITE_0,ITE_1,ITE_t = self.ITE(data,t,x)

        return np.mean(ITE_1-ITE_0), np.mean(ITE_t-ITE_0)