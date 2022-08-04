import numpy as np
import csv
import cvxopt
from mliv.utils import set_seed, cat

example = '''
from mliv.inference import DualIV

model = DualIV()
model.fit(data)
ITE = model.predict(data.train)
ATE,_ = model.ATE(data.train)
'''

def quadprog(H, f, L=None, k=None, Aeq=None, beq=None, lb=None, ub=None):
    """
    Input: Numpy arrays, the format follows MATLAB quadprog function: https://www.mathworks.com/help/optim/ug/quadprog.html
    Output: Numpy array of the solution
    """
    n_var = H.shape[1]

    P = cvxopt.matrix(H, tc='d')
    q = cvxopt.matrix(f, tc='d')

    if L is not None or k is not None:
        assert(k is not None and L is not None)
        if lb is not None:
            L = np.vstack([L, -np.eye(n_var)])
            k = np.vstack([k, -lb])

        if ub is not None:
            L = np.vstack([L, np.eye(n_var)])
            k = np.vstack([k, ub])

        L = cvxopt.matrix(L, tc='d')
        k = cvxopt.matrix(k, tc='d')

    if Aeq is not None or beq is not None:
        assert(Aeq is not None and beq is not None)
        Aeq = cvxopt.matrix(Aeq, tc='d')
        beq = cvxopt.matrix(beq, tc='d')

    sol = cvxopt.solvers.qp(P, q, L, k, Aeq, beq)

    return np.array(sol['x'])
def median_inter(z):
    n = len(z)
    z = z.reshape(n, -1)
    A = np.repeat(z, repeats=n, axis=1)
    B = A.T
    dist = np.abs(A-B).reshape(-1,1)
    vz=np.median(dist)
    return vz

def get_K_entry(x,z,v):
    return np.exp((np.linalg.norm(x-z)**2) / (-2 * (v **2)))

def get_K_matrix(X1,X2,v):
    M = len(X1)
    N = len(X2)
    K_true = np.zeros((M,N))

    for i in range(M):
        for j in range(N):
            K_true[i,j] = get_K_entry(X1[i:i+1,:].T, X2[j:j+1,:].T, v)
    return K_true

def get_K_entry_2d(x,z,Vmat):
    return np.exp((x-z).T @ Vmat @ (x-z) /2)

def get_K_matrix_2d(X1,X2,Vmat):
    M = len(X1)
    N = len(X2)
    K_true = np.zeros((M,N))
    Vmat = np.linalg.inv(Vmat)

    for i in range(M):
        for j in range(N):
            K_true[i,j] = get_K_entry_2d(X1[i:i+1,:].T, X2[j:j+1,:].T, Vmat)
            
    return K_true

def DualIV_trainer(x, y, z):
    N, x_dim = x.shape

    vx = [median_inter(x[:,i]) for i in range(x_dim)]
    vz = median_inter(z)
    vy = median_inter(y)

    K_xx = 1
    for i in range(x_dim):
        K_xx = K_xx * get_K_matrix(x[:,i:i+1], x[:,i:i+1], vx[i])
    K_zz = get_K_matrix(z, z, vz)
    K_yy = get_K_matrix(y, y, vy)

    K = K_xx

    yz = np.concatenate([y,z],-1)
    vyz = 90000
    Vmat = np.array([[vy, vyz], [vyz, vz]])
    L_yzyz = get_K_matrix_2d(yz, yz, Vmat)

    L = L_yzyz

    lambda1 = 0.001
    gamma = N * np.linalg.norm(L @ L, 2) / np.linalg.norm(K @ L, 2) ** 2
    A = L @ L + 1 / N * gamma * L @ (K @ K) @ L + lambda1 * np.eye(N)
    Ainv = np.linalg.inv(A)

    lambda2 = 0.001
    Q = 2 * K.T @ L.T @ Ainv @ L @ K + lambda2 * np.eye(N)
    R = - 2 * K.T @ L.T @ Ainv @ L @ y

    beta = quadprog(Q,R)
    
    return beta, vx

class DualIV(object):
    def __init__(self) -> None:
        self.config = {
                    'methodName': 'DualIV',
                    'num': -1,
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

        x4train = cat([data.train.t[:num], data.train.x[:num]])
        y4train = data.train.y[:num]
        z4train = data.train.z[:num]

        print('Run {}-th experiment for {}. '.format(exp, config['methodName']))

        beta, vx = DualIV_trainer(x4train, y4train, z4train)

        def estimation(data):
            return backResult(x4train, cat([data.t-data.t, data.x]), beta, vx), backResult(x4train, cat([data.t, data.x]), beta, vx)

        print('End. ' + '-'*20)

        self.x4train = x4train
        self.beta = beta
        self.vx = vx
        self.estimation = estimation

    def predict(self, data=None, t=None, x=None):
        if data is None:
            data = self.data.test

        if x is None:
            x = data.x

        if t is None:
            t = data.t

        return backResult(self.x4train, cat([t, x]),self.beta ,self.vx)

    def ITE(self, data=None, t=None, x=None):
        if data is None:
            data = self.data.test

        if x is None:
            x = data.x

        if t is None:
            t = data.t

        ITE_0 = backResult(self.x4train, cat([t-t, x]),self.beta ,self.vx)
        ITE_1 = backResult(self.x4train, cat([t-t+1, x]),self.beta ,self.vx)
        ITE_t = backResult(self.x4train, cat([t, x]),self.beta ,self.vx)

        return ITE_0,ITE_1,ITE_t

    def ATE(self, data=None, t=None, x=None):
        ITE_0,ITE_1,ITE_t = self.ITE(data,t,x)

        return np.mean(ITE_1-ITE_0), np.mean(ITE_t-ITE_0)

def backResult(x, x_vis, beta, vx):
    x_dim = x.shape[1]
    K_Xtest = 1
    for i in range(x_dim):
        K_Xtest = K_Xtest * get_K_matrix(x[:,i:i+1], x_vis[:,i:i+1], vx[i])
        
    y_vis_dual = K_Xtest.T @ beta
    return y_vis_dual

