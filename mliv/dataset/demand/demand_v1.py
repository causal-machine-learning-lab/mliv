from itertools import product
import numpy as np
from numpy.random import default_rng
from pandas import DataFrame
import os

np.random.seed(42)

example = '''
from mliv.dataset.demand.demand_v1 import generate_Demand_train, generate_Demand_test, set_Configuration

config, config_trt, config_val, config_tst = set_Configuration()

train = generate_Demand_train(**config_trt)
valid = generate_Demand_train(**config_val)
test  = generate_Demand_test(**config_tst)
'''

config = {
    'dataName': 'Demand',
    'exps': 10,
    'num': 10000,
    'rho': 0.5,
    'alpha': 1.0,
    'beta': 0.0, 
    'seed': 2022,
    'num_val': 10000,
    'seed_val': 3033,
    'seed_tst': 4044
    }

def set_Configuration(config=config):
    config_trt = {}
    keys_trt = ['num', 'rho', 'alpha', 'beta', 'seed']
    for key in keys_trt:
        config_trt[key] = config[key]

    config_val = {}
    keys_val = ['rho', 'alpha', 'beta']
    for key in keys_val:
        config_val[key] = config[key]
    config_val['num'] = config['num_val']
    config_val['seed'] = config['seed_val']

    config_tst = {}
    keys_tst = ['rho', 'alpha', 'beta']
    for key in keys_tst:
        config_tst[key] = config[key]
    config_tst['seed'] = config['seed_tst']

    return config, config_trt, config_val, config_tst

def h(t):
    return 2 * ((t - 5) ** 4 / 600 + np.exp(-4 * (t - 5) ** 2) + t / 10 - 2)

def f(p, t, s):
    return 100 + (10 + p) * s * h(t) - 2 * p

def generate_Demand_train(num=10000, rho=0.5, alpha=1, beta=0, seed=2021):

    rng=default_rng(seed)
    
    emotion = rng.choice(list(range(1, 8)), (num,1))
    time = rng.uniform(0, 10, (num,1))
    cost = rng.normal(0, 1.0, (num,1))
    noise_price = rng.normal(0, 1.0, (num,1))
    noise_demand = rho * noise_price + rng.normal(0, np.sqrt(1 - rho ** 2), (num,1))
    price = 25 + (alpha * cost + 3) * h(time) + beta * cost + noise_price
    structural = f(price, time, emotion).astype(float)
    outcome = (structural + noise_demand).astype(float)
    
    mu0 = f(price-price, time, emotion).astype(float)
    mut = structural
    
    numpys = [noise_price,noise_demand, cost, time, emotion, time, emotion, price, mu0, mut, structural, outcome]
    
    train_data = DataFrame(np.concatenate(numpys, axis=1),
                          columns=['u1','u2','z1','x1','x2','c1','a1','t1','m0','mt','g1','y1'])
    
    return train_data

def generate_Demand_test(rho=0.5, alpha=1, beta=0, seed=2021):

    rng=default_rng(seed)
    
    noise_price = rng.normal(0, 1.0, (2800,1))
    noise_demand = rho * noise_price + rng.normal(0, np.sqrt(1 - rho ** 2), (2800,1))
    
    cost = np.linspace(-1.0, 1.0, 20)
    time = np.linspace(0.0, 10, 20)
    emotion = np.array([1, 2, 3, 4, 5, 6, 7])
    
    data = []
    price_z = []
    for c, t, s in product(cost, time, emotion):
        data.append([c, t, s])
        price_z.append(25 + (alpha * c + 3) * h(t) + beta * c)
    features = np.array(data)
    price_z = np.array(price_z)[:, np.newaxis]
    price = price_z + noise_price
    
    structural = f(price, features[:,1:2], features[:,2:3]).astype(float)
    outcome = (structural + noise_demand).astype(float)
    
    mu0 = f(price-price, features[:,1:2], features[:,2:3]).astype(float)
    mut = structural
    
    numpys = [noise_price, noise_demand, features, features[:,1:3], price, mu0, mut, structural, outcome]
    
    test_data = DataFrame(np.concatenate(numpys, axis=1),
                          columns=['u1','u2','z1','x1','x2','c1','a1','t1','m0','mt','g1','y1'])
    
    return test_data