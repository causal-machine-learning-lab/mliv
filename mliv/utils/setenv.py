import numpy as np 
import random
import argparse
import os
from numba import cuda

try:
    import torch
except:
    pass
try:
    import tensorflow as tf
except:
    pass

def clear_cache():
    try:
        if torch.cuda.is_available():
            cuda.select_device(0)
            cuda.close()
    except:
        pass

def set_cuda(CUDA='3'):
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA if isinstance(CUDA,str) else str(CUDA)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def set_seed(seed=2021):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def set_tf_seed(seed=2021):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    tf.reset_default_graph()
    tf.compat.v1.set_random_seed(seed)

def get_device(GPU=True):
    device = torch.device('cuda' if torch.cuda.is_available() and GPU else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return device