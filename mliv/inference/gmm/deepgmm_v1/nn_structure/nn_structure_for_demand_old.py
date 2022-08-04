import torch
from torch import nn
from typing import Tuple


def build_net_for_demand_old() -> Tuple[nn.Module, nn.Module]:
    response_net = nn.Sequential(nn.Linear(3, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32),
                                 nn.Tanh(),
                                 nn.Linear(32, 1))

    dual_net = nn.Sequential(nn.Linear(3, 128),
                             nn.ReLU(),
                             nn.Linear(128, 64),
                             nn.ReLU(),
                             nn.Linear(64, 1))

    return response_net, dual_net
