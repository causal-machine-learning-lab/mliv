import torch
from torch import nn
from typing import Tuple


def build_net_for_sin() -> Tuple[nn.Module, nn.Module]:
    response_net = nn.Sequential(nn.Linear(1, 20),
                                 nn.LeakyReLU(),
                                 nn.Linear(20, 3),
                                 nn.LeakyReLU(),
                                 nn.Linear(3, 1))

    dual_net = nn.Sequential(nn.Linear(2, 20),
                             nn.LeakyReLU(),
                             nn.Linear(20, 1))

    return response_net, dual_net
