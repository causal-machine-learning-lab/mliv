from typing import Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


from .nn_structure_for_demand_old import build_net_for_demand_old
from .nn_structure_for_sin import build_net_for_sin
from .nn_structure_for_dsprite import build_net_for_dsprite
from .nn_structure_for_demand_image import build_net_for_demand_image

import logging

logger = logging.getLogger()


def build_extractor(data_name: str) -> Tuple[nn.Module, nn.Module]:
    if data_name == "demand_old":
        logger.info("build old model without image")
        return build_net_for_demand_old()
    elif data_name == "sin":
        return build_net_for_sin()
    elif data_name == "dsprite":
        return build_net_for_dsprite()
    elif data_name == "demand_image":
        return build_net_for_demand_image()
    else:
        raise ValueError(f"data name {data_name} is not valid")
