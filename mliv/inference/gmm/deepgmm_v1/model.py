from typing import List, Optional
import torch
from torch import nn
import numpy as np
import logging

from .utils.pytorch_linear_reg_utils import fit_linear, linear_reg_pred, outer_prod, add_const_col
from .dataclass import TrainDataSet, TestDataSet, TrainDataSetTorch, TestDataSetTorch

logger = logging.getLogger()


class DeepGMMModel:

    def __init__(self,
                 primal_net: nn.Module,
                 dual_net: nn.Module
                 ):
        self.primal_net = primal_net
        self.dual_net = dual_net

    def predict_t(self, treatment: torch.Tensor):
        self.primal_net.train(False)
        return self.primal_net(treatment)

    def predict(self, treatment: np.ndarray):
        treatment_t = torch.tensor(treatment, dtype=torch.float32)
        return self.predict_t(treatment_t).data.numpy()

    def evaluate_t(self, test_data: TestDataSetTorch):
        target = test_data.structural
        with torch.no_grad():
            pred = self.predict_t(test_data.treatment)
        return (torch.norm((target - pred)) ** 2) / target.size()[0]

    def evaluate(self, test_data: TestDataSet):
        return self.evaluate_t(TestDataSetTorch.from_numpy(test_data)).data.item()
