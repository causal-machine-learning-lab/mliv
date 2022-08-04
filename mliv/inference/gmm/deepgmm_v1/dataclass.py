from typing import NamedTuple, Optional
import numpy as np
import torch

def load_TrainDataSet(data):
    try:
        x = data.x
    except:
        x = None

    train_data = TrainDataSet(treatment=data.t,
                              instrumental=data.z,
                              covariate=x,
                              outcome=data.y,
                              structural=data.g)
    
    return train_data
    
def load_TestDataSet(data):
    try:
        z = data.z
    except:
        z = None
    
    try:
        x = data.x
    except:
        x = None

    try:
        y = data.y
    except:
        y = None

    test_data = TestDataSet(treatment=data.t,
                            instrumental=z,
                            covariate=x,
                            outcome=y,
                            structural=data.g)
    
    return test_data

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

    def to(self, device):
        covariate = None
        if self.covariate is not None:
            covariate = self.covariate.to(device)
        return TrainDataSetTorch(treatment=self.treatment.to(device),
                                 instrumental=self.instrumental.to(device),
                                 covariate=covariate,
                                 outcome=self.outcome.to(device),
                                 structural=self.structural.to(device))


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
    def to(self, device):
        covariate = None
        instrumental = None
        outcome = None
        if self.covariate is not None:
            covariate = self.covariate.to(device)
        if self.instrumental is not None:
            instrumental = self.instrumental.to(device)
        if self.outcome is not None:
            outcome = self.outcome.to(device)
        return TestDataSetTorch(treatment=self.treatment.to(device),
                                covariate=covariate,
                                instrumental=instrumental,
                                outcome=outcome,
                                structural=self.structural.to(device))