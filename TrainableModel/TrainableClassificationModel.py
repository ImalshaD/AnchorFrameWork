import torch
from .TrainableModel import TrainableModel
from abc import ABC

import torch.nn as nn

class TrainableClassificationModel(TrainableModel, ABC):
                
    def __init__(self, num_classes ,input_dim,criterion: nn.Module = nn.CrossEntropyLoss()) -> None:
        super(TrainableClassificationModel,self).__init__(criterion,input_dim)
        self.num_classes = num_classes
    def getPrediction(self, outputs):
        return torch.max(outputs.data, 1)