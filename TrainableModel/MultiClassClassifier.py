import torch
from .TrainableClassificationModel import TrainableClassificationModel
from .TrainableModel import TrainableModel

import torch.nn as nn
class MultiClassClassifier(TrainableClassificationModel):
    def __init__(self,input_dim ,num_classes, criterion: nn.Module = nn.CrossEntropyLoss()) -> None:
        super(MultiClassClassifier,self).__init__(num_classes, criterion,input_dim)
        self.fc1 = nn.Linear(self.input_dim, 768)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(768, self.num_classes)
    def forward(self, x):
        x = self.dropout(x)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.fc1(x)
        return x
    def getCopy(self) -> 'TrainableModel':
        return MultiClassClassifier(self.input_dim,self.num_classes,self.criterion)