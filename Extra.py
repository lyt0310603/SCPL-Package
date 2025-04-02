import torch
import torch.nn as nn
from typing import List, Optional, Callable, Any, Tuple, Union
from copy import deepcopy

class ExtraLayer(nn.Module):
    def __init__(self, n_classes, classifier):
        super().__init__()
        if classifier is None:
            self.layers = nn.Sequential(
                nn.LazyLinear(512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Linear(2048, n_classes))
        else:
            self.layers = deepcopy(classifier)
        
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.layers(x)

    def get_loss(self, x, label):
        loss = self.ce(x, label)
        return loss