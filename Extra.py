import torch
import torch.nn as nn
from copy import deepcopy

class ExtraLayer(nn.Module):
    def __init__(self, num_classes, classifier):
        super().__init__()
        if classifier is None:
            self.layers = nn.Sequential(
                nn.LazyLinear(512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes))
        else:
            self.layers = deepcopy(classifier)
        
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, features):
        return self.layers(features)

    def get_loss(self, features, labels):
        loss = self.cross_entropy_loss(features, labels)
        return loss