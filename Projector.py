import torch
import torch.nn as nn


class ProjLayer(nn.Module):
    def __init__(self, proj_type):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Flatten(),
            self._get_proj_head(proj_type)
        )
            
    def _get_proj_head(self, proj_type):
        if proj_type == "i":
            return Identity_Projector()
        elif proj_type == "mlp":
            return MLP_Projector()
        elif proj_type == "l":
            return Linear_Projector()
        elif proj_type == "DeInfo":
            return DeInfo_Projector()
        else:
            raise ValueError(f"Unknown projector type: {proj_type}")
            
    def forward(self, x):
        return self.proj(x)

class Linear_Projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.LazyLinear(1024)

    def forward(self, x):
        return self.layer(x).to(x.device)

class Identity_Projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Identity()

    def forward(self, x):
        return self.layer(x).to(x.device)

class MLP_Projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Linear(512, 1024)
        )
        
    def forward(self, x):      
        return self.layer(x).to(x.device)

class DeInfo_Projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.LazyLinear(2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),  
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048)
        )

    def forward(self, x):      
        return self.layer(x).to(x.device)
        