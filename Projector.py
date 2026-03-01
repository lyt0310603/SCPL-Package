import torch
import torch.nn as nn
import copy

class ProjectorLayer(nn.Module):
    def __init__(self, projector_type, custom_projector):
        super().__init__()

        self.projector = nn.Sequential(
            nn.Flatten(),
            self._get_projector_head(projector_type, custom_projector)
        )
            
    def _get_projector_head(self, projector_type, custom_projector):
        if projector_type == "i":
            return IdentityProjector()
        elif projector_type == "mlp":
            return MLPProjector()
        elif projector_type == "l":
            return LinearProjector()
        elif projector_type == "DeInfo":
            return DeInfoProjector()
        elif projector_type == "c":
            if custom_projector is None:
                raise ValueError("Custom projector is required when projector_type is 'c'")
            if not isinstance(custom_projector, nn.Module):
                raise ValueError("Custom projector must be an instance of torch.nn.Module")
            return copy.deepcopy(custom_projector)
        else:
            raise ValueError(f"Unknown projector type: {projector_type}")
            
    def forward(self, x):
        return self.projector(x)

class LinearProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.LazyLinear(1024)

    def forward(self, x):
        return self.layer(x).to(x.device)

class IdentityProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Identity()

    def forward(self, x):
        return self.layer(x).to(x.device)

class MLPProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Linear(512, 1024)
        )
        
    def forward(self, x):      
        return self.layer(x).to(x.device)

class DeInfoProjector(nn.Module):
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
        