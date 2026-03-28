import torch
import torch.nn as nn
import torch.nn.functional as F


class LossLayer:
    """Resolve and compute local block loss functions."""

    def __init__(self, device, loss_function, num_classes, projector_layer):
        self.device = device
        self.num_classes = num_classes
        self.projector_layer = projector_layer
        self.loss_function = loss_function
        
        if device == 'cpu':
            self.tensor_type = torch.FloatTensor
        else:
            self.tensor_type = torch.cuda.FloatTensor

        if isinstance(loss_function, str):
            if loss_function == "CL":
                self.loss = self.ContrastiveLoss
            elif loss_function == "DeInfo":
                self.loss = self.DeInfoLoss
            elif loss_function == "CE":
                self.loss = self.CELoss
            else:
                raise ValueError(f"Unknown loss function: {loss_function}")

    def ContrastiveLoss(self, features, labels):   
        """Compute supervised contrastive loss.

        Args:
            features: Block feature tensor.
            labels: Class labels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Loss scalar and projected
            features.
        """
        features = self.projector_layer(features)
        projected_output = features
        
        features = nn.functional.normalize(features)
        labels = labels.view(-1, 1)
        batch_size = labels.shape[0]
        mask = torch.eq(labels, labels.T).type(self.tensor_type)
        
        denom_mask = torch.scatter(torch.ones_like(mask, device=self.device), 1,
                                   torch.arange(batch_size, device=self.device).view(-1, 1), 0)
    
        logits = torch.div(torch.matmul(features, features.T), 0.1)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        denom = torch.exp(logits) * denom_mask
        prob = logits - torch.log(denom.sum(1, keepdim=True))
        loss = (denom_mask * mask * prob).sum(1) / mask.sum(1)
        loss = -loss
        loss = loss.view(1, batch_size).mean()   
        return loss, projected_output

    def DeInfoLoss(self, features, labels):
        """Compute DeInfo loss with variance/invariance/covariance terms.

        Args:
            features: Block feature tensor.
            labels: Class labels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Loss scalar and projected
            features.
        """
        features = self.projector_layer(features)
        projected_output = features
        num_features = features.cpu().shape[1]
        normalized_features =  nn.functional.normalize(features)
        batch_size = labels.shape[0]
        
        features_mean = normalized_features - normalized_features.mean(dim=0)
        cov_features = (features_mean.T @ features_mean) / (batch_size)
        cov_loss = off_diagonal(cov_features).pow(2).sum().div(num_features)

        target_onehot = to_one_hot(labels.to(self.device), self.num_classes)
        target_sm = similarity_matrix(target_onehot)
        features_sm = similarity_matrix(normalized_features)
        invar_loss = F.mse_loss(target_sm.to(self.device), features_sm.to(self.device))
        
        features_mean = normalized_features - normalized_features.mean(dim=0)
        std_features = torch.sqrt(features_mean.var(dim=0) + 0.0000001) 
        var_loss = torch.mean(F.relu(1 - std_features)) / (batch_size)

        loss = ( var_loss * 1.0 + invar_loss * 1.0 + cov_loss * 1.0)
        return loss, projected_output

    def CELoss(self, features, labels):
        """Compute cross-entropy loss.

        Args:
            features: Model features.
            labels: Ground-truth labels.

        Returns:
            Tuple[torch.Tensor, None]: Loss output and `None` projector output.
        """
        return F.cross_entropy(features, labels), None
    
    def get_loss(self, features, labels):
        """Dispatch to configured loss implementation.

        Args:
            features: Model features.
            labels: Ground-truth labels.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Loss and optional
            projected output.
        """
        return self.loss(features, labels)


def off_diagonal(x):
    """Return flattened off-diagonal entries of square matrix.

    Args:
        x: Square matrix tensor.

    Returns:
        torch.Tensor: Flattened off-diagonal values.
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def similarity_matrix(x , no_similarity_std = False):
    """Compute pairwise cosine similarity matrix.

    Args:
        x: Input feature tensor with shape `[N, D]`.
        no_similarity_std: Unused compatibility argument.

    Returns:
        torch.Tensor: Similarity matrix with shape `[N, N]`.
    """
    xc = x - x.mean(dim=1).unsqueeze(1)
    xn = xc / (1e-8 + torch.sqrt(torch.sum(xc**2, dim=1))).unsqueeze(1)
    R = xn.matmul(xn.transpose(1, 0)).clamp(-1, 1)
    return R

def to_one_hot(y, n_dims=None):
    """Convert integer labels to one-hot vectors.

    Args:
        y: Integer label tensor.
        n_dims: Number of classes. If `None`, inferred from `y`.

    Returns:
        torch.Tensor: One-hot encoded labels.
    """
    y_tensor = y.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot

