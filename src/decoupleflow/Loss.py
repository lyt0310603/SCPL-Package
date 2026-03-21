import torch
import torch.nn as nn
import torch.nn.functional as F


class LossLayer:
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
            else:
                raise ValueError(f"Unknown loss function: {loss_function}")
        else:
            self.loss = self.normal_loss

    def ContrastiveLoss(self, features, labels):   
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

    def normal_loss(self, features, labels):
        return self.loss_function(features, labels), None
    
    def get_loss(self, features, true_labels):
        return self.loss(features, true_labels)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def similarity_matrix(x , no_similarity_std = False):
    xc = x - x.mean(dim=1).unsqueeze(1)
    xn = xc / (1e-8 + torch.sqrt(torch.sum(xc**2, dim=1))).unsqueeze(1)
    R = xn.matmul(xn.transpose(1, 0)).clamp(-1, 1)
    return R

def to_one_hot(y, n_dims=None):
    y_tensor = y.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot

