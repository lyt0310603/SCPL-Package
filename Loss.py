import torch
import torch.nn as nn
import random
import numpy as np
import math
import torch.nn.functional as F


class LossLayer:
    def __init__(self, device, loss_fn, n_classes, projector):
        self.device = device
        self.n_classes = n_classes
        self.projector = projector
        self.loss_fn = loss_fn
        
        if device == 'cpu':
            self.tensor_type = torch.FloatTensor
        else:
            self.tensor_type = torch.cuda.FloatTensor

        if isinstance(loss_fn, str):
            if loss_fn == "CL":
                self.loss = self.ContrastiveLoss
            elif loss_fn == "DeInfo":
                self.loss = self.DeInfoLoss
            else:
                raise ValueError(f"Unknown loss function: {loss_fn}")
        else:
            self.loss = self.normal_loss

    def ContrastiveLoss(self, x, label):   
        x = self.projector(x)
        proj_out = x
        
        x = nn.functional.normalize(x)
        label = label.view(-1, 1)
        batch_size = label.shape[0]
        mask = torch.eq(label, label.T).type(self.tensor_type)
        
        denom_mask = torch.scatter(torch.ones_like(mask, device=self.device), 1,
                                   torch.arange(batch_size, device=self.device).view(-1, 1), 0)
    
        logits = torch.div(torch.matmul(x, x.T), 0.1)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        denom = torch.exp(logits) * denom_mask
        prob = logits - torch.log(denom.sum(1, keepdim=True))
        loss = (denom_mask * mask * prob).sum(1) / mask.sum(1)
        loss = -loss
        loss = loss.view(1, batch_size).mean()   
        return loss, proj_out

    def DeInfoLoss(self, x, label):
        x = self.projector(x)
        proj_out = x
        num_features = x.cpu().shape[1]
        nor_x =  nn.functional.normalize(x)
        batch_size = label.shape[0]
        
        # covar
        x_mean = nor_x - nor_x.mean(dim=0)
        cov_x = (x_mean.T @ x_mean) / (batch_size)
        cov_loss = off_diagonal(cov_x).pow(2).sum().div(num_features)

        # invar
        target_onehot = to_one_hot(label.to(self.device), self.n_classes)
        target_sm = similarity_matrix(target_onehot)
        x_sm = similarity_matrix(nor_x)
        invar_loss = F.mse_loss(target_sm.to(self.device), x_sm.to(self.device))
        
        # var
        x_mean = nor_x - nor_x.mean(dim=0)
        std_x = torch.sqrt(x_mean.var(dim=0) + 0.0000001) 
        var_loss = torch.mean(F.relu(1 - std_x)) / (batch_size)

        loss = ( var_loss * 1.0 + invar_loss * 1.0 + cov_loss * 1.0)
        return loss, proj_out

    def normal_loss(self, x, label):
        return self.loss_fn(x, label), None
    
    def get_loss(self, x, true_y):
        return self.loss(x, true_y)


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

