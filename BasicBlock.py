from collections import OrderedDict
from Loss import LossLayer
from loss_fn import LocalLoss
from Projector import ProjLayer
from Encoder import EncoderLayer
from Extra import ExtraLayer
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Callable, Any, Tuple, Union


class BasicBlock(nn.Module):
    def __init__(self, layer_list, device, trans_func, proj_type, loss_fn, n_classes, adaptive, classifier):
        super().__init__()
        self.device = device     
        
        self.trans_func = trans_func if trans_func is not None else self._default_trans_func
        
        self.encoder_layer = EncoderLayer(layer_list)
        self.projector_layer = ProjLayer(proj_type)
        self.loss_layer = LossLayer(self.device, loss_fn, n_classes, self.projector_layer)

        
        self.train_step = self.forward_step
        if adaptive:
            self.extra_layer = ExtraLayer(n_classes, classifier)
            self.loss_type = self.adaptive_loss
            self.test_step = self.adaptive_test_step
        else: 
            self.loss_type = self.normal_loss
            self.test_step = self.forward_step
            
    @staticmethod
    def _default_trans_func(*args):
        return args[0]

    def forward_step(self, result: Union[torch.Tensor, List[torch.Tensor]]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        x = self.trans_func(*result)
        result_list, loss_cal = self.encoder_layer(x)
            
        return result_list, loss_cal
    
    def adaptive_test_step(self, result: Union[torch.Tensor, List[torch.Tensor]]):
        x = self.trans_func(*result)
        result_list, loss_cal = self.encoder_layer(x)
        proj_out = self.projector_layer(loss_cal)
        classifier_out = self.extra_layer(proj_out)

        return result_list, loss_cal, classifier_out
        
    def forward(self, result: Union[torch.Tensor, List[torch.Tensor]]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        if self.training:
            return self.train_step(result)
        else:
            return self.test_step(result)

    def normal_loss(self, x, true_y):
        encoder_loss, _ = self.loss_layer.get_loss(x, true_y)
        return encoder_loss, x

    def adaptive_loss(self, x, true_y):
        encoder_loss, proj_out = self.loss_layer.get_loss(x, true_y)

        x = x.detach()
        
        classifier_out = self.extra_layer(proj_out)
        extra_loss = self.extra_layer.get_loss(classifier_out, true_y)
        
        
        loss = encoder_loss + extra_loss*0.001
        return loss, x
        
    def loss_cal(self, x, true_y):     
        return self.loss_type(x, true_y)