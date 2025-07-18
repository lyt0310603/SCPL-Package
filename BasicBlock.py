from Loss import LossLayer
from Projector import ProjectorLayer
from Encoder import EncoderLayer
from Extra import ExtraLayer
import torch
import torch.nn as nn
from typing import List, Optional, Callable, Any, Tuple, Union


class BasicBlock(nn.Module):
    def __init__(self, layer_list, device, transform_func, projector_type, loss_fn, num_classes, is_adaptive, classifier):
        super().__init__()
        self.device = device     
        
        self.transform_func = transform_func if transform_func is not None else self._default_transform_func
        
        self.encoder_layer = EncoderLayer(layer_list)
        self.projector_layer = ProjectorLayer(projector_type)
        self.loss_layer = LossLayer(self.device, loss_fn, num_classes, self.projector_layer)

        self.train_step = self.forward_step
        if is_adaptive:
            self.extra_layer = ExtraLayer(num_classes, classifier)
            self.loss_type = self.adaptive_loss
            self.test_step = self.adaptive_test_step
        else: 
            self.loss_type = self.normal_loss
            self.test_step = self.forward_step
            
    @staticmethod
    def _default_transform_func(*args):
        return args[0]

    def forward_step(self, result: Union[torch.Tensor, List[torch.Tensor]]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        x = self.transform_func(*result)
        result_list, loss_cal = self.encoder_layer(x)
            
        return result_list, loss_cal
    
    def adaptive_test_step(self, result: Union[torch.Tensor, List[torch.Tensor]]):
        x = self.transform_func(*result)
        result_list, loss_cal = self.encoder_layer(x)
        projected_output = self.projector_layer(loss_cal)
        classifier_output = self.extra_layer(projected_output)

        return result_list, loss_cal, classifier_output
        
    def forward(self, result: Union[torch.Tensor, List[torch.Tensor]]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        if self.training:
            return self.train_step(result)
        else:
            return self.test_step(result)

    def normal_loss(self, x, true_y):
        encoder_loss, _ = self.loss_layer.get_loss(x, true_y)
        return encoder_loss, x

    def adaptive_loss(self, x, true_y):
        encoder_loss, projected_output = self.loss_layer.get_loss(x, true_y)

        x = x.detach()
        
        classifier_output = self.extra_layer(projected_output)
        extra_loss = self.extra_layer.get_loss(classifier_output, true_y)
        
        
        loss = encoder_loss + extra_loss*0.001
        return loss, x
        
    def loss_cal(self, x, true_y):     
        return self.loss_type(x, true_y)