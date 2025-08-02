from Loss import LossLayer
from Projector import ProjectorLayer
from Encoder import EncoderLayer
from Extra import ExtraLayer
import torch
import torch.nn as nn
from typing import List, Optional, Callable, Any, Tuple, Union


class BasicBlock(nn.Module):
    def __init__(self, layer_list, device, transform_func, projector_type, loss_fn, num_classes):
        super().__init__()
        self.device = device     
        
        self.transform_func = transform_func if transform_func is not None else self._default_transform_func
        
        self.encoder_layer = EncoderLayer(layer_list)
        self.projector_layer = ProjectorLayer(projector_type)
        self.loss_layer = LossLayer(self.device, loss_fn, num_classes, self.projector_layer)
            
    @staticmethod
    def _default_transform_func(*args):
        return args[0]

    def forward_step(self, result: Union[torch.Tensor, List[torch.Tensor]], mask: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], torch.Tensor]:
        x = self.transform_func(*result)
        result_list, loss_cal = self.encoder_layer(x, mask)
            
        return result_list, loss_cal    
        
    def forward(self, result: Union[torch.Tensor, List[torch.Tensor]], mask: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return self.forward_step(result, mask)

    def loss(self, x, true_y):
        encoder_loss, _ = self.loss_layer.get_loss(x, true_y)
        return encoder_loss, x
        
    def loss_cal(self, x, true_y):     
        return self.loss(x, true_y)
    
class AdaptiveBasicBlock(BasicBlock):
    def __init__(self, layer_list, device, transform_func, projector_type, loss_fn, num_classes, classifier):
        super().__init__(layer_list, device, transform_func, projector_type, loss_fn, num_classes)
        self.extra_layer = ExtraLayer(num_classes, classifier)
        
    def adaptive_test_step(self, result: Union[torch.Tensor, List[torch.Tensor]], mask: Optional[torch.Tensor] = None):
        x = self.transform_func(*result)
        result_list, loss_cal = self.encoder_layer(x, mask)
        projected_output = self.projector_layer(loss_cal)
        classifier_output = self.extra_layer(projected_output)

        return result_list, loss_cal, classifier_output
    
    def forward(self, result: Union[torch.Tensor, List[torch.Tensor]], mask: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        if self.training:
            return super().forward_step(result, mask)
        else:
            return self.adaptive_test_step(result, mask)
    
    def loss(self, x, true_y):
        encoder_loss, projected_output = self.loss_layer.get_loss(x, true_y)

        x = x.detach()
        
        classifier_output = self.extra_layer(projected_output)
        extra_loss = self.extra_layer.get_loss(classifier_output, true_y)
        
        
        loss = encoder_loss + extra_loss*0.001
        return loss, x