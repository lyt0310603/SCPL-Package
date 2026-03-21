from .Loss import LossLayer
from .Projector import ProjectorLayer
from .Encoder import EncoderLayer
from .Extra import ExtraLayer
import torch
import torch.nn as nn
from typing import List, Optional, Callable, Any, Tuple, Union


class BasicBlock(nn.Module):
    """Single decoupled block with encoder, projector, and local loss."""

    def __init__(self, layer_list, device, transform_func, projector_type, custom_projector, loss_fn, num_classes):
        super().__init__()
        self.device = device     
        
        self.transform_func = transform_func if transform_func is not None else self._default_transform_func
        
        self.encoder_layer = EncoderLayer(layer_list)
        self.projector_layer = ProjectorLayer(projector_type, custom_projector)
        self.loss_layer = LossLayer(self.device, loss_fn, num_classes, self.projector_layer)
            
    @staticmethod
    def _default_transform_func(*args):
        """Return the first positional input.

        Args:
            *args: Incoming tensors from previous block outputs.

        Returns:
            torch.Tensor: The first input tensor.
        """
        return args[0]

    def forward_step(self, result: Union[torch.Tensor, List[torch.Tensor]], mask: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Run one forward step for this block.

        Args:
            result: Current block input tensor(s).
            mask: Optional sequence mask used by encoder layers.

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor]: Encoder outputs list and
            block feature used for local loss.
        """
        x = self.transform_func(*result)
        result_list, loss_cal = self.encoder_layer(x, mask)
            
        return result_list, loss_cal    
        
    def forward(self, result: Union[torch.Tensor, List[torch.Tensor]], mask: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Forward entry point of the block.

        Args:
            result: Current block input tensor(s).
            mask: Optional sequence mask.

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor]: Same output as
            `forward_step`.
        """
        return self.forward_step(result, mask)

    def loss(self, x, true_y):
        """Compute local encoder loss.

        Args:
            x: Block feature used for loss.
            true_y: Ground-truth labels on the same device.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Loss tensor and original feature.
        """
        encoder_loss, _ = self.loss_layer.get_loss(x, true_y)
        return encoder_loss, x
        
    def loss_cal(self, x, true_y):     
        """Alias for `loss` to keep training-loop compatibility.

        Args:
            x: Block feature used for loss.
            true_y: Ground-truth labels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Same result as `loss`.
        """
        return self.loss(x, true_y)
    
class AdaptiveBasicBlock(BasicBlock):
    """BasicBlock variant with an auxiliary classifier for adaptive inference."""

    def __init__(self, layer_list, device, transform_func, projector_type, custom_projector, loss_fn, num_classes, classifier):
        super().__init__(layer_list, device, transform_func, projector_type, custom_projector, loss_fn, num_classes)
        self.extra_layer = ExtraLayer(num_classes, classifier)
        
    def adaptive_test_step(self, result: Union[torch.Tensor, List[torch.Tensor]], mask: Optional[torch.Tensor] = None):
        """Run adaptive inference for one block.

        Args:
            result: Current block input tensor(s).
            mask: Optional sequence mask.

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]: Encoder
            outputs, local feature, and classifier logits.
        """
        x = self.transform_func(*result)
        result_list, loss_cal = self.encoder_layer(x, mask)
        projected_output = self.projector_layer(loss_cal)
        classifier_output = self.extra_layer(projected_output)

        return result_list, loss_cal, classifier_output
    
    def forward(self, result: Union[torch.Tensor, List[torch.Tensor]], mask: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Dispatch to training or adaptive-eval forward behavior.

        Args:
            result: Current block input tensor(s).
            mask: Optional sequence mask.

        Returns:
            Union[Tuple[List[torch.Tensor], torch.Tensor], Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]]:
            Training returns base outputs; evaluation returns extra classifier
            logits.
        """
        if self.training:
            return super().forward_step(result, mask)
        else:
            return self.adaptive_test_step(result, mask)
    
    def loss(self, x, true_y):
        """Combine encoder loss with auxiliary classifier loss.

        Args:
            x: Block feature used for local loss.
            true_y: Ground-truth labels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Combined loss and detached
            feature.
        """
        encoder_loss, projected_output = self.loss_layer.get_loss(x, true_y)

        x = x.detach()
        
        classifier_output = self.extra_layer(projected_output)
        extra_loss = self.extra_layer.get_loss(classifier_output, true_y)
        
        
        loss = encoder_loss + extra_loss*0.001
        return loss, x