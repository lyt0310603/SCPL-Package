import torch
import torch.nn as nn
from typing import Tuple, List, Union


class EncoderLayer(nn.Module):
    """Dispatch layer execution by layer type and return block-level features."""

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.process_functions = []
        self.mask = None
        
        for layer in layers:
            self.layers.append(layer)
            if isinstance(layer, nn.LSTM):
                self.process_functions.append(self._process_lstm)
            elif isinstance(layer, (nn.GRU, nn.RNN)):
                self.process_functions.append(self._process_rnn)
            elif isinstance(layer, nn.Embedding):
                self.process_functions.append(self._process_embedding)
            elif isinstance(layer, nn.TransformerEncoderLayer):
                self.process_functions.append(self._process_transformer)
            else:
                if layer.__class__.__module__.startswith('torch.nn'):
                    self.process_functions.append(self._process_standard)
                else:
                    self.process_functions.append(self._process_custom)
        
    def _process_lstm(self, features: torch.Tensor, layer: nn.LSTM) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Process an LSTM layer.

        Args:
            features: Input tensor for the LSTM layer.
            layer: LSTM module.

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor]: `[output, h, c]` and mean
            hidden state used as local loss feature.
        """
        features, (hidden_state, cell_state) = layer(features)
        return [features, hidden_state, cell_state], hidden_state.mean(dim=0)
    
    def _process_rnn(self, features: torch.Tensor, layer: Union[nn.GRU, nn.RNN]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Process a GRU/RNN layer.

        Args:
            features: Input tensor for the recurrent layer.
            layer: GRU or RNN module.

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor]: `[output, h]` and mean
            hidden state used as local loss feature.
        """
        features, hidden_state = layer(features)
        return [features, hidden_state], hidden_state.mean(dim=0)
    
    def _process_embedding(self, features: torch.Tensor, layer: nn.Embedding) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Process an embedding layer.

        Args:
            features: Token indices.
            layer: Embedding module.

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor]: Embedded output and pooled
            feature along sequence dimension.
        """
        features = layer(features)
        return [features], features.mean(dim=1)
    
    def _process_transformer(self, features: torch.Tensor, layer: nn.Transformer) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Process a transformer encoder layer.

        Args:
            features: Input sequence features.
            layer: Transformer encoder layer.

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor]: Layer output and pooled
            sequence feature.
        """
        features = layer(src=features, src_key_padding_mask=self.mask)
        return [features], features.mean(dim=1)
    
    def _process_standard(self, features: torch.Tensor, layer: nn.Module) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Process a standard torch.nn layer.

        Args:
            features: Input features.
            layer: Torch module with standard forward behavior.

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor]: Layer output list and the
            same output used as local loss feature.
        """
        features = layer(features)
        return [features], features

    def _process_custom(self, features: torch.Tensor, layer: nn.Module) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Process a custom non-torch.nn layer.

        Args:
            features: Input features.
            layer: Custom module.

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor]: Output list and loss
            feature.

        Raises:
            ValueError: If custom layer output format is invalid.
        """
        try:
            features = layer(features)
            if isinstance(features, tuple):
                features, loss_value = features
            else:
                loss_value = features
            return [features], loss_value
        except ValueError:
            raise ValueError("Custom layer forward method must return two values: (output, loss_value)")

    def forward(self, features, mask=None):
        """Run all assigned layers in order.

        Args:
            features: Input tensor for this encoder block.
            mask: Optional mask for transformer-style layers.

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor]: Last layer outputs and
            final loss feature.
        """
        self.mask = mask
        for layer, process_function in zip(self.layers, self.process_functions):
            result_list, loss_value = process_function(features, layer)
            features = result_list[0]
        return result_list, loss_value