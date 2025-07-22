import torch
import torch.nn as nn
from typing import Tuple, List, Union


class EncoderLayer(nn.Module):
    def __init__(self, layers):
        super().__init__()
        
        # 初始化時分析每層的處理方式
        self.layers = nn.ModuleList()
        self.process_functions = []
        
        for layer in layers:
            self.layers.append(layer)
            # 根據層的類型決定處理函數
            if isinstance(layer, nn.LSTM):
                self.process_functions.append(self._process_lstm)
            elif isinstance(layer, (nn.GRU, nn.RNN)):
                self.process_functions.append(self._process_rnn)
            elif isinstance(layer, nn.Embedding):
                self.process_functions.append(self._process_embedding)
            else:
                if layer.__class__.__module__.startswith('torch.nn'):
                    self.process_functions.append(self._process_standard)
                else:
                    self.process_functions.append(self._process_custom)
        
    def _process_lstm(self, features: torch.Tensor, layer: nn.LSTM) -> Tuple[List[torch.Tensor], torch.Tensor]:
        features, (hidden_state, cell_state) = layer(features)
        return [features, hidden_state, cell_state], hidden_state.mean(dim=0)
    
    def _process_rnn(self, features: torch.Tensor, layer: Union[nn.GRU, nn.RNN]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        features, hidden_state = layer(features)
        return [features, hidden_state], hidden_state.mean(dim=0)
    
    def _process_embedding(self, features: torch.Tensor, layer: nn.Embedding) -> Tuple[List[torch.Tensor], torch.Tensor]:
        features = layer(features)
        return [features], features.mean(dim=1)
    
    def _process_standard(self, features: torch.Tensor, layer: nn.Module) -> Tuple[List[torch.Tensor], torch.Tensor]:
        features = layer(features)
        return [features], features

    def _process_custom(self, features: torch.Tensor, layer: nn.Module) -> Tuple[List[torch.Tensor], torch.Tensor]:
        try:
            features = layer(features)
            if isinstance(features, tuple):
                features, loss_value = features
            else:
                loss_value = features
            return [features], loss_value
        except ValueError:
            raise ValueError("Custom layer forward method must return two values: (output, loss_value)")

    def forward(self, features):
        for layer, process_function in zip(self.layers, self.process_functions):
            result_list, loss_value = process_function(features, layer)
            features = result_list[0]  # 使用第一個輸出作為下一層的輸入
        return result_list, loss_value