import torch
import torch.nn as nn
from typing import List, Optional, Callable, Any, Tuple, Union


class EncoderLayer(nn.Module):
    def __init__(self, layer_list):
        super().__init__()
        
        # 初始化時分析每層的處理方式
        self.layers = nn.ModuleList()
        self.process_funcs = []
        
        for layer in layer_list:
            self.layers.append(layer)
            # 根據層的類型決定處理函數
            if isinstance(layer, nn.LSTM):
                self.process_funcs.append(self._process_lstm)
            elif isinstance(layer, (nn.GRU, nn.RNN)):
                self.process_funcs.append(self._process_rnn)
            elif isinstance(layer, nn.Embedding):
                self.process_funcs.append(self._process_embedding)
            else:
                if layer.__class__.__module__.startswith('torch.nn'):
                    self.process_funcs.append(self._process_standard)
                else:
                    self.process_funcs.append(self._process_custom)
        
    def _process_lstm(self, x: torch.Tensor, layer: nn.LSTM) -> Tuple[List[torch.Tensor], torch.Tensor]:
        x, (h, c) = layer(x)
        return [x, h, c], h.mean(dim=0)
    
    def _process_rnn(self, x: torch.Tensor, layer: Union[nn.GRU, nn.RNN]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        x, h = layer(x)
        return [x, h], h.mean(dim=0)
    
    def _process_embedding(self, x: torch.Tensor, layer: nn.Embedding) -> Tuple[List[torch.Tensor], torch.Tensor]:
        x = layer(x)
        return [x], x.mean(dim=1)
    
    def _process_standard(self, x: torch.Tensor, layer: nn.Module) -> Tuple[List[torch.Tensor], torch.Tensor]:
        x = layer(x)
        return [x], x

    def _process_custom(self, x: torch.Tensor, layer: nn.Module) -> Tuple[List[torch.Tensor], torch.Tensor]:
        try:
            x = layer(x)
            if isinstance(x, tuple):
                x, loss_cal = x
            else:
                loass_cal = x
            return [x], loss_cal
        except ValueError:
            raise ValueError("Custom layer forward method must return two values: (output, loss_cal)")

    def forward(self, x):
        for layer, process_func in zip(self.layers, self.process_funcs):
            result_list, loss_cal = process_func(x, layer)
            x = result_list[0]  # 使用第一個輸出作為下一層的輸入
            
        return result_list, loss_cal