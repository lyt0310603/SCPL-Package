import torch
import torch.nn as nn
import time
from BasicBlock import BasicBlock, AdaptiveBasicBlock
from utils import Optimizer, LR_Scheduler, CPUThread


class RegSCPLModel(nn.Module):
    def __init__(self, custom_model, device_map, loss_fn="CL", num_classes=None,
                 projector_type="i", custom_projector=None,
                 transform_funcs=None, 
                 optimizer_fn=torch.optim.Adam, optimizer_param={},
                 scheduler_fn: torch.optim.lr_scheduler=None, scheduler_param={},
                 multi_t=True, 
                 is_adaptive=False, patiencethreshold=1, cosinesimthreshold=0.8, classifier=None):
        super().__init__()
        self.multi_t = multi_t
        self.is_adaptive = is_adaptive
        
        self._model_check(custom_model, device_map, transform_funcs, loss_fn, num_classes, is_adaptive, classifier)
        self.model_config, self.device_list = self._get_layer_config(custom_model, device_map, transform_funcs, 
                                                                      loss_fn, projector_type, custom_projector, num_classes, classifier)
        self._build_model()
        self._init_optimizers(optimizer_fn, optimizer_param)
        self._init_schedulers(scheduler_fn, scheduler_param)

        self.train_step = self.train_step
        self.test_step = self.test_step
        if self.is_adaptive:
            self.test_step = self.adaptive_test_step
            self.patiencecount = patiencethreshold
            self.costhreshold = cosinesimthreshold
            self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            
    
    # get the information of the user model
    def _get_layer_config(self, custom_model, device_map, transform_funcs, loss_fn, projector_type, custom_projector, num_classes, classifier):
        layer_config = {}
        balance_idx = 0
        layer_idx = 0
        layers = []
        device = []

        # 從 device_map 生成設備分配列表
        device_distribution = self._generate_device_distribution(device_map)

        for name, layer in custom_model.named_children():
            layers.append(layer)
            if len(layers) == device_distribution[layer_idx]['layer_balance']:
                layer_config[layer_idx] = {"layer_list": layers.copy(),
                                           "device": device_distribution[layer_idx]['device'],
                                           "transform_func": transform_funcs[layer_idx] if transform_funcs != None else None,
                                           "projector_type": projector_type,
                                           "custom_projector": custom_projector,
                                           "loss_fn":loss_fn if (layer_idx < len(device_distribution)-1 or self.is_adaptive) else nn.CrossEntropyLoss(),
                                           "num_classes": num_classes}
                if self.is_adaptive:
                    layer_config[layer_idx]["classifier"] = classifier

                device.append(device_distribution[layer_idx]['device'])
                layer_idx += 1
                balance_idx += 1
                layers.clear()

        return layer_config, device

    def _generate_device_distribution(self, device_map):
        """
        從 device_map 生成設備分配列表
        device_map: {"cuda:0": 1, "cuda:1": 1, "cuda:2": 1, "cuda:3": 3}
        返回: [{"layer_balance": 1, "device": "cuda:0"}, {"layer_balance": 1, "device": "cuda:1"}, ...]
        """
        if device_map is None:
            return None
            
        device_distribution = []
        for device, count in device_map.items():
            device_distribution.append({"layer_balance": count, "device": device})
            
        return device_distribution
    

    def _build_model(self):            
        self.model = []
        for idx, (k, v) in enumerate(self.model_config.items()):
            if self.is_adaptive:
                self.model.append(AdaptiveBasicBlock(**v).to(v['device']))
            else:
                self.model.append(BasicBlock(**v).to(v['device']))
        
        self.model = torch.nn.Sequential(*self.model)

    # check the relation of model layer、balance and num of gpu
    def _model_check(self, custom_model, device_map, transform_funcs, loss_fn, num_classes, is_adaptive, classifier):
        if len(custom_model) != sum(device_map.values()):
            raise ValueError(f'Layers of model don\'t equal to balance, {len(custom_model)} and {sum(device_map.values())}')
        if device_map != None:
            for device, count in device_map.items():
                if count == 0:
                    raise ValueError(f'Layer balance for device {device} is 0, which means no layer will be assigned to this device.')
                if device == None:
                    raise ValueError(f'Device is None, which means no device will be assigned to this layer.')

        if transform_funcs != None:
            if len(transform_funcs) != len(device_map):
                raise ValueError('Cannot distribute transform_funcs, please check the length of transform_funcs')

        if loss_fn == "DeInfo" and num_classes == None:
            raise ValueError('DeInfo Loss need pass class nums')

    # test to print
    def print(self):
        for layer in self.model:
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    print(name, param.data)

    def train_step(self, X, Y, mask=None):
        features_list = []
        labels_list = []
        masks_list = []
        total_loss = 0
        layer_losses = []
        layer_features = []
        threads = []
        layers_num = len(self.model_config)
        for i in range(0, layers_num):
            labels_list.append(Y.to(self.device_list[i]))
            if mask is not None:
                masks_list.append(mask.to(self.device_list[i]))
            else:
                masks_list.append(None)
        features_list.append([X.to(self.device_list[0], non_blocking=True)])        
        
        for i in range(layers_num):
            self.model[i].train()

        for optimizer in self.optimizers:
            optimizer.zero_grad()
            
        for i in range(0, layers_num):
            output, hidden_state = self.model[i](features_list[-1], masks_list[i])
            layer_features.append(hidden_state)
            
            args = (self.model[i], self.optimizers[i].optimizer, hidden_state, labels_list[i])
            
            if not self.multi_t:
                layer_losses.append(self._loss_backward_update(*args))
            else:
                threads.append(CPUThread(target=self._loss_backward_update, args=args))
                threads[-1].start()
                
            detached_list = [t.detach().to(self.device_list[i+1 if i+1 < layers_num else i], non_blocking=True) for t in output]
            features_list.append(detached_list)
            
        
        if self.multi_t:
            for t in range(len(threads)):
                    total_loss += threads[t].get_result().item()
        else:
            for loss in layer_losses:
                total_loss += loss.item()
        
        if self.is_adaptive:
            features = self.model[-1].projector_layer(layer_features[-1])
            classifier_output = self.model[-1].extra_layer(features)
            return classifier_output, total_loss, labels_list[-1]

        return layer_features[-1], total_loss, labels_list[-1]

    def test_step(self, X, Y, mask=None):
        features_list = []
        masks_list = []
        
        y = Y.to(self.device_list[-1], non_blocking=True)
        features_list.append([X.to(self.device_list[0], non_blocking=True)])
        layers_num = len(self.model_config)

        # 初始化 masks_list，確保長度與層數一致
        for i in range(layers_num):
            if mask is not None:
                masks_list.append(mask.to(self.device_list[i]))
            else:
                masks_list.append(None)

        for i in range(layers_num):
            self.model[i].eval()
            
        for i in range(layers_num):
            output, _ = self.model[i](features_list[-1], masks_list[i])
            output = [t.to(self.device_list[i+1 if i+1 < layers_num else i], non_blocking=True) for t in output]
            features_list.append(output)
            
        return output[0], y

    def adaptive_test_step(self, X, Y, mask=None):
        self.patiencecount = 0
        classifier_output_pre = None
        classifier_outputs = []
        features_list = []
        masks_list = []
        
        features_list.append([X.to(self.device_list[0], non_blocking=True)])
        
        layers_num = len(self.model_config)

        # 初始化 masks_list，確保長度與層數一致
        for i in range(layers_num):
            if mask is not None:
                masks_list.append(mask.to(self.device_list[i]))
            else:
                masks_list.append(None)

        for i in range(layers_num):
            self.model[i].eval()

        for i in range(layers_num):            
            output, _, classifier_output = self.model[i](features_list[-1], masks_list[i])
            output = [t.to(self.device_list[i+1 if i+1 < layers_num else i], non_blocking=True) for t in output]
            features_list.append(output)
            classifier_outputs.append(classifier_output)
        
            if i != 0:                
                self.patiencecount += self.AdaptiveCondition(classifier_outputs[i-1], classifier_outputs[i])
                if self.patiencecount >= self.costhreshold:
                    break
            
        y = Y.to(self.device_list[i], non_blocking=True)
        return classifier_output, i, y
                
    def AdaptiveCondition(self, prelayer, nowlayer):
        prelayer_maxarg = torch.argmax(prelayer, dim=1)
        nowlayer = nowlayer.to(prelayer.device)
        nowlayer_maxarg = torch.argmax(nowlayer, dim=1)
        cossimi = torch.mean(self.cos(prelayer, nowlayer))
        condition_maxarg = torch.all(prelayer_maxarg == nowlayer_maxarg)
        
        if condition_maxarg and cossimi > self.cosinesimthreshold:
            return  1
        return 0
    
    def _init_optimizers(self, optimizer_fn, optimizer_param):
        self.optimizers = list()
        for i in range(len(self.model_config)):
            self.optimizers.append(Optimizer(
                self.model[i].parameters(),
                optimizer_function=optimizer_fn,
                optimizer_parameters=optimizer_param
            ))
            
    def _init_schedulers(self, scheduler_fn, scheduler_param):
        if scheduler_fn != None:
            
            if not scheduler_param:
                raise ValueError("lr_scheduler need paramaters")
                
            self.schedulers = list()
            for i in range(len(self.model_config)):
                self.schedulers.append(LR_Scheduler(
                    self.optimizers[i].optimizer,
                    scheduler_fn=scheduler_fn,
                    scheduler_param=scheduler_param
                ))

    def _loss_backward_update(self, layer, optimizer, hidden_state, true_y):
        loss, hidden_state = layer.loss_cal(hidden_state, true_y)
        loss.backward()
        optimizer.step()
        return loss
    
    def scheduler_step(self, *arg):
        for idx, scheduler in enumerate(self.schedulers):
            scheduler.step(*arg)
            
    def get_lr(self):
        return self.optimizers[-1].get_learning_rate()

    def get_config(self):
        return self.model_config

    def MovetoSingle(self, device):
        layers_num = len(self.model_config)
        for i in range(layers_num):
            self.model[i] = self.model[i].to(device)
            self.device_list[i]= device

    def forward(self, X, Y, mask=None):
        if self.training:
            return self.train_step(X, Y, mask)
        else:
            return self.test_step(X, Y, mask)
