import torch
import torch.nn as nn

from BasicBlock import BasicBlock
from utils import Optimizer, LR_Scheduler, CPUThread


class SCPL_model(nn.Module):
    def __init__(self, custom_model, layer_balance, projector_type="i", loss_fn="CL", num_classes=None,
                 transform_funcs=None, gpu_list=None, device_distribution=None, 
                 optimizer_fn=torch.optim.Adam, optimizer_param={},
                 scheduler_fn: torch.optim.lr_scheduler=None, scheduler_param={},
                 multi_t=True, 
                 is_adaptive=False, patiencethreshold=1, cosinesimthreshold=0.8, classifier=None):
        super().__init__()
        self.multi_t = multi_t
        self.is_adaptive = is_adaptive
        
        self._model_check(custom_model, layer_balance, gpu_list, device_distribution, transform_funcs, loss_fn, num_classes, is_adaptive, classifier)
        self.model_config, self.device_list = self._get_layer_config(custom_model, layer_balance, 
                                                                      gpu_list, device_distribution, transform_funcs, 
                                                                      loss_fn, projector_type, num_classes, is_adaptive, classifier)
        self._build_model()
        self._init_optimizers(optimizer_fn, optimizer_param)
        self._init_schedulers(scheduler_fn, scheduler_param)

        if self.is_adaptive:
            self.test = self.adaptive_test_step
            self.patiencecount = patiencethreshold
            self.costhreshold = cosinesimthreshold
            self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        else:
            self.test = self.test_step
    
    # get the information of the user model
    def _get_layer_config(self, custom_model, layer_balance, gpu_list, device_distribution, transform_funcs, loss_fn, projector_type, num_classes, is_adaptive, classifier):
        layer_config = {}
        balance_idx = 0
        layer_idx = 0
        layers = []
        device = []

        for name, layer in custom_model.named_children():
            layers.append(layer)
            if len(layers) == layer_balance[balance_idx]:
                layer_config[layer_idx] = {"layer_list": layers.copy(),
                                           "device": gpu_list[device_distribution[layer_idx]] if gpu_list != None and device_distribution != None else 'cpu',
                                           "trans_func": transform_funcs[layer_idx] if transform_funcs != None else None,
                                           "projector_type": projector_type,
                                           "loss_fn":loss_fn if (layer_idx < len(layer_balance)-1 or self.is_adaptive) else nn.CrossEntropyLoss(),
                                           "num_classes": num_classes,
                                           "is_adaptive":is_adaptive,
                                           "classifier":classifier}
                device.append(gpu_list[device_distribution[layer_idx]] if gpu_list != None and device_distribution != None else 'cpu')
                layer_idx += 1
                balance_idx += 1
                layers.clear()

        return layer_config, device
    

    def _build_model(self):            
        self.model = []
        for idx, (k, v) in enumerate(self.model_config.items()):
            self.model.append(BasicBlock(**v).to(v['device']))
        
        self.model = torch.nn.Sequential(*self.model)

    # check the relation of model layer、balance and num of gpu
    def _model_check(self, custom_model, layer_balance, gpu_list, device_distribution, transform_funcs, loss_fn, num_classes, is_adaptive, classifier):
        if len(custom_model) != sum(layer_balance):
            raise ValueError('Layers of model don\'t equal to balance')
        if device_distribution != None:
            if len(layer_balance) != len(device_distribution):
                raise ValueError('Cannot distribute all blocks, please check the length of balance and distribute')
                
            if gpu_list != None:
                for i in device_distribution:
                    if i >= len(gpu_list):
                        raise ValueError('Cannot find gpu to distribute block')
        if transform_funcs != None:
            if len(transform_funcs) != len(layer_balance):
                raise ValueError('Cannot distribute transform_funcs, please check the length of transform_funcs')

        if loss_fn == "DeInfo" and num_classes == None:
            raise ValueError('DeInfo Loss need pass class nums')

    # test to print
    def print(self):
        for layer in self.model:
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    print(name, param.data)

    def train_step(self, X, Y):
        features_list = []
        labels_list = []
        total_loss = 0
        layer_losses = []
        layer_features = []
        threads = []
        layers_num = len(self.model_config)
        for i in range(0, layers_num):
            labels_list.append(Y.to(self.device_list[i]))
            
        features_list.append([X.to(self.device_list[0], non_blocking=True)])        
        
        for i in range(layers_num):
            self.model[i].train()

        for optimizer in self.optimizers:
            optimizer.zero_grad()
            
        for i in range(0, layers_num):
            output, hidden_state = self.model[i](features_list[-1])
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
            
        return layer_features[-1], total_loss

    def test_step(self, X, Y):
        features_list = list()
        
        y = Y.to(self.device_list[-1], non_blocking=True)
        features_list.append([X.to(self.device_list[0], non_blocking=True)])
        
        layers_num = len(self.model_config)

        for i in range(0, layers_num):
            self.model[i].eval()
            
        for i in range(0, layers_num):
            output, _ = self.model[i](features_list[-1])
            output = [t.to(self.device_list[i+1 if i+1 < layers_num else i], non_blocking=True) for t in output]
            features_list.append(output)
            
        return output[0], y

    def adaptive_test_step(self, X, Y):
        self.patiencecount = 0
        classifier_output_pre = None
        classifier_outputs = []
        features_list = list()
        
        y = Y.to(self.device_list[-1], non_blocking=True)
        features_list.append([X.to(self.device_list[0], non_blocking=True)])
        
        layers_num = len(self.model_config)

        for i in range(0, layers_num):
            self.model[i].eval()

        for i in range(0, layers_num):
            output, _, classifier_output = self.model[i](features_list[-1])
            output = [t.to(self.device_list[i+1 if i+1 < layers_num else i], non_blocking=True) for t in output]
            features_list.append(output)
            classifier_outputs.append(classifier_output)
        return classifier_output, i, y, classifier_outputs
                
    def AdaptiveCondition(self, prelayer, nowlayer):
        prelayer_maxarg = torch.argmax(prelayer)
        nowlayer = nowlayer.to(prelayer.device)
        nowlayer_maxarg = torch.argmax(nowlayer)
        cossimi = torch.mean(self.cos(prelayer , nowlayer))
        if nowlayer_maxarg == prelayer_maxarg and cossimi > self.costhreshold:
            return  1
        
        return 0
    
    def _init_optimizers(self, optimizer_fn, optimizer_param):
        self.optimizers = list()
        for i in range(len(self.model_config)):
            self.optimizers.append(Optimizer(
                self.model[i].parameters(),
                optimizer_fn=optimizer_fn,
                optimizer_param=optimizer_param
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
        return self.optimizers[-1].get_lr()

    def get_config(self):
        return self.model_config

    def forward(self, X, Y):
        if self.training:
            return self.train_step(X, Y)
        else:
            return self.test(X, Y)
