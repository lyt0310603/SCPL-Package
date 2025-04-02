from collections import OrderedDict

import torch
import torch.nn as nn

from BasicBlock import BasicBlock
from utils import Optimizer, LR_Scheduler, CPUThread
from itertools import chain

import numpy as np


class SCPL_model(nn.Module):
    def __init__(self, user_model, user_balance, proj_type="i", loss_fn="CL", n_classes=None,
                 trans_fucs=None, user_gpu=None, user_distribute=None, 
                 optimizer_fn=torch.optim.Adam, optimizer_param={},
                 scheduler_fn: torch.optim.lr_scheduler=None, scheduler_param={},
                 multi_t=True, 
                 adaptive=False, patiencethreshold=1, cosinesimthreshold=0.8, classifier=None):
        super().__init__()
        self.multi_t = multi_t
        self.adaptive = adaptive
        
        self._model_check(user_model, user_balance, user_gpu, user_distribute, trans_fucs, loss_fn, n_classes, adaptive, classifier)
        self.model_config, self.model_device = self._get_layer_config(user_model, user_balance, 
                                                                      user_gpu, user_distribute, trans_fucs, 
                                                                      loss_fn, proj_type, n_classes, adaptive, classifier)
        self._make_model()
        self._init_optimizers(optimizer_fn, optimizer_param)
        self._init_schedulers(scheduler_fn, scheduler_param)

        if self.adaptive:
            self.test = self.adaptive_test_step
            self.countthreshold = patiencethreshold
            self.costhreshold = cosinesimthreshold
            self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        else:
            self.test = self.test_step
    
    # get the information of the user model
    def _get_layer_config(self, user_model, user_balance, user_gpu, user_distribute, trans_fucs, loss_fn, proj_type, n_classes, adaptive, classifier):
        layer_config = {}
        balance_idx = 0
        layer_idx = 0
        layer_list = []
        device = []

        for name, layer in user_model.named_children():
            layer_list.append(layer)
            if len(layer_list) == user_balance[balance_idx]:
                layer_config[layer_idx] = {"layer_list": layer_list.copy(),
                                           "device": user_gpu[user_distribute[layer_idx]] if user_gpu != None and user_distribute != None else 'cpu',
                                           "trans_func": trans_fucs[layer_idx] if trans_fucs != None else None,
                                           "proj_type": proj_type,
                                           "loss_fn":loss_fn if (layer_idx < len(user_balance)-1 or self.adaptive) else nn.CrossEntropyLoss(),
                                           "n_classes": n_classes,
                                           "adaptive":adaptive,
                                           "classifier":classifier}
                device.append(user_gpu[user_distribute[layer_idx]] if user_gpu != None and user_distribute != None else 'cpu')
                layer_idx += 1
                balance_idx += 1
                layer_list.clear()

        return layer_config, device
    

    def _make_model(self):            
        self.model = []
        for idx, (k, v) in enumerate(self.model_config.items()):
            self.model.append(BasicBlock(**v).to(v['device']))
        
        self.model = torch.nn.Sequential(*self.model)

    # check the relation of model layer、balance and num of gpu
    def _model_check(self, user_model, user_balance, user_gpu, user_distribute, trans_fucs, loss_fn, n_classes, adaptive, classifier):
        if len(user_model) != sum(user_balance):
            raise ValueError('Layers of model don\'t equal to balance')
        if user_distribute != None:
            if len(user_balance) != len(user_distribute):
                raise ValueError('Cannot distribute all blocks, please check the length of balance and distribute')
                
            if user_gpu != None:
                for i in user_distribute:
                    if i >= len(user_gpu):
                        raise ValueError('Cannot find gpu to distribute block')
        if trans_fucs != None:
            if len(trans_fucs) != len(user_balance):
                raise ValueError('Cannot distribute trans_fucs, please check the length of trans_fucs')

        if loss_fn == "DeInfo" and n_classes == None:
            raise ValueError('DeInfo Loss need pass class nums')

    # test to print
    def print(self):
        for layer in self.model:
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    print(name, param.data)

    def train_step(self, X, Y):
        Xs = []
        Ys = []
        all_loss = 0
        layer_loss = []
        layer_fs = []
        tasks = []
        layers_num = len(self.model_config)
        for i in range(0, layers_num):
            Ys.append(Y.to(self.model_device[i]))
            
        Xs.append([X.to(self.model_device[0], non_blocking=True)])        
        
        for i in range(layers_num):
            self.model[i].train()

        for opt in self.opts:
            opt.zero_grad()
            
        for i in range(0, layers_num):
            result, hidden = self.model[i](Xs[-1])
            layer_fs.append(hidden)
            
            args = (self.model[i], self.opts[i].optimizer, hidden, Ys[i])
            
            if not self.multi_t:
                layer_loss.append(self._loss_backward_update(*args))
            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()
                
            detached_list = [t.detach().to(self.model_device[i+1 if i+1 < layers_num else i], non_blocking=True) for t in result]
            Xs.append(detached_list)
            
        
        if self.multi_t:
            for t in range(len(tasks)):
                    all_loss += tasks[t].get_result().item()
        else:
            for loss in layer_loss:
                all_loss += loss.item()
            
        return layer_fs[-1], all_loss

    def test_step(self, X, Y):
        Xs = list()
        
        y = Y.to(self.model_device[-1], non_blocking=True)
        Xs.append([X.to(self.model_device[0], non_blocking=True)])
        
        layers_num = len(self.model_config)

        for i in range(0, layers_num):
            self.model[i].eval()
            
        for i in range(0, layers_num):
            result, _ = self.model[i](Xs[-1])
            result = [t.to(self.model_device[i+1 if i+1 < layers_num else i], non_blocking=True) for t in result]
            Xs.append(result)
            
        return result[0], y

    def adaptive_test_step(self, X, Y):
        self.patiencecount = 0
        classifier_out_pre = None
        classifier_outs = []
        Xs = list()
        
        y = Y.to(self.model_device[-1], non_blocking=True)
        Xs.append([X.to(self.model_device[0], non_blocking=True)])
        
        layers_num = len(self.model_config)

        for i in range(0, layers_num):
            self.model[i].eval()

        for i in range(0, layers_num):
            result, _, classifier_out = self.model[i](Xs[-1])
            result = [t.to(self.model_device[i+1 if i+1 < layers_num else i], non_blocking=True) for t in result]
            Xs.append(result)
            classifier_outs.append(classifier_out)
        return classifier_out, i, y, classifier_outs
                
    def AdaptiveCondition(self, prelayer, nowlayer):
        prelayer_maxarg = torch.argmax(prelayer)
        nowlayer = nowlayer.to(prelayer.device)
        nowlayer_maxarg = torch.argmax(nowlayer)
        cossimi = torch.mean(self.cos(prelayer , nowlayer))
        if nowlayer_maxarg == prelayer_maxarg and cossimi > self.costhreshold:
            return  1
        
        return 0
    
    def _init_optimizers(self, optimizer_fn, optimizer_param):
        self.opts = list()
        for i in range(len(self.model_config)):
            self.opts.append(Optimizer(
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
                    self.opts[i].optimizer,
                    scheduler_fn=scheduler_fn,
                    scheduler_param=scheduler_param
                ))

    def _loss_backward_update(self, layer, optimizer, hat_y, true_y):
        loss, hat_y = layer.loss_cal(hat_y, true_y)
        loss.backward()
        optimizer.step()
        return loss
    
    def scheduler_step(self, *arg):
        for idx, scheduler in enumerate(self.schedulers):
            scheduler.step(*arg)
            
    def get_lr(self):
        return self.opts[-1].get_lr()

    def get_config(self):
        return self.model_config

    def forward(self, X, Y):
        if self.training:
            return self.train_step(X, Y)
        else:
            return self.test(X, Y)
