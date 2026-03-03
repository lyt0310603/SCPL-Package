import torch
import torch.nn as nn
import torch.optim as optim
import threading


class Optimizer:
    def __init__(self, model_parameters, optimizer_function, optimizer_parameters):
        """優化器包裝類，支援自訂優化器與參數。"""
        if isinstance(optimizer_function, str):
            if optimizer_parameters:
                self.optimizer = LARS(model_parameters, **optimizer_parameters)
            else:
                self.optimizer = LARS(model_parameters, lr=1e-3)
        else:
            if optimizer_parameters:
                self.optimizer = optimizer_function(model_parameters, **optimizer_parameters)
            else:
                self.optimizer = optimizer_function(model_parameters, lr=1e-3)
    
    def zero_grad(self):
        self.optimizer.zero_grad()
        
    def get_learning_rate(self):
        """取得目前學習率。"""
        return self.optimizer.param_groups[0]['lr']


class LR_Scheduler:
    def __init__(self, opt, scheduler_fn, scheduler_param):
        self.scheduler = scheduler_fn(opt, **scheduler_param)
        
    def step(self, *arg):
        self.scheduler.step(*arg)


# LARS Optimizer
class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])
                
    @staticmethod
    def exclude_bias_and_norm(parameter):
        """判斷是否為 bias 或 normalization 參數。"""
        return parameter.ndim == 1


class CPUThread(threading.Thread):
    def __init__(self, target=None, args=(), **kwargs):
        """CPU 執行緒包裝，支援取得結果與例外處理。"""
        super(CPUThread, self).__init__()
        self._target = target
        self._args = args
        self._kwargs = kwargs

    def run(self):
        if self._target == None:
            return
        try:
            self.__result__ = self._target(*self._args, **self._kwargs)
        except Exception as e:  
            self.__result__ = None
    
    def get_result(self):
        """取得執行緒運算結果，若失敗則丟出例外。"""
        self.join()
        if self.__result__ is None:
            raise ValueError(f"Here are some error in loss backward, please check your model structure")
        else:
            return self.__result__




