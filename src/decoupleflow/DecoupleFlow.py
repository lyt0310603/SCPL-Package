import torch
import torch.nn as nn
import time
from .BasicBlock import BasicBlock, AdaptiveBasicBlock
from .utils import Optimizer, LR_Scheduler, CPUThread


class DecoupleFlow(nn.Module):
    """Split model layers across devices and train each block locally."""

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
        self.device_distribution = self._generate_device_distribution(device_map)
        self._validate_config(
            custom_model=custom_model,
            transform_funcs=transform_funcs,
            loss_fn=loss_fn,
            num_classes=num_classes,
            projector_type=projector_type,
            custom_projector=custom_projector,
            multi_t=multi_t,
            is_adaptive=is_adaptive,
            patiencethreshold=patiencethreshold,
            cosinesimthreshold=cosinesimthreshold,
            optimizer_param=optimizer_param,
            scheduler_fn=scheduler_fn,
            scheduler_param=scheduler_param,
            classifier=classifier,
        )
        self.model_config, self.device_list = self._get_layer_config(custom_model, transform_funcs, 
                                                                      loss_fn, projector_type, custom_projector, num_classes, classifier)
        self._build_model()
        self._init_optimizers(optimizer_fn, optimizer_param)
        self._init_schedulers(scheduler_fn, scheduler_param)

        self.training_step = self.train_step
        self.validation_step = self.test_step
        if self.is_adaptive:
            self.validation_step = self.adaptive_test_step
            self.patiencecount = patiencethreshold
            self.costhreshold = cosinesimthreshold
            self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            
    
    def _get_layer_config(self, custom_model, transform_funcs, loss_fn, projector_type, custom_projector, num_classes, classifier):
        """Build per-block configuration for model partitioning.

        Args:
            custom_model: User model composed of child layers.
            transform_funcs: Optional transform function per block.
            loss_fn: Loss setting or callable.
            projector_type: Projector mode string.
            custom_projector: Optional custom projector module.
            num_classes: Number of classes for class-dependent losses.
            classifier: Optional auxiliary classifier for adaptive mode.

        Returns:
            Tuple[dict, list]: Block configuration dict and block device list.
        """
        layer_config = {}
        balance_idx = 0
        layer_idx = 0
        layers = []
        device = []

        device_distribution = self.device_distribution

        for name, layer in custom_model.named_children():
            layers.append(layer)
            if len(layers) == device_distribution[layer_idx]['layers']:
                layer_config[layer_idx] = {"layer_list": layers.copy(),
                                           "device": device_distribution[layer_idx]['device'],
                                           "transform_func": transform_funcs[layer_idx] if transform_funcs != None else None,
                                           "projector_type": projector_type,
                                           "custom_projector": custom_projector,
                                           "loss_fn":loss_fn if (layer_idx < len(device_distribution)-1 or self.is_adaptive) else 'CE',
                                           "num_classes": num_classes}
                if self.is_adaptive:
                    layer_config[layer_idx]["classifier"] = classifier

                device.append(device_distribution[layer_idx]['device'])
                layer_idx += 1
                balance_idx += 1
                layers.clear()

        return layer_config, device

    def _generate_device_distribution(self, device_map):
        """Normalize user device_map into unified list format.

        Args:
            device_map: Either `dict[device, layers]` or list of
                `{"device": ..., "layers": ...}` mappings.

        Returns:
            Optional[list]: Normalized device distribution list.

        Raises:
            TypeError: If `device_map` has unsupported structure.
            KeyError: If required keys are missing in list-style items.
        """
        if device_map is None:
            return None

        device_distribution = []

        if isinstance(device_map, dict):
            for device, count in device_map.items():
                device_distribution.append({"layers": count, "device": device})
            return device_distribution

        if isinstance(device_map, list):
            for idx, item in enumerate(device_map):
                if not isinstance(item, dict):
                    raise TypeError(f"device_map[{idx}] must be a dict with keys 'device' and 'layers'")
                if "device" not in item:
                    raise KeyError(f"device_map[{idx}] missing required key: 'device'")

                if "layers" in item:
                    layers = item["layers"]
                else:
                    raise KeyError(f"device_map[{idx}] missing required key: 'layers'")

                device_distribution.append({"layers": layers, "device": item["device"]})
            return device_distribution

        raise TypeError("device_map must be dict or list")
    

    def _build_model(self):            
        self.model = []
        for idx, (k, v) in enumerate(self.model_config.items()):
            if self.is_adaptive:
                self.model.append(AdaptiveBasicBlock(**v).to(v['device']))
            else:
                self.model.append(BasicBlock(**v).to(v['device']))
        
        self.model = torch.nn.Sequential(*self.model)

    def _model_check(self, custom_model, transform_funcs, loss_fn, num_classes, is_adaptive, classifier):
        """Validate model partition settings before building blocks.

        Args:
            custom_model: User model to split.
            transform_funcs: Optional transform function list.
            loss_fn: Loss setting.
            num_classes: Number of classes.
            is_adaptive: Whether adaptive mode is enabled.
            classifier: Optional adaptive classifier.

        Raises:
            ValueError: If layer counts, devices, or loss args are invalid.
        """
        device_distribution = self.device_distribution
        if device_distribution is None:
            raise ValueError('device_map cannot be None')

        total_balance = sum([item["layers"] for item in device_distribution])
        if len(custom_model) != total_balance:
            raise ValueError(f'Layers of model don\'t equal to balance, {len(custom_model)} and {total_balance}')

        for item in device_distribution:
            device = item["device"]
            count = item["layers"]
            if count <= 0:
                raise ValueError(f'Layer balance for device {device} must be > 0.')
            if device is None:
                raise ValueError('Device is None, which means no device will be assigned to this layer.')

        if transform_funcs != None:
            if len(transform_funcs) != len(device_distribution):
                raise ValueError('Cannot distribute transform_funcs, please check the length of transform_funcs')

        if loss_fn == "DeInfo" and num_classes == None:
            raise ValueError('DeInfo Loss need pass class nums')
    
    def _validate_config(
        self,
        custom_model,
        transform_funcs,
        loss_fn,
        num_classes,
        projector_type,
        custom_projector,
        multi_t,
        is_adaptive,
        patiencethreshold,
        cosinesimthreshold,
        optimizer_param,
        scheduler_fn,
        scheduler_param,
        classifier,
    ):
        """Run full initialization validation in a single entry point."""
        self._type_check(
            loss_fn=loss_fn,
            transform_funcs=transform_funcs,
            num_classes=num_classes,
            projector_type=projector_type,
            custom_projector=custom_projector,
            multi_t=multi_t,
            is_adaptive=is_adaptive,
            patiencethreshold=patiencethreshold,
            cosinesimthreshold=cosinesimthreshold,
            optimizer_param=optimizer_param,
            scheduler_fn=scheduler_fn,
            scheduler_param=scheduler_param,
        )
        self._model_check(
            custom_model=custom_model,
            transform_funcs=transform_funcs,
            loss_fn=loss_fn,
            num_classes=num_classes,
            is_adaptive=is_adaptive,
            classifier=classifier,
        )
    
    def _type_check(
        self,
        loss_fn,
        transform_funcs,
        num_classes,
        projector_type,
        custom_projector,
        multi_t,
        is_adaptive,
        patiencethreshold,
        cosinesimthreshold,
        optimizer_param,
        scheduler_fn,
        scheduler_param,
    ):
        """Validate runtime parameter types for safer initialization.

        Raises:
            TypeError: If argument types do not match expected interfaces.
        """
        if not isinstance(loss_fn, str):
            raise TypeError("loss_fn must be a string: 'CL' | 'DeInfo' | 'CE'")
        if loss_fn not in {"CL", "DeInfo", "CE"}:
            raise TypeError("loss_fn must be one of: 'CL', 'DeInfo', 'CE'")

        if transform_funcs is not None:
            if not isinstance(transform_funcs, list):
                raise TypeError("transform_funcs must be a list of callables or None")
            for idx, fn in enumerate(transform_funcs):
                if fn is not None and not callable(fn):
                    raise TypeError(f"transform_funcs[{idx}] must be callable or None")

        if num_classes is not None and not isinstance(num_classes, int):
            raise TypeError("num_classes must be int or None")
        if isinstance(num_classes, int) and num_classes <= 0:
            raise TypeError("num_classes must be > 0")

        if not isinstance(projector_type, str):
            raise TypeError("projector_type must be a string")
        if projector_type == "c" and custom_projector is not None and not isinstance(custom_projector, nn.Module):
            raise TypeError("custom_projector must be torch.nn.Module when projector_type is 'c'")

        if not isinstance(multi_t, bool):
            raise TypeError("multi_t must be bool")
        if not isinstance(is_adaptive, bool):
            raise TypeError("is_adaptive must be bool")

        if not isinstance(patiencethreshold, int):
            raise TypeError("patiencethreshold must be int")
        if patiencethreshold < 1:
            raise TypeError("patiencethreshold must be >= 1")

        if not isinstance(cosinesimthreshold, (int, float)):
            raise TypeError("cosinesimthreshold must be int or float")

        if not isinstance(optimizer_param, dict):
            raise TypeError("optimizer_param must be dict")
        if scheduler_fn is not None and not isinstance(scheduler_param, dict):
            raise TypeError("scheduler_param must be dict when scheduler_fn is provided")

    def print(self):
        for layer in self.model:
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    print(name, param.data)

    def train_step(self, X, Y, mask=None):
        """Run one decoupled training step across all blocks.

        Args:
            X: Input batch.
            Y: Label batch.
            mask: Optional mask forwarded to each block.

        Returns:
            Tuple[torch.Tensor, float, torch.Tensor]: Final block feature/logits,
            aggregated loss value, and labels on final device.
        """
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
        """Run standard inference through all blocks.

        Args:
            X: Input batch.
            Y: Label batch.
            mask: Optional mask forwarded to each block.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Final output tensor and labels on
            last block device.
        """
        features_list = []
        masks_list = []
        
        y = Y.to(self.device_list[-1], non_blocking=True)
        features_list.append([X.to(self.device_list[0], non_blocking=True)])
        layers_num = len(self.model_config)

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
        """Run adaptive inference with early-exit check.

        Args:
            X: Input batch.
            Y: Label batch.
            mask: Optional mask forwarded to each block.

        Returns:
            Tuple[torch.Tensor, int, torch.Tensor]: Classifier output at stop
            layer, stop-layer index, and labels on that layer device.
        """
        self.patiencecount = 0
        classifier_output_pre = None
        classifier_outputs = []
        features_list = []
        masks_list = []
        
        features_list.append([X.to(self.device_list[0], non_blocking=True)])
        
        layers_num = len(self.model_config)

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
        """Evaluate whether adjacent classifier outputs are stable.

        Args:
            prelayer: Previous layer classifier logits.
            nowlayer: Current layer classifier logits.

        Returns:
            int: `1` if prediction class and cosine similarity pass threshold,
            otherwise `0`.
        """
        prelayer_maxarg = torch.argmax(prelayer, dim=1)
        nowlayer = nowlayer.to(prelayer.device)
        nowlayer_maxarg = torch.argmax(nowlayer, dim=1)
        cossimi = torch.mean(self.cos(prelayer, nowlayer))
        condition_maxarg = torch.all(prelayer_maxarg == nowlayer_maxarg)
        
        if condition_maxarg and cossimi > self.cosinesimthreshold:
            return  1
        return 0
    
    def _init_optimizers(self, optimizer_fn, optimizer_param):
        """Create optimizer wrappers for each block.

        Args:
            optimizer_fn: Optimizer class or special string selector.
            optimizer_param: Optimizer keyword arguments.
        """
        self.optimizers = list()
        for i in range(len(self.model_config)):
            self.optimizers.append(Optimizer(
                self.model[i].parameters(),
                optimizer_function=optimizer_fn,
                optimizer_parameters=optimizer_param
            ))
            
    def _init_schedulers(self, scheduler_fn, scheduler_param):
        """Create schedulers for each block optimizer.

        Args:
            scheduler_fn: Scheduler constructor.
            scheduler_param: Scheduler keyword arguments.

        Raises:
            ValueError: If scheduler is provided without parameters.
        """
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
        """Compute local loss, backpropagate, and update one block.

        Args:
            layer: Target block.
            optimizer: Optimizer for this block.
            hidden_state: Feature used for local loss.
            true_y: Ground-truth labels.

        Returns:
            torch.Tensor: Local loss tensor.
        """
        loss, hidden_state = layer.loss(hidden_state, true_y)
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
        """Dispatch to train or validation step based on module mode.

        Args:
            X: Input batch.
            Y: Label batch.
            mask: Optional mask batch.

        Returns:
            tuple: Output tuple from selected step function.
        """
        if self.training:
            return self.training_step(X, Y, mask)
        else:
            return self.validation_step(X, Y, mask)
