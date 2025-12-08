# ================================
# EXPERIMENTAL 
# ================================

import torch
from torch.optim import Optimizer, AdamW
from typing import Iterator, Optional, Callable, Tuple, Dict, Any


def get_optimizer(name: str, model_parameters, lr: float, weight_decay: float = 0.0, **kwargs):
    """
    Factory function to get optimizer by name.
    Supports both PyTorch built-in optimizers and custom ones.
    
    Special handling for EnhancedMuon: creates a hybrid optimizer that uses
    Muon for 2D weight matrices (except embeddings/outputs) and AdamW for 
    scalars/vectors/embeddings/outputs following the Muon paper recommendations.
    
    Args:
        name: Optimizer name (e.g., 'Adam', 'AdamW', 'SGD', 'EnhancedMuon')
        model_parameters: Model parameters to optimize (can be named_parameters() or parameters())
        lr: Learning rate
        weight_decay: Weight decay (L2 penalty)
        **kwargs: Additional optimizer-specific arguments
        
    Returns:
        Optimizer instance or HybridOptimizer for EnhancedMuon
    """
    if name == 'EnhancedMuon':
        return create_hybrid_muon_optimizer(model_parameters, lr, weight_decay, **kwargs)
    
    params_list = list(model_parameters)
    if params_list and isinstance(params_list[0], tuple):
        params_list = [p for _, p in params_list]
    
    if hasattr(torch.optim, name):
        optimizer_cls = getattr(torch.optim, name)
        return optimizer_cls(params_list, lr=lr, weight_decay=weight_decay, **kwargs)
    
    if name in globals():
        optimizer_cls = globals()[name]
        return optimizer_cls(params_list, lr=lr, weight_decay=weight_decay, **kwargs)
    
    raise ValueError(f"Optimizer '{name}' not found in torch.optim or custom optimizers")


def create_hybrid_muon_optimizer(model_parameters, lr: float, weight_decay: float = 0.0, **kwargs):
    """
    Create a hybrid optimizer following Muon paper recommendations:
    - Use AdamW for: scalars (0D), vectors (1D), embeddings, and output/classifier layers
    - Use Muon for: 2D weight matrices (except embeddings and output layers)
    
    The function attempts to identify embedding and output layers by name patterns.
    Common patterns: 'embed', 'embedding', 'classifier', 'head', 'fc_out', 'output', 'readout'
    
    Returns:
        A HybridOptimizer wrapper that manages both optimizers
    """
    muon_params = []
    adamw_params = []
    
    for name, param in model_parameters:
        name_lower = name.lower()
        
        is_embedding = any(pattern in name_lower for pattern in 
                          ['embed', 'embedding', 'token', 'position'])
        is_output = any(pattern in name_lower for pattern in 
                       ['classifier', 'head', 'fc_out', 'output', 'readout', 'mlp.layers.3'])
        
        if len(param.shape) >= 2 and not is_embedding and not is_output:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    
    optimizers = []
    
    if muon_params:
        muon_optimizer = EnhancedMuon(muon_params, lr=lr, weight_decay=weight_decay, **kwargs)
        optimizers.append(muon_optimizer)
    
    if adamw_params:
        adamw_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ['ns_iters', 'grad_clip_norm', 'track_stats', 'momentum']}
        if 'momentum' in kwargs:
            adamw_kwargs['betas'] = (kwargs['momentum'], 0.999)
        adamw_optimizer = torch.optim.AdamW(adamw_params, lr=lr, weight_decay=weight_decay, **adamw_kwargs)
        optimizers.append(adamw_optimizer)
    
    return HybridOptimizer(optimizers)


class CustomOptimizer(Optimizer):
    """
    Template for a custom optimizer.
    
    Replace this with your own implementation as needed.
    """
    
    def __init__(self, params, lr=1e-3, weight_decay=0.0, **kwargs):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, weight_decay=weight_decay, **kwargs)
        super(CustomOptimizer, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                p.data.add_(grad, alpha=-group['lr'])
        
        return loss


class HybridOptimizer(Optimizer):
    """
    Wrapper for managing multiple optimizers (e.g., Muon for 2D params + AdamW for 1D params).
    Inherits from Optimizer to be compatible with PyTorch LR schedulers.
    """
    def __init__(self, optimizers):
        self.optimizers = optimizers
        all_params = []
        for opt in optimizers:
            for group in opt.param_groups:
                all_params.extend(group['params'])
        
        defaults = {}
        if optimizers:
            defaults = optimizers[0].defaults.copy()
        super(HybridOptimizer, self).__init__(all_params, defaults)
        
        self._param_groups = []
        for opt in optimizers:
            self._param_groups.extend(opt.param_groups)
    
    @property
    def param_groups(self):
        """Return combined param_groups from all optimizers."""
        return self._param_groups
    
    @param_groups.setter
    def param_groups(self, value):
        self._param_groups = value
        idx = 0
        for opt in self.optimizers:
            num_groups = len(opt.param_groups)
            opt.param_groups = value[idx:idx + num_groups]
            idx += num_groups
    
    def zero_grad(self, set_to_none: bool = False):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)
    
    def step(self, closure=None):
        loss = None
        for optimizer in self.optimizers:
            loss = optimizer.step(closure)
        return loss
    
    def state_dict(self):
        return {
            'optimizers': [optimizer.state_dict() for optimizer in self.optimizers],
            'param_groups': self._param_groups
        }
    
    def load_state_dict(self, state_dict):
        if 'optimizers' in state_dict:
            for optimizer, opt_state in zip(self.optimizers, state_dict['optimizers']):
                optimizer.load_state_dict(opt_state)
        if 'param_groups' in state_dict:
            self._param_groups = state_dict['param_groups']


class EnhancedMuon(Optimizer):
    """
    Enhanced implementation of the Muon optimization algorithm with Nesterov momentum.
    
    Following the Muon paper recommendations:
    - Uses Nesterov-style momentum by default (empirically better than standard SGD momentum)
    - Should be applied only to 2D weight matrices (not embeddings or output layers)
    - For transformers, apply to Q, K, V separately rather than combined QKV layers
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-4)
        ns_iters (int, optional): number of Newton-Schulz iterations (default: 5)
        momentum (float, optional): momentum factor (default: 0.9)
        weight_decay (float, optional): weight decay coefficient (default: 0)
        grad_clip_norm (float, optional): gradient clipping norm (default: None)
        track_stats (bool, optional): whether to track optimization statistics (default: False)
        nesterov (bool, optional): whether to use Nesterov momentum (default: True)
    """
    
    def __init__(self, 
                params: Iterator[torch.nn.Parameter], 
                lr: float = 1e-4,
                ns_iters: int = 5, 
                momentum: float = 0.9, 
                weight_decay: float = 0,
                grad_clip_norm: Optional[float] = None,
                track_stats: bool = False,
                nesterov: bool = True):
        
        defaults = dict(
            lr=lr, 
            ns_iters=ns_iters, 
            momentum=momentum, 
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            track_stats=track_stats,
            nesterov=nesterov
        )
        super(EnhancedMuon, self).__init__(params, defaults)
        
        self.global_stats = {
            'update_magnitudes': [],
            'gradient_norms': []
        }
    
    def newton_schulz_orthogonalize(self, X: torch.Tensor, num_iters: int) -> torch.Tensor:
        """
        Enhanced Newton-Schulz orthogonalization with stability checks.
        """
        if torch.isnan(X).any() or torch.isinf(X).any():
            return torch.eye(X.shape[0], X.shape[1], device=X.device)
        
        norm = torch.norm(X, p='fro')
        if norm < 1e-8:
            return X
        
        X = X / norm
        
        for i in range(num_iters):
            X_T_X = torch.matmul(X.T, X)
            X_new = (3 * X - torch.matmul(X, X_T_X)) / 2
            
            if torch.norm(X_new - X, p='fro') < 1e-6:
                X = X_new
                break
                
            X = X_new
            
        return X
    
    def get_dimension_scaling(self, shape: Tuple[int, ...]) -> float:
        """
        Calculate the appropriate dimension scaling factor for different parameter shapes.
        
        For matrices (linear layers), this is sqrt(d_in * d_out).
        For other parameter types, we use appropriate heuristics.
        
        Args:
            shape (tuple): Shape of the parameter tensor
            
        Returns:
            float: Scaling factor
        """
        if len(shape) == 2:  # Linear layer weights
            d_in, d_out = shape
            return (d_in * d_out) ** 0.5
        elif len(shape) == 1:
            return shape[0] ** 0.5
        elif len(shape) == 4:
            c_out, c_in, k_h, k_w = shape
            return (c_in * c_out * k_h * k_w) ** 0.5
        else:
            return torch.prod(torch.tensor(shape)).float() ** 0.5
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Performs a single optimization step with enhanced capabilities.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            ns_iters = group['ns_iters']
            momentum_factor = group['momentum']
            weight_decay = group['weight_decay']
            grad_clip_norm = group['grad_clip_norm']
            track_stats = group['track_stats']
            
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(group['params'], grad_clip_norm)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.clone()
                
                state = self.state[p]
                
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                    state['step'] = 0
                    state['update_history'] = [] if track_stats else None
                
                state['step'] += 1
                
                momentum_buffer = state['momentum_buffer']
                momentum_buffer.mul_(momentum_factor).add_(grad, alpha=1.0)
                
                if group['nesterov']:
                    grad_to_use = grad + momentum_factor * momentum_buffer
                else:
                    grad_to_use = momentum_buffer
                
                grad_norm = torch.norm(grad, p='fro').item()
                if track_stats:
                    self.global_stats['gradient_norms'].append(grad_norm)
                
                if len(p.shape) >= 2:
                    original_shape = p.shape
                    if len(p.shape) > 2:
                        p_flat = p.reshape(p.shape[0], -1)
                        grad_flat = grad_to_use.reshape(grad_to_use.shape[0], -1)
                    else:
                        p_flat = p
                        grad_flat = grad_to_use
                    
                    ortho_grad = self.newton_schulz_orthogonalize(grad_flat, ns_iters)
                    
                    dim_scaling = self.get_dimension_scaling(original_shape)
                    
                    buffer_norm = torch.norm(grad_flat, p='fro')
                    if buffer_norm > 1e-8:
                        scaling = dim_scaling / buffer_norm
                        update = ortho_grad * scaling
                        
                        # Reshape back if needed
                        if len(p.shape) > 2:
                            update = update.reshape(original_shape)
                        
                        # Apply the update
                        p.add_(update, alpha=-lr)
                        
                        # Apply decoupled weight decay
                        if weight_decay != 0:
                            p.mul_(1 - lr * weight_decay)
                        
                        # Track update magnitude
                        if track_stats:
                            update_mag = torch.norm(update, p='fro').item() * lr
                            state['update_history'].append(update_mag)
                            self.global_stats['update_magnitudes'].append(update_mag)
                
                else:
                    p.add_(grad_to_use, alpha=-lr)
        
        return loss
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Return optimization statistics for analysis.
        
        Returns:
            Dict[str, Any]: Dictionary containing tracked statistics
        """
        stats = {
            'global': self.global_stats,
            'parameters': {}
        }
        
        for group in self.param_groups:
            if group['track_stats']:
                for p in group['params']:
                    if p in self.state and 'update_history' in self.state[p]:
                        state = self.state[p]
                        stats['parameters'][id(p)] = {
                            'shape': p.shape,
                            'updates': state['update_history'],
                            'steps': state['step']
                        }
        
        return stats
