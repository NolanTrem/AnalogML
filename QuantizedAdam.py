import torch
import numpy as np
import math

class QuantizedAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, bits=4):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, bits=bits)
        super(QuantizedAdam, self).__init__(params, defaults)
        self.bits = bits

    def quantize_weights(self, weights):
        quantized_weights = torch.zeros_like(weights)
        max_val = 2 ** (self.bits - 1) - 1
        for i in range(1, self.bits + 1):
            mask = (weights >= 2 ** (i - self.bits - 1)) & (weights < 2 ** (i - self.bits))
            quantized_weights[mask] = 2 ** (i - self.bits - 1)
        quantized_weights = torch.clamp(quantized_weights, -max_val, max_val)
        return quantized_weights

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('QuantizedAdam does not support sparse gradients')
                amsgrad = group['amsgrad']

                # Normalize the gradients
                grad_norm = torch.norm(grad)
                if grad_norm > 0:
                    grad /= grad_norm

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / (bias_correction1 + group['eps'])


                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Quantize the weights to the nearest E24 resistor values
                p.data = self.quantize_weights(p.data)

        return loss
