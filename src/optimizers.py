
# Minimal RAdam and Lookahead implementations to match original script usage.
# If you prefer, you can swap to torch.optim.AdamW instead.

import math
from torch.optim.optimizer import Optimizer

class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(RAdam, self).__init__(params, defaults)

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
                    raise RuntimeError('RAdam does not support sparse gradients')
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = grad.new_zeros(p.data.size())
                    state['exp_avg_sq'] = grad.new_zeros(p.data.size())

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                N_sma_max = 2 / (1 - beta2) - 1
                step = state['step']
                beta2_t = beta2 ** step
                N_sma = N_sma_max - 2 * step * beta2_t / (1 - beta2_t)

                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])

                if N_sma >= 5:
                    rect = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2))
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p.data.addcdiv_(exp_avg, denom, value=-group['lr'] * rect / (1 - beta1 ** step))
                else:
                    p.data.add_(exp_avg, alpha=-group['lr'] / (1 - beta1 ** step))
        return loss

class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = optimizer.param_groups
        self.state = {}

        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = {'slow_param': p.data.clone()}

        self._step = 0

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self._step += 1

        if self._step % self.k != 0:
            return loss

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                slow = self.state[p]['slow_param']
                slow.add_(p.data - slow, alpha=self.alpha)
                p.data.copy_(slow)

        return loss

    def zero_grad(self):
        self.optimizer.zero_grad()
