import torch
from torch.optim.optimizer import Optimizer, required


class ProxSGD(Optimizer):
    """Implementation of Prox-SGD algorithm, proposed in
        "Prox-SGD: Training Structured Neural Networks under Regularization and Constraints", ICLR 2020.
        https://openreview.net/forum?id=HygpthEtvr

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        epsilon (float, optional): initial learning rate (default: 1e-3)
        epsilon_decay (float, optional): decay factor used for decaying the learning rate over time
        rho (float, optional): initial rho value used for computing running averages of gradient (default:  0.9)
        rho_decay (float, optional): decay factor used for decaying the momentum term over time
        beta (float, optional): beta coefficient used for computing
            running averages of the square of the gradient (default:  0.999)
        delta (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        mu (float, optional): regularization constant mu (for L1 penalty) (default: 1e-4)
        step_offset (integer, optional): offset in time for decaying the learning rate as well as the momentum term
            (default: 4)
    """

    def __init__(self, params, epsilon=0.06, epsilon_decay=0.5, rho=0.9, rho_decay=0.5, step_offset=4, beta=0.999, delta=1e-8,
                 mu=None, clip_bounds=(None,None)):
        if not 0.0 <= epsilon:
            raise ValueError("Invalid initial learning rate: {}".format(epsilon))
        if not 0.0 <= epsilon_decay < 1.0:
            raise ValueError("Invalid epsilon decay parameter: {}".format(epsilon_decay))
        if not 0.0 <= rho_decay < 1.0:
            raise ValueError("Invalid rho decay parameter: {}".format(rho_decay))
        if not 0.0 <= rho < 1.0:
            raise ValueError("Invalid rho parameter: {}".format(rho))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta parameter: {}".format(beta))
        if not 0.0 <= delta:
            raise ValueError("Invalid delta value: {}".format(delta))

        defaults = dict(epsilon=epsilon, epsilon_decay=epsilon_decay, rho=rho, rho_decay=rho_decay, beta=beta, delta=delta,
                        mu=mu, step_offset=step_offset, clip_bounds=clip_bounds)
        super(ProxSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ProxSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for x in group['params']:
                if x.grad is None:
                    continue
                grad = x.grad.data

                if grad.is_sparse:
                    raise RuntimeError('Prox-SGD does not support sparse gradients')

                state = self.state[x]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['v_t'] = torch.zeros_like(x.data)
                    # Exponential moving average of squared gradient values
                    state['r_t'] = torch.zeros_like(x.data)

                v_t, r_t = state['v_t'], state['r_t']

                state['step'] += 1
                

                epsilon_t = group['epsilon'] / ((state['step'] + group['step_offset'])**group['epsilon_decay'])
                rho_t     = group['rho'] / ((state['step'] + group['step_offset'])**group['rho_decay'])

                # Decay the first and second moment running average coefficient
                v_t.mul_(1 - rho_t).add_(rho_t, grad)
                r_t.mul_(group['beta']).addcmul_(1 - group['beta'], grad, grad)

                bias_correction = 1 - group['beta'] ** (state['step']+1)

                tau_t = (r_t / bias_correction).sqrt().add_(group['delta'])
                x_tmp = x - v_t / tau_t

                if group['mu'] is not None:
                    mu_t  = group['mu'] / tau_t
                    x_hat = torch.max(x_tmp - mu_t, torch.zeros_like(x_tmp)) - torch.max(-x_tmp - mu_t, torch.zeros_like(x_tmp))
                else:
                    x_hat = x_tmp

                low, up = group['clip_bounds']
                if low is not None or up is not None:
                    x_hat = x_hat.clamp_(low,up)

                x.data.add_(epsilon_t, x_hat - x)

        return loss