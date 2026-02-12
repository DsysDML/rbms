import torch
from torch.optim import Optimizer


class L1Regularization(torch.nn.Module):
    def __init__(self, optimizer: list[Optimizer], lambda_l1: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = optimizer
        self.lambda_l1 = lambda_l1

    def forward(self, input):
        for opt in self.optimizer:
            for p in opt.param_groups[0]["params"]:
                p.grad -= self.lambda_l1 * torch.sign(p)


class L2Regularization(torch.nn.Module):
    def __init__(self, optimizer: list[Optimizer], lambda_l2: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = optimizer
        self.lambda_l2 = lambda_l2

    def forward(self, input):
        for opt in self.optimizer:
            for p in opt.param_groups[0]["params"]:
                p.grad -= self.lambda_l2 * p


class ClipGradNorm(torch.nn.Module):
    def __init__(self, optimizer: list[Optimizer], max_grad_norm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm

    def forward(self, input):
        for opt in self.optimizer:
            torch.nn.utils.clip_grad_norm_(
                opt.param_groups[0]["params"], max_norm=self.max_grad_norm
            )


class NormalizeGrad(torch.nn.Module):
    def __init__(self, optimizer: list[Optimizer], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = optimizer

    def forward(self, input):
        for opt in self.optimizer:
            norm_grad = torch.nn.utils.get_total_norm(
                [p.grad for p in opt.param_groups[0]["params"] if p.grad is not None]
            )
            for p in opt.param_groups[0]["params"]:
                p.grad /= norm_grad


def build_pre_grad_update(
    optimizer: list[Optimizer],
    lambda_l1: float,
    lambda_l2: float,
    normalize_grad: bool,
    max_grad_norm: float,
    **kwargs,
):
    return torch.compile(
        torch.nn.Sequential(
            *[L1Regularization(optimizer=optimizer, lambda_l1=lambda_l1)]
            * (lambda_l1 > 0),
            *[L2Regularization(optimizer=optimizer, lambda_l2=lambda_l2)]
            * (lambda_l2 > 0),
            *[NormalizeGrad(optimizer=optimizer)] * normalize_grad,
            *[ClipGradNorm(optimizer=optimizer, max_grad_norm=max_grad_norm)]
            * (max_grad_norm > 0),
        )
    )
