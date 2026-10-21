from __future__ import annotations

import math

import torch


class _TriangularSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, width: float):
        ctx.save_for_backward(x)
        ctx.width = width
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        width = ctx.width
        surrogate = (1.0 - x.abs() / width).clamp_min(0.0) / width
        return grad_output * surrogate, None


def surrogate_spike(x: torch.Tensor, width: float = 1.0) -> torch.Tensor:
    return _TriangularSpike.apply(x, width)


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))
