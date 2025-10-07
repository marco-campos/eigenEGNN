import torch
import torch.nn.functional as F
from torch.nn import HuberLoss, Module, SmoothL1Loss


def compute_relative_error(input, target, eps=1e-8):
    """Compute maximum relative error of each row of `input` with respect to
    `target`, except for entries where `target` has absolute values less than
    `eps`, in which case we return the absolute error instead."""

    abs_err = torch.abs(input - target)
    rel_err = torch.where(
        torch.abs(target) < eps, abs_err, abs_err / torch.abs(target))
    return torch.amax(rel_err, 1)


def count_correct_rel(out, target):
    REL_EPS = 1e-1
    return int((compute_relative_error(out, target) < REL_EPS).sum())


def count_correct_abs(out, target):
    ABS_EPS = 1e-2
    return int((torch.linalg.vector_norm(out - target, ord=2, dim=1)
                < ABS_EPS).sum())


def PSNR(out, target):
    mse = torch.mean((out - target)**2, dim=1)
    data_range = torch.amax(target, dim=1) - torch.amin(target, dim=1)
    return 20 * torch.log10(data_range) - 10 * torch.log10(mse)


def count_correct_psnr(out, target):
    return int((PSNR(out, target) > 20).sum())


class WeightedHuberLoss(Module):

    def __init__(self, reduction="mean", delta=1.0):
        super().__init__()
        self.reduction = reduction
        self.delta = delta

    def forward(self, output, target):
        criterion = HuberLoss(reduction=self.reduction, delta=self.delta)
        weights = torch.tensor([1 / (x + 1) for x in range(output.shape[1])],
                               device=output.device)
        weights = weights.repeat(output.shape[0], 1)
        loss = criterion(output * weights, target * weights)
        return loss


class WeightedSmoothL1Loss(Module):

    def __init__(self, reduction="mean", beta=1.0):
        super().__init__()
        self.reduction = reduction
        self.beta = beta
        self.criterion = SmoothL1Loss(reduction=self.reduction, beta=self.beta)

    def forward(self, output, target):
        torch.set_printoptions(profile="full")
        weights = torch.reciprocal(
            torch.arange(
                start=1,
                end=output.shape[1] + 1,
                dtype=torch.float,
                device=output.device,
            ))
        weights = weights.repeat(output.shape[0], 1)
        loss = self.criterion(output * weights, target * weights)
        return loss


class RelativeSmoothL1Loss(Module):

    def __init__(self, reduction="mean", beta=1.0, kappa=0.02):
        super().__init__()
        self.reduction = reduction
        self.beta = beta
        self.kappa = kappa
        self.criterion = SmoothL1Loss(reduction=self.reduction, beta=self.beta)

    def forward(self, output, target):
        loss1 = self.criterion(torch.exp(self.kappa * (output - target)),
                               torch.ones_like(target))
        loss2 = self.criterion(torch.exp(self.kappa * (target - output)),
                               torch.ones_like(target))
        return loss1 + loss2


def polar_loss(output, target, lmbda=2.0):
    output_norm = torch.linalg.norm(output, dim=1)
    target_norm = torch.linalg.norm(target, dim=1)
    return torch.stack(
        [
            (1 - torch.sum(F.normalize(output) * F.normalize(target), dim=1)),
            lmbda * torch.abs(output_norm - target_norm) /
            (output_norm + target_norm),
        ],
        dim=-1,
    )


def polar2_loss(output, target, lmbda=1.0):
    output_norm = torch.linalg.norm(output, dim=1)
    target_norm = torch.linalg.norm(target, dim=1)
    return torch.stack(
        [
            (1 -
             torch.sum(F.normalize(output) * F.normalize(target), dim=1)**2),
            lmbda * torch.abs(output_norm - target_norm) /
            (output_norm + target_norm),
        ],
        dim=-1,
    )


class PolarLoss(Module):

    def __init__(self, lmbda=2.0, reduction="mean"):
        super().__init__()
        self.lmbda = lmbda
        self.reduction = reduction
        if self.reduction == "mean":
            self.reduction_fn = torch.mean
        elif self.reduction == "sum":
            self.reduction_fn = torch.sum
        elif self.reduction == "none":
            self.reduction_fn = lambda x: x

    def forward(self, output, target):
        return self.reduction_fn(
            polar_loss(output, target, self.lmbda).sum(dim=-1))


class Polar2Loss(Module):

    def __init__(self, lmbda=2.0, reduction="mean"):
        super().__init__()
        self.lmbda = lmbda
        self.reduction = reduction
        if self.reduction == "mean":
            self.reduction_fn = torch.mean
        elif self.reduction == "sum":
            self.reduction_fn = torch.sum
        elif self.reduction == "none":
            self.reduction_fn = lambda x: x

    def forward(self, output, target):
        return self.reduction_fn(
            polar2_loss(output, target, self.lmbda).sum(dim=-1))


class SmoothPolarLoss(Module):

    def __init__(self, lmbda=1, beta=1.0, reduction="mean"):
        super().__init__()
        self.lmbda = lmbda
        self.beta = beta
        self.reduction = reduction

    def forward(self, output, target):
        output_norm = torch.linalg.norm(output, dim=1)
        target_norm = torch.linalg.norm(target, dim=1)
        loss_tensor = torch.stack(
            [
                (1 -
                 torch.sum(F.normalize(output) * F.normalize(target), dim=1)),
                self.lmbda * torch.abs(output_norm - target_norm) /
                (output_norm + target_norm),
            ],
            dim=-1,
        )
        return F.smooth_l1_loss(
            loss_tensor,
            torch.zeros_like(loss_tensor),
            reduction=self.reduction,
            beta=self.beta,
        )


class SoftmaxLoss(Module):

    def __init__(self, loss, *args, **kwargs):
        super().__init__()
        self.criterion = loss(*args, **kwargs)

    def forward(self, output, target):
        return self.criterion(F.softmax(output, dim=1), F.softmax(target,
                                                                  dim=1))


def RPD(output, target, eps=1e-8):
    return torch.sum(
        torch.abs(output - target) /
        (torch.abs(output) + torch.abs(target) + eps),
        dim=1,
    )


class RPDLoss(Module):

    def __init__(self, eps=1e-8, reduction="mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        if self.reduction == "mean":
            self.reduction_fn = torch.mean
        elif self.reduction == "sum":
            self.reduction_fn = torch.sum
        elif self.reduction == "none":
            self.reduction_fn = lambda x: x

    def forward(self, output, target):
        return self.reduction_fn(RPD(output, target, eps=self.eps))


def SmoothRPD(output, target, eps=1e-8):
    return torch.mean(
        ((output - target) / (torch.abs(output) + torch.abs(target) + eps))**2,
        dim=1)


class SmoothRPDLoss(Module):

    def __init__(self, eps=1e-8, reduction="mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        if self.reduction == "mean":
            self.reduction_fn = torch.mean
        elif self.reduction == "sum":
            self.reduction_fn = torch.sum
        elif self.reduction == "none":
            self.reduction_fn = lambda x: x

    def forward(self, output, target):
        return self.reduction_fn(SmoothRPD(output, target, eps=self.eps))


def inv_lap(pred, target):
    return torch.linalg.vector_norm(torch.abs(1 / pred - 1 / target) /
                                    (1 / target),
                                    ord=2,
                                    dim=1)


def l2_error(output, target):
    return torch.linalg.vector_norm(output - target, ord=2, dim=1)
