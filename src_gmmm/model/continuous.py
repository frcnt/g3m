from abc import ABC
from typing import Any, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model.distributions import DistributionGaussian
from ..nn.layers import AnalogBitsEmbedding, CDCDEmbedding


class SDE(ABC, nn.Module):

    def diffusion(self, t: torch.Tensor):
        raise NotImplementedError

    def forward_drift(self, t: torch.Tensor, zt: torch.Tensor):
        raise NotImplementedError

    def loc_scale(self, t: torch.Tensor):
        raise NotImplementedError

    def reverse_drift(
        self,
        t: torch.Tensor,
        zt: torch.Tensor,
        score: torch.Tensor,
        eta: Optional[torch.Tensor | float] = None,
    ):
        f = self.forward_drift(t, zt)
        g = self.diffusion(t)

        if eta is None:
            eta = g

        return f - 0.5 * (g**2 + eta**2) * score


class LinearLogSNRVPSDE(SDE):
    def __init__(self, log_snr_min: float = -10.0, log_snr_max: float = 10.0) -> None:
        super().__init__()
        self.register_buffer("_log_snr_min", torch.as_tensor(log_snr_min))
        self.register_buffer("_log_snr_max", torch.as_tensor(log_snr_max))

    def gamma(self, t: torch.Tensor):
        return self._log_snr_min + (self._log_snr_max - self._log_snr_min) * t

    def beta(self, t: torch.Tensor):
        dg_dt = self._log_snr_max - self._log_snr_min
        s2 = torch.sigmoid(self._log_snr_min + dg_dt * t)
        beta = s2 * dg_dt
        return beta

    def diffusion(self, t: torch.Tensor):
        return torch.sqrt(self.beta(t))

    def forward_drift(
        self,
        t: torch.Tensor,
        zt: torch.Tensor,
    ):
        return -0.5 * self.beta(t) * zt

    def loc_scale(self, t: torch.Tensor):
        gamma_t = self.gamma(t)
        loc = torch.sigmoid(-gamma_t) ** 0.5
        scale = torch.sigmoid(gamma_t) ** 0.5

        return loc, scale


class KarrasSDE(SDE):
    def __init__(
        self,
    ):
        super().__init__()

    def diffusion(self, t: torch.Tensor):
        return torch.sqrt(self.beta(t))

    def forward_drift(
        self,
        t: torch.Tensor,
        zt: torch.Tensor,
    ):
        return -0.5 * self.beta(t) * zt

    def loc_scale(self, t: torch.Tensor):
        loc = 1.0
        scale = t

        return loc, scale


class BaseDiffusion(ABC, nn.Module):
    def __init__(
        self,
        sde: SDE,
        distribution: DistributionGaussian,
    ):
        super().__init__()
        self.sde = sde
        self.distribution = distribution

    def loss_diffusion(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
        *args: Optional[Any],
    ):
        raise NotImplementedError

    def training_targets(
        self, t: torch.Tensor, x: torch.Tensor, index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a, b = self.sde.loc_scale(t)
        eps = self.distribution.sample(index)

        x_t = a * x + b * eps

        if self.parameterization == "eps":
            target = eps
        elif self.parameterization == "x0":
            target = x
        else:
            raise NotImplementedError

        return x_t, target

    def construct_score(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        pred: torch.Tensor,
    ):

        loc, scale = self.sde.loc_scale(t)

        if self.parameterization == "eps":
            score = -pred / scale
        elif self.parameterization == "x0":
            score = (loc * pred - x_t) / scale**2
        else:
            raise NotImplementedError

        return score

    @torch.inference_mode()
    def reverse_step(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        pred: torch.Tensor,
        dt: torch.Tensor,
        index: Optional[torch.Tensor] = None,
        **_,
    ):

        score = self.construct_score(t=t, x_t=x_t, pred=pred)

        drift_dt = self.sde.reverse_drift(t=t, zt=x_t, score=score) * dt
        diff_dt = (
            self.sde.diffusion(t)
            * self.distribution.sample(index)
            * torch.sqrt(torch.abs(dt))
        )

        return x_t + drift_dt + diff_dt

    @torch.inference_mode()
    def sample_prior(self, index: torch.Tensor):
        return self.distribution.sample(index)


class ContinuousDiffusion(BaseDiffusion):
    def __init__(
        self,
        sde: SDE,
        parameterization: Literal["eps", "x0"],
        distribution: DistributionGaussian,
    ):
        super(ContinuousDiffusion, self).__init__(sde=sde, distribution=distribution)
        self.parameterization = parameterization

    def loss_diffusion(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
        *args: Optional[Any],
    ):
        assert pred.shape == target.shape
        return F.mse_loss(pred, target)


class CategoricalDataContinuousDiffusion(BaseDiffusion):
    def __init__(
        self,
        sde: SDE,
        embedding: CDCDEmbedding,
        distribution: Optional[DistributionGaussian] = None,
    ):
        if distribution is None:
            distribution = DistributionGaussian(
                dim=embedding.embedding_dim, zero_cog=False
            )
        super().__init__(sde=sde, distribution=distribution)
        self.embedding = embedding

    def loss_diffusion(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
        *args: Optional[Any],
    ):
        return F.cross_entropy(pred, target)  # logits

    def training_targets(self, t: torch.Tensor, x: torch.Tensor, index: torch.Tensor):
        x_t, _ = super().training_targets(t=t, x=self.embedding.forward(x), index=index)
        return x_t, x

    def construct_score(self, t: torch.Tensor, x_t: torch.Tensor, pred: torch.Tensor):

        pred = self.embedding.expected_embedding(pred)
        return super().construct_score(t, x_t, pred)

    @property
    def parameterization(self):
        return "x0"


class AnalogBitsContinuousDiffusion(BaseDiffusion):

    def __init__(
        self,
        sde: SDE,
        embedding: AnalogBitsEmbedding,
        distribution: Optional[DistributionGaussian] = None,
        clamp_pred_in_reverse: Optional[Tuple[float, float]] = None,
    ):
        if distribution is None:
            distribution = DistributionGaussian(
                dim=embedding.embedding_dim, zero_cog=False
            )
        super().__init__(sde=sde, distribution=distribution)
        self.embedding = embedding
        self.clamp_pred_in_reverse = clamp_pred_in_reverse

    def loss_diffusion(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
        *args: Optional[Any],
    ):
        return F.mse_loss(pred, target)  # logits

    def training_targets(self, t: torch.Tensor, x: torch.Tensor, index: torch.Tensor):
        x_t, x_bits = super().training_targets(
            t=t, x=self.embedding.forward(x), index=index
        )
        return x_t, x_bits  # return bit version as target

    def construct_score(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        pred: torch.Tensor,
    ):
        if self.clamp_pred_in_reverse:
            assert self.parameterization == "x0"
            pred = torch.clamp(pred, *self.clamp_pred_in_reverse)
        return super().construct_score(t=t, x_t=x_t, pred=pred)

    @property
    def parameterization(self):
        return "x0"
