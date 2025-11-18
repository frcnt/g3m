import math
from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskingSchedule(ABC, nn.Module):
    def __init__(self, eps: float = 1e-4):
        super(MaskingSchedule, self).__init__()
        self.eps = eps

    def forward(self, t: torch.Tensor):
        # this function compute the signal-to-noise ratio (SNR)
        return torch.log(self.alpha(t) / (1.0 - self.alpha(t)))

    def dalpha(self, t):
        return (1.0 - 2 * self.eps) * self._dalpha(t)

    def alpha(self, t: torch.Tensor):
        return (1.0 - 2 * self.eps) * self._alpha(t) + self.eps

    def dgamma_times_alpha(self, t):
        return self.dalpha(t) / (1.0 - self.alpha(t))

    def _alpha(self, t):
        raise NotImplementedError

    def _dalpha(self, t):
        raise NotImplementedError


class CosineMaskingSchedule(MaskingSchedule):
    def _dalpha(self, t):
        return -math.pi / 2.0 * torch.sin(math.pi / 2.0 * (1.0 - t))

    def _alpha(self, t):
        return 1.0 - torch.cos(math.pi / 2.0 * (1.0 - t))


class PolyMaskingSchedule(MaskingSchedule):
    def __init__(self, exponent: float = 1.0, eps: float = 1e-4):
        super().__init__(eps=eps)
        self.exponent = exponent

    def _dalpha(self, t):
        return -self.exponent * t ** (self.exponent - 1.0)

    def _alpha(self, t):
        return 1.0 - t**self.exponent


class MaskingDiffusion(nn.Module):
    """
    NOTE: always consider $t$ to be in [0, 1].
    """

    def __init__(
        self,
        schedule: MaskingSchedule,
        vocab_size: int,
        simple_loss: bool = False,
    ):
        super(MaskingDiffusion, self).__init__()
        self.schedule = schedule

        self.register_buffer(
            "vocab_size", torch.as_tensor(vocab_size, dtype=torch.long)
        )
        self.simple_loss = simple_loss

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        """
        :param t:
        :param x: idx representation
        :return:
        """
        a = self.schedule.alpha(t)  # probability of being unmasked
        un_mask = torch.bernoulli(a).bool()
        out = torch.where(un_mask.view(-1), x, self.vocab_size)
        return out

    def training_targets(self, t: torch.Tensor, x: torch.Tensor, index: torch.Tensor):
        x_t = self.forward(t, x)
        return x_t, x

    def loss_diffusion(
        self, logits: torch.Tensor, x: torch.Tensor, t: torch.Tensor, x_t: torch.Tensor
    ):
        log_p = torch.log_softmax(logits, dim=-1)
        one_hot_x = F.one_hot(x, self.vocab_size)
        neg_cross_ent = one_hot_x * log_p
        neg_cross_ent = torch.where(one_hot_x.bool(), neg_cross_ent, 0.0)
        neg_cross_ent = torch.sum(neg_cross_ent, dim=-1, keepdim=True)

        mask = (
            x_t == self.vocab_size
        ).float()  # only loss for the tokens in x_t that were masked
        masked_neg_cross_ent = mask[..., None] * neg_cross_ent  # [n_nodes, 1]

        if self.simple_loss:
            loss_diff = -masked_neg_cross_ent
        else:
            loss_diff = self.schedule.dgamma_times_alpha(t) * masked_neg_cross_ent
        loss_diff = torch.mean(loss_diff)

        return loss_diff

    @torch.inference_mode()
    def reverse_step(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        pred: torch.Tensor,
        dt: torch.Tensor,
        **_,
    ):

        alpha_t = self.schedule.alpha(t)
        alpha_s = self.schedule.alpha(t + dt)

        mean_preds = torch.softmax(pred, dim=-1)

        unmask_prob = (alpha_s - alpha_t) / (1.0 - alpha_t)
        probs_vocab = unmask_prob * mean_preds

        probs_mask = torch.ones((x_t.shape[0], 1), device=x_t.device) * (
            1.0 - unmask_prob
        )

        probs = torch.cat([probs_vocab, probs_mask], axis=-1)

        to_unmask = torch.multinomial(probs, num_samples=1).view(-1)
        is_mask = x_t == self.vocab_size

        x_s = torch.where(is_mask, to_unmask, x_t)

        return x_s

    @torch.inference_mode()
    def sample_prior(self, index: torch.Tensor):
        """
        Prior is all masked.
        """
        return self.vocab_size * torch.ones(
            (len(index),), device=self.vocab_size.device, dtype=self.vocab_size.dtype
        )
