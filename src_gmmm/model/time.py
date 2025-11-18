from abc import ABC
from typing import Union

import torch
from torch_geometric.data import Batch


class TimeSampler(ABC):
    def __call__(
        self, batch: Batch, device: torch.device
    ) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        raise NotImplementedError


class UniformTimeSampler(TimeSampler):
    def __init__(self, lb: float = 1e-3, ub: float = 1.0):
        self.lb = lb
        self.ub = ub

    def __call__(
        self, batch: Batch, device: torch.device
    ) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        return self.sample(size=batch.num_graphs, device=device, lb=self.lb, ub=self.ub)

    @staticmethod
    def sample(
        size: int,
        device: torch.device,
        lb: float = 0.0,
        ub: float = 1.0,
    ):
        return (lb - ub) * torch.rand(size, device=device) + ub


class AntitheticTimeSampler(TimeSampler):
    def __init__(self, lb: float = 1e-3, ub: float = 1.0):
        self.lb = lb
        self.ub = ub

    def __call__(
        self, batch: Batch, device: torch.device
    ) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        return self.sample(size=batch.num_graphs, device=device, lb=self.lb, ub=self.ub)

    @staticmethod
    def sample(
        size: int,
        device: torch.device,
        lb: float = 0.0,
        ub: float = 1.0,
    ):
        t0 = torch.rand(1, device=device)
        t = 1.0 - ((t0 + torch.linspace(0.0, 1.0, size + 1, device=device)[:-1]) % 1.0)
        return (ub - lb) * t + lb


class PerModalityTimeSampler(TimeSampler):
    def __init__(self, sampler_dict: dict[str, TimeSampler]):
        self.sampler_dict = sampler_dict

    def __call__(self, batch: Batch, device: torch.device):
        t = dict()
        for key, sampler in self.sampler_dict.items():
            t[key] = sampler(batch, device)
        return t
