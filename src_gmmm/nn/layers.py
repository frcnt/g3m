import math
from abc import ABC
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import normal_
from torch_scatter import scatter_mean


def cosine_cutoff(edge_distances: torch.Tensor, cutoff: float):
    return torch.where(
        edge_distances < cutoff,
        0.5 * (torch.cos(torch.pi * edge_distances / cutoff) + 1.0),
        torch.tensor(0.0, device=edge_distances.device, dtype=edge_distances.dtype),
    )


class FourierEmbedding(nn.Module):
    """
    Random Fourier features (sine and cosine expansion).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        std: float = 1.0,
        trainable: bool = False,
    ):
        super(FourierEmbedding, self).__init__()
        assert (out_features % 2) == 0
        weight = torch.normal(mean=torch.zeros(out_features // 2, in_features), std=std)

        self.trainable = trainable
        if trainable:
            self.weight = nn.Parameter(weight)
        else:
            self.register_buffer("weight", weight)

        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = F.linear(x, self.weight)
        cos_features = torch.cos(2 * math.pi * x)
        sin_features = torch.sin(2 * math.pi * x)
        x = torch.cat((cos_features, sin_features), dim=1)

        return x


class EdgeEmbedding(nn.Module):
    def __init__(
        self,
        num_rbf_features: int = 64,
        max_distance: float = 25.0,
        trainable: bool = False,
        norm: bool = True,
        cutoff: bool = True,
    ):
        super().__init__()

        self.norm = norm
        self.num_rbf_features = num_rbf_features
        self.max_distance = max_distance
        self.cutoff = cutoff

        self.register_buffer("delta", torch.tensor(max_distance / num_rbf_features))
        offsets = torch.linspace(
            start=0.0, end=max_distance, steps=num_rbf_features
        ).unsqueeze(0)
        if trainable:
            self.offsets = nn.Parameter(offsets)
        else:
            self.register_buffer("offsets", offsets)

    def forward(
        self,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        norm: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm = self.norm if norm is None else norm
        dest, source = edge_index

        vectors = positions[dest] - positions[source]  # (n_edges, 3) vector (i - > j)

        distances = torch.sqrt(
            torch.sum(vectors**2, dim=-1, keepdim=True) + 1e-6
        )  # (n_edges, 1)
        d = self.featurize_distances(distances)

        cos = F.cosine_similarity(positions[dest], positions[source], dim=-1).unsqueeze(
            1
        )
        edge_features = torch.cat([d, cos], dim=1)

        if norm:
            vectors = vectors / (distances + 1.0)

        return edge_features, vectors

    def featurize_distances(self, distances: torch.Tensor):
        distances = torch.clamp(distances, 0.0, self.max_distance)
        features = torch.exp((-((distances - self.offsets) ** 2)) / self.delta)

        if self.cutoff:
            features = features * cosine_cutoff(distances, cutoff=self.max_distance)

        return features

    @property
    def out_features(self):
        return self.num_rbf_features + 1


class EquivLayerNorm(nn.Module):
    def __init__(
        self,
        dims: tuple[int, Optional[int]],
        eps: float = 1e-6,
        affine: bool = True,
    ):
        super().__init__()

        self.dims = dims
        self.sdim, self.vdim = dims
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight_s = nn.Parameter(torch.Tensor(self.sdim))
            self.bias_s = nn.Parameter(torch.Tensor(self.sdim))
            # self.weight_v = nn.Parameter(torch.Tensor(self.vdim))
        else:
            self.register_parameter("weight_s", None)
            self.register_parameter("bias_s", None)
            # self.register_parameter("weight_v", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight_s.data.fill_(1.0)
            self.bias_s.data.fill_(0.0)
            # self.weight_v.data.fill_(1.0)

    def forward(
        self, s: torch.Tensor, v: torch.Tensor, index: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        batch_size = int(index.max()) + 1
        smean = s.mean(dim=-1, keepdim=True)
        smean = scatter_mean(smean, index, dim=0, dim_size=batch_size)

        s = s - smean[index]

        var = (s * s).mean(dim=-1, keepdim=True)
        var = scatter_mean(var, index, dim=0, dim_size=batch_size)
        var = torch.clamp(var, min=self.eps)
        sout = s / var[index]

        if self.affine and self.weight_s is not None and self.bias_s is not None:
            sout = sout * self.weight_s + self.bias_s

        if v is not None:
            vmean = torch.pow(v, 2).sum(dim=1, keepdim=True).mean(dim=-1, keepdim=True)
            vmean = scatter_mean(vmean, index, dim=0, dim_size=batch_size)
            vmean = torch.clamp(vmean, min=self.eps)
            vout = v / vmean[index]

        else:
            vout = None

        out = sout, vout

        return out


class Preconditioner(ABC, nn.Module):
    def forward(
        self,
        t: torch.Tensor,
        h: torch.Tensor,
        pos: torch.Tensor,
    ):
        raise NotImplementedError


class KarrasPreconditioner(Preconditioner):
    def __init__(
        self,
        sigma_h: Optional[float] = None,
        sigma_pos: Optional[float] = None,
    ):
        super().__init__()
        self.sigma_h = sigma_h
        self.sigma_pos = sigma_pos

    def forward(
        self,
        t: torch.Tensor,
        h: torch.Tensor,
        pos: torch.Tensor,
    ):
        sigma_t = t
        if self.sigma_h is not None:
            sigma_h = torch.sqrt(sigma_t**2 + self.sigma_h**2)
        else:
            sigma_h = 1.0

        if self.sigma_pos is not None:
            sigma_pos = torch.sqrt(sigma_t**2 + self.sigma_pos**2)
        else:
            sigma_pos = 1.0

        return h / sigma_h, pos / sigma_pos


class CDCDEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        norm: float = 1.0,
        init_std: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.norm = norm
        self.init_std = init_std

        self.reset_parameters()

    def reset_parameters(self):
        normal_(self.embedding.weight, std=self.init_std)

    def forward(self, x: torch.Tensor):
        emb = self.normalize(self.embedding(x))
        return emb

    def expected_embedding(self, logits: torch.Tensor):
        prob = F.softmax(logits, dim=-1)
        weight = self.embedding.weight.T  # (vocab_size, embedding_dim)

        out = self.normalize(F.linear(prob, weight))
        assert (
            out.shape[1] == self.embedding_dim
        ), "problem with the expected embedding dim"
        return out

    @property
    def device(self):
        return self.embedding.weight.device

    def normalize(
        self, embedding: torch.Tensor, dim: int = -1, eps: float = 1e-8
    ) -> torch.Tensor:
        norm = torch.norm(embedding, dim=dim, p=2, keepdim=True)
        norm = torch.clamp(norm, min=eps)

        emb = (embedding / norm) * self.norm

        return emb


class AnalogBitsEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        scale: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_bits = math.ceil(math.log2(vocab_size))
        self.scale = scale

    def forward(self, x: torch.LongTensor):
        assert x.ndim == 1
        x_bits = (self.int2bit(x, self.n_bits) * 2.0 - 1.0) * self.scale

        return x_bits

    @classmethod
    def int2bit(cls, x_int: torch.LongTensor, n_bits: int):
        mask = 2 ** torch.arange(
            n_bits - 1, -1, -1, device=x_int.device, dtype=x_int.dtype
        )
        return x_int.unsqueeze(-1).bitwise_and(mask).ne(0).float()

    @classmethod
    def bit2int(cls, x_bit: torch.FloatTensor, threshold: bool = True):
        if threshold:
            x_bit = (x_bit > 0.0).float()
        mask = 2 ** torch.arange(x_bit.shape[-1] - 1, -1, -1).to(
            x_bit.device, x_bit.dtype
        )
        return torch.sum(mask * x_bit, -1).long()

    @property
    def embedding_dim(self):
        return self.n_bits
