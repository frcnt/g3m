import abc
from typing import Literal, Optional, Sequence, Union

import torch
import torch.nn as nn

from ..utils.ops import scatter_center


class Readout(nn.Module, abc.ABC):

    def forward(
        self,
        t: torch.Tensor,
        states: dict[str, torch.Tensor],
        h: torch.Tensor,
        pos: torch.Tensor,
        node_index: torch.Tensor,
        edge_node_index: torch.Tensor,
    ) -> Sequence[torch.Tensor]:
        raise NotImplementedError


class DataPointReadout(Readout):

    def __init__(
        self,
        hidden_dim: int,
        h_output_dim: Optional[int] = 0,
        pred_h: bool = True,
        pred_pos: bool = True,
        zero_cog: bool = True,
        parameterization: Literal["residual-pos", "residual-time-pos"] = "residual-pos",
    ) -> None:
        super(DataPointReadout, self).__init__()

        if pred_h:
            assert h_output_dim > 0
            self.net_h = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, h_output_dim),
            )

        if pred_pos:
            self.net_pos = nn.Linear(in_features=hidden_dim, out_features=1, bias=False)
            self.zero_cog = zero_cog

        self.pred_h = pred_h
        self.pred_pos = pred_pos
        self.parameterization = parameterization

    def forward(
        self,
        t: torch.Tensor,
        states: dict[str, torch.Tensor],
        h: torch.Tensor,
        pos: torch.Tensor,
        node_index: Optional[torch.Tensor],
        edge_node_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        out = dict()

        if self.pred_h:
            out["h"] = self.net_h(states["s"])

        if self.pred_pos:
            out_pos = self.net_pos(states["v"]).squeeze()

            match self.parameterization:
                case "residual-pos":
                    out_pos += pos
                case "residual-time-pos":
                    out_pos = pos + t[node_index] * out_pos
                # otherwise raw output is 'out_pos'

            if self.zero_cog:
                out_pos = scatter_center(out_pos, index=node_index)

            out["pos"] = out_pos

        return out


class KarrasReadout(Readout):
    def __init__(
        self,
        hidden_dim: int,
        h_output_dim: Optional[int] = 0,
        pred_h: bool = True,
        pred_pos: bool = True,
        zero_cog: bool = True,
        sigma_h: float = None,
        sigma_pos: float = None,
    ) -> None:
        super(KarrasReadout, self).__init__()

        if pred_h:
            assert h_output_dim > 0
            assert sigma_h is not None
            self.net_h = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, h_output_dim),
            )

        if pred_pos:
            assert sigma_pos is not None
            self.net_pos = nn.Linear(in_features=hidden_dim, out_features=1, bias=False)
            self.zero_cog = zero_cog

        self.pred_h = pred_h
        self.sigma_h = sigma_h

        self.pred_pos = pred_pos
        self.sigma_pos = sigma_pos

    def forward(
        self,
        t: torch.Tensor,
        states: dict[str, torch.Tensor],
        h: torch.Tensor,
        pos: torch.Tensor,
        node_index: Optional[torch.Tensor],
        edge_node_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        NB: the std at time $t$ is $t$, sigma(t)=t
        """

        out = dict()

        if self.pred_h:
            net_h = self.net_h(states["s"])

            c_skip = self._c_skip(t, self.sigma_h)
            c_out = self._c_out(t, self.sigma_h)
            out_h = c_skip * h + c_out * net_h

            out["h"] = out_h

        if self.pred_pos:
            net_pos = self.net_pos(states["v"]).squeeze()

            c_skip = self._c_skip(t, self.sigma_pos)
            c_out = self._c_out(t, self.sigma_pos)
            out_pos = c_skip * pos + c_out * net_pos

            if self.zero_cog:
                out_pos = scatter_center(out_pos, index=node_index)

            out["pos"] = out_pos

        return out

    def _c_skip(
        self, t: torch.Tensor, sigma_data: Union[torch.Tensor, float]
    ) -> torch.Tensor:
        sigma_t = t
        return sigma_data**2 / (sigma_data**2 + sigma_t**2)

    def _c_out(
        self, t: torch.Tensor, sigma_data: Union[torch.Tensor, float]
    ) -> torch.Tensor:
        sigma_t = t
        return sigma_t * sigma_data / torch.sqrt(sigma_data**2 + sigma_t**2)
