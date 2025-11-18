from typing import Optional

import torch
import torch.nn as nn

from ..nn.encoder import EquivEncoder
from ..nn.layers import Preconditioner
from ..nn.readout import Readout


class EquivariantParameterization(nn.Module):

    def __init__(
        self,
        encoder: EquivEncoder,
        readout: Readout,
        preconditioner: Optional[Preconditioner] = None,
    ):
        super(EquivariantParameterization, self).__init__()
        self.encoder = encoder
        self.readout = readout
        self.preconditioner = preconditioner

    def forward(
        self,
        t: torch.Tensor,
        h: torch.Tensor,
        pos: torch.Tensor,
        node_index: torch.Tensor,
        edge_node_index: torch.Tensor,
    ):

        if self.preconditioner is not None:
            h, pos = self.preconditioner.forward(t, h, pos)

        states = self.encoder.forward(
            t=t,
            h=h,
            pos=pos,
            node_index=node_index,
            edge_node_index=edge_node_index,
        )
        return self.readout.forward(
            t,
            states,
            h=h,
            pos=pos,
            node_index=node_index,
            edge_node_index=edge_node_index,
        )
