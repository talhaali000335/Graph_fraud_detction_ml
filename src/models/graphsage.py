"""
Graph Fraud Detection AI
GraphSAGE Model

Two-layer GraphSAGE network for transaction fraud detection.
"""

import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import SAGEConv


class GraphSAGEFraudDetector(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        output_dim=2,
        dropout=0.30,
    ):
        super().__init__()

        self.conv1 = SAGEConv(
            input_dim,
            hidden_dim,
        )

        self.conv2 = SAGEConv(
            hidden_dim,
            hidden_dim,
        )

        self.classifier = nn.Linear(
            hidden_dim,
            output_dim,
        )

        self.dropout = dropout

    def forward(
        self,
        x,
        edge_index,
    ):

        # First GraphSAGE layer
        x = self.conv1(
            x,
            edge_index,
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        # Second GraphSAGE layer
        x = self.conv2(
            x,
            edge_index,
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        # Fraud classifier
        out = self.classifier(x)

        return out