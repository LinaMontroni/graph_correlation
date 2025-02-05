import numpy as np
from tqdm import tqdm
import torch
import networkx as nx
from scipy.sparse import csgraph
from scipy.linalg import eigh
from sklearn.preprocessing import MinMaxScaler
import pickle
import time
from torch.nn import Linear
import torch.nn.functional as F
from torch.utils.data import random_split
import joblib
import random
from torch_geometric.data import Data, DataLoader
import torch_geometric.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gadp
from torchmetrics.regression import SpearmanCorrCoef
import optuna
from torchmetrics import Metric
from typing import Any, List
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


class DifferentialSpearmanCorrCoef(Metric):
    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = -1.0
    plot_upper_bound: float = 1.0

    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        num_outputs: int = 1,
        **kwargs: Any,
    ) -> None:
       """
        Differentiable Spearman Correlation computation class.

        Args:
            num_outputs (int): Number of spearman correlation coeficcients to ouput

        Returns:
            None
        """
        super().__init__(**kwargs)
        rank_zero_warn(
            "Metric `SpearmanCorrcoef` will save all targets and predictions in the buffer."
            " For large datasets, this may lead to large memory footprint."
        )
        if not isinstance(num_outputs, int) and num_outputs < 1:
            raise ValueError(f"Expected argument `num_outputs` to be an int larger than 0, but got {num_outputs}")
        self.num_outputs = num_outputs

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        preds, target = _spearman_corrcoef_update(preds, target, num_outputs=self.num_outputs)
        self.preds.append(preds.to(self.dtype))
        self.target.append(target.to(self.dtype))

    def rank_data(self,data: Tensor) -> Tensor:
        """
        Differentiable rank computation using soft ranking.

        Args:
            data (Tensor): Input tensor of shape (N,).

        Returns:
            Tensor: Differentiable ranks of the input tensor.
        """
        n = data.size(0)
        pairwise_diffs = data.unsqueeze(1) - data.unsqueeze(0)  # Pairwise differences
        soft_ranks = torch.sigmoid(-pairwise_diffs).sum(dim=1) + 0.5  # Smooth rank approximation
        return soft_ranks

    def spearman_corrcoef_compute(self, preds: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
        """
        Differentiable spearman correlation coeficient.

        Args:
            preds (Tensor): input vector 
            target (Tensor): input vector 

        Returns:
            Tensor: Differentiable spearman correlation between the input tensors.
        """

        if preds.ndim == 1:
            preds = self.rank_data(preds)
            target = self.rank_data(target)
        else:
            preds = torch.stack([self.rank_data(p) for p in preds.T]).T
            target = torch.stack([self.rank_data(t) for t in target.T]).T

        preds_diff = preds - preds.mean(0)
        target_diff = target - target.mean(0)

        cov = (preds_diff * target_diff).mean(0)
        preds_std = torch.sqrt((preds_diff * preds_diff).mean(0))
        target_std = torch.sqrt((target_diff * target_diff).mean(0))

        corrcoef = cov / (preds_std * target_std + eps)
        res = torch.clamp(corrcoef, -1.0, 1.0)
        return res

    def compute(self) -> Tensor:
        """Compute Spearman's correlation coefficient."""
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        
        return self.spearman_corrcoef_compute(preds, target)


class GNNSiamese(torch.nn.Module):
    def __init__(self, conv_hidden_channels,activation,batch_size,readout_layer,num_node_features,n_graphs,gat_heads):
        """
        GNN-Siamese architecture

        Args:
            conv_hidden_channels (Tensor): Number of neurons per layer
            activation (Torch function): Activation function
            batch_size (int): Batch size
            readout_layer (Torch function): Pooling layer
            num_node_features (int): Number of node features
            n_graphs (int): Number of graphs in each set
            gat_heads (int): Number of GAT heads (attention mechanism)
        Returns:
            None
        """
        super(GCNSimpleSiamese, self).__init__()
        self.readout_layer = readout_layer
        self.activation = activation
        self.n_graphs = n_graphs
        self.last_conv_hidden_channel = conv_hidden_channels[-1]
        self.batch_size = batch_size

        self.convs = torch.nn.ModuleList()
        for i in range(len(conv_hidden_channels)):
          in_channels = num_node_features if i == 0 else conv_hidden_channels[i - 1] * gat_heads
          out_channels = conv_hidden_channels[i]
          self.convs.append(nn.GATConv(in_channels, out_channels,heads=gat_heads))
          if i == len(conv_hidden_channels)-1:
              self.convs.append(nn.GATConv(out_channels * gat_heads, 1,heads=gat_heads))

        self.spearman_corrcoef = DifferentialSpearmanCorrCoef(num_outputs=batch_size)

    def forward_once(self, x, edge_index, batch):
        """
        Single forward method: performs a forward on a single set of graphs

        Args:
            x (Tensor): Input data: node features for all nodes in all graphs of a single set
            edge_index (Tuple): Edge index from all nodes in all graphs of a single set. 
            batch: Batch id
        Returns:
            Tensor: a tensor of size [1] for each graph. Size [Z,batch_size] for Z graphs in a set 
        """

        for conv in self.convs:
            x = self.activation(conv(x, edge_index))

        # Compute the mean of heads
        x = x.mean(dim=-1)  # Averaging over the head dimension

        # 2. Readout layer
        x = self.readout_layer(x, batch)

        #reshape
        x = x.view(-1)
        x = x.reshape(self.batch_size,self.n_graphs).T #reshape for n_graphs per set 

        return x

    def forward(self, x1, edge_index1, batch1,x2, edge_index2, batch2):
        """
        Siamese forward method: performs forward_once method on a both set of graphs and computes the spearman correlation between the outputs from forward_once.

        Args:
            x1 (Tensor): Input data: node features for all nodes in all graphs of set 1
            edge_index1 (Tuple): Edge index from all nodes in all graphs of set 1
            batch1: Batch id for all graphs in set 1
            x2 (Tensor): Input data: node features for all nodes in all graphs of set 2
            edge_index2 (Tuple): Edge index from all nodes in all graphs of set 2
            batch2: Batch id for all graphs in set 2
        Returns:
            Tensor: Predicted correlation between graphs sets1 and 2
        """
        # 1. Call foward_once for both pair graphs
        out1 = self.forward_once(x1, edge_index1, batch1)
        out2 = self.forward_once(x2, edge_index2, batch2)

        # 2. Calculate Spearman correlation using the Lambda-like layer
        pred_corr = self.spearman_corrcoef(out1, out2).view(-1)

        return pred_corr
