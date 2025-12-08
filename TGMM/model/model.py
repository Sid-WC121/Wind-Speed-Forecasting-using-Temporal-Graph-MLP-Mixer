from TGMM.model.losses import MaskedCRPSLogNormal
import torch.nn as nn
import torch
from torch_scatter import scatter
from einops import rearrange

from TGMM.model.mlp_mixer import MLPMixerTemporal
from TGMM.model.elements import MLP
from TGMM.model.gnn import GNN
from TGMM.model.readouts import SingleNodeReadout
from TGMM.model.utils import StaticGraphTopologyData
from typing import Dict


class GMMModel(nn.Module):  # Changed from LightningModule to nn.Module

    def __init__(self, cfg, topo_data, metadata: Dict):
        super().__init__()
        assert cfg.metis.n_patches > 0
        self.cfg = cfg
        self.topo_data: StaticGraphTopologyData = topo_data
        self.metadata = metadata
        self.pooling = cfg.model.pool

        n_features = metadata.get('n_features', 18)
        input_dim = n_features * 2 if cfg.model.add_valid_mask else n_features
        self.input_encoder_patch = nn.Linear(
            input_dim, cfg.model.nfeatures_patch)
        self.input_encoder_node = nn.Linear(
            input_dim, cfg.model.nfeatures_node)
        self.edge_encoder = nn.Linear(1, cfg.model.nfeatures_patch)

        self.gnns = nn.ModuleList([
            GNN(nin=cfg.model.nfeatures_patch, nout=cfg.model.nfeatures_patch, nlayer_gnn=1,
                gnn_type=cfg.model.gnn_type, bn=True, dropout=cfg.train.dropout_gnn, res=True)
            for _ in range(cfg.model.nlayer_gnn)
        ])
        self.U = nn.ModuleList([
            MLP(cfg.model.nfeatures_patch, cfg.model.nfeatures_patch,
                nlayer=1, with_final_activation=True)
            for _ in range(cfg.model.nlayer_gnn-1)
        ])

        self.mixer_patch = MLPMixerTemporal(
            n_features=cfg.model.nfeatures_patch,
            n_spatial=cfg.metis.n_patches,
            n_timesteps=cfg.dataset.window,
            n_layer=cfg.model.nlayer_patch_mixer,
            dropout=cfg.train.dropout_patch_mixer,
            with_final_norm=True
        )
        self.mixer_node = MLPMixerTemporal(
            n_features=cfg.model.nfeatures_node,
            n_spatial=self.topo_data.num_nodes,
            n_timesteps=cfg.dataset.window,
            n_layer=cfg.model.nlayer_node_mixer,
            dropout=cfg.train.dropout_node_mixer,
            with_final_norm=True
        )
        self.readout = SingleNodeReadout(
            cfg.model.nfeatures_patch,
            cfg.model.nfeatures_node,
            cfg.dataset.window,
            cfg.dataset.horizon,
            self.topo_data,
            n_layers=cfg.model.nlayer_readout,
            dropout=cfg.train.dropout_readout
        )

    def forward(self, x_raw, valid_x, debug=False, return_full=False):

        if self.cfg.model.add_valid_mask:
            valid_x_float = valid_x.float() if valid_x.dtype == torch.bool else valid_x
            x_in = torch.cat([x_raw, valid_x_float], dim=-1)  # [B, t, n, f+f] or [B, t, n, 2f]
        else:
            x_in = x_raw
        
        x = rearrange(self.input_encoder_patch(x_in), 'B t n f -> B t n f')
        nodes_x = rearrange(self.input_encoder_node(x_in),
                            'B t n f -> B t n f')

        edge_weight = self.edge_encoder(self.topo_data.edge_weight.unsqueeze(-1)).unsqueeze(
            0).unsqueeze(0).repeat(x.shape[0], x.shape[1], 1, 1)

        x = x[..., self.topo_data.subgraphs_nodes_mapper, :]
        e = edge_weight[..., self.topo_data.subgraphs_edges_mapper, :]
        edge_index = self.topo_data.combined_subgraphs
        batch_x = self.topo_data.subgraphs_batch

        for i, gnn in enumerate(self.gnns):
            if i > 0:
                subgraph = scatter(x, batch_x, dim=-2,
                                reduce=self.pooling)[..., batch_x, :]
                x = x + self.U[i-1](subgraph)
                x = scatter(x, self.topo_data.subgraphs_nodes_mapper, dim=-2,
                            reduce='mean')[..., self.topo_data.subgraphs_nodes_mapper, :]
            
            x = gnn(x, edge_index, e)
        
        patch_x = scatter(x, self.topo_data.subgraphs_batch,dim=-2, reduce=self.pooling)

        patch_x = self.mixer_patch(patch_x)

        nodes_x = self.mixer_node(nodes_x)

        out = self.readout(patch_x, nodes_x)

        if return_full:
            return out, patch_x, nodes_x
        return out

