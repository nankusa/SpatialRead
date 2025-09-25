from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules import TransformerDecoderLayer, TransformerDecoder
import torch_geometric
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data, Batch
from torch_geometric.data.collate import collate

import numpy as np

from pymatgen.core.lattice import Lattice

# GNNs
from spatialread.modules.gnn.schnet import SchNet as SchNetO
from spatialread.modules.gnn.painn import PaiNN as PaiNNO
from spatialread.modules.gnn.visnet import ViSNet

from spatialread.utils.log import Log
from spatialread.utils.graph import filter_edge
from spatialread.config import get_config, get_model_config, get_data_config, get_train_config
from spatialread.modules.utils import MLP

from spatialread.modules.jmp.models.gemnet.backbone import GemNetOCBackbone
from spatialread.modules.jmp.models.gemnet.config import BackboneConfig
from spatialread.modules.jmp.utils.goc_graph import (
    Cutoffs,
    Graph,
    MaxNeighbors,
    generate_graph,
    subselect_graph,
    tag_mask,
)

from abc import ABC, abstractmethod
from typing import Literal, Any, Dict, Union, List
from typing_extensions import override


import warnings


class BaseModule(torch.nn.Module, ABC, Log):
    def __init__(self):
        torch.nn.Module.__init__(self)
        ABC.__init__(self)
        Log.__init__(self)

        self.config = get_config()
        self.model_config = get_model_config()
        self.data_config = get_data_config()
        self.train_config = get_train_config()
        self.spnode_config = self.data_config['spnode']
        self.head_type = self.model_config['head']

        self.init_gnn()

        if self.head_type == 'transformer':

            tconfig = self.model_config['transformer']
            self.tconfig = tconfig

            self.spnode_proj = nn.Linear(self.gnn_hid_dim, tconfig['hid_dim'])
            self.atom_proj = nn.Linear(self.gnn_hid_dim, tconfig['hid_dim'])

            decoder_layer = TransformerDecoderLayer(
                tconfig['hid_dim'],
                tconfig['nheads'],
                tconfig['ff_dim'],
                tconfig['dropout'],
                batch_first=tconfig['batch_first'],
            )
            self.transformer = TransformerDecoder(
                decoder_layer, tconfig['nlayers']
            )
            self.pos_emb = nn.Parameter(torch.zeros([self.spnode_config['num'], tconfig['hid_dim']]))
            self.cls_emb = nn.Parameter(torch.zeros([tconfig['hid_dim']]))
        elif self.head_type == 'spnode_mlp' or self.head_type == 'atom_mlp':
            mlp_config = self.model_config['head_mlp']
            self.mlp = MLP([mlp_config['hid_dim']] * mlp_config['nlayers'])
        else:
            raise ValueError(f"Unknown head type {self.head_type}")
        # num_spnode = self.spnode_config['num']
        # self.pos_emb = nn.Paramter([num_spnode, tconfig['hid_dim']])

    def init_gnn(self):
        raise NotImplementedError

    # process single data
    def data_transform(self, data: Data) -> Data:
        assert self.cutoff is not None
        assert self.max_num_neighbors is not None

        # print("DEBUG", self.cutoff, self.max_num_neighbors)

        data.atomic_numbers = data.atomic_numbers.long()
        data.natoms = torch.tensor([len(data.atomic_numbers)]).to(data.atomic_numbers.device)
        data.pos = data.pos.float()
        data.cell = data.cell.reshape(1, 3, 3)

        edge_index, edge_dist, edge_vec, cell_offsets = data.edge_index, data.edge_dist, data.edge_vec, data.cell_offsets
        # edge_index, edge_dist, edge_vec, [cell_offsets] = filter_edge(edge_index, edge_dist, edge_vec, None, None, cutoff, max_num_neighbors, other_content=[cell_offsets])
        edge_index, edge_dist, edge_vec, [cell_offsets] = filter_edge(edge_index, edge_dist, edge_vec, 'p2pv', data.atomic_numbers==self.spnode_config['z'], self.cutoff, self.max_num_neighbors, other_content=[cell_offsets])

        # print("DEBUG", edge_index[0].max(), edge_index[1].max(), edge_index[0].min(), edge_index[1].min())
        # print("DEBUG", data.atomic_numbers.shape)
        # print("DEBUG", edge_index.shape, edge_dist.max())

        data.edge_index = edge_index
        data.edge_dist = edge_dist
        data.edge_vec = edge_vec

        return data

    # collate transformed data
    def collate_fn(self, data_list: List[Data]) -> Batch:
        # normally data_list should be list of torch_geometric.data.Data
        # we directly use torch_geometric.data.collate.collate
        # return collate(data_list[0].__class__, data_list)[0]
        return Batch.from_data_list(data_list)

    # process collated data to target
    def extract_gnn_feat(self, batch_data: Batch):
        # forward process
        raise NotImplementedError


    def forward(self, batch: Dict, **kwargs) -> Dict[str, torch.Tensor]:
        # by default we use graph data
        # batch_graph = self.process_data_list(graph_data)
        if self.model_config['gnn'] == 'gemnet':
            graph_data = batch['gemnet']
        else:
            # default spgraph
            graph_data = batch['spgraph']
        data_list = [self.data_transform(data) for data in graph_data]
        batch_graph = self.collate_fn(data_list)
        feat = self.extract_gnn_feat(batch_graph, **kwargs)
        atom_feat = feat['atom_feat']
        spnode_feat = feat['spnode_feat']

        batch_size = len(graph_data)

        if self.head_type == 'transformer':
            spnode_feat = self.spnode_proj(spnode_feat)
            atom_feat = self.atom_proj(atom_feat)

            assert len(spnode_feat) == self.spnode_config['num'] * batch_size
            spnode_feat = spnode_feat.reshape(batch_size, self.spnode_config['num'], self.model_config['transformer']['hid_dim'])
            spnode_feat += self.pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
            atom_feat, atom_mask = to_dense_batch(atom_feat, batch_graph.batch[batch_graph.atomic_numbers != self.spnode_config['z']])
            # pos_emb = self.pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
            cls_emb = self.cls_emb.unsqueeze(0).unsqueeze(0).repeat([batch_size, 1, 1]) # 1， 1， hid_dim

            emb = torch.cat([cls_emb, spnode_feat], dim=1) # bs, 1+num_spnode, hid_dim

            # repulsion_mask = batch_graph['repulsion_mask'].reshape(batch_size, self.spnode_config['num'])
            
            emb = self.transformer(emb, atom_feat, memory_key_padding_mask=~atom_mask)

            return {
                "cls_feat": emb[:, 0],
                "spnode_feat": emb[:, 1:],
                "atom_feat": atom_feat,
                "atom_mask": atom_mask
            }
        elif self.head_type == 'spnode_mlp':
            spnode_feat = self.mlp(spnode_feat)
            return {
                "spnode_feat": spnode_feat,
                "atom_feat": atom_feat,
                "atom_batch_idx": batch_graph.batch[batch_graph.atomic_numbers != self.spnode_config['z']],
                "spnode_batch_idx": batch_graph.batch[batch_graph.atomic_numbers == self.spnode_config['z']],
            }
        elif self.head_type == 'atom_mlp':
            atom_feat = self.mlp(atom_feat)
            return {
                "atom_feat": atom_feat,
                "spnode_feat": spnode_feat,
                "atom_batch_idx": batch_graph.batch[batch_graph.atomic_numbers != self.spnode_config['z']],
                "spnode_batch_idx": batch_graph.batch[batch_graph.atomic_numbers == self.spnode_config['z']],
            }
        else:
            raise ValueError(f"Unknown head type {self.head_type}")


# class SchNet(BaseModule):
#     def __init__(self):
#         super().__init__()

#         self.model_config = self.model_config["schnet"]
#         self.schnet = SchNetO(
#             **self.model_config,
#             VNODE_Z=120,
#         )

#     def data_transform(self, data: Data) -> Data:
#         return super().data_transform(data)

#     def collate_fn(self, data_list: List[Data]) -> Data:
#         return super().collate_fn(data_list)

#     def extract_gnn_feat(self, batch_data) -> Data:
#         h = None
#         out = self.schnet(
#             batch_data,
#             h=h,
#         )
#         h = out["feat"]
#         return out


class PaiNN(BaseModule):
    def __init__(self):
        super().__init__()

    def init_gnn(self):
        self.transformer_config = self.model_config['transformer']
        self.painn_config = self.model_config["painn"]
        self.cutoff = self.painn_config['cutoff']
        self.max_num_neighbors = self.painn_config.pop('max_num_neighbors')
        self.painn = PaiNNO(
            **self.painn_config,
        )
        # self.proj = nn.Linear(self.painn_config['n_atom_basis'], self.transformer_config['hid_dim'])

        self.gnn_hid_dim = self.painn_config['n_atom_basis']

    def data_transform(self, data: Data) -> Data:
        return super().data_transform(data)

    def extract_gnn_feat(self, batch_data: Batch) -> Dict[str, torch.Tensor]:
        q, mu = self.painn(batch_data)
        # print("DEBUG", q[:10, :10], q[-10:, :10])
        # q = self.proj(q)
        spnode_mask = batch_data.atomic_numbers == self.spnode_config['z']
        return {
            "atom_feat": q[~spnode_mask],
            "spnode_feat": q[spnode_mask]
        }


class JMP(BaseModule):
    def __init__(self):
        super().__init__()

    def init_gnn(self):
        self.gemnet_config = self.model_config["gemnet"]
        self.hid_dim = self.gemnet_config["hid_dim"]

        base_config = BackboneConfig.base()
        # Plus one for Virtual Node
        self.atom_embedding = nn.Embedding(125, self.hid_dim)
        # self.atom_embedding.apply(objectives.init_weights)
        # self.gemnet = nn.ModuleDict(
        #     {
        #         msg_direction: GemNetOCBackbone(base_config, **base_config)
        #         for msg_direction in self.msg_routes
        #     }
        # )
        self.gemnet = GemNetOCBackbone(base_config, **base_config)
        if self.gemnet_config.get("ckpt") is not None:
            self.load_backbone_state_dict(self.gemnet_config["ckpt"])

        self.gnn_hid_dim = self.hid_dim

    def data_transform(self, data: Data) -> Data:
        return data

    def load_backbone_state_dict(self, ckpt_path):
        def filter_state_dict(state_dict: dict[str, torch.Tensor], prefix: str):
            return {
                k[len(prefix) :]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }

        # Due to that pre-trained JMP-S weight is used, the ckpt path is not configurable
        ckpt = torch.load(ckpt_path)
        state_dict = ckpt["state_dict"]
        backbone_state_dict = filter_state_dict(state_dict, "backbone.")
        # due to that we remove symmetric mp, there are some unexpected keys to ignore
        load_ret = self.gemnet.load_state_dict(backbone_state_dict, strict=False)
        print("Load Backbone Statedict Return: ", load_ret)
        # for msg_direction in self.msg_routes:
        #     load_ret = self.gemnet[msg_direction].load_state_dict(
        #         backbone_state_dict, strict=False
        #     )
        #     self.log(
        #         f"Load JMP Backbone Statedict for {msg_direction} Return: {str(load_ret)}",
        #         "info",
        #     )

    def extract_gnn_feat(self, batch_data: Batch) -> Dict[str, torch.Tensor]:
        x = batch_data["atomic_numbers"]
        pos = batch_data["pos"]
        atom_batch_idx = batch_data["batch"]

        h = self.atom_embedding(x)

        # print("DEBUG BEFORE", h[:3, :5])

        # struct
        graph_output = self.gemnet(batch_data, h=h)
        h = graph_output["energy"]

        spnode_mask = batch_data.atomic_numbers == self.spnode_config['z']
        return {
            "atom_feat": h[~spnode_mask],
            "spnode_feat": h[spnode_mask]
        }


class ViSNetPaiNN(BaseModule):
    def __init__(self):
        super().__init__()

        mlp_config = self.model_config['head_mlp']
        self.mlp = MLP([mlp_config['hid_dim']] * mlp_config['nlayers'])
    
    def init_gnn(self):
        self.visnet = ViSNet(**self.model_config['visnet'])
        self.visnet_hid_dim = self.model_config['visnet']['hidden_channels']
        
        self.transformer_config = self.model_config['transformer']
        self.painn_config = self.model_config["painn"]
        self.cutoff = self.painn_config['cutoff']
        self.max_num_neighbors = self.painn_config.pop('max_num_neighbors')
        self.painn = PaiNNO(
            **self.painn_config,
        )

        self.gnn_hid_dim = self.painn_config['n_atom_basis']
        self.proj = nn.Linear(self.visnet_hid_dim, self.gnn_hid_dim)

    def extract_gnn_feat(self, graph: Batch, spgraph: Batch) -> Dict[str, torch.Tensor]:
        # x = graph["atomic_numbers"]
        # pos = graph["pos"]
        # atom_batch_idx = graph["batch"]

        # h = self.atom_embedding(x)

        atom_feat = self.visnet(graph)

        q, mu = self.painn(spgraph)
        # print("DEBUG", q[:10, :10], q[-10:, :10])
        # q = self.proj(q)
        spnode_mask = spgraph.atomic_numbers == self.spnode_config['z']
        spnode_feat = q[spnode_mask]

        # print("DEBUG", gemnet_graph.atomic_numbers.shape, spgraph.atomic_numbers.shape, spnode_mask.sum())

        return {
            "atom_feat": self.proj(atom_feat),
            "spnode_feat": spnode_feat
        }


    def forward(self, batch: Dict, **kwargs) -> Dict[str, torch.Tensor]:
        # by default we use graph data
        # batch_graph = self.process_data_list(graph_data)
        spgraph = batch['spgraph']
        graph = batch['graph']

        spgraph_list = [self.data_transform(data) for data in spgraph]
        batch_spgraph = self.collate_fn(spgraph_list)

        graph_list = [self.data_transform(data) for data in graph]
        batch_graph = self.collate_fn(graph_list)

        feat = self.extract_gnn_feat(batch_graph, batch_spgraph, **kwargs)

        atom_feat = feat['atom_feat']
        spnode_feat = feat['spnode_feat']

        batch_size = len(spgraph_list)

        # spnode_feat = self.spnode_proj(spnode_feat)
        # atom_feat = self.atom_proj(atom_feat)

        atom_feat = self.mlp(atom_feat)
        return {
            "atom_feat": atom_feat,
            "spnode_feat": spnode_feat,
            "atom_batch_idx": batch_graph.batch[batch_graph.atomic_numbers != self.spnode_config['z']],
            "spnode_batch_idx": batch_graph.batch[batch_graph.atomic_numbers == self.spnode_config['z']],
        }

        # assert len(spnode_feat) == self.spnode_config['num'] * batch_size
        # spnode_feat = spnode_feat.reshape(batch_size, self.spnode_config['num'], self.model_config['transformer']['hid_dim'])
        # spnode_feat += self.pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        # atom_feat, atom_mask = to_dense_batch(atom_feat, batch_graph.batch)
        # # pos_emb = self.pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        # cls_emb = self.cls_emb.unsqueeze(0).unsqueeze(0).repeat([batch_size, 1, 1]) # 1， 1， hid_dim

        # emb = torch.cat([cls_emb, spnode_feat], dim=1) # bs, 1+num_spnode, hid_dim

        # # repulsion_mask = batch_graph['repulsion_mask'].reshape(batch_size, self.spnode_config['num'])
        
        # emb = self.transformer(emb, atom_feat, memory_key_padding_mask=~atom_mask)

        # return {
        #     "cls_feat": emb[:, 0],
        #     "spnode_feat": emb[:, 1:],
        #     "atom_feat": atom_feat,
        #     "atom_mask": atom_mask
        # }


class JMPPaiNN(BaseModule):
    def __init__(self):
        super().__init__()

    def init_gnn(self):
        self.gemnet_config = self.model_config["gemnet"]

        self.gemnet_hid_dim = self.gemnet_config["hid_dim"]
        base_config = BackboneConfig.small()
        self.atom_embedding = nn.Embedding(125, self.gemnet_hid_dim)
        self.gemnet = GemNetOCBackbone(base_config, **base_config)
        if self.gemnet_config.get("ckpt") is not None:
            self.load_backbone_state_dict(self.gemnet_config["ckpt"])

        
        self.transformer_config = self.model_config['transformer']
        self.painn_config = self.model_config["painn"]
        self.cutoff = self.painn_config['cutoff']
        self.max_num_neighbors = self.painn_config.pop('max_num_neighbors')
        self.painn = PaiNNO(
            **self.painn_config,
        )

        self.gnn_hid_dim = self.painn_config['n_atom_basis']
        self.proj = nn.Linear(self.gemnet_hid_dim, self.gnn_hid_dim)

    def load_backbone_state_dict(self, ckpt_path):
        def filter_state_dict(state_dict: dict[str, torch.Tensor], prefix: str):
            return {
                k[len(prefix) :]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }

        # Due to that pre-trained JMP-S weight is used, the ckpt path is not configurable
        ckpt = torch.load(ckpt_path)
        state_dict = ckpt["state_dict"]
        backbone_state_dict = filter_state_dict(state_dict, "backbone.")
        # due to that we remove symmetric mp, there are some unexpected keys to ignore
        load_ret = self.gemnet.load_state_dict(backbone_state_dict, strict=False)
        print("Load Backbone Statedict Return: ", load_ret)
        # for msg_direction in self.msg_routes:
        #     load_ret = self.gemnet[msg_direction].load_state_dict(
        #         backbone_state_dict, strict=False
        #     )
        #     self.log(
        #         f"Load JMP Backbone Statedict for {msg_direction} Return: {str(load_ret)}",
        #         "info",
        #     )

    def extract_gnn_feat(self, gemnet_graph: Batch, spgraph: Batch) -> Dict[str, torch.Tensor]:
        x = gemnet_graph["atomic_numbers"]
        pos = gemnet_graph["pos"]
        atom_batch_idx = gemnet_graph["batch"]

        h = self.atom_embedding(x)

        # print("DEBUG BEFORE", h[:3, :5])

        # struct
        graph_output = self.gemnet(gemnet_graph, h=h)
        h = graph_output["energy"]

        atom_feat = h

        q, mu = self.painn(spgraph)
        # print("DEBUG", q[:10, :10], q[-10:, :10])
        # q = self.proj(q)
        spnode_mask = spgraph.atomic_numbers == self.spnode_config['z']
        spnode_feat = q[spnode_mask]

        # print("DEBUG", gemnet_graph.atomic_numbers.shape, spgraph.atomic_numbers.shape, spnode_mask.sum())

        return {
            "atom_feat": self.proj(atom_feat),
            "spnode_feat": spnode_feat
        }


    def forward(self, batch: Dict, **kwargs) -> Dict[str, torch.Tensor]:
        # by default we use graph data
        # batch_graph = self.process_data_list(graph_data)
        spgraph = batch['spgraph']
        gemnet_graph = batch['gemnet']

        spgraph_list = [self.data_transform(data) for data in spgraph]
        batch_spgraph = self.collate_fn(spgraph_list)

        batch_gemnet_graph = self.collate_fn(gemnet_graph)

        feat = self.extract_gnn_feat(batch_gemnet_graph, batch_spgraph, **kwargs)

        atom_feat = feat['atom_feat']
        spnode_feat = feat['spnode_feat']

        batch_size = len(spgraph_list)

        spnode_feat = self.spnode_proj(spnode_feat)
        atom_feat = self.atom_proj(atom_feat)

        assert len(spnode_feat) == self.spnode_config['num'] * batch_size
        spnode_feat = spnode_feat.reshape(batch_size, self.spnode_config['num'], self.model_config['transformer']['hid_dim'])
        spnode_feat += self.pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        atom_feat, atom_mask = to_dense_batch(atom_feat, batch_gemnet_graph.batch)
        # pos_emb = self.pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        cls_emb = self.cls_emb.unsqueeze(0).unsqueeze(0).repeat([batch_size, 1, 1]) # 1， 1， hid_dim

        emb = torch.cat([cls_emb, spnode_feat], dim=1) # bs, 1+num_spnode, hid_dim

        # repulsion_mask = batch_graph['repulsion_mask'].reshape(batch_size, self.spnode_config['num'])
        
        emb = self.transformer(emb, atom_feat, memory_key_padding_mask=~atom_mask)

        return {
            "cls_feat": emb[:, 0],
            "spnode_feat": emb[:, 1:],
            "atom_feat": atom_feat,
            "atom_mask": atom_mask
        }

        

        # return {
        #     "atom_feat": h[~spnode_mask],
        #     "spnode_feat": h[spnode_mask]
        # }


def construct_model():
    model_config = get_model_config()
    gnn = model_config['gnn']

    if gnn == 'painn':
        return PaiNN()
    elif gnn == 'gemnet':
        return JMP()
    elif gnn == 'gemnetpainn':
        return JMPPaiNN()
    elif gnn == 'visnetpainn':
        return ViSNetPaiNN()
    else:
        raise ValueError(f"Unsupported gnn type: {gnn}")
