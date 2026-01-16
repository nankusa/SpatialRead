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
from spatialread.modules.gnn.schnet import SchNet

from spatialread.utils.log import Log
from spatialread.utils.graph import filter_edge, radius_graph_pbc_p2v
from spatialread.utils.coord import sample_virtual_nodes
from spatialread.config import (
    get_config,
    get_model_config,
    get_data_config,
    get_train_config,
)
from spatialread.modules.utils import MLP

from spatialread.modules.gemnet.backbone import GemNetOCBackbone
from spatialread.modules.gemnet.config import BackboneConfig
from spatialread.utils.goc_graph import (
    Cutoffs,
    Graph,
    MaxNeighbors,
    generate_graph,
    subselect_graph,
    tag_mask,
)
from spatialread.utils.finetune_state_dict import (
    retreive_ft_state_dict_from_loaded_ckpt,
    filter_state_dict,
)

from abc import ABC, abstractmethod
from typing import Literal, Any, Dict, Union, List, TypedDict, cast
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
        self.spnode_config = self.data_config["spnode"]
        self.head_type = self.model_config["head"]
        self.mp_between_spnode = self.model_config["mp_between_spnode"]

        self.init_gnn()

        if self.mp_between_spnode and self.head_type in ["transformer", "spnode_mlp"]:
            self.schnet = SchNet(
                hidden_channels=self.model_config["head_mlp"]["hid_dim"]
            )

        if self.head_type == "transformer":
            spconfig = self.spnode_config
            assert isinstance(
                spconfig["num"], int
            ), "currently only support fixed grid for transformer due to pos embed"
            assert spconfig["sample"] == "grid"

            tconfig = self.model_config["transformer"]
            self.tconfig = tconfig

            self.spnode_proj = nn.Linear(self.gnn_hid_dim, tconfig["hid_dim"])
            self.atom_proj = nn.Linear(self.gnn_hid_dim, tconfig["hid_dim"])

            decoder_layer = TransformerDecoderLayer(
                tconfig["hid_dim"],
                tconfig["nheads"],
                tconfig["ff_dim"],
                tconfig["dropout"],
                batch_first=tconfig["batch_first"],
            )
            self.transformer = TransformerDecoder(decoder_layer, tconfig["nlayers"])
            self.pos_emb = nn.Parameter(
                torch.zeros([self.spnode_config["num"] ** 3, tconfig["hid_dim"]])
            )
            self.cls_emb = nn.Parameter(torch.zeros([tconfig["hid_dim"]]))
            # self.mask_emb = nn.Parameter(torch.zeros([tconfig['hid_dim']]))
        elif self.head_type in ["spnode_mlp", "atom_mlp", "global_attn", "gmt"]:
            mlp_config = self.model_config["head_mlp"]
            self.mlp = MLP([mlp_config["hid_dim"]] * mlp_config["nlayers"])
            # self.mlp.apply(init_he_orthogonal)
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

        data.atomic_numbers = data.atomic_numbers.long()
        data.natoms = torch.tensor([len(data.atomic_numbers)]).to(
            data.atomic_numbers.device
        )
        data.pos = data.pos.float()
        data.cell = data.cell.reshape(1, 3, 3)

        num_atoms = len(data.atomic_numbers)
        edge_index, edge_dist, edge_vec = data.edge_index, data.edge_dist, data.edge_vec
        spconfig = self.spnode_config

        # # sample coordinates of spatial nodes
        # with torch.amp.autocast("cuda", enabled=False):
        #     spnode_pos, _ = sample_virtual_nodes(data.pos, spconfig['num'], spconfig['sample'], self.data_config['pbc'], data.cell.reshape(3, 3).cpu().numpy(), repulsion_distance=spconfig['repulsion_distance'], kde_bandwidth=spconfig['kde_bandwidth'])
        #     # assert len(spnode_pos) == spconfig['num']

        #     p2v_edge_index, p2v_edge_dist, p2v_edge_vec = radius_graph_pbc_p2v(data.pos, spnode_pos, data.cell.reshape(3, 3), cutoff=10, max_num_neighbors=5000) # radius returns random max_num_neighbros, we need to maintain all edges and filter in `filter_edge`

        #     data.atomic_numbers = torch.cat([torch.ones(len(spnode_pos)).to(data.atomic_numbers)*spconfig['z'], data.atomic_numbers], dim=0)
        #     data.pos = torch.cat([spnode_pos, data.pos], dim=0)

        #     # concat edge of p2p and p2v
        #     edge_index += len(spnode_pos)
        #     p2v_edge_index[0] += len(spnode_pos)

        #     edge_index = torch.cat([edge_index, p2v_edge_index], dim=-1)
        #     edge_dist = torch.cat([edge_dist, p2v_edge_dist], dim=0)
        #     edge_vec = torch.cat([edge_vec, p2v_edge_vec], dim=0)

        # filter for cutoff and max_num_neighbors
        edge_index, edge_dist, edge_vec = filter_edge(
            edge_index,
            edge_dist,
            edge_vec,
            "p2pv",
            data.atomic_numbers == self.spnode_config["z"],
            self.cutoff,
            self.max_num_neighbors,
            repulsion_distance=spconfig["repulsion_distance"],
        )

        data.edge_index = edge_index
        data.edge_dist = edge_dist
        data.edge_vec = edge_vec

        self.compute_repulsion_mask(data)

        # return spdata
        return data

    # # process single data
    # def data_transform(self, data: Data, spdata: Data) -> Data:
    #     assert self.cutoff is not None
    #     assert self.max_num_neighbors is not None

    #     data.atomic_numbers = data.atomic_numbers.long()
    #     data.natoms = torch.tensor([len(data.atomic_numbers)]).to(data.atomic_numbers.device)
    #     data.pos = data.pos.float()
    #     data.cell = data.cell.reshape(1, 3, 3)

    #     num_atoms = len(data.atomic_numbers)
    #     edge_index, edge_dist, edge_vec = data.edge_index, data.edge_dist, data.edge_vec
    #     spconfig = self.spnode_config

    #     # sample coordinates of spatial nodes
    #     with torch.amp.autocast("cuda", enabled=False):
    #         spnode_pos, _ = sample_virtual_nodes(data.pos, spconfig['num'], spconfig['sample'], self.data_config['pbc'], data.cell.reshape(3, 3).cpu().numpy(), repulsion_distance=spconfig['repulsion_distance'], kde_bandwidth=spconfig['kde_bandwidth'])
    #         # assert len(spnode_pos) == spconfig['num']

    #         p2v_edge_index, p2v_edge_dist, p2v_edge_vec = radius_graph_pbc_p2v(data.pos, spnode_pos, data.cell.reshape(3, 3), cutoff=10, max_num_neighbors=5000) # radius returns random max_num_neighbros, we need to maintain all edges and filter in `filter_edge`

    #         data.atomic_numbers = torch.cat([torch.ones(len(spnode_pos)).to(data.atomic_numbers)*spconfig['z'], data.atomic_numbers], dim=0)
    #         data.pos = torch.cat([spnode_pos, data.pos], dim=0)

    #         # concat edge of p2p and p2v
    #         edge_index += len(spnode_pos)
    #         p2v_edge_index[0] += len(spnode_pos)

    #         edge_index = torch.cat([edge_index, p2v_edge_index], dim=-1)
    #         edge_dist = torch.cat([edge_dist, p2v_edge_dist], dim=0)
    #         edge_vec = torch.cat([edge_vec, p2v_edge_vec], dim=0)

    #     # filter for cutoff and max_num_neighbors
    #     edge_index, edge_dist, edge_vec = filter_edge(edge_index, edge_dist, edge_vec, 'p2pv', data.atomic_numbers==self.spnode_config['z'], self.cutoff, self.max_num_neighbors, repulsion_distance=spconfig['repulsion_distance'])

    #     data.edge_index = edge_index
    #     data.edge_dist = edge_dist
    #     data.edge_vec = edge_vec

    #     self.compute_repulsion_mask(data)

    #     spedge_index, spedge_dist, spedge_vec = filter_edge(spdata.edge_index, spdata.edge_dist, spdata.edge_vec, 'p2pv', spdata.atomic_numbers==self.spnode_config['z'], self.cutoff, self.max_num_neighbors, repulsion_distance=spconfig['repulsion_distance'])
    #     spdata.edge_index = spedge_index
    #     spdata.edge_dist = spedge_dist
    #     spdata.edge_vec = spedge_vec

    #     # print("DEBUG", spedge_index.shape, data.edge_index.shape, float(spdata.edge_dist.min()), float(data.edge_dist.min()), float(spdata.edge_dist.max()), float(data.edge_dist.max()))

    #     self.compute_repulsion_mask(spdata)

    #     # return spdata
    #     return data

    def compute_repulsion_mask(self, data: Data):
        # mask
        spnode_mask = data.atomic_numbers == self.spnode_config["z"]
        atom_pos = data.pos[~spnode_mask]
        spnode_pos = data.pos[spnode_mask]

        dist_sp2atom = torch.cdist(spnode_pos, atom_pos)
        dist_sp2atom_min, _ = torch.min(dist_sp2atom, dim=1)  # N_sp
        data["repulsion_mask"] = (
            dist_sp2atom_min > self.spnode_config["repulsion_distance"]
        )  # N_sp

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
        if self.model_config["gnn"] == "gemnet":
            if self.head_type in ["atom_mlp", "global_attn", "gmt"]:
                graph_data = batch["gemnet"]
            elif self.head_type in ["transformer", "spnode_mlp"]:
                graph_data = batch["spgemnet"]
            else:
                raise ValueError(f"Unknown head type {self.head_type}")
            data_list = [self.data_transform(data) for data in graph_data]
        else:
            # default spgraph
            # we dynamically add p2v edge
            graph_data = None

            if self.head_type in ["transformer", "spnode_mlp"]:
                graph_data = batch["spgraph"]
            elif self.head_type in ["atom_mlp", "global_attn", "gmt"]:
                graph_data = batch["graph"]
            else:
                raise ValueError(f"Unknown head type {self.head_type}")

            data_list = [self.data_transform(data) for data in graph_data]

        # data_list = [self.data_transform(data) for data in graph_data]
        # data_list = [self.data_transform(data, spdata) for data, spdata in zip(graph_data, spgraph_data)]
        batch_graph = self.collate_fn(data_list)
        feat = self.extract_gnn_feat(batch_graph, **kwargs)
        atom_feat = feat["atom_feat"]
        atom_pos = feat["atom_pos"]

        batch_size = len(graph_data)

        if self.head_type in ["transformer", "spnode_mlp"]:
            spnode_feat = feat["spnode_feat"]
            spnode_pos = feat["spnode_pos"]

            if self.mp_between_spnode:
                spnode_feat = self.schnet(
                    spnode_feat,
                    spnode_pos,
                    batch_graph.batch[
                        batch_graph.atomic_numbers == self.spnode_config["z"]
                    ],
                )

        if self.head_type == "transformer":
            repulsion_mask = feat["repulsion_mask"]  # N_spnode
            # print("DEBUG", repulsion_mask.shape, repulsion_mask.sum())
            spnode_feat = self.spnode_proj(spnode_feat)
            atom_feat = self.atom_proj(atom_feat)

            assert len(spnode_feat) == self.spnode_config["num"] ** 3 * batch_size
            # Mask those spatial node that are too closer to atoms
            # spnode_feat[~repulsion_mask] = self.mask_emb.to(spnode_feat)
            spnode_feat = spnode_feat.reshape(
                batch_size,
                self.spnode_config["num"] ** 3,
                self.model_config["transformer"]["hid_dim"],
            )
            spnode_feat += self.pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
            atom_feat, atom_mask = to_dense_batch(
                atom_feat,
                batch_graph.batch[
                    batch_graph.atomic_numbers != self.spnode_config["z"]
                ],
            )
            # pos_emb = self.pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
            cls_emb = (
                self.cls_emb.unsqueeze(0).unsqueeze(0).repeat([batch_size, 1, 1])
            )  # 1， 1， hid_dim

            emb = torch.cat([cls_emb, spnode_feat], dim=1)  # bs, 1+num_spnode, hid_dim

            # repulsion_mask = batch_graph['repulsion_mask'].reshape(batch_size, self.spnode_config['num'])

            emb = self.transformer(emb, atom_feat, memory_key_padding_mask=~atom_mask)

            return {
                "cls_feat": emb[:, 0],
                "spnode_feat": emb[:, 1:],
                "repulsion_mask": repulsion_mask,
                "atom_feat": atom_feat,
                "atom_mask": atom_mask,
            }
        elif self.head_type == "spnode_mlp":
            repulsion_mask = feat["repulsion_mask"]
            spnode_feat = self.mlp(spnode_feat)
            return {
                "spnode_feat": spnode_feat,
                "atom_feat": atom_feat,
                "repulsion_mask": repulsion_mask,
                "atom_batch_idx": batch_graph.batch[
                    batch_graph.atomic_numbers != self.spnode_config["z"]
                ],
                "spnode_batch_idx": batch_graph.batch[
                    batch_graph.atomic_numbers == self.spnode_config["z"]
                ],
                # "voxel_volume": batch_graph.voxel_volume
            }
        elif self.head_type in ["atom_mlp", "global_attn", "gmt"]:
            atom_feat = self.mlp(atom_feat)
            return {
                "atom_feat": atom_feat,
                # "spnode_feat": spnode_feat,
                "atom_batch_idx": batch_graph.batch[
                    batch_graph.atomic_numbers != self.spnode_config["z"]
                ],
                # "spnode_batch_idx": batch_graph.batch[batch_graph.atomic_numbers == self.spnode_config['z']],
                "edge_index": batch_graph.edge_index,
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
        self.transformer_config = self.model_config["transformer"]
        self.painn_config = self.model_config["painn"]
        self.cutoff = self.painn_config["cutoff"]
        self.max_num_neighbors = self.painn_config.pop("max_num_neighbors")
        self.painn = PaiNNO(
            **self.painn_config,
        )
        # self.proj = nn.Linear(self.painn_config['n_atom_basis'], self.transformer_config['hid_dim'])

        self.gnn_hid_dim = self.painn_config["n_atom_basis"]

    # def data_transform(self, data: Data) -> Data:
    #     return super().data_transform(data)

    def extract_gnn_feat(self, batch_data: Batch) -> Dict[str, torch.Tensor]:
        q, mu = self.painn(batch_data)
        # print("DEBUG", q[:10, :10], q[-10:, :10])
        # q = self.proj(q)
        spnode_mask = batch_data.atomic_numbers == self.spnode_config["z"]
        return {
            "atom_feat": q[~spnode_mask],
            "spnode_feat": q[spnode_mask],
            "atom_pos": batch_data.pos[~spnode_mask],
            "spnode_pos": batch_data.pos[spnode_mask],
            "repulsion_mask": batch_data["repulsion_mask"],
        }


class JMP(BaseModule):
    def __init__(self):
        super().__init__()

    def init_gnn(self):
        self.gemnet_config = self.model_config["gemnet"]
        self.hid_dim = self.gemnet_config["hid_dim"]

        base_config = BackboneConfig.base()
        # base_config.scale_basis = False
        base_config.regress_forces = False
        base_config.direct_forces = False
        base_config.scale_file = self.gemnet_config["scale_file"]
        # Plus one for Virtual Node
        self.atom_embedding = nn.Embedding(120, self.hid_dim)
        # self.atom_embedding.apply(objectives.init_weights)
        # self.gemnet = nn.ModuleDict(
        #     {
        #         msg_direction: GemNetOCBackbone(base_config, **base_config)
        #         for msg_direction in self.msg_routes
        #     }
        # )
        self.gemnet = GemNetOCBackbone(base_config, **dict(base_config))
        if self.gemnet_config.get("ckpt") is not None:
            self.load_backbone_state_dict(self.gemnet_config["ckpt"])

        self.gnn_hid_dim = self.hid_dim

    def data_transform(self, data: Data) -> Data:
        self.compute_repulsion_mask(data)
        return data

    def load_backbone_state_dict(self, ckpt_path):
        # Due to that pre-trained JMP-S weight is used, the ckpt path is not configurable
        ckpt = torch.load(ckpt_path)
        # state_dict = ckpt["state_dict"]
        state_dict = retreive_ft_state_dict_from_loaded_ckpt(ckpt, load_emas=True)
        # state_dict = retreive_ft_state_dict_from_loaded_ckpt(ckpt)

        backbone_state_dict = filter_state_dict(state_dict, "backbone.")
        # due to that we remove symmetric mp, there are some unexpected keys to ignore
        load_ret = self.gemnet.load_state_dict(backbone_state_dict, strict=False)
        print("Load Backbone Statedict Return: ", load_ret)

        embedding_state_dict = filter_state_dict(
            state_dict, "embedding.atom_embedding."
        )
        # due to that we remove symmetric mp, there are some unexpected keys to ignore
        load_ret = self.atom_embedding.load_state_dict(
            embedding_state_dict, strict=True
        )
        print("Load Embedding Statedict Return: ", load_ret)

        # for msg_direction in self.msg_routes:
        #     load_ret = self.gemnet[msg_direction].load_state_dict(
        #         backbone_state_dict, strict=False
        #     )
        #     self.log(
        #         f"Load JMP Backbone Statedict for {msg_direction} Return: {str(load_ret)}",
        #         "info",
        #     )

    def extract_gnn_feat(self, batch_data: Batch) -> Dict[str, torch.Tensor]:
        x = batch_data["atomic_numbers"] - 1
        pos = batch_data["pos"]
        atom_batch_idx = batch_data["batch"]

        h = self.atom_embedding(x)

        # print("DEBUG BEFORE", h[:3, :5])

        # struct
        graph_output = self.gemnet(batch_data, h=h)
        h = graph_output["energy"]

        spnode_mask = batch_data.atomic_numbers == self.spnode_config["z"]
        return {
            "atom_feat": h[~spnode_mask],
            "spnode_feat": h[spnode_mask],
            "atom_pos": batch_data.pos[~spnode_mask],
            "spnode_pos": batch_data.pos[spnode_mask],
            "repulsion_mask": batch_data["repulsion_mask"],
        }


def construct_model():
    model_config = get_model_config()
    gnn = model_config["gnn"]

    if gnn == "painn":
        return PaiNN()
    elif gnn == "gemnet":
        return JMP()
    elif gnn == "gemnetpainn":
        return JMPPaiNN()
    elif gnn == "visnetpainn":
        return ViSNetPaiNN()
    else:
        raise ValueError(f"Unsupported gnn type: {gnn}")
