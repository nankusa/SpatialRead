# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

from typing import Optional, Tuple
import torch
from torch import Tensor, nn
from .utils import (
    NeighborEmbedding,
    CosineCutoff,
    OptimizedDistance,
    rbf_class_mapping,
    act_class_mapping,
    scatter,
)
from torch_geometric.nn.pool import radius_graph

from .radius_graph import radius_graph_pbc
from .ocp import get_pbc_distances

import torch

def limit_in_edges(edge_index, edge_weight, max_in_degree=32):
    src, dst = edge_index  # [N], [N]
    N = edge_index.shape[1]

    # Step 1: 排序 - 先按 dst 分组，然后按每个组内部 edge_weight 的绝对值排序
    abs_weight = edge_weight.abs()
    
    # 为每条边生成索引
    edge_idx = torch.arange(N, device=edge_index.device)

    # Lexicographical sort: 先按 dst 升序，再按 abs_weight 降序
    sort_key = dst * abs_weight.max() + abs_weight  # 保证先按 dst，再按 abs_weight 降序
    perm = torch.argsort(sort_key)

    sorted_src = src[perm]
    sorted_dst = dst[perm]
    sorted_weight = edge_weight[perm]
    sorted_edge_idx = edge_idx[perm]

    # Step 2: 计算每个 dst 的入边数量
    unique_dst, counts = torch.unique_consecutive(sorted_dst, return_counts=True)

    # Step 3: 构造掩码：只保留每个 dst 的前 max_in_degree 条边
    mask = torch.zeros(N, dtype=torch.bool, device=edge_index.device)
    start = 0
    for count in counts:
        end = start + count.item()
        mask[start : min(end, start + max_in_degree)] = True
        start = end

    # Step 4: 恢复原始顺序
    selected = perm[mask]
    new_edge_index = edge_index[:, selected]
    new_edge_weight = edge_weight[selected]

    return selected


# 示例使用
# edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 1, 1, 2, 2, 2]], dtype=torch.long)
# edge_weight = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float)
# filtered_edge_index, filtered_edge_weight = filter_edges(edge_index, edge_weight)
# print("Filtered edge_index:", filtered_edge_index)
# print("Filtered edge_weight:", filtered_edge_weight)


class TorchMD_ET(nn.Module):
    r"""Equivariant Transformer's architecture. From
    Equivariant Transformers for Neural Network based Molecular Potentials; P. Tholke and G. de Fabritiis.
    ICLR 2022.

    This function optionally supports periodic boundary conditions with arbitrary triclinic boxes.
    For a given cutoff, :math:`r_c`, the box vectors :math:`\vec{a},\vec{b},\vec{c}` must satisfy
    certain requirements:

    .. math::

      \begin{align*}
      a_y = a_z = b_z &= 0 \\
      a_x, b_y, c_z &\geq 2 r_c \\
      a_x &\geq 2  b_x \\
      a_x &\geq 2  c_x \\
      b_y &\geq 2  c_y
      \end{align*}

    These requirements correspond to a particular rotation of the system and reduced form of the vectors, as well as the requirement that the cutoff be no larger than half the box width.

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_layers (int, optional): The number of attention layers.
            (default: :obj:`6`)
        num_rbf (int, optional): The number of radial basis functions :math:`\mu`.
            (default: :obj:`50`)
        rbf_type (string, optional): The type of radial basis function to use.
            (default: :obj:`"expnorm"`)
        trainable_rbf (bool, optional): Whether to train RBF parameters with
            backpropagation. (default: :obj:`True`)
        activation (string, optional): The type of activation function to use.
            (default: :obj:`"silu"`)
        attn_activation (string, optional): The type of activation function to use
            inside the attention mechanism. (default: :obj:`"silu"`)
        neighbor_embedding (bool, optional): Whether to perform an initial neighbor
            embedding step. (default: :obj:`True`)
        num_heads (int, optional): Number of attention heads.
            (default: :obj:`8`)
        distance_influence (string, optional): Where distance information is used inside
            the attention mechanism. (default: :obj:`"both"`)
        cutoff_lower (float, optional): Lower cutoff distance for interatomic interactions.
            (default: :obj:`0.0`)
        cutoff_upper (float, optional): Upper cutoff distance for interatomic interactions.
            (default: :obj:`5.0`)
        max_z (int, optional): Maximum atomic number. Used for initializing embeddings.
            (default: :obj:`100`)
        max_num_neighbors (int, optional): Maximum number of neighbors to return for a
            given node/atom when constructing the molecular graph during forward passes.
            This attribute is passed to the torch_cluster radius_graph routine keyword
            max_num_neighbors, which normally defaults to 32. Users should set this to
            higher values if they are using higher upper distance cutoffs and expect more
            than 32 neighbors per node/atom.
            (default: :obj:`32`)
        box_vecs (Tensor, optional):
            The vectors defining the periodic box.  This must have shape `(3, 3)`,
            where `box_vectors[0] = a`, `box_vectors[1] = b`, and `box_vectors[2] = c`.
            If this is omitted, periodic boundary conditions are not applied.
            (default: :obj:`None`)
        vector_cutoff (bool, optional): Whether to apply the cutoff to the vector features. This prevents the energy from being discontinuous at the cutoff, but may hinder training.
            (default: :obj:`False`)
        check_errors (bool, optional): Whether to check for errors in the distance module.
            (default: :obj:`True`)

    """

    def __init__(
        self,
        hidden_channels=128,
        num_layers=6,
        num_rbf=50,
        rbf_type="expnorm",
        trainable_rbf=True,
        activation="silu",
        attn_activation="silu",
        neighbor_embedding=True,
        num_heads=8,
        distance_influence="both",
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        max_z=100,
        max_num_neighbors=32,
        check_errors=True,
        box_vecs=None,
        vector_cutoff=False,
        dtype=torch.float32,
        VNODE_Z=None,
        deform=False
    ):
        super(TorchMD_ET, self).__init__()

        assert distance_influence in ["keys", "values", "both", "none"]
        assert rbf_type in rbf_class_mapping, (
            f'Unknown RBF type "{rbf_type}". '
            f'Choose from {", ".join(rbf_class_mapping.keys())}.'
        )
        assert activation in act_class_mapping, (
            f'Unknown activation function "{activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )
        assert attn_activation in act_class_mapping, (
            f'Unknown attention activation function "{attn_activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.attn_activation = attn_activation
        self.neighbor_embedding = neighbor_embedding
        self.num_heads = num_heads
        self.distance_influence = distance_influence
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_z = max_z
        self.dtype = dtype
        self.VNODE_Z = VNODE_Z
        self.max_num_neighbors = max_num_neighbors
        self.deform=deform

        act_class = act_class_mapping[activation]

        self.embedding = nn.Embedding(self.max_z, hidden_channels, dtype=dtype)

        self.distance = OptimizedDistance(
            cutoff_lower,
            cutoff_upper,
            max_num_pairs=-max_num_neighbors,
            return_vecs=True,
            loop=True,
            box=box_vecs,
            long_edge_index=True,
            check_errors=check_errors,
        )
        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )
        self.neighbor_embedding = (
            NeighborEmbedding(
                hidden_channels, num_rbf, cutoff_lower, cutoff_upper, self.max_z, dtype
            )
            if neighbor_embedding
            else None
        )
        self.dropout = nn.Dropout(p=0.2)

        self.attention_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = EquivariantMultiHeadAttention(
                hidden_channels,
                num_rbf,
                distance_influence,
                num_heads,
                act_class,
                attn_activation,
                cutoff_lower,
                cutoff_upper,
                vector_cutoff,
                dtype,
            )
            self.attention_layers.append(layer)

        self.out_norm = nn.LayerNorm(hidden_channels, dtype=dtype)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        if self.neighbor_embedding is not None:
            self.neighbor_embedding.reset_parameters()
        for attn in self.attention_layers:
            attn.reset_parameters()
        self.out_norm.reset_parameters()

    def forward(
        self,
        data,
        # z: Tensor,
        # pos: Tensor,
        # batch: Tensor,
        box: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        s: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        z = data.atomic_numbers
        pos = data.pos
        batch = data.batch

        x = self.embedding(z)

        # print(data.cell)

        # edge_index = radius_graph(pos, 30.0, batch, loop=False, max_num_neighbors=5000)
        # edge_index, cell_offsets, neighbors = radius_graph_pbc(data, self.cutoff_upper, self.max_num_neighbors)
        edge_index, cell_offsets, neighbors = radius_graph_pbc(data, self.cutoff_upper, 100)
        out = get_pbc_distances(
            data.pos,
            edge_index,
            data.cell,
            cell_offsets,
            neighbors,
            return_offsets=True,
            return_distance_vec=True,
        )

        edge_index: torch.Tensor = out["edge_index"]
        edge_weight: torch.Tensor = out["distances"]
        cell_offset_distances: torch.Tensor = out["offsets"]
        edge_vec: torch.Tensor = out["distance_vec"]

        indices = z[edge_index[0]] != self.VNODE_Z
        edge_index = edge_index[:, indices]
        edge_weight = edge_weight[indices]
        edge_vec = edge_vec[indices]


        # edge_weight = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)

        # print("A", edge_index.shape, edge_index[1].max(), edge_index[1].min())

        # print("B", edge_index.shape, edge_index[1].max(), edge_index[1].min(), self.cutoff_upper)

        # print("DEBUG", edge_weight[edge_index[1] == edge_index[1].max()], z[edge_index[1].max()])

        # indices_1 = torch.logical_and(edge_weight < self.cutoff_upper, z[edge_index[1]] != self.VNODE_Z)
        # indices_2 = torch.logical_and(edge_weight < self.cutoff_upper, z[edge_index[1]] == self.VNODE_Z)
        # indices = torch.logical_or(indices_1, indices_2)
        # edge_index = edge_index[:, indices]
        # edge_weight = edge_weight[indices]

        # print("C", edge_index.shape, edge_index[1].max(), edge_index[1].min(), edge_weight.max(), edge_weight.min(), edge_weight)

        indices = limit_in_edges(edge_index, edge_weight, max_in_degree=self.max_num_neighbors)
        edge_index = edge_index[:, indices]
        edge_weight = edge_weight[indices]
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        # print("D", edge_index.shape, edge_index[1].max(), edge_index[1].min(), edge_weight.max(), edge_weight.min(), edge_weight)

        # edge_index, edge_weight, edge_vec = self.distance(pos, batch, box)
        # # print("DEBUG", edge_index.shape, edge_weight, z.shape, edge_index[1].max())
        # edge_index = radius_graph(pos, self.cutoff_upper, batch, loop=True, max_num_neighbors=self.max_num_neighbors)
        # edge_weight = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)

        # This assert must be here to convince TorchScript that edge_vec is not None
        # If you remove it TorchScript will complain down below that you cannot use an Optional[Tensor]
        assert (
            edge_vec is not None
        ), "Distance module did not return directional information"

        edge_attr = self.distance_expansion(edge_weight)
        mask = edge_index[0] != edge_index[1]
        # print("DEBUG BEFORE", edge_vec.max(), edge_vec.min(), torch.norm(edge_vec, dim=1).max(), torch.norm(edge_vec, dim=1).min())
        # indices = torch.norm(edge_vec, dim=1) < 0.01
        # print("DEBUG", z, edge_index[:, indices], z[edge_index[0, indices]], z[edge_index[1, indices]])
        # indices = edge_weight < 0.01
        # print("DEBUGAAAAAA", edge_index[:, indices])
        edge_vec[mask] = edge_vec[mask] / torch.norm(edge_vec[mask], dim=1).unsqueeze(1)
        # print("DEBUG AFTER", edge_vec.max(), edge_vec.min())

        if self.neighbor_embedding is not None:
            x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)

        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device, dtype=x.dtype)

        for attn in self.attention_layers:
            dx, dvec = attn(x, vec, edge_index, edge_weight, edge_attr, edge_vec)
            x = x + dx
            x = self.dropout(x)
            vec = vec + dvec
        feat = self.out_norm(x)

        vn_indices = z == self.VNODE_Z
        pn_indices = z != self.VNODE_Z

        return {
            "pos": pos,
            "vn_pos": pos[vn_indices],
            "pn_pos": pos[pn_indices],
            "vn_indices": vn_indices,
            "pn_indices": pn_indices,
            "x": x,
            "pn_x": x[pn_indices],
            "vn_x": x[vn_indices],
            "v": vec,
            "pn_v": vec[pn_indices],
            "vn_v": vec[vn_indices],
            "data": data,
            "feat": feat,
            "batch": batch,
            "vn_feat": feat[vn_indices],
            "vn_batch": batch[vn_indices],
            "pn_feat": feat[pn_indices],
            "pn_batch": batch[pn_indices],
        }

        # return x, vec, z, pos, batch

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"num_rbf={self.num_rbf}, "
            f"rbf_type={self.rbf_type}, "
            f"trainable_rbf={self.trainable_rbf}, "
            f"activation={self.activation}, "
            f"attn_activation={self.attn_activation}, "
            f"neighbor_embedding={self.neighbor_embedding}, "
            f"num_heads={self.num_heads}, "
            f"distance_influence={self.distance_influence}, "
            f"cutoff_lower={self.cutoff_lower}, "
            f"cutoff_upper={self.cutoff_upper}), "
            f"dtype={self.dtype}"
        )


class EquivariantMultiHeadAttention(nn.Module):
    """Equivariant multi-head attention layer.

    :meta private:
    """

    def __init__(
        self,
        hidden_channels,
        num_rbf,
        distance_influence,
        num_heads,
        activation,
        attn_activation,
        cutoff_lower,
        cutoff_upper,
        vector_cutoff=False,
        dtype=torch.float32,
    ):
        super(EquivariantMultiHeadAttention, self).__init__()
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        self.layernorm = nn.LayerNorm(hidden_channels, dtype=dtype)
        self.act = activation()
        self.attn_activation = act_class_mapping[attn_activation]()
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.q_proj = nn.Linear(hidden_channels, hidden_channels, dtype=dtype)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels, dtype=dtype)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels * 3, dtype=dtype)
        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3, dtype=dtype)

        self.vec_proj = nn.Linear(
            hidden_channels, hidden_channels * 3, bias=False, dtype=dtype
        )

        self.dk_proj = None
        if distance_influence in ["keys", "both"]:
            self.dk_proj = nn.Linear(num_rbf, hidden_channels, dtype=dtype)

        self.dv_proj = None
        if distance_influence in ["values", "both"]:
            self.dv_proj = nn.Linear(num_rbf, hidden_channels * 3, dtype=dtype)
        self.vector_cutoff = vector_cutoff

        self.reset_parameters()

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.vec_proj.weight)
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)
        if self.dv_proj:
            nn.init.xavier_uniform_(self.dv_proj.weight)
            self.dv_proj.bias.data.fill_(0)

    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij):
        x = self.layernorm(x)
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim * 3)

        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec = vec.reshape(-1, 3, self.num_heads, self.head_dim)
        vec_dot = (vec1 * vec2).sum(dim=1)

        dk = (
            self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
            if self.dk_proj is not None
            else None
        )
        dv = (
            self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim * 3)
            if self.dv_proj is not None
            else None
        )
        x, vec = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            vec=vec,
            dk=dk,
            dv=dv,
            r_ij=r_ij,
            d_ij=d_ij,
            dim_size=None,
        )
        x = x.reshape(-1, self.hidden_channels)
        vec = vec.reshape(-1, 3, self.hidden_channels)

        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)

        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec
        return dx, dvec

    def propagate(
        self,
        edge_index: Tensor,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        vec: Tensor,
        dk: Optional[Tensor],
        dv: Optional[Tensor],
        r_ij: Tensor,
        d_ij: Tensor,
        dim_size: Optional[int],
    ) -> Tuple[Tensor, Tensor]:
        # print("DEBUG", edge_index[0].max(), edge_index[1].max(), edge_index[0].min(), edge_index[1].min())
        q_i = q.index_select(0, edge_index[1])
        k_j = k.index_select(0, edge_index[0])
        v_j = v.index_select(0, edge_index[0])
        vec_j = vec.index_select(0, edge_index[0])
        x, vec = self.message(q_i, k_j, v_j, vec_j, dk, dv, r_ij, d_ij)
        return self.aggregate((x, vec), edge_index[1], dim_size=dim_size)

    def message(
        self,
        q_i: Tensor,
        k_j: Tensor,
        v_j: Tensor,
        vec_j: Tensor,
        dk: Optional[Tensor],
        dv: Optional[Tensor],
        r_ij: Tensor,
        d_ij: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # attention mechanism
        if dk is None:
            attn = (q_i * k_j).sum(dim=-1)
        else:
            attn = (q_i * k_j * dk).sum(dim=-1)

        # attention activation function
        cutoff = self.cutoff(r_ij).unsqueeze(1)
        attn = self.attn_activation(attn)

        # print("DEBUG", attn.shape, attn.max(), attn.min())

        # The original ET arquitecture only weights the attention with the cutoff function,
        #  this causes a discontinuity in the energy at the cutoff, since the bias of the dv_proj
        #  layer might be non-zero.
        # This option makes it so that both the scalar and vector features are weighted with the cutoff.
        if self.vector_cutoff:
            v_j = v_j * cutoff.unsqueeze(2)
        else:
            attn = attn * cutoff
        # value pathway
        if dv is not None:
            v_j = v_j * dv
        x, vec1, vec2 = torch.split(v_j, self.head_dim, dim=2)

        # update scalar features
        x = x * attn.unsqueeze(2)
        # update vector features
        vec = vec_j * vec1.unsqueeze(1) + vec2.unsqueeze(1) * d_ij.unsqueeze(
            2
        ).unsqueeze(3)
        return x, vec

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=0, dim_size=dim_size)
        vec = scatter(vec, index, dim=0, dim_size=dim_size)
        return x, vec

    def update(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs
