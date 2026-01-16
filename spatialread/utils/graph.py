import torch
from torch_geometric.data import HeteroData, Data, Batch
from typing import Union, Literal, List
from torch_geometric.nn.pool import radius_graph as pyg_radius_graph
from torch_cluster import radius

from ase.neighborlist import neighbor_list
from ase import Atoms


import torch
from torch import Tensor


def radius_graph_pbc_p2v(
    pnode_pos: Tensor,
    vnode_pos: Tensor,
    cell: Tensor,
    cutoff: float = 5.0,
    max_num_neighbors: int = 32,
):
    """
    Compute edge_index, distances and vectors from pnode to vnode
    with periodic boundary conditions, using minimal-image convention.

    Args:
        pnode_pos: (N, 3) positions of physical nodes
        vnode_pos: (M, 3) positions of virtual nodes
        cell: (3, 3) or (1, 3, 3) cell matrix (a1, a2, a3 as rows)
        cutoff: cutoff radius (Å)
        max_num_neighbors: maximum number of pnodes connected to each vnode.
            If <= 0 or very large, effectively no limit.

    Returns:
        edge_index: (2, E), [0] = pnode indices in [0, N-1],
                              [1] = vnode indices in [0, M-1]
        distances: (E,) distances under PBC minimal image
        rel_vec:   (E, 3) vectors from pnode -> vnode under PBC minimal image
    """
    # ---- sanitize inputs ----
    device = pnode_pos.device
    pnode_pos = torch.as_tensor(pnode_pos, dtype=torch.float32, device=device)
    vnode_pos = torch.as_tensor(vnode_pos, dtype=torch.float32, device=device)

    cell = torch.as_tensor(cell, dtype=torch.float32, device=device)
    if cell.dim() == 3:  # (1, 3, 3) or (B,3,3); 我们只支持单一结构，取第一项
        cell = cell[0]
    assert cell.shape == (3, 3), f"cell must be (3,3), got {cell.shape}"

    N = pnode_pos.shape[0]
    M = vnode_pos.shape[0]

    # ---- 1. 计算分数坐标 (fractional coordinates) ----
    # r_cart = f @ cell  =>  f = r_cart @ cell^{-1}
    cell_inv = torch.inverse(cell)             # (3,3)
    p_frac = pnode_pos @ cell_inv              # (N,3)
    v_frac = vnode_pos @ cell_inv              # (M,3)

    # ---- 2. 对所有 (p, v) 计算最小镜像下的位移 ----
    # delta_f = f_v - f_p, shape (N, M, 3)
    # 然后 wrap 到 [-0.5, 0.5) 保证是最近镜像
    delta_f = v_frac.unsqueeze(0) - p_frac.unsqueeze(1)   # (N, M, 3)
    delta_f = delta_f - torch.round(delta_f)              # wrap to [-0.5, 0.5)

    # 回到笛卡尔坐标：delta_r = delta_f @ cell
    # (N,M,3) @ (3,3) => (N,M,3)
    delta_r = torch.matmul(delta_f, cell)

    # 距离矩阵 (N,M)
    dist = torch.linalg.norm(delta_r, dim=-1)

    # ---- 3. cutoff 掩码 ----
    mask = dist <= cutoff
    if not mask.any():
        # 没有任何边
        edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        distances = torch.empty(0, dtype=torch.float32, device=device)
        rel_vec = torch.empty(0, 3, dtype=torch.float32, device=device)
        return edge_index, distances, rel_vec

    # 找到所有满足 dist <= cutoff 的 (p, v) 对
    p_idx, v_idx = mask.nonzero(as_tuple=True)   # (E,), (E,)
    distances = dist[p_idx, v_idx]               # (E,)
    rel_vec = delta_r[p_idx, v_idx, :]           # (E,3)

    # ---- 4. 按每个 vnode 做 max_num_neighbors 限制（最近 K 个 pnode）----
    if max_num_neighbors is not None and max_num_neighbors > 0:
        # 我们对每个 vnode 独立筛选
        keep_mask = torch.ones_like(distances, dtype=torch.bool)

        # 这里用简单循环，M 一般不大（如 512），开销可以接受
        for v in range(M):
            # 找到所有连到 vnode v 的边的下标
            mask_v = (v_idx == v)
            idx_v = torch.nonzero(mask_v, as_tuple=False).view(-1)
            num_v = idx_v.numel()
            if num_v <= max_num_neighbors:
                continue  # 不需要截断

            # 取距离最近的 max_num_neighbors 个 pnode
            d_v = distances[idx_v]                          # (num_v,)
            # 按距离升序排序
            sorted_d, order = torch.sort(d_v)
            keep_local = idx_v[order[:max_num_neighbors]]   # 保留这些全局边 index

            # 其他的标记为 False
            drop_local = idx_v[order[max_num_neighbors:]]
            keep_mask[drop_local] = False

        # 应用 keep_mask
        p_idx = p_idx[keep_mask]
        v_idx = v_idx[keep_mask]
        distances = distances[keep_mask]
        rel_vec = rel_vec[keep_mask]

    # ---- 5. 构造 edge_index 并返回 ----
    edge_index = torch.stack([p_idx, v_idx], dim=0)   # (2,E)

    return edge_index, distances, rel_vec



# def radius_graph_pbc_p2v(
#     pnode_pos, vnode_pos, cell, cutoff=5.0, max_num_neighbors=32
# ):
#     """
#     Compute edge_index, distances and vectors from pnode to vnode with periodic boundary conditions

#     Args:
#         pnode_pos: pnode position tensor of shape (N, 3)
#         vnode_pos: vnode position tensor of shape (M, 3)
#         cell: unit cell matrix of shape (3, 3)
#         cutoff: cutoff radius
#         max_num_neighbors: maximum number of neighbor nodes

#     Returns:
#         edge_index: edge index tensor of shape (2, E), first row is pnode indices, second row is vnode indices
#         distances: distance tensor of shape (E,)
#         vecs: edge vector tensor of shape (E, 3)
#     """

#     # Ensure inputs are torch tensors
#     pnode_pos = torch.as_tensor(pnode_pos, dtype=torch.float32)
#     vnode_pos = torch.as_tensor(vnode_pos, dtype=torch.float32)
#     cell = torch.as_tensor(cell, dtype=torch.float32)

#     N = pnode_pos.shape[0]  # number of pnodes
#     M = vnode_pos.shape[0]  # number of vnodes

#     # Generate periodic images
#     # Consider the nearest 27 images (including the original cell)
#     shifts = torch.tensor(
#         [[i, j, k] for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1]],
#     ).to(pnode_pos)

#     # Convert image shifts to real space displacements
#     periodic_shifts = shifts @ cell

#     # Create periodic images for each pnode (correct approach)
#     pnode_pos_expanded = pnode_pos.unsqueeze(1) + periodic_shifts.unsqueeze(
#         0
#     )  # (N, 27, 3)
#     pnode_pos_expanded = pnode_pos_expanded.reshape(-1, 3)  # (N*27, 3)

#     # Create corresponding original pnode indices for each image pnode
#     pnode_indices = torch.arange(N, dtype=torch.long).to(pnode_pos.device).repeat_interleave(27)

#     # Use radius_graph to find neighbors
#     # Use vnode_pos as center points, pnode_pos_expanded as neighbor points
#     row, col = radius(
#         x=pnode_pos_expanded,  # neighbor points (image pnodes)
#         y=vnode_pos,  # center points (original vnodes)
#         r=cutoff,
#         max_num_neighbors=max_num_neighbors,
#     )
#     edge_index = torch.stack([col, row])

#     # Convert image indices to original pnode indices
#     original_pnode_indices = pnode_indices[edge_index[0]]
#     vnode_indices = edge_index[1]

#     # Create final edge_index
#     final_edge_index = torch.stack([original_pnode_indices, vnode_indices], dim=0)

#     # Calculate edge vectors and distances
#     p_indices = original_pnode_indices  # original pnode indices
#     v_indices = vnode_indices  # vnode indices

#     # Get image pnode positions and vnode positions
#     mirror_p_pos = pnode_pos_expanded[edge_index[0]]  # image pnode positions
#     v_pos = vnode_pos[v_indices]  # vnode positions

#     # Calculate relative position vectors (directly using image pnode to vnode vectors)
#     rel_pos_cart = v_pos - mirror_p_pos

#     # Calculate distances
#     distances = torch.norm(rel_pos_cart, dim=1)

#     # Create final edge_index (no deduplication, keep all edges)
#     final_edge_index = torch.stack([p_indices, v_indices], dim=0)

#     return final_edge_index, distances, rel_pos_cart


def radius_graph_pbc_ase(data: Data, radius: float, max_num_neighbors: int):
    if data.get('z') is not None:
        z = data.z.cpu().numpy()
    else:
        z = data.atomic_numbers.cpu().numpy()
    positions = data.pos.cpu().numpy()
    cell = data.cell.cpu().numpy()

    atoms = Atoms(numbers=z, positions=positions, cell=cell, pbc=True)

    i, j, D, S = neighbor_list("ijDS", atoms, cutoff=radius, self_interaction=False)

    cell = atoms.cell.array  # (3, 3)

    disp = D

    src = torch.tensor(i, dtype=torch.long)
    dst = torch.tensor(j, dtype=torch.long)
    shift = torch.tensor(disp, dtype=torch.float32)

    dist = torch.norm(shift, dim=-1)
    edge_index = torch.stack([src, dst])

    indices = limit_in_edges(edge_index, dist, max_num_neighbors)
    
    return edge_index[:, indices], dist[indices], shift[indices], S[indices] # edge_index, edge_distance, edge_vec, cell_offset


# return edge_index, edge_dist, edge_vec
def radius_graph(
    data: Data,
    cutoff: float = 6.0,
    max_num_neighbors: int = 30,
    pbc: bool = False,
    rep=[None, None, None],
):
    if isinstance(data, HeteroData):
        raise ValueError(
            "Undefined behavior for HeteroData, please use hetero_to_homo first."
        )

    if isinstance(data, Batch):
        raise ValueError("Undefined behavior for Batch.")

    if pbc:
        return radius_graph_pbc_ase(data, cutoff, max_num_neighbors)
        # edge_index, cell_offsets, neighbors = radius_graph_pbc(data, cutoff, max_num_neighbors, rep=rep)
        # out = get_pbc_distances(
        #     data.pos,
        #     edge_index,
        #     data.cell,
        #     cell_offsets,
        #     neighbors,
        #     return_offsets=True,
        #     return_distance_vec=True,
        # )

        # edge_index: torch.Tensor = out["edge_index"]
        # # edge_dist: torch.Tensor = out["distances"]
        # edge_dist: torch.Tensor = out["distances"]
        # cell_offset_distances: torch.Tensor = out["offsets"]
        # edge_vec: torch.Tensor = out["distance_vec"]

        # return edge_index, edge_dist, edge_vec
    else:
        pos = data.pos

        edge_index = pyg_radius_graph(pos, cutoff, max_num_neighbors=max_num_neighbors)
        edge_dist = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        return edge_index, edge_dist, edge_vec


def edge_indices(
    edge_index: torch.Tensor,
    vn_mask: torch.Tensor,
    msg_direction: Literal["p2p", "p2v", "v2v", "v2p", "p2pv"],
):
    if msg_direction == "p2p":
        indices = torch.logical_and(
            vn_mask[edge_index[0]] == 0, vn_mask[edge_index[1]] == 0
        )
        return indices
    elif msg_direction == "p2v":
        indices = torch.logical_and(
            vn_mask[edge_index[0]] == 0, vn_mask[edge_index[1]] == 1
        )
        return indices
    elif msg_direction == "v2v":
        indices = torch.logical_and(
            vn_mask[edge_index[0]] == 1, vn_mask[edge_index[1]] == 1
        )
        return indices
    elif msg_direction == "v2p":
        indices = torch.logical_and(
            vn_mask[edge_index[0]] == 1, vn_mask[edge_index[1]] == 0
        )
        return indices
    elif msg_direction == "p2pv":
        indices = vn_mask[edge_index[0]] == 0
        return indices
    else:
        raise ValueError(f"Unknown msg_direction: {msg_direction}")


def limit_in_edges(edge_index, edge_dist, max_in_degree):
    src, dst = edge_index  # [N], [N]
    N = edge_index.shape[1]

    # Step 1: 排序 - 先按 dst 分组，然后按每个组内部 edge_dist 的绝对值排序
    abs_weight = edge_dist.abs()

    # 为每条边生成索引
    edge_idx = torch.arange(N, device=edge_index.device)

    # Lexicographical sort: 先按 dst 升序，再按 abs_weight 降序
    sort_key = dst * abs_weight.max() + abs_weight  # 保证先按 dst，再按 abs_weight 降序
    perm = torch.argsort(sort_key)

    sorted_src = src[perm]
    sorted_dst = dst[perm]
    sorted_weight = edge_dist[perm]
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
    new_edge_dist = edge_dist[selected]

    return selected


def filter_edge(
    edge_index: torch.Tensor,
    edge_dist: torch.Tensor,
    edge_vec: torch.Tensor,
    msg_direction: Literal["p2p", "p2v", "v2v", "v2p", "p2pv", None],
    vn_mask: torch.Tensor,
    cutoff: float,
    max_num_neighbors: int,
    other_content: List[torch.Tensor] = [],
    repulsion_distance: float = 0
):
    if msg_direction is not None:
        assert msg_direction in [
            "p2p",
            "p2v",
            "v2v",
            "v2p",
            "p2pv",
        ], f"Unknown msg_direction: {msg_direction}"

        indices = edge_indices(edge_index, vn_mask, msg_direction)
        edge_index = edge_index[:, indices]
        edge_dist = edge_dist[indices]
        edge_vec = edge_vec[indices]

        for content in range(len(other_content)):
            other_content[content] = other_content[content][indices]

    # NOTE: For homo data, it should be noted that some VNs will be connected to no any pn
    # This will result in indexing error in some GNNs using aggregation, due to no message can be aggregated for this vn
    if cutoff is not None:
        indices = edge_dist < cutoff
        edge_index = edge_index[:, indices]
        edge_dist = edge_dist[indices]
        edge_vec = edge_vec[indices]

        for content in range(len(other_content)):
            other_content[content] = other_content[content][indices]

    # indices = ~is_vnode[edge_index[0]]
    # edge_index = edge_index[:, indices]
    # edge_dist = edge_dist[indices]
    # edge_vec = edge_vec[indices]

    if max_num_neighbors is not None:
        indices = limit_in_edges(edge_index, edge_dist, max_in_degree=max_num_neighbors)
        edge_index = edge_index[:, indices]
        edge_dist = edge_dist[indices]
        edge_vec = edge_vec[indices]

        for content in range(len(other_content)):
            other_content[content] = other_content[content][indices]

    indices = edge_dist > repulsion_distance
    edge_index = edge_index[:, indices]
    edge_dist = edge_dist[indices]
    edge_vec = edge_vec[indices]

    for content in range(len(other_content)):
        other_content[content] = other_content[content][indices]

    if len(other_content) > 0:
        return edge_index, edge_dist, edge_vec, other_content

    return edge_index, edge_dist, edge_vec


def calc_edge_index(
    data: Union[HeteroData, Data, List[Union[HeteroData, Data]], Batch],
    cutoff,
    max_num_neighbors,
    pbc,
    msg_direction=None,
    fully_connected: bool = False,
):
    if fully_connected:
        assert cutoff is None, "cutoff should be None for fully connected graph"
        assert (
            max_num_neighbors is None
        ), "max_num_neighbors should be None for fully connected graph"

    if isinstance(data, Batch):
        raise ValueError(
            "Unsupported Batch for calc_edge_index, convert to list first."
        )
        data_list = data.to_data_list()
        return Batch.from_data_list(
            [
                calc_edge_index(
                    d, cutoff, max_num_neighbors, pbc, msg_direction, fully_connected
                )
                for d in data_list
            ]
        )

    if isinstance(data, list):
        data = [
            calc_edge_index(
                d, cutoff, max_num_neighbors, pbc, msg_direction, fully_connected
            )
            for d in data
        ]
        return data

    if isinstance(data, HeteroData):
        homo_data = hetero_to_homo(data)
        # prevent too large graph for pbc system and fully connected graph
        edge_index, edge_dist, edge_vec = radius_graph(
            homo_data, 19, 1000, pbc, rep=[2, 2, 2]
        )
        vn_mask = homo_data.vn_mask
        if fully_connected:
            edge_index, edge_dist, edge_vec = filter_edge(
                edge_index, edge_dist, edge_vec, msg_direction, vn_mask, None, None
            )
        else:
            edge_index, edge_dist, edge_vec = filter_edge(
                edge_index,
                edge_dist,
                edge_vec,
                msg_direction,
                vn_mask,
                cutoff,
                max_num_neighbors,
            )
        # NOTE: assume all pn are before vn, see hetero_to_homo
        num_pn = homo_data.z.shape[0] - vn_mask.sum().long()
        num_vn = vn_mask.sum().long()
        if msg_direction == "p2p":
            data["pnode", "to", "pnode"].edge_index = edge_index
            data["pnode", "to", "pnode"].edge_dist = edge_dist
            data["pnode", "to", "pnode"].edge_vec = edge_vec
        elif msg_direction == "p2v":
            p2v_edge_index = edge_index.clone()
            p2v_edge_index[1] -= num_pn
            data["pnode", "to", "vnode"].edge_index = p2v_edge_index
            data["pnode", "to", "vnode"].edge_dist = edge_dist
            data["pnode", "to", "vnode"].edge_vec = edge_vec
        elif msg_direction == "v2v":
            v2v_edge_index = edge_index.clone()
            v2v_edge_index[0] -= num_pn
            v2v_edge_index[1] -= num_pn
            data["vnode", "to", "vnode"].edge_index = v2v_edge_index
            data["vnode", "to", "vnode"].edge_dist = edge_dist
            data["vnode", "to", "vnode"].edge_vec = edge_vec
        elif msg_direction == "v2p":
            v2p_edge_index = edge_index.clone()
            v2p_edge_index[0] -= num_pn
            data["vnode", "to", "pnode"].edge_index = v2p_edge_index
            data["vnode", "to", "pnode"].edge_dist = edge_dist
            data["vnode", "to", "pnode"].edge_vec = edge_vec
        else:
            raise ValueError(
                f"Unsupported msg_direction for hetero graph: {msg_direction}"
            )
        return data
    elif isinstance(data, Data):
        # assert msg_direction is None, "msg_direction should be None for homo graph"
        edge_index, edge_dist, edge_vec = radius_graph(
            data, 10, 1000, pbc, rep=[2, 2, 2]
        )
        # edge_index, edge_dist, edge_vec = radius_graph(data, cutoff, max_num_neighbors, pbc)
        vn_mask = data.get("vn_mask")
        edge_index, edge_dist, edge_vec = filter_edge(
            edge_index,
            edge_dist,
            edge_vec,
            msg_direction,
            vn_mask,
            cutoff,
            max_num_neighbors,
        )
        data.edge_index = edge_index
        data.edge_dist = edge_dist
        data.edge_vec = edge_vec
        return data
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
