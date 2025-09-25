import torch
from torch_geometric.data import HeteroData, Data, Batch
from typing import Union, Literal, List
from torch_geometric.nn.pool import radius_graph as pyg_radius_graph
from torch_cluster import radius

import warnings


from .radius_graph import radius_graph_pbc
from .ocp import get_pbc_distances

def hetero_to_homo(hetero_data: Union[HeteroData, List[HeteroData], Batch]) -> Data:
    if isinstance(hetero_data, Batch):
        # TODO: to_data_list will fail when extra properties are included such as is_binding
        return Batch.from_data_list(hetero_to_homo(hetero_data.to_data_list()))

    if isinstance(hetero_data, list):
        homo_data_list = [hetero_to_homo(data) for data in hetero_data]
        return homo_data_list

    homo_data = Data()

    pn_data = hetero_data['pnode']
    vn_data = hetero_data['vnode']

    pn_z = pn_data['z']
    pn_pos = pn_data['pos']
    vn_z = vn_data['z']
    vn_pos = vn_data['pos']
    homo_data['z'] = torch.cat([pn_z, vn_z], dim=0)
    homo_data['pos'] = torch.cat([pn_pos, vn_pos], dim=0)
    homo_data['vn_mask'] = torch.cat([torch.zeros(pn_z.shape[0]), torch.ones(vn_z.shape[0])]).bool().to(pn_z.device)
    # other keys in hetero_data
    for key in hetero_data.keys():
        if 'edge_index' in key:
            # warnings.warn(f"Not support edge_index transformation of heterodata now, the {key} will be deleted in homo data")
            continue
        if key not in ['pnode', 'vnode', 'z', 'pos', 'vn_mask']:
            homo_data[key] = hetero_data[key]

    return homo_data

def homo_to_hetero(homo_data: Union[Data, List[Data], Batch]) -> Union[HeteroData, List[HeteroData]]:
    if isinstance(homo_data, Batch):
        return Batch.from_data_list(homo_to_hetero(homo_data.to_data_list()))
    if isinstance(homo_data, list):
        hetero_data_list = [homo_to_hetero(data) for data in homo_data]
        return hetero_data_list

    hetero_data = HeteroData()

    homo_z = homo_data['z']
    homo_pos = homo_data['pos']
    homo_vn_mask = homo_data['vn_mask']
    homo_pn_z = homo_z[homo_vn_mask == 0]
    homo_pn_pos = homo_pos[homo_vn_mask == 0]
    homo_vn_z = homo_z[homo_vn_mask == 1]
    homo_vn_pos = homo_pos[homo_vn_mask == 1]
    hetero_data['pnode'].z = homo_pn_z
    hetero_data['pnode'].pos = homo_pn_pos
    hetero_data['vnode'].z = homo_vn_z
    hetero_data['vnode'].pos = homo_vn_pos

    # other keys in homo_data
    for key in homo_data.keys():
        if 'edge_index' in key:
            warnings.warn(f"Not support edge_index transformation of heterodata now, the {key} will be deleted in homo data")
            continue
        if key not in ['z', 'pos', 'vn_mask']:
            hetero_data[key] = homo_data[key]
    return hetero_data

# return edge_index, edge_weight, edge_vec
def radius_graph(data: Data, cutoff: float = 6.0, max_num_neighbors: int = 30, pbc: bool = False, rep=[None, None, None]):
    if isinstance(data, HeteroData):
        raise ValueError("Undefined behavior for HeteroData, please use hetero_to_homo first.")
    
    if isinstance(data, Batch):
        raise ValueError("Undefined behavior for Batch.")

    if pbc:
        edge_index, cell_offsets, neighbors = radius_graph_pbc(data, cutoff, max_num_neighbors, rep=rep)
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
        # edge_weight: torch.Tensor = out["distances"]
        edge_distance: torch.Tensor = out["distances"]
        cell_offset_distances: torch.Tensor = out["offsets"]
        edge_vec: torch.Tensor = out["distance_vec"]

        return edge_index, edge_distance, edge_vec
    else:
        pos = data.pos

        edge_index = pyg_radius_graph(pos, cutoff, max_num_neighbors=max_num_neighbors)
        edge_weight = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        return edge_index, edge_weight, edge_vec

def edge_indices(edge_index: torch.Tensor, vn_mask: torch.Tensor, msg_direction: Literal['p2p', 'p2v', 'v2v', 'v2p', 'p2pv']):
    if msg_direction == 'p2p':
        indices = torch.logical_and(vn_mask[edge_index[0]] == 0, vn_mask[edge_index[1]] == 0)
        return indices
    elif msg_direction == 'p2v':
        indices = torch.logical_and(vn_mask[edge_index[0]] == 0, vn_mask[edge_index[1]] == 1)
        return indices
    elif msg_direction == 'v2v':
        indices = torch.logical_and(vn_mask[edge_index[0]] == 1, vn_mask[edge_index[1]] == 1)
        return indices
    elif msg_direction == 'v2p':
        indices = torch.logical_and(vn_mask[edge_index[0]] == 1, vn_mask[edge_index[1]] == 0)
        return indices
    elif msg_direction == 'p2pv':
        indices = vn_mask[edge_index[0]] == 0
        return indices
    else:
        raise ValueError(f"Unknown msg_direction: {msg_direction}")

def limit_in_edges(edge_index, edge_weight, max_in_degree):
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

def filter_edge(edge_index: torch.Tensor, edge_weight: torch.Tensor, edge_vec: torch.Tensor, msg_direction: Literal['p2p', 'p2v', 'v2v', 'v2p', 'p2pv', None], vn_mask: torch.Tensor, cutoff: float, max_num_neighbors: int, other_content: List[torch.Tensor]=[]):
    if msg_direction is not None:
        assert msg_direction in ['p2p', 'p2v', 'v2v', 'v2p', 'p2pv'], f"Unknown msg_direction: {msg_direction}"

        indices = edge_indices(edge_index, vn_mask, msg_direction)
        edge_index = edge_index[:, indices]
        edge_weight = edge_weight[indices]
        edge_vec = edge_vec[indices]

        for content in range(len(other_content)):
            other_content[content] = other_content[content][indices]

    # NOTE: For homo data, it should be noted that some VNs will be connected to no any pn
    # This will result in indexing error in some GNNs using aggregation, due to no message can be aggregated for this vn
    if cutoff is not None:
        indices = edge_weight < cutoff
        edge_index = edge_index[:, indices]
        edge_weight = edge_weight[indices]
        edge_vec = edge_vec[indices]

        for content in range(len(other_content)):
            other_content[content] = other_content[content][indices]

    # indices = ~is_vnode[edge_index[0]]
    # edge_index = edge_index[:, indices]
    # edge_weight = edge_weight[indices]
    # edge_vec = edge_vec[indices]

    if max_num_neighbors is not None:
        indices = limit_in_edges(edge_index, edge_weight, max_in_degree=max_num_neighbors)
        edge_index = edge_index[:, indices]
        edge_weight = edge_weight[indices]
        edge_vec = edge_vec[indices]

        for content in range(len(other_content)):
            other_content[content] = other_content[content][indices]

    if len(other_content) > 0:
        return edge_index, edge_weight, edge_vec, other_content

    return edge_index, edge_weight, edge_vec


def calc_edge_index(data: Union[HeteroData, Data, List[Union[HeteroData, Data]], Batch], cutoff, max_num_neighbors, pbc, msg_direction=None, fully_connected: bool = False):
    if fully_connected:
        assert cutoff is None, "cutoff should be None for fully connected graph"
        assert max_num_neighbors is None, "max_num_neighbors should be None for fully connected graph"

    if isinstance(data, Batch):
        raise ValueError("Unsupported Batch for calc_edge_index, convert to list first.")
        data_list = data.to_data_list()
        return Batch.from_data_list([calc_edge_index(d, cutoff, max_num_neighbors, pbc, msg_direction, fully_connected) for d in data_list])

    if isinstance(data, list):
        data = [calc_edge_index(d, cutoff, max_num_neighbors, pbc, msg_direction, fully_connected) for d in data]
        return data

    if isinstance(data, HeteroData):
        homo_data = hetero_to_homo(data)
        # prevent too large graph for pbc system and fully connected graph
        edge_index, edge_weight, edge_vec = radius_graph(homo_data, 19, 1000, pbc, rep=[2, 2, 2])
        vn_mask = homo_data.vn_mask
        if fully_connected:
            edge_index, edge_weight, edge_vec = filter_edge(edge_index, edge_weight, edge_vec, msg_direction, vn_mask, None, None)
        else:
            edge_index, edge_weight, edge_vec = filter_edge(edge_index, edge_weight, edge_vec, msg_direction, vn_mask, cutoff, max_num_neighbors)
        # NOTE: assume all pn are before vn, see hetero_to_homo
        num_pn = homo_data.z.shape[0] - vn_mask.sum().long()
        num_vn = vn_mask.sum().long()
        if msg_direction == 'p2p':
            data['pnode', 'to', 'pnode'].edge_index = edge_index
            data['pnode', 'to', 'pnode'].edge_weight = edge_weight
            data['pnode', 'to', 'pnode'].edge_vec = edge_vec
        elif msg_direction == 'p2v':
            p2v_edge_index = edge_index.clone()
            p2v_edge_index[1] -= num_pn
            data['pnode', 'to', 'vnode'].edge_index = p2v_edge_index
            data['pnode', 'to', 'vnode'].edge_weight = edge_weight
            data['pnode', 'to', 'vnode'].edge_vec = edge_vec
        elif msg_direction == 'v2v':
            v2v_edge_index = edge_index.clone()
            v2v_edge_index[0] -= num_pn
            v2v_edge_index[1] -= num_pn
            data['vnode', 'to', 'vnode'].edge_index = v2v_edge_index
            data['vnode', 'to', 'vnode'].edge_weight = edge_weight
            data['vnode', 'to', 'vnode'].edge_vec = edge_vec
        elif msg_direction == 'v2p':
            v2p_edge_index = edge_index.clone()
            v2p_edge_index[0] -= num_pn
            data['vnode', 'to', 'pnode'].edge_index = v2p_edge_index
            data['vnode', 'to', 'pnode'].edge_weight = edge_weight
            data['vnode', 'to', 'pnode'].edge_vec = edge_vec
        else:
            raise ValueError(f"Unsupported msg_direction for hetero graph: {msg_direction}")
        return data
    elif isinstance(data, Data):
        # assert msg_direction is None, "msg_direction should be None for homo graph"
        edge_index, edge_weight, edge_vec = radius_graph(data, 10, 1000, pbc, rep=[2, 2, 2])
        # edge_index, edge_weight, edge_vec = radius_graph(data, cutoff, max_num_neighbors, pbc)
        vn_mask = data.get("vn_mask")
        edge_index, edge_weight, edge_vec = filter_edge(edge_index, edge_weight, edge_vec, msg_direction, vn_mask, cutoff, max_num_neighbors)
        data.edge_index = edge_index
        data.edge_weight = edge_weight
        data.edge_vec = edge_vec
        return data
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

        