"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
from torch_geometric.data.batch import Batch
from torch_geometric.data import Data, HeteroData
from torch_scatter import segment_coo, segment_csr

from typing import Union, Literal, List, Optional

from ase.neighborlist import neighbor_list
from ase import Atoms


def radius_graph_pbc_cpu(data: Data, radius: float, max_num_neighbors: int):
    odevice = data.pos.device
    data = data.to('cpu')
    device = data.pos.device

    z = data.atomic_numbers.cpu().numpy()
    positions = data.pos.cpu().numpy()
    cell = data.cell.cpu().numpy().reshape(3, 3)      # 单个构型时应该是 (3, 3)

    atoms = Atoms(numbers=z, positions=positions, cell=cell, pbc=True)

    # i: 中心原子，j: 邻居原子
    i, j, d, D, S = neighbor_list("ijdDS", atoms, cutoff=radius, self_interaction=False)

    center_index  = torch.tensor(i, dtype=torch.long, device=device)  # 对齐原来的 index1 含义
    neigh_index   = torch.tensor(j, dtype=torch.long, device=device)  # 对齐原来的 index2 含义
    shift_cart    = torch.tensor(D, dtype=torch.float32, device=device)
    edge_distance = torch.tensor(d, dtype=torch.float32, device=device)
    shift_cell    = torch.tensor(S, dtype=torch.float32, device=device)  # (E, 3)，整数晶格偏移

    # 距离平方（和原函数里的 atom_distance_sqr 对齐）
    atom_distance_sqr = (shift_cart ** 2).sum(dim=-1)

    # 对于单构型，把 natoms 写成形如 tensor([N])
    if isinstance(data.natoms, int):
        natoms = torch.tensor([data.natoms], device=device)
    else:
        natoms = data.natoms.to(device)

    # 直接复用原来的逻辑，得到 mask_num_neighbors 和 num_neighbors_image
    mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
        natoms=natoms,
        index=center_index,                  # 就像以前传 index1 一样
        atom_distance=atom_distance_sqr,
        max_num_neighbors_threshold=max_num_neighbors,
    )

    # 应用 mask 把多余邻居裁掉
    if not torch.all(mask_num_neighbors):
        center_index = center_index[mask_num_neighbors]
        neigh_index  = neigh_index[mask_num_neighbors]
        shift_cart   = shift_cart[mask_num_neighbors]
        shift_cell   = shift_cell[mask_num_neighbors]

    # 为了和原来的方向保持一致：edge_index[0] = 邻居，edge_index[1] = 中心
    edge_index = torch.stack([neigh_index, center_index], dim=0)

    # 也可以顺便返回距离（根号后的）
    dist = torch.sqrt((shift_cart ** 2).sum(dim=-1))
    # print("DEBUG SHIFT CART", dist.min(), dist.max(), dist.mean(), dist.std())

    data = data.to(odevice)
    return edge_index.to(odevice), shift_cell.to(odevice), num_neighbors_image.to(odevice)


def radius_graph_pbc(
    data: Union[Data, HeteroData],
    radius: float,
    max_num_neighbors_threshold: int,
    pbc: list[bool] = [True, True, True],
    rep: List[Optional[int]] = [None, None, None]
):
    # odevice = data.cell.device
    try:
        # data = data.to('cuda:3')
        device = data.cell.device

        # print("DEBUG device", device)

        if isinstance(data.natoms, int):
            data.natoms = torch.tensor([data.natoms], device=device)

        batch_size = len(data.natoms)

        if hasattr(data, "pbc"):
            data.pbc = torch.atleast_2d(data.pbc)
            for i in range(3):
                if not torch.any(data.pbc[:, i]).item():
                    pbc[i] = False
                elif torch.all(data.pbc[:, i]).item():
                    pbc[i] = True
                else:
                    raise RuntimeError(
                        "Different structures in the batch have different PBC configurations. This is not currently supported."
                    )

        # position of the atoms
        atom_pos = data.pos

        # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
        num_atoms_per_image = data.natoms
        num_atoms_per_image_sqr = (num_atoms_per_image**2).long()

        # index offset between images
        index_offset = torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image

        index_offset_expand = torch.repeat_interleave(index_offset, num_atoms_per_image_sqr)
        num_atoms_per_image_expand = torch.repeat_interleave(
            num_atoms_per_image, num_atoms_per_image_sqr
        )

        # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
        # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
        # the following (but 10x faster since it removes the for loop)
        # for batch_idx in range(batch_size):
        #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
        num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
        index_sqr_offset = (
            torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
        )
        index_sqr_offset = torch.repeat_interleave(
            index_sqr_offset, num_atoms_per_image_sqr
        )
        atom_count_sqr = torch.arange(num_atom_pairs, device=device) - index_sqr_offset

        # Compute the indices for the pairs of atoms (using division and mod)
        # If the systems get too large this apporach could run into numerical precision issues
        index1 = (
            torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor")
        ) + index_offset_expand
        index2 = (atom_count_sqr % num_atoms_per_image_expand) + index_offset_expand
        # Get the positions for each atom
        pos1 = torch.index_select(atom_pos, 0, index1)
        pos2 = torch.index_select(atom_pos, 0, index2)

        # Calculate required number of unit cells in each direction.
        # Smallest distance between planes separated by a1 is
        # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
        # Note that the unit cell volume V = a1 * (a2 x a3) and that
        # (a2 x a3) / V is also the reciprocal primitive vector
        # (crystallographer's definition).

        cross_a2a3 = torch.cross(data.cell[:, 1], data.cell[:, 2], dim=-1)
        cell_vol = torch.sum(data.cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

        if pbc[0]:
            if rep[0] is not None:
                rep_a1 = torch.tensor(rep[0], device=device)
            else:
                inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
                rep_a1 = torch.ceil(radius * inv_min_dist_a1)
        else:
            rep_a1 = data.cell.new_zeros(1)

        if pbc[1]:
            if rep[1] is not None:
                rep_a2 = torch.tensor(rep[1], device=device)
            else:
                cross_a3a1 = torch.cross(data.cell[:, 2], data.cell[:, 0], dim=-1)
                inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
                rep_a2 = torch.ceil(radius * inv_min_dist_a2)
        else:
            rep_a2 = data.cell.new_zeros(1)

        if pbc[2]:
            if rep[2] is not None:
                rep_a3 = torch.tensor(rep[2], device=device)
            else:
                cross_a1a2 = torch.cross(data.cell[:, 0], data.cell[:, 1], dim=-1)
                inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
                rep_a3 = torch.ceil(radius * inv_min_dist_a3)
        else:
            rep_a3 = data.cell.new_zeros(1)

        # Take the max over all images for uniformity. This is essentially padding.
        # Note that this can significantly increase the number of computed distances
        # if the required repetitions are very different between images
        # (which they usually are). Changing this to sparse (scatter) operations
        # might be worth the effort if this function becomes a bottleneck.
        max_rep = [rep_a1.max(), rep_a2.max(), rep_a3.max()]

        # Tensor of unit cells
        cells_per_dim = [
            torch.arange(-rep, rep + 1, device=device, dtype=torch.float) for rep in max_rep
        ]
        unit_cell = torch.cartesian_prod(*cells_per_dim)
        num_cells = len(unit_cell)
        unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(len(index2), 1, 1)
        unit_cell = torch.transpose(unit_cell, 0, 1)
        unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(batch_size, -1, -1)

        # Compute the x, y, z positional offsets for each cell in each image
        data_cell = torch.transpose(data.cell, 1, 2)
        pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
        pbc_offsets_per_atom = torch.repeat_interleave(
            pbc_offsets, num_atoms_per_image_sqr, dim=0
        )

        # Expand the positions and indices for the 9 cells
        pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
        pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
        index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
        index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
        # Add the PBC offsets for the second atom
        pos2 = pos2 + pbc_offsets_per_atom

        # Compute the squared distance between atoms
        atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
        atom_distance_sqr = atom_distance_sqr.view(-1)

        # Remove pairs that are too far apart
        mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
        # Remove pairs with the same atoms (distance = 0.0)
        mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
        mask = torch.logical_and(mask_within_radius, mask_not_same)
        index1 = torch.masked_select(index1, mask)
        index2 = torch.masked_select(index2, mask)
        unit_cell = torch.masked_select(
            unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
        )
        unit_cell = unit_cell.view(-1, 3)
        atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

        mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
            natoms=data.natoms,
            index=index1,
            atom_distance=atom_distance_sqr,
            max_num_neighbors_threshold=max_num_neighbors_threshold,
        )

        if not torch.all(mask_num_neighbors):
            # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
            index1 = torch.masked_select(index1, mask_num_neighbors)
            index2 = torch.masked_select(index2, mask_num_neighbors)
            unit_cell = torch.masked_select(
                unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
            )
            unit_cell = unit_cell.view(-1, 3)

        edge_index = torch.stack((index2, index1))

        # data = data.to(odevice)
        # return edge_index.to(odevice), unit_cell.to(odevice), num_neighbors_image.to(odevice)
        return edge_index, unit_cell, num_neighbors_image
    except torch.cuda.OutOfMemoryError:
        # data = data.to(odevice)
        return radius_graph_pbc_cpu(data, radius, max_num_neighbors_threshold)


def get_max_neighbors_mask(
    natoms: torch.Tensor,
    index: torch.Tensor,
    atom_distance: torch.Tensor,
    max_num_neighbors_threshold: int,
):
    """
    Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors.
    Assumes that `index` is sorted.
    """
    device = natoms.device
    num_atoms = natoms.sum()

    # Get number of neighbors
    # segment_coo assumes sorted index
    ones = index.new_ones(1).expand_as(index)
    num_neighbors = segment_coo(ones, index, dim_size=num_atoms)
    max_num_neighbors = num_neighbors.max()
    num_neighbors_thresholded = num_neighbors.clamp(max=max_num_neighbors_threshold)

    # Get number of (thresholded) neighbors per image
    image_indptr = torch.zeros(natoms.shape[0] + 1, device=device, dtype=torch.long)
    image_indptr[1:] = torch.cumsum(natoms, dim=0)
    num_neighbors_image = segment_csr(num_neighbors_thresholded, image_indptr)

    # If max_num_neighbors is below the threshold, return early
    if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
    ):
        mask_num_neighbors = torch.tensor([True], dtype=bool, device=device).expand_as(
            index
        )
        return mask_num_neighbors, num_neighbors_image

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with infinity so we can easily remove unused distances later.
    distance_sort = torch.full([num_atoms * max_num_neighbors], np.inf, device=device)

    # Create an index map to map distances from atom_distance to distance_sort
    # index_sort_map assumes index to be sorted
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
        index * max_num_neighbors
        + torch.arange(len(index), device=device)
        - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance)
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)
    # Select the max_num_neighbors_threshold neighbors that are closest
    distance_sort = distance_sort[:, :max_num_neighbors_threshold]
    index_sort = index_sort[:, :max_num_neighbors_threshold]

    # Offset index_sort so that it indexes into index
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_neighbors_threshold
    )
    # Remove "unused pairs" with infinite distances
    mask_finite = torch.isfinite(distance_sort)
    index_sort = torch.masked_select(index_sort, mask_finite)

    # At this point index_sort contains the index into index of the
    # closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = torch.zeros(len(index), device=device, dtype=bool)
    mask_num_neighbors.index_fill_(0, index_sort, True)

    return mask_num_neighbors, num_neighbors_image


import time
import torch
from ase.io import read
from ase.neighborlist import neighbor_list
from torch_geometric.data import Data
# from your_module import radius_graph_pbc  # 换成你 CUDA 版所在路径


def radius_graph_pbc_ase(atoms, radius, device="cpu"):
    """
    使用 ASE 生成周期性边，返回 PyTorch 张量
    """
    i, j, D = neighbor_list("ijD", atoms, cutoff=radius, self_interaction=False)

    # 确保 cell 是 (3, 3)
    cell = atoms.cell.array  # (3, 3)

    print("cell shape:", atoms.cell.array.shape)
    print("D shape:", D.shape)


    # 计算实际位移
    disp = D @ cell + atoms.positions[j] - atoms.positions[i]

    src = torch.tensor(i, dtype=torch.long, device=device)
    dst = torch.tensor(j, dtype=torch.long, device=device)
    shift = torch.tensor(disp, dtype=torch.float32, device=device)

    return src, dst, shift


if __name__ == "__main__":
    atoms = read("./FITGEW_clean.cif")
    radius = 10.0
    max_num_neighbors_threshold = 20000  # 可以按你代码里的默认值来设

    # === ASE CPU 版本 ===
    t0 = time.time()
    src1, dst1, shift1 = radius_graph_pbc_ase(atoms, radius, device="cpu")
    t1 = time.time()
    print(f"[ASE version] time: {t1 - t0:.4f} s, edges={len(src1)}")

    # === 你的 CUDA 版本 ===
    # 准备 Data 对象（只包含 positions + cell）
    pos = torch.tensor(atoms.positions, dtype=torch.float32)
    cell = torch.tensor(atoms.cell.array, dtype=torch.float32).unsqueeze(0)  # [1,3,3]
    data = Data(pos=pos, cell=cell)
    data.natoms = len(pos)
    print(len(pos))

    t0 = time.time()
    edge_index, unit_cell, num_neighbors_image = radius_graph_pbc(
        data,
        radius=radius,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
        pbc=[True, True, True],
    )
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[CUDA version] time: {t1 - t0:.4f} s, edges={edge_index.shape[1]}")

    # === 简单对比 ===
    print(f"ASE edges: {len(src1)} | CUDA edges: {edge_index.shape[1]}")
