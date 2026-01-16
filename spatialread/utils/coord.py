import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import cKDTree
from pymatgen.core.lattice import Lattice
from typing import Callable, Union, Tuple, Dict, Any, Literal, List, Optional


def farthest_point_sampling(xyz: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Perform farthest point sampling on a set of points.
    
    Args:
        xyz: Array of shape (N, 3) containing point coordinates
        num_samples: Number of points to sample
        
    Returns:
        Array of shape (num_samples, 3) containing sampled points
    """
    num_points = xyz.shape[0]
    if num_samples > num_points:
        raise ValueError(f"Requested {num_samples} samples but only {num_points} points available")
    
    selected_indices = np.zeros(num_samples, dtype=int)
    distances = np.full(num_points, np.inf)
    
    # Start with a random point
    selected_indices[0] = np.random.randint(num_points)
    
    for i in range(1, num_samples):
        # Calculate distances from the last selected point
        dist = np.linalg.norm(xyz - xyz[selected_indices[i - 1]], axis=1)
        # Update distances to be the minimum distance to any selected point
        distances = np.minimum(distances, dist)
        # Select the point with maximum minimum distance
        selected_indices[i] = np.argmax(distances)
    
    return xyz[selected_indices]


def sample_spherical_points(
    centroid: torch.Tensor,
    radius: torch.Tensor,
    num_points: int,
    random_rotation: bool = True
) -> torch.Tensor:
    """
    Sample points uniformly on a sphere surface.
    
    Args:
        centroid: Center of the sphere (Tensor[3])
        radius: Radius of the sphere (scalar Tensor)
        num_points: Number of points to sample
        random_rotation: Whether to apply random rotation
        
    Returns:
        Tensor of shape (num_points, 3) containing sampled points
    """
    device = centroid.device
    dtype = centroid.dtype

    radius = radius.to(device=device, dtype=dtype)

    golden_ratio = (1.0 + torch.sqrt(torch.tensor(5.0, device=device, dtype=dtype))) / 2.0
    
    idx = torch.arange(num_points, device=device, dtype=dtype)
    theta = 2 * torch.pi * idx / golden_ratio
    phi = torch.acos(1 - 2 * (idx + 0.5) / num_points)
    
    x = radius * torch.sin(phi) * torch.cos(theta)
    y = radius * torch.sin(phi) * torch.sin(theta)
    z = radius * torch.cos(phi)
    
    points = torch.stack((x, y, z), dim=1)
    
    if random_rotation:
        rotation_matrix = _generate_random_rotation_matrix().to(device=device, dtype=dtype)
        points = torch.matmul(points, rotation_matrix.T)
    
    return centroid + points


def _generate_random_rotation_matrix() -> torch.Tensor:
    """Generate a random 3x3 rotation matrix (in SO(3))."""
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    # Ensure det(q) == 1 (pure rotation, no reflection)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def detect_surface_atoms(
    atom_positions: np.ndarray, 
    radius: float = 10.0, 
    anisotropy_threshold: float = 0.5
) -> np.ndarray:
    """
    Detect surface atoms based on local anisotropy.
    
    Args:
        atom_positions: Array of shape (N, 3) containing atom positions
        radius: Search radius for neighborhood
        anisotropy_threshold: Threshold for anisotropy ratio
        
    Returns:
        Array of surface atom indices
    """
    kdtree = cKDTree(atom_positions)
    surface_indices: List[int] = []
    
    for i, center in enumerate(atom_positions):
        neighbor_indices = kdtree.query_ball_point(center, r=radius)
        neighbors = atom_positions[neighbor_indices]
        
        if len(neighbors) < 5:
            # Too few neighbors, likely surface atom
            surface_indices.append(i)
            continue
        
        # Center the neighborhood and compute covariance matrix
        neighbors_centered = neighbors - neighbors.mean(axis=0)
        covariance = neighbors_centered.T @ neighbors_centered / (len(neighbors) - 1)
        eigenvalues = np.linalg.eigvalsh(covariance)  # Sorted ascending
        
        # Check anisotropy (ratio of largest eigenvalue to sum)
        anisotropy_ratio = eigenvalues[-1] / (np.sum(eigenvalues) + 1e-8)
        if anisotropy_ratio > anisotropy_threshold:
            surface_indices.append(i)
    
    return np.array(surface_indices, dtype=np.int32)


def sample_virtual_nodes(
    # data: HeteroData,
    pnode_pos: torch.Tensor,
    vnode_num: Union[int, float],
    mode: Literal['uniform', 'density', 'inverse_density', 'grid', 'surface', 'center'] = 'uniform',
    pbc: bool = False,
    cell: Optional[np.ndarray] = None,
    repulsion_distance: Optional[float] = 0.5,
    kde_bandwidth: Optional[float] = 1.0,
) -> torch.Tensor:
    """
    Sample positions for virtual nodes.
    
    Args:
        pnode_pos: positions of physical nodes (Tensor[N, 3])
        vnode_num: Number of virtual nodes (int) or resolution (float, in Å) for grid-like modes
        mode: Sampling mode for virtual nodes
        pbc: Whether to use periodic boundary conditions
        cell: 3x3 cell matrix if pbc=True or vnode_num is float
        repulsion_distance: Minimum distance from real atoms; if None, no repulsion filtering
        kde_bandwidth: Bandwidth for kernel density estimation (for density-based sampling)
        
    Returns:
        Tensor of shape [Nv, 3] with virtual node positions (same device/dtype as pnode_pos)
    """
    if isinstance(vnode_num, float):
        if cell is None:
            raise ValueError("cell must be provided when vnode_num is float (used as a resolution).")
        cell_len = np.linalg.norm(cell, axis=1, ord=2)
        resolution = vnode_num
        vnode_nums: List[int] = []
        for i in range(3):
            # 保证至少 1 个格点，避免 0 导致下游异常
            vnode_nums.append(max(1, min(10, int(cell_len[i] // resolution))))
    elif isinstance(vnode_num, int):
        if vnode_num <= 0:
            raise ValueError(f"vnode_num must be positive, got {vnode_num}")
        resolution = 15 / vnode_num
        vnode_nums = [vnode_num, vnode_num, vnode_num]
    else:
        raise ValueError(f'Unknown vnode_num type: {type(vnode_num)}')
    
    cell_volume = abs(np.linalg.det(cell))
    N_s = np.prod(vnode_nums)
    voxel_volume = float(cell_volume / N_s)
    voxel_volume /= resolution**3

    vnode_num = int(vnode_nums[0] * vnode_nums[1] * vnode_nums[2])
    # print("DEBUG SpNode Num", vnode_num)
    if vnode_num <= 0:
        raise ValueError("Computed total vnode_num is 0; check cell size / resolution.")
    M = int(vnode_num * 2)  # candidate pool size
    
    real_node_positions = pnode_pos
    atom_positions = real_node_positions.detach().cpu().numpy()
    
    candidate_points: np.ndarray

    if mode == 'grid':
        candidate_points = _sample_grid_points(atom_positions, vnode_nums, pbc, cell)
        candidate_points_t = torch.from_numpy(candidate_points).to(real_node_positions)
        return candidate_points_t, {"grid": vnode_nums, "voxel_volume": voxel_volume}

    if mode == 'center':
        lat = Lattice(cell)
        point = lat.get_cartesian_coords([0.5, 0.5, 0.5])
        return torch.from_numpy(point).reshape(1, 3).to(real_node_positions), 1
    
    if mode in ['density', 'inverse_density']:
        candidate_points = _sample_by_density(
            atom_positions, M, mode, pbc, cell, kde_bandwidth
        )
    elif mode == 'surface':
        candidate_points = _sample_surface_points(atom_positions, M)
    elif mode == 'uniform':
        candidate_points = _sample_uniform_points(atom_positions, M, pbc, cell)
    else:
        raise ValueError(f"Unsupported sampling mode: {mode}")
    
    # Apply repulsion filtering
    filtered_candidates = _apply_repulsion_filter(
        candidate_points, real_node_positions, repulsion_distance
    )

    # Select final virtual node positions
    virtual_node_positions_np = _select_final_positions(filtered_candidates, vnode_num)
    virtual_node_positions = torch.from_numpy(virtual_node_positions_np).to(real_node_positions)
    
    return virtual_node_positions, vnode_num


def _sample_by_density(
    atom_positions: np.ndarray,
    num_points: int,
    mode: str,
    pbc: bool,
    cell: Optional[np.ndarray],
    bandwidth: Optional[float] = None,  # Make bandwidth optional
    oversample_factor: int = 10,
) -> np.ndarray:
    """Sample points based on density estimation."""
    if len(atom_positions) == 0:
        raise ValueError("atom_positions is empty in _sample_by_density.")
    
    # Use smaller bandwidth for better sparse region detection
    if bandwidth is None:
        # Calculate optimal bandwidth based on data characteristics
        n_samples = len(atom_positions)
        if n_samples > 1:
            # Silverman's rule of thumb with smaller factor for sensitivity
            bandwidth = np.std(atom_positions, axis=0).mean() * (4 / (3 * n_samples)) ** (1/5) * 0.5
        else:
            bandwidth = 0.1
    
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(atom_positions)
    
    if pbc:
        assert cell is not None, "Cell must be provided for PBC sampling"
        lattice = Lattice(cell)
        fractional_coords = np.random.rand(oversample_factor * num_points, 3)
        samples = lattice.get_cartesian_coords(fractional_coords).astype(np.float32)
    else:
        centroid = np.mean(atom_positions, axis=0)
        distances = np.linalg.norm(atom_positions - centroid, axis=1)
        max_distance = float(np.max(distances))
        samples = _sample_points_in_sphere(centroid, max_distance, oversample_factor * num_points)
    
    # Calculate densities and probabilities
    log_densities = kde.score_samples(samples)
    densities = np.exp(log_densities)
    
    if mode == 'inverse_density':
        # Higher probability for lower density areas (sparse regions)
        inverse_densities = 1.0 / (densities + 1e-5)
        probabilities = inverse_densities
        probabilities = MinMaxScaler().fit_transform(probabilities.reshape(-1, 1)).reshape(-1)
    else:
        # Higher probability for higher density areas
        probabilities = MinMaxScaler().fit_transform(densities.reshape(-1, 1)).reshape(-1)

    probabilities += 0.1  # Add small constant to avoid zero probabilities
    probabilities /= probabilities.sum()
    
    # Sample according to probabilities
    indices = np.random.choice(len(samples), size=num_points, replace=True, p=probabilities)
    return samples[indices]


def _sample_surface_points(atom_positions: np.ndarray, num_points: int) -> np.ndarray:
    """Sample points around surface atoms."""
    if len(atom_positions) == 0:
        raise ValueError("atom_positions is empty in _sample_surface_points.")
    surface_indices = detect_surface_atoms(atom_positions)
    surface_positions = atom_positions[surface_indices] if len(surface_indices) > 0 else atom_positions
    
    if len(surface_positions) == 0:
        print("⚠️ No surface atoms detected. Falling back to full molecule sampling.")
        surface_positions = atom_positions
    
    # Sample around surface atoms with Gaussian noise
    center_samples = surface_positions[np.random.choice(len(surface_positions), size=num_points, replace=True)]
    noise = np.random.normal(loc=0.0, scale=0.5, size=center_samples.shape)
    return (center_samples + noise).astype(np.float32)


def _sample_uniform_points(
    atom_positions: np.ndarray, 
    num_points: int,
    pbc: bool, 
    cell: Optional[np.ndarray], 
) -> np.ndarray:
    """Sample points uniformly."""
    if len(atom_positions) == 0:
        raise ValueError("atom_positions is empty in _sample_uniform_points.")

    if pbc:
        assert cell is not None, "Cell must be provided for PBC sampling"
        lattice = Lattice(cell)
        fractional_coords = np.random.rand(num_points, 3)
        cart_coords = lattice.get_cartesian_coords(fractional_coords).astype(np.float32)
        return cart_coords
    else:
        centroid = np.mean(atom_positions, axis=0)
        distances = np.linalg.norm(atom_positions - centroid, axis=1)
        max_distance = float(np.max(distances))
        samples = sample_spherical_points(
            centroid=torch.tensor(centroid, dtype=torch.float32),
            radius=torch.tensor(max_distance, dtype=torch.float32),
            num_points=num_points,
            random_rotation=True
        )
        return samples.cpu().numpy().astype(np.float32)


def _sample_grid_points(
    atom_positions: np.ndarray, 
    vnode_nums: List[int], 
    pbc: bool, 
    cell: Optional[np.ndarray], 
) -> np.ndarray:
    """
    Sample points on a grid within the unit cell or bounding box.
    """
    if pbc:
        assert cell is not None, "Cell must be provided for PBC sampling"
        lattice = Lattice(cell)

        grid_positions: List[np.ndarray] = []
        nx, ny, nz = vnode_nums
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    fractional_coords = np.array([
                        (i+0.5) / nx,
                        (j+0.5) / ny,
                        (k+0.5) / nz,
                    ], dtype=float)
                    cartesian_coords = lattice.get_cartesian_coords(fractional_coords)
                    grid_positions.append(cartesian_coords.astype(np.float32))
        
        grid_points = np.stack(grid_positions, axis=0)
    else:
        # For non-periodic systems, create a grid in the bounding box
        if len(atom_positions) == 0:
            raise ValueError("atom_positions is empty in _sample_grid_points (non-PBC).")
        min_coords = atom_positions.min(axis=0)
        max_coords = atom_positions.max(axis=0)
        # Simple cubic grid
        grid_points = _create_bounding_box_grid(min_coords, max_coords, vnode_nums)
    
    return grid_points.astype(np.float32)


def _create_bounding_box_grid(
    min_coords: np.ndarray, 
    max_coords: np.ndarray, 
    vnode_nums: List[int]
) -> np.ndarray:
    """Create a grid within a bounding box."""
    nx, ny, nz = vnode_nums
    x = np.linspace(min_coords[0], max_coords[0], nx)
    y = np.linspace(min_coords[1], max_coords[1], ny)
    z = np.linspace(min_coords[2], max_coords[2], nz)
    
    grid_points = np.array(np.meshgrid(x, y, z, indexing='ij')).T.reshape(-1, 3)
    return grid_points.astype(np.float32)


def _sample_points_in_sphere(center: np.ndarray, radius: float, num_points: int) -> np.ndarray:
    """Sample points uniformly within a sphere."""
    if num_points <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.arccos(np.random.uniform(-1, 1, num_points))
    r = radius * np.cbrt(np.random.uniform(0, 1, num_points))
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    return (np.stack((x, y, z), axis=1) + center).astype(np.float32)


def _apply_repulsion_filter(
    candidate_points: np.ndarray, 
    real_positions: torch.Tensor, 
    repulsion_distance: Optional[float]
) -> np.ndarray:
    """Filter candidate points that are too close to real atoms."""
    if repulsion_distance is None:
        return candidate_points
    if candidate_points.shape[0] == 0:
        return candidate_points

    candidate_tensor = torch.tensor(
        candidate_points, dtype=real_positions.dtype, device=real_positions.device
    )
    distance_matrix = torch.cdist(candidate_tensor, real_positions)  # [N_candidates, N_real]
    if distance_matrix.shape[0] == 0:
        return candidate_points
    min_distances = distance_matrix.min(dim=1).values
    mask = min_distances >= repulsion_distance
    return candidate_tensor[mask].cpu().numpy()


def _select_final_positions(
    candidate_points: np.ndarray, 
    num_selected: int, 
    use_fps: bool = True
) -> np.ndarray:
    """Select final virtual node positions from candidates."""
    n_candidates = candidate_points.shape[0]
    if n_candidates == 0:
        # 没有候选点，直接返回空
        return candidate_points

    if n_candidates < num_selected:
        # 候选点不足时直接用全部
        selected_points = candidate_points
    else:
        if use_fps:
            selected_points = farthest_point_sampling(candidate_points, num_selected)
        else:
            # just take random subset (preserves probability bias)
            indices = np.random.choice(n_candidates, size=num_selected, replace=False)
            selected_points = candidate_points[indices]
    
    return selected_points


def reshape_coord_and_calculate(
    func: Callable[..., Union[np.ndarray, Tuple, Dict]], 
    coord: np.ndarray,
    *args: Any,
    **kwargs: Any
) -> Union[np.ndarray, Tuple, Dict]:
    """
    Flatten coord to (N, 3), call func, then reshape outputs back to the original
    coord shape (excluding last dimension).
    """
    assert coord.shape[-1] == 3, f'reshape coord to (-1, 3) failed: coord shape {coord.shape}'
    point_shape = coord.shape[:-1]  # Original shape (excluding last dim)
    coord_flat = coord.reshape((-1, 3))  # Flatten to (N,3)
    
    ret = func(coord_flat, *args, **kwargs)
    
    def _reshape_array(arr: np.ndarray) -> np.ndarray:
        """Reshape array to match original point cloud structure"""
        if arr.ndim == 1:
            return arr.reshape(point_shape)
        return arr.reshape((*point_shape, arr.shape[-1]))
    
    if isinstance(ret, dict):
        # Handle dictionary case
        return {k: _reshape_array(v) for k, v in ret.items()}
    elif isinstance(ret, tuple):
        # Handle tuple case
        return tuple(_reshape_array(r) for r in ret)
    else:
        # Handle single array case
        return _reshape_array(ret)
