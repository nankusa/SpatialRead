import math
from pathlib import Path
from typing import Dict, Any, List, Literal, Union
import multiprocessing as mp
import click

from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm
from torch_geometric.data import Data
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read

from spatialread.utils.log import get_logger
from spatialread.utils.chem import _make_supercell
from spatialread.config import get_data_config, init_config


def farthest_point_sampling(xyz: np.ndarray, M: int) -> np.ndarray:
    N = xyz.shape[0]
    selected = np.zeros(M, dtype=int)
    distances = np.full(N, np.inf)

    selected[0] = np.random.randint(N)
    for i in range(1, M):
        dist = np.linalg.norm(xyz - xyz[selected[i - 1]], axis=1)
        distances = np.minimum(distances, dist)
        selected[i] = np.argmax(distances)
    return xyz[selected]

def get_grid_poses(lat: Lattice, num_spnode: Union[int, float]) -> torch.Tensor:
    """
    如果 num_spnode 是整数：
        与原逻辑一致，三个方向上都是 num_spnode 个点（立方网格）。
    如果 num_spnode 是浮点数：
        视为网格分辨率（单位长度上的步长），
        沿每个晶格矢量的网格点数 = floor(该方向晶格矢量长度 / num_spnode)，向下取整，
        并且至少为 1。
    """
    # -------- 判断类型 --------
    if isinstance(num_spnode, int):
        # 保持原来的立方网格行为
        nx = ny = nz = int(num_spnode)
    else:
        # 视为分辨率（例如 Å/格点）
        res = float(num_spnode)

        # 这里假设 lat 有类似 pymatgen.Lattice 的接口
        # 如果你的 Lattice 不一样，把这一行改成对应的获取三个晶格向量长度的方法即可
        a, b, c = lat.abc   # 或者 lat.lengths / lat.get_lengths() 等

        # 沿每个方向的网格点数 = floor(length / res)，向下取整，并至少为 1
        nx = max(int(a // res), 1)
        ny = max(int(b // res), 1)
        nz = max(int(c // res), 1)

        print("DEBUG resolution based sampling", nx, ny, nz, a, b, c)

    grid_poses = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # 分数坐标：中心采样 (i+0.5)/n
                frac_coords = [
                    (i + 0.5) / nx,
                    (j + 0.5) / ny,
                    (k + 0.5) / nz,
                ]
                grid_pos = lat.get_cartesian_coords(frac_coords)
                grid_poses.append(torch.from_numpy(grid_pos).float())

    grid_poses = torch.stack(grid_poses, dim=0)  # [N, 3]
    return grid_poses

# def get_grid_poses(lat: Lattice, num_spnode: int) -> torch.Tensor:
#     # num_grid = round(num_spnode ** (1/3))
#     # assert num_grid ** 3 == num_spnode
#     num_grid = num_spnode
#     grid_poses = []
#     for i in range(num_grid):
#         for j in range(num_grid):
#             for k in range(num_grid):
#                 grid_pos = lat.get_cartesian_coords(
#                     [i / num_grid + 0.5, j / num_grid + 0.5, k / num_grid + 0.5]
#                 )
#                 grid_poses.append(torch.from_numpy(grid_pos).float())
#     grid_poses = torch.stack(grid_poses)  # [N, 3]
#     return grid_poses

def repulsion_filter_and_sample(pos: npt.NDArray, points: npt.NDArray, repulsion_distance: float, num_points: int):
    # === Repulsion filter
    points = torch.tensor(points, dtype=torch.float32)
    dist_matrix = torch.cdist(points, torch.tensor(pos))  # [N_vnodes, N_real_nodes]
    min_distances = dist_matrix.min(dim=1).values
    repulsion_distance = repulsion_distance
    mask = min_distances >= repulsion_distance
    points = points[mask]

    filtered_candidates = points.cpu().numpy()
    if len(filtered_candidates) < num_points:
        raise ValueError(f"Too few candidate virtual nodes after repulsion filtering: {len(filtered_candidates)} < {num_points}")

    # === FPS to select M virtual node positions
    selected = farthest_point_sampling(filtered_candidates, num_points)
    spnode_pos = torch.tensor(selected, dtype=torch.float32)

    return spnode_pos

def get_uniform_poses(pos: npt.NDArray, lat: Lattice, num_points: int, repulsoin_distance: float):
    oversample = num_points * 3
    frac_coords = np.random.rand(oversample, 3)
    cart_coords = lat.get_cartesian_coords(frac_coords)  # shape [oversample, 3]
    samples = cart_coords.astype(np.float32)  # candidate points

    return repulsion_filter_and_sample(pos, samples, repulsoin_distance, num_points)

def get_inv_density_poses(pos: npt.NDArray, lat: Lattice, num_points: int, repulsion_distance: float, bandwidth: float = 1.0, kernel: str = 'gaussian'):
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    kde.fit(pos)

    # === Uniformly sample fractional coordinates in unit cell
    oversample = num_points * 3
    frac_coords = np.random.rand(oversample, 3)
    cart_coords = lat.get_cartesian_coords(frac_coords)  # shape [oversample, 3]
    samples = cart_coords.astype(np.float32)  # candidate points

    # === Inverse density sampling
    log_dens = kde.score_samples(samples)
    dens = np.exp(log_dens)
    inv_dens = 1.0 / (dens + 1e-5)
    inv_dens = MinMaxScaler().fit_transform(inv_dens.reshape(-1, 1)).reshape(-1)
    inv_dens += 0.1

    probs = inv_dens / inv_dens.sum()
    indices = np.random.choice(len(samples), size=oversample, replace=False, p=probs)
    weighted_candidates = samples[indices]

    return repulsion_filter_and_sample(pos, weighted_candidates, repulsion_distance, num_points)

def add_spatial_nodes(data: Data, spnode_z: int, num_spnode: int, sample_method: Literal['uniform', 'inverse_density', 'grid'], repulsion_distance: float) -> Data:
    cell = data["cell"].tolist()
    pos = data["pos"]
    X = data["atomic_numbers"]

    lattice_obj = Lattice(cell)
    assert sample_method == 'grid'
    if sample_method == 'uniform':
        spnode_pos = get_uniform_poses(pos, lattice_obj, num_spnode, repulsion_distance)
    elif sample_method == 'inverse_density':
        spnode_pos = get_inv_density_poses(pos, lattice_obj, num_spnode, repulsion_distance)
    elif sample_method == 'grid':
        spnode_pos = get_grid_poses(lattice_obj, num_spnode)

    pos_added = torch.concat([spnode_pos, pos], dim=0)
    X_added = torch.concat(
        [
            torch.ones(
                len(spnode_pos),
                dtype=torch.long,
                device=X.device,
            )
            * spnode_z,
            X,
        ],
        dim=0,
    )

    data["atomic_numbers"] = X_added
    data["pos"] = pos_added.float()
    data["cell"] = torch.tensor(cell, dtype=torch.float, device=X.device)
    # data["vn_min_distance"] = min_distances
    # data['repulsion_mask'] = mask

    return data

def build_graph_i(matid: str, data_config: Dict[str, Any]) -> bool:
    """Build graph data for a single material"""
    logger = get_logger(filename=Path(__file__).stem)
    logger.info(f"Starting graph construction for {matid}")

    override = data_config['override']
    
    # Setup paths
    graph_path = data_config['graph_dir'] / f"{matid}.pt"
    spgraph_path = data_config['spgraph_dir'] / f'{matid}.pt'
    
    if graph_path.exists() and spgraph_path.exists():
        if not override:
            logger.info(f"{matid} graph and spgraph already exist, skipping")
            return True
        else:
            logger.warn(f"{matid} graph and spgraph already exist, but override is True, will be overwritten")

    # 1. Build supercell and check constraints
    cif_path = data_config['cif_dir'] / f"{matid}.cif"
    logger.info(f"Processing CIF file: {cif_path}")
    try:
        atoms = read(str(cif_path))
        logger.debug(f"Original structure has {len(atoms)} atoms")
        atoms = _make_supercell(atoms, cutoff=data_config['structure']['min_lat_len'])
        logger.debug(f"Supercell has {len(atoms)} atoms after expansion")

        if data_config.get('scale') is not None:
            olen = len(atoms)
            atoms = _make_supercell(atoms, scale=data_config['scale'])
            print("DEBUG SUPERCELL", data_config['scale'], olen, len(atoms))
        
        # Check supercell constraints
        cell_params = atoms.cell.cellpar()
        logger.debug(f"Cell parameters: {cell_params}")
        for l in cell_params[:3]:
            if data_config['structure']['max_lat_len'] and l > data_config['structure']['max_lat_len']:
                logger.warning(f"{matid}: supercell length {l} exceeds max_length {data_config['max_lat_len']}, but graph will still be saved")
        if data_config['structure']['max_num_atoms'] and len(atoms) > data_config['structure']['max_num_atoms']:
            logger.warning(f"{matid}: {len(atoms)} atoms exceeds max_num_atoms {data_config['structure']['max_num_atoms']}, but graph will still be saved")
            
        struct = AseAtomsAdaptor().get_structure(atoms)
        logger.debug(f"Successfully converted to pymatgen structure")
    except Exception as e:
        logger.error(f"{matid} failed when building supercell: {e}", exc_info=True)
        return False

    try:
        # Save graph data
        data = Data(
            atomic_numbers=torch.tensor([site.specie.Z for site in struct], dtype=torch.long),
            # atomic_symbols=[site.specie.symbol for site in struct],
            pos=torch.from_numpy(np.stack([site.coords for site in struct])).to(torch.float),
            cell=torch.tensor(struct.lattice.matrix, dtype=torch.float),
            matid=matid,
        )
        torch.save(data, graph_path)
        logger.info(f"Saved graph data to {graph_path}")

        spnode_config = data_config['spnode']
        spdata = add_spatial_nodes(data, spnode_config['z'], spnode_config['num'], spnode_config['sample'], spnode_config['repulsion_distance'])

        torch.save(spdata, spgraph_path)
        logger.info(f"Saved spgraph data to {spgraph_path}")


    except Exception as e:
        logger.error(f"Error creating/saving graph data: {e}", exc_info=True)
        return False

    logger.info(f"Successfully completed graph construction for {matid}")
    return True

def process_matids(matids: List[str], config: str, data_config: Dict[str, Any]):
    """Process a batch of matids"""
    try:
        init_config(config)
        logger = get_logger(filename=Path(__file__).stem)
        logger.info(f"Starting process with {len(matids)} materials")
        
        torch.set_default_dtype(torch.double)
        logger.debug("Set default dtype to double")
        
        for matid in tqdm(matids, desc="Processing materials"):
            try:
                logger.info(f"Processing material {matid}")
                success = build_graph_i(
                    matid=matid,
                    data_config=data_config
                )
                if not success:
                    logger.warning(f"Failed to process {matid}")
            except Exception as e:
                logger.error(f"Error processing {matid}: {str(e)}", exc_info=True)
    except Exception as e:
        logger.error(f"Fatal error in process_matids: {e}", exc_info=True)
        raise

def build_graph(config: str):
    """Build graph data for all materials in dataset using multiprocessing."""
    logger = get_logger(filename=Path(__file__).stem)
    logger.info("Starting graph construction")
    
    try:
        init_config(config)
        data_config = get_data_config()
        matids = data_config['matid_df']['matid'].tolist()

        fmatids = []
        for matid in tqdm(matids):
            graph_path = data_config['graph_dir'] / f"{matid}.pt"
            spgraph_path = data_config['spgraph_dir'] / f'{matid}.pt'
            
            if graph_path.exists() and spgraph_path.exists() and not data_config['override']:
                logger.info(f"{matid} graph already exist, skipping")
            else:
                fmatids.append(matid)
        matids = fmatids

        num_process = data_config.get('num_process', 1)
        logger.info(f"Loaded configuration, found {len(matids)} materials to process with {num_process} processes")

        # Split matids into chunks for each process
        chunk_size = (len(matids) + num_process - 1) // num_process
        matid_chunks = [matids[i:i + chunk_size] for i in range(0, len(matids), chunk_size)]
        logger.info(f"Split materials into {len(matid_chunks)} chunks")

        processes = []
        for i, chunk in enumerate(matid_chunks):
            p = mp.Process(
                target=process_matids,
                args=(chunk, config, data_config.copy())
            )
            p.start()
            processes.append(p)
            logger.debug(f"Started process {i+1} (PID: {p.pid}) with {len(chunk)} materials")

        for p in processes:
            p.join()
            logger.debug(f"Process {p.pid} completed")

        logger.info("All processes completed")
    except Exception as e:
        logger.error(f"Error in build_graph: {e}", exc_info=True)
        raise

@click.command()
@click.option('--config', type=str, required=True, help="Path to config file")
def cli(config: str):
    """Command line interface for building graph data."""
    try:
        # Set multiprocessing start method
        mp.set_start_method('spawn')
        init_config(config)
        logger = get_logger(filename=Path(__file__).stem)
        logger.info("Starting CLI execution")
        logger.debug("Set multiprocessing start method to 'spawn'")
        build_graph(config)
        logger.info("CLI execution completed successfully")
    except Exception as e:
        logger.critical(f"CLI execution failed: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    cli()
