import math
from pathlib import Path
from typing import Dict, Any, List
import multiprocessing as mp
import json
import warnings

import click
import numpy as np
import torch
from tqdm import tqdm

import torch
from torch_geometric.data import Data

from ase.io import read
from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from spatialread.utils.chem import _make_supercell
from spatialread.utils.log import get_logger
from spatialread.utils.radius_graph import radius_graph_pbc
from spatialread.utils.graph import filter_edge
from spatialread.utils.ocp import get_pbc_distances
from spatialread.config import get_data_config, init_config


class GraphPreprocessor:
    """Handles Graph data preprocessing and caching."""

    def __init__(self, config: Dict[str, Any], device: Any, sp: bool = True):
        self.logger = get_logger(filename=Path(__file__).stem)

        # Get parameters from config
        self.override = config.get("override", False)
        self.gnn_min_lat_len = config["structure"]["min_lat_len"]
        self.max_num_atoms = config["structure"]["max_num_atoms"]
        self.spnode_z = config['spnode']['z']
        self.num_spnode = config['spnode']['num']

        # Setup directories
        self.cif_dir = config["cif_dir"]
        # self.graph_dir = config["graph_dir"]
        # self.spgraph_dir = config['spgraph_dir']
        if sp:
            self.graph_dir = config['spgraph_dir']
        else:
            self.graph_dir = config['graph_dir']

        self.device = device

        # Load target values
        self.df = config["matid_df"]

    def process_material(self, matid: str) -> bool:
        """Process and save data for a single material."""
        output_path = self.graph_dir / f"{matid}.pt"

        if output_path.exists() and not self.override:
            data = torch.load(output_path, weights_only=False, map_location="cpu")
            if data.get("edge_index") is not None:
                self.logger.info(
                    f"Skipping {matid} - {self.graph_dir / f'{matid}.pt'} edge already processed"
                )
                return True

        def process(device):
            if not (self.graph_dir / f"{matid}.pt").exists():
                self.logger.error(
                    f"Failed to load graph from {self.graph_dir / f'{matid}.pt'}, Please process graph first"
                )
                return False

            data = torch.load(
                self.graph_dir / f"{matid}.pt", weights_only=False, map_location=device
            )
            if self.max_num_atoms is not None and len(data.atomic_numbers) > self.max_num_atoms:
                self.logger.info(
                    f"Skip {matid} - {self.graph_dir / f'{matid}.pt'} with {len(data.atomic_numbers)} atoms"
                )
                return False

            data.natoms = torch.tensor([len(data.atomic_numbers)], device=device)
            data.cell = data.cell.reshape(
                1, 3, 3
            )  # for collate we unsqueeze one dimension

            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                data,
                radius=10,
                max_num_neighbors_threshold=5000,
                pbc=[True, True, True],
                rep=[1, 1, 1]
            )

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
            edge_dist: torch.Tensor = out["distances"]
            edge_vec: torch.Tensor = out["distance_vec"]

            # filter edge is left for training to dynamically set cutoff and max_num_neighbors

            # edge_index, edge_dist, edge_vec, [cell_offsets] = filter_edge(edge_index, edge_dist, edge_vec, 'p2pv', data.atomic_numbers==self.spnode_z, cutoff, max_neighbors, other_content=[cell_offsets])

            # neighbors[0] = edge_index.shape[1]

            data["edge_index"] = edge_index
            data["edge_dist"] = edge_dist
            data["edge_vec"] = edge_vec
            data["cell_offsets"] = cell_offsets
            data['neighbors'] = neighbors

            torch.save(data.to("cpu"), self.graph_dir / f"{matid}.pt")

        try:
            process(self.device)
            self.logger.info(
                f"Successfully process {matid}"
            )
            return True
        except torch.cuda.OutOfMemoryError as e:
            self.logger.info(
                f"OOM to process {matid}: {str(e)}"
            )
            try:
                process('cpu')
                self.logger.info(
                    f"Successfully process {matid} in cpu"
                )
                return True
            # except torch.cuda.OutOfMemoryError as e:
            #     self.logger.info(
            #         f"Failed to process {matid}: {str(e)}, use cpu instaed"
            #     )
            #     try:
            #         process("cpu", 10, 30)
            #     except Exception as e:
            #         self.logger.error(
            #             f"Failed to process {matid}: {str(e)}", exc_info=True
            #         )
            #         return False
            except Exception as e:
                self.logger.error(f"Failed to process {matid}: {str(e)}", exc_info=True)
                return False
        except Exception as e:
            self.logger.error(f"Failed to process {matid}: {str(e)}", exc_info=True)
            return False


def process_chunk(matids: List[str], config: str, data_config: Dict[str, Any], device: str, sp: bool):
    """Process a chunk of materials with a single preprocessor instance."""
    init_config(config)
    preprocessor = GraphPreprocessor(data_config, device, sp)
    for matid in tqdm(matids):
        preprocessor.process_material(matid)


def build_edge(config: str, devices: List[str], sp: bool):
    """Build Graph data for all materials using multiprocessing."""
    logger = get_logger(filename=Path(__file__).stem)
    logger.info("Starting Graph data preprocessing")

    try:
        init_config(config)
        data_config = get_data_config()

        # Get list of materials to process
        matid_df = data_config["matid_df"]
        matids = matid_df["matid"].tolist()

        # Setup multiprocessing
        chunk_size = (len(matids) + len(devices) - 1) // len(devices)
        matid_chunks = [
            matids[i : i + chunk_size] for i in range(0, len(matids), chunk_size)
        ]
        assert len(matid_chunks) == len(devices)

        processes = []
        for device, chunk in zip(devices, matid_chunks):
            p = mp.Process(
                target=process_chunk, args=(chunk, config, data_config, device, sp)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        logger.info("Graph preprocessing completed successfully")

    except Exception as e:
        logger.error(f"Graph preprocessing failed: {str(e)}", exc_info=True)
        raise


@click.command()
@click.option("--config", type=str, required=True, help="Path to config file")
@click.option(
    "--devices",
    type=str,
    required=True,
    help="Device to calculate edge_index, separated by comma",
)
@click.option(
    '--sp',
    is_flag=True
)
def cli(config: str, devices: str, sp: bool):
    """Command line interface for Graph preprocessing."""
    try:
        mp.set_start_method("spawn")
        init_config(config)
        logger = get_logger(filename=Path(__file__).stem)
        logger.info("Starting Graph preprocessing CLI")
        devices = devices.split(",")
        build_edge(config, devices, sp)
        logger.info("Graph preprocessing CLI completed")
    except Exception as e:
        logger.critical(f"Graph CLI failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    cli()
