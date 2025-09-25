import click
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
from functools import partial

import torch
from torch_geometric.data import Data

from pymatgen.core.lattice import Lattice

# JMP
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

from spatialread.config import init_config, get_config, get_model_config, get_data_config, get_train_config
from spatialread.utils.log import get_logger
from spatialread.utils.radius_graph import radius_graph_pbc

from typing import List, Union, Any, Dict


class JMP():
    def __init__(self, config, device):
        self.logger = get_logger(filename=Path(__file__).stem)

        # Get parameters from config
        self.override = config['override']

        self.graph_dir = config["graph_dir"]
        self.gemnet_dir = config['gemnet_dir']

        self.spnode_config = config['spnode']

        self.device = device

        # Load target values
        self.df = config["matid_df"]

    def generate_graphs(
        self,
        data: Data,
        cutoffs: Cutoffs,
        max_neighbors: MaxNeighbors,
        pbc: bool,
    ) -> Data:

        # edge_index, cell_offsets, neighbors = radius_graph_pbc(
        #     data,
        #     radius=10,
        #     max_num_neighbors_threshold=5000,
        #     pbc=[True, True, True],
        #     rep=[2, 2, 2]
        # )
        aint_graph = generate_graph(
            data,
            cutoff=cutoffs.aint,
            max_neighbors=max_neighbors.aint,
            pbc=pbc,
        )
        subselect = partial(
            subselect_graph,
            data,
            aint_graph,
            cutoff_orig=cutoffs.aint,
            max_neighbors_orig=max_neighbors.aint,
        )
        main_graph = subselect(cutoffs.main, max_neighbors.main)
        aeaint_graph = subselect(cutoffs.aeaint, max_neighbors.aeaint)
        qint_graph = subselect(cutoffs.qint, max_neighbors.qint)

        qint_graph = tag_mask(data, qint_graph, tags=[1, 2])

        graphs = {
            "main": main_graph,
            "a2a": aint_graph,
            "a2ee2a": aeaint_graph,
            "qint": qint_graph,
        }

        for graph_type, graph in graphs.items():
            for key, value in graph.items():
                setattr(data, f"{graph_type}_{key}", value)

        return data

    def data_transform(self, data: Data) -> Data:
        data.atomic_numbers = data.atomic_numbers.long()
        data.natoms = len(data.atomic_numbers)
        data.pos = data.pos.float()
        data.cell = data.cell.reshape(1, 3, 3)

        # data = self.add_virtual_nodes(data)

        device = data["atomic_numbers"].device

        data.tags = 2 * torch.ones(data.natoms).to(device)
        data.tags = data.tags.long().to(device)

        data.fixed = torch.zeros(data.natoms, dtype=torch.bool).to(device)

        cutoff = 8
        max_neighbors = 8
        # if data.natoms > 300:
        #     max_neighbors = 5
        # elif data.natoms > 200:
        #     max_neighbors = 10
        # else:
        #     max_neighbors = 20

        # cutoff = 19
        # max_neighbors = 8

        # if data.natoms > 200:
        #     max_neighbors = 5

        data = self.generate_graphs(
            data,
            cutoffs=Cutoffs.from_constant(cutoff),
            max_neighbors=MaxNeighbors.from_goc_base_proportions(max_neighbors),
            pbc=True,
        )

        return data

    def process_material(self, matid: str):
        try:
            if (self.gemnet_dir / f'{matid}.pt').exists() and not self.override:
                self.logger.info(f"Material {matid} already processed, skip")
                return
            self.logger.info(f"Start to process gemnet data of {matid}")
            data = torch.load(self.graph_dir / f'{matid}.pt', weights_only=False, map_location=self.device)
            # if len(data.atomic_numbers) > 512 + 768:
            #     self.logger.info(
            #         f"Skip {matid} - {self.graph_dir / f'{matid}.pt'} with {len(data.atomic_numbers)} atoms"
            #     )
            #     return False
            try:
                graph = self.data_transform(data).to('cpu')
            except torch.cuda.OutOfMemoryError:
                self.logger.warn(f"Fail to process gemnet data of {matid}: out of memory, use cpu instead")
                data = data.to('cpu')
                graph = self.data_transform(data).to('cpu')
            except Exception as e:
                self.logger.error(f"Fail to process gemnet data of {matid}: {e}")
                return
            torch.save(graph, self.gemnet_dir / f'{matid}.pt')
            self.logger.info(f"Finish to process gemnet data of {matid}")
        except Exception as e:
            self.logger.error(f"Fail to process gemnet data of {matid}: {e}")


def process_chunk(matids: List[str], config: str, data_config: Dict[str, Any], device):
    """Process a chunk of materials with a single preprocessor instance."""
    init_config(config)
    preprocessor = JMP(data_config, device)
    for matid in tqdm(matids):
        preprocessor.process_material(matid)


def build_data(config, devices):
    logger = get_logger(filename=Path(__file__).stem)
    logger.info("Starting JMP/GemNet data preprocessing")

    try:
        init_config(config)
        data_config = get_data_config()
        model_config = get_model_config()

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
                target=process_chunk, args=(chunk, config, data_config, device)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        logger.info(f"Graph preprocessing completed successfully")

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
def cli(config: str, devices: str):
    """Command line interface for Graph preprocessing."""
    try:
        mp.set_start_method("spawn")
        init_config(config)
        logger = get_logger(filename=Path(__file__).stem)
        logger.info("Starting prebuilding data for jmp/gemnet")
        devices = devices.split(",")
        build_data(config, devices)
    except Exception as e:
        logger.critical(f"Graph CLI failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    cli()