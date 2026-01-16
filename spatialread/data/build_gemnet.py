import click
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
from functools import partial

import torch
from torch_geometric.data import Data

from pymatgen.core.lattice import Lattice

# JMP
from spatialread.utils.goc_graph import (
    Cutoffs,
    Graph,
    MaxNeighbors,
    generate_graph,
    subselect_graph,
    tag_mask,
)

from spatialread.config import (
    init_config,
    get_config,
    get_model_config,
    get_data_config,
    get_train_config,
)
from spatialread.utils.log import get_logger

# from spatialread.utils.graph import radius_graph_pbc_ase

from typing import List, Union, Any, Dict


class JMP:
    def __init__(
        self,
        config,
        device,
        sp: bool = False,
        cutoff: float = 12.0,
        max_neighbors: int = 30,
    ):
        self.logger = get_logger(filename=Path(__file__).stem)

        # Get parameters from config
        self.override = config["override"]

        self.graph_dir = config["graph_dir"]
        self.gemnet_dir = config["gemnet_dir"]

        self.spgraph_dir = config["spgraph_dir"]
        self.spgemnet_dir = config["spgemnet_dir"]

        self.spnode_config = config["spnode"]

        # Load target values
        self.df = config["matid_df"]

        self.device = device

        self.sp = sp

        self.cutoff = cutoff
        self.max_neighbors = max_neighbors

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

        cutoff = self.cutoff
        max_neighbors = self.max_neighbors
        # # if self.config.conditional_max_neighbors:
        # if data.natoms > 300:
        #     max_neighbors = 5
        # elif data.natoms > 200:
        #     max_neighbors = 10
        # else:
        #     max_neighbors = 30

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
        if not self.sp:
            # try:
            skip = False
            if (self.gemnet_dir / f"{matid}.pt").exists() and not self.override:
                try:
                    torch.load(
                        self.gemnet_dir / f"{matid}.pt",
                        weights_only=False,
                        map_location="cpu",
                    )
                    skip = True
                    self.logger.info(f"Material {matid} already processed, skip")
                except Exception as e:
                    self.logger.info(
                        f"Material {matid}'s gemnet data is broken, reprocess", e
                    )
                    skip = False

            if not skip:
                self.logger.info(f"Start to process gemnet data of {matid}")
                data = torch.load(
                    self.graph_dir / f"{matid}.pt",
                    weights_only=False,
                    map_location=self.device,
                )
                graph = self.data_transform(data).to("cpu")
                torch.save(graph, self.gemnet_dir / f"{matid}.pt")
        else:
            skip = False
            if (self.spgemnet_dir / f"{matid}.pt").exists() and not self.override:
                try:
                    torch.load(
                        self.spgemnet_dir / f"{matid}.pt",
                        weights_only=False,
                        map_location="cpu",
                    )
                    skip = True
                    self.logger.info(
                        f"Material spatial {matid} already processed, skip"
                    )
                except Exception as e:
                    self.logger.info(
                        f"Material {matid}'s gemnet data is broken, reprocess", e
                    )
                    skip = False

            if not skip:
                self.logger.info(f"Finish to process gemnet data of {matid}")
                data = torch.load(
                    self.spgraph_dir / f"{matid}.pt",
                    weights_only=False,
                    map_location=self.device,
                )
                graph = self.data_transform(data).to("cpu")
                torch.save(graph, self.spgemnet_dir / f"{matid}.pt")
                self.logger.info(
                    f"Finish to process gemnet data with spatial nodes of {matid}"
                )


def process_chunk(
    matids: List[str],
    config: str,
    data_config: Dict[str, Any],
    device,
    sp,
    cutoff,
    max_neighbors,
):
    """Process a chunk of materials with a single preprocessor instance."""
    init_config(config)
    preprocessor = JMP(data_config, device, sp, cutoff, max_neighbors)
    for matid in tqdm(matids):
        preprocessor.process_material(matid)


def build_data(config, devices, sp):
    logger = get_logger(filename=Path(__file__).stem)
    logger.info("Starting JMP/GemNet data preprocessing")

    try:
        init_config(config)
        data_config = get_data_config()
        model_config = get_model_config()

        cutoff = model_config["gemnet"]["cutoff"]
        max_neighbors = model_config["gemnet"]["max_num_neighbors"]

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
                target=process_chunk,
                args=(chunk, config, data_config, device, sp, cutoff, max_neighbors),
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
@click.option("--sp", is_flag=True)
def cli(config: str, devices: str, sp: bool):
    """Command line interface for Graph preprocessing."""
    try:
        mp.set_start_method("spawn")
        init_config(config)
        logger = get_logger(filename=Path(__file__).stem)
        logger.info("Starting prebuilding data for jmp/gemnet")
        devices = devices.split(",")
        build_data(config, devices, sp)
    except Exception as e:
        logger.critical(f"Graph CLI failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    cli()
