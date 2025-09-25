import pandas as pd
from tqdm import tqdm
from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, HeteroData, TensorAttr
from torch_geometric.data.collate import collate
import lightning as L
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
from einops import rearrange

from spatialread.config import get_data_config, get_train_config, get_model_config
from spatialread.utils.log import Log
from spatialread.utils.chem import atomic_number_to_symbol
from spatialread.utils.metric import metric_classification, metric_regression
from spatialread.utils.graph import get_pbc_distances

from abc import ABC, abstractmethod
from typing import Union, Dict, List, Any, Literal
import numpy.typing as npt


class BaseDataset(Dataset, Log):
    def __init__(self, split):
        Dataset.__init__(self)
        Log.__init__(self)

        self.split = split
        assert self.split in [
            "train",
            "val",
            "test",
        ], "Split must be 'train', 'val', or 'test'"
        self.data_config = get_data_config()
        self.train_config = get_train_config()
        self.model_config = get_model_config()

        self._init_params()

        self._filter_max_num_atoms()

    def _init_params(self) -> None:
        """Initialize dataset parameters from config."""
        self.root_dir = self.data_config["root_dir"]
        self.cif_dir = self.data_config["cif_dir"]
        self.matid_df = self.data_config[f"{self.split}_matid_df"]

        if self.model_config['head'] == 'atom_mlp':
            self.spgraph_dir = self.data_config["spgraph_dir"]
            self.graph_dir = self.data_config['graph_dir']
        elif self.model_config['head'] == 'spnode_mlp' or self.model_config['head'] == 'transformer':
            self.spgraph_dir = self.data_config["spgraph_dir"]
            self.graph_dir = self.data_config["graph_dir"]
        else:
            raise ValueError(f"Unsupported head {self.model_config['head']}")

        self.gemnet_dir = self.data_config['gemnet_dir']

        self.max_num_atoms = self.data_config['structure']['max_num_atoms']

    def _filter_max_num_atoms(self):
        if self.max_num_atoms is None:
            return
        if 'num_atoms' not in self.matid_df:
            for matid in tqdm(self.matid_df['matid']):
                graph = torch.load(self.graph_dir / f"{matid}.pt", weights_only=False)
                self.matid_df.loc[self.matid_df['matid'] == matid, 'num_atoms'] = graph.atomic_numbers.shape[0]
        if self.max_num_atoms is not None:
            self.matid_df = self.matid_df[self.matid_df['num_atoms'] <= self.max_num_atoms]

        print(f"Filter max num atoms of {self.max_num_atoms}")

    def _load_structure(self, matid: str):
        graph = torch.load(self.graph_dir / f"{matid}.pt", weights_only=False, map_location="cpu")

        graph.natoms = torch.tensor([graph.atomic_numbers.shape[0]])
        graph.cell = graph.cell.reshape(1, 3, 3)

        spgraph = torch.load(self.spgraph_dir / f"{matid}.pt", weights_only=False, map_location="cpu")

        spgraph.natoms = torch.tensor([spgraph.atomic_numbers.shape[0]])
        spgraph.cell = spgraph.cell.reshape(1, 3, 3)


        if 'gemnet' in self.model_config['gnn']:
            gemnet = torch.load(self.gemnet_dir / f"{matid}.pt", weights_only=False, map_location="cpu")
            return {"graph": graph, "spgraph": spgraph, "gemnet": gemnet}
        else:
            return {"graph": graph, "spgraph": spgraph}

        # return {"graph": graph, "token": token, "coord": coord, "distance": distance, "edge_type": edge_type}

    def _load_gemnet(self, matid: str):
        graph = torch.load(self.gemnet_dir / f'{matid}.pt', weights_only=False, map_location='cpu')
        return {
            "gemnet": graph
        }

    def __getitem__(self, idx: int):
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.matid_df)

    def collate_fn(
        self, data_list: List[Dict[str, Union[str, torch.Tensor]]], list_keys=[], stack_keys=[], concat_keys=[]
    ) -> Dict[str, Union[List[str], torch.Tensor]]:
        ret = {}
        for key in list_keys:
            ret[key] = [d[key] for d in data_list]
        for key in stack_keys:
            ret[key] = torch.stack([d[key] for d in data_list])
        for key in concat_keys:
            ret[key] = torch.cat([d[key] for d in data_list])

        return ret


class FinetuneDataset(BaseDataset):
    """
    Custom PyTorch Dataset for loading SpatialRead (Crystal Chemistry Knowledge Database) data.
    Supports loading different types of features (FF and MACE) for crystal structures.
    """

    def __init__(
        self,
        split: Literal["train", "val", "test"],
        mean: Dict[str, float] = None,
        std: Dict[str, float] = None,
    ):
        super().__init__(split)

        self._filter_label()

        if self.task_type == 'Regression':
            self._set_statistic(mean, std)

    def _init_params(self) -> None:
        super()._init_params()
        self.task_name = self.train_config["task_name"]
        self.task_type = self.train_config['task_type']
        assert self.task_type in ['Regression', 'BinaryClassification', 'Classification'], f'Unsupported task type {self.task_type}'

    # def _load_structure(self, matid: str):
    #     ret = super()._load_structure(matid)
    #     graph = ret['graph']

    #     # if self.split == 'train':
    #     #     graph = self.add_noise(graph, scale=self.train_config['loss']['noise']['scale'])

    #     return {
    #         "graph": graph
    #     }

    def _filter_label(self) -> None:
        olen = len(self.matid_df)
        self.matid_df = self.matid_df.dropna(
            subset=[self.task_name]
        )
        self.log(f"Total [{len(self.matid_df)}/{olen}] mats with label")

    def _set_statistic(
        self, mean: Dict[str, float] = None, std: Dict[str, float] = None
    ):
        if mean is not None and std is not None:
            # Use provided statistics
            self._mean = mean
            self._std = std
        elif self.split == "train":
            # Compute statistics from training set
            self._mean = {self.task_name: self.matid_df[self.task_name].mean()}
            self._std = {self.task_name: self.matid_df[self.task_name].std()}
        else:
            raise ValueError(
                "Normalization statistics must be provided for validation and test sets"
            )

    def _load_label(self, idx: int):
        matid = self.matid_df.iloc[idx]["matid"]

        if self.task_type == 'Regression':
            label = self.matid_df.iloc[idx][self.task_name]
            if hasattr(self, "_mean") and hasattr(self, "_std"):
                label = (label - self._mean[self.task_name]) / self._std[self.task_name]
            return {self.task_name: torch.tensor([label]).float()}
        elif self.task_type == 'BinaryClassification':
            label = self.matid_df.iloc[idx][self.task_name]
            return {self.task_name: torch.tensor([label]).float()}
        elif self.task_type == 'Classification':
            label = self.matid_df.iloc[idx][self.task_name]
            label = torch.tensor([label]).long()
            return {self.task_name: label}
        else:
            raise ValueError(f'Unsupported task type: {self.task_type}')

    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        matid = self.matid_df.iloc[idx]["matid"]

        ret = {"matid": matid}
        ret.update(self._load_structure(matid))
        ret.update(self._load_label(idx))

        # ret.update(self._load_gemnet(matid))

        return ret

    def collate_fn(
        self, data_list: List[Dict[str, Union[str, torch.Tensor]]]
    ) -> Dict[str, Union[List[str], torch.Tensor]]:
        # return super().collate_fn(data_list, list_keys=['matid', 'graph'], stack_keys=['pes', 'token', 'coord', 'distance', 'edge_type'], concat_keys=[self.task_name])
        # return super().collate_fn(data_list, list_keys=['matid', 'graph'], stack_keys=[], concat_keys=[self.task_name])
        list_keys = ['matid', 'graph', 'spgraph']
        if 'gemnet' in self.model_config['gnn']:
            list_keys.append('gemnet')
        return super().collate_fn(data_list, list_keys=list_keys, stack_keys=[], concat_keys=[self.task_name])


class BaseDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.data_config = get_data_config()
        self.train_config = get_train_config()
        self.datasets = {}

    @abstractmethod
    def setup(self, DatasetCls, stage) -> None:
        for split in ['train', 'val', 'test']:
            self.datasets[split] = DatasetCls(split)

    def train_dataloader(self) -> DataLoader:
        """Create DataLoader for training data."""
        return DataLoader(
            dataset=self.datasets["train"],
            batch_size=self.train_config["batch_size"],
            num_workers=self.train_config["num_workers"],
            collate_fn=self.datasets["train"].collate_fn,
            shuffle=False,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create DataLoader for validation data."""
        return DataLoader(
            dataset=self.datasets["val"],
            batch_size=self.train_config["batch_size"],
            num_workers=self.train_config["num_workers"],
            collate_fn=self.datasets["val"].collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Create DataLoader for test data."""
        return DataLoader(
            dataset=self.datasets["test"],
            batch_size=self.train_config["batch_size"],
            num_workers=self.train_config["num_workers"],
            collate_fn=self.datasets["test"].collate_fn,
            pin_memory=True,
        )


class FinetuneDatamodule(BaseDataModule):
    def setup(self, stage: str = None) -> None:
        self.datasets["train"] = FinetuneDataset("train")
        task_type = self.train_config['task_type']
        if task_type == 'Regression':
            self._mean = self.datasets["train"]._mean
            self._std = self.datasets["train"]._std

            # Then setup validation and test sets using the same statistics
            self.datasets["val"] = FinetuneDataset("val", self._mean, self._std)
            self.datasets["test"] = FinetuneDataset("test", self._mean, self._std)
        elif task_type == 'BinaryClassification' or task_type == 'Classification':
            self.datasets["val"] = FinetuneDataset("val")
            self.datasets["test"] = FinetuneDataset("test")
        else:
            raise ValueError('Unsupported task type', task_type)