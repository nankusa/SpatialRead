"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import fnmatch
import itertools
import math
from abc import abstractmethod
from collections.abc import Iterable, Mapping
from functools import partial
from logging import getLogger
from pathlib import Path
from typing import Annotated, Any, Generic, Literal, TypeAlias, assert_never, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .lightning import Base, BaseConfig, Field, LightningModuleBase, TypedConfig
from .lightning.data.balanced_batch_sampler import (
    BalancedBatchSampler,
    DatasetWithSizes,
)
from .lightning.util.typed import TypedModuleDict
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import BaseData
from torch_scatter import scatter
from typing_extensions import TypedDict, TypeVar, override

from ...datasets.finetune.base import LmdbDataset
from ...datasets.finetune_pdbbind import PDBBindConfig, PDBBindDataset
from ...models.gemnet.backbone import GemNetOCBackbone, GOCBackboneOutput
from ...models.gemnet.config import BackboneConfig
from ...models.gemnet.layers.base_layers import ScaledSiLU
from ...modules import transforms as T
from ...modules.dataset import dataset_transform as DT
from ...modules.dataset.common import CommonDatasetConfig, wrap_common_dataset
from ...modules.early_stopping import EarlyStoppingWithMinLR
from ...modules.ema import EMAConfig
from ...modules.scheduler.gradual_warmup_lr import GradualWarmupScheduler
from ...modules.scheduler.linear_warmup_cos_rlp import (
    PerParamGroupLinearWarmupCosineAnnealingRLPLR,
)
from ...modules.transforms.normalize import NormalizationConfig
from ...utils.goc_graph import (
    Cutoffs,
    Graph,
    MaxNeighbors,
    generate_graph,
    subselect_graph,
    tag_mask,
)
from ...utils.state_dict import load_state_dict
from ..config import (
    EmbeddingConfig,
    OptimizerConfig,
    OutputConfig,
    optimizer_from_config,
)
from .metrics import FinetuneMetrics, MetricPair, MetricsConfig

log = getLogger(__name__)


class GraphScalarOutputHead(Base[TConfig], nn.Module, Generic[TConfig]):
    @override
    def __init__(
        self,
        config: TConfig,
        reduction: str | None = None,
    ):
        super().__init__(config)

        if reduction is None:
            reduction = self.config.graph_scalar_reduction_default

        self.out_mlp = self.mlp(
            ([self.config.backbone.emb_size_atom] * self.config.output.num_mlps)
            + [self.config.backbone.num_targets],
            activation=self.config.activation_cls,
        )
        self.reduction = reduction

    @override
    def forward(
        self,
        input: OutputHeadInput,
        *,
        scale: torch.Tensor | None = None,
        shift: torch.Tensor | None = None,
    ) -> torch.Tensor:
        data = input["data"]
        backbone_output = input["backbone_output"]

        n_molecules = int(torch.max(data.batch).item() + 1)

        output = self.out_mlp(backbone_output["energy"])  # (n_atoms, 1)
        if scale is not None:
            output = output * scale
        if shift is not None:
            output = output + shift

        output = scatter(
            output,
            data.batch,
            dim=0,
            dim_size=n_molecules,
            reduce=self.reduction,
        )  # (bsz, 1)
        output = rearrange(output, "b 1 -> b")
        return output


class GraphBinaryClassificationOutputHead(Base[TConfig], nn.Module, Generic[TConfig]):
    @override
    def __init__(
        self,
        config: TConfig,
        classification_config: BinaryClassificationTargetConfig,
        reduction: str | None = None,
    ):
        super().__init__(config)

        assert (
            classification_config.num_classes == 2
        ), "Only binary classification supported"

        if reduction is None:
            reduction = self.config.graph_scalar_reduction_default

        self.out_mlp = self.mlp(
            ([self.config.backbone.emb_size_atom] * self.config.output.num_mlps) + [1],
            activation=self.config.activation_cls,
        )
        self.classification_config = classification_config
        self.reduction = reduction

    @override
    def forward(self, input: OutputHeadInput) -> torch.Tensor:
        data = input["data"]
        backbone_output = input["backbone_output"]

        n_molecules = int(torch.max(data.batch).item() + 1)

        output = self.out_mlp(backbone_output["energy"])  # (n, num_classes)
        output = scatter(
            output,
            data.batch,
            dim=0,
            dim_size=n_molecules,
            reduce=self.reduction,
        )  # (bsz, num_classes)
        output = rearrange(output, "b 1 -> b")
        return output


class GraphMulticlassClassificationOutputHead(
    Base[TConfig], nn.Module, Generic[TConfig]
):
    @override
    def __init__(
        self,
        config: TConfig,
        classification_config: MulticlassClassificationTargetConfig,
        reduction: str | None = None,
    ):
        super().__init__(config)

        if reduction is None:
            reduction = self.config.graph_scalar_reduction_default

        self.out_mlp = self.mlp(
            ([self.config.backbone.emb_size_atom] * self.config.output.num_mlps)
            + [classification_config.num_classes],
            activation=self.config.activation_cls,
        )
        self.classification_config = classification_config
        self.reduction = reduction

        self.dropout = None
        if classification_config.dropout:
            self.dropout = nn.Dropout(classification_config.dropout)

    @override
    def forward(self, input: OutputHeadInput) -> torch.Tensor:
        data = input["data"]
        n_molecules = int(torch.max(data.batch).item() + 1)

        x = input["backbone_output"]["energy"]
        if self.dropout is not None:
            x = self.dropout(x)

        x = self.out_mlp(x)  # (n, num_classes)
        x = scatter(
            x,
            data.batch,
            dim=0,
            dim_size=n_molecules,
            reduce=self.reduction,
        )  # (bsz, num_classes)
        return x


class NodeVectorOutputHead(Base[TConfig], nn.Module, Generic[TConfig]):
    @override
    def __init__(
        self,
        config: TConfig,
        reduction: str | None = None,
    ):
        super().__init__(config)

        if reduction is None:
            reduction = self.config.graph_scalar_reduction_default

        self.out_mlp = self.mlp(
            ([self.config.backbone.emb_size_edge] * self.config.output.num_mlps)
            + [self.config.backbone.num_targets],
            activation=self.config.activation_cls,
        )
        self.reduction = reduction

    @override
    def forward(self, input: OutputHeadInput) -> torch.Tensor:
        data = input["data"]
        backbone_output = input["backbone_output"]

        n_atoms = data.atomic_numbers.shape[0]

        output = self.out_mlp(backbone_output["forces"])
        output = output * backbone_output["V_st"]  # (n_edges, 3)
        output = scatter(
            output,
            backbone_output["idx_t"],
            dim=0,
            dim_size=n_atoms,
            reduce=self.reduction,
        )
        return output


class FinetuneModelBase(LightningModuleBase[TConfig], Generic[TConfig]):

    @override
    def __init__(self, hparams: TConfig):
        self.validate_config(hparams)
        super().__init__(hparams)

        # Set up callbacks
        if (ema := self.config.ema) is not None:
            self.register_callback(lambda: ema.construct_callback())

        self._set_rlp_config_monitors()

        self.embedding = nn.Embedding(
            num_embeddings=self.config.embedding.num_elements,
            embedding_dim=self.config.embedding.embedding_size,
        )

        self.backbone = self._construct_backbone()
        self.register_shared_parameters(self.backbone.shared_parameters)

        self.construct_output_heads()

        self.train_metrics = FinetuneMetrics(
            self.config.metrics,
            self.metrics_provider,
            self.config.graph_scalar_targets,
            self.config.graph_classification_targets,
            self.config.node_vector_targets,
        )
        self.val_metrics = FinetuneMetrics(
            self.config.metrics,
            self.metrics_provider,
            self.config.graph_scalar_targets,
            self.config.graph_classification_targets,
            self.config.node_vector_targets,
        )
        self.test_metrics = FinetuneMetrics(
            self.config.metrics,
            self.metrics_provider,
            self.config.graph_scalar_targets,
            self.config.graph_classification_targets,
            self.config.node_vector_targets,
        )

        # Sanity check: ensure all named_parameters have requires_grad=True,
        #   otherwise add them to ignored_parameters.
        self.ignored_parameters = set[nn.Parameter]()
        for name, param in self.named_parameters():
            if param.requires_grad:
                continue
            self.ignored_parameters.add(param)
            log.info(f"Adding {name} to ignored_parameters")

        self.process_freezing()

        if (ckpt_best := self.config.ckpt_best) is not None:
            if (monitor := ckpt_best.monitor) is None:
                monitor, mode = self.primary_metric()
            else:
                if (mode := ckpt_best.mode) is None:
                    mode = "min"

            self.register_callback(lambda: ModelCheckpoint(monitor=monitor, mode=mode))

        if (early_stopping := self.config.early_stopping) is not None:
            if (monitor := early_stopping.monitor) is None:
                monitor, mode = self.primary_metric()
            else:
                if (mode := early_stopping.mode) is None:
                    mode = "min"

            self.register_callback(
                lambda: EarlyStoppingWithMinLR(
                    monitor=monitor,
                    mode=mode,
                    patience=early_stopping.patience,
                    min_delta=early_stopping.min_delta,
                    min_lr=early_stopping.min_lr,
                    strict=early_stopping.strict,
                )
            )

        for cls_target in self.config.graph_classification_targets:
            match cls_target:
                case MulticlassClassificationTargetConfig(
                    class_weights=class_weights
                ) if class_weights:
                    self.register_buffer(
                        f"{cls_target.name}_class_weights",
                        torch.tensor(class_weights, dtype=torch.float),
                        persistent=False,
                    )
                case _:
                    pass

    def construct_graph_scalar_output_head(self, target: str) -> nn.Module:
        return GraphScalarOutputHead(
            self.config,
            reduction=self.config.graph_scalar_reduction.get(
                target, self.config.graph_scalar_reduction_default
            ),
        )

    def construct_graph_classification_output_head(
        self,
        target: BinaryClassificationTargetConfig | MulticlassClassificationTargetConfig,
    ) -> nn.Module:
        match target:
            case BinaryClassificationTargetConfig():
                return GraphBinaryClassificationOutputHead(
                    self.config,
                    target,
                    reduction=self.config.graph_classification_reduction.get(
                        target.name, self.config.graph_classification_reduction_default
                    ),
                )
            case MulticlassClassificationTargetConfig():
                return GraphMulticlassClassificationOutputHead(
                    self.config,
                    target,
                    reduction=self.config.graph_classification_reduction.get(
                        target.name, self.config.graph_classification_reduction_default
                    ),
                )
            case _:
                raise ValueError(f"Invalid target: {target}")

    def construct_node_vector_output_head(self, target: str) -> nn.Module:
        return NodeVectorOutputHead(
            self.config,
            reduction=self.config.node_vector_reduction.get(
                target, self.config.node_vector_reduction_default
            ),
        )

    def construct_output_heads(self):
        self.graph_outputs = TypedModuleDict(
            {
                target: self.construct_graph_scalar_output_head(target)
                for target in self.config.graph_scalar_targets
            },
            key_prefix="ft_mlp_",
        )
        self.graph_classification_outputs = TypedModuleDict(
            {
                target.name: self.construct_graph_classification_output_head(target)
                for target in self.config.graph_classification_targets
            },
            key_prefix="ft_mlp_",
        )
        self.node_outputs = TypedModuleDict(
            {
                target: self.construct_node_vector_output_head(target)
                for target in self.config.node_vector_targets
            },
            key_prefix="ft_mlp_",
        )

    def load_backbone_state_dict(
        self,
        *,
        backbone: Mapping[str, Any],
        embedding: Mapping[str, Any],
        strict: bool = True,
    ):
        ignored_key_patterns = self.config.ckpt_load.ignored_key_patterns
        # If we're dumping the backbone's force out heads, then we need to ignore
        #   the unexpected keys for the force out MLPs and force out heads.
        if (
            not self.config.backbone.regress_forces
            or not self.config.backbone.direct_forces
        ):
            ignored_key_patterns.append("out_mlp_F.*")
            for block_idx in range(self.config.backbone.num_blocks + 1):
                ignored_key_patterns.append(f"out_blocks.{block_idx}.scale_rbf_F.*")
                ignored_key_patterns.append(f"out_blocks.{block_idx}.dense_rbf_F.*")
                ignored_key_patterns.append(f"out_blocks.{block_idx}.seq_forces.*")

        load_state_dict(
            self.backbone,
            backbone,
            strict=strict,
            ignored_key_patterns=ignored_key_patterns,
            ignored_missing_keys=self.config.ckpt_load.ignored_missing_keys,
            ignored_unexpected_keys=self.config.ckpt_load.ignored_unexpected_keys,
        )
        if not self.config.ckpt_load.reset_embeddings:
            load_state_dict(self.embedding, embedding, strict=strict)
        log.critical("Loaded backbone state dict (backbone and embedding).")

    def generate_graphs(
        self,
        data: BaseData,
        cutoffs: Cutoffs,
        max_neighbors: MaxNeighbors,
        pbc: bool,
    ):
        aint_graph = generate_graph(
            data, cutoff=cutoffs.aint, max_neighbors=max_neighbors.aint, pbc=pbc
        )
        aint_graph = self.process_aint_graph(aint_graph)
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

        # We can't do this at the data level: This is because the batch collate_fn doesn't know
        # that it needs to increment the "id_swap" indices as it collates the data.
        # So we do this at the graph level (which is done in the GemNetOC `get_graphs_and_indices` method).
        # main_graph = symmetrize_edges(main_graph, num_atoms=data.pos.shape[0])
        qint_graph = tag_mask(data, qint_graph, tags=self.config.backbone.qint_tags)

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

    def collate_fn(self, data_list: list[BaseData]):
        return Batch.from_data_list(data_list)
