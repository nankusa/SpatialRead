"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import fnmatch
import math
from logging import getLogger
from typing import Annotated, Any, Literal, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from .model.config import BaseConfig
from .config import Field, TypedConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data.data import BaseData
from typing_extensions import TypeVar, override

from spatialread.modules.scheduler.gradual_warmup_lr import GradualWarmupScheduler
from spatialread.modules.scheduler.linear_warmup_cos_rlp import (
    PerParamGroupLinearWarmupCosineAnnealingRLPLR,
)
from .task_config import (
    OptimizerConfig,
    optimizer_from_config,
)

log = getLogger(__name__)


class RLPWarmupConfig(TypedConfig):
    steps: int
    """Number of steps for the warmup"""

    start_lr_factor: float
    """The factor to multiply the initial learning rate by at the start of the warmup"""


class RLPConfig(TypedConfig):
    name: Literal["rlp"] = "rlp"

    monitor: str | None = None
    mode: str | None = None
    patience: int = 10
    factor: float = 0.1
    min_lr: float = 0.0
    eps: float = 1.0e-8
    cooldown: int = 0
    threshold: float = 1.0e-4
    threshold_mode: str = "rel"
    interval: str = "epoch"
    frequency: int = 1
    warmup: RLPWarmupConfig | None = None

    def _to_linear_warmup_cos_rlp_dict(self):
        """
        Params for PerParamGroupLinearWarmupCosineAnnealingRLPLR's RLP
            mode="min",
            factor=0.1,
            patience=10,
            threshold=1e-4,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-8,
            verbose=False,
        """
        return {
            "mode": self.mode,
            "factor": self.factor,
            "patience": self.patience,
            "threshold": self.threshold,
            "threshold_mode": self.threshold_mode,
            "cooldown": self.cooldown,
            "min_lr": self.min_lr,
            "eps": self.eps,
            "verbose": True,
        }


class WarmupCosRLPConfig(TypedConfig):
    name: Literal["warmup_cos_rlp"] = "warmup_cos_rlp"

    warmup_steps: int | None = None
    warmup_epochs: int | None = None
    max_steps: int | None = None
    max_epochs: int | None = None
    warmup_start_lr_factor: float = 0.0
    min_lr_factor: float = 1.0e-2
    last_step: int = -1
    should_restart: bool = False

    rlp: RLPConfig

    @override
    def __post_init__(self):
        super().__post_init__()

        assert self.rlp.warmup is None, "RLP warmup is not supported"


LRSchedulerConfig: TypeAlias = Annotated[
    RLPConfig | WarmupCosRLPConfig, Field(discriminator="name")
]


class ParamSpecificOptimizerConfig(TypedConfig):
    name: str | None = None
    """The name of the parameter group for this config"""

    paremeter_patterns: list[str] = []
    """List of parameter patterns to match for this config"""

    optimizer: OptimizerConfig | None = None
    """
    The optimizer config for this parameter group.
    If None, the default optimizer will be used.
    """

    lr_scheduler: LRSchedulerConfig | None = None
    """
    The learning rate scheduler config for this parameter group.
    If None, the default learning rate scheduler will be used.
    """


class FinetuneConfigBase(BaseConfig):
    optimizer: OptimizerConfig
    """Optimizer to use."""
    lr_scheduler: LRSchedulerConfig | None = None
    """Learning rate scheduler configuration. If None, no learning rate scheduler is used."""

    parameter_specific_optimizers: list[ParamSpecificOptimizerConfig] | None = None
    """Configuration for parameter-specific optimizers"""


TConfig = TypeVar("TConfig", bound=FinetuneConfigBase)


# class FinetuneModelBase(LightningModuleBase[TConfig], Generic[TConfig]):
class FinetuneModelBase:
    def named_parameters_matching_patterns(self, patterns: list[str]):
        for name, param in self.named_parameters():
            # if param in self.ignored_parameters:
            #     continue
            if (
                matching_pattern := next(
                    (pattern for pattern in patterns if fnmatch.fnmatch(name, pattern)),
                    None,
                )
            ) is None:
                continue

            yield name, param, matching_pattern

    def _cos_rlp_schedulers(self):
        if (lr_schedulers := self.lr_schedulers()) is None:
            log.warning("No LR scheduler found.")
            return

        if not isinstance(lr_schedulers, list):
            lr_schedulers = [lr_schedulers]

        for scheduler in lr_schedulers:
            if isinstance(scheduler, PerParamGroupLinearWarmupCosineAnnealingRLPLR):
                yield scheduler

    def _on_train_batch_start_cos_rlp(self):
        for scheduler in self._cos_rlp_schedulers():
            scheduler.on_new_step(self.global_step)

    @override
    def on_train_batch_start(self, batch: BaseData, batch_idx: int):
        match self.config.lr_scheduler:
            case WarmupCosRLPConfig():
                self._on_train_batch_start_cos_rlp()
            case _:
                pass

    def _on_validation_epoch_end_cos_rlp(self, config: WarmupCosRLPConfig):
        # rlp_monitor = self._rlp_metric(config.rlp)
        # log.info(f"LR scheduler metrics: {rlp_monitor}")

        # metric_value: torch.Tensor | None = None
        for scheduler in self._cos_rlp_schedulers():
            if scheduler.is_in_rlp_stage(self.global_step):
                if metric_value is None:
                    metric_value = rlp_monitor.compute()

                log.info(f"LR scheduler is in RLP mode. RLP metric: {metric_value}")
                scheduler.rlp_step(metric_value)

    @override
    def on_validation_epoch_end(self):
        match self.config.lr_scheduler:
            case WarmupCosRLPConfig() as config:
                self._on_validation_epoch_end_cos_rlp(config)
            case _:
                pass

    @override
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        match self.config.lr_scheduler:
            case RLPConfig(warmup=RLPWarmupConfig()):
                lr_scheduler = self.lr_schedulers()
                assert isinstance(lr_scheduler, GradualWarmupScheduler)
                if not lr_scheduler.finished:
                    lr_scheduler.step()
            case _:
                pass

        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    def split_parameters(self, pattern_lists: list[list[str]]):
        all_parameters = list(self.parameters())

        parameters: list[list[torch.nn.Parameter]] = []
        for patterns in pattern_lists:
            matching = [
                p for _, p, _ in self.named_parameters_matching_patterns(patterns)
            ]
            parameters.append(matching)
            # remove matching parameters from all_parameters
            all_parameters = [
                p for p in all_parameters if all(p is not m for m in matching)
            ]

        return parameters, all_parameters

    def _cos_annealing_hparams(
        self, lr_config: WarmupCosRLPConfig, *, lr_initial: float
    ):
        if (warmup_steps := lr_config.warmup_steps) is None:
            if warmup_epochs := lr_config.warmup_epochs:
                assert warmup_epochs >= 0, f"Invalid warmup_epochs: {warmup_epochs}"
                _ = (
                    self.trainer.estimated_stepping_batches
                )  # make sure dataloaders are loaded for self.trainer.num_training_batches
                num_steps_per_epoch = math.ceil(
                    self.trainer.num_training_batches
                    / self.trainer.accumulate_grad_batches
                )
                warmup_steps = warmup_epochs * num_steps_per_epoch
            else:
                warmup_steps = 0
        log.critical(f"Computed warmup_steps: {warmup_steps}")

        if not (max_steps := lr_config.max_steps):
            if max_epochs := lr_config.max_epochs:
                _ = (
                    self.trainer.estimated_stepping_batches
                )  # make sure dataloaders are loaded for self.trainer.num_training_batches
                num_steps_per_epoch = math.ceil(
                    self.trainer.num_training_batches
                    / self.trainer.accumulate_grad_batches
                )
                max_steps = max_epochs * num_steps_per_epoch
            else:
                max_steps = self.trainer.estimated_stepping_batches
                assert math.isfinite(max_steps), f"{max_steps=} is not finite"
                max_steps = int(max_steps)

        log.critical(f"Computed max_steps: {max_steps}")

        assert (
            lr_config.min_lr_factor > 0 and lr_config.min_lr_factor <= 1
        ), f"Invalid {lr_config.min_lr_factor=}"
        min_lr = lr_initial * lr_config.min_lr_factor

        assert (
            lr_config.warmup_start_lr_factor > 0
            and lr_config.warmup_start_lr_factor <= 1
        ), f"Invalid {lr_config.warmup_start_lr_factor=}"
        warmup_start_lr = lr_initial * lr_config.warmup_start_lr_factor

        lr_scheduler_hparams = dict(
            warmup_epochs=warmup_steps,
            max_epochs=max_steps,
            warmup_start_lr=warmup_start_lr,
            eta_min=min_lr,
            should_restart=lr_config.should_restart,
        )

        return lr_scheduler_hparams

    def _construct_lr_scheduler(
        self, optimizer: torch.optim.Optimizer, config: RLPConfig
    ):
        assert config.monitor is not None, f"{config=}"
        assert config.mode is not None, f"{config=}"

        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode=config.mode,
            factor=config.factor,
            threshold=config.threshold,
            threshold_mode=config.threshold_mode,
            patience=config.patience,
            cooldown=config.cooldown,
            min_lr=config.min_lr,
            eps=config.eps,
            verbose=True,
        )
        if config.warmup is not None:
            optim_lr = float(optimizer.param_groups[0]["lr"])
            warmup_start_lr = optim_lr * config.warmup.start_lr_factor

            lr_scheduler = GradualWarmupScheduler(
                optimizer,
                warmup_start_lr=warmup_start_lr,
                warmup_steps=config.warmup.steps,
                after_scheduler=lr_scheduler,
            )
            return {
                "scheduler": lr_scheduler,
                "monitor": config.monitor,
                "interval": config.interval,
                "frequency": config.frequency,
                "strict": False,
                "reduce_on_plateau": True,
            }
        else:
            return {
                "scheduler": lr_scheduler,
                "monitor": config.monitor,
                "interval": config.interval,
                "frequency": config.frequency,
                "strict": True,
            }

    def configure_optimizers_param_specific_optimizers(
        self, configs: list[ParamSpecificOptimizerConfig]
    ):
        params_list, rest_params = self.split_parameters(
            [c.paremeter_patterns for c in configs]
        )
        optimizer = optimizer_from_config(
            [
                *(
                    (
                        self.config.optimizer if c.optimizer is None else c.optimizer,
                        params,
                        c.name or ",".join(c.paremeter_patterns),
                    )
                    for c, params in zip(configs, params_list)
                ),
                (self.config.optimizer, rest_params, "rest"),
            ],
            base=self.config.optimizer,
        )

        out: dict[str, Any] = {
            "optimizer": optimizer,
        }
        if (lr_config := self.config.lr_scheduler) is None:
            return out

        match lr_config:
            case RLPConfig():
                assert all(
                    c.lr_scheduler is None for c in configs
                ), f"lr_scheduler is not None for some configs: {configs=}"

                if (
                    lr_scheduler := self._construct_lr_scheduler(optimizer, lr_config)
                ) is not None:
                    out["lr_scheduler"] = lr_scheduler
            case WarmupCosRLPConfig():
                param_group_lr_scheduler_settings = [
                    *(
                        self._cos_annealing_hparams(
                            (
                                lr_config
                                if c.lr_scheduler is None
                                or not isinstance(c.lr_scheduler, WarmupCosRLPConfig)
                                else c.lr_scheduler
                            ),
                            lr_initial=param_group["lr"],
                        )
                        for c, param_group in zip(configs, optimizer.param_groups[:-1])
                    ),
                    self._cos_annealing_hparams(
                        lr_config, lr_initial=optimizer.param_groups[-1]["lr"]
                    ),
                ]

                log.critical(f"{param_group_lr_scheduler_settings=}")
                lr_scheduler = PerParamGroupLinearWarmupCosineAnnealingRLPLR(
                    optimizer,
                    param_group_lr_scheduler_settings,
                    lr_config.rlp._to_linear_warmup_cos_rlp_dict(),
                    max_epochs=next(
                        (s["max_epochs"] for s in param_group_lr_scheduler_settings)
                    ),
                )
                out["lr_scheduler"] = {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            case _:
                # assert_never(lr_config)
                pass

        return out

    @override
    def configure_optimizers_jmp(self):
        print("DEBUG JMP OPTIMIZE", self.config)

        if self.config.parameter_specific_optimizers is not None:
            return self.configure_optimizers_param_specific_optimizers(
                self.config.parameter_specific_optimizers
            )

        optimizer = optimizer_from_config(
            [(self.config.optimizer, self.parameters())],
        )

        out: dict[str, Any] = {
            "optimizer": optimizer,
        }
        if (lr_config := self.config.lr_scheduler) is None:
            return out

        assert isinstance(
            lr_config, RLPConfig
        ), "Only RLPConfig is supported if `parameter_specific_optimizers` is None"
        if (
            lr_scheduler := self._construct_lr_scheduler(optimizer, lr_config)
        ) is not None:
            out["lr_scheduler"] = lr_scheduler

        return out
