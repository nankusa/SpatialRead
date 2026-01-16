from pydantic import BaseModel, ConfigDict
from pydantic import PrivateAttr as PrivateAttr
from collections.abc import Mapping, MutableMapping
from typing import Any, Optional, ClassVar, TYPE_CHECKING, Literal, cast
from typing_extensions import override
from dataclasses import dataclass, field
from box import Box
import fnmatch
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
)
from torch import Tensor
import torch.optim as optim
from torch.optim import Optimizer

from spatialread.modules.lightning.finetune import FinetuneConfigBase, FinetuneModelBase
from spatialread.modules.lightning.task_config import AdamWConfig
from spatialread.modules.lightning.finetune import (
    RLPConfig,
    WarmupCosRLPConfig,
)
from spatialread.modules.lightning.param_specific_util import (
    make_parameter_specific_optimizer_config,
)

from spatialread.config import get_optimize_config


def set_scheduler(pl_module):
    optimize_config = get_optimize_config()
    if optimize_config["type"] == "transformer":
        return set_scheduler_transformer(pl_module)
    elif optimize_config["type"] == "jmp":
        return set_scheduler_jmp(pl_module)
    elif optimize_config["type"] == "reduce_on_plateau":
        return set_scheduler_reduce_on_plateau(pl_module)
    else:
        raise ValueError(f'Unsupported optimize type {optimize_config["type"]}')


def set_scheduler_jmp(pl_module):
    config = FinetuneConfigBase.draft()

    config.optimizer = AdamWConfig(
        lr=8.0e-5,
        amsgrad=False,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.1,
    )
    config.lr_scheduler = WarmupCosRLPConfig(
        warmup_epochs=5,
        warmup_start_lr_factor=1.0e-1,
        should_restart=False,
        max_epochs=32,
        min_lr_factor=0.1,
        rlp=RLPConfig(mode="min", patience=5, factor=0.1),
    )

    # config.parameter_specific_optimizers = make_parameter_specific_optimizer_config(
    #     config,
    #     6,
    #     {
    #         "embedding": 0.3,

    #         "blocks_0": 0.55,
    #         "blocks_1": 0.40,
    #         "blocks_2": 0.30,
    #         "blocks_3": 0.40,
    #         "blocks_4": 0.55,
    #         "blocks_5": 0.625,
    #         "transformer": 1.0,
    #     },
    # )
    config.parameter_specific_optimizers = make_parameter_specific_optimizer_config(
        config,
        4,
        {
            "embedding": 0.3,
            "blocks_0": 0.35,
            "blocks_1": 0.40,
            "blocks_2": 0.55,
            "blocks_3": 0.625,
            "transformer": 1.25,
            # "blocks_4": 0.55,
            # "blocks_5": 0.625,
        },
    )

    config = config.finalize()

    # a hack method and need to be improved
    pl_module.config = config
    return pl_module.configure_optimizers_jmp()


def set_scheduler_transformer(pl_module):
    optim_config = get_optimize_config()["transformer"]
    lr = optim_config["lr"]
    wd = optim_config["weight_decay"]

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    backbone_names = ["head"]
    end_lr = optim_config["end_lr"]
    decay_power = optim_config["decay_power"]
    optim_type = optim_config["optim_type"]
    warmup_steps = optim_config["warmup_steps"]
    max_steps = optim_config["max_steps"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in pl_module.named_parameters()],
            "weight_decay": wd,
            "lr": lr,
        },
    ]

    if optim_type == "adamw":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.95)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)
    else:
        raise NotImplementedError

    if max_steps != -1:
        if pl_module.trainer.max_steps == -1:
            max_steps = pl_module.trainer.estimated_stepping_batches
        else:
            max_steps = pl_module.trainer.max_steps

    if isinstance(warmup_steps, float):
        warmup_steps = int(max_steps * warmup_steps)

    print(
        f"max_epochs: {pl_module.trainer.max_epochs} | max_steps: {max_steps} | warmup_steps : {warmup_steps} "
        f"| weight_decay : {wd} | decay_power : {decay_power}"
    )

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    elif decay_power == "constant":
        scheduler = get_constant_schedule(
            optimizer,
        )
    elif decay_power == "constant_with_warmup":
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    sched = {"scheduler": scheduler, "interval": "step"}

    return (
        [optimizer],
        [sched],
    )


def set_scheduler_reduce_on_plateau(pl_module):
    optim_config = get_optimize_config()["reduce_on_plateau"]
    lr = optim_config["lr"]
    wd = optim_config["weight_decay"]

    # 配置参数
    end_lr = optim_config.get("end_lr", 1e-6)
    optim_type = optim_config["optim_type"]
    monitor_metric = optim_config.get("lr_monitor", "val_loss")
    monitor_mode = optim_config.get("monitor_mode", "min")
    patience = optim_config.get("lr_patience", 10)
    factor = optim_config.get("lr_factor", 0.1)

    # 创建优化器
    params = pl_module.parameters()
    if optim_type == "adamw":
        optimizer = torch.optim.AdamW(params, lr=lr, eps=1e-8, betas=(0.9, 0.95))
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(params, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: {optim_type}")

    # 准备返回结构
    return_config = {
        "optimizer": optimizer,
    }

    # 没有warmup的情况
    return_config["lr_scheduler"] = {
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=monitor_mode,
            factor=factor,
            patience=patience,
            threshold=1e-4,
            min_lr=end_lr,
            verbose=True,
        ),
        "monitor": monitor_metric,
        "interval": "epoch",
        "frequency": 1,
        "name": "reduce_on_plateau",
    }

    return return_config
