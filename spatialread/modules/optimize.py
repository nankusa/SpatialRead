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

from .scheduler.gradual_warmup_lr import GradualWarmupScheduler
from .scheduler.linear_warmup_cos_rlp import (
    PerParamGroupLinearWarmupCosineAnnealingRLPLR,
)
from ._config.missing import MISSING, validate_no_missing_values
from ._config.missing import AllowMissing as AllowMissing
from ._config.missing import MissingField as MissingField

from spatialread.config import get_optimize_config


_MutableMappingBase = MutableMapping[str, Any]

_DraftConfigContextSentinel = object()


class TypedConfig(BaseModel, _MutableMappingBase):
    _is_draft_config: bool = PrivateAttr(default=False)
    """
    Whether this config is a draft config or not.

    Draft configs are configs that are not yet fully validated.
    They allow for a nicer API when creating configs, e.g.:

        ```python
        config = MyConfig.draft()

        # Set some values
        config.a = 10
        config.b = "hello"

        # Finalize the config
        config = config.finalize()
        ```
    """

    repr_diff_only: ClassVar[bool] = True
    """
    If `True`, the repr methods will only show values for fields that are different from the default.
    """

    MISSING: ClassVar[Any] = MISSING
    """
    Alias for the `MISSING` constant.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        # By default, Pydantic will throw a warning if a field starts with "model_",
        # so we need to disable that warning (beacuse "model_" is a popular prefix for ML).
        protected_namespaces=(),
        validate_assignment=True,
        strict=True,
        revalidate_instances="always",
        arbitrary_types_allowed=True,
        extra="ignore",
    )

    def __draft_pre_init__(self):
        """Called right before a draft config is finalized."""
        pass

    def __post_init__(self):
        """Called after the final config is validated."""
        pass

    @classmethod
    def from_dict(cls, model_dict: Mapping[str, Any]):
        return cls.model_validate(model_dict)

    def model_deep_validate(self, strict: bool = True):
        """
        Validate the config and all of its sub-configs.

        Args:
            config: The config to validate.
            strict: Whether to validate the config strictly.
        """
        config_dict = self.model_dump(round_trip=True)
        config = self.model_validate(config_dict, strict=strict)

        # Make sure that this is not a draft config
        if config._is_draft_config:
            raise ValueError("Draft configs are not valid. Call `finalize` first.")

        return config

    @classmethod
    def draft(cls, **kwargs):
        config = cls.model_construct_draft(**kwargs)
        return config

    def finalize(self, strict: bool = True):
        # This must be a draft config, otherwise we raise an error
        if not self._is_draft_config:
            raise ValueError("Finalize can only be called on drafts.")

        # First, we call `__draft_pre_init__` to allow the config to modify itself a final time
        self.__draft_pre_init__()

        # Then, we dump the config to a dict and then re-validate it
        return self.model_deep_validate(strict=strict)

    @override
    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)

        # Call the `__post_init__` method if this is not a draft config
        if __context is _DraftConfigContextSentinel:
            return

        self.__post_init__()

        # After `_post_init__` is called, we perform the final round of validation
        self.model_post_init_validate()

    def model_post_init_validate(self):
        validate_no_missing_values(self)

    @classmethod
    def model_construct_draft(cls, _fields_set: set[str] | None = None, **values: Any):
        """
        NOTE: This is a copy of the `model_construct` method from Pydantic's `Model` class,
            with the following changes:
            - The `model_post_init` method is called with the `_DraftConfigContext` context.
            - The `_is_draft_config` attribute is set to `True` in the `values` dict.

        Creates a new instance of the `Model` class with validated data.

        Creates a new model setting `__dict__` and `__pydantic_fields_set__` from trusted or pre-validated data.
        Default values are respected, but no other validation is performed.

        !!! note
            `model_construct()` generally respects the `model_config.extra` setting on the provided model.
            That is, if `model_config.extra == 'allow'`, then all extra passed values are added to the model instance's `__dict__`
            and `__pydantic_extra__` fields. If `model_config.extra == 'ignore'` (the default), then all extra passed values are ignored.
            Because no validation is performed with a call to `model_construct()`, having `model_config.extra == 'forbid'` does not result in
            an error if extra values are passed, but they will be ignored.

        Args:
            _fields_set: The set of field names accepted for the Model instance.
            values: Trusted or pre-validated data dictionary.

        Returns:
            A new instance of the `Model` class with validated data.
        """

        values["_is_draft_config"] = True

        m = cls.__new__(cls)
        fields_values: dict[str, Any] = {}
        fields_set = set()

        for name, field in cls.model_fields.items():
            if field.alias and field.alias in values:
                fields_values[name] = values.pop(field.alias)
                fields_set.add(name)
            elif name in values:
                fields_values[name] = values.pop(name)
                fields_set.add(name)
            elif not field.is_required():
                fields_values[name] = field.get_default(call_default_factory=True)
        if _fields_set is None:
            _fields_set = fields_set

        _extra: dict[str, Any] | None = None
        if cls.model_config.get("extra") == "allow":
            _extra = {}
            for k, v in values.items():
                _extra[k] = v
        object.__setattr__(m, "__dict__", fields_values)
        object.__setattr__(m, "__pydantic_fields_set__", _fields_set)
        if not cls.__pydantic_root_model__:
            object.__setattr__(m, "__pydantic_extra__", _extra)

        if cls.__pydantic_post_init__:
            m.model_post_init(_DraftConfigContextSentinel)
            # update private attributes with values set
            if (
                hasattr(m, "__pydantic_private__")
                and m.__pydantic_private__ is not None
            ):
                for k, v in values.items():
                    if k in m.__private_attributes__:
                        m.__pydantic_private__[k] = v

        elif not cls.__pydantic_root_model__:
            # Note: if there are any private attributes, cls.__pydantic_post_init__ would exist
            # Since it doesn't, that means that `__pydantic_private__` should be set to None
            object.__setattr__(m, "__pydantic_private__", None)

        return m

    @override
    def __repr_args__(self):
        # If `repr_diff_only` is `True`, we only show the fields that are different from the default.
        if not self.repr_diff_only:
            yield from super().__repr_args__()
            return

        # First, we get the default values for all fields.
        default_values = self.model_construct_draft()

        # Then, we compare the default values with the current values.
        for k, v in super().__repr_args__():
            if k is None:
                yield k, v
                continue

            # If there is no default value or the value is different from the default, we yield it.
            if not hasattr(default_values, k) or getattr(default_values, k) != v:
                yield k, v
                continue

            # Otherwise, we can skip this field.

    # region MutableMapping implementation
    if not TYPE_CHECKING:
        # This is mainly so the config can be used with lightning's hparams
        #   transparently and without any issues.

        @property
        def _ll_dict(self):
            return self.model_dump()

        # We need to make sure every config class
        #   is a MutableMapping[str, Any] so that it can be used
        #   with lightning's hparams.
        @override
        def __getitem__(self, key: str):
            # Key can be of the format "a.b.c"
            #   so we need to split it into a list of keys.
            [first_key, *rest_keys] = key.split(".")
            value = self._ll_dict[first_key]

            for key in rest_keys:
                if isinstance(value, Mapping):
                    value = value[key]
                else:
                    value = getattr(value, key)

            return value

        @override
        def __setitem__(self, key: str, value: Any):
            # Key can be of the format "a.b.c"
            #   so we need to split it into a list of keys.
            [first_key, *rest_keys] = key.split(".")
            if len(rest_keys) == 0:
                self._ll_dict[first_key] = value
                return

            # We need to traverse the keys until we reach the last key
            #   and then set the value
            current_value = self._ll_dict[first_key]
            for key in rest_keys[:-1]:
                if isinstance(current_value, Mapping):
                    current_value = current_value[key]
                else:
                    current_value = getattr(current_value, key)

            # Set the value
            if isinstance(current_value, MutableMapping):
                current_value[rest_keys[-1]] = value
            else:
                setattr(current_value, rest_keys[-1], value)

        @override
        def __delitem__(self, key: str):
            # This is unsupported for this class
            raise NotImplementedError

        @override
        def __iter__(self):
            return iter(self._ll_dict)

        @override
        def __len__(self):
            return len(self._ll_dict)

        # endregion


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


@dataclass(frozen=True)
class _OptimizerParamGroupConfig:
    cls: type[optim.Optimizer]
    param_group_kwargs: dict[str, Any] = field(default_factory=lambda: {})
    optimizer_kwargs: dict[str, Any] = field(default_factory=lambda: {})


def named_parameters_matching_patterns(pl_module, patterns: list[str]):
    for name, param in pl_module.named_parameters():
        # # ignored_parameters are to ensure all parameters requires grad
        # if param in pl_module.ignored_parameters:
        #     continue
        if (
            matching_pattern := next(
                (pattern for pattern in patterns if fnmatch.fnmatch(name, pattern)),
                None,
            )
        ) is None:
            continue

        yield name, param, matching_pattern


def split_parameters(pl_module, pattern_lists):
    all_parameters = list(pl_module.parameters())

    parameters = []
    for patterns in pattern_lists:
        # NOTE
        patterns = ["model." + p for p in patterns]
        matching, names = [], []
        matching = [
            p for _, p, _ in named_parameters_matching_patterns(pl_module, patterns)
        ]
        names = [
            n for n, _, _ in named_parameters_matching_patterns(pl_module, patterns)
        ]
        print(f"OPTIMIZE PATTERN MATCHING: {patterns}, {names}")
        parameters.append(matching)
        # remove matching parameters from all_parameters
        all_parameters = [
            p for p in all_parameters if all(p is not m for m in matching)
        ]

    return parameters, all_parameters


def _create_dict_from_config(
    config,
    params,
    name=None,
):
    from torch.optim import AdamW

    AdamWKwargs = AdamW

    if not TYPE_CHECKING:
        AdamWKwargs = dict

    if config.lr <= 0:
        raise ValueError(f"Learning rate must be positive, got {config.lr}")

    kwargs = cast(
        dict,
        AdamWKwargs(
            params=params,
            lr=config.lr,
            amsgrad=config.amsgrad,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps,
        ),
    )

    # if name is not None:
    #     kwargs["name"] = name
    return _OptimizerParamGroupConfig(AdamW, param_group_kwargs=kwargs)


def optimizer_from_config(
    param_groups,
    *,
    base=None,
):
    configs = [
        _create_dict_from_config(
            param_group[0],
            param_group[1],
            name=param_group[2] if len(param_group) == 3 else None,
        )
        for param_group in param_groups
    ]
    optimizer_cls_list = [c.cls for c in configs]
    assert len(set(optimizer_cls_list)) == 1, "All optimizers must be of the same type"
    optimizer_cls = optimizer_cls_list[0]

    optimizer_kwargs_list = [c.optimizer_kwargs for c in configs]
    assert (
        len(set(map(str, optimizer_kwargs_list))) == 1
    ), "All optimizers must have the same kwargs"
    optimizer_kwargs = optimizer_kwargs_list[0]

    base_kwargs = {}
    if base is not None:
        base_config = _create_dict_from_config(base, [])
        assert (
            base_config.cls == optimizer_cls
        ), "Base optimizer must be of the same type"
        _ = base_config.param_group_kwargs.pop("params", None)
        base_kwargs.update(base_config.param_group_kwargs)

    param_groups_configs = [c.param_group_kwargs for c in configs]
    optimizer = optimizer_cls(
        params=param_groups_configs,
        **optimizer_kwargs,
        **base_kwargs,
    )
    # detailed log about the optimizer configuration
    param_groups_logs: list[str] = []
    for i, c in enumerate(param_groups_configs):
        c = copy.deepcopy(c)
        params = c.pop("params", None)
        n_params = len(params) if params is not None else 0
        total_param_size = sum(p.numel() for p in params) if params is not None else 0
    return optimizer


def _cos_annealing_hparams(
    pl_module, lr_config: WarmupCosRLPConfig, *, lr_initial: float
):
    if (warmup_steps := lr_config.warmup_steps) is None:
        if warmup_epochs := lr_config.warmup_epochs:
            assert warmup_epochs >= 0, f"Invalid warmup_epochs: {warmup_epochs}"
            _ = (
                pl_module.trainer.estimated_stepping_batches
            )  # make sure dataloaders are loaded for self.trainer.num_training_batches
            num_steps_per_epoch = math.ceil(
                pl_module.trainer.num_training_batches
                / pl_module.trainer.accumulate_grad_batches
            )
            warmup_steps = warmup_epochs * num_steps_per_epoch
        else:
            warmup_steps = 0

    if not (max_steps := lr_config.max_steps):
        if max_epochs := lr_config.max_epochs:
            _ = (
                pl_module.trainer.estimated_stepping_batches
            )  # make sure dataloaders are loaded for self.trainer.num_training_batches
            num_steps_per_epoch = math.ceil(
                pl_module.trainer.num_training_batches
                / pl_module.trainer.accumulate_grad_batches
            )
            max_steps = max_epochs * num_steps_per_epoch
        else:
            max_steps = pl_module.trainer.estimated_stepping_batches
            assert math.isfinite(max_steps), f"{max_steps=} is not finite"
            max_steps = int(max_steps)

    assert (
        lr_config.min_lr_factor > 0 and lr_config.min_lr_factor <= 1
    ), f"Invalid {lr_config.min_lr_factor=}"
    min_lr = lr_initial * lr_config.min_lr_factor

    assert (
        lr_config.warmup_start_lr_factor > 0 and lr_config.warmup_start_lr_factor <= 1
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
    pl_module, optimizer: torch.optim.Optimizer, config: RLPConfig
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


def set_scheduler(pl_module):
    optimize_config = get_optimize_config()
    if optimize_config["type"] == "transformer":
        return set_scheduler_transformer(pl_module)
    elif optimize_config["type"] == "jmp":
        return set_scheduler_jmp(pl_module)
    elif optimize_config['type'] == 'reduce_on_plateau':
        return set_scheduler_reduce_on_plateau(pl_module)
    else:
        raise ValueError(f'Unsupported optimize type {optim_config["type"]}')


def set_scheduler_jmp(pl_module):
    config = get_optimize_config()['jmp']
    config = Box(config)

    # configs = config['parameter_specific_optimizers']
    configs = config.parameter_specific_optimizers
    params_list, rest_params = split_parameters(
        pl_module, [c.paremeter_patterns for c in configs]
    )
    optimizer = optimizer_from_config(
        [
            *(
                (
                    config.optimizer if c.optimizer is None else c.optimizer,
                    params,
                    c.name or ",".join(c.paremeter_patterns),
                )
                for c, params in zip(configs, params_list)
            ),
            (config.optimizer, rest_params, "rest"),
        ],
        base=config.optimizer,
    )

    out: dict[str, Any] = {
        "optimizer": optimizer,
    }
    if (lr_config := config.lr_scheduler) is None:
        return out

    lr_config_dict = lr_config.to_dict()

    lr_config_dict["rlp"] = RLPConfig.from_dict(lr_config.rlp.to_dict())

    lr_config = WarmupCosRLPConfig.from_dict(lr_config_dict)

    match lr_config:
        case RLPConfig():
            assert all(
                c.lr_scheduler is None for c in configs
            ), f"lr_scheduler is not None for some configs: {configs=}"

            if (
                lr_scheduler := _construct_lr_scheduler(pl_module, optimizer, lr_config)
            ) is not None:
                out["lr_scheduler"] = lr_scheduler
        case WarmupCosRLPConfig():
            param_group_lr_scheduler_settings = [
                *(
                    _cos_annealing_hparams(
                        pl_module,
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
                _cos_annealing_hparams(
                    pl_module, lr_config, lr_initial=optimizer.param_groups[-1]["lr"]
                ),
            ]

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
            raise ValueError(f"Unknown lr_config: {lr_config}")

    return out


def set_scheduler_transformer(pl_module):
    optim_config = get_optimize_config()['transformer']
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
    max_steps = optim_config['max_steps']

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
    optim_config = get_optimize_config()['reduce_on_plateau']
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
        optimizer = torch.optim.AdamW(
            params, 
            lr=lr, 
            eps=1e-8, 
            betas=(0.9, 0.95)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(params, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)

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
            verbose=True
        ),
        "monitor": monitor_metric,
        "interval": "epoch",
        "frequency": 1,
        "name": "reduce_on_plateau",
    }

    return return_config
