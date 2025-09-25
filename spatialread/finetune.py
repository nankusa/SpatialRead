import click
from pathlib import Path

from typing import Dict, Any, Optional, List, Literal

from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
import numpy as np
import pandas as pd

import torch
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torch_geometric
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
import lightning as L
from lightning.pytorch import Trainer, loggers
from lightning.pytorch.utilities.types import STEP_OUTPUT

from lightning.pytorch.callbacks import DeviceStatsMonitor, ModelCheckpoint

from spatialread.config import init_config, get_train_config, get_model_config, get_data_config, get_config
from spatialread.data.datamodule import FinetuneDatamodule
from spatialread.modules.model import construct_model
from spatialread.modules.nn import MLP
from spatialread.modules.optimize import set_scheduler
from spatialread.utils.log import Log
from spatialread.utils.metric import metric_regression, metric_classification, metric_binary_classification
from torch_scatter import scatter_mean


class SpatialReadLightningModule(L.LightningModule):
    def __init__(self, config: str, is_predict=False):
        super().__init__()

        self.is_predict = is_predict
        init_config(config, is_predict)
        self.data_config = get_data_config()
        self.model_config = get_model_config()
        self.train_config = get_train_config()

        # Initialize model flags
        self._init_params()

        self.model = construct_model()
        # self.schnet = SchNet()
        self._init_head()
        self._init_loss()

        # Save hyperparameters
        if not is_predict:
            self.save_hyperparameters(get_config())

    def _init_params(self) -> None:
        self.task_name = self.train_config["task_name"]
        self.task_type = self.train_config['task_type']
        self.cls_num = self.train_config['cls_num']

    def _init_loss(self) -> None:
        """Initialize loss functions."""
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def _init_head(self) -> None:
        mconfig = self.model_config
        if self.task_type == 'Classification':
            out_dim = self.cls_num
        elif self.task_type == 'Regression' or self.task_type == 'BinaryClassification':
            out_dim = 1
        else:
            raise ValueError(f'Unsupported task type {self.task_type}')

        if mconfig['head'] == 'transformer':
            dim = mconfig['transformer']['hid_dim']
            self.head = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Tanh(),
                nn.Linear(dim, out_dim)
            )
        elif mconfig['head'] == 'atom_mlp' or mconfig['head'] == 'spnode_mlp':
            dim = mconfig['head_mlp']['hid_dim']
            readout = mconfig['head_mlp']['readout']
            self.head = nn.Linear(dim, out_dim)
            self.readout = aggr_resolver(readout)

    def _compute_loss(
        self,
        feature: Dict[str, Any],
        output: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
        is_validation: bool=False
    ) -> torch.Tensor:
        loss = 0.0
        metric_dict = {}

        # if not is_validation:
        #     noise = feature['noise']
        #     force = feature['force']
        #     denoise_loss = self.mse_loss(-noise, force)
        #     metric_dict[f"denoise/loss"] = denoise_loss
        #     loss += denoise_loss * self.train_config['loss']['noise']['weight']
        #     denoise_metric = metric_regression(-noise, force)
        #     for k, v in denoise_metric.items():
        #         metric_dict[f"denoise/{k}"] = v

        if self.task_type == 'Regression':
            task_loss = self.mse_loss(
                output[self.task_name].reshape(-1), batch[self.task_name].reshape(-1)
            )
            task_metric = metric_regression(
                batch[self.task_name].reshape(-1), output[self.task_name].reshape(-1)
            )
            
            mean, std = self.trainer.datamodule._mean[self.task_name], self.trainer.datamodule._std[self.task_name]
            task_metric_denorm = metric_regression(
                batch[self.task_name].reshape(-1), output[self.task_name].reshape(-1), mean=mean, std=std
            )
            for k, v in task_metric_denorm.items():
                metric_dict[f"{self.task_name}_denorm/{k}"] = v
        elif self.task_type == 'BinaryClassification':
            task_loss = self.bce_loss(
                output[self.task_name].reshape(-1), batch[self.task_name].reshape(-1)
            )
            task_metric = metric_binary_classification(
                batch[self.task_name].reshape(-1), output[self.task_name].reshape(-1), need_sigmoid=True
            )
        elif self.task_type == 'Classification':
            task_loss = self.cross_entropy(
                output[self.task_name].reshape(-1, self.cls_num), batch[self.task_name].reshape(-1).long()
            )
            task_metric = metric_classification(
                batch[self.task_name].reshape(-1), output[self.task_name].reshape(-1, self.cls_num)
            )
        else:
            raise ValueError(f'Unsupported task type {self.task_type}')

        metric_dict[f"{self.task_name}/loss"] = task_loss
        for k, v in task_metric.items():
            metric_dict[f"{self.task_name}/{k}"] = v
        loss += task_loss

        return metric_dict, loss


    def forward(self, batch: Dict[str, Any], is_validation: bool = False) -> List[Dict[str, torch.Tensor]]:
        feature, output = {}, {}


        feature = self.model(batch)

        if self.model_config['head'] == 'transformer':
            cls_feat = feature['cls_feat']
            output[self.task_name] = self.head(cls_feat) # bs, 1
        elif self.model_config['head'] == 'atom_mlp':
            out = self.head(feature['atom_feat'])
            output[self.task_name] = self.readout(out, feature['atom_batch_idx'])
        elif self.model_config['head'] == 'spnode_mlp':
            out = self.head(feature['spnode_feat'])
            output[self.task_name] = self.readout(out, feature['spnode_batch_idx'])
            # print("DEBUG", out.mean(), out.std(), output[self.task_name])

        if self.task_type == 'Regression' or self.task_type == 'BinaryClassification':
            output[self.task_name] = output[self.task_name].squeeze(-1)

        return feature, output

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        # memory_allocated = torch.cuda.max_memory_allocated('cuda:0') / 1024**2
        # # print(torch.cuda.memory_summary())
        # self.log("memory_allocated_MB", memory_allocated, on_epoch=True, batch_size=len(batch['matid']))

        feature, output = self(batch, is_validation=False)
        metric_dict, loss = self._compute_loss(feature, output, batch)

        # Log individual losses
        self.log_dict(
            {f"train_{k}": v for k, v in metric_dict.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=len(batch['matid'])
        )

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch['matid'])
        )
        
        lrs = self.lr_schedulers().get_last_lr()
        lr_avg = sum(lrs)
        lr_avg /= len(lrs)
        self.log("lr", lr_avg)

        # # 记录当前学习率（从优化器中获取）
        # lr = self.optimizers().param_groups[0]["lr"]
        # # print(list(self.optimizers().param_groups[0].keys()))
        # # for name, param in self.named_parameters():
        # #     if "torchmdnet" in name:
        # #         for group in self.trainer.optimizers[0].param_groups:
        # #             if param in group["params"]:
        # #                 self.log("lr_torchmdnet", group["lr"], prog_bar=True, on_step=True, on_epoch=False, batch_size=len(batch['matid']))
        # #                 break
        # #         break  # 只找第一个就退出

        # self.log("lr", lr, prog_bar=True, logger=True, batch_size=len(batch['matid']))

        return loss

    def on_validation_epoch_start(self) -> None:
        self.best_val_metric = 1e5
        self.validation_output = []

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Optional[STEP_OUTPUT]:
        """Validation step with logging."""
        # with torch.enable_grad():
        feature, output = self(batch, is_validation=True)
        metric_dict, loss = self._compute_loss(feature, output, batch, is_validation=True)
        # feature, output = self(batch)
        # metric_dict, loss = self._compute_loss(feature, output, batch, is_validation=True)

        # Log individual losses
        self.log_dict(
            {f"val_{k}": v for k, v in metric_dict.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=len(batch['matid'])
        )

        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch['matid'])
        )

        if self.task_type == 'Regression' or self.task_type == 'BinaryClassification':
            preds = output[self.task_name].reshape(-1)
            targets = batch[self.task_name].reshape(-1)

            self.validation_output.append({"matid": batch['matid'], "preds": preds, "targets": targets})
        elif self.task_type == 'Classification':
            preds = output[self.task_name].reshape(-1, self.cls_num)
            targets = batch[self.task_name].reshape(-1)
            self.validation_output.append({"matid": batch['matid'], "preds": preds, "targets": targets})
        else:
            raise ValueError('Unsupported Task Type')

    def log_result(self, stage: Literal['val', 'test']):
        output = self.validation_output if stage == 'val' else self.test_output

        preds = torch.cat([x["preds"] for x in output])
        targets = torch.cat([x["targets"] for x in output])

        matids = [m for x in output for m in x['matid']]

        # Use datamodule's evaluate method for consistent normalization
        if self.task_type == 'Regression':
            mean, std = self.trainer.datamodule._mean[self.task_name], self.trainer.datamodule._std[self.task_name]
            metric_dict = metric_regression(
                targets[~torch.isnan(preds)], preds[~torch.isnan(preds)], mean=mean, std=std
            )

            df = pd.DataFrame()
            df['matid'] = matids
            df['predict'] = (preds * std + mean).cpu().detach().numpy()
            df['target'] = (targets * std + mean).cpu().detach().numpy()
            df.to_csv(Path(self.logger.log_dir) / f'{stage}_result.csv', index=False)

            if stage == 'val' and metric_dict['mae'] < self.best_val_metric:
                self.best_val_metric = metric_dict['mae']
                df.to_csv(Path(self.logger.log_dir) / 'best_mae_val_result.csv', index=False)

        elif self.task_type == 'BinaryClassification':
            metric_dict = metric_binary_classification(
                targets[~torch.isnan(targets)], preds[~torch.isnan(preds)], need_sigmoid=True
            )

            df = pd.DataFrame()
            df['matid'] = matids
            df['predict'] = F.sigmoid(preds).cpu().detach().numpy()
            df['target'] = targets.cpu().detach().numpy()
            df.to_csv(Path(self.logger.log_dir) / f'{stage}_result.csv', index=False)

            if stage == 'val' and metric_dict['acc'] > self.best_val_metric:
                self.best_val_metric = metric_dict['acc']
                df.to_csv(Path(self.logger.log_dir) / 'best_acc_val_result.csv', index=False)

        elif self.task_type == 'Classification':
            metric_dict = metric_classification(
                targets[~torch.isnan(targets)], preds[~torch.isnan(preds)]
            )
            df = pd.DataFrame()
            df['matid'] = matids
            # sigmoid
            # predict = F.sigmoid(preds)
            predict = preds
            predict = F.softmax(predict, dim=-1)
            predict = torch.argmax(torch.tensor(predict), dim=-1).cpu().detach().numpy()
            df['target'] = targets.cpu().detach().numpy()
            df['predict'] = predict
            df.to_csv(Path(self.logger.log_dir) / f'{stage}_result.csv', index=False)

        else:
            raise ValueError(f'Unsupported Task Type {self.task_type}')

        self.log_dict(
            {f"{stage}_end_{self.task_name}/{k}": v for k, v in metric_dict.items()},
            on_epoch=True,
            # prog_bar=True,
            logger=True,
        )



    def on_validation_epoch_end(self) -> None:
        self.log_result('val')
            
    def on_test_epoch_start(self) -> None:
        self.test_output = []

    def test_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Optional[STEP_OUTPUT]:
        """Validation step with logging."""
        feature, output = self(batch)
        metric_dict, loss = self._compute_loss(feature, output, batch)

        # Log individual losses
        self.log_dict(
            {f"test_{k}": v for k, v in metric_dict.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=len(batch['matid'])
        )

        preds = output[self.task_name]
        targets = batch[self.task_name]

        self.test_output.append({"matid": batch['matid'], "preds": preds, "targets": targets})

    def on_test_epoch_end(self) -> None:
        self.log_result('test')

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
        return set_scheduler(self)
        # return torch.optim.AdamW(self.parameters(), 1e-3, weight_decay=0.0)

    # def predict(self, graph: torch_geometric.data.Data, matid: str = "mat", log_contribution: bool = True):
    #     feature, output = self({
    #         "matid": [matid],
    #         "graph": [graph]
    #     }, log_contribution=log_contribution)
    #     pred_val = output[f'{self.task_name}'].reshape(-1)[0]
    #     if log_contribution:
    #         contribution = output[f'{self.task_name}_contribution'][matid]
    #         return pred_val, contribution
    #     else:
    #         return pred_val

@click.group()
def cli():
    pass


@cli.command()
@click.option("--config", type=str, required=True, help="Path to config YAML file")
def train(config: str) -> None:
    """Main training CLI entry point."""
    init_config(config)
    train_config = get_train_config()

    module = SpatialReadLightningModule(config)
    datamodule = FinetuneDatamodule()

    tb_logger = loggers.TensorBoardLogger(save_dir=Path(train_config["log_dir"]) / train_config['task_name'])

    device_stats = DeviceStatsMonitor()

    if train_config['task_type'] == 'Classification' or train_config['task_type'] == 'BinaryClassification':
        checkpoint_callback = ModelCheckpoint(
            # monitor=f"val_end_{train_config['task_name']}/mae",
            monitor=f"val_end_{train_config['task_name']}/acc",
            # mode="min",
            mode="max",
            save_last=True,
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            monitor=f"val_end_{train_config['task_name']}/mae",
            # monitor=f"val_end_{train_config['task_name']}/acc",
            mode="min",
            # mode="max",
            save_last=True,
        )

    trainer = Trainer(
        callbacks=[device_stats, checkpoint_callback],
        max_epochs=train_config["max_epochs"],
        accelerator=train_config["accelerator"],
        devices=train_config["device"],
        precision=train_config["precision"],
        strategy=train_config["strategy"],
        enable_progress_bar=True,
        logger=tb_logger,
        log_every_n_steps=train_config["log_every_n_steps"],
        gradient_clip_val=train_config["gradient_clip_val"],
        accumulate_grad_batches=train_config["accumulate_grad_batches"],
    )

    ckpt_path = train_config["ckpt_path"]
    resume = train_config['resume']
    if ckpt_path is not None:
        loadret = module.load_state_dict(torch.load(ckpt_path, weights_only=False, map_location='cpu')['state_dict'], strict=False)
        print(f"Load from {ckpt_path}, return", loadret)

    if resume is not None:
        print(f"Resume from {resume}")
        trainer.fit(module, datamodule=datamodule, ckpt_path=resume)
    else:
        trainer.fit(module, datamodule=datamodule)

    trainer.test(module, datamodule=datamodule)

# @cli.command()
# def test(config: str) -> None:
#     """Main training CLI entry point."""
#     init_config(config)
#     train_config = get_train_config()

#     module = SpatialReadLightningModule(config)
#     datamodule = PorousMatDatamodule()

#     tb_logger = loggers.TensorBoardLogger(save_dir=train_config["log_dir"])

#     trainer = Trainer(
#         max_epochs=train_config["max_epochs"],
#         accelerator=train_config["accelerator"],
#         devices=train_config["device"],
#         precision=train_config["precision"],
#         strategy=train_config["strategy"],
#         enable_progress_bar=True,
#         logger=tb_logger,
#         log_every_n_steps=train_config["log_every_n_steps"],
#         gradient_clip_val=train_config["gradient_clip_val"],
#         accumulate_grad_batches=train_config["accumulate_grad_batches"],
#     )

#     ckpt_path = train_config["ckpt_path"]
#     assert ckpt_path is not None, f'test must have ckpt_path'
#     module.load_state_dict(torch.load(ckpt_path, weights_only=False)['state_dict'], strict=True)
#     trainer.test(module, datamodule=datamodule)

# @cli.command()
# @click.option('--ckpt-path', type=str, required=True)
# @click.option('--cif-path', type=str, required=True)
# @click.option('--output-dir', type=str, default='.')
# def predict(ckpt_path: str, cif_path: str, output_dir: str):
#     from torch_geometric.data import Data
#     from pymatgen.core.structure import Structure
#     from pymatgen.io.ase import AseAtomsAdaptor
#     from ase.io import read
#     from spatialread.utils.chem import _make_supercell

#     matid = Path(cif_path).stem
#     ckpt = torch.load(ckpt_path, weights_only=False)

#     """Main training CLI entry point."""
#     config = ckpt['hyper_parameters']
#     init_config(config, is_predict=True)
#     train_config = get_train_config()
#     data_config = get_data_config()

#     atoms = read(str(cif_path))
#     atoms = _make_supercell(atoms, cutoff=data_config['min_lat_len'])
#     struct = AseAtomsAdaptor().get_structure(atoms)
#     graph = Data(
#         atomic_numbers=torch.tensor([site.specie.Z for site in struct], dtype=torch.long),
#         pos=torch.from_numpy(np.stack([site.coords for site in struct])).to(torch.float),
#         cell=torch.tensor(struct.lattice.matrix, dtype=torch.float),
#         matid=matid,
#     )

#     module = SpatialReadLightningModule(config, is_predict=True)

#     assert ckpt_path is not None, f'predict must have ckpt_path'
#     module.load_state_dict(ckpt['state_dict'], strict=True)

#     pred_val, contribution = module.predict(graph, matid=matid, log_contribution=True)

#     output_dir = Path(output_dir)
#     (output_dir / 'contribution').mkdir(exist_ok=True, parents=True)
#     np.save(output_dir / 'contribution' / f'{matid}.npy', contribution)
#     print("Matid: ", matid, 'Predicted Value:', pred_val)
    


if __name__ == "__main__":
    cli()
