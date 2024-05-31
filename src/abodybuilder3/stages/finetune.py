import os

import dvc.api
import lightning.pytorch as pl
import torch
from box import Box
from dvclive.lightning import DVCLiveLogger
from lightning.fabric import Fabric
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from loguru import logger

from abodybuilder3.lightning_module import ABB3DataModule, LitABB3
from abodybuilder3.utils import DelayedEarlyStopping


class FineTuneCallback(Callback):
    def __init__(
        self,
        use_annealing: bool = True,
        dropout: float = 0.0,
        turn_off_scheduler: bool = True,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.use_annealing = use_annealing
        self.used = False
        self.dropout = dropout
        self.turn_off_scheduler = turn_off_scheduler
        self.learning_rate = learning_rate

    def on_train_epoch_start(self, trainer, pl_module):
        if not self.used:
            pl_module.finetune = True

            # fix learning rate and turn of schedular
            if self.turn_off_scheduler:
                trainer.optimizers[0].param_groups[0]["lr"] = self.learning_rate
                trainer.strategy.lr_scheduler_configs = []

            # adjust dropout
            pl_module.model_config.dropout_rate = self.dropout
            for layer in pl_module.model.modules():
                if isinstance(layer, torch.nn.Dropout):
                    layer.p = self.dropout

            # make sure initialise adam once
            self.used = True

            if not self.use_annealing:
                pl_module.loss.dist_and_angle_annealing = 50

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.loss.dist_and_angle_annealing += 1


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    nodes = int(os.environ.get("SLURM_NNODES", 1))
    devices = len(os.environ.get("SLURM_JOB_GPUS", "0").split(","))
    logger.info(f"Computing with {nodes=} with {devices=} GPUs each.")

    # read params and share across nodes
    fabric = Fabric(devices=devices, num_nodes=nodes)
    fabric.launch()
    fabric.barrier()
    if fabric.global_rank == 0:
        config = dvc.api.params_show()
    else:
        config = None
    params = Box(fabric.broadcast(config, src=0))

    pl.seed_everything(params.base.seed)

    fabric.barrier()
    if fabric.global_rank == 0:
        logger.info("SLURM_NTASKS =", os.environ.get("SLURM_NTASKS"))
        logger.info("SLURM_TASKS_PER_NODE =", os.environ.get("SLURM_TASKS_PER_NODE"))
        logger.info("SLURM_JOB_GPUS =", os.environ.get("SLURM_JOB_GPUS"))
        logger.info("SLURM_NNODES =", os.environ.get("SLURM_NNODES"))
        logger.info(f"{torch.cuda.device_count()=}")

    legacy = not params.base.all_data
    data = ABB3DataModule(
        data_dir=params.base.data_dir,
        batch_size=8,
        legacy=legacy,
        edge_chain_feature=params.model.edge_chain_feature,
        num_workers=params.base.num_workers,
        pin_memory=params.base.pin_memory,
        use_plm_embeddings=params.language.model is not None,
    )
    samples_processed_per_step = devices * nodes * 8
    accumulate_grad_batches = int(
        params.finetune.batch_size / samples_processed_per_step
    )

    fabric.barrier()
    if fabric.global_rank == 0:
        logger.info(f"{nodes=}")
        logger.info(f"{devices=}")
        logger.info(f"{accumulate_grad_batches=}")

    # first stage of training
    ckpt = f"checkpoints/first_stage/best_first_stage_{params.finetune.metric}.ckpt"
    logger.info(f"Loading from {ckpt=}")
    model = LitABB3.load_from_checkpoint(ckpt)
    ckpt_epoch = torch.load(ckpt)["epoch"]
    logger.info(f"Resuming from epoch {ckpt_epoch=}")

    # second stage of training
    if params.finetune.use_annealing:
        early_stopping = DelayedEarlyStopping(
            delay_start=50,
            monitor=f"valid/{params.finetune.metric}",
            patience=params.finetune.early_stopping,
            verbose=True,
        )
    else:
        early_stopping = EarlyStopping(
            monitor=f"valid/{params.finetune.metric}",
            patience=params.finetune.early_stopping,
            verbose=True,
        )

    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints/second_stage",
            auto_insert_metric_name=False,
            save_top_k=1,
            filename="best_second_stage_loss",
            monitor=f"valid/loss",
        ),
        ModelCheckpoint(
            dirpath="checkpoints/second_stage/",
            auto_insert_metric_name=False,
            save_top_k=1,
            filename="best_second_stage_cdrh3_rmsd",
            monitor="valid/cdrh3_rmsd",
        ),
        early_stopping,
        LearningRateMonitor(logging_interval="epoch"),
        FineTuneCallback(
            use_annealing=params.finetune.use_annealing,
            dropout=params.finetune.dropout,
            turn_off_scheduler=params.finetune.turn_off_scheduler,
            learning_rate=params.finetune.learning_rate,
        ),
    ]

    if model.model_config.use_plddt:
        callbacks.append(
            ModelCheckpoint(
                dirpath="checkpoints/second_stage/",
                auto_insert_metric_name=False,
                save_top_k=1,
                filename="best_second_stage_plddt",
                monitor="valid/plddt",
            )
        )
    else:
        with open("checkpoints/second_stage/best_second_stage_plddt.ckpt", "a"):
            pass

    max_epochs = ckpt_epoch + params.finetune.epochs
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=devices,
        num_nodes=nodes,
        accumulate_grad_batches=accumulate_grad_batches,
        check_val_every_n_epoch=1,
        logger=DVCLiveLogger(dir="dvclive/second_stage"),
        callbacks=callbacks,
        max_epochs=max_epochs if not params.base.debug else 6,
        strategy="ddp",
        limit_train_batches=1.0 if not params.base.debug else 1,
        limit_val_batches=1.0 if not params.base.debug else 1,
        precision="bf16-mixed",
    )
    trainer.fit(model, data, ckpt_path=ckpt)
