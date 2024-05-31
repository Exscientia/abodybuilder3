import os

import dvc.api
import lightning.pytorch as pl
import ml_collections
import torch
from box import Box
from dvclive.lightning import DVCLiveLogger
from lightning.fabric import Fabric
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from loguru import logger

from abodybuilder3.lightning_module import ABB3DataModule, LitABB3

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

    # model and loss configs
    loss_config = ml_collections.config_dict.ConfigDict(params.loss)
    model_config = ml_collections.config_dict.ConfigDict(params.model)
    optimiser_config = ml_collections.config_dict.ConfigDict(params.optimiser)

    # edge_features dimension
    rel_pos_dim = 64
    c_z = 2 * rel_pos_dim + 1
    if params.model.edge_chain_feature:
        c_z += 3
    model_config.c_z = c_z

    if params.language.model is not None:
        model_config.c_s = 1024

    # log slurm parameters
    pl.seed_everything(params.base.seed)

    fabric.barrier()
    if fabric.global_rank == 0:
        logger.info("SLURM_NTASKS =", os.environ.get("SLURM_NTASKS"))
        logger.info("SLURM_TASKS_PER_NODE =", os.environ.get("SLURM_TASKS_PER_NODE"))
        logger.info("SLURM_JOB_GPUS =", os.environ.get("SLURM_JOB_GPUS"))
        logger.info("SLURM_NNODES =", os.environ.get("SLURM_NNODES"))
        logger.info(f"{torch.cuda.device_count()=}")

    # data module
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
    accumulate_grad_batches = int(params.train.batch_size / samples_processed_per_step)

    fabric.barrier()
    if fabric.global_rank == 0:
        logger.info(f"{nodes=}")
        logger.info(f"{devices=}")
        logger.info(f"{accumulate_grad_batches=}")

    # first stage of training
    model = LitABB3(model_config, loss_config, optimiser_config)
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints/first_stage/",
            auto_insert_metric_name=False,
            save_top_k=1,
            filename="best_first_stage_loss",
            monitor="valid/loss",
        ),
        ModelCheckpoint(
            dirpath="checkpoints/first_stage/",
            auto_insert_metric_name=False,
            save_top_k=1,
            filename="best_first_stage_cdrh3_rmsd",
            monitor="valid/cdrh3_rmsd",
        ),
        EarlyStopping(
            monitor="valid/loss", patience=params.train.early_stopping, verbose=True
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if model.model_config.use_plddt:
        callbacks.append(
            ModelCheckpoint(
                dirpath="checkpoints/first_stage/",
                auto_insert_metric_name=False,
                save_top_k=1,
                filename="best_first_stage_plddt",
                monitor="valid/plddt",
            )
        )
    else:
        with open("checkpoints/first_stage/best_first_stage_plddt.ckpt", "a"):
            pass

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=devices,
        num_nodes=nodes,
        accumulate_grad_batches=accumulate_grad_batches,
        check_val_every_n_epoch=1,
        logger=DVCLiveLogger(dir="dvclive/first_stage"),
        callbacks=callbacks,
        max_epochs=params.train.epochs if not params.base.debug else 3,
        strategy="ddp",
        limit_train_batches=1.0 if not params.base.debug else 1,
        limit_val_batches=1.0 if not params.base.debug else 1,
        precision="bf16-mixed",
    )
    trainer.fit(model, data)
