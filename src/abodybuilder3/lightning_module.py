import lightning.pytorch as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from abodybuilder3.dataloader import ABDataset, collate_fn
from abodybuilder3.loss import ABB3Loss
from abodybuilder3.openfold.model.structure_module import StructureModule
from abodybuilder3.radam import RAdam


class ABB3DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        legacy: bool = False,
        rel_pos_dim: int = 64,
        edge_chain_feature: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        use_plm_embeddings: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.legacy = legacy
        self.rel_pos_dim = rel_pos_dim
        self.edge_chain_feature = edge_chain_feature
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_plm_embeddings = use_plm_embeddings

    def setup(self, stage: str):
        self.train_dataset = ABDataset(
            self.data_dir,
            "train",
            legacy=self.legacy,
            rel_pos_dim=self.rel_pos_dim,
            edge_chain_feature=self.edge_chain_feature,
            use_plm_embeddings=self.use_plm_embeddings,
        )
        self.valid_dataset = ABDataset(
            self.data_dir,
            "valid",
            legacy=self.legacy,
            rel_pos_dim=self.rel_pos_dim,
            edge_chain_feature=self.edge_chain_feature,
            use_plm_embeddings=self.use_plm_embeddings,
        )
        self.test_dataset = ABDataset(
            self.data_dir,
            "test",
            legacy=self.legacy,
            rel_pos_dim=self.rel_pos_dim,
            edge_chain_feature=self.edge_chain_feature,
            use_plm_embeddings=self.use_plm_embeddings,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class LitABB3(pl.LightningModule):
    def __init__(self, model_config, loss_config, optim_config):
        super().__init__()
        model_config["use_plddt"] = loss_config.plddt.weight > 0
        self.save_hyperparameters()
        self.model_config = model_config
        self.loss_config = loss_config
        self.optim_config = optim_config
        self.model = StructureModule(**model_config)
        self.loss = ABB3Loss(loss_config)
        self.finetune = False

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, "train")
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), max_norm=float("inf")
        )
        self.log("grad_norm", grad_norm)
        return loss

    def validation_step(self, batch, batch_idx):
        self._step(batch, "valid")
        self.log(
            "finetune",
            float(self.finetune),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def _step(self, batch, split):
        output = self.model(
            {
                "single": batch["single"],
                "pair": batch["pair"],
            },
            batch["aatype"],
            batch["seq_mask"],
        )
        loss, loss_dict = self.loss(output, batch, self.finetune)
        for loss_name in loss_dict:
            self.log(
                f"{split}/{loss_name}",
                loss_dict[loss_name],
                prog_bar=loss_name == "loss",
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )
        return loss

    def configure_optimizers(self):
        if self.optim_config.optimiser == "RAdam":
            optimizer = RAdam(
                self.parameters(),
                lr=self.optim_config.lr,
                weight_decay=self.optim_config.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.optim_config.T_0,
                T_mult=self.optim_config.T_mult,
                eta_min=self.optim_config.eta_min,
            )
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "learning_rate",
            }
            return [optimizer], [lr_scheduler]
        elif self.optim_config.optimiser == "AdamW":
            optimizer = AdamW(
                self.parameters(),
                lr=self.optim_config.lr,
                betas=(0.9, 0.99),
                eps=1e-6,
            )
            scheduler1 = LinearLR(optimizer, start_factor=1e-3, total_iters=1000)
            scheduler2 = LambdaLR(
                optimizer, lambda epoch: 0.95 if epoch >= 30_000 else 1
            )
            scheduler = SequentialLR(
                optimizer,
                [scheduler1, scheduler2],
                milestones=[
                    1000,
                ],
            )
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "learning_rate",
            }
            return [optimizer], [lr_scheduler]
        else:
            raise ValueError(
                "Expected Adam or RAdam as optimiser. Instead got"
                f" {self.optim_config.optimiser=}."
            )
