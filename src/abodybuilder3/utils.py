import io
import logging

import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping

from abodybuilder3.dataloader import ABDataset
from abodybuilder3.openfold.data.data_transforms import make_atom14_masks
from abodybuilder3.openfold.np.protein import Protein, to_pdb
from abodybuilder3.openfold.np.residue_constants import restype_order_with_x
from abodybuilder3.openfold.utils.feats import atom14_to_atom37
from abodybuilder3.openfold.utils.loss import compute_plddt

log = logging.getLogger(__name__)


def string_to_input(heavy: str, light: str, device: str = "cpu") -> dict:
    """Generates an input formatted for an ABB3 model from heavy and light chain
    strings.

    Args:
        heavy (str): heavy chain
        light (str): light chain

    Returns:
        dict: A dictionary containing
            aatype: an (n,) tensor of integers encoding the amino acid string
            is_heavy: an (n,) tensor where is_heavy[i] = 1 means residue i is heavy and
                is_heavy[i] = 0 means residue i is light
            residue_index: an (n,) tensor with indices for each residue. There is a gap
                of 500 between the last heavy residue and the first light residue
            single: a (1, n, 23) tensor of node features
            pair: a (1, n, n, 132) tensor of edge features
    """
    aatype = []
    is_heavy = []
    for character in heavy:
        is_heavy.append(1)
        aatype.append(restype_order_with_x[character])
    for character in light:
        is_heavy.append(0)
        aatype.append(restype_order_with_x[character])
    is_heavy = torch.tensor(is_heavy)
    aatype = torch.tensor(aatype)
    residue_index = torch.cat(
        (torch.arange(len(heavy)), torch.arange(len(light)) + 500)
    )

    model_input = {
        "is_heavy": is_heavy,
        "aatype": aatype,
        "residue_index": residue_index,
    }
    model_input.update(
        ABDataset.single_and_double_from_datapoint(
            model_input, 64, edge_chain_feature=True
        )
    )
    model_input["single"] = model_input["single"].unsqueeze(0)
    model_input["pair"] = model_input["pair"].unsqueeze(0)

    model_input = {k: v.to(device) for k, v in model_input.items()}
    return model_input


def backbones_from_outputs(outputs: list[dict], aatype: torch.Tensor) -> torch.Tensor:
    """Generates a tensor of size (n, len(outputs), 3) of backbone coordinates.
    Entry backbones[i, j] contains the Ca coordinates of residue i in output j.

    Args:
        outputs (list[dict]): outputs from ABB3 models
        aatype (torch.Tensor): an (n,) tensor of integers encoding the amino acid string

    Returns:
        torch.Tensor: A tensor of Ca coordinates.
    """
    backbones = []
    for output in outputs:
        add_atom37_to_output(output, aatype)
        backbones.append(output["atom37"][:, 1, :])
    backbones = torch.stack(backbones)
    return backbones


def add_atom37_to_output(output: dict, aatype: torch.Tensor):
    """Adds atom37 coordinates to an output dictionary containing atom14 coordinates."""
    atom14 = output["positions"][-1, 0]
    batch = make_atom14_masks({"aatype": aatype.squeeze()})
    atom37 = atom14_to_atom37(atom14, batch)
    output["atom37"] = atom37
    output["atom37_atom_exists"] = batch["atom37_atom_exists"]
    return output


def output_to_pdb(output: dict, model_input: dict) -> str:
    """Generates a pdb file from ABB3 predictions.

    Args:
        output (dict): ABB3 output dictionary
        model_input (dict): ABB3 input dictionary

    Returns:
        str: the contents of a pdb file in string format.
    """
    aatype = model_input["aatype"].squeeze().cpu().numpy().astype(int)
    atom37 = output["atom37"]
    chain_index = 1 - model_input["is_heavy"].cpu().numpy().astype(int)
    atom_mask = output["atom37_atom_exists"].cpu().numpy().astype(int)
    residue_index = np.arange(len(atom37))
    if "plddt" in output:
        plddt = compute_plddt(output["plddt"].squeeze()).unsqueeze(1)
        b_factors = plddt.repeat(1, 37).detach().cpu().numpy()
    else:
        b_factors = np.zeros_like(atom_mask)

    protein = Protein(
        aatype=aatype,
        atom_positions=atom37,
        atom_mask=atom_mask,
        residue_index=residue_index,
        b_factors=b_factors,
        chain_index=chain_index,
    )

    return to_pdb(protein)


class DelayedEarlyStopping(EarlyStopping):
    def __init__(self, delay_start: int = 50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay_start = delay_start
        self.start_epoch = None

    def on_train_start(self, trainer, pl_module):
        # Store the starting epoch
        if self.start_epoch is None:
            self.start_epoch = trainer.current_epoch
        super().on_train_start(trainer, pl_module)

    def on_validation_start(self, trainer, pl_module):
        # Store the starting epoch
        if self.start_epoch is None:
            self.start_epoch = trainer.current_epoch
        super().on_train_start(trainer, pl_module)

    def _should_delay(self, trainer, log_info: bool = False):
        # Calculate the difference between current epoch and starting epoch
        epochs_passed = trainer.current_epoch - self.start_epoch
        epochs_remaining = self.delay_start - epochs_passed
        if epochs_remaining > 0:
            if log_info:
                log.info(
                    "Early Stopping will start monitoring in"
                    f" {epochs_remaining} epochs."
                )
            return True
        return False

    def on_train_epoch_end(self, trainer, pl_module, *args, **kwargs):
        if self._should_delay(trainer):
            return
        super().on_train_epoch_end(trainer, pl_module, *args, **kwargs)

    def on_validation_end(self, trainer, pl_module, *args, **kwargs):
        if self._should_delay(trainer, log_info=True):
            return
        super().on_validation_end(trainer, pl_module, *args, **kwargs)
