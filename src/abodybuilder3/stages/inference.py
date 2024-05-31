import io
import warnings
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
import typer

from abodybuilder3.lightning_module import ABB2DataModule, LitABB2

from abodybuilder3.openfold.np.protein import Protein, to_pdb
from abodybuilder3.openfold.np.relax.cleanup import fix_pdb
from abodybuilder3.openfold.utils.feats import (
    atom14_to_atom37 as openfold_atom14_to_atom37,
)

warnings.filterwarnings("ignore")


def atom14_to_atom37(position: np.ndarray, sample_unshaped: dict) -> np.ndarray:
    batch = {
        "residx_atom37_to_atom14": sample_unshaped["residx_atom37_to_atom14"].numpy(),
        "atom37_atom_exists": sample_unshaped["atom37_atom_exists"].numpy(),
    }
    return openfold_atom14_to_atom37(position, batch)


def compute_plddt(plddt: torch.Tensor) -> torch.Tensor:
    """Computes plddt from the model output. The output is a histogram of unnormalised
    plddt.

    Args:
        plddt (torch.Tensor): (B, n, 50) output from the model

    Returns:
        torch.Tensor: (B, n) plddt scores
    """
    pdf = torch.nn.functional.softmax(plddt, dim=-1)
    vbins = torch.arange(1, 101, 2).to(plddt.device).float()
    output = pdf @ vbins  # (B, n)
    return output


def main(
    model: str = typer.Option("language"),
    metric: str = typer.Option("loss"),
    output_dir: Path = typer.Option("output"),
):
    output_dir = output_dir / f"{model}-{metric}"
    for folder in ["true", "pred", "plddt", "pred_unfixed"]:
        (output_dir / folder).mkdir(exist_ok=True, parents=True)

    data = ABB2DataModule(
        data_dir="data/",
        legacy=False,
        batch_size=1,
        edge_chain_feature=True,
        use_plm_embeddings="language" in model,
    )
    data.setup(None)

    # inference
    ckpt = output_dir / "best_second_stage.ckpt"
    logger.info(f"Loading from {ckpt=}")
    model = LitABB2.load_from_checkpoint(ckpt)
    model.model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("Inference is being done on CPU as GPU not found.")
    model.to(device)

    valid = zip(data.valid_dataset, data.val_dataloader())
    test = zip(data.test_dataset, data.test_dataloader())
    combined = chain(test, valid)
    total = len(data.valid_dataset) + len(data.test_dataset)

    predictions_failed = 0
    for sample_unshaped, sample in tqdm(combined, total=total):
        identifier = sample_unshaped["structure"]
        logger.info(f"Processing {identifier}.")

        for key in sample:
            sample[key] = sample[key].to(device)
        with torch.no_grad():
            output = model.model(
                {
                    "single": sample["single"],
                    "pair": sample["pair"],
                },
                sample["aatype"],
                sample["seq_mask"],
            )

        positions_true14 = (
            sample["atom14_gt_positions"].squeeze().cpu().squeeze().numpy()
        )
        positions_pred14 = output["positions"][-1].squeeze().cpu().numpy()
        positions_true = atom14_to_atom37(positions_true14, sample_unshaped)
        positions_pred = atom14_to_atom37(positions_pred14, sample_unshaped)
        assert positions_true.shape == positions_pred.shape

        # to_pdb
        aatype = sample["aatype"].squeeze().cpu().numpy().astype(int)
        atom_mask = (
            sample_unshaped["all_atom_mask"].numpy().astype(int)
        )  # was all_atom_mask. Changed to atom37_atom_exists then back. For 7vyr_H0-L0 the atom37_atom_exists is incorrect (showing that the first atom exists).
        residue_index = sample["residue_index"].squeeze().cpu().numpy().astype(int) + 1
        chain_index = 1 - sample_unshaped["is_heavy"].numpy().astype(int)

        # write ground truth pdb
        ground_truth = Protein(
            aatype=aatype,
            atom_positions=positions_true,
            atom_mask=atom_mask,
            residue_index=residue_index,
            b_factors=np.zeros_like(atom_mask),
            chain_index=chain_index,
        )
        with open(output_dir / "true" / f"{identifier}.pdb", "w") as f:
            f.write(fix_pdb(io.StringIO(to_pdb(ground_truth)), {}))

        # write plddt
        if "plddt" in output:
            plddt = compute_plddt(output["plddt"]).squeeze().detach().cpu().numpy()
        else:
            plddt = np.zeros(len(aatype), dtype=np.float)
        torch.save(
            {"plddt": plddt, "region": sample_unshaped["region"]},
            output_dir / "plddt" / f"{identifier}.pt",
        )

        # write prediction pdb
        try:
            b_factors = np.expand_dims(plddt, 1).repeat(37, 1)
            prediction = Protein(
                aatype=aatype,
                atom_positions=positions_pred,
                atom_mask=atom_mask,
                residue_index=residue_index,
                b_factors=b_factors,
                chain_index=chain_index,
            )
            with open(output_dir / "pred" / f"{identifier}.pdb", "w") as f:
                f.write(fix_pdb(io.StringIO(to_pdb(prediction)), {}))
            with open(output_dir / "pred_unfixed" / f"{identifier}.pdb", "w") as f:
                f.write(to_pdb(prediction))
            logger.info(f"pred {identifier} success")
        except:
            logger.info(f"pred {identifier} fail")
            predictions_failed += 1

    logger.info(f"{predictions_failed=}")


if __name__ == "__main__":
    typer.run(main)
