"""Script to preprocess data for the ABB3 model.

For each Fv, three main methods are used to extract information.

The first, extract_chain_information, extracts data from raw PDB files.

    residue_mask: torch.tensor (n,) mask for if the residue is standard
    all_atom_positions: torch.tensor (n, 37, 3) coordinates of atoms
    all_atom_mask: all_atom_mask (n, 37) mask of which of the 37 atoms are present
    cdr_mask: torch.tensor (n,) mask for if cdr region
    sequence_3_letters: list of length n of 3 letter codes
    is_heavy: torch.tensor (n,) mask for if the heavy chain

The second, add_sequence_and_aatype, adds the following information from the sequence_3_letters.

    sequence: a string of length n of 1 letter codes (X for non standard)
    aatype: torch.tensor (n,) integer encoding of the seqeunce (X is 20)

The third uses openfold to add information based on the above. We further break this down into a number of transforms.

1. make_atom14_masks. Makes atom exists masks and mapping between atom14 and atom37.
    input:
        aatype
    outputs:
        atom14_atom_exists: torch.tensor(n, 14) mask for which atom14 coords are used
        atom37_atom_exists: torch.tensor(n, 37) mask for which atom37 coords are used
        residx_atom14_to_atom37: torch.tensor(n, 14) used to map from atom14 to atom37
        residx_atom37_to_atom14: torch.tensor(n, 37) used to map from atom37 to atom14

2. make_atom14_positions. Makes masks for atom14 coordinates existing. Includes alternative atom positions and a tensor indicating if an atom can be represented in two ways (i.e. has an alt representation).
    input:
        atom14_atom_exists
        residx_atom14_to_atom37
        all_atom_mask
        all_atom_positions
    outputs:
        atom14_gt_exists: torch.tensor(n, 14) mask for which atom14 coords are used
        atom14_gt_positions: torch.tensor(n, 14, 3) atom14 coordinates
        atom14_alt_gt_exists: torch.tensor(n, 14) mask for which atom14 alterative coords are used
        atom14_alt_gt_positions: torch.tensor(n, 14, 3) alternative atom14 coordinates
        atom14_atom_is_ambiguous:  torch.tensor(n, 14) mask indicating whether an atom has an alt coordinate

3. atom37_to_frames. Generates rigids for the backbone plus 7 rigid groups.

    input:
        aatype
        all_atom_positions
        all_atom_mask
    output:
        rigidgroups_gt_exists: torch.tensor(n, 8) ground truth mask for each rigid group in residue
        rigidgroups_group_exists: torch.tensor(n, 8) - same as rigidgroups_gt_exists as far as I can tell...
        rigidgroups_gt_frames: torch.tensor(n, 8, 4, 4) - 4 x 4 rigid group of each atom/frame
        rigidgroups_alt_gt_frames: torch.tensor(n, 8, 4, 4) - alternative 4 x 4 rigid group
        rigidgroups_group_is_ambiguous torch.tensor(n, 8) - if the 4 x 4 rigid group has two potential values (i.e. if the gt and alt_gt differ)

4. atom37_to_torsion_angles. Generates torsion of the residue.

    input:
        aatype
        all_atom_positions
        all_atom_mask
    output:
        torsion_angles_sin_cos: torch.tensor(n, 7, 2) - torsion angles
        alt_torsion_angles_sin_cos : torch.tensor(n, 7, 2) - alternate torsion angles (accounting for 180-degree symmetry)
        torsion_angles_mask: torch.tensor(n, 7, 2) - mask for which angles are used.

5. make_pseudo_beta. I think this is the approximate positions of the beta carbon atoms (Cβ) in each residue of the protein.

    input:
        aatype
        all_atom_positions
        all_atom_mask
    output:
        pseudo_beta: torch.tensor(n, 3) - coordinate
        pseudo_beta_mask: torch.tensor(n,) - if it exists

6. get_backbone_frames. Extracts the backbone frame from all frames computed using atom37_to_frames.

    input:
        rigidgroups_gt_frames
        rigidgroups_gt_exists
    output:
        backbone_rigid_tensor: torch.tensor(n, 4, 4). Equal to rigidgroups_gt_frames[:, 0, :, :]
        backbone_rigid_mask: torch.tensor(n,). Equal to rigidgroups_gt_exists[:, 0]

7. get_chi_angles. Extracts backbound torsion angles

    input:
        all_atom_mask (just for dtype)
        torsion_angles_sin_cos
        torsion_angles_mask
    output:
        chi_angles_sin_cos: torch.tensor(n, 4, 2). Equal to torsion_angles_sin_cos[:, 3:, :]
        chi_mask: torch.tensor(n, 4, 2). Equal to torsion_angles_mask[:, 3:, :]

Finally, the following information which is at the Fv level

    pdb: pdb string
    fv: sabdab Fv object
"""

from __future__ import annotations

import io
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import typer
from Bio import BiopythonDeprecationWarning
from exs.sabdab import GideonInterface
from exs.sabdab.antibody.fv import Fv
from exs.sabdab.antibody.sabdab_entries import SAbDabEntry
from exs.sabdab.antibody.variable_domain import VariableDomain
from tqdm import tqdm

from exs.abodybuilder2.openfold.data import data_transforms
from exs.abodybuilder2.openfold.np import residue_constants
from exs.abodybuilder2.openfold.np.protein import Protein, to_pdb
from exs.abodybuilder2.openfold.np.relax.cleanup import fix_pdb
from exs.abodybuilder2.openfold.utils.feats import atom14_to_atom37

# Ignore the deprecation warning from Biopython
warnings.simplefilter("ignore", BiopythonDeprecationWarning)


HEAVY_REGION_FW_NAMES = {"fwh1", "fwh2", "fwh3", "fwh4"}
HEAVY_REGION_CDR_NAMES = {"cdrh1", "cdrh2", "cdrh3"}
HEAVY_REGION_NAMES = HEAVY_REGION_FW_NAMES.union(HEAVY_REGION_CDR_NAMES)

LIGHT_REGION_FW_NAMES = {"fwl1", "fwl2", "fwl3", "fwl4"}
LIGHT_REGION_CDR_NAMES = {"cdrl1", "cdrl2", "cdrl3"}
LIGHT_REGION_NAMES = LIGHT_REGION_FW_NAMES.union(LIGHT_REGION_CDR_NAMES)

CDR_NAMES = HEAVY_REGION_CDR_NAMES.union(LIGHT_REGION_CDR_NAMES)
FW_NAMES = HEAVY_REGION_FW_NAMES.union(LIGHT_REGION_FW_NAMES)
REGION_NAMES = HEAVY_REGION_NAMES.union(LIGHT_REGION_NAMES)


class ErrorWithLocation(Exception):
    def __init__(self, reason_for_failure: str, location_of_failure: str):
        self.reason_for_failure = reason_for_failure
        self.location_of_failure = location_of_failure
        super().__init__(
            f"Reason for failure: {reason_for_failure}\nLocation of failure:"
            f" {location_of_failure}"
        )


def extract_chain_information(
    chain: VariableDomain,
) -> dict[str, np.ndarray | list[str] | torch.Tensor]:
    """Extracts raw information from chain, other properties can be derived from this.

    Args:
        chain (VariableDomain):

    Returns:
        A dictionary containing:
            seq_mask: an torch tensor of shape (n, ) where 0 indicates if we want to
            ignore a residue (e.g. because it is a water) or because the structural
            information is not present.

            all_atom_positions: an torch tensor of shape (n, 37, 3) containing atom37 coordinates

            all_atom_mask: an torch tensor of shape (n, 37) containing a 1 if that atom37 coordinate is used.

            cdr_mask: an torch tensor of shape (n, ) which is 1 if the residue is in a CDR region.

            sequence_3_letters: a list of 3-letter amino acid codes.

            is_heavy: a torch tensor of shape (n,) indicating if the residue is a heavy
            chain (1) or light chain (0).

            region: a list of regions per residue from anarci ['fwh1', 'fwh1',...]

            residue_index: a torch tensor of shape (n,) giving an index to a residue.
    """
    seq_mask = []
    all_atom_positions = []
    all_atom_mask = []
    cdr_mask = []
    sequence_3_letters = []
    region = []
    residues = chain.get_residues()

    for residue in residues:
        # extract region and residue
        residue_region = residue.get_region(definition="imgt")

        # check region is recognised
        if residue_region not in REGION_NAMES:
            raise ValueError(
                "Expected numbered residue to be a recognised region. Got"
                f" {residue_region=} at {residue.imgt_number=}"
            )

        # record region, cdr_mask, seq letters
        region.append(residue_region)
        cdr_mask.append(int(residue_region in CDR_NAMES))
        sequence_3_letters.append(residue.resname)

        if residue.is_missing:
            # numbered residue without structure
            all_atom_positions.append(torch.zeros((37, 3)))
            all_atom_mask.append(torch.zeros((37,)))
        else:
            # compute atom 37
            all_atom_positions_entry = torch.zeros((37, 3))
            all_atom_mask_entry = torch.zeros((37,))
            for atom in residue.get_atoms():
                if (
                    atom.name in residue_constants.atom_types
                ):  # atoms such as hydrogen we ignore
                    all_atom_positions_entry[
                        residue_constants.atom_order[atom.name]
                    ] = torch.tensor(atom.coord)
                    all_atom_mask_entry[residue_constants.atom_order[atom.name]] = 1.0
            all_atom_positions.append(all_atom_positions_entry)
            all_atom_mask.append(all_atom_mask_entry)

        if residue.resname not in residue_constants.restype_3to1.keys():
            seq_mask.append(0)
        else:
            seq_mask.append(1)

    all_atom_positions = torch.stack(all_atom_positions)
    all_atom_mask = torch.stack(all_atom_mask)

    if chain.chain_type == "H":
        is_heavy = torch.ones(len(all_atom_mask))
    else:
        is_heavy = torch.zeros(len(all_atom_mask))

    # a hack to make positional encoding not go across chains
    residue_index = torch.arange(len(all_atom_positions))
    if chain.chain_type == "L":
        residue_index = residue_index + 500

    to_return = {
        "seq_mask": torch.tensor(seq_mask),
        "all_atom_positions": all_atom_positions,
        "all_atom_mask": all_atom_mask,
        "cdr_mask": torch.tensor(cdr_mask),
        "sequence_3_letters": sequence_3_letters,
        "is_heavy": is_heavy,
        "region": region,
        "residue_index": residue_index,
    }

    to_return_lengths = {x: len(to_return[x]) for x in to_return}
    if len(set(to_return_lengths.values())) > 1:
        raise ValueError(
            "Expected all return values to have the same length. Got"
            f" {to_return_lengths}."
        )

    return to_return


def cdr_check(cdr_mask: torch.Tensor) -> None:
    """Get unique values of runs and check they are as expected."""
    changes = torch.where(cdr_mask.diff() != 0)[0]
    segment_value_idx = torch.cat((torch.tensor([0]), changes + 1))
    segment_values = cdr_mask[segment_value_idx]
    assert torch.allclose(
        segment_values, torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    )


def add_sequence_and_aatype(
    chain_information: dict[str, np.ndarray | list[str] | torch.Tensor],
) -> dict[str, np.ndarray | list[str] | torch.Tensor]:
    sequence = "".join(
        [
            residue_constants.restype_3to1.get(aa, "X")
            for aa in chain_information["sequence_3_letters"]
        ]
    )
    aatype = torch.tensor(
        [residue_constants.restype_order_with_x[aa] for aa in sequence]
    )

    chain_information.update({"sequence": sequence, "aatype": aatype})
    return chain_information


def add_openfold_data_transforms(
    chain_information: dict[str, np.ndarray | list[str] | torch.Tensor],
) -> dict[str, np.ndarray | list[str] | torch.Tensor]:
    # this is a hack because openfold will put an aatype as unknown even if the residue
    # is known but the structural information is missing. Without this hack the
    # atom14 mask and other various quantities will be incorrect
    aatype = chain_information["aatype"].clone()

    chain_information["aatype"][~chain_information["seq_mask"].bool()] = (
        residue_constants.restype_order_with_x["X"]
    )

    atom14_transforms = [
        data_transforms.make_atom14_masks,
        data_transforms.make_atom14_positions,
    ]

    for transform in atom14_transforms:
        chain_information = transform(chain_information)

    rest_of_transforms = [
        data_transforms.atom37_to_frames,
        data_transforms.atom37_to_torsion_angles(""),
        data_transforms.make_pseudo_beta(""),
        data_transforms.get_backbone_frames,
        data_transforms.get_chi_angles,
    ]

    chain_information["all_atom_positions"] = chain_information[
        "all_atom_positions"
    ].double()
    for transform in rest_of_transforms:
        chain_information = transform(chain_information)

    to_cast_to_float = [
        "all_atom_positions",
        "atom14_gt_positions",
        "torsion_angles_sin_cos",
        "alt_torsion_angles_sin_cos",
        "pseudo_beta",
    ]
    for tensor in to_cast_to_float:
        chain_information[tensor] = chain_information[tensor].float()

    # return aatype to be non-masked version
    chain_information["aatype"] = aatype
    return chain_information


def add_region_numeric(
    chain_information: dict[str, np.ndarray | list[str] | torch.Tensor],
) -> dict[str, np.ndarray | list[str] | torch.Tensor]:
    mapping = {
        "cdrh1": 0,
        "cdrh2": 1,
        "cdrh3": 2,
        "cdrl1": 3,
        "cdrl2": 4,
        "cdrl3": 5,
        "fwh1": 6,
        "fwh2": 7,
        "fwh3": 8,
        "fwh4": 9,
        "fwl1": 10,
        "fwl2": 11,
        "fwl3": 12,
        "fwl4": 13,
    }

    region_numeric = torch.tensor([mapping[x] for x in chain_information["region"]])
    chain_information["region_numeric"] = region_numeric
    return chain_information


def merge_chain_dict(
    heavy_dict: dict[str, np.ndarray | list[str] | torch.Tensor],
    light_dict: dict[str, np.ndarray | list[str] | torch.Tensor],
) -> dict[str, np.ndarray | list[str] | torch.Tensor]:
    if heavy_dict.keys() != light_dict.keys():
        raise ValueError(
            f"Input dicts should have same keys, got:",
            heavy_dict.keys(),
            "and",
            light_dict.keys(),
            sep="\n\n",
        )
    combined_dict = {}
    for key in heavy_dict:
        heavy_value, light_value = heavy_dict[key], light_dict[key]
        if isinstance(heavy_value, torch.Tensor):
            combined_dict[key] = torch.cat((heavy_value, light_value), dim=0)
        elif isinstance(heavy_value, str) or isinstance(heavy_value, list):
            combined_dict[key] = heavy_value + light_value
        else:
            raise ValueError("Unrecognised type", type(heavy_value))
    return combined_dict


def check_chain_length(
    merged_chain_information: dict[str, np.ndarray | list[str] | torch.Tensor],
    variable_domains: list[VariableDomain],
) -> None:
    output_lengths = {}
    numbering = {
        chain.chain_type: chain.get_numbering(scheme="imgt")
        for chain in variable_domains
    }
    output_lengths["numbering"] = len(numbering["H"]) + len(numbering["L"])
    for key in merged_chain_information:
        if key not in [
            "structure",
            "pdb",
            "resolution",
        ]:
            output_lengths[key] = len(merged_chain_information[key])
    if len(set(output_lengths.values())) > 1:
        raise ValueError(
            f"Expected all return values to have the same length. Got {output_lengths}."
        )
    # check numbering and is_heavy
    if len(numbering["H"]) != merged_chain_information["is_heavy"].sum().item():
        raise ValueError(
            f'Got {len(numbering["H"])=} but'
            f' {merged_chain_information["is_heavy"].sum().item()=}'
        )


def get_chain_informations(
    chains: list,
) -> list[dict[str, np.ndarray | list[str] | torch.Tensor]]:
    chain_informations = []
    for chain in chains:
        chain_information = extract_chain_information(chain)
        chain_information = add_region_numeric(chain_information)

        chain_information = add_sequence_and_aatype(chain_information)

        chain_information = add_openfold_data_transforms(chain_information)

        chain_informations.append(chain_information)
    return chain_informations


def get_fv_information(
    fv: Fv, resolution: float
) -> tuple[Protein, dict[str, np.ndarray | list[str] | torch.Tensor]]:
    # variables used to track failures
    # process Fv
    pdb_code = fv.full_id[0]
    # check Fv is just a heavy and light chain
    try:
        if fv.fv_type != "VH-VL":
            raise ValueError(f"Not a VH-VL antibody. {fv.fv_type=}.")
    except Exception as e:
        reason_for_failure = f"{type(e).__name__}: {str(e)}"
        location_of_failure = "VH-VL check"
        raise ErrorWithLocation(reason_for_failure, location_of_failure)

    # check chains are as expected
    try:
        chains = fv.get_variable_domains()
        assert (
            len(chains) == 2
            and chains[0].chain_type == "H"
            and chains[1].chain_type == "L"
        )
    except Exception as e:
        reason_for_failure = f"{type(e).__name__}: {str(e)}"
        location_of_failure = "chain_check"
        raise ErrorWithLocation(reason_for_failure, location_of_failure)

    try:
        chain_informations = get_chain_informations(chains)
    except Exception as e:
        reason_for_failure = f"{type(e).__name__}: {str(e)}"
        location_of_failure = "extract_chain_information"
        raise ErrorWithLocation(reason_for_failure, location_of_failure)

    # combine all data
    try:
        merged_chain_information = merge_chain_dict(*chain_informations)
        merged_chain_information.update(
            {
                "structure": f"{pdb_code}_{fv.id}",
                "pdb": pdb_code,
                "resolution": resolution,
            }
        )
    except Exception as e:
        reason_for_failure = f"{type(e).__name__}: {str(e)}"
        location_of_failure = "combine"
        raise ErrorWithLocation(reason_for_failure, location_of_failure)

    # add full coords
    try:
        merged_chain_information["atom37_gt_positions"] = atom14_to_atom37(
            merged_chain_information["atom14_gt_positions"], merged_chain_information
        )
    except Exception as e:
        reason_for_failure = f"{type(e).__name__}: {str(e)}"
        location_of_failure = "atom37"
        raise ErrorWithLocation(reason_for_failure, location_of_failure)

    # check output lengths are consistent
    try:
        check_chain_length(merged_chain_information, fv.get_variable_domains())
    except Exception as e:
        reason_for_failure = f"{type(e).__name__}: {str(e)}"
        location_of_failure = "output_length_check"
        raise ErrorWithLocation(reason_for_failure, location_of_failure)

    # cdr check
    try:
        cdr_check(merged_chain_information["cdr_mask"])
    except Exception as e:
        reason_for_failure = f"{type(e).__name__}: {str(e)}"
        location_of_failure = "cdr_check"
        raise ErrorWithLocation(reason_for_failure, location_of_failure)

    # Process protein
    try:
        processed_pdb = Protein(
            aatype=merged_chain_information["aatype"].numpy(),
            atom_positions=merged_chain_information["atom37_gt_positions"].numpy(),
            atom_mask=merged_chain_information["all_atom_mask"].numpy(),
            residue_index=merged_chain_information["residue_index"].numpy(),
            b_factors=np.zeros_like(merged_chain_information["all_atom_mask"].numpy()),
            chain_index=1 - merged_chain_information["is_heavy"].numpy().astype(int),
        )
    except Exception as e:
        reason_for_failure = f"{type(e).__name__}: {str(e)}"
        location_of_failure = "process_protein"
        raise ErrorWithLocation(reason_for_failure, location_of_failure)
    return processed_pdb, merged_chain_information


def process_fv(
    fv: Fv,
    processed_pdb_path: Path,
    structure_path: Path,
    resolution: float,
) -> dict[str, bool | str]:
    pdb_code = fv.full_id[0]
    try:
        processed_pdb, merged_chain_information = get_fv_information(fv, resolution)
    except ErrorWithLocation as e:
        success_dict = {
            "structure": f"{pdb_code}_{fv.id}",
            "pdb": pdb_code,
            "fv": fv.id,
            "success": False,
            "reason_for_failure": e.reason_for_failure,
            "location_of_failure": e.location_of_failure,
        }
        return success_dict
    try:
        torch.save(merged_chain_information, structure_path / f"{pdb_code}_{fv.id}.pt")
        with (processed_pdb_path / f"{pdb_code}_{fv.id}.pdb").open("w") as f:
            f.write(fix_pdb(io.StringIO(to_pdb(processed_pdb)), {}))
    except Exception as e:
        success_dict = {
            "structure": f"{pdb_code}_{fv.id}",
            "pdb": pdb_code,
            "fv": fv.id,
            "success": False,
            "reason_for_failure": f"{type(e).__name__}: {str(e)}",
            "location_of_failure": "save_pdb",
        }
        return success_dict
    success_dict = {
        "structure": f"{pdb_code}_{fv.id}",
        "pdb": pdb_code,
        "fv": fv.id,
        "success": True,
        "reason_for_failure": "",
        "location_of_failure": "",
    }
    return success_dict


def process_pdb(
    pdb: SAbDabEntry, processed_pdb_path: Path, structure_path: Path
) -> tuple[int, list[dict[str, bool | str]]]:
    # iterate over fvs
    fvs = pdb.structure.get_fvs()
    success_dict_list = []
    fv_pbar = tqdm(total=len(fvs), desc="Processing", leave=False, disable=False)
    observed_fvs_in_pdb = set()
    for fv in fvs:
        fv_pbar.update(1)
        if fv.id in observed_fvs_in_pdb:
            continue
        observed_fvs_in_pdb.add(fv.id)
        success_dict = process_fv(
            fv, processed_pdb_path, structure_path, pdb.structure_quality.resolution
        )
        success_dict_list.append(success_dict)

    return len(fvs), success_dict_list


def main(
    path: str = typer.Option("data", help="Path to save data."),
    chunk_id: int = typer.Option(help="Id of the chunk to process"),
    chunk: int = typer.Option(help="Process a total of chunk pdbs."),
):
    db = GideonInterface()
    # read pdbs
    path = Path(path)
    pdb_path = path / "pdbs" / f"pdbs_{chunk}_{chunk_id}.json"
    with pdb_path.open("r") as f:
        pdbs = json.load(f)

    entries = db.search_sabdab(pdb_codes=pdbs)
    # location of pytorch structures
    structure_path = path / "structures" / "structures"
    structure_path.mkdir(exist_ok=True, parents=True)

    # location of csvs recording if the computation was successful
    success_df_path = path / "structures" / "success_df"
    success_df_path.mkdir(exist_ok=True)

    # processed pdb path
    processed_pdb_path = path / "structures" / "processed_pdbs"
    processed_pdb_path.mkdir(exist_ok=True)

    success_dict_list = []
    pbar = tqdm(total=len(pdbs), desc="Processing", disable=False)
    total_fvs = 0
    for pdb in entries:
        pbar.set_description(f"Processing {pdb.pdb_code}")
        n_fvs, success_dict_list_pdb = process_pdb(
            pdb, processed_pdb_path, structure_path
        )
        total_fvs += n_fvs
        success_dict_list += success_dict_list_pdb
        pbar.update(1)
    success_df = pd.DataFrame.from_records(
        success_dict_list,
    )
    success_df.to_csv(success_df_path / f"{chunk}_{chunk_id}.csv", index=False)
    assert len(success_df) == total_fvs


if __name__ == "__main__":
    typer.run(main)
