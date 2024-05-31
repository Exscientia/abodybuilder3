"""
A new script to split data into train, validation and test.

The original validation and test set (which I call test sets for the rest of this
docstring) will be subsets of the new test sets. New validation and test sets will
be human only structures at a max resolution of 2.3 (similar to the original test set).

Contamination is avoided by sequence similirty. Further contamination can be avoided by considering sequence clusters. 

The output of this script is split.csv which includes the following columns
- structure - unique identifier
- in_legacy - if the structure is in the original dataset
- split - train, valid, test or unassigned (repeat sequences or sequences in a test set in the same cluster are unassigned).
- legacy_test_split - for structures in the test sets, this is a boolean indicating if its in the original validation or test set 

Furthermore, another dataframe is outputted called valid_test_set_similarity.csv which for each element of the test sets gives the Levenshtein to all sequences in the training set. This is done for full sequence and concatenaded cdr pseudo-sequences.
"""

from itertools import product
from pathlib import Path

import dvc.api
import numpy as np
import pandas as pd
from Levenshtein import ratio
from loguru import logger
from tqdm import tqdm

# from https://github.com/oxpig/ImmuneBuilder/blob/main/data/antibody_data.csv
validation_set = [
    "1mlb_B0-A0",
    "2fb4_H0-L0",
    "3nps_B0-C0",
    "3g5y_B0-A0",
    "1jfq_H0-L0",
    "2w60_A0-B0",
    "3lmj_H0-L0",
    "2e27_H0-L0",
    "2v17_H0-L0",
    "3ifl_H0-L0",
    "2fbj_H0-L0",
    "1oaq_H0-L0",
    "3p0y_H0-L0",
    "3eo9_H0-L0",
    "1gig_H0-L0",
    "4f57_H0-L0",
    "4nzu_H0-L0",
    "4h0h_B0-B1",
    "3hnt_H0-L0",
    "3go1_H0-L0",
    "3gnm_H0-L0",
    "2vxv_H0-L0",
    "3mxw_H0-L0",
    "3i9g_H0-L0",
    "2adf_H0-L0",
    "2r8s_H0-L0",
    "3e8u_H0-L0",
    "1mqk_H0-L0",
    "3hc4_H0-L0",
    "3m8o_H0-L0",
    "3umt_A0-A1",
    "2xwt_A0-B0",
    "4hpy_H0-L0",
    "2d7t_H0-L0",
    "1nlb_H0-L0",
    "4h20_H0-L0",
    "1fns_H0-L0",
    "3giz_H0-L0",
    "1dlf_H0-L0",
    "3liz_H0-L0",
    "1seq_H0-L0",
    "3v0w_H0-L0",
    "2ypv_H0-L0",
    "1mfa_H0-L0",
    "3oz9_H0-L0",
    "1jpt_H0-L0",
]

# this comes from personal communication with Brennan
test_set = [
    "7vux_H0-L0",
    "7zf6_H0-L0",
    "7tp4_H0-L0",
    "7ps3_H0-L0",
    "7sg5_H0-L0",
    "7rt9_C0-D0",
    "7ue9_H0-L0",
    "7t0j_H0-L0",
    "7ryu_H0-L0",
    "7so5_H0-L0",
    "7urq_H0-L0",
    "7sg6_H0-L0",
    "7ps6_H0-L0",
    "7ttm_H0-L0",
    "7sjs_H0-L0",
    "7q0g_H0-L0",
    "7q4q_D0-C0",
    "7vyr_H0-L0",
    "7sem_B0-C0",
    "7rxl_H0-L0",
    "7qu1_A0-B0",
    "7seg_H0-L0",
    "7zwi_B0-C0",
    "7u8c_H0-L0",
    "7z0x_H0-L0",
    "7t0i_D0-C0",
    "7ps4_B0-C0",
    "7rqr_A0-B0",
    "7sn1_H0-L0",
    "7rqq_H0-L0",
    "7rxi_H0-L0",
    "7s4g_H0-L0",
    "7qf0_H0-L0",
    "7rp2_H0-I0",
]


def main(
    path: str,
    seed: int,
    validation_size: int,
    test_size: int,
    valid_test_max_cluster_size: int,
    test_valid_resolution_cutoff: float,
    test_valid_cdrh3_cutoff: int,
    use_cdr_pseudosequence: bool,
):
    if test_size < len(test_set):
        raise ValueError(
            "Original test set is included in the new one. Therefore test_size needs"
            f" to be at least {len(test_set)}. Got {test_size=}"
        )

    path = Path("data")
    np.random.seed(seed)
    summary_df = pd.read_csv(path / "structures" / "summary_df.csv", index_col=0)
    filter_df = pd.read_csv(path / "filters.csv", index_col=0)

    # remove excluded
    split_df = filter_df.query("not exclude")[["exclude"]]

    # add resolution, species, cdrh3_length and number of missing cdr residues which will be used for valid/test
    split_df = split_df.join(summary_df[["resolution", "species"]])
    split_df = split_df.join(summary_df["cdrh3"].map(len).rename("cdrh3_length"))
    split_df["missing_residues"] = (
        summary_df["missing_residues_cdr"] + summary_df["missing_residues_fw"]
    )

    # split based on either sequence of cdr pseudosequence
    summary_df["sequence"] = summary_df["heavy"] + "/" + summary_df["light"]
    summary_df["cdr_pseudosequence"] = (
        summary_df["cdrl1"]
        + summary_df["cdrl2"]
        + summary_df["cdrl3"]
        + summary_df["cdrh1"]
        + summary_df["cdrh2"]
        + summary_df["cdrh3"]
    )

    split_df = split_df.join(summary_df[["sequence"]])
    if use_cdr_pseudosequence:
        split_df["cluster"] = summary_df["cdr_pseudosequence"]
    else:
        split_df["cluster"] = split_df["sequence"]

    # in legacy
    split_df["pdb"] = split_df.index.map(lambda x: x.split("_")[0])
    legacy_pdbs = set(pd.read_csv(path / "legacy" / "antibody_data.csv")["PDB"])
    split_df["in_legacy"] = split_df["pdb"].map(lambda x: x in legacy_pdbs)

    # assign legacy test split
    split_df["split"] = "unassigned"
    split_df["legacy_test_split"] = False
    split_df.loc[test_set, "split"] = "test"
    split_df.loc[test_set, "legacy_test_split"] = True

    # cluster size - valid/test wont be selected from big clusters
    split_df["cluster_size"] = split_df.groupby("cluster")["cluster"].transform("size")

    # assign a validation set
    test_clusters = set(split_df.query('split =="test"')["cluster"])
    valid_pool = (
        split_df.query("cluster not in @test_clusters")
        .query(f"cluster_size <= {valid_test_max_cluster_size}")
        .query(f"resolution < {test_valid_resolution_cutoff}")
        .query(f"resolution > 0")
        .query(f"missing_residues == 0")
        .query('species=="HOMO SAPIENS"')
        .query(f"cdrh3_length <= {test_valid_cdrh3_cutoff}")
    ).drop_duplicates("cluster")

    new_valid_structures = np.random.choice(
        valid_pool.index, validation_size, replace=False
    )
    split_df.loc[new_valid_structures, "split"] = "valid"

    # assign a rest of the test set
    to_sample = test_size - len(test_set)
    valid_and_test_clusters = set(
        split_df.query('split in ["valid", "test"]')["cluster"]
    )

    test_pool = (
        split_df.query("cluster not in @valid_and_test_clusters")
        .query(f"cluster_size <= {valid_test_max_cluster_size}")
        .query(f"resolution < {test_valid_resolution_cutoff}")
        .query(f"resolution > 0")
        .query(f"missing_residues == 0")
        .query('species=="HOMO SAPIENS"')
        .query(f"cdrh3_length <= {test_valid_cdrh3_cutoff}")
    ).drop_duplicates("cluster")

    new_test_structures = np.random.choice(test_pool.index, to_sample, replace=False)
    split_df.loc[new_test_structures, "split"] = "test"

    # assign remaining clusters to the train set
    valid_and_test_clusters = set(
        split_df.query('split in ["valid", "test"]')["cluster"]
    )
    train_clusters = split_df.query("cluster not in @valid_and_test_clusters")[
        "cluster"
    ]
    train_structures = split_df.query("cluster in @train_clusters").index
    split_df.loc[train_structures, "split"] = "train"

    # sanity check
    splits = ["train", "valid", "test"]
    for split1, split2 in product(splits, splits):
        if split1 == split2:
            continue
        assert (
            set(split_df.query(f'split=="{split1}"')["cluster"]).intersection(
                set(split_df.query(f'split=="{split2}"')["cluster"])
            )
            == set()
        )
        assert (
            set(split_df.query(f'split=="{split1}"')["sequence"]).intersection(
                set(split_df.query(f'split=="{split2}"')["sequence"])
            )
            == set()
        )
    assert set(test_set).issubset(set(split_df.query('split=="test"').index))
    for split in ["valid", "test"]:
        assert np.all(split_df.query(f'split=="{split}"')["resolution"] <= 2.3)
        assert np.all(
            split_df.query(f'split=="{split}" and not legacy_test_split')["species"]
            == "HOMO SAPIENS"
        )

    counts = split_df["split"].value_counts()
    assert counts["valid"] == validation_size
    assert counts["test"] == test_size
    for split in ["train", "valid", "test", "unassigned"]:
        logger.info(f"Split {split} size: {counts[split]}")

    split_df[["in_legacy", "split", "legacy_test_split"]].to_csv(path / "split.csv")

    # measure distance to training set
    cdr_columns = ["cdrh1", "cdrh2", "cdrh3", "cdrl1", "cdrl2", "cdrl3"]
    summary_df = summary_df[cdr_columns].dropna()
    summary_df["cdr"] = summary_df[cdr_columns].sum(axis=1)
    split_df = split_df.join(summary_df["cdr"])

    valid_and_test_df = split_df.query("split in ['test', 'valid']")
    train_df = split_df.query("split =='train'")
    total = len(valid_and_test_df) * len(train_df)

    similarity_df = []
    pbar = tqdm(total=total)
    for structure, row in split_df.query("split in ['test', 'valid']").iterrows():
        for train_structure, train_row in split_df.query("split=='train'").iterrows():
            similarity_df.append(
                (
                    structure,
                    train_structure,
                    ratio(row["sequence"], train_row["sequence"]),
                    ratio(row["cdr"], train_row["cdr"]),
                )
            )
            pbar.update(1)

    similarity_df = pd.DataFrame(
        similarity_df,
        columns=[
            "structure",
            "train_structure",
            "sequence_similarity",
            "cdr_similarity",
        ],
    )
    similarity_df.to_csv(path / "valid_test_set_similarity.csv", index=False)


if __name__ == "__main__":
    params = dvc.api.params_show()
    main(path=params["base"]["data_dir"], **dvc.api.params_show()["split"])
