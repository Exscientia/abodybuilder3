from pathlib import Path
from typing import Optional

import dvc.api
import numpy as np
import pandas as pd


def main(
    path: str = "data",
    resolution_cutoff: Optional[float] = None,
    abangle_cutoff: Optional[float] = None,
    cdrh3_length_cutoff: Optional[int] = None,
    total_cdr_length_cutoff: Optional[int] = None,
    rare_species_cutoff: Optional[int] = None,
):
    # read summary df
    path = Path(path)
    try:
        summary_df = pd.read_csv(path / "structures" / "summary_df.csv")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e.strerror}: {e.filename}.\n\n Did you run scripts/data/combine_data_dfs.py?."
        )
    summary_df = summary_df.set_index("structure").rename_axis("structure")

    # compute filter_df
    filter_df = filter_df_from_summary_df(
        summary_df=summary_df,
        resolution_cutoff=resolution_cutoff,
        abangle_cutoff=abangle_cutoff,
        cdrh3_length_cutoff=cdrh3_length_cutoff,
        total_cdr_length_cutoff=total_cdr_length_cutoff,
        rare_species_cutoff=rare_species_cutoff,
    )
    filter_df.to_csv(path / "filters.csv")


def filter_df_from_summary_df(
    summary_df: pd.DataFrame,
    resolution_cutoff: Optional[float] = None,
    abangle_cutoff: Optional[float] = None,
    cdrh3_length_cutoff: Optional[int] = None,
    total_cdr_length_cutoff: Optional[int] = None,
    rare_species_cutoff: Optional[int] = None,
) -> pd.DataFrame:
    """Create a filter dataframe from a summary dataframe.

    summary_df must contain columns resolution, abangle_HL, abangle_HC1, abangle_HC2,
    abangle_LC1, abangle_LC2, abangle_dc, cdrh3 and species.

    The output dataframe contains boolean columns of various filters as well as an
    exclude column which the dataset can be filtered on.

    Args:
        summary_df (pd.DataFrame): summary_df with required columns
        resolution_cutoff (float, optional): filter structures with resolutions above
        this value. Defaults to 3.5.
        abangle_cutoff (float, optional): abangle z-scores are computed, the absolute
        value of the z-score is above this value filter the structure. Defaults to 3.0.
        cdrh3_length_cutoff (int, optional): filter structures with cdrh3 loops above
        this value. Defaults to 30.
        total_cdr_lenght_cutoff (int, optional): filter structures where sum of cdr
        lengths are above this value. Defaults to 68.
        rare_species_cutoff (int, optional): filter out structures if it belongs to a
        species with less than rare_species_cutoff entries in sabdab. Defaults to 15.

    Returns:
        pd.DataFrame: A dataframe with various boolean columns and an exclude columns.
        The index is the same as summary_df.
    """

    # resolution filter
    filter_df = {}
    if resolution_cutoff is not None:
        filter_df["resolution_high"] = summary_df["resolution"] > resolution_cutoff

    # abangle outlier
    if abangle_cutoff is not None:
        abangle_columns = [
            "abangle_HL",
            "abangle_HC1",
            "abangle_HC2",
            "abangle_LC1",
            "abangle_LC2",
            "abangle_dc",
        ]

        for abangle_column in abangle_columns:
            abangle = summary_df[abangle_column]
            zscores = (abangle - abangle.mean()) / abangle.std()
            filter_df[f"{abangle_column}_outlier"] = (
                zscores.abs() > abangle_cutoff
            ) | abangle.isna()

    # sequence cdrh3 length outliers
    if cdrh3_length_cutoff is not None:
        filter_df["cdrh3_high"] = summary_df["cdrh3"].map(len) >= cdrh3_length_cutoff

    # sequence total cdr length outliers
    if total_cdr_length_cutoff is not None:
        cdr_columns = ["cdrh1", "cdrh2", "cdrh3", "cdrl1", "cdrl2", "cdrl3"]
        summary_df["cdr_total_len"] = sum(
            [summary_df[cdr].map(len) for cdr in cdr_columns]
        )
        filter_df["cdr_high"] = summary_df["cdr_total_len"] >= total_cdr_length_cutoff

    # species_filter
    if rare_species_cutoff is not None:
        filter_df["rare_species"] = (
            summary_df.groupby("species")["species"].transform("size")
            <= rare_species_cutoff
        )

    # make exclude column
    if filter_df == {}:
        filter_df = pd.DataFrame({"exclude": np.zeros(len(summary_df), dtype=bool)})
    else:
        filter_df = pd.DataFrame(filter_df)
        filter_df["exclude"] = filter_df.any(axis=1)

    # return filter_df
    filter_df = filter_df.set_index(summary_df.index).rename_axis(summary_df.index.name)
    return filter_df


if __name__ == "__main__":
    main(**dvc.api.params_show()["filter"])
