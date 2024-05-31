from pathlib import Path

import pandas as pd
import torch
import typer
from exs.sabdab import GideonInterface
from exs.scalop.predict import assign
from loguru import logger
from tqdm import tqdm

# sabdab database
db = GideonInterface()


def main(
    path: str = typer.Option("data", help="Path to save data."),
    chunk_id: int = typer.Option(0, help="Id of the chunk to process"),
    chunk: int = typer.Option(50, help="Process a total of chunk pdbs."),
):
    # read pdbs
    path = Path(path)
    path_df = path / "structures" / "success_df" / f"{chunk}_{chunk_id}.csv"
    success_df = pd.read_csv(path_df)

    if not success_df["success"].any():
        logger.warning(f"Dataframe {path_df=} has no successful entries.")
        return

    success_df = success_df.query("success")
    pdbs = list(success_df["pdb"].unique())
    entries = db.search_sabdab(pdb_codes=pdbs)

    df = []
    pbar = tqdm(total=len(pdbs), desc="Processing", disable=False)
    for pdb_entry in entries:
        pbar.set_description(f"Processing {pdb_entry.pdb_code}")

        # iterate over fvs
        fvs = pdb_entry.structure.get_fvs()
        fv_pbar = tqdm(total=len(fvs), desc="Processing", leave=False, disable=False)
        observed_fvs_in_pdb = set()
        for fv in fvs:
            # identifier
            try:
                fv_identifier = f"{pdb_entry.pdb_code}_{fv.id}"
                fv_df_row = [
                    fv_identifier,
                ]
                fv_pbar.set_description(f"Processing {fv_identifier}")
            except:
                continue

            if fv_identifier in observed_fvs_in_pdb:
                continue

            if fv_identifier not in list(success_df["structure"]):
                continue
            vh = fv.heavy
            vl = fv.light
            if vh is None or vl is None:
                logger.warning(
                    f"For {fv_identifier=} we did not get expected heavy-light"
                    " variable domains"
                )
                continue

            # chains
            try:
                fv_df_row.extend([vh.sequence, vl.sequence])
            except:
                fv_df_row.extend(
                    [
                        None,
                    ]
                    * 2
                )

            # cdr regions
            try:
                fv_df_row.extend(
                    [
                        vh.get_region_sequence(region=region, definition="imgt")
                        for region in [
                            "cdrh1",
                            "cdrh2",
                            "cdrh3",
                        ]
                    ]
                    + [
                        vl.get_region_sequence(region=region, definition="imgt")
                        for region in [
                            "cdrl1",
                            "cdrl2",
                            "cdrl3",
                        ]
                    ]
                )
            except:
                fv_df_row.extend(
                    [
                        None,
                    ]
                    * 6
                )

            # scallop classifications
            try:
                scallop_clusters = (
                    vh.scalop_canonicals["imgt"]["cdrh1"],
                    vh.scalop_canonicals["imgt"]["cdrh2"],
                    vl.scalop_canonicals["imgt"]["cdrl1"],
                    vl.scalop_canonicals["imgt"]["cdrl2"],
                    vl.scalop_canonicals["imgt"]["cdrl3"],
                )
                fv_df_row.extend(scallop_clusters)
            except:
                fv_df_row.extend(
                    [
                        None,
                    ]
                    * 5
                )

            # resolution
            try:
                fv_df_row.extend(
                    [
                        pdb_entry.structure_quality.resolution,
                    ]
                )
            except:
                fv_df_row.extend(
                    [
                        None,
                    ]
                )

            # abangles
            try:
                if fv.abangle_result.a_type == "H" and fv.abangle_result.b_type == "L":
                    fv_df_row.extend(
                        [
                            fv.abangle_result.ab,
                            fv.abangle_result.ac1,
                            fv.abangle_result.ac2,
                            fv.abangle_result.bc1,
                            fv.abangle_result.bc2,
                            fv.abangle_result.dc,
                        ]
                    )
                elif (
                    fv.abangle_result.a_type == "L" and fv.abangle_result.b_type == "H"
                ):
                    fv_df_row.extend(
                        [
                            fv.abangle_result.ab,
                            fv.abangle_result.bc1,
                            fv.abangle_result.bc2,
                            fv.abangle_result.ac1,
                            fv.abangle_result.ac2,
                            fv.abangle_result.dc,
                        ]
                    )
                else:
                    raise AssertionError("Unexpected abangle result types.")
            except:
                fv_df_row.extend(
                    [
                        None,
                    ]
                    * 6
                )

            # species

            try:
                fv_sources = [
                    species
                    for var in fv.get_variable_domains()
                    for species in var.parent.source_organisms
                ]
                fv_df_row.extend(
                    [
                        "|".join(set(fv_sources)),
                    ]
                )
            except:
                fv_df_row.extend(
                    [
                        None,
                    ]
                )

            # light chain type
            try:
                fv_df_row.append(fv.light.light_chain_type)
            except:
                fv_df_row.append(None)

            # VJC genes
            try:
                fv_df_row.extend(
                    [
                        vh.v_gene_annotation.germline.split("*")[0],
                        vh.j_gene_annotation.germline.split("*")[0],
                        vl.v_gene_annotation.germline.split("*")[0],
                        vl.j_gene_annotation.germline.split("*")[0],
                    ]
                )
            except:
                fv_df_row.extend([None] * 4)

            # number of missing residies in the CDR and Fw regions
            try:
                missing_residues = [res for res in fv.get_residues() if res.is_missing]

                cdr_missing = len(
                    [
                        res
                        for res in missing_residues
                        if "cdr" in res.get_region(definition="imgt")
                    ]
                )
                fw_missing = len(
                    [
                        res
                        for res in missing_residues
                        if "fw" in res.get_region(definition="imgt")
                    ]
                )
                fv_df_row.extend([cdr_missing, fw_missing])
            except:
                fv_df_row.extend([None, None])
            try:
                fv_df_row.append(fv.scfv)
            except:
                fv_df_row.append(None)
            # record entry
            df.append(fv_df_row)
            fv_pbar.update(1)
            observed_fvs_in_pdb.add(fv_identifier)

        pbar.update(1)

    df = pd.DataFrame(
        df,
        columns=[
            "structure",
            "heavy",
            "light",
            "cdrh1",
            "cdrh2",
            "cdrh3",
            "cdrl1",
            "cdrl2",
            "cdrl3",
            "cdrh1_cluster",
            "cdrh2_cluster",
            "cdrl1_cluster",
            "cdrl2_cluster",
            "cdrl3_cluster",
            "resolution",
            "abangle_HL",
            "abangle_HC1",
            "abangle_HC2",
            "abangle_LC1",
            "abangle_LC2",
            "abangle_dc",
            "species",
            "light_chain_type",
            "heavy_V_gene",
            "heavy_J_gene",
            "light_V_gene",
            "light_J_gene",
            "missing_residues_cdr",
            "missing_residues_fw",
            "is_scfv",
        ],
    )

    output_dir = path / "structures" / "summary_df"
    output_dir.mkdir(exist_ok=True, parents=True)
    df.to_csv(output_dir / f"{chunk}_{chunk_id}.csv", index=False)
    assert len(df) == len(success_df)


if __name__ == "__main__":
    typer.run(main)
