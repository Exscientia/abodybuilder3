import subprocess
import tarfile
from pathlib import Path
from typing import Optional

import dvc.api
import pandas as pd
import torch
from tqdm import trange

from abodybuilder3.language.model import PairedIgT5, ProtBert, ProtT5, ProtTrans


def process_chunk(
    model: ProtTrans,
    sub_df: pd.DataFrame,
    structures_path: Path,
    structures_plm_path: Path,
):
    heavy_sequences, light_sequences = list(sub_df["heavy"]), list(sub_df["light"])
    embeddings = model.get_embeddings(heavy_sequences, light_sequences)
    for structure_name, embedding in zip(sub_df.index, embeddings):
        structure = torch.load(structures_path / f"{structure_name}.pt")
        if structure["all_atom_positions"].size(0) != embedding.size(0):
            raise ValueError(f"Embedding size not correct for {structure_name=}.")
        structure["plm_embedding"] = embedding
        torch.save(structure, structures_plm_path / f"{structure_name}.pt")


def write_compressed_output(filename: Path, output_path: Optional[Path] = None) -> None:
    if output_path is not None:
        subprocess.run(f"tar -czf {filename} {output_path}", shell=True)
    else:
        # writes an empty .tar.gz file
        with tarfile.open(filename, "w:gz") as tar:
            pass


def main(model: Optional[str], chunk_size: int):
    path = Path("data")
    structures_path = path / "structures" / "structures"
    structures_plm_path = path / "structures" / "structures_plm"
    structures_plm_path.mkdir(exist_ok=True)
    structures_plm_compressed_file = path / "structures" / "structures_plm.tar.gz"

    if model is None:
        write_compressed_output(structures_plm_compressed_file)
        return
    elif model == "protbert":
        model = ProtBert()
    elif model == "prott5":
        model = ProtT5()
    elif model == "igt5-paired":
        model = PairedIgT5()
    else:
        raise ValueError(
            f"Expected model to be 'prott5' or 'igt5-paired'. Instead got {model=}."
        )

    df = pd.read_csv(path / "structures" / "summary_df.csv").set_index("structure")
    for start_idx in trange(0, len(df), chunk_size, desc="Chunk"):
        end_idx = min(start_idx + chunk_size, len(df))
        sub_df = df.iloc[start_idx:end_idx]
        process_chunk(model, sub_df, structures_path, structures_plm_path)
    write_compressed_output(structures_plm_compressed_file, structures_plm_path)


if __name__ == "__main__":
    main(**dvc.api.params_show()["language"])
