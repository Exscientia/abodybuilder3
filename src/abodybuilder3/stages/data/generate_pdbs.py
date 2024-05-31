"""
Generates a JSON file of pdbs used in dataset. The data is chunked so the rest of the data preprocessing pipeline (hpc_scripts/data/generate_data.py) easy to parallelise.

Files will be saved in the following structure

arg.path/
├─ pdbs/
│  ├─ pdbs_{chunk}_0.json
│  ├─ pdbs_{chunk}_1.json
⋮   ⋮

Each json file contains a list of pdbs of size chunk.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import typer
from exs.sabdab import GideonInterface


def main(
    path: str = typer.Option("data", help="Path to save data."),
    chunk: int = typer.Option(50, help="Save name of pdbs in chunks of this size."),
    rescut: float = typer.Option(99999, help="Resolution cutoff for entries."),
    gideon_username: str = typer.Option(
        default=None, help="Gideon username, if not set then try environment variable."
    ),
    gideon_password: str = typer.Option(
        default=None, help="Gideon password, if not set then try environment variable."
    ),
    gideon_sql_password: str = typer.Option(
        default=None,
        help="Gideon sql password, if not set then try environment variable.",
    ),
):
    if os.getenv("GIDEON_USERNAME") is None and gideon_username is None:
        raise ValueError(
            "Either GIDEON_USERNAME environment variable or gideon_username argument must be set."
        )
    if os.getenv("GIDEON_PASSWORD") is None and gideon_password is None:
        raise ValueError(
            "Either GIDEON_PASSWORD environment variable or gideon_password argument must be set."
        )
    if os.getenv("GIDEON_SQL_PASSWORD") is None and gideon_sql_password is None:
        raise ValueError(
            "Either GIDEON_SQL_PASSWORD environment variable or gideon_sql_password argument must be set."
        )

    db = GideonInterface(
        username=gideon_username,
        password=gideon_password,
        password_sql=gideon_sql_password,
    )

    entries = db.search_sabdab(resolution_lower_than=rescut)
    pdb_codes = [entry.pdb_code for entry in entries]
    # save full output
    path = Path(path) / "pdbs"
    path.mkdir(exist_ok=True, parents=True)
    with open(path / "pdbs.json", "w") as f:
        json.dump(pdb_codes, f)

    # save chunked data
    for idx, start in enumerate(np.arange(0, len(pdb_codes), chunk)):
        end = min(start + chunk, len(pdb_codes))
        with open(path / f"pdbs_{chunk}_{idx}.json", "w") as f:
            json.dump(pdb_codes[start:end], f)


if __name__ == "__main__":
    typer.run(main)
