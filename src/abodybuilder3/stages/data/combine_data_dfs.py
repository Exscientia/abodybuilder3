from pathlib import Path

import pandas as pd
import typer


def main(
    output_dir: str = typer.Option("data", help="Path to save data."),
) -> None:
    output_dir_path = Path(output_dir)

    # success_df
    dfs = []
    for fname in (output_dir_path / "structures" / "success_df").iterdir():
        dfs.append(pd.read_csv(fname))
    success_df = pd.concat(dfs)

    # summary_df
    dfs = []
    for fname in (output_dir_path / "structures" / "summary_df").iterdir():
        dfs.append(pd.read_csv(fname))
    summary_df = pd.concat(dfs)

    assert len(summary_df) == len(success_df.query("success"))
    success_df.to_csv(output_dir_path / "structures" / "success_df.csv", index=False)
    summary_df.to_csv(output_dir_path / "structures" / "summary_df.csv", index=False)


if __name__ == "__main__":
    typer.run(main)
