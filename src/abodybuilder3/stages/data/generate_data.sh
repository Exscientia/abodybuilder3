#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --mem=8G
#SBATCH --output=slurm_output/%A_%a.out

source ~/.init_conda # activate mamba
conda activate .venv/

python src/exs/abodybuilder2/stages/data/generate_data.py --path data --chunk 50 --chunk-id $SLURM_ARRAY_TASK_ID
python src/exs/abodybuilder2/stages/data/generate_data_summary.py --path data --chunk 50 --chunk-id $SLURM_ARRAY_TASK_ID