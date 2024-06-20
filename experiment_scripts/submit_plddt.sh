# activate env
conda activate .venv/

# pull data
dvc pull

# run script
dvc exp run \
    -S base.cmd="srun python" \
    -S base.debug=false

# push results
dvc exp push origin --rev HEAD