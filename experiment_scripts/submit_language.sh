# activate env
conda activate .venv/

# pull data
dvc pull

# run script
dvc exp run \
    -S base.cmd="srun python" \
    -S base.debug=false \
    -S language.model=prott5

# push results
dvc exp push origin --rev HEAD