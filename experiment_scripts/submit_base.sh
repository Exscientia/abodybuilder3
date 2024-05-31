# activate env
conda activate .venv/

# pull data
dvc pull

# run script
dvc exp run \
    -S base.cmd="srun python" \
    -S base.debug=false \
    -S loss.plddt.weight=0 \
    -S finetune.epochs=400 \
    -S finetune.early_stopping=400 \
    -S finetune.metric=plddt

# push results
dvc exp push origin --rev HEAD