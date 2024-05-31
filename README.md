# ABodyBuilder3

Code for the paper "ABodyBuilder3: Improved and scalable antibody structure predictions"

# Code

## Download data from zenodo

Data and model weights are hosted at https://zenodo.org/records/11354577.

The bash script `download.sh` will download and extract data and model weights into
appropriate directories. 

## Filter and split data 

The repo comes with pre-specified data filtering (specified in `data/filters.csv`) and
splits (specified in `data/split.csv`). If you want to reproduce these steps then run 

1. `python src/abodybuilder3/stages/data/combine_data_dfs.py`
2. `python src/abodybuilder3/stages/data/filter_data.py`
3. `python src/abodybuilder3/stages/data/split_data.py`

## Embed sequences using language model 

Pre-computed language model embeddings are provided in `data/structures/structures_plm`
after running `download.sh`. If you wish to regenerate then run

`python src/abodybuilder3/stages/data/language_model_embeddings.py`

## Train model

The model can be trained using
 
1. `python src/abodybuilder3/stages/train.py`
2. `python src/abodybuilder3/stages/finetune.py`

## Inference and evaluation

The model can be used to predict structures from the validation and test set using 

`python src/abodybuilder3/stages/inference.py`

For general sequences inputs can be prepared using `string_to_input` from
`src/abodybuilder3/utils.py` and the language models in `src.abodybuilder3.language`.

## DVC

Our code is built using dvc pipelines, an alternative way to run the code is via `dvc exp run`.

# Citation

If this code is useful to you please cite our paper using the following bibtex entry.
