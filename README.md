# ABodyBuilder3

Code for the paper [ABodyBuilder3: Improved and scalable antibody structure predictions](https://arxiv.org/abs/2405.20863).

# Code

## Download data from zenodo

Data and model weights are hosted at https://zenodo.org/records/11354577.

The bash script `download.sh` will download and extract data and model weights into
appropriate directories. 

If you only require model weights for inference, these can be downloaded and extracted with the following commands.
```
mkdir -p zenodo
mkdir -p output/
wget -P zenodo/ https://zenodo.org/records/11354577/files/output.tar.gz
tar -xzvf zenodo/output.tar.gz -C output/
```

## Notebook example

A simple example of using the model is given in `notebooks/example.ipynb`.

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

For general sequences inputs can be prepared following the examples given in `notebooks/example.ipynb`.

## DVC

Our code is built using dvc pipelines, an alternative way to run the code is via `dvc
exp run`. See `experiment_scripts` for the configurations we used for the experiments in
the manuscript.

# Citation

If this code is useful to you please cite our paper using the following bibtex entry.

```
@article{kenlay2024abodybuilder3,
  title={ABodyBuilder3: Improved and scalable antibody structure predictions},
  author={Kenlay, Henry and Dreyer, Fr{\'e}d{\'e}ric A and Cutting, Daniel and Nissley, Daniel and Deane, Charlotte M},
  journal={arXiv preprint arXiv:2405.20863},
  year={2024}
}
```