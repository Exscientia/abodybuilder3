#!/bin/bash

mkdir -p zenodo
wget -P zenodo/ https://zenodo.org/records/11354577/files/data.tar.gz
wget -P zenodo/ https://zenodo.org/records/11354577/files/structures_plm.tar.gz
wget -P zenodo/ https://zenodo.org/records/11354577/files/output.tar.gz

mkdir -p output/
tar -xzvf zenodo/data.tar.gz 
tar -xzvf zenodo/structures_plm.tar.gz
tar -xzvf zenodo/output.tar.gz -C output/
