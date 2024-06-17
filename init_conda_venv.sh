#!/bin/bash

# Make sure to bypass a possible $PIP_REQUIRE_VIRTUALENV=true set by users
# This is because pip is unable to detect that it is inside conda's virtual environment - and would throw an error

if lspci | grep -i nvidia > /dev/null; then
    # Use environment file with cudatoolkit
    echo "Setting up GPU environment"
    environment="environment_gpu.yml"
else
    # Use environment file without cudatoolkit
    echo "Setting up CPU environment"
    environment="environment_cpu.yml"
fi

PIP_REQUIRE_VIRTUALENV=false conda env create -f ${environment} --prefix .venv

eval "$(conda shell.bash hook)"
conda activate .venv/
python --version

# install pip dependencies
PIP_REQUIRE_VIRTUALENV=false pip install -e ".[dev]" --constraint pinned-versions.txt
