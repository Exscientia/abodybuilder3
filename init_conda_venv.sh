#!/bin/bash

if ! git rev-parse --git-dir > /dev/null 2>&1; then
  : # This is not a valid git repository and will fail due to scm erroring, so tell the user
    echo "You have not initialised a git repo. Please run ./init_git.sh first and try again"
    exit
fi

_SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
pushd ${_SCRIPT_DIR} # cd to script directory

CONDA_OR_MAMBA=$(which mamba)
if [ -z $CONDA_OR_MAMBA ]; then
    CONDA_OR_MAMBA=$(which conda)
fi
if [ -z $CONDA_OR_MAMBA ]; then
    echo "No mamba or conda executable found."
    exit
fi

# https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/
set -euxo pipefail

echo "Setting up virtualenv"

# Make sure to bypass a possible $PIP_REQUIRE_VIRTUALENV=true set by users
# This is because pip is unable to detect that it is inside conda's virtual environment - and would throw an error

if lspci | grep -i nvidia > /dev/null; then
    # Use environment file with cudatoolkit
    environment="environment_gpu.yml"
else
    # Use environment file without cudatoolkit
    environment="environment_cpu.yml"
fi

env PIP_REQUIRE_VIRTUALENV=false ${CONDA_OR_MAMBA} env create -f ${environment} --prefix .venv

popd # return to original directory

conda activate .venv/
python --version

# install pip dependencies
pip install -e ".[dev]" --constraint pinned-versions.txt
