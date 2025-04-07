#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

if [ -z "$1" ]; then
    echo "Usage: $0 <env_name>"
    exit 1
fi


# Command relevant only for compute canada, comment out if using other machines
# module --force purge
# echo "Loading module done"

# Command relevant only for compute canada, comment out if using other machines
# module load python/3.10 StdEnv/2023 opencv/4.10.0 cuda/12.2 rust/1.76.0 git-lfs postgresql/16.0
ENV_NAME=$1
VENV_DIR=~/$ENV_NAME

echo "Creating new virtual environment: $ENV_NAME with Python 3.10"

# Create virtual environment with Python 3.10
virtualenv $VENV_DIR --python=python3.10
source $VENV_DIR/bin/activate

echo "Virtual environment activated"

# Upgrade pip
pip install --upgrade pip

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Install mmcv 2.2.0 from the specified URL
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html

cd ../../
# Install additional dependencies from multimodal.txt
pip install -r requirements/multimodal.txt

# Install local package (-e .)
pip install -v -e .

# Install local package from faster_coco_eval_repo
cd faster_coco_eval_repo
pip install -v -e .
cd ..

# Install local package from sahi
cd sahi
pip install -v -e .
cd ..

echo "Environment setup complete!"
