# npm1-training
training pipeline for image classification of npm1 mutation status from bone marrow smears

## Programming language
- Python

## Hardware and OS
The code has been developed and tested with linux on the IBM Power9 CPU architecture (ppc64le) using NVIDIA Tesla V100 GPUs.
This version of the code requires only a single NVIDIA Tesla V100 GPU, 4 CPU cores and 4gb of RAM to run.

## Dependencies
All dependencies are listed in environment.yml.

## Installation
We recommend using a virtual conda environment for installation. Installation takes 10 minutes:
´´´
conda env create -f environment.yml
´´´
Activate the environment using:
´´´
conda activate npm1_classification
´´´

## Usage
### Training a single model