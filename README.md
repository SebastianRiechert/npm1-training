# npm1-training
training pipeline for image classification of npm1 mutation status from bone marrow smears

## Programming language
- Python

## Hardware and OS
The code has been developed and tested with linux on the IBM Power9 CPU architecture (ppc64le) using NVIDIA Tesla V100 GPUs.
This version of the code requires only a single NVIDIA Tesla V100 GPU, 4 CPU cores and 4gb of RAM to run.

## Dependencies
All dependencies are listed in `environment.yml`

## Installation
We recommend using a virtual conda environment for installation. Installation takes 5 minutes:
```
conda env create -f environment.yml
```
Activate the environment using:
```
conda activate npm1_classification
```

## Data
The images required for training are released under KAGGLELINK.

## Training
Train a single model using default parameters. Training 50 epochs takes 40 minutes on the specified hardware.
```
python train.py --data-path datasets/npm1_wsi.csv --imgs_path kaggle/wsi
```
- `--imgs-path` must point to the folder containing the images from the dataset.
- use `--data-path` to train on a csv-file (containing img_id to label mappings) which get split automatically into train- and test-sets or
- use `--data-paths` to point to two csv-files containing pre-split datasets
```
python train.py --data-paths datasets/train_set.csv datasets/test_set.csv --imgs_path kaggle/wsi
```
Display configurable training parameters:
```
python train.py -h
```

Hyperparametersearch:
- create experiment:
```
python create_experiment_study.py --name demo
```
- launch hyperparametersearch:
```
python hyper_wsi.py --data-path datasets/npm1_wsi.csv --experiment_name demo
```
The hyperparameter-space to search is defined and can be changed in `hyper_wsi.py`

To view training metrics, launch the MLflow-UI:
```
mlflow ui
```