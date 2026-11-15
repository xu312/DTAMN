# DTAMN

PyTorch implementation of **Dual-Topology Aware Meta-learning Network for Robust and Scalable Traffic Prediction**.

## Table of Contents

- `config_file/`: training and model configurations for each dataset.
- `lib/`: data loading, normalization, evaluation metrics, and utilities.
- `model/`: model implementation and trainer.
- `data/`: datasets and precomputed adjacency matrices.

## Requirements

Python, CUDA-enabled PyTorch, PyTorch Geometric, NumPy, SciPy, pandas, h5py, NetworkX, and nvidia-ml-py. An NVIDIA GPU with installed drivers is required.

## Data Preparation

Datasets and precomputed adjacency matrices will be provided via cloud storage.

**Download:** [Google Drive](https://drive.google.com/file/d/1xr8XHMYsZMKM4Q287Nl5bbfx6dwE3mmW/view?usp=drive_link)

Extract the downloaded files into `data/`, keeping each dataset and its `<dataset>_A.npy` and `<dataset>_cycle.npy` matrices under `data/<dataset>/`. No additional graph preprocessing is required.

## Model Training

Run from the repository root:

```bash
python run.py --dataset <DATASET_NAME> --mode train --device cuda:0
```

Supported datasets: `PEMS04`, `PEMS07`, `PEMS08`, `PEMS07(M)`, `PEMS07(L)`, `NYCBike`, and `NYCTaxi`.

For example:

```bash
python run.py --dataset PEMS08 --mode train --device cuda:0
```

Adjust hyperparameters in `config_file/<dataset>_DTAMN.conf`. Logs and checkpoints are saved to `experiments/<dataset>/<timestamp>/`.

For testing, set `checkpoint_path` in `run.py` to the saved model and use `--mode test` with the same dataset and model configuration.
