# AIOps Module 1 Assignment — Experiment Management & Reproducibility

This repo contains my submission for the Module 1 assignment (Git + Conda, MLflow, DVC). The actual findings and analysis are in `report.pdf` — this file is just about how to set everything up and run the code yourself.

## What's in here

- `train.py` — trains an MLP on MNIST, logs params/metrics/seed/git_commit tag/artifact to MLflow. Used for both Q2's hyperparameter sweep and Q4's capstone run.
- `register_model.py` — registers a trained run's model in the MLflow Model Registry and transitions it to Staging (Q4).
- `generate_filelist.py` — walks `data/` and writes out `filenames.csv`, used for the DVC versioning in Q3.
- `environment.yml` — pinned conda/mamba environment (Python 3.10, pytorch, mlflow, dvc, scikit-learn, pandas).
- `data.dvc`, `filenames.csv.dvc` — DVC pointer files tracking the versioned dataset (currently pointing at v2, 2800 files).
- `screenshots/` — evidence referenced from `report.pdf`.
- `partner_b/` — my partner Arnav's (DA24B027) own write-up and screenshots documenting his independent reproduction of this project (Q4, Parts 2-4). See `partner_b/README.md`.
- `report.pdf` — the actual 1-page report with findings for all 4 questions.

## Setup

### 1. Clone the repo

```bash
git clone git@github.com:AakashAadhithya/aiops-assignment1.git
cd aiops-assignment1
```

### 2. Recreate the environment

Needs conda or mamba installed.

```bash
conda env create -f environment.yml -n aiops
conda activate aiops
```

### 3. Pull the dataset

Needs access to the DVC remote — see note below.

```bash
dvc pull
```

## Running the MLflow sweep (Q2)

Start a local MLflow tracking server first:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

Then, in another terminal, run training with any combination of hyperparameters, e.g.:

```bash
python train.py --hidden_size 128 --lr 0.001 --epochs 20 --run_name my-run
```

Open `http://localhost:5000` in a browser to see the logged runs and compare them.

## DVC versioning (Q3)

The dataset is already versioned in this repo (v1 = 1800 files, v2 = 2800 files, current). To see the version history:

```bash
git log --oneline -- data.dvc
```

To roll back to an older version:

```bash
git checkout <commit-hash>
dvc checkout
```

## Capstone reproduction (Q4)

`train.py` reads the MLflow server address from an environment variable, so it can point at someone else's server:

```bash
export MLFLOW_TRACKING_URI=http://<server-ip>:<port>
python train.py --hidden_size 128 --lr 0.001 --epochs 20 --run_name capstone-reproduced
```

The DVC remote is configured for SSH — to pull data from a different machine, update the remote URL first:

```bash
dvc remote modify sshremote url ssh://<user>@<host>/path/to/dvcstore
```

### Partner B's independent reproduction

My partner Arnav (DA24B027) reproduced this project from scratch on his own machine, over a shared mobile hotspot. His full write-up, exact commands, and screenshots (cloning, environment setup, and the matching training run) are documented in [`partner_b/README.md`](partner_b/README.md). All 20 logged epochs matched my original run exactly.

## Note on the DVC remote

My DVC remote lives on my personal machine, so `dvc pull` will only work if you have SSH access to it. For grading purposes, the actual dataset content isn't essential to re-download — `data.dvc` and `filenames.csv.dvc` in the commit history are the evidence of versioning, and `screenshots/q3_rollback.png` shows the rollback working end to end.

## AI usage disclosure

See `AI_DISCLOSURE.md`.
