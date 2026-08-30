# Q4 Partner B Reproduction

Partner B: Arnav (DA24B027)

## Procedure

I cloned Partner A's repository, checked out the handoff commit `b360abc`, and recreated the supplied Conda environment. The final reproduction was run from commit `58d768a`, which adds configurable MLflow experiment naming without changing the training procedure.

```bash
git clone git@github.com:AakashAadhithya/aiops-assignment1.git
cd aiops-assignment1
git checkout b360abc
conda env create -f environment.yml -n q4
conda activate q4
git checkout 58d768a
python train.py --hidden_size 128 --lr 0.001 --epochs 20 --run_name capstone-q4-reproduced
```

The run used `hidden_size=128`, `lr=0.001`, `batch_size=64`, `epochs=20`, and `seed=42`.

## Result

| Run | Validation accuracy | Training loss |
|---|---:|---:|
| Partner A original | 0.948 | 0.02032236113371596 |
| Partner B reproduction | 0.948 | 0.02032236113371596 |

The absolute validation-accuracy difference was `0.000`. This is within the chosen tolerance of `0.001`, so the reproduction was a match.

- Partner A MLflow run ID: `fe120f946b894c2aae5b9fb611afa853`
- Partner B MLflow run ID: `99055b2e12974d4282cc15571f2a6c30`

## Evidence

- `partner_a_original_run.png`: original run parameters and final metrics.
- `partner_b_mlflow_run.png`: reproduced run parameters, final metrics, comparison note, source commit, and logged model.
- `partner_b_reproduction_terminal.png`: completed Partner B run and MLflow run link.
- `clone_and_checkout.png`: clone and handoff-commit checkout.
- `conda_environment.png`: environment recreation from `environment.yml`.
