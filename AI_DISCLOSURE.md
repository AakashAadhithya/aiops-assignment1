# AI Disclosure

**Tool used:** Claude (Anthropic), via the chat interface.

**How it was used:**
- Broke down the assignment into a step-by-step plan across all 4 questions.
- Guided me through setting up the Ubuntu VM environment: installing git, mamba/conda, dvc, dvc-ssh, and mlflow, and troubleshooting install issues as they came up.
- Helped write `train.py` (MLP-on-MNIST training script with MLflow logging) and `register_model.py` (model registration/staging script) — I reviewed, ran, and iterated on this code myself rather than using it unmodified.
- Helped debug real errors I hit while running the code, including: an MLflow/skops model-saving error (switched to pickle serialization), DVC's `tutorial/ver/data.zip` path being outdated in the lecture slides, SSH server setup and a stale/locked QEMU process, UTM's bridged-network mode failing on a mobile hotspot (worked around with a `socat` TCP relay), and MLflow's CORS/allowed-hosts security settings blocking cross-machine access during the Q4 partner handoff.
- Helped me compress report.pdf to fit the required 1-page limit.

**Impact:** AI assistance shaped nearly every stage of this assignment — from environment setup through debugging to the final report — since I was doing most of this tooling (VM, DVC, MLflow, LaTeX) for the first time. I ran every command myself, made the actual decisions at each step (e.g. which hyperparameters to sweep, how to interpret the results, what to write in the analysis), and understood the reasoning behind the fixes rather than copying them blindly, but the overall workflow and much of the code and debugging would have taken substantially longer without it.
