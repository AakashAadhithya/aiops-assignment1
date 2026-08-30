"""
train.py — MLP on MNIST with MLflow experiment tracking
AIOps Module 1 Assignment, Question 2
"""
import os
import argparse
import subprocess
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss


def get_git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode("ascii").strip()
    except Exception:
        return "unknown"


def load_mnist(train_size=6000, test_size=1000, seed=42):
    print("Loading MNIST (first run downloads + caches it, later runs are fast)...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X / 255.0  # normalize pixel values to [0,1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, test_size=test_size,
        random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int, default=64,
                         help="Units in the single hidden layer")
    parser.add_argument("--lr", type=float, default=0.001,
                         help="Initial learning rate")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "mnist-mlp-classifier"))

    X_train, X_test, y_train, y_test = load_mnist(seed=args.seed)
    classes = np.unique(y_train)

    model = MLPClassifier(
        hidden_layer_sizes=(args.hidden_size,),
        learning_rate_init=args.lr,
        batch_size=args.batch_size,
        max_iter=1,          # epochs controlled manually below
        warm_start=True,
        random_state=args.seed,
        solver="adam",
    )

    run_name = args.run_name or f"hidden{args.hidden_size}-lr{args.lr}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("hidden_size", args.hidden_size)
        mlflow.log_param("lr", args.lr)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("seed", args.seed)
        mlflow.set_tag("git_commit", get_git_commit_hash())

        for epoch in range(args.epochs):
            model.partial_fit(X_train, y_train, classes=classes)

            train_pred_proba = model.predict_proba(X_train)
            train_loss = log_loss(y_train, train_pred_proba, labels=classes)

            val_pred = model.predict(X_test)
            val_accuracy = accuracy_score(y_test, val_pred)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

            print(f"epoch {epoch+1}/{args.epochs}  "
                  f"train_loss={train_loss:.4f}  val_accuracy={val_accuracy:.4f}")

        mlflow.sklearn.log_model(model, "model", serialization_format="pickle")

    print("Run complete.")


if __name__ == "__main__":
    main()
