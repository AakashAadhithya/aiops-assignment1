"""Register the capstone run's model and transition it to Staging (Q4 Part 1)."""
import argparse
import mlflow
from mlflow.tracking import MlflowClient

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True, help="MLflow run ID to register")
    parser.add_argument("--model_name", default="mnist-mlp-classifier")
    args = parser.parse_args()

    mlflow.set_tracking_uri("http://localhost:5000")

    model_uri = f"runs:/{args.run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=args.model_name)
    print(f"Registered {args.model_name} version {result.version}")

    client = MlflowClient()
    client.transition_model_version_stage(
        name=args.model_name,
        version=result.version,
        stage="Staging",
    )
    print(f"Transitioned {args.model_name} v{result.version} to Staging")

if __name__ == "__main__":
    main()
