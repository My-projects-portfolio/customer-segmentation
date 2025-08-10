import argparse
from pathlib import Path

import mlflow


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    mlflow.set_tracking_uri("file:./mlruns")

    # In a real registry flow, you would log the model and transition stages.
    # Here we just demonstrate a placeholder where you could integrate MLflow Model Registry.
    model_dir = Path(args.model_dir)
    if not (model_dir / "kmeans_model.npz").exists():
        raise SystemExit("No model found to register.")

    print("Model ready for registry promotion (hook placeholder).")


if __name__ == "__main__":
    main()
