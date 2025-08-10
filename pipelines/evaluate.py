import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rfm_path", type=str, required=True)
    p.add_argument("--model_dir", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    rfm = pd.read_parquet(args.rfm_path)
    X = rfm[["Recency", "Frequency", "Monetary"]].values

    # Load persisted model
    model_npz = np.load(Path(args.model_dir) / "kmeans_model.npz")
    centers = model_npz["centers_"]
    scale_mean = model_npz["scale_mean_"]
    scale_scale = model_npz["scale_scale_"]

    scaler = StandardScaler()
    scaler.mean_ = scale_mean
    scaler.scale_ = scale_scale
    Xs = scaler.transform(X)

    # Predict labels by nearest center
    dists = ((Xs[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = dists.argmin(axis=1)

    sil = silhouette_score(Xs, labels) if len(np.unique(labels)) > 1 else 0.0

    Path("artifacts/reports").mkdir(parents=True, exist_ok=True)
    Path("artifacts/reports/metrics.json").write_text(json.dumps({"silhouette": sil}, indent=2))
    print(f"Evaluation silhouette={sil:.4f}")


if __name__ == "__main__":
    main()
