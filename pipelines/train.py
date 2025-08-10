import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rfm_path", type=str, required=True)
    p.add_argument("--export_path", type=str, required=True)
    p.add_argument("--sample", type=float, default=None)
    p.add_argument("--ci_gate", type=str, default="False")
    return p.parse_args()


def load_params():
    import yaml

    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    params = load_params()

    df = pd.read_parquet(args.rfm_path)
    if args.sample:
        df = (
            df.sample(frac=args.sample, random_state=params["random_state"])
            if len(df) > 1000
            else df
        )

    X = df[["Recency", "Frequency", "Monetary"]].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = params["k"]
    kmeans = KMeans(n_clusters=k, random_state=params["random_state"], n_init="auto")
    labels = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else 0.0

    # MLflow logging
    mlflow.set_tracking_uri("file:./mlruns")
    with mlflow.start_run(run_name=f"kmeans_k={k}"):
        mlflow.log_params({"k": k, "random_state": params["random_state"]})
        mlflow.log_metric("silhouette", sil)

        export_dir = Path(args.export_path)
        export_dir.mkdir(parents=True, exist_ok=True)

        # Persist minimal artifacts
        model_path = export_dir / "kmeans_model.npz"
        np.savez(
            model_path,
            centers_=kmeans.cluster_centers_,
            scale_mean_=scaler.mean_,
            scale_scale_=scaler.scale_,
        )

        meta = {"silhouette": sil, "k": k}
        (export_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        print(f"Trained KMeans(k={k}) silhouette={sil:.4f}")

        # CI gate (compare vs last metrics if present)
        if str(args.ci_gate).lower() == "true":
            prev_path = Path("artifacts/reports/metrics.json")
            if prev_path.exists():
                prev = json.loads(prev_path.read_text())
                drop = prev.get("silhouette", 0) - sil
                max_drop = load_params()["silhouette_gate_drop"]
                if drop > max_drop:
                    raise SystemExit(
                        f"CI Gate failed: silhouette drop {drop:.4f} > {max_drop}"
                    )

        # write current metrics for next runs
        Path("artifacts/reports").mkdir(parents=True, exist_ok=True)
        Path("artifacts/reports/metrics.json").write_text(
            json.dumps({"silhouette": sil}, indent=2)
        )


if __name__ == "__main__":
    main()
