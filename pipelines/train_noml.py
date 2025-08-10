# pipelines/train_noml.py
import argparse
import json
from pathlib import Path

import numpy as np


def load_rfm(rfm_path: str):
    p = Path(rfm_path)
    if not p.exists():
        raise FileNotFoundError(f"RFM file not found: {p}")

    if p.suffix.lower() == ".parquet":
        # Try pyarrow first (no pandas import)
        try:
            import pyarrow.parquet as pq

            tbl = pq.read_table(p)
            cols = {c.lower(): i for i, c in enumerate(tbl.column_names)}
            for need in ("recency", "frequency", "monetary"):
                if need not in cols:
                    raise ValueError(f"Missing '{need}' column in parquet.")
            X = np.column_stack(
                [
                    tbl.column(cols["recency"]).to_numpy(),
                    tbl.column(cols["frequency"]).to_numpy(),
                    tbl.column(cols["monetary"]).to_numpy(),
                ]
            ).astype(float)
            return X
        except Exception:
            # Fallback to pandas if available
            import pandas as pd

            df = pd.read_parquet(p)
            return df[["Recency", "Frequency", "Monetary"]].to_numpy(dtype=float)

    elif p.suffix.lower() == ".csv":
        # CSV path (simple)
        try:
            import pandas as pd

            df = pd.read_csv(p)
            return df[["Recency", "Frequency", "Monetary"]].to_numpy(dtype=float)
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV: {e}")
    else:
        raise ValueError("Unsupported file type. Use .parquet or .csv")


def standardize(X: np.ndarray):
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std


def kmeans_numpy(X: np.ndarray, k: int, max_iter: int = 100, seed: int = 42):
    rng = np.random.default_rng(seed)
    # KMeans++ init (simple variant)
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=X.dtype)
    idx0 = rng.integers(0, n)
    centers[0] = X[idx0]
    d2 = ((X - centers[0]) ** 2).sum(axis=1)
    for j in range(1, k):
        probs = d2 / d2.sum()
        idx = rng.choice(n, p=probs)
        centers[j] = X[idx]
        d2 = np.minimum(d2, ((X - centers[j]) ** 2).sum(axis=1))

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # Assign
        dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # (n,k)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Update
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = X[mask].mean(axis=0)
            # if empty cluster, keep old center
    return centers, labels


def silhouette_numpy(
    X: np.ndarray, labels: np.ndarray, sample: int | None = None, seed: int = 42
):
    """Compute silhouette score (mean) in NumPy. To keep it fast, you can sample points."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = (
        np.arange(n)
        if (sample is None or sample >= n)
        else rng.choice(n, size=sample, replace=False)
    )
    L = labels[idx]

    # Precompute pairwise distances to avoid recompute for each i
    # For ~4k points this is OK; if larger, reduce 'sample'.
    Xs = X[idx]
    # (m,m) distance matrix
    D = np.sqrt(((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(axis=2))

    s_vals = []
    for i in range(len(idx)):
        same = L == L[i]
        other = ~same
        # a(i): mean intra-cluster distance (excluding self)
        a = D[i, same]
        a = a[a > 0]  # drop self
        a_i = a.mean() if a.size else 0.0

        # b(i): min mean distance to other clusters
        b_i = np.inf
        for c in np.unique(L[other]):
            mask = L == c
            b_i = min(b_i, D[i, mask].mean())
        s = 0.0 if b_i == 0 and a_i == 0 else (b_i - a_i) / max(a_i, b_i)
        s_vals.append(s)
    return float(np.mean(s_vals)) if s_vals else 0.0


def save_artifacts(
    export_path: str,
    centers: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    k: int,
    sil: float,
):
    out = Path(export_path)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "kmeans_model.npz", centers_=centers, scale_mean_=mean, scale_scale_=std
    )
    (out / "meta.json").write_text(
        json.dumps({"silhouette": sil, "k": int(k)}, indent=2)
    )
    Path("artifacts/reports").mkdir(parents=True, exist_ok=True)
    Path("artifacts/reports/metrics.json").write_text(
        json.dumps({"silhouette": sil}, indent=2)
    )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rfm_path", required=True, help="Path to rfm.parquet or rfm.csv")
    ap.add_argument(
        "--export_path", required=True, help="Where to write model artifacts"
    )
    ap.add_argument(
        "--sample",
        type=float,
        default=None,
        help="Optional frac sample (0-1) for speed",
    )
    ap.add_argument("--k", type=int, default=4)
    return ap.parse_args()


def main():
    args = parse_args()
    X = load_rfm(args.rfm_path)

    # optional downsample to speed up
    if args.sample and 0 < args.sample < 1.0 and len(X) > 1000:
        m = int(len(X) * args.sample)
        rng = np.random.default_rng(42)
        rows = rng.choice(len(X), size=m, replace=False)
        X = X[rows]

    Xs, mean, std = standardize(X)
    centers, labels = kmeans_numpy(Xs, k=args.k, max_iter=100, seed=42)

    # silhouette on a capped sample for speed (e.g., 2000 points max)
    m = None if len(Xs) <= 2000 else 2000
    sil = silhouette_numpy(Xs, labels, sample=m)
    print(
        f"Trained (no-MLflow, no-sklearn) KMeans(k={args.k}) silhouette={sil:.4f}, n={len(Xs)}"
    )

    save_artifacts(args.export_path, centers, mean, std, args.k, sil)


if __name__ == "__main__":
    main()
