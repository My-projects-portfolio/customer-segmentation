import argparse
import pandas as pd
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_path", type=str, required=True)
    p.add_argument("--out_path", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerID").agg(
            Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalPrice", "sum"),
        )
    ).reset_index()

    rfm.to_parquet(out_path, index=False)
    print(f"Saved RFM features to {out_path}")


if __name__ == "__main__":
    main()
