import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_path", type=str, required=True)
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--full", type=str, default="False")
    return p.parse_args()


def main():
    args = parse_args()
    raw_path = Path(args.raw_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read
    df = (
        pd.read_csv(raw_path)
        if raw_path.suffix == ".csv"
        else pd.read_excel(
            raw_path,
            dtype={"InvoiceNo": str, "StockCode": str},  # ensure string types on read
        )
    )

    # Basic cleaning
    df = df.dropna(subset=["CustomerID"])  # remove rows with no customer id
    df = df[df["Quantity"] > 0]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])  # ensure valid dates

    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # Optional sampling for quick runs
    if args.full.lower() != "true":
        df = df.sample(frac=0.25, random_state=42) if len(df) > 10000 else df

    df.to_parquet(out_path, index=False)
    print(f"Saved interim parquet to {out_path}")


if __name__ == "__main__":
    main()
