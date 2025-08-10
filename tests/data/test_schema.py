from pathlib import Path
import pandas as pd

def test_transactions_schema_basic():
    p = Path("data/interim/transactions.parquet")
    assert p.exists(), "transactions.parquet not found (CI sample step should have created it)"

    df = pd.read_parquet(p)
    required = ["InvoiceNo", "InvoiceDate", "CustomerID", "Quantity", "UnitPrice", "TotalPrice"]
    new_func(df, required)

def new_func(df, required):
    for col in required:
        assert col in df.columns, f"Missing column: {col}"

    assert (df["Quantity"] > 0).all(), "Quantity must be > 0"
    assert (df["UnitPrice"] >= 0).all(), "UnitPrice must be >= 0"
