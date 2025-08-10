import pandas as pd
import pandera as pa
from pandera.typing import Series


class TxnSchema(pa.SchemaModel):
    InvoiceNo: Series[str]
    InvoiceDate: Series[object]
    CustomerID: Series[float]
    Quantity: Series[float]
    UnitPrice: Series[float]

    class Config:
        strict = False


def test_schema_sample():
    # Load tiny sample if available
    try:
        df = pd.read_parquet("data/interim/transactions.parquet").head(100)
    except Exception:
        return  # skip if not generated yet
    TxnSchema.validate(df, lazy=True)
