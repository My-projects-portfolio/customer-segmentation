from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("🧩 Customer Segmentation Dashboard")

rfm_path = Path("data/processed/rfm.parquet")
if not rfm_path.exists():
    st.info("Run the pipeline first: `make train`.")
else:
    rfm = pd.read_parquet(rfm_path)
    st.subheader("RFM Sample")
    st.dataframe(rfm.head(20))

    st.subheader("Cluster Explorer (after training)")
    meta_path = Path("artifacts/models/meta.json")
    if meta_path.exists():
        import json

        meta = json.loads(meta_path.read_text())
        st.write("**Model Info:**", meta)

        # Simple viz
        st.scatter_chart(rfm, x="Recency", y="Monetary")
    else:
        st.warning("Model not found. Train first: `make train`.")
