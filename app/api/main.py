from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import numpy as np

app = FastAPI(title="Customer Segmentation API")

# Lazy load model artifacts
_model = None
_scale_mean = None
_scale_scale = None
_centers = None


class RFM(BaseModel):
    recency: float
    frequency: float
    monetary: float


def _load():
    global _model, _scale_mean, _scale_scale, _centers
    npz = np.load(Path("artifacts/models/kmeans_model.npz"))
    _centers = npz["centers_"]
    _scale_mean = npz["scale_mean_"]
    _scale_scale = npz["scale_scale_"]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(rfm: RFM):
    global _centers, _scale_mean, _scale_scale
    if _centers is None:
        _load()

    x = np.array([[rfm.recency, rfm.frequency, rfm.monetary]])
    xs = (x - _scale_mean) / _scale_scale
    dists = ((xs[:, None, :] - _centers[None, :, :]) ** 2).sum(axis=2)
    label = int(dists.argmin(axis=1)[0])
    return {"cluster": label}
