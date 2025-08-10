# tests/integration/test_api.py
from pathlib import Path
import pytest

MODEL_READY = Path("artifacts/models/kmeans_model.npz").exists()
pytestmark = pytest.mark.skipif(
    not MODEL_READY, reason="Model artifact not built yet in this stage."
)

if MODEL_READY:
    import httpx
    from httpx import ASGITransport
    from app.api.main import app

    @pytest.mark.asyncio
    async def test_predict_smoke():
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/predict",
                json={"recency": 10, "frequency": 5, "monetary": 200},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "cluster" in data
            assert isinstance(data["cluster"], int)
