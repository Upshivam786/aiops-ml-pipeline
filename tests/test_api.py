import pytest
from fastapi.testclient import TestClient

VALID = {f"f{i}": 0.1 * i for i in range(8)}


@pytest.fixture(scope="module")
def client():
    try:
        from src.serve import app
    except Exception as e:  # no @production model available
        pytest.skip(f"model not available: {e}")
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_binary_class(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    assert r.json()["prediction"] in (0, 1)


def test_predict_rejects_missing_field(client):
    bad = {k: v for k, v in VALID.items() if k != "f3"}
    assert client.post("/predict", json=bad).status_code == 422


def test_metrics_expose_prediction_counter(client):
    client.post("/predict", json=VALID)
    body = client.get("/metrics").text
    assert "ml_predictions_total" in body
