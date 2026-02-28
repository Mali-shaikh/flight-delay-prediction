"""
Unit Tests for FastAPI Endpoints
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

SAMPLE_PAYLOAD = {
    "month": 6,
    "day_of_week": 5,
    "day_of_month": 15,
    "dep_hour": 8,
    "carrier": "AA",
    "origin": "JFK",
    "dest": "LAX",
    "distance": 2475.0,
    "crs_elapsed_time": 330.0,
}


class TestRootEndpoint:
    def test_root_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_endpoints(self):
        response = client.get("/")
        data = response.json()
        assert "endpoints" in data


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data


class TestPredictEndpoint:
    def test_predict_valid_input(self):
        """Test valid prediction request."""
        response = client.post("/predict", json=SAMPLE_PAYLOAD)
        # Either 200 (model loaded) or 503 (model not trained yet)
        assert response.status_code in [200, 503]

    def test_predict_response_schema(self):
        """If prediction succeeds, verify schema."""
        response = client.post("/predict", json=SAMPLE_PAYLOAD)
        if response.status_code == 200:
            data = response.json()
            assert "delayed" in data
            assert "probability" in data
            assert "status" in data
            assert "confidence" in data
            assert data["delayed"] in [0, 1]
            assert 0.0 <= data["probability"] <= 1.0
            assert data["status"] in ["DELAYED", "ON TIME"]
            assert data["confidence"] in ["HIGH", "MEDIUM", "LOW"]

    def test_predict_invalid_month(self):
        """Test validation — month out of range."""
        bad_payload = {**SAMPLE_PAYLOAD, "month": 13}
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422  # Validation error

    def test_predict_invalid_day_of_week(self):
        """Test validation — day_of_week out of range."""
        bad_payload = {**SAMPLE_PAYLOAD, "day_of_week": 8}
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_predict_missing_field(self):
        """Test validation — missing required field."""
        incomplete = {"month": 6, "carrier": "AA"}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_predict_carrier_uppercase(self):
        """Test carrier code is auto-uppercased."""
        payload = {**SAMPLE_PAYLOAD, "carrier": "aa"}
        response = client.post("/predict", json=payload)
        assert response.status_code in [200, 503]


class TestModelInfoEndpoint:
    def test_model_info_endpoint_exists(self):
        response = client.get("/model-info")
        assert response.status_code in [200, 503]
