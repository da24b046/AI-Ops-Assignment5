import math
import uuid

import pytest
from fastapi.testclient import TestClient

import eta_app


URL = "/v1/models/eta:predict"

VALID_ROW = {
    "distance_km": 10,
    "n_stops": 2,
    "hour": 10,
    "weekday": 2,
    "traffic_index": 0.3,
    "rain": 0,
    "weight_kg": 3,
}


@pytest.fixture
def client():
    with TestClient(eta_app.app) as test_client:
        yield test_client


def test_valid_prediction(client):
    response = client.post(URL, json={"instances": [VALID_ROW]})

    assert response.status_code == 200
    body = response.json()

    assert body["model_version"] == "v1"
    assert len(body["predictions"]) == 1

    prediction = body["predictions"][0]
    assert math.isfinite(prediction)
    assert prediction == round(prediction, 1)


def test_batch_preserves_order(client):
    rows = [
        dict(VALID_ROW, distance_km=50),
        dict(VALID_ROW, distance_km=5),
        dict(VALID_ROW, distance_km=90),
    ]

    individual_predictions = []

    for row in rows:
        response = client.post(URL, json={"instances": [row]})
        assert response.status_code == 200
        individual_predictions.append(response.json()["predictions"][0])

    # Different expected values make the ordering check meaningful.
    assert len(set(individual_predictions)) == len(rows)

    response = client.post(URL, json={"instances": rows})

    assert response.status_code == 200
    assert response.json()["predictions"] == individual_predictions


@pytest.mark.parametrize(
    "field,too_low,too_high",
    [
        ("distance_km", 0.09, 500.01),
        ("n_stops", 0, 61),
        ("hour", -1, 24),
        ("weekday", -1, 7),
        ("traffic_index", -0.01, 1.01),
        ("rain", -1, 2),
        ("weight_kg", 0, 1000.01),
    ],
)
def test_out_of_range_values(client, field, too_low, too_high):
    for invalid_value in (too_low, too_high):
        row = dict(VALID_ROW, **{field: invalid_value})
        response = client.post(URL, json={"instances": [row]})

        assert response.status_code == 422
        body = response.json()

        assert body["error"] == "validation_failed"
        assert body["details"]
        assert all(
            {"field", "msg"} <= detail.keys()
            for detail in body["details"]
        )
        assert any(
            field in detail["field"]
            for detail in body["details"]
        )


def test_missing_feature(client):
    row = dict(VALID_ROW)
    del row["hour"]

    response = client.post(URL, json={"instances": [row]})

    assert response.status_code == 422
    assert any(
        "hour" in detail["field"]
        for detail in response.json()["details"]
    )


def test_wrong_feature_type(client):
    row = dict(VALID_ROW, n_stops="not-an-integer")
    response = client.post(URL, json={"instances": [row]})

    assert response.status_code == 422
    assert response.json()["error"] == "validation_failed"


def test_fractional_integer_rejected(client):
    row = dict(VALID_ROW, n_stops=2.5)
    response = client.post(URL, json={"instances": [row]})

    assert response.status_code == 422


def test_infeasible_delivery(client):
    row = dict(VALID_ROW, distance_km=151, n_stops=21)
    response = client.post(URL, json={"instances": [row]})

    assert response.status_code == 422
    assert response.json()["error"] == "validation_failed"
    assert "infeasible" in response.text.lower()


@pytest.mark.parametrize(
    "changes",
    [
        {"distance_km": 150, "n_stops": 21},
        {"distance_km": 151, "n_stops": 20},
    ],
)
def test_feasibility_boundary_allowed(client, changes):
    row = dict(VALID_ROW, **changes)
    response = client.post(URL, json={"instances": [row]})

    assert response.status_code == 200


@pytest.mark.parametrize(
    "batch_size,expected_status",
    [(0, 422), (1, 200), (500, 200), (501, 422)],
)
def test_batch_size_limits(client, batch_size, expected_status):
    response = client.post(
        URL,
        json={"instances": [VALID_ROW] * batch_size},
    )

    assert response.status_code == expected_status

    if expected_status == 200:
        assert len(response.json()["predictions"]) == batch_size


@pytest.mark.parametrize("valid", [True, False])
def test_request_id_echo_on_success_and_error(client, valid):
    instances = [VALID_ROW] if valid else []

    response = client.post(
        URL,
        json={"instances": instances},
        headers={"X-Request-ID": "assignment-test-123"},
    )

    assert response.status_code == (200 if valid else 422)
    assert response.headers["X-Request-ID"] == "assignment-test-123"
    assert response.json()["request_id"] == "assignment-test-123"


@pytest.mark.parametrize("valid", [True, False])
def test_generated_request_id_on_success_and_error(client, valid):
    instances = [VALID_ROW] if valid else []

    response = client.post(URL, json={"instances": instances})

    assert response.status_code == (200 if valid else 422)

    request_id = response.headers["X-Request-ID"]
    assert str(uuid.UUID(request_id)) == request_id
    assert response.json()["request_id"] == request_id


def test_generated_request_ids_are_distinct(client):
    first = client.post(URL, json={"instances": [VALID_ROW]})
    second = client.post(URL, json={"instances": [VALID_ROW]})

    assert first.headers["X-Request-ID"] != second.headers["X-Request-ID"]


def test_metrics_and_scrape_exclusion(client):
    client.post(URL, json={"instances": [VALID_ROW]})
    client.post(URL, json={"instances": []})

    first = client.get("/metrics")
    second = client.get("/metrics")

    assert first.status_code == 200
    assert first.headers["content-type"].startswith("text/plain")

    # Reading metrics must not change the counters or histogram.
    assert first.text == second.text

    text = first.text
    assert (
        'eta_requests_total{path="/v1/models/eta:predict",status="200"}'
        in text
    )
    assert (
        'eta_requests_total{path="/v1/models/eta:predict",status="422"}'
        in text
    )
    assert 'path="/metrics"' not in text

    bucket_lines = [
        line for line in text.splitlines()
        if line.startswith("eta_request_latency_seconds_bucket{")
    ]

    assert len(bucket_lines) >= 4
    assert any('le="+Inf"' in line for line in bucket_lines)

    bucket_counts = [
        int(line.rsplit(" ", 1)[1])
        for line in bucket_lines
    ]

    # Histogram buckets must be cumulative.
    assert bucket_counts == sorted(bucket_counts)

    count_line = next(
        line for line in text.splitlines()
        if line.startswith("eta_request_latency_seconds_count ")
    )
    count = int(count_line.rsplit(" ", 1)[1])
    assert bucket_counts[-1] == count

    assert "eta_request_latency_seconds_sum " in text


def test_inference_exception_returns_generic_500(client, monkeypatch):
    def broken_predict(*args, **kwargs):
        raise RuntimeError("SECRET_INTERNAL_MODEL_PATH")

    monkeypatch.setattr(eta_app.model, "predict", broken_predict)

    response = client.post(
        URL,
        json={"instances": [VALID_ROW]},
        headers={"X-Request-ID": "failure-test-123"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "error": "internal_server_error",
        "request_id": "failure-test-123",
    }
    assert response.headers["X-Request-ID"] == "failure-test-123"
    assert "SECRET_INTERNAL_MODEL_PATH" not in response.text
    assert "Traceback" not in response.text


def test_health_endpoint(client):
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_version"] == "v1"
    assert response.headers["X-Request-ID"] == response.json()["request_id"]


def test_unknown_route_has_request_id(client):
    response = client.get(
        "/does-not-exist",
        headers={"X-Request-ID": "missing-route-123"},
    )

    assert response.status_code == 404
    assert response.headers["X-Request-ID"] == "missing-route-123"
    assert response.json()["request_id"] == "missing-route-123"
