"""SwiftParcel ETA REST service. Run commands from the repository root."""

import logging
import math
import os
import threading
import time
import uuid
from collections import Counter

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.exceptions import HTTPException as StarletteHTTPException


# Load the selected model once when this process starts.
bundle = joblib.load(
    os.environ.get("ETA_MODEL", "data/eta_v1.joblib")
)
model = bundle["model"]

app = FastAPI(title="SwiftParcel ETA")
logger = logging.getLogger(__name__)


class Delivery(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    distance_km: float = Field(ge=0.1, le=500)
    n_stops: int = Field(ge=1, le=60)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    traffic_index: float = Field(ge=0, le=1)
    rain: int = Field(ge=0, le=1)
    weight_kg: float = Field(gt=0, le=1000)

    @model_validator(mode="after")
    def check_feasibility(self):
        if self.distance_km > 150 and self.n_stops > 20:
            raise ValueError(
                "infeasible delivery: distance_km > 150 "
                "and n_stops > 20"
            )
        return self


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instances: list[Delivery] = Field(
        min_length=1,
        max_length=500,
    )


# Fixed-size, cumulative metrics protected against concurrent updates.
metrics_lock = threading.Lock()
request_counts = Counter()

latency_bounds = [
    0.001, 0.005, 0.01, 0.025, 0.05,
    0.08, 0.1, 0.25, 0.5, 1.0, float("inf"),
]
latency_buckets = [0] * len(latency_bounds)
latency_sum = 0.0
latency_count = 0


@app.middleware("http")
async def request_envelope(request: Request, call_next):
    global latency_sum, latency_count

    supplied_id = request.headers.get("X-Request-ID")
    request.state.request_id = (
        supplied_id if supplied_id is not None
        else str(uuid.uuid4())
    )

    started = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled request failure")
        response = JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "request_id": request.state.request_id,
            },
        )

    response.headers["X-Request-ID"] = request.state.request_id

    if request.url.path != "/metrics":
        elapsed = time.perf_counter() - started

        # Route templates prevent arbitrary URLs creating endless labels.
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")

        with metrics_lock:
            request_counts[(path, response.status_code)] += 1
            latency_sum += elapsed
            latency_count += 1

            for index, bound in enumerate(latency_bounds):
                if elapsed <= bound:
                    latency_buckets[index] += 1

    return response


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc):
    details = [
        {
            "field": ".".join(str(item) for item in error["loc"]),
            "msg": error["msg"],
        }
        for error in exc.errors()
    ]

    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_failed",
            "details": details,
            "request_id": request.state.request_id,
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        headers=exc.headers,
        content={
            "error": str(exc.detail),
            "request_id": request.state.request_id,
        },
    )


@app.post("/v1/models/eta:predict")
def predict(body: PredictionRequest, request: Request):
    try:
        rows = [row.model_dump() for row in body.instances]
        frame = pd.DataFrame(
            rows,
            columns=bundle["feature_columns"],
        )

        predictions = [
            round(float(value), 1)
            for value in model.predict(frame)
        ]

        if len(predictions) != len(rows):
            raise ValueError("Unexpected prediction count")

        if not all(math.isfinite(value) for value in predictions):
            raise ValueError("Nonfinite prediction")

        return {
            "predictions": predictions,
            "model_version": bundle["version"],
            "request_id": request.state.request_id,
        }

    except Exception:
        # Detailed errors stay in server logs, not in client responses.
        logger.exception("Inference failed")

        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "request_id": request.state.request_id,
            },
        )


@app.get("/healthz")
def health(request: Request):
    return {
        "status": "ok",
        "model_version": bundle["version"],
        "request_id": request.state.request_id,
    }


@app.get("/metrics")
def metrics():
    with metrics_lock:
        lines = [
            "# HELP eta_requests_total Requests excluding metrics scrapes.",
            "# TYPE eta_requests_total counter",
        ]

        for (path, status), count in sorted(request_counts.items()):
            lines.append(
                f'eta_requests_total{{path="{path}",'
                f'status="{status}"}} {count}'
            )

        lines.extend([
            "# HELP eta_request_latency_seconds Request latency in seconds.",
            "# TYPE eta_request_latency_seconds histogram",
        ])

        for bound, count in zip(latency_bounds, latency_buckets):
            label = "+Inf" if math.isinf(bound) else str(bound)
            lines.append(
                "eta_request_latency_seconds_bucket"
                f'{{le="{label}"}} {count}'
            )

        lines.extend([
            f"eta_request_latency_seconds_sum {latency_sum}",
            f"eta_request_latency_seconds_count {latency_count}",
        ])

    return PlainTextResponse(
        "\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4",
    )
