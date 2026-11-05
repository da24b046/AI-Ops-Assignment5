"""Ray Serve ETA service with stable customer-based canary routing."""

import hashlib
import math
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Header
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ray import serve


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
            raise ValueError("infeasible delivery")
        return self


class CanaryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(min_length=1)
    instances: list[Delivery] = Field(
        min_length=1,
        max_length=500,
    )


def bucket(customer_id):
    """Return a stable value in [0, 1), independent of processes."""
    digest = hashlib.sha256(
        customer_id.encode("utf-8")
    ).digest()

    # Use 53 bits so the resulting fraction is exactly representable.
    value = int.from_bytes(digest[:8], "big") >> 11
    return value / (1 << 53)


@serve.deployment(ray_actor_options={"num_cpus": 0.25})
class EtaModel:
    def __init__(self, bundle_path, extra_work_ms=0.0):
        # Load once when this replica starts.
        self.bundle = joblib.load(bundle_path)
        self.extra_work_ms = extra_work_ms

    def predict(self, rows):
        # Optional CPU work for separate load experiments.
        end = time.perf_counter() + self.extra_work_ms / 1000
        while time.perf_counter() < end:
            pass

        frame = pd.DataFrame(
            rows,
            columns=self.bundle["feature_columns"],
        )

        predictions = [
            round(float(value), 1)
            for value in self.bundle["model"].predict(frame)
        ]

        return {
            "predictions": predictions,
            "model_version": self.bundle["version"],
        }


api = FastAPI(title="SwiftParcel sticky canary")


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.1},
)
@serve.ingress(api)
class EtaRouter:
    def __init__(self, prod, candidate, weight):
        self.prod = prod
        self.candidate = candidate
        self.weight = weight

    @api.post("/v1/models/eta:predict")
    async def predict(
        self,
        body: CanaryRequest,
        x_beta: str | None = Header(default=None),
    ):
        use_candidate = (
            x_beta == "1"
            or bucket(body.customer_id) < self.weight
        )

        handle = (
            self.candidate if use_candidate else self.prod
        )

        output = await handle.predict.remote(
            [row.model_dump() for row in body.instances]
        )

        return {
            **output,
            "arm": "v2" if use_candidate else "v1",
        }


def build(weight=0.1, extra_work_ms=0.0):
    weight = float(weight)
    extra_work_ms = float(extra_work_ms)

    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("weight must be between 0 and 1")

    if not math.isfinite(extra_work_ms) or extra_work_ms < 0:
        raise ValueError("extra_work_ms must be finite and nonnegative")

    data_dir = Path(
        os.environ.get("ETA_DATA_DIR", "data")
    ).resolve()

    prod = EtaModel.options(
        name="eta-prod",
        num_replicas=2,
        max_ongoing_requests=8,
    ).bind(
        str(data_dir / "eta_v1.joblib"),
        extra_work_ms,
    )

    candidate = EtaModel.options(
        name="eta-candidate",
        num_replicas="auto",
        max_ongoing_requests=8,
        autoscaling_config={
            "min_replicas": 1,
            "max_replicas": 3,
            "target_ongoing_requests": 2,
            "upscale_delay_s": 5,
            "downscale_delay_s": 30,
        },
    ).bind(
        str(data_dir / "eta_v2.joblib"),
        extra_work_ms,
    )

    return EtaRouter.bind(prod, candidate, weight)
