"""MLflow registration, fresh-data evaluation and promotion decisions."""

import argparse
import json
import math
import os
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

from eta_data import FEATURES, make_data


class EtaPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.bundle = joblib.load(context.artifacts["bundle"])

    def predict(self, context, model_input, params=None):
        frame = model_input[self.bundle["feature_columns"]]
        predictions = self.bundle["model"].predict(frame)

        return pd.DataFrame({
            "eta_minutes": np.round(predictions, 1),
            "model_version": self.bundle["version"],
        })


def configure_tracking():
    mlflow.set_tracking_uri(
        os.environ.get(
            "MLFLOW_TRACKING_URI",
            "sqlite:///mlflow.db",
        )
    )


def register():
    configure_tracking()
    mlflow.set_experiment("swiftparcel-eta")
    client = MlflowClient()

    # A consistent double-valued schema avoids integer/float
    # mismatches when clients construct DataFrames differently.
    example = make_data(5, seed=5)[FEATURES].astype(float)

    pinned_requirements = [
        f"{package}=={version(package)}"
        for package in [
            "mlflow",
            "scikit-learn",
            "pandas",
            "numpy",
            "joblib",
            "cloudpickle",
        ]
    ]

    registry_rows = []

    for bundle_version, alias in [
        ("v1", "prod"),
        ("v2", "candidate"),
    ]:
        bundle_path = Path(
            f"data/eta_{bundle_version}.joblib"
        ).resolve()

        wrapper = EtaPyfunc()
        wrapper.bundle = joblib.load(bundle_path)
        example_output = wrapper.predict(None, example)

        with mlflow.start_run(run_name=bundle_version):
            info = mlflow.pyfunc.log_model(
                name="eta",
                python_model=wrapper,
                artifacts={"bundle": str(bundle_path)},
                signature=infer_signature(
                    example,
                    example_output,
                ),
                input_example=example,
                pip_requirements=pinned_requirements,
                code_paths=[
                    str(Path(__file__).resolve()),
                    str(Path("eta_data.py").resolve()),
                ],
                registered_model_name="eta-model",
            )

        registry_version = str(info.registered_model_version)

        client.set_registered_model_alias(
            "eta-model",
            alias,
            registry_version,
        )

        registry_rows.append({
            "version": registry_version,
            "bundle": bundle_version,
            "alias": alias,
        })

    Path("partB/registry.json").write_text(
        json.dumps(registry_rows, indent=2) + "\n"
    )

    print("\nRegistry table:")
    print(pd.DataFrame(registry_rows).to_string(index=False))

    return registry_rows


def decide(
    prod,
    cand,
    min_rel_gain=0.03,
    max_p95_ms=10.0,
    max_slice_regression=0.05,
):
    """Pure decision function: no model loading or alias changes."""
    metric_keys = [
        "mae",
        "mae_rush",
        "mae_rain",
        "p95_ms",
    ]

    for metrics in (prod, cand):
        for key in metric_keys:
            value = float(metrics[key])
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    "Metrics must be finite and nonnegative"
                )

    thresholds = [
        min_rel_gain,
        max_p95_ms,
        max_slice_regression,
    ]
    if not all(math.isfinite(float(x)) for x in thresholds):
        raise ValueError("Thresholds must be finite")

    if (
        not 0 <= min_rel_gain <= 1
        or max_p95_ms < 0
        or max_slice_regression < 0
    ):
        raise ValueError("Invalid thresholds")

    criteria = {
        "overall_mae": bool(
            cand["mae"]
            <= prod["mae"] * (1 - min_rel_gain)
        ),
        "latency": bool(
            cand["p95_ms"] <= max_p95_ms
        ),
        "rush_slice": bool(
            cand["mae_rush"]
            <= prod["mae_rush"] * (1 + max_slice_regression)
        ),
        "rain_slice": bool(
            cand["mae_rain"]
            <= prod["mae_rain"] * (1 + max_slice_regression)
        ),
    }

    return {
        "criteria": criteria,
        "promote": all(criteria.values()),
    }


def evaluate(model, fresh):
    features = fresh[FEATURES].astype(float)
    predictions = model.predict(features)["eta_minutes"].to_numpy()
    errors = np.abs(
        predictions - fresh["eta_minutes"].to_numpy()
    )

    rush_mask = fresh["hour"].isin(
        [7, 8, 9, 16, 17, 18, 19]
    ).to_numpy()
    rain_mask = fresh["rain"].eq(1).to_numpy()

    # Exclude model loading and warm-up from inference timing.
    for _ in range(10):
        model.predict(features.iloc[[0]])

    durations_ms = []

    for index in range(200):
        single_row = features.iloc[[index]]
        started = time.perf_counter()
        model.predict(single_row)
        durations_ms.append(
            (time.perf_counter() - started) * 1000
        )

    return {
        "mae": float(errors.mean()),
        "mae_rush": float(errors[rush_mask].mean()),
        "mae_rain": float(errors[rain_mask].mean()),
        "p95_ms": float(np.percentile(durations_ms, 95)),
    }


def promotion_gate(**thresholds):
    configure_tracking()
    fresh = make_data(2000, seed=99)

    metrics = {}

    for alias in ("prod", "candidate"):
        loaded_model = mlflow.pyfunc.load_model(
            f"models:/eta-model@{alias}"
        )
        metrics[alias] = evaluate(loaded_model, fresh)

    settings = {
        "min_rel_gain": 0.03,
        "max_p95_ms": 10.0,
        "max_slice_regression": 0.05,
        **thresholds,
    }

    decision = decide(
        metrics["prod"],
        metrics["candidate"],
        **settings,
    )

    return {
        "metrics": metrics,
        "thresholds": settings,
        **decision,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["register", "gate"],
    )
    args = parser.parse_args()

    if args.action == "register":
        register()
    else:
        results = {
            "default": promotion_gate(),
            "strict_gain_0.60": promotion_gate(
                min_rel_gain=0.60
            ),
        }

        output = json.dumps(results, indent=2)
        Path("partB/gate_results.json").write_text(
            output + "\n"
        )
        print(output)
