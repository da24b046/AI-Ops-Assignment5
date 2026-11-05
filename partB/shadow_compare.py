"""Compare production and candidate using identical HTTP requests."""

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from eta_data import FEATURES


URLS = {
    "prod": "http://127.0.0.1:5101/invocations",
    "candidate": "http://127.0.0.1:5102/invocations",
}

EXPECTED_VERSIONS = {
    "prod": "v1",
    "candidate": "v2",
}

WARMUPS = 5


def main():
    rows = (
        pd.read_csv("data/eta_fresh.csv")[FEATURES]
        .head(200)
        .astype(float)
        .to_dict("records")
    )
    assert len(rows) == 200

    predictions = {"prod": [], "candidate": []}
    latencies = {"prod": [], "candidate": []}
    pairs = []

    with requests.Session() as session:

        def invoke(arm, row):
            started = time.perf_counter()

            response = session.post(
                URLS[arm],
                json={"dataframe_records": [row]},
                timeout=30,
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"{arm}: HTTP {response.status_code}: "
                    f"{response.text}"
                )

            body = response.json()
            elapsed_ms = (time.perf_counter() - started) * 1000

            output = body["predictions"]
            if len(output) != 1:
                raise ValueError("Expected one prediction")

            item = output[0]

            if item["model_version"] != EXPECTED_VERSIONS[arm]:
                raise ValueError(
                    f"Unexpected version from {arm}: {item}"
                )

            prediction = float(item["eta_minutes"])
            if not math.isfinite(prediction):
                raise ValueError("Nonfinite prediction")

            return prediction, elapsed_ms

        # Warm-ups are excluded from all reported measurements.
        for arm in URLS:
            for _ in range(WARMUPS):
                invoke(arm, rows[0])

        for index, row in enumerate(rows):
            # Alternate server order to reduce ordering bias.
            order = (
                ("prod", "candidate")
                if index % 2 == 0
                else ("candidate", "prod")
            )

            pair = {"row_index": index}

            for arm in order:
                prediction, elapsed_ms = invoke(arm, row)
                predictions[arm].append(prediction)
                latencies[arm].append(elapsed_ms)

                pair[f"{arm}_prediction"] = prediction
                pair[f"{arm}_latency_ms"] = elapsed_ms

            pairs.append(pair)

    # Both outputs have one decimal place. Round differences to
    # avoid floating-point noise misclassifying exactly 10 minutes.
    differences = np.round(
        np.abs(
            np.array(predictions["prod"])
            - np.array(predictions["candidate"])
        ),
        1,
    )

    results = {
        "paired_requests": len(rows),
        "warmup_requests_per_server": WARMUPS,
        "verified_versions": EXPECTED_VERSIONS,
        "mean_absolute_difference_minutes": float(
            differences.mean()
        ),
        "percentage_difference_gt_10_minutes": float(
            (differences > 10).mean() * 100
        ),
        "p95_ms": {
            arm: float(np.percentile(values, 95))
            for arm, values in latencies.items()
        },
    }

    Path("partB/shadow_pairs.json").write_text(
        json.dumps(pairs, indent=2) + "\n"
    )

    output = json.dumps(results, indent=2)
    Path("partB/shadow_results.json").write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
