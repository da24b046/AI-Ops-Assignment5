"""Measure end-to-end batch cost with one sequential client."""

import json
import statistics
import time
from pathlib import Path

import pandas as pd
import requests

from eta_data import FEATURES


URL = "http://127.0.0.1:8000/v1/models/eta:predict"
WARMUP_REQUESTS = 10
MEASURED_REQUESTS = 200

rows = pd.read_csv("data/eta_sample.csv")[FEATURES].to_dict("records")
results = []

with requests.Session() as session:
    for batch_size in (1, 100):
        payload = {"instances": rows[:batch_size]}
        assert len(payload["instances"]) == batch_size

        def send_request():
            response = session.post(
                URL,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            body = response.json()

            assert body["model_version"] == "v1"
            assert len(body["predictions"]) == batch_size

        for _ in range(WARMUP_REQUESTS):
            send_request()

        durations_ms = []

        for _ in range(MEASURED_REQUESTS):
            started = time.perf_counter()
            send_request()
            durations_ms.append(
                (time.perf_counter() - started) * 1000
            )

        mean_ms = statistics.mean(durations_ms)

        results.append({
            "batch_size": batch_size,
            "concurrency": 1,
            "warmup_requests": WARMUP_REQUESTS,
            "measured_requests": MEASURED_REQUESTS,
            "mean_request_ms": mean_ms,
            "ms_per_row": mean_ms / batch_size,
        })

output = json.dumps(results, indent=2)
Path("partA/batch_cost.json").write_text(output + "\n")

print(output)

print("\nSummary:")
print(f"{'Batch size':>10} {'Mean request ms':>18} {'ms/row':>12}")
for result in results:
    print(
        f"{result['batch_size']:>10} "
        f"{result['mean_request_ms']:>18.4f} "
        f"{result['ms_per_row']:>12.6f}"
    )
