"""Release smoke gate: exit 0 on success, otherwise exit 1."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import requests


ROW = {
    "distance_km": 10,
    "n_stops": 2,
    "hour": 10,
    "weekday": 2,
    "traffic_index": 0.3,
    "rain": 0,
    "weight_kg": 3,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--expect-version", required=True)
    parser.add_argument("--max-p95-ms", type=float, default=80)
    parser.add_argument("--out")
    args = parser.parse_args()

    checks = []
    versions = set()
    p95_ms = None

    def check(name, passed):
        passed = bool(passed)
        checks.append({"check": name, "passed": passed})
        print(f"{'PASS' if passed else 'FAIL'}: {name}", flush=True)

    try:
        with requests.Session() as session:

            def predict(row):
                response = session.post(
                    args.url,
                    json={"instances": [row]},
                    timeout=10,
                )
                response.raise_for_status()
                body = response.json()

                predictions = body.get("predictions")
                if not isinstance(predictions, list) or len(predictions) != 1:
                    raise ValueError("Expected exactly one prediction")

                value = float(predictions[0])
                if not math.isfinite(value):
                    raise ValueError("Prediction is not finite")

                versions.add(body["model_version"])
                return value, body, response

            base, body, response = predict(ROW)

            check(
                "Single prediction is finite and nonnegative",
                math.isfinite(base) and base >= 0,
            )

            header_id = response.headers.get("X-Request-ID")
            check(
                "Request ID matches between header and body",
                bool(header_id) and header_id == body.get("request_id"),
            )

            longer, _, _ = predict(dict(ROW, distance_km=40))
            check(
                "Model sanity: longer distance increases ETA",
                longer > base,
            )

            heavier_traffic, _, _ = predict(
                dict(ROW, traffic_index=0.9)
            )
            check(
                "Model sanity: heavier traffic does not reduce ETA",
                heavier_traffic >= base,
            )

            invalid_response = session.post(
                args.url,
                json={"instances": [dict(ROW, distance_km=0)]},
                timeout=10,
            )
            check(
                "Invalid input returns HTTP 422",
                invalid_response.status_code == 422,
            )

            # Sanity probes above warm the service before timing.
            durations_ms = []

            for _ in range(30):
                started = time.perf_counter()
                predict(ROW)
                durations_ms.append(
                    (time.perf_counter() - started) * 1000
                )

            check(
                f"All predictions use expected version {args.expect_version}",
                versions == {args.expect_version},
            )

            # Nearest-rank empirical p95.
            durations_ms.sort()
            index = math.ceil(0.95 * len(durations_ms)) - 1
            p95_ms = durations_ms[index]

            check(
                f"p95 {p95_ms:.3f} ms <= {args.max_p95_ms:.3f} ms "
                "(30 timed requests)",
                p95_ms <= args.max_p95_ms,
            )

    except Exception as exc:
        check(
            f"Requests and response processing completed: "
            f"{type(exc).__name__}: {exc}",
            False,
        )

    result = {
        "checks": checks,
        "p95_ms": p95_ms,
        "expected_version": args.expect_version,
        "passed": bool(checks) and all(
            item["passed"] for item in checks
        ),
    }

    output = json.dumps(result, indent=2)

    if args.out:
        Path(args.out).write_text(output + "\n")

    print("\n" + output)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
