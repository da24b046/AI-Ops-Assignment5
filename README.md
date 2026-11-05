# SwiftParcel ETA — Module 4, Assignment 5

CPU-only delivery-time prediction served through FastAPI, MLflow and
Ray Serve, with a release smoke gate.

## Environment

The recorded runs used Python 3.14.4 on Linux. Exact installed package
versions are recorded in requirements-lock.txt.

Run all commands from the repository root.

For a fresh environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
source env.sh
mkdir -p evidence/screenshots
```

On a compatible Python/platform, requirements-lock.txt can be used
instead of requirements.txt to install the recorded environment versions.

In every additional terminal, enter the repository, activate the same
environment, and run:

```bash
source env.sh
```

env.sh configures module imports, SQLite tracking and one native
numerical-library thread per process. It defaults ETA_MODEL to v1.

## Generate data and models

Model files and databases are intentionally excluded from submission.

```bash
python eta_data.py
python - <<'PY'
import pandas as pd
from eta_data import FEATURES

pd.read_csv("data/eta_fresh.csv")[FEATURES].head(200).to_csv(
    "data/eta_sample.csv", index=False
)
PY
```

## Run the tests

```bash
python -m pytest partA/test_eta_app.py -v 2>&1 | tee evidence/pytest_A.txt
python -m pytest partB/test_gate.py -v 2>&1 | tee evidence/pytest_B.txt
```

Recorded results: 28 Part A cases and 18 promotion-gate cases passed.

## Part A: REST service and measurements

In a server terminal:

```bash
python -m uvicorn eta_app:app --host 127.0.0.1 --port 8000 --workers 1
```

In a second prepared terminal:

```bash
curl --fail http://127.0.0.1:8000/healthz

python load_test.py \
  --url http://127.0.0.1:8000/v1/models/eta:predict \
  --data data/eta_sample.csv \
  --concurrency 1 2 4 8 16 --duration 4 --batch 1 \
  --out partA/loadtest_results.json \
  2>&1 | tee evidence/loadtest_baseline.txt

python partA/measure_batch.py 2>&1 | tee evidence/batch_cost.txt

curl -sS http://127.0.0.1:8000/metrics > evidence/metrics.txt
```

Stop the one-worker server with Ctrl+C and restart it with two workers:

```bash
python -m uvicorn eta_app:app --host 127.0.0.1 --port 8000 --workers 2
```

Repeat the load measurement in the second terminal:

```bash
python load_test.py \
  --url http://127.0.0.1:8000/v1/models/eta:predict \
  --data data/eta_sample.csv \
  --concurrency 1 2 4 8 16 --duration 4 --batch 1 \
  --out partA/loadtest_2workers.json \
  2>&1 | tee evidence/loadtest_2workers.txt
```

## Part D: release smoke gate

With the Part A v1 server still running:

```bash
python partD/smoke_eta.py \
  --url http://127.0.0.1:8000/v1/models/eta:predict \
  --expect-version v1 --max-p95-ms 80 \
  --out partD/smoke_results.json \
  2>&1 | tee partD/smoke_output.txt
```

Expected exit status: 0 when every check passes.

Demonstrate rejection using an intentionally incorrect version:

```bash
python partD/smoke_eta.py \
  --url http://127.0.0.1:8000/v1/models/eta:predict \
  --expect-version v999 --max-p95-ms 80 \
  --out evidence/smoke_wrong_version.json \
  > evidence/smoke_wrong_version.txt 2>&1

smoke_status=$?
printf '%s\n' "$smoke_status" > evidence/smoke_wrong_version_exit.txt
cat evidence/smoke_wrong_version.txt
printf 'Exit status: %s\n' "$smoke_status"
```

Expected exit status: 1. Do not enable shell exit-on-error for this
intentional-failure demonstration.

Stop the Part A server before continuing.

## Part B: registration and promotion gate

```bash
python partB/eta_mlflow.py register 2>&1 | tee evidence/registry.txt
python partB/eta_mlflow.py gate 2>&1 | tee partB/gate_output.txt
```

Registration assigns prod to v1 and candidate to v2. Registry version
integers can increase on repeated registration. The gate evaluates
aliases without changing them.

### Shadow comparison

Start production in one prepared terminal:

```bash
mlflow models serve \
  -m 'models:/eta-model@prod' \
  --host 127.0.0.1 -p 5101 --env-manager local
```

Start candidate in another prepared terminal:

```bash
mlflow models serve \
  -m 'models:/eta-model@candidate' \
  --host 127.0.0.1 -p 5102 --env-manager local
```

After both servers are ready, use the control terminal:

```bash
curl --fail http://127.0.0.1:5101/ping
curl --fail http://127.0.0.1:5102/ping

python partB/shadow_compare.py 2>&1 | tee partB/shadow_output.txt
```

The script measures 200 paired requests after five excluded warm-ups
per server and saves both summary and raw paired measurements.

Stop both MLflow servers with Ctrl+C after measurement.

## Part C: Ray Serve verification

Port 8000 must be available. Check for existing listeners:

```bash
ss -ltnp '( sport = :8000 )'
```

Then run:

```bash
python partC/verify_ray.py 2>&1 | tee partC/verify_output.txt
```

The script creates its own two-CPU Ray cluster, verifies allocation,
repeat stickiness, identical redeployment, increasing canary weight and
header overrides, then shuts its application and cluster down.

## Results and report

REPORT.md contains the recorded measurements, explanations, proposed
abort criteria, limitations and attribution. Rerunning measurements
overwrites their result files; update the report to match any new runs.

Screenshots are supplementary evidence under evidence/screenshots.
The assignment submission uses partA/, partB/, partC/ and partD/.
Exclude model bundles, datasets, databases, MLflow artifacts and virtual
environments from the final ZIP.

