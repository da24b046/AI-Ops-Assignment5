# SwiftParcel ETA — Module 4, Assignment 5

## Machine and execution environment

- CPU: Intel Core i7-13620H, 16 logical CPU cores.
- RAM: 14.82 GiB.
- OS: Linux x86_64.
- Python: 3.14.4.
- CPU only; native numerical-library threads limited to one.
- Full installed package versions are recorded in requirements-lock.txt.

## Part A — REST service

### Implementation and tests

The FastAPI service loads the selected model once per worker from
ETA_MODEL, defaulting to data/eta_v1.joblib. It validates feature ranges,
batch sizes and delivery feasibility, preserves prediction order, and
returns predictions rounded to one decimal place.

Responses carry request IDs. Validation errors use the required 422
structure, inference errors return a generic 500, and Prometheus metrics
exclude requests to /metrics. The Part A test suite passed all 28 cases.
Five dependency deprecation warnings were observed, with no test failures.

### Baseline load test: one worker

Method: supplied closed-loop load generator, loopback HTTP, batch size 1,
and a four-second run at each concurrency. Client and server ran on the
same laptop. The supplied generator calculates throughput as completed
requests divided by the requested four-second duration; request draining
can make this a slight overestimate of wall-clock throughput.

| Concurrency | Requests | Requests/s | p50 ms | p95 ms | p99 ms | Errors |
|---|---|---|---|---|---|---|
| 1 | 2459 | 614.8 | 1.6 | 1.9 | 2.3 | 0 |
| 2 | 3247 | 811.8 | 2.4 | 2.8 | 3.7 | 0 |
| 4 | 3218 | 804.5 | 4.7 | 6.2 | 7.5 | 0 |
| 8 | 3202 | 800.5 | 9.7 | 12.5 | 15.3 | 0 |
| 16 | 3193 | 798.2 | 19.7 | 24.8 | 28.8 | 0 |

The baseline saturation knee is around concurrency 2: throughput reaches
811.8 requests/s and then remains approximately flat, while latency rises.
Increasing concurrency from 2 to 16 raises p95 from 2.8 to 24.8 ms without
increasing throughput.

Little's Law gives L = throughput × mean response time. With negligible
client think time, the closed-loop concurrency approximates L, so
concurrency / throughput estimates mean response time. At concurrency 2
this is approximately 2.46 ms; at concurrency 16 it is approximately
20.05 ms. These are estimates, not measured means or p95 values. The
increasing time with flat throughput is consistent with queuing.

All tested concurrency levels satisfy p95 <= 80 ms with zero errors.
The highest measured SLO-compliant baseline throughput is therefore
811.8 requests/s at concurrency 2. This is the best observed capacity
within this sweep, not a proven absolute maximum.

### Batch cost at concurrency 1

Each batch size used ten excluded warm-up requests followed by 200
measured sequential requests. Timing includes HTTP, request serialization,
validation, prediction and response decoding.

| Batch size | Mean request ms | Mean ms/row |
|---|---|---|
| 1 | 1.608542 | 1.608542 |
| 100 | 2.165732 | 0.021657 |

Cost per row is mean request time divided by the actual number of rows.
Batch 100 costs approximately 74.3 times less per row because fixed HTTP
and model-call overhead is shared across rows. Total batch latency still
increases from about 1.61 to 2.17 ms; lower per-row cost does not mean
lower latency for an individual batch request.

### Capacity change: two workers

I increased Uvicorn from one worker process to two to allow service work
to use a second CPU core, retaining one native numerical-library thread
per worker. I repeated the same load-test sweep.

| Concurrency | Requests | Requests/s | p50 ms | p95 ms | p99 ms | Errors |
|---|---|---|---|---|---|---|
| 1 | 2517 | 629.2 | 1.5 | 1.8 | 2.0 | 0 |
| 2 | 4039 | 1009.8 | 1.9 | 2.4 | 2.8 | 0 |
| 4 | 4992 | 1248.0 | 3.3 | 5.1 | 6.1 | 0 |
| 8 | 5594 | 1398.5 | 5.5 | 8.6 | 10.8 | 0 |
| 16 | 5693 | 1423.2 | 10.7 | 16.9 | 21.5 | 0 |

The highest observed SLO-compliant throughput increased from 811.8 to
1423.2 requests/s: a 1.75x improvement, approximately 75.3% higher.
The proposed change did not double capacity. Possible remaining limits
include client overhead, HTTP/JSON processing and process scheduling;
these bottlenecks were not separately profiled.

Throughput growth slows between concurrency 8 and 16 while p95 rises
from 8.6 to 16.9 ms, suggesting another saturation region. Longer runs
and additional concurrency levels would be needed to characterize it.

The generator's replicas column is not a worker count here: it counts
distinct served_by response values, which this API does not expose.
Startup logs provide evidence of the two worker processes. Prometheus
metrics are process-local; production multiworker monitoring would need
aggregation.

### Why p95/p99 instead of only the mean?

p95 and p99 expose slow requests and queuing that an average can hide.
They reveal tail-latency problems affecting users even when most
requests are fast.

### Measurement limitations

Each concurrency measurement was a single four-second run, so results
have limited statistical precision. These measurements describe this
laptop and local client/server setup, not external production capacity.

## Part B — MLflow registry, promotion gate and shadow

### Registry

Each custom pyfunc loads its bundle through load_context and returns a
DataFrame with eta_minutes rounded to one decimal and model_version.
Both models were logged with an explicit signature, input example,
artifact bundle, supporting code and pinned pip requirements.
Tracking and registry use SQLite.

| Registry version | Bundle | Alias |
|---|---|---|
| 1 | v1 | prod |
| 2 | v2 | candidate |


### Multi-criteria promotion gate

The pure decide function requires all four criteria to pass: relative
overall MAE improvement, candidate single-row p95, rush-hour MAE, and
rain-slice MAE. The synthetic test suite passed all 18 cases, including
exact, just-inside and just-outside threshold boundaries.

Evaluation used make_data(2000, seed=99). Rush hours are 7–9 and 16–19
inclusive; the rain slice uses rain == 1. Each model received ten
excluded warm-ups followed by 200 timed single-row pyfunc predictions.
These timings exclude model loading and HTTP transport.

| Gate configuration | Alias | MAE | Rush MAE | Rain MAE | p95 ms |
|---|---|---|---|---|---|
| default | prod | 11.9553 | 19.6799 | 11.6227 | 0.9542 |
| default | candidate | 6.1049 | 8.5795 | 6.2853 | 0.9362 |
| strict_gain_0.60 | prod | 11.9553 | 19.6799 | 11.6227 | 0.9558 |
| strict_gain_0.60 | candidate | 6.1049 | 8.5795 | 6.2853 | 0.9041 |


- **default**: thresholds `{"min_rel_gain": 0.03, "max_p95_ms": 10.0, "max_slice_regression": 0.05}`; criteria `{"overall_mae": true, "latency": true, "rush_slice": true, "rain_slice": true}`; promote = **True**.
- **strict_gain_0.60**: thresholds `{"min_rel_gain": 0.6, "max_p95_ms": 10.0, "max_slice_regression": 0.05}`; criteria `{"overall_mae": false, "latency": true, "rush_slice": true, "rain_slice": true}`; promote = **False**.

Overall MAE improved by 48.94%. This exceeds the default 3%
requirement but falls below 60%, explaining the strict gate rejection.
Gate evaluation did not change either alias.

### Paired shadow comparison

The same 200 fresh-data rows were sent individually
to both MLflow HTTP servers. Server order alternated between pairs,
and 5 warm-up requests per server
were excluded. Returned model versions were checked on every response.
HTTP timings include request serialization and response decoding.

| Measurement | Result |
|---|---|
| Mean absolute prediction difference | 9.4845 minutes |
| Prediction pairs differing by more than 10 minutes | 35.00% |
| Production HTTP p95 | 2.8905 ms |
| Candidate HTTP p95 | 2.8390 ms |

I would recommend v2 for a monitored 10% canary because the default fresh-data quality and inference-latency criteria all passed. The observed prediction differences warrant monitoring
customer impact, while labelled fresh-data overall and slice MAE provide
the accuracy evidence for this recommendation.
An already running MLflow server retains the model loaded when its alias
was resolved, so moving that alias does not replace its in-memory model.
One update strategy is to start and warm a new version, pass readiness
checks, switch a load balancer, and drain the old servers.
A second strategy is a rolling replacement with sufficient ready
replicas, readiness checks and graceful draining to keep serving requests.

## Part C — Ray Serve sticky canary

### Implementation and verification

Each EtaModel replica loads its bundle once. Production has two fixed
replicas; the candidate has one to three autoscaling replicas.
The FastAPI router accepts customer_id and instances, returns the
prediction plus arm, and supports the X-Beta: 1 override.

Verification deployed the application on a Ray cluster explicitly
limited to 2 CPUs and used real HTTP calls.

| Check | Measured result |
|---|---|
| Candidate share at weight 0.2 | 192/1000 = 19.20%; within 16–24% |
| Repeated customer calls | 200/200 kept the same arm |
| Identical application redeployment | 200/200 kept the same arm |
| Increase weight from 0.2 to 0.5 | 0 candidate customers moved back to production |
| X-Beta: 1 override | 50/50 routed to candidate |
| Assignment after removing override | Normal production assignments restored for all 50 |

### Why built-in hash() is unsuitable

Python salts string hashes between interpreter processes, so hash() can
assign the same customer differently after a restart or across replicas.
The router instead uses SHA-256 and converts 53 digest bits into a
deterministic fraction in [0,1). Customers below the weight threshold
receive v2; increasing the threshold preserves existing candidate
assignments.

### Two reasons stickiness matters

For the ETA product, a stable assignment avoids model switching causing
inconsistent estimates for the same customer. For experiment analysis,
customer-level assignment keeps repeated observations in one treatment
group, reducing contamination and supporting meaningful customer-level
comparisons; the allocation share is measured over distinct customers.

### Autoscaling justification

The candidate targets two ongoing requests per replica, with a maximum
of eight admitted ongoing requests per replica to provide headroom.
A five-second upscale delay responds to sustained demand, while a
thirty-second downscale delay reduces rapid scale-down oscillation.
The maximum of three candidate replicas limits resource growth while
the two production replicas remain available.
Each model replica reserves 0.25 CPU and the router reserves 0.1 CPU,
giving a maximum deployment reservation of 1.35 CPUs within the
two-CPU cluster; these are initial settings rather than load-tuned
parameters, and sustained autoscaling behaviour was not tested.

## Part D — Release gate

### Smoke test evidence

The running v1 REST service passed all seven smoke checks: finite and
nonnegative prediction, matching request ID, longer-distance sanity,
heavier-traffic sanity, invalid-input HTTP 422, expected model version,
and a latency threshold.

The empirical p95 across 30 timed requests after sanity probes was
1.990 ms, below the 80 ms limit. The successful run
returned exit 0. Deliberately expecting v999 against v1 failed the
version check and returned exit 1, demonstrating release rejection.

These fixed-input sanity checks do not prove global model monotonicity.
Thirty timed requests provide a lightweight smoke check, not a reliable
production tail-latency estimate.

### Proposed abort criteria for a 10% v2 canary

The following are proposed monitoring rules, not production experiments
that I ran. HTTP metrics must be labelled by model version and aggregated
across replicas.

| Metric | Abort threshold | Evaluation window and minimum sample |
|---|---|---|
| v2 HTTP 5xx rate | Above 1% and more than 0.5 percentage points above v1 | Rolling 5 minutes; at least 100 requests per arm |
| v2 prediction HTTP p95 | Above 80 ms | Three consecutive 1-minute windows; at least 100 v2 requests in each |
| v2 overall MAE against delivered ETA labels | More than 5% above v1 MAE | Trailing 24 hours, evaluated hourly; at least 100 labelled deliveries per arm |
| v2 rush-hour or rain-slice MAE | Either slice more than 5% above the matching v1 slice | Trailing 24 hours, evaluated hourly; at least 50 labelled deliveries per arm in the slice |

Quality criteria can only be evaluated when actual delivery labels
arrive. Insufficient samples mean insufficient evidence, not an automatic
pass. Aborting would route new normal requests back to v1 and allow
in-flight requests to finish.

## Overall testing limitations and NOT TESTED list

Completed: Part A tests and HTTP checks, one/two-worker load sweeps,
batch measurements, registry registration, promotion-gate tests and
fresh-data evaluation, paired MLflow shadow comparison, Ray HTTP
verification, and successful/intentional-failure smoke runs.

NOT TESTED:
- Sustained load that demonstrates actual candidate autoscaling.
- Live production traffic and automated execution of the abort policy.
- Delayed-label monitoring on real deliveries.
- Blue-green/rolling deployment, failover and connection draining.
- Global model monotonicity across the feature space.
- Long-duration or repeated capacity experiments.
- The release smoke script against v2; its measured runs used v1.

## Attribution

The supplied Module4_Assignment_Starter eta_data.py and load_test.py
were used unchanged. No lecture notebook code was used in the steps
documented here. Implementation and terminal guidance were prepared
with OpenAI Codex assistance; the reported execution results were
recorded from my laptop.

API references:
- [MLflow pyfunc API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html)
- [Ray Serve autoscaling](https://docs.ray.io/en/latest/serve/advanced-guides/advanced-autoscaling.html)
