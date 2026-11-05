"""Verify sticky canary routing using real HTTP requests."""

import json
from pathlib import Path

import ray
import requests
from ray import serve

from eta_ray import build


URL = "http://127.0.0.1:8000/v1/models/eta:predict"

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
    result_path = Path("partC/verify_results.json")

    # Prevent an earlier successful result surviving a failed rerun.
    result_path.unlink(missing_ok=True)

    ray.init(
        address="local",
        num_cpus=2,
        include_dashboard=False,
    )

    try:
        cpu_count = ray.cluster_resources()["CPU"]
        assert cpu_count == 2, cpu_count
        print(f"Ray cluster CPU limit: {cpu_count}", flush=True)

        serve.run(
            build(weight=0.2),
            name="eta",
            route_prefix="/",
        )

        with requests.Session() as session:

            def get_arm(customer_id, beta=False):
                response = session.post(
                    URL,
                    json={
                        "customer_id": customer_id,
                        "instances": [ROW],
                    },
                    headers={"X-Beta": "1"} if beta else {},
                    timeout=30,
                )

                if response.status_code != 200:
                    raise RuntimeError(
                        f"HTTP {response.status_code}: "
                        f"{response.text}"
                    )

                body = response.json()
                arm = body["arm"]

                assert arm in {"v1", "v2"}
                assert body["model_version"] == arm
                assert len(body["predictions"]) == 1

                return arm

            customers = [
                f"customer-{index}"
                for index in range(1000)
            ]

            original = {
                customer: get_arm(customer)
                for customer in customers
            }

            candidate_count = sum(
                arm == "v2" for arm in original.values()
            )
            candidate_share = candidate_count / len(customers)

            assert abs(candidate_share - 0.2) <= 0.04, (
                candidate_share
            )

            print(
                f"PASS share: {candidate_count}/1000 "
                f"= {candidate_share:.2%}; required 16%-24%",
                flush=True,
            )

            repeated_customers = customers[:200]

            assert all(
                get_arm(customer) == original[customer]
                for customer in repeated_customers
            )
            print(
                "PASS repeat stickiness: 200/200",
                flush=True,
            )

            # Redeploy the identical application.
            serve.run(
                build(weight=0.2),
                name="eta",
                route_prefix="/",
            )

            assert all(
                get_arm(customer) == original[customer]
                for customer in repeated_customers
            )
            print(
                "PASS identical-redeploy stickiness: 200/200",
                flush=True,
            )

            # Increase the threshold while retaining the same hash.
            serve.run(
                build(weight=0.5),
                name="eta",
                route_prefix="/",
            )

            expanded = {
                customer: get_arm(customer)
                for customer in customers
            }

            moved_back = sum(
                original[customer] == "v2"
                and expanded[customer] == "v1"
                for customer in customers
            )

            assert moved_back == 0, moved_back
            print(
                "PASS monotonic rollout: 0 candidate customers "
                "moved back to production",
                flush=True,
            )

            # Choose customers normally routed to production at 0.5.
            override_customers = [
                customer
                for customer in customers
                if expanded[customer] == "v1"
            ][:50]

            assert len(override_customers) == 50

            assert all(
                get_arm(customer, beta=True) == "v2"
                for customer in override_customers
            )

            # The override must not permanently change assignment.
            assert all(
                get_arm(customer) == "v1"
                for customer in override_customers
            )

            print(
                "PASS beta override: 50/50; normal assignments "
                "restored afterwards",
                flush=True,
            )

        results = {
            "ray_cpu_limit": cpu_count,
            "distinct_customers": len(customers),
            "candidate_count": candidate_count,
            "candidate_share": candidate_share,
            "repeat_customers": len(repeated_customers),
            "redeploy_customers": len(repeated_customers),
            "candidate_moved_back": moved_back,
            "beta_override": len(override_customers),
            "normal_assignment_restored": True,
            "passed": True,
        }

        result_path.write_text(
            json.dumps(results, indent=2) + "\n"
        )
        print("\n" + json.dumps(results, indent=2), flush=True)

    finally:
        try:
            serve.shutdown()
        finally:
            ray.shutdown()


if __name__ == "__main__":
    main()
