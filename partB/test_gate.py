import pytest

from eta_mlflow import decide


PROD = {
    "mae": 100.0,
    "mae_rush": 100.0,
    "mae_rain": 100.0,
    "p95_ms": 2.0,
}

CANDIDATE = {
    "mae": 90.0,
    "mae_rush": 95.0,
    "mae_rain": 95.0,
    "p95_ms": 5.0,
}


def test_all_criteria_pass():
    result = decide(PROD, CANDIDATE)

    assert result["promote"] is True
    assert all(result["criteria"].values())


@pytest.mark.parametrize(
    "field,value,criterion",
    [
        ("mae", 97.0, "overall_mae"),
        ("p95_ms", 10.0, "latency"),
        ("mae_rush", 105.0, "rush_slice"),
        ("mae_rain", 105.0, "rain_slice"),
    ],
)
def test_exact_boundary_passes(field, value, criterion):
    candidate = dict(CANDIDATE, **{field: value})
    result = decide(PROD, candidate)

    assert result["criteria"][criterion] is True
    assert result["promote"] is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("mae", 96.9999),
        ("p95_ms", 9.9999),
        ("mae_rush", 104.9999),
        ("mae_rain", 104.9999),
    ],
)
def test_just_inside_threshold_passes(field, value):
    candidate = dict(CANDIDATE, **{field: value})

    assert decide(PROD, candidate)["promote"] is True


@pytest.mark.parametrize(
    "field,value,criterion",
    [
        ("mae", 97.0001, "overall_mae"),
        ("p95_ms", 10.0001, "latency"),
        ("mae_rush", 105.0001, "rush_slice"),
        ("mae_rain", 105.0001, "rain_slice"),
    ],
)
def test_just_outside_threshold_blocks_promotion(
    field, value, criterion
):
    candidate = dict(CANDIDATE, **{field: value})
    result = decide(PROD, candidate)

    assert result["criteria"][criterion] is False
    assert result["promote"] is False

    # Only the deliberately broken criterion should fail.
    assert all(
        passed
        for name, passed in result["criteria"].items()
        if name != criterion
    )


def test_strict_gain_rejects_ten_percent_improvement():
    result = decide(
        PROD,
        CANDIDATE,
        min_rel_gain=0.60,
    )

    assert result["criteria"]["overall_mae"] is False
    assert result["promote"] is False


@pytest.mark.parametrize(
    "candidate_mae,expected",
    [(40.0, True), (40.0001, False)],
)
def test_sixty_percent_gain_boundary(candidate_mae, expected):
    candidate = dict(CANDIDATE, mae=candidate_mae)
    result = decide(
        PROD,
        candidate,
        min_rel_gain=0.60,
    )

    assert result["promote"] is expected


def test_gain_is_relative_not_absolute():
    prod = dict(PROD, mae=10.0)
    candidate = dict(CANDIDATE, mae=9.8)

    # A 0.2 absolute reduction is only a 2% relative improvement.
    result = decide(prod, candidate)

    assert result["criteria"]["overall_mae"] is False
    assert result["promote"] is False


def test_latency_limit_applies_to_candidate():
    slow_prod = dict(PROD, p95_ms=1000.0)

    # Production latency does not change the candidate's 10 ms limit.
    assert decide(slow_prod, CANDIDATE)["promote"] is True
