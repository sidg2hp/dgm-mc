import numpy as np
import pytest

from dgm_mc.selection import choose_deployment


def test_cost_aware_selection_can_choose_classical_fallback():
    rng = np.random.default_rng(4)
    y = rng.normal(size=200)
    classical = y + rng.normal(scale=.1, size=200)
    neural = rng.normal(size=200)  # intentionally poor / not a useful martingale control
    choice = choose_deployment(
        y, np.column_stack([classical, neural]),
        control_names=("geometric", "neural"), pilot_seconds=.01,
        evaluation_seconds=.001, training_seconds=100., expected_queries=1,
        target_se=.1,
    )
    assert not choice.accepted
    assert "classical" in choice.reason


def test_cost_aware_selection_accepts_neural_only_when_amortized_benefit_wins():
    rng = np.random.default_rng(8)
    y = rng.normal(size=500)
    classical = y + rng.normal(scale=.5, size=500)
    neural = y + rng.normal(scale=.01, size=500)
    choice = choose_deployment(
        y, np.column_stack([classical, neural]),
        control_names=("geometric", "neural"), pilot_seconds=.01,
        evaluation_seconds=.001, training_seconds=.1, expected_queries=1000,
        target_se=.01,
    )
    assert choice.accepted
    assert "neural" in choice.name


def test_selection_rejects_bad_inputs():
    with pytest.raises(ValueError):
        choose_deployment([1, 2], np.zeros((2, 1)), control_names=("neural",),
                          pilot_seconds=0, evaluation_seconds=0)
    with pytest.raises(ValueError):
        choose_deployment([1, 2, 3], np.zeros((3, 1)), control_names=("neural",),
                          pilot_seconds=-1, evaluation_seconds=0)
