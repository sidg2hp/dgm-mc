"""Cost-aware deployment selection for the proposed DGM--MC protocol.

This module does not claim a new variance-reduction theorem. It makes the
deployment choice explicit: compare plain MC, classical controls, and neural
controls using an independent pilot, then select the lowest estimated total
cost for a requested standard error and query count. The neural option is
allowed to lose; a conservative fallback is part of the method.
"""
from dataclasses import dataclass

import numpy as np

from .estimators import fit_controls


@dataclass(frozen=True)
class DeploymentChoice:
    name: str
    coefficients: tuple[float, ...]
    pilot_seconds: float
    evaluation_seconds: float
    training_seconds: float
    predicted_variance: float
    predicted_paths: int
    predicted_total_seconds: float
    accepted: bool
    reason: str


def choose_deployment(
    pilot_payoff,
    pilot_controls,
    *,
    control_names,
    pilot_seconds,
    evaluation_seconds,
    training_seconds=0.0,
    expected_queries=1,
    target_se=1e-3,
    safety_factor=1.25,
):
    """Select the lowest predicted total cost among valid control subsets.

    ``pilot_controls`` columns must be fresh zero-mean controls on the pilot
    paths. This routine chooses among plain MC, each prefix of the supplied
    control ordering, and the full set. It does not use evaluation outcomes.

    The estimate is intentionally conservative: required paths are inflated by
    ``safety_factor**2`` and the neural training cost is charged once per
    ``expected_queries``. A returned ``accepted=False`` means the neural/full
    option was not cheaper than the best non-neural option under the pilot.
    """
    y = np.asarray(pilot_payoff, dtype=float)
    controls = np.asarray(pilot_controls, dtype=float)
    if y.ndim != 1 or controls.ndim != 2 or len(y) != len(controls) or len(control_names) != controls.shape[1]:
        raise ValueError("Pilot shapes and control names must agree")
    if len(y) < 3 or not np.isfinite(y).all() or not np.isfinite(controls).all():
        raise ValueError("Pilot values must be finite and contain at least three paths")
    if any(x < 0 for x in (pilot_seconds, evaluation_seconds, training_seconds)):
        raise ValueError("Costs cannot be negative")
    if expected_queries < 1 or target_se <= 0 or safety_factor < 1:
        raise ValueError("Invalid deployment assumptions")

    candidates = [("plain", np.empty((len(y), 0)), np.zeros(0))]
    # Prefixes make the deployment ladder interpretable: classical first,
    # then neural. The full set may be redundant and is still evaluated.
    for width in range(1, controls.shape[1] + 1):
        current = controls[:, :width]
        beta = fit_controls(y, current)
        candidates.append(("+".join(control_names[:width]), current, beta))

    measured = []
    for name, current, beta in candidates:
        adjusted = y if current.shape[1] == 0 else y - current @ beta
        variance = float(np.var(adjusted, ddof=1))
        paths = max(2, int(np.ceil(variance * safety_factor ** 2 / target_se ** 2)))
        # Evaluation cost is measured per pilot path in this contract. This is
        # deliberately simple and auditable; a production system should record
        # separate batched cost curves.
        eval_cost = evaluation_seconds * paths / len(y)
        has_neural = any("neural" in name.lower() for name in name.split("+"))
        fixed_cost = pilot_seconds + (training_seconds / expected_queries if has_neural else 0.0)
        total = fixed_cost + eval_cost
        measured.append((total, DeploymentChoice(
            name=name, coefficients=tuple(float(x) for x in beta),
            pilot_seconds=float(pilot_seconds if name != "plain" else 0.0),
            evaluation_seconds=float(eval_cost), training_seconds=float(training_seconds if has_neural else 0.0),
            predicted_variance=variance, predicted_paths=paths,
            predicted_total_seconds=float(total), accepted=False, reason="candidate")))
    measured.sort(key=lambda item: item[0])
    best_total, best = measured[0]
    non_neural = [item for item in measured if "neural" not in item[1].name.lower()]
    baseline_total, baseline = min(non_neural, key=lambda item: item[0])
    accepted = "neural" in best.name.lower() and best_total < baseline_total
    reason = "neural option selected by pilot cost model" if accepted else "classical fallback selected by pilot cost model"
    return DeploymentChoice(**{**best.__dict__, "accepted": accepted, "reason": reason})
