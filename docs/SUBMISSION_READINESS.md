# Submission readiness and remaining work

## Decision

**Do not submit this as a demonstrated SOTA method or industry standard.**
The original headline is invalid, and the corrected central estimator has clear
prior art. The new artifact can support a rigorous reconstruction/negative-result
manuscript, subject to human scientific review and venue fit. It does not establish
acceptance at an A/A* venue. No specific conference deadlines or ranking claims
are assumed.

## Delivered evidence

- Separate audited implementation and preserved original projects.
- Mathematical counterexample to incorrect terminal centering.
- Correct predictable martingale controls and independent coefficient pilots.
- Plain, antithetic, geometric-control and scrambled Sobol comparators.
- Three training seeds, DGM/MLP comparison, three heterogeneous-spot scenarios.
- Analytical geometric calibration, time-grid refinement, bad-gradient stress.
- Explicit reference uncertainty, timing, manifests and generated LaTeX tables.
- Learned-control 1D calibration: 80 replications per checkpoint, six checkpoints,
  fixed/pilot coefficients (960 estimates); 1,818 price/Delta/Gamma grid records.
- Cost-aware deployment selector implemented and unit-tested, with an explicit
  classical fallback; it has not yet been promoted to a headline empirical claim.

## Gates not yet met

1. **Novelty:** the defensible candidate is the complete DGM-MC deployment
   protocol—batched surrogate + predictable control + pilot cost selector +
   classical fallback—not the generic existence of neural martingale controls.
   Compare directly with Vidales et al. and Hinds/Tretyakov using their published
   algorithms and matched tasks.
2. **Training convergence:** the 300-step runs are deliberately short screening
   runs, not converged competitors. Predeclare a budget sweep, independent
   validation and hyperparameter selection before a larger held-out test.
3. **Coverage strength:** increase the 80 learned-control and 40 oracle-control
   replications to several hundred per configuration; add multidimensional
   analytical neural-control targets and adverse market regimes.
4. **Generalization:** hundreds of held-out full-dimensional states; heterogeneous
   volatility/correlation, rate and maturity shifts; near-expiry and tail regimes.
   Include different payoff families and stochastic volatility/jump models.
5. **Efficiency:** equal-wall-time and time-to-RMSE curves across path budgets,
   strong RQMC constructions, geometric/multi-asset controls, direct learned
   integrands, and Deep BSDE with time-discretization refinement. Charge training,
   data generation, pilot and training costs. Report amortization break-even
   queries and evaluate the cost-aware selector itself.
6. **High dimensions:** neural d=25/50/100 runs with multiple seeds and memory
   measurements. Classical-only results do not fill this gap.
7. **Risk outputs:** extend the delivered 1D analytical price/Delta/Gamma checks
   to multidimensional independent pathwise references and hedging-error tests.
   The present near-expiry Greek errors and negative surrogate values require
   improvement. Autograd is not a guarantee of accurate financial Greeks.
8. **Engineering:** independent clean-machine reproduction, CPU/GPU comparisons,
   peak memory, robust checkpoint interchange and long-run recovery. Confirm
   dependency availability and audit source licensing before a public release.
9. **Human review:** resolve final authorship, affiliations, licensing, desired
   venue template and page limit. The current generic article is not represented
   as compliant with a particular conference's current rules.

## Recommended scientific direction

Test the implemented residualized neural martingale and cost-aware selector on the
component unexplained by a cheap analytic control. Add a finite-sample selection
bound and robust matched-cost gains against a correctly implemented, time-refined
FBSDE. That could constitute a substantive contribution. Neither is claimed here.
If classical methods dominate, preserve that finding rather than weakening
baselines or relabelling oracle performance as a neural result.
