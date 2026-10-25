# Research audit — 26 October 2026

## Verdict

The inherited results do not support the proposed near-optimal/SOTA headline.
An A/A* submission on those claims would face fundamental correctness,
novelty, and experimental-validity objections. Venue acceptance cannot be
guaranteed by software changes. The replacement manuscript is a reproducibility
and estimator-validation study, not a disguised claim that these issues were
already solved.

Scope: both sibling projects' numerical modules, trainers, experiment callers,
tests, result JSONs/configurations, reporting scripts, READMEs and manuscripts.
Two independent read-only audits covered numerical correctness and artifact
provenance respectively. Binary checkpoints were not loaded or trusted.

## Blocking findings

| Finding | Evidence in original workspace | Consequence / replacement |
|---|---|---|
| Invalid terminal control mean | `dgm_option_pricing/evaluation/monte_carlo.py:198–223`; `neurips_submission/experiments/run_hybrid_mc.py:139–162` | A network's time-zero prediction is not the known expectation of its near-terminal prediction. Replace with a predictable zero-mean stochastic integral. |
| Saved price contradicts zero-bias claim | `neurips_submission/results/hybrid_mc/hybrid_mc_summary.json:2–11` | 1D price 0.095156352459 versus exact 0.104505835722, about 8.946% low, despite SE 4.32e-7. |
| Derived numbers presented alongside runs | `dgm_option_pricing/scratch_fix.py:10–29` and `scratch_fix2.py` | SEs rescaled from another experiment; coefficient hard-coded. Quarantine legacy numbers rather than treat them as new runs. |
| Metric substitution | `neurips_submission/paper.tex:389–407` versus `results/scaling/d_5/result.json` and `results/paper/comparison/d_5/comparison_results.json` | 2.40% is five-point MRE, not 100-state relative L2 (11.0255%). Regenerate all tables from explicit schemas. |
| Train/eval function mismatch | both `models/dgm_network.py:129–135` | Smoothed payoff becomes kinked at inference. Use smooth raw networks and exact-payoff terminal MSE. |
| False basket boundary target | both `losses/boundary_loss.py:23–41` | A basket call need not vanish when one asset is small. Use whole-space collocation, no artificial face penalty. |
| Biased squared stochastic residual | both `pde/operator.py:92–97,149–163` and `losses/pde_loss.py:49–55` | E[Rhat²]=R²+Var(Rhat). Use deterministic exact Hessian contraction. |
| Adaptive branch cannot differentiate | both `samplers/adaptive_sampler.py:88–106` | `no_grad` causes fallback; no importance weights. Omit this unsupported feature. |
| Incorrect antithetic uncertainty | both `evaluation/monte_carlo.py:82–99` | Use independent pair averages and sample variance (ddof=1). |
| Test-set checkpoint selection | `dgm_option_pricing/experiments/run_scaling.py:90–128` | Fixed training budgets and separate RNG namespaces; no test-based selection. |
| Non-deterministic L-BFGS closure | both `training/trainer.py:216–256` | New random trace probes each closure. Exact residual and no L-BFGS in the reference protocol. |
| Requested versus actual paths | `dgm_option_pricing/experiments/run_hybrid_mc.py:130–159` | 50k cap reported as requested 100k. Report actual paths and pilot costs separately. |

## Reporting and interpretation findings

- D's scaling aggregate and per-run files disagree on d=3 error and runtime,
  d=5 width/parameter count, d=7 steps, and d=10 runtime. Some d>=25 aggregate
  rows have no matching scaling run.
- `experiments/update_hmc.py` changes aggregates without changing their sources.
- Deep BSDE ATM errors are plotted with five-state mean errors. MC SE is
  uncertainty, not realized error. Five equal-asset diagonal states do not
  establish full-dimensional generalization.
- N's saved comparisons give DGM 6/20 wins, all ITM, not 7/20 or ATM superiority.
- The d=2 Greeks figure has a delta jump and negative gamma near ATM and is not
  evidence for the claimed d=1 analytical Greek errors.
- The resilience experiment compares a near-terminal prediction to a realized
  payoff, not a conditional price reference. Each training stage also inherits
  extra L-BFGS work.
- No replicated training-seed study was found. Missing checkpoints/config
  hashes and raw predictions prevent reliable reconstruction of some claims.
- Some validation paths print success for NaN because comparisons with NaN
  are false. The new implementation rejects nonfinite numerical records.
- Baseline citation DOI `10.1016/j.cam.2021.113508` resolves to Jeon and Kwak,
  **Pricing variable annuity with surrender guarantee**, not the stated neural
  regression paper (DOI resolver checked 2026-10-26). Fidelity to the stated
  baseline was not established; do not label it a verified replication.

## Checks performed during the audit

In each original folder, read-only runs of `python -B -m pytest -p no:cacheprovider
tests/test_pde_operator.py tests/test_monte_carlo.py tests/test_analytical.py
tests/test_payoffs.py -q` passed 24 tests. These suites do not exercise the key
hybrid estimator, high-dimensional stochastic loss, adaptive branch or selection
leakage. Passing them is not a validation of the central claims.

Small in-memory checks confirmed terminal-control bias even for an exact
Black–Scholes predictor, train/eval mismatch, antithetic SE error, and stochastic
loss noise. The new test suite independently protects the corrected behavior.
Full original training was not rerun; OmegaConf is absent from the active Python.

## Preservation and attribution

The new project does not modify, delete, reformat, stage, or commit the original
trees. `audit/source_manifest.json` fingerprints eligible public research files;
it is an integrity inventory, not a claim that every file was individually read.
No original checkpoint, environment, private configuration, or git history is
copied. Differences establish reporting problems, not intent to deceive.
