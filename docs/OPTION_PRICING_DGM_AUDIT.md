# Audit of `option-pricing-DGM`

The remote repository is the source of the current DGM-MC narrative. Its code
and README were inspected from the `main` branch on 2026-10-26, alongside the
local inherited artifacts already fingerprinted in `audit/source_manifest.json`.

## What is real and useful

- It contains a DGM implementation for multi-asset Black--Scholes PDEs, DGM/MLP
  models, validation, scaling, ablations, Greeks, an FBSDE wrapper, timing code,
  and a hybrid MC experiment.
- Its valid `price_with_delta_cv` path uses a left-endpoint stochastic integral;
  this is the right mathematical direction for a robust DGM-MC estimator.
- The DGM direct surface idea and the hybrid fallback idea are complementary:
  direct inference can be fast after training, while MC remains a correctness
  anchor.

## Blocking issues in the remote claims

1. `evaluation/monte_carlo.py:163–234` exposes a terminal prediction CV with a
   purported mean equal to the network's initial price. That mean is not known
   for an approximate network, and the saved 1D price is about 8.95% below the
   analytical value.
2. `experiments/run_hybrid_mc.py:130–160` caps the actual CV paths at 50,000 but
   records the requested `n_mc`, normally 100,000.
3. `experiments/run_comparison.py:174–187` compares a 50k-path, 30-step hybrid
   estimate to a 1M-path reference. It reports hybrid sampling error, not a
   matched total-cost RMSE comparison.
4. `baselines/fbsde_pricer.py` trains only at ATM and returns `NaN` off ATM. The
   five-state comparison therefore cannot support a general DGM-versus-FBSDE
   claim.
5. `experiments/run_time_analysis.py:47–59` silently uses an untrained model if
   no checkpoint exists. It times 10,000 neural states but extrapolates MC from
   only 100 states and does not charge training or compare at a common error.
6. The README/paper call the hybrid result “>99.999%” and “>1000x” while the
   saved legacy values, estimator variants and aggregate rows are inconsistent.
7. The original hard terminal wrapper, artificial boundary loss, stochastic
   Hutchinson squared residual, adaptive fallback, and antithetic SE issues are
   documented in `docs/AUDIT.md`.

## Correct interpretation

The remote repo is a promising prototype of a DGM-MC **direction**, not evidence
that DGM-MC is already better than MC or FBSDE. The new `dgm_mc` package keeps the
useful direction, removes the invalid terminal control, adds the cost-aware
selection contract, and states the exact evidence needed for a stronger claim.
