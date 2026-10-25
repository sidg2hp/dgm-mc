# DGM-MC positioning and corrected claim set

## What the project is trying to do

The project is best understood as a **hybrid deployment method**, not as a
claim that a neural PDE solver universally replaces Monte Carlo or Deep BSDE:

1. **DGM surface:** after offline training, one network evaluates a batch of
   prices and autodiff Greeks without launching new paths for every query.
2. **DGM-MC estimator:** the same model contributes a predictable left-endpoint
   stochastic integral. Its expectation is zero on fresh paths, so it can reduce
   Monte Carlo variance without inheriting the network's price bias.
3. **Cost-aware fallback:** an independent pilot compares plain MC, cheap
   analytic controls, neural controls and combined controls. The system uses the
   neural branch only when its predicted amortized cost to reach a requested
   error is lower; otherwise it falls back to the classical control.

This is a meaningful systems/protocol contribution if demonstrated with matched
accuracy, matched compute, multiple payoffs/models, and direct implementations of
the strongest prior methods. It is not a new martingale theorem.

## Claims that are allowed now

- “DGM-MC is a proposed cost-aware hybrid protocol combining a DGM surrogate,
  a predictable martingale control, and an independent-pilot fallback rule.”
- “The estimator remains conditionally unbiased under the stated integrability
  and fresh-path conditions; approximate gradients affect variance, not the
  control expectation.”
- “In the completed CPU screening study, neural controls reduced estimated
  standard errors by approximately 4.5–5.4x versus plain MC at central states.”
- “The same study found cheap geometric controls were substantially faster and
  therefore did not establish neural practical superiority.”
- “DGM gives a batched surrogate, but the current short-run Greek and direct
  price errors are too large for a production claim.”

## Claims that must not appear without new evidence

- “Novel neural control variates” or “first DGM-MC method.” Neural martingale
  control variates and unbiased deep PDE solvers predate this project.
- “State of the art,” “industry standard,” “guaranteed speedup,” or “better than
  FBSDE” without matched target error, total cost, training amortization,
  time-grid refinement and a full FBSDE implementation.
- “>99.999% variance reduction” from the inherited terminal-value CV. That result
  is biased and its reported path count was capped below the requested budget.
- “100-dimensional neural pricing” based on rows whose source run is absent or
  whose price can be negative in deep OTM states.
- “Inference time” based on one neural forward pass versus an MC subset
  extrapolation. Include training, pilot, paths, hardware, batching and target
  error.

## The strongest publishable framing

> We introduce and audit DGM-MC, a cost-aware hybrid protocol for high-dimensional
> European option pricing. DGM is used in two distinct roles: as an amortized
> batched surrogate and as a predictable martingale control for a fresh Monte
> Carlo estimator. An independent pilot selects the cheapest control combination
> for a requested error, with classical controls as a mandatory fallback. We prove
> conditional unbiasedness of the estimator and evaluate the complete training,
> pilot and inference cost rather than reporting variance ratios alone.

The last sentence is a proposed contribution. It becomes an empirical claim only
after the selection rule is benchmarked against matched FBSDE, MC, RQMC and
classical-control baselines.

## Required experiment for “better than current technologies”

For each model/payoff/dimension, define a target RMSE and report total wall time
to reach it, with uncertainty over at least 10 independent seeds:

- plain MC, antithetic MC, RQMC and analytic controls;
- the published/prior-art neural martingale control method;
- a correctly implemented Deep BSDE baseline with time-step refinement;
- direct DGM surface inference;
- DGM-MC with and without the cost-aware selector.

Charge offline training once per deployment query count, then report break-even
query counts. Keep separate price, Delta, Gamma, and hedging-error targets.
Reference arithmetic-basket prices must use independent high-precision RQMC or
validated multi-level MC. The current repository does not yet meet this gate.
