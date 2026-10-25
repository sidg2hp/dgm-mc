# Statistical and computational protocol

## Scope and success criteria

The implemented target is an equal-weight arithmetic basket European call under
constant-coefficient multivariate Black–Scholes. General covariance, heterogeneous
volatilities, nonpositive rates, zero volatility, and singular positive-semidefinite
correlation are supported by the numerical core. The measured main study keeps
volatility/rate/maturity fixed. This is not a calibration to market data.

Success means mathematical validity, correct sampling units, repeatable runs,
comparable targets, complete cost reporting, and explicit failure disclosure.
It does not mean that a neural method must win.

## Model and training

- Log-price x=log(S/K); exact deterministic Hessian contraction at all dimensions.
- Smooth tanh DGM and MLP; exact arithmetic-payoff terminal MSE, weight 10.
- No embedded kink, train/eval switch, artificial basket boundary, random trace,
  unsupported adaptive sampling, checkpoint fallback, or implicit resume.
- CPU float64; seed before initialization; separate collocation generator.
- Fresh 300-step models with width 32, depth 2, batch 64 in the completed-study
  protocol `cpu_verified.json`. Gaussian collocation std=0.3; uniform time.
  Fixed Adam LR=0.001. The larger 600-step protocol was interrupted and excluded.
- Three initialization/training seeds (11,29,47); no test-based model selection.
- Equal width is **not** equal parameter count or equal compute. Parameter counts
  and training times are recorded, and architecture comparisons are descriptive.

## Estimation

For x_{k+1}=x_k+mu*dt+B*dW_k with BB'=A, the neural control is
sum exp(-r*t_k) grad_x u(t_k,x_k)' B*dW_k. Every gradient is evaluated before
its fresh Brownian increment. Exact GBM simulation removes discretization bias
in the payoff; a coarse control grid affects variance, not expectation.

An additional geometric terminal control has its exact analytical expectation
subtracted. Pilot OLS fits either the neural coefficient alone or both controls,
using 1024 independent pilot paths. An SVD handles collinearity. Evaluation uses
1024 new paths. The fixed coefficient-one variant requires no pilot.

The pilot is fresh for each repeated evaluation. Conditional on training and
pilot information the estimator is unbiased; model quality and pilot estimation
affect variance. Bad controls can increase variance. There is no claimed
finite-pilot monotonic improvement from adding a control.

## Comparators and evaluation states

- Plain terminal MC, antithetic MC, pilot-fitted geometric CV, and scrambled Sobol.
- Neural coefficient one, neural pilot OLS, and combined geometric/neural OLS.
- Baseline evaluation path count is equal, but work is not: terminal simulation
  needs one increment while neural controls use 8 and gradient evaluations.
- Report evaluation, pilot, and training costs separately. Do not infer speedup
  from standard-error ratios. Training amortization is an explicit deployment
  choice; the report shows both warm evaluation and cold-start costs.
- Three declared moneyness centers 0.85, 1, 1.15. For d>1, component spots are
  m*exp(linspace(-.12,.12,d)), off the equal-asset diagonal. These three vectors
  are a limited benchmark, not a full price-surface validation.
- Learned models: d=1,5,10. Classical-only extension: d=25,50,100, **not neural
  scaling evidence**. All methods share the same target per scenario.
- Eight MC evaluation repetitions per scenario/model; three training seeds.
  Repeated evaluations of one model are not independent training replications.
  Training-seed dispersion is reported separately; no pooled significance claim.

## References and uncertainty

- d=1 exact analytical value. Other arithmetic baskets use an independently
  seeded 16-scramble x 32768-point RQMC reference. Reference SE is retained.
- MC SE uses sample variance (ddof=1). Antithetic sampling units are pair means;
  Sobol units are independent scramble means, not Sobol points.
- Student-t intervals are asymptotic diagnostics for nonnormal sampling units,
  not guaranteed finite-sample coverage. Geometric analytical experiments
  provide actual coverage checks without a noisy reference.
- RMSE against an approximate reference is labelled as such: reference error
  is shared across evaluations and is not silently treated as zero.
- Forty independent repetitions for geometric-payoff oracle calibration across
  d=1,5,10, grids 4/16/64, and deliberately scaled gradients. Wilson intervals
  describe coverage uncertainty. These are analytic controls, not learned wins.
- Negative variance reduction, poor direct pricing and undercoverage are kept.

## Provenance

Each run refuses an existing destination and records configuration, source hashes,
software versions, seed namespaces, checkpoint hashes and completion hashes.
Only completed runs can generate reports. Raw records are immutable inputs to
generated tables. No old aggregate enters the new experiments.

## Remaining publication gates

See `SUBMISSION_READINESS.md`. A valid estimator does not imply a novel method,
SOTA accuracy, superior cost, good Greeks, or an accepted paper.
