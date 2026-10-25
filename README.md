# DGM–MC: an audited research reconstruction

This is an isolated, self-contained reconstruction of the sibling
`dgm_option_pricing/` and `neurips_submission/` research projects. Neither
original folder is modified. It uses Python/PyTorch, but deliberately does not
import legacy modules or reuse their checkpoints or numerical claims.

**Status:** research artifact, not an industry standard and not a demonstrated
state-of-the-art method. The legacy terminal-value control variate is biased;
its tiny reported standard error is not evidence of accurate pricing.
See `docs/AUDIT.md` before using any old results.

The corrected method subtracts a predictable, left-endpoint stochastic integral
from an exactly simulated discounted payoff. A frozen approximate neural model
can change variance, but does not change the estimator's expectation. Coefficients
are fitted on an independent pilot, never on final evaluation paths. Geometric
basket controls and scrambled Sobol provide strong non-neural comparators.

## Reproduce

Run commands **inside this directory**, not either legacy project:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m dgm_mc.benchmark --config configs/smoke.json --output results/smoke
python -m dgm_mc.benchmark --config configs/cpu_study.json --output results/cpu_study
python -m dgm_mc.report --input results/cpu_study
```

Run directories are immutable: a second run at an existing output path fails
rather than silently reusing stale results. Use a new destination for reruns.
Training and evaluation are CPU/float64 by design in this reference version.
The actual run manifest records software, hardware class, source hashes, command,
seeds, and configuration. No paid compute or external service is required.

## Layout

- `dgm_mc/`: validated market model, smooth DGM/MLP, exact PDE residual,
  Monte Carlo estimators, benchmark and reporting commands.
- `tests/`: mathematical and statistical regression checks.
- `configs/`: small smoke run and declared CPU study.
- `results/`: newly measured records, checkpoints, reference uncertainty,
  generated CSV/figures and summaries. Legacy evidence stays separate.
- `audit/`: hashes and machine-readable checks of inherited artifacts.
- `docs/`: audit, statistical protocol, literature and submission limitations.
- `paper/`: standalone LaTeX manuscript and generated result tables.

No license is assigned to the inherited projects. This reconstruction cites
their provenance and the underlying published methods; licensing and final
authorship should be confirmed by the research owner before broader reuse.
