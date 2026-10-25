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

## Delivered research package

| Artifact | Location |
|---|---|
| Compiled 9-page manuscript | `paper/main.pdf` |
| LaTeX and bibliography | `paper/main.tex`, `paper/references.bib` |
| Main results and readable comparison | `results/cpu_verified/reports/RESULTS.md` |
| All scenario/method summaries | `results/cpu_verified/reports/summary.csv` |
| Learned-control coverage and Greeks | `results/diagnostics/` |
| Original-work audit | `docs/AUDIT.md`, `audit/legacy_checks.json` |
| Protocol and publication gaps | `docs/PROTOCOL.md`, `docs/SUBMISSION_READINESS.md` |

**Actually executed:** 18 fresh models (DGM/MLP, d=1/5/10, three seeds), 1,872
pricing estimates, 840 oracle-calibration estimates, 960 learned-control coverage
estimates and 1,818 price/Greek grid records. Classical baselines extend to d=100;
that is **not** a neural d=100 result. The main run took about 753 seconds on this
CPU environment. Only 300 training steps were used: this is a screening study,
not evidence of converged or production-ready models.

**Scientific finding:** central-scenario DGM neural controls reduce estimated SE
by roughly 4.5–5.4x versus plain MC, not the inherited >1000x claim. The cheap
geometric control is substantially faster than the neural variants. Near-expiry
Greek errors and some negative surrogate outputs remain. All are disclosed.

## Reproduce

Run commands **inside this directory**, not either legacy project:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m dgm_mc.benchmark --config configs/smoke.json --output results/smoke
python -m dgm_mc.benchmark --config configs/cpu_verified.json --output results/reproduction
python -m dgm_mc.report --input results/reproduction
```

Run directories are immutable: a second run at an existing output path fails
rather than silently reusing stale results. Use a new destination for reruns.
Training and evaluation are CPU/float64 by design in this reference version.
The actual run manifest records software, hardware class, source hashes, command,
seeds, and configuration. No paid compute or external service is required.

Regenerate the delivered paper from committed measurements (no training needed):

```bash
python -m dgm_mc.report --input results/cpu_verified --paper-dir paper/generated
python -m dgm_mc.diagnostic_report --input results/diagnostics --output paper/generated
python -m dgm_mc.build_paper
python -m dgm_mc.verify
```

The PDF build needs `pdflatex` and `bibtex`. It uses standard packages, disables
shell escape and MiKTeX automatic package installation, and does not change
global settings. Final author/affiliation and conference-format review remain.

Reproduce the independent 1D extension:

```bash
python -m dgm_mc.diagnostics --models results/reproduction/models --output results/reproduction_diagnostics --repeats 80 --paths 1024
```

Try a different spot vector under a checkpoint's recorded market:

```bash
python -m dgm_mc.price --model results/cpu_verified/models/dgm_d5_seed11.json --spots 0.9 0.95 1.0 1.05 1.1 --paths 4096 --steps 16 --seed 901
```

Checkpoint loading uses PyTorch's restricted weights-only mode, validates hashes
and matches market/architecture metadata. It never silently substitutes an
untrained model. The returned interval is an asymptotic research diagnostic.

For this machine, single-thread BLAS settings were used for the measured study:
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` (set before
Python starts). The manifest records Python 3.10.8, NumPy 2.2.6, SciPy 1.15.3
and PyTorch 2.12.0+cpu. Different numerical libraries/hardware may change timings
and random realizations; scientific replication is not a promise of identical
floating-point bytes. Git preserves the committed evidence's bytes for hash checks.

The optional legacy integrity check requires the two original sibling folders:
`python -m dgm_mc.audit --legacy-parent .. --verify`. It verified 299 eligible
original research files unchanged. No other command requires those folders.

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
