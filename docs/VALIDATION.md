# Validation record

All commands below were executed in the isolated new project on 2026-10-26
(Asia/Kolkata); run manifests store UTC timestamps.

| Check | Actual result |
|---|---|
| `python -B -m pytest -q` | 39 passed; final full run approximately 24 seconds |
| Smoke benchmark (`configs/smoke.json`) | Complete: one model and 22 estimate records |
| Verified CPU benchmark (`configs/cpu_verified.json`) | Complete: 18 models, 1,872 pricing estimates, 840 oracle-calibration estimates |
| Independent 1D diagnostics, 80 repetitions | Complete: 960 learned-control estimates and 1,818 price/Greek records |
| `python -B -m dgm_mc.report --input results/cpu_verified --paper-dir paper/generated` | 126 scenario/method summaries generated from hash-verified measurements |
| `python -B -m dgm_mc.diagnostic_report --input results/diagnostics --output paper/generated` | Greek and coverage tables/figure generated |
| `python -B -m dgm_mc.build_paper` | PDF compiled successfully; no undefined references or overfull boxes |
| PDF inspection | Nine-page manuscript inspected, including tables and all three figures |
| `python -B -m dgm_mc.verify` | Main records, all checkpoints, diagnostics, reports and paper hashes verified |
| `python -B -m dgm_mc.audit --legacy-parent .. --verify` | 299 original research source/result files unchanged |
| Pricing CLI with five heterogeneous spots | Executed successfully with restricted checkpoint loading and independent pilot |

The earlier larger run timed out and is excluded, as documented in `RUN_LOG.md`.
The failed initial LaTeX build was fixed without installing packages or changing
global configuration. SHA-covered outputs retain their original byte-level line
endings in git; ordinary source files use LF for portability.

These checks establish artifact consistency and targeted numerical correctness,
not full research validity, convergence, SOTA, or production suitability. There
is no separately configured lint/typecheck pipeline. No GPU run, independent
machine replication, paid compute, market-data calibration or conference
submission was performed.
