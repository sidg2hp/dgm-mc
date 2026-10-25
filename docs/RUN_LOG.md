# Execution log

- Numerical regression suite: 35 tests passed before launching experiments.
- `configs/smoke.json`: complete end-to-end smoke run (22 estimates, one model).
  Excluded from scientific conclusions.
- `configs/cpu_study.json` -> `results/cpu_study`: interrupted by the shell tool's
  600-second timeout after completing the 1D block and part of the 5D block.
  No `completion.json` exists. Original partial artifacts remain locally,
  are ignored by git, and are **not** inputs to the paper or result compilation.
  No Python process remained after interruption (checked with Windows tasklist).
- Resource adjustment, made before the replacement run: retain 18 fresh models,
  six classical dimensions and 40 coverage replications; reduce training from
  600 to 300 steps, evaluation from 12 to 8 replications, paths from 2048 to 1024,
  and control grid from 16 to 8. Independent reference precision is unchanged.
  This is a compute-budget screening study, not a convergence study.
- `configs/cpu_verified.json` is the new, separately recorded protocol. Its
  status and actual runtime are determined by `results/cpu_verified/completion.json`,
  not by this log or a plan. No partially trained legacy model is reused.
- Replacement run completed: 753.07 seconds, 18 models, 1,872 pricing estimates,
  54 direct-price records, 840 oracle-calibration records.
- Separate learned diagnostics completed: six frozen 1D models, 80 repetitions
  each, fixed and pilot coefficients (960 estimates); 101 spots x three times x
  six models (1,818 raw price/Greek records). No new fitting or model selection.
- Expanded regression suite: 39 tests passed, including safe checkpoint loading
  and rejection of hash-inconsistent or incomplete artifacts.
- New 5D pricing CLI exercised successfully with heterogeneous spots and an
  independent pilot. This is an execution check, not a separate headline result.
- LaTeX first encountered missing `natbib` and a malformed host PATH entry.
  The manuscript now uses standard numeric BibTeX citations; the portable build
  command sanitizes only its child-process PATH and disables package installation.
  No global configuration was changed. The PDF builds with no undefined
  references or overfull boxes; the nine-page output was visually inspected.
