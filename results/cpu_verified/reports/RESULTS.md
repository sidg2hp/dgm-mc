# Measured results

Generated from a completed, hash-verified run. No legacy performance is included.

- Fresh models: 18; estimate records: 1872.
- Analytic calibration records: 840.
- Total recorded wall time: 753.1 seconds.
- Training: 300 steps; seeds [11, 29, 47].
- Evaluation paths: 1024; pilot paths: 1024; time steps: 8.

## Central moneyness scenario

RMSE is against the recorded independent reference, not necessarily exact truth.
Warm cost includes pilot; cold cost adds full training. No equal-compute claim.

| d | Method | Architecture | RMSE | RMS SE | Warm ms | Cold s |
|---|---|---|---:|---:|---:|---:|
| 1 | Antithetic | classical | 0.00142 | 0.00318 | 4.62 | 0.00 |
| 1 | Geometric CV | classical | 1.2e-17 | 6.42e-18 | 4.79 | 0.00 |
| 1 | MC | classical | 0.00403 | 0.00461 | 0.70 | 0.00 |
| 1 | RQMC | classical | 0.000397 | 0.000435 | 30.80 | 0.03 |
| 1 | Combined CV | dgm | 1.55e-17 | 8.07e-18 | 383.89 | 19.69 |
| 1 | Neural fixed | dgm | 0.000662 | 0.000956 | 183.46 | 19.49 |
| 1 | Neural pilot | dgm | 0.000672 | 0.000948 | 383.89 | 19.69 |
| 1 | Combined CV | mlp | 1.58e-17 | 8.39e-18 | 88.84 | 5.52 |
| 1 | Neural fixed | mlp | 0.000875 | 0.00112 | 43.29 | 5.47 |
| 1 | Neural pilot | mlp | 0.00077 | 0.00104 | 88.84 | 5.52 |
| 5 | Antithetic | classical | 0.00232 | 0.00189 | 2.43 | 0.00 |
| 5 | Geometric CV | classical | 0.000333 | 0.000324 | 3.75 | 0.00 |
| 5 | MC | classical | 0.00283 | 0.00318 | 1.43 | 0.00 |
| 5 | RQMC | classical | 0.000183 | 0.000177 | 19.81 | 0.02 |
| 5 | Combined CV | dgm | 0.000217 | 0.00027 | 377.62 | 46.95 |
| 5 | Neural fixed | dgm | 0.000506 | 0.000601 | 189.79 | 46.76 |
| 5 | Neural pilot | dgm | 0.000511 | 0.000594 | 377.62 | 46.95 |
| 5 | Combined CV | mlp | 0.000279 | 0.0003 | 84.92 | 10.42 |
| 5 | Neural fixed | mlp | 0.000783 | 0.000811 | 41.58 | 10.38 |
| 5 | Neural pilot | mlp | 0.000669 | 0.000717 | 84.92 | 10.42 |
| 10 | Antithetic | classical | 0.000488 | 0.0017 | 2.00 | 0.00 |
| 10 | Geometric CV | classical | 0.000119 | 0.000259 | 4.49 | 0.00 |
| 10 | MC | classical | 0.00276 | 0.00291 | 2.02 | 0.00 |
| 10 | RQMC | classical | 0.000167 | 0.000208 | 37.55 | 0.04 |
| 10 | Combined CV | dgm | 0.00025 | 0.000231 | 325.54 | 80.54 |
| 10 | Neural fixed | dgm | 0.000524 | 0.000654 | 173.24 | 80.39 |
| 10 | Neural pilot | dgm | 0.000522 | 0.000647 | 325.54 | 80.54 |
| 10 | Combined CV | mlp | 0.00024 | 0.000234 | 117.16 | 21.31 |
| 10 | Neural fixed | mlp | 0.000843 | 0.000953 | 60.77 | 21.26 |
| 10 | Neural pilot | mlp | 0.000741 | 0.000802 | 117.16 | 21.31 |
| 25 | Antithetic | classical | 0.00126 | 0.00156 | 3.59 | 0.00 |
| 25 | Geometric CV | classical | 0.000231 | 0.000219 | 7.96 | 0.01 |
| 25 | MC | classical | 0.00241 | 0.00274 | 3.97 | 0.00 |
| 25 | RQMC | classical | 0.000176 | 0.000192 | 57.80 | 0.06 |
| 50 | Antithetic | classical | 0.000847 | 0.00154 | 5.29 | 0.01 |
| 50 | Geometric CV | classical | 8.83e-05 | 0.000202 | 16.22 | 0.02 |
| 50 | MC | classical | 0.00267 | 0.00273 | 6.27 | 0.01 |
| 50 | RQMC | classical | 0.000219 | 0.000169 | 97.05 | 0.10 |
| 100 | Antithetic | classical | 0.00144 | 0.00151 | 14.79 | 0.01 |
| 100 | Geometric CV | classical | 0.000209 | 0.000192 | 27.47 | 0.03 |
| 100 | MC | classical | 0.00384 | 0.00269 | 12.84 | 0.01 |
| 100 | RQMC | classical | 0.000163 | 0.000139 | 165.48 | 0.17 |

## Interpretation limits

- Three states per dimension are not a full-dimensional surface validation.
- Classical d=25/50/100 results are not learned-model scaling evidence.
- The geometric control is exact for a one-asset basket; roundoff-level errors are expected.
- Forty calibration replications give wide coverage intervals; no coverage guarantee is inferred.
- Repeated evaluation paths are not additional training seeds. Per-seed dispersion is in summary.csv.
- Negative findings are retained. Statistical validity is distinct from efficiency and novelty.
