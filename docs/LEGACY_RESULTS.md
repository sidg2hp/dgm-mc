# Legacy high-dimensional results: transparent inclusion

The public `option-pricing-DGM` repository reports DGM results at d=7, 10, 25,
50 and 100. They were not included in the verified main tables because the audit
could not connect every aggregate value to an independently reproducible run and
because the same dimensions use incompatible metrics and experiment streams.

They are now included in `audit/legacy_high_dim_reported.json` and the paper's
legacy-results table. The values are transcribed from the remote repository at
commit `a2ef13292e9975f3c1c60c1f35f737260903dbc7`; no approximate or fabricated
values were inserted.

| d | Reported scaling relative L2 | Reported comparison MRE | Reported hybrid SE reduction | Reported hybrid variance reduction |
|---:|---:|---:|---:|---:|
| 7 | 13.14% | 6.74% | 710x | 99.9998% |
| 10 | 28.47% | 5.65% | 499x | 99.9996% |
| 25 | 3.55% | 3.55% | 4.9x | 95.9% |
| 50 | 70.76% | 5.56% | 4.8x | 95.7% |
| 100 | 60.69% | 60.69% | 5.1x | 96.2% |

## Why these are not main claims

- The d=10 scaling error (28.47%) and comparison MRE (5.65%) are different
  metrics/evaluation streams, not interchangeable estimates.
- The remote hybrid runner caps actual CV paths at 50,000 while recording a
  requested 100,000 budget.
- The remote paper's FBSDE endpoint is ATM-only and returns no valid values at
  the other comparison states.
- Some high-dimensional aggregate rows were generated or patched outside a
  complete per-run provenance chain; checkpoint/configuration hashes and raw
  predictions are not sufficient for independent reproduction.
- The legacy terminal-value CV is biased. The valid delta-integral direction is
  retained in the new DGM-MC implementation, but these old hybrid numbers cannot
  be interpreted as evidence for it.

The table is useful for preserving the research record and identifying the next
experiment. It is not a claim that the reported values are false, exact, or
reproduced. It is a source-attributed report requiring rerun before submission.
