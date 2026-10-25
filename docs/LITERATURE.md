# Literature and novelty assessment

Primary-source search checked on 2026-10-26. This is a targeted prior-art check,
not an exhaustive systematic review or a certification of the latest SOTA.
No source is evidence that our implementation achieves its published results.

| Work | Verified identifier | Relevance and consequence |
|---|---|---|
| Sirignano and Spiliopoulos, *DGM: A deep learning algorithm for solving partial differential equations* (2018) | arXiv:1708.07469v5; DOI 10.1016/j.jcp.2018.08.029 | Source of the mesh-free PDE-residual architecture. DGM itself is not new. |
| Sabate Vidales, Siska and Szpruch, *Unbiased deep solvers for linear parametric PDEs* (v4, 2022; first preprint 2018) | arXiv:1810.05094v4; DOI 10.1080/1350486X.2022.2030773 | Sections 2–3 explicitly combine approximate neural gradients and martingale control variates; experiments include 100D. The central proposed reframing already exists. |
| Hinds and Tretyakov, *Neural variance reduction for stochastic differential equations* (2023 journal reference) | arXiv:2209.12885v2; DOI 10.21314/JCF.2023.010 | Brownian and Levy-driven neural controls, direct variance minimization, two-pass training/evaluation and cost comparisons. Essential future implementation comparator. |
| Han, Jentzen and E, *Solving high-dimensional partial differential equations using deep learning* (2018) | arXiv:1707.02568v3; DOI 10.1073/pnas.1718942115 | Deep BSDE baseline; cited but not falsely claimed as reproduced in the new study. |
| Guo, Langrene and Wu, *Simultaneous upper and lower bounds of American-style option prices with hedging via neural networks* (2025 journal reference) | arXiv:2302.12439; DOI 10.1080/14697688.2025.2472874 | Adjacent stopping/hedging literature. Extending to American options is not automatically novel; not evaluated here. |

The first four abstract records were checked; the central mathematical and
algorithmic sections of the Vidales and Hinds papers were also inspected in
arXiv HTML. HTML conversion displays a rendering date that should not be
mistaken for a publication date; the bibliography uses version metadata.

The legacy Zhou reference is not verified: its DOI resolves to Jeon and Kwak's
*Pricing variable annuity with surrender guarantee*, JCAM 393, 113508 (2021).
The new manuscript omits this unsupported attribution.

## Defensible contribution now

An auditable reconstruction, explicit failure counterexample, independently
piloted composite controls, correct sampling-unit uncertainty, transparent
compute accounting, and a reproducible negative/positive screening benchmark.
Neither martingale unbiasedness nor linear-control orthogonalization is claimed
as a new theorem. Combining established components is not by itself A/A* novelty.

## Plausible next research contribution (not a completed result)

Develop a cost-aware residual-control learner that targets only variance left
after strong classical controls, with independent selection and an excess-risk
bound accounting for training, pilot fitting, and grid resolution. Compare
direct integrand learning to PDE-gradient learning and analytic controls across
multiple model/payoff families. The present composite-control experiment is a
testable starting point, not proof of this broader method's superiority.
