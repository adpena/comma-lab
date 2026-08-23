HR3 landed as an honest storage-bounded negative in commit `701bb78e08`.

Best receiver-closed row: **362,473 residual-equivalent bytes**, producing a **463,601 B container**—325,615 B over the 137,986 B cap and 3,193 B worse than HG1. Exact parse-back and repeat reproduce the source categorical SHA.

The residual is highly structured: 98.806% of its modeled cost lies in BL1’s top 1% positions, 86.09% is within eight pixels of a generated boundary, and 94.36% of lag-1 overlaps retain the same target class.

The prior prediction remains **unadjudicated**. APDataStore fell to 241 MiB free, forcing the charter-permitted bounding reduction:

- Widths 8/16: eight complete real-coder rows.
- Widths 32/64/96: counted model-only bounds; residuals unmeasured.
- Width 128: model alone exceeds the residual budget.

No scorer, Modal, Metal, CUDA, MPS, or `upstream/` mutation occurred.

Artifacts:

- [Evidence memo](/Users/adpena/Projects/pact/.omx/research/ddm_hr3_residual_implicit_carrier_20260823.md)
- [Reviewed runner](/Users/adpena/Projects/pact/experiments/ddm_hr3_residual_implicit_carrier.py)
- [RESULT.json](/Volumes/APDataStore/pact/ddm_hr3_residual_implicit_carrier/RESULT.json)
- [Retention manifest](/Volumes/APDataStore/pact/ddm_hr3_residual_implicit_carrier/manifest.json)

Own-vehicle frontier unchanged: **DX2 — S `0.14821987563243377` @ `180,368 B` `[contest-CUDA T4, n600]`**.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — disposition: full-field price widths 32/64/96 and receiver-close the winner; owner: MAIN-assigned HR3 successor; consumer store: `/Volumes/APDataStore/pact/ddm_hr3_residual_implicit_carrier/manifest.json`; fire trigger: APDataStore has at least 6 GiB free with no competing bulk writer, followed by a reviewed scope-expansion migration. Fire no scorer unless residual-equivalent bytes reach ≤36,858.

## LIVE-HYPOTHESES

- Widths 32/64/96 remain testable because their counted models fit the budget and training loss continued falling, though width 96 leaves only about 10.2 KB for residual and framing.
- A sparse class/edge/temporal hybrid remains plausible because the residual is strongly concentrated, boundary-heavy, and temporally class-stable.

## DEAD-ENDS

- Width-8/16 Fourier-coordinate, per-pair-FiLM carriers: all eight complete rows lose to HG1.
- Width 128 under this serialization: its smallest model is 39,481 B before residual or framing.
- Nominating one coder: Brotli wins the model while LZMA2 wins the residual.
- Large float32 NumPy `matmul` on this environment: it produced non-finite arithmetic and was replaced by deterministic finite `einsum`.
- Claiming complete container-family closure from this run: widths 32/64/96 were not full-field measured.