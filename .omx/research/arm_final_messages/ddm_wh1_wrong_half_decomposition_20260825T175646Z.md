Executed and landed the WH1 charter.

Key n600 findings:

- WRONG-indicator: 76,470.790 B; WHICH-class: 2,500.167 B.
- Lane: 38.646% of WRONG bits from 0.5856% of positions, a 65.998× enrichment.
- Margins below 4 bits: 59.867% of WRONG bits from 0.7278% of positions.
- 86.044% of WRONG bits are `STATIC_IN_IMAGE`; 99.236% lie on decoded-token boundaries.
- The decoder-known class × margin × 16×16-tile formulation has a −980.640 B aggregate gap: the existing adaptive coder already beats that partition. This is confirmatory, not a new GB1 lever.
- The prior-law prediction passed; the near-uniform falsifier did not fire.

Artifacts:

- [Verdict memo](/Users/adpena/Projects/pact/.omx/research/ddm_wh1_wrong_half_decomposition_20260825.md)
- [Reusable Stage-B instrument](/Users/adpena/Projects/pact/tools/token_wrong_half_ledger.py)
- [RESULT.json](/Volumes/APDataStore/pact/ddm_wh1_wrong_half_decomposition/measurement_v1/RESULT.json)
- [Custody manifest](/Volumes/APDataStore/pact/ddm_wh1_wrong_half_decomposition/measurement_v1/CLEANUP_MANIFEST.json)

The 2.9 GiB retained ledger passed full hash verification; 30/30 checkpoints, 117,964,800/117,964,800 tokens matched. Four focused tests passed. Landed commits: `f5b4b007be`, `45cba91422`, `5d86a411ba`, and memo `edffc69b77`. Unrelated staged launcher moves were preserved.

No scorer, candidate, archive mutation, Modal dispatch, or score claim occurred. Frontier remains **gb1 — S 0.14811799921260607 @ 180,215 B `[contest-CUDA T4, n600]`**.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN-designated Stage-B producer; consumer store: `/Volumes/APDataStore/pact/ddm_s1_trained_renderer_diagonal/stage_b/`; fire trigger: MAIN selects a Stage-A seed/window whose fingerprint contract binds the moved runtime, archive, n600 receiver-consumed token field, and exact hashes, then run WH1 smoke → full replay → analyze → verify with explicit custody flags.

## LIVE-HYPOTHESES

- The moved Stage-B object may expose a positive decoder-known gap because renderer movement changes the prediction and calibration distribution; the present negative applies only to GB1.
- A genuinely causal recurrence feature may exploit the 86.044% image-static concentration. This is plausible from the measured recurrence, but the current G4 label uses realized errors and cannot itself be treated as free conditioning.

## DEAD-ENDS

- GB1 class × margin × fixed-tile conditioning: closed at formulation scope because its aggregate bound is 980.640 B worse than the shipped adaptive coder.
- Position sidecars: remain closed because naming concentrated cells incurs the previously measured address tax.
- G4 xi-proxy exploitation: only 93 wrong positions and 0.0417% of WRONG bits; additionally non-physical and not decoder-free.
- Calibration probing and aggregate DX2/GB1 attribution reruns: already settled and should not be repeated without a moved object.