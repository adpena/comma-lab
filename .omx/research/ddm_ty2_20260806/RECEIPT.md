# ddm_ty2 synergy + hybrid implementation sweep - 2026-08-06

Status: **SCORER-FREE / COMPOSITION-LEDGER / score_claim=false**.

Own-vehicle frontier line: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`. Contest pointer remains the borrowed `0.19108`; TY2 did not run `upstream/evaluate.py`, did not run a new SegNet/PoseNet population scorer, and did not move either pointer.

## Top S-stakes Rows

| rank | row | S-stakes | weakness coverage |
|---:|---|---|---|
| 1 | `TY2-H1-renderer-v14-base-trained-residual` | It is the renderer leg for TK1 Route S: PR130-class d_seg plus PR130 pose and a 3.3KB renderer projects to `S=0.157385863 @ 168,892 B`; sub-0.15 needs `11,092` fewer bytes at the same distortion or lower d_seg/d_pose. | Covers TK2 C1/C2 TOY paint by making them failure-map structure only; covers v14 by treating the remaining gap as RGB/scorer projection, not receiver plumbing; turns D1 scratch-train into residual-train. |
| 2 | `TY2-H2-tokens-semantic-base-annulus-latent-residual` | It is the rate leg: TK1's selected semantic stream is `142,001 B`, and every `1,501.81956 B` costs `0.001 S`; residual bytes must be annulus/edge-value justified. | Covers the small learned-prior TOY result by keeping KT exact coding as the base and using learned payload only in the m91 boundary annulus; folds #869 as a queued adaptive-granularity byte lever. |
| 3 | `TY2-H4-pose-warp-rank1-frame0-free-actuator` | It supplies the pose leg for the same Route S: replacing current pose with PR130-class pose changes the TK1 projection from `0.219958695` to `0.157385863`, a `0.062572832 S` swing. | Covers RV1/X6 post-hoc pose as NAIVE, respects P1's shared-low-rank negative, and supplies the missing pose-bound cure for ET3-style solve-within+CVP. |

## Ledger Counts

- Total rows: `10`
- Hybrid rows: `4`
- Synergy rows: `6`
- Rows fired by TY2: `0`
- Rows folded into existing evidence/plans: `3` (`TY2-S1`, `TY2-S4`, `TY2-S6`)
- Rows queued with explicit fire order: `7`

## Single Cheapest Fire-order

Fire `TY2-H3-coder-kt-backbone-small-learned-mixer` first. It is scorer-free and byte-only:

1. Materialize or locate the exact TK1 `142,001 B` semantic stream as the durable input.
2. Run KT baseline versus small counted learned context mixer on the same stream.
3. Require exact decode equality and count mixer/model/header bytes.
4. Fold immediately if `mixer + coded_stream >= 142,001 B` or any symbol mismatches.

This cannot produce a new score by itself, but it is the cheapest valid test because it consumes no scorer slot and any win translates directly through `25 / 37,545,489 = 6.658589531e-7 S/B`.

## Verified Required Examples

- **CVP x solve-within:** verified through SW1, DK1, and ET3. The composition exists and should not be duplicated by TY2. ET3 reached `eta=0.3562364032` and `eta/bar=2.083` on n32, but held because pose max was `1.1284`, above the bound.
- **KT context-arith x learned context-mixing:** routed as `TY2-H3`. TK1's KT stream is the exact `142,001 B` base; TY1's pure learned prior loses by `+539,728 B`, so the only admissible learned form is a counted residual probability/context mixer that beats KT after model bytes.
- **PE3 conditioning x latent renderer:** routed as `TY2-S2`. PE3 is byte-closed and parse-back proven (`74,408 B` section in the 75KB hybrid row), but runtime RGB/scorer survival is unmeasured, so it may condition H1's latent residual only after an adapter/runtime gate.

## RECALL EVIDENCE

- `.omx/tmp/codex_runs/ty2_prompt.md` and `.omx/tmp/codex_runs/_common_contract.md`: established the scorer-free charter, protected files, required fields, and final reporting shape. Plan impact: TY2 writes ledger/receipts only, with `score_claim=false`.
- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`, and `.omx/state/main_hot_state.md`: confirmed no-fake, upstream immutability, serializer, current pointer, and active scorer-lane constraints. Plan impact: no scorer/evaluator run, no protected edits, and current frontier updated to `S=0.7534578126155775 @ 357,837 B`.
- `.omx/research/ddm_ty1_20260806/TOY_LEDGER.jsonl`: provided 37 graded rows and weakness labels. Plan impact: each TY2 row maps its composition to a specific TY1 weakness instead of treating a killed formulation as a killed family.
- `.omx/research/ddm_tk1_20260806/RECEIPT.md`: provided semantic-stream bytes, Route S projections, boundary-pixel geometry, and residual crossover limits. Plan impact: H1/H2/H3/H4 S-stakes and byte ceilings are TK1-derived, not guessed.
- `.omx/research/ddm_tk2_20260806/RECEIPT.md`: provided the n4 flat/template/boundary paint smoke. Plan impact: TK2 paint is used only as failure-map evidence for H1, not as renderer proof.
- `.omx/research/ddm_sw1_20260806/RECEIPT.md`, `.omx/research/ddm_dk1_20260806/RECEIPT.md`, and `.omx/research/ddm_et3_20260806/RECEIPT.md`: verified solve-within plus CVP and the ET3 pose-bound hold. Plan impact: `TY2-S1` is folded, not re-fired.
- `.omx/research/ddm_pe3_20260805/PE3_RECEIPT_20260805.md` and `ddm_pe3_hybrid_receipt.json`: verified PE3 byte closure and parse-back without runtime/scorer survival. Plan impact: PE3 is a conditioning row, not a score row.
- `.omx/research/pr86_pr130_fullstack_intake_20260728.md` and `.omx/research/codex_findings_ddm_p1_frame0_pose_quotient_carrier_20260725T143303Z_codex.md`: provided PR130 external pose/renderer lessons and the P1 negative scope. Plan impact: H4 is pair-conditioned/joint/warp-base scoped and does not copy PR130 or resurrect P1.
- `.venv/bin/python tools/list_canonical_equations.py --json` filtered for rate/pose/boundary/context equations: confirmed the relevant canonical laws for rate price, boundary annulus waterfill, pose/frame_0 asymmetry, and scorer-free byte accounting. Plan impact: all queued rows are priced against counted bytes and scoped to scorer authority.

## Boundaries

- No score is claimed. All `S` values in TY2 are copied from receipts or recomputed from receipt-provided components as projections.
- No full n600 scorer job was launched. Active scorer ownership remains outside TY2.
- No `upstream/` file was edited.
- No protected file was edited.
- No `/tmp` evidence is cited or persisted.
- External PR130 facts are used as lessons and target shapes only; no external bytes, weights, basis values, or constants are imported.
- PE3 remains runtime-survival-unmeasured. It cannot be treated as an RGB/scorer row until a receiver consumption gate lands.

## Outcome

TY2 produced a typed 10-row ledger with four build-ready hybrid rows and six synergy rows. The highest-value path is H1+H2+H4 around TK1 Route S; the cheapest next concrete action is H3's byte-only KT versus learned context-mixer race.
