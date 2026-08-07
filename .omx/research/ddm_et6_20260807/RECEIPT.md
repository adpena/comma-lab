# ddm_et6 receipt

## Three-family table

| family | strict receiver-covered flips (proj n600) | priced support flips (proj n600) | support frac in n32 ET4 sample | projected bytes | best coder | B/support flip | xW | support net dS | verdict |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| token_cell_edits_support_priced | 0.0 | 84056.2 | 100.00% | 119137.5 | brotli | 1.417354 | 1.113 | 0.008073 | FOLDED: priced support only; no inverse-token receiver proof |
| tq1_snap_menu_idx_only | 0.0 | 84056.2 | 100.00% | 0.0 | strict-empty; geom payload brotli | n/a | n/a | 0.000000 | FOLDED: strict reproduction coverage is zero |
| road_lane_grammar_lane_crop | 0.0 | 20212.5 | 24.05% | 142931.2 | brotli | 7.071429 | 5.554 | 0.078038 | FOLDED: support price above W and receiver grammar absent |

Axis: `[macOS-CPU advisory]`; sample: stratified-random n=32, seed 20260807, not prefix.
The strict receiver-covered column is zero where no token/grammar receiver proof exists. Support pricing is retained as a description-price measurement only.

## Denominators

- ET4 net flips denominator: `78302`.
- Break-even W from ET4 adjudication: `1.27310821533` B/net-flip.
- `S_per_flip = 8.477105034722222e-07`; `rate_per_byte = 6.658589531221714e-07`.

## Sample Pairs

`14, 23, 47, 60, 77, 97, 120, 143, 150, 178, 197, 214, 232, 260, 266, 291, 316, 320, 341, 367, 375, 402, 422, 432, 455, 474, 498, 518, 538, 549, 570, 588`

## Verdict

- Falsifier triggered: `True`.
- Summary: All ET6 priced support families are above W or have zero strict reproduction, so ET4 is unshippable at these description granularities.
- No SegNet/PoseNet scorer run was launched. No archive was built. No pointer moved.
- Token-cell pricing is a block16 support re-description of ET4 correction bands; it is not a receiver-survived token edit.
- TQ1 menu strict reproduction is zero because the available menu is global all-pair snap moves, while ET4 patches are pair-local image-domain CVP deltas.
- Road/Lane grammar pricing covers only the Road<->Lane off-diagonal reduction proxy; this measured band crop is still above W and is folded.

## Custody

- ET4 rows: `/Volumes/VertigoDataTier/pact/ddm_et4_20260806/et4_solve_within_cvp_rows.jsonl` sha256 `183695482076af8da33aa7fd48f8850696b19f38fc92dc73830ad3952e85cec0`.
- ET4 patch records checked: `600`; all row hashes matched: `True`.
- Parent archive: `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/candidate_archives/move_0023_snap_r00_c12_L13.zip.receipt-bytes` sha256 `b35e756829306a85ec2ad51634bde74523d89df9046c682253176b393bd59c06`; bytes `357837`.
- Parent archive IX2 token decode succeeded: `True`.
- Pre-move token file equals parent archive tokens: `False`.
- GT `lstars` opened through ZIP_STORED memmap: `True`.

## RECALL EVIDENCE

- Read .omx/tmp/codex_runs/et6_prompt.md and _common_contract.md; obeyed scorer-free ET6 scope and receipt targets.
- Read PROGRAM.md, CLAUDE.md, AGENTS.md, docs/operating_manual_craft_handoff.md, and .omx/state/main_hot_state.md for governing constraints and live pointer.
- Read ddm_et5 RECEIPT, PRICING_TABLE, and CAMPAIGN_984_ROUTE: ET5 image/i16 patch stream folded at 84.476 B/full-flip.
- Read ddm_et4 TWELFTH_MOVE_ADJUDICATION and RECEIPT: ET4 solved corrections are real image-domain CVP patches, and the receipt warns not to pretend those are token edits.
- Read ddm_et2 phase-field summary and ET4/SQ1 scripts: ET4 correction band rederived as target != lstar with exact 2x2 scorer snap.
- Read ddm_rl1, ddm_se3, ddm_pe1, and per-edge optimality directive: Road<->Lane crop/per-edge grammar is the relevant edge-local route.
- Read ddm_rh1 and ddm_sv2: live token stream must be priced with the real IX2TOK01 coder; remaining headroom is content, not base-rule recoding.
- Searched MEMORY.md for current frontier and exact/advisory separation; no direct ET6 memory hit found.

## Boundaries

- Axis label: `[macOS-CPU advisory]`.
- `score_claim=false`; `promotion_eligible=false`.
- All prices are real coder outputs over serialized sampled descriptions. n600 bytes and flips are stratified-sample projections unless marked as prior.
- Support net dS uses ET4's sampled net-flip projection. It is not admitted score movement because the strict receiver-covered mass is zero for every family in this ET6 run.

## Follow-on Disposition

- FIRED: ET6 scorer-free pricing receipt over stratified-random n=32 sample.
- FOLDED: token_cell_edits_support_priced; support price `1.417354` B/flip is above W and no inverse-token receiver proof exists.
- FOLDED: tq1_snap_menu_idx_only; strict reproduction coverage is zero.
- FOLDED: road_lane_grammar_lane_crop; support price `7.071429` B/flip is above W and receiver grammar is absent.
- QUEUED-WITH-FIRE-ORDER: none from ET6.
