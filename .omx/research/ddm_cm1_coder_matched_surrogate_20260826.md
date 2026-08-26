# DDM CM1 — coder-matched rate-surrogate verdict

**Date:** 2026-08-26  
**Axis:** `[macOS-CPU advisory / scorer-free EXACT byte targets]`  
**Task:** `ddm_no1_row1_three_term_objective`  
**Verdict scope:** **FORMULATION**, on 8 retained current-field perturbations (6 dx2/19-family + 2 fs2/13-family); this is not a population-level family kill.  
**Score claim:** none. No scorer, Modal dispatch, archive mutation, or `upstream/` write occurred.

## Result first

**NOT FIREABLE.** The cheapest candidate clearing both held-out gates (`Pearson >= 0.9` and `Spearman >= 0.9`) is the real F26/HPAC 128-frame prefix at `(0.962088, 0.942857)`, `n=6`, but it costs **167.911 s/evaluation** and is a non-differentiable coder output. The exact edit-support window reaches `(0.999999, 1.0)`, but without a restartable coder-state cache it still costs the full-state median **897.675 s/evaluation**. No static form reached the correlation gate.

Therefore row 1 must **RE-PRICE TO RESTARTABLE EXACT-INCREMENTAL / OUTER-LOOP CODING**. The other prerequisite, wd3's single-seed variation, remains MAIN-owned.

## Typed candidate rows

All byte targets are archive-byte deltas from retained real F26/HPAC receipts whose unchanged control reproduced the shipped token stream byte-identically. Prefix and edit-support predictions are sums of retained real-coder per-frame code-bit ledgers, never entropy estimates.

| family | rho Pearson | rho Spearman | held-out n | wall s/eval | typed verdict |
|---|---:|---:|---:|---:|---|
| `exact_full_f26_hpac_reencode` | 1.000000 | 1.000000 | 6 | 897.675 | exact physical target; not inner-loop trainable |
| `windowed_exact_prefix_32` | 0.175818 | 0.028571 | 6 | 38.421 | refused: misses correlation gate |
| `windowed_exact_prefix_64` | 0.814205 | 0.657143 | 6 | 80.198 | refused: misses correlation gate |
| `windowed_exact_prefix_128` | 0.962088 | 0.942857 | 6 | 167.911 | **correlation winner; not trainable-cost** |
| `windowed_exact_prefix_256` | 0.991862 | 1.000000 | 6 | 340.460 | passes correlation; dominated by prefix-128 on cost |
| `windowed_exact_edit_support_halo_0_full_state` | 0.999999 | 1.000000 | 6 | 897.675 | local exact support; full-state cost remains |
| `windowed_exact_edit_support_halo_1_full_state` | 1.000000 | 1.000000 | 6 | 897.675 | local exact support; full-state cost remains |
| `windowed_exact_edit_support_halo_2_full_state` | 1.000000 | 1.000000 | 6 | 897.675 | local exact support; full-state cost remains |
| `windowed_exact_edit_support_halo_4_full_state` | 1.000000 | 1.000000 | 6 | 897.675 | local exact support; full-state cost remains |
| `windowed_exact_edit_support_halo_8_full_state` | 1.000000 | 1.000000 | 6 | 897.675 | local exact support; full-state cost remains |
| `static_sm2_bank_transfer` | -0.128194 | -0.214286 | 8 | 3.654 | refused: 152-row SM2 fit fails on current-field rows |
| `static_sm2_bank_plus_fs2_to_dx2_heldout` | 0.207665 | 0.028571 | 6 | 3.654 | refused: 152 bank + 2 fresh fs2 fit fails on held-out dx2 |
| `static_marginal_entropy` | -0.397012 | -0.833333 | 8 | 3.654 | refused: anti-correlated LOOCV |
| `static_hard_entropy_pair` | -0.395273 | -0.833333 | 8 | 3.654 | refused: anti-correlated LOOCV |
| `static_context_compact` | -0.159089 | -0.452381 | 8 | 3.654 | refused: direction/mismatch features still fail LOOCV |

The static wall time is measured offline reconstruction plus retained-field feature extraction, not an optimized inner-loop implementation. It is quoted only to show that the cheap forms are inaccurate. The two-point fs2-only SM2 transfer has a mechanically perfect rank correlation because `n=2`; it is degenerate and is not promotion evidence. On all 8 rows it is negative, and after fitting those 2 fs2 rows its held-out dx2 result is `0.207665/0.028571`.

The prefix costs are real control-prefix F26/HPAC runs. The edit-support candidates use exact local sums from full-context per-frame ledgers; producing those ledgers currently requires the full encode. Halo 0 already has only 0.806 B MAE across all 8 rows, so the missing object is restartable coder state, not a wider halo.

## Prior-law adjudication

The prior-law prediction survived this bounded test. Every static top-level candidate stayed below Pearson 0.5, while only real-coder forms crossed 0.9. The result does **not** close all learned or stateful surrogate families: the live bank is only 8 selected perturbations, not stratified-random `n>=32`, and the successful edit-support calculation exposes a plausible restartable-state implementation rather than a mechanism wall.

SM2's deterministic modulo-5 bank split (`121 train / 31 held out`) gave Pearson `0.352556`, Spearman `0.732775`, RMSE `36,095.1 B`. This is consistent with its earlier non-promotion and with rsf1's direction-of-failure law.

## RECALL EVIDENCE

Searched the full relevant corpus before building:

- `.omx/research/`, arm receipts, and source by content with queries including `rate surrogate`, `entropy`, `F26`, `HPAC`, `adaptive context`, `match structure`, `bits_per_frame`, `real reencode`, `token rate`, and `coder-matched`.
- The canonical registry via `.venv/bin/python tools/list_canonical_equations.py --json`, including `token_rate_model_direction_dependence_v1` and `section_coding_axis_closure_v1`.
- `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks, design/spec files, `.omx/state/main_hot_state.md`, and the canonical task ledger for `ddm_no1_row1_three_term_objective`.
- Governing sources `PROGRAM.md`, `AGENTS.md`/identical `CLAUDE.md`, `docs/operating_manual_craft_handoff.md`, the charter, and the common contract.

Beyond the charter seeds, the search found six LD1 exact re-encodes on the same pinned `cc10a7b0...` current token field, with 2,500–60,000 real edits and byte-identical dx2/19-family controls. That changed the plan from a two-row fs2 check into a six-row same-coder held-out surface plus a two-row cross-corrector check. It also found retained per-frame real-coder ledgers, which made the prefix and explicit edit-support races possible without regenerating the eight full payloads. The canonical direction-dependence law changed the proxy set to include edit direction and spatial/temporal state; those additions still failed.

No additional existing coder-matched surrogate meeting the charter's gate was found in those scopes.

## Custody

Primary retained store: `/Volumes/APDataStore/pact/ddm_cm1_coder_matched_surrogate/`

- `RESULTS.json` — 16,230 B — sha256 `833e4100e59aa8f7206d21c4be3a0a4c0fa704611cc9f3ebc15919cb468514df`
- `ROWS.jsonl` — sha256 `5ff3decc0ff6daaffbe4f61ca9598ac1bd069b487c65f451880f8336c2d04543`
- `MANIFEST.json` — sha256 `fa89e76a3510b1758b3b66de0a88d5438cb7cc31bdd0261af761d678f857517a`
- final safe-run receipt `analysis_safe_run_status_attempt6.json` — sha256 `6119095399015630ddfca709139a97d9e26f17736f2bddd32369ec1488bde7f9`
- executed source script — 27,725 B — sha256 `82c14100fa6af36b75916cf301ce3c47bc72d11232dbe94f1ec866469bb3739c`; this exact pre-commit working-tree source fact is recorded in both `RESULTS.json` and `MANIFEST.json`.
- eight reconstructed candidate token fields, each 117,964,800 B, retained under `retained/fields/`; the manifest records each path and sha256.
- real prefix payloads: `w032` 6,031 B sha `8dbf040d...`, `w064` 12,106 B sha `a872c430...`, `w128` 25,262 B sha `5c62d452...`, `w256` 48,508 B sha `774ff32b...`.

The manifest independently verified 14 retained artifact facts before the edit-support extension; the final harness additionally verifies candidate and control per-frame ledger hashes before use. All candidate fields were persisted before static measurement. No materialized payload was discarded.

## Verification

- `ruff check` and `py_compile`: green.
- Two genuine `review_tracker.py` file-review passes: completed after correctness/custody review and independent result review.
- Independent SciPy recomputation matched every prefix Pearson/Spearman value to `<1e-12` and verified retained artifact byte counts and hashes.
- SM2 soft-histogram count implementation matched the original SM2 vector implementation to `<1e-12` on symbol and temporal-delta controls.

## Ledger receipt and fire order

`tools/canonical_task_status.py update` appended the final row at `2026-08-26T13:04:24.934287Z`: actor/session `ddm_cm1`, owner retained as `MAIN`, status `blocked`, test status `green`. Earlier CM1 results were explicitly superseded after adding the charter's edit-local candidate and executed-source custody; the final row cites the current `RESULTS.json` sha above.

- **QUEUED** — owner `MAIN`; consumer `.omx/state/canonical_task_status.jsonl::ddm_no1_row1_three_term_objective`; fire only when a restartable F26/HPAC coder-state cache or exact-incremental outer-loop implementation makes the near-perfect edit-support signal cheap enough for the intended optimizer and validates both correlations at `>=0.9` on stratified-random `n>=32` current-field rows.
- **QUEUED** — owner `MAIN`; consumer the same canonical row; fire wd3 seed variation only after the Metal lane is available and MAIN owns the single-flight claim. CM1 did not touch this prerequisite.

**GESTALT-DELTA:** coder state, not another static statistic, is the missing sufficient object: local real-code contributions already predict the byte delta almost exactly, but accessing them currently costs the full adaptive encode.

## LIVE-HYPOTHESES

- A restartable F26/HPAC state checkpoint immediately before edited pairs can preserve halo-0's near-perfect ordering while cutting the 897.675 s full-state cost, because halo 0 already has 0.806 B all-row MAE once the true coder state is available.
- The prefix transition between 64 and 128 frames suggests an effective adaptive-context horizon near that scale on this candidate bank; a cached-state 128-frame local replay is plausible, but it needs random held-out rows.
- A differentiable learned surrogate of the coder's match-length and adaptive-context state may succeed where marginal/context-count features failed, because sv2 identifies those internal states as the paid mechanism and CM1 did not model them directly.

## DEAD-ENDS

- Marginal entropy and hard entropy plus temporal entropy are closed for this formulation/current-field bank: both are strongly anti-correlated with real bytes.
- The existing 152-row SM2 affine model is not transferable to these current F26/HPAC perturbations: its all-row correlation is negative, and adding the two fresh fs2 rows leaves held-out dx2 at `0.207665/0.028571`.
- Static direction, position-cost, and spatial/temporal mismatch counts are closed for this formulation: their LOOCV correlation is negative.
- Prefixes of 32 and 64 frames are too short on this bank: neither clears both correlation gates.
- Full-state edit-support replay is not itself a trainable surrogate: it is accurate but costs the same 897.675 s median as the full encode.

Own-vehicle frontier: **S 0.14811799921260607 @ 180,215 B [contest-CUDA T4, n600]** — unchanged by CM1.
