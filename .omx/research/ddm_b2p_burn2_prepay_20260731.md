---
schema: ddm_b2p_burn2_blocker_prepay.v1
date_utc: 2026-07-30
arm: ddm_b2p (burn-2 blocker prepay — QA75 / QA81 / QA80, during the QA24 burn window)
ledger_row: "#783"
research_only: true
score_claim: false
promotion_eligible: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
axis: "[macOS-CPU advisory — scorer-free numpy materialization + producers; NO SegNet/PoseNet run, NO Metal, NO paid dispatch, NO pointer mutation]"
consumes: [ddm_bc1_qa24_compose_and_fire_20260731 (§7 the three burn-2 headlines), ddm_ph3_realization_hybrid_adaptive_convocation_20260731 (§10 the menu),
  ddm_sg1_segnet_typing_and_reburn_20260731 (§5 the QA75 MATERIALIZATION BLOCKER), ddm_ms2r_r3_box_tolerance_solve_20260725 (04_candidate v10 archive),
  tac.witness_dsl.v10_production_receiver (scorer-free decode), tac.canonical_equations.segnet_head_rank4_flipdist_20260715 (the flip-distance law + measured pair-norms),
  experiments/results/mlx_fleet_gt_cache/gt_n{96,600}.npz (cached margins + argmax), codexwt/ddm_cb1_perclass_carrier_byteclose_20260725 (the cb1 carrier branch)]
consumers: [ddm_bc1 burn-2 (QA75 distill targets + QA80 budget field ready), MAIN post-burn boundary (compose burn-2 immediately), the parallel session (QA81 unblock is sequencing on its WIP commit)]
tokens: [p0-ledger-ok]
---

# ddm_b2p — burn-2 blocker prepay: QA75 UNBLOCKED · QA80 producer BUILT · QA81 TYPED BLOCKER

## §0 POINTER HONESTY FIRST (means/ends firewall)

**The exact frontier did NOT move. `0.1910828242 [contest-CPU]` is UNMOVED.** This unit is BUILD +
MATERIALIZATION prepay only — it moves NO trained byte and runs NO scorer. Its entire value is that
burn-2 warm-starts with two of its three §10 blockers cleared and the third precisely diagnosed, so the
post-burn boundary composes burn-2 without a materialization stall. Every number below is
`[macOS-CPU advisory]`; `score_claim=false`. The QA24 seg re-burn (pid 68621) was untouched (never read
or written its out-dir; no scorer/Metal compute while it held the slot).

## §1 PER-LEG STATUS (the report MAIN needs)

| leg | design | prepay verdict | artifact |
|---|---|---|---|
| **QA75** (solve-frame distill targets) | ph3 §10.1 / sg1 §5 MATERIALIZATION BLOCKER | **UNBLOCKED** | 600 per-pair frames on SSD + typed loader landed |
| **QA80** (margin-bounded photometric budget) | ph3 §10.2 | **PRODUCER BUILT** (exact burn-frame n600 field = post-burn scorer step) | `src/tac/boundary_math/margin_budget_field.py` + n96/n600 conservative demo fields |
| **QA81** (cb1 Lane/hood carrier composite) | ph3 §10.3 | **TYPED BLOCKER (not separable)** | cb1 commit requires the WIP `direct_description_carrier_compose.py` |

## §2 QA75 UNBLOCKED — EXACT C1 solve frames materialized (the sg1 §5 / lv1 blocker cleared)

The EXACT-solve archive was decoded to per-pixel target frames via the v10 production receiver's
**scorer-free** decode + factor-2 realization path (`parse_packet` → `decode_y_plane_pair` →
`realize_pair_frame1`). NO SegNet was run (logit/margin distill precompute is the deliberately-deferred
POST-BURN step).

- **Source:** `/Volumes/VertigoDataTier/pact/ddm_ms2r_r3_box_tolerance_solve_20260725T030551Z/stage_checkpoints/04_candidate`
  · archive.zip sha256 `e3d0581ff4a3f475…` (291,205,400 B) · receiver `tac.witness_dsl.v10_production_receiver.v1`.
- **Geometry (MEASURED from the packet header):** 600 pairs · camera 874×1164×3 RGB · `frame0_policy_id =
  description-frame0.v1` → **frame0 is independently described** (both frames materialized, distinct) ·
  `residual_codec_id = None`.
- **Materialized:** `/Volumes/VertigoDataTier/pact/ddm_b2p_20260731/qa75_solve_frames/` — 600 ×
  `pair-NNNNNN.npy` (each `(2,874,1164,3)` uint8 = frame0,frame1), 3.4 GB · `manifest.json` (schema
  `qa75_solve_frame_targets.v1`, **per-pair sha256 for all 600**) · `materialize_receipt.json`.
- **Determinism (proven, not asserted):** `determinism_spotcheck` decoded the packet twice and realized
  pairs {0,300,599} twice → **identical = True** (the v10 realize path is integer-only). Wall 203.1 s.
  Sample `verify_sha` {0,150,300,450,599} all True.
- **Typed loader landed:** `tac.witness_dsl.qa75_solve_frame_targets.SolveFrameTargets` —
  `.frame1(pair_id)` / `.frame0(pair_id)` return memmapped `(H,W,C)` uint8; `verify=True` re-checks the
  sha; fail-closed on schema/range/sha drift. The burn-2 §10.1 distill stage consumes this by `pair_id`.
- **Fail-closed guard (pass-2 review):** the materializer REFUSES a residual-carrying archive (it realizes
  the nullspace plane only) — a future residual archive routes to `inflate_archive`, never a silent wrong
  frame.
- **Owed post-burn (named):** the LOGIT/MARGIN distill FIELD (SegNet on these frames at compress time) —
  the scorer pass that ph3 §10.1 wants — is the post-burn step; the frames it runs on now exist.

## §3 QA80 PRODUCER BUILT — per-pixel flip-distance / margin-budget field `d=|m|/‖Δw‖`

Recall-first FIRST: the flip-distance law already exists as the canonical equation
`segnet_head_rank4_linear_flipdist_v1` (`head_flip_distance_feature_space`, `head_pair_normals_from_weight`,
+ **MEASURED** `HEAD_PAIR_NORMS` — the 10 class-pair `‖w_c − w_c'‖`), and the per-pixel margin field is
CACHED (`gt_n{96,600}.npz['margins']` + `['lstars']`). So QA80's producer is a **thin composition over
existing measured surfaces**, not a rebuild.

- **Landed:** `src/tac/boundary_math/margin_budget_field.py` (+ `src/tac/tests/test_margin_budget_field.py`).
  - `exact_flip_distance_field(margin, winner, runner)` — the EXACT rank-4 law per pixel (needs the
    per-pixel runner-up class).
  - `conservative_budget_field(margin, argmax)` — a **SOUND** budget from the CACHE alone (argmax + margin
    magnitude): divides `|margin|` by the MAX pair-norm over pairs involving argmax → `d_cons ≤ d_exact`
    for any runner-up → provably-safe flip-distance lower bound.
  - `budget_field_from_gt_cache(npz, mode)` + `MarginBudgetField` (summary + save + sha manifest).
- **Runs on real cached data (scorer-free):** conservative budget fields produced from `gt_n96` and
  `gt_n600` → `/Volumes/VertigoDataTier/pact/ddm_b2p_20260731/qa80_margin_budget/` (n96 75 MB q50 1.5363,
  n600 471 MB q50 1.553; float32 + `.manifest.json` sha each; `qa80_demo_receipt.json`).
- **The "needed form" gap (honest, per the task's own fallback):** the EXACT per-pixel field needs the
  RUNNER-UP class, which is NOT in the gt cache (only argmax + margin magnitude are), and computing it on
  the BURN frames requires a SegNet pass = **POST-BURN scorer step** (marked, not run). The conservative
  GT-frame field is a valid safe PRIOR + the producer contract is ready; the burn-frame exact n600 field
  is the named post-burn call `exact_flip_distance_field(margin, winner, runner)`. Pixel-space conversion
  = the pp1 band lemma (registered LAW), the consumer's step.

## §4 QA81 TYPED BLOCKER — cb1 carrier is NOT separable from the WIP compose file

The cb1 branch (`codexwt/ddm_cb1_perclass_carrier_byteclose_20260725T203310Z`, 1 commit `2721704ab2`) lands
the per-class carrier as changes to **`src/tac/optimization/direct_description_carrier_compose.py`** — it
adds `encode_static_class_mask_rule` (the MyCar/hood static-mask carrier codec) + the receiver wildcard
class-code path. **All three cb1 code files depend on that symbol:** `ddm_runtime_exporter.py` (import),
`ddm_runtime_receiver.py` (import), and `tools/measure_ddm_cb1_perclass_carrier_byteclose.py`
(`from …compose import encode_static_class_mask_rule; mycar_rule = encode_static_class_mask_rule(…)`). The
symbol does NOT exist on main.

`direct_description_carrier_compose.py` is a **parallel-session WIP file** (dirty on main with an UNRELATED
change: `requires_pose6_transport` + pose6 transport, disjoint regions from cb1's). Landing cb1 therefore
requires either (a) TOUCHING that WIP file — forbidden by this arm's binding constraint, or (b) clobbering
the parallel session's uncommitted work. Landing only the new files is ALSO impossible: their top-level
`from …compose import (…)` would ImportError on main → landing import-broken code (a fake landing).

**Not separable → TYPED BLOCKER.** It is a **SEQUENCING** blocker, not a design conflict: cb1's and the
WIP's hunks touch DISJOINT regions of the file, so the moment the parallel session COMMITS (or stashes)
its pose6-transport WIP, the cb1 commit cherry-picks onto clean main without textual conflict.
**Unblock owner = MAIN / the parallel session** (commit the WIP), then a clean `git cherry-pick 2721704ab2`
through the landing review gate. All of cb1's non-compose deps (`ddm_hood_static_reassert`,
`ddm_realized_flip_menu`, `ddm_rg4_g3_blocks_and_active_tube`, `ddm_ws1_warm_start`,
`direct_description_measurement_ladder`) are already present on main.

## §5 HYGIENE

- **venv gate0 = ALREADY canonical on main (verified functional).** `tools/launch_tr1_run.py`
  `venv_custody_gate0` already PREFERS `tools/check_venv_src_custody.py` (present on main, commit
  `0af5f1a7e5`) and falls back to the inline `.pth`/`__editable__` scan. Verified end-to-end this session:
  tool present = True, `venv_custody_gate0(repo)` → None (clean) via the tool path. No change needed; the
  bc1 §5 "tool not present" concern was a worktree-visibility artifact, closed on main.
- **DSL stubs LEFT AS-IS (fold-and-delete contract honored).** The three
  `src/tac/witness_dsl/ph3_s10_frontloaded_levers_20260731.py` stubs fold ONLY when their real
  implementation lands (a `spec_tr1_renderer` Lever + trainer flag). This unit prepaid the DATA / PRODUCER
  objects (the distill frames, the budget-field surface) — it made NO trainer flag real — so no stub folds
  (that is burn-2's spec_tr1 landing). The stub docstrings already point at the ledger rows this memo
  flips; no stub edit warranted.

## §6 TRIALITY / verdict scope / STORES CONSULTED

- **DAG:** this memo + `ddm_b2p_burn2_prepay_DAG_FEED_20260731.md` + commit `a7968d9a62` (the two producers
  + n4 tests). **DSL:** no lever became real (QA75/QA80/QA81 flags remain DESIGNED-stubs; the producers are
  data/utility surfaces the burn-2 levers will consume) → `[p0-ledger-ok]`, DSL leg deliberately untouched
  per the fold contract. **equations:** consumed `segnet_head_rank4_linear_flipdist_v1` (measured pair-norms
  + the exact law) — no new equation (QA80 is an application of the registered law, not a new one).
- **verdict_scope:** QA75 = INSTANCE-cleared (frames materialized + determinism-proven + loader tested).
  QA80 = PRODUCER built + unit-tested + demonstrated on real cache; the exact burn-frame field is a named
  post-burn scorer step (not a negative). QA81 = SEQUENCING blocker (parallel-session WIP), not a cb1
  falsification.
- **STORES CONSULTED:** CLAUDE.md (NO-FAKE, OPTIMAL-FORM, never-touch-WIP, DSL-fold contract, review-gate,
  serializer post-edit shas, SSD certify-or-block); docs/operating_manual_craft_handoff; bc1 §7 (the three
  burn-2 headlines) + §5 (gate0); ph3 §10; sg1 §5 (the materialization blocker) + §1 (QA74 typing); the
  v10 production receiver decode contract; the flip-distance canonical equation + measured pair-norms; the
  gt cache (`margins`/`lstars`); the cb1 branch diff + import graph; MEMORY (canonical class order, #141
  margin-saliency, SegNet-fractal flip-distance law).

## §7 SSD custody (certify-or-block)

`/Volumes/VertigoDataTier/pact/ddm_b2p_20260731/` (278 GiB free at write): `qa75_solve_frames/` (3.4 GB,
600 per-pair-sha manifest, deterministically rebuildable from the source archive sha `e3d0581f…` via
`materialize_solve_frames`), `qa80_margin_budget/` (522 MB, per-field sha manifests, rebuildable from
`gt_n{96,600}.npz` via `budget_field_from_gt_cache`), `logs/`. All rebuildable, all sha-manifested, no
`/tmp` in evidence.
