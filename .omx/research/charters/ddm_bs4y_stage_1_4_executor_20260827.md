# ddm_bs4y_stage_1_4_executor — build the missing stage 1–4 executor and run the born-small three-way measurement per the sealed FIRE_ORDER

## MANDATE

The #1304 leg-1 chain is GREEN through Stage-0: sr3 reclaimed AP to 69,386,371,072 B free
(receipt `/Volumes/APDataStore/pact/ddm_ai1_20260809/SR3_VERIFICATION_RECEIPT.json`, status
RECLAIMED_VERIFIED) and Stage-0 r4 returned **READY_FOR_STAGE_1 with zero blockers**
(checkpoint `/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/checkpoints/stage_00_source_preflight_r4.json`,
121,056 B, written 2026-08-26). The immutable order in
`/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/FIRE_ORDER.json`
(schema `ddm_bs3_resolved_carrier_fire_order.v1`, sha
`d684c9bc859f825e5d5341c822dcd8c989f91d3a8e7aef1a44316ced3b333db5`) now demands stages 1–4 —
and the stage 1–4 EXECUTOR IS UNBUILT: `experiments/ddm_bs3_born_small_resolved_carrier.py::main()`
only rebuilds BODY_RESULT/FIRE_ORDER (read its main() at lines 513–545 first; it OWNS the
stage_10/20/30/40 checkpoint names — extend that lineage, do not invent a parallel one).

1. **READ BEFORE CODE:** FIRE_ORDER.json in full · the bs4 memo
   `.omx/research/ddm_bs4_born_small_stage_fire_20260826.md` · the bs4x cure memo
   `.omx/research/ddm_bs4x_stage0_cure_and_stage_fire_20260826.md` ·
   `experiments/ddm_bs3_born_small_resolved_carrier.py` · the three sha-pinned reference
   implementations below. Build ONLY what the fire order demands and the executor lacks.
2. **BUILD the executor** (new `experiments/ddm_bs4_born_small_stage_executor.py` or an
   extension of the bs3 module — smallest honest form) implementing the immutable order:
   - **Stage 1** — born-small frame-1 masters from the exact BO2 receiver (raw pinned in
     FIRE_ORDER; revalidate its sha before consumption).
   - **Stage 2** — DX2 600×12 int12 carrier decode + QS5 central-difference 6×12 PoseNet
     Jacobians. ~19.6 GB projected materialization: re-run the storage waterfall BEFORE this
     stage and fail closed if the mandated root cannot hold it.
   - **Stage 3** — RJ2 CPR1→CAP1→DX2→RR5→Brotli q9/w16 production re-encode, carrier
     section ONLY.
   - **Stage 4** — three-way realized d_seg/d_pose measurement: GB1/DX2 base vs
     BO2-stale-carrier vs born-small-fresh-solve, on the retained n=32 seed-20260826 sample
     (selection NPY sha-pinned in FIRE_ORDER `scorer_scope.selection`; NEVER a prefix).
   - **Stage 5 (learned-implicit screen): CONDITIONAL GATE — stays QUEUED-BEHIND-THE-EXACT-SOLVE.
     Do NOT fire.**
3. Reference implementations (provenance pins — consume these mechanisms, byte-verify each
   file's sha before lifting code from it):
   - DX2 carrier decode: `experiments/ddm_po1_t4_error_feedback_pose_compensation.py`
     sha256 `7d68c0c716ed37279cc99ec077ff0804f54005298648ce2e24849f612e2295a8` (45,542 B)
   - DX2 surface + production encoder: `experiments/ddm_rj2_joint_renderer_object_change.py`
     sha256 `59fa4498227c1586ac80e5c262ad75309c138db7ff6f0e6345eec03cc3cc6815` (61,608 B)
   - QS5 exact solve: `experiments/ddm_qs5_resolve_compensation.py`
     sha256 `19a781cdc527ee18750213a913359aae5b959646903d6e7673afb74050ebc2bc` (42,614 B)
4. Per-stage ADDITIVE checkpoints under the mandated root
   (`/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/checkpoints/`, stage_10/20/30/40
   names per the bs3 lineage; on collision use the `_rN` versioning cure from
   `experiments/ddm_bs4_born_small_stage0_preflight.py` — never clobber, never delete).
   Each checkpoint records every materialized payload's path+sha256+bytes.
5. Axis label on EVERY scored row: `[macOS-CPU advisory, seeded uniform random n=32 from
   n600] NON-PROMOTABLE`, `score_claim=false`, `promotable=false`. One full-scorer chunk at a
   time per FIRE_ORDER `scorer_scope` (chunk_max_pairs=32).

## HARD CONSTRAINTS

- `upstream/` READ-ONLY (scorer weights = pinned inputs). NO Modal. Local SegNet/PoseNet
  forwards ARE in scope (this is the scorer-slot arm) — n32 local CPU advisory only, never a
  score claim.
- ALWAYS KEEP THE PAYLOAD (P0 DEF CON 1000): every materialized frame/decode/Jacobian/scorer
  payload persisted under the mandated APDataStore root with sha256+bytes in the stage
  checkpoint; certify-or-block — nothing deleted or moved; the SR3 archive
  (`/Volumes/APDataStore/pact/ddm_ai1_20260809/SR3_ORIGINAL_TREE.tar.zst`) is UNTOUCHABLE.
- qs2/qs4 lesson (cross-regime constant transfer): every QS5 compensation is re-derived
  IN-COMPILE on THIS object — assert it in code; never carry another object's compensation.
- `.py` edits = 2 genuine review passes + `tools/review_tracker.py mark-file`; serializer
  commits w/ post-edit shas; on the #1293 git-objects denial the serializer auto-retains a
  bundle (rc=17) — report it, MAIN cherry-picks. Memo corrections APPEND-ONLY.
- STOP AND REPORT typed blockers on: any pin mismatch (BO2 raw, DX2 runtime/seal,
  BODY_RESULT, scorer weights — #1237's census is live context) · storage waterfall failure
  mid-stage · scorer-slot contention · anything touching sealed custody. Do NOT improvise
  around a failing pin.
- Resumability P0: the executor takes `--resume-from` and continues from the last completed
  stage checkpoint; a crash loses at most one stage.

## PRIOR NEGATIVE SIGNAL

- #1262: born-small REFUSED at 209× on the bo2 distortion row — the stage-4 three-way
  measurement is the adjudication of that verdict at optimal form (fresh solve vs the stale
  carrier bo2 measured). Do not pre-assume either direction.
- bs4_stage0 r2/r3 rc=1 deadlock (cured by the `_rN` versioning) — reuse that cure for any
  checkpoint-name collision; never weaken `atomic_json_once`.
- qs4's +2.4e-4 pose disaster (stale Schur compensation carried across objects) — the
  IN-COMPILE re-derivation assert is mandatory, not advisory.
- #821: N sites of one copied pattern = ONE fact — lift shared mechanisms from the reference
  implementations by import/refactor where practical, not by triple copy-paste.

## OPTIMAL FORM

- Family REFERENCE w/ provenance pins: the three sha-pinned reference implementations above ·
  the Stage-0 tool (`experiments/ddm_bs4_born_small_stage0_preflight.py`, guards + `_rN` cure)
  · FIRE_ORDER.json sha `d684c9bc859f825e5d5341c822dcd8c989f91d3a8e7aef1a44316ced3b333db5` ·
  Stage-0 r4 checkpoint (121,056 B, READY_FOR_STAGE_1).
- SCOPE reductions declared: n=32 seeded random sample (pre-declared by the bs4 charter's
  retained draw — LEGAL; n600 verdicts are downstream consumers, none claimed here).
  MECHANISM reductions FORBIDDEN: real BO2 receiver, real DX2 int12 decode, real QS5
  central-difference Jacobians, real RJ2 coder chain at q9/w16 — no synthetic fixtures, no
  scalar-only runs, no toy stand-ins for any stage.
- **PRIOR-LAW PREDICTION (falsifiable):** stages 1–4 complete with all payloads retained and
  zero pin drift; the stage-4 three-way row measures the born-small fresh solve's realized
  distortion BELOW the stale-carrier BO2 row (the #1262 209× refusal was measured on the
  STALE carrier — the fresh-solve leg has never been measured). FALSIFIER: fresh-solve
  distortion ≥ the stale row → #1262's refusal extends to the fresh object and the
  born-small route closes at MEASURED scope — that is a finding, not a failure.

## DELIVERABLE

`.omx/research/ddm_bs4y_stage_1_4_execution_20260827.md` — executor build summary (diff +
review passes) + stage 1–4 checkpoint table (named checkpoint · bytes · sha256 · payloads
retained) + the three-way stage-4 measurement table w/ axis labels + typed handoff for MAIN
(confirm/revise/blocker vs #1262's 209× refusal) + ledger rows
(tools/canonical_task_status.py, actor ddm_bs4y) + GESTALT-DELTA line. Serializer commits
(or bundles). End with the own-vehicle frontier line.
