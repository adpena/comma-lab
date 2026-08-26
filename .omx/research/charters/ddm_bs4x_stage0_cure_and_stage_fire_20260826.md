# ddm_bs4x_stage0_cure_and_stage_fire — cure the Stage-0 re-run deadlock, then execute BS4 stages 1–4 per the sealed fire order

## MANDATE

The #1304 triple fire's leg 1 is BLOCKED by a tool-design deadlock (#893 genus: the
ladder cannot represent its own instances). `experiments/ddm_bs4_born_small_stage0_preflight.py`
jointly refuses EVERY re-run:

- `CHECKPOINT` is hardcoded to `checkpoints/stage_00_source_preflight.json` (line 35)
  and written via `atomic_json_once` (line 480), whose payload embeds per-run
  provenance (`sys.argv`, git HEAD, platform — lines 465–478) → the bytes differ on
  every run → the append-only guard (line 174, "refusing to replace different
  checkpoint") raises. Receipt: bs4_stage0_r2 rc=1.
- The mandated-root guard (line 369, "BS4 Stage 0 may write only the charter-mandated
  APDataStore consumer root") refuses any alternative `--output`. Receipt:
  bs4_stage0_r3 rc=1 with `--output .../r2_20260826`.

Yet its own memo (`.omx/research/ddm_bs4_born_small_stage_fire_20260826.md`) contracts
"resume stages 1–4 with new additive checkpoints" when storage clears — which it now
has: sr3's reclaim left AP at ~48,109,191,168 B free vs the 28,220,450,048 B launch
floor.

1. **CURE (small .py edit, 2 genuine review passes):** versioned additive Stage-0
   checkpoints INSIDE the mandated root — e.g. on collision at
   `stage_00_source_preflight.json`, probe `stage_00_source_preflight_r2.json`,
   `_r3.json`, … and write the first name that is absent (or byte-matches). BOTH
   guards stay: the mandated-root guard (line 369) untouched; `atomic_json_once`
   append-only semantics untouched — the ORIGINAL RETAINED-REFUSED checkpoint
   (121,250 B, sha `bfd33e8dc9e6407c218aef14b9095ec40887566705ea1d0ca1e11b8c6ed4e2a7`)
   is NEVER modified or replaced. Executed controls both directions: (a) POSITIVE —
   run Stage-0 twice; the second run must NOT clobber (writes `_rN+1` or byte-matches);
   original checkpoint byte-identical before/after; (b) NEGATIVE — hand the writer a
   path that exists with different bytes at the SAME versioned name mid-race → it must
   still raise. Add a focused test.
2. **RUN Stage-0** post-cure → expect `READY_FOR_STAGE_1` (storage now clears; pins
   revalidate; scorer slot check per the tool). rc=2 with a NEW refusal = typed
   blocker, stop and report — do NOT improvise around a failing pin.
3. **EXECUTE stages 1–4** per the immutable order in the bs4 memo + `FIRE_ORDER.json`
   (schema `ddm_bs3_resolved_carrier_fire_order.v1`, sha `d684c9bc…`), via the
   existing executor lineage (`experiments/ddm_bs3_born_small_resolved_carrier.py` owns
   the stage_10/20/30/40 checkpoint names — read it and the fire order BEFORE
   writing any new code; build only what the fire order demands and the executor
   lacks). Retained n=32 seed-20260826 sample (NPY sha `1d088e908e74de6051…`), per-stage
   additive checkpoints under the mandated root, axis label
   `[macOS-CPU advisory, seeded uniform random n=32 from n600] NON-PROMOTABLE`,
   `score_claim=false`. Stage-5 learned-implicit screen: CONDITIONAL GATE — stays
   QUEUED-BEHIND-THE-EXACT-SOLVE, do NOT fire.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY (weights are pinned inputs, never touched). NO Modal. Local
  SegNet/PoseNet forwards are IN SCOPE per the fire order (this is the scorer-slot
  arm) — n32 local CPU advisory only, never a score claim.
- ALWAYS KEEP THE PAYLOAD (P0 DEF CON 1000): every materialized frame/scorer payload
  persisted under the mandated APDataStore root with sha256+bytes in the stage
  checkpoint; certify-or-block — no BS3 retained tree deleted or moved; storage
  waterfall re-checked before the heavy stage-2 materialization (~19.6 GB projected).
- `.py` edits = 2 genuine review passes; serializer commits w/ post-edit shas; on the
  #1293 git-objects denial the serializer auto-retains a bundle (rc=17) — report it,
  MAIN cherry-picks. Memo corrections APPEND-ONLY.
- NEVER WEAKEN either guard; no blanket waivers; placeholder rationales rejected.
- STOP AND REPORT typed blockers on: any pin mismatch (DX2 runtime/seal, BODY_RESULT,
  BO2 raw, scorer weights) · storage waterfall failure mid-stage · scorer-slot
  contention · anything touching sealed custody.

## PRIOR NEGATIVE SIGNAL

- bs4_stage0_r2 + r3 rc=1 receipts (the deadlock trace above) — the cure must make
  BOTH failure modes unreachable for re-runs while preserving both guards.
- #1237 pin-consistency law: `check_pin_consistency` on the DX2 runtime is already in
  the tool (line 395); do not bypass it.
- qs2/qs4 lesson (cross-regime constant transfer): stage-2 QS5 solves must re-derive
  compensation IN-COMPILE on THIS object, never carry another object's compensation.
- m141: harden-as-default — the r-run versioning cure is exactly a charter-or-fix
  rough edge; fix it structurally, not with a one-off deletion of the old checkpoint
  (deletion would violate append-only AND certify-or-block).

## OPTIMAL FORM

- Family REFERENCE w/ provenance pins: the Stage-0 tool at HEAD (guards at lines
  174/369/480) · bs4 memo (stage table + storage census + fire order) ·
  `experiments/ddm_bs3_born_small_resolved_carrier.py` (stage executor lineage) ·
  FIRE_ORDER.json sha `d684c9bc859f825e5d5341c822dcd8c989f91d3a8e7aef1a44316ced3b333db5`.
- SCOPE reductions declared: n=32 seeded random sample (pre-declared by the bs4
  charter's retained draw — LEGAL scope reduction; n600 verdicts are downstream
  consumers, none claimed here). MECHANISM reductions FORBIDDEN: real scorers, real
  QS5 retention surface, real coders — no synthetic fixtures, no scalar-only runs.
- **PRIOR-LAW PREDICTION (falsifiable):** the cure lands ≤30 changed lines; Stage-0 r4
  returns READY_FOR_STAGE_1 (AP ~48.1 GB > 28.22 GB floor, 151/151 pins expected to
  re-pass); stages 1–4 complete with all payloads retained and zero pin drift.
  FALSIFIER: a pin mismatch (most likely the DX2 runtime tree after recent SSD
  re-pin work, #1237's 11/23 census) → the arm stops with the typed blocker naming
  the exact control row — that outcome is a #1237 residue finding, not a failure.

## DELIVERABLE

`.omx/research/ddm_bs4x_stage0_cure_and_stage_fire_20260826.md` — cure diff summary +
executed controls both directions + Stage-0 r4 receipt + stage 1–4 checkpoint table
(named checkpoint · bytes · sha256 · payloads retained) + ledger rows
(tools/canonical_task_status.py, actor ddm_bs4x) + GESTALT-DELTA line + typed
handoff for MAIN (what the three-way stage-4 measurement says vs the born-small
route's 209× refusal at #1262 — confirm, revise, or blocker). Serializer commits
(or bundles). End with the own-vehicle frontier line.
