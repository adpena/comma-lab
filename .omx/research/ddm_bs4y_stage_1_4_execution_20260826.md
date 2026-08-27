# ddm_bs4y — born-small stages 1–4 execution verdict (2026-08-26)

**Status: STAGES_1_THROUGH_4_COMPLETE** (checkpoint `checkpoints/stage_40_three_way_measurement.json`,
sha256 c02191f98a7ad60f0193acab8c58dfa82e5b855828792e1a4ce306f2a0d91344; stage_5 NOT fired, gated by design).
Axis: **[macOS-CPU advisory, seeded uniform random n=32 from n600] NON-PROMOTABLE**, `score_claim=false`,
`promotion_eligible=false`. Subsample mode: seeded uniform random from the sealed 32-pair sample — NOT a
prefix (the [[m88]]/bp2 prefix-bias genus honored by construction). 20 of 32 sealed pairs measured,
60 scorer forwards. All materialized payloads retained under
`/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/retained/bs4y/` (charter: nothing deleted or moved;
SR3 tarball untouched; `upstream/` untouched).

## The three-way measurement (distortion-S, per-pair mean over the 20 measured pairs)

| leg | distortion S | vs base |
|---|---|---|
| gb1/dx2 base (frontier body) | 0.07624409308700053 | — |
| born-small + STALE carrier | 3.88433486543213 | +3.808 |
| born-small + FRESH solve | 3.5308745779508395 | +3.455 |
| born-small @ PERFECT pose (floor) | 1.2316650424494517 | **+1.155** |

- `pose_s_gap_stale_minus_base` = 2.6527; the fresh solve recovers 0.3535 of it →
  **pose_gap_recovered_fraction = 13.32%**.
- `carrier_is_frame0_only=true` and `d_seg_identical_across_carrier_legs=true` — the carrier legs do not
  move seg; every seg/rate excess lives in the born-small BODY.
- gb1 contest row quoted in the checkpoint as CONTEXT ONLY (different authority surface, never inserted
  into the n32 legs).

## Adjudication vs #1262 (born-small REFUSED at 209×)

**The fresh-solve rescue hypothesis is REFUTED at INSTANCE scope** (sealed BS3 random-n32 born-small
object through the exact DX2 receiver, carrier and scorers):

1. Fresh in-compile solving of the carrier — the qs5-proven compensation discipline, applied at the
   receiver-materialized level — recovers only **13.3%** of the pose gap. Carrier staleness is NOT the
   route's binding defect.
2. Even granting **perfect pose** (a floor no realizable carrier reaches), the born-small body sits at
   distortion S 1.232 vs base 0.076 — **16.2× worse with the entire pose term deleted**. The route dies
   on the seg side of the body itself.
3. This is concordant with, and mechanistically decomposes, #1262's 209× refusal: the two measurements
   used different instruments and agree in sign and order. The #1247 "byte-feasible by 36,858 B" framing
   stands as a RATE fact only — the byte headroom cannot be purchased because the object's distortion
   floor exceeds the entire budget it would free (per the [[m124]] two-readings law: at ~6.66e-7 S/B,
   +1.155 S ≈ 1.7M B equivalent — 47× the byte win).

**Route disposition:** born-small (bs2/bs3/bs4 lineage) CLOSED as a rescue-by-fresh-solve candidate.
verdict_scope: INSTANCE (this sealed object, this receiver); the FAMILY question "can a born-small object
with a co-designed body reach base distortion" is untouched by this row and remains routed through the
#1270/S1 trained-renderer diagonal and the #1215 2×2 diagonal cell (field AND model move together) — this
measurement entered the diagonal for THIS object and it refused.

## Execution record (the two resume cures + the storage unblock)

- r2 (rc=1, 2,135 s): stage-3 parse-back pin mismatch — the executor hashed the RAW compressed semantic
  section where `rj2.fresh_process_parseback` hashes the RECEIVER-MATERIALIZED `parts.semantic_blob`
  ([[m99]] units/level genus). Cured in-compile via `receiver_semantic_sha()` (same-reader equality, no
  carried constant), commit b2374599c8; positive+negative controls executed.
- r3 (rc=1, 99 s): `rj2.atomic_bytes` correctly refused to overwrite the retained r2 FAILING transcript
  (retention discipline working). Cured with additive `_rN` transcript versioning (never clobber),
  commit 33a06db3bb.
- r4 (rc=1, 21 s): stage-3 PASSED; stage-4 storage waterfall refused pair 73 — AP free 4,294,836,224 B
  sat 131,072 B under the 4 GiB floor + 6,104,016 B stack. Cured by a certified MOVE (never delete) of
  `mlx_strict_score_calibration_pr101_pose_axis_20260522` (8 files, 5,662,338,053 B, SC3-cold-store
  content) AP → `/Users/adpena/pact_local_coldstore/ap_overflow_20260826/`, per-file sha256 verified
  byte-identical post-copy; machine-readable relocation manifest at the original AP parent
  (`…RELOCATED_TO_LOCAL_20260826.json`). AP freed to 9.3 Gi.
- r5 (rc=0, 63 s): stages 3–4 completed; per-pair stage-4 artifacts from r4 were retained and skipped
  idempotently — only the tail ran.

## Consumers / routing

- #1304 bs4y leg: DONE. Remaining #1304 legs: w96b aligned seeds (retention demand >22.32 GB post-dedup
  still exceeds AP's 9.3 Gi → stays routed to the #1165 Vertigo 08-27 cold-move boundary) + rb1 sealed
  configs (fire when a slot and storage clear).
- GESTALT-DELTA (per [[all-signal-informs-evolving-gestalt]]): the sub-0.12 "different object" search
  loses its second-best candidate to a measured body-floor, not to carriage — reinforcing the sy2
  object-change law: a new object must beat the dx2 base's distortion floor BEFORE its rate story counts.
