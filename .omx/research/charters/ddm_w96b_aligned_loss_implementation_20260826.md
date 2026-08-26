# ddm_w96b_aligned_loss_implementation — build the REAL aligned-config loss (exact expected-flip margin) + content-addressed retention that shrinks the two-seed storage demand

## MANDATE

w96a (memo `.omx/research/ddm_w96a_aligned_config_renderer_window_20260826.md`, commit
`fc915c771f`) closed honest-BLOCKED on two gates without measuring the aligned W96
hypothesis. This arm pays BLOCKER 2 (implementation) in full and attacks BLOCKER 1
(storage 45,521,567,744 B two-seed retention vs APDataStore 22,319,071,232 B free) at the
DEMAND side via w96a's own live-hypothesis 3. NO launch from this arm — MAIN fires the
two seeds sequentially per w96a's recorded fire-orders once both gates are green.

1. IMPLEMENT the aligned objective FOR REAL — w96a's checklist verbatim: exact
   expected-flip MARGIN objective (CE1's law, NOT WD3's calibrated target-probability
   disagreement — renaming is a closed dead-end), tau schedule, cosine floor, STEP-ZERO
   pose gate (live-hypothesis 2: pose active from step zero to prevent the pose-dominated
   failures of every prior W96 diagonal), resume identity, tests, 2 genuine review
   passes. Wire into the WD3/S1A trainer surface as a selectable loss law (default OFF,
   byte-identical when off), DSL-registered per the lever discipline.
2. BUILD lossless content-addressed retention for the aligned window's evaluation trees
   (live-hypothesis 3): the retained trees contain large repeated fields — dedup by
   content hash with EVERY payload identity + sha256 independently recoverable (ALWAYS
   KEEP THE PAYLOAD is satisfied by content-addressed storage; a discarded byte is not).
   MEASURE the resulting two-seed retention demand on the real retained OFF replay
   (`/Volumes/APDataStore/pact/ddm_w96a_aligned_window/off_baseline_s1e_rerun.json` +
   the 35-checkpoint trees) — emit the derived post-dedup demand vs the 22,319,071,232 B
   available.
3. VERDICT + HANDOFF: implementation-green receipt (tests + review passes + resume
   identity proof) + the measured post-dedup storage demand + an updated sealed
   fire-order for MAIN (seeds 20260815, 20260816 sequential, S1E n60 re-screen per
   checkpoint, ≥5× OFF-screen improvement gate before any n600 claim). Ledger rows via
   tools/canonical_task_status.py (actor ddm_w96b), superseding w96a's `blocked` rows
   where paid.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO training launch, NO Metal claim, NO scorer, NO Modal from
  this arm. Serializer commits w/ post-edit shas; `.py` = 2 genuine review passes; on the
  #1293 git-object denial retain a serializer-authored bundle + shas (MAIN lands).
- ALWAYS KEEP THE PAYLOAD → `/Volumes/APDataStore/pact/ddm_w96a_aligned_window/` (shared
  AP root, additive only; w96a's receipts read-only). Respect the storage contract:
  no local-disk fallback, no symlinks-as-storage (closed dead-end).
- No naive/toy/generic (operator 08-26): the loss law must be the EXACT expected-flip
  margin form, derived and tested — a lesser surrogate re-creates the defect w96a
  refused. Never recall from working memory alone (m44).

## PRIOR NEGATIVE SIGNAL (binding — w96a's dead-ends carry forward)

- Renaming WD3's loss "aligned" is CLOSED — different executed law, real build required.
- Local-disk fallback / symlinks / discarded payloads CLOSED by the storage contract.
- Higher-LR-as-cure CLOSED on the ancestor formulation (6e-5 weakened, 1e-5 null).
- SVD-r32 FORBIDDEN (unconditional prior verdict).
- The 35 OFF-config rows are NOT an aligned-family closure (different formulation).

## OPTIMAL FORM

- Family REFERENCE exemplars w/ provenance pins: CE1's measured objective findings
  (#1089: 81.19% of LR budget to the worst-aligned objective, scale-invariant; #1091:
  the seg wall is 92.7% configuration, 13.6× closure) · w96a's verdict memo + storage
  blocker receipt (`STORAGE_PREFLIGHT_BLOCKER.json`) + OFF replay (commit `fc915c771f`)
  · the wd3 trainer surface at HEAD.
- SCOPE reductions declared (e.g. dedup measured on the existing OFF trees, not a
  hypothetical). MECHANISM reductions FORBIDDEN: the loss law is exact or the arm
  reports blocked, never approximated silently.
- **PRIOR-LAW PREDICTION (falsifiable):** (a) the exact expected-flip margin law lands
  green (build-scope, no unknown physics), and (b) content-addressed dedup cuts the
  two-seed retention demand BELOW 22,319,071,232 B (the repeated-field hypothesis) —
  making the storage gate free. FALSIFIER for (b): post-dedup demand stays above
  APDataStore free → storage expansion is the true binding gate, routed to the #1165
  Vertigo-reclaim boundary (pk4 cold-move 08-27) with the measured demand as its input.

## DELIVERABLE

`.omx/research/ddm_w96b_aligned_loss_implementation_20260826.md` — implementation-green
receipt (tests, review passes, resume identity) + measured post-dedup two-seed demand +
updated sealed fire-order for MAIN + ledger receipts + GESTALT-DELTA line + payload
shas. Serializer commit (or bundle per #1293). End with the own-vehicle frontier line.
