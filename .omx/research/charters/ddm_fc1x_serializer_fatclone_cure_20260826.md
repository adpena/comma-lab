# ddm_fc1x_serializer_fatclone_cure — TWO-LANDING: the serializer fallback must never full-clone the checkout (#1302)

## MANDATE

bs3's serializer fallback attempted a FULL-CHECKOUT CLONE onto APDataStore — allocated
8.4 GiB during live storage pressure, breached the reserve, and produced NO landing
artifact (custody fell to MAIN). fb2 certified-and-removed ~12 GiB of the scratch
(certificate + pinned tree `26b27dce163fa2be966b980aa651d8b828e83f1e`). The GOOD path
already exists: hd1's auto-bundle-on-denial (`a08ea28d77`, rc=17) produces small
verified bundles — mg1 and pc2/hv2/jf2 all landed through it. The fat-clone path is a
DISTINCT (legacy or conditional) branch that must die. Pay the two-landing rule:

1. LANDING 1 — cure at source: find the clone-based fallback branch in
   `tools/subagent_commit_serializer.py` (and any sister emitters), make the fallback
   BUNDLE-ONLY: never `git clone` of the checkout, bounded artifact size (bundle +
   receipts only), and a storage-reserve check before ANY SSD write (refuse with a
   typed rc if free space minus the projected artifact would drop below the reserve —
   read the reserve from the same constant the storage waterfall uses, never invent
   one). Preserve the rc=17 contract and receipts schema hd1 landed — mg1's consumption
   this session is the live regression case (its bundle flow must still work
   identically; run the fallback's existing test suite + the hd1 controls).
2. LANDING 2 — class guard: a check refusing re-introduction of clone-based fallback
   (scan the serializer + sister landing tools for `git clone` invocations aimed at the
   repo checkout in a fallback/recovery context; same-line waiver with real rationale
   for legitimate uses elsewhere). POSITIVE CONTROL EXECUTED: a synthetic reintroduction
   must fire the guard; negative direction verified clean.

## HARD CONSTRAINTS

- The serializer is a SHARED LIVE TOOL (sr3 is committing through it right now): edits
  must be small, behavior-preserving on the success path, and the full existing
  serializer test suite must be green before commit. `.py` = 2 genuine review passes.
- `upstream/` READ-ONLY. NO Modal, NO scorer. Serializer commits w/ post-edit shas; on
  the #1293 denial retain the bundle (the very path under edit — if the denial strikes,
  the OLD committed serializer produces the bundle; do not bootstrap through your own
  uncommitted edit).
- Receipts: the bs3 arm final message dead-end line + fb2's cleanup certificate are the
  incident receipts; cite them, do not re-derive the incident.

## PRIOR NEGATIVE SIGNAL

- #1293/#1300 (memo `.omx/research/ddm_hd1_apparatus_two_landings_20260826.md`): the
  denial class is real and recurrent (6 instances) — the bundle path is proven; this
  arm must not regress it.
- #1219 (the #1122 recurrence row; cure receipts in the AppleDouble incident notes of
  the harness ledger): a PREVENTION cure never repairs — the guard (landing 2)
  prevents; landing 1 is the repair; both are owed, neither substitutes for the other.
- #1302 (this task's incident receipts): the bs3 arm final message
  `.omx/research/arm_final_messages/ddm_bs3_born_small_resolved_carrier_20260826T203546Z.md`
  dead-end line + fb2's certificate above.
- m102: control-plane failures are silent — the storage-reserve refusal must be LOUD
  (typed rc + receipt), never a silent skip.

## OPTIMAL FORM

- Family REFERENCE exemplars w/ provenance pins: hd1's bundle-fallback implementation,
  commit `a08ea28d77` (`tools/subagent_commit_serializer.py` +
  `src/tac/tests/test_subagent_commit_serializer_bundle_fallback.py`) + hd1's landing
  memo `.omx/research/ddm_hd1_apparatus_two_landings_20260826.md` (commit `f3d6aba3e1`)
  carrying the executed synthetic + LIVE denial controls + fb2's cleanup certificate
  (pinned tree `26b27dce163fa2be966b980aa651d8b828e83f1e`, memo
  `.omx/research/ddm_fb2_route_table_gb1_20260826.md`).
- SCOPE: the serializer + direct sister landing tools only. MECHANISM reductions
  FORBIDDEN: the positive control is executed, not asserted; the reserve constant is
  read from the canonical storage-waterfall source.
- **PRIOR-LAW PREDICTION (falsifiable):** the fat-clone branch is a reachable legacy
  path (pre-hd1 code or a conditional hd1 missed), and removing it plus the reserve
  check costs <100 changed lines with zero behavior change on the success + bundle
  paths. FALSIFIER: the clone came from OUTSIDE the serializer (another tool bs3
  invoked) — then the cure lands at THAT tool, named explicitly, and the serializer
  gets only the reserve check.

## DELIVERABLE

`.omx/research/ddm_fc1x_serializer_fatclone_cure_20260826.md` — the located clone
path + both landings + executed controls (positive/negative) + serializer suite green +
ledger rows closing #1302 + GESTALT-DELTA line. Serializer commit (or bundle). End with
the own-vehicle frontier line.
