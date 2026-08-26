# ddm_hd1_apparatus_two_landings — pay the two owed structural cures: serializer auto-bundle fallback on git-object denial (#1293 leg 2) + (sha,bytes) pin-consistency check (#1237 leg 2)

## MANDATE

Two defect classes hit the fleet repeatedly on 08-26 and both owe their structural
(two-landing) cure. (1) #1293: codex-sandbox `.git/objects` write denial — 4 instances in
ONE day (pc2, hv2, jf2, d3b; d3a/d3c/or1 unaffected — INTERMITTENT). Each arm improvised
a bundle/format-patch and MAIN landed custody by hand. (2) #1237: HALF-UPDATED PIN, 2
instances (dg2's candidate runtimes; jf2's four runtimes — ARCHIVE_SHA256 re-pinned
correctly per candidate, ARCHIVE_BYTES=180_368 inherited from the dx2 template → every
advisory run refused rc=1 at _verify_input; MAIN cure receipt
`/Volumes/APDataStore/pact/ddm_jf2_terminal_diagonal_harvest/scorer/runtime_fix/FIX_NOTE.json`).
Both are DETERMINIZABLE (m141/ac1 law: lessons learned twice become deterministic).

## SCOPE

1. #1293 LEG 2 — serializer auto-degrade: in `tools/subagent_commit_serializer.py`, when
   the commit fails with the git-object write-denial signature ("Operation not permitted"
   touching `.git/objects`), automatically (a) author a `git bundle` (or format-patch
   chain) of the intended commit content to the arm's SSD receipt dir, (b) write a typed
   notify row (canonical_task_status or the receipts JSONL) w/ bundle path + per-file
   shas + the failing environment context (cwd, sandbox markers, uid, mount flags — the
   diagnosis data #1293 says is unowned), (c) exit w/ a distinct rc + printed
   BUNDLE_FALLBACK line so keepers/MAIN see it structurally. EXECUTED positive control
   mandatory (simulate the denial via a read-only .git/objects in a THROWAWAY clone —
   never the live repo) + negative control (normal commit unchanged).
2. #1237 LEG 2 — pin-consistency check: a small canonical checker (new tool or extension
   of an existing preflight surface) that, given a candidate runtime dir, verifies
   sha256(archive.zip) == ARCHIVE_SHA256 AND stat.st_size == ARCHIVE_BYTES inside
   inflate.py — refusing internally-inconsistent (sha,bytes) pairs BEFORE any scorer/fire
   consumes the runtime. Wire it where runtimes are consumed:
   tools/fire_local_advisory.py preflight + make_candidate_seal.py (verify the seal
   producer path doesn't already do this — extend, don't duplicate; grep first, m53).
   Executed positive control on a copy of jf2's ORIGINAL broken runtime (the real
   defect, preserved on APDataStore) + negative control on the fixed copy.
3. Tests for both (positive/negative/waiver where applicable), 2 genuine review passes on
   every .py, serializer commits. Class-population line per landing (the M1 meta-cure,
   #1146): state the measured instance count + the search scope that bounded it.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal, NO scorer. Never point destructive/simulation steps at
  the live repo (.git) — throwaway clones only. Serializer commits w/ post-edit shas; if
  the arm ITSELF hits the #1293 denial, the format-patch fallback it is building is the
  documented path (bundle + shas, MAIN lands).
- Smallest-correct cures (np1 law): extend existing surfaces, no new frameworks; the
  serializer is fcntl-locked shared infrastructure — keep the lock discipline intact and
  the failure path OUTSIDE the lock's critical section where possible.

## PRIOR NEGATIVE SIGNAL

- #1219: a PREVENTION cure never repairs — both cures here must handle the ALREADY-BROKEN
  case (the checker refuses existing bad runtimes; the serializer fallback fires on the
  live denial), not only prevent future authoring.
- #1086/sw2: vacuous controls are the named trap — every control EXECUTED, both
  directions, rc shown in the memo.
- m100: the detector zeroes on the cure — after landing, re-run each detector on the
  cured surface and show zero.

## OPTIMAL FORM

- Family REFERENCE exemplars w/ provenance pins (receipt-backed): the candidate-seal
  contract's control discipline (36 executed controls, commit `361608c875`) · the np1
  smallest-correct pattern (commit `499ffd68a1`) · the serializer's own lock/refusal
  architecture (tools/subagent_commit_serializer.py at HEAD — read before touching) ·
  jf2's FIX_NOTE receipt (path above).
- SCOPE reductions declared (which consumers wired now vs queued). MECHANISM reductions
  FORBIDDEN: real executed controls, real defect artifacts as positive-control inputs.
- **PRIOR-LAW PREDICTION (falsifiable):** the pin checker finds ≥1 MORE internally-
  inconsistent runtime among existing candidate dirs on the SSDs beyond the 2 known
  instances (the class is live, not historical). FALSIFIER: sweep finds 0 → the class
  was fully drained by the two known instances; say so w/ the swept denominator (m50).

## DELIVERABLE

`.omx/research/ddm_hd1_apparatus_two_landings_20260826.md` — both cures + executed
control transcripts + the sweep denominator + class-population lines + ledger receipts.
Serializer commits (.py 2 passes). End w/ the own-vehicle frontier line.
