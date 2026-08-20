# ddm_cp2 — Turn N independent byte wins into ONE measured row. The bottleneck is composition, not bytes.

**Owner:** codex arm · **Base:** PR130 CPR1 · **scorer-free BUILD** (that is the point — the scorer
slot is held by `ddm_ai1`) · `score_claim=false`

## OPTIMAL FORM (read first)

Reference form: a receiver + composition harness such that ANY subset of the landed section-level
candidates can be assembled into ONE real `archive.zip`, parsed back by the REAL `inflate.sh`, and
handed to `upstream/evaluate.py` as a single n600 row — reference pin, verify at source: base
archive 191,052 B, commit `113b52fdb1` (gauge closure), commit `d3650d6c68` (sm3's eight archives).
Declared reductions: SCOPE only — you may build for the currently-landed candidate set and leave the
schema open for later sections. MECHANISM reductions are TOY-BRACKET: a harness that concatenates
byte counts instead of building a real archive; a receiver that reproduces payload in a research
path but not through the shipped `inflate.sh`; a "composed" figure that is arithmetic over separate
runs rather than one stat of one file.

**Authority:** full online research, full OSS, full internal leverage — our own code, docs, and
unwired modules are off the shelf to use, adapt, refactor, extend, or fork. PR130 repro code is
off-the-shelf authorized. Cite path + commit for anything reused.

## WHY THIS, NOW

Measured today, all byte-closed, none scored:

| candidate | section | Δ bytes | state |
|---|---|---:|---|
| `ai1` ANS + temporal_reversion | tokens | −2,416 | `evaluator_runnable: false` — not wired into shipped inflate.py |
| `sm3` pointwise low-rank r32 | semantic | −6,272 | 8 archives retained, 38/38 parse-back, **no receiver for SM3R mode** |
| `sm3` joint vector/scale VQ32 | semantic | −4,648 | same |
| prior SD1 mixed q3/q4 | semantic | −848 | has a measured semantic-leg improvement; pose unmeasured |

Composed (ai1 + sm3 leader) = **−8,688 B = 26.1%** of the 33,252 B sub-0.15 target → 182,364 B.
Marginals are ADDITIVE (measured superadditivity gap −20 B, 0.0267%), so composition is expected to
hold — but "expected" is not measured, and that is precisely what this harness exists to settle.

The binding constraint is no longer finding bytes. It is that **each candidate currently needs its
own scorer pass, and one arm holds the slot.** One composed archive = one pass = every candidate
adjudicated together, including their interactions.

## WHAT TO BUILD

1. **The SM3R receiver.** sm3's own named unmeasured obligation. Decode path for the low-rank /
   VQ semantic modes, wired into the REAL `inflate.sh`, fail-closed on an unknown mode, and
   **byte-identical to the legacy path when the new field is absent** (the sv2 three-case pattern:
   absent → legacy, known tag → new path, anything else → REFUSE). Do not invent a fourth case.
2. **The composition harness.** Given a set {section: candidate}, emit ONE `archive.zip`, stat it,
   sha256 it, parse it back through the real receiver, and prove determinism by double-build
   equality (sm3 already did this per-candidate — reuse its protocol, do not reinvent).
3. **The interaction measurement.** Build the composed archive and compare its ACTUAL size against
   the sum of the individual deltas. If they disagree, that disagreement is a first-class finding —
   the −20 B superadditivity gap was measured on the ORIGINAL sections, not on these replacements,
   and it does not automatically transfer.
4. **Hand off, do not race.** Emit a scorer queue in sm3's existing `SM3_SCORER_QUEUE.json` schema
   listing exactly which composed archives need d_seg/d_pose, ranked. `ddm_ai1` owns the slot; when
   it frees, MAIN fires the queue. **Do not run the scorer in this arm.**

## ALWAYS KEEP THE PAYLOAD (P0, DEF CON 1000 — operator 2026-08-09)

CLAUDE.md `## ⛔ ALWAYS KEEP THE PAYLOAD`. Persist every composed archive with sha256 + byte count,
not just the winner — `ddm_pk4` retained 864 candidate payloads and `ddm_sm3` retained 16 archive/
repeat ZIPs; that is the standard. Run `tac.payload_retention_gate` on anything you write. Known
limitation: it tests PERSISTENCE, not reachability-to-a-writer, so a payload consumed only by
`sha256()` or a parse-back check still counts as LOST
(`.omx/research/ddm_main_p0_triage_20260810.md`).

## HARD RULES

- Bulk artifacts → `/Volumes/VertigoDataTier/pact/ddm_cp2_20260810/`. No `/tmp` in evidence.
- Commits via `tools/subagent_commit_serializer.py`, POST-EDIT `--expected-content-sha256` per file,
  tags `[no-triality] [p0-ledger-ok]`, **no attribution trailer of any kind**.
- `.py` files: 2 × `tools/review_tracker.py mark-file <f> --status reviewed`; never
  `REVIEW_GATE_OVERRIDE=1` with a `.py`.
- Write ONLY under your own `ddm_cp2` paths. Two arms sharing an output directory happened today
  (pk3/pk4) and destroyed the independence of both results.
- `upstream/` IMMUTABLE. Intake clones READ-ONLY. Sister arms' landed artifacts APPEND-ONLY —
  sm3's and ai1's outputs are INPUTS here, never edits.
- Every number carries its axis; macOS is `[macOS-CPU advisory]`, never `[contest-CPU]`. No Modal
  without operator GO. **No scorer run in this arm** — that is the whole design.

## DELIVERABLE

The SM3R receiver with its three-case fail-closed test · the composition harness · ONE composed
archive with exact bytes + sha256 + double-build determinism + real parse-back · the measured
interaction (actual composed size vs summed deltas) · a ranked scorer queue for MAIN to fire.
If composition turns out NOT to be additive on these replacements, say so with the number — that
is a more valuable finding than a byte win, because every downstream budget on this campaign
currently assumes additivity.
