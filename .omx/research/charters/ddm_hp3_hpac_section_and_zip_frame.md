# ddm_hp3 — The last unattacked section (hpac, 15,092 B) + the container itself.

**Owner:** codex arm · **Base:** PR130 CPR1 · scorer-free until the final gate ·
`[macOS-CPU advisory]` · `score_claim=false` until a byte-closed archive is evaluated

## OPTIMAL FORM (read first)

Reference form: the hpac section re-represented and the archive CONTAINER re-framed so the measured
`archive.zip` shrinks at unchanged d_seg/d_pose, proven by the real stat — reference pin: the
reproduced base is 191,052 B sha
`0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`, and any candidate must be
diffed against exactly those bytes. Declared reductions: SCOPE only — you may iterate on a
stratified-random subset (NEVER a prefix, per m88/m96), but the winner is re-measured at full n600.
MECHANISM reductions are TOY-BRACKET and cannot produce a family verdict: a projected size instead
of a real stat; a container change that is not parsed back by the real `inflate.sh`; a saving that
silently drops a section the receiver needs.

**Authority:** full online research, full OSS, full internal leverage — our own code, docs, and
unwired modules are off the shelf to use, adapt, refactor, extend, or fork. PR130 repro code is
off-the-shelf authorized. Cite path + commit for anything reused.

## THE BUDGET THIS SERVES (computed, not asserted)

At the PR130 base the rate term is **0.127213685 = 73.9% of S**. On this vehicle, rate IS the score.

| target | ΔS | bytes to cut | % of archive |
|---|---:|---:|---:|
| sub-0.16 | 0.012141 | **18,234 B** | 9.5% |
| sub-0.15 | 0.022141 | **33,252 B** | 17.4% |

Section marginals (exact leave-one-out, superadditivity gap −20 B so they are ADDITIVE):
tokens 116,980 B (61.23%, `ddm_ai1` live) · semantic 36,580 B (19.15%, `ddm_sm3` live) ·
pose 23,384 B (12.24%, `ddm_pk2`) · **hpac 15,092 B (7.90%, THIS ARM)** · ZIP overhead 104 B.

hpac is the smallest term, and that is exactly why it is worth one bounded arm and not more: it is
the last section with no attack on record, and closing it honestly completes the four-section sweep
so the campaign stops guessing about where the remaining bytes are.

## WHAT TO MEASURE

1. **hpac decomposition.** What are those 15,092 B, structurally? The archive stat is the fact; a
   params×bits projection is not. If the packed size does not reconcile with the structure, that
   discrepancy IS the first finding.
2. **The representation race**, each to REAL bytes in a REAL archive: re-quantization at measured
   sensitivity · structural factorization · pruning at measured-zero cost. Race, do not adopt by
   reputation (#940 — races won in BOTH directions on the same day). NOTE the coder axis is already
   CLOSED (#996, all four sections vs their own memoryless bound) and the gauge family is CLOSED
   (`113b52fdb1`: best 64 B against a 2,000 B gate). **Do not re-run either.** What is open is a
   smaller thing to code, not a better way to code it.
3. **The CONTAINER, which nobody has examined.** ZIP overhead measures 104 B on the leave-one-out,
   but that is the *marginal* figure — measure the ACTUAL container cost: local headers, central
   directory, per-member names, stored-vs-deflated choice, alignment. `upstream/evaluate.py:63`
   charges the whole file, so container bytes are real bytes. Small, certain, and unexamined beats
   large and speculative. If it is genuinely ~100 B, say so and close it.
4. **One byte-closed n600 `evaluate.py` row** for anything that survives, with exact sha256 + size.

## ALWAYS KEEP THE PAYLOAD (P0, DEF CON 1000 — operator 2026-08-09)

CLAUDE.md `## ⛔ ALWAYS KEEP THE PAYLOAD`. Persist every candidate's bytes with sha256 + byte count,
not just the winner's — `ddm_pk4` retained all 864 candidate payloads and that is the standard.
Run `tac.payload_retention_gate` on anything you write. Known detector limitation: it tests
PERSISTENCE, not reachability-to-a-writer, so a payload consumed only by `sha256()` or a
decode-verify still counts as LOST (`.omx/research/ddm_main_p0_triage_20260810.md`).

## HARD RULES

- Bulk artifacts → `/Volumes/VertigoDataTier/pact/ddm_hp3_20260810/`. No `/tmp` in evidence.
- Commits via `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256` per
  file, tags `[no-triality] [p0-ledger-ok]`, **no attribution trailer of any kind**.
- `.py` files: 2 × `tools/review_tracker.py mark-file <f> --status reviewed`; never
  `REVIEW_GATE_OVERRIDE=1` with a `.py`.
- Write ONLY under your own `ddm_hp3` output paths. Two arms sharing an output directory happened
  today (pk3/pk4) and destroyed the independence of both their results — do not repeat it.
- Do NOT touch tokens (`ddm_ai1`, holds the scorer slot) or semantic (`ddm_sm3`). If you need the
  scorer, queue and say so rather than racing.
- `upstream/` IMMUTABLE. Intake clones READ-ONLY. Every number carries its axis; a macOS run is
  `[macOS-CPU advisory]`, never `[contest-CPU]`. No Modal without operator GO.

## DELIVERABLE

The hpac decomposition · the raced candidates with REAL bytes each · the measured container cost
with its own breakdown · one byte-closed n600 row for anything that survives. If hpac is within a
few hundred bytes of its floor, say so with the measurement that shows it. Closing the last section
honestly is what lets the campaign stop looking here — and it is exactly what `ddm_pk4` did for the
gauge family.
