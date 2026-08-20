# ddm_sg2 — The seg axis on the PR130 base: 16.6% of S, zero arms, and every instrument we own points at a retired vehicle.

**Owner:** codex arm · **Base:** PR130 CPR1 · **scorer-free until the final gate** (`ddm_ai1` holds
the slot) · `[macOS-CPU advisory]` · `score_claim=false`

## OPTIMAL FORM (read first)

Reference form: the PR130 seg term (0.028609) decomposed on the SHIPPING vehicle — reference pin,
verify at source: base archive 191,052 B, commit `113b52fdb1`, sha256 prefix `0491d5df84fc70b6`
(full pin in PROVENANCE below) — with each candidate lever priced in S units against a MEASURED
per-class/per-edge attribution, not a transferred one. Declared reductions: SCOPE only — n120
seeded stratified-random is legal for iteration (NEVER a prefix, per m88/m96, and note seg prefix
bias is ≈0.96× so it is the *mild* axis), winner re-run at n600. MECHANISM reductions are
TOY-BRACKET: a d_seg number from a proxy instead of the frozen CPU-torch SegNet argmax; a lever
priced on the TR1/burn vehicle and asserted to transfer; a per-CLASS table used where the structure
is per-EDGE.

**Authority:** full online research, full OSS, full internal leverage — our own code, docs, and
unwired modules are off the shelf to use, adapt, refactor, extend, or fork. PR130 repro code is
off-the-shelf authorized. Cite path + commit for anything reused.

## WHY THIS ARM EXISTS

Three-axis decomposition of the bar (computed, not asserted):

| axis | value | share of S | cut needed ALONE for sub-0.15 |
|---|---:|---:|---|
| rate | 0.127214 | 73.9% | 17.4% |
| **seg** | **0.028609** | **16.6%** | 77.4% |
| pose | ~0.0152 | 8.8% | 145.7% — impossible alone |

Rate is correctly where the fleet is (`ai1` tokens · `cp2` semantic+composition · `hp3` hpac).
We hold **−8,688 B byte-closed = 26.1%** of the rate-only target; if it survives distortion we land
≈0.166356 and still owe **0.016356**. Buying that last piece purely on rate costs another 24,564 B
on sections already under attack. A MIX is cheaper — and the seg axis has **zero** arms.

**The blocking fact this arm must fix first (#917):** our entire seg lever inventory was built and
priced on the RETIRED TR1/burn vehicle. Those numbers do not transfer, and the standing law is that
citing them as if they do is the cross-regime constant-transfer genus. Assume nothing carries.

Named entry point, MEASURED but UNVERIFIED-BY-YOU: `#906` reports chroma siting sensitivity
2.2791e-4 = **79.66% of PR130's ENTIRE seg term** (commits `afa34a0860`, `38e08900c3`).
**Re-derive it at source before building on it.** If it does not reproduce, that is your first
finding and it outranks everything else in this charter.

## WHAT TO MEASURE

1. **Decompose PR130's 0.028609 on the SHIPPING vehicle.** Per-EDGE, not per-class — the standing
   measured law (`ddm_pc2`) is that Road participates in 87.8% of all flips and Road↔Lane alone is
   49.2%; a per-class table splits one separatrix across two rows and hides the structure. Report
   the edge table with its own denominator.
2. **Verify #906's chroma claim at source.** Reproduce the 2.2791e-4 sensitivity, or report the
   number you actually get. SegNet reads RGB, so chroma is a genuine argmax actuator — but the
   claim's magnitude (79.66% of the whole term) is extraordinary and deserves its own control.
3. **Price the top-2 levers in S units.** For each: what does it cost in bytes, what does it buy in
   d_seg, and what is the exchange rate against the rate axis (at this base, 1,000 B ≈ 0.000666 S,
   so a lever must beat that to be worth its own bytes). A seg lever that costs more rate than it
   saves seg is a LOSS — say so.
4. **One byte-closed n600 row** for anything that survives, queued for MAIN to fire (do NOT race
   `ai1` for the scorer). Emit the queue in `ddm_sm3`'s existing `SM3_SCORER_QUEUE.json` schema.

## ALWAYS KEEP THE PAYLOAD (P0, DEF CON 1000 — operator 2026-08-09)

CLAUDE.md `## ⛔ ALWAYS KEEP THE PAYLOAD`. Persist every candidate's bytes with sha256 + count, not
just the winner's — `ddm_pk4` retained 864 payloads, `ddm_sm3` retained 16 archive/repeat ZIPs.
Run `tac.payload_retention_gate` on anything you write. Known limitation: it tests PERSISTENCE, not
reachability-to-a-writer, so a payload consumed only by `sha256()` or a parse-back still counts as
LOST (`.omx/research/ddm_main_p0_triage_20260810.md`).

## PROVENANCE PINS (verify each at source; a pin that does not reproduce is a STOP)

- base archive `/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/reproduction/archive.zip`
  191,052 B sha256 `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`
- base S = 0.172141297491896447 `[contest-CUDA, DALI GT, n600]` — the bar, not our vehicle
- semantic leg reproduced here: DALI-GT seg contribution 0.0002857038709852431 = 0.998650× published
- pose axis CLOSED by `ddm_pk2`: baseline WON; basis symbols are ALREADY signed int5 (−15…15)
- `upstream/` IMMUTABLE; intake clone `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/`
  READ-ONLY — never edit, never `git add` inside

## HARD RULES

- Bulk artifacts → `/Volumes/VertigoDataTier/pact/ddm_sg2_20260810/`. No `/tmp` in evidence.
- Commits via `tools/subagent_commit_serializer.py`, POST-EDIT `--expected-content-sha256` per file,
  tags `[no-triality] [p0-ledger-ok]`, **no attribution trailer of any kind**.
- `.py`: 2 × `tools/review_tracker.py mark-file <f> --status reviewed`; never `REVIEW_GATE_OVERRIDE=1`
  with a `.py`.
- Write ONLY under your own `ddm_sg2` paths — two arms sharing an output dir happened today
  (pk3/pk4) and destroyed the independence of both results.
- Sister arms' landed artifacts are APPEND-ONLY inputs. Do not touch tokens (`ai1`), semantic
  (`cp2`), or hpac (`hp3`).
- Every number carries its axis. macOS = `[macOS-CPU advisory]`, never `[contest-CPU]`. No Modal
  without operator GO — if a lever's verification genuinely requires the DALI-vs-AV job, NAME it as
  a blocker and stop rather than substituting a proxy.

## DELIVERABLE

The per-EDGE decomposition of 0.028609 with its denominator · #906 verified or corrected at source ·
the top-2 levers priced in S units against the 1,000 B ≈ 0.000666 S exchange rate · a scorer queue
for MAIN. If the seg axis on this base turns out to be within noise of its own floor, say so with
the measurement — closing 16.6% of S honestly is worth as much as a win, and it tells the campaign
to put everything back on rate.
