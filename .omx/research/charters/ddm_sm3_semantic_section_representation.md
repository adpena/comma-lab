# ddm_sm3 — The semantic section is 19.15% of the archive and nobody has attacked its REPRESENTATION.

**Owner:** codex arm · **Base:** PR130 CPR1 · scorer-free until the final n600 gate ·
`[macOS-CPU advisory]` · `score_claim=false` until a byte-closed archive is evaluated

## OPTIMAL FORM (read first)

Reference form: the semantic decoder's 66,339 parameters re-represented so the archive's semantic
section shrinks at **equal-or-better** measured d_seg, proven by a real `archive.zip` stat and a real
n600 `evaluate.py` row — not a projected byte count. Declared reductions: SCOPE only — n120
stratified-random (NEVER a prefix, per m88/m96) is legal for the fast iteration loop, and every
surviving candidate is re-run at full n600 before it is quoted. MECHANISM reductions are
TOY-BRACKET and cannot produce a family verdict: a parameter count that is never packed into real
bytes; a d_seg from a proxy instead of the frozen CPU-torch SegNet argmax; a "projected" section
size instead of the actual leave-one-out archive delta.

**Authority:** full online research, full OSS, full internal leverage — our own code, docs, and
unwired modules are off the shelf to use, adapt, refactor, extend, or fork. PR130 repro code is
off-the-shelf authorized (operator grant 2026-08-06). Cite path + commit for anything reused.

## WHY THIS SECTION, AND WHY NOW

Exact leave-one-out marginals on the reproduced 191,052 B archive:

| section | bytes | share | marginal S |
|---|---:|---:|---:|
| tokens | 116,980 | 61.23% | 0.0778922 |
| **semantic** | **36,580** | **19.15%** | **0.0243571** |
| pose | 23,384 | 12.24% | 0.0155704 |
| hpac | 15,092 | 7.90% | 0.0100491 |

Superadditivity gap is −20 B (0.0267%), so the marginals are ADDITIVE and budgetable — a byte cut
here does not have to be re-measured against the others to be real.

**Tokens are taken** (`ddm_ai1`, scorer slot). **Pose was attacked** (`ddm_pk2`). **The coder axis
is CLOSED** (#996: all 4 sections measured against their own memoryless bound). **The gauge family
is CLOSED** (`ddm_pk4`, `113b52fdb1`: best saving 64 B against a 2,000 B gate, 432 candidates, real
coders). What remains on this section is not a better coder — it is a **smaller thing to code.**

Semantic is REPRODUCED and measurable here: `quant_bits=4`, 66,339 params, through
`train_semantic_quantized.evaluate_all`, DALI-GT d_seg contribution **0.0002857038709852431** =
0.998650× the published Ada figure, 19 s per n600 on Metal. So you can iterate fast and verify
against a known-good number.

## WHAT TO MEASURE

1. **Where the 36,580 B actually sit.** Decompose the section by tensor/layer before proposing
   anything — params × bits is a projection, the archive stat is the fact. If the packed size does
   not reconcile with 66,339 × 4 bits, that discrepancy IS the first finding.
2. **The representation race**, each measured to REAL bytes in a REAL archive: per-tensor bit
   re-allocation by measured d_seg sensitivity · low-rank / factorized structure · shared codebook
   or VQ across tensors · pruning at measured-zero d_seg cost · sub-4-bit where sensitivity permits.
   Race them; do not adopt by reputation (the SMEVR/LOTTO lesson, #940 — races won in BOTH
   directions on the same day).
3. **The d_seg price of every byte saved**, through the frozen CPU-torch SegNet argmax on the real
   decode path. A cut that costs more d_seg than it saves rate is a LOSS: at this operating point
   1,000 B ≈ 0.000666 S, so the exchange rate is explicit and you can price each candidate.
4. **One byte-closed n600 `evaluate.py` row** for the winner. `upstream/evaluate.py:63` charges
   `archive.zip` and nothing else.

## ALWAYS KEEP THE PAYLOAD (P0, DEF CON 1000 — operator 2026-08-09)

CLAUDE.md `## ⛔ ALWAYS KEEP THE PAYLOAD`. Every run that materializes a payload MUST persist it;
a scalar-only artifact is FORBIDDEN, and this binds at the TYPING moment, not at review. Persist
every candidate's packed bytes with sha256 + byte count, not just the winner's — `ddm_pk4` retained
all 864 candidate payloads and that is the standard. Run `tac.payload_retention_gate` on anything
you write. Known detector limitation, do not be misled by it: it tests PERSISTENCE, not
reachability-to-a-writer, so a payload consumed only by `sha256()` or a decode-verify still counts
as LOST (`.omx/research/ddm_main_p0_triage_20260810.md`).

## PROVENANCE PINS (verify each at source; a pin that does not reproduce is a STOP)

- reproduced archive `/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/reproduction/archive.zip`
  191,052 B sha `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`
- base S = 0.172141297491896447 `[contest-CUDA, DALI GT, n600]` — the bar, not our vehicle
- gauge-family closure `113b52fdb1` + `.omx/research/ddm_pk3_rate_aware_gauge_preflight_20260809/`
- `upstream/` is IMMUTABLE. The intake clone at
  `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/` is READ-ONLY: never edit, never
  `git add` inside. Copy out to work.

## HARD RULES

- Bulk artifacts → `/Volumes/VertigoDataTier/pact/ddm_sm3_20260810/`. No `/tmp` in evidence.
- Commits via `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256` per
  file, tags `[no-triality] [p0-ledger-ok]`, **no attribution trailer of any kind**.
- `.py` files: 2 × `tools/review_tracker.py mark-file <f> --status reviewed`; never
  `REVIEW_GATE_OVERRIDE=1` with a `.py`.
- Do NOT touch the token sections — `ddm_ai1` owns them and holds the scorer slot. If your work
  needs the scorer, queue it and say so rather than racing.
- Every number carries its axis. A macOS run is `[macOS-CPU advisory]`, never `[contest-CPU]`.
  No Modal without operator GO.

## DELIVERABLE

The section decomposition · the raced candidates with REAL packed bytes and measured d_seg each ·
one byte-closed n600 row for the winner with its exact archive sha256 and size. If the section
turns out to be within a few hundred bytes of its own representational floor, say so plainly with
the measurement that shows it — closing 19.15% of the archive honestly is worth as much as a win,
and it is what `ddm_pk4` just did for the gauge family.
