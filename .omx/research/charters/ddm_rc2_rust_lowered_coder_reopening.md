# ddm_rc2 — Two coder families were excluded on PURE-PYTHON wall-clock, unmeasured. Race them lowered into Rust.

**Owner:** codex arm · **Base:** PR130 CPR1 · scorer-free (coder races are byte measurements) ·
`[macOS-CPU advisory]` · `score_claim=false`

## OPTIMAL FORM (read first)

Reference form: per-symbol adaptive arithmetic coding and iterative BP/LDPC-syndrome decode, each at
its family's REFERENCE form, measured to REAL bytes on the REAL PR130 payloads, with decode cost
measured in a NATIVE lowering — not pure Python. Reference pin, verify at source: base archive
191,052 B sha `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`. Declared
reductions: SCOPE only (fewer payloads, fewer variants). MECHANISM reductions are TOY-BRACKET: a
pure-Python prototype used to produce a FAMILY verdict; a projected byte count instead of a real
coded stream; a decode-time number used as an admissibility gate (FORBIDDEN, see below).

**Authority:** full online research, full OSS, full internal leverage — our own code, docs, and
unwired modules are off the shelf to use, adapt, refactor, extend, or fork. PR130 repro code is
off-the-shelf authorized. Cite path + commit.

## WHY — an exclusion made on the slowest possible implementation

`.omx/research/ddm_iv4_missing_piece_hunt_20260728.md` row **A14**, verbatim:

> "Per-symbol adaptive-arithmetic in PURE PYTHON is the classic 30-min killer; iterative BP over 1M
> sites likewise. A byte-winning coder that busts 1800 s is INADMISSIBLE"

Severity FATAL(conditional); the row's own text says the cost is **"unmeasured."** So two coder
families were excluded as a CLASS, on a PROJECTED wall-clock cost, measured against pure Python.

**Both premises are now void:**
1. **Operator binding 2026-08-09:** *"Don't disqualify anything decoder due to time unless I give
   authorization."* Decode wall-clock is NOT an admissibility criterion. The score has no time term
   (`upstream/evaluate.py:92`); `timeout-minutes: 30` bounds the WHOLE CI job. This is now in
   `tac.subagent_contract` (commit `e2700086f2`) — read it.
2. **Operator 2026-08-09:** *"we can also lower into rust."* We have a BUILT, PROVEN native path:
   13 crates in `runtime-rs/` (incl. `tac-levelset-inflate`, `tac-boundary-decode`, `qma-codec`,
   `residual-codec`, `tac-packet-compiler`), golden-vector suites, and end-to-end parity harnesses
   (`lane_end_to_end_parity.py`, `generate_golden_vectors.py`). Task #214 ALREADY lowered a decode
   to fit the budget once. CLAUDE.md's native-runtime discipline exists for exactly this: native
   code is ALLOWED when it "expands the legal witness-program class."

**Consequence you must state plainly in your report:** task #996 records "CODER AXIS CLOSED on the
PR130 base." If these families were pre-filtered by A14, that closure was taken over a RESTRICTED
candidate set and its scope needs correcting. Determine whether they were in #996's race or not —
at source, from #996's own receipts — and say which.

## WHAT TO MEASURE

1. **Were they actually raced?** Read #996's coder-race receipts. If adaptive-arithmetic / BP were
   in the candidate set and LOST ON BYTES, this arm is CLOSED-AS-STALE and that is a fine answer —
   report it and stop. Only proceed if they were absent or excluded on time.
2. **Race them on bytes, at reference form.** Real coded streams on the real PR130 sections
   (tokens 116,980 B · semantic 40,252 B raw · pose 23,054 B · hpac 20,179 B). Adaptive-arithmetic
   with a real context model; BP/LDPC-syndrome on the flip/residual structure where it applies.
   Compare against the incumbents and against each section's own memoryless bound.
3. **Measure decode cost as a FACT, in the native lowering.** If a family wins bytes, lower the
   decode into `runtime-rs/` (reuse the existing crates + golden-vector parity pattern — do NOT
   invent a new parity scheme) and report seconds. **Report it; never gate on it.** A slow decode
   is a Rust-lowering task, not a verdict.
4. **Bit-identity is the REAL gate.** Native lowering is admissible only with a Python reference
   oracle and byte-identical parity, per CLAUDE.md's native-runtime discipline. That is where your
   rigor goes — not into wall-clock.

## ALWAYS KEEP THE PAYLOAD (P0, DEF CON 1000 — operator 2026-08-09 ×2)

CLAUDE.md `## ⛔ ALWAYS KEEP THE PAYLOAD` + operator this session: *"keep all archives and outputs
and everything."* Persist EVERY coded stream from EVERY variant with sha256 + byte count — not just
the winner. A coder race is exactly the measure-and-discard shape that the P0 exists to kill (the
anchor incident was a coder measurement whose bytes were dropped). Run
`tac.payload_retention_gate` on anything you write. Known detector limit: it tests PERSISTENCE, not
reachability-to-a-writer, so bytes consumed only by `len()` or `sha256()` still count as LOST
(`.omx/research/ddm_main_p0_triage_20260810.md`).

## HARD RULES

- Bulk → `/Volumes/APDataStore/pact/ddm_rc2_20260810/` (tier-2, ~997 GiB free; tier-1
  VertigoDataTier is 98% full — do NOT write bulk there). No `/tmp` in evidence.
- Commits via `tools/subagent_commit_serializer.py`, POST-EDIT `--expected-content-sha256` per file,
  tags `[no-triality] [p0-ledger-ok]`, **no attribution trailer of any kind**.
- `.py`: 2 × `tools/review_tracker.py mark-file <f> --status reviewed`; never `REVIEW_GATE_OVERRIDE=1`
  with a `.py`. Rust: keep the golden-vector + parity contract.
- Write ONLY under `ddm_rc2` paths. Sister arms' artifacts are APPEND-ONLY inputs.
- `upstream/` IMMUTABLE. Intake clones READ-ONLY. No Modal without operator GO. Every number
  carries its axis.
- **rule-118 binds:** the coder ALGORITHM lowered into inflate is FREE; any video-derived table or
  learned model it needs is COUNTED in archive.zip. State which side each artifact lands on.

## DELIVERABLE

The #996-scope determination (were they raced, or time-filtered?) · real coded bytes per family per
section vs incumbent and vs the memoryless bound · decode seconds reported as a FACT with its
implementation named (Python reference vs Rust) · a Rust-lowering plan for anything that wins bytes,
reusing the existing crates. If both families lose on BYTES at reference form, that closes them
honestly and finally — which is worth as much as a win, because it converts a time-excluded class
into a byte-measured one.

## RESPAWN NOTE (MAIN, 2026-08-10) — your blocker was OURS, and it is cured

You stopped correctly at the contract's SSD condition: `/Volumes/APDataStore`
returned `Operation not permitted`. That was NOT capacity (997 GiB free) and NOT
your error — the arm keeper granted `--add-dir` for tier-1 ONLY, so every arm
routed to tier-2 inherited a sandbox forbidding its own output directory. You are
the third arm it killed today (with `ddm_sd2`, `ddm_vh2`). Fixed at the spawn site
in `tools/codex_arm_queue.py`, commit `002208ddf6`, both-direction controls
executed. Write to APDataStore normally now; the stop condition stands unchanged.
