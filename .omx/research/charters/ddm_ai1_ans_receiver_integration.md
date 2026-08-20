# ddm_ai1 — The −2,416 B is REAL and NOT SCOREABLE. Wire ANS into the shipped receiver and close that gap.

**Owner:** codex arm · **Base:** PR130 CPR1 · owns the scorer slot · `[macOS-CPU advisory]` ·
`score_claim=false` until an evaluator-runnable archive exists

## OPTIMAL FORM (read first)

Reference form: the ANS token coder wired into the SHIPPED `inflate.py` decode path, such that the
archive is evaluator-runnable end-to-end — `inflate.sh` produces bit-identical frames to the range-coded
control, and `upstream/evaluate.py` scores it. Declared reductions: SCOPE only — n120 stratified-random
is legal for the fast iteration loop, and the winner is re-run at full n600 before any row is quoted.
MECHANISM reductions are TOY-BRACKET and cannot produce a verdict: a decode that reproduces tokens in a
research harness but not through the real `inflate.sh`; a "projected" byte count instead of the actual
`archive.zip` stat; a determinism claim from one run.

**Authority:** full online research, full OSS, full internal leverage — our own code, docs, and
unwired modules are off the shelf to use, adapt, refactor, extend, or fork. PR130 repro code is
off-the-shelf authorized. Cite path + commit for anything reused.

## THE MEASURED WIN, AND EXACTLY WHY IT IS BLOCKED

`ddm_tm1` (`measurement_complete: true`) measured, on the reproduced PR130 archive:

| candidate | archive bytes | Δ vs shipped range |
|---|---:|---:|
| shipped range coder | 191,052 | — |
| ANS control | 188,932 | **−2,120 B** |
| + `temporal_reversion` sidecar | 188,636 | **−2,416 B** |

`temporal_reversion` carries `projected_rate_score_delta: -0.001608715230743166`, real token bytes
114,528, a 39 B packed sidecar (codec `raw`, `verified_exact: true`), and a +36 B model-bundle delta.
Decode verified: 117,964,800 tokens, `exact_target_equality: true`, `ans_terminal_state_empty: true`,
4.16 s decode wall.

**The blocker, verbatim from tm1's own result:** `evaluator_runnable: false`,
`receiver_complete: false`, `evaluator_blocker: "research ANS control is not wired into the shipped
inflate.py"`. So the bytes are real and the score is not claimable. **That gap is this arm's whole job.**

Tokens are **61.23%** of the archive (116,980 of 191,052 B, exact leave-one-out) — this is THE rate axis.
The coder axis was otherwise measured shut: ANS sits +0.0071% over the model's own cross-entropy, so
~8 B of slack remain in 116,980. There is no second bite here. Realize this one.

## WHAT TO BUILD

1. **The decode path.** The shipped receiver uses `constriction.stream.queue.RangeDecoder` (QUEUE
   semantics, forward). ANS is `constriction.stream.stack.AnsCoder` (STACK semantics — `encode_reverse`,
   then decode forward). Same library, already a declared dependency, so **rule-118 is clean and no new
   dep is introduced** — verify that claim at source before relying on it. Wire the ANS branch into the
   real `inflate.py` with a fail-closed selector, and keep the range path intact and byte-identical when
   the ANS field is absent (legacy archives must still decode).
2. **Byte-close.** Build the real `archive.zip`, stat it (`upstream/evaluate.py:63` charges that file
   and nothing else), and record sha256 + bytes.
3. **Prove determinism.** Decode twice; frames bit-identical. tm1 retained `archive.repeat.zip` per
   candidate — reuse that protocol, do not invent one.
4. **Score it.** `upstream/evaluate.py`, n600. Report S with every component and its axis label.
5. **Then, and only then**, adjudicate `temporal_reversion` on top (the extra −296 B vs the ANS control).

## ALWAYS KEEP THE PAYLOAD (P0, DEF CON 1000 — operator 2026-08-09)

CLAUDE.md `## ⛔ ALWAYS KEEP THE PAYLOAD`. Every run that materializes a payload MUST persist it;
a scalar-only artifact is FORBIDDEN and this rule binds at the typing moment, not at review.
`tac.payload_retention_gate` (commit `77160d7418`) is the detector — run it on anything you write.
This arm exists *because* that rule was broken once already: the original ANS run measured both coder
payloads and wrote only their lengths, costing two full re-encodes and delaying this very win.

## PROVENANCE PINS (verify each at source; a pin that does not reproduce is a STOP)

- reproduced archive `/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/reproduction/archive.zip`
  191,052 B sha `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`
- retained ANS payload `/Volumes/VertigoDataTier/pact/ddm_dt1_20260809/retained/ans_n600.bin`
  114,860 B sha `a0b18dc0803ef541d3eb265bba5380f7aa067593f6af584b0891ded5bdd74488`
- retained range payload `/Volumes/VertigoDataTier/pact/ddm_ap1_20260809/retained/range_n600.bin` 116,980 B
- tm1 result `/Volumes/VertigoDataTier/pact/ddm_tm1_20260809/measurement_v2/tm1_result.json`;
  per-candidate payloads under `.../measurement_v2/candidates/<name>/`
- ANS-control archive sha `447d7697f60b86e2d5e26a70f48f497dd852ce19ef2fd78f2901d952a8535b42`, 188,932 B
- the landed tool `experiments/ddm_tm1_token_model_lever.py`
- `upstream/` is IMMUTABLE. The intake clone at
  `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/` is READ-ONLY: never edit, never
  `git add` inside. Copy out to work.

## HARD RULES

- Bulk artifacts → `/Volumes/VertigoDataTier/pact/ddm_ai1_20260809/`. No `/tmp` in evidence.
- Commits via `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256` per file,
  tags `[no-triality] [p0-ledger-ok]`, **no attribution trailer of any kind**.
- `.py` files: 2 × `tools/review_tracker.py mark-file <f> --status reviewed`; never
  `REVIEW_GATE_OVERRIDE=1` with a `.py`.
- Every number carries its axis. A macOS run is `[macOS-CPU advisory]`, never `[contest-CPU]`.
  No Modal without operator GO.

## DELIVERABLE

An evaluator-runnable archive, its exact byte count and sha256, a determinism receipt, and the n600
`evaluate.py` row with components. If the wiring reveals that ANS cannot be made evaluator-runnable
inside the 30-min decode budget or under deterministic decode, say so plainly with the measured reason —
that is a real finding and it retires a −2,416 B mirage before it wastes anyone else's time.
