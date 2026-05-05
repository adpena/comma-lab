---
name: Grand Council adversarial review — PCC3 stats.json internal-consistency STRICT preflight check
description: 2026-04-30 ~23:30 UTC. Inner council 10-member deliberation on PCC3 (stats.json internal-consistency check). Three design decisions: (DD1) per-script vs global MIN_SEC_PER_EPOCH constant, (DD2) producer-side assertion vs preflight-only, (DD3) --smoke handling. Live audit found 2 violations (train_segmap.py, train_segmap_film_canvas.py); train_imp_cycle.py was the originating bug fixture (200 epochs in 3.5s impossible) and ships with the producer-side assertion already.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## User mandate (2026-04-30 ~22:55 UTC)

> "permanently fix all bugs and bug classes and metabugs and everything and have all design decisions and ultimate experiment subject to extreme paranoia and adversarial grand council reviews"

The IMP cycle 0 = 1.98 KILL was a measurement bug: stats.json claimed `{"epochs": 200, "elapsed_sec": 3.47}` — physically impossible (200 epochs of fine-tune in 3.5 seconds). The producer-side assertion landed in `experiments/train_imp_cycle.py` (commit pending). PCC3 enforces the pattern system-wide: every stats.json producer must carry an internal-consistency assertion, and the preflight scanner walks the codebase to find new offenders.

## Live audit

Scanner finds `json.dump`/`json.dumps` calls whose dict literal contains both an EPOCH-like key (`epochs`, `steps`, `iterations`, `n_epochs`, `n_steps`, `num_epochs`, `num_steps`, `epoch_count`) AND an ELAPSED-like key (`elapsed_sec`, `elapsed`, `wall_time`, `wall_seconds`, `total_seconds`, `duration_sec`, `duration_s`, `elapsed_s`, `elapsed_seconds`, `wall_clock_sec`). Live count after `experiments/train_imp_cycle.py` fix:

| File | Line | Function | Status |
|---|---|---|---|
| `experiments/train_imp_cycle.py` | 270 (`_save_state` dispatched from main) | `_save_state` writes `meta`; assertion lives in `main()` at line 366 BEFORE the save call (line 394) | PASSES (assertion above caller) |
| `experiments/train_segmap.py` | 456 | `main()` writes `summary` dict; NO assertion comparing `elapsed` to `args.epochs` | VIOLATION → fix in this batch |
| `experiments/train_segmap_film_canvas.py` | 337 | `main()` writes `summary` dict; NO assertion | VIOLATION → fix in this batch |

Two real producer scripts; both fixed in the same PCC3 landing wave.

## DD1: per-script vs global MIN_SEC_PER_EPOCH constant

### Options

**Option A** — single global `MIN_SEC_PER_EPOCH = 0.05` in `tac.training_floors` consumed by every producer
- Pros: one source of truth; uniform enforcement
- Cons: lane-specific workloads vary 100×: SegMap on T4 (~2 s/epoch real), distill on L40S (~5 s/epoch real), tiny IMP fine-tune on CUDA (~0.05 s/epoch is the conservative floor). A single global constant either fails noisy lanes or under-detects fast lanes.

**Option B** — per-script constant chosen by the script author who knows the workload
- Pros: matches the actual physics of each lane; assertion catches stub-loop only when it would be 10–100× off the true floor
- Cons: discipline burden; risk of someone setting a too-loose floor that misses a stub bug

**Option C** — per-profile constant (one per `tac.profiles` entry)
- Pros: structured; profile-aware
- Cons: profiles describe MODELS not training durations; same model can train at very different per-epoch wall times depending on dataset/device/batch

### Council vote (10 inner members)

| Member | Vote | Rationale |
|---|---|---|
| Shannon (LEAD) | B | Information theory: each script encodes a different rate-distortion regime; one global is information-destroying |
| Dykstra (CO-LEAD) | B | Per-stage feasibility set; per-script floor matches per-stage convexity |
| Yousfi | B | Detection-error tradeoff is per-workload — loose global = false negatives |
| Fridrich | A → B | First instinct global; Shannon's argument convincing — switch to B with mandatory comment-justifying-the-number |
| Contrarian | B | Global is exactly the silent-default-trap CLAUDE.md forbids |
| Quantizr | B | Quantizr's own pipeline has 5 stages with 5 different per-step costs; one global is wrong |
| Hotz | B | "Pick the number you'd defend in a code review"; per-script forces that |
| Selfcomp | B | Selfcomp's training has anchor → finetune → joint → QAT → final, each a different per-step cost |
| MacKay | B | MDL: the floor is a model parameter, not a system constant |
| Ballé | B | Each codec stage has its own R(D) floor; global is incoherent |

**Result: 10/10 Option B (per-script constant with mandatory justifying comment).**

The number must come with a comment naming the device/lane class and citing what a "stub loop" would look like (e.g. "0.05 s/epoch is below even toy training on CPU; stub loops on CUDA come in at ~0.017s/epoch").

## DD2: producer-side assertion vs preflight-only

### Options

**Option A** — Preflight-only: scan codebase, fail at preflight if no assertion present, but trust producers to never have stub loops
- Pros: less producer-side code
- Cons: preflight runs at COMMIT time; the bug fires at RUN time. Preflight cannot catch a stub loop in already-committed code; it only catches the absence of the assertion structure.

**Option B** — Producer-side assertion (raise RuntimeError at runtime if elapsed < epochs * floor) + preflight enforces presence
- Pros: defense in depth; the assertion fires at training time and FAILS LOUD before stats.json is written; preflight ensures every new producer adds the assertion
- Cons: every producer carries ~10 lines of assertion + comment

**Option C** — Producer-side only (no preflight)
- Pros: simplest
- Cons: no enforcement that NEW producers add the assertion; bug class re-emerges every time a new training script lands

### Council vote (10 inner members)

| Member | Vote | Rationale |
|---|---|---|
| Shannon | B | Defense in depth; preflight catches structural absence, runtime assertion catches dynamic violations |
| Dykstra | B | Both gates close the feasibility set from different sides |
| Yousfi | B | Steganalysis: never trust one detector; ensemble |
| Fridrich | B | Adversary thinks of all the ways to bypass; we need both gates |
| Contrarian | B | Preflight-only is too easy to bypass with a comment trick; runtime is unfakeable |
| Quantizr | B | Same pattern as EMA: producer-side wired AND preflight enforces (Check 88) |
| Hotz | B | "Belt + suspenders" |
| Selfcomp | B | Matches the existing EMA + auth-eval pattern (producer + preflight) |
| MacKay | B | Bayesian: posterior over bug existence is much lower with both detectors |
| Ballé | B | Same as Shannon — defense in depth is the canonical rate-distortion-with-side-information pattern |

**Result: 10/10 Option B (producer-side assertion AND preflight enforcement).**

This matches the existing pattern for EMA (Check 88), auth-eval-on-best (Check 7), and eval_roundtrip (Check 5): wire the producer correctly AND preflight enforces the wiring.

## DD3: --smoke handling

### Options

**Option A** — Assertion always fires; smoke tests cannot use `args.smoke=True` to skip work
- Pros: no escape hatch ever
- Cons: legitimate smoke tests (CI, dev-loop) deliberately skip the expensive work; assertion would force them to fake elapsed_sec

**Option B** — Assertion gated on `not args.smoke and args.epochs > 0`
- Pros: matches user intent (smoke = "I am deliberately skipping work, don't enforce wall-clock"); keeps real-run safety; matches the train_imp_cycle.py implementation already landed
- Cons: a buggy script could set args.smoke=True in production by accident; need a separate guard for "smoke flag in dispatch script"

**Option C** — Assertion always fires; smoke mode must be a separate script
- Pros: cleanest separation
- Cons: large refactor; doubles the script count; CI overhead

### Council vote (10 inner members)

| Member | Vote | Rationale |
|---|---|---|
| Shannon | B | Information theory: --smoke is metadata declaring the run is not a real measurement; assertion respects that |
| Dykstra | B | Smoke is a different feasibility set; gating is correct |
| Yousfi | B | Steganalysis distinguishes test-set from operational-set; same here |
| Fridrich | C → B | Wanted hard separation; persuaded by Hotz's "we don't want to double the script count" |
| Contrarian | B + audit | OK with B but PCC3 must ALSO scan dispatch scripts for `--smoke` in production wave; "smoke flag in remote_lane_*.sh" should be its own future check |
| Quantizr | B | Quantizr ships smoke tests in the same script; gating is the standard pattern |
| Hotz | B | "Don't double the surface area" |
| Selfcomp | B | All Selfcomp scripts use `--smoke` gated assertions; this is the established convention |
| MacKay | B | The smoke/real binary IS a model parameter; conditioning on it is principled |
| Ballé | B | Same as Shannon |

**Result: 10/10 Option B (gate on `not args.smoke and args.epochs > 0`).**

Contrarian's audit-future-check note is captured as TODO for the dispatch-script scanner: any `remote_lane_*.sh` invoking a training script with `--smoke` in a production wave is a separate bug class to tag for a future strict check.

## Implementation summary

1. **Producer-side**: each violating script (`experiments/train_segmap.py`, `experiments/train_segmap_film_canvas.py`) gains an assertion of the form:

   ```python
   MIN_WALL_PER_EPOCH_SEC = <per-script-justified-constant>  # see council vote
   if not args.smoke and args.epochs > 0:
       expected_min = args.epochs * MIN_WALL_PER_EPOCH_SEC
       if elapsed < expected_min:
           raise RuntimeError(
               f"PCC3 STUB-LOOP DETECTED: claimed {args.epochs} epochs in "
               f"{elapsed:.2f}s — below floor {expected_min:.2f}s ..."
           )
   ```
   placed BEFORE the `json.dump` / `json.dumps` of `summary` / `meta`.

2. **Preflight check**: `check_stats_json_internal_consistency(strict=True, verbose=False)` in `src/tac/preflight.py`:
   - AST scans `scripts/`, `experiments/`, `src/tac/`, `submissions/robust_current/` for `*.py`
   - Walks each function looking for `json.dump`/`json.dumps` whose first arg is (or refers to) a dict containing both an EPOCH-like and an ELAPSED-like key
   - Walks BACKWARDS from the dump line within the same function for an `assert` / `if … raise` that compares an elapsed-like name to an epochs-like name with `Mult` or `Div` op
   - Inter-function flow handled via the `# PCC3-WAIVED-INTERFUNCTION: <reason>` waiver marker (for the train_imp_cycle.py case where `_save_state(meta)` is called from `main()` which holds the assertion)
   - Whitelist: `# PCC3-WAIVED: <reason>` same-line waiver for legitimate cases where elapsed_sec=0 is intentional
   - Strict mode raises `MetaBugViolation` listing each `file:line:dict-snippet`

3. **Wired into `preflight_all()`** at `strict=True` initially after the 2 producer fixes land. Live count after fix: 0 violations.

## Internal-consistency check

- The new check passes on `experiments/train_imp_cycle.py` (assertion lives in main() at line 366; the `# PCC3-WAIVED-INTERFUNCTION` waiver covers the helper call site).
- The new check passes on `experiments/train_segmap.py` and `experiments/train_segmap_film_canvas.py` AFTER the assertion is added.
- The new check rejects a synthetic test fixture mimicking the IMP bug (200 epochs in 3.5 s with no assertion).

## Reactivation criteria

- If a new training script lands a stats.json producer without the assertion, PCC3 STRICT FAILS at preflight commit time.
- If a producer's assertion is too LOOSE (e.g. `MIN_WALL_PER_EPOCH_SEC = 0.0001` to bypass), the runtime stub-loop bug remains — this is a follow-up audit (PCC3-extended) that the council leaves for a future round once the absolute-floor pattern proves itself.
- If `--smoke` mode is used in production, that is a separate bug class for a future scanner (`check_no_smoke_in_production_wave`).

## Cross-references

- `feedback_grand_council_imp_permanent_fix_review_20260430.md` (the originating IMP-permanent-fix design)
- `experiments/train_imp_cycle.py` (the reference implementation of the producer-side assertion)
- CLAUDE.md "FORBIDDEN PATTERNS" (the bug class catalog)
- CLAUDE.md "Meta-bug class catalog" (where PCC3 will be cataloged once landed STRICT)
