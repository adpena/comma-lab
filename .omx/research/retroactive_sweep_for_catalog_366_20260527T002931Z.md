<!-- HISTORICAL_PROVENANCE: APPEND-ONLY per Catalog #110/#113. Retroactive sweep per Catalog #348 for newly-landed Catalog #366 (inflate shim import drift). -->
<!-- HISTORICAL_SCORE_LITERAL_OK: this memo cites historical Cascade C' fc-call_id anchors; no NEW score literal claims. -->

# Retroactive sweep for Catalog #366 — inflate shim import drift

**Date:** 2026-05-27T00:29:31Z
**Per:** CLAUDE.md "Operator gates must be wired and used" non-negotiable + Catalog #348 "EVENT-DRIVEN RETROACTIVE VERDICT-TAINT SWEEP" self-protection.

## 1. Bug-class symptom signature

A trainer wrapper's `_write_runtime` emits an inflate.py shim with:
```python
from tac.substrates.<X>.inflate import <name>
```
where `<name>` is NOT actually exported by the canonical inflate module at `src/tac/substrates/<X>/inflate.py`. The shim crashes at the first import attempt with:
```
ImportError: cannot import name '<name>' from 'tac.substrates.<X>.inflate'
```

The canonical pattern uses `from ... import main_cli as main` when the canonical module exports `main_cli` (per Catalog #146 contest 3-arg contract); legacy shims importing bare `main` are drifted.

## 2. Pre-fix window

The bug-class drift was empirically demonstrated **once** in the recent session: Cascade C' Wave 2 inflate shim raised `ImportError: cannot import name 'main' from 'tac.substrates.cascade_c_prime_frame_1_segnet_waterfill.inflate'` because the trainer wrapper's emitted shim imported `main` but the canonical module exports `main_cli`. Fix landed commit `3c2ce7fc2` (2026-05-26 18:30:44) used `from ... import main_cli as main`.

## 3. Historical-KILL/DEFER/FALSIFY search results

Searched repo for all `from tac.substrates.<X>.inflate import <name>` patterns in string-literal Constants across `experiments/train_substrate_*.py` + `experiments/train_renderer*.py` (93 files total). Inspected exports of every target inflate module.

Live count BEFORE Catalog #366 landing: **0**.

**Cross-checked all 28 currently-emitted shims:** 26 import `inflate_one_video` (canonical for SCAFFOLD-style trainers per HNeRV parity L4); 2 import `main_cli` (canonical for trainers using the helper's CLI entry); 0 import `main` or other non-canonical names.

No historical KILL / DEFER / FALSIFY memos cite the inflate shim import drift bug class. This is a NEW bug class surfaced 2026-05-26 specifically by the Cascade C' substrate scaffold. The 18-trainer Catalog #226 + Catalog #361 OVERNIGHT-GG submission-dir-preserve waves did not introduce this drift because each trainer's `_write_runtime` was either:
1. Emitting `inflate_one_video` (matching the canonical SCAFFOLD module export), OR
2. Emitting `main_cli as main` (matching the canonical CLI helper export pattern).

The Cascade C' scaffold landed with `main` because the operator was working from a slightly stale local memory of which canonical name was current (`main` was historical-canonical for some HNeRV-family trainers pre-Catalog-146).

No historical verdicts require RE-EVAL because:
1. The bug class is structural (ImportError fires at first invocation, NOT a silent semantic regression).
2. Every previously-dispatched substrate trainer used a matching import name (empirical proof: their inflate.py was successfully imported).
3. The bug-class extinction lives at Catalog #361 (Modal artifact harvester preserves submission-dir bytes) + Catalog #146 (contest-compliant inflate template) + Catalog #295 (PYTHONPATH self-containment) + Catalog #205 (canonical device selector) + Catalog #366 (import-NAME matches export); the 5-surface coverage is now complete.

## 4. Per-finding RE-EVAL-priority assignment

| Historical Finding | RE-EVAL Priority | Rationale |
|---|---|---|
| N/A — no historical kill/defer/falsify verdicts apply | N/A | This bug class is NEW (2026-05-26) and has no historical precedent. The Cascade C' Wave 2 empirical anchor is the only known incident and was fixed in the same commit batch. |

## 5. Cross-references

- CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable
- CLAUDE.md "Operator gates must be wired and used" non-negotiable
- Catalog #361 sister gate (`check_modal_artifact_filter_preserves_submission_dir` — same META class at output/submission preservation surface)
- Catalog #146 (contest-compliant inflate template)
- Catalog #295 (PYTHONPATH self-containment for submissions/*/inflate.py)
- Catalog #205 (canonical select_inflate_device)
- Catalog #348 retroactive verdict-taint sweep discipline
- Catalog #287 placeholder-rationale rejection
- Cascade C' Wave 2 verdict commit `67280b4c9` + inline fix `3c2ce7fc2`

## 6. Discipline declarations

- Catalog #229 PV: full git log inspection + AST audit of all 93 trainer files emitted shim patterns + cross-checked against all target inflate module exports
- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — this memo is NEW; zero mutations
- Catalog #287 substantive-rationale rejection — placeholder literals rejected throughout
- Catalog #348 4-field contract: bug-class symptom signature ✓ + pre-fix window ✓ + historical search results ✓ + per-finding RE-EVAL-priority assignment ✓

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
