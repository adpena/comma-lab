<!-- HISTORICAL_PROVENANCE: APPEND-ONLY per Catalog #110/#113. Retroactive sweep per Catalog #348 for newly-landed Catalog #365 (canonical helper signature drift). -->
<!-- HISTORICAL_SCORE_LITERAL_OK: this memo cites historical Cascade C' fc-call_id anchors; no NEW score literal claims. -->

# Retroactive sweep for Catalog #365 — canonical helper signature drift (gate_auth_eval_call kwarg drift)

**Date:** 2026-05-27T00:29:31Z
**Per:** CLAUDE.md "Operator gates must be wired and used" non-negotiable + Catalog #348 "EVENT-DRIVEN RETROACTIVE VERDICT-TAINT SWEEP" self-protection.

## 1. Bug-class symptom signature

A trainer wrapper invokes `tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call(...)` with one or more of the following drift modes:

1. **Deprecated kwarg names** — uses `archive=` (canonical `archive_zip=`), `json_out=` (canonical `output_json=`), `lane_id=` (not in canonical signature), or `substrate_id=` (canonical `substrate_tag=`).
2. **Missing canonical required kwargs** — omits `args=` (argparse.Namespace), `archive_zip=`, `output_json=`, `contest_auth_eval_script=`, or `substrate_tag=`.

The canonical signature inspected at `src/tac/substrates/_shared/smoke_auth_eval_gate.py:261` is fully keyword-only (separator `*,`) with the 7 required kwargs above + 7 optional kwargs.

## 2. Pre-fix window

The bug-class drift was empirically demonstrated **once** in the recent session: Cascade C' subagent C combined Stage 7 auth_eval failed `TypeError` on the Modal T4 dispatch `fc-01KSK7GTPEF27FX0AAH2319GVR` (2026-05-26 17:50:31, rc=0 in 5.2s with stages 1-6 SUCCESS but stage 7 TypeErrored). Pre-fix kwargs were:
```
gate_auth_eval_call(
    archive=submission_dir / "0.bin",         # deprecated
    inflate_sh=top_inflate_sh,
    json_out=auth_eval_json_path,             # deprecated
    device=auth_eval_axis,
    upstream_dir=args.upstream_dir,
    lane_id=lane_id,                          # not in signature
    substrate_id=substrate_id,                # deprecated
)
```
Fix landed commit `a885ea2e5` (2026-05-26 17:50:31).

## 3. Historical-KILL/DEFER/FALSIFY search results

Searched repo for all `gate_auth_eval_call(` invocations across:
- `experiments/train_substrate_*.py` (92 files)
- `experiments/train_renderer*.py` (any matches)

Live count BEFORE Catalog #365 landing: **0** (the Cascade C' fix `a885ea2e5` cleaned the only known violation in the same commit batch; sister Catalog #226 18-trainer-refactor wave commits 7b2fece4a + 27db6f6bd + 9839a3a73 had already routed all 18 substrate trainers through the canonical helper).

**No historical KILL / DEFER / FALSIFY memos cite the kwarg drift bug class.** This is a NEW bug class surfaced 2026-05-26 specifically by the Cascade C' substrate scaffold's trainer wrapper. The 18-trainer Catalog #226 refactor wave did not introduce this drift because each refactored trainer used the canonical kwargs from byte one; the Cascade C' scaffold landed with drifted kwargs because the operator was working from a slightly stale local memory of the helper signature.

No historical verdicts require RE-EVAL because:
1. The bug class is structural (kwarg name mismatch produces `TypeError` at first invocation, NOT a silent semantic regression).
2. Every previous Cascade B / Cascade A / sister substrate trainer that successfully ran auth_eval used the canonical kwargs (empirical proof: their auth_eval stages did NOT TypeError).
3. The bug-class extinction lives at Catalog #226 (canonical helper routing) + Catalog #365 (canonical kwarg names within the helper); both are STRICT-from-byte-one.

## 4. Per-finding RE-EVAL-priority assignment

| Historical Finding | RE-EVAL Priority | Rationale |
|---|---|---|
| N/A — no historical kill/defer/falsify verdicts apply | N/A | This bug class is NEW (2026-05-26) and has no historical precedent. The Cascade C' subagent C empirical anchor is the only known incident and was fixed in the same commit batch. |

## 5. Cross-references

- CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable
- CLAUDE.md "Operator gates must be wired and used" non-negotiable
- Catalog #226 sister gate (`check_trainer_auth_eval_uses_canonical_helper` — same META class at the call-EXISTS surface)
- Catalog #164 (canonical scorer-loss helper routing)
- Catalog #205 (canonical inflate device-fork)
- Catalog #218 (canonical mini-batch reconstruct)
- Catalog #348 retroactive verdict-taint sweep discipline
- Catalog #287 placeholder-rationale rejection
- Cascade C' subagent C combined verdict commit `f661770aa..994cc673c` + fix `a885ea2e5`

## 6. Discipline declarations

- Catalog #229 PV: full git log inspection + canonical helper signature inspection + AST audit of all 93 trainer files at preflight wire-in time
- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — this memo is NEW; zero mutations
- Catalog #287 substantive-rationale rejection — placeholder literals rejected throughout
- Catalog #348 4-field contract: bug-class symptom signature ✓ + pre-fix window ✓ + historical search results ✓ + per-finding RE-EVAL-priority assignment ✓

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
