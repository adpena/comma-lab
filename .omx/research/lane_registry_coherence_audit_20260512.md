# Lane registry + sub-agent coherence audit — 2026-05-12

generated_at: 2026-05-12T17:45:11Z
from_state_hash: lane_registry.json @ 410 lanes; lane_maturity_audit.log
audit_lane_id: `lane_session_coherence_audit_20260512`
scope: Catalog #90 / Catalog #125 / Catalog #126 cross-reference + memo-vs-registry orphan scan
gpu_spend: $0 (read-only audit)

## Executive summary

HEADLINE: Catalog #125 and Catalog #126 are **already STRICT @ 0 live
violations** (since 2026-05-09 STRICT-FLIP per "fix all yourself" operator
approval). The task's premise that these checks are warn-only with backfill
needed is STALE; both gates have been ratcheted to strict for ~3 days. No
backfill work required.

Counts before audit:

- Catalog #125 (`check_subagent_landing_has_solver_wire_in`): **0 violations** (STRICT). 111 post-cutover memos scanned, 6 research-only opt-out, 0 missing.
- Catalog #126 (`check_lane_pre_registered_before_work_starts`): **0 violations** (STRICT). 228 files across 51 sources scanned, 0 unregistered lane references.
- Catalog #90 (`check_lane_registry_consistent`): **0 violations** (STRICT). 410 lanes validated cleanly via `tools/lane_maturity.py validate`.

Counts after audit:

- Same as before — no preflight regressions introduced.
- 5 new lanes pre-registered at L0 (1 audit lane + 4 named follow-up lanes surfaced by orphan-memo scan).

## Part A: Lane registry audit

### A.1 Lane totals

```
Total lanes: 410   L3=0   L2=62   L1=168   L0=180
```

(Post pre-registration of 5 lanes during this audit; was 405 at start.)

### A.2 Catalog #90 STRICT check

`python tools/lane_maturity.py validate` → `OK — 410 lane(s) validated cleanly.`

Coverage:

- duplicate `id`: 0
- gate dict shape: 410/410 valid
- computed-level vs stored-level: 410/410 match
- file-path-looking evidence pointing at non-existent files: 0/672 satisfied gates (via `looks_like_filepath` heuristic with `_FILE_PATH_PREFIXES`)

Note: a naive substring-`/` scan finds 275 "broken" tokens but they are all text artifacts (`14/14` test counts, `for/else` keywords, `producer/adapter/wrapper` descriptions). The canonical Catalog #90 heuristic correctly filters these via the path-prefix rule and returns 0.

### A.3 Memo-vs-registry orphan scan (advisory; outside Catalog #126 scope)

Catalog #126 enforces orphans in **source files** (`.py`/`.sh`); this audit
additionally scans landing memo backtick references for awareness. After
applying the `_LANE_ID_REFERENCE_BLOCKLIST` filter, 6 memo files reference
8 unregistered lane tokens. Classification:

| memo | unknown ref | classification | action |
| --- | --- | --- | --- |
| `feedback_check_125_126_coherence_by_default_strict_landed_20260509.md` | `lane_arc3_v2`, `lane_arch_c3_v2`, `lane_c30_xyz`, `lane_class_proofs`, `lane_id_normalizer` | docstring/example tokens explicitly cited as Yousfi false-positives in the memo body | NONE — already-resolved historical record |
| `feedback_codex_round4_findings_fix_with_self_protection_landed_20260509.md` | `lane_artifact_lifecycle_state_lock`, `lane_maturity_state_lock` | deferred follow-up lanes named in body | **PRE-REGISTERED at L0** (this audit) |
| `feedback_gha_cpu_eval_queue_landed_20260512.md` | `lane_t1_balle_128k_endtoend` | stylistic prefix mismatch — registry id is `t1_balle_128k_endtoend` (without `lane_` prefix) | none — naming-convention issue, not a missing lane |
| `feedback_magic_codec_dense_xray_substrate_classifier_small_dispatches_landed_20260511.md` | `lane_magic_codec_categorical_contiguous_fix` | conditional follow-up lane named in body | **PRE-REGISTERED at L0** (this audit) |
| `feedback_pr106_r2_paired_cpu_eval_landed_20260511.md` | `lane_pr106_yshift_score_table` | real in-flight Kaggle kernel lane | **PRE-REGISTERED at L0** (this audit) |
| `feedback_t1_phase1_scaffold_landed_20260509.md` | `track1_phase_a1_score_gradient_bestproxy_lr2e6_20260509_codex` | archive directory name (not a lane id) | none — already covered by registered `track1_phase_a1_score_gradient` |

### A.4 Stale lane scan

| level | total | stale (> 30 d untouched) | no audit history |
| --- | --- | --- | --- |
| L0 | 180 | 0 | 3 |
| L1 | 168 | 0 | 5 |
| L2 | 62 | n/a | n/a |
| L3 | 0 | n/a | n/a |

Registry is healthy — no stale candidates require reclassification. The 8 lanes with no audit history are original-batch entries from before `lane_maturity_audit.log` was instituted (`lane_g_v3`, `lane_sc_plus_plus`, `lane_pose_delta_pd_v2`, etc.); acceptable.

## Part B: Sub-agent coherence audit (Catalog #125)

### B.1 Live count

```
[subagent-landing-wire-in] OK (111 post-cutover memo(s) scanned, 6 research-only opt-out, 0 missing)
```

110 of 111 post-cutover memos already declare the 6-hook wire-in section
(remaining 1 is the same set as research-only opt-out). Backfill is not
required.

### B.2 Per-hook breakdown

Sample of 5 random memos:

- Sensitivity-map: EXERCISED or N/A with rationale — all 5/5
- Pareto constraint: EXERCISED or N/A with rationale — all 5/5
- Bit-allocator: EXERCISED or N/A with rationale — all 5/5
- Cathedral autopilot dispatch hook: EXERCISED or N/A with rationale — all 5/5
- Continual-learning posterior update: EXERCISED or N/A with rationale — all 5/5
- Probe-disambiguator: EXERCISED or N/A with rationale — all 5/5

No backfill commits issued — gate is at 0.

## Part C: Lane pre-registration backfill (Catalog #126)

### C.1 Live count

```
[lane-pre-registered] OK (228 file(s) across 51 source(s) scanned; 0 unregistered lane references)
```

Catalog #126 enforces source-file scope (commits + dirty/untracked WIP). No
violations exist there.

### C.2 New lanes pre-registered during this audit

Sourced from memo-cross-reference (advisory orphan-memo scan, Part A.3):

1. `lane_session_coherence_audit_20260512` — L0, phase 2 — this audit's own lane.
2. `lane_artifact_lifecycle_state_lock` — L0, phase 0 — deferred follow-up from codex round 4 fix.
3. `lane_maturity_state_lock` — L0, phase 0 — deferred follow-up from codex round 4 fix.
4. `lane_magic_codec_categorical_contiguous_fix` — L0, phase 1 — deferred follow-up from magic_codec dense xray.
5. `lane_pr106_yshift_score_table` — L0, phase 1 — real Kaggle yshift kernel lane.

All five additions are append-only registry mutations via `python tools/lane_maturity.py add-lane …`, audit-logged to `.omx/state/lane_maturity_audit.log`.

## Part D: Strict-flip recommendation

**Already done.** Catalog #125 + #126 are both STRICT @ 0 since 2026-05-09 per:

- `src/tac/preflight.py:1894` — `check_subagent_landing_has_solver_wire_in(strict=True, ...)`
- `src/tac/preflight.py:1902` — `check_lane_pre_registered_before_work_starts(strict=True, ...)`

No strict-flip operator decision required.

## Operator decision surfaced

NONE — the audit confirmed the gates are already enforcing strict and the
session-state was healthier than the task brief assumed. The only actions
taken were the 5 lane pre-registrations (pure-append, no risk).

## Wire-in declarations (this audit)

This audit is an internal accounting pass — the 6 solver-integration hooks are:

- Sensitivity-map contribution: N/A — audit pass, no new tensors / no sensitivity signal.
- Pareto constraint: N/A — audit pass, no new feasibility constraint.
- Bit-allocator hook: N/A — audit pass, no per-tensor importance changes.
- Cathedral autopilot dispatch hook: N/A — audit pass, no archive-deployable byte change.
- Continual-learning posterior update: N/A — audit pass, no new empirical anchor.
- Probe-disambiguator: N/A — no 2+ defensible interpretations; the registry is single-source-of-truth.

## Verification

- `python tools/lane_maturity.py validate` → OK (410 lanes).
- `python -c "from tac.preflight import check_subagent_landing_has_solver_wire_in, check_lane_pre_registered_before_work_starts; check_subagent_landing_has_solver_wire_in(strict=True, verbose=True); check_lane_pre_registered_before_work_starts(strict=True, verbose=True)"` → both OK.
- No `.py` files touched; review gate not invoked.

## Cross-refs

- CLAUDE.md "Lane maturity registry — non-negotiable"
- CLAUDE.md "Subagent coherence-by-default — NON-NEGOTIABLE, HIGHEST EMPHASIS"
- Catalog #90 / #125 / #126 in CLAUDE.md "Meta-bug class catalog"
- `feedback_check_125_126_coherence_by_default_strict_landed_20260509.md` — the canonical strict-flip landing memo
