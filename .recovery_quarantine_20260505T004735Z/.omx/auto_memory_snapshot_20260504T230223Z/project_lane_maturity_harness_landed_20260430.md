---
name: PROJECT — Lane Maturity Harness LANDED (registry + CLI + Check 90 + tests + report + CLAUDE.md)
description: 2026-04-30. Built the canonical Lane Maturity Harness that mechanically tracks every lane's Level (0/1/2/3) status against the 7-gate Level-3 checklist defined in feedback_production_hardened_standard_definition_20260430.md. Makes it impossible for a lane to claim Level 2/3 without registered evidence.
type: project
originSessionId: lane-maturity-harness-20260430
---

## What landed

Six artifacts, six logical commits via tools/subagent_commit_serializer.py:

1. **`.omx/state/lane_registry.json`** — canonical JSON registry (schema_version=1) seeded with 23 lanes from the Phase 1/1.5/2/3 audit. Each lane has 7 gates (impl_complete, real_archive_empirical, contest_cuda, strict_preflight, three_clean_review, memory_entry, deploy_runbook) with status+evidence per gate, plus computed level.
2. **`tools/lane_maturity.py`** — argparse-based CLI with subcommands: `audit` (default, color table), `mark`, `unmark`, `validate`, `report`, `add-lane`. Every mutation appends a JSONL forensics record to `.omx/state/lane_maturity_audit.log`.
3. **Preflight Check 90 `check_lane_registry_consistent`** in `src/tac/preflight.py` — wired STRICT @ 0 violations in `preflight_all()`. Validates schema, no duplicate ids, all 7 gates present per lane, stored level matches computed-from-gates, every file-path-looking evidence string points to a real file on disk.
4. **`src/tac/tests/test_lane_maturity_harness.py`** — 42 tests covering: registry load + schema mismatch, all 4 level-computation cases (0 / 1 / required-only / 6-of-7 / all-7), mark/unmark/add-lane validation, evidence-path file-existence enforcement, level/gates mismatch detection, duplicate-id detection, missing-gate detection, audit log JSONL append, report generation, real-registry CLI smoke tests, file-path heuristic parametrized tests, Check 90 wiring. All 42 pass in 0.57s.
5. **`reports/lane_maturity.md`** — auto-generated report grouped by Phase with summary stats and per-gate ✓/✗ table per lane.
6. **CLAUDE.md** updates: new "Lane maturity registry — non-negotiable" section after the Meta-bug catalog; Check 90 added to the catalog (STRICT @ 0 entry).

## Initial seed state (23 lanes)

| Level | Count | Lanes |
|-------|-------|-------|
| L3    | 1     | lane_g_v3 (the only currently Level 3 lane per audit) |
| L2    | 4     | lane_pose_delta_pd_v2, lane_omega_w_v2, lane_lct, lane_j_nwc |
| L1    | 11    | lane_sc_plus_plus, lane_stc_clean_source, lane_psd, lane_multi_pass_inflate, lane_joint_admm, lane_multi_pass_inflate_phase15, lane_10_admm_realcodec, lane_12_nerv_mask_codec, lane_17_imp_10cycle, lane_19_segnet_logit_margin, lane_20_balle_hyperprior |
| L0    | 7     | lane_gp_rerun, lane_sensitivity_map, lane_bit_level_archive_opt, lane_mdl_bayesian, lane_raft_radial_pose, lane_decoder_systems_rewrite, lane_rl_pufferlib_bandit |

Match exactly to feedback_production_hardened_standard_definition_20260430.md per-lane audit.

## How to use the CLI

```bash
# Audit table (default, no args)
.venv/bin/python tools/lane_maturity.py
.venv/bin/python tools/lane_maturity.py audit

# Mark a gate satisfied (for subagents shipping work)
.venv/bin/python tools/lane_maturity.py mark lane_omega_w_v2 \
    --gate contest_cuda \
    --evidence "0.97 [contest-CUDA] reports/raw/2026-04-30-omega-w-v2-cuda/auth_eval.json"

# Unmark (revert) — reason required
.venv/bin/python tools/lane_maturity.py unmark lane_x \
    --gate three_clean_review --reason "Round 4 found CRITICAL bug, counter reset"

# Validate registry (preflight Check 90 also calls this)
.venv/bin/python tools/lane_maturity.py validate

# Regenerate the report (commit it)
.venv/bin/python tools/lane_maturity.py report

# Register a new lane at L0
.venv/bin/python tools/lane_maturity.py add-lane lane_new \
    --name "New Lane" --phase 2 --notes "..."
```

## How preflight Check 90 enforces consistency

Wired into `preflight_all()` STRICT after Check 91 (lane GP basis-fit kill). Runs on every preflight invocation: pre-commit hook, CI, every direct preflight_all call. Failures raise `MetaBugViolation` with the specific errors.

What it catches:
- Hand-edits that bumped `level` without bumping the corresponding gates
- Hand-edits that mark a gate true with a path that doesn't exist
- Copy-paste bugs that produce duplicate lane ids
- Schema drift (older registries with schema_version != 1)
- Missing gates (some hand-edit dropped a gate field)

What it does NOT catch (acceptable trust):
- An operator marking 7 gates true with descriptive text only (e.g. "documented in council notes"). The system trusts evidence the same way the rest of the codebase trusts proxy/auth tags. The audit log preserves the literal evidence for post-hoc forensics.
- An operator lying about a `[contest-CUDA]` score. Same trust model as the rest of the codebase.

## Adversarial self-review (Shannon / Quantizr / Hotz / Contrarian)

**Shannon (information-preserving math):** Gate-to-level mapping is 7 booleans → 4 ordinal levels. Lossy by design but the level rules favor specific informative thresholds (impl_complete + real_archive_empirical for L2; all 7 for L3). Validator catches stored-vs-computed drift. OK.

**Quantizr (would my own registration be blocked?):** Tested edge cases — `[empirical:Modal-T4-CUDA call_id fc-xxxx]` and `[contest-CUDA] 1.05 score from Vast.ai 4090` correctly classified as descriptive (not paths). Path heuristic only fires on real path shapes. OK.

**Hotz (CLI ergonomic enough that subagents will use it?):** 5-token mark command. Argparse choices for --gate. Color audit. Compared to manual JSON + level recompute + audit log append, it's a clear win. OK.

**Contrarian (registered-Level-3 that doesn't actually meet the bar?):** A subagent could mark all 7 gates true with descriptive text only (no path enforcement). MITIGATION: audit log preserves literal evidence; CLAUDE.md describes what each gate means; reviewers can audit. NOT a CRITICAL bug — accepts the same trust model as the rest of the codebase. Filed as known weakness.

A second Contrarian concern: `evidence` for `contest_cuda` is not format-enforced. Could be "1.05" or "1.05 [contest-CUDA]" or "report path". MITIGATION: gate semantics documented in CLAUDE.md + memory; bare numbers are caught by `check_scores_have_lane_tag` (separate STRICT preflight check) on the markdown surfaces (run_log/findings/BATTLE_PLAN). NOT a CRITICAL bug.

## Cross-refs

- `feedback_production_hardened_standard_definition_20260430.md` — the 4 maturity levels + 7-gate Level-3 standard + per-lane audit
- `feedback_subagent_serializer_temp_index_landed_20260430.md` — commit serializer used for all 6 commits
- CLAUDE.md "Lane maturity registry — non-negotiable" section (appended this session)
- CLAUDE.md "Meta-bug class catalog" — Check 90 added at #90

## Test pass

42/42 pass in `pytest src/tac/tests/test_lane_maturity_harness.py -v` (~0.57s).

## Files created/modified

- CREATED: `.omx/state/lane_registry.json` (23 lanes seeded)
- CREATED: `tools/lane_maturity.py` (CLI)
- CREATED: `src/tac/tests/test_lane_maturity_harness.py` (42 tests)
- CREATED: `reports/lane_maturity.md` (auto-generated)
- MODIFIED: `src/tac/preflight.py` (Check 90 wired STRICT)
- MODIFIED: `CLAUDE.md` (new section + Check 90 in catalog)
