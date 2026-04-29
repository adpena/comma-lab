# DX + Telemetry bug audit — 2026-04-28 (post-verify_vast fix)

Trigger: 3 DX bugs landed in `scripts/verify_vast_instances.py` (commit
`21784862`). Audit charter: find the same bug classes elsewhere in DX,
telemetry, and monitoring tools across the repo before they cost more $$
on Vast.ai or break the review gate.

## Bugs fixed (3 commits)

| Commit | File | Bug class | Severity |
|---|---|---|---|
| `6ceed18e` | `scripts/launch_lane_on_vastai.py` | NoneType / KeyError on dict.get + TIMEOUT vs failure conflation + int() crash | HIGH |
| `dcb088af` | `tools/review_tracker.py` | Silent-success on non-tracked file → ungated commits | HIGH |
| `f4417422` | `src/tac/deploy/vastai/cli.py` | datetime.fromisoformat crash mid-fleet (3 sites) | HIGH |

## Per-file audit findings

### `scripts/verify_vast_instances.py`
Status: **CLEAN** — fix from commit `21784862` is correct. Verified `classify()`
has only one caller (line 192) and `ssh_succeeded` is computed correctly.

### `scripts/launch_lane_on_vastai.py` (~720 LOC)
3 HIGH bugs fixed in commit `6ceed18e`:
1. **KeyError on partial state** — `_resolve_host_port` (was line 684) used
   `info["ssh_host"]` and `int(info["ssh_port"])` directly. If Vast.ai
   reports `actual_status=running` but ssh fields haven't propagated yet
   (race condition seen 2026-04-28), the launcher KeyError'd. Fix: explicit
   None-checks + RuntimeError naming both fields.
2. **TIMEOUT vs failure conflation** — `extract_remote` printed
   `extract failed: TIMEOUT` with no actionable signal. Now: distinct
   timeout branch with retry guidance.
3. **int() crash on non-numeric ID** — `create_instance` did `int(instance_id)`
   without try/except. If Vast.ai returns the string `"None"`, the launcher
   crashes mid-launch leaving an orphan instance. Now: explicit RuntimeError
   naming the bad value.

### `scripts/probe_nvdec.sh` (~451 LOC)
Status: **CLEAN**. Codex R5-r6 fix #4 already added classification tokens
(NVDEC_MISSING / DALI_BUILD / FIXTURE / UNKNOWN) with corresponding exit
codes 2/3/4/5. Defensive against fixture corruption (Stage 0 bytes/ftyp
validation) and the lightweight pre-probe correctly exits non-zero on
NVDEC_MISSING and exit 5 on UNKNOWN (do NOT auto-destroy on UNKNOWN).
The order `nvdec_markers` > `fixture_markers` > `dali_markers` is correct
because DALI surfaces NVDEC failures via its own Pipeline class.

### `scripts/remote_setup_full.sh` (~159 LOC)
Status: **CLEAN with 1 LOW note**. Stage 0.5 lightweight pre-probe before
DALI install saves $0.05+ per bad host. Stage 5 venv has the idempotency
guard `if [ ! -d "$WORKSPACE/.venv" ]`. Stage 6.5 strips AppleDouble files.
Stage 8 emits provenance.json. Heartbeat is NOT written by setup_full.sh —
that's deferred to each lane script per Check 41 (correct design).

LOW: `set -euo pipefail` + `grep -q` pattern on lines 89, 94 is documented
as a pitfall on line 84-87 and the `grep -q` guards capture full output
first via `SCALE_HELP=$(...)`. Defensive — looks fine.

### `tools/review_tracker.py` (~1746 LOC)
1 HIGH bug fixed in commit `dcb088af`:
- **`cmd_mark_file` silent-success** — printed "No entities matching" and
  exited 0 when called with `.sh`/`.md`/etc. files. The tracker only
  ingests `.py` files, so operators saw no error and assumed the gate
  passed. Fix: classify by extension, emit clear NOTICE, return non-zero;
  main propagates via `sys.exit(rc)`.

LOW finding (not fixed): the `scan_all_modules()` function does NOT include
`scripts/` directory. Python files under `scripts/` (e.g.,
`scripts/launch_lane_on_vastai.py`, `scripts/verify_vast_instances.py`)
are silently ungated — they cannot be tracked, marked, or reviewed.
Currently the workaround is REVIEW_GATE_OVERRIDE=1 for those files.
Consider expanding scan scope to include scripts/ in a future cycle.

### `src/tac/preflight.py` Checks 41/47/48
Status: **HEALTHY**. Verified live state via `preflight_all()`:
- Check 41 (lane heartbeat): 0 violations across 69 scripts
- Check 47 (archive size assertion): 0 violations across 69 scripts
- Check 48 (orphan src/tac modules): 38 warnings (intentional warn-only,
  cleanup target).

The Check 47 heuristic correctly fires only when both `_ARCHIVE_BUILD_MARKERS`
AND `_AUTH_EVAL_MARKERS` appear — lane scripts that delegate to a Python
helper (Lane G v3, Lane EC v2) are correctly exempted.

### `src/tac/deploy/vastai/cli.py` (~351 LOC)
3 HIGH bugs fixed in commit `f4417422`:
- `datetime.fromisoformat(inst.created_at)` crashed at three sites
  (status:128, results:217, destroy:261) when created_at was missing/None
  on tracker entries from an older client version. The exception killed
  the entire status/results/destroy command mid-loop, potentially leaving
  downstream instances un-destroyed (cost burn).
  Fix: try/except with yellow warning + safe fallback at each site.

### `tools/vastai_orphan_cleanup.py`
Status: **CLEAN**. `_age_minutes()` returns None on parse failure (caller
filters to None). `gpu_util` parsed defensively with try/except.

### Lane scripts (5 sampled)
Sample: `remote_lane_hm_homography_motion.sh`, `remote_lane_i_darts_dims.sh`,
`remote_lane_ec_v2.sh`, `remote_lane_omega_v3_rate_frontier.sh`,
`remote_lane_g_v3_corrected_kl_weight.sh`.

| Script | heartbeat | ARCHIVE_BYTES | AppleDouble | [contest-CUDA] | set -euo |
|---|---|---|---|---|---|
| hm | ✓ | (delegates) | (in setup_full) | ✓ | ✓ |
| i_darts | ✓ | (delegates) | (in setup_full) | DARTS variant | ✓ |
| ec_v2 | ✓ | (delegates) | (in setup_full) | ✓ | ✓ |
| omega_v3 | ✓ | ✓ | (in setup_full) | ✓ | ✓ |
| g_v3 | ✓ | (delegates) | (in setup_full) | ✓ | ✓ |

All 5 PASS. The "delegates" pattern means archive build is done in a
Python helper (`build_baseline_archive` etc.) and the lane script consumes
it — correct, no size assertion needed at shell level. Check 47 confirms 0
violations across all 69 lane scripts.

## Top 3 patterns observed

1. **`.get()` without default → NoneType crash on optional fields.**
   3 of 9 fixes were this pattern: `dict.get("field")` → None →
   `int(None)` / `datetime.fromisoformat(None)` / `str[:N]` all crash. The
   fix is always: explicit None check + actionable error message.

2. **Silent-success on no-match → operator assumes gate passed.**
   review_tracker `cmd_mark_file` and previously `verify_vast_instances`
   IDLE classification both fall in this class. The fix is always: emit
   a clear NOTICE explaining what went wrong + return non-zero. The
   "silent-success cascade" is one of the most expensive bug classes
   because it masks itself.

3. **Conflated error states with no actionable diagnostic.**
   `extract_remote: TIMEOUT`, `verify: UNREACHABLE` (was: SSH failed +
   heartbeat absent), `probe_nvdec: every-error → exit-2`. The fix is
   always: distinct error branches with operator-actionable retry
   guidance and a classification token the next layer up can dispatch on.

## Memory entry candidate

Pattern name: `feedback_dx_telemetry_silent_success_audit_20260428`.

Recurring DX/telemetry bug class across 4 files in 1 day. The bugs all
share the same shape: a tool reports SUCCESS or returns 0 when the
underlying operation actually failed silently. Operators react to the
exit code, not the log text, so the silent-success masks itself.

**Mitigation rule**: every CLI tool that performs a single logical
operation MUST exit non-zero on any failure mode, including no-match,
empty input, and missing required fields. Operators reading exit codes
must be able to distinguish operation-succeeded vs operation-skipped.

## Files changed

- `scripts/launch_lane_on_vastai.py` (commit `6ceed18e`, +40 −4)
- `tools/review_tracker.py` (commit `dcb088af`, +42 −7)
- `src/tac/deploy/vastai/cli.py` (commit `f4417422`, +37 −3)

Total: 3 HIGH bugs fixed across 3 files. Preflight remains green.
Review tracker selftest passes (8323 entities, 7/7 tests).

---

## Codex review addendum — 2026-04-28 PM (3 more findings, same patterns)

Lane EC + Lane EBR went LIVE on Vast.ai with all three bugs latent. Codex
caught them before the lanes hit their failure stages (gradient corruption +
EMA crash + budget under-spend). Three new commits:

| Commit | File | Bug class | Severity |
|---|---|---|---|
| `6de52150` | `src/tac/bit_allocator.py` | Bisection bracket too tight → budget under-spent | MEDIUM |
| `38f81188` | `experiments/precompute_gradient_corrections.py` | Sign-only int8 writes + fast-model byte budget vs real packed format | HIGH |
| `fed09e10` | `src/tac/training.py` | Module registered AFTER EMA snapshot → KeyError on update | HIGH |

### Patterns we now catch (regression-test anchored)

1. **"Sign-only quantization throws away gradient magnitude"** — encode the
   actual gradient values via per-tensor symmetric quantization. Anchor:
   `test_greedy_preserves_gradient_magnitude_codex_finding1` (asserts SIGN
   of negative gradient survives + at least one |val| < 127).
2. **"Closed-form byte estimate vs actual packed format"** — never trust
   `bytes ≈ overhead + n × per_pixel`. Always feed the result through the
   actual packer in a drop-tail loop. Anchor:
   `test_enforce_packed_byte_cap_drops_tail_to_fit_budget_codex_finding1_bug2`.
3. **"Module registered after EMA/optimizer snapshot"** — register first,
   snapshot second. Plus harden the snapshot to tolerate late additions.
   Anchor: `test_codex_finding2_ema_safe_when_module_added_after_construction`.
4. **"Monotonic bisection bracket on the wrong side of target"** — test the
   upper bracket value first; grow exponentially until it crosses. Anchor:
   `test_codex_finding3_bracket_grows_when_initial_c_hi_underspends`.

### Same shape as the prior audit
Patterns 1, 2, 3 above are all variants of the **silent-success** class from
the original audit: a tool reports success (returns gradients, packs an
archive, snapshots EMA) but the underlying state is corrupt. Detection
requires VALUE/SIGN regression tests at the boundary, not just SUCCESS
return-code asserts. Going forward: every Codex round 1 of a new lane MUST
include at least one anchor that asserts a NUMERIC property of the produced
artifact, not just "returned without exception".
