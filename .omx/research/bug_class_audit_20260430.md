# Bug Class Audit — 2026-04-30 PM

**Owner:** BUG-CLASS-HARDENING-AGENT
**Trigger:** User mandate "keep hardening and fixing all bug classes and metabugs"
**Baseline preflight count at audit start:** 90+ STRICT checks (60s runtime)
**Audit deliverable:** prioritized list of new bug classes + proposed STRICT checks

## Investigation methodology

Sherlock-Holmes audit:
1. Read CLAUDE.md FORBIDDEN PATTERNS and existing meta-bug class catalog
2. Read recent failure-mode memory files (Ω-W-V2 regression, Lane GP v4 KILL, swarm recovery)
3. Read commit-serializer.log to identify lock-contention / hook-runtime patterns
4. Grep codec / pose-fit modules for patterns matching prescribed bug classes
5. Audit Modal `.spawn()` callsite registration discipline
6. Identify new patterns visible in the past 48h of repo activity

## Findings (sorted by severity)

### Class A — PoseNet-sensitivity-blind codec (CRITICAL)

**Pattern:** A codec module mutates `renderer.bin` weights (touches `state_dict()` /
`*.weight`) but contains no PoseNet-protection mention (no docstring assertion, no
sensitivity-weighted Hessian, no PoseNet-FastViT-input-derivative weighting).

**Incident:** Lane Ω-W-V2 stack on 2026-04-30 burnt $0.05 + 50s GPU and produced
score 1.07 vs Lane G v3 baseline 1.05. Codec saved -0.034 rate but cost +0.052 PoseNet
distortion (PoseNet went from 0.003455 → 0.005644, +63.4%). Memory:
`feedback_owv2_savings_correction_conv_vs_full_renderer_20260430.md`.

**Audit data (state_dict mentions vs PoseNet protection mentions):**

| Module | state_dict | PoseNet | Status |
|---|---:|---:|---|
| `src/tac/neural_weight_codec.py` | 2 | 0 | UNPROTECTED |
| `src/tac/water_filling_codec_v2.py` | 1 | 0 | UNPROTECTED |
| `src/tac/balle_hyperprior_codec.py` | 5 | 0 | UNPROTECTED |
| `src/tac/block_fp_codec.py` | 11 | 0 | UNPROTECTED |
| `src/tac/owv2_renderer_archive.py` | 12 | 1 | UNDERPROTECTED (1 mention is the bug fix promise, not real protection) |
| `src/tac/water_filling_codec.py` | 17 | 2 | partial (Hessian-aware) |
| `src/tac/contrib/vqvae_codec.py` | 10 | 2 | partial |
| `src/tac/network_codec.py` | 15 | 13 | well-protected |
| `src/tac/nerv_mask_codec.py` | 15 | 2 | mask codec — exempt by pattern |
| `src/tac/neural_weight_codec_sensitivity.py` | 0 | 3 | wrapper (no state_dict mutate) |

**Severity: CRITICAL** — exact incident class that just bit a $25 stack a few hours ago.
4 codec modules currently could ship a regression like Ω-W-V2.

**Proposed Check N+1: `check_renderer_codec_has_posenet_protection`**

- Detection: AST-scan `src/tac/*codec*.py` for state_dict/.weight mutation
  (`state_dict()`, `state_dict[`, `module.weight =`, `param.data =`).
- Require: docstring or inline mention of one of:
  `[posenet-protected]`, `[posenet-sensitivity-weighted]`, `[mask-codec-not-renderer]`,
  `[per-channel-hessian]`, `[fp16-only-fallback]`, OR explicit waiver
  `# RENDERER_CODEC_POSENET_PROTECTION_WAIVED:<reason>`.
- Exempt: mask-only codecs (`mask_codec.py`, `nerv_mask_codec.py`,
  `argmax_codec.py`), pose-only codecs (`pose_delta_codec*.py`), magic registries
  (`codec_magic_registry.py`), library-only utilities (`arithmetic_qint_codec.py`),
  benchmark modules (`benchmark_codecs.py`).
- Waiver pattern: same-line `# RENDERER_CODEC_POSENET_PROTECTION_WAIVED:`.

### Class B — White-noise pose-stream assumption (HIGH)

**Pattern:** A new pose-fit / pose-basis / pose-residual module assumes the pose
trajectory is smooth (low-frequency, Runge-friendly, polynomial/spline/DCT-fittable)
without first running an empirical white-noise test against actual Lane G v3
baseline poses.

**Incident:** Lane GP v4 (B-spline + DCT + natural cubic spline candidates) all
plateaued at avg RMSE ≈ 1.15-1.59 (near signal std 1.5-2.3) because pose dims 1-5
are white-noise (`diff_std/signal_std ≈ 1.35` ≈ √2). Memory:
`project_lane_gp_v4_killed_basis_fit_infeasible_20260430.md`. This is mostly already
covered by Check 91 `check_pose_basis_fit_kill_acknowledged`, BUT that check is a
file-existence + kill-marker check. It does NOT enforce that the empirical
white-noise test was actually run.

**Severity: HIGH** — Check 91 already covers part of this. Class is mostly extinct
but the empirical-test-runs-before-claim discipline is not yet enforced.

**Proposed Check N+2: `check_pose_fit_module_has_white_noise_test`**

- Detection: any file matching `experiments/fit_pose_*.py` or
  `src/tac/pose_*_fit.py` / `pose_*_basis.py` / `pose_*_polynomial.py` etc. (same
  scope as Check 91).
- Require: matching test file `src/tac/tests/test_<module_name>_white_noise.py`
  OR an inline marker `# WHITE_NOISE_CHECK_DEFERRED:<reason>` (Lane GP v4 KILL
  marker counts as the strongest deferral). The test must call the module on
  `experiments/results/lane_g_v3_landed/optimized_poses.pt` and assert RMSE ≥ 0.5
  for dims 1-5 (i.e., the basis CANNOT fit white noise, which is the negative
  result that justifies the kill verdict).
- Lands STRICT @ 0 violations (Lane GP v4's KILL markers cover all current
  candidates; new files start with the marker).

### Class C — Whole-repo preflight thundering-herd (HIGH operational)

**Pattern:** `tools/preflight_hook.py` runs `tac.preflight` (full ~60s scan) on
every commit. When N subagents commit in parallel, each waits its turn for the
exclusive lock + 60s hook = total wall-clock O(N × 60s).

**Incident data (from `.omx/state/commit-serializer.log` last 24h):**

- Max wait time observed: 361.594 s
- Max commit duration: 160.144 s
- Sum of (wait + commit): 5+ minutes for a single subagent
- Multiple `commit_failed` from full-repo violations the subagent never touched
  (e.g., MDL Bayesian codec `quantizer-roundtrip-tests` failure blocking
  unrelated subagent commits)

**Severity: HIGH operational** — directly costs subagent throughput AND causes
spurious commit failures (subagent A blocks on subagent B's uncommitted file).

**Proposed Refactor: `tools/preflight_hook.py --changed-files-only` mode**

- Default for pre-commit: scan only files matching `git diff --cached --name-only`.
- Cache: hash `git diff --cached --name-only --diff-filter=ACMR` + each file's
  blob SHA. If a recent run (<300s) exists for the same input hash, reuse its
  result. Cache file: `.omx/state/preflight_cache.json` (max 50 entries).
- `--full` flag (or env `PREFLIGHT_FULL=1`): run whole-repo scan. Used by CI /
  nightly.
- Fast checks (e.g., `check_no_mps_fallback_default`, `check_python_files_compile`)
  with per-file logic remain effective on changed-files subset.
- Slow checks that depend on cross-file state (e.g., `preflight_filename_contract`,
  `preflight_arity`, `check_lane_registry_consistent`) would skip in
  changed-files mode UNLESS a relevant file is in the changeset.
- Risk: changed-files mode might miss cross-file regressions (e.g., a profile
  removed from `profiles.py` while a `.sh` script still references it).
  **Mitigation:** every change to `profiles.py` / `preflight.py` triggers full
  scan via heuristic (in changed-files set OR matches `**/preflight*.py` /
  `**/profiles.py` glob).

**Severity: HIGH** — only operational, not correctness, but the contention
manifests AS correctness incidents (subagent commit_failed messages).

### Class D — Rule-attribution false-positive verification (MEDIUM)

**Pattern:** Check 83 (`check_no_proxy_metric_drives_decision`) was over-firing
on legitimate rule-citations like "per CLAUDE.md MPS rule". Already fixed in
commit 591b7a43 (added `per CLAUDE.md` / `Council #N` exemption).

**Severity: MEDIUM** — already fixed. Stress-test needed to verify no remaining
false-positive surface.

**Proposed: 5 new synthetic stress tests added to
`src/tac/tests/test_callsite_contracts_and_no_mps_decision.py`** simulating:

1. "Per CLAUDE.md non-negotiable, NO MPS for promote/kill" decision
2. "Council #271 ruled: KILL Lane MM (MPS noise, not architecture)" rule restatement
3. "WITHDRAWN: earlier KILL on MPS evidence superseded by [contest-CUDA] result"
4. "[advisory only] [MPS-PROXY] forecast: -0.05 vs baseline; promote pending CUDA"
5. "POST-MORTEM: 2026-04-25 KILL on MPS PoseNet 0.245 was 23x noise; FALSIFIED"

All 5 SHOULD NOT fire. Stress-test catches regression in exemption regex.

### Class E — Subagent commit-serializer lock contention (MEDIUM)

**Pattern:** Same root cause as Class C — when preflight is slow, the lock is
held longer, queueing N subagents.

**Already implemented mitigation:** generation-3 per-PID temp index (commit
b860710c per memory `feedback_subagent_serializer_temp_index_landed_20260430.md`).
That fix eliminated the staging-race body-shuffle bug class.

**Class C refactor (above) also addresses this.** Once preflight runs in <5s
on changed-files-only mode, the lock contention dissolves.

**Severity: MEDIUM** — addressed by Class C refactor.

### Class F — Newly discovered patterns (audit findings)

#### F1: Modal `.spawn()` discipline already canonical

Verified: `experiments/modal_train_lane.py:427` writes `modal_metadata.json` to
`experiments/results/lane_<label>_modal/` with call_id immediately after spawn.
`tools/harvest_modal_calls.py` reads these. The other `.spawn()` site at
`src/tac/deploy/modal/archive/modal_renderer_smoke_deploy.py:206` is in `archive/`
(per-path-name "explicitly archived; do not invoke"). No new check needed.

#### F2: Codec docstring claim-vs-reality dual-tagging gap

The Ω-W-V2 incident showed docstring "40.98% byte savings" was eligible-conv-
subset only, NOT full renderer (actual 20.59%). The CLAUDE.md "empirical-claim-
without-evidence-tag" forbidden pattern (Check 84 deferred) doesn't yet cover
the special "subset-vs-full" aliasing case. **Filed as future Check 84a follow-up.**
Out of scope for this round (Check 84 base check itself isn't landed yet).

#### F3: `check_python_files_compile` already strict

Compile check runs in 0.75s for 631 .py files, well-suited for changed-files
mode. Already fast; still benefits from caching.

#### F4: Test-coverage drift on new src/tac modules

35 modules touched in last 24h have no test_<module>.py file. The
`check_test_files_imports_resolve` only verifies that EXISTING tests import
correctly. No check requires that new src/tac modules SHIP with a test file.
**Filed as future check, out of scope this round.**

## Prioritization for Phase 2 (3 STRICT checks to land)

**Pick:**
1. **Class A — `check_renderer_codec_has_posenet_protection`** (CRITICAL, recent
   incident, 4 unprotected codec modules visible RIGHT NOW)
2. **Class B — `check_pose_fit_module_has_white_noise_test`** (HIGH, pairs with
   Check 91, Lane GP v4 KILL provides clean baseline)
3. **Class C — `tools/preflight_hook.py --changed-files-only` refactor** (HIGH
   operational, dissolves the thundering-herd lock contention)

(Class D stress tests + Class E will land alongside these as part of the diff.)

## Live counts pre-fix (0-violation strict-promotion plan)

- Class A: 4-5 unprotected codec modules. Fix path: add `[posenet-protected]`
  marker (or explicit waiver) to each before strict-promotion. Lands warn-only
  first, sweep, then strict.
- Class B: 0 violations after Lane GP v4 kill markers (already in repo).
  Lands STRICT @ 0 directly per Lane A pattern.
- Class C: refactor; benchmark before/after.

## Cross-refs

- `feedback_owv2_savings_correction_conv_vs_full_renderer_20260430.md` (Class A)
- `project_lane_gp_v4_killed_basis_fit_infeasible_20260430.md` (Class B)
- `project_swarm_recovery_state_20260430.md` (Class C)
- `feedback_concurrent_subagent_commit_message_swap_20260429.md` (Class E)
- `feedback_subagent_serializer_temp_index_landed_20260430.md` (Class E mitigation)
- `feedback_check_82_83_landed_council_round3_prescription_20260429.md` (Class D)
- `project_preflight_unblock_landed_591b7a43_20260430.md` (Class D fix)
