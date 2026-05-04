# Codex 5-finding adversarial review — all 5 fixed (2026-04-28/29)

BINDING: After running /codex:adversarial-review and getting a
"needs-attention" verdict, work the findings as a checklist and commit
each as a separate `Codex F<N>: <fix>` commit. Bundle uncommitted
prior-session work that's load-bearing for the fix into the SAME commit
(don't strip it out — it's prerequisite). Add ONE regression test file
that anchors all findings with magnitude/value assertions.

## The 5 findings + canonical fix patterns

### F1 [HIGH] Vastai tracker leaked 75 live records (~$305 spend) into git
- **Symptom**: `.omx/state/vastai_active_instances.json` committed with
  75 active records. Other checkouts/CI could destroy live instances.
- **Root cause**: file was tracked, no .gitignore entry. Per-machine
  operational state must NEVER be in shared repo.
- **Canonical fix**: `git rm --cached <file>` (file stays LOCAL with
  records intact) + add to .gitignore. Same treatment for
  `instance_setup_first_seen.json`.
- **Pattern**: any file with per-machine live-state semantics (instance
  IDs, hostnames, ephemeral PIDs) belongs in .gitignore from day 1.

### F2 [MEDIUM] CLI flag parsed but never threaded → silent no-op
- **Symptom**: `experiments/qat_finetune.py` parsed `--protected-pattern-set
  {posenet_prior,segnet_prior}` but never wired into QATConfig. Lane SG
  runs were byte-for-byte identical to legacy QAT despite operator
  believing they were testing a different protection set.
- **Root cause**: argparse → action; action → never read. Same dead-flag
  class as `feedback_dead_flag_wiring_pattern` and pose_dim
  (`feedback_pose_dim_dead_resolver`) — the **dead-resolver pattern**.
- **Canonical fix**:
  1. Add field to QATConfig dataclass
  2. Thread `args.X` into QATConfig() construction
  3. Pass `cfg.X` to the actual function that needs it
  4. Function APPLIES the value (not just records it)
  5. Add a test that proves DIFFERENT values produce DIFFERENT behavior
- **Behavioral test pattern** (this is what makes the test load-bearing):
  build a sensitivity profile that ranks PoseNet-prior layers as
  FP4-tolerable, run with both `posenet_prior` and `segnet_prior`,
  assert the resulting bit allocations DIFFER.

### F3 [MEDIUM] caller's pattern list treated as ADDITIVE not REPLACEMENT
- **Symptom**: `swap_renderer_convs_with_self_compress` always used
  `SC_PROTECTED_NAME_PATTERNS` and added caller's list as extras. So
  `segnet_prior` produced PoseNet ∪ SegNet protection (both lists
  protected), NOT the disjoint SegNet-only set the docstring promised.
- **Root cause**: helper function had `extra_protected_patterns=` only,
  no replacement primitive. Operator's mental model ("protect ONLY this
  set") had no API surface.
- **Canonical fix**: add `protected_patterns=None` kwarg. When provided
  (non-None), it REPLACES the default list. `extra_protected_patterns=`
  still works as additive on top.
- **Behavioral test pattern**: build a real model with both lists' named
  layers, swap with each pattern_set, assert `posenet ∩ segnet == ∅`.
  This catches the additive bug (which would yield non-empty intersection).
- **Diagnostic surface fix**: return dict adds `protected_patterns_used`
  field so post-hoc audit can verify which list was applied.

### F4 [MEDIUM] BF16 falsely advertised on T4/P100
- **Symptom**: `get_supported_quantization_modes()` initialized
  `modes = {"fp16", "bf16"}` for ALL CUDA. T4 (CC 7.5) and P100 (CC 6.0)
  don't have hardware BF16 → silent FP32 fallback OR runtime error,
  defeating the fail-fast gate. AWS spot fleet uses T4.
- **Root cause**: capability mapping omitted the BF16-needs-Ampere fact.
- **Canonical fix**: gate `bf16` by CC >= 8.0. Update the requirements
  dict in `assert_quantization_hardware_supported` to name Ampere
  explicitly.
- **Test pattern**: mock `torch.cuda.get_device_capability` for each
  CC tier (6.0, 7.5, 8.0, 8.9), assert the right mode set.

### F5 [HIGH OPERATIONAL] Lane RM-d crashed at extracted/0.mkv missing
- **Symptom**: Lane RM-d trained 1+ hour, built archive successfully,
  then `contest_auth_eval` crashed at Stage 3 with "Error opening input
  file extracted/0.mkv. No such file or directory".
- **Root cause**: `scripts/launch_lane_on_vastai.py:_enumerate_python_and_shell`
  only included .py/.sh/.json/.toml/.md/.txt files, silently dropping
  `submissions/robust_current/config.env`. Without that file, inflate.sh
  could not read `PYTHON_INFLATE=renderer` and fell into legacy ffmpeg
  path that expects extracted/0.mkv (which never exists in renderer
  archives). Memory `feedback_partial_tarball_deploy_traps` had
  documented this exact bug class for Spain Lane M+N.
- **Canonical fix is LAYERED** (defense in depth):
  1. Launcher: add `.env` to `allowed_suffixes`
  2. `experiments/contest_auth_eval.py`: hard-fail at startup if
     `inflate_sh.parent / config.env` is missing OR
     `PYTHON_INFLATE=renderer` is not set (placed AFTER upstream check
     so existing tests preserve order)
  3. Lane scripts: defensive in-script grep before contest_auth_eval call
  4. NEW Check 63 in preflight.py: every `remote_lane_*.sh` calling
     contest_auth_eval MUST either route through canonical module OR
     have its own PYTHON_INFLATE pre-check
- **Pattern for "silently-missing canonical config"**: when a remote
  artifact needs a config file that the launcher might exclude, ALWAYS
  fail-fast inside the consumer (contest_auth_eval) AND in a static
  preflight check, not just in the launcher.

## Per-finding commit metadata

| Finding | Commit SHA | Tests added | Behavioral test | Live count |
|---------|-----------|-------------|-----------------|------------|
| F1      | 6be2daf0  | 1           | gitignore literal grep | n/a |
| F2      | 38ab14db  | 2           | DIFFERENT allocations on same profile | n/a |
| F3      | 70462819  | 4           | DISJOINT protected sets posenet vs segnet | n/a |
| F4      | 6cd20d0f  | 5           | mock CC=7.5/6.0/8.0/8.9 → expected mode set | n/a |
| F5      | f8a93912  | 3 + Check 63 | grep .env literal + config.env content | 0 violations |

All 15 regression tests in `src/tac/tests/test_codex_review_fixes_20260428.py`
pass. Total relevant suite (5 test files): 77 tests pass.

## Preflight catalog

Was 62 STRICT preflight checks. Now **63 STRICT** after Check 63
(`check_lane_scripts_set_up_inflate_environment`) lands at 0 live
violations (verified via `.venv/bin/python -c "from tac.preflight import
check_lane_scripts_set_up_inflate_environment; check_...(strict=False)"`).

## Redeploy outcome

Launched 4 instances (3 phase2-dispatch + 1 fresh Lane RM-V2 with F5 fix):
- 35802893: Lane H-V3-b retry (after first instance auto-destroyed by
  Check 59 on launch failure)
- 35802727: Lane F-V5-b retry (after first instance auto-destroyed by
  Check 59 on CUDA Error 803)
- 35803182: Lane RM-V2 retry (after first instance auto-destroyed by
  Check 59 on cuda_available=False)
- 35800070: Lane J-JBL-c phase2 still pending (sshd boot still slow)

Auto-destroy on bad-host (Check 59) worked correctly 3 times. Each
bad-host costs ~$0.01 because phase2-extract catches it immediately
after 2-3 min of boot, NOT after hours of training.

## Patterns that worked

- **Fail-fast at every layer** (launcher excludes file → consumer
  refuses to start → preflight check blocks future regressions). Don't
  rely on a single guard.
- **Layered defense with named entry points** (Check 63, F5 in
  contest_auth_eval, F5 in lane script) lets each layer be tested
  independently.
- **Behavioral tests anchored on REAL models** (F3) catch dead-flag
  cases that helper-output tests miss. Test the actual behavior on the
  actual entry point.
- **Bundle prerequisite uncommitted work** into the same commit when
  it's load-bearing for the test (e.g., F3 commit included the Lane SG
  helper additions to self_compress.py because the F3 test wouldn't
  compile without them).

## What NOT to do

- **DON'T** assume parsed flag = applied flag. ALWAYS write a
  behavioral test that proves DIFFERENT values produce DIFFERENT
  behavior on the actual entry point.
- **DON'T** add an `extras=` parameter when callers actually want a
  REPLACEMENT primitive. Mental model mismatch produces silent bugs.
- **DON'T** initialize a "default" set with hardware-conditional values
  (BF16) without per-CC gating.
- **DON'T** skip suffix in launcher's tarball file-list. Any file
  named `*.env`, `*.cfg`, `*.ini`, `*.yaml`, `*.yml` is potentially
  load-bearing — review the suffix list whenever a new lane fails
  with "missing config".
