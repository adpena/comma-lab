# Loop-session permanent bug-class extinction (2026-05-01)

Eight bug classes burned a Vast.ai instance dispatch each (~$0.30 + 5-10 min
forward velocity) during the 2026-05-01 loop session. This memo documents
each bug class plus the permanent code-level fix + STRICT preflight check.

## Why "permanent" not "patched"

User mandate: "always permanently ensure all bugs and bug classes and metabugs
fixed permanently". For each bug below: the code fix lands in the canonical
path AND a static-detectable preflight check ensures any future regression is
blocked at commit time (warn-only first commit, flip STRICT after sweep).

## Bug class inventory + permanent fixes

### Bug Class #1 — uv not on PATH on fresh `pytorch:cuda` Vast.ai image

**Symptom**: `bash: uv: command not found` on Stage 1.
**Root cause**: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel ships no uv binary.
**Fix in code**: `scripts/ensure_remote_uv.sh` (canonical bootstrap, already
extant). Used by `scripts/remote_archive_only_eval.sh:46-58 bootstrap_runtime_deps()`.
**Preflight**: PCC5 `check_remote_archive_eval_self_bootstraps_uv_and_ffmpeg`.
**Reference**: `feedback_uv_not_on_path_vast_instance_20260501.md`.

### Bug Class #2 — Vast.ai CUDA driver too old → uv-default torch wheel triggers silent CPU fallback

**Symptom**: `inflate.sh` runs to completion but contest_auth_eval reports
PoseNet=NaN; investigation reveals torch was installed as the CPU wheel.
**Root cause**: driver_major < 580 cannot satisfy cu13 wheel; uv falls back
to CPU wheel without raising.
**Fix in code**: `scripts/remote_archive_only_eval.sh:88-95` auto-pins
`INFLATE_TORCH_SPEC=torch==2.5.1+cu124` for driver_major < 580 + sets
`UV_EXTRA_INDEX_URL` and `UV_INDEX_STRATEGY=unsafe-best-match`.
**Preflight**: indirectly covered by PCC5 (the bootstrap function carries
the pin). The driver detection is itself self-bootstrap.
**Reference**: `feedback_vast_cuda_driver_too_old_silent_cpu_fallback_20260501.md`.

### Bug Class #3 — `.venv/bin/python: No module named pip`

**Symptom**: probe_nvdec.sh DALI install path crashes with `No module named pip`.
**Root cause**: `uv venv` does NOT install pip; downstream `python -m pip ...`
fails before any helpful error.
**Fix in code**:
1. NEW: `scripts/ensure_remote_pip.sh` — canonical pip bootstrap (companion
   to `ensure_remote_uv.sh`). Idempotent. Pip-already-present fast path.
2. `scripts/probe_nvdec.sh` — auto-invoke `ensure_remote_pip.sh` BEFORE the
   pip install path so a fresh-host probe self-heals.
3. `scripts/remote_lane_nwc.sh` — invokes `ensure_remote_pip.sh` immediately
   after `uv venv` creation.
**Preflight**: PCC6 `check_venv_creators_use_ensurepip` — every `uv venv` /
`virtualenv` call must have `ensurepip` / `ensure_remote_pip.sh` / `uv pip
install` / `# NO_PIP_NEEDED:` annotation within 12 lines. (`python -m venv`
ships pip in stdlib bundle and is not flagged.)
**Live count after fix**: 0 violations.

### Bug Class #4 — System ffmpeg 4.4.2 lacks `in_primaries` scale option

**Symptom**: `inflate.sh require_ffmpeg_parity` errors: "scale filter is
missing required option 'in_primaries'".
**Root cause**: Ubuntu 22.04 ships ffmpeg 4.4.2 without modern color filter
options that `submissions/robust_current/inflate.sh` requires.
**Fix in code**: `scripts/remote_archive_only_eval.sh:132-165` auto-downloads
BtbN ffmpeg static build with retry-on-truncation when system ffmpeg lacks
the required scale options.
**Preflight**: indirectly covered by PCC5 (`bootstrap_runtime_deps` carries
the BtbN download).

### Bug Class #5 — macOS `._*` resource forks in `upstream/videos/`

**Symptom**: `experiments/contest_auth_eval.py` validator errors:
"uncompressed-dir contamination".
**Root cause**: SCP'd tarball from a macOS local laptop carries
AppleDouble files alongside `.mkv` videos.
**Fix in code**: `scripts/remote_archive_only_eval.sh:69-72` strips
`upstream/**/._*` and `.DS_Store` files at Stage 0 of bootstrap.
**Preflight**: pre-existing Check 37 `check_lane_scripts_strip_macos_resource_forks`.

### Bug Class #6 — 30 GB Vast.ai disk too small for 6-candidate chain eval

**Symptom**: chain eval crashes at candidate #5 with "no space left on device".
**Root cause**: 5 GB uv-managed torch wheels + 6 × 3.6 GB inflated frames =
27 GB working set, hitting 30 GB ceiling once OS + container overhead is
counted.
**Fix in code**:
1. `scripts/launch_lane_on_vastai.py` — bumped default `--disk` from 35 → 60,
   added `--min-disk-gb` flag, `find_offer` floors `disk_space>min_disk_gb`,
   and `create_instance` warns when `disk_gb < 60`.
2. `src/tac/deploy/base.py InstanceSpec.disk_gb` — bumped default from 40 → 60.
3. `scripts/check_vastai.py` — bumped hardcoded `--disk 40` → `--disk 60`.
**Preflight**: PCC7 `check_vastai_create_uses_min_disk_60` — every
`vastai create instance` invocation must use `--disk >= 60` OR carry
`# SINGLE_CANDIDATE_DISK_OK: <why>` annotation. Python `--disk str(int(disk_gb))`
parameterized form is allowed (the dataclass-default check covers it).
**Live count after fix**: 0 violations.

### Bug Class #7 — Per-candidate disk cleanup missing in chain drivers

**Symptom**: same as Bug Class #6 surface — disk fills mid-chain.
**Root cause**: chain driver loops `for entry in CANDIDATES` and runs
`remote_archive_only_eval.sh` per candidate. Each leaves
`LOG_DIR/eval_work/inflated/` (~3.6 GB) on disk.
**Fix in code**: `experiments/results/lane_g_v3_owv3_wave3_refinement_20260501/wave3_chain_driver.sh`
adds `rm -rf eval_work/{inflated,extracted,archive.zip}` at the end of
each iteration.
**Preflight**: PCC8 `check_remote_chain_drivers_clean_inflated_per_candidate`
— any `*chain*.sh` file with a for-loop calling `remote_archive_only_eval.sh`
must `rm -rf` `eval_work/inflated` (or use `--no-keep-work-dir` / annotate).
**Live count after fix**: 0 violations.

### Bug Class #8 — Vast.ai instance preemption mid-chain

**Symptom**: instance silently destroyed mid-chain; logs end abruptly.
**Root cause**: spot-tier offers can be preempted; chain has no checkpoint
or recovery mechanism.
**Workaround (not yet permanent)**: prefer `interruptible=False` in
`vastai create` when available; or fall back to Modal which has persistent
volumes. The current tactical fix is to keep chain durations < 30 min so
preemption windows don't overlap; the Modal recovery harness (memory
`feedback_modal_spawn_result_cache_pattern_20260429.md`) is the
strategic counterpart.
**Preflight**: not added in this commit — needs a `--interruptible=False`
flag scan once the Vast.ai CLI surface is verified to support it. Tracked
as deferred work.

## STRICT preflight checks landed (PCC5-PCC8)

All 4 ship `strict=False` (warn-only) on this commit. Per the Lane A pattern
in CLAUDE.md "Meta-bug class catalog (strict-mode preflight)", once a check
sits at 0 live violations across the codebase, flip `strict=True` in
`preflight_all()`.

| Check | Live count after fix | Promotion candidate |
|-------|----------------------|---------------------|
| PCC5  | 4 (4 NWC scripts)    | After NWC scripts adopt bootstrap_runtime_deps |
| PCC6  | 0                    | NEXT COMMIT |
| PCC7  | 0                    | NEXT COMMIT |
| PCC8  | 0                    | NEXT COMMIT |

## Files modified in this commit

Code:
- `scripts/ensure_remote_pip.sh` (NEW, 60 lines)
- `scripts/probe_nvdec.sh` (self-heal pip path, +14 lines)
- `scripts/remote_lane_nwc.sh` (post-`uv venv` ensurepip, +9 lines)
- `scripts/launch_lane_on_vastai.py` (--min-disk-gb=60 default, ~30 lines)
- `scripts/check_vastai.py` (--disk 40 → 60)
- `src/tac/deploy/base.py` (InstanceSpec.disk_gb 40 → 60)
- `experiments/results/.../wave3_chain_driver.sh` (per-cand cleanup)
- `src/tac/preflight.py` (4 NEW STRICT-eligible checks PCC5-PCC8, +400 LOC)

Tests:
- `src/tac/tests/test_check_loop_session_extinction_pcc5_pcc8.py` (NEW, 26 cases)

## Cross-references

- `feedback_uv_not_on_path_vast_instance_20260501.md` (Bug Class #1)
- `feedback_vast_cuda_driver_too_old_silent_cpu_fallback_20260501.md` (Bug Class #2)
- `feedback_vastai_dispatch_failures_20260501.md` (general loop-session)
- `feedback_vastai_correct_launch_pattern.md` (canonical launcher pattern)
- `feedback_canonical_remote_bootstraps` (the bootstrap discipline)

## Verification

```bash
.venv/bin/python -c "from tac.preflight import preflight_all; preflight_all(check_codebase=True, verbose=False); print('PREFLIGHT_ALL OK')"
.venv/bin/python -m pytest src/tac/tests/test_check_loop_session_extinction_pcc5_pcc8.py -q
# 26 passed
```

## What would change my mind (KILL retraction criteria)

If a future Vast.ai dispatch hits ANY of bug classes #1, #3, #4, #5, #6, #7
again, this extinction is not permanent and a deeper root-cause hunt is
required. Bug classes #2 and #8 are tactical-only and known incomplete.
