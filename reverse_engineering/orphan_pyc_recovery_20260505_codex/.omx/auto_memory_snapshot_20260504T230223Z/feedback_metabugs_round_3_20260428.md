---
name: 3 new metabugs found in V5 launcher round (2026-04-28 PM)
description: 3 lanes deployed via V5 launcher; each found a NEW metabug. (1) Missing upstream/*.py files (frame_utils not in tarball) — fixed via auto-include all upstream/*.py. (2) NVDEC probe fails AFTER 5-min DALI install — should be pre-DALI lightweight check. (3) Lane scripts source env.sh which doesn't exist if setup fails — should fail loud earlier.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Metabug A: tarball missed upstream/*.py imports

**Symptom**: Lane Ω-V2 crashed at lane.log: `ModuleNotFoundError: No module named 'frame_utils'`. Setup_full.sh COMPLETED ✓ but lane runtime needs `from upstream.frame_utils import ...` and frame_utils.py wasn't in the tarball.

**Root cause**: V5 launcher's canonical includes were `upstream/{evaluate,modules,__init__}.py` only, not all `upstream/*.py`. `frame_utils.py` was missing.

**Fix**: V5 launcher now auto-includes ALL `upstream/*.py` files (small, ~30KB total). Adding new upstream module → auto-included.

**Permanent guard (TIER-1 next session)**: Check 44 — scan for `from upstream.X import` and `import upstream.X` in all .py + .sh; verify `upstream/X.py` exists locally and is in launcher includes.

## Metabug B: NVDEC probe runs AFTER 5-min DALI install — FIXED 2026-04-28 commit 58e55890

**Symptom**: Lane EC failed at setup_full.sh Stage 4 (NVDEC probe). Probe is correct (caught the bad host) but burns 5 min of DALI install before failing.

**Root cause**: `scripts/probe_nvdec.sh` requires DALI installed — runs in `setup_full.sh` AFTER apt deps + DALI pip install (~5 min total). Failure costs $0.05+ per attempt.

**FIX LANDED** (commit 58e55890): `probe_nvdec.sh --lightweight` flag dlopens `libnvcuvid.so.1` + `libcuda.so.1` via ctypes, calls `cuvidGetDecoderCaps()` for H264/8bit/4:2:0, exits in ~3s with zero install cost. `remote_setup_full.sh` Stage 0.5 runs it BEFORE the 5-min DALI install at Stage 3. Classification dispatch reuses OK/NVDEC_MISSING/UNKNOWN exit codes from the deep probe.

A pass at Stage 0.5 does NOT skip the deep DALI-based probe at Stage 4 (still authoritative for `fn.experimental.inputs.video` MIXED operator). The lightweight catches ~95% of NVDEC-missing hosts at zero cost; the deep catches edge cases like NVDEC-engine-present-but-DALI-version-mismatch.

**Permanent guard**: existing `check_remote_scripts_have_nvdec_probe` (Check 11) already enforces probe-before-GPU-work. The Stage 0.5 lightweight probe satisfies it AND provides cost protection.

## Metabug C: Lane scripts hard-source env.sh, fail noisy without it

**Symptom**: Lane EC's `lane.log` showed: `scripts/remote_lane_ec_engineered_corrections.sh: line 54: /workspace/pact/env.sh: No such file or directory`. The lane script ran AFTER setup_full.sh failed (because nohup wrapper continues regardless of setup exit code).

**Root cause**: `setup_full.sh` writes `env.sh` only at SUCCESS (Stage 7). When NVDEC probe fails at Stage 4, env.sh never written. Lane script then sources non-existent file.

**Fix candidate**:
- Update wrapper script (`run_lane.sh` written by `execute_lane_in_tmux`) to check `setup_full.sh` exit code BEFORE running the lane: `bash setup_full.sh && bash lane.sh` instead of running both regardless.
- OR: lane scripts pre-check `[ -f env.sh ] || { echo "FATAL: env.sh missing — setup likely failed" >&2; exit 1; }`.

**Permanent guard**: Update launcher's `execute_lane_in_tmux` to make `run_lane.sh` use `&&` between setup and lane (currently uses `;`).

## Cross-references
- `feedback_launcher_v5_auto_discovery_tarball_20260428` — V5 launcher
- `feedback_check_43_tarball_anchor_parity_20260428` — sibling check class
- `feedback_vastai_nvdec_host_variation` — NVDEC variability
- `feedback_canonical_remote_bootstraps` — setup_full.sh canonical
