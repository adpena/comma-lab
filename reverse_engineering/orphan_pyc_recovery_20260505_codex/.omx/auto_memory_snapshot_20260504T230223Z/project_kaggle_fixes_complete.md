---
name: Kaggle Deployment Fixes Complete
description: Three root causes found and fixed. Mount path, bootstrap scope, read-only filesystem. v8 running.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Three Root Causes (all fixed)

1. **Bootstrap __main__ guard** (all versions before v10): _kaggle_setup() inside `if __name__ == "__main__"` — Python parses imports before executing __main__, so torch/click fail before setup runs. Fix: call at module scope.

2. **Dataset mount path** (v10+): Kaggle changed from /kaggle/input/<slug>/ to /kaggle/input/datasets/<owner>/<slug>/. Debug kernel confirmed. Fix: search both paths.

3. **Read-only filesystem** (v6-v7): Training script writes to /kaggle/src/results/ which is read-only. Fix: symlink to /kaggle/working/ or TAC_RESULTS_DIR env override.

## Current State
- v8 pushed with all 3 fixes, status: RUNNING
- Preflight retry loop (5x 60s) handles dataset propagation delay
- Debug kernel (no GPU) available for quick mount verification

## Deployment Checklist (CLAUDE.md)
1. Bump pyproject.toml version
2. Update deploy_config.py BASE_FLAGS
3. Rebuild wheel
4. Upload to Kaggle dataset, wait 30+ min
5. Build kernels, push
6. Monitor for >30 min before declaring success
