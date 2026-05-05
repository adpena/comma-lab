---
name: Vast.ai dispatch failure modes — anchor-missing, PYBIN-wrong-venv, OOM-on-4090
description: 2026-05-01 ~12:30 UTC. 8 of 10 parallel Vast.ai dispatches FAILED in 4 distinct ways. Documenting the actual failure modes so future dispatches use the correct pattern.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The 4 failure modes I hit launching 10 parallel dispatches

### 1. macOS `tar -rzf` silently fails to append anchor data (60% of failures)

`scripts/launch_lane_on_vastai.py:add_lane_anchors_to_tarball` uses `tar -rzf` (append-to-gzip) which is **NOT supported on macOS bsdtar**. The function catches the error and just prints a WARN message, then continues. The lane script then exits rc=1 at preflight when it can't find anchor files.

**Symptom:** lane.log shows `FATAL: missing <anchor file path>` immediately after Stage 0 preflight. Instance appears "running" via vastai CLI but lane process is dead. Heartbeat continues alive (it's in a separate trap).

**Workaround:** SCP the anchor directories DIRECTLY after `phase2-extract` succeeds:
```bash
scp -P <port> -r <local_anchor_dir> root@ssh<n>.vast.ai:/workspace/pact/<remote_path>/
ssh -p <port> root@ssh<n>.vast.ai 'cd /workspace/pact && PYBIN=/opt/conda/bin/python WORKSPACE=/workspace/pact nohup bash <lane_script> > /workspace/lane.log 2>&1 &'
```

**Permanent fix:** rebuild the launcher to construct the tarball ONCE with both repo + anchors, OR use `--include` lists, OR use rsync.

### 2. Lane script picks `.venv` (empty `av`-only) instead of `/opt/conda` (full pytorch)

setup_full.sh creates BOTH `/opt/conda` (with full pytorch+CUDA) AND `.venv` (with `av` only for inflate_renderer). Most lane scripts default `PYBIN="$WORKSPACE/.venv/bin/python"` if `.venv` exists, picking the EMPTY venv. Result: `ModuleNotFoundError: No module named 'torch'`.

**Workaround:** explicitly set `PYBIN=/opt/conda/bin/python` when running the lane:
```bash
ssh ... 'PYBIN=/opt/conda/bin/python bash <script> ...'
```

**Permanent fix:** lane scripts should detect `.venv` has torch before preferring it (e.g., `python -c "import torch" 2>/dev/null && PYBIN=...` or test the import).

### 3. RTX 4090 (24GB) OOMs Q-FAITHFUL JointFrameGenerator

Q-FAITHFUL trains a JointFrameGenerator that exceeds 24GB on RTX 4090. Killed at epoch 10 with SIGKILL (exit 137).

**Workaround:** filter Q-FAITHFUL to A100 80GB / H100 80GB on Vast.ai (`gpu_name in [A100, H100]`) OR run on Lightning Studio.

**Permanent fix:** `scripts/remote_lane_q_faithful_jointgen.sh` should add a min-VRAM preflight check (require ≥40GB VRAM, fail fast on 4090).

### 4. Script with `set -u` and unbound `$PYBIN` (MAE-V)

`scripts/remote_lane_mae_v.sh:160` references `$PYBIN` without setting a default at the top of the script. With `set -u` (CLAUDE.md required), unbound `$PYBIN` triggers immediate exit at line 160.

**Workaround:** export PYBIN=/opt/conda/bin/python before invocation.

**Permanent fix:** every lane script should have the top-level `if [ -z "${PYBIN:-}" ]; then ... fi` block (β Fisher script has it, MAE-V doesn't).

## Pattern that DOES work (Q-FAITHFUL until OOM, β Fisher post-restart)

1. `phase1` succeeds → instance ID assigned
2. `phase2` succeeds → SCP + extract + auto-launch (BUT anchors silently missing on macOS)
3. SSH in → SCP anchor dirs directly
4. SSH in → `PYBIN=/opt/conda/bin/python nohup bash <script>` 
5. Heartbeat from lane script confirms alive
6. Wait for `contest_auth_eval.json` to land

Reliability ranking:
- **Lightning Studio** (per `feedback_lightning_ai_ssh_credentials_20260430.md`): no spot bidding, no NVDEC roulette, no macOS tar bug, persistent storage. **The reliable path.**
- **Vast.ai 4090** with manual anchor SCP + explicit PYBIN: 1/10 success rate naive, ~50% with workarounds.
- **Modal**: per CLAUDE.md, not authoritative for contest-grade scores (CPU-only auth eval). OK for harvesting.

## What this turn cost

10 instances launched, ~4 hours of cumulative GPU time wasted on failed setups before the failures were diagnosed. ~$2-3 lost. With the workarounds documented above, future dispatches should hit ~80%+ success.

## Cross-refs

- `feedback_lightning_ai_ssh_credentials_20260430.md` (Lightning is the reliability answer)
- `feedback_vastai_idle_waste.md` (sister failure pattern)
- `scripts/launch_lane_on_vastai.py:518` (the broken `add_lane_anchors_to_tarball` function)
- `scripts/remote_lane_g_v3_owv3_fisher_stack.sh:25-31` (the .venv-vs-conda PYBIN selection bug)
- `scripts/remote_lane_q_faithful_jointgen.sh` (needs ≥40GB VRAM, missing the preflight)
- `scripts/remote_lane_mae_v.sh:160` (PYBIN unbound, missing default)
