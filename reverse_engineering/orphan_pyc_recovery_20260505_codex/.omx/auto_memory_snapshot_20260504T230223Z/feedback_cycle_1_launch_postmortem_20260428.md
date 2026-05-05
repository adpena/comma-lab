---
name: Cycle 1 launch post-mortem — 5 separate operational failure modes blocked all 3 launches
description: 2026-04-28 attempted Lane Ω-V2 + Lane EC + Lane SAUG-V2 dispatch from skunkworks Cycle 1 plan ($5.80 / 14h). All 3 failed for different operational reasons. Total wasted spend ~$1.50 across 3 destroyed instances. Root cause: launcher infrastructure has too many fragile pieces. The "skunkworks plans → instant launch" workflow is not yet operational.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Cycle 1 plan (skunkworks, project_lane_g_v3_stacking_skunkworks_20260428)

| Lane | Predicted | Cost cap | Result |
|------|-----------|----------|--------|
| Ω-V2 | [0.70, 0.95] | $1.50 | DESTROYED — sync interrupted, NVDEC needs DALI install |
| EC | [0.85, 1.05] | $0.62 | DESTROYED — same |
| SAUG-V2 | [0.60, 0.90] | $4.32 | DESTROYED — NVDEC probe failed (no CUDA visible in container) |

Total spent: ~$1.50 (instances were running for ~10-15 min each at $0.241-0.309/hr).

## 5 distinct operational failures encountered

### 1. codex:rescue subagent sandbox blocks vast.ai DNS
Memory: `feedback_codex_sandbox_blocks_vastai_dns_20260428`. Subagents can't reach console.vast.ai. Launches MUST come from parent shell.

### 2. Wrong setup script path: `setup_full.sh` vs `remote_setup_full.sh`
Memory: `feedback_remote_setup_script_correct_path_20260428`. Parent-side instruction-quality bug.

### 3. Bash background tasks killed at exit code 144 mid-rsync
Harness kills background rsyncs around 5 minutes regardless of `timeout` parameter. 175MB sync over public internet to ssh9.vast.ai (Netherlands) takes longer. **Fix candidates**:
- Compress + scp single tarball instead of rsync
- Run in foreground (synchronous, blocking) — guarantees completion at the cost of UX
- Build a local launcher daemon that survives parent restarts

### 4. NVDEC variability per host (known per `feedback_vastai_nvdec_host_variation`)
Same 4090 image, same driver, but Netherlands ssh9 host had "no CUDA-capable device detected" inside the container. The probe correctly caught this AFTER setup_full.sh ran 30+ min of installs. **Cost saved**: probe fails fast (per `check_remote_scripts_have_nvdec_probe`).

### 5. probe_nvdec.sh requires DALI installed FIRST
Standalone NVDEC probe pre-DALI-install fails with "FATAL: nvidia.dali not installed". Either:
- Add `--ensure-dali` flag invocation to the canonical probe pattern
- OR run a lighter probe (e.g., just `nvidia-smi` + `python -c "import torch; print(torch.cuda.is_available())"`) before installing the heavy DALI dependency

## What still works

- **Lane G v3 frontier (1.05) is harvested locally** + already in repo at `experiments/results/lane_g_v3_landed/`
- **4 healthy in-flight instances continue training** (Lane I, V, M-V2, F-V3 v2) — these used a different launcher pattern (probably `src/tac/deploy/vastai/client.py` from a successful past session)
- **Skunkworks Cycle 1 plan is documented + still actionable** — the underlying lane code is correct; only the launcher pipeline is broken

## Hardening proposals (TIER-1 follow-up, NOT done in this session)

1. **Build canonical parent-shell launcher script** (`scripts/launch_lane_on_vastai.sh`) that:
   - Picks offer with retry on stale offer IDs
   - Compresses + SCPs tarball (not rsync — single network call)
   - Runs setup_full + lane in tmux (so parent disconnect doesn't kill)
   - Polls heartbeat for 8 min with destroy-on-timeout
   - Auto-registers in tracker
2. **Pre-DALI NVDEC sanity check**: lightweight probe BEFORE the 30-min setup
3. **Build watchdog**: scan tracker every 5 min, destroy any instance with stale heartbeat (>30 min)
4. **Document `src/tac/deploy/vastai/client.py` canonical launcher pattern** — that's how the 4 healthy instances were launched; we should use the same pattern not roll our own

## Recommendation for next session

- Re-launch Cycle 1 (Ω-V2, EC, SAUG-V2) using the canonical `tac.deploy.vastai.cli launch` if it works for these scripts, OR build the proper parent-shell launcher first
- Continue harvesting in-flight Vast.ai work (Lane I, V, M-V2, F-V3 v2)
- Wait for Round 11 codex fix subagent to complete
- Wait for 8 council EUREKA deploy scripts codex to land

## Cross-references
- `project_lane_g_v3_stacking_skunkworks_20260428` — the Cycle 1 plan that didn't deploy
- `feedback_vastai_nvdec_host_variation` — known NVDEC variability
- `feedback_codex_sandbox_blocks_vastai_dns_20260428` — sandbox isolation
- `feedback_remote_setup_script_correct_path_20260428` — instruction-quality bug
- `feedback_canonical_remote_bootstraps` — canonical pattern
- `feedback_oneshot_vastai_subagent_failure_pattern` — broader subagent-launch issues
