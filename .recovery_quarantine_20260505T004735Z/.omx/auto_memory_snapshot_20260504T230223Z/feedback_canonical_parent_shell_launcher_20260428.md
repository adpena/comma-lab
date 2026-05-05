---
name: scripts/launch_lane_on_vastai.py — canonical parent-shell Vast.ai launcher
description: 2026-04-28 built the canonical launcher addressing all 5 Cycle 1 failure modes. Single deterministic command takes a lane script path → finds offer → creates instance → builds tarball → SCPs → extracts → CUDA probe → tmux execution → 8-min heartbeat verify → registers tracker → fail-loud destroy on any error. Replaces ad-hoc launch attempts.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Usage

```bash
.venv/bin/python scripts/launch_lane_on_vastai.py \
  --lane-script scripts/remote_lane_omega_v2_lagrangian.sh \
  --label lane_omega_v2 \
  --predicted-band 0.70 0.95 \
  --estimated-cost 1.50 \
  --council-priority 1
```

Optional flags:
- `--dry-run`: search offer + verify env, don't create instance (smoke test)
- `--max-dph 0.50`: cost cap (default $0.50/hr)
- `--anchor-dirs experiments/results/lane_a_landed/iter_0 ...`: extra dirs in tarball
- `--council-priority 1`: tracker metadata

## 9 stages

0. Find offer (cheapest non-KR 4090 with reliability > 0.96, max $0.50/hr)
1. Create instance with --label + --ssh
2. Wait 90s for boot
3. Wait for SSH ready (poll up to 3 min)
4. Build minimal tarball (~175MB) excluding noise (.venv, .git, results/, precomputed/, kaggle_kernels/, archives/, work/)
5. SCP tarball to /workspace/pact.tar.gz
6. Extract into /workspace/pact + delete tarball
7. **Lightweight CUDA probe** (pre-DALI): `nvidia-smi` + `torch.cuda.is_available()` — catches NVDEC-broken hosts before 30-min DALI install
8. Start `bash remote_setup_full.sh && bash <lane>` in tmux (so parent disconnect doesn't kill)
9. Poll heartbeat for 8 min — destroy on timeout

Tracker registration happens RIGHT AFTER instance creation (stage 1) so any subsequent failure can clean up. Removal happens automatically on launch failure path.

## Failure modes addressed (from feedback_cycle_1_launch_postmortem_20260428)

| Failure | Mitigation |
|---------|-----------|
| codex sandbox blocks vast.ai DNS | Runs in parent shell only |
| Wrong setup script path | Hardcodes `scripts/remote_setup_full.sh` |
| Bash bg killed at 144 mid-rsync | Tarball SCP (single network call, not multi-rsync) |
| NVDEC host variability | Stage 7 lightweight probe BEFORE DALI install |
| probe_nvdec.sh needs DALI installed | Stage 7 uses raw `torch.cuda` check, no DALI |
| Idle instance burning $$$ | Stage 9 timeout → auto-destroy |
| Lost work on disconnect | tmux detaches; parent can disconnect freely |

## Companion: scripts/verify_vast_instances.py

Run periodically to maintain fleet health:
```bash
.venv/bin/python scripts/verify_vast_instances.py --auto-destroy-stale --stale-minutes 30
```

## Smoke-tested

`--dry-run` passed: found offer 25753562 < $0.50/hr in seconds. Real launches deferred until next session per user's pacing.

## Cross-references
- `feedback_cycle_1_launch_postmortem_20260428` — the 5 motivating failures
- `feedback_codex_sandbox_blocks_vastai_dns_20260428` — why parent-shell only
- `feedback_remote_setup_script_correct_path_20260428` — script path metabug
- `feedback_per_instance_verify_pattern_20260428` — runtime watchdog companion
- `feedback_vastai_launch_returns_success_before_lane_starts` — heartbeat as canonical
- `feedback_canonical_remote_bootstraps` — remote_setup_full.sh standard
- `feedback_vastai_nvdec_host_variation` — why probe early
