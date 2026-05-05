---
name: Vast.ai NVDEC roulette ≈85% bad-host rate today — Modal is the reliability path
description: 2026-04-29 night session: 5 rounds of TIER-1 dispatches (~$5), 0 lanes finished training. Vast.ai's NVDEC roulette + SSH unreliability dominated tonight. Modal T4 ($0.59/hr vs $0.25/hr) sidesteps both. Use Modal for >2h training runs; Vast.ai only when cheap moonshot + retry budget exists. Bug catalog from this session is also load-bearing.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Empirical reliability data (2026-04-29 night, ~6h session)

| Round | Lanes attempted | Survived to training | Failure modes |
|-------|-----------------|----------------------|---------------|
| v1 | 6 | 0 | git reset --hard wiped anchors |
| v2 | 6 | 0 | same bug, hadn't fully landed yet |
| v3 | 6 | 0 | anchor regex couldn't match `="${VAR:-...}"` |
| v4 (retry-wrapper) | 6 | 4 dispatched, 0 trained | UnboundLocalError shadowing import |
| v5 (bug fixed) | 6 | 1 dispatched, 0 trained | NVDEC roulette + Lane J corpus issue |

**Empirical Vast.ai 4090 NVDEC bad-host rate today: ~85%.** Same offer filter, same image — different host = different NVDEC capability. Even with `--max-retries 3` per lane, 4/5 lanes exhausted retries. Need `--max-retries 6+` to get >50% survival.

## Decision rule (UPDATED)

| Lane property | Vast.ai 4090 ($0.25/hr) | Modal T4 ($0.59/hr) |
|---------------|------------------------|---------------------|
| Training < 2h, cheap moonshot | ✓ with retry-wrapper + 3 retries | over-priced |
| Training 2-6h, validated lane | risky on bad NVDEC night | ✓ reliable |
| Training > 6h, important | NO — too risky | ✓ canonical |
| First-of-class lane (no proof) | NO — pre-deploy E2E smoke first | NO — same |
| Auth eval (~30 min) | NO — random failures | ✓ |
| Moonshot we cannot afford to lose | NO unless retry budget = $50+ | ✓ canonical |

When Vast.ai NVDEC bad-host rate is high (>50% per round), **don't dispatch.** Wait or pivot to Modal.

## Bug catalog from tonight (8 commits, all permanently prevented)

These are the bugs each round of dispatches surfaced. All have STRICT preflight checks now (Checks 66-71).

1. **`git reset --hard origin/main` wipes local-only anchors** — commit 705ebad7 + Check 66 STRICT
2. **Anchor regex couldn't match `="${VAR:-...}"` form** — commit 28155a8a (the canonical lane idiom)
3. **phase2-wait required `--lane-script`** — commit 6320375c (made optional)
4. **`from X import Y` inside func body shadows module import → UnboundLocalError** — commit 56da3edf + Check 71 STRICT
5. **phase2-launch only polled NVDEC for 60s, missed post-setup crashes** — commit 8c7d0033 (extended to 240s + LANE_CRASHED detection via run.log freshness)
6. **`pgrep -f python` self-matches** — preflight blocked at write-time (memory `feedback_pose_loader_and_preflight_2026_04_26`)
7. **launcher's `subprocess.run` without `check=`** — preflight blocked, fixed inline

## Permanent prevention infrastructure landed

72 STRICT preflight checks total (was 65 at session start). Total proactive cost ~3s.

| Check | What it catches |
|-------|-----------------|
| 66 | `git reset --hard` in remote_lane_*.sh |
| 67 | py_compile fails (631 .py in 0.75s) |
| 68 | bash -n fails (109 .sh in 0.45s) |
| 69 | ANCHOR_* paths must exist locally |
| 70 | pytest --collect-only must succeed |
| 71 | shadowed-module-import-before-use (UnboundLocalError trap) |

Plus:
- `scripts/launch_lane_with_retry.py` — wraps phase1+phase2 with auto-retry on phase2 failures
- `phase2-launch` Stage 2 (240s) detects: NVDEC_BAD | LANE_CRASHED | SETUP_COMPLETE | RUNNING

## SZ paradigm fundamental incompatibility

The 5h+ SZ-Phase2-c training produced a remarkable 3KB renderer (92,371 params packed to 3,028 bytes). BUT:

1. **Renderer-only archive is non-compliant** per Yousfi PR #35 — needs SegNet at inflate time → would need 48MB scorer in archive (defeats the rate gain)
2. **SZ paradigm uses pose_dim=1** (zoom scalar) — incompatible with 6-DOF G v3/Lane A poses
3. **SZ paradigm masks reconstructed from luma in-renderer** — needs custom inflate path (not yet wired)

For SZ to score meaningfully:
- Train SZ-style renderer with pose_dim=6 + use Lane G v3 masks/poses (rate ~0.30, distortion TBD)
- OR wire luma-mask-reconstruction into inflate_renderer.py

Both deferred to next session.

## Cross-references

- Memory `feedback_canonical_lane_lifecycle_DECISION_TREE_20260428` — recover-before-destroy + auth-eval-everywhere
- Memory `feedback_git_reset_nukes_anchors_20260429` — the canonical-git-sync-pattern was wrong
- Memory `feedback_vastai_nvdec_host_variation` — original NVDEC variability finding
- Recovered SZ artifacts: `experiments/results/recovered_35793092_lane_sz_phase2_c/`
- Modal auth eval runner: `experiments/modal_auth_eval.py`
- Retry wrapper: `scripts/launch_lane_with_retry.py`
