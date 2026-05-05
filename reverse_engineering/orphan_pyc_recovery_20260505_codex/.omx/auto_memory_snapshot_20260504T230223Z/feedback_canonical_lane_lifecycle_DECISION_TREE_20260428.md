---
name: CANONICAL lane lifecycle decision tree (recover-before-destroy + auth-eval-everywhere)
description: 2026-04-28 PM after Lane RM-d wasted $1.16 of 3.5h training because we destroyed without recovery, AND $10+ from NVDEC roulette. PERMANENT canonical workflow for every lane lifecycle event. Replaces ad-hoc destroy-on-failure with structured triage. Tools: recover_lane_artifacts.py, auth_eval_local.py, modal_auth_eval.py, Check 64 (E2E smoke proof), Check 65 (lane-class proof).
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The CANONICAL workflow

### Before dispatching any new lane class
1. Author lane script following `tools/canonical_lane_template.py`
2. **Run `experiments/canonical_local_auth_eval_smoke.py --lane <X>`** — proves E2E pipeline works locally on tiny synthetic data (~1.5s)
3. Smoke writes proof to `.omx/state/lane_e2e_smoke_proofs.json`
4. Without proof, **Check 64 STRICT preflight blocks dispatch**
5. If new LANE CLASS (e.g., pose-replacement, encoder-only): also need at least one entry in `.omx/state/lane_class_proofs.json` (Check 65 warn-only → STRICT after backfill)

### When dispatching to Vast.ai
1. Use `scripts/launch_lane_on_vastai.py` (V6+) — has Stage 0.5 lightweight NVDEC probe
2. Phase2-launch Stage 2 polls setup.log for ~60s — auto-destroys NVDEC_BAD hosts (saves ~$0.05 vs hours of idle billing)
3. If MOONSHOT or long-running (>6h): consider Modal instead via `experiments/modal_auth_eval.py` for reliability — $7/12h reliable vs $2.50/12h effective on Vast.ai at 30% NVDEC success

### When a lane crashes / underperforms / completes
**NEVER destroy without recovery.** New canonical destroy:
```bash
.venv/bin/python tools/recover_lane_artifacts.py <instance_id> --lane-label <label> --then-destroy
```
This:
1. SSH+SCP-pulls all archive-relevant artifacts (renderer.bin, masks.mkv, optimized_poses.pt, archive*.zip, run.log) to `experiments/results/recovered_<id>_<label>/`
2. Saves recovery metadata (timestamp, sizes, what was found)
3. THEN destroys instance
4. Idempotent — re-running refreshes

If instance is unreachable: `--no-recover` skips recovery (e.g., NVDEC_BAD where no training output exists)

### When auth eval crashes (like Lane RM-d's 0.mkv bug)
**Don't lose the trained artifacts!** New workflow:
1. Recover via `tools/recover_lane_artifacts.py`
2. Run auth eval LOCALLY: `.venv/bin/python tools/auth_eval_local.py --archive-dir experiments/results/recovered_<id>_<label>/`
3. Or run on Modal: `modal run experiments/modal_auth_eval.py --archive <path>`
4. The local/Modal runner has the F5 0.mkv guard built in — won't crash on the same bug

## What we lost today (cost analysis)

- Lane RM-d: 3.5h × $0.33/hr = **$1.16 wasted** (destroyed before recovery)
- 20+ NVDEC_BAD lanes: ~$10 wasted before auto-destroy fix (~$0.05 per host with fix)
- Q-FAITHFUL crashes: 2 × $0.27/hr × ~30 min each = ~$0.27 (Stage 0 failures, no training output to recover)
- Total today before fixes: **~$11**
- Total today after fixes: **<$1** for same volume of dispatches

## Why the bug class existed

Static-analysis preflight (63 STRICT checks before today) all guard CODE PATTERNS — never run the FULL pipeline locally. The "first lane of a new class" was always the canary. Lane RM-d was the canary for pose-replacement-only.

Check 64 (E2E smoke proof, STRICT) + Check 65 (lane-class proof, warn-only initially) close this gap PERMANENTLY for all future lane classes.

## When to use Modal vs Vast.ai (decision rule)

| Lane property | Vast.ai 4090 | Modal T4 |
|---------------|--------------|----------|
| Training < 4h, cheap moonshot | ✓ ($0.25/hr) | over-priced |
| Training > 6h, important | risky (NVDEC roulette) | ✓ ($0.59/hr reliable) |
| First-of-class lane (no proof yet) | NO — local smoke first | NO — local smoke first |
| Inference / auth eval (~30 min) | NO — random failures | ✓ |
| Moonshot we cannot afford to lose (Lane SZ Phase 2) | with backup recovery | ✓ canonical |

## Cross-references
- `tools/recover_lane_artifacts.py` — pull artifacts before destroy
- `tools/auth_eval_local.py` — run contest_auth_eval anywhere
- `experiments/modal_auth_eval.py` — Modal-portable
- `experiments/canonical_local_auth_eval_smoke.py` — pre-dispatch E2E smoke
- Check 64 STRICT — `.omx/state/lane_e2e_smoke_proofs.json` required per lane
- Check 65 warn-only → STRICT — `.omx/state/lane_class_proofs.json` required per class
- `feedback_artifact_recovery_canonical_workflow_20260428.md` — recovery workflow detail
- `project_modal_vs_vastai_reliability_analysis_20260428.md` — Modal cost rubric
- commit `169ecff4` — the canonical hardening
