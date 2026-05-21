# Codex Findings: HFV Same-Lane Dispatch Claim Unblock

**UTC:** 2026-05-21T20:31:23Z  
**Lane:** `hfv_sidecar_frontier_decision_packet`  
**Verdict:** PROCEED — HFV9 exact-eval dispatch is no longer blocked by unrelated live claims.

## What changed

`tools/plan_hfv_sidecar_frontier_decision.py` now reads the dispatch claim ledger and distinguishes:

- same-lane active claims for the recommended HFV paired exact-eval lanes, which still block dispatch;
- unrelated active claims, which remain coordination warnings but no longer block HFV exact eval.

The previous packet used a global `active_dispatch_claim_count > 0` blocker. After stale Modal rows were reconciled, that policy became over-conservative: the remaining live Selfcomp A100 call is unrelated to HFV9, and the stale local master-gradient row is not an HFV exact-eval lane.

## Live decision packet

Output:

- `experiments/results/hfv_sidecar_frontier_decision_packet_20260521T203123Z/hfv_sidecar_frontier_decision_packet.json`
  - SHA-256: `0143ae4805959bc40f73936e385252994f3f9d931f03a61be65f742060a4e6e0`
- `experiments/results/hfv_sidecar_frontier_decision_packet_20260521T203123Z/hfv_sidecar_frontier_decision_packet.md`
  - SHA-256: `c3fdacc5c714b029b8653f7fc450a3d490a691dc2a3384cd4010ba9b085aaa64`

Key fields:

```json
{
  "dispatch_blocked": false,
  "dispatch_blockers": [],
  "active_dispatch_claim_count": 2,
  "active_dispatch_claims_known": true,
  "same_lane_conflicts": 0,
  "unrelated_active": 2,
  "recommended": "hfv9_magic_explicit_row"
}
```

Recommendation remains `hfv9_magic_explicit_row`:

- archive bytes: `178553`
- archive SHA-256: `9a32b1311da1076b1659ff6652481383527279905c8a135eeedff6ec913888ac`
- compliance profile: `row_archive_contained_magic_identified`

## Selfcomp A100 poll

The live Selfcomp A100 job is still not terminal:

```text
call_id=fc-01KS5YG9W26T72D6Z8Y3N44JEN
lane=lane_overnight_xx_selfcomp_tier_2_paid_modal_a100_first_anchor_dispatch_20260521
status=not_ready
```

Poll artifact:

- `experiments/results/selfcomp_a100_harvest_poll_20260521T202908Z.json`

## Verification

Commands:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  tools/plan_hfv_sidecar_frontier_decision.py \
  src/tac/tests/test_plan_hfv_sidecar_frontier_decision.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_plan_hfv_sidecar_frontier_decision.py -q

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/plan_hfv_sidecar_frontier_decision.py \
  --output-dir experiments/results/hfv_sidecar_frontier_decision_packet_20260521T203123Z
```

Result:

- planner unit tests: `4 passed`;
- live packet: `dispatch_blocked=false`;
- exact eval not yet executed in this memo.

## Next action

Dispatch paired Modal exact eval for `hfv9_magic_explicit_row` on contest CPU and contest CUDA. The remaining active claims are unrelated to the HFV exact-eval lanes, so they are coordination warnings rather than blockers.
