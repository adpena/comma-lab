# Dispatch Protocol Complete - Boyd Feasibility Umbrella - 2026-05-15

Operator insight: individual gates catch individual slips, but dispatch needs
the conjunction. A trainer can satisfy `min_vram_gb` while missing
`autocast_fp16`; a recipe can declare `pyav_decode_strategy` while the driver
omits Modal/NVML hygiene; a META decorator can exist while stale CLI routing
still fires. The canonical fix is a feasibility-set umbrella:

```text
dispatch_protocol_complete =
  tier1_engineering
  AND tier2_hardware_correctness
  AND tier3_substrate_correctness
```

Implementation landed:

- `src/tac/deploy/dispatch_protocol.py` defines the structured report and
  `require_dispatch_protocol_complete(...)`.
- `tools/operator_authorize.py` calls the umbrella for native provider/local
  dispatches before local pre-deploy, native provider preflight, lane-claim
  creation, or provider dispatch.
- `tools/check_dispatch_protocol_complete.py` exposes the same check as a
  JSON CLI for operator/runbook use.
- `src/tac/tests/test_dispatch_protocol_complete.py` covers clean, missing
  Modal env hygiene, missing trainer optimization/auth-eval tier flags,
  strict refusal, and non-native/no-op plan surfaces.
- `src/tac/tests/test_operator_authorize_canonical_tool.py` pins the runtime
  order: dispatch protocol -> local pre-deploy -> provider preflight -> lane
  claim -> dispatch.

Tier mapping:

- Tier 1 engineering: dispatch enabled, no declared blockers, canonical lane
  id, native platform support, cost-band epoch budget, remote driver, trainer.
- Tier 2 hardware correctness: `min_vram_gb`, `min_smoke_gpu`,
  `video_input_strategy`, `pyav_decode_strategy`, `target_modes`,
  `canary_status`, GPU capability order, and Modal env hygiene.
- Tier 3 substrate correctness: autocast/TF32/torch.compile/no-grad coverage
  or explicit waivers, plus canonical auth-eval helper for promotion-surface
  contest/production recipes.

Scope choice:

This is a runtime dispatch gate, not a broad repo-wide strict preflight scan.
That avoids blocking unrelated partner WIP and plan-only/no-op recipes while
still refusing exactly the failure mode that burns GPU: a native dispatch about
to create a claim and spend provider time with an empty feasibility
intersection.

Verification:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  src/tac/tests/test_dispatch_protocol_complete.py \
  src/tac/tests/test_operator_authorize_canonical_tool.py
# 25 passed

.venv/bin/ruff check \
  src/tac/deploy/dispatch_protocol.py \
  tools/check_dispatch_protocol_complete.py \
  tools/operator_authorize.py \
  src/tac/tests/test_dispatch_protocol_complete.py \
  src/tac/tests/test_operator_authorize_canonical_tool.py
# All checks passed
```

DP1 result:

```bash
PYTHONPATH=src .venv/bin/python tools/check_dispatch_protocol_complete.py \
  --recipe .omx/operator_authorize_recipes/substrate_pretrained_driving_prior_modal_t4_dispatch.yaml \
  --strict
# dispatch_protocol_complete=true
```

D1 result:

```bash
PYTHONPATH=src .venv/bin/python tools/check_dispatch_protocol_complete.py \
  --recipe .omx/operator_authorize_recipes/substrate_d1_segnet_margin_polytope_modal_t4_dispatch.yaml
# dispatch_protocol_complete=false
# tier1 blocks on dispatch_enabled=false and explicit D1 overlay blockers
# tier2/tier3 pass
```

No score claim is created by this landing.
