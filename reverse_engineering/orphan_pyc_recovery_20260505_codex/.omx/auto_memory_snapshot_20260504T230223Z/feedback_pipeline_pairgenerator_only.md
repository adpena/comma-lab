---
name: Pipeline downstream is AsymmetricPairGenerator-only
description: 2026-04-26 — pipeline.py + auth_eval_renderer + optimize_poses + renderer_export hardcode AsymmetricPairGenerator class assumptions. PairGenerator-class profiles (use_zoom_flow=False) need a 4-layer refactor + interface forwarding shim before they can deploy end-to-end.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
The deploy pipeline was built around AsymmetricPairGenerator. PairGenerator-
class profiles cannot complete end-to-end without an interface unification:

1. **pipeline.step_export** — hardcoded `AsymmetricPairGenerator(...)` constructor.
   FIXED via `build_renderer()` dispatch (commit c5214993).
2. **pipeline.step_export** — checkpoints with parametrize hooks (DEN's self-
   compression path) need keys normalized: `<prefix>.parametrizations.weight.original`
   → `<prefix>.weight`, drop codebooks. FIXED in c5214993.
3. **renderer_export.load_asymmetric_checkpoint_fp4** — same hardcoded
   AsymmetricPairGenerator. FIXED via `build_renderer()` (c5214993).
4. **renderer_export._infer_asymmetric_config** — read `model.use_dsconv` but
   PairGenerator stores it on `model.renderer.use_dsconv`. FIXED with renderer-
   attr fallback (c5214993).
5. **PairGenerator interface forwarding shim** — added `pose_dim`, `use_dsconv`,
   `use_zoom_flow`, `padding_mode`, `use_dilation`, etc. as forwarded attrs from
   inner MaskRenderer + MotionPredictor so consumers (`auth_eval_renderer`,
   `optimize_poses`, `qat_finetune`) work without modification. STAGED in working
   tree, blocked by review-gate on renderer.py (needs 2 distinct approvers + 1
   human). 9436fb15 has just the pipeline-side soft-skip.
6. **pipeline.step_pose_tto** — soft-skip when pose_dim=0 (no FiLM): pose-space
   TTO is architecturally meaningless. FIXED in 9436fb15.

**Why:** DEN-V2 burned ~$1 of GPU (~1h compress attempts) discovering each layer
sequentially. Each error was a different KeyError/AttributeError exposing the
next AsymmetricPairGenerator-only assumption.

**How to apply:**
- Any new profile with use_zoom_flow=False (→ PairGenerator) MUST verify the
  full deploy chain before launching expensive remote runs.
- A preflight rule should instantiate `build_renderer(profile)` and check the
  resulting model exposes every attribute the deploy pipeline reads. Any missing
  attribute → fail at preflight, not at runtime.
- The forwarding shim on renderer.py needs proper greenup (2 approvers + human)
  before merge. Until then, PairGenerator-class profiles cannot deploy.
