# Proposed CLAUDE.md addition — Eval-roundtrip + autograd-YUV6 in training inner loop

**Operator:** paste the section below into CLAUDE.md immediately AFTER the
existing "eval_roundtrip — NON-NEGOTIABLE, HIGHEST EMPHASIS" section
(currently around the "EMA — NON-NEGOTIABLE" boundary). It EXTENDS the
existing rule to the NeRV/HNeRV training path where the rule has demonstrably
not been applied (per binary-forensics finding 2026-05-09).

Cross-references for the operator:

- Source dossier: `.omx/research/hnerv_leaderboard_binary_forensics_dossier_20260509.md`
- 16-secret table: `.omx/research/hnerv_forensics_critical_findings_for_a1a9359d_20260509.md`
- PR #95 oracle (Finding A): `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source/submissions/belt_and_suspenders/src/stages/common.py:179-194`
- PR #95 oracle (Finding B): `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source/submissions/belt_and_suspenders/src/data.py:51-81`
- Canonical implementation: `src/tac/differentiable_eval_roundtrip.py`
- Probe disambiguator: `tools/probe_yuv6_differentiability_disambiguator.py`
- Module tests: `src/tac/tests/test_differentiable_eval_roundtrip.py` (31 pass)
- Trainer wire-in tests: `tests/paradigm_delta_epsilon_zeta/test_eval_roundtrip_in_training_loop.py` (18 pass)
- Subagent forensics: a30f2ade flagged this as the single biggest "thing they did right that we didn't"

---

## Eval-roundtrip + autograd-YUV6 in training inner loop — NON-NEGOTIABLE, HIGHEST EMPHASIS

This rule extends "eval_roundtrip — NON-NEGOTIABLE, HIGHEST EMPHASIS" (above)
into the **NeRV/HNeRV training-path scope** where the rule has demonstrably
not been applied. PR #95 / PR #106 (Aaron's "belt_and_suspenders") shipped
this fix and it materially closed the proxy-auth gap; we shipped without it
and pose plateau'd at noise floor (cf. binary forensics dossier 2026-05-09;
Aaron's quote: *"pose plateaued at 142 across 2500+ epochs"* without the YUV
monkey-patch).

### Two requirements (both NON-NEGOTIABLE):

1. **Eval-roundtrip baked into the TRAINING inner loop** (Finding A)

   Every inner training step that consumes rendered RGB MUST apply the FULL
   contest eval roundtrip (bicubic-up to camera resolution → bilinear-down to
   scorer resolution → clamp(0, 255) → STE-round) BEFORE the loss is computed.
   It is NOT sufficient to apply this only at end-of-epoch eval. The
   contest-eval gradient diverges from the proxy gradient by 2-11× when this
   is done at end-of-epoch only (per existing CLAUDE.md
   `feedback_proxy_auth_math_useless`); shipping training without it is the
   `WASTED RUN` failure class.

   Canonical primitive: `tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training`.

2. **Autograd-preserving rgb_to_yuv6 in training** (Finding B)

   Upstream `frame_utils.rgb_to_yuv6` is decorated with `@torch.no_grad()` AND
   uses in-place `clamp_()`; PoseNet's `preprocess_input` delegates to it, so
   without a patch the pose loss gradient is ZERO through the YUV6 op (the
   pose loss never reaches the renderer; pose stays pinned at random-init).

   Every training path that backprops PoseNet loss MUST either:
   - call `tac.differentiable_eval_roundtrip.patch_upstream_yuv6_globally()`
     at process start (Aaron's PR #95 verified-working recipe), OR
   - load scorers via `tac.scorer.load_differentiable_scorers` (which calls
     `make_scorers_differentiable` per-instance), OR
   - explicitly route YUV6 calls through `tac.differentiable_eval_roundtrip.differentiable_rgb_to_yuv6`
     instead of upstream's `rgb_to_yuv6`.

   The non-arbitrariness probe at `tools/probe_yuv6_differentiability_disambiguator.py`
   is the canonical arbitrator; it returns a recommendation that the trainer
   `--yuv6-mode auto` flag honors.

### Trainer wire-in contract

Every `experiments/train_*.py` that backprops through SegNet+PoseNet MUST
expose:

- `--enable-eval-roundtrip-in-training` (default True)
- `--enable-differentiable-yuv6` (default True)
- `--yuv6-mode {monkey_patch_global, tac_differentiable_routing, auto}` (default `auto`)

Defaults match the PR #95 verified-working recipe. The opt-out flags exist
ONLY for ablation studies that intentionally regress the bug class to
re-measure the gap.

Per-step provenance MUST emit:

- `eval_roundtrip_active=true/false`
- `yuv6_mode=<mode>`
- `gradient_through_yuv6_nonzero=true/false` (verified at process start via
  the in-process probe)

### STRICT preflight check

`check_training_inner_loop_uses_eval_roundtrip` (catalog #120, owned by FIX-A
review subagent, held warn-only initially with strict-flip after stability
period). Refuses any `experiments/train_*.py` that:

- Calls a renderer/decoder forward producing RGB AND
- Computes a SegNet/PoseNet loss in the same `loss.backward()` graph AND
- Does NOT route the rendered RGB through the canonical roundtrip primitive
  (`tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training`
  OR `tac.training.simulate_eval_roundtrip` for backward-compat OR
  `tac.renderer.simulate_eval_roundtrip` directly), unless the file carries
  a same-line `# EVAL_ROUNDTRIP_INNER_LOOP_OK:<reason>` waiver.

The check ALSO refuses any trainer that loads scorers via the upstream
`from frame_utils import rgb_to_yuv6` path AND backprops through them
without first calling `patch_upstream_yuv6_globally` or
`load_differentiable_scorers`.

### Why this is HIGHEST EMPHASIS

The May-4 race window cost analysis (CLAUDE.md "Race-mode rigor inversion")
showed PR #107 landed at 0.229 (~11th place) when a 241-LOC fix on PR100
substrate (rem2's silver) landed at 0.195. Per binary forensics, PR #95's
verified recipe — which we had ACCESS to in `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/`
since 2026-05-04 — would have closed the proxy-auth gap that kept our pose
loss noise-floored. This is the single biggest "thing they did right that we
didn't." Future-proofing this into a STRICT non-negotiable closes the bug
class permanently.

### Cross-references

- Existing CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE": this rule extends.
- Existing CLAUDE.md "MPS auth eval is NOISE": this rule is its training-side
  sister (training-time gradient drift instead of eval-time hardware drift).
- CLAUDE.md "Subagent landing checklist" (per coherence-by-default): hooks
  1 (sensitivity-map), 5 (continual-learning posterior), 6 (probe).
- Memory entries:
  - `feedback_eval_roundtrip_inner_loop_yuv6_replication_landed_20260509.md`
  - `feedback_unified_lagrangian_action_principle_GR_style_20260509.md`
  - `feedback_design_tension_ship_both_interpretations_let_math_arbitrate_20260509.md`
- Predicted impact: -0.010 to -0.050 score on Phase 1 trainer dispatch.
  Tagged `[predicted; PR #95 eval_roundtrip+yuv6_monkey_patch in training inner loop on Phase 1 trainer]`
  per CLAUDE.md `forbidden_score_claims`. NO empirical anchor exists yet —
  this is a prerequisite fix, not a measured score lane.
