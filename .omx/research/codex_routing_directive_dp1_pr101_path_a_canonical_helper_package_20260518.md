# Codex routing directive: DP1+PR101 Path A canonical helper package
# Date: 2026-05-18
# Operator: closure campaign master memo `closure_campaign_pursue_and_confirm_master_20260518.md` op-routable OPR-CLOSE-4
# Authority: `.omx/research/dp1_pr101_composition_design_memo_20260518.md` (THE design memo; 116.4 KB; Path A canonical helper spec at §5.4)
# Sister: `.omx/research/codex_routing_directive_dp1_pr101_op1_op2_zero_cost_probes_20260518.md` (the upstream OP-1 + OP-2 $0 probe directive)
# Per CLAUDE.md "Subagent coherence-by-default" + Catalog #245 4-layer pattern + #325 per-substrate symposium + #240 recipe-vs-trainer-state + #220 substrate L1+ operational mechanism + #272 distinguishing-feature integration contract

## CANONICAL POINTERS (read FIRST)

1. `/Users/adpena/Projects/pact/CLAUDE.md` (FULL; especially Catalog #325 6-step per-substrate symposium contract + #220 substrate L1+ operational mechanism + #272 distinguishing-feature integration + #324 post-training Tier-C validation + #240 recipe-vs-trainer-state)
2. `/Users/adpena/Projects/pact/AGENTS.md`
3. `.omx/research/dp1_pr101_composition_design_memo_20260518.md` (THE AUTHORITY; Path A canonical helper spec at §5.4)
4. `.omx/research/codex_routing_directive_dp1_pr101_op1_op2_zero_cost_probes_20260518.md` (upstream $0 probe gating)
5. `.omx/research/closure_campaign_pursue_and_confirm_master_20260518.md` (parent coordinator memo)
6. `src/tac/substrates/pretrained_driving_prior/` (DP1 canonical impl; codebook + distillation)
7. `src/tac/substrates/pr101_lc_v2_clone/` (PR101 GOLD consumer; HNeRV trainer)
8. `experiments/train_substrate_pretrained_driving_prior.py` (DP1 canonical trainer pattern reference)
9. `experiments/train_substrate_pr101_lc_v2_clone.py` (PR101 canonical trainer pattern reference)

## STRATEGIC CONTEXT

The DP1+PR101 Path A canonical helper unifies the 2-stage Path A composition per design memo §5.4:
- **Stage 1**: DP1 codebook weight init for PR101 HNeRV (warm-start)
- **Stage 2**: PR101 HNeRV refinement training on contest video `upstream/videos/0.mkv`

Predicted band per memo §3.2: optimistic `-0.012` / pessimistic `+0.002` per Dykstra-feasibility analysis (Catalog #287 axis tag `[prediction]`).

**Upstream gates** (per upstream OP-1 + OP-2 $0 probe directive):
- OP-1 OOD-similarity probe must return `OOD_SIMILAR_ENOUGH` (Comma2k19 vs `upstream/videos/0.mkv`)
- OP-2 architecture-compatibility probe must return `LOADABLE` (DP1 codebook state_dict → PR101 HNeRV)

**Per-substrate symposium gate** per Catalog #325: NEW T3 per-substrate symposium memo at `.omx/research/council_t3_per_substrate_symposium_dp1_pr101_composition_path_a_20260518.md` must satisfy 6-step contract before dispatch.

This directive routes the full Path A canonical helper build to Codex.

## OP-1: Layer 1 — substrate canonical helper package

**Target**: `src/tac/substrates/dp1_pr101_path_a/__init__.py` + sister modules

**Public API**:

```python
# src/tac/substrates/dp1_pr101_path_a/__init__.py
from .architecture import Dp1Pr101PathASubstrate
from .stage_1_warm_start import init_pr101_hnerv_from_dp1_codebook
from .stage_2_refinement import refine_pr101_hnerv_on_contest_video
from .score_aware_loss import Dp1Pr101PathAScoreAwareLoss
from .archive import build_dp1_pr101_path_a_archive, decode_dp1_pr101_path_a_archive
from .anchor_writer import append_dp1_pr101_path_a_anchor

__all__ = [
    "Dp1Pr101PathASubstrate",
    "init_pr101_hnerv_from_dp1_codebook",
    "refine_pr101_hnerv_on_contest_video",
    "Dp1Pr101PathAScoreAwareLoss",
    "build_dp1_pr101_path_a_archive", "decode_dp1_pr101_path_a_archive",
    "append_dp1_pr101_path_a_anchor",
]
```

**Implementation requirements** (per design memo §5.4):

- `Dp1Pr101PathASubstrate(architecture_spec, dp1_codebook, pr101_hnerv_config)` substrate class
- `init_pr101_hnerv_from_dp1_codebook(pr101_model, dp1_codebook_state_dict) -> pr101_model_warm_started` (Stage 1)
- `refine_pr101_hnerv_on_contest_video(warm_started_model, contest_video_path, epochs, scorers) -> trained_model` (Stage 2 with canonical eval_roundtrip + EMA 0.997 + score-aware loss per CLAUDE.md non-negotiables)
- `Dp1Pr101PathAScoreAwareLoss` per Catalog #164 canonical scorer-loss helper routing (`tac.substrates._shared.score_aware_common.score_pair_components`)
- Archive grammar at `src/tac/substrates/dp1_pr101_path_a/archive.py` with magic header `DPR1\x00` + length-prefix sections + monolithic single-file `0.bin` per HNeRV parity L3
- Inflate runtime ≤ 100 LOC at `submissions/dp1_pr101_path_a/inflate.py` per HNeRV parity L4

**Canonical-vs-unique decision per layer** (per Catalog #290 + design memo §5.4):

| Layer | Decision | Rationale |
|---|---|---|
| DP1 codebook consumption | ADOPT canonical (`tac.substrates.pretrained_driving_prior.compose_with`) | Per Catalog #211 canonical composition helper |
| PR101 HNeRV architecture | ADOPT canonical (`tac.substrates.pr101_lc_v2_clone.architecture`) | PR101 GOLD architectural primitives |
| Score-aware loss | ADOPT canonical (`tac.substrates._shared.score_aware_common.score_pair_components`) | Catalog #164 + #226 |
| EMA | ADOPT canonical (`tac.training.EMA` decay 0.997) | CLAUDE.md "EMA — NON-NEGOTIABLE" |
| eval_roundtrip | ADOPT canonical (`tac.differentiable_eval_roundtrip`) | CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE" |
| Archive grammar | FORK_BECAUSE_PRINCIPLED_MISMATCH | DPR1\x00 magic + DP1+PR101 composition format unique |
| Inflate runtime | FORK_BECAUSE_PRINCIPLED_MISMATCH | Path-A-specific decode logic |
| auth_eval routing | ADOPT canonical (`tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call`) | Catalog #226 |
| Hardware-substrate detection | ADOPT canonical (`tac.substrates._shared.trainer_skeleton.detect_hardware_substrate`) | Catalog #190 |
| Inflate device selection | ADOPT canonical (`select_inflate_device`) | Catalog #205 |

## OP-2: Layer 2 — trainer + recipe

**Trainer target**: `experiments/train_substrate_dp1_pr101_composition_path_a.py`

Required elements per CLAUDE.md "Dispatch optimization protocol" (Catalog #270) + per-substrate symposium 6-step contract (Catalog #325):

- `TIER_1_OPERATOR_REQUIRED_FLAGS` module-level dict per Catalog #151 wire-in discipline
- `--enable-autocast-fp16` argparse per Catalog #172
- TF32 via `device_or_die` per Catalog #178
- `--enable-torch-compile` argparse per Catalog #179
- `with torch.no_grad():` for all eval forwards per Catalog #180
- canonical `gate_auth_eval_call(...)` for auth eval per Catalog #226
- `select_inflate_device` per Catalog #205
- `_full_main` MUST IMPLEMENT full path (NO `NotImplementedError` per Catalog #240)
- `_smoke_main` for 5-100ep CPU/MPS smoke

**Recipe target**: `.omx/operator_authorize_recipes/substrate_dp1_pr101_composition_path_a_modal_a100_dispatch.yaml`

Required fields per multi-Catalog sister gates:

```yaml
# .omx/operator_authorize_recipes/substrate_dp1_pr101_composition_path_a_modal_a100_dispatch.yaml
substrate_id: dp1_pr101_composition_path_a
trainer_path: experiments/train_substrate_dp1_pr101_composition_path_a.py
lane_script: scripts/remote_lane_substrate_dp1_pr101_composition_path_a.sh
platform: modal
gpu: A100

# Per Catalog #170 minimum VRAM (PR101 HNeRV + DP1 codebook joint)
min_vram_gb: 40

# Per Catalog #171 video input strategy
video_input_strategy: per_dispatch_local_copy

# Per Catalog #181 pyav decode strategy
pyav_decode_strategy: cpu_thread_async_upload

# Per Catalog #182 target_modes
target_modes:
  - contest_one_video_replay

# Per Catalog #173 canary status
canary_status: independent_substrate

# Per Catalog #215 smoke compute class
min_smoke_gpu: A100

# Per Catalog #240 recipe-vs-trainer-state consistency
# INITIAL LANDING: research_only=true + dispatch_enabled=false
# Flip to dispatch_enabled: true ONLY after per-substrate symposium PROCEED per Catalog #325
research_only: true
dispatch_enabled: false

# Per Catalog #324 post-training Tier-C validation
predicted_band: [0.180, 0.190]
predicted_band_validation_status: pending_post_training
predicted_band_reactivation_criterion: "Stage 2 Modal A100 50ep smoke lands [contest-CUDA T4] anchor; sister Tier-C density measurement on landed archive"

# Per Catalog #152 required input files
required_input_files:
  - flag: --dp1-codebook-archive
    default: experiments/results/lane_dp1_phase_2_landed/dp1_codebook_archive.bin
    required_input_file: true
  - flag: --pr101-base-archive
    default: experiments/results/lane_pr101_lc_v2_clone_landed/pr101_lc_v2_archive.zip
    required_input_file: true
  - flag: --contest-video-path
    default: upstream/videos/0.mkv
    required_input_file: true

# Per Catalog #325 per-substrate symposium evidence
symposium_memo_path: .omx/research/council_t3_per_substrate_symposium_dp1_pr101_composition_path_a_20260518.md

# Cost band
cost_band:
  epochs: 50
  expected_cost_usd: 10.0  # Modal A100 50ep ~$10
```

## OP-3: Layer 3 — STRICT preflight gates (claim + wire)

**Note**: this substrate inherits ALL existing strict preflight gates per the cross-cutting non-negotiables. NO new gate needed — but the substrate MUST PASS every existing gate including:
- Catalog #146 (Phase 1 trainer contest-compliant runtime)
- Catalog #164 (canonical scorer-preprocess routing)
- Catalog #166 (Modal HEAD-parity)
- Catalog #220 (substrate L1+ operational mechanism)
- Catalog #226 (canonical auth_eval helper)
- Catalog #240 (recipe-vs-trainer-state)
- Catalog #270 (dispatch optimization protocol)
- Catalog #272 (distinguishing-feature integration contract)
- Catalog #305 (observability surface declaration in design memo header)
- Catalog #324 (post-training Tier-C validation)
- Catalog #325 (per-substrate symposium evidence)

**Distinguishing feature declaration** per Catalog #272:
```yaml
# In lane_registry.json
distinguishing_feature_name: "dp1_warm_start_codebook"
distinguishing_bytes_path: "<archive_path>/dp1_codebook_section"  # in archive.zip
inflate_consumer_function: "src/tac/substrates/dp1_pr101_path_a/archive.py::decode_dp1_pr101_path_a_archive"
byte_mutation_smoke_passes: false  # initial; flips to true after empirical smoke
```

## OP-4: Layer 4 — integration wire-ins per Catalog #125 6-hook

**Hook #1 sensitivity-map**: DP1 codebook entry sensitivity → per-entry axis weight contribution
**Hook #2 Pareto constraint**: warm-start initialization adds Pareto constraint per design memo §5.4
**Hook #3 bit-allocator**: DP1 codebook byte allocation tier (high-utility codes → tier-1)
**Hook #4 cathedral autopilot**: extend `tools/cathedral_autopilot_autonomous_loop.py`:
```python
def adjust_predicted_delta_for_dp1_pr101_path_a_smoke_verdict(predicted_delta, archive_sha256):
    """Post-Stage-2 smoke verdict:
    - lands within predicted [0.180, 0.190] → +0.005 reward
    - lands outside band → 0 (no adjustment; sister Catalog #324 triggers reactivation)"""
    ...
```
**Hook #5 continual-learning**: `append_dp1_pr101_path_a_anchor` → `.omx/state/dp1_pr101_path_a_anchors.jsonl`
**Hook #6 probe-disambiguator**: OP-1 OOD-similarity + OP-2 architecture-compatibility per upstream directive

## OP-5: Per-substrate symposium memo per Catalog #325 6-step contract

**Target**: `.omx/research/council_t3_per_substrate_symposium_dp1_pr101_composition_path_a_20260518.md`

Required v2 frontmatter (per Catalog #300) + 6 sections (per Catalog #325):

```yaml
---
council_tier: T3
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Hafner, Hinton, Tishby memorial, Atick, Schmidhuber, Selfcomp, Carmack, Hotz]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS  # or PROCEED if no revisions
council_dissent: [...]
council_assumption_adversary_verdict: [...]
council_decisions_recorded: [...]
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: "n/a"
predicted_band: [0.180, 0.190]
predicted_band_validation_status: pending_post_training
horizon_class: frontier_pursuit
deferred_substrate_id: dp1_pr101_composition_path_a
---

# Per-substrate symposium for DP1+PR101 Path A

## 1. Cargo-cult audit per assumption (Catalog #303)
[...]

## 2. 9-dim checklist evidence (Catalog #294)
[...]

## 3. Observability surface (Catalog #305)
[...]

## 4. Sextet pact deliberation (with grand council attendees)
[...]

## 5. Per-substrate reactivation criteria (Catalog #325)
[...]

## 6. Catalog #324 post-training Tier-C validation
predicted_band_validation_status: pending_post_training (reactivation = post-training Tier-C re-measurement on the landed archive sha)
```

## OP-6: Stage-2 Modal A100 50ep smoke (operator-gated; $5-15)

**Smoke command** (operator-runnable after symposium PROCEED):

```bash
# 30-second pre-deploy harness fires per Catalog #243 + #271 + #313
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_dp1_pr101_composition_path_a_modal_a100_dispatch \
    --smoke-before-full \
    --estimated-cost-usd 12.00
# Codex pre-dispatch review (Catalog #271) fires; predecessor probe outcome check (Catalog #313) fires
# Local pre-deploy check (Catalog #243) fires; dispatch optimization protocol (Catalog #270) fires
# All gates green → Modal A100 50ep smoke fires
```

## OP-7: Memory entry per CLAUDE.md "Subagent coherence-by-default"

`feedback_dp1_pr101_path_a_canonical_helper_package_landed_<utc>.md` with all 6 Catalog #294 / #303 / #305 / #296 / #290 / #125 sections.

## OP-8: canonical_task_status row emission

8 task rows (OP_1 through OP_8) per closure verifier dependency.

## Discipline checklist

Same as sister directives + per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable:
- Initial recipe `research_only: true` until symposium PROCEED per Catalog #325
- Flip to `dispatch_enabled: true` ONLY after operator-gated review of symposium verdict
- Empirical Stage-2 smoke results validate predicted band per Catalog #324; reactivation criteria pinned

## Dependencies on sister OPs

- **OP-AUDIT-1** (master-gradient 6-archive) — DP1 archive needs projector landing per OP-AUDIT-1 OP-7 (currently fail-closed)
- **OP-AUDIT-3 FISHER** — Stage-2 refinement consumes Fisher diagonal for natural-gradient step
- **OP-AUDIT-3 RIEM** — Stage-2 refinement uses Riemannian-Newton META-substrate inheritance
- **Upstream $0 probe directive** — OP-1 OOD-similarity + OP-2 architecture-compatibility must pass before Stage-2 fires

Begin.
