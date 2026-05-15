# Canonical Substrate Contest-CUDA Chain Pattern (2026-05-15)

**Lane**: `lane_canonicalize_substrate_contest_chain_20260515`
**Audience**: every future substrate landing subagent + operator dispatch flow.

This document codifies the canonical 8-hop pattern from "subagent writes a substrate trainer" to "Modal dispatch produces a `[contest-CUDA]` anchor and reseeds the cost-band posterior". Cite specific working substrates as references for each hop.

## The 8 hops

| # | Surface | Canonical example | Catalog citation |
|---|---|---|---|
| 1 | Substrate package | `src/tac/substrates/c6_e4_mdl_ibps/` | Catalog #124 (8 evidence fields) |
| 2 | Trainer | `experiments/train_substrate_c6_e4_mdl_ibps.py` | Catalog #168 (AnnAssign manifest) |
| 3 | TIER_1 manifest | `TIER_1_OPERATOR_REQUIRED_FLAGS` dict | Catalog #151 (env→CLI wire-up) |
| 4 | Auth-eval canonical helper | `gate_auth_eval_call(...)` | Catalog #226 |
| 5 | Operator-authorize recipe | `.omx/operator_authorize_recipes/substrate_<id>_modal_<gpu>_dispatch.yaml` | Catalog #176 |
| 6 | Smoke-before-full wrapper | `scripts/operator_authorize_substrate_<id>_modal_<gpu>_dispatch.sh` | Catalog #167 |
| 7 | Remote lane driver | `scripts/remote_lane_substrate_<id>.sh` | Catalog #163 (sentinel) |
| 8 | Modal dispatch | `experiments/modal_train_lane.py` | Catalog #166 (HEAD parity) |

## Per-hop canonical pattern

### Hop 1 — substrate package layout

Reference: `src/tac/substrates/c6_e4_mdl_ibps/` and `src/tac/substrates/d4_wyner_ziv_frame_0/`.

Required modules:
- `architecture.py` — model, encoder, decoder
- `archive.py` — `pack_archive(...)` + `parse_<magic>_archive_bytes(...)` + `<magic>_SECTION_ROLES`
- `inflate.py` — script-side `inflate.py` template
- `score_aware_loss.py` — uses `tac.substrates.score_aware_common.score_pair_components` (Catalog #164)
- `__init__.py` — re-exports + Catalog #124 8-field docstring

### Hop 2 — trainer surface

Reference: `experiments/train_substrate_c6_e4_mdl_ibps.py`.

```python
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {  # Catalog #168
    "--video-path": {"env": "VIDEO_PATH", "required_input_file": True, ...},
    "--enable-autocast-fp16": {...},  # Catalog #172
    # ... per-substrate flags
}

def _smoke_main(args) -> int:
    # 100-epoch ~$0.30 smoke; trainer_artifact_v1 contract
    ...

def _full_main(args) -> int:
    # production training; ends with gate_auth_eval_call(...)
    # NEVER raise NotImplementedError here unless paired with
    # `research_only: true` in the recipe (per CLAUDE.md
    # "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY")
    ...
```

### Hop 3 — TIER_1 operator-required flags manifest

Per Catalog #151, every required CLI flag has an env-var alias. Operator-authorize wrappers read recipe-declared env vars and thread them into the trainer subprocess. Catalog #168 requires `ast.AnnAssign` extraction since substrate trainers use `dict[str, dict[str, Any]] = {...}` syntax.

### Hop 4 — auth-eval canonical helper

Per Catalog #226, every substrate trainer's `_full_main` MUST end with:

```python
from tac.substrates._shared.smoke_auth_eval_gate import gate_auth_eval_call
result = gate_auth_eval_call(
    archive_path=archive_path,
    inflate_sh=inflate_sh,
    json_out=json_out,
    device="cuda",
    ...
)
```

The helper validates `result["score_claim_valid"] == True` AND `result["score_axis"] == "contest_cuda"` before any cost-band posterior write (Catalogs #127, #175, #177, #192, #193).

### Hop 5 — operator-authorize recipe

Reference: `.omx/operator_authorize_recipes/substrate_c6_e4_mdl_ibps_modal_t4_dispatch.yaml`.

Required fields per CLAUDE.md "FIX-HARDEN-OPT" + Catalogs #170/#171/#173/#181/#182:

```yaml
schema_version: 1
name: substrate_<id>_modal_<gpu>_dispatch
lane_id: lane_substrate_<id>_<YYYYMMDD>
summary: ...

platform: modal
gpu: "${MODAL_GPU:-A100}"  # or T4/L4/A10G/L40S/H100
min_vram_gb: 40            # Catalog #170
min_smoke_gpu: "A100"      # Catalog #215
video_input_strategy: per_dispatch_local_copy   # Catalog #171
pyav_decode_strategy: cpu_thread_async_upload   # Catalog #181
target_modes: [contest_one_video_replay, ...]   # Catalog #182
canary_status: canary | post_canary_dependent | independent_substrate  # Catalog #173

cost_band:
  epochs: 2000
  hand_calibrated_fallback_p50_usd: 5.00
  platform_key: modal
  gpu_key: A100

predicted_delta: "..."
predicted_delta_basis: ".omx/research/..."

remote_driver: scripts/remote_lane_substrate_<id>.sh

required_input_files:
  - flag: --video-path
    default_path: upstream/videos/0.mkv
```

### Hop 5 (research-only branch)

If `_full_main` is intentionally `NotImplementedError` (council-gated) OR the substrate is structurally research-only, the recipe MUST explicitly tag:

```yaml
research_only: true
dispatch_blockers:
  - phase_2_council_approval_required_to_lift_full_main_NotImplementedError
```

OR (for design-stage scaffolds with no trainer):

```yaml
dispatch_enabled: false
research_only: true
defer_reason: |
  Trainer pending; recipe declared first per export-first discipline.
```

### Hop 6 — smoke-before-full wrapper

Reference: `scripts/operator_authorize_substrate_c6_e4_mdl_ibps_modal_t4_dispatch.sh`.

Per Catalog #167, every paid full-canary dispatch MUST route through `tools/run_modal_smoke_before_full.py` which fires a 100-epoch ~$0.30 smoke first, validates green, then fires the full canary.

### Hop 7 — remote lane driver

Reference: `scripts/remote_lane_substrate_c6_e4_mdl_ibps.sh`.

Per Catalog #163, every `remote_lane_*.sh` that sources `scripts/remote_archive_only_eval.sh` MUST set `REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1` before sourcing.

### Hop 8 — Modal dispatch

`experiments/modal_train_lane.py` (canonical) reads the manifest, builds the image with all hard runtime deps (Catalog #203 + #224), threads sentinel files (Catalog #201), records HEAD parity ledger (Catalog #166), and invokes the trainer subprocess.

## Anti-patterns (forbidden)

1. **The "built-but-not-firing" trap (Z3 v2 anchor 2026-05-15)**: substrate's `_full_main` exists, env→CLI wired, auth-eval routed — BUT recipe is `smoke_only: true`. The full code path NEVER runs on Modal. The bug class has TWO failure modes:
   - **Sub-class 1**: recipe `smoke_only: true` AND trainer `_full_main` is implemented production code (the v2 path that needs to land becomes dead code on every dispatch). Fix: split into TWO recipes — `_modal_<gpu>_smoke_only_dispatch.yaml` for diagnostic AND `_modal_<gpu>_dispatch.yaml` for full contest-CUDA. OR delete the smoke-only recipe and rely on smoke-before-full wrapper.
   - **Sub-class 2**: recipe NOT smoke_only, NOT research_only, NOT dispatch_enabled:false — but trainer's `_full_main` raises `NotImplementedError` (Z4 + Z5 anchor). Modal dispatch reaches trainer and crashes pre-auth-eval. Fix: tag recipe `research_only: true` until council approval lands (extincts the Z3 v2 bug class at the recipe-vs-trainer-state boundary).

2. **Phantom contest-CUDA claim**: marking a lane registry row L2+ with `[contest-CUDA]` evidence when the producing dispatch was actually `smoke_only: true`. Catalog #105 / #139 / #220 catch the runtime-effect symptoms; the new Catalog #232 catches the recipe-vs-trainer-state divergence at design time.

3. **Orphan recipe**: recipe exists, trainer doesn't. The audit identifies these (9 currently); they MUST carry `dispatch_enabled: false` so no operator-authorize call fires them.

## Verification — local pre-deploy harness (30s, $0)

Per operator 2026-05-15: `tools/local_pre_deploy_check.py --trainer experiments/train_substrate_<id>.py --recipe substrate_<id>_modal_<gpu>_dispatch [--strict]` runs 6 checks in ~10s including the new `recipe_status_consistent_with_trainer_state` check that catches the Z3 v2 / Z4 / Z5 bug class.

## Cross-references

- This audit: `.omx/research/substrate_contest_cuda_chain_audit_20260515.md`
- Audit JSON (machine-readable): `.omx/state/substrate_contest_cuda_chain_audit.json`
- Z3 v2 anchor memory: `feedback_z3_v2_smoke_green_but_v2_path_inactive_diagnostic_layout_anchor_20260515.md` (referenced but NOT yet created — this audit IS the supporting evidence)
