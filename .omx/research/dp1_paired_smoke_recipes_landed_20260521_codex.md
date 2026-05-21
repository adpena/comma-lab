# DP1 Paired-Smoke Recipe Landing (Codex)

timestamp_utc: 2026-05-21T02:20:36Z
agent: codex
scope: dp1 procedural-codebook first paired-smoke readiness
verdict: LANDED_OPERATOR_GATED_RECIPES_NO_DISPATCH

## Summary

This landing closes the remaining DP1 procedural-codebook paired-smoke recipe
surface without firing spend or making a score claim.

Authored three operator-gated recipes under `.omx/operator_authorize_recipes/`:

- `substrate_pretrained_driving_prior_original_baseline_modal_t4_paired_dispatch.yaml`
- `substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch.yaml`
- `substrate_pretrained_driving_prior_null_exploit_codebook_modal_t4_paired_dispatch.yaml`

All three recipes retain:

- `dispatch_enabled: false`
- `research_only: true`
- `score_claim: false`
- `promotion_eligible: false`
- `rank_or_kill_eligible: false`
- `ready_for_exact_eval_dispatch: false`

All three recipes also set:

- `DPP_LANE_ID` to the recipe lane id, so the remote worker verifies the same
  dispatch claim that `operator_authorize.py` creates.
- `DPP_SKIP_AUTH_EVAL: "1"`, so the training recipe cannot emit a one-axis
  score claim while the recipe metadata says `score_claim: false`. Paired
  CPU/CUDA auth eval remains a separate harvest gate on the emitted archive.

## Wiring Fixed

The remote DP1 driver already had a trainer surface for procedural-codebook
replacement, but the remote script did not forward the corresponding env knobs
into the trainer CLI. This landing wires:

- `DPP_PROCEDURAL_CODEBOOK_REPLACEMENT`
- `DPP_PROCEDURAL_CODEBOOK_SEED_HEX`
- `DPP_PROCEDURAL_CODEBOOK_GENERATOR_KIND`
- `DPP_PROCEDURAL_CODEBOOK_NULL_EXPLOIT_CONTROL`
- `DPP_PROCEDURAL_VARIANT_PROVENANCE_PATH`
- `DPP_PROCEDURAL_VARIANT_DISTILLATION_SKIP`

The shared `DPP_PROCEDURAL_ARGS` vector is appended to both the smoke path and
the full-training path in
`scripts/remote_lane_substrate_pretrained_driving_prior.sh`.

The remote driver now reads `LANE_ID` from `DPP_LANE_ID` with the historical
scaffold lane as fallback, closing the recipe-lane-vs-driver-lane dispatch claim
misattribution class.

`DPP_PROCEDURAL_CODEBOOK_VALIDATE_DOMAIN=0` now forwards
`--no-procedural-codebook-validate-domain`; the default recipes keep validation
enabled. The reserved `--procedural-variant-distillation-skip` flag now fails
closed until a real no-distillation procedural training path exists.

The smoke trainer path in `experiments/train_substrate_pretrained_driving_prior.py`
now also exercises the procedural archive mutation and procedural-aware parser
when `--enable-procedural-codebook-replacement` is set. This prevents the smoke
path from silently testing only the original archive grammar while the full path
tests the procedural variant.

The exported submission runtime now vendors the procedural-aware inflate helper,
the seed-derived codebook generator submodule, and the DP1 prior-application
module. This closes the `ModuleNotFoundError` class where a procedural archive
could be built but the emitted `submission/` tree could not import its own
inflate dependencies.

## Lane Registry

Registered L0 recipe lanes:

- `lane_dp1_original_baseline_first_paired_anchor_20260520`
- `lane_dp1_procedural_codebook_replacement_first_paired_smoke_20260520`
- `lane_dp1_null_exploit_codebook_replacement_control_paired_smoke_20260520`

Each lane is marked `research_only=true` with notes preserving the operator-gated,
non-promotional status.

## Verification

Commands run:

```bash
bash -n scripts/remote_lane_substrate_pretrained_driving_prior.sh
.venv/bin/python -m py_compile experiments/train_substrate_pretrained_driving_prior.py
.venv/bin/python -m pytest -q src/tac/tests/test_dp1_remote_driver_contract.py src/tac/substrates/pretrained_driving_prior/tests/test_dispatch_ready_extension.py src/tac/substrates/pretrained_driving_prior/tests/test_procedural_variant.py
.venv/bin/python tools/operator_authorize.py --list | rg 'pretrained_driving_prior_(original|procedural|null)|substrate_pretrained_driving_prior_(original|procedural|null)'
.venv/bin/python tools/operator_authorize.py --recipe substrate_pretrained_driving_prior_original_baseline_modal_t4_paired_dispatch --dry-run
.venv/bin/python tools/operator_authorize.py --recipe substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch --dry-run
.venv/bin/python tools/operator_authorize.py --recipe substrate_pretrained_driving_prior_null_exploit_codebook_modal_t4_paired_dispatch --dry-run
.venv/bin/python experiments/train_substrate_pretrained_driving_prior.py --smoke --procedural-variant-distillation-skip
git diff --check
```

Results:

- Shell syntax: pass
- Python compile: pass
- Focused pytest: `49 passed in 1.60s`
- Recipe discovery: all three recipes listed by `operator_authorize.py --list`
- Dry-run behavior: all three print the dispatch plan and then refuse dispatch
  because `dispatch_enabled=false`
- Reserved distillation-skip flag: exits `1` with an explicit fail-closed
  message rather than silently claiming to skip distillation
- Whitespace check: pass

## Remaining Gate

No paired smoke was launched in this landing. The next action remains an
operator-intentional dispatch flip for the baseline and procedural recipes,
followed by separate paired CPU/CUDA auth-eval harvest on the emitted archives
before any empirical anchor for
`procedural_codebook_from_seed_compression_savings_v1` is registered.
