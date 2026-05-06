# Alpha/Lane 12 Large-Move Next Actions

Date: 2026-05-01

This runbook is for the mask-payload path only. It does not create score
evidence. Score truth remains exact CUDA archive evaluation through
`archive.zip -> inflate.sh -> upstream/evaluate.py`, preferably
`experiments/contest_auth_eval.py --device cuda`.

## Current Anchor

- Baseline comparator: PFP16 A++ final deploy bundle.
- Archive:
  `experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/archive/archive.zip`
- Archive SHA-256:
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`
- Archive bytes: `686635`
- Recomputed score: `1.043987524793892`
- Exact eval source:
  `experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/eval/contest_auth_eval.json`

Lane 12 `jsonfix40` is scoped negative evidence for that measured
implementation/config only:

- Exact eval:
  `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/contest_auth_eval.json`
- Recomputed score: `26.03719330455429`
- PoseNet: `49.7784996`
- SegNet: `0.03528685`
- Archive bytes: `296478`
- Archive SHA-256:
  `864549cc648f0b3a023076c11812ccd0f10b1d013ed3fd6bb24d20bbcde85c97`

## Dispatch 1: Parallel Mask-Codec Screen

Purpose: cheaply screen Wavelet, VQ, and grayscale-LUT mask representations
against the decoded PFP16 archive masks before any retraining spend.

Command:

```bash
.venv/bin/python experiments/paradigm_alpha_real_archive_eval.py \
  --archive experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/archive/archive.zip \
  --mask-member masks.mkv \
  --output-dir experiments/results/alpha_mask_codec_probe_20260501 \
  --candidates alpha2,alpha3,alpha4
```

Use only as empirical/no-score evidence. A candidate only graduates when it
is archive-integrated, passes Alpha-Geo diagnostics, has regenerated pose
provenance when masks change, and passes exact CUDA auth eval.

## Dispatch 2: Alpha-Geo Residual Region Materialization

Purpose: expand the current repair packet into a larger deterministic region
set for sparse residual design. This is CPU-only and does not require Lane 12
L2 clearance.

Command:

```bash
.venv/bin/python experiments/diagnose_nerv_geometry.py \
  --baseline experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/archive/archive.zip \
  --baseline-member masks.mkv \
  --candidate experiments/results/lane_12_nerv_20260430_codex_jsonfix40/archive_lane_12_nerv.zip \
  --candidate-member masks.nrv \
  --num-frames 1200 \
  --height 384 \
  --width 512 \
  --threshold-preset none \
  --residual-region-count 1000 \
  --visual-component-classes 1,2 \
  --visual-disable-temporal-tracks \
  --output-json experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_residual_regions_1000_20260501.json
```

Next implementation target after this command is a charged sparse-residual
payload builder. All residual side information must live inside `archive.zip`.

## Dispatch 3: Decoded-Baseline NeRV Retraining

Purpose: rerun NeRV against decoded archive masks and the primitive contract,
not fresh SegNet targets. This is the shortest path back to a large Alpha
rate move, but it is currently blocked by Lane 12 L2 policy.

Before any remote claim or launch, run the local trainer-input preflight:

```bash
.venv/bin/python experiments/preflight_lane12_decoded_baseline_build.py --force
```

This preflight decodes the baseline `masks.mkv`, validates its canonical
uint8 decoded-mask SHA against the `alpha_geo_primitive_contract_v1`, builds the
weighted sampling scaffold, checks the remote script guardrails, and writes:

```text
experiments/results/lane12_l2_unblock_readiness_20260502/decoded_baseline_build_preflight.json
```

It does not train, dispatch, write `.omx/state/lane12_nerv_l2_clearance.json`,
or create score evidence. A green decoded-baseline contract preflight only
means the trainer inputs are coherent; L2 clearance and passing Alpha-Geo
geometry are still required before remote build-only training.

Blocked command template:

```bash
WORKSPACE=/workspace/pact \
LOG_DIR=/workspace/pact/lane_12_nerv_decoded_baseline_results \
RUN_AUTH_EVAL=0 \
GT_MASKS_SOURCE=decoded-baseline \
DECODED_BASELINE_PATH=/workspace/pact/experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/archive/archive.zip \
DECODED_BASELINE_MEMBER=masks.mkv \
ALPHA_PRIMITIVE_CONTRACT=/workspace/pact/experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_1_vs_pfp16_repair_regions_20260501T080036Z.primitive_contract.json \
bash scripts/remote_lane_nerv.sh
```

Required blocker before dispatch: a valid
`.omx/state/lane12_nerv_l2_clearance.json` with
`cleared_for_retraining_unblock=true`, `lane12_l2=true`,
`geometry_gate_passed=true`, `grand_council_clean_passes>=3`, and cited
evidence paths. The launcher is expected to fail closed until that packet
exists.

The Alpha-Geo geometry record used for clearance must also carry archive
custody, not just aggregate rates: baseline and candidate source SHA-256 values,
candidate member `masks.nrv`, `diagnostic_config.threshold_preset=promotion`,
and `pass_fail.overall_pass=true`. For any later exact-eval dispatch readiness,
POSE_REGEN provenance must match the passing Alpha-Geo record by
`candidate_archive_sha256`; independent green-looking geometry and pose records
for different candidates are a fail-closed mismatch, not clearance.
