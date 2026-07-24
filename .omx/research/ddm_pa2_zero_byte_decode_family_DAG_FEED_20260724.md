---
title: DDM PA2 zero-byte decode-transform family DAG feed
date_utc: 2026-07-24
lane_id: lane_ddm_pa2_zero_byte_decode_family_20260724
research_only: true
execution_allowed: false
score_claim: false
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
verdict: IC2_FRAME1_XIHAT_ADMITTED_AT_ZERO_BYTES
verdict_scope: "THREE INSTANCE bases; n600 batch32 frozen scorers; no contest-axis or promotion verdict"
pointer: "0.1910828242 [contest-CPU]"
pointer_moved: false
main_landing_review_required: true
---

# Node

`DDM_PA2_ZERO_BYTE_DECODE_FAMILY_20260724`

Purpose: enumerate and measure generic receiver transforms whose runtime inputs
are limited to already-decoded frames plus frozen video-independent geometry.
The counted archive is immutable. Any per-pair table, target-derived
coefficient, selected gauge position, residual, class map, or tone table is a
counted payload and cannot ride in `inflate.py` as free code.

# Stores consulted

- `CLAUDE.md` and `AGENTS.md`
- `reports/latest.md`
- `.omx/state/lane_registry.json`
- `.omx/state/subagent_progress.jsonl`
- `.omx/state/master_gradient_anchors.jsonl`
- `.omx/state/modal_call_id_ledger.jsonl`
- `.omx/state/cost_band_posterior.jsonl`
- `.omx/state/continual_learning_posterior.jsonl`
- `.omx/state/probe_outcomes.jsonl`
- latest `council_t3_*`, design, Codex findings, and session-summary surfaces
- #580 resize/stem, #583 rank-4 prototype, PA1, IC1, IC2, and MS2R receipts
- per-arm and broadcast inboxes through `2026-07-24T20:01:23Z`

# Inputs

| Upstream node | Edge | Consumed authority |
|---|---|---|
| PA1 | ANCHORS | Existing free frame-0 YUV6 moment/BN inverse; already in IC1/IC2, freshly tested on MS2R. |
| #401 blind coordinate | EXECUTES | Exact frozen-resize blind mask; arbitrary generic fill is scorer-input identity. |
| #580 exact resize | DERIVES | All candidates recurse through the real `874x1164 -> 384x512` bilinear receiver surface. |
| SegNet stride-2 stem | DERIVES | Spatial support is the exact 2x2 scorer lattice, never a disk/global menu. |
| exact scorer factorization | TYPES | Frame 0 is Seg-free; frame 1 is joint Seg/Pose. |
| PoseNet YUV6 | DERIVES | Temporal xi-hat comes from decoded-frame luma-gradient displacement; half blend follows two-frame symmetry. |
| #583 rank-4 prototypes | BLOCKS | Feature prototypes lack class assignment and RGB/uint8 receiver pullback. |
| gauge-orbit energy | BLOCKS | Sample-specific coordinates lack a generic RGB/uint8 receiver pullback. |
| IC1 E4 packet | MEASURES | `W_joint + PA1`, 131,582 exact bytes. |
| IC2 E4 packet | MEASURES | `W_seg + PA1`, 131,154 exact bytes. |
| MS2R q4/q8 packet | MEASURES | 208 q4 / 392 q8 exact quotient control, 291,205,400 bytes. |

# Executable DAG

```text
SHA-bound typed config
  |
  +-> bind IC1 / IC2 / MS2R archive bytes and custody receipts
  |
  +-> exact receiver base
  |     IC1/IC2: real E4 inflate output
  |     MS2R: selected q4/q8 scorer plane -> certified factor-2 uint8 preimage
  |
  +-> generic free-interpreter family
  |     #401 blind zero fill
  |     stride-2 stem boundary residual
  |     decoded-frame xi-hat -> frame0 companion blend
  |     decoded-frame xi-hat -> frame1 proposal blend
  |     PA1 decoded YUV6 moments -> frozen BN target (MS2R only)
  |
  +-> n600 x batch32 frozen SegNet + PoseNet
  |     immutable checkpoint per batch
  |
  +-> greedy round
  |     compile current stack + one candidate
  |     fresh joint remeasure
  |     admit iff conditional delta S < 0 and archive delta = 0
  |     repeat; never sum singleton deltas
  |
  +-> append structurally score-neutral #401
  |
  +-> preserve 19 selected receiver-output stages on SSD
  |
  +-> rehash exact archive before/after
        -> aggregate receipt
```

# Measured outputs

| Base | Baseline S | Selected score-moving stack | Joint S | Delta S | Archive delta |
|---|---:|---|---:|---:|---:|
| IC1 `W_joint+PA1` | 23.66179213623354 | none | 23.66179213623354 | 0 | 0 B |
| IC2 `W_seg+PA1` | 28.00173925293584 | frame1 xi-hat | 25.244396496399435 | -2.7573427565364064 | 0 B |
| MS2R q4/q8 | 194.42556028324648 | none | 194.42556028324648 | 0 | 0 B |

The IC2 admitted arm moves `d_seg` from `0.024124510023328993` to
`0.07160721672905816` and `d_pose` from `65.03498712932134` to
`32.38684246616016`. Round 2 freshly remeasured both remaining candidates:
spatial added `+0.6990042965872441` score units and frame0 xi-hat added
`+10.779291276362141`; neither was admitted.

# Typed blockers

- `pa2_gauge_orbit_rgb_pullback_v1`:
  `missing_generic_rgb_uint8_receiver_pullback`. Per-pair orbit positions or
  coefficients are counted. This blocks the present zero-byte realization,
  not the gauge family.
- `pa2_rank4_class_tone_gamma_v1`:
  `rank4_feature_prototypes_lack_rgb_pullback`. Per-pair class/tone/gamma maps
  are counted. This blocks decoded-frame-only use of the current feature
  artifact, not a future frozen generic RGB pullback.
- #401 saves exactly zero bytes on all three pure-generator bases because none
  stores a camera-resolution residual section. Its scorer-input identity is
  nevertheless proved on all n600 pairs.

# Triality

- DSL:
  `.omx/research/configs/ddm_pa2_zero_byte_decode_family_20260724.json`
- DAG: this file plus immutable SSD batch stages
- equation:
  `ddm_pa2_zero_byte_conditional_greedy_v1`
- callable receiver:
  `tac.optimization.ddm_pa2_zero_byte_decode_family`
- aggregate receipt:
  `.omx/research/ddm_pa2_zero_byte_decode_family_20260724T194836Z/receipt.json`

# Feed row

```json
{
  "node_id": "DDM_PA2_ZERO_BYTE_DECODE_FAMILY_20260724",
  "status": "READY_FOR_MAIN_REVIEW",
  "research_only": true,
  "score_claim": false,
  "pointer_moved": false,
  "admitted_base": "IC2_W_seg_PA1",
  "admitted_member": "pa2_temporal_xihat_frame1_proposal_v1",
  "conditional_delta_S": -2.7573427565364064,
  "archive_byte_delta": 0,
  "canonical_equation_id": "ddm_pa2_zero_byte_conditional_greedy_v1",
  "remaining_blockers": [
    "GAUGE_RGB_UINT8_PULLBACK_ABSENT",
    "RANK4_FEATURE_TO_RGB_PULLBACK_ABSENT",
    "CONTEST_CPU_CUDA_REPLAY_NOT_RUN"
  ],
  "main_landing_review_required": true
}
```

# Quarantine and non-edges

- PR98/L28 and old public lane material were not imported, copied, or treated
  as authority. They remain historical signal only.
- No paid dispatch, remote run, GPU actuation, contest evaluation, pointer
  mutation, or score claim occurred.
- The IC2 result is not automatically transferred to IC1, MS2R, contest CPU,
  or contest CUDA.

# MAIN landing review

MAIN must independently verify:

1. all three archive byte counts and SHA-256 values before accepting the
   zero-byte claim;
2. exact IC1/IC2 E4 and MS2R quotient receiver custody;
3. decoded-frame-only inputs and absence of video-derived code constants;
4. all n600 batch32 rows and the IC2 non-telescoping round-2 results;
5. selected 19-stage output chains and archive before/after identity;
6. typed blocker scope, false-authority labels, and pointer immobility;
7. the canonical equation registry append and three clean review passes.
