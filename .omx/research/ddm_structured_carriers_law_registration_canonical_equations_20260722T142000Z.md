---
schema: canonical_equation_landing_note.v1
date_utc: 2026-07-22T14:20:00Z
equation_id: ddm_describe_line_rate_distortion_bracket_v1
task: 540
master_task: 578
research_only: true
score_claim: false
main_landing_review_required: true
---

# Canonical equation - DDM describe-line rate/distortion bracket

For measured receiver-closed points `p=(B,D_seg,D_pose)`, define the joint gate

`G(p; B_box) = 1[B<=B_box] * 1[D_seg<=0.00116] * 1[D_pose<=0.00025]`.

The registered three-leg bracket is

```text
v7 exact values:      G(p; 200000)=0 only because rate is red
v8 sparse pixels:     G(p; 200000)=0 because rate and both evaluator legs are red
v9 structured base:   G(p; 154600)=0 because both evaluator legs are red
```

with the open curve

`p_v9 -> p_(G2CS1 + xi events) -> p_v7`.

The callable predicts only the scoped categorical receipt verdict from measured rows. Its three
`predicted_vs_empirical_residual` values are genuine categorical prediction residuals; the n600
byte interpolation is returned separately as

`B_600 = 51668 + (600-64)*(72397-51668)/(256-64) = 2628875/24 B`

and labeled `DERIVED_FROM_MEASURED_N64_N256_NOT_MEASURED_N600`.

## Empirical anchors

| anchor | primary cross receipt | SHA-256 | axis |
|---|---|---|---|
| v7 exact-value rate wall | `.omx/research/ddm_v7_solved_plane_tolerance_waterfill_603_613_20260722T102423Z.receipt.json` | `64658a05a8975707f98db308223cefff78b5352975bb59cc2aa8a4ff2f8d50fb` | `[macOS-CPU frozen-scorer advisory]` |
| v8 sparse-pixel ERF wall | `.omx/research/ddm_v8_margin_gated_correction_603_613_20260722T115052Z.receipt.json` | `7051927df863a3ab01a6e1494550a914829715b00faeae15baa3abb951a49d1c` | `[macOS-CPU frozen-scorer advisory]` |
| v9 structured in-box base | `.omx/research/ddm_v9_carrier_compose_byteclose_SHA_RECEIPT_20260722.json` | `97bf956179ec46a52be0ccc8a5e16e399a682e11e94d1e0b9e33497a32bfafe6` | `[macOS-CPU frozen-scorer advisory]` |

Each anchor also binds its n64 and n256 receipt hashes, for nine verified files total. Provenance is
non-promotable by construction; `score_claim=false` and `promotion_eligible=false` survive JSONL
roundtrip.

## Scope

- v7 negative: opaque solved-plane site/value payload formulation only.
- v8 negative: finite-tau post-hoc exact pixel values over the inherited photometrically alien
  predictor only. ERF evidence explains the mechanism without widening the receipt verdict.
- v9 negative: exact composed carrier instance with zero measured G2CS1 symbols. Structured carrier,
  joint chart/event, and corrected-inner-Jacobian families remain open.
- Anti-pattern: not registered because the v8 receipt is narrower than the class-level negative
  registry contract.

## Producers and consumers

Producers: `tools.run_direct_description_entropy_priced_member`,
`tools.run_ddm_v9_carrier_compose`.

Consumers: `tac.optimization.v10_constructive_solver`,
`tac.optimization.direct_description_entropy_priced_member`,
`tac.witness_control.costate_organ_v2`.

## STORES CONSULTED

STORES CONSULTED: all ten predecessor equation drafts; v7/v8/v9 cross and window receipts; v9
compose source/config/checkpoint SHA receipt; canonical-equation schema/registry/list tooling;
canonical anti-pattern taxonomy; SegNet ERF factorization; Brenier/Splay crosswalk notes; current
DAG, pointer, lane, task, and subagent stores; delegated inboxes.

`0.1910828242 [contest-CPU]` is unchanged. MAIN landing review is required.

