---
schema: ddm_v8_margin_gated_correction_landing.v1
task: 603
feeds_task: 613
master_task: 578
lane_id: ddm_v8_margin_gated_correction
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
research_only: true
score_claim: false
d_seg_claim: false
d_pose_claim: false
candidate_archive: false
producer_commit: 46d3a79370c80258963d46c338660b2cdde03eb3
main_landing_review_required: true
---

# DDM v8 margin-gated correction

## Outcome

**MEASURED, FORMULATION-scoped:** the encode-side Fisher/margin mask collapses the opaque correction
payload by about 94%, but it does not preserve either evaluator leg. At n256, the tight mandatory
predictor-mismatch mask covers 4.0182% of sites and produces a 9,360,569-byte archive—already
46.80x the 200 KB box—at advisory `d_seg=0.025907576084` and `d_pose=113.918588951715`.
Increasing tau adds mostly Boundary support and improves Pose, but d_seg becomes worse rather than
approaching its target. The only measured joint-feasible endpoint remains the SHA-bound v7 exact
row at 171,332,654 bytes.

This closes the registered finite-tau, exact-value, margin-gated opaque-correction formulation. It
does not close learned/analytic carriers, corrected-inner-Jacobian realization, or the unmeasured
argmax-safe in-mask quantization axis.

## Receiver-closed tau ladder

Every finite row is **MEASURED** on `[macOS-CPU frozen-scorer advisory]`; bytes are exact final ZIP
lengths and all claim flags are false.

| tau | n64 mask | n64 bytes | n64 d_seg | n64 d_pose | n256 mask | n256 bytes | n256 d_seg | n256 d_pose |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 4.530207% | 2,629,076 | 0.029359102249 | 117.509345241081 | 4.018213% | 9,360,569 | 0.025907576084 | 113.918588951715 |
| 0.1 | 4.668339% | 2,711,431 | 0.029486179352 | 110.196784712230 | 4.149423% | 9,675,861 | 0.026060779889 | 108.363288404228 |
| 0.5 | 5.256947% | 3,062,403 | 0.030277411143 | 67.326188740405 | 4.711958% | 11,003,067 | 0.026977002621 | 72.738685040068 |
| 1.0 | 6.074444% | 3,547,665 | 0.031822125117 | 17.403719517738 | 5.478674% | 12,811,245 | 0.028795083364 | 31.003158645132 |
| settled exact endpoint | 100% | 43,112,153 | 0.000171422958 | 0.000081666650 | 100% | 171,332,654 | 0.000154534976 | 0.000104117518 |

The tight mask reduces bytes versus v7 exact by 93.90% (n64) and 94.54% (n256), but still costs
13.15x/46.80x the box and misses d_seg by 0.028199102249/0.024747576084. Therefore byte collapse is
real but not useful constraint preservation.

## Per-stream binders and Pose guard

| tight-mask stream | n64 ZIP-home bytes | n256 ZIP-home bytes |
|---|---:|---:|
| Road | 790,533 | 2,745,416 |
| Lane | 37,384 | 137,424 |
| Undrivable | 48,596 | 352,076 |
| Movable | 985,888 | 3,390,676 |
| MyCar | 446 | 35,052 |
| Boundary | 714,216 | 2,627,176 |

Movable, Road, and Boundary account for 94.73%/93.62% of the tight archive. Boundary is the largest
added stream at every adjacent tau transition in both windows while overall d_pose falls; at n256
its three additions are 307,876, 1,119,452, and 924,860 bytes. This is measured allocation
attribution, not a causal leave-one-out PoseNet result. A causal stream binder remains explicitly
unmeasured rather than inferred.

Pose never passes at a finite tau. The monotone Pose improvement alongside worsening Seg shows why
the Fisher/margin selector alone is insufficient: placing exact RGB values at selected pixels does
not account for the receiver-to-scorer inner Jacobian and convolutional context.

## Resize/preimage freedom

The cheap derived 2x-bilinear linear probe places 18.40%-24.57% of masked RGB-delta L2 energy in
the resize nullspace. Only one to three active 2x2 blocks per rung are exactly mean-zero, so no
counted-byte saving is claimed. This is `DERIVED_PREIMAGE_FREEDOM_NOT_SCORE`, not a through-R or
evaluator result.

## Receiver, custody, and resumability

- The mask is encode-side only. Counted correction positions and exact RGB values are consumed by
  the inherited v7 receiver; no scorer weights, mask semantics, or GT argmax table are shipped.
- Each of four tau stages and four candidate stages is atomically preserved per window. Compiler
  replay, parse/re-encode identity, deterministic receiver replay, and exact ZIP byte homes pass.
- Round-1 sealed validation binds committed producer custody, candidate-table hash, exact candidate
  checkpoint rows/argv, all six tau frames and section ledgers, archive/receiver/frame homes,
  per-stream rows, resume paths, and the SHA-bound v7 endpoint.
- Same-output sealed validation measured 5.21 seconds (n64) and 17.79 seconds (n256).
- Local evidence totals 24,022,174 B and 86,496,913 B. SSD is read-only under delegated authority,
  so certify-or-block preserves all bytes and records no deletion or move.

Receipt anchors:

- n64 `73217e3bd8649978c8da8dc3f1f30215c3022838f354304b90673f6aa1cc683f`
- n256 `99e0e1e9f639a2d42140c97306c9d8b50fd3b883cddd42ba4e2b55bcf267886e`
- cross-window `7051927df863a3ab01a6e1494550a914829715b00faeae15baa3abb951a49d1c`

## Round-1 adversarial review

Disposition: `PASS_AFTER_THREE_CUSTODY_FIXES`.

1. Embedded settled-v7 configs initially failed strict Python-mode tuple parsing. Fixed by
   revalidating the bound object through strict JSON mode; regression added.
2. The inherited exact endpoint used a nested evaluator bridge but the first joint-gate read used a
   flat schema. Fixed with one validated bridge extractor and explicit inherited Seg/Pose gate.
3. Completed-receipt replay initially under-bound candidate/checkpoint/tau state. Fixed with the
   full sealed validator above and fail-closed receiver manifest checks.

The two pre-round1 receipts are preserved under their original output roots with SHA-256
`6efe099e7357493b736d6ce2c9b68e39156acd5e9c0a5028f5b5a7b79d0f31c1` and
`10063fc4226ce136509a1818d88205015fb21e1b526ee3a8a997c828abeade52`.
Ruff and the focused suite are green: **51 passed**.

## Blocker delta

- GREEN, local advisory: solved-plane margin mask, exact receiver bytes per tau, joint Seg/Pose
  measurement, per-stream byte/mask decomposition, derived resize-null probe, and sealed resume.
- RED, formulation-scoped: any measured `d_seg <= 0.00116` endpoint within 200 KB; the only joint
  endpoint remains 43.1/171.3 MB.
- RED: causal Pose stream ablation, optional argmax-safe in-mask quantization, n600, contest CPU/CUDA,
  candidate archive, score claim, and promotion.
- Canonical #603 remains `8/19` on this branch. MAIN reviews the append-only draft row.

## Bounded re-derivation argv

Both same-output invocations perform sealed validation in far under ten minutes on this host:

```text
.venv/bin/python tools/run_direct_description_entropy_priced_member.py --config .omx/research/ddm_v8_margin_gated_correction_n64_603_613_20260722T104341Z.config.json --output-dir .omx/research/ddm_v8_margin_gated_correction_n64_603_613_20260722T104341Z_rerun1 --execution-allowed false
.venv/bin/python tools/run_direct_description_entropy_priced_member.py --config .omx/research/ddm_v8_margin_gated_correction_n256_603_613_20260722T104341Z.config.json --output-dir .omx/research/ddm_v8_margin_gated_correction_n256_603_613_20260722T104341Z_run1 --execution-allowed false
```

## STORES CONSULTED

- `CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`
- `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`
- SHA-bound v7 receipts `8db93c4e...`, `d68f1d9e...`, and `64658a05...`
- `gt_n600` solved planes, lstars, margins, and gt_poses frozen scorer cache
- 2026-07-19 reverse-waterfill, Fisher/margin, corrected-inner-Jacobian, curvelet/shearlet, and xi directives

Pointer honesty: **0.1910828242 [contest-CPU] — unchanged.**
