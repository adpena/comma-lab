---
schema: dag_feed.v1
task: 603
feeds_task: 613
master_task: 578
lane_id: ddm_v8_margin_gated_correction
research_only: true
execution_allowed: false
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
main_landing_review_required: true
---

# FEED-603 -> 613: margin-gated correction DAG

```text
SHA-bound solved target planes + frozen SegNet
  -> batch16 solved top1/top2 margins and argmax
SHA-bound v6 fixed_ar1_hold24 predictor + frozen SegNet
  -> predictor argmax
  -> G_tau = [abs(target margin)<tau] OR [predictor argmax != target argmax]
  -> solved-argmax six-stream partition with Boundary precedence
  -> exact target RGB corrections on G_tau; predictor elsewhere
  -> v7 Brotli-Q11/LZMA-XZ-9e section tournament
  -> receiver-closed exact ZIP (no scorer, GT argmax table, or receiver mask semantics)
  -> frozen SegNet + official-YUV6 PoseNet advisory bridge
  -> exact ZIP homes + mask/stream table + derived resize-null probe
  -> discrete measured Pareto envelope at 25/37,545,489
  -> n64/n256 finite-tau formulation falsifier
  -> immutable cross-window receipt/register draft -> FEED-613
```

## Edge-state delta

| edge | state | evidence |
|---|---|---|
| margin gate and solved-argmax partition | GREEN_LOCAL_ADVISORY | four exact tau checkpoints, batch16, six disjoint streams |
| receiver closure | GREEN_LOCAL_ADVISORY | exact final ZIP, parseback, x2 compiler, no scorer/GT table |
| joint Seg/Pose guard | GREEN_LOCAL_ADVISORY | both evaluator legs measured at every tau |
| tight-mask byte collapse | GREEN_BUT_INSUFFICIENT | 93.90% n64 / 94.54% n256 reduction versus v7 exact |
| finite-tau `d_seg <= 0.00116` | RED_FORMULATION_SCOPE | best finite n256 d_seg 0.025907576084 |
| finite-tau Pose guard | RED_FORMULATION_SCOPE | best finite n256 d_pose 31.003158645132 |
| feasible correction `<=200 KB` | RED_FORMULATION_SCOPE | tight n256 already 9,360,569 B; exact endpoint 171,332,654 B |
| resize preimage freedom | DERIVED_ONLY | 18.40%-24.57% null-energy; no byte or score claim |
| contest CPU/CUDA / score | REFUSE | unauthorized; pointer unchanged |

## Unified-stack wire-in

1. Sensitivity map: every row records selected/mismatch/low-margin sites by solved stratum. Boundary
   dominates added margin payload; Movable/Road/Boundary dominate the mandatory mismatch archive.
2. Pareto constraint: the measured route is tau0 -> tau0.1 -> tau0.5 -> tau1 -> exact. The first
   rate break is tau1 -> exact in both windows; no finite point satisfies evaluator constraints.
3. Bit allocator: finite-tau exact-value corrections are ineligible. The optional in-mask quantized
   axis remains a distinct unmeasured formulation, not an inferred rescue.
4. Cathedral/autopilot: emit no launch. Refuse n600 for this formulation after the n256 46.80x
   tight-mask rate floor and evaluator misses.
5. Continual learning: cross-window receipt
   `7051927df863a3ab01a6e1494550a914829715b00faeae15baa3abb951a49d1c` is the empirical anchor;
   canonical #603 remains 8/19 pending MAIN review.
6. Probe disambiguator: margin/Fisher selection and corrected-inner-Jacobian realization are now
   separated. A successor must arbitrate exact-value masking versus realized argmax-safe updates on
   receiver-closed bytes, with a causal Pose stream ablation if claiming a pose binder.

Pointer `0.1910828242 [contest-CPU]` unchanged.
