# DAG FEED — BEV staticity v2 absolute trajectory

`research_only=true`; `[macOS-CPU advisory]`; pointer unchanged; no launch, score, promotion, or
dispatch authority. MAIN landing review is required.

```text
hash-pinned gt_n600 RGB + f1 labels ──> singleton f0/f1 scorer custody: 0/600 mismatch
hash-pinned G1 calibration ────────────┐
exact PoseNet f1[t-1] -> f0[t] ────────┼─> phase-consistent A_f0[t], A_f1[t]
cached PoseNet f0[t] -> f1[t] ─────────┘                 │
                                                        v
                               D0 largest bottom-connected hood
                         n64 p50=0.0 px, static=0.922991: PASS
                                                        │
                                                        v
                  n600 Road/Lane oriented shallow-side absolute-BEV D1/D2
                     Road p50=39.0226, static=0.043093: NEGATIVE
                     Lane p50=47.1192, static=0.043713: NEGATIVE
                                                        │
                                                        v
            D3 BLOCKED: no static coefficients / xi B-spline / sparse events packet
              U1/U2=false · U5=false · P0=false · dispatch_authority=false
```

## Typed edge dispositions

| producer | consumer | edge | disposition |
|---|---|---|---|
| singleton f0/f1 scorer stages | absolute-trajectory D0 | source/label custody | **PASS 0/600** |
| exact cross + cached within PoseNet targets | dual absolute charts | cross-then-within SE(3) | **PASS** |
| `A_f0` + bottom-connected hood | all downstream ground interpretation | positive control | **PASS n64/n600** |
| `A_f1` + oriented Road/Lane boundaries | D1 staticity | necessary condition | **FAILED this chart** |
| directrix/ruling residual | D2 developability | C3-safe statistic | Road/Lane fractions about 0.043 |
| D1/D2 hold | D3 describe-line packet | admission gate | **BLOCKED** |

## Reactivation criteria

1. Supply a new hash-custodied absolute-motion source or independently admitted calibration. True
   absolute ego GT is in scope; reusing the v1 pairwise-as-absolute error is not.
2. Preserve exact singleton f0/f1 custody and the n64/n600 hood gate: p50 <=1 scorer px and static
   fraction >=0.5.
3. At n600, require both Road and Lane p50 <=1 scorer px and static fraction >=0.5 in the same
   absolute chart before running D3.
4. If D3 opens, keep the operator's Fisher/margin reverse-waterfill and curvelet/shearlet-only
   residual policy, stop at rate break-even, and require through-real-receiver scorer admission
   before any byte or score claim.

The exact G1-calibrated PoseNet chart is closed for static-ground D3. This feed does not close true
absolute ego GT, other admitted calibrations, or the broader BEV/worldsheet representation family.
