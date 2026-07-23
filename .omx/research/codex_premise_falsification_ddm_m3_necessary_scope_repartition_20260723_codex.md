---
schema: ddm_m3_necessary_scope_repartition_findings.v1
date_utc: 2026-07-23
lane_id: lane_ddm_m3_necessary_scope_repartition_20260723
research_only: true
execution_allowed: false
score_claim: false
promotion_eligible: false
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
verdict: NUMERIC_TRUE_SCOPE_NOT_CERTIFIABLE_CURRENT_CUSTODY
verdict_scope: "INSTANCE:c1 x v14 x v19b x landed G2/G2G2 evidence; no full-stratum solve, v19c saturation, solve-then-correct common master, or current-vehicle frame0 Pose-preimage"
pointer: "0.1910828242 [contest-CPU Linux x86_64]"
pointer_moved: false
main_landing_review_required: true
---

# DDM M3 necessary-scope repartition — premise falsification

## Operator answer first

**No.** The current system is not multicoefficient inverse-solving everything possible before
preparing #366 joint descent. G2G2 measured only six selected pairs and 20 Lane centerline
coordinates, admitted `0/6`, and explicitly refused n600. The later G2 atlas covers all 600 pairs
but has `INNER_ENCODER_JACOBIAN_NOT_MEASURED_PRIMARY_PATH` and
`BLOCKED_NO_RECEIVER_DELTA_DSEG`; it is not an inverse-solve reach measurement.

The opposite error is also forbidden: current custody cannot prove that all 2,377,273 c1 residual
errors need #366. V19b already demonstrates that the pre-synergy partition is stale: its exact
n600 common master removes 73,945 net Road+Undrivable+MyCar errors and 29,377 net Lane+Movable
errors for one shared `+3,884 B` stack, with `+0.080496721217` score units of amplification.
However, that v19b row starts from v15, not from a full-stratum inverse-solved master. V19c is
absent. The solve-then-correct conditional gain is therefore unmeasured.

The requested verdict **“#366 over-scoped by X%” is not numerically certifiable**. The measured
stale-partition evidence is `73,945 / 2,377,273 = 3.110496775%`; subtracting it gives a
**counterfactual current-master residual** of 2,303,328. Neither number is the certified
post-saturation #366 scope. The honest true-scope interval remains `[0, 2,377,273]` until the
missing composed measurement lands.

This is not an `ESCALATE`-to-cancel result: no receipt shows solve+correct nearly closes the box.
It is an **ESCALATE-BEFORE-FIRE scope blocker**: MAIN/operator should not interpret the cleared
#366 apparatus as proof that its 2.377M-error target is minimal.

## SOLVE/DESCEND frontier

All v19b class rows are exact net effects through the same real receiver and frozen scorer.
`+flip` means fewer errors; `-flip` means net harm. The 3,884 bytes are shared by the whole stack
and cannot be allocated per class. Off-target collateral is not decomposed per class in the
n600 receipt; Undrivable's net harm is directly measured. `UNKNOWN` is an epistemic status, not
zero.

| stratum | frame | c1/v14 control errors | multicoefficient SOLVE reach | v19b correction on v15 common master | v19c top-up | frame separation | certified-infeasible residual |
|---|---|---:|---|---|---|---|---|
| Road | frame_1 | 2,210,770 | **UNKNOWN**; no full-stratum receiver solve, bytes/collateral unknown | `+82,824` flips; class `Delta d_seg=-0.003021996606`; 2,127,946 remain; shared +3,884 B | **PENDING** | Seg-bearing | **UNKNOWN** |
| Lane | frame_1 | 300,563 | **SUBSET ONLY**; G2G2 six pairs/20 centerline coordinates, 0/6 admitted, n600 refused | `+2,003`; class `Delta d_seg=-0.002900212702`; 298,560 remain; shared +3,884 B | **PENDING** | Seg-bearing | **UNKNOWN** |
| Undrivable | frame_1 | 236,896 | **UNKNOWN**; no full-stratum receiver solve, bytes/collateral unknown | **`-25,191` net harm**; class `Delta d_seg=+0.000431254666`; 262,087 remain; shared +3,884 B | **PENDING** | Seg-bearing | **UNKNOWN** |
| Movable | frame_1 | 425,853 | **UNKNOWN**; no full-stratum receiver solve, bytes/collateral unknown | `+27,374`; class `Delta d_seg=-0.018745142348`; 398,479 remain; shared +3,884 B | **PENDING** | Seg-bearing | **UNKNOWN** |
| MyCar | frame_1 | 66,446 | **UNKNOWN**; no full-stratum receiver solve, bytes/collateral unknown | `+16,312`; class `Delta d_seg=-0.000543851005`; 50,134 remain; shared +3,884 B | **PENDING** | Seg-bearing | **UNKNOWN** |
| Road | frame_0 | 0 Seg obligation | N/A for Seg | N/A for Seg | N/A | exact Seg-null; Pose-only preimage **open** | **0 Seg errors** |
| Lane | frame_0 | 0 Seg obligation | N/A for Seg | N/A for Seg | N/A | exact Seg-null; Pose-only preimage **open** | **0 Seg errors** |
| Undrivable | frame_0 | 0 Seg obligation | N/A for Seg | N/A for Seg | N/A | exact Seg-null; Pose-only preimage **open** | **0 Seg errors** |
| Movable | frame_0 | 0 Seg obligation | N/A for Seg | N/A for Seg | N/A | exact Seg-null; Pose-only preimage **open** | **0 Seg errors** |
| MyCar | frame_0 | 0 Seg obligation | N/A for Seg | N/A for Seg | N/A | exact Seg-null; Pose-only preimage **open** | **0 Seg errors** |

Frame 0's zero is structural: `SegNet.preprocess_input` selects only `x[:, -1]`. PoseNet reads
both frames, so Pose remains a global six-output constraint and has no honest per-class
allocation. The current-vehicle conditional frame-0 Pose rate is unmeasured. Under the explicit
quarantine waiver, R1 is cited only as the nontransferable comparator
`d_pose=0.001610, 7,195 complete bytes`; no R1 bytes or weights are consumed here.

## Coverage gap — what is not being inverse-solved

The “everything possible” audit is red on every Seg-bearing stratum:

1. Road, Undrivable, Movable, and MyCar have no full-stratum multicoefficient receiver solve.
2. Lane has only the G2G2 six-pair/20-centerline subset; nonlinear relinearized coordinates,
   full grammar families, and n600 are open.
3. G2's n600 `compact_parabolic_shearlet`, `rank4_head_chart`, and other operator byte rows do not
   cure the gap: receiver `Delta d_seg` is absent, so its KKT status is blocked.
4. The current-vehicle frame-0 conditional Pose-preimage race is preregistered but unmeasured.
5. V19c saturation and a solve-then-correct sequential master are absent.

Until these exist, “joint descent is absolutely necessary for the remainder” is a hypothesis,
not a certificate.

## Named missed synergies

- **SOLVE x SE correction — unmeasured.** V19b starts from v15; independent subtraction after a
  hypothetical solve is invalid because SegNet squeeze-excite makes same-frame responses
  nonadditive.
- **Correction x correction — measured positive.** V19b measures
  `+0.080496721217` amplified gain; eight of ten ordered moves amplify, up to the cited `9.058356x`.
- **Frame_1 Seg x frame_0 Pose — structurally separable in incidence, rate unmeasured.** Frame 0
  owes no Seg fidelity, but PoseNet couples both frames. The correct missing race fixes frame 1
  and minimizes a conditional frame-0 Pose preimage before assigning that work to joint descent.
- **Per-class correction effects are not additive byte homes.** The 3,884-byte v19b stack is one
  shared receiver program. Per-class byte division would be fake precision.

## Scope decision for #366

Do not change the #366 config from this branch. At MAIN/operator review:

- preserve #366 as a prepared fallback/trunk apparatus;
- withhold the claim that 2,377,273 errors are its **minimal** necessary scope;
- require one sequential exact chain
  `v15 -> full-stratum solve -> correction saturation -> conditional frame split`;
- only then compute `X%` and decide whether to shorten/narrow #366 or cancel it.

The resulting row must include complete bytes, exact per-class errors, helpful/harmful collateral,
Pose, v19c saturation status, and immutable stage hashes. A failed finite formulation narrows only
its named scope.

## Re-derivation

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  tools/audit_ddm_m3_necessary_scope_repartition.py \
  --c1-ledger .omx/research/ddm_c1_composed_candidate_ledger_603_613_20260723.json \
  --v14-receipt .omx/research/ddm_v14_realization_fidelity_n600_20260722T215500Z/ddm_v14_realization_fidelity_n600_receipt.json \
  --v19b-receipt .omx/research/ddm_v19b_joint_remeasure_stack_20260723T051914Z/ddm_v19b_joint_remeasure_stack_receipt.json \
  --g2-aggregate-ledger .omx/research/ddm_g2_solve_diff_op_mining_n600_20260722T194000Z/aggregate_ledger.json \
  --output .omx/research/ddm_m3_necessary_scope_repartition_receipt_20260723.json
```

The helper SHA-pins all four inputs, cross-checks class/bucket sums, and refuses to promote G2
while receiver `Delta d_seg` or the inner encoder Jacobian is absent.

## Evidence and stores consulted

- c1 ledger SHA-256 `14fdf1570b43df65ac949fe157e68ea328ff584f7df1331acf25cca8f900d936`.
- v14 receipt SHA-256 `82d3249908d42a86575c407ab3d7acdf9b3706b31225f2e46862b2472966e5a9`.
- v19b receipt SHA-256 `4bb5d6b4b793b667c7cbe15e37cbf9a27f6c0e75451374839fb5df8ca1c1b8e8`.
- G2 aggregate SHA-256 `061220fd8c1ca047b210841235fc805194a96175e933ee110ba4ac8bb2077d84`.
- G2G2 durable receipt at historical commit `8894c03db5ca1c68bd6865abf890d4a81b122fdf`;
  full SSD receipt SHA-256 `928d3cd74cc92ef52aa9f821229ada12fbf4c3e9dad772e8a76adffcfcfcb078`.
- `CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`; c1 spec/equations/DAG;
  #602 MDL receipt; #49/S12 and #580 full-kernel custody; M1 frame-0 audit/equations/DAG;
  R1 comparator receipt; current frontier, lane, task, and inbox surfaces.

No launch, paid dispatch, archive mutation, #366 config edit, exact contest eval, score claim, or
pointer movement occurred.

