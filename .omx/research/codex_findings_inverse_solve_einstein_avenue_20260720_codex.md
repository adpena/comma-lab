# Codex findings — inverse-solve Einstein Avenue (2026-07-20)

`lane_id=inverse_solve_einstein_avenue_20260720T092250Z` · `research_only=true` · `$0 local` · no training · no provider/GPU dispatch · pointer unchanged

## Verdict

**FORMULATION-SCOPED EMPTY INTERSECTION.** No currently custodied exact or receiver-closed description satisfies all three fixed-C1 constraints: `archive.zip <= 216,223 B`, `d_seg <= 0.00015196`, and `d_pose <= 0.00010184`. The best exact byte-closed fixed-distortion evidence remains the carried C1 archive at `409,526,925 B`, `d_seg=0.00015196`, `d_pose=0.00010184`, and canonical `S=272.73427793588485` on `[contest-CPU Linux x86_64]`; this lane produced no new score row and makes no pointer claim.

Machine-readable receipt: `inverse_solve_einstein_avenue_20260720T092250Z_receipt.json`.

## The corrected intersection

The measured C1 distortion term is

`100(0.00015196) + sqrt(10(0.00010184)) = 0.04710838004286111`.

Against the live `[contest-CPU]` pointer `0.1910828242`, the continuous byte crossing is `216223.63637531892 B`. Because the score must be strictly lower, the maximum integer archive size is **216,223 B**:

| archive bytes | derived score at fixed C1 distortion | pointer relation |
|---:|---:|---:|
| 216,223 | 0.19108240046379638 | below by 4.237362036e-7 |
| 216,224 | 0.1910830663227495 | above by 2.421227495e-7 |
| 216,300 | 0.19113367160318678 | above by 5.084740319e-5 |
| 264,320 | 0.22310821853211343 | above by 0.03202539433 |

Thus the handoff's `~216,300 B` is a useful order-of-magnitude pointer but not an admissible strict cap. The `264,320 B` outer box is also not a fixed-C1 crossing. Reaching the exact intersection requires removing **409,310,702 B** from C1, a `1894.00x` reduction to `360.372 B/pair` amortized.

## What was actually tested or already settled

| coordinate/formulation | custodied result | admission |
|---|---|---|
| exact C1 scorer planes | `409,526,925 B`; exact shipped bytes; C1 distortions; contest-CPU exact eval | distortion-valid, rate-invalid |
| positive Seg/Pose band | `1,474,579.92 B/pair`, `d_seg=0`, `d_pose=2.521975392e-5` on n24 | range-coordinate only; no spatial/RGB receiver or archive; also `4091.83x` the amortized cap |
| PDW2 quotient target | `138 B` raw / `133 B` Brotli-q11, exact target parse-back | target-valid; no spatial/RGB pullback, through-R distortion, or archive score |
| Cole–Hopf/Gibbs target | full-n600 codomain target in `2.5133 s`, mean top-one `0.9958448` | initializer only; no RGB/uint8 preimage |
| settled R1 direct-RGB + xi | `89,772 B`, `d_seg=0.0045491197374`, `d_pose=0.001609547154`, `S=0.6415553932` `[macOS-CPU advisory]` | rate-valid outer-box instance, but `d_seg` is `29.9363x` C1; archive not retained |
| fresh xi packet | `91,062 B`, SHA-custodied | strict receiver gate failed on undefined `_CP_XI_FX`; no decode and no distortion row |

The measured exact-plane lossless family is already rate-walled and was not rerun. The 133-byte PDW2 result proves that quotient target complexity is tiny; it does **not** prove that the conditional spatial/RGB pullback is tiny. That missing pullback—not another entropy coder—is the binding object.

## Exact formulation scope

This verdict walls the intersection of the tested exact-plane, range-coordinate, PDW2 target-only, Cole–Hopf target-only, and settled R1 direct-generator instances. It does **not** reject inverse solving, `W=(G,xi,T)`, curvelet/shearlet spatial carriers, corrected xi receivers, or branchwise active-set preimage solvers.

The empty set is:

`{a in current custodied rows : bytes(a)<=216223 and d_seg(a)<=0.00015196 and d_pose(a)<=0.00010184} = empty`.

No new exact eval was fired because there was no new member of that typed set: reevaluating unchanged C1 adds no information, while scoring a target-only or failed-receiver packet would create false authority.

## Next optimal-form recursion

After MAIN review, the next no-training build should be a **PDW2-conditioned Fisher/inner-Jacobian spatial pullback reverse-waterfill**:

1. Keep the native-float32 133-byte PDW2 head target fixed.
2. Solve only its missing per-stratum spatial/RGB pullback, factor Pose through `xi`, and represent residual edge corrections in a curvelet/shearlet basis.
3. Rank cells by winner-rival Fisher margin and the operator-routed necessity/resize field; predict realization with the corrected first-order + secant + active-set QP inner Jacobian, not naive first-order.
4. Charge container, runtime, target, xi, and residual bytes together. At fixed C1 distortion the entire archive must be at most `216,223 B`; the 133-byte target leaves at most `216,090 B` before container/runtime charges.
5. First produce one n24 archive with strict parse-back, hard CPU-Torch through-R distortions, retained mismatch bitmap, and per-stratum byte ledger. Do not project it to n600 authority. Only a receiver-closed pass authorizes an n600 exact archive/eval.

For a coarsening that saves `B_saved>0` bytes, the exact KKT admission test is

`100*Delta_d_seg + sqrt(10*(d_pose+Delta_d_pose)) - sqrt(10*d_pose) < 25*B_saved/37545489`.

For a repair adding `B_add>0` bytes, admit only when the same distortion-term delta plus `25*B_add/37545489` is negative.

This recursion attacks the conditional preimage code length directly and remains inside the frozen scorer/no-training charter.

## Round-1 adversarial review

- **Cap attack:** `216,300 B` and `264,320 B` were recomputed, not inherited. Both fail the fixed-C1 pointer inequality; the strict cap is `216,223 B`.
- **Coordinate-type attack:** positive-band bytes are range residuals, and PDW2/Cole–Hopf objects are targets. None were relabeled as receiver/archive bytes.
- **Cheap-archive attack:** the `89,772 B` xi row was not composed with C1 distortion; its measured distortion and advisory axis remain attached.
- **Failed-packet attack:** the `91,062 B` packet supplies bytes plus an implementation blocker only. No score or distortion is inferred.
- **Naive-negative attack:** the whole inverse family remains open. The untested optimal receiver-closed formulation is named above.

Round-1 result: **PASS WITH FORMULATION-SCOPED WALL**.

## Triality, DAG feed, and pointer delta

- **DSL:** no typed DSL mutation in this evidence-only lane. Any follow-on must reuse declared native-float32 receiver, xi, and curvelet/shearlet surfaces; no invented flags.
- **DAG (`FEED-inverse-solve-einstein-avenue-20260720`):** exact C1 custody -> exact cap derivation -> formulation/type audit -> empty admitted intersection -> PDW2-conditioned receiver pullback -> n24 parse-back/hard-oracle gate -> n600 archive/eval gate.
- **Equations:** frozen score law, strict integer cap equation, and marginal KKT admission law above.
- **Pointer delta:** none; `0.1910828242 [contest-CPU Linux x86_64]` remains unchanged.

## MAIN landing review required

MAIN must review before merge or follow-on execution, focusing on:

1. the `216,223 B` strict-cap correction;
2. the evidence boundary between target/range bytes and actual archive bytes;
3. whether the PDW2-conditioned receiver-closed recursion is the right optimal-form next build.

No promotion, pointer movement, paid dispatch, or full n600 launch is authorized by this landing.
