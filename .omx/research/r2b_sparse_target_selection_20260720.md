# R2b sparse target selection — n600 measured negative

**Verdict:** `FORMULATION_RATE_DEAD_KKT_STOP_ZERO`. The one-bit, source-sign-chosen, fixed-magnitude same-rounded-bin factor-2 stream is a real receiver-replayable lever and reduces the hard SegNet error, but it is not economical. This is a formulation-scoped negative; sparse target selection, corrected inner-Jacobian selection, curvelet/shearlet coordinate grammars, and xi-factorized Pose remain open.

Axis: `[macOS-CPU advisory]`, `score_claim=false`, `promotion_eligible=false`. Pointer `0.1910828242 [contest-CPU Linux x86_64]` **UNMOVED**. MAIN review and merge are required.

## Measured outcome

The official-geometry CPU-Torch replay re-derived the capstone gap as `17,926` Seg flips, `d_seg=0.00015196058485243057`, `d_pose=0.00010184347386600314`, and nonrate `S=0.047108982805336805`. The live batch-16 target differs from the sha-pinned batch-32 cache at exactly three labels; live batch-16 labels are score authority, while the cache is used only for edge context. This reproduces the inherited report-rounded `0.00015196 / 0.00010184` row without laundering the three-cell geometry difference.

Of the `17,926` flip cells, `16,751` had an exact bounded-uint8 signed rounding-bin realization; `1,175` were `HARD_REJECT`. The full feasible stream is `27,213 B`; its canonical one-member charged archive is `27,313 B`, below the `70,748 B` absolute ceiling. Nevertheless, the very first 64-decision knee costs `342 B` and returns only a scheduled `5.4253472e-5 S`, or `1.58636e-7 S/B`, below the contest price `6.65859e-7 S/B`. The reverse-waterfill/KKT stop is therefore **zero decisions**.

I still hard-scored the full feasible knee as a bounded falsification control. It fixes `1,585/16,751 = 9.462%` of selected cells, reducing `d_seg` to `0.00013852437337239583`, but worsens `d_pose` to `0.0001025492627354498`. Net nonrate recovery is only `0.0012332316583976016 S`; the charged rate cost is `0.018186605586625867 S`, so the composed delta is a `+0.016953373928228266 S` regression. At the realized gain the stream must fall to `<=1,852.09 B`, a further `14.75x` byte reduction, merely to break even.

## Contribution histogram

Every Seg flip carries `8.477105034722222e-7 S`. Source-class counts are Road `7,831`, Lane `2,556`, Undrivable `3,622`, Movable `1,346`, MyCar `2,571`. Necessity is concentrated on edges: Road-Lane `5,193`, other edges `12,468`, non-edge only `265`.

Margin/Fisher strata are: `[1e-6,1e-5)` `16`; `[1e-5,1e-4)` `143`; `[1e-4,1e-3)` `1,448`; `[1e-3,1e-2)` `9,187`; `[1e-2,1e-1)` `7,110`; `[1e-1,1)` `22`. Thus tie-tight `<1e-3` holds `1,607` flips (`0.00136227 S`), the `[1e-3,1)` margin band holds `16,319` (`0.01383379 S`), and there are zero `>=1` interior flips.

Pose error is almost entirely dimension 0: per-dimension MSE is `[6.106607479e-4, 2.108630524e-7, 1.429034401e-7, 1.659929928e-8, 3.912838189e-9, 2.581665010e-8]`. A frame1-only cell stream therefore attacks the smaller Seg term while leaving the dominant Pose coordinate structurally unfactored.

## Charged curve

| decisions | charged B | scheduled recovered Seg S upper bound | marginal S/B | admitted |
|---:|---:|---:|---:|:---:|
| 0 | 0 | 0 | — | yes |
| 64 | 342 | 0.0000542535 | 1.586e-7 | no |
| 256 | 847 | 0.000217014 | 3.223e-7 | no |
| 1,024 | 2,535 | 0.000868056 | 3.857e-7 | no |
| 4,096 | 8,292 | 0.00347222 | 4.523e-7 | no |
| 8,375 | 15,533 | 0.00709958 | 5.009e-7 | no |
| 12,563 | 21,906 | 0.01064979 | 5.571e-7 | no |
| 16,751 | 27,313 | 0.01420000 | 6.566e-7 | no |

The curve is an oracle-scheduled upper bound from baseline flip mass; only the full-feasible candidate is a measured hard-oracle row. Its realized nonrate recovery is `8.68%` of the scheduled Seg upper bound, directly exposing the missing interaction/inner-Jacobian term.

## Receiver, custody, and cleanup

The stream stores sorted gap-ULEB coordinates plus bit-packed signs, Brotli-Q11 compressed, with strict canonical parse-back. For each decoded decision the receiver derives the exact signed numerator from the rounded plane, coefficient gcd, and rounding-bin boundary, solves the bounded four-tap integer block, and refuses unless the numerator is exact and the rounded byte is unchanged.

External receipt: `/Volumes/VertigoDataTier/pact/evidence/r2b_sparse_target_selection_20260720T1621Z/receipt.json`, `8,195 B`, SHA-256 `c86d15ff905c1de912c6a66fe63c2122bd69115a536f70265c59d9ae8cd34a68`. Stream SHA-256 `cf304b13fedb081ee2323a7c1631cf5e1f9b411533063a2d934b825b8d893c02`; charged archive SHA-256 `095353f3ae6cd217965a2d80073393bf3eff30fdecba2781ca48697dd618a023`.

The `3,662,409,600 B` candidate raw (SHA-256 `9b30d56acf7ebc201fcb126832b1c337e00acdbe9aed8f9e698388b91d9b1f5f`) was deterministic rebuildable scorer scratch. After all 38 batch stages and its hash were preserved, a fsynced cleanup receipt (SHA-256 `8bdea15e8943e42c3b2f069926bbc8e7adeca96a7332df90f52bcc09b744d0bf`) authorized deletion; source and target raws remain untouched.

## Bounded self-review

The single bounded review found that the first source-distance implementation squared `int16` differences before summing, which can overflow. It was corrected to promote pixels to `int32` before subtraction and squaring, and `test_choose_source_sign_uses_nonoverflowing_distance` fixes the bug class. The invalid pre-fix scorer stages are preserved but quarantined as `candidate_stages_superseded_int16_distance_3be0f925`; they are not consumed. Current stage directories are bound to target-raw SHA, baseline-raw SHA, scorer-binding SHA, and candidate-stream SHA.

## Reformulation edge and triality

The next coordinate is not another blanket sign stream. Use corrected first-order + secant + QP inner-Jacobian selection to raise the measured `9.46%` realization fraction, encode positions in a receiver-bound curvelet/shearlet grammar to target the required `14.75x` byte reduction, and factor the dominant Pose dimension through xi rather than frame1-only cells.

- DSL/receiver: real parse-back and exact bounded-uint8 consumer over the existing V10 rounded-plane descriptor; no launch lever.
- Equations: consumes the bounded factor-2, Fisher/margin, reverse-waterfill/KKT, inner-Jacobian, and curvelet/shearlet laws; registers no new law.
- DAG: `M2 exact-source rate-dead -> R2b signed sparse stream -> KKT zero + measured 9.46% realization -> corrected-Jacobian/curvelet/xi successor`.

Stores consulted: delegated authority; `CLAUDE.md`; `AGENTS.md`; operating manual; v7.5 §8; inherited M2 memo/receipt/tool; V10 capstone and y-hat receipts; sha-pinned n600 cache; C1 and M2 exact raw custody; broadcast through `2026-07-19T19:48:01Z`.

**Pointer delta: 0. MAIN landing review is mandatory.**
