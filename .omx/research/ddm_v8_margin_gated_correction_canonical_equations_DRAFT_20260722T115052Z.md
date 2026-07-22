---
schema: canonical_equation_candidate_note.v1
task: 603
feeds_task: 613
lane_id: ddm_v8_margin_gated_correction
structured_carriers_anchor_ordinal: 5
research_only: true
registry_promotion: false
main_landing_review_required: true
---

# DRAFT fifth structured-carriers anchor — margin gate is selector, not realization law

This is a registry draft. MAIN decides whether to register it after reviewing the code and
SHA-bound receipts; this branch does not mutate the canonical equations registry.

Let `T_t` be the solved uint8 pair, `P_t` the bound v6 predictor, and let frozen SegNet on the last
plane produce target winner `c*_t(x)`, target top1-top2 margin `m_t(x)`, and predictor winner
`c^P_t(x)`. For registered threshold `tau`, define the encode-side support

```text
G_tau(t,x) = 1[abs(m_t(x)) < tau OR c^P_t(x) != c*_t(x)].
```

For the six disjoint solved-argmax strata `M_s` (Boundary has precedence), the measured v8
correction is

```text
C_{s,tau}(t,p,x) = M_s(t,x) G_tau(t,x) [T_t(p,x) - P_t(p,x)],  p in {0,1}
W_tau            = P + sum_s C_{s,tau}
A_tau            = ReceiverClosedZIP(P, {EntropyCode(C_{s,tau})}_s)
B_tau            = len(A_tau).
```

`G_tau`, scorer weights, and target argmax labels are not receiver inputs. Correction positions and
values are counted in `A_tau`; the receiver only applies them.

The evaluator and admission law extend v7 unchanged:

```text
D_seg(tau)  = mean 1[argmax SegNet(R(W_tau)) != lstar]
D_pose(tau) = mean (PoseNet(YUV6(W_tau))[:6] - gt_pose)^2
D(tau)      = 100 D_seg(tau) + sqrt(10 D_pose(tau))
S(tau)      = D(tau) + (25/37,545,489) B_tau.
```

The constrained knee is `argmin B_tau` subject to `D_seg<=0.00116` and
`D_pose<=0.00025`, over measured finite tau plus the SHA-bound exact endpoint. Both windows select
the exact endpoint; n256 requires 171,332,654 bytes.

Crucially, `G_tau` is a Fisher/margin **selection** law, not a realization theorem. Pixelwise exact
replacement on `G_tau` does not guarantee the target scorer cell because SegNet has spatial
receptive-field coupling and the receiver-to-logit inner Jacobian is omitted. Empirically,
increasing tau from 0 to 1 worsens d_seg in both windows while improving Pose. Therefore the fifth
anchor is the fail-closed separation

```text
margin_selector != argmax_realizer.
```

A successor may replace `T-P` with a corrected-inner-Jacobian/secant/QP realized update and/or an
argmax-safe quantizer, but it must retain exact receiver bytes and the same joint evaluator.

For the cheap resize probe, orthogonally decompose masked RGB delta under the linear 2x block-mean
operator `R_lin`:

```text
delta = Pi_row delta + Pi_null delta,
||delta||^2 = ||Pi_row delta||^2 + ||Pi_null delta||^2.
```

The measured null-energy fraction is 0.183956747048-0.245672690936. This is a derived linear
preimage fact only; it does not authorize a through-R or byte-saving claim.

Receipt anchor:
`7051927df863a3ab01a6e1494550a914829715b00faeae15baa3abb951a49d1c`.

STORES CONSULTED: DAG `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md` (FEED-603-v7 falsifier row + FEED-REALIZATION-VERDICT-20260721T210858Z #612 fork + FEED-V9-RECURSIVE-FRACTAL-ALREADY-BUILT + FEED-STANDING-RECALL-V7-V8-V9-FIRST); v7/v8 SHA-bound receipts (64658a05 / 8db93c4e / d68f1d9e / 7051927d) + bounded replays ×2; memory `segnet_recursive_fractal_factorization_20260715.md` (ERF r50≈85px — the mechanism convergence) + the standing SegNet-sees-REGIONS law (CLAUDE.md §unified flow + flip-pixel NO-GO ×3); SPEC_366 header (FALLBACK_ONLY_TRIGGER_NOT_MET — routing correction) + `direct_description_minimizer_PRIMARY_SPEC_20260721T214800Z.md`; canonical_equations registry (structured-carriers law still unregistered, #540 debt — this draft extends its anchor candidates); task ledger #603/#612/#613/#578. NOT consulted this pass: graph-memory recall tool (custody performed against primary artifacts directly).
