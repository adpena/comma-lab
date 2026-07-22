---
schema: canonical_equation_candidate_note.v1
task: 603
feeds_task: 613
lane_id: ddm_v7_solved_plane_tolerance_waterfill
research_only: true
registry_promotion: false
main_landing_review_required: true
---

# DRAFT candidate law — solved-plane tolerance ladder as evaluator

This note is a registry draft. MAIN decides whether to register it after reviewing the code and
SHA-bound receipts; this branch does not mutate the canonical equations registry.

Let `T_t in uint8^(2x384x512x3)` be solved-plane pair `t`, `P_t` the bound receiver output from the
v6 fixed-hold predictor, and `{M_s(t)}` the six disjoint supports obtained by self-detecting the five
canonical roles and applying Boundary first. For stratum `s` and rung `r`, define

```text
C_{s,exact}(t) = M_s(t) [T_t - P_t]
C_{s,q}(t)     = M_s(t) Q_q(T_t - P_t),            q in {4,16,64}
C_{s,hold}(t)  = C_{s,exact}(k(t)),                k=fixed24 or xi-keyed, gap <=24
C_{s,drop}(t)  = 0.

W_{pi,t} = P_t + sum_s C_{s,pi_s}(t),
A_pi     = ReceiverClosedZIP(P, {EntropyCode(C_{s,pi_s})}_s),
B_pi     = len(A_pi).
```

`Q_q` is the implemented uint8 residual quantizer. Each section coder is the shorter exact-parseback
result of Brotli-Q11 and LZMA-XZ preset-9-extreme. No scorer weights, stratum labels, or GT argmax
table are members of `A_pi`.

The evaluator is realized through the actual receiver and frozen preprocessing:

```text
D_seg(pi)  = mean_{t,y,x} 1[argmax SegNet(R(W_{pi,t})) != lstar_{t,y,x}],
D_pose(pi) = mean_{t,j<6} (PoseNet(YUV6(W_{pi,t}))_j - gt_pose_{t,j})^2,
D(pi)      = 100 D_seg(pi) + sqrt(10 D_pose(pi)),
S(pi)      = D(pi) + (25/37,545,489) B_pi.
```

The discrete waterfill is not a handwritten policy order. Sort measured states by increasing `B`,
retain only strict record improvements in `D`, then for adjacent Pareto states `i -> j` compute

```text
g_{i->j} = [D(i)-D(j)] / [B_j-B_i],    B_j>B_i and D(j)<D(i).
```

Stop before the first `g < 25/37,545,489`. The constrained evaluator knee is

```text
pi* = argmin_pi B_pi  subject to D_seg(pi) <= 0.00116.
```

For the measured opaque site/value family, `pi*=exact_all` at both windows, while
`B_pi* in {43,112,153; 171,332,654}`. Therefore

```text
1[B_exact <= 200,000] = 0
```

is a FORMULATION-level falsifier only. It does not imply that the Kolmogorov complexity of `T` under
a learned or analytic receiver exceeds 200 KB. The successor law must replace opaque site/value
payloads with structured carriers and retain the same evaluator, archive, and waterfill definitions.

Receipt anchor:
`64658a05a8975707f98db308223cefff78b5352975bb59cc2aa8a4ff2f8d50fb`.

STORES CONSULTED: DAG `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md` (FEED-603-coder-survey / v4 / v5 / v6 rows — the measured grounds this ladder consumes); v5/v6 SHA-bound receipts (ab5332f2/3b2ea4c9 + the v6 bridge receipt) + the v7 window receipts (8db93c4e/d68f1d9e/64658a05); target-plane receipt a8d94f0f (600-pair solved planes); canonical_equations registry via `tac.canonical_equations` (structured-carriers law NOT yet registered — this draft is its 4th-anchor candidate; #540 debt); SPEC_v75/SPEC_v8 + `direct_description_minimizer_PRIMARY_SPEC_20260721T214800Z.md`; MEMORY.md CURRENT-STATE hooks (L17 trained-generator doctrine, opportunity-pools non-additive law); task ledger #603/#613/#578. NOT consulted: graph-memory recall tool (custody performed against primary artifacts directly).

CONSUMED-BY: `ddm_describe_line_rate_distortion_bracket_v1` v7 exact-value anchor; registration landing `.omx/research/ddm_structured_carriers_law_registration_20260722T142000Z.md`; MAIN review required.
