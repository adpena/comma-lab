---
title: "P0 backward closer: exact-costate reuse, sparse-adjoint closure, and terminal skip"
date_utc: "2026-07-13"
lane_id: "lane_p0_backward_closer_20260713"
research_only: true
score_claim: false
pointer_moved: false
training_performed: false
paid_dispatch: false
live_run_mutated: false
---

# P0 backward closer — build, n600 measurement, and composed accounting

## Outcome first

> **K2_N600_RESULT_PENDING.** The sealed real-state replay is still running; every final number in
> this section is intentionally withheld until its three stage manifests, 600 unique pair records,
> aggregate, and trusted receipt hash close. No partial prefix is evidence.

The other two survivor verdicts are already closed:

1. **Terminal exact-metric #396: LANDED, formulation-scoped.** The existing `(1+1)-ES` control
   evaluates the exact contest objective composed over all 600 stored Seg/Pose cells, makes no
   gradient or costate call, and measured `Delta S=-8.229284904293088e-6` in 64 exact function
   evaluations. The six changed cells were freshly checked in the canonical 16-pair frozen scorer
   chunk; the unchanged 594 cells retain the STOP-sealed n600 authority fixture. This is an n600-
   composed terminal result, **not** a fresh full-600 scorer replay and not a training-loop speedup.
2. **Sparse / low-rank adjoint: `NO_GO_DENSE_FULLRANK` for the measured formulation.** All 600
   exact input costates are numerically dense. At 4.7366% oracle output-mask area, input-costate
   relative L2 error is `0.363536`; raw `r95=68/120`, `r99=100/120`, and rank-64 error is
   `0.238192`. The ideal unavailable custom spatial-kernel ceiling is `2.208577x`; exact and
   current dense-kernel realized arithmetic saving are **`1.0x`**. A subsequent sibling landing
   built a default-off compact Metal Conv2d input-VJP primitive and measured `40/40` bit-identical
   NumPy-fp32 CPU parity trials, but Metal parity and wall time are unavailable in this execution
   custody and the accuracy provider is still missing. The primitive is LANDED; sparse-adjoint
   admission and measured wall saving remain NO-GO/BLOCKED. No ideal factor is multiplied into the
   composed stack.

## Ranking by measured net win

| rank | survivor | evidence | admitted effect | stack factor |
|---:|---|---|---|---:|
| 1 | guarded event-controlled K=2 exact-costate reuse | `PENDING_SEALED_N600` | pending all-600 gate; live trainer activation remains refused | `PENDING` |
| 2 | #396 exact-metric terminal route | MEASURED n600-composed `Delta S=-8.229284904293088e-6`; 64 calls; 56.8432 s local advisory | 100% costate skip **inside this terminal route only** | not composable with bulk training fraction |
| 3 | masked / fixed-low-rank SegNet adjoint + compact-kernel child | MEASURED n600 density plus heldout-120 VJP/spectrum; MEASURED CPU parity 40/40 | primitive LANDED default-off; accuracy NO-GO, Metal wall BLOCKED; retain full dense VJP | measured current realized `1.0x` |

## K=2 law and admission contract

Let `p=A/n` be the accepted guarded reuse fraction over **all** `n=600` assigned states, including
terminal/ineligible/fallback rows in the denominator, and let `f=0.1784755863` be the prior
diagnostic exact-teacher forward fraction. Over two opportunities:

```text
baseline exact backward calls       = 2n
guarded exact backward calls        = 2n - A
exact-call amortization             = 2/(2-p)
exact-backward-call reduction       = p/2
diagnostic teacher-slice speedup    = 2/[2-p(1-f)]
forward-elimination Amdahl ceiling  = 1/(1-f) = 1.2172492787x
reuse admission rate gate           = p > 2f/(1-f) = 0.4344985574
```

These are **DERIVED diagnostic accounting**, not in-loop wall clock. Even `p=1` gives at most
`1.697107707x` for this charged forward-guard teacher slice, not a literal 2x whole-epoch win.

Every attempted reuse is event-controlled, never fixed cadence:

```text
exact anchor + durable costate payload + full-facet baseline
  -> one changed-frame stale reuse attempt in the same control/event scope
  -> exact forward-only CE, d_seg, d_pose guard
       accept iff CE strictly descends and d_seg/d_pose do not worsen
       otherwise byte-exact rollback + full exact teacher refresh
  -> exact anchor required again (K_max=2)
```

Stage, event, scorer/objective, frame, payload, or control-scope drift forces refresh. The retained
payload is re-opened and SHA-verified on construction, checkpoint restore, exact-anchor record, and
immediately before reuse. The offline fidelity gate additionally requires every behaviorally
accepted row to have renderer-gradient relative L2 `<1` and stale-minus-exact matched-step
`d_seg<=0`. The latter two are accepted-row fidelity certificates; they are not claims about rows
charged to fallback.

## K=2 n600 measured result

`PENDING_SEALED_N600_RECEIPT_AND_NUMBERS`

Authority boundary: the cached replay measures fidelity and exact-call accounting on immutable real
n600 states. It does **not** mutate a trainer, perform training, measure whole-epoch wall clock, or
move a contest score. Live trainer activation remains fail-closed because the hot trainer has no
reviewed current-costate provider/checkpoint seam. The explicit operator-GO packet names that owed
wire-in and paired timing treatment.

## Sparse / BCR-style cheap adjoint — measured closure

The flagship receipt regenerated and hash-matched `600/600` task455 costates. The decisive measured
facts are:

- exact-zero output and input spatial support: `0/600` states;
- 4.7366% oracle-mask input-costate error: relative L2 `0.363536`, cosine `0.954554`;
- deployable source-margin mask error: relative L2 `0.793434`, cosine `0.775455`;
- frozen scorer support obstruction: exact local halo `685` pixels plus `23` global squeeze-excite
  reductions;
- raw cohort spectrum: `r95=68`, `r99=100` of 120; rank-64 relative Frobenius error `0.238192`;
- ideal custom spatial arithmetic ceiling `2.208577x`, current dense arithmetic `1.0x`.

Thus the measured BCR/ICM analogy does not supply the missing operator block: one exact discrete
SegNet VJP is already linear in pixels, and no hierarchical low-rank off-diagonal representation of
the **state-dependent frozen CNN Jacobian transpose** was demonstrated. `verdict_scope` is the
source-bound task455 frozen EfficientNet-B2 U-Net, 4.7366% output masks, and fixed linear cohort
bases. It does not kill a pre-SE/local scorer, a learned current-witness larger mask with custom
kernels, nonlinear manifolds, or a replacement provider.

The post-flagship `custom_sparse_adjoint_kernel` child closes the implementation question without
manufacturing admission. Its compact grouped/depthwise/stride/rank Conv2d input-VJP authority is
**MEASURED bit-identical in 40/40 deterministic NumPy-fp32 CPU trials**. Its 125 real frozen-SegNet
convolution replay re-derives **DERIVED** nominal and exact-valid-tap ceilings of `2.208615x` and
`2.200230x`, respectively. But the current sandbox exposes no Metal device, so N=10 cross-process
Metal parity, achieved wall factor, and achieved/ceiling efficiency are **UNMEASURED/BLOCKED**.
Moreover the useful 20% post-hoc oracle mask has rel-L2 `0.026206`, while the available source-margin
mask has `0.540174`; their `0.513967` predictor gap remains open. The code primitive is therefore
default-off and supplies no multiplicative measured saving to this stack.

## Terminal gradient-free costate skip

The admitted child is not a bulk SPSA/ES gradient approximation. It is the existing #396
post-training exact-metric accept/reject route:

- **MEASURED:** all 600 stored base cells enter every score composition; six pair-local cells vary;
  `(1+1)-ES` reaches the exact six-bit optimum at 64 calls, `Delta S=-8.229284904293088e-6`;
- **DERIVED:** the terminal route makes zero costate calls, hence skips 100% of costate calls *inside
  that route*;
- **NOT COMPOSABLE:** no baseline count of training-loop teacher calls or terminal-band occupancy was
  measured, so its bulk-training teacher reduction remains `UNQUANTIFIED_NOT_COMPOSABLE`;
- **SCOPED NO-GO:** true SPSA/ES gradient estimation is admitted only with deterministic effective
  dimension `<=2`; measured search support is six and prior tensor ranks are far larger. No such
  certificate exists.

The handoff is a fixed-SHA, recursively verified receipt plus a default-off argv-inert DSL policy.
It revalidates committed compact source/fixture snapshots and the tracked harness sources before
returning the #396 costate-skip action. Those snapshots bind the original ignored receipt, STOP-
sealed fixture, candidate archive, scorer, and GT-cache hashes; the original local bytes remain
restorable/auditable by SHA but are not required for a fresh-clone policy import. Missing/tampered
committed custody returns the ordinary full-teacher/#396 route.

## Composed cheap-backward stack

Only admitted, quantified factors may compose:

```text
guarded K2 exact-call factor                 = PENDING_N600
times realized sparse/low-rank factor        = 1.0x
plus terminal route-local costate skip       = not multiplicative; bulk fraction unmeasured
-------------------------------------------------------------------------------
quantified bulk teacher-cost reduction       = PENDING_N600_K2_ONLY
whole-epoch wall-clock speedup                = UNKNOWN; operator-GO timer owed
```

The diagnostic `82.15%` backward share remains a premise to test, not current in-loop authority.
The sparse replay itself observed a different diagnostic ordering under its own substrate. Neither
diagnostic ratio can be promoted to the live loop. The composed result will therefore report exact
call reduction and conditional teacher-slice speed only; it will not manufacture whole-epoch time.

## Triality and system intelligence

- **DSL:** `exact_costate_reuse_k2_guarded` and `terminal_exact_metric_costate_skip_396` are named
  factories under `src/tac/witness_dsl/`; both compile empty trainer argv until a reviewed provider
  seam exists. Exact-costate controller state and payload custody are additive/resumable.
- **Equation:** `exact_costate_reuse_k2_guarded_v1` records guarded cost, exact-call, full-facet, and
  terminal-skip laws. Sparse closure remains in
  `sparse_adjoint_mask_error_and_se_support_closure_v1`.
- **DAG:** `FEED-p0-backward-closer-20260713` records admission, scoped negatives, fallbacks, and the
  operator-GO edge; shared hot DAG append is deferred to main review.
- **Sensitivity / Pareto:** K2 records fidelity regret against exact call reduction; sparse records
  approximation error against ideal FLOPs; terminal records exact objective progress per call.
- **Bit allocator:** no direct archive-byte actuator is introduced. #396 retains its existing exact
  pack/re-score accept gate.
- **Autopilot:** consume only a trusted admitted receipt. Fallback is full teacher; no autonomous
  heavy dispatch is authorized.
- **Continual learning:** receipts, equations, tests, standalone DAG FEED, this memo, and the GO
  packet are the durable posterior update.

## Literature boundary

- Lazy/stale gradient literature motivates guarded communication/computation skipping, but does not
  admit this state-dependent witness gradient: <https://arxiv.org/abs/1805.09965>.
- BCR-Net supplies the multiscale low-rank integral-operator analogy; it is not evidence that the
  frozen SegNet adjoint has the required block structure: <https://arxiv.org/abs/1810.08754>.
- SPSA motivates dimension-robust zeroth-order estimation, but the exact #396 accept/reject route and
  a training-gradient estimator remain different mechanisms:
  <https://www.jhuapl.edu/spsa/PDF-SPSA/Spall_TAC00.pdf>.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`; v7.5 §8 and v8 specs;
`reports/latest.md`; canonical lane/subagent/task/gradient/modal/cost-band/probe/equation stores;
latest sister findings/session/design/council memos and recent operator directives;
`.omx/research/p0_costate_reuse_gradfree_20260713.md`;
`.omx/research/p0_sparse_adjoint_costate_vjp_20260713.md` and its receipt/DAG/equation;
`.omx/research/custom_sparse_adjoint_kernel_20260713.md`, static receipt, DAG, and equation;
`.omx/research/invprob_operator_fold_20260713.md`;
`.omx/research/ugc_terminal_polish_ab_20260712.md` and its sealed local receipts;
the immutable V9/task455 checkpoints, frozen scorer weights, GT cache, and cached n600 state
assignments. No live trainer/run, provider, GPU, paid dispatch, evaluator, submission, or frontier
pointer was actuated.

---

## 2026-07-14 CORRECTION / SUPERSESSION — corrected n600 K2 closure

Everything above this divider is retained byte-for-byte as **HISTORICAL_PROVENANCE**. In
particular, its `K2_N600_RESULT_PENDING`, placeholder economics, and deferred shared-DAG language
describe the state on 2026-07-13; they are not current claims. This dated block supersedes those
claims without erasing them. The shared DAG feed is **NOW LANDED** under the exact heading
`FEED-p0-backward-closer-20260713`.

**Actual verdict:** `K2_CORRECTED_NOT_ADMITTED_DEFAULT_OFF` (`NOT_ADMITTED`, default-off,
pointer unchanged). **verdict_scope=FORMULATION:** this negative applies only to the bounded direct
raw-input-costate zero-order-hold K2 formulation (`exact_costate_reuse_k2_guarded_v1`), evaluated
under the strict rule that every accepted row must have stale-minus-exact `d_seg` regret `<=0`, on
three V9 checkpoint states (`v9_ep150_ema_best`, `v9_ep251_stage_octave1`,
`v9_ep275_ema_final`; 200 rows each) and the sealed
`[macOS-CPU advisory; Torch/NumPy-fp32 training-gradient MEANS only]` axis. The costate-reuse family
remains intact for a newly preregistered guard/tolerance formulation; no sibling family is killed.

The corrected wrapper is
`experiments/results/p0_costate_reuse_k2_n600_v3_20260713/corrected_adjudication_receipt.json`
(397028 bytes; file SHA-256
`2102912bc8bd9711f00869746414fb21ea723729bcd26e612274547c6ca73d59`; adjudication-content
SHA-256 `4f7f2a6ef95f9989b734cd6e785d8b55dca7a77d7d9f03c693ff18287dea6e6e`). The
tracked compact custody summary is
`.omx/research/p0_costate_reuse_k2_corrected_adjudication_receipt_20260714.json`. A clean clone does
**not** contain the full wrapper or its 600 source rows; revalidation requires restoration of the
hash-matching 606-file sealed source tree from certified cold storage or another certified source.
No cold-store destination is recorded by the wrapper, so that restoration location remains an
explicit custody gap.

**MEASURED n600:** 523 rows were eligible, 456 were behavioral full-facet accepts (`p=0.76`), 67
were actual guard fallbacks, and 77 were terminal/blocked. Thus total nonaccept is `144=67+77`
(`q=0.24`); `144` must not be reported as 144 actual guard fallbacks. Accepted renderer-gradient
relative-L2 was `456/456 <1` (median `0.03072912052372636`, p90 `0.0518675255971356`, max
`0.1432164947042975`). Accepted stale-minus-exact `d_seg<=0` held for only `308/456` (median `0`,
p90 `0.000020345052083335646`, max `0.000091552734375`). The corrected gate therefore has
`passed=false` solely because `all_accepted_stale_d_seg_regret_lte_exact=false`; the earlier verdict
status is `SUPERSEDED_INVALID_FALLBACK_CHARGE`.

**DERIVED diagnostic economics, behind the failed gate:** with baseline `2.0`,
`alpha=0.1784755863`, and `p=0.76`, guarded expected cost is
`2+alpha-p=1.4184755862999998` and diagnostic teacher-slice speedup is
`2/(2+alpha-p)=1.4099643443401577x`. The exact-backward factor
`2/(2-p)=1.6129032258064517x` and reduction `p/2=0.38` are a counterfactual for this rejected
formulation. They are never multiplied into admitted composition. Admitted composition is K2
`1.0x` times sparse realized `1.0x`, for **0% admitted bulk teacher-cost reduction**. Terminal #396
remains a local exact-metric win only and is nonmultiplicative. Whole-epoch wall-clock effect is
`UNKNOWN`.

**Latest canonical timing caveat:** commit
`e59f69a79cb2d974ec29fcaf75c6c855bd782a7a` and
`.omx/research/frozen_scorer_verdict_wallclock_n96_20260714.json` report a **MEASURED** n96
`0.621 s/pair` on `[macOS-CPU-torch 1-thread advisory wall-clock] NON-PROMOTABLE`, with measured
SegNet/PoseNet forward shares `0.774/0.226`. `6.21 min/n600` (`372.6 s`) is a **DERIVED linear
extrapolation**, not an n600 measurement. This forward-only authority-verdict result supersedes use
of the old backward-share diagnostic as a global 95%-kill claim. Zero K2 benefit is admitted and
the whole-epoch effect remains unknown.

The in-loop timer packet is now classified `FIDELITY_BLOCKED_FUTURE_TEMPLATE`: it does not solicit
current GO and cannot activate this formulation. A diagnostic timer may run only after a newly
preregistered formulation/provider is fidelity-admitted.
