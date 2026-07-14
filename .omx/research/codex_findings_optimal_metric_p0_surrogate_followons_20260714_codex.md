# Codex findings — optimal-metric P0 surrogate follow-ons — 2026-07-14

**POINTER STATUS: UNCHANGED.** `[contest-CPU Linux x86_64]` submittable
`0.1910828242`; borrowed defensive bank `0.1880443979880752` is non-submission.
No byte-closed exact row was produced.

## Verdict summary

The canonical metric and its active-provider gates are built and locally
verified. The three requested full-real-n600 numerical verdicts are not
recoverable from current bytes:

| Requested result | Strict status | Concrete custody |
|---|---|---|
| Identity vs categorical-Fisher-natural vs winner-rival-margin-Fisher-natural M selection | `NO_VERDICT_DATA_CUSTODY` | K2 has 600 exact real-R outcomes but no candidate pullbacks/logits/Jacobians; round 2 has 120 exact/candidate pullbacks but no matched finite real-R outcomes. |
| Centered-logit whole-teacher/Jacobian student rho, first-order gain, and charged cost | `NO_VERDICT_DATA_CUSTODY` | strict n600 manifest absent; blocker receipt says `n_pairs=0`, `fit_steps=0`, `teacher_calls=0`; no fitted student exists. |
| Numerical null-removal vs renderer-anisotropy vs decision-reweighting decomposition | `NO_VERDICT_DATA_CUSTODY` | n120 rows lack the same-state renderer Jacobian/range basis; scalar reductions cannot reconstruct it. |
| Categorical finite-Bregman/dual-Euclidean prediction of through-R agreement | `NO_VERDICT_DATA_CUSTODY` | helper is built, but retained rows do not jointly contain candidate/reference centered logits and the exact same-candidate real-R outcome. |

These are data-admission classifications only. They do not constitute an
`INSTANCE`, `FORMULATION`, `FAMILY`, or `PARADIGM` fidelity negative. The
optimal-form student and M family remain intact.

Machine receipt:
`.omx/research/optimal_metric_p0_data_custody_receipt_20260714.json`.

## What landed

### 1. One reusable fidelity law

`src/tac/scorer_surrogate/vjp_fidelity.py` owns:

- NumPy-fp32 renderer pullback and semantic array hashes;
- typed PSD preconditioner receipts;
- identity, categorical-Fisher-natural, and winner-rival-margin-Fisher-natural candidates;
- canonical `rho`, norm ratio, `eta`, relative L2, and matrix-free receipt paths;
- optional MLX-fp32 advisory reduction;
- strict exactly-n600 M selection by exact real-R argmax prediction;
- null-space/anisotropy/reweighting decomposition.

The selector binds canonical preconditioner names, ordered per-state matrices,
ordered state IDs, every rho sequence, and the exact through-R outcome vector.

The choice to invert the pulled-back Fisher is typed, not stylistic: `h` is a
cotangent and an optimizer step is produced by the inverse tangent metric.
Categorical Fisher uses `diag(p)-pp^T`; winner-rival Fisher uses exact two-class
curvature `2 sigmoid(m)(1-sigmoid(m))`. The inherited `0.978` margin/Fisher
correlation motivates the candidate but is not embedded as a scale or promoted
to an n600 result.

The canonical equation delegates its `M=I` reduction to this helper, avoiding
two numerical laws. Its research-only inherited real-n600-source/n120 anchor is
registered under one stable `equation_id=argmax_native_vjp_fidelity_v1`; it
carries no full-n600 M-selection, score, or live-provider authority. The
append-only ledger preserves the pre-grounding event and a later Bregman-
grounded event; latest-event reduction selects the latter.

The metric is now grounded in Bregman geometry without collapsing distinct
objects:

- fixed-state `M` is exactly the Hessian of the local quadratic generator
  `F_M(h)=1/2 h^T M h`;
- categorical log-partition Bregman equals `KL(p_T||p_S)`;
- `||softmax(z_T)-softmax(z_S)||_2` is the cheap exact distance for the
  squared-Hessian geometry `H_F^2`, not finite KL, Fisher-natural cotangent
  geometry, or reachable VJP `rho`;
- sampled KL uses the pointwise-nonnegative extended integrand
  `log(p/q)+q/p-1`, while the naive mean is diagnostic only.

A state-varying `M` is not advertised as one globally integrable finite
Bregman divergence without a separate path/integrability proof.

### 2. Active raw-cosine authority removed

The full map is
`.omx/research/optimal_metric_p0_raw_cosine_audit_20260714.md`.

- Generic scorer-gradient provider decisions now require a canonical reachable
  renderer-gradient receipt with positive `rho` and `eta`, plus their exact
  teacher functional step. Ambient input-costate cosine/L2 is diagnostic only.
- On-policy probing no longer flips an exact CE/through-R decision using raw
  cosine.
- Whole-teacher admission requires exactly-n600 selected-M custody and exact
  functional validation. The law ID, state-receipt schema, selector schema,
  selected-M receipt, n600 fidelity aggregate, and exact-functional receipt are
  bound separately. All three files are rehashed and semantically verified at
  admission; worst-pair `rho` and `eta` must be positive. Backend parity cosine
  remains valid only because it compares NumPy and framework implementations
  of the identical student VJP.
- INSTANT projected-adjoint renderer gradients now go through the canonical
  helper under `M=I`.
- The legacy whole-teacher ambient-fit driver is implementation-blocked from
  claiming the requested optimal form.
- The metric policy now exposes a V9·CGauge DSL lever plus a strict binding
  manifest (`LawRef`, consumer, distinct schemas, content-bound receipt
  paths/hashes, exact-teacher fallback). Drift and unknown fields refuse. The
  exclusive V9 config/provenance-bijection owner composed that argv-inert Lever
  into the 432 and ideal mod19/mod32 expected sets and embedded the returned
  binding wholesale in both the outer manifest and the content-hashed
  provenance bijection. Independent compilation verified exact object equality,
  `activation=false`, `provider_current=false`, all three receipt paths absent,
  and fallback `full_frozen_teacher`; no owner hot file was edited by this arm.

Round-2/round-3 historical cosine fields are preserved for provenance and are
explicitly first-cut instance evidence only.

### 3. Disentanglement design

For one sealed `(g_T,g_S,J_R,M)` state, the helper reports:

```text
ambient g
  -> U^T g                     null-space removal only
  -> J_R^T g under I           renderer singular-value anisotropy
  -> J_R^T g under M           decision/optimizer reweighting
```

This directly answers the earlier confound: the observed 12.5–51x lift cannot
be assigned wholly to null removal. The n600 numerical increments remain
unmeasured because the range basis/Jacobian was not retained.

## Measured and inherited numbers — no laundering

- **MEASURED-INHERITED, real-n600-source heldout n120, first-cut round 2:**
  renderer-pullback `rho=0.0176974146`, same-LR `eta=0.00230361`, relative L2
  `1.00614916`, positive-dot states `60.8333%`. This is not full n600 and not an
  optimal-form student result.
- **MEASURED-INHERITED, advisory n96:** Fisher versus negative margin
  correlation `0.978`. This motivates a candidate M only.
- **UNMEASURED:** selected M, full-n600 optimal-form rho/gain, student VJP cost,
  anchor-update cost, effective cadence cost, 95%-kill economics, exact
  one-step/trajectory parity, and the three disentanglement increments.

No raw `0.0014` RGB cosine is repeated as a fidelity conclusion.

## Round-1 adversarial review

1. **Tangent/cotangent type confusion:** closed by constructing a pulled-back
   tangent Fisher and applying its damped/Moore–Penrose inverse to cotangents.
2. **State-dependent M custody:** closed by hashing the ordered per-state matrix
   set, not pretending one Fisher matrix was global.
3. **Wrong-costate admission after removing cosine:** caught by the wider test
   pass; closed by requiring canonical reachable positive `rho`/`eta` in
   addition to the exact functional receipt.
4. **Receipt schema masquerading as metric ID:** closed and broadcast;
   consumers now persist law ID and schemas separately.
5. **Naive-driver negative:** closed by emitting `NO_VERDICT_DATA_CUSTODY` and
   blocking the ambient-fit driver from optimal-form execution.
6. **Sub-n600 promotion:** selector refuses every count other than exactly 600.
7. **Degenerate selector rows:** zero-variance M predictions remain visible as
   uninformative and cannot win; zero-variance real-R outcomes refuse the
   selection; exact Spearman+Pearson ties emit no selected M.
8. **Dormant raw-threshold authority:** periodic provider configs may now omit
   the four legacy ambient cosine/L2/norm threshold fields. When present for
   parse-back they remain diagnostic and cannot admit, reject, or rescope a
   provider; malformed/empty ambient diagnostics likewise cannot reject valid
   canonical reachable evidence.
9. **Preconditioner alias/name laundering:** selector mapping keys must equal
   the content-described preconditioner name, and the exact through-R outcome
   vector is now semantically hashed in the selection receipt.
10. **Invented student receipt hashes:** closed by decision-time rehash and JSON
    semantic validation of selection, n600 aggregate, and exact-functional
    receipts, including cross-receipt SHA and worst-pair value equality.
11. **V9 rebinding changed an A/B arm:** the first owner integration rebound the
    ideal mod32 arm to mod19. Read-only round-1 testing caught it; provenance is
    now rebound after each ideal-arm update, preserving the registered mod19/32
    A/B.
12. **V9 provenance erased specific LawRef/legacy custody fields:** the same pass
    caught the T1 start-law overwrite and missing hosc replacement keys. The
    owner restored the flicker-floor LawRef, literal `N7 BUILD-OWED`, and the
    live-argv/replaced-value compatibility fields without weakening the new
    bijection.
13. **Cheap dual metric could launder a different geometry:** closed by typing
    it as squared-Hessian `H_F^2`, reserving `rho` for reachable VJP alignment,
    and explicitly retaining the Fisher-natural cotangent solve.
14. **Naive MC KL can be negative:** closed by the pointwise-nonnegative
    extended f-divergence estimator with iid-from-reference custody and
    overflow/nonfinite refusal.
15. **V9 gauge covariance could remain a paper-only LawRef:** read-only audit
    found declarative `cgauge_master_action_v1` references, but no explicit
    runtime affine-Legendre gauge-transform pair, invariance test, or
    content-bound covariance receipt in the trainer/launcher/DSL consumers.
    This is an implementation-custody gap, not a CGauge formulation or family
    negative; it was routed to the exclusive provenance owner without editing
    its hot files.

Final local seal: Ruff passed, followed by three clean focused passes of
`188 passed`; the final machine custody receipt SHA-256 is
`d9ac102f19942d439f6fe8308b2702e8b0e0547f5277dfd9f2b9259ebae811e1`.
The post-integration V9 group passed three clean `82 passed` groups; independent strict
compiles of `v9_cgauge_432`, `v9_cgauge_ideal_mod19`, and
`v9_cgauge_ideal_mod32` all verified the exact metric binding and frozen-teacher
fallback in the hashed bijection.

## Ranked next actions

1. **Highest EV — custody-complete no-training n600 replay.** From preserved,
   hashed stage checkpoints, retain exact/candidate renderer pullbacks,
   centered logits/probabilities, winner/rival margins and Jacobians,
   equal-trust finite candidate steps, exact real-R before/after argmax changes,
   Pose deltas, and content hashes. Storage preflight, stage checkpoints, and
   auto-clean certification are mandatory. This produces the first lawful M
   selection and disentanglement receipt.
2. **Fit the optimal-form centered-logit value+Jacobian student.** Only after
   action 1 supplies real n600 geometry: use the selected M, preserve
   value-only/Jacobian/functional/on-policy-refresh stages, NumPy authority,
   MLX parity, EMA, optimizer state, and exact fallback. Measure rho, eta,
   worst pair/class, and fully charged cost.
3. **Bregman/trust-region functional gate.** At matched trust radius, compare
   exact categorical Bregman/KL, the separately typed dual-Euclidean predictor,
   winner-rival crossing behavior, exact CE, d_seg/d_pose, and short-trajectory
   regret. Any sampled KL uses the extended non-negative form. Static rho never
   licenses replacement alone.
4. **Per-class/per-head M.** Preregister shared-M versus class/head-block M on a
   design split; select only if untouched real-R validation improves without
   Pose harm or worst-class regression.
5. **Compose with #455/#465 economics.** Use measured student VJP/update/anchor
   costs and the existing `C_student + (C_teacher+C_update)/K` law. A 95%-kill
   claim requires the full charged ratio, not an isolated kernel time.
6. **On-policy importance correction.** Remains
   `BLOCKED_DISTRIBUTION_CUSTODY` until sealed live/cache density ratios,
   support masks, and effective sample size exist.

## Triality and pointer delta

- **DSL:** generic scorer-gradient and whole-teacher admission surfaces consume
  canonical metric custody and fail closed; sibling #430 DSL consumes the same
  distinct metric/schema identifiers; the V9 binding API supplies the sole DSL
  lever/manifest/consumer contract and exact-teacher fallback, and the exclusive
  owner has composed and verified it in all three compiled V9 vehicle programs.
- **DAG:**
  `.omx/research/optimal_metric_p0_surrogate_followons_DAG_FEED_20260714.md`.
- **Equation:** one stable canonical ID,
  `argmax_native_vjp_fidelity_v1`, delegates to the helper and retains only the
  inherited n120 anchor. The append-only history has its initial event plus the
  latest Bregman-grounded event; latest-event reduction carries the local
  quadratic and categorical Bregman laws plus the separately typed dual metric.
  No new empirical anchor is added because no new n600 measurement exists.
- **Pointer delta:** exactly zero.

## Inbox coordination consumed

Consumed mission/broadcast directives through `2026-07-14T14:02:13Z`: the
training-loss sibling integrated the clarified metric/schema contract and
retained n600 custody blocking; #497/#502/#503 consume the same single metric;
the exclusive provenance owner consumed the binding/round-1 corrections through
`2026-07-14T13:58:53Z`. No duplicate helper, student, curriculum, or basis
implementation was created. Serializer handoffs from other lanes were not
absorbed.

The 14:02 Bregman directive was applied to the metric helper, canonical
equation, V9 binding contract, custody probe, tests, DAG, and audit. Its broad
wording that the dual form removes a Fisher solve is scoped to squared-Hessian
geometry; applying it to the Fisher-natural cotangent metric would change the
metric and was therefore refused.

The new all-surfaces Bregman lane owns adjacent new modules and explicitly will
not edit the canonical helper. It consumed the API/type correction. The genuine
basis lane reported no measured Hessian/Gram custody, so no basis ranking was
promoted into this metric receipt.

The exact owned-file serializer reached `git add` and failed closed with
`rc=128` (`unable to create temporary file: Operation not permitted`); it staged
nothing and produced no commit SHA. Exact file hashes are therefore part of the
final handoff for privileged harvest.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`; v7.5/v8 specs; `reports/latest.md`;
lane/subagent/task/gradient/modal/posterior state; latest sister
findings/session/design/council/directive memos; round-2/round-3 receipts and
cleanup manifests; K2 n600 rows; whole-teacher blocker receipt and source;
canonical GT-cache custody; live inbox at every checkpoint.
