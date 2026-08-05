# AR8_RECEIPT - arXiv 2602.15293 softmax information geometry

## Verdict

AR8 is an **ADOPT-AS-FORMULATION** paper, not a launch trigger. Its main theorem is
a clean external derivation of the same object our live metric line already uses:
softmax-head Bregman/Fisher geometry, with the natural steering direction obtained
by solving a local covariance/Fisher system instead of adding a Euclidean probe
vector. The new value is not a new score row; it is a stricter off-target
preservation formulation for collateral guards and seg-hold-while-pose-descends.

No scorer job was run. No archive was built. Pointer unmoved.

## Ranked Crosswalk

| rank | disposition | claim | named consumer | falsifier | cost / follow-on disposition |
|---:|---|---|---|---|---|
| 1 | **ADOPT** | Dual steering gives the missing theorem shape for "change target, minimize off-target KL": move in dual coordinates, equivalently solve `Cov[gamma | lambda] delta_lambda = beta` with damping. For our frozen SegNet head this is the same categorical Fisher/Bregman object as margin-Fisher, but only after pulling it through the actual receiver/scorer Jacobian. | `p0_fisher_full_leverage_20260717`, `ddm_ms4d_direct_metric_completion`, and the metric-active solve line (`ms2/ms3/ms4d`) | On a cached metric-custody point, the Fisher-natural/covariance solve disagrees in sign with the existing margin-Fisher winner/runner direction after the same pullback, or the first-order predicted off-target KL is worse than Euclidean under identical target gain. | `$0`, scorer-free. **QUEUED-WITH-FIRE-ORDER AR8-P1:** on next metric-ledger touch, compute dual-steering direction vs current margin-Fisher direction on the SHA-bound MS4D/PF2 rows; no scorer slot. |
| 2 | **ALREADY-EMBODIED** | The paper's Bregman geometry is already represented locally: `g = diag(p) - p p^T` is the categorical Fisher/Hessian of logsumexp; raw dual-Euclidean is not the Fisher-natural metric. | #500/#504 optimal-metric line, `src/tac/canonical_equations/optimal_metric_unification_20260714.py`, `src/tac/canonical_equations/bregman_v9_surfaces_20260714.py` | A direct source or test shows local code using raw `||delta_eta||` as verdict-bearing Fisher-natural distance. | No new equation. **FOLDED** into existing #500/#504/#550/#552 metric doctrine. |
| 3 | **ADOPT** | Collateral accounting should be stated as an off-target KL/protected-distribution constraint before it is stated as a Euclidean or raw-flip constraint. This does not replace v19 realized collateral pricing; it supplies the predicted trust-region geometry that v19 then falsifies or prices. | v15 zero-collateral family, v19 realized-collateral ledger, #920 Lane x ANNIHILATE / Road-Lane x PHASE protection primitives | Same target flip count and same counted bytes, but dual/off-target-KL projection produces equal or worse realized harmful off-target flips than the current constraint on a stratified `n>=32` replay. | `$0` first-order, then existing realized replay if positive. **QUEUED-WITH-FIRE-ORDER AR8-P2:** add an off-target-KL column to the next v19/collateral replay receipt, not a standalone scorer run. |
| 4 | **ADOPT-PRECONDITIONED** | For `jd1`, the transfer is product-metric, not literal paper reuse: PoseNet target movement is a low-rank continuous quadratic, SegNet hold is softmax Fisher/Bregman. Dual steering supplies the "hold off-target while moving target" proof shape for the Seg side of seg-hold-while-pose-descends. | `jd1` recursive joint pose-finish, #383 conditioning gate, terminal pose-finish fire gates | Product-metric step at matched predicted pose gain does not reduce Seg KL/protected-cell violations versus the current #383 seg-hold gate, or it loses net S after receiver parse-back. | `$0` on existing pose quadratic + seg metric rows before any launch. **QUEUED-WITH-FIRE-ORDER AR8-P3:** when jd1 next touches its gate receipt, add a product-metric projection row using existing MS4D pose and Seg custody. |
| 5 | **ALREADY-EMBODIED / LOWER PRIORITY** | The paper's probing setup assumes a learned linear probe. Our Seg head is frozen and known; the frozen-head margin field and rank-4 prototype bank are stronger than a fitted probe when they are available. | #583 prototype-bank users, PF2/MS4D metric custody, hard-tail ranking | Exact frozen-head prototype/margin custody absent for a surface; then dual mean-difference probe may be a fallback, explicitly labelled inferred. | **FOLDED**: no replacement of exact head surfaces. |
| 6 | **N-A DIRECT CODE** | The GitHub implementation is useful as an algorithm sketch, not vendorable code for Pact. It targets LLM/CLIP embeddings and top-k vocab/image softmaxes, with no receiver pullback, uint8/R parse-back, or contest axis. | none | A local adapter proves byte-closed receiver-pullback parity and strict metric-custody ingestion. | **FOLDED** as reading only. If coded later, implement natively against our K=5/rank-4 head and MS4D schemas. |

## Paper And Code Facts

- `arxiv.org/abs/2602.15293` is reachable. It reports v1 submitted 2026-02-17 and v2 revised
  2026-05-29, with the GitHub code link and ICML 2026 proceedings reference.
- The paper defines softmax distributions from representations `lambda` and output vectors
  `gamma_y`, with `A(lambda)=log sum_y exp(lambda^T gamma_y)`.
- It identifies the induced KL as the Bregman divergence of `A`, and the dual map as
  `phi(lambda)=grad A(lambda)=E[gamma | lambda]`.
- Euclidean steering is `lambda_t = lambda_0 + t beta_W`; the paper calls this a type
  mismatch because `beta_W` is dual-space data.
- Dual steering is `phi(lambda_t)=phi(lambda_0)+t beta_W`.
- The theorem's useful content for us: among points on the probe hyperplane, the dual
  steering path minimizes KL change, and under the paper's concept-factorization assumption
  it preserves the off-target distribution.
- Practical method: solve the local softmax covariance system with damping/regularization.
  The current GitHub `information_geometry/steering/method.py` exposes `e_steering` and
  `m_steering`; `m_steering` builds logits, probabilities, mean vectors, and a CG solve of the
  regularized covariance action before stepping in primal coordinates.

## Transfer Validity

The theorem is directly about representations feeding a softmax head. Pact does **not**
directly edit the SegNet head state. Pact edits receiver/archive parameters or pixels, then
the fixed receiver, resize, uint8, SegNet backbone, and softmax head map those edits to logits.

Therefore every AR8 adoption is preconditioned on a pullback:

`delta_archive -> delta_receiver_rgb -> delta_prehead_lambda -> softmax Fisher/Bregman metric`

Without that pullback, "dual steering" would be the same name-preserving fake as the old
raw-dual/no-solve shortcut. With the pullback and MS4D/PF2 custody, it is a derivation of the
metric-active direction we already require.

## Recall Evidence

Sources searched, with exact query surfaces:

- Requested run charter and common contract under the repo-local codex_runs directory.
- Governing files: `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`,
  `.omx/state/main_hot_state.md`.
- Memory registry query: `AR8|2602.15293|softmax|dual steering|dual-metric|Fisher|#583|m65|#504|Bregman|v15|v19|lg1|jd1|#920`.
- Corpus queries over `.omx/research`, `.omx/state`, docs, code, reports:
  `#583`, `Fisher-margin`, `corrected-inner-Jacobian`, `dual-metric`,
  `Euclid-vs-Fisher`, `Bregman`, `Nielsen`, `zero-collateral`, `priced-collateral`,
  `lane-guard`, `ANNIHILATE`, `PHASE protection`, `joint pose-finish`, `linear probe`,
  `prototype bank`, `natural gradient`.
- Canonical equation pass: `tools/list_canonical_equations.py --json`, then filtered by
  Fisher/Bregman/softmax/natural-gradient surfaces.

What changed beyond the charter seeds:

- Found the local #500/#504 result is already sharper than the paper's headline: raw dual
  Euclidean is a squared-Hessian metric, not Fisher-natural. That made direct code vendoring
  unsafe and turned the result into a formulation/crosswalk adoption, not a new equation.
- Found `p0_fisher_full_leverage_20260717` already lists six live Fisher surfaces:
  Fisher-density training, rank-4 head natural-gradient, dual-metric telemetry, bit allocation,
  Fisher-mass triggers, and EMA/selection. AR8 should feed those surfaces, not fork a parallel
  "softmax steering" line.
- Found #550 Nielsen and #552 SPD-submanifold receipts already resolved the two-chart/one-geometry
  and rank-4 natural-gradient questions. AR8 confirms the same direction rather than replacing it.
- Found the #583/fr1 caution: the rank-1 Fisher candidate was only a first-order VJP candidate, not
  an executable corrected-inner-Jacobian actuator. This is the exact failure class AR8 could repeat
  if we forget the receiver pullback.
- Found p4x/lg1 evidence that Lane annihilation is component-level, not per-pixel margin alone.
  Therefore dual steering can inform Lane protection only after the protected distribution is
  defined over the right component/support units.

Scoped negative: I did not find an existing AR8-specific receipt in memory or `.omx/research` under
`AR8`, `2602.15293`, or `dual steering` in the searched scopes.

## NEXT_IF_RESUMED

1. **AR8-P1, scorer-free:** compute the K=5/rank-4 dual-steering direction from existing MS4D/PF2
   metric custody and compare it to the current margin-Fisher direction. Accept only if metric ID,
   pullback, and row hashes are explicit.
2. **AR8-P2, scorer-free first:** add an off-target-KL/preserved-distribution column to the next
   v19 collateral receipt. Realized replay remains the authority; the AR8 column is only the
   predictor/trust-region leg.
3. **AR8-P3, jd1 gate:** add a product metric row: target PoseNet quadratic descent, protected
   SegNet Fisher/Bregman hold. Do not launch from this receipt alone.

## Boundaries

- Full text was reachable; this is not an abstract-only crosswalk.
- Code repo was reachable; no code was imported or run.
- No scorer slot was claimed.
- No n600 scorer work, archive build, CUDA/CPU contest eval, or public-pointer update happened.
- No persisted evidence path uses a system temporary directory.

End state:

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
