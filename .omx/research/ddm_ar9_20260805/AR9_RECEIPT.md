# AR9_RECEIPT - Nielsen Bregman MEB Crosswalk

Tags: [no-triality] [p0-ledger-ok]

Date: 2026-08-05

Scope: scorer-free literature and corpus crosswalk. No launcher, scorer, archive
mutation, exact eval, CUDA/CPU promotion, or frontier pointer movement.

Sources read:

- arXiv abstract page: https://arxiv.org/abs/2607.24197
- arXiv HTML v1: https://arxiv.org/html/2607.24197v1
- arXiv PDF v1: https://arxiv.org/pdf/2607.24197
- ResearchGate mirror: intentionally not retried after the charter-declared 403.

## Answer First

AR9 is useful, but it is not a new vehicle by itself. The actionable delta is a
representative-choice rule:

> When a packet cell, stratum, prototype, or codeword is priced as one
> representative for a set, test the minimax representative in the geometry that
> actually owns the error term. Do not silently substitute the mean/centroid for
> the minimax center.

Nielsen's paper makes the finite left-Bregman minimum enclosing ball equivalent
to a weighted point-set minimum enclosing ball under power distance after the
dual-gradient transform. It also identifies the Bregman Badouiu-Clarkson update
with the Frank-Wolfe update for that power-MEB problem. In our corpus this does
not replace the existing Laguerre/power witness identity; it sharpens where a
minimax center may beat an arithmetic/Bregman centroid and where any radius must
be tagged as target-geometry only.

The only follow-on I would queue from AR9 is a small, named-consumer minimax
representative probe against an existing receiver-visible codebook/stratum. The
#539 power-diagram route is already embodied and still blocked on receiver
arithmetic and spatial/RGB pullback. The v16/v17 trust-radius surfaces should
record the Bregman/power radius only as a target-space diagnostic, never as a
replacement for receiver-realized validity radius. The #504 Bregman leg is
sharpened but not promoted to a new canonical equation without a real consumer
receipt.

## Ranked Crosswalk

| Rank | Verdict | Consumer | AR9 delta | Falsifier | Cost and dispatch |
|---:|---|---|---|---|---|
| 1 | ADOPT as queued probe | `rg3` / `hope_bn_capacity.py` FISHER_MARGIN_SITE_LOCAL_PER_STRATUM_CODEBOOK first; v14 prototype colors or #869 adaptive-quant cell reps only if their current representative and receiver path are named | Mean-like representatives and minimax representatives answer different questions. AR9 gives a cheap way to compute the minimax Bregman/power center for a finite stratum before spending archive bytes. | Same counted bytes and receiver path, but minimax center fails to reduce worst-case margin debt or receiver-through-R `d_seg`; or no receiver-visible cell can be named, making it target-only. | QUEUED-WITH-A-FIRE-ORDER. $0 if it reuses existing cached stratum features/margins; no scorer launch in this receipt. |
| 2 | ALREADY-EMBODIED, no new build | #539 power-diagram witness parametrization; `src/tac/boundary_math/power_diagram_witness.py` | AR9's lifting/farthest-power language matches the existing affine-head Laguerre/power quotient and may support later support-set/coreset compression. It does not solve the nonlinear feature-field pullback, RGB realization, or declared receiver arithmetic blocker. | Any claim that AR9 alone makes a spatial witness receiver-closed; frame195-style arithmetic mismatch or absence of paired feature/RGB realization refutes promotion. | FOLDED into existing #539/PDW backlog. New build forbidden until the receiver arithmetic contract is named and an n600 receiver path is available. |
| 3 | ADOPT as metadata discipline, not authority | v16/v17 trust-region and `ddm_dynamic_quantum_calibration_20260724.py` validity-radius surfaces | A Bregman-MEB radius is a ball radius in target/head geometry after choosing the divergence and coordinate side. It can label candidate set diameter, not certify receiver survival. | Any attempt to use target-space Bregman radius as a universal validity radius or to skip the measured receiver-realized rho gate. | QUEUED-WITH-A-FIRE-ORDER on the next trust-radius touch: add a precondition tag that states geometry, side, and receiver-boundary status. No standalone scorer. |
| 4 | ALREADY-EMBODIED, no new equation today | #504 Bregman leg; `bregman_all_surfaces_504_derivation_20260715.md`; closed scorer canonical equation | AR9 extends the same information-geometry family from centroid/projection facts to minimax balls. It sharpens the distinction between centroid policy and minimax policy but does not change the existing gauge identity. | A new canonical equation would be fake unless tied to a landed minimax consumer or a measured receiver/trust-radius effect. | FOLDED. Keep as equation-worthy only after rank-1 or rank-3 produces a receipt. |
| 5 | N-A until consumer | Generic Frank-Wolfe MEB implementation | The algorithm is tiny and relevant only as a subroutine for rank-1 minimax representatives or a later #539 support-set compressor. | A standalone FW-MEB helper with no caller, no geometry tag, or no receiver/coder effect is pure means-as-ends. | N-A now. If rank-1 fires, implement locally inside that probe or as a private helper with the consumer named in the same patch. |

## Paper-To-Pact Mapping

MEASURED here means measured in prior Pact artifacts and only recalled here. AR9
itself was not experimentally measured in this turn.

| AR9 object | Pact object | Status |
|---|---|---|
| Weighted point-set MEB under power distance | Frozen Seg head as a 5-site Laguerre/power diagram in rank-4 quotient space | PRIOR MEASURED/VERIFIED locally in `ddm_sx1_separatrix_carrier_20260803.md`, `power_diagram_witness_20260718.md`, and canonical equation code. AR9 is confirmatory, not novel. |
| Bregman minimax center | Codeword/prototype/cell representative under a chosen task geometry | NEW DECISION RULE. Needs a named cell/stratum and receiver/coder comparison before adoption. |
| Frank-Wolfe power-MEB approximator | Possible minimax representative solver or #539 support-set reducer | USEFUL ONLY WITH CONSUMER. No helper landed. |
| Farthest Bregman Voronoi / farthest power diagram | Candidate support-set interpretation for active worst-case cells | DIAGNOSTIC. Could rank hard cells/edges, but must not replace receiver-through-R evidence. |
| Bregman radius | Target-space trust diameter | LABEL ONLY until calibrated against receiver-realized validity radius. |

## Representative Rule

Current Pact artifacts already distinguish several representative regimes:

- #504 derived right-data Bregman centroids as inverse-gradient averages and
  separately records the opposite-orientation arithmetic mean.
- RG3/HOPE selects capacity by exact n600 frozen-Seg input measure and
  Fisher-margin stratum tables; its canonical code explicitly lacks rate
  denominators and ships nothing.
- v14/#869-style prototype/color/cell representatives are score-relevant only
  if the receiver path and counted bytes survive.

AR9's minimax center is not interchangeable with those means. A centroid
minimizes aggregate divergence; a minimum enclosing ball center minimizes the
maximum divergence/radius. The likely value is in cells where one hard tail
controls `d_seg`, not in smooth mass-dominated strata. The correct probe is:

1. Select one existing receiver-visible stratum/cell with current representative
   policy and cached feature/margin evidence.
2. Compute the incumbent representative and the AR9 minimax center in the same
   declared geometry and coordinate side.
3. Price identical receiver/coder payload shape, or stop as target-only if that
   shape is unavailable.
4. Compare worst-case margin debt first, then receiver-through-R `d_seg` and
   counted bytes only when a legal receiver path exists.

## #539 Power-Diagram Impact

AR9 does not reopen the core #539 conclusion. The corpus already has:

- An exact affine-head to Laguerre/power quotient identity in real arithmetic.
- PDW1/PDW2 packet constructions that reduce deterministic target description
  bytes in the head quotient.
- A documented blocker: target-only power packets are not equivalent to spatial
  RGB witnesses, and the n600 attempt hit a frame195 receiver-arithmetic
  mismatch.

The only AR9 addition is conceptual compression pressure: if a future receiver
contract closes arithmetic and RGB realization, the active MEB support set might
be a smaller support certificate than all target cells. That is not a present
promotion path.

## Trust-Region Impact

AR9 makes the geometry-tag precondition stricter. A trust ball must say where it
lives:

- head quotient / target logit geometry;
- categorical/Bregman output geometry;
- RGB input after R;
- receiver-realized post-R scorer surface.

The v16/v17/dynamic-quantum validity-radius law remains authoritative only after
receiver-realized calibration. A Bregman/power radius can be a precondition or a
candidate-pruning statistic; it cannot certify byte-closed safety or replace the
measured rho gate.

## #504 Gauge And Canonical Equation Custody

The #504 Bregman leg already has finite-divergence identities, centroid
orientation guards, and an implementation-custody gap label. The closed scorer
canonical equation already includes rank-4 Laguerre/power Seg cells plus
categorical Bregman debt. AR9 therefore sharpens language, not custody:

- centroid policy = average-risk representative;
- MEB policy = worst-case-risk representative;
- gauge identity = still the existing affine/power quotient, not a new model
  factorization fact.

No canonical equation should be added from AR9 alone. It becomes
canonical-equation-worthy only if a minimax representative or trust-radius
receipt lands with a concrete consumer.

## Follow-On Ledger

### QUEUED-WITH-A-FIRE-ORDER - ar9_minimax_representative_probe

Consumer: first available one of:

- RG3/HOPE `FISHER_MARGIN_SITE_LOCAL_PER_STRATUM_CODEBOOK` stratum with existing
  cached feature/margin rows and a named current representative.
- v14 margin-optimal prototype color cell, if its receiver path and current
  representative code are named.
- #869 adaptive-quant cell representative, if its receiver path and current
  representative code are named.

Fire order:

1. Read the consumer's current representative implementation and cached evidence.
2. Declare divergence, coordinate side, and target/receiver boundary before
   computing any center.
3. Compute incumbent representative vs AR9 minimax Bregman/power center on the
   same finite set.
4. If no legal receiver/coder payload exists, stop as `TARGET_ONLY_DIAGNOSTIC`.
5. If a payload exists, compare same-byte receiver-through-R `d_seg` on the
   smallest authority-compatible cached sample first; promote only after the
   governing n/sample rule is satisfied.

### FOLDED - #539 AR9 lifting route

Fold into the existing PDW/#539 backlog. Do not build a new power-diagram
parametrizer from AR9 alone. The receiver arithmetic and spatial pullback
blocker still owns the next legitimate fire order.

### QUEUED-WITH-A-FIRE-ORDER - trust_radius_geometry_tag

On the next v16/v17/dynamic-quantum trust-radius edit, add an explicit
`bregman_power_radius_scope` or equivalent receipt field that states the target
geometry and says whether it is receiver-realized. Do not run a scorer just for
this metadata.

### FOLDED - #504 canonical equation extension

Do not add a new canonical equation from literature alone. Reconsider only after
the minimax representative probe or trust-radius tag produces a landed artifact.

### N-A - standalone Frank-Wolfe MEB helper

No caller, no helper. If the minimax representative probe needs FW-MEB, it may
land with that probe and name the consumer in the same commit.

## Recall Evidence

Stores consulted:

- `.omx/tmp/codex_runs/ar9_prompt.md`
- `.omx/tmp/codex_runs/_common_contract.md`
- `PROGRAM.md`
- `CLAUDE.md`
- `AGENTS.md` from the user-provided context
- `docs/operating_manual_craft_handoff.md`
- `.omx/state/main_hot_state.md`
- arXiv abstract, HTML, and PDF for `2607.24197`
- local `.omx/research`, `src/tac`, `docs`, and canonical equation registry

Queries run:

- `Bregman|Laguerre|power diagram|power-distance|Voronoi|MEB|circumcenter|minimax|farthest`
- `#539|tropical|argmax partition|terminal-feature|frozen head|power witness|Laguerre`
- `#504|Bregman centroid|centroid|mean centroid|FISHER_MARGIN_SITE|margin-optimal|prototype|adaptive-quant|representative|trust region|validity radius|gauge`
- canonical-equation registry search for `closed_scorer_action`, `Laguerre`,
  `Bregman`, `FISHER_MARGIN`, `validity_radius`, `dynamic_quantum`, and `gauge`

Key local evidence opened:

- `.omx/research/ddm_sx1_separatrix_carrier_20260803.md`
- `.omx/research/nielsen_infogeo_crosswalk_20260719_codex.md`
- `.omx/research/bregman_all_surfaces_504_derivation_20260715.md`
- `.omx/research/operator_directive_per_edge_optimality_criteria_20260805.md`
- `.omx/research/power_diagram_witness_20260718.md`
- `.omx/research/v10_power_diagram_byteclose_findings_20260718.md`
- `.omx/research/power_diagram_byteclose_DAG_FEED_20260718.md`
- `.omx/research/ddm_ms1_min_description_lattice_solve_canonical_equations_20260724.md`
- `src/tac/boundary_math/power_diagram_witness.py`
- `src/tac/canonical_equations/closed_scorer_variational_de_20260721.py`
- `src/tac/canonical_equations/seg_rate_breakeven_and_head_gauge_laws_20260719.py`
- `src/tac/canonical_equations/ddm_v17_validity_radius_law_20260723.py`
- `src/tac/canonical_equations/ddm_dynamic_quantum_calibration_20260724.py`
- `src/tac/canonical_equations/hope_bn_capacity_per_stratum_20260727.py`
- `src/tac/optimization/hope_bn_capacity.py`

Found beyond charter seeds and impact:

- The prior Nielsen information-geometry crosswalk already names #539/PDW as
  the main power-diagram consumer and warns that Bregman radii do not price
  bytes by themselves. AR9 is therefore a refinement, not a first discovery.
- PDW1/PDW2 are already implemented and byte-described in target space, but the
  frame195 arithmetic mismatch and target-only/non-equivalence warnings remain
  the live blocker.
- The HOPE/RG3 codebook surface is a better first minimax-representative
  consumer than a new #539 build because it already owns finite strata and
  Fisher-margin site-local codebook language.
- The v17 and dynamic-quantum equations explicitly require family-specific
  receiver-realized validity radii; AR9 cannot override that with a target-space
  radius.
- The closed scorer canonical law already owns rank-4 Laguerre/power Seg cells
  and categorical Bregman debt, so a new literature-only canonical equation
  would duplicate settled law.

No exact scorer, public evaluator, archive mutation, or GPU/CPU dispatch was
run. No `S` component was recomputed from AR9 because AR9 produced no candidate
archive and no receiver-realized row.

## NEXT_IF_RESUMED

Resume only if a named consumer is selected. Start with RG3/HOPE unless a newer
operator directive names v14 or #869 first.

Minimal next unit:

1. Locate the exact current representative implementation for one finite stratum.
2. Extract the finite point set and current representative without launching a
   scorer.
3. Compute the minimax Bregman/power center in a geometry-tagged scratch result.
4. Decide `TARGET_ONLY_DIAGNOSTIC` vs receiver/coder-compatible probe before any
   measured claim.
5. If receiver-compatible, queue through the normal scorer lane discipline and
   record denominator, axis, archive bytes, and pointer status.

Expected stop conditions:

- Stop as FOLDED if no current representative implementation can be named.
- Stop as TARGET_ONLY_DIAGNOSTIC if the minimax center cannot be represented by
  the same receiver/coder payload.
- Stop as NO-PROMOTION if the center changes target-space radius but not
  receiver-through-R `d_seg` or counted bytes.

Frontier status: S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory];
contest pointer borrowed/unmoved.
