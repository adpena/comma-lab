# Costate ORGAN v2 exact-anchor implementation spec (2026-07-21)

**Lane:** `p0_costate_organ_factorization_grounded_ABC`  
**Authority:** delegated `costate_exact_anchor_v2`; `advisory_only=true`; `actuation=NONE`; MAIN landing review required.  
**Verdict scope:** implementation spec derived before code. No score, promotion, stop, launch, live-config, or pointer authority.

## Objective and closed composition

The v2 readback is a sibling of the existing v1 factorized adjoint, never a replacement:

`lambda(pair, site) = exact_gap * visibility * realizability * byte_price`.

The implementation will be a pure, deterministic, NumPy-free critical path under
`tac.witness_control.costate_organ_v2`. It will expose the four named factors, their
provenance, factor ablations, apparatus-validity gates, opportunity-pool identity, dual
Euclidean/Fisher readback, and `_dev`/`_prod` maturity. No learned parameter is admitted;
an optional residual field remains default-off and reports zero parameters.

The exact Seg leg uses the frozen rank-4 head chart and an explicit four-tap bilinear
resize adjoint. It must preserve the all-class gauge null and `lambda_seg(frame_0)=0`.
The rate leg consumes `realization_breakeven_bytes_v1` by registry ID and callable, and
pool ordering consumes registered `witness_measured_reverse_waterfill_v1`; the distinct
07-18 same-pool non-additivity law has no registry row as of this sweep, so its own
formal registration remains `FORMALIZATION_PENDING` rather than receiving an invented ID.

## Findings to Organ consumption matrix

| Finding | SENSE input | DECIDE factor | Readback format | Apparatus-validity precondition | Not applicable with reason |
|---|---|---|---|---|---|
| #547/#549 exact inverse solve; practical n600 anchor `d_seg=0.00015196`, `d_pose=0.00010184` | current realized-through-R `d_seg,d_pose`; exact anchor custody | `exact_gap` in score units, nonnegative | current, anchor, Seg and Pose score debt | exact bytes/scorer axis named; fp32 canonical support fill | #549 n24 zero-band is confirmation, not the n600 value source |
| Tie-aware support fill is canonical fp32 exact | fill policy/custody | `exact_gap` admission | `support_fill=fp32_exact_canonical` | no tie-aware substitute | none |
| Frozen rank-4 Seg head (#486/#487) | pair direction, centered head weights | closed `lambda_seg` chart | rank, gauge-null residual, LawRef | frozen scorer weights | no learned class direction |
| Exact four-tap resize | site tap weights/channel | `visibility` and camera adjoint | tap count, weights, visible mass | exact resize geometry | never stale axis-aligned mask |
| Full real-linear nullity 80.6742% (#580) | `nullity_full_kernel` | visible fraction `1-nullity` | full-kernel vs stale-mask values side by side | direct-sum law geometry | does not imply cheap bytes |
| Frame 0 is Seg-free | frame index | `visibility_seg=0` | explicit zero theorem | SegNet last-frame input contract | Pose remains two-frame |
| Pose chroma 2x2 box | channel/frequency support | per-channel visibility | Seg and Pose visibility separately | distinguish full-res Seg from Pose | sub-2px chroma is Pose-invisible, not Seg-invisible |
| r1b6/r1b7 realization gate | formulation, requested/feasible/survived writes | `realizability` | 30.1% design anchor plus formulation result | uint8/resize/parse-back custody | r1b7 fixed magnitude is not a distinct arm |
| 30.1% M1 design-time anchor | band-design route | default design `realizability=0.301` | numerator/denominator | design-only | not a measured n600 recovery fraction |
| `realization_breakeven_bytes_v1` domain refinement | realized recovery and charged bytes | `byte_price` | equation ID, latest registry event, break-even bytes | latest event must be `domain_refined` | scheduled upper bound never substitutes for realized recovery |
| 07-18 exclusive pools | pool A/B/C identity and remaining ceiling | same-pool KKT marginal; never sum | pool, ceiling, used/remainder | registered KKT consumer ID present | dedicated pool equation absent: `FORMALIZATION_PENDING` |
| Per-class priors Road 50/Lane 19/Undrivable 13; Lane 77% skip-limited | class/pair/site | exact-gap allocation prior only | raw prior and skip-limited flag | instance/date scope preserved | not a universal class law |
| Fisher bank `765457d4...` | bank header/site row | tie-break / EV order only | bank SHA, candidate count, Fisher terms | exact bank SHA and schema | never fabricates missing site rows |
| Optional xi transport | previous-pair site and topology event | optional transport multiplier | applied/refused reason | no sparse-topology event | excluded on birth/death/saddle events |
| 52% head-norm affine gauge, dense fixed grammar rate-neutral | direction projected into gauge/complement | rate lambda zero in gauge; rank complement | gauge fraction and grammar scope | tested dense fixed-shape grammar | not a general entropy-free claim |
| Euclidean plus Fisher-cosine | candidate/reference vectors | never blended into one metric | both cosines, relative norms, sign flip | vector custody and compatible basis | absent vectors produce unavailable, not zero |
| `--ckpt-every 1` observer poison | launch/telemetry cadence | exclude contaminated window | contamination flag/reason | clean checkpoint cadence | contaminated rows never enter correlation |
| EMA lag after resume | resume and EMA-shadow custody | correction only when verified | applied/unknown plus evidence | explicit EMA reset/lag evidence | otherwise no numeric correction |
| 90.6% edge flicker; flat-amplitude exhaustion | stratum, temporal persistence, formulation | temporal duty and formulation gate | edge fraction, flicker, verdict scope | trained-witness/post-hoc-flat match | never promoted to family-wide floor |
| `_dev`/`_prod` maturity | recommendation maturity | promotion eligibility | maturity and pointer eligibility | explicit tag | untagged recommendations fail safe to `_dev` |
| ACME Vol.4 Pontryagin/LQR conformance route | fixed synthetic scalar plant/cost/control only | no DECIDE factor; validates adjoint convention | analytic/FD errors, sweep residuals, Riccati-root disposition | fixture-only, bounded projection, no runtime state | cannot authorize curriculum or live control; nonlinear training plant remains unformalized |

## Validation plan

1. Build a read-only historical corpus from the existing #205 as-of trajectory decisions
   and the C2 carrier-smoke receipts. Each row keeps source path/hash, pre-intervention
   factors, realized `DeltaS`, contamination, and formulation scope.
2. Compare Spearman rank correlation of current DECIDE, exact-anchor v2, and realized
   `DeltaS`; report ties and excluded contaminated rows.
3. Ablate each factor independently and report rank correlation, not just the full product.
4. Require v2 to improve over old DECIDE on the combined retrospective corpus. This is a
   development/retrospective gate only; it cannot establish live generalization or promotion.
5. Wire the v2 row beside v1 in `tools/costate_digest.py` and
   `tools/costate_shadow_report.py`; defaults remain read-only and fail-open.
6. Run a fixed synthetic scalar-LQR forward/backward conformance fixture: compare
   backward costates and bounded projected controls with the finite-horizon Riccati
   solution, central-difference both Hamiltonian derivatives, require monotone relaxed
   sweep residual, and reject the non-stabilizing algebraic Riccati root. The fixture is
   validation-only and supplies no live-control authority.

## Triality and wire-in

- **DSL:** no new trainer flag or actuator. Inputs are typed recommendation/readback fields.
- **DAG:** a dated `costate_organ_v2_exact_anchor_DAG_FEED_*.md` will record producer,
  validation, consumer, and authority boundaries.
- **Equations:** existing LawRefs are consumed by ID. The v2 composition law remains
  `FORMALIZATION_PENDING` until the backtest receipt exists; only then may a canonical
  equation registration carry the empirical anchor.
- **Sensitivity map:** exact-gap/pair/site rows are reusable marginal inputs.
- **Pareto/bit allocator:** pool-aware marginal and canonical rate price are emitted; no
  candidate is admitted by a proxy-only gain.
- **Autopilot:** advisory readback only, `actuation=NONE`, `_prod` required for any future
  pointer-routable recommendation.
- **Continual learning:** backtest receipt and per-factor ablation are the durable anchor.
- **Probe disambiguator:** Euclidean/Fisher and transported/untransported reads remain
  separate; no hidden blend.

## Stores consulted

CLAUDE.md; AGENTS.md; v7.5/v8 SPECs; canonical equation registry (2026-07-15..20);
required `graph_memory_recall.py 'costate organ'`; 2026-07-15..20 memory sweep; #205
costate design/backtest surfaces; C2 per-class and witness-own decomposition receipts;
r1b6/r1b7 receipts; null-compiler/full-kernel and head-gauge receipts; Fisher EV bank.
