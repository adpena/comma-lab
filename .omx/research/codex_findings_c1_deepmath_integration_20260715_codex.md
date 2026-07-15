# C1 deep-math representation/optimization integration — raw findings

UTC: `2026-07-15`

Lane: `c1_deepmath_integration_20260715`

Authority: source re-derivation + existing durable receipts + local deterministic tests only. No
training, scorer-forward, archive creation/mutation, provider dispatch, exact evaluator call, score,
promotion, or pointer mutation was performed.

Verdict: **`C1_DEEPMATH_NOT_CERTIFIABLE_YET__APPARATUS_HARDENED`**.

Pointer: **UNMOVED** — `0.1910828242 [contest-CPU Linux x86_64]` submittable;
`0.1880443979880752` remains defensive-bank/non-submission custody.

## Status vocabulary

- `BUILT`: implementation bytes exist in this checkout, not merely a memo/receipt from another tree.
- `V9-WIRED`: a typed DSL/config edge reaches the real trainer/receiver consumer and composes with the
  selected V9 program. Policy metadata without a semantic consumer is not wired.
- `TESTED`: the named implementation has executable structural/numerical tests. A passing toy identity
  is not a through-R verdict.
- `MEASURED n600 through-R`: all 600 pairs traverse actual R and frozen SegNet on receiver-valid RGB.
  Saved-OFF/advisory, unequal-byte, or non-current-config receipts remain explicitly scoped.
- `ON`: admitted into the C1 optimal-form selection. This is stricter than “a base mechanism executes.”
  An unmeasured base mechanism is reported in `V9-WIRED` but is **not** certified `ON` here.

## Per-finding integration matrix

| finding | BUILT? | V9-WIRED / composes? | TESTED bit-identity / structure? | MEASURED real n600 through-R no-regression? | ON in certified C1? |
|---|---|---|---|---|---|
| **#171 task-space softmax-of-SDF level-set vehicle** | **YES** — `lever_b_levelset_generator.py`, MLX trainer, byte-close receiver | **YES as the base vehicle**, including K-field `out_sdf`; not a newly switchable representation lever | **YES** — NumPy/MLX/receiver and argmax/SDF invariants exist | **NO for current C1 V9 config.** Prior feasibility `1.27e-5` and n96 HOSC `0.00124` are through-R/advisory formulation anchors, not a current byte-closed n600 C1 row | **NO certification**; structurally executing base, but current-C1 n600 custody is owed |
| **#180 standalone Morse-Smale partition codec** | **YES** | **NO** as V9 carrier; measured-dominated standalone is intentionally deferred | **YES** | **NO current C1.** Prior standalone result `d_seg=5.57e-4 @ 740 B/frame` is advisory/formulation evidence and rate-dominated by the SDF witness | **NO** |
| **#180 Morse-Smale birth-completion / persistence control** | **YES** — `birth_completion.py`, persistence topology loss, DSL `v75_birth_completion_event` | **YES structurally** in the inherited V9 lever set; it is control logic, not the standalone codec | **YES** — focused birth/counterforce and resume tests | **NO current-C1 isolated n600 A/B** | **NO certification**; active mechanism, isolated no-regression owed |
| **#284 Amortizing-the-Argmax laws** (Laguerre/tropical/Maslov/Fisher/Γ-limit) | **YES as equations + several DSL/control consumers**, not one actuator | **PARTIAL** — power-diagram, τ/eikonal, stage-order and head-offset pieces are wired; no monolithic “#284 lever” exists | **YES** for the eight canonical laws and consuming helpers | **NO monolithic/current-C1 n600 receipt**; individual historical anchors retain their own scopes | **NO monolithic ON claim** |
| **#286 τ-coupled eikonal/interface-width control** | **YES** — typed `unified_tau_eikonal_hold` and rung controller | **YES** — ideal config emits `0.01 -> 0.05` before lower τ rung | **YES** — equation/controller/resume tests | **NO exact current-C1 n600 delta.** Base `0.01` and end `0.05` are measured/ledger anchors; interpolation/order is DERIVED | **NO certification**; compiled mechanism, A/B owed |
| **#217 post-Muon leap-residual reheat** | **NO** — design-only; no trainer flag or semantic consumer | **NO** | **NO implementation test** | **NO** | **NO**; exact build gate is #216 saddle signature plus an actual gain-decoupled sphere/Stiefel block |
| **#500 `argmax_native_vjp_fidelity_v1` optimal metric** | **YES** — canonical metric, policy binding, Bregman/Mahalanobis guards | **POLICY-WIRED, MODEL-ACTIVATION FALSE** — no measured renderer/Fisher pullback Gram or trainer preconditioner consumer | **YES** — source/binding/equation tests | **NO** — `NO_VERDICT_DATA_CUSTODY` for live n600 selected metric | **NO** |
| **#504 Bregman/Nielsen identities** | **YES** — stable KL/Bregman, centroids, sigma propagation, affine gauge checks | **NO live model factorization/covariance consumer**; declarative V9 binding only | **YES** — exact identities, extreme-logit guard, 600 synthetic SPD states; dual-vs-H² error about `9e-13` | **NO** — synthetic/local geometry, not through-R | **NO** |
| **#500/#504 Fisher-natural `H^-1` solve + trust radius** | **YES after this landing** for a gauge-fixed SPD chart: typed `numpy.linalg.solve` plus `H`-norm trust-ball projection | **NO** — categorical quotient basis, live pullback `H`, damping law, optimizer consumer, checkpoint receipt absent | **YES locally** — interior, boundary, zero cotangent, invalid radius, no explicit inverse | **NO** | **NO** |
| **#502 legacy polar-directional Fourier control** | **YES** | **YES** — default `--basis legacy_fourier_ab_control` | **YES** — train/checkpoint/inflate legacy path | **YES only for bounded historical formulation:** ep675 OFF `0.004244`; not current full V9 C1. It is not a curvelet win | **CONTROL ONLY**, never labeled optimal |
| **#502 current `windowed_curvelet` trainer token** | **YES as code, but INVALIDATED AS LITERAL CURVELET** — spatial Gabor-like wave packet | **NO** — trainer refuses V9 IPE+taper; name cannot be used as literal-family custody | **YES for localization/NumPy/MLX/receiver structure**, but those tests do not cure family misidentification | **NO admissible family row**; invalidated round-1 rankings cannot be reused | **NO / FORBIDDEN as literal curvelet** |
| **#502 literal finite polar-frequency-wedge curvelet** | **RECEIPT-BUILT, SOURCE ABSENT HERE** — clean structural proof SHA `677a2252c43c1272ec0e2e83d65ce1b82d23b8ddb089d73a111a5f0b26d46d25`; named `localized_basis_frames.py` is missing | **NO** train/MLX/checkpoint/generated-inflate consumer | **STRUCTURAL PROOF CLEAN**, but executable source cannot be re-opened in this checkout | **ADVISORY saved-OFF n600 receiver only:** `0.5048239560`; equal 109,559 scalar/support values, not equal bytes, not fresh training, not promotable | **NO** |
| **#502 compact shearlet** | **YES primitive/equation** | **NO trainer basis token/receiver consumer** | **YES structural tests** | **ADVISORY saved-OFF n600 receiver only:** `0.4288604312`; same unequal-byte/non-fresh scope | **NO** |
| **#503 recursive-fractal DecisionCarrierBundle / palette-chroma** | **NO in this checkout** — prior owner recorded source hashes but serializer failed; claimed modules are absent | **NO** — `V9_INTEGRATION_BLOCKED_OWNER`, real parser/disabled consumers absent | **Historical local tests/receipts only**, not re-openable implementation custody here | **NO.** Pair-0 palette rank is 15/15; n600 is a 12.0068 h DERIVED projection. No alternate archive/receiver A/B or achieved saving | **NO**; `NO_VERDICT_RECEIVER_RATE_CUSTODY` |

## Measured through-R rows — exact scope

| row | value | axis/custody | admissible conclusion |
|---|---:|---|---|
| legacy polar-Fourier, self-orient OFF, ep675 | `d_seg=0.004244` | n600 bounded warm-start, macOS-CPU advisory | control formulation only |
| self-oriented legacy Fourier, along8, ep675 | `d_seg=0.0042590586` | same bounded n600 formulation | no realized improvement over OFF; not a localized-frame verdict |
| self-oriented legacy Fourier, along26 | `d_seg=0.004286` | same bounded n600 formulation | no realized improvement over OFF; not a localized-frame verdict |
| saved-OFF receiver, legacy Fourier | `d_seg=0.4097223155` | n600 receiver RGB through R, CPU-torch batch32, advisory/non-score | finite-truncation receiver row only |
| saved-OFF receiver, compact shearlet | `d_seg=0.4288604312` | same, equal scalar/support count but not equal bytes | family stays open |
| saved-OFF receiver, literal finite polar curvelet | `d_seg=0.5048239560` | same, equal scalar/support count but not equal bytes | family stays open |
| fixed non-PoU mix | `d_seg=0.5303014119` | same; explicitly not a decoder-boundary PoU | mix formulation only |

There is **no** new measurement row from this arm. None of these rows is a current full-lever V9 versus
fresh stripped-literal-curvelet byte-closed A/B. The spectral `1.7–2.0x` and OMP `1.09x` values are
upper-bound/capacity evidence (`score_claim=false`), not through-R no-regression authority.

## Curvelet composition resolution

### Guard audit

The refusal at trainer lines 3996–4015 is partly over-broad but cannot be safely relaxed into a
curvelet activation:

1. `self_orient` concatenation is basis-generic in the current trainer, and ground-frame alone calls
   `_basis_feats_np(chart_coords)` generically. Their curvelet-specific refusal is broader than the
   downstream mechanics. Ground-frame plus self-orient remains genuinely incompatible because it mixes
   frame-coordinate tangents with ground-coordinate basis features.
2. d-seg taper mathematically reweights arbitrary feature columns, but current windowed-curvelet MLX
   training retains the pre-taper MLX tensor when no per-pair cache is active. More importantly, the
   generated receiver never regenerates or folds the GT-derived taper. This is a receiver-bijection gap
   for the full Fourier control too, not permission to relax the guard.
3. IPE is genuinely Fourier-specific. It computes one scalar attenuation from each global frequency
   vector `B`. A literal curvelet atom is a sum over a radial/angular frequency wedge; correct IPE must
   attenuate each wedge frequency before summation. Reusing the Fourier scalar is a name-preserving fake.
4. The selected `windowed_curvelet` implementation itself was invalidated as the literal family after
   the round-1 build. Relaxing its guard would integrate the wrong mathematical object.

### C1 decision surface

The decision is therefore **option (b)**, with two required comparisons after build closure:

1. **Optimal-form decision A/B:** current full-lever legacy-Fourier control versus stripped
   literal-polar-curvelet treatment (`render_aa=none`, d-seg taper OFF until receiver-folded, self-orient
   OFF). This answers which complete candidate config wins, but is intentionally multi-lever.
2. **Causal basis A/B:** matched stripped legacy-Fourier versus matched stripped literal-polar-curvelet,
   differing only in the basis family and its required receiver manifest. This attributes the basis.

`v9_ideal_mod32_basis_ab_configs()` remains a pure typed compile receipt and now reports
`BLOCKED_FAIL_CLOSED_BUILD_THEN_OPERATOR_GO`, `launch_ready=false`, and all three exact blockers. It no
longer describes the basis-only argv delta as a runnable A/B.

Exact build needed before either fire:

- recover/land and re-open the clean literal polar-frequency-wedge source bytes;
- add a distinct honest family ID and DSL Lever (do not reuse invalidated semantics under the same name);
- implement NumPy authority, MLX parity, checkpoint/resume, generated-inflate and byte-close consumers;
- implement wedge-correct IPE **or** keep the treatment stripped;
- fold d-seg taper into `in_proj.weight` for receiver parity or remove it from both matched arms; never
  ship GT-derived taper state as uncounted free decoder input;
- prove exact packet/uint8 bit identity and basis-manifest mutation sensitivity;
- then operator-GO `curvelet_through_R_dseg_ab`, n600, real R, exact byte-closed archives, both declared
  contest axes kept separate.

## C1 deep-math config delta landed here

| surface | before | after |
|---|---|---|
| basis A/B readiness | `PREPARED_NOT_FIRED_OPERATOR_GO_REQUIRED` despite treatment trainer refusal | `BLOCKED_FAIL_CLOSED_BUILD_THEN_OPERATOR_GO`; three source-derived blockers + `launch_ready=false` |
| default basis | `legacy_fourier_ab_control` | **UNCHANGED** control; no silent optimal claim |
| literal curvelet | absent from live train/receiver | **UNCHANGED / blocked**; no fake substitution |
| #503 DCB | absent from this checkout/live V9 | **UNCHANGED / blocked**; no policy-only fake wire |
| Fisher-natural geometry | gauge-fixed `H^-1` solve exists | adds typed trust-radius projection; remains model-inactive |
| score/pointer | 0.19108 / 0.18804 bank | **UNCHANGED** |

No trainer argv, model, checkpoint grammar, receiver packet, score ledger, or pointer was changed.

## Triality and equations

- **DSL:** `WindowedCurveletBasis` and `LegacyFourierABControl` are mapped by
  `lever_registry.lever_factories()` to the real `--basis` flag. `activation_ledger.duty_to_measure()`
  keeps `WindowedCurveletBasis` owed until a real fired+measured close event. The compiled V9 receipt is
  now explicitly blocked. No Bregman Lever was invented because no live Hessian/model consumer exists.
- **DAG:** `.omx/research/c1_deepmath_integration_DAG_FEED_20260715.md` is the append-only concurrent
  feed for canonical union; every negative carries a formulation/implementation-custody scope.
- **Equations:** `windowed_curvelet_parabolic_capacity_v1` remains upper-bound/localization scope only;
  `curvelet_directional_basis_dseg_reduction_v1` retains its bounded legacy self-orient caveat;
  `bregman_dual_metric_squared_hessian_v1` owns
  `||delta_eta||^2=delta_theta^T H^2 delta_theta` and forbids calling it Fisher-natural;
  `argmax_of_sdf_is_additively_weighted_power_diagram_v1` owns the Morse-Smale separatrix identity;
  `eikonal_retention_couples_to_tau_rung_v1` owns the `0.01 -> 0.05` rung law; the #284 Maslov,
  Fisher-caustic, mirror-descent/natural-gradient, `tau_eps_hbar`, Modica-Mortola and MCF laws remain
  separately scoped.

## Operator-GO gates and exact blockers

1. **Not GO-able yet:** literal curvelet train/receiver source and parity are absent.
2. **Not GO-able yet:** full-lever control's GT-derived d-seg taper lacks receiver folding/regeneration.
3. **After those builds:** operator-GO `curvelet_through_R_dseg_ab`; governed launcher, storage
   preflight, resumable periodic + every-stage preserved checkpoints, exact archive custody, n600 real R.
4. **#500/#504:** live quotient-chart/pullback Hessian + damping/radius law + optimizer consumer +
   checkpoint/telemetry receipt before any natural-gradient fire.
5. **#503:** recover exact source bytes, real parser/disabled consumer, DCB encoder/parser, alternate
   archive and receiver A/B, and achieved byte-saving through-R receipt before composition.
6. **#217:** build #216 signature and a gain-decoupled admissible manifold block before a leap-residual
   stage can become a DSL Lever.
7. **C1 certification:** one clean converged V9 C0 byte-closed n600 row plus isolated A/B rows for every
   candidate marked active. Until then, every activation above is MEANS, not an optimal-form verdict.

## Own round-1 review

| question | verdict |
|---|---|
| Is every certified `ON` claim n600 through-R measured? | **YES vacuously: no deep-math candidate is certified ON.** Structural base consumers are explicitly separated from C1 certification. |
| Is the curvelet crux resolved? | **YES as an A/B-decision owed, not silently sided.** Full composition is mathematically blocked by IPE and implementation-custody blocked by the missing literal source. |
| Are unwired items wired+measured or blocked with exact build? | **YES.** #503, literal curvelet, standalone Morse-Smale, #500 model activation and #217 each name the missing consumer/custody. |
| Is Bregman solve built or ticketed? | **BUILT before this arm; trust-radius projection built here.** Live quotient/pullback/damping remains ticketed. |
| Did any proxy become score authority? | **NO.** OMP/spectral, mask, saved-OFF receiver, pair0 rank and synthetic SPD identities remain scoped. |
| Pointer delta | **0.** |

## STORES CONSULTED

`docs/operating_manual_craft_handoff.md`; `CLAUDE.md`; `AGENTS.md`; project current-state memory;
`src/tac/witness_dsl/{spec_v9_cgauge,optimal_basis_20260714,windowed_curvelet_basis_lever_20260714,lever_registry,activation_ledger}.py`;
the MLX trainer and byte-close receiver; #500/#502/#503/#504 findings/specs/receipts; #171/#180/#217/#284/#286
research and equation surfaces; lane/task state; inbox through `2026-07-14T20:32:37Z`; audit git object
`617cacc047`. No later inbox directive was present at this review checkpoint.

## Transactional custody receipt

- Hash-bound serializer invocation: **ATTEMPTED** with all nine owned files, each modified file's
  `HEAD` SHA-256, each new file declared `base=new`, every post-edit SHA-256, and
  `triality_legs=dag,dsl,equations`.
- Result: **BLOCKED_SANDBOX_GIT_OBJECT_WRITE**, before staging or commit.
- Literal refusal: `git add failed (rc=128): error: unable to create temporary file: Operation not
  permitted; .omx/state/lane_maturity_audit.log: failed to insert into database; fatal: updating files
  failed`.
- Scope: filesystem/git-object custody only. It is not a content, test, review, preflight, scientific,
  or launch verdict. The exact reviewed worktree delta is intentionally preserved for MAIN harvesting.
- Repository-wide `lane_maturity.py validate` also reports 110 historical missing-evidence paths; none
  names `c1_deepmath_integration_20260715`. Scope: **INHERITED_GLOBAL_REGISTRY_DEBT**, not this lane.
