---
title: "Codex findings: SOAP, Muon, and Beyond crosswalk for Pact DDM / Muon surfaces"
date_utc: "2026-07-24T12:17:43Z"
lane_id: "lane_ddm_px1_soap_muon_beyond_20260724"
research_only: true
score_claim: false
authority_axis: "source-custodied advisory"
verdict_scope: "FORMULATION x optimizer-comparison measurement contract; no optimizer adoption or score verdict"
pointer_before: "0.1910828242 [contest-CPU Linux x86_64]"
pointer_after: "0.1910828242 [contest-CPU Linux x86_64]"
pointer_delta: 0
main_review_required: true
---

# Executive verdict

**ADOPT-MEASUREMENT-CONTRACT; DO-NOT-ADOPT-OPTIMIZER-CONSTANTS.**

Khona et al. make two directly useful points for Pact:

1. optimizer A/Bs are not interpretable until the realized update RMS is matched at the actual
   parameter-group boundary; and
2. Muon's Newton-Schulz quality must be measured on actual optimizer update matrices against an
   exact-polar reference, rather than inferred from the iteration name or a synthetic unit test.

Those points strengthen the gates for the incumbent final Muon stage, the historical #195
MD-decoupling arm, #366's grammar-coordinate descent engine, and pending #556
`FilmPolarSPDNormalMomentum`. They do **not** authorize importing the paper's `beta1=0.9`,
analytic approximately-`0.2` RMS correction, `16` PolarExpress iterations, epsilon, weight decay,
SOAP covariance factors, or layer-wise distributed optimizer into Pact.

The highest-EV `$0` retrofit is a default-off **shadow measurement receipt**, not a training
change: replay the same copied checkpoint/state and held gradient sequence through incumbent and
candidate update rules; log per-group raw-direction RMS, realized-update RMS, Newton-Schulz polar
residual, exact-SVD gap, and wall time; fit any scale transfer from those local observations; then
reset before a governed A/B. No local update-matrix custody was found in the inspected artifacts,
so a numerical answer today would be fake: **`NO_UPDATE_MATRIX_CUSTODY`**.

No training, GPU/paid dispatch, config mutation, frontier move, or score claim was performed.

# Source custody and claim verification

| Source | Custody read | What was verified |
|---|---|---|
| Khona et al., *SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales*, arXiv:2607.20548v1 | Full 32-page PDF read; downloaded PDF SHA-256 `f4ffc0bfd4ec865c659ce341f70fe027db06d07cfbbc1cf4c0309c6690b1d187` | Algorithms, experiment regimes, figures, appendices, limitations, update-RMS derivation, SOAP staleness mechanism, Muon orthogonalization caveat, distributed layer-wise design |
| NVIDIA `Emerging-Optimizers` | Released repository read at commit `bf77c4fb2471d90f9238206711add567d2d13c3d` | Muon defaults and scaling modes; Newton-Schulz dtype/coefficients; exact-polar comparison tests; SOAP current-gradient ordering; QR/eigh and KL-Shampoo controls |
| Pact checkout | Worktree base `f40abd6af2e969f8846a0b822b7ada6b076f4270` | Incumbent Muon construction and trainer wiring; #195 disposition; #304/#321 confound classes; #366 coordinate surface; #552/#556 status |

Primary links:

- Paper abstract/PDF: <https://arxiv.org/abs/2607.20548>,
  <https://arxiv.org/pdf/2607.20548>
- Released code: <https://github.com/NVIDIA-NeMo/Emerging-Optimizers/tree/bf77c4fb2471d90f9238206711add567d2d13c3d>
- Muon implementation at the pinned commit:
  <https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/bf77c4fb2471d90f9238206711add567d2d13c3d/emerging_optimizers/orthogonalized_optimizers/muon.py>
- SOAP implementation at the pinned commit:
  <https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/bf77c4fb2471d90f9238206711add567d2d13c3d/emerging_optimizers/soap/soap.py>

## Verified paper/code claims

| Claim | Verification | Pact implication |
|---|---|---|
| Paper regimes are large-scale language-model pretraining | **MEASURED by authors:** 8B dense GPT; 3B-active/30B MoE; 8B-active/72B hybrid MoE; 1T/3T-token horizons; 25M-token baseline global batches and experiments up to 100M tokens | Severe regime gap from Pact's approximately `1e5`-`3e5` parameter, n600, single-device MLX witness optimization |
| SOAP rotation preserves Frobenius norm | **DERIVED in paper:** orthonormal left/right rotations preserve `||Delta W||_F` | Update-RMS matching can compare SOAP/Adam-like directions, but says nothing about realized through-R score debt |
| Muon RMS correction depends on momentum damping | **DERIVED under the paper's assumptions:** `sqrt((1-beta1)/(1+beta1))`, approximately `0.2` at `beta1=0.9`; Algorithm 1 separately applies `sqrt(max(in,out))` Kimi scaling | Not a Pact constant: current Muon uses momentum `0.95` by default, Nesterov by default, different runtime/scaling semantics, and final-stage nonstationarity |
| More accurate orthogonalization can matter | **MEASURED by authors:** exact-SVD MOP and SOAP are slightly below Muon loss in the controlled Qwen-3-30B-A3B plot; appendix compares five-step quintic with 16-step PolarExpress | Opens an actual-update measurement, not a global iteration-count change |
| Released tests show PolarExpress approaches the exact polar factor better than five-step quintic on constructed ill-conditioned matrices | **CODE TEST, not Pact measurement:** 10-step comparison to SVD and 16-step near-orthogonal Gram assertions | Useful reference implementation; insufficient for Pact adoption without actual group matrices and wall-clock custody |
| SOAP loss spikes come from a stale preconditioner | **MEASURED by authors:** every-step basis refresh is still insufficient if the current gradient is excluded; current-gradient accumulation plus per-step basis update resolves their spike class; KL-Shampoo adds secondary stability | Distinct from Pact's measured #304/#321 apparatus confounds; do not claim a fix transfer |
| QR and eigendecomposition are similar at every-step refresh | **MEASURED by authors:** negligible loss difference in their controlled comparison; QR chosen as cheaper | A future Pact SOAP probe should test local wall time; no universal QR adoption |
| Layer-wise distribution addresses optimizer memory/comms at scale | **SYSTEMS DESIGN in paper:** full matrices are assigned by layer across data-parallel ranks and asynchronously all-gathered | `N/A-WHY` for current single-device MLX/n600 unless a measured optimizer-state memory bottleneck appears |

## Release-code caveat

The released library defaults are not the paper recipe. Its Muon constructor defaults to momentum
`0.95`, Nesterov off, decoupled weight decay `0.01`, five-step `quintic`, spectral shape scaling,
and `extra_scale_factor=1.0`; the docstring says an extra factor such as `0.2` may be used for RMS
matching. The paper recipe uses momentum `0.9`, no Nesterov, decoupled weight decay `0.1`, 16
PolarExpress iterations, and epsilon `1e-7`. This difference is itself evidence against
paper-title/config cargo culting.

The release's `newton_schulz` requires FP32 input, normalizes by the Frobenius norm, and under
PyTorch `"medium"` matmul precision explicitly converts the iteration to BF16 and back. Its
16-step PolarExpress unit test checks the smaller Gram against identity at approximately `1e-5`
absolute tolerance on random matrices. These are release-code properties, **not measured Pact
MLX properties**.

# Row-by-row regime-transfer ledger

| Axis | Paper/source regime | Pact target regime | Transfer verdict |
|---|---|---|---|
| Objective | smooth next-token cross-entropy | grammar/renderer parameters, quantization/resize `R`, frozen SegNet/PoseNet, exact archive rate | **MECHANISM-ONLY**; score acceptance must remain realized through `R` |
| Parameter count | billions of parameters | approximately `1e5`-`3e5` learned parameters | **FORMULATION OPEN, NUMBERS BLOCKED** |
| Batch | 25M-token baseline; up to 100M tokens | n600 video pairs with small device batches | large-batch noise/critical-batch explanation does not transfer |
| Horizon | 1T/3T tokens | staged witness training / compact grammar fitting | do not transfer schedule or endpoint claims |
| Hardware | large multi-GPU NVIDIA training | current M5/MLX development; contest CPU/CUDA score axes separate | no timing/comms transfer |
| Matrix shapes | large transformer projections and MoE matrices | small renderer/FiLM matrices, grammar variables, mixed scalar/vector/matrix groups | measure group by group; some groups have no valid Muon matrix surface |
| Optimizer semantics | paper recipe and released PyTorch library | MLX built-in Muon plus Pact wrappers; current environment has no importable `mlx` | **`MLX_IMPLEMENTATION_CUSTODY_BLOCKED`** for live-source confirmation on this host |
| Acceptance | training/validation loss and downstream evals | `100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489`, exact bytes/hardware custody | proxy convergence cannot promote a Pact arm |

# Crosswalk to the five requested surfaces

## 1. Muon orthogonalization quality: incumbent finisher, PR95 control, #469, #552, #556

### Current Pact state

- `src/tac/optimization/muon_finisher_mlx.py` routes eligible 2-D hidden weights to MLX's built-in
  Muon and the other leaves to AdamW. Defaults are momentum `0.95`, Nesterov on, and five
  Newton-Schulz steps.
- `experiments/train_levelset_witness_realized_through_R_mlx.py` exposes the final-stage switch and
  logs configuration/counts, but the inspected switch receipt does not log per-group update RMS,
  actual-update polar residual, or exact-polar gap.
- `src/tac/optimization/md_decoupling.py` has a separate five-step repeated-coefficient Muon-like
  surface with shape factor `sqrt(max(1, rows/cols))`; this is not the paper's
  `sqrt(max(rows,cols))` spectral scaling.
- Existing tests prove that a synthetic five-step matrix's singular-value spread collapses. They
  intentionally accept an approximate band, not exact orthogonality. They do not answer whether the
  incumbent's actual final-stage matrices need 5, 8, 10, or 16 iterations.
- #469 is implemented but remains negative-scoped without the required holistic n600 authority.
  #552 is spec-only. #556 is pending and must not be described as live.

### Decision

**KEEP five-step incumbent unchanged. Add a future actual-update audit gate before any step-count or
coefficient change.**

For a rectangular momentum/update matrix `M_g`, define the exact polar direction
`P_g = U_g V_g^T` from an FP32/FP64 SVD reference and choose the smaller Gram:

```text
O_g = NS(M_g)
E_gram(g) =
  ||O_g O_g^T - I||_F / sqrt(rows_g),  rows_g <= cols_g
  ||O_g^T O_g - I||_F / sqrt(cols_g),  otherwise
E_polar(g) = ||O_g - P_g||_F / max(||P_g||_F, tiny)
```

Compare five-step incumbent, candidate PolarExpress iteration counts, and exact SVD on the **same
actual copied updates**. Record `E_gram`, `E_polar`, update RMS, dtype/matmul mode, group shape,
matrix condition summary, and microseconds/update. The chosen formulation must then face a
matched-wall-clock, matched-update-RMS, full-facet A/B. A smaller `E_polar` alone is not an
acceptance metric.

**Effect on #556:** strengthen its existing gate with independent Q/H receipts:

- `Q` direction: tangent-projected direction RMS, retraction displacement, Gram/polar residual;
- `H` direction: metric ID, SPD condition number, positivity margin, direction/update RMS;
- composed `QH`: reconstruction error, realized function displacement, split-resume parity.

One scale, momentum, or Newton-Schulz count may not be silently shared between Q and H.

**PR95 status:** sealed historical control/lesson only. This memo does not reopen it as the live
vehicle, remeasure it, or borrow its settings.

## 2. Update-RMS matching: A/B arms, #195, #366, family-(d)

This is the strongest transferable method, but it must be localized.

For group `g` with `n_g` scalar entries:

```text
rms_g(X) = ||X||_F / sqrt(n_g)
d_ref[t,g]  = reference direction before learning-rate multiplication
d_cand[t,g] = candidate direction before learning-rate multiplication

rho_g = median_t rms_g(d_ref[t,g]) / max(rms_g(d_cand[t,g]), tiny)
eta_cand[g] = eta_ref[g] * rho_g
```

The median is a preregistered robust estimator over a held sequence from copied state. It is **not
a new magic constant**. Weight-decay contribution must be either included in both realized updates
or separated in both. After estimating `rho_g`, reset to the same checkpoint and replay; do not
continue from the calibration trajectory.

### Proposed default-off receipt

Schema name for a future owner: `pact.optimizer_update_scale_receipt.v1`.

Required fields:

- source git/config/checkpoint hashes, optimizer-state schema, seed, stage/event, group path and
  shape;
- reference/candidate optimizer IDs and complete semantics: momentum, Nesterov, coefficient
  schedule/count, dtype/matmul mode, epsilon roles, weight-decay method;
- `raw_direction_rms`, `weight_decay_rms`, `realized_update_rms`, LR, fitted `rho_g`;
- `E_gram`, `E_polar`, exact-reference dtype, and per-update wall time for matrix groups;
- reset/replay digest, resume-state additions, protected facets, and authority axis;
- explicit `score_claim=false` until a byte-closed contest-axis evaluation exists.

### #195

The ledger says the MD arm was stable by construction but under-stepped `d_seg` at scale and must
have its own LR sweep if reactivated. Because no per-group update-RMS receipt was found, that
negative remains **INSTANCE x TESTED SCALE**, not a family death. RMS matching can remove one
comparison confound, but cannot resurrect #195 without a new governed arm and full-facet result.

### #366 / family-(d)

#366 is now a compact grammar-parametrized joint descent through `R` and frozen scorers, not a
generic neural-network weight optimizer. Transfer only the measurement principle:

- define groups over grammar/template/SE(3)/continuous fitting coordinates;
- compare proposal displacement RMS in each coordinate's declared metric;
- record the realized `R`/scorer acceptance and protected facets;
- do not insert SOAP Kronecker factors unless their curvature alignment to the score-quotient
  functional is separately measured.

The paper preconditions smooth matrix gradients of language-model loss. #366's grammar coordinates
include discontinuous realization and heterogeneous geometry. That blocks direct algorithm
transfer.

## 3. SOAP / preconditioned second order vs #366 descent engine

**DO NOT ADOPT SOAP NOW; FAMILY OPEN.**

SOAP is second-order-like in the operational sense of maintaining left/right gradient covariance
factors and optimizing in their evolving eigenbasis. It is not a Hessian oracle, and its paper
benefit lives in a very different scale/noise regime. Its state and per-step QR costs could dominate
Pact's small matrices; conversely, exact local factorizations could be cheap. The direction is
empirical, not inferential.

If a later owner opens a bounded local formulation, the minimum probe is:

1. deterministic NumPy-FP32 formula reference for one matrix group;
2. MLX parity on a real copied gradient sequence;
3. current gradient accumulated **before** the every-step basis update;
4. basis-age, current-gradient-inclusion, factor condition, pre/post RMS, update wall time, and
   memory telemetry;
5. exact split-resume state for both factors, basis, first/second moments, and step;
6. matched-update-RMS incumbent shadow control;
7. only then a default-off n24 negative filter, with n600 required for any positive conclusion.

The released code provides a useful ordering reference: it casts the current gradient to FP32,
updates Kronecker factors with that gradient, updates the eigenbasis/moments, and then projects the
same gradient. Its defaults are not the paper's full KL-SOAP recipe (`use_kl_shampoo=False` by
default), another reason to name the exact formulation.

## 4. Loss-spike elimination vs #304/#321

**DISTINCT MECHANISM CLASSES.**

The paper's SOAP spike chain is:

```text
stale / current-gradient-excluding covariance
  -> lagged eigenbasis
  -> oscillating gradient norms
  -> LM-loss spikes / divergence
```

Pact's measured #304/#321 chain included:

```text
viscous-eikonal unit mismatch
  + shared grad-clip starvation
  + legacy absorbing median freeze
  + inert adaptive epsilon
  + resume-drift stiff-term injection
  + missing liveness stamp
  -> frozen weights mislabeled "converging"
```

Therefore neither SOAP nor per-step QR is a cure for Pact's recorded spike history. The transferable
lesson is observability: any future SOAP arm must expose basis age/current-gradient inclusion and
must inherit Pact's liveness, loss-term domination, skip fraction, resume-drift, and protected-facet
gates. Verdict scope is **FORMULATION x SOAP_STALENESS**, not optimizer family or spike paradigm.

## 5. Layer-wise memory / communications

**`N/A-WHY x CURRENT_SINGLE_DEVICE_MLX`.**

The paper's layer-wise optimizer keeps whole matrices on assigned data-parallel ranks and
asynchronously all-gathers layer updates aligned with forward buckets. Pact's current target is a
single-device small model, so there is no data-parallel optimizer-shard communication to remove.
Do not build this surface until telemetry shows optimizer-state memory or multi-device communication
is load-bearing. If that changes, preserve whole-matrix semantics and compare transfer time against
the actual forward bucket schedule.

# `$0` retrofit specification

This memo specifies but does not build or fire the receipt.

## DSL leg

Add a future typed, default-off `OptimizerUpdateScaleReceipt` treatment with:

```text
enabled = false
mode = shadow_only
reference_optimizer_id
candidate_optimizer_id
calibration_steps
group_metric_ids
exact_polar_reference = true|false
reset_after_calibration = true
score_claim = false
```

Compiler refusals:

- no checkpoint/source/config hash;
- no complete optimizer/resume semantics;
- candidate changes more than the named optimizer treatment;
- missing group metric or heterogeneous groups collapsed into one RMS;
- borrowed numeric scale with no local calibration receipt;
- no reset/replay digest;
- any score/promotion language on shadow measurements.

## DAG leg

```text
custodied checkpoint + held batch/order + source hashes
  -> clone state A/B
  -> incumbent shadow directions
  -> candidate shadow directions
  -> per-group RMS + polar/SVD + wall receipt
  -> fit local rho_g
  -> RESET both states
  -> deterministic replay parity
  -> matched-RMS / matched-budget negative filter
  -> realized-through-R protected-facet gate
  -> governed n600 only after explicit owner/lane/dispatch approval
  -> exact archive/contest-axis evaluation before any score claim
```

## Equation leg

The RMS/polar equations above are the measurement equations. Acceptance remains Pact's action:

```text
S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489
```

with exact axis/custody labels. RMS matching is a nuisance-control constraint, not an objective and
not a promise that the two optimizers take equivalent functional steps.

# Unified-solver wire-in disposition

This is a `research_only=true` advisory; no code or empirical anchor was produced.

| Required hook | Disposition |
|---|---|
| Sensitivity map | **N/A until measurement:** future receipt groups should be keyed to the actual parameter/grammar sensitivity consumer |
| Pareto constraint | **ACTIVE in proposed gate:** no protected Seg/Pose/rate regression; RMS is not sufficient |
| Bit allocator | **N/A-WHY:** optimizer telemetry has no measured byte marginal yet |
| Cathedral/autopilot | **BLOCKED:** no dispatchable treatment until typed DSL, reset/replay, and local receipt exist |
| Continual-learning posterior | **N/A:** no empirical Pact anchor; paper/source facts stay advisory |
| Probe-disambiguator | **SPECIFIED:** compare incumbent five-step vs explicit coefficient/count alternatives and exact SVD on the same matrices |

# No-borrowed-constants ledger

| Source value/form | Status in Pact |
|---|---|
| `sqrt((1-beta1)/(1+beta1)) approximately 0.2` at `beta1=0.9` | **NOT ADOPTED**; local `rho_g` must be measured |
| `sqrt(max(in,out))` scaling | **NOT ADOPTED**; current surfaces differ and runtime MLX source unavailable here |
| 16 PolarExpress iterations | **NOT ADOPTED**; actual-update residual/wall/full-facet gate required |
| paper Muon `beta1=0.9`, `wd=0.1`, `eps=1e-7` | **NOT ADOPTED** |
| KL-SOAP `beta_kron=0.95`, `beta2=0.95`, `p=-1`, `F=1`, `eps=1e-8` | **NOT ADOPTED** |
| no Nesterov in the paper regime | **NOT ADOPTED**; a local same-budget A/B would be required |
| Conv1D stays AdamW in the paper architecture | **MECHANISM-SPECIFIC N/A** for Pact architecture |
| QR instead of eigendecomposition | **NOT ADOPTED**; local timing/parity first |

# Directive-consumption table

| Authority/directive | Consumption |
|---|---|
| Wrapped delegated authority, SHA-256 `a31adb80cab055ea1962daca14e4c6a5af3dc8091fb067e081ccea939a0a1d39` | Governs isolated branch, lane, deliverable, no-launch, review, pointer, and MAIN-review requirements |
| `CLAUDE.md`, `AGENTS.md`, operating manual | Fully read before research; NO-FAKE, resumability, authority axes, triality, serializer, and advisory boundaries applied |
| `SPEC_v75_optimal_single_trunk_20260708.md` / `SPEC_v8_perclass_decomposition_20260708.md` | Incumbent vehicle and resumability contract preserved; no launch or vehicle replacement |
| Current DDM directives `ddm_is1_directive_fivetype_layerstack_20260724.md`, `ddm_is1_directive_upstream_solve_as_oracle_20260724.md`, and `ddm_is1_directive7_score_quotient_functional_family_20260724.md` | #366 treated as grammar/score-quotient fitting consumer, not generic LLM optimizer surface |
| `muon_dig_directive_adamw_optimality_20260713.md` | Optimal-form and fair-comparison discipline applied; no paper settings declared optimal |
| Operator broadcast, 2026-07-19 EV/Fisher directives | No residual/basis action opened; future acceptance remains realized score action, not Euclidean proxy |
| Task-specific inbox | Checked before source work and review; no stop/superseding directive present |

# Adversarial review record

## Round 1 findings and corrections

1. **Release recipe conflation risk:** the initial synthesis treated "released Muon" and the paper's
   Algorithm 1 as one configuration. Corrected by separately recording release defaults and paper
   recipe.
2. **RMS-factor overreach:** an analytic factor could have been presented as a retrofit. Corrected:
   only the measurement procedure transfers; `rho_g` is fit locally from copied-state directions.
3. **Spike-family overreach:** "SOAP removes spikes" could be read as a Pact cure. Corrected with
   explicit causal chains and `FORMULATION x SOAP_STALENESS` scope.
4. **#556 composition risk:** one Muon scale/iteration count could be silently applied to both polar
   factors. Corrected with separate Q/H metric, RMS, retraction, condition, reconstruction, and
   resume receipts.
5. **Layer-wise cargo-cult risk:** distributed layer ownership is irrelevant without data-parallel
   state/communication pressure. Marked `N/A-WHY`.

## Three clean passes

| Pass | Fresh question | Result |
|---|---|---|
| Clean 1 — source re-derivation | Does every paper/code claim above trace to the full PDF or pinned released code, with paper recipe separated from library defaults? | **CLEAN** |
| Clean 2 — regime and custody | Did any paper number become a Pact setting, score, or efficacy claim; did any failed instance kill a family? | **CLEAN** |
| Clean 3 — apparatus and triality | Are DSL/DAG/equations, resumability/reset, directives, inbox, no-launch, pointer honesty, and MAIN review explicit? | **CLEAN** |

# Pointer-delta honesty and MAIN review

- Pointer remains **`0.1910828242 [contest-CPU Linux x86_64]`**.
- `score_claim=false`; this memo has no candidate archive and no CPU/CUDA/MLX score.
- No Torch or MLX tensor probe was run. The host could not import `mlx`; this is recorded as a
  runtime-custody blocker, not worked around.
- Lane remains L0/advisory and `research_only=true`.
- **MAIN landing review is required.** MAIN should review:
  1. that the proposed receipt is telemetry/shadow-only and introduces no default/config change;
  2. that #195's negative stays instance-scoped and #552/#556 remain spec-only/pending;
  3. that the #366 transfer is coordinate-metric/RMS observability only, not SOAP adoption;
  4. that no `0.2`, 16-step, PolarExpress, SOAP, QR, epsilon, momentum, or weight-decay constant is
     promoted without measured local custody.
