# V9 CGauge witness-training sweep specification — 2026-07-14

**Status:** `DESIGN_COMPLETE / DSL_COMPILE_HELD / NO_LAUNCH / $0`
**Lane:** `witness_train_sweep_spec`
**Scientific base:** `v9_cgauge_432` (task #432); the locally named #445 CUDA arm is an execution substrate, not a second typed scientific config in the current tree.
**Pointer:** submit-ready `0.1910828242 [contest-CPU Linux x86_64]` **UNMOVED**; borrowed defensive bank `0.1880443980` is **NON-SUBMISSION** and is not promoted here.
**Goal gap:** submit-ready-to-0.15 = `0.0410828242`; borrowed-bank-to-0.15 = `0.0380443980`.

This is the operator-requested fire list for “generate sweeps and train something,” but this unit only
specifies the campaign. It launches no trainer, scorer, provider, GPU job, evaluator, or archive mutation.
Every eventual treatment is dual-use:

1. an exact-row candidate that must be byte-closed, parsed back, inflated deterministically, and evaluated;
2. a new real trajectory for the costate organ, whose current corpus is one run, nine intervals, and three
   regimes. More trajectories improve predictive regime coverage; they do **not** magically identify causal
   run-level lever effects without matched treatments and controls.

## Stores consulted

- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`, and
  `tac.subagent_contract`.
- v7.5 operating contract §8 and the v8 inherited contract.
- `tools/costate_digest.py --json`, the activation ledger, curriculum-candidate pool, and
  `tools/corpus_query.py` over the named sweep terms.
- `src/tac/witness_dsl/spec_v9_cgauge.py`, `curriculum_dsl.py`, `lever_registry.py`, the current trainer
  parser, and pure compiles of `v9_cgauge_432`, ideal-mod19, and ideal-mod32.
- `experiments/results/v9_cgauge_432_coherent_arm_20260711/{launch.sh,run.log,costate_shadow.jsonl}` and
  the #205 telemetry harvest.
- `next_launch_all_levers_ticket_20260713.md`, `default_off_comprehensive_sweep_20260710.md`,
  `curriculum_candidate_pool_p0_20260710.md`, `hcm_causal_attribution_dig_20260713.md`, and the current
  V9 design/fresh Codex review surfaces.
- Both live inboxes through `2026-07-14T17:00:15Z`; no later directive existed at the last pre-write read.

## Re-derived campaign state

| Surface | Current read | Meaning |
|---|---:|---|
| activation ledger | `82` known, `80` never-fired, `81` owed | the duty queue is real and still mostly undrained |
| ranked duty top three | Taper `78.9%`; Horizon `47.3%`; Step `34.2%` | the percentages normalize against the **borrowed** 0.188044-to-0.15 gap; they rank design EV, not pointer authority |
| curriculum pool | `60` total; `45` owed; `20` built-never-fired | readiness pool, not evidence that every row composes with V9 |
| repository DSL completeness | `301/407` mapped, `106` unmapped, `0` stale | global inventory gap remains; never invent missing routes |
| current V9 pure compile | #432 `378` argv tokens / `14` expected levers | base construction succeeds in the current worktree |
| ideal V9 pure compiles | mod19 and mod32 `417` argv tokens / `20` expected levers each | later typed descendants construct, but are not substituted for the operator-named #432 base in this sweep |
| base memory | `34.258 GiB` governed peak RSS; MLX peak `30.293 GiB` | measured on the stopped #432 run; use `38 GiB` per-run planning floor before governor/system headroom |
| base wall | `80,747.41 s / 288 epochs = 4.673 min/epoch`; separate #205 harvest `5.14 min/epoch` | derived full-run envelope `9.74–10.71 d` for 3000 epochs, before treatment-specific overhead |
| scorer geometry | n600, CPU-torch verdict, batch `32` | batch 8 is invalid for historical comparison; the 2026-07-14 correction is binding |

The stopped #432 run is trajectory evidence, not the new campaign control: its source/provenance state and
partial horizon do not satisfy the clean-tree launch contract. The sweep begins with a fresh, immutable C0
control only after all gates below pass.

## Measurement contract and band convention

Let `B = S_control - S_treatment`; positive `B` is improvement. A “screening band” below is a derived
addressable interval or queue ceiling, **not** a confidence interval and not a score claim. The exact total
`S` on archive bytes decides every row. If only a component ceiling is available, it is labeled `B_seg` and
cannot override Pose/rate regressions.

Every arm freezes seed, pair order, n600 cache/hash, optimizer-step budget, base program, exact scorer/R
geometry, verdict batch 32, checkpoint cadence, EMA semantics, and archive receiver. Structural treatments
start fresh. A suffix fork is permitted only when the typed resume registry explicitly proves that both
branches restore identical model/optimizer/RNG/stage state and differ solely in the declared treatment.

All future runs must emit:

- typed config, compiled argv, expected-active-lever manifest, LawRef/value-provenance receipt, source/tree
  hash, GT/upstream hashes, seed/order, device/axis, and causal treatment manifest;
- per-epoch loss terms plus `costate_shadow.jsonl`, per-class d_seg/part-fraction/flip-mass, d_pose, rate,
  event transitions, accepted-update state, and deterministic component timing;
- atomic periodic checkpoints and distinct complete EMA-shadow checkpoints at every stage boundary;
- terminal EMA/live/Polyak byte-close selection, exact archive bytes/SHA, parse-back proof, deterministic
  inflate proof, `[contest-CPU]` and `[contest-CUDA]` rows kept separate, and a literal pointer decision.

## C0 — mandatory matched control (unranked)

Pseudo-DSL:

```text
C0 := compile_v9_cgauge_432_launch_config(
  num_pairs=600, epochs=3000, seed=0,
  verdict_device="cpu", verdict_batch=32,
  stage_checkpoints=true, periodic_checkpoint_every=25
)
```

No hand-authored flags are allowed. The fresh compile must preserve all 14 expected base levers and the
exclusive config-provenance bijection. C0 itself is both an exact-row candidate and the reference trajectory.
For fresh structural arms, C0 must be re-run in the same admitted campaign window; an old heterogeneous run
is not a causal substitute.

## Ranked treatment fire list

### 1. TAPER-ISO — isolate `DsegAwareTaper` by removing it

- **Treatment:** `C0 - DsegAwareTaper`; control is C0 with the existing
  `DsegAwareTaper(strength=1.0, scale=0.0, floor=0.05)` ON. This is a fresh structural A/B because taper
  changes the input feature amplitudes from epoch 0.
- **Why first:** duty rank `78.9%`; queue estimate `B≈0.030` is `ESTIMATED`. The old +18% negative was
  under-converged and retracted; converged anchors reversed sign. The live #432 run included taper but did
  not isolate it, so it supplies trajectory context, not a causal effect.
- **Screening band:** `B ∈ [0, 0.030]`, derived as null-to-current queue ceiling. This is not an asserted
  lower bound.
- **Falsifier:** at identical seed/order/steps and terminal byte-close, if taper-ON does not have lower exact
  `S` and lower n600 d_seg than taper-OFF, the V9 instance-level benefit is falsified; retain the better exact
  row, even if that is OFF.
- **Organ regime:** margin-saliency spectral allocation across birth, boundary migration, and terminal
  annulus concentration; supplies the missing ON/OFF treatment identity around the existing lane-erosion
  trajectory.
- **Envelope:** `38 GiB` per-run planning floor; `9.74–10.71 d` per fresh 3000-epoch branch. Same compute
  shape as C0.

### 2. HORIZON-ISO — `HorizonWeightedMargin` in its measured reducible band

- **Treatment pseudo-DSL:** append
  `HorizonWeightedMargin(weight=LawRef(HWM_V9_STAGE_SHARE), target=0.5, margin_lo=0.3,
  margin_hi=0.5, row_lo=96, row_hi=288, start_epoch=<typed stage boundary>)` to C0.
- **Weight derivation:** no literal is permitted. At the frozen eligible C0 stage checkpoint, measure the raw
  n600 horizon term `L_h` and the other active loss `L_o`, then resolve
  `w_h = (0.15/0.85) * L_o / max(L_h, eps)` so the treatment occupies exactly the settled 15% single-force
  loss-share cap at that boundary. The weight remains fixed within the stage. If that LawRef/measurement is
  absent, the arm stays HELD.
- **Why second:** duty rank `47.3%`; zero archive bytes; the frozen scorer measured an oracle addressable
  ceiling of `0.012–0.024 S` across the stable GT-margin slices, while the `<0.05` region is label-noise-like
  and intentionally excluded.
- **Screening band:** `B ∈ [0, 0.024]`; `0.024` is the measured oracle ceiling, not promised realization.
- **Falsifier:** exact `B≤0`, or no shift of surviving flips toward higher GT margins within rows 96–288,
  falsifies this V9 treatment formulation. Pose and bytes must not erase any d_seg gain.
- **Organ regime:** reducible confident-horizon versus low-margin noise; adds a targeted Road/Lane boundary
  regime absent from the current three-prototype organ corpus.
- **Envelope:** `38 GiB` planning floor. A full fresh arm inherits `9.74–10.71 d`; a shorter identical-state
  suffix fork may be used only after resume-drift custody is proven. Incremental masked-loss wall is
  `UNKNOWN` until the mandatory timing smoke.

### 3. STEP-ISO — `StepNativeActivation` as a fresh activation-basin arm

- **Treatment pseudo-DSL:** replace C0's activation scientific declaration with
  `StepNativeActivation(beta_start=1.0, beta_end=8.0, anneal="linear", basis="annealed_hosc",
  omega=1.0, finer_bias_init=false)`; never stack FreSh or another initialization treatment.
- **Why third:** duty rank `34.2%`; measured screen moved from `-18.7%` at n100 to `-4.5%` at n600 and the
  queue records `0.013 S` as the owed adoption estimate. A fixed beta is forbidden saturation death.
- **Screening band:** `B ∈ [0, 0.013]`, null-to-queue ceiling.
- **Falsifier:** exact `B≤0`, no reduction in ring/edge survival error, or a saturation/dead-gradient trace
  falsifies this registered V9 formulation at the matched horizon; it does not kill step-native families.
- **Organ regime:** continuous-to-step activation stiffness, topology birth survival, and beta-rung response;
  this is a new control geometry rather than another sample of the current smooth-HOSC regime.
- **Envelope:** `38 GiB`; `9.74–10.71 d`. The operator set is unchanged, so no whole-epoch speedup or penalty
  is asserted.

### 4. AA-SUPER2 — replace IPE with exact 2x supersample coverage

- **Treatment pseudo-DSL:** replace C0's `render_aa="ipe"` declaration with
  `AACoverageRender(mode="supersample", ss=2, grid_h=384, grid_w=512)`; fresh start required.
- **Why fourth:** the authority probe measured oracle-R floor `0.00091` versus `0.00247` and Lane recall
  `0.56→0.94` (`+0.38`). This directly attacks alias-erased thin Lane structure and has high scientific
  blast radius despite compute cost.
- **Screening band:** current Lane fraction is `690,639/117,964,800=0.00585462`; therefore the measured
  recall lift gives an optimistic `B_seg ∈ [0, 100*0.00585462*0.38] = [0, 0.2225]`. This is a Lane-only
  oracle ceiling, not a total-score forecast.
- **Falsifier:** no positive n600 Lane-recall/d_seg movement, exact `B≤0`, any receiver mismatch, or decode
  time beyond 30 minutes. A supersample loss does not falsify IPE or the AA family.
- **Organ regime:** observation-operator aliasing versus coverage integration; adds subpixel Lane-birth and
  annulus-survival dynamics.
- **Envelope:** base `34.258 GiB` and `9.74–10.71 d` are lower bounds. The witness-forward portion scales
  as `ss²=4`, but the whole-epoch component split is unmeasured, so a numeric whole-run multiplier would be
  fake. Require a governed real-shape timing/RSS smoke before GO; this arm may be refused on memory/decode.

### 5. ETF-HEAD — byte-free rare-class `HeadGeometry(head="etf")`

- **Treatment pseudo-DSL:** append `HeadGeometry(head="etf", additive_margin=0.0)` to C0; do not compose
  the additive-margin sibling in this isolation arm.
- **Why fifth:** one of the 20 built-never-fired pool rows; it targets the neural-collapse minority-norm
  deficit for Lane/Movable without trainable head bytes. At the current V9 epoch-275 surface, the measured
  Lane+Movable score-term debt is `0.13694+0.00259=0.13954`.
- **Screening band:** `B_seg ∈ [0, 0.1395]`, null-to-current rare-class-debt ceiling. It does not assume the
  ETF recovers all of that debt.
- **Falsifier:** rare-class d_seg does not improve, total exact `B≤0`, or the supposed rate win does not
  survive archive parse-back. The verdict is scoped to frozen simplex ETF on this head.
- **Organ regime:** minority-class norm/head geometry, complementing the existing trajectory's data- and
  schedule-driven lane erosion.
- **Envelope:** `38 GiB`; `9.74–10.71 d`. Same head width and no added trainable payload.

### 6. POLAR-FINISH — function-preserving FiLM polar MCSD/SPEL finisher

- **Treatment pseudo-DSL:** at the governed Muon finishing boundary, append
  `FilmPolarChartSPELManifoldMuon(start_epoch=<the typed Muon boundary>)`; preserve `W=QH0`, all Q/H0,
  tangent momentum, Q-EMA, optimizer, RNG, and rollback state.
- **Why sixth:** built-never-fired and registry-resumable; NumPy/MLX parity is measured, and it attacks the
  terminal optimizer geometry without replaying a naive unit-Stiefel projection that changes the function.
- **Screening band:** no measured V9 treatment-effect prior exists. The only honest numeric bound is
  `B_seg ∈ [0, 100*(0.0409147-0.005318)] = [0, 3.5597]`, the current smooth-label excess ceiling; this broad
  interval is why the row ranks below levers with actual effect anchors.
- **Falsifier:** the matched finishing suffix fails to improve exact `S`, violates polar/tangent invariants,
  or loses split-resume determinism. Scope is this MCSD/SPEL fallback, not manifold optimization generally.
- **Organ regime:** terminal tangent geometry, finisher conditioning, and accepted/rolled-back step response;
  it adds a late-stage regime absent from the current ep≤275 corpus.
- **Envelope:** `38 GiB`; wall is the identical C0 prefix plus an `UNKNOWN` finisher delta. The owner must
  land a real-checkpoint timing smoke before projecting a total.

### 7. HORIZON×STEP — conditional non-additivity arm

- **Treatment pseudo-DSL:** compose the exact, already-measured HORIZON-ISO and STEP-ISO declarations on a
  fresh C0 lineage. Fire **only if both isolated arms have `B>0` and pass their mechanism guards**.
- **Why seventh:** the two levers act on different objects—reducible horizon support and activation chart—so
  the interaction tests whether the gains compose or collide. It is deliberately sequenced after isolation,
  not used to hide attribution.
- **Screening band:** `B ∈ [0, 0.024+0.013] = [0, 0.037]`; the additive upper bound is derived from the two
  source ceilings, while overlap can reduce it to zero.
- **Falsifier:** it must beat the **better single arm**, not merely C0. Otherwise there is no positive
  interaction and the campaign retains the best isolated exact row.
- **Organ regime:** cross-regime coupling between boundary support and activation stiffness; supplies the
  first direct non-additivity trajectory once the single-arm effects are known.
- **Envelope:** `38 GiB`; `9.74–10.71 d` plus the HWM loss overhead measured by its smoke.

## Dual-purpose organ map

| Run | Exact-row role | New organ information |
|---|---|---|
| C0 | fresh #432 candidate/control | clean reference trajectory with current telemetry schema |
| TAPER-ISO | OFF candidate and causal contrast | spectral saliency allocation ON/OFF |
| HORIZON-ISO | zero-byte support-loss candidate | reducible horizon versus noise regime |
| STEP-ISO | activation-basin candidate | beta/stiffness/topology response |
| AA-SUPER2 | observation-operator candidate | aliasing/coverage and subpixel Lane survival |
| ETF-HEAD | byte-free head candidate | minority-norm/head-geometry response |
| POLAR-FINISH | terminal optimizer candidate | late-stage tangent/rollback regime |
| HORIZON×STEP | conditional stack candidate | measured non-additivity between two passing levers |

The ingestion unit is one immutable run record with treatment identity and chronological intervals. Do not
inflate the sample count by calling 600 correlated pairs or many epochs independent runs. The organ must
report `n_runs`, `n_intervals`, regimes, and walk-forward performance separately; persistence remains the
baseline until multi-run held-out evidence beats it.

## Unmapped, incompatible, or intentionally deferred surfaces

1. **Protected V9 provenance is doing its job.** Directly appending each proposed lever to the current #432/
   ideal typed object was read-only tested and refused. HWM lacks seven scientific LawRefs; Step's beta-end
   `8.0` conflicts with the sealed `3.177`; AA lacks the supersample scientific binding; ETF lacks head
   bindings; Film lacks its finisher binding. TAPER-ISO needs a reviewed expected-lever-manifest removal.
   The exclusive provenance owner must create each distinct typed variant on the clean tree; raw extra flags
   are forbidden.
2. **Global completeness remains open:** `106` real trainer flags are not referenced by the repository DSL,
   including relevant `--aa-self-orient-fine-*`, `--basis-family`, `--config-provenance-required`, several
   event companions, resume controls, and timing surfaces. None is smuggled into this spec.
3. **#445 mapping:** current local evidence names #445 as the CUDA training arm, but no canonical typed
   `#445` scientific config object was found. This spec therefore keeps #432 as the scientific base and treats
   #445 only as a future execution-axis ticket until the canonical pointer says otherwise.
4. **HardnessOversample is not in the fire list.** Although its factory is real and the pool says
   built-never-fired, the later V9 ticket records that enlarged weighted order is truncated to the original P
   consumed visits. It needs the repair and a fixed-oversample weighted-vs-uniform receipt before firing.
5. **MarginCompandedGroundChart is held.** It is a real factory with measured proxy geometry, but the current
   implementation is structurally incompatible with the V9 IPE path and still owes a counted receiver-close
   A/B. Do not make AA removal an unacknowledged co-treatment.
6. **D18 k90 truncation and mod19/mod32 remain terminal byte-close A/Bs**, not separate training trajectories.
   Apply them to every terminal candidate where their receiver contract is valid; do not count them as organ
   runs.

## Launch gates — current status, do not bypass

| Gate | Status | Required transition |
|---|---|---|
| `governor_measured_growth_fix` | **BLOCKED / three-pass sealed but not landed**; its canonical serializer reached `git add` and failed `rc128` (`Operation not permitted`) | land the reviewed measured-growth admission fix; then re-run real system admission |
| shared tree disentanglement | **BLOCKED** | serialize/review the live-arm files into a reproducible main-tree state; contested DSL/equation sources must have one owner |
| per-variant V9 provenance compile | **BLOCKED** | reviewed typed variants, complete LawRefs, zero missing/duplicate/stale owners, exact active-lever manifests |
| storage waterfall | **BLOCKED in latest ticket** | authorized SSD workload root and ≥`1,026,048,000 B` certified reservation; no local-disk fallback |
| real-shape timing/RSS | **OWED** | governed non-score smoke per treatment, especially AA and polar finish; no guessed multiplier |
| deterministic/resume seal | **OWED** | full resume registry, atomic periodic + all stage checkpoints, EMA shadow, split-resume equality |
| operator containment | **NO GO RECEIVED** | explicit operator GO for each launch after all prior gates pass |
| exact evaluation | **NOT RUN** | byte-close and separate exact contest-CPU/CUDA custody on the exact archive |

A governor refusal is information. A dirty-tree or provenance refusal is also information. Neither may be
worked around with raw Python, invented flags, launch.sh edits, `--resume-allow-lever-drift`, or a provider
dispatch outside `tools/launch_witness_run.py` plus the canonical lane claim.

## Stop and selection rules

- Rank eventual candidates by exact total `S`, never proxy loss and never d_seg alone.
- Keep all per-class facets visible: Road/Lane/Undrivable/Movable/MyCar d_seg, island birth/area, d_pose
  versus need, bytes, runtime, and receiver survival.
- A failed arm gets the narrowest supported `INSTANCE`/`FORMULATION` verdict and a named reactivation
  condition; it does not kill a family.
- Preserve the exact best archive bytes. The first submittable exact row below `0.1910828242` moves the
  submit-ready pointer; the borrowed `0.1880443980` bank remains separately labeled.

## Triality and pointer delta

- **DSL:** this memo specifies pseudo-DSL only. Actual factories already exist, but each V9 treatment needs
  exclusive-owner provenance/LawRef compilation on a clean tree.
- **DAG:** `.omx/research/witness_train_sweep_spec_DAG_FEED_20260714.md`.
- **Equations:** existing taper, horizon-margin, step-native, AA, head/geometry, and polar laws are consumed.
  No contested equation file was edited and no unmeasured effect was registered as a law.
- **Pointer delta:** exactly zero. This unit is $0 design and launch containment only.
