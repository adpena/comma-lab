# Campaign meta adversarial review: v9c2 -> v9c3 -> v10

Date: 2026-07-18  
Lane: `lane_campaign_meta_review_v9c2_to_v10_20260718`  
Mode: `research_only=true`; advisory; no launch; no score; no pointer move  
Audit cutoff: v9c2 telemetry through the epoch-950 checkpoint and the last n600 verdict at epoch 925; canonical `main` through phase-probe merge `7c90b25577`; registered v10 branch SPEC through `49ecbb2538`  
Sacred source: `experiments/results/levelset_n600_witness_20260717T113932Z/` remained read-only

## Verdict first

**[DERIVED] v9c3 pre-launch verdict: `REVISE`.** Do not launch the currently implied
"ep725 + corrected events + #270" restart. After the revisions in this memo, **PROCEED only as a
bounded, pre-registered signal-harvest campaign whose success criterion is information and v10
readiness, not a lower score or a claim of optimality.** This applies the operator's 2026-07-18
reframe literally.

**[MEASURED] The single highest-value campaign finding is that the banked ep725 source is an EMA
deploy snapshot, not an exact pre-Muon resumable state.** The defensive bank has no optimizer/RNG/
event-history sidecar. The state-bearing ep726 sidecar exists only in the sacred run and is already
post-switch. Therefore a v9c3 "resume" from ep725 is a new optimizer/EMA/RNG **fork**, while an
absolute Muon@726 would fire immediately. In that immediate-fire form, warm-Muon has no conditioned
outgoing AdamW moment to transfer and cannot test the intended mechanism.

**[MEASURED, custody-limited] A second decisive update and operator correction landed during this
review:** on an n24 sampled, fp32-EMA, camera-uint8/frozen-CPU-SegNet advisory probe, an ad-hoc
per-pixel-independent min-norm Road->Lane intervention produced relative `Delta d_seg = +56.6%` to
`+161.1%`. **[MEASURED]** Code reinspection confirms `base=-(m/gn2)*grad` is composed independently
per selected pixel. It is neither #424 conditioning in the training loop nor the real #425 coherent
per-pair/per-dash codec. `verdict_scope=DIAGNOSIS + STRAWMAN FORMULATION + MEASUREMENT-SAMPLE`:
valid findings are that 96.7% of sampled Road->Lane flips lie on the phase band and the stratum is
amplitude-open; every phase-efficacy inference is **RETRACTED**. Both real modes remain untested.
The cited raw receipts are absent, the probe is n24 rather than the §14.5 n600 gate, and it never
exercises the actual #425 carrier receiver; therefore §14.5 remains open.

**[DERIVED] v10 verdict: `NOT_LAUNCH_READY`.** The direction remains open, but #528-#532, explicit
birth-before-phase enforcement, phase efficacy in the actual train-side/receiver formulation,
state persistence, counted seed custody, and exact byte-close/parse-back remain launch blockers.
If the new §14.7 Morse-Smale/FiLM representation is selected, its topology and collateral-locality
receipt is an additional conditional admission gate, not an already-proven property. §14.8's
cell-complex/xi/seed framing is likewise a representation-sufficiency hypothesis, not an evaluator
identity.

**[MEASURED] Recursive-review state: `3/3 CLEAN` after the last reset.** This seals the confound
inventory at the stated cutoff only. It is not a v9c3 config seal and not a v10 launch seal.

## Claim labels and verdict scope

- **[MEASURED]** means directly re-derived from bytes, code, committed primary artifacts, or run
  telemetry in this pass. A reported measurement whose raw receipt is missing is explicitly marked
  custody-limited.
- **[DERIVED]** means arithmetic or a necessary consequence of measured/code facts.
- **[INFERRED]** means the evidence supports the explanation but the causal isolation is incomplete.
- **[ASSUMED]** means an unverified input; it cannot authorize a launch.
- Negative verdicts below state their scope. No instance/formulation finding is promoted to a family
  or paradigm verdict.

## Stores consulted and trust boundary

**[MEASURED] STORES CONSULTED:**

1. `tools/graph_memory_recall.py` queries for resume-event geometry, cold-Muon, spike median freeze,
   v9c2, phase-stack efficacy, and v10 capstone; graph recall was used as an index, never as result
   authority.
2. `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md` and the historical confound
   memos it routes, especially `confound_hunt_synthesis_20260705.md` and the warm-start/resume law.
3. `reports/latest.md:61`, `.omx/state/active_lane_dispatch_claims.md`, lane/task registries, and the
   operator inbox directives at `2026-07-18T03:20:58Z`, `2026-07-18T03:37:35Z`, and the corrective
   directive at `2026-07-18T03:57:13Z`.
4. Sacred v9c2 `launch.sh`, `run.log`, `constants_manifest.json`, checkpoint manifests/NPZ schemas,
   and the defensive bank; donor mod32cap telemetry was read as a non-randomized comparison.
5. Source and tests for the trainer, `curriculum_dsl.py`, `typed_config.py`, `gauge.py`, v9 compiler,
   spike/liveness monitoring, and phase probe.
6. `claude/p0_518_resume_warmup_geometry_20260717` and
   `claude/p0_521_spec_v10_capstone_20260717` through `49ecbb2538`, read by Git object without
   modifying either branch.
7. The main-merged phase-probe memo/tool/tests at `f2ca1bdd67`, `f344285c8a`, and merge
   `7c90b25577`.
8. The main-merged checkpoint-maturity apparatus at `d3610ffc3a`/`5b2994cbf4` and the
   factorized realization-regime rows already present at this branch base; the naming-independent
   canonical-doc registry at `8866186378`/`3cdf094475` was also checked before writing this memo.

**[MEASURED] Re-derived rather than trusted:** pointer identity, bank file inventory and SHA-256s,
checkpoint roles/schemas, epoch schedules, telemetry trajectories, live/EMA gaps, liveness fields,
Muon switch path, typed-DSL composition limits, v10 compiler behavior, score arithmetic, and the
absence of the cited phase raw receipts in the canonical/local SSD paths checked.

**[ASSUMED] Taken as directive rather than empirical fact:** v9c3 is intentionally a signal/v10-
readiness vehicle. **[MEASURED, custody-limited] Taken from a committed report rather than
independently replayed:** the n24 phase-probe numeric deltas, because its cited raw JSON/log/ledger
rows are absent. **[MEASURED] Re-derived from code and the operator correction:** the intervention's
per-pixel-independent identity and its non-equivalence to #424/#425.

## Canonical state and exact-score firewall

- **[MEASURED]** The submittable pointer is `[contest-CPU Linux x86_64] 0.1910828242`, archive SHA
  prefix `ad02b0124cbb`, lane `lane_clickpolish_pr110_frontier_20260710`
  (`reports/latest.md:61`).
- **[MEASURED]** The `0.18804439798807521` PR128 splice is borrowed/non-submission custody and is not
  a promotion target.
- **[MEASURED]** The v9c2 bank's ep725 EMA d_seg is `0.003457972208659`.
- **[DERIVED]** Its Seg term alone is `100*d_seg = 0.3457972208659`, exceeding the exact pointer by
  `0.1547143966659` even under impossible zero pose and zero bytes.
- **[DERIVED]** A contender needs `d_seg < 0.001910828242` even with free pose/rate. v9c3 recovery
  to ep725 therefore cannot itself be the fastest lower-exact-row proof.
- **[DERIVED]** This arithmetic does not make v9c3 useless: under the operator criterion, its value
  is causal information per wall-clock and de-risking of v10.
- **[MEASURED]** New vehicle-shaped checkpoint names without an explicit `_prod` token are
  non-promotable at the canonical pointer gate. v9c3 experiments must be named `_dev`; v10 remains
  `_dev` until a separately reviewed, immutable `_prod` bank is created.
- **[DERIVED]** An `_prod` token clears only the maturity-name gate. It does not prove operator GO,
  exact bytes, custody, CPU/CUDA parity, or any other promotion gate.

## Apparatus validity and positive control

| Surface | Finding | Authority and scope |
|---|---|---|
| Sacred-run custody | **[MEASURED]** launch/manifests and bank hashes are present; the run directory was only read. | Artifact custody; no score authority claimed. |
| Optimization liveness | **[MEASURED]** all 300 epoch loss rows from ep651-950 have `spike_skipped:false`, `accepted_frac:1`, and `weights_stepped:true`; no `frozen_epoch:true` or run confound alarm was found. | Clears the legacy median-freeze failure for this run instance. |
| n600 verdicts | **[MEASURED]** all 11 post-baseline CPU verdicts are 75/75 batches and not frozen, with EMA-primary `d_seg` plus paired live-weight gap readouts; `--verdict-pairs 0` is explicit. | Advisory witness-training verdict axis, not contest archive score. |
| EMA-vs-live | **[MEASURED]** live-gap telemetry is on every 25 epochs, so EMA masking is visible rather than silent. | Readout is trusted for gap sign/magnitude. |
| Static d_seg canary | **[MEASURED]** the trainer runs a synthetic known-effect suite once and copies the result into later verdict rows. It passed even though resume `baseline_v0` was corrupted. | Positive control for scorer/liveness/trend plumbing only; it does not validate schedule positioning, checkpoint identity, or lever causality. |
| Phase probe | **[MEASURED]** camera uint8 rounding and frozen CPU SegNet are exercised, but only n24/fp32 EMA under an ad-hoc per-pixel-independent intervention; the memo reports overlapping deterministic writers after a phantom-death relaunch, while raw receipts are absent. | Valid for the phase-band/amplitude-open diagnosis and this strawman's harm only; not a #424/#425 efficacy assay. Duplicate-writer identity is not independently checkable; §14.5 n600, carrier int8/dequant, archive parse-back, and #425 receiver remain untested. |
| Checkpoint maturity | **[MEASURED]** main's `_dev`/`_prod` parser and pointer gate refuse `_dev` and untagged vehicle names. The prod-immutability helper refuses an existing `_prod` directory when called, but no production bank writer calls it yet. | Clears silent dev-to-pointer promotion on current main; bank immutability remains helper-only. |

**[DERIVED] Apparatus verdict:** sufficient to support the scoped v9c2 trajectory, phase diagnosis,
and ad-hoc-intervention warning; insufficient to support any phase efficacy verdict, resume
equivalence, per-lever efficacy, v10 byte closure, or a score.

## Primary v9c2 trajectory, re-derived

| Epoch | EMA d_seg | live d_seg | Lane d_seg | Interpretation |
|---:|---:|---:|---:|---|
| 675 | 0.003633 | 0.003876 | — | **[MEASURED]** fresh-AdamW fork is training. |
| 700 | 0.003470 | 0.003550 | — | **[MEASURED]** subpixel + phase + satisfice engage together; attribution becomes stacked. |
| 725 | 0.0034579722 | 0.003719 | 0.21319242 | **[MEASURED]** defensive-bank minimum immediately before the absolute Muon switch. |
| 750 | 0.004294 | 0.005311 | 0.23789418 | **[MEASURED]** EMA +24.18%, live +42.81%, Lane +11.59% after cold Muon. |
| 900 | 0.003891 | 0.004782 | — | **[MEASURED]** partial recovery, still above ep725. |
| 925 | 0.003847 | 0.004693 | — | **[MEASURED]** EMA is 18.03% below live and 11.25% above ep725; live is 26.2% above ep725. |

**[MEASURED]** The pose conditioning gate is `DEGENERATE_GUARD_TRIPPED` from the transition region
through ep950 and never fires. **[MEASURED]** Movable initializes at zero, but ordinary training
already reduces its class error to roughly 0.034 by ep675; "unborn-by-design" is true of the
configured initialization, not the observed endpoint.

**[MEASURED]** The factorized realization observer further reports ep900 per-class sub-LSB flip mass
`Lane=0.281` (`MIXED`) and `Movable=0.533` (`realization-limited`) on the advisory axis. This supports
a Movable representation/realization blocker, but does not by itself identify island-birth time or
prove an endpoint is literally unborn.

**[MEASURED]** Donor mod32cap also degrades across its cold flat-LR Muon boundary: ep725
`0.003414` -> ep750 `0.004351` (+27.45%), close to v9c2's +24.18%.
**[INFERRED]** Cold Muon is the primary shared discontinuity and Force3 alone is not established as
the cause. `verdict_scope=INSTANCE + NONRANDOMIZED-COMPARISON`; the Muon family remains open, and the
phase/satisfice/subpixel stack remains unisolated.

## Campaign confound-reduction ledger

Status is exactly one of `fixed`, `carried`, `new-from-fix`, or `unaddressed`. Signature is
`D/S/M = default-harmful / silent / measurement-corrupting` at campaign scale; `Y`, `N`, or `latent`.

| ID | Confound or load-bearing assumption | Bit vehicle / evidence | v9c3 treatment | v10 disposition | Status | D/S/M | Negative verdict scope |
|---|---|---|---|---|---|---|---|
| C01 | **[MEASURED]** accepted-only spike median can freeze all updates while telemetry continues | Pre-v9 synthesis; current ep651-950 rows all step | Keep rollback mode and liveness assertions | Compiler must preserve fail-loud liveness | `fixed` | Y/Y/Y | Historical implementation; current instance cleared |
| C02 | **[MEASURED]** adaptive epsilon can be inert | Pre-v9; v9c2 uses eikonal weight 0 | Hold excluded | Prove exclusion or real adaptive law in emitted argv | `carried` | Y/Y/Y | Dormant config family, not eikonal paradigm |
| C03 | **[MEASURED]** eikonal unit/normalization mismatch | Pre-v9; normalized code exists, v9c2 weight 0 | Hold excluded | Compiler/manifest proof owed | `carried` | Y/Y/Y | Dormant configuration |
| C04 | **[MEASURED]** shared global clipping lets one group hijack all groups | Pre-v9; v9c2 emits per-group clipping | Preserve per-group | Must compile and test | `carried` | Y/Y/Y | Config portability; v9c2 instance fixed |
| C05 | **[MEASURED]** frozen closed-loop controller can look active | Pre-v9; current liveness/controller rows are live | Preserve liveness stamps | Require same self-protection | `fixed` | Y/Y/Y | Historical implementation; current instance cleared |
| C06 | **[MEASURED]** static positive canary is blind to run-specific schedule corruption | v9c2 `baseline_v0=0.208537` while static canary passes | Add fork-baseline reproduction gate | Cold v10 still needs per-stage known-effect controls | `new-from-fix` | Y/Y/Y | Apparatus formulation |
| C07 | **[MEASURED]** one-shot clear/reset guard can silently refuse or double-clear | Pre-v9; current guard logs/fails loud | Preserve and assert one boundary | Persist/register every boundary | `fixed` | Y/Y/Y | Historical implementation; current guard is loud |
| C08 | **[MEASURED]** stale optimizer moments were avoided by weights-only restart, but replaced by a fresh-state transient | v9c2 fresh AdamW at ep651 | Name as fork; recondition before treatment | Cold v10 eliminates resume leg | `new-from-fix` | Y/N/Y | Fork configuration, not optimizer family |
| C09 | **[MEASURED]** EMA can flatter worsening live weights | ep925 EMA 0.003847 vs live 0.004693 | Mandatory live+EMA terminal readouts and clearance law | Fresh stage-specific EMA law still owed | `carried` | Y/N/Y | Readout/config; current live-gap rows make it loud; not EMA family |
| C10 | **[MEASURED]** palette/structured init flags can be overwritten yet reported configured | v9c2 now logs `applied:false`; v10 #532 structured init semantics remain weak | No init claim on ep725 fork | Real `InitHeadSolve`, per-class detectors, receiver survival | `carried` | Y/N/Y | Initialization formulation; historical implementation was silent, current v9c2 is loud |
| C11 | **[MEASURED]** resume baseline evaluated at the wrong tau/beta/form position | v9c2 0.208537 phantom baseline | #517 positioning must land and reproduce bank d_seg | Cold v10 avoids resume, but compiler tests positioning laws | `carried` | Y/Y/Y | Resume implementation |
| C12 | **[MEASURED]** ramp/event state is incompletely persisted | #518 residuals: `last_boundary_epoch`, `engaged_epoch`, detector history/mode | Block launch until all state is sidecar-persisted | Same P0 persistence gate | `carried` | Y/Y/Y | Persistence implementation |
| C13 | **[MEASURED]** 24-pair default can masquerade as n600 | Historical; v9c2 explicitly uses 0/all | Keep explicit n600 verdicts | #529 must emit and parse exact setting | `fixed` | Y/Y/Y | Historical config; current v9c2 instance fixed, v10 proof owed |
| C14 | **[MEASURED]** duplicate/last-wins argv flags can falsify the intended arm | Historical; typed v9c2 launch is clean | Dedicated exact expected-lever manifest | Real compiler/parser round-trip | `fixed` | Y/Y/Y | Historical compiler; current v9c2 launch is clean |
| C15 | **[MEASURED]** eikonal anneal/re-entry can destabilize a later stage | Prior vehicle; excluded in v9c2 | Do not reactivate | Explicit v10 decision, not inheritance | `carried` | Y/Y/Y | Dormant schedule |
| C16 | **[MEASURED]** periodic reorientation can be logged without semantically re-arming a new boundary | Prior campaign; #518 forces ramp-end reorient but persistence remains | Assert event-relative boundary and exactly-one reorient | Cold stage registration test | `carried` | Y/latent/Y | Schedule implementation |
| C17 | **[MEASURED]** absolute seed/persistence anneals after a fork can execute at the wrong relative time | Prior resume class; v9c2 absolute epochs | Re-anchor every fork-relative lever | Cold v10 should use cold-relative stages only | `carried` | Y/Y/Y | Fork schedule |
| C18 | **[MEASURED]** sibling/base-trainer guards can diverge from the levelset entry point | Historical spread class; stale gauge comments still name base-only flags although levelset flags now exist | Compile against real levelset parser | #529 must prove one emitted/parsed surface | `carried` | Y/Y/Y | Integration/portability |
| C19 | **[MEASURED]** full-stack result cannot be attributed to individual levers | v9/C0 and v9c2 stack | Fixed-factor factorial arms only | Cold v10 admits stages only after isolated evidence | `carried` | Y/N/Y | Experimental design |
| C20 | **[MEASURED]** subpixel, phase, satisfice, and moment reset all occur at ep700 | v9c2 `run.log`/launch | ep725 fork can test late treatment response, not original efficacy | Enforce island-birth before phase and stagger stages | `carried` | Y/N/Y | Vehicle instance/order |
| C21 | **[MEASURED]** absolute Muon@726 fires 75 epochs after resume and overrides a held/stale event sensor | v9c2 fire-via-cap row | Re-anchor after a measured AdamW recondition window; forbid absolute 726 | Cold event-relative boundary | `unaddressed` | Y/N/Y | v9c2/v9c3 config |
| C22 | **[MEASURED]** cold momentum and flat LR are changed together at a harmful boundary | v9c2 and donor comparison | Independent 2x2 momentum x LR arms | Factor effects before adopting in v10 | `unaddressed` | Y/N/Y | Schedule formulation; Muon family open |
| C23 | **[MEASURED]** defensive bank preserves deploy EMA/stage EMA but omits the full state sidecar | Bank inventory/hashes | Preserve/hash ep726 full sidecar; name ep725 arms forks | Cold v10 not affected, but custody law generalizes | `new-from-fix` | Y/Y/Y | Checkpoint custody |
| C24 | **[MEASURED]** named `MuonWarmStart` bundles momentum and LR; independent gauges are descriptive, not a named v9c3 compiler | DSL/code audit | Use two independent `TypedLever`s and a dedicated manifest | Compiler must expose factor identity | `unaddressed` | Y/Y/Y | Config/attribution implementation |
| C25 | **[MEASURED]** Movable has no applied initial seed; `--structured-init` was configured but `applied:false`, and the pretrain partition had zero Movable mass | v9c2 init row + class verdicts; later nonzero learning is not birth telemetry | Measure nucleus count/area/birth epoch; no endpoint overclaim | Cold explicit birth gate before phase | `carried` | Y/Y/Y | Initialization/observability |
| C26 | **[MEASURED]** pose gate remains degenerate and fresh detector state can be confused with recovered state | v9c2 through ep950 | Rebase/persist detector, ramp w_pose, label fresh state | Exact score-domain pose composition | `carried` | Y/Y/Y | Gate implementation; pose family open |
| C27 | **[DERIVED]** an ep725 fork inherits phase-conditioned weights and cannot test birth-before-phase | schedule chronology | Use only for late train-side recovery; separate source assay | Enforce cold birth-before-phase structurally | `carried` | Y/N/Y | Vehicle ordering |
| C28 | **[MEASURED, custody-limited]** the ad-hoc per-pixel-independent min-norm intervention produces collateral Lane->Road overshoot | n24 phase report plus code identity at `base=-(m/gn2)*grad` | Use only as a warning about this strawman; test the actual #424 and coherent #425 modes | No phase-efficacy inference; §14.2/#425 remain open | `new-from-fix` | Y/N/Y | Strawman per-pixel-independent formulation only |
| C29 | **[MEASURED]** phase report lacks raw receipt/hash; its memo reports two briefly concurrent writers after a phantom-death relaunch, and the tool has no output lock/merge/fingerprint | committed memo/tool + path search; raw writer identity is not independently checkable | Preserve raw outputs and lock/fingerprint resume state | Raw custody is required before the result can be independently replayed | `unaddressed` | Y/Y/Y | Provenance/duplicate-writer custody |
| C29b | **[MEASURED]** the phase result is n24/fp32-EMA/ad-hoc-independent and does not exercise carrier int8/dequant, archive parse-back, or the #425 receiver | These sample/axis/formulation limits are disclosed in the committed memo; the intervention-identity correction is operator-supplied and code-confirmed | Run matched actual-mode assays; do not promote the strawman result | §14.5 n600 byte-close gate remains | `carried` | Y/N/Y | Disclosed measurement/formulation limits; not silent |
| C30 | **[MEASURED]** #528 squares the intended pose marginal under score-domain loss | v10 spec audit | Not a v9c3 lever | Set score-domain `w_pose=1`; test emitted objective | `unaddressed` | Y/Y/Y | v10 objective implementation |
| C31 | **[MEASURED]** #529 has no successful emitted argv/manifest/parse path; `spec_v10_status().clear` omits `post_gate_fold_owed` and can falsely report readiness, although `compile_v10_capstone_launch_config()` itself always raises | v10 branch source | v9c3 needs its own real compiler and complete status too | Build compiler success path; include the fold in status; keep compiler fail-closed | `unaddressed` | Y/latent/Y | v10 readiness-status/compiler implementation |
| C32 | **[MEASURED]** #530 cold v10 references fork-only head/EMA laws | v10 spec/compiler | Keep fork laws v9c3-only | Split `InitHeadSolve` from `ForkHeadSolve/ForkEmaClearance` | `unaddressed` | Y/Y/Y | Vehicle composition |
| C33 | **[MEASURED]** #531 has no explicit class/cell-conditioned quotient residual `T` | flat/textured error gap 0.0416 vs 0.0048 | v9c3 may localize residuals but cannot close design | Give `T` unique custody and train only quotient residual | `unaddressed` | Y/Y/Y | Representation formulation; factorized paradigm open |
| C34 | **[MEASURED]** #532 exact range(A) over reals fails after uint8 clip/round; max discrepancy 62.74 | v10 branch probes | Not a v9c3 claim | Constrain through realized R and parse-back; fix init detectors | `unaddressed` | Y/Y/Y | Projection/receiver formulation |
| C35 | **[MEASURED]** phase event stream is 29,958 B: 16.6x the 1.8 KB upper estimate and 33.3x the 0.9 KB lower estimate | v10 spec §13/14 | Measure only decision-relevant phase forms | Actual entropy/receiver budget is a gate | `unaddressed` | Y/N/Y | Rate formulation |
| C36 | **[DERIVED]** cold-start seeds can be mislabeled FREE although video-derived seed bytes are counted | v10 rule-118 boundary | Inventory any harvested checkpoint statistic separately | Count, hash, parse, and receiver-consume every video-derived seed | `unaddressed` | Y/Y/Y | Custody/compliance |
| C37 | **[MEASURED]** advisory axes can be promoted by inference | MPS/MLX and n24 phase are non-promotable | Keep contest CPU/CUDA and advisory axes separate | Exact archive bytes on both governed axes | `carried` | Y/Y/Y | Evidence axis |
| C38 | **[MEASURED]** development checkpoints can be mistaken for promotion candidates | New `_dev`/`_prod` pointer gate on main; untagged v9c2 bank defaults safe-side | Name all v9c3 runs/banks `_dev` | v10 stays `_dev` until reviewed dev-to-prod creation; only explicit `_prod` is maturity-eligible | `fixed` | Y/Y/Y | Historical promotion risk; current pointer gate fixes this surface, other promotion gates remain |
| C39 | **[MEASURED]** Movable's remaining errors are partly realization-limited, not only a missing birth label | ep900 per-class sub-LSB fraction 0.533 advisory | Measure nuclei and realization regime together; do not equate class error with birth | Give Movable an explicit birth/representation path before phase | `unaddressed` | Y/N/Y | Instance/representation diagnosis; family open |
| C40 | **[MEASURED]** `_prod` bank immutability is a helper with no production bank-writer consumer | `assert_bank_dir_writable`/`bank_dir_name` docs and call-site search | Route every v9c3 banker through the helper even though v9c3 is dev | Wire all v10 bank writers; create new dated prod banks, never mutate | `unaddressed` | Y/Y/Y | Banking integration/custody |
| C41 | **[MEASURED]** name-anchored search can create a second canonical v10 spec and split authority | Canonical-doc registry finds the branch SPEC; duplicate gate is warn-only and capped at 64 refs; §14.6 is now folded there at `d72020522b` | No new v9c3 spec outside its named compiler/memo | Preserve that single SoT and fold this review there; no second v10 SPEC | `carried` | Y/Y/Y | Process/search guard; not proof of global uniqueness |
| C42 | **[DERIVED]** paired ep725 forks without a persisted common RNG state can confound momentum/LR effects | ep725 EMA has no RNG/data-order state; the factorial previously specified identical weights but not identical recorded stochastic inputs | Record one seed, data order, initialization, and source hash for every paired arm; preregister replication or a deterministic single-seed scope | Cold v10 must preserve the same deterministic identity and label any single-seed inference | `unaddressed` | Y/Y/Y | Experimental-design/replication; optimizer family open |
| C43 | **[MEASURED]** §14.7 joins an affine-head feature-space pullback, Morse-Smale duality, and collateral-free FiLM locality; **[DERIVED]** the latter two do not follow | At `a1539fbeb2`, the exact active boundary also needs head bias and a top-logit condition; `F=f(phi)`, MS equivalence, and scalar-critical-point identity are unproved; current FiLM is a global broadcast actuator | No v9c3 claim; a cheap oriented-stratum collateral probe may inform v10 | Preserve the rank-4 affine-head pullback as a valid nucleus; relabel MS/FiLM as hypotheses and require receiver-closed n600 topology/collateral evidence before admitting that representation | `unaddressed` | Y/N/Y | Formulation/equivalence and actuator-support guarantee; not an MS or FiLM family verdict |
| C44 | **[MEASURED]** §14.8 equates evaluator outputs with candidate carriers (`d_seg=partition rate`, `d_pose=xi`, `bytes=seed`) and calls pixel/subpixel detail invisible; **[DERIVED]** only the discrete argmax-Hamming nucleus is exact | At `49ecbb2538`, d_pose authority remains frozen-PoseNet first-six-output MSE, archive authority is every exact zip byte, and pixel/subpixel changes matter whenever they survive R and cross a margin; pure topology also omits raster geometry and #531's `T` | No v9c3 claim; preserve exact receiver readouts | Rephrase cell complex, xi, and seed as factorization candidates; gate partition-only vs `+T`, xi pose sufficiency, and exact archive accounting through the real receiver | `unaddressed` | Y/N/Y | Formulation/representation sufficiency; not a cell-complex, xi, or seed family verdict |
| C45 | **[MEASURED]** the probe memo mislabeled an ad-hoc per-pixel-independent intervention as "fire the phase stack" and an oracle ceiling for #424/#425 | Operator correction `2026-07-18T03:57:13Z`; code composes independent min-norm camera displacements, while #424 is in-loop conditioning and #425 is coherent per-pair/per-dash coding | Bind every assay to an exact actuator/program identity before interpreting its result | Retract efficacy verdict, correct §14.6's superseded #425 ruling, and separately test both actual modes through their real receivers | `new-from-fix` | Y/Y/Y | Measurement/intervention identity; no phase-family verdict |

### Campaign-scale triple-signature ranking

**[DERIVED] Highest current `DEFAULT-HARMFUL x SILENT x MEASUREMENT-CORRUPTING` risks:**

1. Phase intervention-identity conflation: an ad-hoc independent-pixel treatment was labeled as
   #424/#425 and as an efficacy ceiling until the operator correction.
2. #529 incomplete readiness status: `spec_v10_status().clear` can omit the always-owed post-gate
   fold and appear clear; the compiler itself still fail-closes.
3. #528 objective composition.
4. #531 implicit/double-paid `T`.
5. #532 real-only projection and non-receiver init claim.
6. #518 incomplete boundary/event persistence.
7. Phase-probe raw-receipt and duplicate-writer custody.
8. Ep725 deploy snapshot mislabeled as exact resume state.

**[DERIVED] Not currently silent:** EMA masking (live-gap rows expose it), simultaneous ep700
engagement (logged), and the cold Muon firing (logged). Their causal meaning is still confounded.

## v9c3: signal-maximizing revised contract

### Checkpoint roles; never pool "checkpoint soup"

| Source | Measured custody | Legitimate question | Forbidden claim |
|---|---|---|---|
| v9c2 ep725 EMA BEST, SHA-256 `b0a431e9259cd3c54ae53b677076823f36e096b27eb0d9ba74ed7c54c9113cef` | **[MEASURED]** 460,448 B, 59 keys, epoch 725, no optimizer/RNG/event state | Primary named v9c3 **fork**: fork reconditioning, live/EMA clearance, independent Muon factors, late train-side phase response | Exact continuation or pre-Muon optimizer resume |
| v9c2 stage EMA ep726, SHA-256 `88aa0b503e953c501a1e959c3096c328a2acbb9007c9f775d56cefd95676ca64` | **[MEASURED]** 460,808 B; parameter arrays are identical to ep725 because saved before first Muon update | Custody/control that the stage boundary itself did not alter EMA parameters | Independent treatment arm |
| sacred full ep726 sidecar, SHA-256 `68af8e2b608bce4fc1d48be75c7a030c124bb4e2052430fff5878df9912aac63` | **[MEASURED]** 1,554,048 B, 187 keys, optimizer/RNG/live/EMA; already post-switch | Bit-faithful post-switch replay/control; optional same-state flat-vs-anneal continuation | Pre-Muon warm-start source |
| donor mod32cap ep650 | **[MEASURED]** pre-v9c2 phase engagement, but not a cold seeded v10 birth | Separate pre-phase train-side phase on/off assay after first measuring existing nuclei | Proof of cold birth-before-phase or v9c3 identity |
| any other prior checkpoint | **[ASSUMED]** until catalogued/hash-verified | Only a preregistered, source-specific question | Pooling trajectories or treating incomparable states as replicates |

### Required revisions before operator GO can be consumed

1. **[DERIVED] Source custody:** copy/preserve/hash the full ep726 sidecar in a newly named
   `v9c3_dev_*` bank and record source roles. Label every ep725 deploy-based arm `fork`, never
   `resume`; wire the real bank writer through `assert_bank_dir_writable` before relying on the
   prod-immutability policy.
2. **[DERIVED] Land/review #517/#518 on current main. [MEASURED]** The audited branch commit is not
   an ancestor of main. Close R1-R3 persistence (`last_boundary_epoch`, `engaged_epoch`, detector
   history/mode) and make BEST/deploy limitations explicit.
3. **[DERIVED] Dedicated typed v9c3 compiler:** emit argv, program/manifest hashes, parser receipt,
   expected-active-lever manifest, exact source hashes, and a real success path. No raw invented
   flags.
4. **[DERIVED] Fork baseline gate:** schedule-position the source and require the initial n600 EMA
   d_seg to reproduce `0.0034579722` within a predeclared tolerance. The static canary is insufficient.
5. **[DERIVED] Recondition before Muon:** compare the CONFIG 8-epoch and beta2-DERIVED 27-epoch
   AdamW windows, or default to 27 if only one arm is affordable. Re-anchor Muon to the resulting
   boundary/event; absolute epoch 726 is forbidden. Require `muon_warm_seeded_leaves > 0` in warm arms.
6. **[DERIVED] Independent factorial identity:** use two independent `TypedLever`s for momentum
   cold/warm and LR flat/cosine. The named `MuonWarmStart` bundle is forbidden for this attribution
   experiment. Existing trainer flags and DSL primitives suffice; a small compiler/manifest build,
   not a trainer build, is owed.
7. **[DERIVED] Hold Force3 and all other losses fixed** across the Muon factorial. Do not infer the
   original ep700 lever efficacy from inherited ep725 weights.
8. **[DERIVED] Checkpoint every reached treatment boundary:** fork baseline, post-rewarm, pre-Muon,
   Muon +1, +25, and +50; preserve all full state sidecars atomically. Reach +100 and pose engagement
   only if the preregistered earlier separation/stop gates have not terminated the arm.
9. **[DERIVED] Readouts:** n600 EMA and live per-class d_seg; EMA gap; Road/Lane/Movable nucleus
   counts/area/birth epoch; sigma-min trajectory; pose-gate state; event ages; optimizer transition
   telemetry; exact factor identity.
10. **[DERIVED] Pose:** rebase/persist the detector, distinguish fresh detector history from restored
    state, ramp w_pose, and preserve a terminal live/EMA comparison.
11. **[DERIVED] Phase:** the ep725 fork may separately test the actual #424 train-side late-recovery
    program and the actual coherent #425 receiver; never substitute the independent-pixel strawman.
    It cannot test birth-before-phase. A mod32cap ep650 auxiliary assay must first report whether the
    relevant nuclei already exist; if they do, it also cannot answer cold ordering.
12. **[DERIVED] RNG and stop rule:** paired arms must share recorded seed, data order, initialization,
    and source bytes. Predeclare replication or label a deterministic single-seed scope. Evaluate
    milestones sequentially and stop at the first preregistered separating gate; +100 is conditional,
    not mandatory. No score, promotion, "optimal vehicle", or indefinite continuation claim. Skip any
    arm whose question is already answered by a cheaper offline/paired assay.

### Recommended staged design

**[DERIVED] Stage A:** two ep725 EMA forks with identical source bytes, recorded seed/data order,
and fresh AdamW initialization, 8 versus 27 epochs, with fork-baseline and terminal n600
live/EMA/per-class readouts. Select the shorter window only if it is equivalent under preregistered
geometry and moment-settling criteria.

**[DERIVED] Stage B:** from the selected fully checkpointed AdamW state, run the 2x2 momentum
cold/warm x LR flat/cosine under one common capped window. Evaluate matched arms sequentially at the
same milestones and stop all arms at the first preregistered separating milestone; `+100` remains
conditional. This identifies #270's two mechanisms without changing Force3.

**[DERIVED] Stage C:** use the full ep726 post-switch sidecar only as a fidelity/continuation control
or a same-state LR continuation test. Do not use it to claim warm-start efficacy.

**[DERIVED] Stage D:** if still decision-relevant, run a separately named pre-phase train-side
phase/control assay from mod32cap ep650 and a distinct coherent-#425 receiver assay if that codec is
still proposed. Each must bind its actual actuator identity. They inform application mode; only a
cold v10 construction can validate the full birth-before-phase law.

**[DERIVED] Wall-clock-worth verdict:** yes for this capped staged information experiment while v10
blockers are fixed in parallel; no for a full v9c3 optimization continuation or any v9c3 work that
delays the actual v10 compiler/receiver. Ordinary epochs are roughly 115-123 seconds, while an n600
verdict is much more expensive; terminal verdict count must be planned, not sprayed.

## v10 corrections and launch gate

| Item | Current finding | Required correction | Gate |
|---|---|---|---|
| #528 objective | **[MEASURED]** score-domain pose loss already carries the nonlinear marginal; proposed w_pose squares it | `w_pose=1` under score-domain loss; weight-domain form must be a separately named formulation | Unit/gradient test on emitted objective |
| #529 readiness/compiler | **[MEASURED]** status probes symbols/files and can report `clear` without `post_gate_fold_owed`; the compiler itself then fail-closes and has no program/argv/manifest/hash/parser success receipt | Include the post-gate fold in status; compile one complete `WitnessProgram`; resolve LawRefs; emit exact argv/constants; parse with real trainer; bind hashes | Status cannot clear before a real compile; compiler stays fail-closed and passes an actual success test |
| #530 cold birth | **[MEASURED]** cold spec references fork-only head/EMA laws | Implement `InitHeadSolve`; keep `ForkHeadSolve`/`ForkEmaClearance` exclusive to resumed forks | Cold program contains no resume source or fork-only event state |
| #531 quotient residual | **[MEASURED]** generic textured residual can double-pay geometry; measured flat/textured gap 0.0416 vs 0.0048 | Explicit class/cell-conditioned `T` with unique custody after deterministic geometry quotient | Structural parameter/gradient-route invariant prevents trunk ownership; ablation measures effect but is not proof of non-relearning |
| #532 realized projection/init | **[MEASURED]** real range(A) equality fails uint8 clip/round by max 62.74; structured-init detectors are not class-complete | Optimize/verify through actual R, int8/dequant, parse-back; implement class-specific cold init | n600 byte-closed receiver proof |
| §14.2 order | **[ASSUMED]** island-birth-before-phase is a SPEC-mandated but empirically unvalidated ordering hypothesis | Compiler validation must require per-class nucleus-ready before any phase stage until the hypothesis is tested | Negative test rejects phase-before-birth program; matched cold A/B supplies efficacy evidence |
| §14.5 phase | **[MEASURED, custody-limited]** n24 independent-pixel min-norm moves are harmful, but this ad-hoc third mode supplies no efficacy verdict for #424 or #425; only the phase-band/amplitude-open diagnosis survives. The phase-efficacy/#425 ruling in current §14.6 is superseded by the operator correction. | Correct §14.6, preserve raw custody, and separately test actual #424 train-side conditioning and the actual coherent #425 encode/decode/receiver if retained | n600 through-R, int8/dequant, receiver-consumed, matched treatment-identity control |
| Phase rate | **[MEASURED]** 29,958 B event stream is 16.6x the 1.8 KB upper estimate and 33.3x the 0.9 KB lower estimate | Measure entropy/persistence-class generator and include receiver bytes | Exact archive byte delta and score units/byte |
| Persistence | **[MEASURED]** ramp/event keys are incomplete on #518 | Persist all boundary/detector/optimizer/RNG state per stage | Crash/restart equivalence at every boundary |
| Seeds | **[DERIVED]** video-derived cold seed payload is counted even when generator code is free | Inventory/hash/count/receiver-consume every seed | Archive parser and custody manifest |
| Maturity/custody | **[MEASURED]** main refuses `_dev`/untagged vehicle pointer promotion; prod immutability is helper-only; the registered branch SPEC contains §14.6 at `d72020522b`, whose maturity part remains valid while its phase-efficacy/#425 ruling is now superseded | Name build/eval outputs `v10_dev_*`; wire bank writers; correct and keep future findings in that canonical SPEC; create a new reviewed `v10_prod_*` bank only after all exact gates | Pointer refresh records zero maturity refusal and bank-writer immutability test passes for the selected prod row |
| §14.7 MS/FiLM representation | **[MEASURED]** the affine-head boundary is a bias-inclusive, top-class-restricted feature-space pullback and current FiLM broadcasts global scales/shifts; **[DERIVED]** MS duality and one-boundary collateral-free FiLM do not follow | Relabel the equivalence/locality claims as empirically gated hypotheses; preserve the rank-4 pullback nucleus | Conditional on selection: receiver-closed n600 matched-byte MS-seed vs weight-blob comparison, decoded topology fidelity, and full oriented-stratum collateral/d_pose/R/int8/archive receipt |
| §14.8 evaluator level | **[MEASURED]** d_seg is post-R argmax-map Hamming disagreement, d_pose is frozen-PoseNet-output MSE, and rate is exact archive bytes; **[DERIVED]** cell complex/xi/seed are candidate sufficient statistics, not identities, and pixel/subpixel effects are conditional on R/margin survival | Rephrase as a receiver-equivalence/factorization hypothesis that retains raster geometry, pose appearance, `T`, and grammar costs where needed | Conditional on selection: n600 partition-only vs `+T`, xi-only pose-sufficiency, and exact archive-byte ablations through the actual receiver |

**[DERIVED] v10 formulation verdict:** the factorized/witness-compiler paradigm remains open.
`verdict_scope=FORMULATION + IMPLEMENTATION + CUSTODY`; no family/paradigm rejection follows from
the current blockers or the n24 independent-pixel strawman. That result carries no phase-efficacy
direction.

## Residual confound surface, ranked for v10

1. **[MEASURED] #529 false-clear readiness status** — `spec_v10_status().clear` can omit the
   always-owed post-gate fold; the compiler itself remains fail-closed.
2. **[MEASURED] #528 exact-objective mismatch** — corrupts every optimization comparison.
3. **[MEASURED] #531 implicit `T` custody** — can erase the claimed Kolmogorov/factorization advantage.
4. **[MEASURED] #532 realized-R/receiver gap** — real-valued exactness is not shipped exactness.
5. **[MEASURED, custody-limited] actual-mode phase efficacy** — the ad-hoc independent-pixel strawman
   is harmful but non-probative; #424 train-side and coherent #425 receiver modes are both unmeasured.
6. **[MEASURED] incomplete boundary/event persistence** — violates P0 resumability and changes treatment identity.
7. **[MEASURED] phase carrier rate and receiver consumption** — current byte assumption is false.
8. **[MEASURED] birth-before-phase enforcement and nucleus observability are absent;
   [ASSUMED] ordering efficacy** — the SPEC-mandated order is not executable and its efficacy remains
   empirically unvalidated.
9. **[DERIVED] counted cold-seed custody** — generic decoder code is free; video-derived sufficient statistics are not.
10. **[MEASURED] Movable realization/birth split** — 0.533 sub-LSB flip mass is a representation warning,
    while nucleus chronology is still absent.
11. **[MEASURED] contest-axis and maturity custody** — advisory MLX/macOS/n24 or `_dev` findings cannot
    authorize CPU/CUDA pointer promotion; `_prod` naming alone is also insufficient.
12. **[MEASURED] production-bank and canonical-spec guard scope** — the immutability helper is not in
    production banking; §14.6 is now folded, but the duplicate-SPEC guard remains warn-only and
    bounded to 64 refs.
13. **[MEASURED] §14.7 joins pullback, MS duality, and FiLM locality; [DERIVED] the equivalence/locality
    claims are overbroad** — affine-head pullback is valid, but Morse-Smale duality and collateral-free
    global FiLM are unproved; this becomes binding only if v10 selects that representation.
14. **[MEASURED] §14.8 collapses evaluator outputs into proposed carriers; [DERIVED] representation
    sufficiency remains open** — argmax Hamming is exact, while cell topology/xi/seed alone are not
    proven sufficient for raster geometry, frozen PoseNet, `T`, or exact archive grammar.
15. **[MEASURED] phase intervention identity and stale §14.6 ruling** — the independent-pixel
    strawman is not #424/#425, both actual modes remain untested, and the canonical phase-efficacy/
    #425 ruling must be corrected before the SPEC can authorize design closure.

## Monotone de-confounding and optimality

**[DERIVED] Monotone de-confounding is `FALSE` for the vehicles as presently buildable.** v9c3
intends to cure absolute-event geometry but currently introduces/retains fork identity, incomplete
persistence, bundled treatment identity, inherited phase conditioning, and no typed config. v10
eliminates the warm-start class by cold construction but presently introduces #528-#532 and does
not enforce its own curriculum order. The **revised plan** is conditionally monotone only after the
gates above are real and tested.

**[DERIVED] Monotone optimality must be split by vehicle purpose.** It is the wrong criterion to ask
whether v9c3 is a from-scratch score-optimal run. Its target is decision-relevant information per
wall-clock and v10 readiness. On that target, the revised factorial/checkpoint plan is more optimal
than continuing the single stacked trajectory. For v10, the means-vs-ends firewall remains binding:
no more design accumulation before an emitted, byte-closed, parse-backed, exact-evaluated candidate.

**[DERIVED] Means-hoarding risk is high.** The pointer is unmoved, the named phase rate assumption
failed, and the actual train-side/receiver phase effect is still absent. A new design memo or long
unfactored run that does not close one of the ranked gates is negative-EV.

## Recommended ordering and fastest lower-exact-row path

1. **[DERIVED] Preserve/canonicalize the phase raw receipt if recoverable; otherwise rerun only the
   minimum custody-complete measurement needed.** Consume the n24 result only for its phase-band/
   amplitude-open diagnosis and independent-pixel-strawman warning; retract all efficacy direction
   and do not treat it as §14.5 clearance.
2. **[DERIVED] In parallel, land/review #517/#518 and build the dedicated v9c3 typed compiler plus
   full-state checkpoint manifest.** Do not launch until persistence and fork-baseline reproduction pass.
3. **[DERIVED] Run only the capped v9c3 stages that answer unresolved v10 choices:** 8-vs-27 fork
   geometry, independent Muon momentum/LR, pose-gate recovery, and train-side/constrained phase mode.
4. **[DERIVED] In parallel, close v10 #528-#532, make birth-before-phase executable, and make #425
   receiver consumption/rate honest.** The §14.6 fold is present at `d72020522b`, but its phase-
   efficacy/#425 ruling is superseded by the 03:57 correction; preserve the maturity law and correct
   the phase part. Fold this review and every later correction into that registered canonical branch SPEC rather
   than creating a second v10 SPEC. Relabel §14.7 MS duality/FiLM locality as hypotheses and require
   its receiver-closed topology/collateral receipt if selected. Treat §14.8's cell-complex/xi/seed
   objects as candidate sufficient statistics and run partition-only-vs-`+T`, xi-pose, and exact-byte
   ablations if selected. v9c3 does not gate cold-only laws such as `InitHeadSolve`.
5. **[DERIVED] Compile cold `v10_dev`, byte-close every stage, parse back exact bytes, and run short
   receiver smoke.** After independent landing review, create a new immutable `v10_prod_*` bank and
   run exact contest-CPU and contest-CUDA replay through the governed launcher. Only an eligible
   `_prod` exact row can move the pointer.

**[DERIVED] Fastest-path rule:** if a cheaper offline/paired probe answers a v9c3 question, skip that
arm. If v9c3 and v10 build work compete for the critical path, v10 compiler/receiver wins; v9c3 may
run concurrently only when it produces uniquely decision-relevant signal.

## Recursive adversarial review and campaign seal

Every new unaddressed confound reset the counter. Discovery rounds before the final seal found:

1. ep725 deploy snapshot lacks exact resume state;
2. static-canary blindness to schedule-positioned baseline corruption;
3. absolute Muon immediate-fire and #270 bundled treatment identity;
4. #518 boundary/detector persistence gaps;
5. Movable endpoint overclaim and inherited phase-order limitation;
6. v10 #528-#532, rate, receiver, and order gaps;
7. late-landed independent-pixel collateral plus raw/sample/actuator mismatch, initially mislabeled
   as a #424/#425 post-hoc ceiling;
8. independent Muon axes exist only as primitives/typed-lever escape surface, not a named v9c3 compiler;
9. the canonical `_dev`/`_prod` maturity axis and its safe-side pointer/immutability semantics;
10. Movable's 0.533 realization-limited advisory fraction, distinct from the unmeasured birth chronology;
11. prod-bank immutability is helper-only rather than consumed by production banking;
12. the initial §14.6 handoff gap was closed during review by the registered branch update
    `d72020522b`, resetting the seal before the current-state recheck;
13. paired ep725 treatment arms lacked an explicit identical-RNG/data-order/replication contract;
14. the phase memo's duplicate-writer identity is not independently verifiable without the missing raw receipts.
15. §14.7's rank-4 pullback nucleus was promoted into unproved Morse-Smale equivalence and
    collateral-free global-FiLM locality at `a1539fbeb2`; the formulation/guarantee overclaim reset
    the seal without becoming a family verdict.
16. §14.8 at `49ecbb2538` promoted cell complex, xi, and seed candidates into evaluator identities
    and overgeneralized a per-pixel formulation failure into pixel/subpixel invisibility; the exact
    receiver-sufficiency scope reset the seal again.
17. the `2026-07-18T03:57:13Z` operator correction plus code reinspection showed the phase probe was
    an ad-hoc independent-pixel third mode, not #424 or #425; all efficacy inferences were retracted
    and the seal reset again.

| Final pass | Surfaces attacked | Result |
|---:|---|---|
| 1 | Canonical pointer/main, SPEC cutoff through §14.8, corrected §14.6 scope, phase intervention identity, and raw-custody paths | **[MEASURED] CLEAN: zero new confounds.** The already-recorded custody and conditional-representation gaps remained. |
| 2 | Checkpoint hashes/roles/schemas, #517/#518 ancestry, independent Muon primitives, RNG/stop contract, and named v9c3 compiler/manifest | **[MEASURED] CLEAN: zero new confounds.** The already-recorded compiler/persistence gaps were confirmed. |
| 3 | v10 #528-#532, §14 order, §14.7/§14.8 formulation, phase scope, exact-score arithmetic, pointer/axis custody, ordering/self-attack | **[MEASURED] CLEAN: zero new confounds.** |

**[MEASURED] Campaign-review seal: `3/3 CLEAN` at the cutoff.**  
**[DERIVED] Seal scope:** inventory and recommendations only. v9c3 remains `REVISE`; v10 remains
`NOT_LAUNCH_READY`.

## Attack on this review

1. **[MEASURED] The live trajectory partially recovered after Muon.** A reviewer can refute the
   stronger phrase "not recovering"; the defensible statement is "had not reattained ep725 by the
   last n600 verdict at ep925."
2. **[INFERRED] Cold Muon is the primary discontinuity, but v9c2 and donor are not randomized matched
   controls.** The review does not assign a Muon-family negative.
3. **[MEASURED] Movable was absent at configured initialization but learned later.** This review
   rejects "unborn endpoint" and asks for nucleus telemetry rather than extrapolating from class error.
4. **[MEASURED, custody-limited] The phase numbers are committed-report facts whose raw receipt is
   missing.** Their direction is not consumed as phase efficacy at all: only the diagnosis and the
   independent-pixel-strawman warning survive, never an independently verified §14.5 result.
5. **[DERIVED] A multi-arm v9c3 can itself become means-hoarding.** The staged stop rule and
   cheapest-probe substitution are necessary; if offline assays answer the questions, skip v9c3.
6. **[ASSUMED] Absence searches covered the canonical repo, available worktrees, and named SSD tiers,
   not every possible external store.** The claim is "not present in consulted custody," not universal loss.
7. **[DERIVED] Three clean passes can miss a confound shared by all reviewed artifacts.** MAIN must
   independently review this branch diff, especially the checkpoint-role logic, phase verdict scope,
   and ordering. The campaign seal must reset if MAIN supplies new primary evidence.
8. **[DERIVED] §14.7's exact pullback nucleus does not prove its chosen representation or actuator.**
   MAIN should reject any reading of this review that kills MS/FiLM as families or admits them without
   the conditional receiver-closed topology/collateral probe.
9. **[DERIVED] §14.8's "right level" is a useful candidate factorization, not an evaluator theorem.**
   MAIN should preserve its MDL direction while restoring exact authority to the joint receiver
   equivalence class and demanding the conditional sufficiency ablations.

## Final disposition

- **[DERIVED] v9c3:** `REVISE`; after all gates, `PROCEED` only as capped signal/v10-readiness work.
- **[DERIVED] v10:** preserve the paradigm; rewrite/close the named design/compiler/receiver gates;
  no launch today.
- **[MEASURED] Pointer:** `0.1910828242` unchanged; no score, archive, dispatch, or live-run mutation
  resulted from this review.
- **[DERIVED] Landing authority:** MAIN review is mandatory before this memo/DAG feed becomes canonical.
