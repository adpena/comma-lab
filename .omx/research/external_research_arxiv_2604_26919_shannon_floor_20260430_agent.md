# External Research Intake: arXiv 2604.26919v1 And Shannon-Floor Relevance

**Date:** 2026-04-30  
**Agent:** Codex  
**Target paper:** arXiv:2604.26919v1, *Causal Learning with Neural Assemblies*  
**Primary question:** does this paper offer contest-compliant ideas for pushing the comma video compression system toward the Shannon floor?  
**Evidence status:** external research only. Nothing here promotes, ranks, kills, or scores any lane. CUDA exact auth eval on exact archive bytes remains the only promotion/kill/ranking truth.

## Executive Verdict

The paper is not a video compression paper, not a learned codec paper, and not an entropy-model paper. It does not contain a direct path to a smaller `archive.zip`, a better mask representation, a neural weight codec, or a scorer-aware image/video model.

Its usable contribution for this project is narrower: an auditable mechanism pattern for learning and validating directed dependencies using sparse winners, adaptive gain scheduling, and dual readouts. The best transfer is as a **diagnostic and allocation discipline** for beta sensitivity work and alpha failure localization:

- Use **adaptive warm-ramp schedules** instead of abrupt scorer-sensitive loss/quantization transitions.
- Use **two independent readouts** for sensitivity claims: a structural score such as Fisher/gradient/channel asymmetry and a functional score such as held-out perturbation propagation into PoseNet/SegNet components.
- Use **hard Top-K/sparse winner selection** as a byte-aware allocator for protected channels, residual pixels, or temporal regions, but solve it against charged archive bytes rather than copying neural-assembly dynamics.
- Use a **known-structure causal audit graph** to localize failures across mask payload, renderer, pose stream, and score components.

Estimated direct score value is low: `0.000` unless translated into existing alpha/beta lanes. Indirect value is plausible but modest, roughly `[prediction] 0.002-0.015` if it prevents OWV3-style PoseNet regressions or focuses residual side information better. It is not a Shannon-floor paradigm shift by itself.

## Source Intake

Primary sources read:

- arXiv abstract page: https://arxiv.org/abs/2604.26919
- arXiv PDF: https://arxiv.org/pdf/2604.26919
- arXiv API metadata: https://export.arxiv.org/api/query?id_list=2604.26919
- arXiv source tarball: https://arxiv.org/e-print/2604.26919

Related primary sources and OSS checked:

- Papadimitriou et al., *Brain computation by assemblies of neurons*, PNAS 2020: https://www.pnas.org/doi/10.1073/pnas.2001893117
- Original Assembly Calculus / NEMO code cited by the PNAS paper: https://github.com/dmitropolsky/assemblies
- Dabagia et al., *Assemblies of neurons learn to classify well-separated distributions*, COLT/PMLR 2022: https://proceedings.mlr.press/v178/dabagia22a.html
- COLT learning-with-assemblies code repo: https://github.com/mdabagia/learning-with-assemblies
- Dabagia et al., *Computation with Sequences in a Model of the Brain*, arXiv:2306.03812: https://arxiv.org/abs/2306.03812
- Dabagia et al., *Coin-Flipping In The Brain*, arXiv:2406.07715: https://arxiv.org/abs/2406.07715
- `neural-assemblies` PyPI package and GitHub fork: https://pypi.org/project/neural-assemblies/ and https://github.com/Caerii/assemblies

No official DIRECT implementation for arXiv:2604.26919 was found from exact-title, mechanism-name, and author/topic searches. The arXiv source contains TeX, references, and figures, not executable experiment code.

## Precise Paper Summary

arXiv:2604.26919v1 was submitted on 2026-04-29 by Evangelia Kopadi and Dimitris Kalles. The title is *Causal Learning with Neural Assemblies*. It is categorized under cs.LG, cs.AI, and cs.NE. The paper is 8 pages with 11 figures.

The paper asks whether neural assemblies can internalize causal direction. A neural assembly is a sparse group of neurons that becomes a reusable representation through repeated co-activation, k-Winner-Take-All competition, and Hebbian plasticity. The paper explicitly does **not** claim de novo causal discovery from observational data. It assumes a known ground-truth DAG/SCM and tests whether the neural-assembly mechanism can learn directed bindings under that provided structure.

The pipeline is:

1. Ingest a domain table plus a specification containing variable order, directed links, and value mapping.
2. Encode categorical variable values into sparse neural spike patterns, using either value-to-firing-rate encoding or an index-based active set.
3. Stabilize assemblies through repeated k-WTA and Hebbian formation.
4. Apply DIRECT, short for DIRectional Edge Coupling/Training, on each provided edge `u -> v`.
5. Read out directed links using both synaptic-strength asymmetry and functional propagation overlap.
6. Evaluate Top-K structural recovery with `K` fixed to the number of ground-truth links.

The core mechanism, DIRECT, repeatedly co-activates a source assembly `A_u` and target assembly `A_v` while temporarily increasing forward plasticity on `u -> v`. The reverse path still exists, but receives less strengthening. The update is local and winner-sparse. For active source winner `j` and target winner `i`, the connectome entry is multiplied by `1 + area_to_area_beta` under the temporarily boosted forward gain.

The reported schedule is `adaptive_soft` with:

```text
warm_beta = 0.09
max_beta = 0.16
ramp_steps = 20
beta_s = beta_start + ((s - 1) / (R - 1)) * (beta_max - beta_start)
R = ramp_steps
approx increment = 0.00368 per ramp step
```

The authors argue that the warm-ramp schedule preserves assembly stability during early winner-set drift, then strengthens directional separation after overlap-based convergence evidence accumulates.

## Mathematical Claims

The central mathematical object is directional asymmetry, not compression rate.

Synaptic-strength readout:

```text
S_fwd(u,v) = mean(A_u -> A_v)
S_rev(u,v) = mean(A_v -> A_u)
Delta(u,v) = S_fwd(u,v) - S_rev(u,v)
```

A positive `Delta(u,v)` and high rank among ordered pairs are taken as evidence for `u -> v` under the known-structure setting.

Propagation-overlap readout:

```text
input_to_v[i] = sum_{j in A_u} W_{u->v}[j,i]
Ahat_v = top_k(input_to_v)
overlap(u -> v) = |Ahat_v intersect A_v| / k
Delta_prop(u,v) = overlap(u -> v) - overlap(v -> u)
```

Top-K metrics:

```text
Precision@K = TP@K / (TP@K + FP@K)
Recall@K = TP@K / |ground-truth links|
```

`K` is fixed to the number of ground-truth links, so the predicted edge count is forced to match the answer-key edge count.

The do-calculus validation uses the standard backdoor adjustment:

```text
E[Y | do(X = x)] = sum_z E[Y | X = x, Z = z] P(Z = z)
```

The claim is empirical/procedural: with known structure and bounded perturbations, the directed bindings remain inspectable and recoverable. This is not a new rate-distortion theorem, not a Shannon bound, and not a proof that such assemblies can discover graph structure from raw observations.

## Reported Results

The main experiments use small known-structure SCMs, including an Alzheimer SCM with 10 variables and 12 directed links, plus an education/student-dropout SCM. OULAD mappings are retained as supplementary portability evidence.

Reported results include:

- Top-K structural recovery matched the ground-truth link set in the reported fixed-supervision runs.
- On the Alzheimer SCM, all 12 ground-truth links had positive propagation asymmetry.
- Forward propagation overlaps ranged from about `0.74` to `1.00`; reverse overlaps were about `0.04` to `0.15`.
- In R3 robustness, six random seeds across three encoding-separation conditions retained `Precision@K = 1.0`, `Recall@K = 1.0`, and propagation pass rate `1.0`.
- R3 mean `Delta_prop` values were approximately `0.801 +/- 0.030` for base `15x`, `0.811 +/- 0.026` for milder `10x`, and `0.787 +/- 0.062` for harder `6.7x`.
- Interventional and counterfactual checks are reported as directionally consistent under SCM-generated interventions and backdoor-adjusted estimates.

The paper's own discussion sets important boundaries: no observational causal discovery, no universal OOD guarantee, time-aware lagged training is exploratory, and the method requires externally supplied structure.

## Related Paper And OSS Intake

The paper sits in the Assembly Calculus/NEMO line rather than the compression line.

`dmitropolsky/assemblies` is the original public Assembly Calculus/NEMO repo cited by the PNAS 2020 paper. It is MIT-licensed, small, Python/MATLAB/C++, and contains simulation code such as `brain.py`, `learner.py`, `parser.py`, and `simulations.py`. This is the safest reference implementation for the foundational assembly operations, but it does not implement the new DIRECT causal learner.

`mdabagia/learning-with-assemblies` supports the COLT 2022 learning paper. It is notebook-heavy, MIT-licensed, and mirrors the PMLR supplement. It is useful for understanding class-assembly formation and k-WTA simulation, not for direct video compression.

`neural-assemblies` on PyPI is an alpha package published 2026-03-13 and backed by `Caerii/assemblies`, a fork/extension of the original repo. It advertises core assembly calculus, NEMO, language surfaces, and optional GPU paths with CuPy/Torch. It should **not** be installed into this contest repo without a separate supply-chain review. It is alpha, not directly tied to arXiv:2604.26919, and the repository contains a `.claude/settings.local.json` file. That is not by itself proof of compromise, but this project already has strict supply-chain posture and no dependency need exists here.

The Dabagia/Papadimitriou/Vempala related papers contribute useful primitives:

- COLT 2022: assemblies can form class representations from few samples when classes are separated.
- arXiv:2306.03812: sequences can be memorized through ordered assemblies and finite-state behavior can be learned through presented sequences.
- arXiv:2406.07715: noisy assembly dynamics can encode statistical transition behavior and sample from learned Markov-like structure.

For this project, these are analogy sources for sparse, local, inspectable state machines. They are not codec components.

## Implementation Feasibility For This Repo

Direct implementation feasibility is high for toy diagnostics and low for contest runtime value.

What would be easy:

- Reimplement DIRECT from the paper in a small standalone training-only diagnostic module using NumPy or PyTorch.
- Build a directed graph over known contest components: `mask regions -> renderer activations/channels -> PoseNet/SegNet component distortions`.
- Use a warm-ramp schedule and Top-K winner selection to choose protected channel groups or residual regions.
- Emit a JSON audit artifact with structural readout, functional perturbation readout, stability split, and source provenance.

What would be hard or unattractive:

- Shipping a neural-assembly simulator inside `archive.zip` would add runtime code/bytes without an obvious rate win.
- Sparse assemblies do not compress the 421KB mask stream, 686KB PFP16 frontier archive, or renderer payload by themselves.
- The paper's experiments use small tabular SCMs, not image/video fields or neural renderer weights.
- The paper relies on provided graph structure. Our score-affecting dependencies are partly discovered empirically through exact scorer response, not known a priori.

Practical feasibility verdict: implement only as a diagnostic pattern inside beta sensitivity/alpha audit work if already touching those systems. Do not create a new full lane centered on neural assemblies.

## Likely Score-Impact Hypotheses

All numbers here are predictions or derivations. They are not score evidence.

### H1: Adaptive Warm-Ramp For Scorer-Sensitive Training

Replace abrupt activation of scorer-sensitive losses, QAT pressure, KL auxiliary terms, or mixed-precision thresholds with a warm-ramp schedule modeled on DIRECT:

```text
beta_s = beta_start + ((s - 1) / (R - 1)) * (beta_max - beta_start)
```

Contest translation:

- Lane 19 logit-margin loss: ramp boundary/margin weight after baseline fidelity stabilizes.
- OWV3: ramp quantization/protection threshold rather than flipping many channels to a new action at once.
- Alpha mask redesign: ramp temporal/boundary/embedding preservation terms after decoded-baseline-mask CE stabilizes.

Likely impact: `[prediction] 0.000 to -0.005` direct, mostly by reducing catastrophic scorer drift. This is a training-stability hypothesis, not a codec.

### H2: Dual-Readout Sensitivity Validation

Use the paper's "weight asymmetry plus propagation overlap" pattern to harden component sensitivity artifacts:

- Structural readout: Fisher/gradient/channel sensitivity, separated into PoseNet, SegNet, and combined maps.
- Functional readout: held-out perturbation actually moves `pose_dist` and `seg_dist` in the predicted direction/magnitude.

Contest translation:

- For OWV3, no channel map promotes unless rank stability and held-out perturbation response agree.
- For alpha residuals, no residual region is charged unless local mask disagreement propagates to renderer embeddings or score components.

Likely impact: `[prediction] 0.002 to -0.020` indirectly through avoiding OWV2-style regressions and making beta allocation less brittle. Actual score impact requires standalone archive eval and stacked archive eval.

### H3: Hard Top-K Sparse Winner Allocation

Use k-WTA as a design metaphor for byte allocation:

```text
choose Top-K groups by marginal expected score protection per charged byte
```

Candidate groups:

- Conv output channels in OWV3.
- PoseNet-sensitive mask boundary regions.
- Temporal-diff mask regions.
- Connected-component centroid stabilizers.

Contest translation must be byte-aware. The objective is not "largest sensitivity" but:

```text
max expected component distortion reduction / charged archive byte
```

Likely impact: `[prediction] 0.002 to -0.015` if it replaces threshold heuristics that overprotect expensive channels or misses rare high-risk regions.

### H4: Known-Structure Causal Audit Graph For Failure Localization

Use a fixed graph of contest streams and components:

```text
archive bytes -> masks/poses/renderer payloads
masks/poses/renderer -> rendered frames
rendered frames -> SegNet/PoseNet components
components -> contest score
```

Then attach per-edge diagnostics: bytes, perturbation response, component movement, and exact-eval custody when available.

Likely impact: `[engineering]` fewer false kills, less wasted exact eval, and cleaner adversarial reviews. No direct score delta.

### H5: Temporal-Precedence Diagnostics For Alpha

The paper's future-work idea of temporal-precedence-guided discovery is not validated, but it maps naturally to alpha diagnostics:

- Detect which mask regions at `t` explain boundary/pose drift at `t+1`.
- Rank temporal-diff pixels or connected components by propagation into renderer embeddings.
- Charge sparse residuals only for high-precedence/high-propagation regions.

Likely impact: `[prediction] 0.000 to -0.010` if it helps Alpha-Geo-2 residual side information stay small and score-effective. It cannot promote without exact archive CUDA eval.

## Contest Compliance Risks

1. External paper results are not contest evidence. They cannot rank, promote, kill, or retire any lane.
2. No third-party assembly package should be installed or imported into this repo without supply-chain review. There is no need to vendor `neural-assemblies`.
3. Any training-only diagnostic must write deterministic provenance if used to guide dispatch.
4. Any runtime use would have to be fully inside `archive.zip` or fixed contest code, deterministic, zip-safe, and inflate-budget proven. This is not recommended.
5. Stochastic sparse winner selection can introduce nondeterminism unless seeds, ordering, dtype, and tie-breaking are fixed and recorded.
6. A causal graph inferred from scorer probes is a diagnostic model, not proof of causal mechanism. Do not label it as causal discovery unless interventions, controls, and exact component response evidence support that narrow claim.
7. The paper fixes `K` to the ground-truth link count. Copying this without care can inflate precision-like metrics. For contest artifacts, report full ranked curves and byte/error tradeoffs, not only Top-K wins.
8. If this idea is used for sensitivity, promotion artifacts still need CUDA-authored component maps, calibration/holdout stability, response curves, exact custody linkage, and later exact archive eval.

## Experiments To Run

These are proposed experiments only. No code was edited for this intake.

### E0: No-Lane Decision

Do not launch a neural-assembly codec lane. Record this paper as diagnostic input for beta/alpha only. This is the default recommended action.

### E1: Beta Dual-Readout Sensitivity Audit

Add, when already touching beta tooling, a report that pairs:

- structural channel sensitivity: PoseNet/SegNet/combined Fisher or gradient energy;
- functional propagation: held-out perturbation deltas in `pose_dist`, `seg_dist`, and recomputed score contribution;
- rank stability: calibration vs holdout rank distance;
- byte accounting: charged bytes for each protection action.

Promotion gate remains exact archive CUDA eval. This report only decides whether a candidate is worth building/evaluating.

### E2: OWV3 Top-K Byte-Aware Protection Sweep

Run build-only sweeps over `K` protected channel groups using the objective:

```text
estimated_score_protection / charged_archive_byte
```

Reject candidates that exceed PFP16 A++ bytes unless a reviewed distortion-reduction justification exists. Exact eval only for byte-plausible candidates.

### E3: Alpha Temporal-Propagation Residual Ranking

For alpha redesign, rank candidate residual pixels/regions by:

- decoded-baseline-mask disagreement,
- boundary-ring membership,
- temporal pair-diff disagreement,
- connected-component centroid movement,
- renderer embedding drift,
- held-out component perturbation response.

Use a Top-K or knapsack selection for residual side info. Do not spend exact eval until diagnostics meet Alpha-Geo exploratory targets.

### E4: Warm-Ramp Loss/Quantization Ablation

For one already-scheduled scorer-sensitive training run, compare abrupt vs warm-ramp activation of the risky term. Local success criteria:

- lower component collapse rate in diagnostics,
- no archive byte regression,
- equal or better held-out component proxy movement.

Only exact archive CUDA eval can establish score impact.

### E5: Negative Control

Implementing a DIRECT-style assembly learner as a mask or renderer runtime codec should be rejected unless a concrete byte model beats existing payloads before eval. Expected result: no rate advantage after archive charging.

## Adversarial Objections

1. The perfect recovery result is less impressive than it looks because the graph is supervised and `K` is fixed to the number of true links.
2. The work does not solve causal discovery from observations. The paper states this boundary clearly.
3. The graphs are tiny relative to contest image/video/renderer state.
4. The method has no entropy model, no learned video representation, no arithmetic coding, and no archive construction contribution.
5. Neural-assembly terminology can make ordinary sparse selection and gain scheduling sound deeper than it is for our use case.
6. The paper reports mechanism-level auditability, but this does not imply better rate-distortion behavior.
7. Translating "causal direction" to "score-sensitive channel" is an analogy until intervention/perturbation response validates it.
8. Top-K selection can hide false positives outside K. Contest diagnostics should report full ranked tradeoff curves.
9. The adaptive gain values are tuned for the paper's toy settings. Copying `warm_beta=0.09`, `max_beta=0.16`, or `ramp_steps=20` literally would be arbitrary in our lanes.
10. The PyPI/GitHub implementation ecosystem is not needed for contest work and introduces avoidable dependency risk.

## Recommended Intake

Classify arXiv:2604.26919v1 as **Translate/Watch**, not Copy.

Use:

- dual-readout validation for component sensitivity artifacts;
- warm-ramp scheduling for scorer-sensitive training transitions;
- Top-K sparse selection as a byte-aware allocator;
- fixed causal audit graphs for failure localization.

Do not use:

- neural assemblies as a runtime codec;
- external paper scores as contest evidence;
- PyPI assembly packages as dependencies;
- broad causal-learning language in paper/deploy claims.

Roadmap priority: below alpha mask redesign, beta component sensitivity, OWV3 byte-aware redesign, and exact-eval custody work. It can sharpen those lanes, but it does not replace them.
