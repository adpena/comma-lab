# External Optimizer Research Intake For Metalagrangian Archive Search

Date: 2026-05-02  
Author: Codex worker  
Evidence status: external motivation and local design only. No score claim. No GPU jobs dispatched.  
Frontier anchor for all comparisons: C067/Apogee, exact T4 A++ score `0.31561703078448233`, archive bytes `276214`, archive SHA-256 `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`.  
Canonical score source remains: `archive.zip -> inflate.sh -> upstream/evaluate.py`, preferably through `experiments/contest_auth_eval.py --device cuda`.

## Retrieval Metadata

All URLs below were retrieved on 2026-05-02.

1. KellerJordan/modded-nanogpt
   - Repository: https://github.com/KellerJordan/modded-nanogpt
   - Track 3 optimization: https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_3_optimization
   - Raw track-3 README: https://raw.githubusercontent.com/KellerJordan/modded-nanogpt/master/records/track_3_optimization/README.md
   - Raw main README: https://raw.githubusercontent.com/KellerJordan/modded-nanogpt/master/README.md
   - Raw current speedrun trainer: https://raw.githubusercontent.com/KellerJordan/modded-nanogpt/master/train_gpt.py
   - Raw track-3 trainer: https://raw.githubusercontent.com/KellerJordan/modded-nanogpt/master/records/track_3_optimization/train_gpt_simple.py
   - Git HEAD at retrieval: `bd1758ac36884edf537bbedc46e024d8507b3664` on `master`.
   - License: MIT per repository page.

2. Evolution Strategies at the Hyperscale / EGGROLL
   - Project page: https://eshyperscale.github.io/
   - PDF: https://eshyperscale.github.io/imgs/paper.pdf
   - PDF HTTP metadata at retrieval: `Last-Modified: Wed, 04 Mar 2026 16:07:18 GMT`, `ETag: "69a858b6-275a28"`, `Content-Length: 2578984`.
   - PDF SHA-256 at retrieval: `02d77e6be9353d5f650caaf771f488aa8aec8decc1f09dfb6db1528b302f4c9f`.
   - Project citation on page: Sarkar et al., "Evolution Strategies at the Hyperscale", arXiv/eprint `2511.16652`.
   - Code repository: https://github.com/ESHyperscale/HyperscaleES
   - HyperscaleES HEAD at retrieval: `b77f7d6f91238fd575313e946b9cad21e0a74b32` on `main`.
   - Nano-EGG repository: https://github.com/ESHyperscale/nano-egg
   - Nano-EGG HEAD at retrieval: `7c6da585c3dad43256b325d7b13e4cc27902cf94` on `main`.
   - License note: HyperscaleES and nano-egg are GPL-3.0. Use ideas and equations; do not copy code into Pact unless license compatibility is explicitly approved.

## What The Sources Actually Contribute

### modded-nanogpt, main speedrun

The main repo is a live, contest-style optimizer and systems benchmark for training a GPT-like model to a fixed validation target as fast as possible on 8xH100. Its durable contributions for Pact are not "LLM tricks" directly; they are a competition operating model:

- Every record is tied to a fixed target metric, fixed dataset stream rules, same-hardware timing, logs, and readable code.
- Recent speed comes from compounding many small algorithm, schedule, communication, dtype, data-loading, and kernel changes rather than one large magic method.
- The README's current technique list includes Muon/NorMuon, FP8 matmul for the head, softcapped/asymmetrically rescaled logits, zero-init projections, value/skip embedding structure, Flash Attention 3, long-short attention, sparse gates, cautious weight decay, batch/sequence length schedules, partial key offset, multi-token prediction, paired-head attention, fused kernels, sparse gradient communication, reduce-scatter, compile-aware optimizer layout, and logging of the exact code used.
- The main record table, at retrieved commit, reached `1.406` minutes for record #80 with paired-head Muon changes. This is external timing context only.

Contest mapping: treat archive search like their speedrun: a fixed metric, tight hardware/repro boundaries, small deltas, record-level custody, no silent rule drift, and no promotion from convenience proxies.

### modded-nanogpt, track 3 optimization

Track 3 deliberately separates "optimization algorithm quality" from wall-clock speed. It keeps data, batch size, and architecture fixed, then optimizes step count. It requires:

- no multiple forward/backward passes per step;
- no validation-driven early stopping;
- a statistical gate: `(3.28 - mu) * sqrt(n) >= 0.004` under the stated sigma assumption;
- full code included in the logfile, with third-party optimizer code copied into the submitted script rather than imported;
- results history that explicitly marks underpowered early runs and stronger n=5/n=10/n=20 evidence.

The current track-3 result table at retrieval emphasizes optimizer and hyperparameter discipline:

- AdamH and MuonH add a hyperball constraint on hidden matrices plus per-module init stds and cooldown choices.
- NorMuonH combines Muon Newton-Schulz direction, Adafactor-style row/column variance preconditioning, and a hyperball constraint, reaching `3250` steps with `n=10` evidence.
- The guidelines say weight decay is usually the most sensitive hyperparameter, then learning rate, then other knobs; short-run tuning may be useful for optimizer-specific knobs, but full-run retuning of weight decay and learning rate remains necessary.

Contest mapping: our equivalent is not val-loss significance but non-cherry-picked exact archive evidence. Track 3 supports turning atom-search claims into a benchmark with fixed archive/runtime custody, antithetic paired probes, repeated exact checks when variance matters, and full-code/manifest replay.

### EGGROLL paper and project page

The EGGROLL paper contributes a black-box optimizer architecture for non-differentiable or hard-to-backprop systems:

- Evolution Strategies can optimize arbitrary fitness functions, but naive full-matrix Gaussian perturbations are memory-bandwidth inefficient on GPUs.
- EGGROLL structures each matrix perturbation as a low-rank factor pair, rank `r`, so each population member can use a unique perturbation while preserving throughput close to normal batched inference.
- The paper frames a rank-`r` perturbation as `E = A B^T / sqrt(r)` and uses an approximate Gaussian score function. Individual perturbations are low-rank, but the population update can be high-rank.
- Antithetic perturbations, population-scale parallelism, noise reuse, and group/baseline-normalized fitness are central practical tools.
- Appendix arithmetic-intensity analysis argues that low-rank ES can saturate compute at large batch sizes, while naive Gaussian matrix ES stays bandwidth-bound.
- Experiments include reasoning fine-tuning and RL with explicit HPO discipline: Bayesian or random-search sweeps, then multiple seeds and mean/SEM reporting.

Contest mapping: EGGROLL is a strong template for optimizing our discrete archive compiler, where the "model" is a deterministic archive builder and the "fitness" is score or component-response proxy. Low-rank perturbations should be applied to atom-selection logits, rate multipliers, row/column foveation fields, pose basis weights, or decoder thresholds, not to scorer-side models.

### HyperscaleES and nano-egg code context

The related repos add implementation patterns, again as ideas only:

- HyperscaleES splits a `Noiser` from a `Model`. The noiser perturbs parameters, converts raw fitnesses, and updates parameters; the model owns initialization and forward execution.
- Its noiser class separates standard params, matmul params, embedding params, and excluded params. That is directly analogous to Pact's typed archive atoms: mask runs, pose deltas, renderer weights, latent bytes, and excluded custody fields.
- The EGGROLL noiser uses antithetic pairs, deterministic key folding, noise reuse, group-size normalized fitness, and optional batched update buckets for parameters with identical shape to reduce compiler overhead.
- The baseline-subtraction variant reserves group positions for zero-noise controls and computes fitness relative to that baseline.
- Nano-EGG's `QEggRoll` shows a quantized, integer-only evolution path with fixed random matrices, thresholded one-step parameter changes, antithetic paired fitness, and explicit population/group/tokens-per-update controls.

Contest mapping: the best immediate transfer is a deterministic `ArchiveNoiser`/`ArchiveModel` abstraction for archive atoms, with antithetic paired candidate construction and no-op controls. Do not import GPL code; implement the concepts independently if pursued.

## Mapping To Pact Pipeline

| External pattern | Pact equivalent | Existing hook |
|---|---|---|
| Fixed benchmark rules and record log | C067-anchored exact archive custody and dated research ledgers | `.omx/research/*`, `reports/latest.md`, `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/contest_auth_eval.json` |
| Full code embedded in logfile | Manifest captures generator source, commit, exact command, inputs, bytes, SHA, runtime tree | `experiments/contest_auth_eval.py`, builder `build_manifest.json` files |
| No early stopping on validation | No candidate chosen because one diagnostic looks lucky; promote only exact archive bytes | dispatch claim + exact eval artifacts |
| Statistical gate for repeated noisy runs | Paired +/- response curves, same-run zero baseline, duplicate exact eval if runtime custody differs | `experiments/profile_component_sensitivity_official.py --same-run-zero-baseline --require-passed` |
| Weight decay/LR sensitivity | Rate lambda, curvature strength, pair/frame antagonism, class synergy, atom count, body bytes | `experiments/plan_yousfi_fridrich_field_equations.py`, `experiments/optimize_component_response_stack.py` |
| Short-run tuning then full-run retune | Small candidate-size atom policy sweep, then full ledger retune of rate/class/foveation knobs | field-policy JSON -> CMG3A builder -> exact eval |
| Low-rank ES perturbations | Rank-1 perturbations of atom logits and low-dimensional policy fields | new additive planner or deterministic loop around field-policy weights |
| Antithetic paired samples | Build `+epsilon` and `-epsilon` archive variants against same baseline | `experiments/build_component_response_perturbation_plan.py` |
| Baseline subtraction / zero controls | `epsilon=0`, no-op payload checks, unchanged source SHA checks | response-plan zero point and archive validators |
| Shape-bucketed batched update | Batch archive candidates by same builder and same member layout | `build_cmg3_adaptive_runs_candidate.py`, `build_mixed_qzs_block_candidate.py`, `build_imp_c067_bridge_candidates.py` |
| GPL codebase | Idea-only intake unless license review approves code reuse | report/runbook only |

## Top 5 Immediately Actionable Ideas

The ranking is by expected score reduction per wall-clock minute, not by theoretical elegance. All "expected" numbers are EV/proxy judgments, not score claims.

### 1. Antithetic official component-response gate for the best existing C067 atom families

Category: diagnostic exact eval, then stack planning.  
Expected score reduction per wall-clock minute: highest, because candidate builders and many artifacts already exist.  
Source transfer: track-3 statistical discipline plus EGGROLL antithetic/baseline-subtracted fitness.

Rationale:

- Several C067 byte-screen families already exist: CMG3/CMG3A, PMG-hotspot, IMP-C067 bridge, blockfp/QBF1, fixed-slice segment mixes.
- Many failures are likely nonlinear scorer cliffs. One-sided or byte-only signals are too weak.
- A small paired `-eps/0/+eps` exact-response suite turns these into usable curvature and no-op evidence.

Concrete hooks and commands:

```bash
# Build deterministic +/- archive variants from an existing sensitivity artifact.
# First verify the artifact baseline SHA equals C067:
# 226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a
.venv/bin/python experiments/build_component_response_plan_from_sensitivity_artifacts.py \
  --sensitivity-artifact-dir <C067_MATCHING_SENSITIVITY_ARTIFACT_DIR> \
  --baseline-archive experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip \
  --baseline-contest-auth-eval-json experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/contest_auth_eval.json \
  --output-dir experiments/results/external_optimizer_intake_c067_response_plan_20260502 \
  --epsilon -2 --epsilon -1 --epsilon 0 --epsilon 1 --epsilon 2 \
  --max-archive-bytes 276214 \
  --max-archive-byte-delta 0

# Before any remote exact eval, claim the lane. Do not run without a non-conflicting claim.
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id extopt_c067_antithetic_response \
  --platform lightning \
  --instance-job-id <planned_job_id> \
  --agent codex:gpt-5.5 \
  --predicted-eta-utc <UTC> \
  --status eval \
  --notes "antithetic exact response; no score claim until contest_auth_eval json"

# Exact CUDA response evaluation, once claimed and scheduled on CUDA hardware.
.venv/bin/python experiments/profile_component_sensitivity_official.py \
  --baseline-archive experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip \
  --baseline-contest-auth-eval-json experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/contest_auth_eval.json \
  --perturbation-plan experiments/results/external_optimizer_intake_c067_response_plan_20260502/official_response_plan/official_component_response_plan.json \
  --output-dir experiments/results/external_optimizer_intake_c067_response_exact_20260502 \
  --device cuda \
  --same-run-zero-baseline \
  --require-passed

# Offline stack optimizer after exact response curves exist.
.venv/bin/python experiments/optimize_component_response_stack.py \
  experiments/results/external_optimizer_intake_c067_response_exact_20260502 \
  --output-json experiments/results/external_optimizer_intake_c067_response_exact_20260502/stack_optimizer.json \
  --archive-bytes-budget 276214 \
  --top-k 10 \
  --infeasible-top-k 10 \
  --max-stack-size 3
```

GREEN:

- The perturbation plan has paired nonzero epsilons and `epsilon=0`.
- `profile_component_sensitivity_official.py --require-passed` exits green.
- `contest_auth_eval.json` exists for each exact point, with CUDA, 600 samples, archive SHA/bytes, and recomputed score components.
- Stack optimizer recommends a candidate whose projected score is below C067 and whose archive bytes do not exceed 276214.

RED:

- One-sided response only, failed zero repro, no-op source reuse, CPU/MPS/proxy output, stale eval JSON, archive byte increase without component savings, or any missing dispatch claim before remote work.

### 2. EGGROLL-style low-rank perturbations over atom-policy fields, not over scorer models

Category: build-only first; diagnostic exact eval only after byte-screen.  
Expected score reduction per wall-clock minute: high if it finds a safe C067 trust-region candidate; low cost because it starts with existing atom ledgers.

Rationale:

- EGGROLL's rank-1 perturbation idea maps cleanly to our atom field, where a policy vector can be parameterized as low-rank factors over `(frame, row, class, pair/hardness)` rather than independent per-atom scalars.
- This should reduce arbitrary scalar sweeps and make candidate search more sample-efficient.
- Start with deterministic policies generated from existing ledgers; do not add random GPU search yet.

Concrete hooks and commands:

```bash
# Practical field-policy planner over existing C067 row-run atom ledgers.
.venv/bin/python experiments/plan_yousfi_fridrich_field_equations.py \
  --ledger-json experiments/results/c067_cmg3_rowspan_escape_atoms_20260502/stride1_dynamic_ego_foveal_atom_ledger.json \
  --ledger-json experiments/results/c067_cmg3_rowspan_escape_atoms_20260502/stride1_c067_trace_foveal_atom_ledger.json \
  --output-json experiments/results/external_optimizer_intake_yf_field_20260502/policies.json \
  --mode contest \
  --candidate-sizes 8,16,32,64,128,256 \
  --max-source-atoms 1024 \
  --interaction-model sparse_pair_frame_class \
  --positive-proxy-only \
  --policy-prefix extopt_yf_rank1

# Build byte-screen archives from the resulting explicit policies.
.venv/bin/python experiments/build_cmg3_adaptive_runs_candidate.py \
  --frontier-archive experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip \
  --decoded-mask-array experiments/results/c067_multimask_reconciliation_20260502/cmg3_rowspan_stride1.decoded_mask_array.npz \
  --output-dir experiments/results/external_optimizer_intake_yf_field_20260502/top0064 \
  --base-runs-per-row 1 \
  --field-policy-json experiments/results/external_optimizer_intake_yf_field_20260502/policies.json \
  --field-policy-id extopt_yf_rank1_top0064 \
  --compressor auto \
  --force
```

GREEN:

- Build manifests record `score_claim=false`, source policy SHA, no duplicate/unmatched atoms, matching `base_runs_per_row`, and deterministic archive bytes.
- Candidate archive is byte-neutral or byte-better versus C067, or has a documented component-repair hypothesis strong enough to justify paid diagnostics.
- Local decode/parity checks pass before any exact eval claim.

RED:

- Duplicate selected atoms, unmatched field atoms, base-run mismatch, negative field-energy policy without an explicit cliff-mapping note, or sampled body-budget search described as exact.

### 3. Convert modded-nanogpt logfile discipline into candidate manifests

Category: build-only / reproducibility hardening.  
Expected score reduction per wall-clock minute: high indirectly, by preventing invalid jobs and making fast continuation possible.

Rationale:

- modded-nanogpt's most transferable pattern is "the log contains the exact code needed to reproduce."
- Pact already has many builders, but fast-moving candidates can still lose source custody when produced by dirty worktrees or ephemeral remote scripts.

Concrete hook:

- For each new candidate builder or remote script, require its manifest to include: builder source SHA-256, exact command, input archive bytes/SHA, source policy bytes/SHA, runtime tree hash if exact-evaled, and `score_claim=false` until exact eval.
- Existing files to enforce/reuse: `experiments/contest_auth_eval.py`, `scripts/launch_lightning_batch_job.py`, `tools/claim_lane_dispatch.py`, builder `build_manifest.json` outputs.

Minimal command for candidate source audit:

```bash
shasum -a 256 \
  experiments/plan_yousfi_fridrich_field_equations.py \
  experiments/build_cmg3_adaptive_runs_candidate.py \
  experiments/optimize_component_response_stack.py \
  experiments/contest_auth_eval.py \
  tools/claim_lane_dispatch.py
```

GREEN:

- Every dispatchable candidate has replayable source and input hashes in JSON before remote work.

RED:

- Human log only, missing source hash, missing command, missing input archive SHA, or same archive SHA compared across different runtime trees without explicitly classifying it as runtime-custody evidence.

### 4. Short-run hyperparameter transfer: tune non-rate knobs on cheap policies, then retune rate on full policy

Category: build-only -> diagnostic exact eval.  
Expected score reduction per wall-clock minute: medium-high because it attacks the current arbitrary constant risk.

Rationale:

- Track 3's advice is directly applicable: tune stable structural knobs on shorter runs, then retune the most sensitive knobs at full length.
- Pact analogues:
  - "weight decay" -> rate lambda / target body bytes / max archive byte delta;
  - "learning rate" -> atom-count or perturbation epsilon;
  - other knobs -> pair antagonism, frame antagonism, class synergy, foveal weights, rank decay.

Concrete cheap sweep:

```bash
for curvature in 0.02 0.04 0.08 0.12; do
  for pair_antag in 0.0000005 0.000001 0.000002; do
    out="experiments/results/external_optimizer_intake_yf_sweep_20260502/c${curvature}_p${pair_antag}.json"
    .venv/bin/python experiments/plan_yousfi_fridrich_field_equations.py \
      --ledger-json experiments/results/c067_cmg3_rowspan_escape_atoms_20260502/stride1_dynamic_ego_foveal_atom_ledger.json \
      --output-json "$out" \
      --mode contest \
      --candidate-sizes 16,32,64 \
      --max-source-atoms 256 \
      --curvature-strength "$curvature" \
      --pair-antagonism "$pair_antag" \
      --positive-proxy-only \
      --policy-prefix "extopt_c${curvature}_p${pair_antag}"
  done
done
```

GREEN:

- The short sweep only chooses structural knobs. Full-size policies must retune body bytes / selected atom count before exact eval.

RED:

- Promoting a short-policy winner without retuning rate/atom count, or treating byte-screen/body-search proxy as exact optimizer output.

### 5. Batched shape-bucket candidate generation for archive builders

Category: build-only systems optimization.  
Expected score reduction per wall-clock minute: medium, by increasing candidate throughput under deadline.

Rationale:

- EGGROLL's batched-update bucket is a compiler lesson: group identical-shape operations so the compiler/runtime does less redundant work.
- Pact's equivalent is grouping archive candidates by same source archive, same member layout, same decoded mask array, same builder, and only varying policy fields.

Concrete hooks:

- `experiments/build_cmg3_adaptive_runs_candidate.py` for CMG3A policy variants.
- `experiments/build_mixed_qzs_block_candidate.py` for QZS block-size alternatives.
- `experiments/build_imp_c067_bridge_candidates.py` for IMP bridge sweeps.

GREEN:

- Build sweeps emit a single index manifest with all candidate archives, bytes, SHA, build command, and no score claims.
- Candidates are sorted by unchanged-distortion rate delta and explicit component-risk class before exact eval.

RED:

- Parallel build scripts that overwrite shared result dirs, omit source archive SHA, or hide no-op candidates.

## Top 5 Longer-Burn Ideas For 24h H100/H200/A100-Style Compute

### 1. ArchiveNoiser / ArchiveModel ES loop over differentiable-like atom fields

Category: training/search; exact eval only for selected archives.  
Compute shape: 24h H100/H200/A100, many build/proxy loops, sparse exact eval budget.

Design:

- Implement a Pact-native `ArchiveModel` that deterministically maps policy parameters to archive candidates.
- Implement an `ArchiveNoiser` that samples antithetic rank-1 perturbations over low-dimensional atom fields: frame basis, row basis, class basis, hard-pair basis, pose basis, and rate multiplier.
- Fitness hierarchy:
  1. byte/no-op/decode parity;
  2. component sensitivity projection;
  3. L40S/H100 diagnostic exact eval for Pareto candidates only;
  4. T4/equivalent promotion for identical bytes only.

Implementation hooks:

- New code should live in a new additive planner, not existing launchers.
- Reuse `experiments/plan_yousfi_fridrich_field_equations.py`, `experiments/build_cmg3_adaptive_runs_candidate.py`, `experiments/optimize_component_response_stack.py`.
- All selected archives must remain deterministic and byte-closed.

Dispatch rule:

- GREEN only if the selected archives already pass local build/parity and a non-conflicting lane claim exists.
- RED if the ES loop directly consumes scorer outputs without archive custody or uses any sidecar at inflate time.

### 2. NorMuonH-style constrained optimizer for trainable codec/residual lanes

Category: training.  
Compute shape: 24h H100/H200/A100.

Design:

- For future learnable residual, SegMap, SJ-KL successor, or renderer self-compression training, test hyperball constraints on hidden/residual matrices plus per-module init stds.
- Treat rate/distortion as a constrained optimizer problem: keep hidden update norms inside a trust region and decay/retune rate pressure late.
- Do not touch active SJ-KL v2 state.

Implementation hooks:

- `src/tac/experiments/train_renderer.py`
- `src/tac/training.py`
- `src/tac/profiles.py`
- future additive profile only; old behavior default unchanged.

Dispatch rule:

- GREEN if focused tests prove opt-in profile wiring, checkpoint manifest records optimizer code/hparams, and exact eval is on a built archive.
- RED if it changes default training behavior, loads scorers at inflate time, or cannot export a deterministic archive.

### 3. Component-response benchmark suite modeled after track 3

Category: diagnostic exact eval plus research infrastructure.  
Compute shape: 24h mixed CUDA.

Design:

- Create a fixed response benchmark over C067:
  - fixed baseline archive and runtime tree;
  - fixed perturbation basis;
  - paired epsilons;
  - same-run zero baseline;
  - no early stopping or cherry-picking;
  - report mean/variance over repeated exact evals if any stochastic runtime variance appears.
- Use it as the "optimizer track" for archive atom policies.

Implementation hooks:

- `experiments/build_component_response_perturbation_plan.py`
- `experiments/profile_component_sensitivity_official.py`
- `experiments/optimize_component_response_stack.py`
- `scripts/launch_lightning_batch_job.py`

Dispatch rule:

- GREEN if exact CUDA response artifacts pass `--require-passed` and every point has archive SHA/bytes.
- RED if curves are built from proxy maps alone or from missing-custody harvested logs.

### 4. Low-rank pose and renderer active-subspace search

Category: training/search -> diagnostic exact eval.  
Compute shape: 24h H100/H200/A100.

Design:

- Use component traces and scorer-weighted pose atoms to fit a low-dimensional basis for pose deltas and renderer local changes.
- Search low-rank antithetic combinations rather than independent per-pair tweaks.

Implementation hooks:

- `experiments/plan_scorer_weighted_pose_atoms.py`
- `experiments/plan_pose_manifold_waterfill_candidates.py`
- `experiments/line_search_pose_refinement.py`
- `experiments/contest_component_trace.py`

Dispatch rule:

- GREEN if local plan records charged bytes per atom, target pair/frame/class, expected break-even, and exact archive build command.
- RED if pose regeneration decodes more frames than the contest window, omits `--skip-proxy-score` when proxy is irrelevant, or uses a non-reviewed renderer loader.

### 5. Integer/quantized search for decoder/runtime knobs

Category: build-only -> training/search.  
Compute shape: 24h, mostly build/proxy plus selected exact eval.

Design:

- Nano-EGG's quantized thresholded updates suggest a way to search integer knobs cheaply: QZS block sizes, quantizer scales, foveation radii, residual thresholds, class LUT entries, and entropy packer modes.
- Use thresholded one-step updates: a knob changes only if paired normalized fitness exceeds a preset confidence threshold.

Implementation hooks:

- `experiments/build_mixed_qzs_block_candidate.py`
- `experiments/build_qzs3_postprocess_candidate.py`
- `experiments/build_blockfp_c067_archive.py`
- `src/tac/quantizr_qzs3_codec.py`
- `src/tac/qbf1_renderer_codec.py`

Dispatch rule:

- GREEN if the raw payload changed, decoded magic validates, runtime consumes the new bytes, and archive bytes are Pareto-improving before exact eval.
- RED if it is a no-op, format-wrapper-only change, malformed ZIP reliance, or parser divergence dependency.

## Failure Modes And Contest-Compliance Risks

1. External optimizer overclaim
   - Risk: treating MuonH, NorMuonH, or EGGROLL as evidence that a Pact archive improves.
   - Guard: all external results are motivation only. Pact score truth is exact CUDA auth eval of the exact archive bytes.

2. License contamination
   - Risk: copying GPL-3.0 HyperscaleES/nano-egg code into Pact.
   - Guard: reimplement concepts independently or get license approval. Do not paste GPL implementation.

3. Proxy overfitting
   - Risk: low-rank ES optimizes sensitivity maps, byte deltas, or L40S diagnostics and then fails T4.
   - Guard: record proxy artifacts as planning-only; T4/equivalent exact eval is required for promotion.

4. No-op controls missing
   - Risk: packer changes container metadata but scored runtime ignores it.
   - Guard: require payload changed, decode parity, runtime apply log, member SHA changes, and same-run zero controls.

5. Runtime custody mismatch
   - Risk: identical archive SHA scores differently under changed repo-local inflate Python.
   - Guard: compare runtime tree SHA from `contest_auth_eval.py`; classify mismatches as runtime-custody evidence, not archive evidence.

6. Active-lane conflict
   - Risk: duplicating guarded SJ-KL v2 or Q-FAITHFUL work.
   - Guard: no edits to active lane files/state; every remote job must first claim with `tools/claim_lane_dispatch.py claim`.

7. Antithetic search violates archive closure
   - Risk: ES loop reads external sidecars or mutates runtime state outside `archive.zip`.
   - Guard: archive builder must charge every score-affecting byte and manifest payload closure.

8. Early stopping / cherry picking
   - Risk: running many diagnostics and promoting only a lucky one without recording negatives.
   - Guard: pre-register candidate set, include negatives, and record exact archive SHA/bytes and command for every run.

9. Sample count or hardware drift
   - Risk: scoring on partial samples, CPU/MPS, or diagnostic GPU and treating it as A++.
   - Guard: require CUDA, 600 samples, T4/equivalent for promotion, and recomputed formula.

10. Unbounded candidate sweeps
    - Risk: random/grid sweeps consume deadline wall-clock without exact-eval likelihood.
    - Guard: stop unless byte screen, component-risk reason, and exact eval budget are all explicit.

## Exact Implementation Hooks In This Repo

Do not invent flags. The current parser surfaces support the following:

- `experiments/plan_yousfi_fridrich_field_equations.py`
  - Real flags include `--ledger-json`, `--output-json`, `--mode`, `--max-source-atoms`, `--candidate-sizes`, `--interaction-model`, `--curvature-strength`, `--pair-antagonism`, `--frame-antagonism`, `--class-synergy`, `--low-rank-modes`, `--positive-proxy-only`, `--allow-negative-field-energy`, `--policy-prefix`.
  - Use for field-policy JSON generation. Output is planning-only.

- `experiments/build_cmg3_adaptive_runs_candidate.py`
  - Real flags include `--frontier-archive`, `--decoded-mask-array`, `--output-dir`, `--target-extra-runs`, `--target-body-bytes`, `--base-runs-per-row`, `--adaptive-max-runs-per-row`, `--compressor`, `--hard-frame-indices`, `--hard-pair-indices`, `--class-weights-json`, `--hard-frame-multiplier`, `--foveal-row-weight`, `--foveal-col-weight`, `--boundary-detail-weight`, `--rank-decay`, `--body-search-mode`, `--field-policy-json`, `--field-policy-id`, `--force`.
  - Use for concrete CMG3A archives from field policies. Output is not score evidence.

- `experiments/build_component_response_perturbation_plan.py`
  - Builds deterministic archive variants for official response curves. It is scorer-free and non-promotional.

- `experiments/build_component_response_plan_from_sensitivity_artifacts.py`
  - Real flags include `--sensitivity-artifact-dir`, `--baseline-archive`, `--baseline-contest-auth-eval-json`, `--output-dir`, `--epsilon`, `--max-mutated-bytes`, `--max-abs-byte-delta`, `--max-raw-l1-delta`, `--max-archive-bytes`, `--max-archive-byte-delta`.
  - Use to attach sensitivity projections before exact response eval.

- `experiments/profile_component_sensitivity_official.py`
  - Real flags include `--baseline-archive`, `--baseline-contest-auth-eval-json`, `--perturbation-plan`, `--output-dir`, `--contest-auth-eval-script`, `--inflate-sh`, `--upstream`, `--video-names-file`, `--device cuda`, `--inflate-timeout`, `--evaluate-timeout`, `--max-relative-error`, `--zero-repro-tolerance`, `--min-observed-delta`, `--allow-directional`, `--same-run-zero-baseline`, `--require-passed`.
  - Use for official exact CUDA component-response curves.

- `experiments/optimize_component_response_stack.py`
  - Real flags include input paths, `--output-json`, `--archive-bytes-budget`, `--max-posenet-dist`, `--max-segnet-dist`, `--top-k`, `--infeasible-top-k`, `--max-stack-size`, `--max-enumerated-stacks`, `--allow-calibration-inputs`.
  - Use for deterministic offline stack planning. It never claims composition without a stacked exact eval.

- `experiments/contest_auth_eval.py`
  - Real flags include `--archive`, `--inflate-sh`, `--upstream-dir`, `--video-names-file`, `--device`, `--work-dir`, `--inflate-timeout`, `--evaluate-timeout`, `--keep-work-dir`, `--expected-runtime-tree-sha256`.
  - Use for exact CUDA score truth.

- `tools/claim_lane_dispatch.py`
  - Real flags include `claim`, `--claims-path`, `--lane-id`, `--platform`, `--instance-job-id`, `--agent`, `--predicted-eta-utc`, `--status`, `--notes`, `--ttl-hours`, `--now-utc`, `--allow-parallel`, `--child-of`, `--parallel-reason`, `--force`, `--dry-run`.
  - Use before any training, eval, or remote-GPU dispatch.

## GREEN / RED Dispatch Rules

### Build-only candidate

GREEN:

- New archive is deterministic, byte-closed, zip-safe, and has a manifest with source archive SHA/bytes, builder source hash, command, input policy SHA, and `score_claim=false`.
- Raw payload bytes changed in the intended member, or the candidate is explicitly marked as a no-op control.
- Decode/inflate local smoke proves the runtime consumes the new payload when that is the hypothesis.

RED:

- Missing manifest, no-op disguised as candidate, hidden sidecar, resource fork, duplicate ZIP member, malformed ZIP reliance, uncharged sidecar, or source archive mismatch.

### Diagnostic exact eval

GREEN:

- Non-conflicting active dispatch claim exists.
- Candidate passed build-only gates.
- Evaluation uses CUDA and canonical `archive.zip -> inflate.sh -> upstream/evaluate.py`.
- Output includes structured `contest_auth_eval.json`, archive bytes/SHA, recomputed components, logs, and runtime tree hash/provenance.
- L40S/H100/A100 diagnostics are labeled diagnostic unless the promotion standard explicitly accepts them.

RED:

- CPU/MPS/proxy, missing JSON, human-log-only score, stale job, missing claim, incomplete sample count, or exact eval run against a different archive/runtime than the manifest.

### T4 promotion

GREEN:

- Exact T4/equivalent CUDA eval on identical archive bytes.
- `n_samples=600`.
- `score_recomputed_from_components < 0.31561703078448233`.
- Component gates passed; no PoseNet/SegNet collapse.
- Archive bytes/SHA match manifest and runtime tree hash is recorded.
- If archive uses external public-source segments, attribution and compliance notes are present.

RED:

- Score improvement only on L40S/H100 diagnostic, rate-only formula below frontier with worse components, runtime tree mismatch unresolved, or exact eval lacks payload closure.

### Training lane

GREEN:

- Additive profile or script; old behavior unchanged.
- No scorer load at inflate time.
- Deterministic checkpoint/export manifest with hparams, seeds, source hashes, selected tensors, byte accounting, and exclusion reasons.
- Built archive exact-evaled before score language.

RED:

- Default behavior changes, hidden dependencies, non-deterministic export, sidecar checkpoint use, MPS fallback for score, or active SJ-KL/Q-FAITHFUL state touched.

## Adversarial Review Of This Report

1. The most likely overreach is mapping neural optimizer results onto an archive codec contest. Mitigation: every recommendation is framed as proposal-generation or methodology, not score evidence.

2. EGGROLL's throughput claims depend on large matrix multiplications and very large populations. Pact archive builders are not the same workload. The transferable part is low-rank antithetic perturbation and baseline subtraction, not the headline speedup.

3. Track-3 statistical significance is not directly applicable to a deterministic contest evaluator. The useful transfer is anti-cherry-pick discipline, full code custody, paired controls, and repeated exact checks only where runtime variance or custody mismatch appears.

4. The field-policy commands are build-only and could still produce destructive scorer cliffs. Exact CUDA eval remains required, and build-only byte wins cannot promote.

5. HyperscaleES/nano-egg are GPL-3.0. Copying implementation would be risky; this report recommends only independent reimplementation of abstract ideas.

6. Active SJ-KL v2 is guarded and must not be duplicated. This report treats SJ-KL only as future training/search context, not as an instruction to touch its live state.

7. The current C067 archive path is named `exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z`, but local artifacts identify it as the byte-identical C067/Apogee anchor by SHA and bytes. Any future command should re-check SHA before use.

## Implementation Decision

No implementation was warranted in this worker pass. The allowed new code scope would be useful for a future `ArchiveNoiser` or external-optimizer intake helper, but adding it now would duplicate existing field-policy and response-curve tools before a specific lane has been chosen. This pass therefore leaves only this report and no GPU dispatch.
