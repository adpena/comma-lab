# DDM DIO1 — Dion3 crosswalk (2026-08-14)

**Disposition:** paper/repository triage complete; no training or scorer job launched; no candidate archive materialized; no frontier movement claimed.  
**Claim axes:** paper claims are reported on their stated GH200/PyTorch/FSDP surfaces. Local timing is either a real-run receipt or is explicitly labelled a CPU toy bracket for MLX/Metal.  
**Pins:** arXiv [`2608.11612v1`](https://arxiv.org/html/2608.11612v1); official repository [`microsoft/dion`](https://github.com/microsoft/dion), HEAD resolved 2026-08-14 as `71dc64f9fde8c3ee8643431390a16698bf32396f`; current effective-frontier archive SHA-256 `f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de`.

## Bottom line

Dion3's advertised optimizer-step acceleration addresses a different wall: large LLM matrices, PyTorch on NVIDIA GH200, and in some cases FSDP communication. The current Pact Muon surfaces are six small matrices on one MLX/Metal host. The exact standard-NS5 work is only 9.689 GFLOP per old-V9 epoch or 1.789 GFLOP per current TR1 epoch. A deterministic CPU toy bracket projects 0.027% and 0.062% of the corresponding real epoch medians. Direct MLX/Metal isolation was **not measured** because this sandbox exposes no Metal device. Even allowing for that boundary, the current TR1 implementation would need to be over 81 times slower than the CPU toy bracket to reach the 5% materiality gate; the old-V9 threshold is over 182 times. Gram-NS, CuteDSL symmetric kernels, CUDA graphs, and distributed megabatching are therefore **N-A / CURRENT-SCALE** as throughput work.

The transferable idea is Dion3's row-subset update with momentum error feedback. It is not a speed claim here and it is not an imported constant. It is an **ADOPT-CLASS / SHADOW-ONLY** optimizer treatment for the already-built, never-fired WP1/JD1 Muon finisher. Its first test is a scorer-free replay over retained real gradients and momentum, with actual update-RMS matching as the authority. It has no consumer in RX2, whose live work is fixed-token rate rather than gate-aware distortion. It may be folded into JS1/#982 only if that existing treatment later exposes eligible 2-D optimizer blocks and a retained gradient/momentum capture surface.

## Claim triage

| Dion3 claim | What the paper/repository actually measures or implements | Pact-scale adjudication | Consumer and falsifier |
|---|---|---|---|
| Up to about 6.5× faster optimizer steps than Muon | Optimizer-step-only benchmarks, excluding forward/backward, on 1/4 NVIDIA GH200 GPUs, roughly 1B–14B model scales. The paper itself estimates Newton–Schulz at about 1%–17% of end-to-end training depending on scale. | **N-A / CURRENT-SCALE**. It is not a Pact end-to-end speed prediction. Our matrices and single-device execution remove the cubic and communication walls that make the headline possible. | No build. Reopen only if a real MLX profile places polar work at or above 5% of epoch time or the minimum eligible matrix dimension grows materially. |
| Gram Newton–Schulz cuts orthogonalization FLOPs | Exact-arithmetic reformulation around a Gram matrix. The paper also documents half-precision instability from spurious negative eigenvalues/eigenvector drift and adds reset/dtype discipline. The package routes this through a separate `gram_newton_schulz` dependency. | **LESSON-ONLY / CURRENT-SCALE**. Fewer formal FLOPs do not imply a faster small-matrix Metal kernel, and the paper does not establish better small-matrix numerical behavior than our standard NS5. | Burn-4/TR1 Muon and `tac` MLX optimizer stack only if the 5% gate fires. Falsified for speed at present by the bound below; any future implementation must also match polar residual and update direction. |
| Symmetry-aware CuteDSL kernels provide another speed layer | Custom NVIDIA kernels aimed at Hopper/Blackwell; current repository setup includes NVIDIA-specific optional dependencies. | **N-A** for MLX/Metal and **N-A** for Modal T4, which is an evaluation lane rather than a training consumer. | None. A future NVIDIA training lane would need its own owner, hardware receipt, and deterministic resume contract. |
| Row-subset orthogonalization improves Dion | Dion3 selects rows by top L1 momentum magnitude, orthogonalizes the selected submatrix, updates selected rows, and decays only selected momentum rows so omitted information remains as error feedback. Fractions below 1 change the optimizer trajectory. At fraction 1 the method matches Muon only in exact arithmetic; finite precision still matters. Paper convergence comparisons are LLM training runs, not Pact. | **ADOPT-CLASS / SHADOW-ONLY** as a stochastic subspace treatment, not as a speed import. No fraction is adopted. | Primary: WP1/JD1 Muon-finisher owner. Conditional: JS1/#982. Falsifier: after update-RMS calibration, no subset fraction enters the incumbent A/A update-direction band while preserving bounded starvation, polar quality, and deterministic replay. |
| Scale learning rate by `1/sqrt(fraction)` | An analytic starting normalization justified by the selected update's Frobenius norm, then tested in the paper's LLM setting; `f=1/8` and `f=1/4` results are task-specific. | **ADOPT as an initialization prior only**. The existing px1 actual-update receipt remains authority; no Dion fraction or learning rate transfers directly. | Canonical optimizer methodology, folded into the existing update-RMS law. Falsifier: measured full-tensor update RMS or clipping/saturation disagrees with the analytic prior. |
| Megabatching reduces communication | Same-shape parameter matrices are grouped to reduce FSDP all-to-all rounds. The reported large gain is a specific communication-bound 1B/eight-shard case; other cases benefit little. | **N-A** for the single-host MLX line. This is distinct from `--micro-batch-pairs`, which batches forward/backward examples and remains the only locally plausible large batching lever. | No new consumer. Falsifier for reopening: a profile must show optimizer launch scheduling, rather than model forward/backward, as a material wall. |
| Dion is a drop-in Muon replacement | The repository targets PyTorch ≥2.7 and DTensor/DDP/FSDP2. `Dion3` currently aliases the row-subset `NorDion2` implementation, with Torch compilation and optional Triton/CUDA-graph/NVIDIA kernel paths. | **N-A as a package**. The algorithm can be re-expressed in MLX, but the package is not reusable in the current trainer. | Port only the minimal selector/error-feedback state after a shadow win. Do not port distributed or NVIDIA machinery. |
| Comparable-or-better loss | Demonstrated on the paper's language-model training tasks and budgets, with selection fraction and learning-rate tuning. | **LESSON-ONLY**. It does not predict d_seg, d_pose, rate, or exact contest score. | Any Pact adoption requires an existing governed treatment and exact receiver/public-evaluator closure after scorer-free preflight. |

## Our-scale NS receipt

### Real trainer surfaces

The old V9 Muon surface in `muonh_manifold_muon_dig_20260713.md` contains six matrices: `96×80`, `768×19`, and four `96×96` matrices (59,136 of 87,575 parameters). The current TR1 lotto renderer checkpoint flattens six convolution weights to `24×108`, `3×216`, and four `24×216` matrices. Tokens, biases, and gains remain on Adam in the existing WP1 finisher design.

For standard five-iteration Newton–Schulz, counted as the actual three GEMMs per iteration after orienting each matrix with the shorter side as rows:

| Surface | Exact GEMM FLOP/update | Updates/epoch | Exact GEMM FLOP/epoch | Real epoch reference | Time that would equal 5% |
|---|---:|---:|---:|---:|---:|
| Old V9 six-matrix Muon | 129,189,870 | 75 | 9,689,240,250 | 169.7 s median `[macOS-MLX research-signal]`, #306 | 8.485 s/epoch; 113.13 ms/update |
| Current TR1 lotto six-matrix candidate | 11,927,790 | 150 | 1,789,168,500 | 85.0926 s median `[macOS-MLX research-signal]` | 4.2546 s/epoch; 28.3642 ms/update |

The TR1 epoch reference is **MEASURED** from 108 adjacent epoch intervals, epochs 1767–1875, in `/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/tr1_jd4_cont_ep1766_q3on/telemetry.jsonl` (SHA-256 `eff15166d36d2a8fc78f406033d63fc6475cc88f8a22e5726f3eff3ab901a03f`). Its q05/q50/q95 interval is 84.5781 / 85.0926 / 114.6589 seconds. Every row reports `jd1_finisher: off`; this is an epoch denominator, not a direct Muon timing.

The old V9 real run switched to six Muon parameters at epoch 726. Its before/after epoch timing shows no visible increase, but it is not an isolating measurement because the stage also changed and its bootstrap interval is wide. It is supporting absence-of-a-large-jump evidence only, not the NS fraction receipt.

### Deterministic toy bracket

A fresh deterministic Torch-fp32/one-CPU-thread benchmark used seed `20260814`, 25 warmups, and 400 repetitions of the exact standard-NS5 shape sets. No candidate/video payload was created. This is **TOY-BRACKET `[macOS-CPU advisory; not MLX/Metal]`**, not a trainer measurement.

| Surface | All-six q05/q50/q95 | Median projected per epoch | Fraction of real epoch median | Determinism receipt |
|---|---:|---:|---:|---|
| Old V9 | 0.6103 / 0.6210 / 0.6362 ms | 0.0466 s | 0.0274% | input SHA `354cb57c3ea36ecf78be9ba8ff47d85f0f49ec17706339eabec49ecf005da182`; output/repeat SHA `85d67ecc6f3126816cfc7f2642ffe4dbbea053b018b916e7fbb528d91f30a702` |
| TR1 lotto | 0.3434 / 0.3490 / 0.3606 ms | 0.0524 s | 0.0615% | input SHA `98721f2bfd2ceb944c5c8965c4c3ac268382cd228e7d0e52ea0172ec07d2e7a8`; output/repeat SHA `411c1cb5d6aa36620913886ed148a7d5382a5b06cd98655bfe54d83b0302e9bf` |

All generated inputs, outputs, bit-identical repeat outputs, and raw timing vectors are retained under `/Volumes/VertigoDataTier/pact/ddm_dio1_20260814/retained/`. The machine-readable receipt is `ns5_toy_bracket_receipt.json`, 3,178 B, SHA-256 `85e5ab624e34be6acb0e79b4ee96c2e0a502c9ae6056a9707f4dff39788e4cb4`; the retained benchmark source is 3,918 B, SHA-256 `49ceb538cd8addd93e67a0a076ec706602408543687ffea21d4f72adff14c686`.

A direct MLX benchmark was attempted but **NOT MEASURED**: MLX initialization failed with `RuntimeError: [metal::load_device] No Metal device available`. The resulting bound is therefore deliberately conservative in wording: TR1 standard NS5 would have to take at least 28.36 ms per all-six update to cross the 5% gate, versus a 0.3490 ms CPU toy median, a factor above 81 for the update itself. On the old V9 surface the corresponding ratio is above 182. These ratios do not prove Metal time; they show how large the unmeasured platform reversal would have to be.

This agrees with the earlier #443 receipt: its current-scale estimate put Muon NS at no more than about 1 GFLOP/epoch on that older schedule and rejected custom NS kernels; it measured local GPU steps around 407–414 ms under load with only about 0.6% Python overhead. The fresh exact counts update the denominator without changing the decision.

## Transfer crosswalk and ranked verdicts

1. **ADOPT — actual update-RMS matching as the fairness authority.** Dion3's analytic fraction normalization reinforces, but does not replace, px1's `pact.optimizer_update_scale_receipt.v1`: record full-tensor candidate and reference update RMS per group, fit a robust scale ratio, and preserve checkpoint, gradient, polar, reset, replay, and resume identities. Consumer: existing canonical optimizer-methodology laws. Falsifier: the analytic scaling changes selection/clipping behavior or cannot enter the incumbent A/A band.

2. **ADOPT-CLASS / SHADOW-ONLY — top-L1 row selection with error feedback.** This could change the optimization trajectory by concentrating a step on the currently largest-momentum rows while retaining omitted momentum. At Pact scale, that trajectory effect is the only plausible value; speed is not. Consumer: the built-but-never-fired WP1/JD1 Muon finisher. Falsifier: no fraction/control meets update-direction, starvation, polar, and replay gates at matched update RMS.

3. **LESSON-ONLY — stable Gram-NS reset/dtype discipline.** The finite-precision failure analysis is valuable if a future polar-quality receipt fails, but GNS is not a current speed task. Consumer: future `tac` MLX polar implementation only after the 5% gate. Falsifier: current standard NS already meets residual/replay requirements and stays below the wall.

4. **N-A — symmetric NVIDIA kernels, CUDA graphs, and FSDP megabatching.** Their hardware/runtime/communication surfaces are absent. `--micro-batch-pairs` is not evidence for optimizer megabatching; it attacks the forward/backward denominator instead.

5. **N-A — direct package reuse.** A narrow MLX selector/error-feedback implementation is cheaper and safer than adapting a Torch/DTensor package whose acceleration layers cannot run here.

## #552 / #556 geometry boundary

Dion3 selects rows of an ambient matrix `W`. The #552 line instead factors `W=QH`, assigns separate momenta and retractions to the orthogonal factor `Q` and SPD factor `H`, and records a metric with a cross term. The exact differential/pullback receipt in `matrixcalc_18s096_crosswalk_20260720T180108Z.md` makes this a real geometry choice, not an interchangeable implementation detail.

An ambient row mask does not automatically preserve the tangent constraints of `Q`, positive-definiteness of `H`, or the chosen product metric. Applying Dion3 independently to factor rows would therefore be a new optimizer, not a valid composition of two settled ideas. #556 `FilmPolarSPDNormalMomentum` remains **DEFERRED under its existing terminal-solve owner**. It may reopen only with current code/objective equivalence plus a chart-specific selector, pullback, retraction, metric ID, factor-condition receipt, and exact `W` reconstruction. No new #552/#556 task row is created by this memo.

## Named-consumer routing

| Named surface | Disposition | Reason |
|---|---|---|
| Burn-4/TR1 Muon stage | **QUEUED-WITH-A-FIRE-ORDER** for scorer-free shadow replay only | WP1 already built a default-off renderer-only Muon finisher with NS5 and additive resume state, but current JD4/JD6 telemetry shows it never fired. This is the smallest honest consumer. |
| JS1/#982 joint gate-aware treatment | **FOLDED-INTO-EXISTING-OWNER, CONTINGENT** | Dion3 does not solve the representation/gate problem. The optimizer class becomes testable only after #982's existing no-scorer identity/price/resume preflight exposes eligible 2-D blocks and retained gradients/momentum. |
| RX2 successor training rounds | **FOLDED / NO CONSUMER** | RX2 is a fixed-token rate harvest, not a gate-aware distortion training engine. Routing a Dion treatment there would invent an owner and duplicate the live plan. |
| Canonical equations optimizer laws | **FOLDED** | Existing Muon-final-stage, switch-condition, and px1 update-RMS laws already cover the admissible methodology. No equation or constant changes without a measured receipt. |
| #469 MuonH and #552/#556 product chart | **FOLDED / EXISTING DEFER UNCHANGED** | Ambient row selection is not chart-safe by construction. |

## $0 probe designs

No probe below authorizes a training or scorer launch.

### P1 — TR1/WP1 row-subset shadow replay

- **Input gate:** a retained, scorer-free sequence of real per-step gradients plus pre-step momentum for the six renderer matrices, with checkpoint SHA, optimizer state, seed, step IDs, and exact flatten/unflatten mapping. Existing stage checkpoints are insufficient by themselves; no such per-step trace was found in the searched stores.
- **Treatments:** full-row incumbent A/A repeat; deterministic top-L1 selection at fractions 1, 1/2, 1/4, and 1/8; deterministic random-row controls at the same fractions. Simulate selected-row momentum decay/error feedback without mutating training weights.
- **Retained outputs:** every treatment's full-tensor update trace, selector indices, momentum state, polar residual, update RMS/cosine, per-row selection interval, starvation maximum, wall time, and repeat hash. Persist every candidate, not only the winner, under `/Volumes/VertigoDataTier/pact/ddm_dio1_20260814/retained/` with bytes and SHA-256.
- **Pass:** at least one subset treatment enters the measured incumbent A/A band for full-tensor update RMS/direction, maintains complete bounded row coverage, matches polar-quality tolerance, and repeats byte-identically.
- **Falsifier:** no fraction beats its random control or none enters the incumbent band without starvation/polar/replay failure. That closes this treatment on the current WP1/TR1 renderer formulation.

### P2 — update-RMS calibration

- Use px1's real-update receipt, with the analytic inverse-square-root fraction scale only as the first trial value.
- Fit the candidate/reference per-group RMS ratio from the shadow trace; do not use paper loss curves or nominal learning rate as equivalence evidence.
- **Falsifier:** calibrated scaling requires clipping/saturation, changes selector ordering discontinuously, or still misses the A/A band. In that case the analytic normalization is not transferable to this optimizer state.

### P3 — conditional stable Gram-NS bracket

- **Fire only if** a direct real-trainer profile shows incumbent polar work at or above 5% of epoch time, or a future eligible matrix's short dimension grows enough to cross the same budget.
- Compare standard NS5 against the paper's stable reset/dtype Gram form on identical real matrices. Retain outputs and report wall time, polar residual, update cosine/RMS, dtype, reset points, repeat hash, and peak memory.
- **Falsifier:** standard NS remains below 5%, or Gram-NS misses residual/determinism parity. At current scale this probe is **FOLDED, NOT FIRED**.

### P4 — conditional #982 optimizer shadow

- Reuse P1 only after the existing #982 owner completes its no-scorer preflight and exposes eligible 2-D blocks plus retained gradients/momentum. Do not add a separate training lane.
- **Falsifier:** #982 has no eligible matrix surface, its resume state cannot preserve selector/error-feedback state additively, or the shadow receipt fails P1.

## MLX port price

These are planning estimates, not measured implementation times.

- Minimal deterministic MLX top-L1 selector, selected-row NS call, error-feedback momentum state, additive checkpoint migration, typed-DSL config, telemetry, and replay tests: **4–7 engineer-days**, contingent on P1 winning in shadow.
- Stable Gram-NS math/parity prototype with reset/dtype controls and a direct Metal timing receipt: **1–2 engineer-days**, but it has no current fire trigger because the throughput wall is absent.
- Full repository feature parity, including distributed megabatching, Torch/DTensor behavior, Triton/CuteDSL kernels, or CUDA graph capture: **not priced for implementation** because Pact has no consumer for those surfaces.

## What was measured, what was not

**MEASURED:** the current TR1 real epoch denominator; exact shape-level standard-NS5 GEMM counts; deterministic CPU toy timing; current checkpoint/telemetry optimizer routing; repository implementation surfaces; current frontier and own-vehicle pointers from live custody records.

**NOT MEASURED:** direct MLX/Metal NS wall time; GNS speed or numerical parity on current matrices; any Dion3 convergence effect on Pact; any d_seg/d_pose/rate effect; any exact contest score; any #982 transfer. No scorer, trainer, archive build, or paid job ran.

**BOUNDARY:** the throughput negative is **INSTANCE/CURRENT-SCALE** for the present six-matrix single-host MLX surfaces. It is not a family-wide claim that Gram-NS or custom polar kernels can never matter.

## RECALL EVIDENCE

The recall pass searched beyond the charter seeds before adjudication.

- Full-corpus content searches under `.omx/research`, `.omx/state`, and the sub-0.15 DAG used queries covering `Dion|Gram Newton|Newton.Schulz|row.subset|orthogonal|Muon|muon|optimizer_update_scale_receipt|FilmPolar|SPD.*momentum|micro.batch.pairs|jd1.finisher|#982|RX2`. Task-ledger and hot-state searches resolved current ownership and prevented a new RX2 or #556 row.
- The canonical registry was inspected with `.venv/bin/python tools/list_canonical_equations.py --json`. Beyond the charter seeds it located `pr95_family_l15_muon_optimizer_final_stage_only_v1`, `muon_finisher_schedule_warmstart_and_lr_anneal_v1`, and `muon_switch_conditioning_criterion_v1`. **Plan change:** fold update scaling into existing optimizer laws; do not create or edit a canonical equation.
- `ddm_wp1_20260805/RECEIPT.md` showed that a default-off JD1 Muon finisher already exists, with renderer tensors on Muon, tokens/biases/gains on Adam, warm-started momentum, NS5, and additive resume handling. Current JD4/JD6 telemetry shows `jd1_finisher: off`. **Plan change:** make this built-but-unfired surface the first shadow consumer rather than designing another trainer.
- `spd_submanifold_momentum_20260719_codex.md` plus `matrixcalc_18s096_crosswalk_20260720T180108Z.md` established the separate `Q/H` product geometry and exact pullbacks. **Plan change:** reject direct ambient-row masking of #552/#556 and preserve #556's existing defer.
- `kernel_stack_sweep_443_20260711.md` and `per_lever_compute_audit_20260705.md` already showed a forward/backward-dominated local wall and rejected custom small-matrix NS work. **Plan change:** perform a fresh exact shape count and toy bracket, but make 5% a fail-closed build gate rather than porting the paper's kernels.
- `ddm_na7_negative_signal_audit_20260814.md` established that RX2 is fixed-token rate work and #982 is the later joint gate-aware treatment. **Plan change:** remove RX2 as a consumer and make #982 conditional under its existing owner.
- The official paper and repository were searched for `benchmark`, `GH200`, `optimizer step`, `Gram`, `reset`, `Dion3`, `fraction`, `top`, `l1`, `megabatch`, `FSDP`, `CUDA graph`, `DTensor`, and `torch.compile`. Repository files inspected included the README, Dion3/NorDion2 implementation, Gram-NS wrapper/setup, and megabatch base. `git ls-remote` resolved the HEAD above; a later full clone was unavailable because DNS resolution failed, so no claim here depends on an uninspected local package execution.

## Frontier honesty

This memo produced no archive and moved no exact pointer. The effective frontier remains MC36 Variant C: **S=0.1619344578804448 @ 186,269 B `[contest-CUDA T4, n600]`**, archive SHA-256 `f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de`. The own-vehicle frontier remains LC2: **S=0.16959899569230852 @ 187,226 B `[contest-CUDA T4, n600]`**.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — WP1/JD1 row-subset shadow replay.** Owner: WP1/JD1 Muon-finisher owner. Consumer store: `/Volumes/VertigoDataTier/pact/ddm_dio1_20260814/retained/`. Fire trigger: the storage waterfall passes, a governed scorer-free capture exists for real per-step gradients and pre-step momentum for all six renderer matrices with checkpoint/seed/step identity, and the trainer lane is idle; then run P1 and P2 without mutating weights.
- **FOLDED-INTO-EXISTING-OWNER, CONTINGENT — #982 optimizer shadow.** Owner: JS1/#982 joint gate-aware treatment owner. Consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/`. Fire trigger: RX2 harvest is complete, #982's existing no-scorer identity/price/resume preflight passes, and its trainer exposes eligible 2-D blocks plus retained gradient/momentum traces; then reuse P1, not a new lane.

## LIVE-HYPOTHESES

- Selective top-L1 row updates may help the WP1/TR1 Muon finisher by reducing simultaneous interference between renderer rows while error feedback preserves omitted momentum. This is plausible because the finisher has six concrete matrix blocks and the paper's mechanism changes which subspace receives each step; it remains untested on real Pact gradients.
- The paper's inverse-square-root fraction scaling may land near the correct first update-RMS bracket, because selecting a fraction of rows reduces the full-tensor update norm in the idealized setting. Actual per-group update RMS may invalidate it through nonuniform row norms, clipping, or selector feedback.
- Stable Gram-NS may become useful for numerical quality, rather than speed, if a future larger witness trunk produces ill-conditioned polar inputs or low-precision residual failures. The paper gives a concrete finite-precision failure mechanism and cure, but current small matrices have no observed polar-quality wall.

## DEAD-ENDS

- Porting the 6× Dion3 throughput headline to current Pact training is closed at the **current-scale instance**: the paper times optimizer-only GH200/FSDP surfaces, while our exact NS5 work is far below the 5% materiality gate and direct Metal evidence has not shown a contrary wall.
- Porting CuteDSL symmetric kernels, CUDA graphs, or FSDP megabatching is closed for the current MLX/Metal line: the required NVIDIA/distributed runtime and communication bottleneck are absent.
- Reusing the `microsoft/dion` package directly is closed for the current trainer: it is a PyTorch/DTensor implementation, and the usable idea is smaller than the compatibility port.
- Treating `--micro-batch-pairs` as evidence for Dion megabatching is closed: one batches training examples through forward/backward, while the other groups same-shape optimizer matrices to reduce distributed collectives.
- Routing Dion3 into RX2 is closed: RX2 is a fixed-token rate harvest, not the joint gate-aware training consumer.
- Applying ambient row selection directly to #552/#556 `Q/H` factors is closed without new chart mathematics and receipts: it does not preserve the settled tangent/SPD/product-metric contract by construction.
- Importing paper fractions (`1/4`, `1/8`) or learning rates as Pact constants is closed: their loss evidence is task- and scale-specific, and px1's actual-update receipt is the local authority.
