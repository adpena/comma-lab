# Meta engineering vision

What `comma-lab` is building toward, in three sentences: every arbitrary constant and config in the compression pipeline gets replaced — over time, against signal — by a learned or discovered optimal. Continual learning is the mechanism. Per-element decisions, derived from canonical equations composed by canonical helpers, are the destination.

The individual substrates are evidence for the META. The META is the headline.

---

## 1. The META

Most compression engineering looks like this: pick an architecture, pick its hyperparameters, pick its bit budgets, pick its training schedule. Each "pick" is a value lifted from a canonical reference paper or a sibling implementation or a defensible-sounding default. The picks accumulate. After enough of them, the system works — but no one can point at any single constant and say *why that exact value, against what evidence, derived how*.

The shift here is structural. Every arbitrary constant becomes a candidate for replacement by a learned or discovered optimal. Bit allocation per archive byte derived from per-byte sensitivity, not heuristic. Per-pair codec parameters derived from per-pair difficulty atlas, not uniform. Per-class chroma allocation derived from scorer decomposition per SegNet class, not handpicked. Per-substrate fit ranking derived from M_inflated vs M_contest comparison, not authorial preference. Per-frame transform selector derived from offline scorer-targeted search, not random.

The mechanism is continual learning. Every empirical anchor — paid GPU dispatch, paired CPU+CUDA auth-eval result, falsified prediction, ratified prediction — refines a canonical-equations registry, a cathedral autopilot ranker, a canonical Provenance posterior. The next decision is shaped by cumulative evidence, not by hand-picked priors. The system gets smarter monotonically as long as the signal pipeline stays closed.

The destination is per-element optimal across the entire compression + dispatch + design surface. Not just "the codec is well-tuned" but "every byte the codec emits has a derivation chain back to canonical equations and empirical anchors."

**Why this matters beyond a contest leaderboard.** Each `0.01` / `0.001` / `0.0001` of score lowering looks small in isolation. At production-fleet scale across time — comma devices running on cars; video uplink / on-device-storage / energy cost per device per hour — small per-archive improvements compound. The contest reads as R&D probe for production-relevant compression infrastructure: the discipline that produces reproducible per-element-optimal derivations is the leverageable artifact even when the per-clip score gap looks marginal.

This is what the operational surfaces below are *for*.

---

## 2. Operational surfaces

| Surface | Role in the META | Canonical path |
|---|---|---|
| **xray** | Observability primitive. Six facets (inspectable per layer; decomposable per signal; diff-able across runs; queryable post-hoc; cite-able; counterfactual-able). You cannot replace a constant with a learned value if you cannot observe per-element behavior — xray is what makes the rest possible. | `src/tac/xray/` |
| **Atoms** | Typed-element primitive. Every signal, measurement, candidate becomes a typed atom the solver consumes. Replaces ad-hoc "this is a row of data" with structured discipline. | `src/tac/atom/` |
| **Meta-Lagrangian / Pareto solver** | Canonical convex solver (Boyd & Vandenberghe + Dykstra alternating projections) over typed atoms + constraints + canonical equations + empirical anchors → KKT/ADMM-derived per-element allocation. Replaces arbitrary multi-objective weights with derived dual variables. | `src/tac/findings_lagrangian/` + `src/tac/findings_lagrangian_pp/` + per-pair Lagrangian-dual planner |
| **Canonical equations registry** | Replaces tribal knowledge with queryable models. Every empirical finding becomes an equation; every equation has predicted-vs-empirical residual posterior; every new anchor refines it. | `src/tac/canonical_equations/` |
| **Master gradient extractor** | Canonical SIGNAL source for per-element sensitivity. Decomposes the additive scorer across the parser-known payload domain at zero GPU cost via a Taylor + Cauchy-Schwarz first-order bound. M_contest / M_archive / M_inflated comparison surfaces. | `tools/extract_master_gradient.py` + `src/tac/master_gradient_consumers.py` |
| **Per-pair Lagrangian planner** | Canonical convex solver for per-pair allocation. Replaces uniform allocation + heuristic per-pair weights with derived dual-variable allocation per the per-pair difficulty atlas. | `src/tac/master_gradient_consumers.py` |
| **Canonical Provenance** | Every score claim carries axis + hardware + grade + cite-chain. Replaces "we believe this number" with "this number is grade X derived from anchor Y under hardware Z." | `src/tac/provenance/` |
| **Cathedral autopilot ranker + auto-discovery** | Canonical CONSUMER of all the above. Ingests candidates → ranked dispatch recommendations. Consumer packages plug in via a canonical Protocol contract — no manual ranker-cascade edits when new signal lands. | `tools/cathedral_autopilot_autonomous_loop.py` + `src/tac/cathedral/consumer_contract.py` + `src/tac/cathedral_consumers/` |
| **Continual-learning posterior** | Canonical STATE evolved by signal + empirical anchors. Replaces "we tried this and got that" memory with structured Bayesian posterior over equations + candidates + verdicts. | `src/tac/continual_learning.py` (+ council sister) |
| **Wiring + integration discipline** | The discipline that prevents the META from regressing. Every new helper declares its six wire-in hooks (sensitivity / Pareto / bit-allocator / cathedral autopilot / continual learning / probe disambiguator). Orphan signals are surfaced and corrected. | 6-hook declaration; canonical Protocol contract; auto-discovery |
| **~300 STRICT preflight gates** | Structural extinction of recurring bug classes. Each gate is a per-bug-class learned-optimal that supersedes the prior arbitrary "be careful" engineering norm. META-meta gates protect the catalog itself from drift. | `src/tac/preflight.py` |

The table is the menu. The depth is in the canonical paths.

---

## 3. The META in practice

**FEC6 selector as per-frame discovered optimal.** The bolt-on shipped in PR #110 picks one of K=16 frame transforms per pair, against the upstream scorer's response, via offline scorer-targeted search. Where the predecessor used a uniform per-frame heuristic, the selector derives a per-frame value from a small finite alphabet against the actual signal. Same shape, smaller numbers: the per-element-discovered-optimal pattern at frame granularity.

**Per-byte bit allocation via M_archive top-K + Lagrangian dual.** Empirically, byte-level sensitivity is concentrated — the top-K bytes carry most of the score impact on entropy-coded archives (canonical equation: `per_byte_leverage_uniformly_distributed_v1`, validated against PR #101 with top-1% byte leverage at 6.4%). The per-pair Lagrangian-dual planner consumes the master-gradient signal and emits per-element allocation derived from the dual variables — not from a heuristic budget split.

**Cargo-cult-unwind as per-assumption discovered optimal.** NSCS06 Strip-Everything: v6 at `105.15 [contest-CUDA T4]`. One iteration of per-assumption HARD-EARNED-vs-CARGO-CULTED classification + per-assumption unwind. v7 at `58.89 [contest-CUDA T4]`. 44% reduction. The methodology is the META applied at the design-assumption level: each inherited assumption is treated as a candidate for replacement by an empirically-tested optimal, not a default to carry forward.

The pattern repeats across surfaces. The substrates listed in [`asymptotic_floor_candidate_inventory.md`](./asymptotic_floor_candidate_inventory.md) are evidence at a particular granularity. The methodology is the same.

---

## 4. What is empirically validated

Five lanes have paired or single-axis empirical anchors on contest-1:1 hardware: this PR, A1 substrate engineering, PR #101 replay, PR #106 latent-score-table on the CUDA axis, NSCS06 v7 on the CUDA axis. C6 IBPS at 24-dim latent is implementation-falsified at `3.04 [contest-CUDA A10G]` against a design-time predicted band of `[0.113, 0.163]` — the canonical example of paradigm-intact / implementation-failed distinction the apparatus is built to preserve.

The other ~50 substrates listed in the inventory are scaffolded or designed but not run end-to-end. The methodology + tooling + meta-engineering rigor are real; the empirical validation per candidate is uneven. The inventory memo's Section F catalogs what each candidate is stuck on, honestly.

This is the calibration: the META is the leverageable artifact; the per-candidate empirical state is mixed. Both true.

**PR110/fec6 as methodology-proving artifact (Wave 3 cross-candidate sensitivity comparison, 2026-05-20).** The +0.000794 advantage of fec6 frontier (`0.19205 [contest-CPU]`) over PR101 GOLD (`0.19284 [contest-CPU]`) is not a backbone refinement — the 21-pair cross-candidate per-byte sensitivity matrix established empirically that the 178,158-byte HNeRV backbone has IDENTICAL per-axis aggregate sensitivity between PR101 and fec6 (Pearson seg ρ = 0.961, pose ρ = 0.971). The entire +794 ppm advantage is concentrated in **+259 bytes of FEC6 selector + Huffman k=16 frame-exploit orthogonal overhead**. PR110 IS the proof-of-concept that orthogonal selector overlays on the saturated HNeRV-class backbone are a valid score-lowering path; the structural insight propagates beyond the immediate submission into the selector-extensions class of future substrate work. The discipline that produced this finding ([`docs/per_byte_sensitivity_comparative_analysis_methodology.md`](./per_byte_sensitivity_comparative_analysis_methodology.md)) is now reusable infrastructure; the three canonical equations codified (per_byte_leverage_cross_hardware_aware_v2 + hnerv_backbone_sensitivity_saturated_across_medal_cluster_v1 + cross_codec_super_additive_orthogonality_predictor_v1 per Catalog #344 commit 80484241f) are first-class artifacts independent of PR110's submission use.

---

## 5. The full meal

The depth surfaces:

- [`docs/asymptotic_floor_candidate_inventory.md`](./asymptotic_floor_candidate_inventory.md) — per-paradigm-class candidate enumeration with canonical reference papers.
- [`docs/cargo_cult_unwind_methodology.md`](./cargo_cult_unwind_methodology.md) — the paradigm-rescue discipline that produced the NSCS06 44%-in-one-iteration anchor.
- [`docs/canonical_equations_tour.md`](./canonical_equations_tour.md) — tour of the initial canonical equations as first-class artifacts.
- [`docs/master_gradient_extractor_tour.md`](./master_gradient_extractor_tour.md) — tool tour for the per-element sensitivity extractor and the ten exploits it enables.
- [`docs/strict_preflight_catalog_summary.md`](./strict_preflight_catalog_summary.md) — browseable summary of the structural bug-class extinction surface.
- [`docs/standout_undersold_candidates_spotlight.md`](./standout_undersold_candidates_spotlight.md) — 90-second skim of ten candidates worth attention.
- [`docs/ai_assisted_inverse_steganalysis_persona_council.md`](./ai_assisted_inverse_steganalysis_persona_council.md) — the META concepts that ran underneath the substrate work: inverse-steganalysis on the contest's information space + the named-persona 4-tier council methodology grounded in Anthropic's persona-vectors research.
- Sister library [`adpena/tac`](https://github.com/adpena/tac) — task-aware-compression research primitives and reusable tooling used during development; not required by the PR #110 submission runtime.
- Working repo [`adpena/comma-lab`](https://github.com/adpena/comma-lab) — the broader apparatus.
- [PR #110](https://github.com/commaai/comma_video_compression_challenge/pull/110) — the immediate contest submission the META lands into.

---

## 6. Honest qualifications

This is solo-developer scope. Per-candidate empirical validation is uneven; several substrates are scaffolded or designed without an end-to-end run on contest hardware. The methodology, the canonical operational surfaces, and the meta-engineering discipline are leverageable infrastructure — but the apparatus is ~3 orders of magnitude larger than what has been paid for in GPU hours.

**CPU-vs-CUDA axis as an open frontier.** The contest leaderboard ranks by `--device cpu` evaluation; CUDA evaluation tracks a parallel score on the same archive bytes that is systematically higher by roughly +0.030–0.040 across the HNeRV cluster (canonical reference: [`docs/paper/04_results.md` §4.8](./paper/04_results.md), [`docs/writeup/cuda_cpu_drift_methodology.md`](./writeup/cuda_cpu_drift_methodology.md)). The per-axis frontiers are different archives: the CPU-axis frontier is the PR110 fec6 selector lane (sha-prefix `6bae0201`); the CUDA-axis frontier is the PR106 format0d latent-score-table lane (sha-prefix `9cb989ce`). CPU-optimal engineering may require techniques X, Y, Z while CUDA-optimal engineering may require A, B, C — they are not the same target, and the relationship between them is itself a research question. The Wave 3 engineering analysis [`.omx/research/cpu_vs_cuda_drift_engineering_analysis_20260520T173724Z.md`](../.omx/research/cpu_vs_cuda_drift_engineering_analysis_20260520T173724Z.md) catalogs the four candidate mechanism hypotheses (FALSIFIED: FastViT TF32-attention; REJECTED-for-non-decode: dataset/loader drift; PARTIAL: DALI/NVDEC vs PyAV decoder drift + FastViT FP32 forward-kernel noise floor) and registers three canonical equations for the cluster. The CUDA-axis frontier is less explored than the CPU-axis frontier; both axes get dual-eval per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable on every shippable archive.

Collaboration could accelerate the empirical-validation cycle substantially. A collaborator stepping into one of the scaffolded paradigm classes inherits the tooling (cathedral autopilot ranker; per-pair master gradient; canonical equations registry; deterministic packet compiler; canonical Provenance; ~300 strict preflight gates) without rebuilding it. The bug-class extinction work + canonical-helper consolidation + per-substrate symposium discipline are reusable across paradigm classes.

The META is the durable artifact. The contest submission is the validated tip. Most of what comes next depends on whether the apparatus gets exercised against more empirical anchors.

---

## 7. Who built this (a brief aside)

A Python developer with around eight years of programming and no formal background in video compression. Drawn in by comma.ai's philosophy — open-source hardware and software democratizing self-driving, with tinygrad and openpilot as visions of robotics-OS futures. Spent close to two months solo-grinding with AI assistance. The methodology + broader personal context lives in the [persona-council methodology memo](./ai_assisted_inverse_steganalysis_persona_council.md). Comma.ai's mission and the Yousfi + Fridrich + Quantizr + medal-class-PR-author (`@AaronLeslie138`, `@EthanYangTW`, `@BradyMeighan`, `@SajayR`, `@rem2`) lineage are the important things; this is a small contribution.
