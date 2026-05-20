# comma-lab overview

Working repo for the comma video compression challenge. Submission lands as PR [#110 (`hnerv_fec6_fixed_huffman_k16`)](https://github.com/commaai/comma_video_compression_challenge/pull/110). This document is the operator-facing introduction to what exists beyond the submission packet.

---

## 1. TLDR

PR #110 ships `~1140 LOC` across four files. The repo carries the rest: 52 substrate packages, 47 cathedral consumers, 235 strict preflight gates, 11 canonical equations, ~700 tools, ~1000 lanes in the registry.

The submission's CPU score is `0.192051 [contest-CPU]` paired with `0.226210 [contest-CUDA T4]`, both on the same archive bytes (`sha256: 6bae0201fb08…`, 178,517 bytes, single ZIP member `x` stored uncompressed). That's `-0.000794` below PR [#101](https://github.com/commaai/comma_video_compression_challenge/pull/101)'s `0.192845 [contest-CPU]` on the axis the leaderboard ranks. Within the HNeRV-family cluster — PR [#95](https://github.com/commaai/comma_video_compression_challenge/pull/95) [#100](https://github.com/commaai/comma_video_compression_challenge/pull/100) [#101](https://github.com/commaai/comma_video_compression_challenge/pull/101) [#102](https://github.com/commaai/comma_video_compression_challenge/pull/102) [#103](https://github.com/commaai/comma_video_compression_challenge/pull/103) plus this — the local floor sits inside `~0.0008`. Class-shift to a different substrate paradigm is the visible next direction; most of the apparatus exists to make that step deliberate rather than expensive.

---

## 2. What is empirically validated

End-to-end measurements on contest-1:1 hardware. CPU evals on `linux_x86_64_cpu` Modal containers matching the GitHub-Actions `ubuntu-latest` runner family; CUDA evals on Modal Tesla T4. Anything tagged `[advisory only]` or `[macOS-CPU advisory]` is development signal, not promotion evidence.

| Lane | Score | Paired axes |
|---|---|---|
| `hnerv_fec6_fixed_huffman_k16` (PR #110) | `0.192051 [contest-CPU]` + `0.226210 [contest-CUDA T4]` | yes |
| `lane_a1_*` substrate engineering | `0.19285 [contest-CPU]` + paired CUDA | yes |
| `hnerv_ft_microcodec` (PR #101 replay) | `0.192845 [contest-CPU]` (bot-recomputed) | CPU only |
| `pr106_format0d_latent_score_table` | `0.205330 [contest-CUDA T4]` | CUDA only |
| `nscs06` Strip-Everything v6→v7 | `105.15` → `58.89 [contest-CUDA T4]` after one cargo-cult-unwind pass (44% reduction) | CUDA only |
| `c6_ibps` 50ep IB smoke | `3.04 [contest-CUDA A10G]` vs design-time predicted `[0.113, 0.163]` | implementation-falsified |

Outside the HNeRV-family cluster, no other paradigm has been pushed to a paired CPU + CUDA anchor on contest hardware. Several have implementation-falsified verdicts at specific configs (24-dim IB collapse, naive-PTQ at int4, the NSCS06 7-cargo-cult v6 stack, the Wunderkind G1 v2 reducer, ATW V2's `INDEPENDENT` conditioning verdict at `MI = 0.006385 bits/symbol`). These are implementation-level falsifications, not paradigm kills; reactivation criteria are pinned per candidate.

---

## 3. What is scaffolded

Eleven paradigm classes are inventoried in [`asymptotic_floor_candidate_inventory.md`](./asymptotic_floor_candidate_inventory.md) with the inspiration / why-it-fits per class and the canonical reference paper(s). Compact table:

| Class | Canonical reference | Substrates | Empirical state |
|---|---|---|---|
| Predictive-coding world models | [Rao & Ballard 1999](https://www.nature.com/articles/nn0199_79); [Hafner 2023 DreamerV3](https://arxiv.org/abs/2301.04104); [Dao & Gu 2024 Mamba-2](https://arxiv.org/abs/2405.21060) | Z6 / Z6-v2 / Z7-LSTM / Z7-Mamba-2 / Z8 hierarchical | scaffolded |
| Cooperative-receiver framings | [Atick & Redlich 1990](https://www.mitpressjournals.org/doi/10.1162/neco.1990.2.3.308); [Wyner & Ziv 1976](https://ieeexplore.ieee.org/document/1055508) | Z4 / ATW V1 / ATW V2 / ATW V2-1 Faiss-IVF-PQ | ATW V2 falsified at INDEPENDENT verdict |
| Information bottleneck | [Tishby & Zaslavsky 2015](https://arxiv.org/abs/1503.02406) | C6 IBPS / β_ib sweep / Tishby IB-pure | C6 IBPS falsified at 24-dim latent |
| Pretrained driving priors | [Schafer et al. 2018 comma2k19](https://arxiv.org/abs/1812.05752) | DP1 Phase 2 + DP1+PR101 composition | scaffolded |
| Pose-axis / foveation / spatial-sparse | [Gibson 1950](https://archive.org/details/perceptionofvisu00gibs); [Teed & Deng 2020 RAFT](https://arxiv.org/abs/2003.12039) | RAFT-derived poses / LAPose / SAR / SABOR / TT5L V2 telescopic foveation | scaffolded; FF foveation L0 |
| NeRV-family beyond HNeRV | [Chen et al. 2023 HNeRV](https://arxiv.org/abs/2304.02633); [Chen et al. 2021 NeRV](https://arxiv.org/abs/2110.13903) | TCNeRV / BlockNeRV / FFNeRV / DSNeRV / HiNeRV / e_nerv / ego_nerv / nervdc | scaffolded; several hit dispatch-time API crashes (research_only) |
| Non-NeRV substrate architectures | [Sitzmann 2020 SIREN](https://arxiv.org/abs/2006.09661); [van den Oord 2017 VQ-VAE](https://arxiv.org/abs/1711.00937); [Park 2019 SPADE](https://arxiv.org/abs/1903.07291); [Tan 2021 CLADE](https://arxiv.org/abs/2012.04644); [Ladune 2023 Cool-Chic](https://arxiv.org/abs/2212.05458); [Kim 2023 C3](https://arxiv.org/abs/2312.02753); [He 2021 MAE](https://arxiv.org/abs/2111.06377) | Cool-Chic / C3 / wavelet residual / SIREN / VQ-VAE / hybrid renderer + residual / SPADE+CLADE conditioning via `dp_sims_renderer` / MAE mask augmentation / Quantizr-faithful (historical 0.33-0.41) / grayscale-LUT (PR #56 extension) / diffusion renderer | scaffolded; SIREN reactivation = longer-budget 4090; Cool-Chic+C3 export-contract gates open |
| Codec primitives + entropy coding | [Ballé 2018 hyperprior](https://arxiv.org/abs/1802.01436); [Daubechies 1988](https://onlinelibrary.wiley.com/doi/10.1002/cpa.3160410705); [Holub-Fridrich-Denemark 2014 UNIWARD](https://jis-eurasipjournals.springeropen.com/articles/10.1186/1687-417X-2014-1) | Wyner-Ziv layer (landed); hierarchical WZ quadruple (designed); STC-Dasher arithmetic-coding maximalism; Ballé hyperprior + CompressAI primitives; block-FP + Hessian-block-FP (PR #56 lineage); UNIWARD texture-aware encoding; water-filling Lane Ω-W (V3 launch-ready) | LANDED primitives + scaffolded compositions |
| Self-compression | [Hinton et al. 2014 distillation](https://arxiv.org/abs/1503.02531) | SC++ KL-distilled SegNet surrogate; MDL FP4 TTO; lane_17_imp (iterative magnitude pruning, council-deferred); MAE mask augmentation | scaffolded |
| Composition substrates + stacking | (operator-pioneered Carmack-Hotz Strip-Everything) | NSCS06 v6→v7→v8 (44% v6→v7 empirical); NSCS01 nullspace-split; NSCS02 downsampled; NSCS03 Ballé end-to-end joint; stack_of_stacks recipe framework; S2SBS byte-stuffing; SAR pose-axis composition; V8 Faiss IVF-PQ learned compression | v7 empirical; others scaffolded |
| Higher-order optimization | [Boyd 2011 ADMM](https://www.nowpublishers.com/article/Details/MAL-016); [Dykstra 1983](https://www.jstor.org/stable/2288033) | Riemannian-Newton; Tropical d_seg solver; Joint-ADMM coordinator; 3-set Venn classifier (empirical) | scaffolded + designed |

Most of these are L1 SCAFFOLD or DESIGN-ONLY. The honest accounting in the inventory memo's Section F catalogs what each candidate is stuck on (substrate-engineering cost, cargo-cult-vs-hard-earned classification, score-axis surrogate cost, implementation-vs-paradigm falsification).

---

## 4. What tooling and methodology exists

The submission packet is intentionally narrow. The apparatus that produced it is the rest of the repo. Compact bullets:

- **Cathedral autopilot ranker** at `tools/cathedral_autopilot_autonomous_loop.py` ingests candidates and emits ranked dispatch recommendations. 47 consumers under `src/tac/cathedral_consumers/` plug in via a canonical Protocol contract with auto-discovery (no manual ranker-cascade edits).
- **Per-pair master-gradient extractor** at `tools/extract_master_gradient.py` decomposes the additive scorer `S = 100·d_seg + sqrt(10·d_pose) + 25·R` across the parser-known payload domain at zero GPU cost. Per-pair Lagrangian-dual treatment plan is the downstream consumer. See companion tour at [`master_gradient_extractor_tour.md`](./master_gradient_extractor_tour.md).
- **Deterministic packet compiler** at `src/tac/packet_compiler/deterministic_compiler.py` is the canonical archive-grammar surface. Companions: Ballé hyperprior + Cheng2020 + cooperative-receiver grammars + custom binary container + magic codec + PR100 schema-driven decoder + PR101 conv4 storage perms.
- **Canonical equations registry** at `src/tac/canonical_equations/` holds 11 empirically-calibrated equations (Brotli cascade bounded per stream, MPS-vs-CUDA drift architecture-class dependent, per-byte leverage uniformly distributed on entropy-coded archives, per-pair master-gradient Taylor + Cauchy-Schwarz bound, ...). Bayesian posterior updating against landed anchors. See [`canonical_equations_tour.md`](./canonical_equations_tour.md).
- **Modal call-id ledger + harvester** at `src/tac/deploy/modal/call_id_ledger.py` closes the spawn-and-lose failure mode on Modal's detached function-call cache (24h TTL). `tools/harvest_modal_calls.py` + `tools/parallel_harvest_actuator.py`.
- **Frontier pointer canonical helper** at `src/tac/canonical_frontier_pointer.py` is the single source of truth for local-best CPU/CUDA anchors; auto-refreshed on dispatch completion. CLI: `tools/refresh_canonical_frontier.py`.
- **Per-X optimal codec planner + canonical DuckDB** at `src/tac/empirical_per_x_optimal_codec_planner/` unifies per-pair sensitivity queries.
- **Probe-outcomes ledger** at `src/tac/probe_outcomes_ledger.py` prevents re-firing already-adjudicated dispatches within a 30-day staleness window. 7-verdict taxonomy.
- **Canonical Provenance helper** at `src/tac/provenance/` attaches to every score-claiming row in persisted state. Distinguishes contest-archive-member from research-sidecar from predicted from advisory grades.
- **Wyner-Ziv deliverability proof builder** at `src/tac/wyner_ziv_deliverability/proof_builder.py` distinguishes truly deliverable side-information savings from research-sidecar phantom savings.
- **Pre-dispatch adversarial review automation** at `tools/run_codex_review_for_dispatch.py` invokes an external reviewer pass before any paid dispatch above `$1` estimated cost.
- **Compress-time freezing + gradient-stop toolkit** at `src/tac/freezing/` — eight focused helpers (pose-gradient-stop-after-warmstart, LoRA-style renderer adapter, frozen-teacher KL-distillation, SWA checkpoint averaging, lottery-ticket extraction, EMA freeze-at-eval, training-curriculum freezing-after-warm-start). Channels [Frankle-Carbin 2019](https://arxiv.org/abs/1803.03635), [Hinton 2014](https://arxiv.org/abs/1503.02531), [Hu 2021 LoRA](https://arxiv.org/abs/2106.09685), [Izmailov 2018 SWA](https://arxiv.org/abs/1803.05407).
- **Engineered MPS-vs-CUDA corrections** at `src/tac/engineered_corrections.py` — Kahan summation, softmax epsilon, fp32 matmul; calibrated against the `mps_drift_architecture_class_dependent_v1` canonical equation.
- **235 strict preflight gates** at `src/tac/preflight.py` — fail-closed on empirically-encountered bug classes. Meta-gates protect the catalog itself from drift (duplicate numbers, strict-text-matches-callsite, callsites-have-canonical-row, live-count drift, claim-via-serializer, quota brake at gate count 400). See [`strict_preflight_catalog_summary.md`](./strict_preflight_catalog_summary.md).

Discipline: per-substrate adversarial-council symposium before paid dispatch above threshold (canonical 6-step contract: cargo-cult audit + 9-dim checklist + observability surface + sextet pact + reactivation criteria + post-training Tier-C validation), recursive 3-clean-pass adversarial review with per-round assumption-challenge axis, 4-tier council hierarchy with explicit quorum and tie-break (T1 unbounded / T2 ≤3/day / T3 ≤3/week / T4 ≤2/month), per-deliberation HARD-EARNED-vs-CARGO-CULTED assumption surfacing. See [`cargo_cult_unwind_methodology.md`](./cargo_cult_unwind_methodology.md) for the NSCS06 v6→v7 anchor (44% reduction in one iteration).

Sister library [`adpena/tac`](https://github.com/adpena/tac) holds the task-aware compression primitives the submission runtime imports from.

---

## 5. Honest state and collaboration

Five lanes have paired or single-axis empirical anchors on contest hardware; the other ~50 substrates are scaffolded or designed but not run end-to-end. Most class-shift candidates are stuck on substrate-engineering cost (`$15`-`$300` per honest attempt), cargo-cult-vs-hard-earned classification (which canonical-reference assumptions transfer to dashcam-video and which do not), or score-axis surrogate cost (training against the contest scorer directly is GPU-bound and an unmeasured distilled surrogate is not safe given the PoseNet gap at minor numerical perturbations).

The intent is to push more candidates through the pipeline. The submission packet is the validated tip; the apparatus is ~3 orders of magnitude larger and most of it has not been paid for in GPU hours. The bug-class extinction work, canonical-helper consolidation, and meta-engineering discipline are leverageable — a collaborator stepping into one of the scaffolded paradigm classes inherits the tooling without rebuilding it.

References: [PR #110 anchor](https://github.com/commaai/comma_video_compression_challenge/pull/110) · [inventory](./asymptotic_floor_candidate_inventory.md) · [cargo-cult unwind methodology](./cargo_cult_unwind_methodology.md) · [canonical equations tour](./canonical_equations_tour.md) · [master-gradient extractor tour](./master_gradient_extractor_tour.md) · [strict preflight catalog summary](./strict_preflight_catalog_summary.md) · [sister library `adpena/tac`](https://github.com/adpena/tac).
