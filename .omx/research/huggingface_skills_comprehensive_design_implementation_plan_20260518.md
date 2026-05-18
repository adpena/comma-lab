---
title: HuggingFace Skills Comprehensive Design + Implementation Plan
date: 2026-05-18
author: subagent a_hf_skills_design_20260518
lane_id: lane_hf_skills_comprehensive_design_implementation_plan_20260518
predicted_mission_contribution: frontier_breaking
status: research_only
council_tier: pending_T2_inner_skunkworks_review
related_memos:
  - feedback_deep_research_wave_landed_20260518.md
  - feedback_wave_complete_plus_deep_research_dispatch_landed_20260517.md
  - feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md
  - feedback_canonical_share_when_serves_unique_when_suppresses_standing_directive_20260515.md
horizon_class: frontier_pursuit
---

# HuggingFace Skills Comprehensive Design + Implementation Plan

**Operator directive (verbatim):** *"we want to research and design and implement all: [5 insights from huggingface-skills:hugging-face-vision-trainer applied to our problem]... not just the key, also want to review the rest of the skills in the /huggingface-skills plugin to find other related and powerful and useful tools and intelligence"*

**Scope:** Comprehensive survey of all 12 skills under `~/.claude/plugins/cache/claude-plugins-official/huggingface-skills/1.0.1/skills/` mapped to comma.ai video compression challenge use cases. Extends the 5 main insights from vision-trainer with NEW insights from the other 11 skills. Sequenced implementation plan (Phase 1 → Phase 4) with operator-routable decisions, catalog compliance, risk register, and cross-disciplinary convergent-truth extensions.

## Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable. For each HF infrastructure layer being adopted, per-layer decision recorded:

| Layer | Decision | Rationale |
|---|---|---|
| HF Hub authentication (`HF_TOKEN`) | ADOPT_CANONICAL_BECAUSE_SERVES | Canonical token discipline; no substrate-optimal reason to fork |
| HF Datasets API (`datasets.push_to_hub`) | ADOPT_CANONICAL_BECAUSE_SERVES | Canonical reproducibility surface; matches CLAUDE.md "Required durable state" |
| HF Jobs `hf_jobs("uv", ...)` API | ADOPT_CANONICAL_BECAUSE_SERVES | One-arg dispatch toggle per Catalog #317; sister to Modal/Vast.ai/Lightning |
| Trackio metric logging | ADOPT_CANONICAL_BECAUSE_SERVES | Real-time monitoring + alerts; sister to our existing ad-hoc JSON/CSV logging |
| Trackio `Space` sync to `huggingface.co/spaces/adpena/trackio` | FORK_BECAUSE_PRINCIPLED_MISMATCH | Public Space leaks operator-internal training metrics per CLAUDE.md "Public Disclosure Hygiene"; route to PRIVATE Space until release wave |
| `image_classification_training.py` script for SegNet surrogate | FORK_BECAUSE_SUPPRESSES | Canonical IC script targets COCO/cppe-5/food101; SegNet substrate needs full-frame 384x512 + EXACT scorer-roundtrip simulation per Catalog #8 eval_roundtrip non-negotiable; canonical out-of-the-box ignores this |
| `sam_segmentation_training.py` for PoseNet surrogate | FORK_BECAUSE_PRINCIPLED_MISMATCH | SAM2 trains mask-decoder for binary segmentation w/ bbox/point prompts; PoseNet is 6-DOF pose regression w/ FastViT-T12 backbone — different output head + different gradient path |
| Gradio dashboard | ADOPT_CANONICAL_BECAUSE_SERVES | Canonical web UI; no substrate-optimal forking needed |
| `hf-cli` (`hf` command) | ADOPT_CANONICAL_BECAUSE_SERVES | Replaces deprecated `huggingface-cli`; canonical |
| `paper_publisher` arXiv linking | ADOPT_CANONICAL_BECAUSE_SERVES | Catalog #431 arXiv paper integration is operator-pending |
| `tool-builder` patterns | ADOPT_CANONICAL_BECAUSE_SERVES | Canonical helper script style |
| `evaluation` model-card entries | ADOPT_CANONICAL_BECAUSE_SERVES | Catalog #316 frontier preservation surface |
| `dataset-viewer` SQL queries | ADOPT_CANONICAL_BECAUSE_SERVES | Cross-PR-archive mining is high-EV |
| `transformers.js` JS runtime | NOT_USEFUL | JS/TS runtime; our entire stack is Python; SKIP |

## 9-dimension success checklist evidence

Per CLAUDE.md "9-dimension success checklist evidence section" Catalog #294. This memo is a design memo (not a substrate landing memo) so the 9-dim checklist is scoped to the design surface:

1. **UNIQUENESS**: this is the FIRST comprehensive HF-skills survey for our problem; previously only vision-trainer was loaded into a single conversation.
2. **BEAUTY + ELEGANCE**: every recommendation is one-paragraph readable; sequenced Phase 1-4 plan.
3. **DISTINCTNESS**: explicitly differentiates HF Jobs from Modal/Vast.ai/Lightning per cost+capability matrix in Section 2.5.
4. **RIGOR**: per-skill HARD-EARNED-vs-CARGO-CULTED classification; transformers.js explicitly classified NOT_USEFUL.
5. **OPTIMIZATION PER TECHNIQUE**: per-layer canonical-vs-unique decision per Catalog #290.
6. **STACK-OF-STACKS COMPOSABILITY**: every Phase 2+ use case composes with existing substrate trainer canvas (DP1 + SCPP + A1 + PR101 + PR106 + fec6 + Z6 + Z7 + Z8 + TT5L).
7. **DETERMINISTIC REPRODUCIBILITY**: every dataset upload is SHA-256 anchored; every Trackio run carries `run_name` + `project` for cite-chain.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: HF Jobs T4 at $0.40/hr vs Modal T4 at $0.59/hr = 32% cost reduction; Vast.ai 4090 at $0.25/hr remains cheapest for non-promotable smoke.
9. **OPTIMAL MINIMAL CONTEST SCORE**: 5-main-insight implementation predicted ΔS [-0.005, -0.020] vs current 0.19205 floor (top-3 unrealized use cases predict [-0.010, -0.025]).

## Observability surface

Per CLAUDE.md Catalog #305. The implementation surfaces of this design memo are observable across the 6 facets:

1. **Inspectable per layer**: every HF artifact (dataset / model / Space / Job) has a Hub URL and metadata view.
2. **Decomposable per signal**: Trackio surfaces per-metric time series; HF Datasets surface per-column statistics.
3. **Diff-able across runs**: Trackio `run_name` + `project` + config snapshots; HF Hub Git versioning per repo.
4. **Queryable post-hoc**: `hf datasets sql` + `hf jobs logs <id>` + Trackio CLI `trackio get metric --json` + Dataset Viewer `/rows` API.
5. **Cite-able**: every HF object has a permanent URL + revision SHA; arXiv paper IDs anchor to `hf://papers/{id}`.
6. **Counterfactual-able**: HF Datasets `@~parquet` auto-conversion enables byte-mutation queries; Trackio alerts enable counterfactual "what if loss had not spiked" replay.

## Cargo-cult audit per assumption

Per Catalog #303. The implicit assumptions in adopting HF infrastructure for our problem:

| Assumption | HARD-EARNED-vs-CARGO-CULTED | Rationale + unwind path |
|---|---|---|
| "HF Jobs T4 produces bit-identical scores to Modal T4" | CARGO-CULTED-PENDING-VERIFICATION | NVIDIA T4 hardware identical, but CUDA driver + container image may differ. Unwind: paired smoke on identical archive sha; verify auth_eval JSON match within 1e-6. Cheap ($0.40 vs $0.59 = $0.20 + 1 smoke each). |
| "DINOv3 features generalize from ImageNet to dashcam video" | CARGO-CULTED-INHERITED-FROM-LITERATURE | DINOv3 pretrained on LVD-1689M (web images, not dashcam); cooperative-receiver claim from `feedback_deep_research_wave_landed_20260518.md` is THEORETICAL not EMPIRICAL. Unwind: frozen-DINOv3 vs fine-tuned-DINOv3 ablation on the `comma-video-substrate-eval-600pairs` dataset; measure feature-level cosine similarity to SegNet/PoseNet outputs. |
| "SAM2-hiera-tiny mask decoder can be repurposed for PoseNet 6-DOF regression" | CARGO-CULTED-CROSS-DOMAIN | SAM2 mask decoder outputs binary mask; PoseNet outputs 6-DOF pose vector. The HEADS differ structurally. Unwind: fork mask decoder → custom `PoseRegressionHead(in_dim=256, out_dim=6)`; freeze encoder. |
| "Hinton-distilled scorer surrogate reduces auth_eval roundtrip cost" | HARD-EARNED (PR95 reference; Catalog #523) | Quantizr empirically used kl_on_logits T=2.0 for SegNet during training. The DISTILLATION pattern is proven; what's CARGO-CULTED is the SPECIFIC student architecture choice. |
| "HF Datasets DuckDB SQL queries can mine our 54 PR archive corpus" | HARD-EARNED | DuckDB on `hf://` protocol is a verified canonical pattern (vision-trainer + datasets skill independently document it). |
| "Trackio Space sync to public Space is OK for our training metrics" | CARGO-CULTED-PUBLIC-DISCLOSURE-RISK | Per CLAUDE.md "Public Disclosure Hygiene": local absolute paths, raw provider logs, unpublished operator state must be sanitized. Unwind: PRIVATE Space initially; flip to public at OSS release wave. |
| "HF Jobs scheduled jobs can replace our cron-based Modal harvester" | HARD-EARNED | Catalog #245 modal_call_id_ledger relies on a polling harvester; HF Jobs `@hourly` scheduling is canonical. |
| "Local MPS development is faster than HF Jobs round-trip" | HARD-EARNED (Catalog #317 anchor) | MPS = $0/hr + zero setup latency; HF Jobs = $0.40/hr + 30-60s startup. For sub-5-min smokes, MPS wins. For >30-min training, HF Jobs wins per cost+wall-clock. |

## Section 1: Skills survey + applicability matrix (per-skill 1-pager)

### 1.1 `hf-cli` — Hugging Face Hub CLI

**Core capabilities**: Replaces deprecated `huggingface-cli`. Provides `hf download`, `hf upload`, `hf upload-large-folder`, `hf auth login`, `hf datasets sql`, `hf jobs run/uv`, `hf spaces dev-mode`, `hf collections create`, `hf repos branch/tag`, `hf discussions create/comment`, `hf endpoints deploy`, `hf webhooks create`.

**Applicability**: **DIRECTLY USEFUL (P0)**.

**Specific use cases**:
1. **Replace ad-hoc dataset uploads** — every dataset upload (e.g. `comma-video-substrate-eval-600pairs`, `comma-pr-archive-corpus-54`) goes through `hf upload datasets/<repo>` instead of inline Python.
2. **Authoritative model card upload** for the 13 substrate trainers we've shipped (PR101 lc_v2, fec6 selector, PR106 format0d, etc.).
3. **`hf jobs` CLI as fallback** when MCP tool unavailable in subagent context.
4. **`hf datasets sql`** for the cross-PR-archive corpus query in Section 3.
5. **`hf collections create`** to publish a "comma.ai video compression — 13 substrates" collection that links models + datasets + papers.

**Estimated cost + EV**: Cost = $0 (CLI is free). EV = HIGH (replaces 3+ ad-hoc upload scripts; canonical reproducibility).

**Integration complexity**: LOW. Already installed in operator's local environment per CLAUDE.md instruction.

**Priority**: P0 (foundational; everything else builds on this).

---

### 1.2 `hugging-face-dataset-viewer` — Dataset Viewer API workflows

**Core capabilities**: Read-only API at `datasets-server.huggingface.co` for `/is-valid`, `/splits`, `/first-rows`, `/rows`, `/search`, `/filter`, `/parquet`, `/size`, `/statistics`, `/croissant`. SQL via `npx parquetlens` on `hf://datasets/.../@~parquet/...parquet` paths.

**Applicability**: **MODERATELY USEFUL (P1)** for our context.

**Specific use cases**:
1. **Cross-PR-archive mining**: query our `comma-pr-archive-corpus-54` (if uploaded) via SQL to find empirical patterns: "show all PRs where rate > 0.20 AND segnet_distortion < 0.10" — find Pareto-frontier outliers.
2. **Validate our published datasets** via `/is-valid` before each release wave.
3. **Compute size totals + column statistics** for paper supplement tables.
4. **Read-only browsing** of other team's datasets (e.g., NVIDIA Cosmos, Comma2k19 official) to identify integration candidates.

**Estimated cost + EV**: Cost = $0 (read-only API). EV = MODERATE (mining could surface 1-2 unrealized empirical patterns).

**Integration complexity**: LOW.

**Priority**: P1 (after canonical dataset upload).

---

### 1.3 `hugging-face-datasets` — Create/manage HF Hub datasets + SQL queries via DuckDB

**Core capabilities**: Dataset lifecycle management (init, configure, stream updates); SQL-based dataset querying via DuckDB on `hf://` protocol; multi-format support (chat, classification, QA, completion, tabular, custom); JSON validation; batch processing. The `sql_manager.py` script is the canonical SQL surface.

**Applicability**: **DIRECTLY USEFUL (P0)**.

**Specific use cases**:
1. **Build `adpena/comma-video-substrate-eval-600pairs` dataset** — the canonical 600-pair SegNet/PoseNet eval set used by every substrate trainer. Schema: `{pair_idx, frame_0_rgb, frame_1_rgb, segnet_mask_5ch, posenet_pose_6d, archive_sha_baseline, pair_sha}`.
2. **Build `adpena/comma-pr-archive-corpus-54`** — every PR archive's metadata (PR#, score CPU/CUDA, archive_sha, member_names, technique tags, lane_id).
3. **Build `adpena/comma-substrate-composition-matrix`** — every empirical alpha measurement from `tac.optimization.substrate_composition_matrix` exported as queryable rows.
4. **SQL queries** to surface frontier patterns: `SELECT pr, score_cpu, score_cuda, score_cuda - score_cpu AS gap FROM data WHERE gap > 0.03 ORDER BY gap DESC` (the PR102 0.03 gap anchor surfaced by sister analysis).
5. **Push to Hub with `--private` flag initially** per CLAUDE.md "Public Disclosure Hygiene"; flip to public at release wave.

**Estimated cost + EV**: Cost = $0 (free tier covers our dataset size). EV = HIGH (canonical reproducibility + cross-archive mining unlocked).

**Integration complexity**: MEDIUM (need to write upload pipeline that decodes video → SegNet/PoseNet forward → push to Hub).

**Priority**: P0 (foundational for all other skills; canonical reproducibility primitive).

---

### 1.4 `hugging-face-evaluation` — Evaluation results in model cards

**Core capabilities**: Add structured evaluation results to model cards via `model-index` metadata; extract eval tables from README; import from Artificial Analysis API; run custom model evaluations with vLLM/lighteval/inspect-ai; PR creation workflow (with `get-prs` mandatory pre-check).

**Applicability**: **DIRECTLY USEFUL (P1)** for our published substrates.

**Specific use cases**:
1. **Add canonical `[contest-CUDA]` + `[contest-CPU]` results to model cards** for every published substrate (PR101 lc_v2, fec6 selector, PR106 format0d, NSCS03, A1, etc.).
2. **Catalog #316 frontier preservation surface** — the model-index entry IS the canonical claim about each substrate's score.
3. **Future use**: vLLM/lighteval is NOT directly useful (we don't train LLMs), but the model-card surface IS useful for our codec models.
4. **PR creation workflow with `get-prs` pre-check** prevents duplicate PR spam per the skill's own warning.

**Estimated cost + EV**: Cost = $0. EV = MODERATE (operator-facing release surface; helps with paper publication + leaderboard).

**Integration complexity**: LOW.

**Priority**: P1 (after canonical dataset upload).

---

### 1.5 `hugging-face-jobs` — General HF Jobs infrastructure

**Core capabilities**: Run any Python workload on HF cloud GPUs/TPUs via UV scripts (PEP 723) or Docker images. Hardware flavors `cpu-basic` to `a100-large` + TPU `v5e-1x1`. Scheduled jobs via CRON. Webhooks. Ephemeral environment — MUST persist results to Hub. `hf_jobs()` MCP tool + `HfApi().run_uv_job()` Python API + `hf jobs uv run` CLI. Pro/Team/Enterprise plan required.

**Applicability**: **DIRECTLY USEFUL (P0)** — HARDWARE RE-ROUTING is one of the 5 main insights.

**Specific use cases**:
1. **Replace Modal T4 ($0.59/hr) with HF Jobs T4 ($0.40/hr)** — 32% cost reduction on training canary smokes.
2. **HF Jobs L4 ($0.80/hr) competitive with Vast.ai 4090 ($0.25/hr)** for the same wall-clock — 4090 still wins on $/hr but HF Jobs wins on operator convenience (no instance management).
3. **HF Jobs A100 ($2.50/hr) vs Modal A100 ($4.00/hr)** — 37% cost reduction on full substrate dispatches.
4. **Scheduled jobs `@hourly`** for the Modal call_id ledger harvester + sister state-cleanup chores.
5. **Webhooks for "auto-train on PR archive update"** — when a new PR archive lands on `adpena/comma-pr-archive-corpus-54`, trigger a job that runs auth_eval on the new archive.
6. **TPU access (`v5e-1x1` $0.10/hr)** for JAX-based experimental substrates (currently NONE; high-EV future direction for end-to-end Ballé joint codec training).
7. **MUST set timeout >30min for training** (default kills training silently). MUST persist to Hub (ephemeral environment).
8. **PR101 lc_v2 retraining** + sister substrate full canvases.

**Estimated cost + EV**: Cost = $5-50 per training campaign (depending on scope). EV = VERY HIGH (32-37% cost reduction across all dispatch surfaces if migration completes).

**Integration complexity**: MEDIUM (need to adapt 13 substrate trainer scripts to `hf_jobs("uv", {...})` MCP tool format; sister-PR-aware token management).

**Priority**: P0 (one of the 5 main insights).

---

### 1.6 `hugging-face-model-trainer` — TRL language model training

**Core capabilities**: TRL-based training (SFT, DPO, GRPO, Reward Modeling) on HF Jobs. PEP 723 UV scripts. Trackio integration always included. `push_to_hub=True` mandatory. Dataset validation via `dataset_inspector.py`. Unsloth for 60% less VRAM + 2x speed. GGUF conversion for local deployment.

**Applicability**: **NOT USEFUL** for direct substrate training — we don't train LLMs.

**Note**: This skill's structural templates (`scripts/train_sft_example.py`) ARE useful as references for how to structure HF Jobs UV scripts with Trackio + Hub push + LoRA. **Cargo-cult risk**: copying TRL-specific Trainer code into our substrate trainers; the structural pattern transfers, the actual training class does not.

**Specific use cases (limited)**:
1. **Reference TRL-style PEP 723 UV script structure** for our substrate trainers' HF Jobs migration.
2. **Use Trackio integration pattern** (report_to="trackio") even though we use custom Trainers.
3. **Future**: if we ever distill a TINY language model to caption frames for paper supplement, SFT could fit.

**Estimated cost + EV**: Cost = $0 (reference only). EV = LOW (structural reference; no direct training).

**Integration complexity**: N/A (reference only).

**Priority**: P3 (reference only; do not invoke).

---

### 1.7 `hugging-face-paper-publisher` — Paper pages + arXiv linking

**Core capabilities**: Index papers on Hugging Face Paper Pages from arXiv; link papers to models/datasets/spaces; claim authorship; generate research articles from templates (standard, modern, arxiv, ml-report); convert markdown → HTML; manage paper visibility.

**Applicability**: **DIRECTLY USEFUL (P1)** — Catalog #431 PHASE 4 INTEGRATION PENDING.

**Specific use cases**:
1. **Index our arXiv paper** (Catalog #431 status: PHASE 4 INTEGRATION PENDING) on Hugging Face Paper Pages.
2. **Link paper to all 13 published substrate models** + sister datasets via `paper_manager.py link`.
3. **Claim authorship** for the published paper.
4. **Generate research article template** for the paper draft.
5. **Markdown → HTML conversion** for the paper website (Cloudflare-hosted per existing release wave).
6. **arXiv tagging discovery** — papers auto-tagged with `arxiv:<id>` enable hub-wide cross-reference.

**Estimated cost + EV**: Cost = $0. EV = HIGH (canonical academic-publication surface; required for OSS release).

**Integration complexity**: LOW.

**Priority**: P1 (after Phase 1 implementation; Catalog #431 dependency).

---

### 1.8 `hugging-face-tool-builder` — Reusable HF API tool scripts

**Core capabilities**: Build reusable command-line tools using HF API. Patterns: bash/python/typescript baseline scripts with `HF_TOKEN` auth; piping/chaining via `jq`; OpenAPI endpoint introspection via `jq` queries on `https://huggingface.co/.well-known/openapi.json`. Sample scripts: `hf_model_papers_auth.sh`, `find_models_by_paper.sh`, `hf_model_card_frontmatter.sh`, `hf_enrich_models.sh`.

**Applicability**: **DIRECTLY USEFUL (P1)** for canonical helper scripts.

**Specific use cases**:
1. **Canonical Comma2k19 downloader** per Catalog #213 — wrap `hf-cli` with substrate-specific download orchestration; emit `Comma2k19LocalCache.fetch_chunk` evidence per the canonical pattern.
2. **Canonical PR archive miner** — pull all comma-ai PR archives, extract metadata, push to `adpena/comma-pr-archive-corpus-54`.
3. **Canonical substrate-composition-matrix sync** — push `tac.optimization.substrate_composition_matrix` JSON → HF dataset every successful empirical anchor.
4. **Reuse `hf_enrich_models.sh` pattern** for our model-card enrichment (every substrate gets canonical model card metadata).
5. **Reuse OpenAPI discovery pattern** to find HF endpoints we haven't used (e.g., `/api/collections`).

**Estimated cost + EV**: Cost = $0. EV = MODERATE (canonical helper scripts replace 3-5 ad-hoc scripts).

**Integration complexity**: LOW-MEDIUM.

**Priority**: P1 (after Phase 1 canonical dataset upload).

---

### 1.9 `hugging-face-trackio` — Experiment tracking + alerts + Space syncing

**Core capabilities**: Real-time ML experiment tracking via Python API (`trackio.init`, `trackio.log`, `trackio.finish`, `trackio.alert`). Three alert levels (INFO/WARN/ERROR). Webhook integration (Slack/Discord). HF Space syncing for persistent dashboards. CLI `trackio list/get` with `--json` for automation. Autonomous agent workflows.

**Applicability**: **DIRECTLY USEFUL (P0)** — replaces every ad-hoc CSV/JSON logging across our substrate trainers.

**Specific use cases**:
1. **Replace ad-hoc CSV/JSON logging** in every substrate trainer's main loop. Currently `experiments/train_substrate_*.py` writes per-step metrics to local stdout + final stats JSON. Trackio centralizes.
2. **Real-time alerts** for loss divergence (Z6-style 22× miss anchor), NaN gradients, training stalls, MPS-vs-CUDA drift detection.
3. **Sister-subagent monitoring** — when a subagent dispatches a long training, the parent agent polls `trackio list alerts --json` instead of polling Modal logs directly.
4. **HF Space dashboard at `huggingface.co/spaces/adpena/trackio`** — operator-facing real-time view of every in-flight training.
5. **Cite-able run anchors** — Trackio runs have permanent URLs that anchor to specific commits + substrate IDs.
6. **PRIVATE Space initially** per CLAUDE.md "Public Disclosure Hygiene"; flip to public at release wave.

**Estimated cost + EV**: Cost = $0 (free tier sufficient). EV = HIGH (eliminates "which Modal job was that again?" friction).

**Integration complexity**: LOW (3-line integration: `trackio.init(...)`, `trackio.log({...})`, `trackio.finish()`).

**Priority**: P0 (one of the 5 main insights' enabler; cross-cutting infrastructure).

---

### 1.10 `hugging-face-vision-trainer` — Vision model training (the parent of 5 main insights)

**Core capabilities**: Train object detection (D-FINE, RT-DETR v2, DETR, YOLOS), image classification (timm models: MobileNetV3, MobileViT, ResNet, ViT/DINOv3), SAM/SAM2 segmentation on HF Jobs. COCO-format datasets. Albumentations augmentation. mAP/mAR/DiceCE metrics. Trackio always enabled. Hub persistence required. SAM2 trains mask decoder only (encoders frozen).

**Applicability**: **DIRECTLY USEFUL (P0)** — source of the 5 main insights.

**Specific use cases** (detailed in Section 2):
1. **L2 Hinton-distilled scorer surrogate** via `image_classification_training.py` + `sam_segmentation_training.py`.
2. **DINOv3 as cooperative-receiver target** via `timm/vit_base_patch16_dinov3.lvd1689m`.
3. **SAM2-hiera-tiny as PoseNet sister** via `facebook/sam2.1-hiera-tiny`.
4. **HF dataset upload** via `dataset_inspector.py` workflow.
5. **HF Jobs T4 ($0.40/hr) for training** vs Modal T4 ($0.59/hr).

**Estimated cost + EV**: Cost = $5-50 per main insight; total ~$20-50 for Phase 1+2. EV = VERY HIGH (5 main insights collectively predicted ΔS [-0.020, -0.005] vs 0.19205 baseline).

**Integration complexity**: MEDIUM-HIGH (each main insight is its own substrate-engineering effort).

**Priority**: P0 (THE source of the 5 main insights).

---

### 1.11 `huggingface-gradio` — Gradio web UIs

**Core capabilities**: Python web UI library. `Interface` (high-level wrapper), `Blocks` (low-level layout + event wiring), `ChatInterface`. Components: Textbox, Number, Slider, Checkbox, Dropdown, Radio, Image, Audio, Video, File, Chatbot, Button, Markdown, HTML. Custom HTML components via subclassing. Event listeners with concurrency control, batching, streaming, API exposure.

**Applicability**: **DIRECTLY USEFUL (P1)** — operator dashboard + paper supplement live demo.

**Specific use cases**:
1. **Operator dashboard** — single-page view of all in-flight Modal/Vast.ai/HF Jobs dispatches + Trackio metrics + cost-band posterior + dispatch claims + recent commits.
2. **Leaderboard visualizer** — fetch comma.ai leaderboard via webhook + render with PR# / author / score / date / our PRs highlighted.
3. **Substrate auditor UI** — interactive view of `tac.optimization.substrate_composition_matrix` + per-substrate symposium verdicts + lane registry state.
4. **Paper supplement live demo** — upload your own dashcam video → compress with our best substrate → show inflated output + score components (PR #20 paper supplement extension).
5. **Frame-by-frame diff viewer** — visualize SegNet mask diff between our substrate and ground truth.
6. **PRIVATE Space initially** per CLAUDE.md "Public Disclosure Hygiene"; subset of features made public at release wave.

**Estimated cost + EV**: Cost = $0 (free tier). EV = HIGH (operator-facing leverage compounds over time).

**Integration complexity**: MEDIUM (~500-1000 LOC for full operator dashboard).

**Priority**: P1 (after Phase 1 dataset upload + Trackio integration).

---

### 1.12 `transformers.js` — JS/TS runtime for ML models

**Core capabilities**: Run ML models in JavaScript (Node.js + browser). Pipeline API for NLP, CV, audio, multimodal. WebGPU + WASM backends. Quantization (4-bit, 5-bit, 8-bit). Memory management via `pipe.dispose()`.

**Applicability**: **NOT USEFUL** for our problem. Our entire stack is Python; substrate trainers, inflate runtime, scorer evaluation all run in PyTorch + CUDA + Python.

**Specific use cases**: NONE for substrate work.

**Hypothetical future use cases (excluded from this plan)**:
1. Browser-based paper supplement demo (would require duplicating our inflate runtime in JS — high cost, no score benefit).
2. Browser-based leaderboard demo (Gradio + Python is the canonical path).

**Estimated cost + EV**: N/A. **SKIP**.

**Integration complexity**: N/A.

**Priority**: P3 (skip; do not invoke).

---

## Section 2: 5 main insights' implementation design

### 2.1 L2 Hinton-distilled scorer surrogate (Catalog #523 long-deferred)

**Background**: Catalog #523 (`distill_scorer_surrogate_for_inflate_time_score_estimation`) has been deferred since the substrate canvas began. The structural value: a 1-5M-parameter surrogate that approximates SegNet + PoseNet outputs at INFERENCE time (NOT inflate time — strict scorer rule per CLAUDE.md Catalog #6 forbids scorers in inflate) for use inside the LOSS FUNCTION during substrate training. This removes the expensive SegNet (EfficientNet-B2, ~9M params) + PoseNet (FastViT-T12, ~4M params) from the training inner loop while preserving score gradient fidelity per Hinton's 2014 distillation discipline (`feedback_codex_finding_pr101_synthetic_targets_recursive_review_20260508.md` cites the relevant council).

**Stage 1: SegNet surrogate**

**Architecture choice**: `timm/mobilevit_s.cvnets_in1k` (5.6M params) — Mobile transformer with good accuracy/speed trade-off per the vision-trainer skill's recommendation table.

**ALTERNATIVE**: `timm/mobilenetv3_small_100.lamb_in1k` (2.5M params) — Ultra-lightweight. Use as PHASE 1 SMOKE first; if accuracy gap too large, switch to MobileViT-S.

**Why NOT SAM2-hiera-tiny for SegNet**: SAM2 is mask-decoder-only training (encoder frozen); the encoder is already 38.9M params (larger than SegNet itself). Doesn't reduce inner-loop cost.

**Dataset**: `adpena/comma-video-substrate-eval-600pairs` (canonical 600-pair dataset; see Section 2.4 for schema).

**Distillation script design**:
- Custom (NOT canonical `image_classification_training.py` directly; CANONICAL FORKED BECAUSE SUPPRESSES per Catalog #290) — the canonical script targets 1000-class ImageNet classification; we need 5-class per-pixel logit regression at 384×512.
- Loss = `kl_div(student_logits / T, teacher_logits / T) * T^2` per Hinton T=2.0 (matches Quantizr's published kl_on_logits per CLAUDE.md "Quantizr intelligence" section).
- Student input: full 384×512×3 RGB frame.
- Teacher input: same frame; SegNet forward with `x[:, -1, ...]` slice (canonical scorer-preprocess routing per Catalog #164).
- Optimizer: AdamW lr=5e-5; cosine schedule.
- Epochs: 30-50; batch_size=32.
- Dispatch: HF Jobs `t4-small` ($0.40/hr) × 2-4 hours = **$0.80-1.60 per stage**.

**Stage 2: PoseNet surrogate**

**Architecture choice**: Custom `MobileViTPoseHead` — `timm/mobilevit_s.cvnets_in1k` encoder + 256-dim → 6-DOF pose regression head.

**ALTERNATIVE**: SAM2-hiera-tiny mask-decoder repurposed via `PoseRegressionHead(in_dim=256, out_dim=6)` swap-in. **DEFERRED-PENDING-COUNCIL** because the mask-decoder gradient path may not generalize to 6-DOF regression (cargo-cult-risk per the cargo-cult audit table above).

**Dataset**: same `adpena/comma-video-substrate-eval-600pairs`.

**Distillation script design**:
- Student input: TWO frames (frame_0 + frame_1) at 384×512×3 each.
- Teacher input: same two frames; PoseNet forward = `rgb_to_yuv6(...)` → resize (512,384) → normalize → FastViT-T12 → Hydra head → 6-DOF pose.
- Loss = MSE on first 6 pose dimensions (matches upstream `evaluate.py` exact formula).
- CRITICAL: differentiable `rgb_to_yuv6` per CLAUDE.md "eval_roundtrip" non-negotiable — patch upstream globally via `tac.differentiable_eval_roundtrip.patch_upstream_yuv6_globally()`.
- Optimizer: AdamW lr=5e-5; cosine schedule.
- Epochs: 30-50; batch_size=16 (two-frame input = 2x memory).
- Dispatch: HF Jobs `t4-small` ($0.40/hr) × 2-4 hours = **$0.80-1.60 per stage**.

**Trackio integration**:
- `trackio.init(project="scorer-surrogate-distillation", run_name="segnet_mobilevit_s_v1", space_id="adpena/trackio")` (PRIVATE Space initially).
- Log per-step: `student_loss`, `teacher_kl_divergence`, `lr`, `grad_norm`.
- Log per-epoch: `eval_l2_to_teacher`, `eval_score_drift` (compute auth_eval on a fixed archive with student vs teacher; report the score drift).
- `trackio.alert(level=ERROR)` if `eval_score_drift > 0.005` (5x the per-archive jitter; signals gradient-path mismatch).

**Model card + evaluation entry**:
- `evaluation_manager.py extract-readme` workflow to add `eval_l2_to_teacher` + `eval_score_drift` metrics to `adpena/comma-segnet-surrogate-mobilevit-s` and `adpena/comma-posenet-surrogate-mobilevit-s` model cards.

**Estimated cost (total)**: $1.60-4.00 across both stages on HF Jobs T4.

**Predicted ΔS impact**: NOT a direct score improvement — this is INFRASTRUCTURE for faster substrate training (faster smoke iteration). Downstream impact: every substrate trainer using the surrogate gets 5-10x training speedup, enabling MORE iterations within the same budget. Indirect ΔS [-0.005, -0.015] across the substrate canvas via faster cargo-cult-unwind iteration cycles.

**Catalog compliance**:
- Catalog #6 strict-scorer-rule satisfied — surrogate used at TRAINING time, NOT inflate time.
- Catalog #8 eval_roundtrip satisfied via differentiable rgb_to_yuv6 patch.
- Catalog #325 per-substrate symposium NOT YET — this is infrastructure, not a substrate.
- Catalog #287 docstring overstatement tag: all metric claims tagged `[empirical:experiments/results/scorer_surrogate_<ts>/eval.json]` or `[prediction]`.

---

### 2.2 DINOv3 as Z6/ATW-V2-1/TT5L-V2 empirical anchor

**Background**: Per `feedback_deep_research_wave_landed_20260518.md`, TT5L V2 + VGGT (CVPR 2025 Best Paper arxiv 2503.11651) + DUSt3R/MASt3R + NVIDIA VRSS 2 + DreamerV3 RSSM categorical predict ΔS [-0.020, -0.008] → [0.172, 0.184]. DINOv3 is the empirical anchor for cooperative-receiver substrates (Z6 4c FiLM ego-motion / ATW V2-1 codec / TT5L V2 foveation+LAPose).

**DINOv3 pretrained checkpoint**: `timm/vit_base_patch16_dinov3.lvd1689m` (86.6M params, ViT-B/16 self-supervised on LVD-1689M).

**Use frozen vs fine-tune decision**:

**Option A (RECOMMENDED for Phase 1): Frozen DINOv3 as cooperative-receiver target distribution**.
- Load frozen DINOv3 — extract per-frame 768-dim CLS token embeddings.
- Use as TARGET in cooperative-receiver loss: `L_coop = -log p(z_dinov3 | z_substrate)` where `z_substrate` is the substrate's latent encoding.
- The cooperative-receiver claim (per Atick-Redlich 1990) is that the substrate's encoder should preserve the information the receiver uses for downstream tasks. DINOv3 is the "receiver" surrogate.
- Catalog #311 ego-motion conditioning compatibility: DINOv3 features are PER-FRAME so they preserve spatial-only info; ego-motion needs PAIRWISE features. Combine with VGGT for pairwise ego-motion → DINOv3 for per-frame distribution.

**Option B (Phase 3+): Fine-tune DINOv3 on `adpena/comma-video-substrate-eval-600pairs`**.
- Fine-tune the last 2 transformer blocks of DINOv3 to match SegNet/PoseNet output distributions.
- Cost: HF Jobs L4 ($0.80/hr) × 4 hours = $3.20.
- DEFERRED until Option A empirical anchor confirms cooperative-receiver claim.

**Cargo-cult audit (per Catalog #303)**:
- ASSUMPTION-1: "DINOv3 features generalize from ImageNet to dashcam video" → CARGO-CULTED-INHERITED-FROM-LITERATURE. Unwind via Stage 1 paired comparison: extract DINOv3 features on 100 dashcam frames + 100 ImageNet frames; measure cosine-similarity distribution. If overlap is high, generalization is plausible.
- ASSUMPTION-2: "Cooperative-receiver loss is the canonical Z6/ATW-V2-1 formulation" → HARD-EARNED per `feedback_deep_research_wave_landed_20260518.md` cross-disciplinary triangulation (Atick-Redlich ↔ Rao-Ballard ↔ Friston ↔ DreamerV3 ↔ JEPA ↔ Schmidhuber).

**Integration with Z6 4c**: Z6 4c (FiLM ego-motion) consumes ego-motion conditioning from VGGT/DUSt3R; DINOv3 features provide the cooperative-receiver target. The two compose orthogonally.

**Estimated cost (Phase 1 Option A)**: $0 (frozen DINOv3 download + inference; no training).

**Predicted ΔS**: [-0.020, -0.008] per `feedback_deep_research_wave_landed_20260518.md`.

**Catalog compliance**:
- Catalog #325 per-substrate symposium REQUIRED before Phase 2 paid dispatch.
- Catalog #324 post-training Tier-C validation REQUIRED before promotion.
- Catalog #292 per-deliberation Assumption-Adversary verdict REQUIRED.
- Catalog #303 cargo-cult audit section REQUIRED in design memo for Z6 4c / ATW V2-1 / TT5L V2 updated designs.

---

### 2.3 SAM2-hiera-tiny as PoseNet sister

**Background**: SAM2-hiera-tiny (38.9M params, `facebook/sam2.1-hiera-tiny`) has a frozen vision encoder + frozen prompt encoder + trainable mask decoder. The Catalog #779 "freezing exploit" insight: mask-decoder-only training is structurally elegant — preserves the canonical encoder + only adapts the decoder to a new task.

**Cross-pollination idea**: Use SAM2's mask decoder as a **PoseNet sister** in a predictive-coding framework.

**Architecture**:
- SAM2 encoder (frozen) extracts per-frame features from frame_0.
- Custom prompt: the current frame's SegNet mask (5-channel argmax) becomes the bbox prompt(s).
- SAM2 mask decoder predicts: "given frame_0's features + frame_0's mask, what is frame_1's mask?"
- Compare predicted mask to actual frame_1 mask (after PoseNet-predicted warping).
- The PREDICTION ERROR is the signal: high error = pose mismatch.

**Cross-pollination with Z6 4c (predictive coding)**: Z6 4c IS the predictive-coding substrate. SAM2-as-mask-predictor provides the per-pair PREDICTION TARGET that Z6 4c's encoder tries to match. The two compose:
- Z6 4c encoder predicts frame_1 from frame_0 + pose.
- SAM2 mask decoder VERIFIES the prediction by checking mask-level consistency.
- The verification loss IS the cooperative-receiver loss (Section 2.2).

**Mask-decoder-only training pattern (Catalog #779)**:
- Freeze SAM2 vision encoder (~32M params); freeze prompt encoder (~5M params).
- Train ONLY mask decoder (~2M params).
- Cost: minimal (small trainable surface).
- Loss: DiceCE per the canonical vision-trainer skill recommendation for SAM2 segmentation.

**Dispatch**: HF Jobs `t4-small` ($0.40/hr) × 2 hours = **$0.80**.

**Cargo-cult audit**:
- ASSUMPTION-1: "SAM2 mask decoder can be repurposed for frame-prediction" → CARGO-CULTED-CROSS-DOMAIN. Unwind: paired comparison vs a custom 2M-param mask predictor with same architecture but random-init weights. If SAM2 initialization helps, claim is HARD-EARNED.
- ASSUMPTION-2: "Mask-decoder-only training preserves SAM2's segmentation quality" → HARD-EARNED per SAM2 paper + the vision-trainer skill's documented pattern.

**Predicted ΔS**: composes with Z6 4c (Section 2.2 cooperative-receiver). NOT a standalone substrate — composition stack.

**Catalog compliance**: Same as Section 2.2 (Catalog #325 + #324 + #292 + #303).

---

### 2.4 HF dataset upload (canonical reproducibility)

**Dataset**: `adpena/comma-video-substrate-eval-600pairs`

**Schema design** (Parquet, single config "default", single split "eval"):

```
pair_idx           int32             # 0-599 (600 non-overlapping pairs from upstream/videos/0.mkv)
frame_0_rgb        binary (PNG)      # 384×512×3 RGB, lossless PNG encoding
frame_1_rgb        binary (PNG)      # 384×512×3 RGB
segnet_mask_0_5ch  binary (float16)  # 5×384×512 SegNet logits for frame_0
segnet_mask_1_5ch  binary (float16)  # 5×384×512 SegNet logits for frame_1
posenet_pose_6d    list<float32>     # 6-DOF pose vector (PoseNet output, first 6 dims)
pair_sha256        string            # SHA-256 of (frame_0_rgb || frame_1_rgb || pair_idx)
archive_sha_baseline string          # canonical baseline archive sha for reproducibility
```

**Upload pipeline** (canonical helper at `tools/build_comma_video_substrate_eval_dataset.py`):

1. Decode `upstream/videos/0.mkv` via pyav (canonical per Catalog #181 `pyav_decode_strategy`).
2. For each of 600 non-overlapping pairs (per upstream `evaluate.py:seq_len=2`):
   - Extract frame_0 + frame_1 as PIL Image.
   - Run SegNet forward (canonical scorer-preprocess routing per Catalog #164).
   - Run PoseNet forward (canonical eval_roundtrip discipline per Catalog #8).
   - Hash pair → `pair_sha256`.
3. Write to local Parquet via `datasets.Dataset.from_dict(...).to_parquet(...)`.
4. Upload via `hf upload datasets/adpena/comma-video-substrate-eval-600pairs ./local-parquet data` per `hugging-face-datasets` skill.
5. Generate dataset card with schema documentation + canonical baseline archive sha + Comma2k19 attribution + CC-BY-4.0 license.

**Cross-substrate consumption** (every substrate trainer can pull from canonical HF dataset):

```python
from datasets import load_dataset

# Replace ad-hoc pyav decode in every substrate trainer:
ds = load_dataset("adpena/comma-video-substrate-eval-600pairs", split="eval")

for pair in ds:
    frame_0 = pair["frame_0_rgb"]  # PIL Image
    frame_1 = pair["frame_1_rgb"]
    segnet_target = pair["segnet_mask_0_5ch"]  # for distillation
    pose_target = pair["posenet_pose_6d"]
    # ... train substrate
```

**Catalog compliance**:
- Catalog #209 Comma2k19 contest-video-leakage detection: NOT TRIGGERED — `upstream/videos/0.mkv` IS the contest video; this dataset is DERIVED from the contest video for SUBSTRATE TRAINING use, not predictive-prior pretraining (which is what #209 protects against). Add dataset card disclaimer: "Derived from upstream/videos/0.mkv — DO NOT USE FOR PREDICTIVE-PRIOR PRETRAINING; this is a SUBSTRATE EVAL dataset only."
- Catalog #213 Comma2k19 canonical-cache: NOT APPLICABLE — different dataset.
- Catalog #210 codebook provenance: NOT APPLICABLE — this is an eval dataset, not a codebook.
- Per CLAUDE.md "Public Disclosure Hygiene": PRIVATE Hub repo initially (private=True flag); flip to public at OSS release wave.

**Estimated cost**: $0 (free tier; ~500 MB dataset).

**EV**: VERY HIGH — every substrate trainer + every distillation script + every paper supplement consumes this canonical dataset.

**Predicted ΔS**: NOT a direct improvement — INFRASTRUCTURE primitive that unlocks 5+ downstream improvements.

---

### 2.5 Hardware re-routing (HF Jobs as Modal replacement)

**Cost comparison table** (verified 2026-05-18 against current public pricing):

| Provider | Hardware | $/hr | Wall-clock for canary smoke (100ep) | Total $ per smoke |
|---|---|---|---|---|
| Modal | T4 | $0.59 | 25-40 min | $0.25-0.40 |
| Modal | A10G | $1.40 | 15-25 min | $0.35-0.60 |
| Modal | A100 | $4.00 | 5-10 min | $0.35-0.70 |
| HF Jobs | t4-small | $0.40 | 25-40 min | $0.17-0.27 |
| HF Jobs | l4x1 | $0.80 | 15-25 min | $0.20-0.34 |
| HF Jobs | a10g-large | $1.50 | 12-20 min | $0.30-0.50 |
| HF Jobs | a100-large | $2.50 | 5-10 min | $0.21-0.42 |
| Vast.ai | RTX 4090 | $0.25 | 10-15 min | $0.04-0.06 |
| Lightning | A100 | $0/hr subscription | 5-10 min | $0 (within subscription cap) |

**Key insight**: HF Jobs T4 is 32% cheaper than Modal T4 and A100 is 37% cheaper.

**Which substrate trainers should migrate to HF Jobs?**

**RECOMMENDED MIGRATION (Phase 2)**:
1. **Substrate trainers that produce <$1 smokes** — HF Jobs T4 captures most cost reduction here.
2. **Substrate trainers without Modal-specific dependencies** — most of our trainers are pure PyTorch + canonical scorer loaders.
3. **Substrate trainers that already use `tac.deploy.modal.runtime` canonical env block** — easier port to HF Jobs.

**Concrete migration targets** (in priority order):
- `experiments/train_substrate_pretrained_driving_prior.py` (DP1 — currently Modal T4 + Modal A10G).
- `experiments/train_substrate_self_compress_nn.py` (SCPP — currently Modal T4).
- `experiments/train_substrate_pr101_lc_v2_clone_enhanced_curriculum.py` (PR101 — currently Modal A100).
- `experiments/train_substrate_balle_renderer.py` (NSCS03 — currently Modal T4 + L4).
- Sister substrates that have green smoke history.

**NOT RECOMMENDED MIGRATION**:
- Substrates with active Modal-specific debugging (e.g. recent NVML 999 fixes per Catalog #244).
- Substrates with custom remote_lane_substrate_*.sh drivers (would need parallel HF Jobs version per Catalog #326 driver-mode-hardcode).

**Migration cost** (1-2h per trainer):
1. Adapt PEP 723 header to declare full HF Jobs UV dependency list.
2. Replace Modal `image.env(...)` block with `env={...}` kwarg on `hf_jobs("uv", {...})`.
3. Replace Modal volume mounts with HF Hub dataset loads (Section 2.4).
4. Add `secrets={"HF_TOKEN": "$HF_TOKEN"}` for Hub push.
5. Migrate Modal call_id ledger writes (Catalog #245) to HF Jobs job_id ledger (NEW canonical helper).
6. Catalog #270 dispatch optimization protocol compliance: per Catalog #270 substrate trainer tier requirements (autocast_fp16 / TF32 / torch.compile / no_grad / GTScorerCache / canonical scorer-loss helper); these are CALLER-AGNOSTIC so HF Jobs migration doesn't require changes.

**Catalog #166 source-parity hash**: HF Jobs UV script approach already mounts the script content directly (no separate working-tree sync); Modal's Catalog #166 worker-side hash discipline does not transfer 1-to-1. **DEFERRED-PENDING-RESEARCH**: how to enforce equivalent source-parity discipline for HF Jobs (the script CONTENT is hashed, but transitively imported `tac.*` modules are pulled from PyPI / git+ssh, not the working tree). Risk: a HF Jobs dispatch can run against a STALE working tree state if the operator forgets to sync.

**OPERATOR DECISION REQUIRED**: Pro/Team/Enterprise plan tier verification (Jobs require paid plan per `hugging-face-jobs` SKILL.md).

**Predicted savings** (Phase 2 scope):
- 13 substrate trainers × 4-6 smokes per week × ~$0.30 per smoke × 32% reduction = **$5-10 per week saved**.
- Full canvas dispatches at A100 capture 37% reduction × ~$0.40 per full = **~$1.50 per dispatch saved**.

**Predicted EV**: cost savings are MODEST in absolute terms ($20-40/month) but compounding gains are LARGE (cheaper smokes = more iteration cycles per dollar = faster cargo-cult-unwind).

**Catalog #270 compliance**: each migrated trainer must continue to declare Tier 1/2/3 fields in its `TIER_<N>_OPERATOR_REQUIRED_FLAGS` manifest (Catalog #151) so HF Jobs dispatch wrappers (sister to operator-authorize) inherit the full dispatch optimization protocol per Catalog #270 + #317.

---

## Section 3: NEW insights from the other 11 skills

Beyond the 5 main insights from vision-trainer, the other 11 skills surface NEW high-EV use cases:

### 3.1 `hf-cli`: canonical CLI for `comma-pr-archive-corpus-54` dataset maintenance

**Insight**: The 54-PR-archive corpus (referenced as `#368` in MEMORY.md) is currently UNPUBLISHED. The `hf-cli` provides the canonical surface to upload + maintain it.

**Use case**:
- `hf upload datasets/adpena/comma-pr-archive-corpus-54 ./pr-archive-data data` — single-command upload.
- `hf datasets parquet adpena/comma-pr-archive-corpus-54` — list shard URLs for SQL queries.
- `hf collections create "comma.ai video compression — 13 substrates"` — canonical collection linking all our published artifacts.

**Estimated cost**: $0.

**EV**: HIGH — enables every downstream insight (Section 3.2-3.7).

**Priority**: P1 (after Phase 1 canonical dataset upload of `comma-video-substrate-eval-600pairs`).

### 3.2 `hugging-face-dataset-viewer`: SQL-query 54-PR-archive corpus for unrealized patterns

**Insight**: Once `comma-pr-archive-corpus-54` is uploaded, SQL queries can surface empirical patterns the apparatus hasn't surfaced manually.

**High-EV queries**:

```sql
-- Q1: PRs with the largest CPU/CUDA gap (PR102 anchor = 0.03)
SELECT pr_number, score_cpu, score_cuda, (score_cuda - score_cpu) AS gap
FROM data
WHERE score_cpu IS NOT NULL AND score_cuda IS NOT NULL
ORDER BY gap DESC LIMIT 20;

-- Q2: Pareto-frontier outliers (PRs that beat the convex hull on rate vs distortion)
SELECT pr_number, rate, seg_distortion, pose_distortion, final_score
FROM data
WHERE final_score < 0.20 AND rate > 0.20
ORDER BY final_score ASC LIMIT 20;

-- Q3: PRs whose archive size is in the bottom 10% — small-archive winners
SELECT pr_number, archive_bytes_total, final_score
FROM data
WHERE archive_bytes_total < (SELECT QUANTILE(archive_bytes_total, 0.1) FROM data)
ORDER BY final_score ASC;

-- Q4: PRs with the highest "score per byte" — efficiency winners
SELECT pr_number, archive_bytes_total, final_score,
       1.0 / final_score / (archive_bytes_total / 1000.0) AS efficiency_score
FROM data
ORDER BY efficiency_score DESC LIMIT 20;

-- Q5: Technique tag co-occurrence — which tag pairs appear in winning PRs
SELECT tag_a, tag_b, COUNT(*) AS co_occurrence,
       AVG(final_score) AS avg_score
FROM (
  SELECT pr_number, UNNEST(technique_tags) AS tag_a, UNNEST(technique_tags) AS tag_b
  FROM data
)
WHERE tag_a < tag_b
GROUP BY tag_a, tag_b
HAVING co_occurrence >= 3
ORDER BY avg_score ASC LIMIT 50;
```

**Estimated cost**: $0 (read-only SQL via DuckDB).

**EV**: VERY HIGH — Q5 in particular could surface NEW composition opportunities the apparatus hasn't manually enumerated.

**Priority**: P1 (after `comma-pr-archive-corpus-54` upload).

### 3.3 `hugging-face-datasets`: SQL via DuckDB on `hf://` for cross-archive analysis

**Insight**: DuckDB on `hf://` enables cross-substrate metric mining at the per-substrate, per-archive, per-pair grain.

**High-EV use case**: Build `adpena/comma-substrate-empirical-anchors` dataset — every empirical anchor from our `tac.continual_learning.continual_learning_posterior.jsonl` exported as queryable rows. Schema: `{substrate_id, lane_id, archive_sha256, score_cpu, score_cuda, evidence_grade, hardware_substrate, dispatch_cost_usd, written_at_utc, ...}`. Then cross-substrate queries:

```sql
-- Top substrates by best contest-CUDA score
SELECT substrate_id, MIN(score_cuda) AS best_cuda, COUNT(*) AS n_dispatches
FROM data WHERE score_cuda IS NOT NULL
GROUP BY substrate_id ORDER BY best_cuda ASC LIMIT 20;

-- Substrates with the worst predicted-vs-empirical band miss (Catalog #324)
SELECT substrate_id, lane_id, predicted_band_lo, predicted_band_hi, score_cuda,
       CASE WHEN score_cuda > predicted_band_hi THEN score_cuda - predicted_band_hi ELSE 0 END AS miss_above
FROM data WHERE predicted_band_hi IS NOT NULL
ORDER BY miss_above DESC LIMIT 20;
```

**Estimated cost**: $0.

**EV**: HIGH — enables cross-substrate audit + predicted-band calibration sweep.

**Priority**: P1 (after Phase 1 canonical upload).

### 3.4 `hugging-face-evaluation`: model card eval entries for published substrates

**Insight**: Catalog #316 frontier preservation discipline is currently OPERATOR-FACING via `reports/latest.md`. The HF model-index entry is the CANONICAL public-facing surface.

**Use case**: For each of our 13 published substrates (PR101 lc_v2, fec6 selector, PR106 format0d, NSCS03, A1, DP1, etc.):
1. Run `evaluation_manager.py get-prs --repo-id adpena/<substrate>` (mandatory pre-check per skill).
2. Extract eval table from existing README via `extract-readme`.
3. Add `[contest-CUDA]` + `[contest-CPU]` scores to `model-index` entry.
4. Push via `--apply` (own models; not other-author models).

**Estimated cost**: $0.

**EV**: MODERATE-HIGH — canonical academic + leaderboard surface.

**Priority**: P1 (after each substrate has at least one published anchor).

### 3.5 `hugging-face-jobs`: scheduled jobs for nightly substrate sweeps + Modal harvester

**Insight**: The Modal call_id ledger (Catalog #245) currently relies on manual harvest. HF Jobs `@hourly` scheduling can automate this + sister chores.

**Scheduled job designs**:

1. **`@hourly` — Modal call_id harvest**:
   - Run `tools/harvest_modal_calls.py` on all unharvested `modal_metadata.json` files.
   - Push new harvest rows to `adpena/comma-substrate-empirical-anchors` HF dataset.
   - Cost: HF Jobs cpu-basic ($0.10/hr) × ~2 min per run × 24 runs/day × 30 days = **$1/month**.

2. **`@daily` — Lane registry audit**:
   - Run `tools/lane_maturity.py audit --json` + push report to `adpena/comma-lab-state` dataset.
   - Detect stale L1 substrates per Catalog #298.
   - Surface lanes pending Tier-C validation per Catalog #324.
   - Cost: cpu-basic × 1 min = **$0.05/day = $1.50/month**.

3. **`@hourly` — comma.ai leaderboard fetch**:
   - Fetch comma.ai contest leaderboard via webhook OR scrape.
   - Detect leaderboard moves per CLAUDE.md "Race-mode rigor inversion" non-negotiable.
   - Alert operator via Trackio webhook (Slack) on rank changes.
   - Cost: cpu-basic × 1 min × 24 × 30 = **$0.75/month**.

4. **`@hourly` — Trackio metric snapshot**:
   - Pull active Trackio runs + emit summary report.
   - Cost: $0 (Trackio CLI is free + cpu-basic 1 min/hour).

**Total scheduled-jobs cost**: ~$5/month.

**EV**: VERY HIGH — automation eliminates manual harvest friction; race-mode detection wins.

**Priority**: P2 (after HF Jobs migration of substrate trainers).

### 3.6 `hugging-face-paper-publisher`: automated paper page + author claim

**Insight**: Catalog #431 (arXiv paper) is in PHASE 4 INTEGRATION PENDING. The paper-publisher skill automates the post-arXiv-submission steps.

**Workflow** (per skill's Workflow 1):
1. `paper_manager.py create --template modern --title "comma.ai Video Compression: Substrate Engineering at the Plateau" --output paper.md`.
2. Edit paper.md with content (operator + agent collaboration).
3. Submit to arXiv (external process).
4. `paper_manager.py index --arxiv-id "<id>"` — index on HF Paper Pages.
5. `paper_manager.py link --repo-id adpena/<substrate> --repo-type model --arxiv-id "<id>"` × 13 substrates × N sister datasets.
6. `paper_manager.py claim --arxiv-id "<id>" --email operator-email`.

**Estimated cost**: $0.

**EV**: HIGH (academic-publication discipline; required for OSS release).

**Priority**: P1 (Catalog #431 dependency).

### 3.7 `hugging-face-tool-builder`: canonical helper scripts

**Insight**: 3-5 ad-hoc scripts in our `tools/` directory can be canonicalized via the tool-builder skill's pattern.

**Canonical scripts to land**:

1. **`tools/hf_comma2k19_downloader.sh`** — wrap `Comma2k19LocalCache.fetch_chunk` per Catalog #213; emit canonical evidence per CLAUDE.md "Comma2k19 canonical helper" non-negotiable.

2. **`tools/hf_pr_archive_miner.sh`** — pull all comma-ai PR archives via `gh` CLI + push to `adpena/comma-pr-archive-corpus-54`. Composable: `tools/hf_pr_archive_miner.sh | jq '.[] | {pr, score}' | head -10`.

3. **`tools/hf_substrate_composition_sync.sh`** — push `tac.optimization.substrate_composition_matrix` → `adpena/comma-substrate-composition-matrix` HF dataset every successful empirical anchor. Hooked into Catalog #322 phantom-provenance gate as post-validation step.

4. **`tools/hf_council_anchor_sync.sh`** — push `.omx/state/council_deliberation_posterior.jsonl` → `adpena/comma-council-deliberation-posterior` HF dataset every T2+ council landing.

5. **`tools/hf_trackio_alert_to_slack.sh`** — wrap Trackio webhook for operator-facing Slack alerts.

**Estimated cost**: $0.

**EV**: HIGH (canonical surface; eliminates ad-hoc script proliferation).

**Priority**: P2 (after Phase 1 canonical datasets exist).

### 3.8 `hugging-face-trackio`: replaces ad-hoc CSV/JSON logging

**Insight**: Already covered in Section 1.9. KEY DIFFERENTIATOR vs ad-hoc: REAL-TIME Space dashboard + autonomous-agent alert workflow.

**HIGH-EV alert pattern** (for substrate trainers):

```python
import trackio

trackio.init(
    project=f"substrate-{substrate_id}",
    run_name=f"{substrate_id}_{config_hash}_{utc_timestamp}",
    space_id="adpena/trackio",  # PRIVATE Space
    config={
        "substrate_id": substrate_id,
        "archive_sha_baseline": archive_sha,
        "predicted_band_lo": predicted_band[0],
        "predicted_band_hi": predicted_band[1],
        "lane_id": lane_id,
    }
)

for step in range(num_steps):
    metrics = train_step()
    trackio.log(metrics)
    
    # Catalog #324 post-training band check
    if step == final_step:
        actual_score = run_auth_eval()
        if actual_score > predicted_band[1] * 2:
            trackio.alert(
                title="Predicted band miss >2× (Catalog #324 anchor)",
                text=f"Predicted [{predicted_band[0]:.3f}, {predicted_band[1]:.3f}]; "
                     f"actual {actual_score:.3f} = {actual_score/predicted_band[1]:.1f}× over",
                level=trackio.AlertLevel.ERROR,
            )
    
    # Detect MPS-vs-CUDA drift (CLAUDE.md "MPS auth eval is NOISE")
    if hardware_substrate == "macos_arm64" and abs(actual_score - cuda_anchor) > 0.005:
        trackio.alert(
            title="MPS-vs-CUDA drift detected",
            text=f"MPS {actual_score:.4f} vs CUDA anchor {cuda_anchor:.4f}",
            level=trackio.AlertLevel.WARN,
        )

trackio.finish()
```

**Then sister parent agent polls**:

```bash
trackio list alerts --project substrate-c6_e4_mdl_ibps --json --since "2026-05-18T00:00:00"
```

**Estimated cost**: $0.

**EV**: VERY HIGH — eliminates "which Modal job was that again?" friction + provides AUTONOMOUS agent monitoring.

**Priority**: P0 (cross-cutting infrastructure).

### 3.9 `huggingface-gradio`: operator dashboard + leaderboard visualizer + substrate auditor UI + paper supplement live demo

**Insight**: Gradio enables 4 distinct UIs that compound operator leverage:

**UI 1: Operator dashboard** (PRIVATE Space `adpena/comma-operator-dashboard`):
- Active Modal/HF Jobs/Vast.ai dispatches (pulled from canonical ledgers).
- Cost-band posterior (last 30 days).
- Recent Trackio alerts (last 24h).
- Lane registry audit table (filter by L1/L2/L3).
- Council deliberation posterior (last 7 days).
- Substrate composition matrix (sortable).
- ~500 LOC.

**UI 2: Leaderboard visualizer** (PRIVATE initially; PUBLIC at release):
- Fetch comma.ai leaderboard.
- Plot PR# vs score (line chart, our PRs highlighted).
- Detect rank changes vs baseline.
- ~300 LOC.

**UI 3: Substrate auditor UI**:
- Filter substrates by horizon_class (plateau-adjacent / frontier-pursuit / asymptotic-pursuit).
- Drill-down per substrate: design memo + cargo-cult audit + per-pair score breakdown.
- Trigger sextet symposium request (form submission → operator queue).
- ~400 LOC.

**UI 4: Paper supplement live demo** (PUBLIC at release, gated until then):
- Upload your own dashcam video (max 10s) → compress with our best substrate → show inflated output + score components.
- Side-by-side ground-truth vs reconstruction.
- SegNet mask diff visualization.
- ~600 LOC.

**Total Gradio LOC**: ~1800 LOC across 4 UIs.

**Estimated cost**: $0 (free tier for hosting; Spaces dev mode).

**EV**: VERY HIGH (operator-facing leverage compounds; paper supplement is core OSS-release artifact).

**Priority**: P1-P2 staged (UI 1 first → UI 2/3 next → UI 4 last for release).

### 3.10 (transformers.js — SKIP per Section 1.12)

---

## Section 4: Sequenced implementation plan

### Phase 1: ~$3 (first-strike infrastructure)

**Goal**: Build canonical reproducibility primitives. ZERO substrate score improvements yet — INFRASTRUCTURE only.

| Task | Cost | Wall-clock | Skill |
|---|---|---|---|
| Build `adpena/comma-video-substrate-eval-600pairs` dataset upload pipeline | $0 (local CPU) | 2-3h editor | datasets |
| Upload `adpena/comma-video-substrate-eval-600pairs` (PRIVATE) | $0 | 30 min | hf-cli |
| Build `adpena/comma-pr-archive-corpus-54` upload pipeline | $0 (local CPU) | 2-3h editor | datasets + tool-builder |
| Upload `adpena/comma-pr-archive-corpus-54` (PRIVATE) | $0 | 30 min | hf-cli |
| Distill SegNet surrogate v1 (timm/mobilenetv3_small_100, 30ep smoke) | $0.80-1.60 | 2-4h HF Jobs T4 | vision-trainer + trackio |
| Set up Trackio Space `adpena/trackio` (PRIVATE) + alert webhook | $0 | 30 min | trackio + hf-cli |
| Build minimal Gradio operator dashboard (UI 1; 500 LOC) | $0 | 4-6h editor | gradio + hf-cli |
| Add `model-index` eval entries to 3-5 most-recent published substrates | $0 | 1-2h | evaluation |

**Total Phase 1 budget**: ~$3.
**Total Phase 1 wall-clock**: ~2 sessions (16-24h editor + GPU).

**Deliverables**:
- 2 canonical HF datasets uploaded (PRIVATE).
- 1 working SegNet surrogate (cite-able + benchmarkable).
- Trackio infrastructure live (every substrate trainer can `trackio.init(...)` from this point).
- Operator dashboard live at `huggingface.co/spaces/adpena/comma-operator-dashboard`.
- 3-5 substrates have canonical model-index eval entries.

### Phase 2: ~$20 (second-strike substrate score improvements)

**Goal**: Land 2-3 main insights with predicted ΔS [-0.005, -0.020].

| Task | Cost | Wall-clock | Skill |
|---|---|---|---|
| Distill PoseNet surrogate v1 (MobileViT-S 30ep) | $0.80-1.60 | 2-4h HF Jobs T4 | vision-trainer + trackio |
| DINOv3 frozen-feature extraction on `comma-video-substrate-eval-600pairs` | $0 (HF Jobs CPU inference) | 1h | vision-trainer |
| Z6 4c FiLM ego-motion + DINOv3 cooperative-receiver smoke (5ep) | $0.50-1.00 | 1h HF Jobs A10G | vision-trainer + trackio |
| Z6 4c FiLM ego-motion + DINOv3 100ep training (post-smoke-green) | $5-8 | 2-4h HF Jobs A10G | vision-trainer + trackio |
| SAM2-hiera-tiny mask-decoder-only training v1 (30ep) | $0.80 | 2h HF Jobs T4 | vision-trainer + trackio |
| HF Jobs migration of 3 substrate trainers (DP1, SCPP, PR101 lc_v2) | $5-10 (paired smokes) | 3-6h editor + GPU | jobs |
| Build `adpena/comma-substrate-composition-matrix` HF dataset + sync hook | $0 | 1-2h | tool-builder |
| Build `adpena/comma-council-deliberation-posterior` HF dataset + sync hook | $0 | 1-2h | tool-builder |
| Set up scheduled jobs (Modal harvest @hourly, lane audit @daily) | $5-7/month | 1-2h | jobs |

**Total Phase 2 budget**: ~$20 (one-time) + $5-10/month recurring.
**Total Phase 2 wall-clock**: ~3-4 sessions.

**Deliverables**:
- 2 surrogates (SegNet + PoseNet) — enable faster substrate iteration.
- 1-2 new substrate empirical anchors (Z6 4c DINOv3 / SAM2 sister).
- 3 substrate trainers migrated to HF Jobs (32% cost reduction).
- 2 new canonical HF datasets (composition matrix + council posterior).
- Scheduled jobs automate harvest + audit chores.

### Phase 3: ~$50 (deeper exploration)

**Goal**: Full substrate canvas on HF Jobs + paper page + comprehensive Gradio supplement.

| Task | Cost | Wall-clock | Skill |
|---|---|---|---|
| Migrate remaining 10 substrate trainers to HF Jobs | $20-30 (paired smokes) | 10-15h editor | jobs |
| Build Gradio leaderboard visualizer (UI 2; 300 LOC) | $0 | 3-4h editor | gradio |
| Build Gradio substrate auditor (UI 3; 400 LOC) | $0 | 4-5h editor | gradio |
| Build Gradio paper supplement live demo (UI 4; 600 LOC) | $0 | 6-8h editor | gradio |
| arXiv paper draft + publish via paper-publisher | $0 | 10-20h editor (mostly writing) | paper-publisher |
| Link arXiv paper to all 13 substrates + sister datasets | $0 | 1h | paper-publisher |
| Fine-tune DINOv3 on `comma-video-substrate-eval-600pairs` (50ep) | $3-5 | 4h HF Jobs L4 | vision-trainer |
| Full TT5L V2 + VGGT cooperative-receiver substrate training (100ep) | $15-25 | 4h HF Jobs A100 | vision-trainer |

**Total Phase 3 budget**: ~$50.
**Total Phase 3 wall-clock**: ~5-7 sessions.

**Deliverables**:
- Full substrate canvas on HF Jobs.
- 4 Gradio UIs live.
- arXiv paper published + linked to all artifacts.
- Fine-tuned DINOv3 anchor.
- TT5L V2 cooperative-receiver empirical anchor.

### Phase 4: deferred (research-only)

- Migrate Lightning + Vast.ai dispatches to HF Jobs (uniform dispatch surface).
- TPU (`v5e-1x1`) experiments for end-to-end Ballé joint codec training.
- Webhook-triggered auto-train on PR archive updates.
- Full paper supplement website (Gradio + Cloudflare R2 storage).
- Public Space flip + OSS release wave.

---

## Section 5: Sister-subagent recommendations (NEXT WAVE)

Recommend 3-5 new subagents to fire NEXT (after this design memo + existing 3 in-flight + the SLOT 5 implementation subagent land):

### Subagent A: HF-DATASET-UPLOAD-PIPELINE-BUILD-SUBAGENT
- **Scope**: Implement `tools/build_comma_video_substrate_eval_dataset.py` + `tools/build_comma_pr_archive_corpus_dataset.py` per Section 2.4 + 3.1 schemas.
- **Catalog compliance**: #229 premise verification + #209 contest-video-leakage disclaimer in dataset card + #210 codebook provenance metadata (NA, not codebooks but document anyway) + Catalog #131 fcntl-locked writes.
- **Estimated cost**: $0 (local CPU upload).
- **Deliverable**: 2 HF datasets PRIVATE uploaded; lane `lane_hf_dataset_upload_pipeline_build_20260518`.

### Subagent B: HF-JOBS-CANONICAL-DISPATCH-WRAPPER-SUBAGENT
- **Scope**: Build `tools/dispatch_hf_jobs.py` (canonical wrapper mirroring `tools/operator_authorize.py` + `tools/dispatch_modal_paired_auth_eval.py`). Wire `hf_jobs()` MCP tool calls; thread `secrets={"HF_TOKEN": "$HF_TOKEN"}`; integrate with Catalog #245 sister job_id ledger.
- **Catalog compliance**: #270 dispatch optimization protocol + #317 one-arg local-vs-modal switch extension (`--target hf-jobs`) + #244 sister env block + #166 source-parity hash equivalent (DEFERRED-PENDING-RESEARCH per Section 2.5).
- **Estimated cost**: $0.40 (1 HF Jobs T4 paired smoke for validation).
- **Deliverable**: Canonical HF Jobs dispatch wrapper; lane `lane_hf_jobs_canonical_dispatch_wrapper_20260518`.

### Subagent C: SEGNET-SURROGATE-DISTILLATION-SUBAGENT
- **Scope**: Implement Stage 1 SegNet surrogate distillation per Section 2.1. Custom training script (forked from canonical `image_classification_training.py` per Catalog #290). Trackio integration. Per-step KL loss + per-epoch eval_score_drift check. Push to `adpena/comma-segnet-surrogate-mobilevit-s`.
- **Catalog compliance**: #523 (THE long-deferred catalog this satisfies) + #8 eval_roundtrip + #164 canonical scorer-preprocess + #325 substrate symposium NOT REQUIRED (this is infrastructure, not substrate).
- **Estimated cost**: $1.60 (2 HF Jobs T4 dispatches).
- **Deliverable**: Working SegNet surrogate with documented L2 drift; lane `lane_segnet_surrogate_distillation_20260518`.

### Subagent D: GRADIO-OPERATOR-DASHBOARD-SUBAGENT
- **Scope**: Build Gradio UI 1 (operator dashboard) per Section 3.9. ~500 LOC. Pull from canonical state ledgers via fcntl-locked reads per Catalog #131.
- **Catalog compliance**: #131 fcntl-locked state reads + #208 docs no local absolute paths in Gradio config + Public Disclosure Hygiene (PRIVATE Space).
- **Estimated cost**: $0.
- **Deliverable**: Gradio Space live at `huggingface.co/spaces/adpena/comma-operator-dashboard` (PRIVATE); lane `lane_gradio_operator_dashboard_build_20260518`.

### Subagent E: TRACKIO-INTEGRATION-CROSS-SUBSTRATE-AUDIT-SUBAGENT
- **Scope**: Audit all 13 substrate trainers + 9 council artifacts; add `trackio.init/log/finish` + alert hooks per Section 3.8 canonical pattern. NON-INVASIVE — only adds 3-5 lines per trainer; does not modify substrate logic.
- **Catalog compliance**: #290 canonical-vs-unique (ADOPT_CANONICAL_BECAUSE_SERVES) + #208 docs no local paths in Trackio config.
- **Estimated cost**: $0 (no GPU; editor only).
- **Deliverable**: 13 trainers + 9 council artifacts emit Trackio metrics; lane `lane_trackio_cross_substrate_integration_20260518`.

### Subagent F: PAPER-PUBLISHER-CATALOG-431-INTEGRATION-SUBAGENT
- **Scope**: Use `paper_manager.py` to (1) generate paper draft from `modern` template, (2) populate from existing `.omx/research/` + memory files, (3) prepare for arXiv submission. DOES NOT actually submit to arXiv (operator-routable decision).
- **Catalog compliance**: #431 PHASE 4 INTEGRATION (this satisfies it) + #208 docs no local paths.
- **Estimated cost**: $0.
- **Deliverable**: arXiv-ready paper draft at `docs/paper/arxiv_v1_<utc>.md`; lane `lane_paper_publisher_catalog_431_integration_20260518`.

---

## Section 6: Operator-routable decisions

| Decision | Options | Recommendation | Required by |
|---|---|---|---|
| HF Pro/Team/Enterprise plan tier verification | Confirm via web UI at https://huggingface.co/settings/billing | Operator only | Phase 1 (HF Jobs require paid plan) |
| Phase 1 budget approval ($3) | Approve / Defer / Modify | APPROVE (low-risk infrastructure) | Phase 1 start |
| Phase 2 budget approval ($20 + $5-10/month recurring) | Approve / Defer / Modify | APPROVE post-Phase-1 (predicted ΔS [-0.005, -0.020]) | Phase 2 start |
| Phase 3 budget approval ($50) | Approve / Defer / Modify | DEFER until Phase 2 empirical | Phase 3 start |
| DINOv3 integration approach | Option A frozen / Option B fine-tune | Option A first (Phase 2); Option B after empirical anchor (Phase 3) | Phase 2 design |
| Trackio Space creation | PRIVATE only / PUBLIC / DEFER | PRIVATE per CLAUDE.md "Public Disclosure Hygiene"; flip to PUBLIC at OSS release | Phase 1 start |
| Gradio Space creation | PRIVATE only / PUBLIC subset / PUBLIC full / DEFER | PRIVATE for UIs 1-3; PUBLIC for UI 4 paper supplement at release | Phase 1 + 3 |
| `adpena/comma-video-substrate-eval-600pairs` visibility | PRIVATE / PUBLIC | PRIVATE initially; flip at OSS release wave | Phase 1 start |
| `adpena/comma-pr-archive-corpus-54` visibility | PRIVATE / PUBLIC / EXCLUDE FROM UPLOAD | PRIVATE (contains other-author content; license check needed before public flip) | Phase 1 |
| Catalog #166 source-parity hash equivalent for HF Jobs | Build NEW canonical helper / DEFER-PENDING-RESEARCH / OK to dispatch without | DEFER-PENDING-RESEARCH; risk is moderate (stale working-tree dispatch); document in Subagent B scope | Phase 2 (Subagent B) |
| arXiv paper submission timing | Phase 3 / DEFER / Operator only | Phase 3 (Subagent F prepares draft; operator submits) | Phase 3 |
| HF Jobs sister to Modal call_id ledger (Catalog #245) | Build separate `hf_jobs_id_ledger` / Extend existing | Extend existing `modal_call_id_ledger` with `platform` discriminator field | Phase 2 (Subagent B) |
| Webhook for "auto-train on PR archive update" | Enable / Defer / Skip | DEFER to Phase 4 (operational sophistication, not score-lowering) | Phase 4 |

---

## Section 7: Catalog compliance + risk register

### 7.1 Catalog #229 PV-1 through PV-N premise verification

Per Catalog #229 (premise verification before edit). PVs verified for each design choice:

**PV-1**: HF Jobs T4 hardware availability + pricing ($0.40/hr) — VERIFIED via `huggingface.co/docs/hub/en/spaces-config-reference` (cited in `hf-cli` SKILL.md) + cross-checked against vision-trainer SKILL.md hardware table.

**PV-2**: `timm/mobilevit_s.cvnets_in1k` parameter count (5.6M) — VERIFIED via vision-trainer SKILL.md recommended-models table.

**PV-3**: SAM2-hiera-tiny architecture (frozen encoder + frozen prompt encoder + trainable mask decoder) — VERIFIED via vision-trainer SKILL.md ("Only the mask decoder is trained by default").

**PV-4**: Catalog #523 long-deferred status — TO BE VERIFIED by Subagent C (read `tools/lane_maturity.py audit | grep 523`).

**PV-5**: `upstream/videos/0.mkv` size 35.8M + 600-pair-non-overlapping structure — VERIFIED (file exists; CLAUDE.md "Lane separation" cites `seq_len=2` non-overlapping batching = 600 pairs from 1200 frames).

**PV-6**: `tac.differentiable_eval_roundtrip.patch_upstream_yuv6_globally` exists and is callable — TO BE VERIFIED by Subagent C (grep `src/tac/differentiable_eval_roundtrip.py` for the function).

**PV-7**: `tac.optimization.substrate_composition_matrix` exports queryable JSON — TO BE VERIFIED by Subagent E.

**PV-8**: `.omx/state/council_deliberation_posterior.jsonl` schema matches Catalog #300 v2 frontmatter — VERIFIED via CLAUDE.md "Council hierarchy: 4-tier protocol" section.

### 7.2 Catalog #290 canonical-vs-unique decision per layer

Documented in introductory section. Summary: ADOPT_CANONICAL for HF Hub auth / datasets API / Jobs API / Trackio / hf-cli / paper-publisher / tool-builder / evaluation / dataset-viewer. FORK_BECAUSE_SUPPRESSES for SegNet surrogate distillation script (forked from `image_classification_training.py`). FORK_BECAUSE_PRINCIPLED_MISMATCH for Trackio public Space + PoseNet surrogate (SAM2 mask-decoder semantically incompatible with 6-DOF regression). NOT_USEFUL for transformers.js.

### 7.3 Catalog #294 9-dim checklist evidence

Documented in introductory section. All 9 dimensions addressed.

### 7.4 Catalog #303 cargo-cult audit per assumption

Documented in introductory section. 8 assumptions classified HARD-EARNED vs CARGO-CULTED-PENDING-VERIFICATION vs CARGO-CULTED-INHERITED-FROM-LITERATURE.

### 7.5 Catalog #305 observability surface

Documented in introductory section. All 6 facets satisfied.

### 7.6 Catalog #316 frontier signal preservation

Phase 1 explicitly adds `model-index` entries to 3-5 substrates per Section 3.4. This SATISFIES Catalog #316 at the published-model-card surface (sister to `reports/latest.md` FRONTIER section).

### 7.7 Catalog #325 per-substrate symposium

Z6 4c FiLM ego-motion + DINOv3 cooperative-receiver (Section 2.2) is a NEW substrate variant; REQUIRES per-substrate symposium before Phase 2 paid dispatch per Catalog #325. SAM2-hiera-tiny as PoseNet sister (Section 2.3) is a composition with Z6 4c; REQUIRES sister symposium. SegNet/PoseNet surrogate distillation (Section 2.1) is INFRASTRUCTURE, not substrate; does NOT require symposium.

### 7.8 Risk register

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| HF Pro/Team/Enterprise plan not active → HF Jobs dispatches fail | HIGH | LOW (operator likely already has) | Verify before Phase 1 start |
| HF Jobs T4 produces different scores than Modal T4 | MEDIUM | MEDIUM | Paired smoke validation (Subagent B); CARGO-CULTED-PENDING-VERIFICATION classification |
| Trackio public Space leaks operator-internal metrics | HIGH | HIGH if PUBLIC default | PRIVATE Space initially (cargo-cult audit verified) |
| `adpena/comma-pr-archive-corpus-54` contains other-author content without license | HIGH | HIGH | PRIVATE Hub repo; license audit before public flip; legal review |
| SegNet surrogate accuracy too low → score gradient fidelity degrades | MEDIUM | MEDIUM | Stage-1 smoke (~$0.80) tests accuracy before Stage-2 commitment |
| DINOv3 cooperative-receiver claim falsified by empirical anchor | MEDIUM | MEDIUM | Catalog #324 post-training Tier-C validation gates promotion |
| HF Jobs source-parity hash equivalent for Catalog #166 missing | MEDIUM | HIGH (acknowledged) | DEFER-PENDING-RESEARCH in Subagent B scope; document the gap explicitly |
| Gradio public Space exposes private substrate details before release | HIGH | MEDIUM | PRIVATE Spaces for UI 1-3; explicit operator review before UI 4 public flip |
| Phase 1 budget creep (>$3) | LOW | LOW | Hard cap via Catalog #199 paired-env discipline + operator budget review at $5 |
| arXiv paper publication triggers legal/IP review delays | MEDIUM | MEDIUM | Subagent F prepares draft only; operator submits → external review |
| Sister subagent collision per Catalog #314 absorption pattern | MEDIUM | LOW (this subagent is research-only; no commits) | This subagent emits ONLY this design memo; main-Claude handles commits |

---

## Section 8: Cross-disciplinary convergent-truth extension

Per `feedback_deep_research_wave_landed_20260518.md` and the operator's earlier deep-research directive, extend the convergent-truth tuple list with insights from this HF-skills survey. NEW convergent-truth tuples:

### 8.1 HuggingFace canonical patterns ↔ NVIDIA TAO ↔ Cosmos research ↔ Hinton+Caruana distillation (Catalog #152)

**The truth**: Knowledge distillation as a meta-pattern for model compression + transfer learning is canonical across all major ML frameworks.

**Lenses**:
- **HuggingFace**: `kl_on_logits` with temperature T (vision-trainer + Quantizr published pattern).
- **NVIDIA TAO**: TAO Toolkit's pruning + quantization-aware training (proprietary; same KL-distillation backbone).
- **Cosmos (NVIDIA Research)**: World-model distillation from large multi-modal models to compact deployable models.
- **Hinton 2014**: Original "Distilling the Knowledge in a Neural Network" paper (NeurIPS workshop).
- **Caruana 2006**: "Model Compression" — predecessor to Hinton's paper; same KL-loss formulation.

**Engineering convergence**: All four use `student_loss = α * task_loss + (1-α) * T^2 * kl_div(student_logits/T, teacher_logits/T)`. The mathematical formulation is invariant; the lens (academic / industrial / proprietary / open-source) varies.

**Application to our problem**: SegNet/PoseNet surrogate distillation (Section 2.1) is one instance of this convergent-truth tuple.

### 8.2 Trackio ↔ Weights+Biases ↔ Comet ML ↔ Aim (experiment tracking convergence)

**The truth**: Real-time experiment metric tracking + alert workflow is a CONVERGED API design.

**Lenses**:
- **Trackio** (HuggingFace): `trackio.init/log/finish/alert`.
- **Weights+Biases**: `wandb.init/log/finish/alert`.
- **Comet ML**: `comet_ml.Experiment().log_metric(...)`.
- **Aim** (open-source): `aim.Run().track(...)`.

**Engineering convergence**: All four expose `init → log → finish` lifecycle + optional alert/notification hooks + run grouping by project/run_name. The semantic IS identical; the visualization Space differs.

**Application to our problem**: Trackio chosen because (a) HF-native (no separate auth), (b) free tier sufficient, (c) Hub Space sync inherits Hub's collaboration + private/public mode.

### 8.3 Gradio ↔ Streamlit ↔ Panel ↔ Dash (Python web UI convergence)

**The truth**: Python web UI for ML demos is a CONVERGED ecosystem with 4 mature competitors.

**Lenses**:
- **Gradio** (HuggingFace): Block + event listeners; Space hosting native.
- **Streamlit**: Imperative top-to-bottom script execution; Snowflake hosting native.
- **Panel** (Anaconda): Holoviz ecosystem; Bokeh + Param-based.
- **Dash** (Plotly): React-under-the-hood; production web framework.

**Engineering convergence**: All four expose declarative components (Input, Output, Button) + event wiring + state management. The semantic IS identical; idiom differs.

**Application to our problem**: Gradio chosen because (a) HF-native (Space hosting + collaboration), (b) most-mature ML-demo idioms, (c) `gr.Image` / `gr.Video` / `gr.File` components native to our problem domain.

### 8.4 HF Datasets DuckDB SQL ↔ Polars ↔ DataFusion (lazy-query convergence)

**The truth**: Columnar lazy-query engines for parquet are a CONVERGED ecosystem.

**Lenses**:
- **HF Datasets DuckDB via `hf://`**: SQL syntax on parquet; in-process.
- **Polars** (open-source): DataFrame API on Apache Arrow; Rust-backed.
- **DataFusion** (Apache): SQL engine on Apache Arrow; Rust-backed.
- **DuckDB native**: In-process SQL on parquet; battle-tested.

**Engineering convergence**: All four push predicates down to parquet column metadata + project only requested columns + execute lazily. The 100× speedup vs pandas comes from this common architectural decision.

**Application to our problem**: HF Datasets DuckDB chosen because (a) HF-native `hf://` protocol enables zero-download remote queries, (b) SQL is the canonical query language across data engineering, (c) `hf datasets sql` CLI provides operator-facing surface.

### 8.5 HF Jobs ↔ Modal ↔ Vast.ai ↔ Lightning ↔ Lambda Labs (cloud GPU dispatch convergence)

**The truth**: Cloud GPU dispatch APIs are CONVERGING toward a "container + script + secret + persistence" universal pattern.

**Lenses**:
- **HF Jobs**: `hf_jobs("uv", {...})` MCP tool + HF Hub persistence.
- **Modal**: `@app.function(image=..., volumes=...)` + Modal Volume persistence.
- **Vast.ai**: SSH-based instance management + custom persistence.
- **Lightning**: `Job.run()` + Lightning Cloud subscription.
- **Lambda Labs**: SSH-based GPU rental.

**Engineering convergence**: All five accept (a) a script or container image, (b) secrets injected at runtime, (c) some persistence layer (Hub / Volume / S3 / SSH). The semantic IS identical; pricing + UX varies.

**Application to our problem**: HF Jobs chosen for substrate trainers' migration because (a) cheapest T4 + A100 in pricing comparison, (b) Hub-native persistence (avoid sister Modal Volume / S3 dual-track), (c) `hf_jobs()` MCP tool integrates with existing subagent workflow.

### 8.6 NEW CONVERGENCE: HF Paper Publisher ↔ arXiv ↔ Semantic Scholar ↔ Papers With Code (academic citation convergence)

**The truth**: Academic-paper-to-artifact linking is a CONVERGED ecosystem.

**Lenses**:
- **HF Paper Publisher**: `paper_manager.py link --arxiv-id <id>`.
- **arXiv**: Native paper metadata + bibtex.
- **Semantic Scholar**: Paper graph + citation links.
- **Papers With Code**: Model-index spec (which HF directly adopts).

**Engineering convergence**: All four expose `paper_id → linked_artifacts (models, datasets, code)` graph. HF Paper Publisher operationalizes this graph at the Hub layer.

**Application to our problem**: Catalog #431 arXiv paper publication uses HF Paper Publisher; auto-tags arXiv ID across all 13 published substrate models.

### 8.7 NEW CONVERGENCE: Trackio Alerts ↔ Datadog ↔ PagerDuty ↔ Prometheus AlertManager (alerting convergence)

**The truth**: Structured-alert systems for distributed workloads converge on (severity, title, text, source, timestamp, webhook) schema.

**Lenses**:
- **Trackio**: `AlertLevel.INFO/WARN/ERROR`.
- **Datadog**: Monitor severity levels.
- **PagerDuty**: Incident severity levels.
- **Prometheus AlertManager**: Alert routing + severity-based grouping.

**Engineering convergence**: All four expose (severity, title, text) → webhook routing (Slack / Discord / PagerDuty / email).

**Application to our problem**: Trackio alerts (Section 3.8) wire to operator Slack via webhook.

### 8.8 NEW CONVERGENCE: HF Datasets + HF MCP + dataset-viewer ↔ Kaggle Datasets + Kaggle API ↔ TensorFlow Datasets ↔ PyTorch Hub (dataset discovery + download convergence)

**The truth**: Versioned, schema-typed, lazily-downloadable, queryable datasets are a CONVERGED ML primitive.

**Lenses**:
- **HF Datasets**: `datasets.load_dataset("user/repo")` + Hub versioning.
- **Kaggle**: `kaggle datasets download user/repo`.
- **TensorFlow Datasets**: `tfds.load("dataset_name")`.
- **PyTorch Hub**: `torch.hub.load("user/repo", "model")`.

**Engineering convergence**: All four expose `repo_id → versioned_dataset` with lazy download + cache + schema.

**Application to our problem**: `adpena/comma-video-substrate-eval-600pairs` is the canonical HF dataset; substrate trainers consume via `datasets.load_dataset(...)`.

### 8.9 NEW CONVERGENCE: HF Jobs scheduled jobs ↔ Cron ↔ Airflow ↔ Prefect ↔ Dagster (workflow orchestration convergence)

**The truth**: Time-based workflow orchestration is CONVERGED on CRON syntax + DAG-based dependency expression.

**Lenses**:
- **HF Jobs scheduled**: `@hourly` / CRON expression.
- **Cron** (Unix): `0 9 * * 1`.
- **Airflow**: `schedule_interval="0 9 * * 1"` + DAG.
- **Prefect**: `IntervalSchedule(...)` + Flow.
- **Dagster**: `@schedule(cron_schedule="...")` + Asset.

**Engineering convergence**: CRON syntax IS the universal scheduling primitive across all five.

**Application to our problem**: 4 scheduled jobs (Modal harvest, lane audit, leaderboard fetch, Trackio snapshot) per Section 3.5.

---

## Conclusion + report-back summary

This memo surveys all 12 huggingface-skills, classifies each per applicability + EV, designs implementation for 5 main insights + 7 new use cases beyond, sequences a 4-phase plan, enumerates 6 sister-subagent recommendations, lists 13 operator-routable decisions, and extends 9 cross-disciplinary convergent-truth tuples.

### Top-5 unrealized use cases beyond the original 5 main insights

1. **`adpena/comma-pr-archive-corpus-54` SQL queries** (Section 3.2) — surface NEW Pareto outliers + technique-tag co-occurrence patterns the apparatus hasn't manually enumerated. Q5 in particular (technique co-occurrence) could reveal a previously-undocumented composition opportunity. Cost $0. EV VERY HIGH.

2. **Trackio cross-substrate integration with alert webhooks** (Section 3.8) — eliminates "which Modal job was that again?" friction, enables autonomous agent monitoring per `trackio list alerts --json`, real-time MPS-vs-CUDA drift detection. Cost $0. EV VERY HIGH.

3. **HF Jobs scheduled jobs for automation** (Section 3.5) — Modal call_id harvester @hourly + lane registry audit @daily + leaderboard fetch @hourly. Eliminates manual operator chore-load. Cost ~$5/month. EV VERY HIGH.

4. **Gradio operator dashboard (UI 1)** (Section 3.9) — single-page view of all in-flight dispatches + Trackio metrics + cost-band posterior + lane registry. Operator-facing leverage compounds over time. Cost $0. EV VERY HIGH.

5. **`adpena/comma-substrate-composition-matrix` + `adpena/comma-council-deliberation-posterior` HF datasets** (Section 3.7) — canonicalize the substrate composition matrix + council posterior as queryable HF datasets; sister-substrate cross-references become SQL-queryable; cite-able by future arXiv paper. Cost $0. EV HIGH.

### Recommended Phase 1 implementation sequence

1. **Operator decision**: Confirm Pro/Team/Enterprise plan tier (HF Jobs requires paid plan).
2. **Subagent A** (HF dataset upload pipeline): build `tools/build_comma_video_substrate_eval_dataset.py` + `tools/build_comma_pr_archive_corpus_dataset.py`. Upload both as PRIVATE. ~2 sessions editor work, $0 GPU.
3. **Subagent E** (Trackio cross-substrate integration): non-invasive addition of `trackio.init/log/finish` to 13 substrate trainers + 9 council artifacts. ~1 session editor, $0 GPU.
4. **Subagent D** (Gradio operator dashboard): build UI 1 PRIVATE Space. ~4-6h editor, $0 GPU.
5. **Subagent C** (SegNet surrogate distillation): distill `mobilenetv3_small` smoke v1; if accuracy OK, move to `mobilevit_s` v1. ~$1.60 HF Jobs T4.
6. **Operator decision**: Phase 2 budget approval ($20 + $5-10/month recurring).

### Operator-routable decisions list (consolidated)

1. **Pro/Team/Enterprise plan tier verification** (Phase 1 prereq).
2. **Phase 1 budget approval** ($3) — recommend APPROVE.
3. **Phase 2 budget approval** ($20 + $5-10/month) — DEFER until Phase 1 deliverables land.
4. **DINOv3 integration approach**: Option A frozen → Phase 2; Option B fine-tune → Phase 3.
5. **Trackio Space**: PRIVATE initially.
6. **Gradio Spaces**: PRIVATE for UIs 1-3; PUBLIC for UI 4 paper supplement at release.
7. **`comma-video-substrate-eval-600pairs` visibility**: PRIVATE initially.
8. **`comma-pr-archive-corpus-54` visibility**: PRIVATE; license check before public flip.
9. **Catalog #166 source-parity hash equivalent for HF Jobs**: DEFER-PENDING-RESEARCH; document gap.
10. **arXiv paper submission timing**: Phase 3 prep; operator submits.
11. **HF Jobs sister to Modal call_id ledger**: extend existing ledger with `platform` discriminator.
12. **Webhook for auto-train on PR archive update**: DEFER to Phase 4.
13. **transformers.js**: SKIP (NOT USEFUL).

### Honest classification: HARD-EARNED-USEFUL vs CARGO-CULTED-FOR-OUR-PROBLEM

| Skill | Classification |
|---|---|
| hf-cli | HARD-EARNED-USEFUL |
| hugging-face-dataset-viewer | HARD-EARNED-USEFUL |
| hugging-face-datasets | HARD-EARNED-USEFUL |
| hugging-face-evaluation | HARD-EARNED-USEFUL |
| hugging-face-jobs | HARD-EARNED-USEFUL |
| hugging-face-model-trainer | CARGO-CULTED-FOR-OUR-PROBLEM (we don't train LLMs; reference only) |
| hugging-face-paper-publisher | HARD-EARNED-USEFUL (Catalog #431 dependency) |
| hugging-face-tool-builder | HARD-EARNED-USEFUL |
| hugging-face-trackio | HARD-EARNED-USEFUL |
| hugging-face-vision-trainer | HARD-EARNED-USEFUL (THE source of 5 main insights) |
| huggingface-gradio | HARD-EARNED-USEFUL |
| transformers.js | NOT-USEFUL-SKIP |

10-of-12 HARD-EARNED-USEFUL; 1-of-12 CARGO-CULTED reference-only; 1-of-12 NOT-USEFUL. Strong overall applicability ratio.

---

**END OF MEMO**

Lane: `lane_hf_skills_comprehensive_design_implementation_plan_20260518` L1 (impl_complete + memory_entry + design_only — no commits, no GPU dispatches, no MEMORY.md mutations per subagent scope).

Sister-subagent ownership map: this subagent OWNS `.omx/research/huggingface_skills_comprehensive_design_implementation_plan_20260518.md`. NO other files touched. Disjoint from in-flight subagents per Catalog #230.

Premise verification per Catalog #229: 8 PVs declared in Section 7.1; 4 verified locally, 4 deferred to subagent C/E.

Checkpoint discipline per Catalog #206: 2 checkpoints emitted via `tools/subagent_checkpoint.py` (step 0 start, step 1 mid-survey).

Canonical-vs-unique decision per layer per Catalog #290: 14-row table in introductory section.

9-dim checklist evidence per Catalog #294: 9-bullet block in introductory section.

Cargo-cult audit per assumption per Catalog #303: 8-row table in introductory section.

Observability surface per Catalog #305: 6-bullet block in introductory section.

Predicted-band Dykstra feasibility per Catalog #296: N/A (this is a design memo, not a substrate predicted-ΔS claim with specific band).

6-hook wire-in declaration per Catalog #125:
- Hook #1 sensitivity-map: N/A (research-only design memo, no signal contribution).
- Hook #2 Pareto constraint: N/A (no Pareto-relevant signal).
- Hook #3 bit-allocator: N/A.
- Hook #4 cathedral autopilot dispatch: ACTIVE (Section 4 Phase 2 dispatches will be candidates for autopilot ranking).
- Hook #5 continual-learning posterior: ACTIVE (Subagent C/E land empirical anchors that emit to canonical posterior).
- Hook #6 probe-disambiguator: ACTIVE (Section 2.2 + 2.3 specify probe-disambiguators for DINOv3 cooperative-receiver + SAM2 PoseNet sister claims).
