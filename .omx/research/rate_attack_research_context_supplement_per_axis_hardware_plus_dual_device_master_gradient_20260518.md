# Rate-attack research context supplement: per-axis hardware exploits + dual-device master-gradient + comprehensive available-signal inventory
# Date: 2026-05-18
# Operator clarification verbatim: "by cpu vs gpu exploits, i mean there may be different hardware exploits that are optimal and available for both on rate attack frontier or only for one or the other, and that should be taken into account, and all available master gradient information (should it be run on cpu vs gpu to help to understand more deeply?) and other information and signal available should be made available as context"
# Sister to: in-flight PRIMARY rate-attack subagent a703d2b74784d4f00 + in-flight ADVERSARIAL paradigm-challenger subagent a74313c6276748ad8
# Per CLAUDE.md "Apples-to-apples evidence discipline" + "MPS auth eval is NOISE" + Catalog #316 frontier scan

## PART 1: PER-AXIS HARDWARE EXPLOIT LANDSCAPE (operator clarification of G-category)

The G-category in my prior brainstorm framed "CPU vs GPU exploits" as "CPU-axis-specific optimization (G1) since leaderboard is CPU". **OPERATOR CLARIFICATION**: that framing was too narrow. The actual landscape is:

### Structural framing

Some rate-attack hardware exploits are:
- **CPU-ONLY** (only work on contest-CPU axis): AVX-512 SIMD bit-packing, MKL kernel-specific numerics, x86 80-bit FPU extended precision, CPU cache-line alignment
- **GPU-ONLY** (only work on contest-CUDA axis): NVDEC hardware video decode, NVJPEG hardware JPEG decode, tensor-core native formats (fp4/fp8), CUDA sparse tensor formats, NVIDIA DALI hardware pipeline
- **BOTH-WITH-DIFFERENT-OPTIMAL-CONFIG** (different optimal config per axis): VVC/H.266 codec (different decoder paths CPU vs GPU), AV1 mode/profile selection, JPEG quantization tables (different decode paths), bit-packing alignment (64-byte CPU cache vs 128-byte GPU transactions)
- **AXIS-INVARIANT** (same exploit, same config both axes): ZIP STORED method, ZIP header minimization, entropy-coded payloads via standard libs

### Per-axis optimality matrix (must be in PRIMARY subagent's deliverable)

```
                    contest-CUDA (T4)    contest-CPU (GHA Linux x86_64)
                    ─────────────────    ─────────────────────────────
NVDEC AV1           OPTIMAL              N/A (no hardware decode)
NVJPEG              OPTIMAL              N/A
Tensor cores fp8    OPTIMAL              N/A
CUDA sparse 2:4     OPTIMAL              N/A
DALI pipeline       OPTIMAL              N/A
AVX-512 SIMD        N/A                  OPTIMAL
MKL kernels         N/A                  OPTIMAL
80-bit FPU          N/A                  OPTIMAL (some configs)
CPU cache-line      LOW-PRIORITY         OPTIMAL
VVC software decode AVAILABLE            OPTIMAL (CPU-native libvvenc)
AV1 mode selection  per-profile          per-profile (different optimal)
JPEG q-tables       NVJPEG-decode        libjpeg-turbo decode
Bit-packing         128-byte tx          64-byte cache-line
ZIP STORED          AXIS-INVARIANT       AXIS-INVARIANT
ZIP minimal hdr     AXIS-INVARIANT       AXIS-INVARIANT
Entropy coding      AXIS-INVARIANT       AXIS-INVARIANT
```

### Strategic implication per the per-axis matrix

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable + PR102 empirical +0.033 gap:
- **The contest LEADERBOARD ranks by contest-CPU**, so CPU-ONLY + BOTH-CPU-OPTIMAL exploits have HIGHER LEADERBOARD EV than GPU-ONLY exploits
- **BUT** the contest also REQUIRES contest-CUDA scoring as transparency. If a CPU-optimized exploit catastrophically DESTROYS contest-CUDA score (>1.0 gap), the submission is invalid per public-disclosure-hygiene + bot-comment-expectations
- **Optimal strategy**: prioritize AXIS-INVARIANT exploits (ZIP overhead / entropy coding) + BOTH-WITH-DIFFERENT-OPTIMAL-CONFIG (VVC / JPEG q-tables / AV1 mode selection) + CPU-ONLY exploits that don't catastrophically break CUDA
- **Cardinality**: the cross-axis matrix above implies ~16+ distinct per-axis-optimal configurations; the rate-attack subagent should produce a per-vector per-axis ranking

## PART 2: DUAL-DEVICE MASTER-GRADIENT EXTRACTION RESEARCH QUESTION (operator's "should it be run on cpu vs gpu")

Codex's just-landed ITEM_3 master-gradient extractor (Phase A; commits per `codex_persistent_session_state.jsonl` 2026-05-18T18:19) computed per-pair fp64 master-gradient on PR101_lc_v2 anchor `f174192aeadf...` on a SINGLE device (per `codex_master_gradient_multi_archive_extractor_phase_a_20260518_codex.md` — verify device per the artifact metadata).

**The operator's research question**: should master-gradient be computed on BOTH CPU and GPU to surface the per-axis sensitivity divergence?

### Hypothesis space

H1. **Master-gradient is axis-invariant** → single-device computation sufficient; dual-device adds no information.
H2. **Master-gradient has small axis-divergence** (e.g., < 1% RMS per-byte difference) → dual-device shows structural correlation; minor exploit opportunities.
H3. **Master-gradient has SIGNIFICANT axis-divergence** (e.g., ≥ 5% RMS per-byte difference; some bytes have OPPOSITE-sign gradients across axes) → BIG exploit opportunity; bytes that have positive CUDA gradient + negative CPU gradient (or vice versa) are STRUCTURAL ASYMMETRY MARKERS.

### Empirical proof-path

Per Catalog #229 premise-verification-before-edit: BEFORE the rate-attack subagent commits to per-axis hardware exploits as a TOP-5 strategy, propose extending Codex's `tools/extract_master_gradient.py` to compute master-gradient on BOTH devices (CPU + CUDA) for at least 1 anchor (PR101_lc_v2). Cost: $0-$2 (CPU compute likely free; CUDA compute ~$2 on a Modal T4 smoke if needed).

The dual-device extension would produce:
- `.omx/state/master_gradient_anchors_cpu.jsonl` (per-pair fp64 on x86 CPU)
- `.omx/state/master_gradient_anchors_cuda.jsonl` (per-pair fp64 on T4)
- A canonical helper `tac.master_gradient_consumers.compute_axis_divergence(archive_sha256)` that computes per-byte CPU-vs-CUDA gradient delta + identifies BYTES WITH AXIS-OPPOSITE-SIGN-GRADIENTS

### Routing recommendation

This is a SEPARATE routing directive — write it post-rate-attack-subagent-completion since the subagent's findings inform whether dual-device extension is worth the $2 dispatch. The primary subagent should explicitly answer H1/H2/H3 hypothesis in its research findings; the adversarial subagent challenges the answer; then routing directive lands per consensus.

## PART 3: COMPREHENSIVE AVAILABLE-SIGNAL INVENTORY (operator's "all other information and signal available should be made available as context")

The PRIMARY + ADVERSARIAL rate-attack subagents have access to ALL of the following — list them explicitly to ensure no signal is missed:

### State ledgers (per Catalog #245 4-layer pattern + Catalog #131 fcntl-locked)
- `.omx/state/canonical_task_status.jsonl` (PRIMARY work queue; 12 completed / 9 pending / 2 in-progress per execution-monitoring synthesis)
- `.omx/state/codex_persistent_session_state.jsonl` (Codex autonomous execution log)
- `.omx/state/council_deliberation_posterior.jsonl` (all council anchors)
- `.omx/state/master_gradient_anchors.jsonl` (per-pair fp64 anchors; PR101_lc_v2 `f174192aeadf...`)
- `.omx/state/probe_outcomes.jsonl` (Catalog #313 probe verdicts)
- `.omx/state/modal_call_id_ledger.jsonl` (Catalog #245 Modal dispatch history; today's 14+ dispatches)
- `.omx/state/cost_band_posterior.jsonl` (Catalog #175 cost-band priors)
- `.omx/state/lane_registry.json` (all lanes including today's L1+ landings)
- `.omx/state/substrate_composition_matrix.json` (Catalog #322 composition_alpha cells)
- `.omx/state/wyner_ziv_deliverability/` (Catalog #319 deliverability proof artifacts)

### Canonical helpers (existing in `src/tac/`)
- `src/tac/master_gradient.py` (Codex's just-landed extractor module)
- `src/tac/master_gradient_consumers.py` (consumers of per-pair gradient)
- `src/tac/codec/wyner_ziv_layer.py` (Wyner-Ziv pipeline-stage primitive)
- `src/tac/wyner_ziv_deliverability/` (DeliverabilityTier 1-4 classification per Catalog #319)
- `src/tac/procedural_codebook_generator/` (Codex landed 7c13abda3; canonical procedural-gen primitive)
- `src/tac/null_space_exploiter/` (Codex landed 7c13abda3; HIGHEST-EV single helper per all design memos)
- `src/tac/optimization/substrate_composition_matrix.py` (Catalog #322 composition_alpha canonical)
- `src/tac/cost_band_calibration.py` (Catalog #175 cost-band posterior)
- `src/tac/canonical_task_status.py` (canonical work queue)
- `src/tac/council_continual_learning.py` (canonical posterior writer; Catalog #300 hook #5)
- `src/tac/probe_outcomes_ledger.py` (Catalog #313 canonical helper)
- `src/tac/codex_to_claude_inbox.py` (PLANNED per routing directive 745fc2e19; not yet built by Codex)
- `src/tac/sensitivity_map.py` (Catalog #275 sensitivity-map module + axis-level reweighting API per #586)
- `src/tac/preflight.py` (276 STRICT preflight gates per CLAUDE.md catalog)

### Scorer source code (per CLAUDE.md "Exact scorer architectures")
- `upstream/modules.py` — PoseNet Hydra: vision(2048) → summary(512) → ResBlock → 12-dim pose → first 6 used. SegNet: smp.Unet('tu-efficientnet_b2', classes=5). Critical for F-category Hydra exploits.
- `upstream/evaluate.py` — the canonical scorer (CPU + CUDA paths)
- `upstream/distortion.py` — distortion computation; SegNet argmax + PoseNet MSE
- `upstream/__init__.py` — module-level constants

### Submission archives (concrete byte anchors)
- `submissions/exact_current/` (pinned upstream snapshot; DO NOT EDIT per CLAUDE.md mutation frontier; READ for inflate.py reference)
- `submissions/a1/` (A1 substrate; sub-0.193 CPU anchor)
- `submissions/pr101_lc_v2_clone/` (PR101 GOLD consumer; canonical empirical anchor)
- `submissions/pr106_format0d/` (PR106 0.20533 CUDA frontier per Catalog #316)
- `submissions/dp1*/` (DP1 substrate archives if landed)
- `experiments/results/*_modal/` (Modal dispatch artifacts; harvested archives + metadata)

### Contest video (decoder-side info source per B1 vector)
- `upstream/videos/0.mkv` (37.5 MB; THE only video; YUV-native MKV container; AV1-encoded)
- This file IS the contest-video-as-codebook substrate for B1 vector

### Today's design memos (~14 + ongoing subagents)
- `.omx/research/n_set_venn_classification_design_memo_pair_region_class_frame_axis_20260518.md` (930 lines; 3-set Venn classifier)
- `.omx/research/phase_1_fisher_precondition_canonical_helper_design_memo_20260518.md` (1514 lines)
- `.omx/research/riemannian_newton_substrate_engineering_design_memo_20260518.md`
- `.omx/research/tropical_d_seg_solver_design_memo_20260518.md` (1463 lines)
- `.omx/research/cross_stack_synthesis_9_design_landings_unified_framework_20260518.md` (1449 lines; 9×9 matrix)
- `.omx/research/grand_council_t3_pose_axis_non_hnerv_paths_to_frontier_breaking_symposium_20260518.md` (892 lines)
- `.omx/research/dp1_pr101_composition_design_memo_20260518.md` (1466 lines; Path A canonical; Path B rate-infeasible)
- `.omx/research/multi_loop_codex_goal_design_memo_20260518.md` (1308 lines; 5 loops)
- `.omx/research/design_stack_full_hypergraph_model_design_memo_20260518.md` (2012 lines; 10 typed node categories incl. deterministic_byte_derivation META-category 10)
- `.omx/research/wiring_integration_orphan_audit_post_12_landings_20260518.md` (841 lines; 48/96 cells PLANNED_BUT_UNROUTED)
- `.omx/research/closure_campaign_pursue_and_confirm_master_20260518.md` (1100 lines; 5 OP-AUDIT closure operations)
- `.omx/research/execution_monitoring_synthesis_post_b_landing_20260518.md` (867 lines; today's session in numbers)

### Today's routing directives (~11 directives)
- `.omx/research/codex_routing_directive_*.md` glob — 11 directives landed today routing work to Codex

### Public-PR archives (Catalog #109 vendored intake clones)
- `experiments/results/public_pr101_intake_*/` (PR101 GOLD; HNeRV-family)
- `experiments/results/public_pr102_intake_*/` (PR102 silver)
- `experiments/results/public_pr103_intake_*/` (PR103 silver)
- `experiments/results/public_pr106_intake_*/` (PR106)
- `experiments/results/public_pr107_intake_*/` (PR107 our submission)
- `reverse_engineering/` (curated public-submission deconstruction)

### Memory entries
- `/Users/adpena/.claude/projects/-Users-adpena-Projects-pact/memory/MEMORY.md` (top 50 entries per CLAUDE.md auto-memory)
- `/Users/adpena/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_*.md` (per-topic memory)
- Today's `feedback_*_landed_20260518.md` files (per-subagent completion memos)

### Reports / dashboards
- `reports/latest.md` (current frontier per Catalog #316; STALE per R5-3 reactivation; OPERATIONAL CONCERN)
- `reports/lane_maturity.md` (regenerated via `tools/lane_maturity.py report`)

### Audit tools
- `tools/audit_*.py` family (multiple operator-facing audit surfaces)
- `tools/lane_maturity.py audit` (lane-level audit table)
- `tools/canonical_task_status.py --list-pending`
- `tools/scan_best_anchor_per_axis.py` (Catalog #316 frontier scan)
- `tools/operator_briefing.py` (operator-facing briefing)

## STRATEGIC ROUTING TO IN-FLIGHT SUBAGENTS

Both PRIMARY (`a703d2b74784d4f00`) and ADVERSARIAL (`a74313c6276748ad8`) rate-attack subagents should INCORPORATE this supplement:

1. **PRIMARY**: extend G-category to include the per-axis hardware exploit matrix (Part 1 above) + add H2/H3 hypothesis testing for dual-device master-gradient (Part 2 above) as a SUB-VECTOR within G-category + reference Part 3 inventory in cross-references
2. **ADVERSARIAL**: critique the per-axis matrix completeness (Tao mathematical rigor + Carmack engineering practicality + Hotz CPU/GPU empirical instinct + Boyd convex-feasibility-across-axes) + surface any per-axis vectors PRIMARY misses + challenge whether dual-device master-gradient is empirically necessary OR cargo-culted

Both subagents will be notified via SendMessage referencing this supplement.

— Main-Claude 2026-05-18 (operator clarification incorporation)
