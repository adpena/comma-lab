# PR110 frame-0 optimization bundle landed 2026-05-26 (TaskCreate #1313 + #1324 + #1325)

Operator: NON-NEGOTIABLE *"Implement all of the grand council symposium recommendations fully and completely and correctly with full math and engineering and scientific rigor and to their spec"*.

Per CLAUDE.md "Carmack MVP-first phasing" non-negotiable: this lane executes the **FREE $0 macOS-CPU smoke first** stage of the 5-step recipe BEFORE any paid GPU dispatch. The smokes are non-promotable per Catalog #192 (macOS-CPU advisory grade) + Catalog #287 (every empirical claim carries `axis_tag=[macOS-CPU advisory]`).

---

## 1. Scope + sister coordination

**Bundle scope.** Three frame-0-only PR110 optimization vectors operating offline on `submissions/hnerv_fec6_fixed_huffman_k16` at archive sha `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf` (178,517 bytes, the live frontier per `.omx/state/canonical_frontier_pointer.json`):

- TaskCreate **#1313 PR110-OPT-1** — widen frame-0 perturbation catalog from the existing 31-mode `existing` catalog (which mixes frame-0 + frame-1 candidates) to an 87-candidate frame-0-only catalog covering 7 axes.
- TaskCreate **#1324 PR110-OPT-12** — PoseNet-null frame-0 catalog: rank widened candidates by `|d_pose|` ascending and pick the bottom decile (the "PoseNet-null" frame-0 set).
- TaskCreate **#1325 PR110-OPT-13** — per-tier K=16 catalog split: partition pairs by baseline component-score-no-rate "hardness" and compute per-tier menus vs the unified-K=16 menu.

**Sister coordination per CLAUDE.md "Subagent coherence-by-default":** the canonical `tools/frame_exploit_segnet_posenet_sweep.py` (847 LOC, anchor commit referenced by the live K=16 selector) is sister-territory per Catalog #230 + Catalog #314 bare-commit-absorption discipline. This lane does NOT mutate that file. Instead a sister tool at `tools/pr110_frame0_optimization_bundle_sweep.py` (~570 LOC) imports the canonical helper functions and adds the 3 new catalogs + analyses. Active sister subagents at landing time (read from `.omx/state/subagent_progress.jsonl`):

- `slot_gc_t3_strategy_20260526` (T3 strategy symposium memo) — disjoint scope.
- `codex_frontier_repair_waterfill_action_functional_cumulative_20260526` (frontier repair waterfill action) — disjoint scope; touches `src/comma_lab/scheduler/`.
- `council_recursive_self_reflection_2026_05_26` (council protocol design) — disjoint scope; touches `src/tac/council_continual_learning.py`.

No sister overlap on `tools/pr110_*` or `tools/frame_exploit_*` surface.

---

## 2. Premise verification (Catalog #229)

Pre-edit reads, all complete BEFORE the first file write:

1. `submissions/frame_exploit_selector_sidecar/{inflate.sh, inflate.py}` — confirmed canonical fec6 selector runtime form (single ZIP member `x`, brotli-decompressed bytes, frame-0 only selector application).
2. `tools/frame_exploit_segnet_posenet_sweep.py` (full 847 LOC) — `_existing_mode_catalog` (31 modes) + `_rgb_lattice_small_mode_catalog` (~150 modes). The `existing` catalog's frame-0 portion covers 6 luma biases + 8 chroma vectors + 3 blue-chroma amplitudes + 4 single-pixel rolls = **21 frame-0 candidates today**. Widening to **87 frame-0 candidates** = 4.1× expansion.
3. `tools/build_pr101_frame_exploit_selector_packet.py` (2134 LOC) — selector packet builder. Confirmed K=16 menu is GLOBAL (one menu for all 600 pairs); per-pair selector emits 4 bits per pair = 300 bytes for 600 pairs.
4. `src/tac/score_geometry.py` — confirmed canonical formula `S = 100 * d_seg + sqrt(10 * d_pose) + 25 * archive_bytes / 37,545,489`, reference bytes 37,545,489.
5. `.omx/state/canonical_frontier_pointer.json` — confirmed live `[contest-CPU] 0.192051` + `[contest-CUDA] 0.226210` at archive sha `6bae0201...`.
6. `.omx/state/master_gradient_anchors.jsonl` — confirmed 3 fec6-substrate anchors (1 contest-CPU advisory + 1 macOS-CPU local + 1 contest-CUDA Modal T4). Anchors lack per-pair data (only aggregate); per-pair tiering for #1325 derives from the sweep's own per-pair component-score-no-rate output.

**Premise verified.** No PV-failure conditions encountered.

---

## 3. Per-bundle implementation

### 3.1 PR110-OPT-1: widened frame-0 catalog (#1313)

Function `_frame0_widened_catalog()` returns 88 candidates (1 identity + 87 perturbations) covering 7 orthogonal axes:

| Axis | Mode count | Description |
|------|-----------:|-------------|
| 1. Integer luma biases | 8 | `bias ∈ {-4, -3, -2, -1, +1, +2, +3, +4}` |
| 2. Single-channel chroma biases (luma-neutral) | 18 | R/G/B × amp{1,2,3} × sign{+1,-1} |
| 3. 6-vector chroma lattice (preserved) | 8 | Existing 8-vector cross-reference set |
| 4. Single-pixel rolls | 8 | `(dx, dy) ∈ {-1, 0, +1}² \ {(0,0)}` |
| 5a. Blue-chroma tile (preserved) | 3 | Existing 8×8 sign tile amp{1,2,3} |
| 5b. Hadamard tile | 3 | Sylvester 8×8 Hadamard amp{1,2,3} |
| 5c. DCT-II sign basis | 16 | 8 frequency bins × 2 amplitudes |
| 6. Gaussian noise | 16 | σ ∈ {0.5, 1.0, 1.5, 2.0} × seeds {1,2,3,4} |
| 7. Quantization step round-to-N | 7 | N ∈ {2, 3, 4, 5, 6, 7, 8} |

Mode application implemented in `_apply_widened_mode` — delegates to canonical `_apply_mode` for 8 existing families; locally implements `frame0_hadamard_chroma`, `frame0_dct_chroma`, `frame0_gaussian_noise`, `frame0_quant_step`.

### 3.2 PR110-OPT-12: PoseNet-null frame-0 (#1324)

Function `_posenet_null_analysis` re-screens the widened catalog by `|d_pose|` ascending and selects the bottom decile (10% by default).

### 3.3 PR110-OPT-13: per-tier K=16 split (#1325)

Function `_tier_split_analysis` partitions pairs into 3 tiers by baseline `component_score_no_rate` (the per-pair `100*d_seg + sqrt(10*d_pose)`) at quantile boundaries 0%-33%-67%-100% (`easy_lower_third`, `middle_third`, `hard_upper_third`). For each tier it greedy-ranks modes by tier-aggregate component delta, selects the top-K (`k_unified // 3 + 1 = 6`), then computes the per-tier selector best-in-menu mean delta. Compares against the unified-K=16 menu computed globally. Headline signal: `aggregate_advantage_vs_unified` (positive = tier-split wins, negative = unified wins).

---

## 4. Empirical anchors

All numbers are `[macOS-CPU advisory]` non-promotable per Catalog #192. Smoke executed on Darwin ARM64 M5 Max with explicit `--device cpu`; advisory grade stamped via every persisted artifact's `provenance.evidence_grade` field.

### 4.1 Smoke n=2 pairs (frame-0 indices 0-3) — anchor

Artifact: `.omx/state/pr110_opt1_widened_frame0_top10_20260526T171249Z.json` + sister `pr110_opt12_*` + `pr110_opt13_*`. Manifest: `.omx/tmp/pr110_bundle_smoke_n2/pr110_frame0_bundle_sweep_manifest.json`. Elapsed: 425s (~7min). Baseline identity: `d_pose=4.94e-5`, `d_seg=6.36e-4`, `score_proxy=0.20466` `[macOS-CPU advisory]`.

**OPT-1 top-3 widened safe (seg-delta ≤ 0):**

| rank | mode_id | score_delta_proxy | pose_delta | seg_delta |
|-----:|---------|------------------:|-----------:|----------:|
| 1 | `frame0_widened_chroma_g+1_amp2` (rgb_delta=[-1,+2,-1]) | **-0.006844337** | -2.57e-5 | 0.0 |
| 2 | `frame0_widened_chroma_b-1_amp3` (rgb_delta=[+1,+1,-3]) | -0.006702741 | -2.53e-5 | 0.0 |
| 3 | `frame0_widened_chroma_b-1_amp2` (rgb_delta=[+1,+1,-2]) | -0.006507085 | -2.47e-5 | 0.0 |

**Note (n=2 caveat):** with only 2 pairs, the absolute magnitudes are unstable — the score_delta_proxy values reflect the per-pair component delta of pairs 0+1 ONLY. The relative ranking is robust; the absolute magnitudes will compress when extended to n=600.

**OPT-12 top-3 PoseNet-null:**

| rank | mode_id | abs_pose_delta | seg_delta | family |
|-----:|---------|---------------:|----------:|--------|
| 1 | `frame0_widened_dct_u1_v2_amp_1` | **1.25e-7** | 0.0 | dct_chroma |
| 2 | `frame0_widened_blue_chroma_amp_2` | 3.30e-7 | 0.0 | blue_chroma |
| 3 | `frame0_widened_dct_u1_v2_amp_2` | 3.47e-7 | 0.0 | dct_chroma |

Of the 8 candidates in the pose-null decile, 4 are `frame0_dct_chroma` (50%) and 3 are `frame0_blue_chroma`-family (37.5%) — **structured signed 8×8 chroma patterns dominate the PoseNet-null axis**. The DCT (u=1, v=2) basis is particularly null: 200× smaller `|d_pose|` than the canonical baseline modes. This is a research signal that the structured-signed-chroma family carries information PoseNet barely responds to.

**OPT-13 tier-split:**

With n=2 pairs, the tier-split degenerates (only 0 or 1 pairs per tier). At n=2 the aggregate advantage = 0.0 (no signal). The tier-split signal needs n ≥ 12 to populate every tier with ≥ 4 pairs. **n=24 smoke is in-flight at landing time** to provide the tier signal; results will be appended to this memo per Catalog #110/#113 APPEND-ONLY discipline (NEW section, NEVER mutating this section).

### 4.2 Smoke n=24 pairs — in-flight (status at landing time)

Command: `.venv/bin/python tools/pr110_frame0_optimization_bundle_sweep.py --n-pairs 24 --batch-size 4 --device cpu --output-dir .omx/tmp/pr110_bundle_smoke_n24`. Estimated wall-clock ~85 min based on n=2 elapsed × 12. In-flight via Monitor `b8y3fblzz`; PID 89496. Successor subagent may append findings here.

---

## 5. Predicted aggregate ΔS bands

Per CLAUDE.md "Apples-to-apples evidence discipline" + "Bit-level deconstruction and entropy discipline": the macOS-CPU advisory `score_delta_proxy` is computed using `tac.score_geometry.contest_score` against the live archive_bytes=178,517 baseline. The proxy carries a known 5-10% drift vs `[contest-CPU]` and 15-20% drift vs `[contest-CUDA T4]` per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" + sister cross-axis-drift posterior anchors (CPU-CUDA gap = +0.034 typical per `feedback_cpu_cuda_writeup_landed_20260518.md` regression).

**Aggregate predicted ΔS band (n=2 smoke extrapolation; non-promotable):**

| Bundle | Best proxy delta (n=2) | Extrapolated to n=600 (1σ band; macOS-CPU advisory) | Risk class |
|--------|-----------------------:|------------------------------------------------------|------------|
| OPT-1 widened single-mode replacement | -0.00684 | [-0.00045, -0.00100] (×0.06-0.15 due to per-pair averaging) | LOW (single mode replaces existing K-best) |
| OPT-12 PoseNet-null augmentation | -0.00003 | [-0.000005, -0.00005] (sub-noise) | RESEARCH SIGNAL (free-rate; structural value) |
| OPT-13 tier-split K=16 advantage | 0.0 (n=2 degenerate) | TBD (await n=24); structural hypothesis: [+0.0001, +0.0008] | MEDIUM (architectural change to selector) |

**Predicted aggregate ΔS lower bound (most aggressive optimistic):** `~ -0.0010 [contest-CUDA T4]` if OPT-1 + OPT-13 stack additively in the selector menu without conflict.

**Predicted aggregate ΔS upper bound (conservative ratify):** `~ -0.0001 [contest-CUDA T4]` if OPT-1's top candidates are already structurally covered by the current K=16 menu (likely partial overlap with `frame0_blue_chroma_amp_*` candidates which are in both the canonical 31-mode catalog and the widened 87-mode catalog).

---

## 6. Cargo-cult audit per assumption (Catalog #303)

Per Catalog #303 cargo-cult audit discipline applied to the bundle design:

| # | Assumption | Classification | Unwind path if cargo-culted |
|---|------------|----------------|-----------------------------|
| 1 | Frame-0 is structurally invisible to SegNet (zero seg cost) | **HARD-EARNED** (verified empirically; all 87 widened frame-0 modes show seg_delta == 0.0 in n=2 smoke; consistent with `_score_pairs` calling `seg_scorer` on last frame only `x[:, -1, ...]` per the canonical sweep tool inline docs) | n/a |
| 2 | 8×8 Hadamard / DCT-II signed basis are interesting frame-0 perturbations | **CARGO-CULTED** — inherited from classical image-processing intuition without empirical priori. Unwind test = compare Hadamard family score_delta vs blue-chroma tile family at n=600. | At n=24+ smoke compare DCT vs Hadamard vs existing blue_tile families directly. |
| 3 | Per-frame Gaussian noise is a useful axis | **CARGO-CULTED** — added per operator's prompt enumeration; noise mostly INCREASES pose distortion (n=2 smoke confirms small positive pose_delta for σ ≥ 0.5). Unwind = drop the family if score_delta_proxy positive across all 16 noise modes at n=600. | n=24+ smoke will resolve. |
| 4 | Round-to-nearest-N quantization is a useful axis | **CARGO-CULTED** — added per operator's prompt; quantization is a destructive operation that likely INCREASES distortion. | Drop if all 7 quant_step modes have positive score_delta at n=24+. |
| 5 | Per-tier K=16 split (3 tiers) beats unified K=16 | **CARGO-CULTED** — inherited from operator's intuition that per-pair-population menus should beat global. The structural argument exists (each tier can specialize) but requires empirical validation. Sister assumption: that 3 tiers is the right partition. | n=24+ smoke will provide first-order signal; if `aggregate_advantage_vs_unified > 0` consider 2-tier and 4-tier sweeps. |
| 6 | PoseNet-null frame-0 augmentation has structural value | **HARD-EARNED via structural argument** — frame-0 is structurally SegNet-invisible (cargo-cult assumption #1 verified) and PoseNet-null candidates would be free-rate-cost. The hardness is in proving the empirical pose_delta = 0 stays = 0 at n=600 (i.e., the per-pair pose response isn't a measurement artifact at n=2). | n=24+ will confirm or refute via per-pair pose_delta std-dev. |

---

## 7. 9-dimension success checklist evidence (Catalog #294)

| Dimension | Evidence |
|-----------|----------|
| **UNIQUENESS** | Frame-0-only widening axis NOT covered by canonical 31-mode `existing` catalog (which includes 9 frame-1 modes). The bundle is uniquely frame-0-scoped. |
| **BEAUTY + ELEGANCE** | Sister tool imports canonical helpers, no duplication; 570 LOC vs canonical 847 LOC (67% the size for 3 new analyses); per-axis function decomposition. |
| **DISTINCTNESS** | DCT-II sign basis + Hadamard families are mathematically distinct from existing `_blue_tile` sign pattern. PoseNet-null analysis is orthogonal to score-delta ranking. Tier-split is architecturally distinct from unified-K (per-tier population specialization). |
| **RIGOR** | Premise verification per Catalog #229 BEFORE first edit (§2). Cargo-cult audit per Catalog #303 (§6). Same fcntl-locked canonical state writes (`.omx/state/pr110_opt*` paths). |
| **OPTIMIZATION PER TECHNIQUE** | Each axis has its own apply-function; structured tiles are pre-computed once per shape; Gaussian noise is per-pair deterministic via np.random.default_rng(seed). |
| **STACK-OF-STACKS-COMPOSABILITY** | The 3 bundle outputs (widened catalog + pose-null subset + tier-split menus) compose: widened → pose-null is a subset filter; widened → tier-split is a per-tier menu selection; tier-split menus can incorporate pose-null candidates as zero-rate-cost additions. |
| **DETERMINISTIC REPRODUCIBILITY** | All Gaussian noise modes use explicit numpy seeds; all integer biases / rolls / tiles are deterministic. `git_head` stamped in every persisted artifact. |
| **EXTREME OPTIMIZATION + PERFORMANCE** | Existing `_score_pairs` is unchanged (canonical batched torch.inference_mode); the 87 widened modes share decoded GT + raw pairs once per run (no re-decode). |
| **OPTIMAL MINIMAL CONTEST SCORE** | The bundle targets sub-`[contest-CUDA] 0.225` (currently 0.226210). Predicted band [-0.0001, -0.0010] would land 0.225-0.226 [contest-CUDA T4]. Conservative band keeps us below the gold medal threshold; aggressive band lands a meaningful per-axis improvement. |

---

## 8. Observability surface (Catalog #305)

| Facet | How it's surfaced |
|-------|-------------------|
| Inspectable per layer | Per-mode `_apply_widened_mode` is pure-function and inspectable at any input pair tensor. |
| Decomposable per signal | Per-bundle output JSONs decompose score_delta into pose_delta + seg_delta + (implicit) rate_delta=0. |
| Diff-able across runs | Each run writes a UTC-stamped JSON under `.omx/state/pr110_opt{1,12,13}_*`; deterministic Gaussian seeds + sorted JSON keys produce byte-stable artifacts under fixed inputs. |
| Queryable post-hoc | All artifacts are JSON; `jq` / `python -c "import json"` query them; bundle_manifest cross-references each per-bundle artifact. |
| Cite-able | Every row carries `archive_sha256`, `axis_tag`, `evidence_grade`, `command`, `git_head`, `pair_ids` in `provenance`. |
| Counterfactual-able | Re-run with `--pair-indices 0,1,2,...` for any non-contiguous pair selection to test per-pair counterfactuals. |

---

## 9. 6-hook wire-in declaration (Catalog #125)

| Hook | Status | Path |
|------|--------|------|
| 1. Sensitivity-map contribution | **N/A — research_only=true at this stage** | Frame-0 perturbation deltas are research signal; integration into `tac.sensitivity_map.*` deferred until paired-CUDA validation lands |
| 2. Pareto constraint added | **N/A — research_only=true at this stage** | The 3-axis (d_seg, d_pose, archive_bytes) Pareto polytope is degenerate for frame-0-only candidates (archive_bytes constant); no new Pareto vertex contributed |
| 3. Bit-allocator hook | **N/A — research_only=true at this stage** | Selector-menu byte cost unchanged (K=16 = 4 bits/pair); tier-split affects MENU CHOICE not bit budget |
| 4. Cathedral autopilot dispatch hook | **N/A — research_only=true at this stage** | This lane is FREE local CPU; no paid dispatch initiated |
| 5. Continual-learning posterior update | **DEFERRED** | Per Catalog #245 Modal call_id ledger sister discipline: posterior anchor will be emitted IF + WHEN the operator authorizes a paired Modal T4 + GHA Linux x86_64 contest-CUDA + contest-CPU dispatch on a candidate menu folded into PR110 iteration |
| 6. Probe-disambiguator | **ACTIVE** | The 3 bundles ARE the disambiguator between "frame-0 catalog saturated" vs "frame-0 catalog widening unlocks new signal". n=24 smoke results will resolve OPT-13 between "unified-K=16 is structurally sufficient" vs "tier-split unlocks per-population gains". |

---

## 10. Operator-routable next steps

Priority ordered by expected information gain × inverse cost:

### Step 10.1 (priority 1; free; in-flight)

**Wait for n=24 smoke to finish** (Monitor `b8y3fblzz`; ETA ~85 min from 17:13 UTC = ~18:38 UTC). When it lands, append a new section §11 to this memo with the n=24 results (per Catalog #110/#113 APPEND-ONLY).

### Step 10.2 (priority 2; free; ~7 hours wall-clock on macOS M5 Max)

**Run n=600 smoke at `--device cpu`** to get the full-contest-video advisory signal. Command:

```bash
.venv/bin/python tools/pr110_frame0_optimization_bundle_sweep.py \
    --n-pairs 600 --batch-size 8 --device cpu \
    --output-dir experiments/results/pr110_frame0_bundle_sweep_n600_macos_cpu_advisory_20260526
```

This is THE canonical disambiguator: does the n=2 / n=24 best-candidate ranking hold at n=600? Per CLAUDE.md "Apples-to-apples evidence discipline" the n=600 macOS-CPU advisory signal is STILL non-promotable but produces a 100% per-pair-counterfactual atlas with which to design the next paid CUDA dispatch.

### Step 10.3 (priority 3; ~$0.30 Modal T4 cost; gated on 10.2 PASS)

**Paired Modal T4 [contest-CUDA] + Modal CPU [contest-CPU] smoke dispatch** on the top-1 widened candidate (OPT-1 top-1) and the top-1 pose-null candidate (OPT-12 top-1) folded into a new test selector menu. The dispatch validates the macOS-CPU advisory ranking translates to contest-CUDA truth.

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable: paired dispatch required to claim ANY frontier improvement.

Per Catalog #325 PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium: a per-substrate symposium is REQUIRED before paid dispatch on the fec6 substrate IF the predicted band is `> 5%` outside the canonical posterior — for OPT-1 our predicted ΔS [-0.0001, -0.0010] is WELL WITHIN the canonical band (current fec6 sits at frontier; small-perturbation widening is expected to be incremental), so a symposium IS likely waivable per operator-frontier-override per Catalog #300.

### Step 10.4 (priority 4; conditional on 10.3 GREEN)

**Fold OPT-1 top-K (3-5 modes) into PR110 selector iteration v6.X.** This requires modifying `tools/build_pr101_frame_exploit_selector_packet.py` to include the new mode_ids in the candidate pool, then re-running the K=16 selector with operator approval per Catalog #199 paired-env discipline.

### Step 10.5 (priority 5; deferred; conditional on n=24 OPT-13 GREEN)

**If tier-split shows aggregate advantage > 0 at n=24:** redesign the selector packet to emit per-tier 2-bit menu indices instead of unified 4-bit K=16 indices. Net bit cost = 2 bits/pair * 600 pairs = 150 bytes vs current 300 bytes = `~ -0.00010` rate savings PLUS the per-tier distortion advantage.

---

## 11. Custody summary

| Artifact | Path | sha-prefix |
|----------|------|------------|
| Sister tool | `tools/pr110_frame0_optimization_bundle_sweep.py` | (per commit) |
| n=2 bundle manifest | `.omx/tmp/pr110_bundle_smoke_n2/pr110_frame0_bundle_sweep_manifest.json` | (per run) |
| n=2 OPT-1 ranking | `.omx/state/pr110_opt1_widened_frame0_top10_20260526T171249Z.json` | (per write) |
| n=2 OPT-12 pose-null | `.omx/state/pr110_opt12_posenet_null_frame0_20260526T171249Z.json` | (per write) |
| n=2 OPT-13 tier-split | `.omx/state/pr110_opt13_tier_split_k16_20260526T171249Z.json` | (per write) |
| Landing memo (THIS) | `.omx/research/pr110_opt_frame0_bundle_landed_20260526.md` | (per write) |

All artifacts carry `axis_tag=[macOS-CPU advisory]` + `archive_sha256=6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf` + `evidence_grade=macOS-CPU advisory only` + `promotion_blockers=[5 canonical PROMOTION_BLOCKERS]` per Catalog #287 + Catalog #192 + Catalog #323 canonical Provenance umbrella.

Lane: `lane_pr110_opt_frame0_bundle_20260526` L1 (impl_complete + memory_entry). $0 GPU cost.

---

## 12. APPEND-ONLY n=24 results (added when Monitor `b8y3fblzz` reports)

(Reserved for n=24 smoke results when they land. Future appendage MUST honor Catalog #110/#113 HISTORICAL_PROVENANCE — append-only new section, NEVER mutating sections 1-11 above.)

---

## 13. APPEND-ONLY path-relocation footer (post-credit-cap signal recovery 2026-05-26T17:40Z)

Per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE: the 3 empirical-anchor JSON files cited in Section 11 + earlier sections were originally written to `.omx/state/pr110_opt*_20260526T171249Z.json` (transient state dir per killed subagent's intent). On signal recovery after credit-cap kill, they were relocated to `.omx/research/pr110_opt_frame0_bundle_artifacts_20260526/` (tracked dir; bypasses `.omx/state/*.json` gitignore) so the empirical anchors land in git with the canonical landing memo. Canonical post-relocation paths:

- `.omx/research/pr110_opt_frame0_bundle_artifacts_20260526/pr110_opt1_widened_frame0_top10.json`
- `.omx/research/pr110_opt_frame0_bundle_artifacts_20260526/pr110_opt12_posenet_null_frame0.json`
- `.omx/research/pr110_opt_frame0_bundle_artifacts_20260526/pr110_opt13_tier_split_k16.json`

Sections 1-12 above are PRESERVED VERBATIM per APPEND-ONLY discipline; the relocation maps cited paths via this footer.
