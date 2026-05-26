# Cascade C: P19 PoseNet-null + P18 SegNet-region-waterfill + P11 per-region selector codec — LANDED 2026-05-26

**Subagent_id:** `cascade-c-posenet-null-segnet-region-waterfill-per-region-selector-codec-20260526`

**Operator approval:** 2026-05-26 *"absolutely and enthusiastically"* for EXPLOIT 1.

**Mission alignment per Catalog #300:** `frontier_breaking_enabler` (canonical entropy-position discipline empirically validated at SCORER-ENTROPY attack class; alternative reducer pathway surfaced per Catalog #308).

**Sister coordination per CLAUDE.md "Subagent coherence-by-default":** active sister subagents (NSCS06 v8 stacked Modal T4 RE-FIRE / Z7-Mamba-2 v2 L2 / BoostNeRV Variant C-ii) DISJOINT from this scope per `.omx/state/subagent_progress.jsonl` snapshot.

---

## Headline verdict

**EMPIRICAL IMPLEMENTATION-LEVEL FALSIFICATION per Catalog #307.** All 3 candidate per-region selector codec partitions (V1 3-region, V2 2-region, V3 4-region) measured **+62 to +91 bytes WORSE** than canonical FEC6 wire (249 bytes). PARADIGM (scorer-entropy attack via P18+P19+P11 cascade) **INTACT** — alternative reducers per Catalog #308 are surfaced as operator-routable.

**Predicted ΔS per canonical contest rate formula:** `25 × +91 / 37_545_489 = +0.0000606` (WORSE, not better). Cascade C is NOT a frontier-push win at the P11 per-region codec axis.

**Per CLAUDE.md "Forbidden premature KILL without research exhaustion":** DEFERRED-PENDING-RESEARCH on alternative reducer pathways (fold P19 PoseNet-null into FEC8 Markov; SegNet-class waterfill as frame-1 selector design insight; trainer-design research signal).

---

## 1. Pre-execution gate verdict (Catalog #229 / Catalog #305)

Reference: `.omx/research/cascade_c_posenet_null_segnet_region_waterfill_per_region_codec_pre_execution_gate_report_20260526.md`

All gate items PASS: premise verification + sister scope disjoint + entropy-position declared + full-stack fractal decomposition + compress-time-only invariant + canonical Provenance umbrella + MLX-LOCAL only.

---

## 2. Empirical results (compress-time only; `[macOS-CPU advisory]` per Catalog #192)

### 2.1 Baseline FEC6 decode (canonical anchor)

| Metric | Value |
|--------|-------|
| Archive sha256 | `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf` |
| Archive bytes (total) | 178,517 |
| n_pairs | 600 |
| FEC6 selector_payload wire | **249 bytes** (6-byte header + 243-byte Huffman stream) |
| H(mode) marginal | 3.2116 bits/pair |
| Shannon floor (n=600, K=16) | 241 bytes |
| FEC6 Huff slack vs Shannon | 0.0284 bits/pair (~2 bytes) |
| FEC8 Markov (sister #1336) | 245 bytes (-4 vs FEC6) |

### 2.2 Cascade C P11 per-region codec measurements

Three candidate region partitions tested:

| Variant | n_regions | Region structure | Region-idx bits/pair | Total wire (bytes) | Δ vs FEC6 |
|---------|----------:|------------------|---------------------:|-------------------:|----------:|
| V1 | 3 | A(4modes 426p) + B(4modes 101p) + C(8modes 73p) | 1.15 (Huff-best) | 316 | **+67** |
| V2 | 2 | H(4modes 426p) + T(12modes 174p) | 1.00 (binary Huff) | 340 | **+91** |
| V3 | 4 | L2(2modes 263p) + L3(2modes 163p) + L5(4modes 101p) + L7(8modes 73p) | 1.83 (Huff-best) | 311 | **+62** |

**All 3 variants FALSIFIED.** Best variant (V3) is still +62 bytes worse than FEC6, predicting `+0.000041 ΔS` (worse).

### 2.3 Root cause analysis (per entropy-position discipline)

Per **just-landed entropy-position-discipline standing directive Lesson 1+2**:
- **P11 partition operates AT entropy coder boundary** (Lesson 2 territory: integer-codeword constraint bounds achievable)
- **Chain rule**: `H(region, mode) = H(region) + H(mode|region) = H(mode)` — partition preserves joint entropy exactly
- **Region-idx stream is NEW overhead** that the un-partitioned single-Huffman stream did not pay
- **Per-region Huffman codebooks** add ~48-56 bytes overhead per partition design
- **Net result**: total wire ALWAYS ≥ Shannon floor + region-partition overhead = ~241 + ~70-100 = 311-340 bytes

This is the **canonical entropy-position lesson**: partitioning at the entropy-coder boundary does NOT extract savings; it only re-arranges entropy with new overhead. Per Lesson 1 (BEFORE entropy coder wins), the only winning move is shifting the underlying distribution (FEC8 Markov 1st-order conditional H(X|prev)=2.94 vs marginal H(X)=3.21).

### 2.4 Information-theoretic decomposition (V1 example)

| Component | Bits/pair | Notes |
|-----------|----------:|-------|
| H(mode) marginal | 3.2116 | Joint entropy floor |
| H(region) V1 | 1.1533 | Region-idx stream entropy |
| H(mode\|region) sum | 2.0583 | = 3.2116 - 1.1533 (chain rule verification) |
| Wasted region-idx bits/pair | 0.8467 | Fixed 2-bit minus H(region) = 0.85 bits/pair × 600 = 64 bytes pure overhead |

The 0.85 bit/pair "wasted region-idx bits" represents the redundancy between fixed-width region encoding and the actual region-idx entropy. Even with optimal Huffman on region-idx (collapsing to H(region) bits/pair), there is no path to net savings because the **PARTITION ITSELF adds wire bytes** the un-partitioned coder did not pay.

---

## 3. P19 PoseNet-null analysis (preserved as RESEARCH SIGNAL)

Per sister #1324 OPT-12 PoseNet-null bottom-decile artifact (`.omx/research/pr110_opt_frame0_bundle_artifacts_20260526/pr110_opt12_posenet_null_frame0.json`):

- Top-3 PoseNet-null modes: `frame0_widened_dct_u1_v2_amp_1` (|d_pose|=1.25e-7), `frame0_widened_blue_chroma_amp_2` (3.30e-7), `frame0_widened_dct_u1_v2_amp_2` (3.47e-7)
- **50% of bottom-decile are DCT-chroma**; 37.5% are blue-chroma-family
- Structured signed 8×8 chroma patterns dominate the PoseNet-null axis

**Cascade C P19 hypothesis tested**: pairs assigned to PoseNet-null modes can use REDUCED selector menu. **Empirically**: pairs already assigned `none` (134 pairs = 22.3%) ARE the canonical PoseNet-null subset in the LIVE FEC6 menu. The remaining structured-chroma modes (DCT/blue_chroma) are NOT in the current K=16 menu (which uses `frame0_blue_chroma_amp_{1,3}` only, not DCT).

**Research signal**: a future PR111 selector iteration could replace 2-3 low-frequency rgb_bias modes with DCT-chroma modes (frame0_dct_u1_v2_amp_{1,2}) — but this is an axis-shift in the MENU DESIGN, not a per-region partition of the EXISTING menu.

## 4. P18 SegNet-class-region waterfill (structural validation)

Per CLAUDE.md "Exact scorer architectures — VERIFIED from upstream modules.py":
- SegNet: `smp.Unet('tu-efficientnet_b2', classes=5)` processes ONLY last frame `x[:, -1, ...]`
- **Frame-0 perturbations have seg_delta = 0 STRUCTURALLY** (confirmed by canonical sweep tool for all 87 widened frame-0 modes)
- SegNet's stride-2 stem loses half resolution → artifacts below (256, 192) invisible per CLAUDE.md

**Cascade C P18 hypothesis tested**: per-class-region SegNet logit-margin waterfill enables LARGER frame-0 perturbations in low-margin regions. **Empirically structurally invalid** for FRAME-0 perturbations (SegNet is structurally insensitive to frame-0 regardless of region). However, **structurally valid for FRAME-1 perturbations** which is the PR110 frame-1 selector axis NOT explored in this lane.

**Research signal**: a future Cascade C' variant could attack the FRAME-1 perturbation selector menu with per-class-region SegNet-margin waterfill. This is the canonical Yousfi-Fridrich inverse-steganalysis pattern applied to the FRAME-1 axis.

---

## 5. Carmack-dissent verdict per Catalog #307

**Falsification classification:** **IMPLEMENTATION-LEVEL FALSIFICATION** of the specific "per-region selector codec replaces global K=16" implementation.

**PARADIGM INTACT:** The Cascade C scorer-entropy attack class (P18+P19+P11 cross-position composition) remains the canonical Fridrich-Yousfi inverse-steganalysis frame. The IMPLEMENTATION (3-region partition) fails because:
1. Chain rule preserves joint entropy → no information gain from partition
2. Region-idx stream is NEW overhead not amortized by any per-region savings
3. K=16 alphabet is already near Shannon-floor (0.0284 bits/pair slack) — there is no per-region "fat" to trim

**Per CLAUDE.md "Forbidden premature KILL without research exhaustion":** verdict is **DEFERRED-PENDING-RESEARCH on alternative reducer pathways** per Catalog #308. NOT KILLED.

---

## 6. Alternative reducers per Catalog #308

Per Catalog #308 enumeration of alternative probe methodologies (REFUSED dispatch without ≥3 alternatives):

### Alternative reducer A: fold P19 PoseNet-null into FEC8 Markov 2nd-order (#1336 extension)

The 1st-order Markov (#1336) captures pair-to-pair conditional distribution. A 2nd-order conditional `P(mode_t | mode_{t-1}, pose_null_flag_t)` would directly exploit the PoseNet-null clustering temporally. **Estimated savings**: -2 to -8 additional bytes beyond #1336's -4.

### Alternative reducer B: SegNet-class waterfill applied to FRAME-1 selector

PR110 currently encodes ONLY frame-0 perturbations (16-mode K=16). A sister FRAME-1 selector menu could apply SegNet-class waterfill (high-margin regions = preserve clean RGB; low-margin regions = absorb perturbation). **Estimated savings**: research signal; requires new substrate engineering.

### Alternative reducer C: trainer-design research signal

The PoseNet-null bottom-decile structure (DCT + blue_chroma family) suggests that training curriculum should explicitly include structured-signed-chroma data augmentation. **Estimated savings**: indirect (improves substrate quality); operator-routable for future c1a_l3 curriculum stage.

---

## 7. Catalog #344 candidate equation (proposed; awaits operator approval)

**Proposed canonical equation:** `per_region_selector_codec_savings_v1`

**Mathematical form:**
```
delta_wire_bytes(n_pairs, n_regions, region_partition) =
  header_bytes(n_regions, codebook_overhead)
+ region_idx_bytes(n_pairs, H_region)
+ sum_r mode_in_region_bytes(n_r, H(mode|region=r))
- baseline_huffman_wire_bytes
```

**Predicted EMPIRICAL anchor** for live PR110 fec6 archive (sha `6bae0201...`):
- For n_regions ∈ {2, 3, 4}: predicted delta_wire ∈ [+62, +91] bytes (NET WORSE than FEC6)
- Chain rule invariant: `H(region) + sum_r P(r) * H(mode|region=r) = H(mode_marginal)`

**Empirical anchor measurement** (this lane): 3 partition variants tested, all FALSIFIED at the savings axis.

**Operator-decision protocol per Catalog #344:** this equation, if registered, would document the entropy-position invariant that partition-at-entropy-coder-boundary preserves joint entropy with overhead cost. The equation is **negative-result canonical** (codifies what NOT to attempt). Operator may approve registration as a learning artifact OR defer because it's a negative-result that's already implicit in the entropy-position discipline standing directive.

---

## 8. Drift surface declaration per MLX↔CUDA bidirectional drift anticipation standing directive

Per sister standing directive `feedback_mlx_cuda_bidirectional_drift_anticipation_standing_directive_20260526.md`:

**Drift sources for Cascade C**: N/A. This lane is COMPRESS-TIME ENTROPY-ANALYSIS only; no MLX↔CUDA computation involved. The entropy calculation is float64 Python arithmetic on per-pair mode-assignment integers — exact, deterministic, byte-stable across platforms. No drift mitigation needed.

---

## 9. Canonical-vs-frontier-push decision per sub-ingredient (per pushing-frontier standing directive)

Per sister standing directive `feedback_pushing_the_frontier_of_research_on_optimization_algorithms_standing_directive_20260526.md`:

| Sub-ingredient | Canonical-vs-frontier-push | Rationale |
|---|---|---|
| P19 PoseNet-null subset selection | CANONICAL (use existing sister #1324 OPT-12 artifact) | Already-landed canonical helper |
| P19 reduced-menu encoding for bottom-decile | FRONTIER-PUSH (no canonical helper) | Novel selector-design axis |
| P18 SegNet logit-margin map | CANONICAL (use existing canonical sweep tool sensitivity output) | Already-landed canonical helper |
| P18 Fridrich-UNIWARD reallocation | FRONTIER-PUSH (no canonical helper) | Novel scorer-design axis |
| P11 region partition design | FRONTIER-PUSH (no canonical helper) | Novel codec-design axis |
| P11 per-region menu Huffman codebook | CANONICAL (use existing fec6 Huffman builder) | Already-landed canonical helper |
| P11 per-pair region-idx stream | FRONTIER-PUSH (no canonical helper) | Novel codec-design axis |

3 of 7 sub-ingredients used canonical helpers; 4 of 7 were frontier-push because Cascade C IS the first attempt at the SCORER-ENTROPY attack class.

---

## 10. 6-hook wire-in declaration (Catalog #125)

| Hook | Status | Path |
|------|--------|------|
| 1. Sensitivity-map contribution | **N/A — research_only=true** | Per-pair PoseNet-gradient is structural signal; integration deferred until paired-CUDA validation |
| 2. Pareto constraint | **N/A — empirical falsification** | The 3-axis (d_seg, d_pose, archive_bytes) polytope shows positive Δ archive_bytes for all 3 variants; no new Pareto vertex |
| 3. Bit-allocator hook | **N/A — research_only=true** | Selector-menu bit budget unchanged |
| 4. Cathedral autopilot dispatch hook | **N/A — research_only=true** | This lane is FREE local MLX/CPU; no paid dispatch initiated |
| 5. Continual-learning posterior update | **ACTIVE (negative-result anchor)** | Per CLAUDE.md "Apples-to-apples evidence discipline" — the EMPIRICAL FALSIFICATION is itself a posterior signal. Will be persisted to `.omx/state/continual_learning_posterior.jsonl` via canonical helper IF operator approves (negative-result anchors are LOW-PRIORITY per posterior signal-density principle) |
| 6. Probe-disambiguator | **ACTIVE** | The 3 partition variants ARE the disambiguator between "per-region partition can win" (FALSIFIED) vs "joint entropy invariance prevents win" (RATIFIED) |

---

## 11. HORIZON-CLASS declaration per Catalog #309

**HORIZON-CLASS: plateau_adjacent** — predicted band [+0.000041, +0.000061] (NEGATIVE direction; worse-than-FEC6). The K=16 selector stream sits at the canonical 0.196-0.199 plateau frontier; per-region partition cannot escape it because the entropy is already near-Shannon-floor.

**Per CLAUDE.md "Horizon-class evaluation axis"**: plateau-adjacent classification means the next high-EV move at this position is NOT another within-class partition variant but a **class-shift** to a different entropy-position (P19 trainer-design / P18 frame-1 selector / sister Markov 2nd-order).

---

## 12. Observability surface (Catalog #305)

| Facet | How surfaced |
|-------|--------------|
| Inspectable per layer | Per-variant function `per_region_codec_huffman_idx(regions_map, codes, n_pairs)` is pure-function; inputs = K=16 partition + decoded codes; outputs = wire-byte budget decomposition |
| Decomposable per signal | Output JSON decomposes `total_wire_bytes` into `header_bytes` + `region_idx_bytes` + `mode_in_region_bytes` + `codebook_overhead_bytes` |
| Diff-able across runs | All artifacts under `.omx/research/cascade_c_artifacts_20260526/` are JSON with sorted keys (deterministic byte-stable output for fixed inputs) |
| Queryable post-hoc | All artifacts are JSON; `jq` queryable; sha256 of each artifact recorded |
| Cite-able | Every artifact carries `archive_sha256` + `axis_tag=[macOS-CPU advisory]` + `provenance.subagent_id` + `provenance.method` |
| Counterfactual-able | Re-run with different `REGIONS` dict to test per-partition counterfactuals |

---

## 13. Custody summary

| Artifact | Path | sha-prefix |
|----------|------|------------|
| Pre-execution gate report | `.omx/research/cascade_c_posenet_null_segnet_region_waterfill_per_region_codec_pre_execution_gate_report_20260526.md` | (per write) |
| Per-region codec V1 empirical (3-region 4-4-8) | `.omx/research/cascade_c_artifacts_20260526/cascade_c_per_region_codec_empirical.json` | `f76132209cea59cd...` |
| Alternative reducers V1+V2+V3 empirical | `.omx/research/cascade_c_artifacts_20260526/cascade_c_alternative_reducers_empirical.json` | `bde9db8e8df31355...` |
| Landing memo (THIS) | `.omx/research/cascade_c_posenet_null_segnet_region_waterfill_per_region_codec_landed_20260526.md` | (per write) |

All artifacts carry `axis_tag=[macOS-CPU advisory]` + `archive_sha256=6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf` + `evidence_grade=macOS-CPU-advisory-only` + `promotable=False` + `score_claim=False` + `ready_for_exact_eval_dispatch=False` + 5 canonical promotion_blockers per Catalog #287 + #192 + #323 canonical Provenance umbrella.

---

## 14. Operator-routable next steps

Priority ordered by expected information gain × inverse cost (free → costly):

### Step 14.1 (priority 1; free; immediate)

**Acknowledge Cascade C P11 per-region codec EMPIRICAL FALSIFICATION + log negative-result canonical equation.** No further per-region partition variants warranted; chain-rule invariance is structural.

### Step 14.2 (priority 2; free; ~30 min wall-clock)

**Surface Alternative Reducer A (FEC8 Markov 2nd-order extension)** to a sister subagent. Extend #1336 FEC8 Markov implementation with `P(mode_t | mode_{t-1}, pose_null_flag_t)` 2nd-order conditional; measure wire delta vs FEC8 245-byte baseline. Estimated -2 to -8 bytes additional savings beyond #1336's -4.

### Step 14.3 (priority 3; deferred; conditional on Step 14.2 win)

**Design Cascade C' variant**: SegNet-class waterfill applied to FRAME-1 selector menu. PR110 currently uses frame-0 selector only; a sister frame-1 selector with Fridrich-UNIWARD weighting could extract scorer-entropy savings at the previously-unexploited frame-1 axis. Requires new substrate engineering + paired-CUDA validation.

### Step 14.4 (priority 4; trainer-design signal; deferred to PR111+ iteration)

**Trainer curriculum augmentation**: explicitly include structured-signed-chroma data augmentation (DCT + blue_chroma family per PoseNet-null bottom-decile evidence) in the c1a_l3 curriculum stage. Indirect substrate-quality improvement; no direct codec savings.

### Step 14.5 (priority 5; documentation; immediate)

**Catalog #344 candidate equation `per_region_selector_codec_savings_v1`** awaits operator decision on registration (negative-result canonical). Per Catalog #344 operator-decision protocol: equation registration is operator-routable; sister subagent will NOT register unilaterally.

---

## 15. Cargo-cult audit per assumption (Catalog #303)

Per Catalog #303 cargo-cult audit discipline applied retrospectively:

| # | Assumption | Classification | Empirical verdict |
|---|------------|----------------|-------------------|
| 1 | Per-region selector codec can extract savings vs global K=16 | **CARGO-CULTED** | Empirically FALSIFIED. Chain rule preserves joint entropy. |
| 2 | 3-region partition is the right partition count | **CARGO-CULTED** | All 3 partition counts (2, 3, 4) FALSIFIED. Partition count is NOT the lever. |
| 3 | PoseNet-null pairs can use reduced selector menu | **HARD-EARNED** | Structurally true — 134 pairs (22.3%) already use `none` mode (PoseNet-null by construction). |
| 4 | SegNet-class waterfill applies to frame-0 | **CARGO-CULTED** | Structurally FALSIFIED — SegNet processes ONLY last frame; frame-0 perturbations have seg_delta=0 by construction. The waterfill heuristic applies to FRAME-1 not frame-0. |
| 5 | Cascade C composition is byte-disjoint with #1336 FEC8 | **NOT-TESTED** | Cascade C P11 falsified before composition tested. Sister Alternative Reducer A (Step 14.2) is the canonical next test. |
| 6 | The K=16 menu has within-region "fat" to trim | **CARGO-CULTED** | Empirically FALSIFIED — FEC6 Huffman is 0.0284 bits/pair from Shannon-floor (already saturated). |

---

## 16. Cross-references

- **`.omx/research/pr110_opt_frame0_bundle_landed_20260526.md`** (sister #1313+#1324+#1325)
- **`.omx/research/pr110_opt3_mode_distribution_20260526T170000Z.md`** (sister #1315)
- **`.omx/research/pr110_opt3_variant_b_markov_landed_20260526.md`** (sister #1336 FEC8 Markov -4 bytes; canonical alternative reducer A pathway anchor)
- **`.omx/research/pr110_opt3_variant_c_variable_k_escape_mechanism_landed_20260526.md`** (sister #1343 Variant C falsification — sister entropy-position discipline Lesson 2)
- **`feedback_entropy_position_discipline_in_full_stack_pipeline_standing_directive_20260526.md`** (just-landed; canonical doctrine this lane operationalizes)
- **`feedback_pr95_sniped_lesson_full_stack_mlx_first_per_candidate_standing_directive_20260526.md`** (just-elevated GUIDING PRINCIPLE; full-stack fractal optimization)
- **`feedback_pushing_the_frontier_of_research_on_optimization_algorithms_standing_directive_20260526.md`** (per-sub-ingredient canonical-vs-frontier-push decision)
- **`feedback_mlx_cuda_bidirectional_drift_anticipation_standing_directive_20260526.md`** (drift surface declaration; N/A for this lane)
- **CLAUDE.md "Strict scorer rule"** (compress-time only; no scorer load at inflate)
- **CLAUDE.md "Fridrich inverse steganalysis"** (UNIWARD pattern; canonical doctrine)
- **CLAUDE.md "Forbidden premature KILL without research exhaustion"** (deferred-pending-research; alternative reducers surfaced)
- **Catalog #307** (paradigm-vs-implementation falsification distinction; PARADIGM INTACT)
- **Catalog #308** (alternative probe methodologies enumeration; 3 alternative reducers surfaced)
- **Catalog #344** (canonical equation registry; negative-result candidate equation proposed)

---

## 17. Lane registration

Lane: `lane_cascade_c_posenet_null_segnet_region_waterfill_per_region_codec_20260526` L1 (impl_complete + memory_entry + research_only=true)

**Cost summary:** $0 GPU + ~30 min wall-clock + 0 paid dispatches. Per operator standing "Remember all on MLX" + per CLAUDE.md "Carmack MVP-first phasing" + per just-landed entropy-position discipline.
