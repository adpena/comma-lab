# Deeper granularity addition directive: boundaries + master-gradient boundaries + xray + hard pairs + sensitive bytes + sensitivity map
# Date: 2026-05-18
# Per CLAUDE.md "Subagent coherence-by-default" inter-agent directive pattern
# SUPPLEMENTS the deeper-granularity discovery scope in the in-flight subagent a468f72aa4ca0b50a and any subsequent subagents

## Audience

All in-flight + future subagents working on multi-granularity analytical surfaces inventory, master-gradient null-exploitation, Wyner-Ziv Tier-2 design, or any related cross-stack synthesis work.

## Operator additional context (verbatim 2026-05-18)

> *"and boundaries and master gradient boundaries and xray and hard pairs and sensitive bytes and sensitivity map"*

This EXTENDS the prior 12-granularity list (bit / bytes / zeros / ones / pixel / frame / pair / master-gradient / regions / labels / categories / Venn-diagram) with 6 more granularities. The full deeper-discovery scope is now 18 granularity classes.

## 6 ADDITIONAL granularities to incorporate

### 1. Boundaries (per-boundary)

- **SegNet boundary smoothing** (existing canonical at 3-LOC inflate.py per ab66ac8a / Top EIG/$ #1 lane) — what other boundary-classification surfaces exist?
- **Inter-class boundaries** (5-class × 4 pairwise = 10 boundary types) — per-boundary codec specialization
- **Frame boundaries** (transitions between video frames; per-pair compositions)
- **Region boundaries** (between contiguous same-class regions)
- **Temporal boundaries** (per-pair-pair transitions; motion-vector discontinuities)
- **Quantization boundaries** (where int4 → int6 → int8 → fp16 transitions live in sensitivity_mask_aware_quantizr_v1)
- **Archive section boundaries** (where one canonical helper's output ends and another's begins; magic-codec auto-selector decisions)

### 2. Master gradient boundaries (per-MG-boundary)

This is the inflection-point granularity:

- **Sign-flip boundaries**: where the master gradient direction reverses — these are MAXIMALLY-sensitive points (small mods cross zero-cost surfaces)
- **Magnitude-cliff boundaries**: where ||master_gradient|| drops by 10x+ between adjacent bytes — these are MINIMALLY-sensitive points (good targets for null-space-aligned reductions)
- **cos(seg, pose) boundaries**: where cos transitions from rank-degenerate (0.9+) to orthogonal (0-) — these mark the END of the null-space-exploitable region
- **Cross-pair MG boundaries**: where pair_i and pair_{i+1} have radically different MG profiles — these mark hard-pair clusters

### 3. xray (per-primitive)

13 canonical xray primitives per synthesis (`src/tac/xray/registry.py`). The synthesis flagged that wiring them as named features in the SLIM ranker increases orthogonal feature count from ~5 to ~18 (cross-stack synergy #3 — 3.6× orthogonal-feature richness for ZERO additional risk budget).

For EACH of the 13 primitives:
- Document current consumer count (per `tools/cathedral_autopilot_autonomous_loop.py` + sister rankers)
- Identify the highest-EV NEXT consumer (autopilot v2 cascade is the obvious target)
- Quantify the orthogonality vs other primitives (avoid redundant features in SLIM)

### 4. Hard pairs (per-pair-difficulty-tier)

Existing canonical at `tac.master_gradient_consumers.per_pair_difficulty_atlas`. Deeper exploration:

- **What MAKES a pair hard?** Decompose: high MG magnitude / high cos(seg, pose) variance / high per-pair Wyner-Ziv tier divergence / high motion-vector entropy / high SegNet logit-margin variance
- **Hard-pair clusters**: identify regions of the 600-pair space where N+ consecutive pairs are all hard
- **Per-difficulty-tier codec routing**: top-50 hardest pairs get fp16 / next-100 int8 / etc. — analogous to sensitivity_mask_aware_quantizr_v1 but at the PAIR granularity instead of byte granularity
- **Hard-pair × Venn classification intersection**: do hard pairs concentrate in any Venn cell (HIGH_PAIR_INVARIANT / HIGH_PAIR_SPECIFIC)?

### 5. Sensitive bytes (per-sensitivity-tier)

Existing canonical at `tac.empirical_per_x_optimal_codec_planner.per_byte_strategy.sensitivity_mask_aware_quantizr_v1` (top-2% fp16 / next-5% int8 / next-20% int6 / 73% int4). Deeper exploration:

- **What MAKES a byte sensitive?** Decompose into per-pair contribution × per-pixel/region contribution × per-class contribution × per-axis contribution
- **Sensitivity-tier × Catalog #319 v2 Venn classification** intersection: which sensitivity tiers concentrate in HIGH_PAIR_INVARIANT vs HIGH_PAIR_SPECIFIC?
- **Sensitivity gradients along section boundaries**: do sensitivity tiers correlate with archive section boundaries (renderer.bin vs masks.mkv vs poses.pt)?
- **Sensitive-byte × null-space basis**: bytes in the null subspace are by definition LOW-sensitivity; verify empirically + design the codec hierarchy

### 6. Sensitivity map (per-map)

Existing canonical at `src/tac/sensitivity_map/{__init__.py,axis_weights.py,wyner_ziv_reweight.py}`. Deeper exploration:

- **Multi-axis sensitivity maps**: per-axis (seg/pose/rate) sensitivity is partially canonical; what about cross-axis joint distributions?
- **Per-substrate sensitivity maps**: cross-paradigm differences in sensitivity map structure
- **Sensitivity map evolution across training epochs**: does the map converge to a fixed shape post-warmup?
- **Sensitivity map orthogonality matrix** across substrates: which substrates have orthogonal vs aligned sensitivity maps? (informs composition_alpha per Catalog #322)
- **Cathedral autopilot integration**: the synthesis hook #1 sensitivity-map gap (6 DORMANT per-pixel helpers) — wire ALL into autopilot v2 cascade

## CROSS-GRANULARITY INTERSECTIONS (high-EV discovery surfaces)

The richest discovery surfaces are at the INTERSECTIONS:

| Intersection | Example mechanism | Predicted EV |
|---|---|---|
| **boundaries × master-gradient-boundaries** | Targeted reductions at sign-flip points where both signals agree | HIGH |
| **xray × sensitivity-map** | Per-xray-primitive sensitivity weight in autopilot v2 cascade | HIGH |
| **hard-pairs × sensitive-bytes** | Combined per-pair × per-byte sensitivity routing | HIGHEST |
| **hard-pairs × Venn-classification × per-class** | Triple-intersection codec routing (per-pair × per-region × per-class) | EXTREME |
| **sensitive-bytes × null-space × hash-seed** | Bytes in null subspace replaced with PRNG seed expansion | EXTREME (cross-stack #2 + #1 combined) |
| **master-gradient-boundaries × xray** | Use xray primitives to PREDICT MG boundary locations across substrates | MEDIUM-HIGH |

## INSTRUCTION TO IN-FLIGHT SUBAGENT a468f72aa4ca0b50a

If you are this subagent, INCORPORATE the 6 additional granularities + 6 cross-granularity intersections into your DELIVERABLE 2 (`deeper_granularity_discovery_*.md`). The 18-granularity expanded inventory is THE foundation for subsequent orchestration-queue subagents.

## INSTRUCTION TO FUTURE SUBAGENTS

Read this directive at pre-flight scan along with `comprehensive_analytical_surfaces_inventory_plus_synthesis_design_memo_20260518.md`. The 18-granularity expanded landscape supersedes the 15-granularity scope in the synthesis memo.

— Main-Claude (relayed on behalf of operator 2026-05-18)
