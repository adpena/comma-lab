# Evaluator response atlas -> LF waterfiller DISPATCH ORDER wiring — landed 2026-06-09

**Subagent:** `atlas_waterfiller_dispatch_order_20260609` (the #36 -> #46 consumer
wiring the atlas author named "the single most important consumer wiring to do next").
**Evidence grade:** `[macOS-CPU advisory]` / mechanism-only. No score claims; no
dispatch. $0 local, NO cloud, NO paid GPU, NO MPS. PROPOSAL surface — every emitted
row is `promotable=False` / `requires_exact_remeasure=True`.
**Frontier at landing** (pointer-only, per CLAUDE.md "Frontier scores are
pointer-only"): unchanged. This wiring is a *dispatch-order refinement*; it ranks
WHICH PAIRS to coarsen first across the 600-pair video, it does not move the frontier.

## What landed

The 600-pair `EvaluatorResponseAtlas` (#36) cross-video ranking is now wired into the
SNeRV LF payload rate-distortion reverse-waterfill planner (#46) as the **DISPATCH
ORDER**: the targeting layer the per-pixel cone cannot reach. The band×orientation
atlas tells the planner *which section* is sensitive; the #35 cone tells it *which
frame1 pixel inside that section* has free budget; the #36 atlas tells it *which PAIRS
across the 600-pair video* carry the most joint-safe budget — coarsen the high-budget
temporal segments FIRST, protect the fragile clusters LAST.

### The integration surface (new types / fields / flags)

**Core (`src/tac/optimization/lf_payload_rate_distortion.py`) — gated, default-OFF,
backward-compatible (the 62 existing tests pass unchanged):**

- `TemporalSegment` — a contiguous run of pairs (e.g. 437-440). Fail-closed: refuses
  non-contiguous / unsorted / negative pair runs (a temporal segment's run-length mask
  compressibility — its rent advantage — DEPENDS on contiguity). Carries
  `mean_pair_budget` + `is_protect` + `role`.
- `EvaluatorAtlasDispatch` — the typed cross-video dispatch order. `from_atlas(atlas, ...)`
  extracts the contiguous high-budget temporal segments (among the top-k highest
  `pair_budget` pairs) + the contiguous fragile clusters (among the top-k most-fragile
  pairs). **Drops any high-budget pair also flagged fragile** (never coarsen a protected
  pair; the high segment is split to its non-fragile sub-runs). Carries `pair_budget`
  map + `source_path` + `source_sha256` (fail-closed provenance; `/tmp` rejected; empty
  budget refused). `coarsen_pair_mask(segment)` builds the per-pair boolean sidecar.
- `ACTION_QUANTIZE_TEMPORAL_SEGMENT` — new action kind. Coarsens the section's frame1
  payload ONLY on a high-budget temporal segment's pairs.
- `build_temporal_segment_quantize_action(section, estimate, baseline, dispatch, segment,
  step)` — builds the temporal-masked action + `TemporalSegmentActionAccounting`. Returns
  `None` for protect (fragile) segments, frame0-only sections, scope-invalid sections.
- `temporal_mask_coding_cost_bytes(coarsen_pair_mask)` — REAL coding cost of the per-pair
  coarsen mask (bit-pack + brotli q=11, CLAUDE.md L32). The temporal mask pays rent
  exactly like the spatial cone mask.
- `CandidateActionEvaluation` gains `dispatch_order` (0 = coarsen FIRST), `protect_set`
  (True for a fragile cluster, never coarsened), `segment` (the temporal segment) — all
  optional, default None/False (backward-compatible).
- `plan_lf_payload_actions(..., response_atlas=None, temporal_quantize_steps=None)` —
  two new kwargs. When `response_atlas` is supplied: ALSO proposes a temporal-masked
  quantize per frame1-touching section per high-budget segment per step; assigns every
  ranked action a `dispatch_order` (temporal segments first, sorted by the atlas budget
  ranking; remaining ranked actions follow their value-per-byte order); surfaces a
  PROTECT marker per fragile cluster (`protect_set=True`, `delta_bytes=0`, never ranked
  as a coarsen action); emits a `response_atlas_dispatch` provenance block. When `None`,
  the plan is byte-identical to before (`dispatch_order` None, no temporal actions).

**CLI (`tools/snerv_lf_payload_rate_distortion.py`):**

- `--response-atlas <path>` — the 600-pair `evaluator_response_atlas.jsonl` index. The
  CLI loads it via `EvaluatorResponseAtlas.from_jsonl_lines` + `from_atlas`, sha-cites the
  file (fail-closed: missing / empty / invalid index refused; `/tmp` rejected), and
  records `response_atlas_sha256` in the plan `inputs`.
- `--temporal-quantize-steps` — comma-separated steps for the temporal actions (defaults
  to `--quantize-steps`).
- `--atlas-top-k-budget` / `--atlas-top-k-fragile` — how many top pairs to cluster into
  high-budget / fragile segments (default 10, the atlas headline top-10).

### The accounting (THE LAW, extended temporally)

For a frame1-touching, scope-valid section coarsened on a high-budget temporal segment
at quantize step Δ:

    pair_fraction  = segment_length / n_pairs       (temporal share of bytes coarsened)
    budget_discount = mean_segment_budget / max_pair_budget   (in (0, 1])
    advantage      = budget_discount * (1 - MIN_RESIDUAL=0.10)
    dist_weight    = pair_fraction * (1 - advantage)   (< pair_fraction; the discount)
    gross_freed    = section.bytes * pair_fraction * (Δ / (1+Δ))
    temporal_rent  = brotli_q11(bitpack(coarsen_pair_mask))     (the temporal mask pays rent)
    net_freed      = gross_freed - temporal_rent
    delta_bytes    = -net_freed                  (>= 0, ADDS bytes, when rent > gross)
    est_delta_d_seg  = section_seg_value  * dist_weight * (1 - value_kept)
    est_delta_d_pose = section_pose_value * dist_weight * (1 - value_kept)

Then the unmodified THE LAW is applied — ONE currency with the downstream exact
waterfiller. A temporal mask whose run-length coding cost exceeds the bytes it frees
yields `net_freed <= 0` -> `delta_bytes >= 0` -> THE LAW rejects it (the prompt's "a mask
whose bytes exceed its savings is rejected"). The high-budget segments are BY ATLAS
CONSTRUCTION the low-sensitivity pairs (large usable cone budget), so coarsening them
gives up DISPROPORTIONATELY less distortion per byte than a whole-video coarsen of the
same pair-count fraction. A low-budget segment gets discount ~0 -> `dist_weight ~
pair_fraction` (no claimed advantage — the honest fail-safe).

## The known-optimum (the falsifiable NO-FAKE proof)

`test_temporal_segment_beats_whole_video_value_per_byte`: a synthetic dispatch mirroring
the REAL clustering (high-budget run 437-440, max-budget ceiling pair 442). On a 1 MB
LF section at Δ=1:

- **WHOLE-VIDEO** quantize: value_per_byte 4.46e-7 (gives up the full section value over
  500,000 bytes freed).
- **TEMPORAL-SEGMENT** (437-440, 0.667% of the video): value_per_byte 6.27e-7 — a
  **1.41× higher value-per-byte** — because the budget discount (0.916) shrinks the
  distortion weight from pair_fraction 0.667% down to 0.117%, while a 16-byte run-length
  temporal mask rent is trivial vs the 3,333 gross bytes freed.

Every test would FAIL if the builder returned canonical constants instead of computing
the temporal-mask byte math + budget-weighted distortion (Slot EEE Class 2). The
`test_temporal_mask_coding_cost_*` pair proves the mask cost MEASURES the mask
(contiguous run is cheap, scattered selection is dear).

## Tests

- **NEW: 24 behavioral tests** in `src/tac/tests/test_lf_payload_rate_distortion.py`
  (TemporalSegment fail-closed contiguity/sorting/negative; dispatch fail-closed empty
  budget / `/tmp` / nonpositive n_pairs; temporal mask coding cost contiguous-cheap vs
  scattered-dear; temporal-beats-whole-video known-optimum; budget-discount weighting;
  mask-rent-rejection; net = gross − rent; builder None for protect / frame0-only;
  planner emits temporal actions + dispatch provenance; dispatch_order temporal-first +
  contiguous 0-based; fragile protect markers last + 0-byte + not-ranked; backward-compat
  OFF; false-authority contract on temporal rows; from_atlas against the REAL 600-pair
  index extracting the real 437-440 budget + 517-519 fragile clusters; CLI --response-atlas
  end-to-end against the real index + fail-closed on missing/empty index).
- **62 existing tests: GREEN, unchanged** (backward-compatible — `response_atlas` defaults
  to `None`).
- **Total: 86 passed.** Ruff clean on all 3 touched files.

## End-to-end smoke (REAL artifacts, $0)

CLI run against the REAL `evaluator_response_atlas.jsonl` (600-pair index, #36 output) +
the REAL `scorer_spectral_atlas_fast_20260609` atlas + a synthetic G1b + an in-scope
section-map (LF payload placed at the lowest-sensitivity REAL measured cell: band 3,
yuv/y, isotropic, frame1_only, amp 1.0). Result:

- `n_temporal_segment_actions = 21` (7 high-budget segments × 3 steps), `n_protect = 6`
  (the 6 real fragile clusters), `n_ranked = 25`.
- **best action (dispatch_order 0) = `lf_payload_bytes::quantize_temporal_segment_442-442_delta=2`**
  — pair 442 is the single highest-budget pair in the real atlas (196,569 budget); with
  budget_discount 1.0 the distortion weight is scaled from pair_fraction 0.00167 to
  0.000167 (the full advantage), freeing 1,097 net bytes after a 14-byte run-length
  temporal mask rent.
- The high-budget segments extracted from the real atlas: 442, 426, 577, 437-440, 579,
  145, 532 (matches the atlas memo headline top-10 budget pairs, contiguous-clustered).
- The fragile (protect) segments: 133, 177-178, 510, 514-515, 517-519, 522 (matches the
  atlas memo top-10 fragile pairs).

Smoke artifacts (durable SSD, rebuildable; NOT evidence — rebuild from the CLI):
`/Volumes/VertigoDataTier/pact/atlas_waterfiller_dispatch_smoke_20260610/` (g1b.json +
section_map*.json + plan*.json). The plan files are the dispatch-order proposal; every
row is `requires_exact_remeasure=True`.

## READY note for Branch-B round-3 (the composed temporal+spatial rung)

Branch-B's round-1 ran UNIFORM rate-attack rungs (quantize/drop the whole LF payload);
the #35 wiring added the round-2 SPATIALLY-MASKED rung (cone-masked Δ on frame1 pixels).
The NEXT round (round-3) should run the COMPOSED TEMPORAL × SPATIAL rung this wiring
enables:

> **Predicted best composed rung (Branch-B round-3):**
> `quantize_temporal_segment` on the LF wavelet payload, restricted to the atlas
> HIGH-BUDGET temporal segments only (`437-440`, `577-579`, `426`, `442`, `579`, `145`,
> `532` — dispatched in atlas budget order), AND cone-masked WITHIN those pairs (Δ on
> the cone-FREE frame1 pixels only, full precision on the fragile set). The atlas
> selects WHICH PAIRS; the cone selects WHICH PIXELS within them; both masks pay
> run-length / brotli rent.
>
> **Falsifiable prediction:** at a fixed coarsen step, the composed temporal+spatial rung
> frees `pair_fraction × free_pixel_fraction` of the bytes a uniform whole-video rung
> frees, but gives up only `budget_discount-scaled × free_set_sensitivity_share` of the
> distortion — a compounded distortion-per-byte advantage over BOTH the uniform rung
> AND the spatial-only rung. On the REAL inputs: pair 442 budget_discount 1.0 (full 90%
> temporal advantage) × the cone's pair-0 free-set sensitivity share 0.40 (the spatial
> advantage) -> the composed rung's value_per_byte should exceed the spatial-only rung's
> by the temporal budget discount factor (~1.4× on the highest-budget segment).
>
> **The disconfirming outcome:** if the exact receiver re-measure shows the high-budget
> segments' d_seg/d_pose is NOT lower than `budget_discount × uniform_distortion` (i.e.
> the atlas's high-budget pairs are NOT actually low-sensitivity on the receiver), the
> dispatch-order advantage is falsified at the receiver and the temporal rung collapses
> to the uniform rung minus the temporal mask rent (strictly worse). Run the COMPOSED
> rung AFTER the uniform + spatial-only rungs so the exact ΔS of all three is on the same
> base archive (`requires_recompute_after_accept`). The atlas pose_binds_fraction 0.7286
> says the POSE budget binds for ~73% of pixels video-wide, so the high-budget segments'
> advantage should manifest most on the pose axis (the marginal-value flip).

This is a PROPOSAL: every temporal rung is `requires_exact_remeasure=True`. The downstream
authority is `evaluator_action_waterfill.CandidateActionEvaluation` (exact measured
d_seg/d_pose/bytes vs the base archive). Do NOT touch the in-flight Branch-B run dir
(`snerv_branch_b_rate_attack_*`) — this rung is for its NEXT round.

## 6-hook wire-in (Catalog #125)

1. **sensitivity-map** — ACTIVE. The atlas `pair_budget` per pair IS the per-pair
   cross-video sensitivity index the dispatch order consumes.
2. **Pareto constraint** — ACTIVE. THE LAW (unchanged) is the rate/distortion admission
   predicate; the atlas refines its distortion term temporally (budget discount).
3. **bit-allocator hook** — ACTIVE (PRIMARY). The temporal-segment quantize IS a
   cross-video bit allocator: it spends bytes on the pairs the atlas says are free.
4. **cathedral autopilot dispatch** — N/A. PROPOSAL surface; the downstream
   `evaluator_action_waterfill` row is the dispatch-eligible surface once measured.
5. **continual-learning posterior** — N/A. No empirical anchor promoted (advisory,
   non-promotable); the plan is recomputed per archive + atlas.
6. **probe-disambiguator** — ACTIVE. `dispatch_order` + `protect_set` +
   `budget_discount` disambiguate a temporally-targeted action from a whole-video /
   spatial-only / protect action.

## Per-layer canonical-vs-unique decision (Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| THE LAW / score-unit primitives | ADOPT_CANONICAL | one currency with the exact waterfiller; never re-derived |
| atlas JSONL schema + query surface | ADOPT_CANONICAL (#36) | `EvaluatorResponseAtlas.from_jsonl_lines` + `top_budget_pairs` / `most_fragile_pairs`; never re-authored |
| temporal mask coding cost | ADOPT_CANONICAL (brotli q=11, CLAUDE.md L32) | the canonical sidecar coder; sister of the cone mask rent |
| `TemporalSegment` + `EvaluatorAtlasDispatch` | FORK_PRINCIPLED | genuinely new: the contiguous-run typed dispatch input |
| budget-discount distortion weight | FORK_PRINCIPLED | genuinely new: the dispatch-order advantage made numeric |
| temporal action builder + protect markers | FORK_PRINCIPLED | genuinely new: temporal mask + budget-weighted accounting + protect set |

## Files

- Core: `src/tac/optimization/lf_payload_rate_distortion.py` (extended; +~430 lines).
- CLI: `tools/snerv_lf_payload_rate_distortion.py` (extended; 4 new flags + loader).
- Tests: `src/tac/tests/test_lf_payload_rate_distortion.py` (62 -> 86 tests).
- This memo.

## Reproduce

```bash
PYTHONPATH=src:upstream .venv/bin/python -m pytest \
    src/tac/tests/test_lf_payload_rate_distortion.py -q   # 86 passed

PYTHONPATH=src:upstream .venv/bin/python tools/snerv_lf_payload_rate_distortion.py \
    --g1b-verdict <g1b.json> --atlas <scorer_spectral_sensitivity.v2.json> \
    --section-map <section_map.json> \
    --frame1-cone-map /Volumes/VertigoDataTier/pact/frame1_joint_safe_cone_*/frame1_joint_safe_cone_summary.json \
    --response-atlas /Volumes/VertigoDataTier/pact/evaluator_response_atlas_20260610T001515Z/evaluator_response_atlas.jsonl \
    --output <plan.json>
```
