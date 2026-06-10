# Frame1 cone -> LF payload waterfiller wiring — landed 2026-06-09

**Subagent:** `frame1_cone_waterfiller_wiring_20260609` (the #35 -> #46 consumer
wiring its author named "the single most important consumer wiring to do next").
**Evidence grade:** `[macOS-CPU advisory]` / mechanism-only. No score claims; no
dispatch. $0 local, NO cloud, NO paid GPU. PROPOSAL surface — every emitted row is
`promotable=False` / `requires_exact_remeasure=True`.
**Frontier at landing** (pointer-only, per CLAUDE.md "Frontier scores are
pointer-only"): unchanged. This wiring is a *budget surface refinement*; it ranks
what to re-measure first, it does not move the frontier.

## What landed

The frame1 JOINT SAFE CONE (#35) is now wired into the SNeRV LF payload
rate-distortion reverse-waterfill planner (#46) as the per-frame1-pixel SPATIAL
coarsening budget. The band×orientation atlas (`scorer_spectral_sensitivity.v2`)
tells the planner *which section* is sensitive; the cone tells it *which frame1
pixel inside that section* has free budget — the resolution the band cells cannot
reach.

### The integration surface (new fields/flags)

**Core (`src/tac/optimization/lf_payload_rate_distortion.py`) — gated, default-OFF,
backward-compatible (the 37 existing tests pass unchanged):**

- `Frame1ConeMap` — typed wrapper over the #35 cone arrays. Fields:
  `joint_cone_radius` (H,W), `fragile_cone_mask` (H,W bool), optional
  `joint_sensitivity` (H,W), `fragile_radius_threshold`, `source_path`, `axis_tag`.
  Properties: `free_pixel_fraction`, `fragile_pixel_fraction`, `n_free_pixels`,
  `free_mask()`, and the load-bearing `free_set_sensitivity_share` (the fraction of
  total joint sensitivity that lives on the FREE pixels — the cone's structural
  claim made numeric). `from_npz(...)` reads the EXACT `cone_pair_*.npz` arrays the
  #35 CLI writes (`joint_cone_radius` + `fragile_cone_mask` + `joint_sensitivity`);
  the schema was grepped from `tools/build_frame1_joint_safe_cone.py`, never invented.
  **Fail-closed**: an all-zero radius (#35 "gradient not reachable" / empty-cone
  signature) is REFUSED — a non-reachable cone can never silently produce an
  all-permissive everything-free plan. `/tmp` paths rejected.
- `ACTION_QUANTIZE_CONE_MASKED` — new action kind. Coarsens the section's
  coefficients ONLY at cone-FREE frame1 pixels (radius >= threshold), preserving full
  precision on the fragile set.
- `build_cone_masked_quantize_action(section, estimate, baseline, cone, step)` —
  builds the masked action + a `ConeMaskedActionAccounting` audit struct. Returns
  `None` for frame0-only sections (no frame1 constraint), scope-invalid sections
  (fail-closed; routed to needs_exact_remeasure by the plain builders), and
  all-fragile cones (no free pixel to coarsen).
- `estimate_mask_coding_cost_bytes(free_mask)` — REAL coding cost of the per-pixel
  keep/coarsen mask: bit-pack + brotli q=11 (CLAUDE.md L32). The mask MUST pay rent.
- `ConeMaskedActionAccounting` — per-action audit row: `free_pixel_fraction`,
  `gross_bytes_freed`, `mask_coding_cost_bytes`, `net_bytes_freed`,
  `value_kept_fraction`, `distortion_weight`, `used_sensitivity_share`.
- `plan_lf_payload_actions(..., frame1_cone_map=None, cone_quantize_steps=None)` —
  two new kwargs. When `frame1_cone_map` is supplied, the planner ALSO proposes a
  masked quantize per frame1-touching section at each `cone_quantize_steps` step
  (defaults to `quantize_steps`) and emits a `frame1_cone` provenance block
  (free fractions, threshold, per-action accounting). When `None`, the plan is
  byte-identical to before.

**CLI (`tools/snerv_lf_payload_rate_distortion.py`):**

- `--frame1-cone-map <path>` — accepts EITHER a `cone_pair_*.npz` OR a
  `frame1_joint_safe_cone_summary.json` (first `map_manifest` entry loaded). `/tmp`
  rejected.
- `--cone-quantize-steps` — comma-separated steps for the masked actions.
- `--cone-fragile-radius-threshold` — override the free/fragile threshold.

### The accounting (THE LAW, extended spatially)

For a frame1-touching, scope-valid section at quantize step Δ:

    f              = cone free-pixel fraction (spatial share of coarsenable coeffs)
    share          = cone free-set SENSITIVITY share (< f when sensitivity concentrates
                     on the preserved fragile set — the cone's structural advantage)
    gross_freed    = section.bytes * f * (Δ / (1+Δ))
    mask_rent      = brotli_q11(bitpack(free_mask))         # the mask pays rent
    net_freed      = gross_freed - mask_rent
    delta_bytes    = -net_freed                              # >= 0 (ADDS bytes) when
                                                            # mask_rent > gross_freed
    est_delta_d_seg  = section_seg_value  * share * (1 - value_kept)
    est_delta_d_pose = section_pose_value * share * (1 - value_kept)

Then the unmodified THE LAW (`keep iff -ΔS_distortion > 25·Δbytes/37,545,489`) is
applied to `delta_bytes` + the cone-weighted distortion — ONE currency with the
downstream exact waterfiller. A mask whose coding cost exceeds the bytes it frees
yields `net_freed <= 0` -> `delta_bytes >= 0` -> THE LAW rejects it (the prompt's
"a mask whose bytes exceed its savings is rejected"). When no `joint_sensitivity`
map is present, `share` falls back to the conservative pixel-count fraction `f`
(no claimed advantage — the fail-safe default that never over-admits).

## The known-optimum (the falsifiable NO-FAKE proof)

`test_cone_masked_pays_rent_where_unmasked_does_not`: a synthetic section whose
distortion sensitivity concentrates on a small (10%) fragile band. At B=300,000:

- **UNMASKED** quantize gives up the FULL section value -> ΔS_total +0.010 ->
  does NOT pay rent.
- **CONE-MASKED** quantize gives up only the free set's ~0.88% sensitivity share ->
  ΔS_total -0.089 -> PAYS rent.

The cone unlocks a rent-paying action the band-only planner could not see.
`test_cone_masked_value_per_byte_beats_unmasked` confirms ~7× higher
value-per-byte when both pay. Every test would FAIL if the builder returned
canonical constants instead of computing the byte+distortion math (Slot EEE Class 2).

## Tests

- **NEW: 25 behavioral tests** in `src/tac/tests/test_lf_payload_rate_distortion.py`
  (cone construction + properties; sensitivity-share < pixel-fraction; fail-closed
  all-zero / /tmp / shape / non-2D; mask coding cost coherent-cheap vs salt-and-pepper-
  expensive; masked-pays-where-unmasked-does-not known-optimum; value-per-byte beats
  unmasked; distortion-weight == sensitivity-share; mask-rent-rejection; net = gross −
  rent; frame0-only skip; scope-invalid fail-closed; no-free-pixel None; pixel-fraction
  fallback without sensitivity map; planner backward-compat OFF; planner emits masked +
  provenance; masked outranks unmasked; false-authority contract on masked rows;
  npz round-trip against the EXACT #35 schema; npz wrong-schema + /tmp rejection).
- **37 existing tests: GREEN, unchanged** (backward-compatible — `frame1_cone_map`
  defaults to `None`).
- **Total: 62 passed.** Ruff clean on all 3 touched files.

## End-to-end smoke (REAL artifacts, $0)

CLI run against the REAL `cone_pair_00000.npz` (#35 output) + REAL
`scorer_spectral_atlas_fast_20260609` atlas + a synthetic G1b + an in-scope
section-map (LF payload at band 5 = high-freq / low-sensitivity, frame1_only/y/
vertical/amp 1.0). Result: best action = `lf_payload_bytes::quantize_cone_masked_
delta=2`; real mask rent 13,928 B subtracted from 72,271 B gross -> 58,343 B net
freed; `distortion_weight = 0.403` < `free_pixel_fraction = 0.542`. The cone's
advantage holds on real data (1.345× — the real free 54% of frame1 pixels carry
only 40% of the joint sensitivity).

## READY note for Branch-B round-2 (the spatially-masked rung)

Branch-B's current ladder runs UNIFORM rate-attack rungs (quantize/drop the whole
LF payload). The NEXT round should run the SPATIALLY-MASKED rung this wiring adds:

> **Predicted best spatially-masked rung (Branch-B round-2):**
> `quantize_cone_masked_delta=2` on the LF wavelet payload, masked by the #35 frame1
> joint safe cone (`frame1_joint_safe_cone_20260609T235339Z`, pair-aggregated). On
> the REAL cone (pair 0): free-pixel fraction 0.542, free-set sensitivity share
> 0.403, real mask rent ~13.9 KB.
>
> **Falsifiable prediction:** at a fixed coarsen step, the cone-masked rung frees
> ~`f`=54% of the bytes the uniform rung frees, but gives up only ~`share`=40% of
> the distortion the uniform rung gives up (a 1.345× distortion-per-byte advantage
> after the ~13.9 KB mask rent is paid). The uniform rung's exact ΔS will therefore
> be MORE negative on the rate axis (it frees more bytes) but the masked rung's exact
> ΔS-per-byte (value_per_byte) will be HIGHER. The disconfirming outcome: if the
> exact receiver re-measure shows the masked rung's d_seg/d_pose is NOT lower than
> `share × uniform_distortion` (i.e. the cone's free pixels are NOT actually
> low-sensitivity on the receiver), the cone advantage is falsified at the
> receiver and the masked rung collapses to the uniform rung minus the mask rent
> (strictly worse). Run the masked rung AFTER the uniform rungs so the exact ΔS of
> both is on the same base archive (the planner's `requires_recompute_after_accept`).

This is a PROPOSAL: every masked rung is `requires_exact_remeasure=True`. The
downstream authority is `evaluator_action_waterfill.CandidateActionEvaluation`
(exact measured d_seg/d_pose/bytes vs the base archive). Do NOT touch the in-flight
`snerv_branch_b_rate_attack_20260609T230000Z` run dir — this rung is for its NEXT
round.

## 6-hook wire-in (Catalog #125)

1. **sensitivity-map** — ACTIVE. The cone's `joint_sensitivity` (P18/P19 coupling)
   IS the per-frame1-pixel sensitivity map the masked distortion weight consumes.
2. **Pareto constraint** — ACTIVE. THE LAW (unchanged) is the rate/distortion
   admission predicate; the cone refines its distortion term spatially.
3. **bit-allocator hook** — ACTIVE (PRIMARY). The masked quantize IS a spatially-
   resolved bit allocator: it spends bytes where the cone says frame1 has free budget.
4. **cathedral autopilot dispatch** — N/A. PROPOSAL surface; the downstream
   `evaluator_action_waterfill` row is the dispatch-eligible surface once measured.
5. **continual-learning posterior** — N/A. No empirical anchor promoted (advisory,
   non-promotable); the plan is recomputed per archive.
6. **probe-disambiguator** — ACTIVE. `used_sensitivity_share` + `atlas_scope_valid`
   disambiguate a cone-weighted estimate from a pixel-count fallback / scope refusal.

## Per-layer canonical-vs-unique decision (Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| THE LAW / score-unit primitives | ADOPT_CANONICAL | one currency with the exact waterfiller; never re-derived |
| cone array schema | ADOPT_CANONICAL (#35 .npz) | grepped from the #35 CLI; never invented |
| mask coding cost | ADOPT_CANONICAL (brotli q=11, CLAUDE.md L32) | the canonical sidecar coder |
| `Frame1ConeMap` wrapper | FORK_PRINCIPLED | genuinely new: the typed spatial-budget input |
| sensitivity-share distortion weight | FORK_PRINCIPLED | genuinely new: the cone's structural advantage made numeric |
| masked action builder | FORK_PRINCIPLED | genuinely new: spatial mask + mask-rent accounting |

## Files

- Core: `src/tac/optimization/lf_payload_rate_distortion.py` (extended; +~350 lines).
- CLI: `tools/snerv_lf_payload_rate_distortion.py` (extended; 3 new flags).
- Tests: `src/tac/tests/test_lf_payload_rate_distortion.py` (37 -> 62 tests).
- This memo.

## Reproduce

```bash
PYTHONPATH=src:upstream .venv/bin/python -m pytest \
    src/tac/tests/test_lf_payload_rate_distortion.py -q   # 62 passed

PYTHONPATH=src:upstream .venv/bin/python tools/snerv_lf_payload_rate_distortion.py \
    --g1b-verdict <g1b.json> --atlas <scorer_spectral_sensitivity.v2.json> \
    --section-map <section_map.json> \
    --frame1-cone-map /Volumes/VertigoDataTier/pact/frame1_joint_safe_cone_20260609T235339Z/frame1_joint_safe_cone_summary.json \
    --output <plan.json>
```
