# UNIWARD cost-map → bit_allocator → NSCS06 v8 chroma wire-in — HALT-CLEAN with real structural blocker (RESUME #2)

- **Lane:** `lane_uniward_cost_map_to_bit_allocator_to_nscs06_v8_wire_in_20260530`
- **Status:** HALT-CLEAN (the operator's explicitly-named correct outcome, not a fake landing)
- **Date:** 2026-05-31
- **Resumed-from-predecessor-checkpoint:** YES — predecessor `uniward_cost_map_bit_allocator_nscs06_v8_wire_in_20260530` (pid 53019) HALTed CLEAN at `status=blocked` 2026-05-31T02:27Z with the corrected real-API map in its checkpoint `next_action`. This resume re-grepped every API at HEAD per the operator's search-before-build directive and reached an INDEPENDENT structural-infeasibility verdict that supersedes the predecessor's "harness instability" blocker with a HARD structural blocker.
- **Mission contribution (Catalog #300):** `apparatus_maintenance` (preserves canonical posterior + lane registry from a phantom-API fake landing per CLAUDE.md "NO FAKE IMPLEMENTATIONS" Slot EEE 5 classes + "Forbidden premature KILL"; this is DEFER-pending-substrate-redesign, NOT a kill).
- **Horizon class (Catalog #309):** `plateau_adjacent` (NSCS06 v8 chroma path; not asymptotic).
- **Paid spend:** $0 (numpy/CPU inspection only; zero GPU dispatch).

## Verified-at-HEAD API map (grepped facts only — zero assertions from the prompt)

> **Harness note:** `git` and `grep` are rtk-proxied in this environment and return
> mangled/empty output (`git rev-parse HEAD` returned the git-dir; `grep` returned
> nothing before the sentinel). ALL verification below was done via `.venv/bin/python` /
> `python3` file-read + `pathlib.glob` (Python is not rtk-proxied), then Read tool on
> the captured-to-file output. This matches the predecessor's "output-capture
> instability" note but I worked around it deterministically rather than HALTing on it.

### CONFIRMED REAL (matches predecessor map)

1. **`pack_archive`** in `src/tac/substrates/nscs06_carmack_hotz_strip_everything/archive.py:91`
   is **keyword-only**:
   ```python
   def pack_archive(
       *,
       palette: "GrayscalePalette",
       chroma_rgb: np.ndarray,          # (5, 3) uint8 per-class chroma anchors
       cdf: "ClassConditionalCDF",
       cls_indices: np.ndarray,         # (H, W) uint8 per-pixel class indices
       seg_residual: np.ndarray | None = None,
   ) -> bytes:
   ```
   **There is NO `chroma_weight_map=`, NO `use_uniward_weighting=`, NO bit-budget / quantization param.** Predecessor map ✓ confirmed.

2. **The chroma section is a FIXED 15 raw bytes.** `archive.py:72 _encode_chroma_anchors`
   returns `np.asarray(chroma_rgb, dtype=np.uint8).tobytes()` = `5 × 3 × 1 = 15 bytes`,
   **uncompressed, unquantized, no length-variable encoding.** This is the load-bearing
   fact for the infeasibility verdict (see below).

3. **`build_chroma_palette`** in `palette.py:69` is
   `build_chroma_palette(rgb_pairs, class_labels, *, num_classes=5) -> (num_classes, 3) uint8`
   = per-class mean-RGB anchor. **5 anchors, NOT a 64-entry per-index LUT.** Predecessor map ✓ confirmed.
   (Re-exported from `archive.py:244` for callers.)

### PHANTOM AT HEAD (do NOT exist — these break the wire-in)

4. **`src/tac/bit_allocator/` does NOT exist.** `allocate_per_byte` / `PerByteAllocationPlan` /
   `tac.bit_allocator.per_byte` are **phantom at HEAD** (Python `pathlib` existence check +
   exhaustive `def allocate_per_byte` content-grep across all `src/**/*.py` → 0 matches;
   `PerByteAllocationPlan` → 0 matches; `tac.bit_allocator` import → 0 matches).
   **The operator's warning was correct:** the predecessor mapped this as "VERIFIED real",
   but it is NOT present at HEAD. (Either it was reverted with the predecessor's phantom
   artifacts, or it was never committed.) The prompt's assertion that `allocate_per_byte`
   is "VERIFIED REAL" is itself phantom at HEAD.

5. **`src/tac/composition/` does NOT exist** — my primary scoped surface is phantom
   (`tac.composition` import → 0 matches across all `src/**/*.py`).

6. **The 87bd1c355 UNIWARD integration module does NOT exist at HEAD** —
   `weight_map_per_lut_index.py`, `lut_derivation_uniward_weighted.py`,
   `nscs06_v8_chroma_lut_integration`, `src/tac/substrates/uniward_per_pixel_distortion/`
   all → 0 glob matches.

### THE ONLY REAL UNIWARD CODE (sister-owned, consume-not-modify)

7. `src/tac/substrates/pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1/uniward_cost.py`
   (187 lines) exposes:
   - `compute_uniward_cost_map(frame, *, sigma=1.0, eps=1e-8) -> (H,W) float64` (LOW=texture/blind, HIGH=smooth/sensitive)
   - `aggregate_cost_per_class(cost_map, cls_indices, *, num_classes=5) -> (5,) float64`
   - `cost_to_sensitivity(per_class_cost) -> (5,) float64` (normalized to sum 1)
   Its own module footer (lines 134-138) declares: *"owned by lane_pr110_opt7_\* and is
   consumed (NOT modified) by downstream bit-allocation wire-ins per CLAUDE.md
   UNIQUE-AND-COMPLETE-PER-METHOD."* This lane is **in-flight** (`pr110_opt7_l1_promotion_via_yousfi_t1`,
   pid in checkpoint store, status=in_progress) → building against it now would collide
   per Catalog #302/#340 sister-subagent ownership.

## What predecessor mapped vs what I confirmed

| Item | Predecessor's map | My re-grep at HEAD | Verdict |
|---|---|---|---|
| `pack_archive` keyword-only, no UNIWARD param | correct | confirmed | ✓ |
| chroma = 5 per-class RGB anchors | correct | confirmed `(5,3)` | ✓ |
| `build_chroma_palette` 5-anchor | correct | confirmed | ✓ |
| `allocate_per_byte` "VERIFIED real" | mapped REAL | **PHANTOM at HEAD** (0 matches) | ✗ predecessor map was wrong on this one |
| 87bd1c355 module "EXISTS, locate it" | said EXISTS | **PHANTOM at HEAD** (0 glob hits) | ✗ |
| chroma section byte count | not measured | **15 fixed raw bytes** | NEW (the load-bearing infeasibility fact) |

The predecessor correctly caught the `pack_archive` / 5-anchor phantoms but its own asserted
`allocate_per_byte`-is-real claim was itself imprecise at HEAD — vindicating the operator's
directive to re-grep EVERY API including the predecessor's verified map.

## The real structural blocker (the operator's explicitly-named HALT-CLEAN criterion)

The operator wrote verbatim: *"If after re-grep the wire-in is genuinely infeasible (e.g.
the per-class 5-anchor chroma has too few bytes for UNIWARD weighting to matter), HALT-CLEAN
like the predecessor and document the real blocker — that is the CORRECT outcome, not a fake
landing."*

**This is exactly that case, on two independent axes:**

**Axis A — the bit-allocator does not exist at HEAD.** A UNIWARD-weighted bit-allocation
wire-in requires `allocate_per_byte` (sensitivity → bits) + a `tac.composition` surface to
host it. Both are phantom at HEAD. Building them from scratch would be a NEW substrate
(`lane_class=substrate_engineering`), not a "wire-in", and would duplicate the sister
`pr110_opt7` lane's UNIWARD primitives.

**Axis B — the wire-in TARGET has nothing to allocate.** Even if the allocator existed, a
bit-allocator distributes a *variable* bit budget across bytes weighted by sensitivity. The
NSCS06 v8 chroma section is **15 fixed uint8 bytes** (5 RGB triples at full 8-bit precision)
with **no quantization step and no length-variable encoding**. There is no variable budget to
distribute and no lower-precision option to spend on scorer-blind classes. UNIWARD weighting
("spend bits where the scorer is BLIND, save where SENSITIVE") cannot bias an allocation that
does not exist — every chroma byte is already at fixed full precision. This is precisely the
"too few bytes for UNIWARD weighting to matter" structural mismatch the operator named.

A faithful UNIWARD-weighted chroma path would require **first** redesigning the NSCS06 v8
chroma grammar to support per-class variable-bit quantization (e.g. 8/6/4-bit chroma anchors
selected per-class by scorer sensitivity), which is a substrate-engineering redesign of
`archive.py` + `inflate.py` — explicitly OUT of this lane's consume-don't-modify scope and a
council-grade design decision per CLAUDE.md "Design decisions — non-negotiable" + Catalog #325
per-substrate symposium.

## Reactivation criteria (per CLAUDE.md "Forbidden premature KILL" — this is DEFER, not KILL)

The paradigm (UNIWARD-cost-map biases chroma bit-allocation toward scorer-blind classes) is
INTACT per Catalog #307 (this is an IMPLEMENTATION-INFEASIBLE-AT-HEAD blocker, not a paradigm
falsification). Reactivate when ALL of:

1. **A real `tac.bit_allocator.allocate_per_byte` (sensitivity → bits) lands** with the
   signature the prompt asserts (`sensitivity, *, total_bits, method, min_bits, max_bits) ->
   PerByteAllocationPlan(.bits_per_byte)`), OR an equivalent canonical per-class bit-allocation
   helper is committed to HEAD; AND
2. **The NSCS06 v8 chroma grammar is redesigned** (via per-substrate symposium per Catalog
   #325) to support per-class variable-bit chroma quantization — i.e. there is a VARIABLE bit
   budget for UNIWARD weighting to bias; AND
3. **The sister `pr110_opt7` lane lands** (so `uniward_cost.py` is stable and its
   consume-not-modify contract is honored without collision per Catalog #302/#340), OR the
   UNIWARD cost primitives are promoted to a shared canonical surface both lanes consume.

Re-route would then be: `compute_uniward_cost_map` (real, sister) → `aggregate_cost_per_class`
→ `cost_to_sensitivity` → `allocate_per_byte` (must-land) → per-class variable-bit chroma
quantization in a redesigned `pack_archive` (must-redesign).

## 6-hook wire-in declaration (Catalog #125)

No code landed, so no hooks fire. Declared for completeness of the DEFER record:
- hook #1 sensitivity-map = N/A-AT-HALT (the cost-map WOULD be the sensitivity source once the allocator + variable-bit grammar exist)
- hook #2 Pareto constraint = N/A-AT-HALT
- hook #3 bit-allocator = N/A-AT-HALT (the allocator itself is the missing dependency)
- hook #4 cathedral autopilot dispatch = N/A-AT-HALT
- hook #5 continual-learning posterior = ACTIVE (this DEFER + reactivation criteria become the canonical posterior anchor for the wire-in via the probe-outcomes ledger per Catalog #313)
- hook #6 probe-disambiguator = ACTIVE (this memo IS the disambiguator between "wire-in feasible" vs "structurally infeasible at HEAD pending allocator + variable-bit grammar")

## NO FAKE IMPLEMENTATIONS verification (Slot EEE 5 classes)

- Class 1 (returns-canonical-markers-without-doing-work): N/A — nothing returned; HALT-CLEAN.
- Class 2 (tests-verify-constants): N/A — no tests landed (would have over-fit on phantom APIs).
- Class 3 (synthetic-fixture-instead-of-real-input): AVOIDED — refused to build a wire-in against phantom `allocate_per_byte` / `tac.composition` / 87bd1c355 module.
- Class 4 (placeholder-string-in-canonical-field): AVOIDED — this memo records the REAL blocker, no `pending_ratification`/`TBD` in canonical fields.
- Class 5 (enum-padding): N/A.

The CORRECT outcome per the operator's directive: HALT-CLEAN with a documented real
structural blocker rather than fabricating a wire-in against three phantom dependencies and a
15-byte fixed-precision target with no allocation surface.

## Working-tree state

ALL_CLEAN — no source files created or modified (only this `.omx/research/` memo +
lane registry + probe outcome + checkpoint). No phantom-API artifacts. Sister-DISJOINT vs
DreamerV3 (`dreamer_v3_rssm`), Z7 (`z7_mamba2`), STC/Selfcomp (`lane_stc`), and the
sister-owned `pr110_opt7` UNIWARD lane per Catalog #302/#340.
