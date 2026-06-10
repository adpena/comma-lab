# EVALUATOR INVISIBILITY BASIS — landed 2026-06-10

**Subagent:** `evaluator_null_space_compiler_20260609` (task #47).
**Evidence grade:** TIER 1 = `mathematical-derivation` (hardware-independent,
exact); TIER 2 = `[macOS-CPU advisory]` (the #36 atlas's scorer forwards).
Mechanism-only. No score claims; no dispatch; `promotable=false`. $0 local, NO
cloud, NO paid GPU, NO MPS.
**Frontier at landing** (orphan inventory `da62505aa`): contest-CPU
**0.19198533** (archive `b7106c9b…`, 178,493 B). The basis does not change the
frontier; it is a *certified free-byte / free-perturbation budget surface*
downstream actuators (#46 waterfiller, PR110++ atom generator) consume.

## What landed — the EVALUATOR NULL-SPACE COMPILER

PR110 and the CPU frontier stumbled into free bytes by trial. This task
FORMALIZES what they found: the joint scorer map
`M = (SegNet∘resize∘slice_frame1) ⊕ (PoseNet∘yuv6∘resize)` has a CERTIFIED
INVISIBILITY BASIS — a closed-form set of camera-pixel perturbations that
produce a BIT-IDENTICAL scorer input (residual == 0.0 exactly), at amplitude
unlimited up to uint8 clipping. Those directions are certified free bytes /
free perturbation room.

The structural key (`upstream/modules.py:73` PoseNet, `:109` SegNet via
`upstream/frame_utils.py:11/13`): BOTH scorer heads share an identical FIRST
preprocessing step — a fixed bilinear
`F.interpolate((874,1164) → (384,512), align_corners=False)`. That resize is a
fixed LOW-RANK LINEAR projection `R`; its null space is, by construction, the
set of camera-pixel perturbations invisible to BOTH heads.

### Two TIERS (kept structurally SEPARATE per Catalog #385: measured ≠ certified)

**TIER 1 — CERTIFIED EXACT** (`mathematical-derivation`):
- **1a. Resize zero-weight pixels** (both heads, camera-pixel domain) — the input
  rows/cols the downsample DROPS entirely (total interpolate weight EXACTLY 0).
  Single-pixel, axis-aligned, single-channel, amplitude-unlimited up to clipping.
  **The robustly-certifiable primary basis.**
- **1b. Resize full null space** (both heads) — the complete `ker(R)`. Its
  DIMENSION is closed-form exact (camera_pixels − rank); 1a is an axis-aligned
  subset.
- **Frame0 corollary** — SegNet slices `x[:,-1,...]` (frame1), so ALL of frame0
  is SegNet-invisible by construction; frame0's resize zero-weight pixels are
  invisible to BOTH heads.

**TIER 2 — MEASURED LOW-SENSITIVITY** (`[macOS-CPU advisory]`): the nonlinear
network part has no exact null space, but the #36 atlas measured the per-pair
joint-cone / pose-Jacobian / seg-margin fields. Tier 2 projects those into scoped
per-pair (and optional per-region) low-singular-direction budgets, each carrying
a `MeasurementScope` (Catalog #385) so a measured budget can never masquerade as
a certified zero.

## The derived null-space dimensions (the headline numbers)

Closed-form, exact, hardware-independent (the `_resize_1d_matrix` derivation
matches `F.interpolate` to fp64 roundoff, 9.9e-15):

| Quantity | Value | Fraction |
|---|---:|---:|
| zero-weight camera ROWS dropped | **106 / 874** | 12.13% |
| zero-weight camera COLS dropped | **140 / 1164** | 12.03% |
| **tier-1a zero-weight pixels / channel** | **230,904 / 1,017,336** | **22.6969%** |
| tier-1a directions per frame (×3 RGB) | 692,712 | — |
| **tier-1b full resize null dim / channel** | **820,728 / 1,017,336** | **80.6742%** |
| **frame0 SegNet-invisible (corollary)** | **3,052,008 / 3,052,008** | **100%** |

So **~22.7% of every camera channel is tier-1 certified invisible to BOTH scorer
heads** (single-pixel, amplitude-unlimited), and the full resize null space is
**80.67%** of camera-pixel directions (matching the deep_math memo's 80.7%). The
ENTIRE frame0 is SegNet-invisible.

## Certification test results (the heart — all pass)

`src/tac/tests/test_evaluator_invisibility_basis.py` (30 tests) +
`..._consumers.py` (11 tests) = 41 dedicated tests; 127 incl sister waterfiller
regression. The certification suite:

- **IN-BASIS → BIT-IDENTICAL.** A **255-amplitude** perturbation at EVERY
  zero-weight pixel (all 3 channels), pushed through the real `F.interpolate`,
  yields scorer-input residual == **0.0 EXACTLY** (not merely small). Negative
  / over-range (`1234.5`) amplitudes equally invisible — amplitude-unlimited.
- **OUT-OF-BASIS → DIFFERS.** A perturbation at a non-zero-weight pixel changes
  the scorer input (residual 33.5 in the spot-check) — the basis discriminates;
  a FAKE basis calling everything invisible is caught.
- **CLIPPING HONORED.** A zero-weight pixel set to 99999 then uint8-clipped to
  255 is still invisible (residual 0.0).
- **DERIVATION FIDELITY.** The closed-form 1D matrix reproduces `F.interpolate`
  to 1e-12 on both contest axes (the null space is computed against the REAL
  operator, not an approximation).
- **REAL-SCORER NO-FAKE (slow).** The invisibility survives the FULL upstream
  preprocessing of BOTH heads (SegNet resize; PoseNet resize+`rgb_to_yuv6`),
  bit-identical, on a real-scale frame pair — AND the converse: a frame0 ch0
  +255 perturbation NOT at zero-weight DOES change PoseNet's frame0 input
  (proving frame0 is SegNet-free but NOT PoseNet-free outside the zero-weight
  set — the corollary's exact boundary).

## The artifact (`evaluator_invisibility_basis.v1`)

`tools/build_evaluator_invisibility_basis.py` materialises the basis on the SSD
waterfall (`/Volumes/VertigoDataTier/pact` → `/Volumes/APDataStore/pact` → local
opt-in), timestamped, with a sha-cited manifest. NO `/tmp`.

Built artifact (tier-1 + 600 tier-2 rows from the #36 atlas):
`/Volumes/VertigoDataTier/pact/evaluator_invisibility_basis_20260610T011902Z/`
- `evaluator_invisibility_basis.jsonl` — header (tier-1 summary + corollary +
  provenance) + one line per tier-2 row (header sha `0596d7c6…`).
- `tier1_resize_null_space.npz` — zero-weight row/col index arrays + the boolean
  `(874,1164)` per-pixel invisibility mask (the queryable spatial surface).
- `manifest.json` — sha256 of both files + tier-1 summary + tier-2 atlas ref.

Round-trip verified: `from_jsonl_lines` re-DERIVES tier-1 from sizes (the
certified basis is reproducible from sizes alone) and rebuilds 600 scoped tier-2
rows (pair 442 usable_budget_fraction 0.590, scope non-empty).

Query API (by frame_role / channel / pixel-region / pair): `tier1_pixel_invisible`,
`tier1_frame0_segnet_invisible`, `zero_weight_pixel_mask`, `tier2_by_pair`,
`tier2_by_region`, `tier1_free_byte_fraction_per_channel`.

## Consumer hooks (WIRED, not just documented)

`src/tac/optimization/evaluator_invisibility_basis_consumers.py` (imports the
just-rewired `lf_payload_rate_distortion`, does NOT edit it):

**(a) #46 waterfiller `null_basis` action builder** —
`count_section_tier1_free_bytes(section, basis, pixel_locations, sha)` →
`SectionTier1Accounting`; then `build_null_basis_recode_action` /
`build_null_basis_drop_action` produce a `CandidateActionEvaluation` declaring
**CERTIFIED zero distortion** (`est_delta_d_seg = est_delta_d_pose = 0.0`,
DERIVED not estimated) for bytes that provably encode tier-1-invisible pixels.
Under THE LAW these have negative ΔS_total + positive value_per_byte — the
strongest possible free-bytes branch of the reverse waterfill. Fail-closed:
out-of-bounds / non-certifiable bytes are NOT counted free. Cited by basis sha.

**(b) PR110++ certified-free atom generator** —
`generate_pr110_certified_free_atoms(basis, frame_role, channels, sha, max_atoms)`
yields `CertifiedFreePerturbationAtom`s (one per channel × zero-weight pixel =
free repair room: perturbing along them carries payload at zero scorer cost).
`certified_free_pixel_capacity` reports the total free-repair-room budget.
**Exposed via the existing `tac.null_space_exploiter` surface** (extended per the
orphan-inventory REUSE plan — both the byte-space null basis and this pixel-space
certified complement now live on the one canonical null-space surface).

## Highest-value consumer use

**The #46 waterfiller `null_basis` recode/drop action is the highest-value use.**
Every payload section whose bytes encode tier-1-invisible camera pixels (≈22.7%
of any frame's pixel budget, certified, amplitude-unlimited) can be coarsened or
dropped at PROVABLY zero distortion — not an atlas estimate, a closed-form
correctness fact about the scorer INPUT. That converts the waterfill's
free-bytes branch from "lossless recode assumed zero-cost" into "certified
zero-cost", which is exactly the rigor PR110/the CPU frontier lacked when they
found free bytes by stumbling. The basis tells the waterfiller WHERE the free
bytes provably are, before any exact re-measure spend.

## REUSE (no-duplicative-code; orphan inventory 2026-06-09)

- **Estimator → certified**: `tac.xray.bilinear_resize_nullspace` is the
  Monte-Carlo estimator (80.7% sampled); this module is its closed-form
  certification (derived, exact, with the residual==0.0 proof). Not duplicated —
  promoted to derivation grade.
- **Tier-2 source**: `tac.optimization.evaluator_response_atlas` (#36) +
  `frame1_joint_safe_cone` (#35) — consumed by reference (cone-map path + sha),
  no tensors copied.
- **Scope guard**: `tac.substrates._shared.constants_provenance_manifest.MeasurementScope`
  (Catalog #385) reused for tier-2 measurement scoping.
- **Null-space surface**: `tac.null_space_exploiter` extended (additive
  re-export), not forked — the byte-space null basis (already wired into
  `unified_action.py`) and this pixel-space certified complement coexist.
- **Waterfiller**: `lf_payload_rate_distortion` imported, NOT edited (per the
  directive: it was just rewired — consume its API).

## 6-hook wire-in (Catalog #125)

1. **sensitivity-map** — ACTIVE. The zero-weight pixel mask IS a per-camera-pixel
   certified-invisibility sensitivity map (weight 0 = certified free).
2. **Pareto constraint** — ACTIVE (advisory). Tier-1 directions are the
   amplitude-unlimited feasible boundary; perturbations there cost zero scorer
   distortion.
3. **bit-allocator hook** — ACTIVE (PRIMARY). The `null_basis` recode/drop action
   is a certified-zero-distortion free-byte allocator for the #46 waterfill.
4. **cathedral autopilot dispatch** — N/A. Advisory budget surface; no archive
   bytes emitted by this lane (it informs lanes that do).
5. **continual-learning posterior** — N/A. Tier-1 is a closed-form derivation
   (not an empirical anchor); tier-2 is `[macOS-CPU advisory]` non-promotable,
   recomputed per archive.
6. **probe-disambiguator** — ACTIVE. The tier-1 (certified) vs tier-2 (measured)
   split IS the disambiguator between a provably-free direction and a
   measured-low-sensitivity budget; the certification test (in-basis vs
   out-of-basis) is the regime-conditional probe.

## Per-layer canonical-vs-unique decision (Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| bilinear resize kernel | FORK_PRINCIPLED | closed-form DERIVATION (the xray sister is a sampler; certification needs the exact kernel) |
| tier-1 zero-weight basis | FORK_PRINCIPLED | genuinely new: the certified-exact axis-aligned basis + residual==0.0 proof |
| tier-2 scoped rows | ADOPT_CANONICAL (MeasurementScope, #36 atlas) | reuse the Catalog #385 scope guard + the atlas as source |
| null-space surface | ADOPT_CANONICAL (extend `null_space_exploiter`) | additive re-export onto the canonical surface |
| waterfiller action | ADOPT_CANONICAL (`CandidateActionEvaluation`) | consume the just-rewired #46 API, do not refork |

## Reproduce

```bash
# Build the artifact (tier-1 closed-form + tier-2 from the #36 atlas):
PYTHONPATH=src:upstream .venv/bin/python tools/build_evaluator_invisibility_basis.py \
    --atlas-jsonl /Volumes/VertigoDataTier/pact/evaluator_response_atlas_*/evaluator_response_atlas.jsonl
# Tests (41 dedicated; the certification suite is the heart):
PYTHONPATH=src:upstream .venv/bin/python -m pytest \
    src/tac/tests/test_evaluator_invisibility_basis.py \
    src/tac/tests/test_evaluator_invisibility_basis_consumers.py -q   # 41 passed
```

## Commits

- `a0744896d` — kernels + 30 tests (closed-form tier-1 + corollary + tier-2 scaffold).
- `7e4ac638b` — consumers (null_basis waterfiller action + PR110++ atom generator
  via null_space_exploiter) + artifact CLI + 11 consumer tests.

Lane: `lane_evaluator_invisibility_basis_20260609` (L1: impl_complete +
three_clean_review + memory_entry). `research_only=true` budget surface.
