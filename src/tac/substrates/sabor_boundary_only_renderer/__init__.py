"""tac.substrates.sabor_boundary_only_renderer — Stable-Argmax Boundary-Only Renderer.

L0 SKETCH substrate per operator directive 2026-05-13 (PAIR T+OPT3). Grand
Council O1 first-principles hypothesis (Shannon LEAD): the contest scorer's
SegNet distortion is ``(argmax(out1) != argmax(out2)).float().mean()`` — only
logit ORDERING affects the score, not magnitudes. Pixels deeply interior to
argmax-stable regions are "free bytes" — their RGB values can be perturbed
without changing the SegNet argmax map (and the cost into PoseNet is bounded
by the 4-neighbor stability margin documented in
``.omx/research/sabor_boundary_audit_20260513.md``).

φ1 SABOR audit empirically confirmed (2026-05-13, ``[macOS-CPU advisory]``):

* 99.27% of pixels survive ε=32 RGB uniform-noise perturbation (K=2 samples)
  with identical SegNet argmax.
* Interior fraction (4-neighbor stable) ≥ 0.97 of stable fraction at every ε
  — stable pixels form **large contiguous clusters**.
* Free-byte capacity at ε=32 (conservative 1ch×1bit): **14.6 MB per video**.
  This is **82× the entire current frontier archive** (PR101 0.193, ~178 KB).

The SABOR substrate explicitly factors the per-pair frame into:

1. **Boundary mask** (sparse): the union of (a) Canny edge detection +
   (b) SegNet argmax disagreement at 4-neighbor — typically 1-3% of pixels.
   Stored as packbits + brotli.
2. **Boundary pixel RGB** (high-fidelity int8 RGB): exact RGB at every
   boundary pixel. Cost: ~3 bytes × boundary_pixel_count × num_frames.
3. **Texture fill rules** (~3 KB): per-class color statistics (mean RGB per
   SegNet class) + per-pair small bias correction. The interior pixels are
   reconstructed as ``mean_rgb[class(pixel)] + bias_pair`` at inflate time.
4. **Decoder state** (~10-30 KB): tiny SegNet-segmentation-conditioned
   refinement decoder (FiLM blocks) that lifts the texture-filled RGB to
   the final per-pair output.

At inflate time the algorithm is deterministic, ≤ 200 LOC, brotli + torch
runtime closure (substrate-engineering opt-out for the >100-LOC budget).
The substrate is **score-aware**: gradients flow from contest scorers
through the decoder + bias + per-class-mean parameters to the boundary
mask via the differentiable eval-roundtrip + patched yuv6 (PR #95/#106
monkey-patch contract).

This is the φ-arm follow-up to the φ1 audit. The audit proved the capacity
exists; this substrate consumes it. The boundary mask payload is
information-theoretically necessary (boundaries carry the SegNet argmax
signal); the texture fill is the cheapest possible interior representation
(per-class mean + per-pair bias). Together they target a sub-0.20 ``[contest-
CPU]`` anchor at $1-3 dispatch cost (smaller than the time-traveler L5
substrate because boundary-only avoids the full per-pair decoder forward).

Sister substrates (different surfaces; coordinated 2026-05-13):

* ``tac.substrates.time_traveler`` (φ2 + φ3 + φ4 / O2 + O3 + O4 arms).
* ``tac.substrates.time_traveler_l5_autonomy``.
* ``tac.substrates.s2sbs_byte_stuffing`` (φ3 / O3 stride-2 byte stuffing).

Lane registration::

    python tools/lane_maturity.py add-lane \\
        lane_sabor_boundary_only_renderer_substrate_20260513 \\
        --name "SABOR boundary-only renderer substrate (Council F O1)" --phase 2

13 HNeRV parity-discipline lessons compliance — design-time declaration:

| Lesson | Status |
|---|---|
| L1 substrate must be score-aware | PASS (score-aware Lagrangian wired in score_aware_loss.py) |
| L2 export-first archive grammar | PASS (archive.py SBO1 declared BEFORE training) |
| L3 monolithic 0.bin | PASS (single-file fixed-offset grammar SBO1) |
| L4 inflate <= 100 LOC, <= 2 deps | substrate_engineering exception (~180 LOC; torch + brotli) |
| L5 full RGB renderer | PASS (boundary + texture-fill produces RGB; NOT a mask codec) |
| L6 score-domain Lagrangian | PASS (alpha*B/N + beta*d_seg + gamma*sqrt(d_pose) + delta*boundary_consistency) |
| L7 bolt-on <= 350 LOC | substrate_engineering exception (~900 total) |
| L8 eval-roundtrip + diff yuv6 | PASS (wired via tac.differentiable_eval_roundtrip + score_pair_components) |
| L9 runtime closure | PASS (torch + brotli; numpy is torch transitive) |
| L10 mask/pose coupling | N/A (renderer replaces full slot; boundary mask is byte payload not SegNet input) |
| L11 no-op detector | PASS (Catalog #139 _build_no_op_proof + byte-mutation smoke planned) |
| L12 single-LOC review discipline | PASS (each file reviewable in 30s; 4 files total) |
| L13 KILL last resort | PASS (DEFERRED-pending-anchor reactivation path documented) |

Catalog #124 archive-grammar 8 fields:

* ``archive_grammar``: monolithic single-file 0.bin SBO1 fixed offsets.
* ``parser_section_manifest``: parse_archive() -> SaborArchive(boundary_mask,
  boundary_rgb, class_means, decoder_state_dict, meta).
* ``inflate_runtime_loc_budget``: <= 200 LOC (substrate-engineering exception).
* ``runtime_dep_closure``: torch, brotli.
* ``export_format``: brotli(state_dict + class_means) + packbits(boundary_mask)
  + int8(boundary_rgb) + utf8-json(meta).
* ``score_aware_loss``: alpha*B/N + beta*d_seg + gamma*sqrt(d_pose) +
  delta*boundary_consistency.
* ``bolt_on_loc_budget``: ~900 LOC (substrate_engineering tag).
* ``no_op_detector_planned``: Catalog #139 _build_no_op_proof byte-mutation
  smoke during archive emission.

Reactivation criteria (L0 -> L1):
    First archive build emits a valid SBO1 0.bin AND macOS-CPU advisory
    proxy returns a score in band [0.15, 0.25] AND CPU smoke trainer
    completes 3 epochs without NaN.

Reactivation criteria (L1 -> L2):
    Modal T4 smoke ($0.10-$0.30) emits a CUDA auth-eval JSON with
    score_claim_valid AND in plausible band; macOS-CPU proxy on the same
    archive lands in [0.15, 0.20].

Predicted band: ``[0.165, 0.185] [contest-CPU prediction]`` per Council F
O1 derivation; the band is intentionally wider than the φ1 audit's
14.6 MB capacity because the operating gap between capacity and achievable
varies with boundary-pixel ratio (1-3%) and per-class mean fidelity.

Composition examples (cross-paradigm with sister substrates):

* SABOR × A1 LAPose sidecar: SABOR replaces the RGB renderer; LAPose
  sidecar adds the foveal pose residual on top. Boundary mask is
  preserved; LAPose residual writes only to interior texture-filled
  pixels (PoseNet sees the upgrade; SegNet sees no argmax change because
  interior pixels are stable by construction).
* SABOR × wavelet residual: wavelet detail-band residual adds high-
  frequency texture to interior pixels (same argmax-stability proof).
* SABOR × magic_codec sidecar: magic_codec's free-byte capacity rides on
  SABOR's interior-pixel free-byte capacity (compositional capacity is
  additive when both substrates respect SegNet argmax stability).

CLAUDE.md compliance:

* No silent device defaults (caller passes device explicitly).
* No scorer loading inside this module.
* No /tmp paths.
* No KILL verdicts in the scaffold (DEFER-pending-anchor only).
* score_claim=false, promotion_eligible=false, ready_for_exact_eval_dispatch
  =false until empirical CUDA anchor lands.

Cross-references:

* φ1 audit: ``.omx/research/sabor_boundary_audit_20260513.md``
* φ1 measurement tool: ``tools/measure_segnet_argmax_stable_interior.py``
* Time-traveler design context:
  ``.omx/research/time_traveler_architecture_reverse_engineered_20260513.md``
* Sister exemplars: ``tac.substrates.a1_plus_lapose``,
  ``tac.substrates.a1_plus_wavelet_residual``, ``tac.substrates.grayscale_lut``.
* Catalog #124 archive-grammar non-negotiable.
* CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" non-negotiable.
"""

from .architecture import (
    SaborBoundaryOnlyConfig,
    SaborBoundaryOnlyRenderer,
    detect_boundary_mask_canny_segnet,
)
from .archive import (
    SBO1_MAGIC,
    SBO1_SCHEMA_VERSION,
    SaborArchive,
    pack_archive,
    parse_archive,
)
from .score_aware_loss import (
    SaborBoundaryOnlyLossWeights,
    SaborBoundaryOnlyScoreAwareLoss,
)

__all__ = [
    "SBO1_MAGIC",
    "SBO1_SCHEMA_VERSION",
    "SaborArchive",
    "SaborBoundaryOnlyConfig",
    "SaborBoundaryOnlyLossWeights",
    "SaborBoundaryOnlyRenderer",
    "SaborBoundaryOnlyScoreAwareLoss",
    "detect_boundary_mask_canny_segnet",
    "pack_archive",
    "parse_archive",
]
