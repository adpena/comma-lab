# SPDX-License-Identifier: MIT
"""D1 — SegNet margin polytope encoder substrate (sub-0.188 path #1, lowest cost).

D1 operationalizes the SegNet argmax-margin manifold from
``.omx/research/deep_math_geometry_manifolds_synthesis_20260514.md`` §3.6
into a constructive **frame-1 perturbation encoder**. The substrate exploits
the *geometric* nullspace of the SegNet argmax (per-pixel logit margin) where
YUCR exploits the *structural* nullspace of frame-0 (SegNet code-discards it).

Core insight (deep-math memo §3.6 + codex eureka #6):

* SegNet outputs 5-class logits ``z(x, y, c)`` per pixel; the argmax decision
  ``c*(x, y) = argmax_c z(x, y, c)`` is **stable** under perturbations
  ``δ`` to the logits with ``||δ||_inf < m(x, y)`` where ``m(x, y) =
  top1 - top2`` is the **logit margin**.
* In input pixel space the corresponding **safe polytope** is
  ``Pi_1(x, y) = {δ ∈ R^3 : ||J_seg(x, y) δ||_inf < m(x, y)}``.
* The volume ``vol(Pi_1)`` scales with ``m^3 / det(J_seg)``: high-margin
  pixels (≈ 80% of a typical frame) have **large** polytopes and can absorb
  large noise budgets without flipping the SegNet argmax; low-margin
  boundary pixels (≈ 20%) shrink toward a singleton.
* The **per-pixel safe-noise budget** is the Newton-step distance to the
  nearest decision boundary: ``B_safe(x, y) = m(x, y) / ||grad_pixel
  logit||``. Allocate quantization noise / dithering only where
  ``|noise(x, y)| < B_safe(x, y)`` (polytope interior).

D1 is the **frame-1 sibling** of YUCR's frame-0 cooperative-receiver
exploit:

* **YUCR** = frame-0 perturbations (SegNet *code-discards* frame 0 at
  ``upstream/modules.py:108``; safe by code).
* **D1**  = frame-1 perturbations (SegNet *sees* frame 1; safe by
  *geometry* — only the polytope interior).
* **Together** they exhaust the bidirectional bit-allocation: YUCR captures
  the structural nullspace; D1 captures the geometric nullspace.

L2 OPERATIONAL contest-CPU score band: ``[A1_anchor + Delta]`` where
``Delta ∈ [-0.012, -0.005]`` ``[first-principles-bound]``. As of
2026-05-14 L2 INTEGRATION lands the post-renderer polytope-interior
overlay via
:func:`tac.substrates.d1_segnet_margin_polytope.overlay.apply_l2_overlay_for_video_list`,
so the D1 sidecar bytes operationally consume into per-pixel frame_1 RGB
perturbations at camera resolution. The archive cost can be reduced 16×
by dispatching with
:data:`tac.substrates.d1_segnet_margin_polytope.margin_map.MARGIN_MAP_SHRUNK_RESOLUTION`
``(96, 128)`` (~2.7 KB total vs ~43 KB at full resolution). **NOT a
score claim** — score authority requires both ``[contest-CUDA]`` AND
``[contest-CPU]`` paired auth eval on 1:1 contest-CI hardware per
CLAUDE.md evidence discipline.

Catalog #124 STRICT archive-grammar 8 fields (declared inline so the AST
walker observes them):

- ``archive_grammar``: monolithic single-file ``0.bin`` (HNeRV parity L3)
- ``parser_section_manifest``: D1POLY1 header + 5 length-prefixed sections
  (margin_map_int8 + safe_budget_payload + base_substrate_id + base_sha256
  + meta JSON)
- ``inflate_runtime_loc_budget``: <= 200 LOC substrate-engineering waiver
  (full inverse-margin unpack + safe-budget-weighted noise allocator)
- ``runtime_dep_closure``: torch + brotli + numpy (HNeRV parity L4 <= 2
  deps + numpy as universal-stdlib-equivalent for buffer protocol)
- ``export_format``: D1POLY1 monolithic single-zip-member ``0.bin``
- ``score_aware_loss``: :class:`D1PolytopeScoreAwareLoss` routes through
  the canonical
  :func:`tac.substrates.score_aware_common.score_pair_components` per
  Catalog #164 + adds margin-preserving hinge term that penalizes pushing
  pixels below the safe margin threshold
- ``bolt_on_loc_budget``: ``lane_class=substrate_engineering`` (HNeRV
  parity L7); margin map + polytope encoder + composability wrapper
  exceed the 350 LOC bolt-on cap
- ``no_op_detector_planned``: pack/parse roundtrip is byte-stable; L2
  inflate runtime applies per-pixel polytope-interior noise overlay via
  :func:`tac.substrates.d1_segnet_margin_polytope.overlay.apply_l2_overlay_for_video_list`;
  bytes_changed and pairs_modified diagnostic ensures sidecar bytes
  produce real frame changes (Catalog #220 OPERATIONAL)

Distinction from sister substrates (this is NOT a duplicate):

* **yucr** uses Atick-Redlich cost map + STC water-fill on frame-0
  perturbations. D1 uses argmax-margin polytope encoding on frame-1
  perturbations. Different scorer-side blind subspace; complementary, NOT
  redundant.
* **a1_plus_wavelet_residual** encodes wavelet residual sidecar on per-pair
  pose-axis. D1 operates on per-pixel SegNet-margin axis. Different
  signal source.
* **time_traveler_l5_autonomy** adds the Atick-Redlich loss only. D1
  ships a polytope-aware bit allocator that constructively encodes
  noise within the safe geometric region.

Cross-references:

* Sister substrate (frame-0 cooperative-receiver):
  :mod:`tac.substrates.yucr`
* Sister substrate (pose-axis sidecar):
  :mod:`tac.substrates.a1_plus_wavelet_residual`
* Canonical scorer-input contract:
  :mod:`tac.substrates.score_aware_common`
* Canonical eval-roundtrip primitive:
  :mod:`tac.differentiable_eval_roundtrip`
* Canonical inflate runtime helpers:
  :mod:`tac.substrates._shared.inflate_runtime`
* Source blueprint:
  ``.omx/research/deep_math_geometry_manifolds_synthesis_20260514.md``
  §3.6 + §10 D1.

**No KILL verdicts** — per CLAUDE.md "KILL is LAST RESORT" non-negotiable.
**No /tmp paths** — per CLAUDE.md "Forbidden /tmp paths in any persisted
artifact". **No score claims** without paired ``[contest-CUDA]`` AND
``[contest-CPU]`` evidence on 1:1 contest-CI hardware.

Lane: ``lane_d1_segnet_margin_polytope_encoder_20260514``
"""

from tac.substrates.d1_segnet_margin_polytope.architecture import (
    D1POLY_BASE_SUBSTRATE_IDS,
    D1POLY_DEFAULT_BASE_SUBSTRATE,
    D1POLY_DEFAULT_BUDGET_BITS,
    D1POLY_OVERHEAD_TARGET_BYTES_MAX,
    D1POLY_OVERHEAD_TARGET_BYTES_MIN,
    D1PolytopeConfig,
    D1PolytopeSidecar,
    _BaseArchiveDescriptor,
    compose_with_base,
    estimate_overhead_bytes,
)
from tac.substrates.d1_segnet_margin_polytope.archive import (
    D1POLY1_HEADER_FMT,
    D1POLY1_HEADER_SIZE,
    D1POLY1_MAGIC,
    D1POLY1_SCHEMA_VERSION,
    D1POLY1_SECTION_ROLES,
    D1PolytopeArchive,
    build_readiness_manifest,
    pack_archive,
    parse_archive,
    parse_d1poly1_archive_bytes,
    update_d1poly1_meta,
)
from tac.substrates.d1_segnet_margin_polytope.diagnostics import (
    D1OverlayDiagnostics,
    analyze_d1_overlay_effect,
)
from tac.substrates.d1_segnet_margin_polytope.margin_map import (
    MARGIN_MAP_DEFAULT_RESOLUTION,
    MarginMapMode,
    compute_logit_margin_map,
    compute_logit_margin_map_dummy,
    dequantize_margin_map_int8,
    quantize_margin_map_int8,
)
from tac.substrates.d1_segnet_margin_polytope.overlay import (
    D1_OVERLAY_AMPLITUDE_SCALES,
    D1_OVERLAY_CHANNEL_POLICIES,
    D1_OVERLAY_SIGN_POLICIES,
)
from tac.substrates.d1_segnet_margin_polytope.polytope_encoder import (
    POLYTOPE_DEFAULT_BUDGET_BITS,
    POLYTOPE_LATTICE_LEVELS,
    POLYTOPE_LATTICE_VALUES,
    PolytopeAllocationResult,
    allocate_noise_within_polytope,
    compute_safe_perturbation_budget,
    decode_polytope_payload,
    encode_polytope_payload,
)
from tac.substrates.d1_segnet_margin_polytope.score_aware_loss import (
    D1PolytopeLossWeights,
    D1PolytopeScoreAwareLoss,
)

__all__ = [
    "D1POLY1_HEADER_FMT",
    "D1POLY1_HEADER_SIZE",
    "D1POLY1_MAGIC",
    "D1POLY1_SCHEMA_VERSION",
    "D1POLY1_SECTION_ROLES",
    "D1POLY_BASE_SUBSTRATE_IDS",
    "D1POLY_DEFAULT_BASE_SUBSTRATE",
    "D1POLY_DEFAULT_BUDGET_BITS",
    "D1POLY_OVERHEAD_TARGET_BYTES_MAX",
    "D1POLY_OVERHEAD_TARGET_BYTES_MIN",
    "D1_OVERLAY_AMPLITUDE_SCALES",
    "D1_OVERLAY_CHANNEL_POLICIES",
    "D1_OVERLAY_SIGN_POLICIES",
    "MARGIN_MAP_DEFAULT_RESOLUTION",
    "POLYTOPE_DEFAULT_BUDGET_BITS",
    "POLYTOPE_LATTICE_LEVELS",
    "POLYTOPE_LATTICE_VALUES",
    "D1OverlayDiagnostics",
    "D1PolytopeArchive",
    "D1PolytopeConfig",
    "D1PolytopeLossWeights",
    "D1PolytopeScoreAwareLoss",
    "D1PolytopeSidecar",
    "MarginMapMode",
    "PolytopeAllocationResult",
    "_BaseArchiveDescriptor",
    "allocate_noise_within_polytope",
    "analyze_d1_overlay_effect",
    "build_readiness_manifest",
    "compose_with_base",
    "compute_logit_margin_map",
    "compute_logit_margin_map_dummy",
    "compute_safe_perturbation_budget",
    "decode_polytope_payload",
    "dequantize_margin_map_int8",
    "encode_polytope_payload",
    "estimate_overhead_bytes",
    "pack_archive",
    "parse_archive",
    "parse_d1poly1_archive_bytes",
    "quantize_margin_map_int8",
    "update_d1poly1_meta",
]
