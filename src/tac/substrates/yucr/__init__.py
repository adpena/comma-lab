# SPDX-License-Identifier: MIT
"""YUCR — Yousfi-UNIWARD-Cooperative-Receiver substrate.

YUCR unifies two seemingly distinct results from compression theory and
steganalysis into a single bit-allocator:

* **Atick-Redlich 1990** (*Neural Computation* 2:308-320) — efficient-coding
  theorem for cooperative receivers. The encoder needs to transmit only
  ``H(X | W_scorer, A_scorer, P_scorer)``: bits the scorer cannot extract from
  its own preprocessing pipeline. The orthogonal complement of the scorer's
  receptive subspace is, by construction, free.

* **Yousfi/Fridrich UNIWARD + Filler 2011 syndrome-trellis codes (STC)**
  (IEEE TIFS) — inverse steganalysis. Place quantization noise where the
  scorer's CNN is BLIND. The embedding cost map ``cost(x,y)`` is the inverse
  of per-pixel scorer detectability; STC minimizes ``sum(C * n)`` for a fixed
  payload bit budget via a syndrome-trellis water-fill.

**The synthesis** (`feedback_yousfi_uniward_cooperative_receiver_synthesis_landed_20260513`):
``cost(x,y)`` IS the orthogonal-complement projector of ``H(X | scorer)``.
UNIWARD's per-pixel detectability map equals
``||grad_pixel d_seg|| + sqrt(10) * ||grad_pixel d_pose||`` (already
differentiable via :mod:`tac.differentiable_eval_roundtrip` and
:mod:`tac.substrates.score_aware_common`). The Atick-Redlich theorem tells you
the FLOOR; UNIWARD-STC tells you HOW to constructively reach it.

YUCR is a **SIDECAR** substrate: it composes with a base substrate (A1, PR101,
time-traveler L5 autonomy, sane_hnerv, DP1) by computing the cost map for the
base's reconstruction and STC-allocating a small int8 noise residual that
charges only on the scorer-blind subspace.

**Predicted contest-CPU score band**: ``[A1_anchor + Delta]`` where
``Delta in [-0.020, -0.040]`` ``[time-traveler-prediction]``. From the
verified A1 anchor 0.192848 ``[contest-CPU-1to1]``, this projects YUCR into
the band 0.153 - 0.173. **NOT a score claim** — score authority requires
both ``[contest-CUDA]`` AND ``[contest-CPU]`` paired auth eval on 1:1
contest-CI hardware per CLAUDE.md "Submission auth eval — BOTH CPU AND
CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable.

Catalog #124 STRICT archive-grammar 8 fields (declared inline so the AST
walker observes them):

- ``archive_grammar``: monolithic single-file ``0.bin`` (HNeRV parity L3)
- ``parser_section_manifest``: YUCR1 header + 4 length-prefixed sections
  (cost_map_int8 + stc_payload + base_archive_id + meta JSON)
- ``inflate_runtime_loc_budget``: <= 200 LOC substrate-engineering waiver
  (full inverse-STC unpack + cost-map-weighted dequantization)
- ``runtime_dep_closure``: torch + brotli + numpy (HNeRV parity L4 <= 2 deps
  + numpy as universal-stdlib-equivalent for buffer protocol)
- ``export_format``: YUCR1 monolithic single-zip-member ``0.bin``
- ``score_aware_loss``: :class:`YUCRScoreAwareLoss` routes through the
  canonical :func:`tac.substrates.score_aware_common.score_pair_components`
  per Catalog #164 + adds cost-map-weighted L1 reconstruction term
- ``bolt_on_loc_budget``: ``lane_class=substrate_engineering`` (HNeRV parity
  L7); cost-map + STC water-fill + composability wrapper exceed the 350 LOC
  bolt-on cap
- ``no_op_detector_planned``: pack/parse roundtrip is byte-stable; archive
  payload is structurally consumed by inverse-STC unpack in inflate.py

Distinction from sister substrates (this is NOT a duplicate):

* **wyner_ziv_cooperative_receiver** uses Slepian-Wolf cosets + side-info
  predictor at the decoder. YUCR uses UNIWARD cost map + STC syndrome bits.
  WZ binning ``H(X|Y)`` is generic; YUCR cost-map is scorer-gradient-derived
  (Atick-Redlich orthogonal-complement projector). Complementary, NOT
  redundant — they target different stages of the codec stack.
* **time_traveler_l5_autonomy** uses the Atick-Redlich loss only. YUCR adds
  the constructive STC allocator that places noise where the scorer is
  blind. Time-Traveler is the loss; YUCR is the bit-allocator.

Cross-references:

* Sister substrate (cooperative-receiver loss only):
  :mod:`tac.substrates.time_traveler_l5_autonomy`
* Sister substrate (Slepian-Wolf binning):
  :mod:`tac.substrates.wyner_ziv_cooperative_receiver`
* Sister substrate (sidecar precedent / composability):
  :mod:`tac.substrates.a1_plus_wavelet_residual`
* Canonical scorer-input contract:
  :mod:`tac.substrates.score_aware_common`
* Canonical eval-roundtrip primitive:
  :mod:`tac.differentiable_eval_roundtrip`
* Canonical inflate runtime helpers:
  :mod:`tac.substrates._shared.inflate_runtime`

**No KILL verdicts** — per CLAUDE.md "KILL is LAST RESORT" non-negotiable.
**No /tmp paths** — per CLAUDE.md "Forbidden /tmp paths in any persisted
artifact". **No score claims** without paired ``[contest-CUDA]`` AND
``[contest-CPU]`` evidence on 1:1 contest-CI hardware.

Lane: ``lane_yucr_yousfi_uniward_cooperative_receiver_20260514``
"""

from tac.substrates.yucr.architecture import (
    YUCR_BASE_SUBSTRATE_IDS,
    YUCR_DEFAULT_BASE_SUBSTRATE,
    YUCR_DEFAULT_STC_PAYLOAD_BITS,
    YUCR_OVERHEAD_TARGET_BYTES_MAX,
    YUCR_OVERHEAD_TARGET_BYTES_MIN,
    YUCRConfig,
    YUCRSubstrate,
    compose_with_base,
)
from tac.substrates.yucr.archive import (
    YUCR1_HEADER_FMT,
    YUCR1_HEADER_SIZE,
    YUCR1_MAGIC,
    YUCR1_SCHEMA_VERSION,
    YUCRArchive,
    build_readiness_manifest,
    pack_archive,
    parse_archive,
)
from tac.substrates.yucr.cost_map import (
    YUCRCostMapMode,
    compute_cost_map,
    compute_cost_map_dummy,
    quantize_cost_map_int8,
)
from tac.substrates.yucr.score_aware_loss import (
    YUCRLossWeights,
    YUCRScoreAwareLoss,
)
from tac.substrates.yucr.stc_encoder import (
    STC_DEFAULT_BUDGET_BITS,
    STC_LATTICE_LEVELS,
    STCAllocationResult,
    decode_stc_payload,
    encode_stc_payload,
    waterfill_allocate,
)

__all__ = [
    "STC_DEFAULT_BUDGET_BITS",
    "STC_LATTICE_LEVELS",
    "STCAllocationResult",
    "YUCR1_HEADER_FMT",
    "YUCR1_HEADER_SIZE",
    "YUCR1_MAGIC",
    "YUCR1_SCHEMA_VERSION",
    "YUCRArchive",
    "YUCRConfig",
    "YUCRCostMapMode",
    "YUCRLossWeights",
    "YUCRScoreAwareLoss",
    "YUCRSubstrate",
    "YUCR_BASE_SUBSTRATE_IDS",
    "YUCR_DEFAULT_BASE_SUBSTRATE",
    "YUCR_DEFAULT_STC_PAYLOAD_BITS",
    "YUCR_OVERHEAD_TARGET_BYTES_MAX",
    "YUCR_OVERHEAD_TARGET_BYTES_MIN",
    "build_readiness_manifest",
    "compose_with_base",
    "compute_cost_map",
    "compute_cost_map_dummy",
    "decode_stc_payload",
    "encode_stc_payload",
    "pack_archive",
    "parse_archive",
    "quantize_cost_map_int8",
    "waterfill_allocate",
]
