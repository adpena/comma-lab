# SPDX-License-Identifier: MIT
"""tac.substrates.nscs06_carmack_hotz_strip_everything — NO-neural radical codec (L1 SCAFFOLD).

Per grand-reunion symposium #4 composite design (2026-05-15;
``feedback_grand_reunion_fields_grade_passion_full_council_debrief_vision_strategy_design_whiteboard_session_20260515.md``
§"Composite #4 - The Carmack-Hotz Strip-Everything Stack"). Predicted band
``[0.10, 0.20]`` ``[prediction; first-principles-bound; HIGH VARIANCE]`` —
$5-15 dispatch; NEVER built per MEAT-ON-THE-BONE audit Axis 1 finding #1.

The radical Carmack-Hotz architecture rebuilds the codec from first principles:

  * **NO neural codec.** Zero learned weights, zero entropy bottleneck, zero
    hyperprior. The archive contains a HAND-ROLLED binary container only.
  * **Closed-form bit allocation.** Compress side has FREE access to the
    contest scorers (per CLAUDE.md "Contest compliance"), so we use SegNet to
    compute per-pixel class-importance + arithmetic-code the residual with
    class-conditional CDFs derived from the scorer's own predictions. NO
    score-aware-loss training; the scorer's argmax IS the allocation.
  * **Quantizr-style grayscale-LUT for masks.** PR #56 Selfcomp paradigm:
    odd-frame grayscale ONLY (even frames warped from odd via pose deltas).
  * **Carmack-style "simplest thing that could possibly work"**: every byte
    tested for elimination; iterate from minimal.
  * **Hotz-style "burn the canonical helpers and rebuild"**: codec module is
    self-contained — no sane_hnerv / NeRV / Cool-Chic / Ballé inheritance.

This is the canonical CLASS-SHIFT substrate per the within-class-vs-class-shift
directive: architecture class = fundamentally NOT neural; decode-time
contract = arithmetic-decode-only; training-time paradigm = NO training.

13 HNeRV parity-discipline lessons compliance — design-time declaration:

| Lesson | Status | Notes |
|---|---|---|
| L1 substrate must be score-aware | PASS | scorer queried at COMPRESS time; class-conditional CDFs |
| L2 export-first archive grammar | PASS | CH06 grammar declared BEFORE any compress code (this file) |
| L3 monolithic 0.bin | PASS | single-file fixed-offset CH06 grammar |
| L4 inflate <= 100 LOC, <= 2 deps | PASS | target ~95 LOC; numpy + Pillow only (NO torch) |
| L5 full RGB renderer | PASS | per-pair RGB output via grayscale-LUT + arith-decoded residual |
| L6 score-domain Lagrangian | N/A | NO training; bit allocation is closed-form analytical |
| L7 bolt-on <= 350 LOC | substrate_engineering exception | total ~900 LOC across 5 files |
| L8 eval-roundtrip + diff yuv6 | N/A | NO training; 384->874->uint8->384 simulated at compress only |
| L9 runtime closure | PASS | numpy + Pillow; both wheel-resolvable on contest T4 |
| L10 mask/pose coupling | PASS | pose deltas drive frame-1 warp from frame-0 grayscale |
| L11 no-op detector | PASS | Catalog #139 byte-mutation smoke planned + scaffolded |
| L12 single-LOC review discipline | PASS | each file reviewable in 30s; codec.py is the densest |
| L13 KILL last resort | PASS | DEFERRED-pending-analytical-renderer-anchor reactivation |

Catalog #124 archive-grammar 8 fields:

    archive_grammar:            monolithic single-file 0.bin CH06 fixed offsets
    parser_section_manifest:    parse_archive() -> (palette, grayscale_q, pose, residual_bits, meta)
    inflate_runtime_loc_budget: <= 100 LOC (Pillow + numpy only)
    runtime_dep_closure:        numpy, Pillow
    export_format:              custom (CH06: hand-rolled binary; no fp16/brotli/torch)
    score_aware_loss:           custom (NO TRAINING; closed-form bit allocator at compress)
    bolt_on_loc_budget:         ~900 LOC (substrate_engineering exception per L7)
    no_op_detector_planned:     Catalog #139 _build_no_op_proof + byte-mutation smoke

Catalog #272 distinguishing-feature contract:

    UNIQUE per method (vs all 36 other substrates):
      - ZERO neural network weights in the archive
      - ZERO PyTorch dependency at inflate time
      - Closed-form bit allocation via SegNet-derived class-conditional CDFs
      - Quantizr PR #56 grayscale-LUT paradigm (odd-frame ONLY; even warped)
      - Per-pair arithmetic-coded residual w/ per-class CDF

Canonical-vs-unique decision per layer (per UNIQUE-AND-COMPLETE-PER-METHOD
operating mode + standing directive *"all possible should be pulled into
the decorator or similar reusable and shareable tools and helpers"*):

| Layer | Decision | Rationale |
|---|---|---|
| Architecture (this package) | 100% UNIQUE | No neural; no PyTorch; hand-rolled. The radical premise IS the substrate. |
| Compress-side bit allocator | 100% UNIQUE | Closed-form formula; not implementable via score_aware_common helpers. |
| Inflate runtime | UNIQUE | Pure numpy+Pillow arith-decode + LUT lookup; NO scorer; NO torch. |
| Auth eval routing | ADOPT canonical | `tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call` (Catalog #226). |
| NVML/Modal/CUDA env hygiene | ADOPT canonical | Catalog #244 NVML block in remote driver; `tac.deploy.modal.runtime` constants. |
| Mount manifest | ADOPT canonical | `tac.deploy.modal.mount_manifest.build_training_image` (Catalog #153). |
| eval_roundtrip simulation | PRESERVE HARD-EARNED | 384->874->uint8->384 simulated at compress-time for per-pixel importance. |
| strict-scorer-rule | PRESERVE HARD-EARNED | inflate.py imports ZERO scorer code (NOT torch, NOT smp, NOT efficientnet). |
| Catalog #220 operational mechanism | PRESERVE HARD-EARNED | archive_bytes_added declared; byte-mutation smoke proves consumption. |
| Trainer skeleton (`_pin_seeds`, `_utc_now_iso`, `_git_head_sha`, etc.) | ADOPT canonical | `tac.substrates._shared.trainer_skeleton` shared helpers. |
| Real-pair decode (`decode_real_pairs`) | ADOPT canonical | Same canonical helper used at COMPRESS time (not training). |
| `device_or_die` | ADOPT canonical | Per Catalog #178 TF32-helper consolidation. |
| Lane registry | ADOPT canonical | `tools/lane_maturity.py` per Catalog #126 pre-registration. |
| SubstrateContract decoration | ADOPT canonical | `@register_substrate(CARMACK_HOTZ_SUBSTRATE_CONTRACT)` per Catalog #241/#242. |
| Catalog #244 remote NVML block | ADOPT canonical | auto-emitted by `tac.substrate_registry.driver_generator`. |

Reactivation criteria (L1 -> L2):
    First Modal smoke produces a byte-closed 0.bin archive + finite contest-CUDA
    auth-eval score in the predicted band [0.10, 0.20]. The HIGH VARIANCE
    rationale: dominant gain on RATE (-0.05 to -0.08) via stripping (no neural
    weights = no decoder bytes); possible large LOSS on distortion (+0.01 to
    +0.05) if the analytical renderer cannot reconstruct enough chroma /
    texture from grayscale + pose alone.

CLAUDE.md compliance:
- No silent device defaults (compress side runs scorer on cuda via canonical helper)
- No scorer loading at inflate (inflate.py has ZERO torch / scorer imports)
- No /tmp paths (compress writes under args.output_dir; inflate honors $1/$2/$3)
- No KILL verdicts in the scaffold (DEFER-pending-anchor only)
- Apples-to-apples axis labels on every score claim ([contest-CUDA] only)
"""

from .archive import (
    CH06_HEADER_FMT,
    CH06_HEADER_SIZE,
    CH06_MAGIC,
    CH06_SCHEMA_VERSION,
    CHROMA_BYTES_PER_CLASS,
    CarmackHotzArchive,
    build_chroma_palette,
    decode_class_label_stream,
    encode_class_label_stream,
    pack_archive,
    parse_archive,
)
from .codec import (
    ArithmeticCoder,
    ClassConditionalCDF,
    GrayscalePalette,
    allocate_bits_closed_form,
    build_grayscale_palette,
)
from .inflate import inflate_one_video, main_cli

__all__ = [
    "CH06_HEADER_FMT",
    "CH06_HEADER_SIZE",
    "CH06_MAGIC",
    "CH06_SCHEMA_VERSION",
    "CHROMA_BYTES_PER_CLASS",
    "ArithmeticCoder",
    "CarmackHotzArchive",
    "ClassConditionalCDF",
    "GrayscalePalette",
    "allocate_bits_closed_form",
    "build_chroma_palette",
    "build_grayscale_palette",
    "decode_class_label_stream",
    "encode_class_label_stream",
    "inflate_one_video",
    "main_cli",
    "pack_archive",
    "parse_archive",
]
