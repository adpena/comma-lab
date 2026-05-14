# SPDX-License-Identifier: MIT
"""D4 Wyner-Ziv frame-0 substrate (deep-math M7) — sub-0.188 path #3.

Per ``.omx/research/deep_math_geometry_manifolds_synthesis_20260514.md`` §3.5
"Wyner-Ziv frame-0 reconstruction (codex eureka #7, deepened)" and §6 M7
"Wyner-Ziv frame-0 reconstruction" (top-ranked mechanism by predicted ΔS in
the entire memo). This is the **strongest single-substrate bet for sub-0.188
gate clearance** (predicted ΔS -0.025 to -0.045 [first-principles-bound +
mathematical-derivation]); single substrate could clear sub-0.155 cumulatively
when composed with M2 + M13 (top-3 stack -0.030 to -0.055).

Core insight (deep-math memo §3.1 + §3.5)
----------------------------------------

Per upstream/modules.py:103-109 the SegNet contest scorer slices ``x[:, -1, ...]``
and so sees ONLY frame 1 of each pair; frame 0 is in the SegNet *structural
nullspace*. PoseNet sees both frames (12-channel YUV6 ``(2*6, H/2, W/2)``).
Therefore frame 0 can be DERIVED at decode time from frame 1 + a low-rate
side-info channel (Wyner-Ziv 1976 / Slepian-Wolf 1973), and the saved bytes
from NOT encoding frame 0 directly amortize into a large rate-term Δ.

::

    R_min(frame_0 | frame_1) = H(motion) + H(photometric_residual)
                             ≈ 24 B/pair (SE(3)) + ~5 KB/pair (quantized residual)
                             ≈ 3,030 B/pair total ≪ raw frame_0 entropy ~250 KB/pair

Distinction from sister substrate ``tac.substrates.wyner_ziv_cooperative_receiver``
-----------------------------------------------------------------------------------

The sister substrate (alien-tech N3, lane 20260513) implements DISCUS-style
Slepian-Wolf coset binning of the SOURCE pair against the SegNet+PoseNet
scorer as cooperative-receiver side information. It bins both frames against
the scorer and transmits coset indices.

THIS substrate (D4, lane 20260514) implements a DIFFERENT mechanism: per-pair
parametric motion (SE(3) ego-motion OR optical-flow-field residual; both
modes shipped per probe-disambiguator wire-in) plus a photometric residual,
all conditioned on frame 1 (which is provided by a BASE substrate at decode
time — e.g. A1, PR101, HDM8). The receiver side-info is the BASE substrate's
reconstructed frame 1; the D4 archive carries only the frame_0 - warp(frame_1)
residual plus the motion parameters.

The two substrates are structurally orthogonal and could compose: WZ
substrate carries the base renderer + DISCUS cosets; D4 sidecar carries the
frame-0 derivation parameters. For this landing D4 is a STANDALONE substrate
that composes with an external base archive (specified by ``--base-archive-path``).

Probe-disambiguator (Catalog #125 hook #6)
------------------------------------------

Two defensible interpretations of "low-rate motion side info":

1. **SE(3) parametric motion** — 6 floats per pair (3 translation +
   3 rotation in axis-angle), ~24 B/pair × 600 pairs = 14.4 KB. Best when
   ego-motion is well-conditioned (forward driving). Closed-form Jacobian for
   gradient-based training.
2. **Optical-flow-field residual** — per-pixel ``(u, v)`` flow at coarse
   resolution (e.g. 12×16), quantized to int8 + brotli-packed.
   Non-parametric; better for scenes with parallax, lane changes, vehicle
   tracking. ~2-4 KB/pair after quantization.

Both modes ship as callable interfaces per the design-tension memo
``feedback_design_tension_ship_both_interpretations_let_math_arbitrate_20260509.md``
+ Catalog #125 hook #6. The probe is at
``tools/probe_d4_motion_model_disambiguator.py``.

Catalog #124 archive-grammar 8 fields (declared inline so the AST walker observes them)
---------------------------------------------------------------------------------------

- ``archive_grammar``: monolithic single-file ``0.bin`` (substrate-engineering)
- ``parser_section_manifest``: WZF01 header + (a) motion-params section
  (24 B × 600 = 14.4 KB SE(3) OR brotli-packed flow grid) + (b) residual
  section (per-pair int8 quantized + brotli) + (c) base substrate sha
  pointer (32 B sha256 hex bytes) + (d) sorted-keys JSON meta
- ``inflate_runtime_loc_budget``: ≤ 200 LOC substrate-engineering waiver
  (full motion-warp + residual decode + base-substrate inflate composition)
- ``runtime_dep_closure``: torch + brotli only (HNeRV parity L4 ≤ 2 deps)
- ``export_format``: WZF01 monolithic single-zip-member ``0.bin``
- ``score_aware_loss``: ``WynerZivFrame0ScoreAwareLoss`` routes through the
  canonical ``score_pair_components`` per Catalog #164; trains motion +
  residual jointly to minimize PoseNet pose-error on the reconstructed pair
- ``bolt_on_loc_budget``: ``lane_class=substrate_engineering`` (HNeRV parity
  L7); the per-pair SE(3) + flow + residual + base-substrate composition is
  substrate engineering
- ``no_op_detector_planned``: emit/parse roundtrip preserves bytes byte-for-byte;
  archive payload is structurally consumed by every section of inflate.py
  (motion → warp; residual → add; base_sha → load base; mismatch → raise)

target_modes: ``contest_one_video_replay``, ``contest_generalized``
lane_class: ``substrate_engineering``
research_only: false (export-first, all 8 fields declared)
canary_status: ``independent_substrate`` (structurally distinct from HNeRV-family
  AND from sister DISCUS Wyner-Ziv)

Predicted score band (NOT a claim):
- contest-CUDA: 0.148-0.168 ``[mathematical-derivation; first-principles-bound]``
  (PR101 baseline 0.193 minus predicted Δ -0.025 to -0.045)

Cross-references
----------------

- Master memo: ``.omx/research/deep_math_geometry_manifolds_synthesis_20260514.md``
  §3.5 (Wyner-Ziv duality) + §6 D4 description + §9 Shannon 1959 vector R(D)
- Sister substrate: ``tac.substrates.wyner_ziv_cooperative_receiver`` (DISCUS)
- Canonical primitives: ``tac.codec.cooperative_receiver`` (Atick-Redlich
  cooperative-receiver MI-max loss; D4 imports only if available, otherwise
  uses the scorer-conditional rate term inline)
- Canonical scorer contract: ``tac.substrates.score_aware_common.score_pair_components``
- Canonical inflate runtime helpers: ``tac.substrates._shared.inflate_runtime``
- Canonical trainer skeleton: ``tac.substrates._shared.trainer_skeleton``

Lane: ``lane_d4_wyner_ziv_frame_0_substrate_20260514``
"""

from tac.substrates.d4_wyner_ziv_frame_0.architecture import (
    BASE_SHA_HEX_LEN,
    EVAL_HW,
    NUM_PAIRS,
    PER_PAIR_SE3_PARAMS,
    TOTAL_ARCHIVE_TARGET_BYTES_MAX,
    TOTAL_ARCHIVE_TARGET_BYTES_MIN,
    MotionModelMode,
    WynerZivFrame0Config,
    WynerZivFrame0Substrate,
)
from tac.substrates.d4_wyner_ziv_frame_0.archive import (
    WZF01_MAGIC,
    WZF01_SCHEMA_VERSION,
    WynerZivFrame0Archive,
    pack_archive,
    parse_archive,
)
from tac.substrates.d4_wyner_ziv_frame_0.frame0_synthesis import synthesize_frame_0
from tac.substrates.d4_wyner_ziv_frame_0.motion_model import (
    OpticalFlowField,
    SE3MotionParams,
    apply_optical_flow,
    apply_se3_motion,
)
from tac.substrates.d4_wyner_ziv_frame_0.residual_codec import (
    decode_residual_blob,
    encode_residual_blob,
)
from tac.substrates.d4_wyner_ziv_frame_0.score_aware_loss import (
    WynerZivFrame0LossWeights,
    WynerZivFrame0ScoreAwareLoss,
)

__all__ = [
    "BASE_SHA_HEX_LEN",
    "EVAL_HW",
    "MotionModelMode",
    "NUM_PAIRS",
    "OpticalFlowField",
    "PER_PAIR_SE3_PARAMS",
    "SE3MotionParams",
    "TOTAL_ARCHIVE_TARGET_BYTES_MAX",
    "TOTAL_ARCHIVE_TARGET_BYTES_MIN",
    "WZF01_MAGIC",
    "WZF01_SCHEMA_VERSION",
    "WynerZivFrame0Archive",
    "WynerZivFrame0Config",
    "WynerZivFrame0LossWeights",
    "WynerZivFrame0ScoreAwareLoss",
    "WynerZivFrame0Substrate",
    "apply_optical_flow",
    "apply_se3_motion",
    "decode_residual_blob",
    "encode_residual_blob",
    "pack_archive",
    "parse_archive",
    "synthesize_frame_0",
]
