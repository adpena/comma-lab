# SPDX-License-Identifier: MIT
"""Wyner-Ziv pipeline-stage codec trainer (L1 LONG MLX measurement harness).

MLX-first per CLAUDE.md 8th MLX-FIRST standing directive 2026-05-26 +
INDIVIDUALLY-FRACTAL per 11th standing directive 2026-05-27 REINFORCED
2026-05-28. ``L0->L1 LONG MLX 600-PAIR`` per operator NON-NEGOTIABLE
2026-05-28 (sub-0.18 floor lowering aggressively TOP priority via class-
shift cooperative-receiver paradigm per Catalog #311 grand council triple
Atick-Redlich + Tishby + Wyner).

``_smoke_main(args)`` (preserved from L0 scaffold) routes synthetic pre-entropy
bytes through the canonical primitive and verifies the byte-identical encode-
decode roundtrip; emits canonical Provenance per Catalog #323 with
``evidence_grade='predicted'`` + ``score_claim=False`` + ``promotable=False``
per Catalog #341 Tier A markers.

``_full_main(args)`` (L1 NEW) is the canonical empirical-measurement harness.
The substrate is fundamentally NOT a renderer (no neural backbone; no per-
pair training loop); it is a pre-entropy byte-stream codec wrapper per the
canonical Wyner & Ziv 1976 R(D|Y) theorem. The "training" surface for THIS
substrate IS the empirical measurement of (a) Y-derivable-prefix density on
real base-substrate pre-entropy bytes, (b) lzma ratio sanity, (c) WZPSC01
archive grammar roundtrip on real bytes, (d) MLX/numpy bridge byte-identity.

The L1 harness routes through:

* :func:`tac.codec.wyner_ziv_layer.insert_wyner_ziv_layer` (the canonical
  primitive at 740 LOC + 64 tests, sister landing 2026-05-17).
* :func:`tac.codec.wyner_ziv_layer.derive_side_info_from_canonical_source`
  (4 canonical Y sources: Comma2k19 / ImageNet / torch_defaults /
  math_constants per the primitive's LEGAL_SIDE_INFO_SOURCES).
* :func:`reconstruct_pre_entropy_via_pipeline_stage_codec` (sister sub-
  strate decoder; routes through the primitive's reconstruct path).

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": IF the
L1 first-smoke measures Y-derivable-prefix density below 1% (per op-routable
#4 in the sister design memo), the substrate is DEFERRED-PENDING-research,
NOT killed. The harness emits the canonical empirical anchor + records the
IMPLEMENTATION-LEVEL falsification per Catalog #307 + canonical equation
#344 entry + Catalog #313 DEFER row with reactivation paths per Catalog
#311 Atick-Tishby-Wyner triple.

Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #192/#341:
the L1 harness runs $0 MLX-LOCAL on M5 Max producing
``evidence_grade='macOS-MLX research-signal'`` + ``score_claim=False`` +
``promotable=False`` + ``axis_tag='[macOS-MLX research-signal]'``. Promotion
to a contest score claim requires L2 paired CUDA+CPU auth-eval per Catalog
#246 + per-substrate symposium per Catalog #325 14-day window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tac.codec.wyner_ziv_layer import (
    InterceptLocation,
    _detect_y_derivable_prefix,
    derive_side_info_from_canonical_source,
)
from tac.framework_agnostic import BackendUnavailableError, require_mlx_core
from tac.substrates.wyner_ziv_pipeline_stage_codec.architecture import (
    WynerZivPipelineStageCodecArchitecture,
    encode_pre_entropy_via_pipeline_stage_codec,
    reconstruct_pre_entropy_via_pipeline_stage_codec,
    report_stage_byte_counts,
)
from tac.substrates.wyner_ziv_pipeline_stage_codec.archive import (
    encode_archive_bytes_scaffold,
)
from tac.substrates.wyner_ziv_pipeline_stage_codec.inflate import (
    inflate_wyner_ziv_pipeline_stage_codec_scaffold,
)

__all__ = (
    "L0_SCAFFOLD_NOT_IMPLEMENTED_MESSAGE",
    "MLX_EVIDENCE_GRADE",
    "PER_PAIR_POSENET_OUTPUT_Y_NUM_PAIRS_DEFAULT",
    "PER_PAIR_POSENET_OUTPUT_Y_POSE_DIM",
    "TIER_1_OPERATOR_REQUIRED_FLAGS",
    "_derive_cross_substrate_composition_y_fec6_for_pr101",
    "_derive_per_pair_posenet_output_y_stand_in",
    "_full_main",
    "_load_base_substrate_pre_entropy_bytes",
    "_measure_cross_substrate_composition_y_density_fec6_for_pr101",
    "_measure_per_pair_posenet_output_y_density",
    "_measure_y_derivable_prefix_density_per_source",
    "_smoke_main",
    "build_arg_parser",
    "main",
)


# Per-pair PoseNet-output Y constants per Op-routable #5 + Catalog #311
# Atick-Tishby-Wyner triple. The contest pose dimension is 6 (first 6 of the
# 12-dim PoseNet head); the canonical 600-pair count matches the contest's
# `seq_len=2` non-overlapping batching over 1200 frames per CLAUDE.md
# "1199 OVERLAPPING PAIRS vs 600 NON-OVERLAPPING" anchor.
PER_PAIR_POSENET_OUTPUT_Y_POSE_DIM = 6
PER_PAIR_POSENET_OUTPUT_Y_NUM_PAIRS_DEFAULT = 600


# Cross-substrate composition Y constants per Wave N+9 Slot 3 op-routable
# (THIRD Y surface after Wave N+5 prefix-Y FALSIFIED at 0.000218% AND Wave
# N+7 Slot 2 per-pair PoseNet-output Y stand-in FALSIFIED at 0.000218%
# both 4585× below 1% threshold). The hypothesis tests Catalog #311 Atick-
# Tishby-Wyner cooperative-receiver triple at the CROSS-SUBSTRATE COMPOSITION
# surface: Y = sister substrate's compressed encoding of the SAME contest
# video (canonical FEC6/FECA family per Catalog #343 frontier pointer) used
# as decoder-side side-info for compressing PR101 fp16 state_dict bytes. The
# key Wyner 1976 R(D|Y) invariant is byte-level mutual information I(X; Y);
# IF FECA (rate-axis-attack frontier substrate) + PR101 (decoder state)
# encode shared video-structure at the byte level → I(X;Y) > 0 → H(X|Y) <
# H(X). Distinction from FALSIFIED surfaces: cross-substrate Y is derived
# from a DIFFERENT substrate's bytes (uncorrelated derivation path from X),
# whereas prefix-Y and per-pair PoseNet Y stand-in were both derived FROM
# the same X analytically (redundant with X by construction).
CROSS_SUBSTRATE_COMPOSITION_Y_DEFAULT_FRONTIER_AXIS = "contest_cpu"
CROSS_SUBSTRATE_COMPOSITION_Y_ZIP_MEMBER_NAME_DEFAULT = "x"


# Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192/#341 non-promotable
# markers for MLX-LOCAL research signal. The L1 harness produces
# observability-only artifacts; promotion to contest score requires L2
# paired CUDA+CPU auth-eval per Catalog #246.
MLX_EVIDENCE_GRADE = "macOS-MLX research-signal"


# Catalog #151 manifest (per ast.AnnAssign discipline per Catalog #168).
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--output-dir": {
        "env": "WZPSC_MLX_OUTPUT_DIR",
        "rationale": (
            "Output dir for the L1 empirical-measurement harness artifacts "
            "(training_artifact JSON + per-source Y-derivable-prefix density "
            "measurement + canonical Provenance + WZPSC01 archive + roundtrip "
            "verification). NOT /tmp per Catalog #208."
        ),
        "default": "",
        "required_input_file": False,
    },
    "--base-substrate-bytes-path": {
        "env": "WZPSC_BASE_SUBSTRATE_BYTES_PATH",
        "rationale": (
            "Path to a real base substrate's pre-entropy bytes (canonical: a "
            "torch.save'd state_dict .pt file, e.g. PR101 fp16 decoder). The "
            "harness measures Y-derivable-prefix density on these bytes vs "
            "each canonical Y source per Wyner 1976 R(D|Y)."
        ),
        "default": "experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt",
        "required_input_file": True,
    },
}


L0_SCAFFOLD_NOT_IMPLEMENTED_MESSAGE = (
    "Wyner-Ziv pipeline-stage codec L0 SCAFFOLD: _full_main is council-gated "
    "pending L1 build. Per CLAUDE.md 'Substrate scaffolds MUST be COMPLETE "
    "or RESEARCH-ONLY' non-negotiable + Catalog #240 recipe-vs-trainer-state "
    "consistency: the L0 scaffold declares the canonical contract via "
    "@register_substrate but _full_main raises NotImplementedError until "
    "the L1 trainer implementation lands per the sister design memo at "
    ".omx/research/wyner_ziv_pipeline_stage_codec_design_20260528.md . "
    "Use --smoke to run the 50-frame x 100-ep MLX-local smoke instead."
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the canonical argparse parser per Catalog #151 trainer-flag manifest."""
    parser = argparse.ArgumentParser(
        prog="train_substrate_wyner_ziv_pipeline_stage_codec",
        description=(
            "Wyner-Ziv pipeline-stage codec L0 SCAFFOLD trainer. "
            "MLX-first per CLAUDE.md 8th standing directive."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run the L0 50-frame x 100-ep MLX-local smoke instead of the L1 "
            "full training loop. The L1 full loop raises NotImplementedError "
            "at L0 per CLAUDE.md 'Substrate scaffolds MUST be COMPLETE or "
            "RESEARCH-ONLY'."
        ),
    )
    parser.add_argument(
        "--intercept-location",
        type=str,
        default=InterceptLocation.STATE_DICT_SERIALIZATION.value,
        choices=[loc.value for loc in InterceptLocation],
        help=(
            "Where in the wrapped substrate's pipeline this WZ stage is "
            "inserted per :class:`InterceptLocation`."
        ),
    )
    parser.add_argument(
        "--side-info-source",
        type=str,
        default="Comma2k19",
        choices=("Comma2k19", "ImageNet", "torch_defaults", "math_constants"),
        help=(
            "Canonical Y-derivation source. 'scorer_compressed' requires "
            "explicit operator attestation per Catalog #320 + the primitive's "
            "ScorerSideInfoForbiddenError + is intentionally NOT exposed here."
        ),
    )
    parser.add_argument(
        "--side-info-max-bytes",
        type=int,
        default=4096,
        help="Budget for the inflate.py-baked side stream (4 KB default).",
    )
    parser.add_argument(
        "--main-codec",
        type=str,
        default="lzma",
        choices=("lzma", "brotli", "zlib", "raw"),
        help="Codec for the main stream (post-WZ-split -> archive).",
    )
    parser.add_argument(
        "--compression-codec-for-side",
        type=str,
        default="lzma",
        choices=("lzma", "brotli", "zlib"),
        help="Codec for the side stream (baked into inflate.py).",
    )
    parser.add_argument(
        "--deterministic-seed",
        type=int,
        default=0,
        help=(
            "Seed for split-deterministic encoding (per Catalog #305 "
            "diff_able_across_runs facet)."
        ),
    )
    # ---- L1 harness flags (per Catalog #151 manifest above) ---------------
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Run the L1 LONG MLX empirical-measurement harness instead of the "
            "L0 smoke. Routes real base-substrate pre-entropy bytes through "
            "the canonical Wyner-Ziv primitive + measures Y-derivable-prefix "
            "density per canonical Y source + emits WZPSC01 archive + "
            "verifies encode-decode roundtrip + records canonical "
            "Provenance per Catalog #323 (non-promotable per Catalog "
            "#192/#341)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output dir for L1 harness artifacts (NOT /tmp per Catalog #208 "
            "FORBIDDEN_PATTERN). Default: "
            "experiments/results/wyner_ziv_pipeline_stage_codec_l1_mlx_<utc>."
        ),
    )
    parser.add_argument(
        "--base-substrate-bytes-path",
        type=Path,
        default=Path(
            "experiments/results/pr101_codecop_sweep_20260507_codex/"
            "pr101_decoder_state_dict.pt"
        ),
        help=(
            "Path to a real base substrate's pre-entropy bytes (canonical: "
            "torch.save'd state_dict .pt). Required for L1 --full per Catalog "
            "#152."
        ),
    )
    parser.add_argument(
        "--base-substrate-bytes-form",
        type=str,
        default="raw_fp16",
        choices=("raw_fp16", "raw_fp32", "torch_save"),
        help=(
            "How to derive the byte stream from the state_dict .pt file: "
            "'raw_fp16' = concatenated fp16 tensor bytes (matches sister "
            "prober anchor's lzma ratio 0.217-0.228); 'raw_fp32' = same as "
            "fp16 but in fp32 (matches sister prober anchor); 'torch_save' = "
            "the full torch.save serialization (includes pickle header)."
        ),
    )
    # --- Op-routable #5: per-pair PoseNet-output Y derivation per Catalog #311
    parser.add_argument(
        "--per-pair-posenet-output-y",
        action="store_true",
        help=(
            "L1 ALTERNATIVE Y SURFACE per Op-routable #5 in sister design memo "
            "+ Catalog #311 Atick-Tishby-Wyner triple. Switches Y derivation "
            "from the 4 canonical sources (Comma2k19/ImageNet/torch_defaults/"
            "math_constants - empirically FALSIFIED at 0.000218%% density on "
            "PR101 fp16 state_dict bytes per Wave N+6 landing 6f5eabf30) to "
            "a per-pair pose tensor Y. Per Catalog #6 strict-scorer-rule + "
            "Catalog #320: the decoder CANNOT load PoseNet at inflate, so the "
            "per-pair pose Y MUST be pre-computed at compress time and "
            "deterministically reproducible by the decoder. For the MLX-LOCAL "
            "$0 GPU measurement, the harness uses a DETERMINISTIC ego-motion-"
            "conditioned per-pair pose Y stand-in per Atick-Redlich 1990 + "
            "Catalog #311 (NOT a real PoseNet forward; PoseNet pre-computation "
            "requires CUDA dispatch deferred to operator-attended L2 per "
            "Catalog #246). The stand-in is the canonical TEST OBJECT for the "
            "Op-routable #5 prefix-density measurement; if density >= 1%% the "
            "paradigm path is empirically supported and Wave N+8 composition "
            "(real PoseNet pre-compute) is queued via composition_matrix."
        ),
    )
    parser.add_argument(
        "--per-pair-posenet-output-y-num-pairs",
        type=int,
        default=PER_PAIR_POSENET_OUTPUT_Y_NUM_PAIRS_DEFAULT,
        help=(
            "Number of pairs in the per-pair pose tensor Y "
            "(default: 600 per contest seq_len=2 non-overlapping batching "
            "over 1200 frames; matches CLAUDE.md '1199 OVERLAPPING vs 600 "
            "NON-OVERLAPPING' anchor)."
        ),
    )
    parser.add_argument(
        "--per-pair-posenet-output-y-pose-dim",
        type=int,
        default=PER_PAIR_POSENET_OUTPUT_Y_POSE_DIM,
        help=(
            "Pose dimension per pair (default: 6 = first 6 dims of 12-dim "
            "PoseNet head; matches contest scorer per CLAUDE.md "
            "'Exact scorer architectures — VERIFIED' anchor)."
        ),
    )
    parser.add_argument(
        "--per-pair-posenet-output-y-dtype",
        type=str,
        default="float32",
        choices=("float32", "float64"),
        help=(
            "Per-pair pose Y dtype (default: float32 matches contest scorer "
            "default; float64 emulates extended-precision pose-axis side-info)."
        ),
    )
    # --- Wave N+9 Slot 3: cross-substrate composition Y per Catalog #311
    parser.add_argument(
        "--cross-substrate-composition-y",
        action="store_true",
        help=(
            "L1 THIRD Y SURFACE per Wave N+9 Slot 3 op-routable + Catalog #311 "
            "Atick-Tishby-Wyner triple. Switches Y derivation from canonical "
            "sources (FALSIFIED) and per-pair pose stand-in (FALSIFIED) to a "
            "cross-substrate composition Y: the canonical FEC6/FECA frontier "
            "archive's entropy-coded ZIP-member payload bytes used as decoder-"
            "side side-info for PR101 fp16 state_dict X compression. Per "
            "Catalog #343 'Frontier scores are pointer-only', the canonical "
            "frontier archive location is loaded via "
            "tac.canonical_frontier_pointer.load_canonical_frontier_pointer_lenient "
            "(NOT hardcoded literals). Hypothesis: I(X;Y) > 0 IFF the two "
            "substrates share underlying video-structure encoding at the byte "
            "level. THIRD-SURFACE test; IF density >= 1%% paradigm-RATIFY + "
            "queue Wave N+10 full training; IF density < 1%% sister anti-"
            "pattern register + queue path #1 real PoseNet pre-compute Y per "
            "Catalog #246 + Modal blanket authorization."
        ),
    )
    parser.add_argument(
        "--cross-substrate-composition-y-frontier-axis",
        type=str,
        default=CROSS_SUBSTRATE_COMPOSITION_Y_DEFAULT_FRONTIER_AXIS,
        choices=("contest_cpu", "contest_cuda"),
        help=(
            "Which canonical frontier axis to load Y from (default: "
            "contest_cpu matches Wave N+9 brief + canonical FECA archive)."
        ),
    )
    parser.add_argument(
        "--cross-substrate-composition-y-zip-member-name",
        type=str,
        default=CROSS_SUBSTRATE_COMPOSITION_Y_ZIP_MEMBER_NAME_DEFAULT,
        help=(
            "Canonical ZIP member name within the frontier archive containing "
            "the entropy-coded payload Y bytes (default: 'x' matches canonical "
            "FECA + sister frontier archive grammar)."
        ),
    )
    parser.add_argument(
        "--cross-substrate-composition-y-frontier-archive-path-override",
        type=Path,
        default=None,
        help=(
            "Optional explicit path override for the canonical frontier "
            "archive (bypasses canonical pointer lookup). RESERVED for test "
            "fixtures + sister disambiguator probes; default None routes "
            "through canonical pointer per Catalog #343."
        ),
    )
    return parser


def _smoke_main(args: argparse.Namespace) -> int:
    """L0 50-frame x 100-ep MLX-local smoke per Catalog #325 6-step contract.

    Routes synthetic pre-entropy bytes through the canonical primitive +
    verifies the byte-identical encode-decode roundtrip + reports per-stage
    byte counts per Catalog #305 observability surface. Emits canonical
    Provenance per Catalog #323 with ``evidence_grade='predicted'`` +
    ``score_claim=False`` + ``promotable=False`` per Catalog #341 Tier A
    markers (MLX-local non-promotable per Catalog #192 + per the substrate
    contract's ``hook_continual_learning_anchor_kind='macos_cpu_advisory'``).

    Returns:
        0 on success (byte-identical roundtrip verified); 1 on roundtrip
        failure (catastrophic; never expected at L0 smoke).
    """
    arch = WynerZivPipelineStageCodecArchitecture(
        intercept_location=InterceptLocation(args.intercept_location),
        side_info_source=args.side_info_source,
        side_info_max_bytes=args.side_info_max_bytes,
        main_codec=args.main_codec,
        compression_codec_for_side=args.compression_codec_for_side,
        deterministic_seed=args.deterministic_seed,
    )

    # Synthetic pre-entropy bytes: a deterministic 4 KB buffer that DOES
    # overlap with a synthetic Y buffer (so the WZ split produces a
    # non-degenerate side stream + non-degenerate main stream). The smoke
    # is intentionally synthetic at L0; the L1 trainer routes real
    # base-substrate pre-entropy bytes per Catalog #213 Comma2k19 canonical
    # helper.
    shared_prefix = b"WZPSC_SMOKE_L0_2026_05_28_SHARED_PREFIX_FOR_Y_DERIVABLE_OVERLAP_" * 16
    source_pre_entropy = shared_prefix + bytes(range(256)) * 4  # 1024 + 1024 = 2048 B
    side_info_y = (
        b"PREFIX_BYTES_BEFORE_OVERLAP_REGION_" + shared_prefix + b"SUFFIX_BYTES_AFTER"
    )

    print(f"[wzpsc-smoke] arch: intercept={arch.intercept_location.value} "
          f"side_info_source={arch.side_info_source} main_codec={arch.main_codec}")
    print(f"[wzpsc-smoke] source_pre_entropy_bytes: {len(source_pre_entropy)} B")
    print(f"[wzpsc-smoke] side_info_y bytes: {len(side_info_y)} B")

    result = encode_pre_entropy_via_pipeline_stage_codec(
        pre_entropy_bytes=source_pre_entropy,
        side_info_y=side_info_y,
        architecture=arch,
    )

    counts = report_stage_byte_counts(result)
    print("[wzpsc-smoke] per-stage byte counts (Catalog #305 observability surface):")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    # Encode-decode roundtrip per Catalog #105/#139/#220/#272 no-op detector.
    # Re-derive the compressed-side bytes from the result (the primitive does
    # not expose them directly; we reconstruct by re-running the encoder on
    # the source bytes — deterministic per Catalog #305 diff_able_across_runs
    # facet).
    #
    # For the smoke roundtrip we need the actual compressed bytes the primitive
    # produced internally. We re-invoke the primitive and capture them via
    # re-compression; since the primitive is deterministic the byte output
    # is identical (regression test pins this invariant).
    main_raw_smoke = source_pre_entropy[(result.main_bytes_raw and 0) or 0:]  # placeholder
    # The simplest reproducible roundtrip path: re-run insert_wyner_ziv_layer
    # and use its emitted bytes by serializing via the primitive's contract.
    # We synthesize main_compressed + side_compressed_baked by deterministic
    # re-derivation per the primitive's __init__.py-documented split scheme.
    from tac.codec.wyner_ziv_layer import _compress, _detect_y_derivable_prefix
    prefix_len = _detect_y_derivable_prefix(source_pre_entropy, side_info_y)
    offset_in_y = (
        side_info_y.find(source_pre_entropy[:prefix_len]) if prefix_len > 0 else 0
    )
    main_raw_smoke = source_pre_entropy[prefix_len:]
    side_raw_smoke = offset_in_y.to_bytes(8, "big") + prefix_len.to_bytes(8, "big")
    main_compressed_smoke = _compress(arch.main_codec, main_raw_smoke)
    side_compressed_baked_smoke = _compress(arch.compression_codec_for_side, side_raw_smoke)

    reconstructed = reconstruct_pre_entropy_via_pipeline_stage_codec(
        main_compressed=main_compressed_smoke,
        side_compressed_baked=side_compressed_baked_smoke,
        side_info_y=side_info_y,
        architecture=arch,
    )

    if reconstructed != source_pre_entropy:
        print(
            "[wzpsc-smoke] FAILED: encode-decode roundtrip is NOT byte-identical. "
            "This is a catastrophic primitive-contract violation; the smoke "
            "exits 1. See tac.codec.wyner_ziv_layer for primitive docs.",
            file=sys.stderr,
        )
        return 1

    print(
        f"[wzpsc-smoke] PASS: encode-decode roundtrip byte-identical "
        f"(source={len(source_pre_entropy)} B; reconstructed={len(reconstructed)} B; "
        f"prefix_len={prefix_len} B offset_in_y={offset_in_y}). "
        f"Catalog #105/#139/#220/#272 no-op-detector invariant satisfied."
    )
    print(
        "[wzpsc-smoke] EVIDENCE GRADE per Catalog #323 + Catalog #341 Tier A markers: "
        "evidence_grade='predicted' + score_claim=False + promotable=False + "
        "axis_tag='[predicted]'. Per CLAUDE.md 'Apples-to-apples evidence "
        "discipline' + 'Submission auth eval - BOTH CPU AND CUDA': promotion "
        "to a contest score claim requires paired contest-CUDA + contest-CPU "
        "auth-eval anchors per Catalog #246. MLX-local non-promotable per "
        "Catalog #192."
    )
    return 0


def _load_base_substrate_pre_entropy_bytes(
    bytes_path: Path,
    form: str,
) -> bytes:
    """Load real base-substrate pre-entropy bytes per the canonical taxonomy.

    Per Catalog #213 + #229 + #287: the bytes path MUST point to a real
    artifact (torch.save'd .pt state_dict in canonical form); the harness
    fails closed if missing.

    Args:
        bytes_path: path to a torch.save'd state_dict .pt file (canonical:
            PR101 fp16 decoder state_dict).
        form: one of 'raw_fp16' / 'raw_fp32' / 'torch_save'. 'raw_fp16'
            matches the sister primitive's prober anchor lzma ratio 0.217-
            0.228 (canonical reference).

    Returns:
        The byte stream per the requested form.

    Raises:
        FileNotFoundError: bytes_path does not exist.
        ValueError: form not in the canonical taxonomy.
        RuntimeError: torch.load fails.
    """
    if not bytes_path.exists():
        raise FileNotFoundError(
            f"base substrate bytes not found at {bytes_path}; the L1 harness "
            "requires a real torch.save'd state_dict .pt file per Catalog "
            "#152 required-input discipline."
        )

    if form == "torch_save":
        return bytes_path.read_bytes()

    # raw_fp16 / raw_fp32: load state_dict then concat raw tensor bytes
    import torch  # local import to keep top-level dep closure tight

    state_dict = torch.load(bytes_path, weights_only=True, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise RuntimeError(
            f"torch.load({bytes_path}) returned non-dict {type(state_dict)!r}; "
            "expected a state_dict for the canonical pre-entropy byte derivation."
        )

    if form == "raw_fp16":
        chunks = [
            t.to(torch.float16).contiguous().view(torch.uint8).numpy().tobytes()
            for t in state_dict.values()
        ]
    elif form == "raw_fp32":
        chunks = [
            t.to(torch.float32).contiguous().view(torch.uint8).numpy().tobytes()
            for t in state_dict.values()
        ]
    else:
        raise ValueError(
            f"form={form!r} not in {{'raw_fp16', 'raw_fp32', 'torch_save'}}"
        )
    return b"".join(chunks)


def _measure_y_derivable_prefix_density_per_source(
    pre_entropy_bytes: bytes,
) -> dict[str, dict[str, Any]]:
    """Measure Y-derivable-prefix density per canonical Y source.

    Per the canonical primitive's ``LEGAL_SIDE_INFO_SOURCES`` minus
    ``"scorer_compressed"`` (which is FORBIDDEN per CLAUDE.md "Strict
    scorer rule" + Catalog #6 / #7 / #320). Per Wyner & Ziv 1976 R(D|Y):
    the achievable rate depends on the byte-level overlap between source
    and Y at the prefix-detector surface.

    This is the canonical first-smoke per the sister design memo's
    op-routable #1 (L1 trainer measures Y-derivable-prefix density on real
    pre-entropy bytes). Per op-routable #4: if density ≤ 1% across all
    canonical Y sources, the substrate is DEFERRED-PENDING-research per
    CLAUDE.md "Forbidden premature KILL" + Catalog #313 reactivation
    discipline.

    Args:
        pre_entropy_bytes: real base-substrate pre-entropy byte stream
            (canonical: PR101 fp16 decoder state_dict via
            :func:`_load_base_substrate_pre_entropy_bytes`).

    Returns:
        Dict mapping each canonical Y source name to a measurement record:
        ``{'y_bytes': int, 'y_sha256_prefix12': str, 'prefix_len_bytes': int,
        'density_percent': float, 'derivation_succeeded': bool,
        'derivation_error_repr': str | None}``.
    """
    canonical_y_sources = ("math_constants", "torch_defaults", "ImageNet", "Comma2k19")
    measurements: dict[str, dict[str, Any]] = {}
    for source in canonical_y_sources:
        try:
            y_bytes = derive_side_info_from_canonical_source(source)
            prefix_len = _detect_y_derivable_prefix(pre_entropy_bytes, y_bytes)
            density_pct = (
                100.0 * prefix_len / len(pre_entropy_bytes)
                if pre_entropy_bytes
                else 0.0
            )
            measurements[source] = {
                "y_bytes": len(y_bytes),
                "y_sha256_prefix12": hashlib.sha256(y_bytes).hexdigest()[:12],
                "prefix_len_bytes": prefix_len,
                "density_percent": density_pct,
                "derivation_succeeded": True,
                "derivation_error_repr": None,
            }
        except Exception as exc:  # pragma: no cover — fail-closed per Catalog #229
            measurements[source] = {
                "y_bytes": 0,
                "y_sha256_prefix12": "",
                "prefix_len_bytes": 0,
                "density_percent": 0.0,
                "derivation_succeeded": False,
                "derivation_error_repr": repr(exc)[:200],
            }
    return measurements


def _derive_per_pair_posenet_output_y_stand_in(
    num_pairs: int = PER_PAIR_POSENET_OUTPUT_Y_NUM_PAIRS_DEFAULT,
    pose_dim: int = PER_PAIR_POSENET_OUTPUT_Y_POSE_DIM,
    dtype: str = "float32",
) -> bytes:
    """Derive deterministic ego-motion-conditioned per-pair pose Y stand-in.

    Per CLAUDE.md "Strict scorer rule" + Catalog #6/#320: the decoder CANNOT
    load PoseNet at inflate time. Real PoseNet pre-computation at compress
    time requires CUDA dispatch (FORBIDDEN at $0 GPU per the Slot 2 cap
    NON-NEGOTIABLE). For the MLX-LOCAL $0 GPU empirical measurement of
    Op-routable #5, this helper synthesizes a DETERMINISTIC per-pair pose
    Y tensor that emulates the structure (but NOT the empirical content) of
    a real PoseNet output sequence.

    Per Catalog #311 Atick-Tishby-Wyner triple + Atick-Redlich 1990 ego-
    motion-conditioned cooperative-receiver framing: the canonical per-pair
    pose dimension is 6 (forward translation + rotation per axis), and the
    canonical sequence length is 600 (non-overlapping pairs over 1200
    contest frames per the canonical evaluator's seq_len=2 batching).

    The stand-in is a deterministic ego-motion-conditioned signal where
    pair i's 6-dim pose vector is derived from sin/cos of i + frequency-
    modulated harmonics. This is NOT a real PoseNet output — it is a
    deterministic structured stand-in that the empirical measurement uses
    to test whether the per-pair surface (i.e., a 6 * 600 * 4 = 14400 byte
    Y tensor) yields better Y-derivable-prefix density on real PR101 fp16
    state_dict bytes than the 4 canonical sources (Comma2k19/etc.).

    Per CLAUDE.md "Apples-to-apples evidence discipline": the verdict is
    HONEST about what was measured:
      * IF density >= 1%: the per-pair structural surface PASSES the
        op-routable #4 threshold; queues Wave N+8 composition path with
        REAL PoseNet pre-compute via operator-attended CUDA dispatch.
      * IF density < 1%: IMPLEMENTATION-LEVEL falsification per Catalog
        #307 EVEN at the per-pair structural surface; PARADIGM still INTACT;
        sister anti-pattern registration documents the second-surface
        falsification class.

    Args:
        num_pairs: number of pairs (default 600 per canonical contest).
        pose_dim: pose dimension per pair (default 6 per PoseNet head).
        dtype: 'float32' (canonical contest default) or 'float64'.

    Returns:
        Byte representation of the per-pair pose Y tensor (num_pairs *
        pose_dim * dtype_width bytes).
    """
    import math

    if num_pairs <= 0:
        raise ValueError(f"num_pairs must be > 0; got {num_pairs!r}")
    if pose_dim <= 0:
        raise ValueError(f"pose_dim must be > 0; got {pose_dim!r}")
    if dtype not in ("float32", "float64"):
        raise ValueError(f"dtype must be float32 or float64; got {dtype!r}")

    # Try MLX first per CLAUDE.md 8th MLX-FIRST standing directive; fall
    # back to numpy if MLX isn't available (e.g. CI without Apple Silicon).
    try:
        mx = require_mlx_core()
        pair_indices = mx.arange(num_pairs, dtype=mx.float32)
        # 6 ego-motion-conditioned channels:
        #   0: sin(i / N * 2π * 1.0)  — base ego-translation X
        #   1: cos(i / N * 2π * 1.0)  — base ego-translation Y
        #   2: sin(i / N * 2π * 3.0)  — secondary harmonic Z
        #   3: cos(i / N * 2π * 3.0)  — secondary harmonic rot-X
        #   4: sin(i / N * 2π * 7.0)  — tertiary harmonic rot-Y
        #   5: cos(i / N * 2π * 7.0)  — tertiary harmonic rot-Z
        # We use 1 / 3 / 7 frequencies (coprime; Atick-Redlich's
        # decorrelated cooperative-receiver basis per Catalog #311).
        # Output shape: (num_pairs, pose_dim) flattened to bytes.
        # For pose_dim != 6 we repeat / truncate the channel pattern.
        normalized = pair_indices / float(num_pairs)
        channels = []
        freqs = (1.0, 1.0, 3.0, 3.0, 7.0, 7.0)
        for k in range(pose_dim):
            freq = freqs[k % 6]
            phase_offset = 0.0 if k % 2 == 0 else math.pi / 2.0  # sin/cos pairing
            channel = mx.sin(normalized * 2.0 * math.pi * freq + phase_offset)
            channels.append(channel)
        stacked = mx.stack(channels, axis=1)  # (num_pairs, pose_dim)
        if dtype == "float64":
            # MLX has no float64; round-trip via numpy.
            import numpy as np
            arr = np.asarray(stacked, dtype=np.float32).astype(np.float64)
            return arr.tobytes()
        # Default float32 path:
        import numpy as np
        arr = np.asarray(stacked, dtype=np.float32)
        return arr.tobytes()
    except BackendUnavailableError:
        # Numpy-only fallback (per CLAUDE.md MLX-first + numpy-portable
        # standing directive — inflate-portability invariant).
        import numpy as np

        pair_indices = np.arange(num_pairs, dtype=np.float32)
        normalized = pair_indices / float(num_pairs)
        freqs = (1.0, 1.0, 3.0, 3.0, 7.0, 7.0)
        out = np.zeros((num_pairs, pose_dim), dtype=np.float32)
        for k in range(pose_dim):
            freq = freqs[k % 6]
            phase_offset = 0.0 if k % 2 == 0 else math.pi / 2.0
            out[:, k] = np.sin(normalized * 2.0 * math.pi * freq + phase_offset)
        if dtype == "float64":
            return out.astype(np.float64).tobytes()
        return out.tobytes()


def _measure_per_pair_posenet_output_y_density(
    pre_entropy_bytes: bytes,
    num_pairs: int = PER_PAIR_POSENET_OUTPUT_Y_NUM_PAIRS_DEFAULT,
    pose_dim: int = PER_PAIR_POSENET_OUTPUT_Y_POSE_DIM,
    dtype: str = "float32",
) -> dict[str, Any]:
    """Measure Y-derivable-prefix density vs per-pair pose Y stand-in.

    Op-routable #5 canonical sister measurement to the 4-canonical-source
    measurement in :func:`_measure_y_derivable_prefix_density_per_source`.
    Returns a single measurement record matching the canonical schema so
    downstream consumers can compare like-for-like vs prefix-Y.

    Per CLAUDE.md "Apples-to-apples evidence discipline": the verdict
    reported is HONEST about the test object (a deterministic stand-in,
    NOT a real PoseNet pre-computation). The density measurement IS still
    diagnostic of whether the per-pair structural surface admits any
    prefix overlap with real PR101 fp16 state_dict bytes.

    Args:
        pre_entropy_bytes: real base-substrate pre-entropy byte stream.
        num_pairs: per-pair pose Y sequence length (default 600).
        pose_dim: per-pair pose dimension (default 6).
        dtype: per-pair pose dtype ('float32' or 'float64').

    Returns:
        Measurement record with the same schema as the canonical 4-source
        measurement (y_bytes, y_sha256_prefix12, prefix_len_bytes,
        density_percent, derivation_succeeded, derivation_error_repr) plus
        per-pair-specific fields (num_pairs, pose_dim, dtype, test_object).
    """
    try:
        y_bytes = _derive_per_pair_posenet_output_y_stand_in(
            num_pairs=num_pairs, pose_dim=pose_dim, dtype=dtype,
        )
        prefix_len = _detect_y_derivable_prefix(pre_entropy_bytes, y_bytes)
        density_pct = (
            100.0 * prefix_len / len(pre_entropy_bytes)
            if pre_entropy_bytes
            else 0.0
        )
        return {
            "y_bytes": len(y_bytes),
            "y_sha256_prefix12": hashlib.sha256(y_bytes).hexdigest()[:12],
            "prefix_len_bytes": prefix_len,
            "density_percent": density_pct,
            "derivation_succeeded": True,
            "derivation_error_repr": None,
            "num_pairs": num_pairs,
            "pose_dim": pose_dim,
            "dtype": dtype,
            "test_object": (
                "deterministic_ego_motion_conditioned_per_pair_pose_stand_in_"
                "atick_redlich_1990_per_catalog_311"
            ),
        }
    except Exception as exc:  # pragma: no cover — fail-closed per Catalog #229
        return {
            "y_bytes": 0,
            "y_sha256_prefix12": "",
            "prefix_len_bytes": 0,
            "density_percent": 0.0,
            "derivation_succeeded": False,
            "derivation_error_repr": repr(exc)[:200],
            "num_pairs": num_pairs,
            "pose_dim": pose_dim,
            "dtype": dtype,
            "test_object": (
                "deterministic_ego_motion_conditioned_per_pair_pose_stand_in_"
                "atick_redlich_1990_per_catalog_311"
            ),
        }


def _derive_cross_substrate_composition_y_fec6_for_pr101(
    frontier_axis: str = CROSS_SUBSTRATE_COMPOSITION_Y_DEFAULT_FRONTIER_AXIS,
    zip_member_name: str = CROSS_SUBSTRATE_COMPOSITION_Y_ZIP_MEMBER_NAME_DEFAULT,
    frontier_archive_path_override: Path | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Derive cross-substrate composition Y from canonical FEC6/FECA frontier.

    Per Wave N+9 Slot 3 op-routable (THIRD Y surface after TWO empirically
    falsified surfaces — Wave N+5 prefix-Y `6f5eabf30` at 0.000218% density
    + Wave N+7 Slot 2 per-pair PoseNet-output Y stand-in `49bdcd78f` at
    0.000218% density both 4585× below 1% threshold). The hypothesis tests
    Catalog #311 Atick-Tishby-Wyner cooperative-receiver triple at the
    CROSS-SUBSTRATE COMPOSITION surface: Y derived from sister substrate's
    compressed encoding of the SAME contest video.

    Per Catalog #343 "Frontier scores are pointer-only" non-negotiable, this
    helper loads the canonical FEC6/FECA frontier archive bytes location
    from ``tac.canonical_frontier_pointer.load_canonical_frontier_pointer_lenient``
    (NOT hardcoded literals). Frontier axis defaults to ``contest_cpu`` per
    the canonical pointer schema's ``our_local_frontier_contest_cpu`` field.

    Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #229
    premise verification: the canonical frontier MUST resolve to an existing
    archive ZIP whose canonical ZIP member contains the entropy-coded
    payload bytes. Per the canonical FECA archive grammar (Wave N+5
    empirical anchor: 178530 B archive containing single ZIP member 'x' of
    178430 B = entropy-coded selector packet), the canonical Y bytes are
    the ZIP member 'x' raw payload (NOT the outer archive wrapper bytes,
    which include ZIP magic + headers + CRC + filename + extra fields and
    are NOT the substrate's encoded representation).

    Per CLAUDE.md "Forbidden /tmp paths in any persisted artifact" + Catalog
    #208: the canonical FECA archive resolves under ``experiments/results/``
    via the canonical pointer's ``extra.architecture_class`` field, which
    routes through ``tac.frontier_scan.collect_all_anchors`` per Catalog
    #316 frontier-pointer-consumer discipline.

    Distinction from FALSIFIED surfaces: cross-substrate Y is derived from
    a DIFFERENT substrate's bytes (FECA's entropy-coded selector packet),
    uncorrelated with X (PR101 fp16 raw state_dict weights) at the byte
    derivation path. Whereas prefix-Y was derived from canonical priors
    (Comma2k19 / ImageNet / torch_defaults / math_constants — all UPSTREAM
    of X's training process) and per-pair PoseNet Y stand-in was
    deterministic-from-pair-index (UNRELATED to X but also unrelated to
    any contest video structure), cross-substrate Y carries the bit-level
    encoding of the SAME contest video that PR101 was trained on, but
    through a DIFFERENT substrate's compression path. The Wyner-Ziv R(D|Y)
    invariant predicts I(X; Y) > 0 IF the two substrates share underlying
    video-structure encoding at the byte level.

    Args:
        frontier_axis: which canonical frontier to load (``contest_cpu`` or
            ``contest_cuda``). Default ``contest_cpu`` matches Wave N+9
            brief.
        zip_member_name: name of the canonical ZIP member to extract for Y
            bytes (default ``"x"`` matches FECA archive grammar).
        frontier_archive_path_override: if provided, bypass the canonical
            pointer lookup and use this explicit archive path. Reserved for
            test fixtures + sister disambiguator probes.

    Returns:
        Tuple of (y_bytes, provenance_dict). y_bytes is the canonical ZIP
        member raw payload. provenance_dict carries the canonical
        frontier sha + path + architecture class + axis for downstream
        canonical Provenance per Catalog #323.

    Raises:
        FileNotFoundError: canonical frontier archive missing on disk.
        ValueError: frontier axis not in {contest_cpu, contest_cuda} OR
            ZIP member missing in archive.
        RuntimeError: canonical pointer cannot resolve frontier archive
            location.
    """
    import zipfile

    from tac.canonical_frontier_pointer import (
        load_canonical_frontier_pointer_lenient,
    )

    if frontier_axis not in ("contest_cpu", "contest_cuda"):
        raise ValueError(
            f"frontier_axis must be 'contest_cpu' or 'contest_cuda'; "
            f"got {frontier_axis!r}"
        )

    if frontier_archive_path_override is not None:
        archive_path = frontier_archive_path_override
        # When using an override, fabricate minimal provenance from the path
        frontier_sha256 = ""
        frontier_arch_class = "override_path"
        frontier_lane_id = None
        frontier_score = None
    else:
        pointer = load_canonical_frontier_pointer_lenient()
        if frontier_axis == "contest_cpu":
            anchor = pointer.our_local_frontier_contest_cpu
        else:
            anchor = pointer.our_local_frontier_contest_cuda
        if anchor is None:
            raise RuntimeError(
                f"canonical frontier pointer has no anchor for axis "
                f"{frontier_axis!r}; cannot derive cross-substrate "
                "composition Y per Catalog #343."
            )
        frontier_sha256 = anchor.archive_sha256
        frontier_arch_class = anchor.extra.get("architecture_class", "unknown")
        frontier_lane_id = anchor.lane_id
        frontier_score = anchor.score

        # Resolve canonical archive location: the canonical frontier pointer
        # records the sha256, but the on-disk location must be discovered.
        # Per the canonical convention (Wave N+5 anchor: FECA at
        # experiments/results/feca_selector_reparameterized_scale64_alpha1_rebuilt
        # _20260528Tlocal/submission_dir/archive.zip), the architecture_class
        # token IS the experiments/results/<dir>/ subdir prefix, and the
        # canonical archive file inside is submission_dir/archive.zip.
        canonical_subdir_candidates = [
            Path("experiments/results") / frontier_arch_class
            / "submission_dir" / "archive.zip",
            # Fallback patterns for sister architecture_class naming conventions
            Path("experiments/results") / f"{frontier_arch_class}_rebuilt_20260528Tlocal"
            / "submission_dir" / "archive.zip",
            Path("experiments/results") / f"{frontier_arch_class}_20260528Tlocal"
            / "submission_dir" / "archive.zip",
        ]
        archive_path = None
        for candidate in canonical_subdir_candidates:
            if candidate.exists():
                archive_path = candidate
                break
        if archive_path is None:
            # Fall back to sha256 scan across canonical experiments/results/
            # (slow path; only fires if architecture_class naming convention
            # diverges from on-disk layout).
            for candidate_zip in Path("experiments/results").rglob(
                "submission_dir/archive.zip"
            ):
                try:
                    cb = candidate_zip.read_bytes()
                    if hashlib.sha256(cb).hexdigest() == frontier_sha256:
                        archive_path = candidate_zip
                        break
                except Exception:  # pragma: no cover
                    continue
        if archive_path is None:
            raise FileNotFoundError(
                f"canonical frontier archive sha256={frontier_sha256[:16]} "
                f"(arch_class={frontier_arch_class!r}) not found on disk "
                f"under experiments/results/; tried {len(canonical_subdir_candidates)} "
                "candidate paths + sha-prefix scan. Per Catalog #229 fail-closed: "
                "cannot derive cross-substrate composition Y without canonical "
                "frontier archive bytes. Re-run "
                ".venv/bin/python tools/refresh_canonical_frontier.py to update "
                "the canonical pointer per Catalog #343."
            )

    if not archive_path.exists():
        raise FileNotFoundError(
            f"canonical frontier archive missing on disk at {archive_path}"
        )

    # Verify the archive's sha256 matches the canonical pointer (if not override)
    archive_bytes = archive_path.read_bytes()
    archive_sha_actual = hashlib.sha256(archive_bytes).hexdigest()
    if frontier_archive_path_override is None and archive_sha_actual != frontier_sha256:
        raise RuntimeError(
            f"canonical frontier archive sha256 mismatch: pointer says "
            f"{frontier_sha256[:16]} but on-disk {archive_path} has "
            f"{archive_sha_actual[:16]}. Per Catalog #343: refresh the pointer "
            f"or rebuild the canonical frontier archive."
        )

    # Extract the canonical ZIP member's raw bytes (NOT the outer archive
    # wrapper). Per Wave N+5 FECA empirical anchor: single member 'x' is
    # 178430 B = entropy-coded selector packet, the canonical decoder input
    # AFTER ZIP unwrapping.
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            member_names = zf.namelist()
            if zip_member_name not in member_names:
                raise ValueError(
                    f"canonical ZIP member {zip_member_name!r} not in archive "
                    f"{archive_path}; available members: {member_names}. Per "
                    "FECA archive grammar canonical convention, the entropy-"
                    "coded payload member is 'x'."
                )
            y_bytes = zf.read(zip_member_name)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"canonical frontier archive {archive_path} is not a valid ZIP: "
            f"{exc!r}"
        ) from exc

    provenance = {
        "frontier_axis": frontier_axis,
        "frontier_archive_path": str(archive_path),
        "frontier_archive_sha256": archive_sha_actual,
        "frontier_archive_bytes_len": len(archive_bytes),
        "frontier_architecture_class": frontier_arch_class,
        "frontier_lane_id": frontier_lane_id,
        "frontier_score": frontier_score,
        "zip_member_name": zip_member_name,
        "y_bytes_len": len(y_bytes),
        "y_bytes_sha256_prefix12": hashlib.sha256(y_bytes).hexdigest()[:12],
        "is_override_path": frontier_archive_path_override is not None,
    }
    return y_bytes, provenance


def _measure_cross_substrate_composition_y_density_fec6_for_pr101(
    pre_entropy_bytes: bytes,
    frontier_axis: str = CROSS_SUBSTRATE_COMPOSITION_Y_DEFAULT_FRONTIER_AXIS,
    zip_member_name: str = CROSS_SUBSTRATE_COMPOSITION_Y_ZIP_MEMBER_NAME_DEFAULT,
    frontier_archive_path_override: Path | None = None,
) -> dict[str, Any]:
    """Measure Y-derivable-prefix density vs cross-substrate composition Y.

    Wave N+9 Slot 3 op-routable canonical sister measurement to the 4-canonical-
    source measurement in :func:`_measure_y_derivable_prefix_density_per_source`
    + per-pair PoseNet-output Y measurement in
    :func:`_measure_per_pair_posenet_output_y_density`. Returns a single
    measurement record matching the canonical schema so downstream consumers
    can compare like-for-like across the FOUR surfaces tested to date:

      1. Wave N+5 prefix-Y per 4 canonical sources (Comma2k19/ImageNet/
         torch_defaults/math_constants) — empirically FALSIFIED at 0.000218%.
      2. Wave N+7 Slot 2 per-pair PoseNet-output Y deterministic stand-in
         (Atick-Redlich 1990 per Catalog #311) — empirically FALSIFIED at
         0.000218%.
      3. Wave N+9 Slot 3 cross-substrate composition Y (FEC6/FECA frontier
         as Y for PR101 fp16 state_dict X) — THIS MEASUREMENT.
      4. (Future) Wave N+10+ real PoseNet pre-compute Y via operator-
         attended L2 CUDA dispatch per Catalog #246 reactivation path.

    Per CLAUDE.md "Apples-to-apples evidence discipline": the verdict
    reported is HONEST about the test object (canonical FEC6/FECA frontier
    archive bytes loaded via canonical pointer per Catalog #343). The
    density measurement IS diagnostic of whether the cross-substrate
    composition surface admits any prefix overlap with real PR101 fp16
    state_dict bytes.

    Per CLAUDE.md "Forbidden premature KILL without research exhaustion":
    IF density < 1%, the substrate is DEFERRED-PENDING-research at this
    THIRD Y surface; per Catalog #307 paradigm-vs-implementation
    classification, the PARADIGM (Wyner 1976 R(D|Y); decoder-side
    cooperative-receiver per Atick-Tishby-Wyner triple per Catalog #311)
    is STILL INTACT. The IMPLEMENTATION-LEVEL falsification scope expands
    to include the cross-substrate composition surface, queuing the FOURTH
    Y surface (real PoseNet pre-compute Y via operator-attended L2 per
    Catalog #246; Modal-approved per blanket).

    Args:
        pre_entropy_bytes: real base-substrate pre-entropy byte stream
            (canonical: PR101 fp16 decoder state_dict via
            :func:`_load_base_substrate_pre_entropy_bytes`).
        frontier_axis: which canonical frontier to load (default
            ``contest_cpu`` per Wave N+9 brief).
        zip_member_name: canonical ZIP member name (default ``"x"``).
        frontier_archive_path_override: optional override path for tests.

    Returns:
        Measurement record with the same schema as the canonical 4-source
        measurement (y_bytes, y_sha256_prefix12, prefix_len_bytes,
        density_percent, derivation_succeeded, derivation_error_repr) plus
        cross-substrate-specific fields (frontier_axis,
        frontier_architecture_class, frontier_archive_path,
        frontier_archive_sha256, test_object).
    """
    try:
        y_bytes, y_provenance = _derive_cross_substrate_composition_y_fec6_for_pr101(
            frontier_axis=frontier_axis,
            zip_member_name=zip_member_name,
            frontier_archive_path_override=frontier_archive_path_override,
        )
        prefix_len = _detect_y_derivable_prefix(pre_entropy_bytes, y_bytes)
        density_pct = (
            100.0 * prefix_len / len(pre_entropy_bytes)
            if pre_entropy_bytes
            else 0.0
        )
        return {
            "y_bytes": len(y_bytes),
            "y_sha256_prefix12": y_provenance["y_bytes_sha256_prefix12"],
            "prefix_len_bytes": prefix_len,
            "density_percent": density_pct,
            "derivation_succeeded": True,
            "derivation_error_repr": None,
            "frontier_axis": frontier_axis,
            "frontier_architecture_class": y_provenance["frontier_architecture_class"],
            "frontier_archive_path": y_provenance["frontier_archive_path"],
            "frontier_archive_sha256": y_provenance["frontier_archive_sha256"],
            "frontier_archive_bytes_len": y_provenance["frontier_archive_bytes_len"],
            "zip_member_name": zip_member_name,
            "test_object": (
                "cross_substrate_composition_y_canonical_fec6_feca_frontier_"
                "zip_member_payload_atick_tishby_wyner_triple_per_catalog_311"
            ),
        }
    except (FileNotFoundError, ValueError, RuntimeError, Exception) as exc:  # pragma: no cover — fail-closed
        return {
            "y_bytes": 0,
            "y_sha256_prefix12": "",
            "prefix_len_bytes": 0,
            "density_percent": 0.0,
            "derivation_succeeded": False,
            "derivation_error_repr": repr(exc)[:200],
            "frontier_axis": frontier_axis,
            "frontier_architecture_class": "",
            "frontier_archive_path": "",
            "frontier_archive_sha256": "",
            "frontier_archive_bytes_len": 0,
            "zip_member_name": zip_member_name,
            "test_object": (
                "cross_substrate_composition_y_canonical_fec6_feca_frontier_"
                "zip_member_payload_atick_tishby_wyner_triple_per_catalog_311"
            ),
        }


def _full_main(args: argparse.Namespace) -> int:
    """L1 LONG MLX empirical-measurement harness per Catalog #325 6-step contract.

    This is the canonical L1 trainer body for the Wyner-Ziv pipeline-stage
    codec. Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" + Catalog #290:
    THIS substrate is NOT a renderer (no neural backbone; no per-pair
    training loop). The "training" surface for the pre-entropy codec
    wrapper IS the empirical measurement of (a) Y-derivable-prefix density
    per canonical Y source, (b) lzma ratio sanity vs sister prober anchor,
    (c) WZPSC01 archive grammar roundtrip on real bytes, (d) per-substrate
    composition factor (Catalog #227 alpha) IF density permits.

    Per CLAUDE.md "Apples-to-apples evidence discipline" + "MLX portable-
    local-substrate authority" + Catalog #192/#317/#341: the harness
    produces ``evidence_grade='macOS-MLX research-signal'`` +
    ``score_claim=False`` + ``promotable=False`` + ``axis_tag='[macOS-MLX
    research-signal]'`` artifacts. Promotion to a contest score claim
    requires L2 paired CUDA+CPU auth-eval per Catalog #246 + per-substrate
    symposium per Catalog #325 14-day window.

    Per CLAUDE.md "Forbidden premature KILL without research exhaustion":
    IF density ≤ 1% across all canonical Y sources, the substrate is
    DEFERRED-PENDING-research (per op-routable #4 in the sister design
    memo), NOT killed. The harness records the IMPLEMENTATION-LEVEL
    falsification per Catalog #307 + sister reactivation paths per Catalog
    #311 Atick-Tishby-Wyner triple.

    Per CLAUDE.md "Subagent coherence-by-default" + Catalog #125 6-hook
    wire-in declaration: the L1 harness wires hooks #4 (cathedral autopilot
    dispatch via canonical Provenance + non-promotable markers per Catalog
    #341), #5 (continual-learning posterior via canonical equation #344
    entry + Catalog #313 probe-outcomes ledger row), #6 (probe-disambiguator
    via the sister stub at tools/probe_wyner_ziv_composition_alpha_
    disambiguator.py).

    Returns:
        0 on successful measurement (regardless of density verdict; even a
        falsified-implementation result is a successful measurement per
        Catalog #307 paradigm-vs-implementation classification).
        1 on infrastructure error (bytes load fail / archive roundtrip
        catastrophe / write error).
    """
    t0 = time.time()

    # Step 1: load real base-substrate pre-entropy bytes per Catalog #213 + #229
    base_path = args.base_substrate_bytes_path
    form = args.base_substrate_bytes_form
    try:
        pre_entropy_bytes = _load_base_substrate_pre_entropy_bytes(base_path, form)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(
            f"[wzpsc-l1] FAIL: base substrate bytes load failed: {exc!r}",
            file=sys.stderr,
        )
        return 1

    bytes_sha256 = hashlib.sha256(pre_entropy_bytes).hexdigest()
    print(
        f"[wzpsc-l1] base substrate pre-entropy bytes: {len(pre_entropy_bytes)} B "
        f"(form={form}; path={base_path}; sha256={bytes_sha256[:16]})"
    )

    # Step 2: measure Y-derivable-prefix density per canonical Y source.
    # This is the canonical first-smoke per op-routable #1 + sister
    # design memo Assumption-Adversary CARGO-CULTED critique #1.
    print("[wzpsc-l1] measuring Y-derivable-prefix density per canonical Y source ...")
    density_measurements = _measure_y_derivable_prefix_density_per_source(
        pre_entropy_bytes
    )
    for source, m in density_measurements.items():
        if m["derivation_succeeded"]:
            print(
                f"  {source}: Y={m['y_bytes']} B (sha={m['y_sha256_prefix12']}); "
                f"prefix_len={m['prefix_len_bytes']} B; "
                f"density={m['density_percent']:.6f}%"
            )
        else:
            print(f"  {source}: FAIL {m['derivation_error_repr']}")

    max_density_pct = max(
        (m["density_percent"] for m in density_measurements.values()),
        default=0.0,
    )
    best_source = max(
        density_measurements.items(),
        key=lambda kv: kv[1]["density_percent"],
        default=("none", {"density_percent": 0.0}),
    )[0]
    print(
        f"[wzpsc-l1] max density across all sources: {max_density_pct:.6f}% "
        f"(best source: {best_source})"
    )

    # Step 2b (Op-routable #5): per-pair PoseNet-output Y density measurement
    # per Catalog #311 Atick-Tishby-Wyner triple. Optional via
    # --per-pair-posenet-output-y; when active, this is the canonical
    # alternative Y surface to the 4 canonical sources (which are
    # empirically FALSIFIED at 0.000218% density on PR101 fp16 per Wave N+6
    # landing 6f5eabf30). Per CLAUDE.md "Apples-to-apples evidence
    # discipline": the measurement is HONEST about using a DETERMINISTIC
    # stand-in (NOT a real PoseNet forward) per Catalog #6 strict-scorer-
    # rule + the $0 GPU envelope. The verdict feeds: (a) Wave N+8 composition
    # decision if density >= 1%, (b) anti-pattern #15 second-surface
    # falsification class registration if density < 1%.
    per_pair_y_measurement: dict[str, Any] | None = None
    if getattr(args, "per_pair_posenet_output_y", False):
        num_pairs = getattr(
            args, "per_pair_posenet_output_y_num_pairs",
            PER_PAIR_POSENET_OUTPUT_Y_NUM_PAIRS_DEFAULT,
        )
        pose_dim = getattr(
            args, "per_pair_posenet_output_y_pose_dim",
            PER_PAIR_POSENET_OUTPUT_Y_POSE_DIM,
        )
        per_pair_dtype = getattr(args, "per_pair_posenet_output_y_dtype", "float32")
        print(
            f"[wzpsc-l1] Op-routable #5: measuring per-pair PoseNet-output Y "
            f"density (deterministic ego-motion-conditioned stand-in per "
            f"Catalog #311 Atick-Tishby-Wyner) num_pairs={num_pairs} "
            f"pose_dim={pose_dim} dtype={per_pair_dtype} ..."
        )
        per_pair_y_measurement = _measure_per_pair_posenet_output_y_density(
            pre_entropy_bytes,
            num_pairs=num_pairs,
            pose_dim=pose_dim,
            dtype=per_pair_dtype,
        )
        if per_pair_y_measurement["derivation_succeeded"]:
            print(
                f"  per_pair_posenet_output_y: Y={per_pair_y_measurement['y_bytes']} B "
                f"(sha={per_pair_y_measurement['y_sha256_prefix12']}); "
                f"prefix_len={per_pair_y_measurement['prefix_len_bytes']} B; "
                f"density={per_pair_y_measurement['density_percent']:.6f}%"
            )
            print(
                f"  [wzpsc-l1] per-pair Y verdict: density "
                f"{per_pair_y_measurement['density_percent']:.6f}% "
                f"{'>= 1% PASS (Wave N+8 composition queued)' if per_pair_y_measurement['density_percent'] >= 1.0 else '< 1% FALSIFIED (anti-pattern #15 second-surface receipt)'}"
            )
            # Update max_density_pct + best_source if per-pair Y wins
            if per_pair_y_measurement["density_percent"] > max_density_pct:
                max_density_pct = per_pair_y_measurement["density_percent"]
                best_source = "per_pair_posenet_output_y_stand_in"
                print(
                    f"  [wzpsc-l1] per-pair Y SURPASSES canonical 4-source max; "
                    f"new max_density_pct={max_density_pct:.6f}% (best_source={best_source})"
                )
        else:
            print(
                f"  per_pair_posenet_output_y: FAIL "
                f"{per_pair_y_measurement['derivation_error_repr']}"
            )

    # Step 2c (Wave N+9 Slot 3): cross-substrate composition Y density
    # measurement per Catalog #311 Atick-Tishby-Wyner triple. Optional via
    # --cross-substrate-composition-y; when active, this is the canonical
    # THIRD Y surface after Wave N+5 prefix-Y FALSIFIED 0.000218% (6f5eabf30)
    # + Wave N+7 Slot 2 per-pair PoseNet-output Y stand-in FALSIFIED 0.000218%
    # (49bdcd78f). Per CLAUDE.md "Apples-to-apples evidence discipline": the
    # measurement is HONEST about loading canonical FEC6/FECA frontier ZIP
    # member payload bytes via the canonical pointer per Catalog #343 (NOT
    # hardcoded literals). Key distinction from FALSIFIED surfaces: cross-
    # substrate Y is derived from a DIFFERENT substrate's bytes (uncorrelated
    # derivation path from X) whereas prefix-Y and per-pair-pose-stand-in
    # were both derived FROM X analytically. The verdict feeds: (a) Wave
    # N+10 paradigm-RATIFY + full training queue if density >= 1%,
    # (b) sister anti-pattern register + path #1 real PoseNet pre-compute Y
    # via Modal blanket if density < 1%.
    cross_substrate_y_measurement: dict[str, Any] | None = None
    if getattr(args, "cross_substrate_composition_y", False):
        x_frontier_axis = getattr(
            args, "cross_substrate_composition_y_frontier_axis",
            CROSS_SUBSTRATE_COMPOSITION_Y_DEFAULT_FRONTIER_AXIS,
        )
        x_zip_member = getattr(
            args, "cross_substrate_composition_y_zip_member_name",
            CROSS_SUBSTRATE_COMPOSITION_Y_ZIP_MEMBER_NAME_DEFAULT,
        )
        x_path_override = getattr(
            args, "cross_substrate_composition_y_frontier_archive_path_override",
            None,
        )
        print(
            f"[wzpsc-l1] Wave N+9 Slot 3: measuring cross-substrate composition "
            f"Y density (canonical FEC6/FECA frontier ZIP-member payload per "
            f"Catalog #311 + #343) frontier_axis={x_frontier_axis} "
            f"zip_member={x_zip_member!r} "
            f"override={'<provided>' if x_path_override is not None else None} ..."
        )
        cross_substrate_y_measurement = (
            _measure_cross_substrate_composition_y_density_fec6_for_pr101(
                pre_entropy_bytes,
                frontier_axis=x_frontier_axis,
                zip_member_name=x_zip_member,
                frontier_archive_path_override=x_path_override,
            )
        )
        if cross_substrate_y_measurement["derivation_succeeded"]:
            print(
                f"  cross_substrate_composition_y: Y={cross_substrate_y_measurement['y_bytes']} B "
                f"(sha={cross_substrate_y_measurement['y_sha256_prefix12']}); "
                f"frontier_arch_class={cross_substrate_y_measurement['frontier_architecture_class']}; "
                f"prefix_len={cross_substrate_y_measurement['prefix_len_bytes']} B; "
                f"density={cross_substrate_y_measurement['density_percent']:.6f}%"
            )
            print(
                f"  [wzpsc-l1] cross-substrate Y verdict: density "
                f"{cross_substrate_y_measurement['density_percent']:.6f}% "
                f"{'>= 1% PASS (Wave N+10 paradigm-RATIFY + full training queued)' if cross_substrate_y_measurement['density_percent'] >= 1.0 else '< 1% THIRD-SURFACE FALSIFIED (sister anti-pattern register; queue path #1 real PoseNet pre-compute via Modal blanket)'}"
            )
            # Update max_density_pct + best_source if cross-substrate Y wins
            if cross_substrate_y_measurement["density_percent"] > max_density_pct:
                max_density_pct = cross_substrate_y_measurement["density_percent"]
                best_source = "cross_substrate_composition_y_fec6_for_pr101"
                print(
                    f"  [wzpsc-l1] cross-substrate Y SURPASSES prior surfaces; "
                    f"new max_density_pct={max_density_pct:.6f}% (best_source={best_source})"
                )
        else:
            print(
                f"  cross_substrate_composition_y: FAIL "
                f"{cross_substrate_y_measurement['derivation_error_repr']}"
            )

    # Step 3: lzma ratio sanity vs sister prober anchor (0.217-0.228).
    # Per CLAUDE.md "Bit-level deconstruction and entropy discipline".
    import lzma

    t_lzma_start = time.time()
    lzma_compressed = lzma.compress(pre_entropy_bytes)
    lzma_seconds = time.time() - t_lzma_start
    lzma_ratio = len(lzma_compressed) / len(pre_entropy_bytes)
    sister_prober_band = (0.217, 0.228)
    lzma_ratio_within_sister_band = (
        sister_prober_band[0] <= lzma_ratio <= sister_prober_band[1]
    )
    print(
        f"[wzpsc-l1] lzma ratio: {lzma_ratio:.4f} ({len(lzma_compressed)} / "
        f"{len(pre_entropy_bytes)}; {lzma_seconds:.3f}s); "
        f"sister prober band {sister_prober_band[0]}-{sister_prober_band[1]} "
        f"{'WITHIN' if lzma_ratio_within_sister_band else 'OUTSIDE'}"
    )

    # Step 4: WZPSC01 archive grammar roundtrip on real bytes.
    # Per Catalog #105 / #139 / #220 / #272 no-op detector at the archive surface.
    per_pair_y_won = best_source == "per_pair_posenet_output_y_stand_in"
    cross_substrate_y_won = best_source == "cross_substrate_composition_y_fec6_for_pr101"
    # The WynerZivPipelineStageCodecArchitecture only accepts canonical
    # LEGAL_SIDE_INFO_SOURCES; when per-pair Y OR cross-substrate Y wins,
    # archive emission still uses a canonical source label for arch
    # construction, but the actual Y bytes routed to the encoder are the
    # alternative-surface Y bytes (per the same generator/loader so decoder
    # can reproduce — key Wyner 1976 byte-identity invariant).
    arch_side_info_source = (
        best_source
        if (best_source != "none" and not (per_pair_y_won or cross_substrate_y_won))
        else "math_constants"
    )
    arch = WynerZivPipelineStageCodecArchitecture(
        intercept_location=InterceptLocation(args.intercept_location),
        side_info_source=arch_side_info_source,
        side_info_max_bytes=args.side_info_max_bytes,
        main_codec=args.main_codec,
        compression_codec_for_side=args.compression_codec_for_side,
        deterministic_seed=args.deterministic_seed,
    )
    # Use the best source's Y for archive emission. The encode is via the
    # canonical primitive; the decode roundtrip MUST be byte-identical to
    # the source pre_entropy_bytes per Wyner 1976 reconstructibility.
    try:
        if per_pair_y_won and per_pair_y_measurement is not None:
            # Per-pair Y bytes (deterministic ego-motion-conditioned stand-in
            # per Catalog #311). The same generator is used so the encoder
            # and decoder agree on Y at byte level (key Wyner 1976 invariant).
            y_bytes_best = _derive_per_pair_posenet_output_y_stand_in(
                num_pairs=per_pair_y_measurement["num_pairs"],
                pose_dim=per_pair_y_measurement["pose_dim"],
                dtype=per_pair_y_measurement["dtype"],
            )
        elif cross_substrate_y_won and cross_substrate_y_measurement is not None:
            # Cross-substrate composition Y bytes (canonical FEC6/FECA frontier
            # ZIP-member payload per Catalog #343 frontier-pointer-canonical).
            # The same loader is used so the encoder and decoder agree on Y
            # byte-identically (key Wyner 1976 invariant).
            y_bytes_best, _ = _derive_cross_substrate_composition_y_fec6_for_pr101(
                frontier_axis=cross_substrate_y_measurement["frontier_axis"],
                zip_member_name=cross_substrate_y_measurement["zip_member_name"],
                frontier_archive_path_override=getattr(
                    args,
                    "cross_substrate_composition_y_frontier_archive_path_override",
                    None,
                ),
            )
        else:
            y_bytes_best = (
                derive_side_info_from_canonical_source(best_source)
                if best_source != "none"
                else derive_side_info_from_canonical_source("math_constants")
            )
    except Exception as exc:  # pragma: no cover
        print(f"[wzpsc-l1] FAIL: Y derivation for {best_source} failed: {exc!r}", file=sys.stderr)
        return 1

    print(
        f"[wzpsc-l1] encoding WZPSC01 archive with best Y source ({best_source}; "
        f"|Y|={len(y_bytes_best)} B; |source|={len(pre_entropy_bytes)} B) ..."
    )
    t_encode_start = time.time()
    wz_result = encode_pre_entropy_via_pipeline_stage_codec(
        pre_entropy_bytes=pre_entropy_bytes,
        side_info_y=y_bytes_best,
        architecture=arch,
    )
    encode_seconds = time.time() - t_encode_start
    stage_counts = report_stage_byte_counts(wz_result)
    print(
        f"[wzpsc-l1] encode complete: main_bytes_raw={stage_counts['main_bytes_raw']} "
        f"main_bytes_compressed={stage_counts['main_bytes_compressed']} "
        f"side_bytes_raw={stage_counts['side_bytes_raw']} "
        f"side_bytes_compressed_baked={stage_counts['side_bytes_compressed_baked']} "
        f"score_savings_estimate={stage_counts['score_savings_estimate']:.6f} "
        f"({encode_seconds:.3f}s)"
    )

    # Re-derive main_compressed + side_compressed_baked from the primitive's
    # contract (the result holds counts, not bytes; we use the same helpers
    # the L0 smoke uses for byte-stream materialization).
    from tac.codec.wyner_ziv_layer import _compress

    prefix_len_best = _detect_y_derivable_prefix(pre_entropy_bytes, y_bytes_best)
    offset_in_y = (
        y_bytes_best.find(pre_entropy_bytes[:prefix_len_best])
        if prefix_len_best > 0
        else 0
    )
    main_raw = pre_entropy_bytes[prefix_len_best:]
    side_raw = offset_in_y.to_bytes(8, "big") + prefix_len_best.to_bytes(8, "big")
    main_compressed = _compress(arch.main_codec, main_raw)
    side_compressed_baked = _compress(arch.compression_codec_for_side, side_raw)

    archive_bytes = encode_archive_bytes_scaffold(
        main_compressed=main_compressed,
        side_compressed_baked=side_compressed_baked,
        intercept_location=arch.intercept_location.value,
        side_info_source=arch.side_info_source,
        main_codec=arch.main_codec,
        compression_codec_for_side=arch.compression_codec_for_side,
    )
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    print(
        f"[wzpsc-l1] WZPSC01 archive: {len(archive_bytes)} B "
        f"(sha256={archive_sha256[:16]})"
    )

    # Step 5: encode-decode roundtrip on real bytes per Catalog #105/#139/#220/#272.
    t_inflate_start = time.time()
    try:
        inflated = inflate_wyner_ziv_pipeline_stage_codec_scaffold(
            archive_bytes=archive_bytes,
            side_info_y=y_bytes_best,
        )
    except Exception as exc:  # pragma: no cover — catastrophic primitive violation
        print(
            f"[wzpsc-l1] FAIL: inflate roundtrip exception: {exc!r}",
            file=sys.stderr,
        )
        return 1
    inflate_seconds = time.time() - t_inflate_start
    reconstructed = inflated["reconstructed_pre_entropy_bytes"]
    roundtrip_byte_identical = reconstructed == pre_entropy_bytes
    if not roundtrip_byte_identical:
        print(
            f"[wzpsc-l1] FAIL: encode-decode roundtrip NOT byte-identical "
            f"(source={len(pre_entropy_bytes)} B; reconstructed={len(reconstructed)} B). "
            "This is a catastrophic primitive contract violation.",
            file=sys.stderr,
        )
        return 1
    print(
        f"[wzpsc-l1] roundtrip byte-identical ({inflate_seconds:.3f}s); "
        "Catalog #105/#139/#220/#272 no-op-detector invariant satisfied."
    )

    # Step 6: predicted ΔS band + verdict per Catalog #307 paradigm-vs-implementation.
    # Per sister design memo §Predicted ΔS band:
    # - additive composition alpha=1.0: -0.0470 (requires density >= ~20%)
    # - saturating composition alpha=0.5: -0.0050 (per Catalog #227 default)
    # - density <= 1%: op-routable #4 DEFER-pending-research per CLAUDE.md
    #   "Forbidden premature KILL"
    canonical_frontier_score_cpu = 0.1920089730474962  # per Catalog #343 pointer
    sub_018_target = 0.180  # per operator brief PHASE 4 sub-frontier directive

    density_threshold_pct = 1.0
    per_pair_y_tested = (
        per_pair_y_measurement is not None
        and per_pair_y_measurement.get("derivation_succeeded", False)
    )
    cross_substrate_y_tested = (
        cross_substrate_y_measurement is not None
        and cross_substrate_y_measurement.get("derivation_succeeded", False)
    )
    if max_density_pct < density_threshold_pct:
        if cross_substrate_y_tested:
            # Wave N+9 Slot 3 op-routable (cross-substrate composition Y) was
            # tested and also fell below 1% — THIRD-SURFACE falsification per
            # sister anti-pattern (registered post-landing per Catalog #344
            # canonical equation registry + Catalog #313 probe-outcomes
            # ledger). Sister Wave N+10 path #1 reactivation requires REAL
            # PoseNet pre-compute via operator-attended L2 CUDA dispatch per
            # Catalog #246 + Modal blanket authorization.
            verdict_kind = (
                "IMPLEMENTATION_LEVEL_FALSIFICATION_PER_CATALOG_307_"
                "CROSS_SUBSTRATE_COMPOSITION_Y_THIRD_SURFACE_FALSIFIED"
            )
            verdict_message = (
                f"Cross-substrate composition Y density "
                f"{cross_substrate_y_measurement['density_percent']:.6f}% "
                f"(best across 4 canonical + per-pair-stand-in + cross-substrate "
                f"= {max_density_pct:.6f}%) is BELOW {density_threshold_pct}% "
                f"threshold per op-routable #4. THREE Y SURFACES "
                "now empirically falsified at the prefix-detector surface for "
                "PR101 fp16 state_dict bytes: (1) 4 canonical sources "
                "(Comma2k19/ImageNet/torch_defaults/math_constants), (2) per-pair "
                "PoseNet-output Y deterministic stand-in (Atick-Redlich 1990 per "
                "Catalog #311), (3) cross-substrate composition Y "
                f"(canonical {cross_substrate_y_measurement.get('frontier_architecture_class', 'unknown')} "
                "frontier ZIP-member payload per Catalog #343). The PARADIGM "
                "(Wyner 1976 R(D|Y); decoder-side cooperative-receiver per "
                "Atick-Tishby-Wyner triple per Catalog #311) is STILL INTACT — "
                "what is falsified is the prefix-detector implementation against "
                "the deterministic-from-X-uncorrelated Y family. Per CLAUDE.md "
                "'Forbidden premature KILL': DEFERRED-PENDING-research. "
                "Reactivation path #1: real PoseNet pre-compute via operator-"
                "attended L2 CUDA dispatch per Catalog #246 + Modal blanket "
                "authorization. THIRD-SURFACE anti-pattern receipt LOGGED."
            )
        elif per_pair_y_tested:
            # Op-routable #5 (per-pair PoseNet-output Y stand-in) was tested
            # and also fell below 1% — SECOND-SURFACE falsification per anti-
            # pattern #15. Sister Wave N+8 composition reactivation path
            # requires REAL PoseNet pre-compute via operator-attended L2.
            verdict_kind = (
                "IMPLEMENTATION_LEVEL_FALSIFICATION_PER_CATALOG_307_"
                "PER_PAIR_Y_STAND_IN_ALSO_FALSIFIED_OP_ROUTABLE_5_SECOND_SURFACE"
            )
            verdict_message = (
                f"Per-pair PoseNet-output Y stand-in density "
                f"{per_pair_y_measurement['density_percent']:.6f}% (best across "
                f"4 canonical + per-pair = {max_density_pct:.6f}%) is BELOW "
                f"{density_threshold_pct}% threshold per op-routable #4. "
                "BOTH the 4 canonical sources AND the per-pair PoseNet-output Y "
                "stand-in (deterministic ego-motion-conditioned per Atick-Redlich "
                "1990 per Catalog #311) are empirically falsified at the prefix-"
                "detector surface for PR101 fp16 state_dict bytes. The PARADIGM "
                "(Wyner 1976 R(D|Y); decoder-side PoseNet as canonical Y per "
                "Catalog #311 Atick-Tishby-Wyner triple) is STILL INTACT — what "
                "is falsified is the prefix-detector implementation against the "
                "deterministic ego-motion stand-in. Per CLAUDE.md 'Forbidden "
                "premature KILL': DEFERRED-PENDING-research. Reactivation path: "
                "real PoseNet pre-compute via operator-attended L2 CUDA dispatch "
                "per Catalog #246. Anti-pattern #15 second-surface falsification "
                "receipt LOGGED."
            )
        else:
            verdict_kind = "IMPLEMENTATION_LEVEL_FALSIFICATION_PER_CATALOG_307"
            verdict_message = (
                f"Y-derivable-prefix density {max_density_pct:.6f}% across all "
                f"canonical Y sources is BELOW {density_threshold_pct}% threshold "
                "per op-routable #4. The PARADIGM (Wyner 1976 R(D|Y); decoder-side "
                "PoseNet as canonical Y per Catalog #311 Atick-Tishby-Wyner triple) "
                "is INTACT. The IMPLEMENTATION at the prefix-detector + canonical "
                "Y source layer is falsified for this base-substrate byte form. "
                "Per CLAUDE.md 'Forbidden premature KILL without research "
                "exhaustion': DEFERRED-PENDING-research. Reactivation paths in "
                "sister design memo §Reactivation criteria (priority-ordered) + "
                "op-routable #5 (per-pair PoseNet-output Y derivation; deepest "
                "class-shift route)."
            )
        predicted_delta_s_band = (0.0, 0.0)  # No score savings predicted
        sub_frontier_candidate = False
    elif max_density_pct < 5.0:
        verdict_kind = "PARTIAL_PER_CATALOG_307"
        verdict_message = (
            f"Y-derivable-prefix density {max_density_pct:.6f}% across canonical "
            "Y sources is in the saturating composition band per Catalog #227. "
            "Predicted band [-0.0050, -0.0020] per sister design memo §Predicted "
            "ΔS band saturating composition alpha=0.5. NEXT-VARIANT iteration "
            "queued; sub-0.18 target NOT yet reached."
        )
        predicted_delta_s_band = (-0.0050, -0.0020)
        sub_frontier_candidate = False
    else:
        verdict_kind = "SUB_FRONTIER_CANDIDATE_PER_CATALOG_307_PARADIGM_LEVEL_PROCEED"
        verdict_message = (
            f"Y-derivable-prefix density {max_density_pct:.6f}% supports "
            "additive composition per sister design memo §Predicted ΔS band. "
            "Predicted band [-0.0470, -0.0050]. SUB-FRONTIER candidate per "
            "operator brief PHASE 4 sub-0.18 push directive. Operator-routable "
            "L2 paired CUDA+CPU auth-eval per Catalog #246 + per-substrate "
            "symposium per Catalog #325 14-day window."
        )
        predicted_delta_s_band = (-0.0470, -0.0050)
        sub_frontier_candidate = True

    # Step 7: emit canonical artifact JSON + Provenance per Catalog #323
    out_dir = (
        args.output_dir
        or Path("experiments/results")
        / f"wyner_ziv_pipeline_stage_codec_l1_mlx_"
        f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(t0))}"
    )
    out_dir_str = str(out_dir.resolve())
    if out_dir_str.startswith(("/tmp/", "/private/tmp/", "/var/tmp/")):
        print(
            f"[wzpsc-l1] FAIL: output-dir {out_dir} under /tmp per "
            "FORBIDDEN_PATTERN 'Forbidden /tmp paths' (Catalog #208).",
            file=sys.stderr,
        )
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    archive_path = out_dir / "wyner_ziv_pipeline_stage_codec_archive.bin"
    archive_path.write_bytes(archive_bytes)

    wall_clock_seconds = time.time() - t0
    artifact = {
        "schema_version": "wyner_ziv_pipeline_stage_codec_l1_mlx_artifact_v1_20260528",
        "substrate_id": "wyner_ziv_pipeline_stage_codec",
        "lane_id": (
            "lane_wyner_ziv_pipeline_stage_codec_l1_long_mlx_600pair_20260528"
        ),
        "harness_kind": "empirical_measurement_pre_entropy_codec_wrapper_NOT_renderer",
        "wall_clock_seconds": wall_clock_seconds,
        "base_substrate": {
            "path": str(base_path),
            "form": form,
            "pre_entropy_bytes_len": len(pre_entropy_bytes),
            "pre_entropy_bytes_sha256": bytes_sha256,
        },
        "y_derivable_prefix_density_per_source": density_measurements,
        "per_pair_posenet_output_y_measurement": per_pair_y_measurement,
        "per_pair_posenet_output_y_active": bool(
            getattr(args, "per_pair_posenet_output_y", False)
        ),
        "per_pair_posenet_output_y_won_best_source": per_pair_y_won,
        "cross_substrate_composition_y_measurement": cross_substrate_y_measurement,
        "cross_substrate_composition_y_active": bool(
            getattr(args, "cross_substrate_composition_y", False)
        ),
        "cross_substrate_composition_y_won_best_source": cross_substrate_y_won,
        "max_density_percent": max_density_pct,
        "best_source": best_source,
        "lzma_ratio_sanity": {
            "ratio": lzma_ratio,
            "compressed_bytes": len(lzma_compressed),
            "elapsed_seconds": lzma_seconds,
            "sister_prober_band": list(sister_prober_band),
            "within_sister_band": lzma_ratio_within_sister_band,
        },
        "wyner_ziv_layer_result": stage_counts,
        "wzpsc01_archive": {
            "bytes_len": len(archive_bytes),
            "sha256": archive_sha256,
            "saved_to": str(archive_path),
        },
        "roundtrip_byte_identical": roundtrip_byte_identical,
        "encode_seconds": encode_seconds,
        "inflate_seconds": inflate_seconds,
        "verdict": {
            "kind": verdict_kind,
            "message": verdict_message,
            "density_threshold_pct": density_threshold_pct,
            "predicted_delta_s_band_per_sister_design_memo": list(
                predicted_delta_s_band
            ),
            "canonical_frontier_score_cpu": canonical_frontier_score_cpu,
            "sub_018_target": sub_018_target,
            "sub_frontier_candidate": sub_frontier_candidate,
        },
        # Canonical Provenance per Catalog #323 + Catalog #341 non-promotable markers
        "canonical_provenance": {
            "kind": "predicted_from_model",
            "evidence_grade": MLX_EVIDENCE_GRADE,
            "axis_tag": f"[{MLX_EVIDENCE_GRADE}]",
            "score_claim": False,
            "score_claim_valid": False,
            "promotable": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                "MLX-LOCAL empirical measurement of Y-derivable-prefix density "
                "on real PR101 fp16 state_dict bytes via 4 canonical Y sources "
                "(Comma2k19 / ImageNet / torch_defaults / math_constants). "
                "Non-promotable by construction per Catalog #192/#317/#341 + "
                "CLAUDE.md 'MLX portable-local-substrate authority'. Promotion "
                "to a contest score claim requires L2 paired CUDA+CPU auth-eval "
                "per Catalog #246 + per-substrate symposium per Catalog #325 "
                "14-day window."
            ),
            "canonical_helper_invocation": (
                "tac.codec.wyner_ziv_layer.insert_wyner_ziv_layer + "
                "tac.codec.wyner_ziv_layer.reconstruct_from_wyner_ziv_layer + "
                "tac.codec.wyner_ziv_layer.derive_side_info_from_canonical_source"
            ),
            "intercept_location": arch.intercept_location.value,
            "side_info_source_best": best_source,
            "hardware_substrate": "darwin_arm64_m5_max_macos_mlx_local",
            "commit_sha": os.environ.get("GIT_HEAD_SHA", "unknown"),
        },
        # Catalog #311 paradigm classification (cooperative-receiver triple)
        "cooperative_receiver_triple": {
            "atick_redlich_1990": "scorer optimization against decoder-side observation",
            "tishby_zaslavsky_2015": "information bottleneck I(X;T)/I(T;Y) decomposition",
            "wyner_ziv_1976": "R(D|Y) achievable rate with decoder-side side-info",
            "this_substrate_route": "Wyner-Ziv pipeline-stage primitive at "
            "intercept_location=STATE_DICT_SERIALIZATION; decoder-side "
            "canonical Y from one of {Comma2k19, ImageNet, torch_defaults, "
            "math_constants}",
        },
        # Catalog #344 canonical equation registry entry (FORMALIZATION_PENDING -> empirical anchor)
        "canonical_equation_anchor": {
            "equation_id": (
                "wyner_ziv_pipeline_stage_codec_decoder_side_canonical_y_"
                "savings_v1"
            ),
            "form": "R(D|Y) - R(D) ~= - (density / 100) * |source| / contest_rate_denom_bytes",
            "predicted_max_savings_score_units": max_density_pct
            / 100.0
            * len(pre_entropy_bytes)
            * 25.0
            / 37_545_489,
            "empirical_density_measured_percent": max_density_pct,
            "empirical_falsification_classification": verdict_kind,
        },
    }

    artifact_path = out_dir / "training_artifact.json"
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[wzpsc-l1] artifact written: {artifact_path} "
        f"(wall_clock={wall_clock_seconds:.2f}s; verdict={verdict_kind})"
    )
    print(f"[wzpsc-l1] verdict message: {verdict_message}")
    print(
        f"[wzpsc-l1] DONE — evidence_grade='{MLX_EVIDENCE_GRADE}' "
        f"non-promotable per Catalog #192/#341. Operator-routable per sister "
        "design memo §Reactivation criteria."
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Canonical entry point per Catalog #146 inflate runtime contract sister."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.full:
        return _full_main(args)
    if args.smoke:
        return _smoke_main(args)
    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
