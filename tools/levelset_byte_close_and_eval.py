#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""LEVEL-SET WITNESS byte-close -> full-output inflate -> realized d_seg/d_pose parity -> staged exact eval.

Sister of ``tools/witness_byte_close_and_eval.py`` (the RGB witness path), keyed on the
LEVEL-SET checkpoint (``levelset_witness_ema_mlx.npz`` produced by
``experiments/train_levelset_witness_realized_through_R_mlx.py``). The RGB tool is HARD-KEYED to
``params["out.weight"]`` (a single RGB head); the level-set witness has NO ``out.weight`` -- it
has ``out_sdf`` (K SDF fields) + ``out_tex`` (pose-carrying RGB texture) + ``palette`` (K,3) +
``code`` (per-(pair,frame) FiLM). So the RGB tool KeyErrors on a level-set npz; THIS tool is the
missing level-set exact-eval path -- THE row-enabler (per the FEED-df recursive review memo
``.omx/research/yousfi_levers_optimal_form_review_20260627T063335Z.md``: until it exists, ALL the
level-set d_seg descent is advisory-only and the pointer CANNOT move).

ONE command turns a trained level-set checkpoint into a self-contained contest packet and proves
it end-to-end ON CPU at $0:

  (a) archive.zip       -- int8+brotli LEARNED-payload blob (the MEASURED rate term st_size).
                           Matches ``lever_b_levelset_generator.quantize_levelset_blob`` accounting
                           (one brotli stream for the base weights + one for ``code``).
  (b) inflate.py        -- numpy-only (numpy + brotli + torch-for-R [+ scipy only when the
                           checkpoint used self-orient]) FULL-OUTPUT decoder: emits the complete
                           (2*n_pairs, 874, 1164, 3) uint8 ``.raw`` (frame0 AND frame1 per pair,
                           camera-res, no header) -- the evaluator's expected layout. Runs the ONE
                           CODEPATH (``levelset_rgb_forward_numpy``, inlined) + the torch R
                           (bicubic up to camera -> round -> uint8), byte-identical to the trainer
                           verdict (``_torch_R_to_camera_uint8``).
  (c) realized d_seg/d_pose on the INFLATED FRAMES (read back from the .raw) via the FROZEN
      CPU-torch SegNet/PoseNet vs GT -- NOT a field-level proxy, NOT a generator byte-repro.
  (d) the staged contest-CPU exact-eval command (NOT run here).

RULE 118 (FREE vs COUNTED -- the rate game):
  * FREE (regenerated in inflate.py from cfg, NOT in archive.zip): the curvelet/shearlet
    directional bank (a deterministic parametric polar frequency grid from the 5 scalars
    n_scales/n_orient0/f0/base/n_iso + max_freq) AND, when the checkpoint used self-orientation,
    the directional feats (a fixed-point of the DECODER'S OWN argmax -- GT-free, reconstructible).
    These are generic ALGORITHM, not video-derived -> 0 bytes (CLAUDE.md "inflate.py is a FREE
    interpreter").
  * COUNTED (int8+brotli in archive.zip): the LEARNED video-derived payload -- in_proj / film /
    hidden.* / out_sdf / out_tex / palette weights + the per-frame ``code`` table.

SELF-ORIENT (the mod-32 target uses it): when ``in_proj.weight`` has MORE input columns than the
curvelet bank produces, the checkpoint was trained with ``--self-orient`` (directional feats
concatenated to the curvelet feats). The trainer reconstructs those feats at deploy via a
FIXED-POINT on the decoder's own argmax (cos 0.89-0.91 vs GT, the byte-closeable -48% directional
lever). inflate.py reproduces the same fixed point -> GT-free, 0 extra bytes. PARITY-GAP CAVEAT
(loud, NO-FAKE): the deploy fixed point converges on the FINAL weights, whereas the trainer's
reported implied_S used the dir feats accumulated along the training TRAJECTORY (reorient every N
epochs). They are close (the fixed point is the design) but NOT bit-identical -> the realized
d_seg on the inflated frames is the TRUTH; any gap vs the trainer's number is itself a finding.
ALSO: the trainer does NOT persist the self-orient params (freq_across/freq_along/tau/iters) into
the npz (a trainer gap, flagged) -> they default to the trainer defaults here and are overridable
via ``--so-*`` (generic hyperparameters, rule-118 clean, a handful of manifest bytes).

POSE (the verdict this tool RESOLVES): the scorer computes
``d_pose = MSE(PoseNet(generated_pair)[:6], PoseNet(original_pair)[:6])`` ON THE FRAMES. A stored
6-scalar pose sidecar is just BYTES the scorer never reads -- it does NOT change the rendered
frames, so it does NOT lower realized d_pose. A POSE-BLIND render (the w_pose=0 mod-32 config)
stays d_pose ~O(100) regardless of any sidecar. Therefore the level-set ROW REQUIRES a
POSE-TRAINED render (``--w-pose > 0`` supervising the texture/code to hit the 6 PoseNet targets in
the SegNet-null space) -- exactly like the RGB witness carries pose in its per-(pair,frame) codes.
``--fold-pose-sidecar`` is provided for parity with the RGB tool but is OFF by default and, when
on, LOUDLY records that it adds COUNTED bytes WITHOUT lowering realized d_pose (the scorer reads
frames, not the sidecar).

AUTHORITY: ``[macOS-CPU advisory] NON-PROMOTABLE``. CPU only (no MPS, no CUDA, no paid eval). The
realized d_seg/d_pose here is the frozen CPU-torch mirror of evaluate.py over the measured pair
subset. The reported S is advisory until the SAME packet runs through ``upstream/evaluate.py`` on
contest-compliant Linux x86_64 CPU (the staged command). NO score/frontier/promotion claim;
pointer UNMOVED unless a real byte-closed sub-frontier exact-eval row lands.

Usage:
    .venv/bin/python tools/levelset_byte_close_and_eval.py \\
        --ckpt-dir experiments/results/<run> \\
        [--max-pairs 8] [--gt-cache ...gt_n6.npz] [--keep-packet] [--out reports/...json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import zipfile
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "src", _REPO / "experiments", _REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import train_witness_realized_through_R_mlx as twr  # noqa: E402  (frozen CPU-torch verdict + GT + R)

from tac.decode_memory_tier import (  # noqa: E402  (#224 Wave D decode memory-tier surface)
    DECODE_MEMORY_TIERS,
    DEFAULT_TIER_NAME,
    require_contest_tier,
    resolve_eval_device,
    resolve_tier,
    tier_inflate_env,
)

from tac.boundary_math.lever_b_levelset_generator import (  # noqa: E402
    CurveletBankConfig,
    _int8_symmetric,
    curvelet_directional_B,
    levelset_band_forward_numpy,
    levelset_rgb_forward_numpy,
    quantize_levelset_blob,
)
from tac.boundary_math.windowed_curvelet_frame import (  # noqa: E402
    WindowedCurveletConfig,
    windowed_curvelet_feats,
)
from tac.boundary_math.compact_shearlet_frame import (  # noqa: E402
    CompactShearletConfig,
    compact_shearlet_feats,
)
from tac.boundary_math.curvelet_placement import (  # noqa: E402
    fold_taper_into_in_proj_numpy,
    native_orientation_fixed_point_numpy,
    orientation_metadata_from_atom_specs,
    projective_jacobian_numpy,
    transform_normal_covector_numpy,
)
from tac.boundary_math.localized_basis_frames import (  # noqa: E402
    ATOM_SPECS as LITERAL_CURVELET_ATOM_SPECS,
    BasisProgramConfig,
    basis_features_numpy as literal_curvelet_feats_numpy,
    charted_fine_feats_cache_numpy,
    charted_pair_feats_numpy,
    generated_inflate_source as literal_curvelet_generated_source,
)
from tac.witness_dsl.basis_control import (  # noqa: E402
    COMPACT_SHEARLET,
    LEGACY_FOURIER_AB_CONTROL,
    LITERAL_POLAR_CURVELET,
    GENUINE_FRAME_FEATURE_WIDTH,
    genuine_frame_compact_shearlet_config,
    genuine_frame_windowed_curvelet_config,
    normalize_basis_family,
)

from tac.boundary_math.analytic_lane_render_band import (  # noqa: E402  (#224 Wave E canonical band)
    DEFAULT_DASH_FORWARD_MAX_M,
    LaneBandRenderConfig,
    build_lane_band_pairs_from_lstars,
    composite_band_on_render,
    deserialize_lane_band_any,  # Wave-F: LBND1/LBND2 magic dispatch
    lane_band_rd_rate_report,   # Wave-F: measured per-lever byte accounting
    rasterize_lane_coverage_range_dependent,
    render_config_from_header,
    serialize_lane_band,
    serialize_lane_band_any,    # Wave-F: format-preserving re-serialize (capped inflate)
    serialize_lane_band_rd,     # Wave-F: LBND2 optimal RD serializer
    serialize_lane_band_res,    # LBND4: RD grid + ξ delta/context residual entropy stage (default OFF)
    witness_uncertainty_mask,
)
from tac import contest_score as _cscore  # noqa: E402  (canonical S = 100*d_seg + sqrt(10*d_pose) + 25*rate helpers)
from tac.local_acceleration import torch_levelset_inflate as _tli  # noqa: E402  (canonical FREE-table regen)
from tac.boundary_math import warp_real_luma_frame0 as _wrl  # noqa: E402  (#205 pose carrier: warp-real-luma frame0)
from tac.boundary_math import xi_pose_coder as _xip  # noqa: E402  (#257 store-nothing derive-H + ξ entropy coder)
from tac.boundary_math import witness_crosstensor_codec as _wxc  # noqa: E402  (lossless joint weight/code storage)
from tac import witness_run_artifacts as _wra  # noqa: E402  (canonical run-artifact filename CONTRACT)
from tac.through_r.blind_coordinate import apply_blind_fill  # noqa: E402  (#401/D21a FREE fill)
from tac.codec.levelset_palette_residual import (  # noqa: E402
    EKPR1_APPLICATION,
    EKPR1_CODEC,
    apply_palette_residual,
    cap_palette_residual,
    decode_palette_residual,
    validate_palette_residual_binding,
)

# canonical FREE-table regen fns (rule-118 curvelet bank + self-orient dir feats) — the bit-exact
# oracle reference reuses these so the gate compares against the SAME free tables the inflate uses.
_canon_coords_grid = _tli.coords_grid
_canon_curvelet_B = _tli.curvelet_B
_canon_curvelet_feats = _tli.curvelet_feats
_canon_dir_feats = _tli.dir_feats

# (#328 clip_profile Phase-2, measured-no-regression) per-clip camera/rate constants sourced
# from the canonical MEASURED SoT tac.clip_profile (cache: .omx/state/clip_profiles/<sha>.json).
# value-provenance: MEASURED-ANCHOR (clip_profile cache) > HARDCODED-fallback. On 0.mkv the
# auto-measured values AGREE bit-exactly with the historical literals (asserted in
# test_clip_profile_rewire_byte_close), so this is BYTE-IDENTICAL; the fallback literals keep
# the score path standalone-runnable when the profile cache is absent. NOTE: the xi homography
# below reuses fx for BOTH axes (fx_native == fy_native == 910 on 0.mkv), matching the historical
# single-_XI_FX form. The lane-IPM v_horizon (174, swept-optimal #327) and lane cam-height (1.2)
# DISAGREE with the profile (175 median / 1.22) and are DELIBERATELY NOT sourced here — those are
# the two routed-to-reconciliation discrepancy findings (FEED-clipprofile2), not stale bugs.
try:
    from tac.clip_profile import for_video as _cp_for_video

    _CP = _cp_for_video("upstream/videos/0.mkv")
    CAMERA_H, CAMERA_W = int(_CP.camera.native_h), int(_CP.camera.native_w)
    RATE_DENOM = float(_CP.video_bytes)
    _CP_XI_FX = float(_CP.camera.fx_native)
    _CP_XI_CX = float(_CP.camera.cx_native)
    _CP_XI_CY = float(_CP.camera.cy_native)
    _CP_XI_D = float(_CP.device_height_m)
except Exception:  # standalone fallback (clip_profile cache absent) — documented literals
    CAMERA_H, CAMERA_W = 874, 1164
    RATE_DENOM = 37_545_489.0
    _CP_XI_FX, _CP_XI_CX, _CP_XI_CY, _CP_XI_D = 910.0, 582.0, 437.0, 1.22
_XI_CAMERA_CONTRACT_MARKER = "# __EMIT_HASH_BOUND_XI_CAMERA_CONTRACT__"
_MAGIC = b"LVLS1\x00"  # level-set softmax-of-SDF carrier v1
_PCAR_MAGIC = b"PCAR1\x00"  # #205 pose carrier: warp-real-luma frame0 (stored keyframe luma + per-pair homography)
_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")
_BLIND_COORDINATE_PROOF_SCHEMA = "blind_coordinate_proof.v1"
_BLIND_COORDINATE_RECEIVER_SCHEMA = "blind_coordinate_receiver_binding.v1"
_BLIND_COORDINATE_N_PAIRS = 600
_BLIND_COORDINATE_N_PX = 230_904


def validate_blind_coordinate_n600_receipt(path: Path) -> dict[str, Any]:
    """Fail closed unless ``path`` proves n600 bit identity through both scorer inputs."""

    receipt_path = Path(path)
    if not receipt_path.is_file():
        raise FileNotFoundError(
            f"D21a blind-coordinate fill requires an existing n600 proof receipt; missing {receipt_path}"
        )
    raw = receipt_path.read_bytes()
    try:
        receipt = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"D21a blind-coordinate receipt is not valid JSON: {receipt_path}") from exc
    if receipt.get("schema") != _BLIND_COORDINATE_PROOF_SCHEMA:
        raise ValueError(
            "D21a blind-coordinate receipt schema mismatch: "
            f"expected={_BLIND_COORDINATE_PROOF_SCHEMA}, got={receipt.get('schema')!r}"
        )
    fraction = receipt.get("blind_fraction")
    bit_identity = receipt.get("bit_identity_through_R")
    if not isinstance(fraction, dict) or not isinstance(bit_identity, dict):
        raise ValueError("D21a receipt needs blind_fraction and bit_identity_through_R objects")
    if fraction.get("schema") != "blind_coordinate_fraction.v1":
        raise ValueError("D21a receipt blind_fraction schema mismatch")
    if bit_identity.get("schema") != "blind_coordinate_bit_identity.v1":
        raise ValueError("D21a receipt bit_identity_through_R schema mismatch")
    expected = {
        "n_pairs": _BLIND_COORDINATE_N_PAIRS,
        "all_bit_identical": True,
        "max_abs_diff_pose": 0.0,
        "max_abs_diff_seg": 0.0,
        "n_failures": 0,
        "failing_pairs": [],
    }
    mismatches = {
        key: {"expected": value, "got": bit_identity.get(key)}
        for key, value in expected.items()
        if bit_identity.get(key) != value
    }
    if fraction.get("n_blind_px") != _BLIND_COORDINATE_N_PX:
        mismatches["n_blind_px"] = {
            "expected": _BLIND_COORDINATE_N_PX,
            "got": fraction.get("n_blind_px"),
        }
    if fraction.get("retained_subgrid_hw") != [768, 1024]:
        mismatches["retained_subgrid_hw"] = {
            "expected": [768, 1024],
            "got": fraction.get("retained_subgrid_hw"),
        }
    if mismatches:
        raise ValueError(f"D21a n600 zero-delta receipt gate failed: {mismatches}")
    return {
        "schema": _BLIND_COORDINATE_RECEIVER_SCHEMA,
        "active": True,
        "proof_receipt_path": str(receipt_path),
        "proof_receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "n_pairs": _BLIND_COORDINATE_N_PAIRS,
        "n_blind_px_per_frame": _BLIND_COORDINATE_N_PX,
        "delta_d_seg": 0.0,
        "delta_d_pose": 0.0,
        "all_scorer_inputs_bit_identical": True,
        "lawref": "blind_coordinate_rate_lever_v1",
        "status": "MEASURED_N600_ZERO_DELTA_RECEIPT_ACCEPTED",
        "verdict_scope": (
            "receiver fill admissibility only; direct archive saving is zero for a pure generator "
            "until a camera-resolution residual or sidecar stores the retained subgrid"
        ),
        "score_claim": False,
    }


def _is_linux_x86_64() -> bool:
    """True iff the host is the contest 1:1 CPU/CUDA substrate family (Linux x86_64)."""
    import platform

    return platform.system() == "Linux" and platform.machine().lower() in ("x86_64", "amd64")


def _advisory_axis_label() -> str:
    """Host-truthful advisory authority for the LOCAL (non-exact, genuinely-CPU) print paths.
    macOS CPU-torch is NOT 1:1 with the contest Linux x86_64 CPU runner -> [macOS-CPU advisory];
    a Linux non-x86_64 host (e.g. aarch64) is NOT macOS and NOT contest-1:1 -> its own label
    (2026-07-06 pointer-authority review LOW: never call a Linux ARM box "macOS"); only a real
    Linux x86_64 host earns [contest-CPU advisory]. Always NON-PROMOTABLE here (no MPS/CUDA/paid
    axis on these paths)."""
    import platform

    if _is_linux_x86_64():
        return "[contest-CPU advisory] NON-PROMOTABLE"
    if platform.system() == "Linux":
        return "[Linux-non-x86_64-CPU advisory] NON-PROMOTABLE"
    if platform.system() == "Darwin":
        return "[macOS-CPU advisory] NON-PROMOTABLE"
    return "[non-contest-CPU advisory] NON-PROMOTABLE"


_AUTHORITY = _advisory_axis_label()


def _axis_and_authority(device: str) -> tuple[str, str]:
    """(score_axis, authority) for an EXACT upstream/evaluate.py row, computed from the ACTUAL
    ``--device`` passed to evaluate.py FIRST, then the host platform.

    CPU and CUDA are SEPARATE evidence spaces, never inferred from each other (CLAUDE.md
    "Apples-to-apples evidence discipline"). The pre-fix code derived the axis purely from the
    host platform (``_AUTHORITY``), so a real ``--eval-device cuda`` run on Linux x86_64 (the
    documented decode_t4_16gb tier) persisted score_axis "[contest-CPU]" — a mislabeled-axis
    provenance corruption (2026-07-06 pointer-authority review CRITICAL).

    Mapping (axis labels are load-bearing provenance):
      cuda + Linux x86_64  -> ("[contest-CUDA]", "[contest-CUDA]")
      cuda + other host    -> ("[non-contest-CUDA advisory] NON-PROMOTABLE", same)
      cpu  + Linux x86_64  -> ("[contest-CPU]", "[contest-CPU advisory] NON-PROMOTABLE")
      cpu  + Darwin        -> ("[macOS-CPU advisory] NON-PROMOTABLE", same)
      cpu  + Linux other   -> ("[Linux-non-x86_64-CPU advisory] NON-PROMOTABLE", same)
    MPS is refused upstream of this function (never a score authority)."""
    dev = str(device).strip().lower()
    if dev == "mps":
        raise ValueError("MPS is NEVER a score authority (CLAUDE.md).")
    if dev.startswith("cuda"):
        if _is_linux_x86_64():
            return "[contest-CUDA]", "[contest-CUDA]"
        label = "[non-contest-CUDA advisory] NON-PROMOTABLE"
        return label, label
    if dev == "cpu":
        label = _advisory_axis_label()
        if _is_linux_x86_64():
            return "[contest-CPU]", label
        return label, label
    raise ValueError(f"_axis_and_authority: unknown device {device!r} (expected cpu/cuda).")


def _refuse_tmp(path: Path, field: str) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{field}={path!r} is a /tmp-class path; use the SSD/repo tier per CLAUDE.md.")


def receiver_env_manifest() -> dict[str, Any]:
    """#402 DEPENDENCY PINNING + CROSS-MICROARCH NOTE (advisory PR128 §8.2).

    Record the EXACT versions of every decode-path dependency the shipped inflate.py imports, so a
    byte-close row's provenance names the environment that produced it (a fresh unfrozen sync can
    resolve a different numpy/torch/brotli and the deterministic-reproducibility contract needs the
    versions on record). ``constriction`` is deliberately N/A: unlike PR128, the level-set receiver's
    ξ payload uses a pure-stdlib arithmetic coder (``_ar_decode``), so it carries no constriction
    dependency to pin.

    Cross-host bit-identity note: the AUTHORITY forward is the numpy-fp32 reference
    (``levelset_rgb_forward_numpy``, fp64 by default -- the deterministic verdict), NOT torch. The
    ONLY torch op on the decode path is ``_R`` = ``torch.nn.functional.interpolate(mode='bicubic')``
    whose last-bit rounding CAN differ across CPU microarchitectures/BLAS. Therefore this tool makes
    NO portable sha256-of-.raw claim (grep confirms none is emitted); the .raw is verified PER HOST
    via the bit-exact round-trip gate (``--verify-bit-exact``) on the host that produced it, and
    CPU/CUDA are separate evidence axes never inferred from each other."""
    import importlib
    import platform

    versions: dict[str, str] = {"python": platform.python_version()}
    for mod in ("numpy", "torch", "brotli", "scipy"):
        try:
            versions[mod] = str(getattr(importlib.import_module(mod), "__version__", "unknown"))
        except Exception as exc:  # noqa: BLE001 — absent optional dep is provenance, not fatal
            versions[mod] = f"absent ({type(exc).__name__})"
    versions["constriction"] = ("N/A (level-set receiver uses a pure-stdlib arithmetic ξ coder "
                                "(_ar_decode), not constriction)")
    return {
        "decode_path_versions": versions,
        "host": {"platform": platform.platform(), "machine": platform.machine(),
                 "processor": platform.processor() or "unknown",
                 "is_contest_linux_x86_64": _is_linux_x86_64()},
        "authority_forward": ("numpy-fp32 reference (tac.boundary_math.lever_b_levelset_generator."
                              "levelset_rgb_forward_numpy; fp64 default) -- the deterministic verdict; "
                              "torch is NOT the authority."),
        "cross_host_bit_identity": (
            "The only torch op on the decode path is _R = torch.nn.functional.interpolate(mode="
            "'bicubic'), whose LSBs can vary across CPU microarch/BLAS (advisory PR128 §8.2). This "
            "tool emits NO portable sha256-of-.raw authority claim; the .raw is verified PER HOST via "
            "the --verify-bit-exact round-trip gate. CPU and CUDA are separate evidence axes."),
    }


def xi_receiver_camera_contract() -> dict[str, Any]:
    """Return the canonical, hash-bound camera constants emitted into ``inflate.py``.

    ``_INFLATE_PY`` is a raw source template, so module globals are not in scope when
    the shipped receiver executes.  Keep the clip-profile values in one canonical JSON
    string and bind that exact string before materializing the four runtime constants.
    The values are generic receiver mechanism (rule 118), not video-derived payload.
    """

    values = {
        "cx": float(_CP_XI_CX),
        "cy": float(_CP_XI_CY),
        "device_height_m": float(_CP_XI_D),
        "fx_native": float(_CP_XI_FX),
    }
    canonical_json = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return {
        "schema": "xi_receiver_camera_contract.v1",
        "values": values,
        "canonical_json": canonical_json,
        "canonical_json_sha256": hashlib.sha256(canonical_json.encode("ascii")).hexdigest(),
        "source": "tac.clip_profile.for_video(upstream/videos/0.mkv) with documented fallback",
        "counted_payload_bytes": 0,
    }


def _render_hash_bound_xi_camera_contract() -> str:
    contract = xi_receiver_camera_contract()
    canonical_json = contract["canonical_json"]
    digest = contract["canonical_json_sha256"]
    return "\n".join(
        (
            f"_XI_CAMERA_CONTRACT_JSON = {canonical_json!r}",
            f"_XI_CAMERA_CONTRACT_SHA256 = {digest!r}",
            "if hashlib.sha256(_XI_CAMERA_CONTRACT_JSON.encode('ascii')).hexdigest() != _XI_CAMERA_CONTRACT_SHA256:",
            "    raise RuntimeError('xi camera contract hash mismatch')",
            "_xi_camera_contract = json.loads(_XI_CAMERA_CONTRACT_JSON)",
            "_CP_XI_FX = float(_xi_camera_contract['fx_native'])",
            "_CP_XI_CX = float(_xi_camera_contract['cx'])",
            "_CP_XI_CY = float(_xi_camera_contract['cy'])",
            "_CP_XI_D = float(_xi_camera_contract['device_height_m'])",
            "del _xi_camera_contract",
        )
    )


def _bind_xi_camera_contract(inflate_template: str) -> str:
    marker_count = inflate_template.count(_XI_CAMERA_CONTRACT_MARKER)
    if marker_count != 1:
        raise RuntimeError(
            "inflate source must contain exactly one xi camera-contract marker; "
            f"found {marker_count}"
        )
    return inflate_template.replace(
        _XI_CAMERA_CONTRACT_MARKER,
        _render_hash_bound_xi_camera_contract(),
    )


# ---------------------------------------------------------------------------
# checkpoint loading -- separate LEARNED params from the __cfg/__bank/__render scalars.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# weights-arm selection (B1 confound-F1 fix): ema / live / polyak candidate arms.
# The trainer exports up to three deploy npz candidates for one run:
#   levelset_witness_ema_mlx.npz     — the EMA shadow (default deploy arm)
#   levelset_witness_live_mlx.npz    — the LIVE weights (the EMA-lag escape arm, FEED-br)
#   levelset_witness_polyak_mlx.npz  — the R7 Polyak tail-average finisher CANDIDATE
# Before this fix the byte-close/eval consumed ONE arm and NEVER ranked polyak vs {ema,live}
# (the "picks the better candidate" consumer did not exist). ``select_best_weights_arm`` byte-closes
# + scores every AVAILABLE arm and RECORDS the N-way selection (per-arm scores + winner + margins).
# Fail-open: a missing arm npz is simply not in the set => older runs (ema-only) are unchanged.
_ARM_NPZ: dict[str, str] = {
    "ema": _wra.EMA_NPZ,
    "live": _wra.LIVE_NPZ,
    "polyak": _wra.POLYAK_NPZ,
}
# The order the loader's default candidate search prefers (ema first) — used to label a default run.
_ARM_DEFAULT_ORDER = ("ema", "live", "polyak")


def _arm_label_for_npz(npz_name: str | None) -> str:
    """Map a resolved npz filename to its weights-arm label ('ema'/'live'/'polyak'), or 'explicit'
    for a custom filename, or 'default(ema)' when the loader fell back to its default search order
    (npz_name is None => ema is preferred + is what byte-closes on a normal run)."""
    if not npz_name:
        return "default(ema)"
    base = Path(npz_name).name
    for arm, fn in _ARM_NPZ.items():
        if base == fn:
            return arm
    return "explicit"


def discover_available_arms(ckpt_dir: Path) -> list[str]:
    """The weights arms whose npz is present in ``ckpt_dir`` (fail-open: only what actually exists),
    in the canonical ema/live/polyak order. An older run with just the EMA npz yields ``['ema']``."""
    return [arm for arm in _ARM_DEFAULT_ORDER if (ckpt_dir / _ARM_NPZ[arm]).exists()]


def _load_levelset_ckpt(
    ckpt_dir: Path, npz_name: str | None = None
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load (params, cfg) from a level-set run dir. NO-FAKE: missing files raise.

    Params = every npz key NOT prefixed ``__`` (the learned weights + ``code``). cfg = the
    ``__cfg_*`` / ``__bank_*`` / ``__render_hw`` scalars (parsed; defaults + tensor-shape inference
    fill any that an older save block omitted, with a loud warning)."""
    candidates = [npz_name] if npz_name else [_wra.EMA_NPZ, _wra.LIVE_NPZ]
    npz = None
    for c in candidates:
        p = ckpt_dir / c
        if p.exists():
            npz = p
            break
    if npz is None:
        raise FileNotFoundError(
            f"level-set checkpoint missing in {ckpt_dir} (looked for {candidates}); "
            "refusing to fabricate (NO-FAKE). If the run is still training, its npz is saved only "
            "at the END of the loop -- wait for completion or point at a finished run dir."
        )
    z = np.load(npz, allow_pickle=False)
    params: dict[str, np.ndarray] = {}
    raw_cfg: dict[str, Any] = {}
    for k in z.files:
        if k.startswith("__"):
            a = z[k]
            raw_cfg[k] = a.item() if a.size == 1 else a.tolist()
        else:
            params[k] = np.asarray(z[k], dtype=np.float32)

    for req in ("code", "in_proj.weight", "out_sdf.weight", "out_tex.weight", "palette"):
        if req not in params:
            raise ValueError(f"level-set npz {npz} lacks required param {req!r} (NO-FAKE).")

    cfg: dict[str, Any] = {"npz_name": npz.name}
    cfg["n_classes"] = int(params["out_sdf.weight"].shape[0])
    cfg["hidden_dim"] = int(raw_cfg.get("__cfg_hidden_dim", params["in_proj.weight"].shape[0]))
    cfg["n_hidden"] = int(raw_cfg.get(
        "__cfg_n_hidden", sum(1 for k in params if k.startswith("hidden.") and k.endswith(".weight"))))
    cfg["mod_dim"] = int(params["code"].shape[1])
    cfg["n_pairs"] = int(params["code"].shape[0] // 2)
    cfg["in_feat"] = int(params["in_proj.weight"].shape[1])
    cfg["activation"] = str(raw_cfg.get("__cfg_activation", "hosc"))
    cfg["softmax_temp"] = float(raw_cfg.get("__cfg_softmax_temp", 0.05))
    cfg["chroma"] = bool(int(raw_cfg.get("__cfg_chroma", 1)))
    cfg["wire_w0"] = float(raw_cfg.get("__cfg_wire_w0", 20.0))
    cfg["wire_s0"] = float(raw_cfg.get("__cfg_wire_s0", 10.0))
    cfg["hosc_beta"] = float(raw_cfg.get("__cfg_hosc_beta", 4.0))
    cfg["hosc_omega"] = float(raw_cfg.get("__cfg_hosc_omega", 1.0))
    cfg["basis"] = normalize_basis_family(raw_cfg.get("__cfg_basis", LEGACY_FOURIER_AB_CONTROL))
    if cfg["basis"] == LITERAL_POLAR_CURVELET:
        program_json = raw_cfg.get("__cfg_basis_program_json")
        program_sha = str(raw_cfg.get("__cfg_basis_program_sha256", ""))
        if not isinstance(program_json, str) or not program_sha:
            raise ValueError(
                "literal_polar_curvelet checkpoint lacks BasisProgramConfig JSON/hash custody"
            )
        try:
            program = BasisProgramConfig.from_dict(json.loads(program_json))
        except (TypeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid literal BasisProgramConfig: {exc}") from exc
        if program.canonical_sha256() != program_sha:
            raise ValueError("literal BasisProgramConfig canonical SHA-256 mismatch")
        cfg["basis_program"] = program
        cfg["basis_program_train_sha256"] = program_sha
        if program.taper_enabled:
            if int(raw_cfg.get("__cfg_basis_taper_folded", -1)) != 0:
                raise ValueError("literal training checkpoint must carry explicitly unfolded taper state")
            taper = np.asarray(raw_cfg.get("__basis_taper_unfolded", []), dtype=np.float32)
            if taper.shape != (program.feature_width,):
                raise ValueError(
                    "literal taper custody width does not match BasisProgramConfig feature_width"
                )
            cfg["basis_taper_unfolded"] = taper
        if program.chart_enabled:
            # (#497 gap-a) COUNTED-CHART-PAYLOAD custody: the quantized startup pose table +
            # scales the trainer built its chart from (build_from_xi on the DEQUANTIZED values).
            # Byte-close ships these as the counted 7th section; missing custody = the chart the
            # receiver would rebuild is UNDEFINED -> fail closed (NO-FAKE).
            _cq = raw_cfg.get("__chart_pose_q")
            _cs = raw_cfg.get("__chart_pose_scales")
            if _cq is None or _cs is None:
                raise ValueError(
                    "literal chart_enabled checkpoint lacks __chart_pose_q/__chart_pose_scales "
                    "custody (the counted chart payload) -- cannot byte-close a chart the "
                    "receiver cannot rebuild (NO-FAKE fail-closed)."
                )
            chart_q = np.asarray(_cq, dtype=np.int16)
            chart_scales = np.asarray(_cs, dtype=np.float32)
            if chart_q.ndim != 2 or chart_q.shape[1] != 6 or chart_scales.shape != (6,):
                raise ValueError(
                    f"literal chart custody has wrong shapes: q {chart_q.shape}, "
                    f"scales {chart_scales.shape} (expected (P,6) int16 + (6,) fp32)"
                )
            cfg["chart_pose_q"] = chart_q
            cfg["chart_pose_scales"] = chart_scales
    cfg["bank_n_scales"] = int(raw_cfg.get("__bank_n_scales", 4))
    cfg["bank_n_orient0"] = int(raw_cfg.get("__bank_n_orient0", 6))
    cfg["bank_f0"] = float(raw_cfg.get("__bank_f0", 2.0))
    cfg["bank_base"] = float(raw_cfg.get("__bank_base", 2.0))
    cfg["bank_n_iso"] = int(raw_cfg.get("__bank_n_iso", 4))
    mbf = raw_cfg.get("__cfg_max_bank_freq")
    cfg["max_bank_freq"] = None if (mbf is None or float(mbf) < 0) else float(mbf)
    rh = raw_cfg.get("__render_hw")
    if rh is None:
        cfg["render_h"], cfg["render_w"] = 384, 512
        print("[WARN] npz lacks __render_hw -> defaulting render 384x512 (may NOT match the "
              "trained render res; pass --render-h/--render-w if known).", flush=True)
    else:
        cfg["render_h"], cfg["render_w"] = int(rh[0]), int(rh[1])
    cfg["lane_edge_weight"] = float(raw_cfg.get("__cfg_lane_edge_weight", 0.0))  # provenance only
    # #417 texture trunk (#395): the boundary-annulus attenuation power the receiver forward needs
    # (default 0.0 => no attenuation; matches the trainer default). Shapes for tex_trunk/out_tex_h/
    # decoupled_head come from the param arrays; only annulus_power is a non-shape scalar the receiver
    # must know. Absent (pre-#395 ckpt / shared head) => 0.0 => no manifest key emitted (byte-identical).
    cfg["texture_trunk_annulus_power"] = float(raw_cfg.get("__cfg_texture_trunk_annulus_power", 0.0))
    missing = [k for k in ("__cfg_activation", "__bank_n_scales", "__render_hw") if k not in raw_cfg]
    if missing:
        print(f"[WARN] npz omitted cfg keys {missing} (older save block) -> using inferred/defaults; "
              "provenance-incomplete (the deploy render may diverge if a default is wrong).", flush=True)
    return params, cfg


def _bank_cfg(cfg: dict[str, Any]) -> CurveletBankConfig:
    return CurveletBankConfig(
        n_scales=cfg["bank_n_scales"], n_orient0=cfg["bank_n_orient0"],
        f0=cfg["bank_f0"], base=cfg["bank_base"], n_iso=cfg["bank_n_iso"],
    )


def _curvelet_feat_width(cfg: dict[str, Any]) -> int:
    """2 * (number of curvelet bank columns after the optional max_freq cap)."""
    B = curvelet_directional_B(_bank_cfg(cfg), max_freq=cfg["max_bank_freq"])
    return 2 * int(B.shape[1])


def _basis_feat_width(cfg: dict[str, Any]) -> int:
    """Feature width regenerated by the selected FREE basis compiler."""
    family = normalize_basis_family(cfg.get("basis", LEGACY_FOURIER_AB_CONTROL))
    if family == "windowed_curvelet":
        return GENUINE_FRAME_FEATURE_WIDTH
    if family == COMPACT_SHEARLET:
        return GENUINE_FRAME_FEATURE_WIDTH
    if family == LITERAL_POLAR_CURVELET:
        return int(cfg["basis_program"].feature_width)
    return _curvelet_feat_width(cfg)


def detect_self_orient(cfg: dict[str, Any], so_overrides: dict[str, Any]) -> dict[str, Any]:
    """dir_w = in_feat - curvelet_feat_width. >0 => self-orient was used; n_dir_freqs = dir_w//4.
    freq_across/freq_along/tau/iters are NOT persisted by the trainer -> sourced from CLI overrides
    (defaults = trainer defaults). Returns a dict the manifest + inflate consume."""
    curv_w = _basis_feat_width(cfg)
    dir_w = int(cfg["in_feat"]) - curv_w
    if dir_w < 0:
        raise ValueError(
            f"in_feat ({cfg['in_feat']}) < curvelet_feat_width ({curv_w}): the bank cfg in the npz "
            "does NOT reproduce the trained in_proj input width. The __bank_* scalars are wrong/"
            "missing (NO-FAKE: cannot regenerate the free bank that matches the weights). Pass the "
            "correct --bank-* overrides or use a checkpoint with full __bank_* provenance.")
    if dir_w == 0:
        return {"self_orient": False, "curvelet_feat_width": curv_w, "dir_w": 0}
    if dir_w % 4 != 0:
        raise ValueError(
            f"self-orient dir_w ({dir_w}) not divisible by 4 (= 4*n_dir_freqs). The bank/in_feat "
            "mismatch means the free bank does not reproduce the weights -- refuse (NO-FAKE).")
    return {
        "self_orient": True,
        "curvelet_feat_width": curv_w,
        "dir_w": dir_w,
        "n_dir_freqs": dir_w // 4,
        "freq_across": float(so_overrides["freq_across"]),
        "freq_along": float(so_overrides["freq_along"]),
        "tau": float(so_overrides["tau"]),
        "iters": int(so_overrides["iters"]),
    }


# ---------------------------------------------------------------------------
# byte-close: int8 + brotli (matches quantize_levelset_blob accounting; bank is FREE).
# ---------------------------------------------------------------------------
def build_levelset_blob(
    params: dict[str, np.ndarray], cfg: dict[str, Any], so: dict[str, Any], pose_sidecar: bytes | None,
    lane_band_bytes: bytes | None = None, lane_manifest: dict[str, Any] | None = None,
    pose_carrier_bytes: bytes | None = None, pose_carrier_manifest: dict[str, Any] | None = None,
    cross_tensor_codec: bool = False,
    blind_coordinate_fill: bool = False,
    palette_residual_bytes: bytes | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Layout: magic | u32 manifest_len | manifest_json | u32 base_brotli_len | base_brotli |
            u32 code_brotli_len | code_brotli | u32 pose_len | pose_sidecar |
            manifest-gated lane/pose-carrier/chart/EKPR1 optional blocks.
    base = int8(all params except code) concat -> ONE brotli stream (== quantize_levelset_blob);
    code = int8(code) -> a SECOND brotli stream. The curvelet bank + the lane COVERAGE raster are
    NOT stored (free, rule 118); only the lane MANIFOLD COORDS (``lane_band_bytes``) are counted.
    Absent ``lane_band_bytes`` the manifest + blob are BYTE-IDENTICAL to the pre-Wave-E grammar.
    """
    import brotli

    # `pose_carrier.*` (xi_stored / dxi) are NOT INR weights -- they are the trained ego-twist table
    # whose ONLY legitimate home is the pose-carrier SECTION (the coded ξ payload), never the base
    # weight blob. Older saves leaked them into `params`, so they were int8-quantized into the base
    # blob as DEAD counted bytes (inflate never reads them from base). Exclude them here so the base
    # blob is INR-only and the ξ rate is attributed cleanly to the pose-carrier section (#238).
    # #417: exclude the rule-118 FREE tables (``B`` / ``*_B`` -- e.g. the tex_trunk ``bank_B`` Gabor
    # bank) from the COUNTED base blob, EXACTLY as the canonical quantize_levelset_blob does (they are
    # regenerated free at decode from cfg, never stored -> a 4.7M-value bank would else be counted as
    # dead int8 rate). Shared-head witnesses carry no ``_B`` key => this is a no-op => byte-identical.
    base_order = [k for k in params if k != "code" and not k.startswith("pose_carrier.")
                  and not (k == "B" or k.endswith("_B"))]

    # #417 FAIL-CLOSED receiver-consumption bijection gate (NO-FAKE #8). Every COUNTED base
    # param MUST be consumed by the receiver -- the _INFLATE_PY forward -- else it pays rate
    # but is INERT through R, so a scored A/B on that lever silently renders the shared-head
    # CONTROL and reports a FAKE verdict. Sister of the pose_carrier.* hand-exclusion above;
    # this makes the fix STRUCTURAL for every future group (v7.5.3 tex_trunk / v8 decoupled_head).
    # Escape hatch (loud, explicit): TAC_ALLOW_UNCONSUMED_ARCHIVE_GROUPS=<prefix,...>.
    import os as _os

    from tools.levelset_receiver_bijection_gate import assert_receiver_bijection
    _allow = frozenset(
        g.strip() for g in _os.environ.get("TAC_ALLOW_UNCONSUMED_ARCHIVE_GROUPS", "").split(",") if g.strip()
    )
    assert_receiver_bijection(
        base_order, _INFLATE_PY, context="build_levelset_blob base blob", allow_unconsumed=_allow,
    )

    shapes: dict[str, list[int]] = {}
    scales: dict[str, float] = {}
    for name in base_order:
        a = np.asarray(params[name], dtype=np.float32)
        _, scale = _int8_symmetric(a)
        shapes[name] = list(a.shape)
        scales[name] = float(scale)

    code = np.asarray(params["code"], dtype=np.float32)
    qc, code_scale = _int8_symmetric(code)
    base_plan = None
    code_plan = None
    if cross_tensor_codec:
        base_plan = _wxc.derive_base_permutation_plan(params, base_order)
        code_plan = _wxc.derive_code_transform_plan(code)
        base_raw = _wxc.encode_base_quantized(params, base_order, base_plan.transposed_names)
        code_raw = _wxc.encode_code_quantized(qc, code_plan.transform)
    else:
        base_raw = _wxc.encode_base_quantized(params, base_order, ())
        code_raw = _wxc.encode_code_quantized(qc, _wxc.CODE_TRANSFORM_RAW)
    base_brotli = brotli.compress(base_raw, quality=11)
    code_brotli = brotli.compress(code_raw, quality=11)

    parsed_palette_residual = (
        decode_palette_residual(
            palette_residual_bytes,
            expected_n_pairs=int(cfg["n_pairs"]),
            expected_n_classes=int(cfg["n_classes"]),
        )
        if palette_residual_bytes is not None
        else None
    )
    manifest = {
        "format_version": 1,
        "n_pairs": int(cfg["n_pairs"]),
        "n_classes": int(cfg["n_classes"]),
        "hidden_dim": int(cfg["hidden_dim"]),
        "n_hidden": int(cfg["n_hidden"]),
        "mod_dim": int(cfg["mod_dim"]),
        "activation": str(cfg["activation"]),
        "softmax_temp": float(cfg["softmax_temp"]),
        "chroma": bool(cfg["chroma"]),
        "wire_w0": float(cfg["wire_w0"]), "wire_s0": float(cfg["wire_s0"]),
        "hosc_beta": float(cfg["hosc_beta"]), "hosc_omega": float(cfg["hosc_omega"]),
        "bank_n_scales": int(cfg["bank_n_scales"]), "bank_n_orient0": int(cfg["bank_n_orient0"]),
        "bank_f0": float(cfg["bank_f0"]), "bank_base": float(cfg["bank_base"]),
        "bank_n_iso": int(cfg["bank_n_iso"]),
        "max_bank_freq": (None if cfg["max_bank_freq"] is None else float(cfg["max_bank_freq"])),
        "render_h": int(cfg["render_h"]), "render_w": int(cfg["render_w"]),
        "camera_h": CAMERA_H, "camera_w": CAMERA_W,
        "base_param_order": base_order,
        "base_shapes": shapes,
        "base_scales": scales,
        "code_shape": list(code.shape),
        "code_scale": float(code_scale),
        "self_orient": bool(so["self_orient"]),
        "n_dir_freqs": int(so.get("n_dir_freqs", 0)),
        "so_freq_across": float(so.get("freq_across", 0.0)),
        "so_freq_along": float(so.get("freq_along", 0.0)),
        "so_tau": float(so.get("tau", 4.0)),
        "so_iters": int(so.get("iters", 0)),
        "has_pose_sidecar": bool(pose_sidecar is not None),
    }
    chart_payload_bytes: bytes | None = None  # #497 gap-a: set only by the literal chart branch
    if normalize_basis_family(cfg.get("basis", LEGACY_FOURIER_AB_CONTROL)) == "windowed_curvelet":
        # Generic, video-free receiver program state (rule 118 FREE).  Emitted only for the selected
        # treatment so a polar/default packet retains its historical manifest bytes exactly.
        manifest["basis_family"] = "windowed_curvelet"
        manifest["windowed_curvelet_config"] = asdict(genuine_frame_windowed_curvelet_config())
    elif normalize_basis_family(cfg.get("basis", LEGACY_FOURIER_AB_CONTROL)) == COMPACT_SHEARLET:
        # Same rule-118 boundary as the curvelet treatment: the deterministic atom table is
        # regenerated in inflate.py; only learned coefficients remain counted.
        manifest["basis_family"] = COMPACT_SHEARLET
        manifest["compact_shearlet_config"] = asdict(genuine_frame_compact_shearlet_config())
    elif normalize_basis_family(cfg.get("basis", LEGACY_FOURIER_AB_CONTROL)) == LITERAL_POLAR_CURVELET:
        program: BasisProgramConfig = cfg["basis_program_deploy"]
        if program.taper_enabled and not program.deploy_fold_receipt_sha256:
            raise ValueError("literal taper deploy program lacks the mandatory fold receipt hash")
        manifest["basis_family"] = LITERAL_POLAR_CURVELET
        manifest["basis_program"] = program.to_dict()
        manifest["basis_program_sha256"] = program.canonical_sha256()
        if program.chart_enabled:
            # (#497 gap-a) the COUNTED chart payload (7th block): quantized startup pose table +
            # scales. Manifest flag gates the READ; chart CONFIG rides basis_program (no scalar
            # duplication -> the flag carries only what the parser needs, honest minimal rate).
            _chart_q = cfg.get("chart_pose_q")
            _chart_scales = cfg.get("chart_pose_scales")
            if _chart_q is None or _chart_scales is None:
                raise ValueError(
                    "literal chart_enabled blob build lacks chart_pose_q/scales custody "
                    "(loader should have refused earlier; NO-FAKE fail-closed)."
                )
            chart_payload_bytes = _chart_payload_bytes(
                np.asarray(_chart_q, np.int16), np.asarray(_chart_scales, np.float32)
            )
            manifest["chart_payload"] = {"n_pairs": int(np.asarray(_chart_q).shape[0])}
    if blind_coordinate_fill:
        # Rule-118 FREE receiver program.  The measured proof receipt stays outside the counted
        # packet; only this generic activation/law identity is needed to reproduce the decode.
        manifest["blind_coordinate_fill"] = {
            "schema": _BLIND_COORDINATE_RECEIVER_SCHEMA,
            "lawref": "blind_coordinate_rate_lever_v1",
            "n_blind_px_per_frame": _BLIND_COORDINATE_N_PX,
        }
    if cross_tensor_codec:
        # Compact receiver contract: ``p`` holds indices into base_param_order whose 2-D storage is
        # transposed; ``c=1`` is frame-separated modulo-256 temporal delta for code[2*pair+frame].
        # The human-readable derivation lives in the breakdown/receipt, not the counted manifest.
        manifest["xcodec"] = {
            "p": [base_order.index(name) for name in base_plan.transposed_names],
            "c": int(code_plan.transform == _wxc.CODE_TRANSFORM_FRAME_DELTA_MOD256),
        }
    # #224 Wave E: the analytic-lane RENDER-BAND cfg (ONLY when active -> default-off byte-identical).
    # The lane MANIFOLD COORDS ride the 5th block (counted); this cfg (scalars) rides the manifest so
    # inflate reproduces the coverage raster + composite decode-consistently (rule 118 FREE rasterizer).
    if lane_band_bytes is not None and lane_manifest is not None:
        manifest["lane_render_band"] = lane_manifest
    # #417 texture trunk (#395): the ONLY non-shape config its receiver forward needs is the annulus
    # power (default 0.0 => no boundary attenuation). Emitted ONLY when tex_trunk params are counted, so
    # a shared-head witness's manifest JSON is BYTE-IDENTICAL (the receiver defaults absent -> 0.0).
    if any(k.startswith("tex_trunk.") for k in base_order):
        manifest["texture_trunk_annulus_power"] = float(cfg.get("texture_trunk_annulus_power", 0.0))
    # #205 warp-real-luma pose carrier (6th block). Manifest flag gates the READ (so the reader
    # knows to expect the trailing block); default-off -> byte-identical to the pre-#205 grammar.
    if pose_carrier_bytes is not None and pose_carrier_manifest is not None:
        manifest["pose_carrier"] = pose_carrier_manifest
    if parsed_palette_residual is not None:
        manifest["palette_residual"] = parsed_palette_residual.manifest
    mj = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    out = _io_pack(mj, base_brotli, code_brotli, pose_sidecar, lane_band_bytes, pose_carrier_bytes,
                   chart_payload_bytes, palette_residual_bytes)
    # cross-check our accounting against the canonical quantize_levelset_blob (same int8 grammar).
    # #238: the base blob is INR-only (pose_carrier.* live in the pose-carrier SECTION, not base), so
    # cross-check against the canonical on the SAME INR-only param set (else the check spuriously fails).
    canon = quantize_levelset_blob(
        {k: np.asarray(v, np.float32) for k, v in params.items() if not k.startswith("pose_carrier.")})
    breakdown = {
        "n_params": int(sum(int(np.prod(s)) for s in shapes.values())) + int(np.prod(code.shape)),
        "manifest_bytes": len(mj),
        "base_int8_brotli_bytes": len(base_brotli),
        "code_int8_brotli_bytes": len(code_brotli),
        "pose_sidecar_bytes": (len(pose_sidecar) if pose_sidecar else 0),
        "lane_band_counted_bytes": (len(lane_band_bytes) if lane_band_bytes else 0),
        "pose_carrier_counted_bytes": (len(pose_carrier_bytes) if pose_carrier_bytes else 0),
        "chart_payload_counted_bytes": (len(chart_payload_bytes) if chart_payload_bytes else 0),
        "palette_residual_counted_bytes": (
            len(palette_residual_bytes) if palette_residual_bytes is not None else 0
        ),
        "magic_and_prefixes_bytes": (len(_MAGIC) + 16 + (4 if lane_band_bytes is not None else 0)
                                     + (4 if pose_carrier_bytes is not None else 0)
                                     + (4 if chart_payload_bytes is not None else 0)
                                     + (4 if palette_residual_bytes is not None else 0)),
        "total_0bin_bytes": len(out),
        "canonical_quantize_blob_bytes": int(canon["total_quantized_blob_bytes"]),
        "accounting_matches_canonical": bool(
            len(base_brotli) == canon["base_int8_brotli_bytes"]
            and len(code_brotli) == canon["code_int8_brotli_bytes"]),
        "cross_tensor_codec": {
            "active": bool(cross_tensor_codec),
            "base_permutation": (base_plan.to_json() if base_plan is not None else None),
            "code_transform": (code_plan.to_json() if code_plan is not None else None),
            "quantized_state_lossless_by_construction": True,
        },
    }
    return out, breakdown


def _xcodec_transposed_names(manifest: dict[str, Any]) -> tuple[str, ...]:
    xc = manifest.get("xcodec") or {}
    order = manifest["base_param_order"]
    indices = tuple(int(i) for i in xc.get("p", ()))
    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(order) for i in indices):
        raise ValueError(f"LVLS1 xcodec has invalid base permutation indices {indices}")
    return tuple(order[i] for i in indices)


def _decode_base_params(manifest: dict[str, Any], base_brotli: bytes) -> dict[str, np.ndarray]:
    import brotli

    q = _wxc.decode_base_quantized(
        brotli.decompress(base_brotli),
        manifest["base_param_order"],
        manifest["base_shapes"],
        _xcodec_transposed_names(manifest),
    )
    return {
        name: (arr.astype(np.float32) * float(manifest["base_scales"][name]))
        for name, arr in q.items()
    }


def _xcodec_code_transform(manifest: dict[str, Any]) -> str:
    code = int((manifest.get("xcodec") or {}).get("c", 0))
    if code not in (0, 1):
        raise ValueError(f"LVLS1 xcodec has unknown code transform id {code}")
    return (
        _wxc.CODE_TRANSFORM_FRAME_DELTA_MOD256
        if code == 1
        else _wxc.CODE_TRANSFORM_RAW
    )


def _decode_code(manifest: dict[str, Any], code_brotli: bytes) -> np.ndarray:
    import brotli

    q = _wxc.decode_code_quantized(
        brotli.decompress(code_brotli),
        manifest["code_shape"],
        _xcodec_code_transform(manifest),
    )
    return q.astype(np.float32) * float(manifest["code_scale"])


def _encode_code_brotli(q: np.ndarray, manifest: dict[str, Any]) -> bytes:
    import brotli

    return brotli.compress(
        _wxc.encode_code_quantized(q, _xcodec_code_transform(manifest)), quality=11
    )


def _io_pack(
    manifest: bytes, base: bytes, code: bytes, pose: bytes | None,
    lane_band: bytes | None = None, pose_carrier: bytes | None = None,
    chart: bytes | None = None,
    palette_residual: bytes | None = None,
) -> bytes:
    """Pack the LVLS1 blob. The 5th ``lane_band`` block (#224 Wave E), the 6th
    ``pose_carrier`` block (#205 warp-real-luma frame0), and the 7th ``chart`` block (#497 gap-a:
    the COUNTED chart payload — quantized int16 (P,6) startup pose table + fp32 (6,) scales) are
    followed by the optional 8th ``palette_residual`` EKPR1 block. They are appended, in that
    order, ONLY when non-None -> absent all, the output is
    BYTE-IDENTICAL to the pre-Wave-E 4-block grammar (the default-off guarantee). Trailing blocks
    are gated at READ time by the manifest flags (``lane_render_band`` / ``pose_carrier`` /
    ``chart_payload`` / ``palette_residual``), so the reader knows how many trailing blocks to expect -- NEVER by a bare
    ``off < len(raw)`` (which would misread a lone pose_carrier block as lane). ``lane_band`` = the
    COUNTED lane manifold coords; ``pose_carrier`` = the COUNTED real-luma keyframe payload +
    per-pair homography (rule 118: keyframe COUNTED, warp decoder FREE); ``chart`` = the COUNTED
    video-derived table the ground chart is rebuilt from (rule 118: the chart MATH — expmap /
    plane-motion / homography composition — is FREE generic code; only the q/scales bytes count)."""

    buf = bytearray()
    buf += _MAGIC
    for chunk in (manifest, base, code, (pose or b"")):
        buf += struct.pack("<I", len(chunk))
        buf += chunk
    for opt in (lane_band, pose_carrier, chart, palette_residual):
        if opt is not None:
            buf += struct.pack("<I", len(opt))
            buf += opt
    return bytes(buf)


# ---------------------------------------------------------------------------
# #497 gap-a: COUNTED chart payload (7th optional block) serialize / parse.
#   The literal ground chart MUST be a counted receiver program: the trainer built its chart
#   from the quantize->dequantize round-tripped startup pose table (counted_chart_payload —
#   NOT counted_pose_carrier_xi: the carrier's xi_eff = xi_stored + TRAINED dxi does not exist
#   at ep0, so the chart binds its OWN payload). This section ships exactly that table:
#   q int16 (P,6) little-endian + scales fp32 (6,) little-endian. The receiver dequantizes
#   op-for-op ``xi_pose_coder.dequantize_xi`` (q.astype(f8) * scales.astype(f8)) and rebuilds
#   the identical fp64 homography chain -> trainer chart == receiver chart bit-for-bit.
# ---------------------------------------------------------------------------
def _chart_payload_bytes(q: np.ndarray, scales: np.ndarray) -> bytes:
    q = np.asarray(q)
    scales = np.asarray(scales)
    if q.ndim != 2 or q.shape[1] != 6 or q.dtype != np.int16:
        raise ValueError(f"chart payload q must be int16 (P,6); got {q.dtype} {q.shape}")
    if scales.shape != (6,) or scales.dtype != np.float32:
        raise ValueError(f"chart payload scales must be fp32 (6,); got {scales.dtype} {scales.shape}")
    # explicit little-endian on the wire (host-portable; np.frombuffer '<i2'/'<f4' on read).
    return q.astype("<i2", copy=False).tobytes() + scales.astype("<f4", copy=False).tobytes()


def _parse_chart_payload(chart_b: bytes, n_pairs: int) -> dict[str, np.ndarray]:
    n_pairs = int(n_pairs)
    expected = n_pairs * 6 * 2 + 6 * 4
    if len(chart_b) != expected:
        raise ValueError(
            f"chart payload is {len(chart_b)} B != expected {expected} B for n_pairs={n_pairs} "
            "(int16 (P,6) + fp32 (6,)) -- fail closed (NO-FAKE)."
        )
    q = np.frombuffer(chart_b[: n_pairs * 12], dtype="<i2").reshape(n_pairs, 6)
    scales = np.frombuffer(chart_b[n_pairs * 12:], dtype="<f4")
    return {"q": q.astype(np.int16, copy=False), "scales": scales.astype(np.float32, copy=False)}


# ---------------------------------------------------------------------------
# #205 POSE CARRIER (warp-real-luma frame0): serialize / warp / parse.
#   The scorer reads d_pose = MSE(PoseNet(gen_pair)[:6], PoseNet(gt_pair)[:6]) ON THE FRAMES.
#   A dead pose SIDECAR (block 3) is bytes the render never reads -> does NOT lower realized d_pose.
#   The POSE CARRIER instead REPLACES the render's FRAME0 (SEG-free: SegNet reads only frame1,
#   upstream/modules.py:108) with a STORED REAL keyframe SE(3)-warped by the stored per-pair ego
#   twist -> PoseNet reads real warped motion and the pose becomes measurable through the FRAMES
#   (the ``tac.boundary_math.warp_real_luma_frame0`` semantics). Rule-118 boundary:
#     * COUNTED (archive.zip): the stored real keyframe luma (video-derived) + per-pair homography.
#     * FREE (inflate.py): the inverse-warp bilinear + R (generic algorithm). The dual-use twist xi
#       is stored (fp16) for provenance; the per-pair H (fp64) is what the decode uses (bit-exact,
#       no exp_se3 needed in the decode path -> the shipped inflate == numpy authority by
#       construction). H is a deterministic function of xi (byte-optimal design stores ONLY xi and
#       derives H FREE; H stored here is the decode-simplicity choice, ~72 B/pair, negligible vs
#       the keyframe payload).
# ---------------------------------------------------------------------------
def _pcar_warp_frame0_from_H(src_native: np.ndarray, H: np.ndarray, native_hw: tuple[int, int]) -> np.ndarray:
    """Inverse-warp a native ``(H,W,3)`` real-luma frame by the fp64 homography ``H`` -> uint8.

    OP-FOR-OP identical to ``tac.boundary_math.warp_real_luma_frame0.warp_frame0_native_numpy`` FROM
    ``Hinv = inv(H)`` ONWARD (bilinear inverse-sample + persist fallback + round/clip), so that with
    ``H = homography_from_xi_numpy(xi, geom)`` (fp64) this returns the module authority
    ``warp_frame0_uint8_numpy(src, xi, geom)`` BIT-FOR-BIT (proven 0-diff, tests). The shipped
    inflate inlines a VERBATIM copy of this body -> shipped == this oracle by construction."""
    src = np.asarray(src_native, dtype=np.float64)
    Hh, Ww = int(native_hw[0]), int(native_hw[1])
    if src.shape[:2] != (Hh, Ww):
        raise ValueError(f"pose-carrier keyframe native {src.shape[:2]} != {(Hh, Ww)} (NO-FAKE)")
    C = src.shape[2]
    flat = src.reshape(-1, C)
    us, vs = np.meshgrid(np.arange(Ww), np.arange(Hh))
    grid = np.stack([us.ravel(), vs.ravel(), np.ones(Hh * Ww)], 0).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        Hinv = np.linalg.inv(np.asarray(H, np.float64))
        src_h = Hinv @ grid
        z = src_h[2]
        su = src_h[0] / z
        sv = src_h[1] / z
    valid = (np.isfinite(su) & np.isfinite(sv) & (z > 0)
             & (su >= 0) & (su <= Ww - 1) & (sv >= 0) & (sv <= Hh - 1))
    su_c = np.clip(su, 0.0, Ww - 1)
    sv_c = np.clip(sv, 0.0, Hh - 1)
    x0 = np.floor(su_c).astype(np.int64)
    y0 = np.floor(sv_c).astype(np.int64)
    x1 = np.minimum(x0 + 1, Ww - 1)
    y1 = np.minimum(y0 + 1, Hh - 1)
    wx = (su_c - x0)[:, None]
    wy = (sv_c - y0)[:, None]
    Ia = flat[y0 * Ww + x0]
    Ib = flat[y0 * Ww + x1]
    Ic = flat[y1 * Ww + x0]
    Id = flat[y1 * Ww + x1]
    top = Ia * (1.0 - wx) + Ib * wx
    bot = Ic * (1.0 - wx) + Id * wx
    sampled = top * (1.0 - wy) + bot * wy
    out = np.where(valid[:, None], sampled, flat).reshape(Hh, Ww, C)
    return np.clip(np.round(out), 0.0, 255.0).astype(np.uint8)


def serialize_pose_carrier(
    H_stack: np.ndarray, xi_stack: np.ndarray, keyframes: list[np.ndarray],
    kf_of_pair: list[int], hdr_extra: dict[str, Any],
) -> bytes:
    """Serialize the pose-carrier section (NOT re-brotli'd; keyframes are brotli'd individually).

    Layout: PCAR1 | u32 hdr_len | hdr_json | H(P*9 fp64) | xi(P*6 fp16) | u32 n_kf |
            [u32 blen | brotli(uint8 (kf_h,kf_w,3))]*n_kf.
    ``kf_of_pair[p]`` indexes into ``keyframes``; H/xi are per-pair (P rows)."""
    import brotli

    P = int(H_stack.shape[0])
    hdr = {
        "n_pairs": P,
        "native_h": CAMERA_H, "native_w": CAMERA_W,
        "kf_of_pair": [int(k) for k in kf_of_pair],
        "n_keyframes": len(keyframes),
        **hdr_extra,
    }
    hj = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    buf = bytearray()
    buf += _PCAR_MAGIC
    buf += struct.pack("<I", len(hj)); buf += hj
    buf += np.asarray(H_stack, dtype=np.float64).tobytes()
    buf += np.asarray(xi_stack, dtype=np.float16).tobytes()
    buf += struct.pack("<I", len(keyframes))
    for kf in keyframes:
        kb = brotli.compress(np.ascontiguousarray(kf, dtype=np.uint8).tobytes(), quality=11)
        buf += struct.pack("<I", len(kb)); buf += kb
    return bytes(buf)


def serialize_pose_carrier_store_nothing(
    xi_stack: np.ndarray, hdr_extra: dict[str, Any], *, coder: str = "delta_ar", q_levels: int = 4096,
) -> tuple[bytes, dict[str, Any]]:
    """#257 STORE-NOTHING v2 serializer -- stores ONLY the (quantized, coded) per-pair ego twist ξ.

    The redundant fp64 H (43,200 B/600 pairs = 83% of the old section, FINDING-1) is DROPPED and
    DERIVED FREE at decode (``_xip.homographies_from_xi``, rule-118). The useless
    ``kf_of_pair=[0]*P`` header list is also DROPPED (unused when n_keyframes=0). Layout:
        PCAR1\\x00 | u32 hdr_len | hdr_json | u32 xi_payload_len | xi_payload | u32 n_kf(=0)
    ``xi_payload`` (``tac.boundary_math.xi_pose_coder``) carries the per-channel scales + q
    (``coder='none'`` raw int16 ~0.005 rate, or ``coder='delta_ar'`` temporal-Δ + arithmetic ~0.002).
    Returns (blob, quant_report). The shipped inflate inlines the inverse (ξ decode + derive-H)."""
    q, scales = _xip.quantize_xi(np.asarray(xi_stack, dtype=np.float64), q_levels=int(q_levels))
    xi_payload = _xip.serialize_xi_payload(q, scales, coder=coder)
    P = int(q.shape[0])
    hdr = {
        "n_pairs": P,
        "native_h": CAMERA_H, "native_w": CAMERA_W,
        "n_keyframes": 0,
        "pcar_store_nothing_v": 2,          # discriminator: v2 = derive-H, NO stored H block
        "xi_coder": str(coder),
        "xi_q_levels": int(q_levels),
        **hdr_extra,
    }
    hj = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    buf = bytearray()
    buf += _PCAR_MAGIC
    buf += struct.pack("<I", len(hj)); buf += hj
    buf += struct.pack("<I", len(xi_payload)); buf += xi_payload
    buf += struct.pack("<I", 0)              # n_kf = 0 (store-nothing: NO keyframe luma)
    quant_report = {
        "xi_coder": str(coder), "xi_q_levels": int(q_levels),
        "xi_payload_bytes": len(xi_payload),
        "xi_raw_bytes_ref": len(_xip.serialize_xi_payload(q, scales, coder="none")),
    }
    return bytes(buf), quant_report


def parse_pose_carrier(blob: bytes) -> dict[str, Any]:
    """Inverse of ``serialize_pose_carrier`` / ``serialize_pose_carrier_store_nothing`` (tool-side;
    the shipped inflate inlines the same). Discriminates on the header ``pcar_store_nothing_v``:
    v2 store-nothing DERIVES H from the coded ξ (no stored H block); everything else reads the
    legacy fp64-H block (warp_real_luma stays byte-identical)."""
    import brotli

    assert blob[: len(_PCAR_MAGIC)] == _PCAR_MAGIC, "bad pose-carrier magic"
    off = len(_PCAR_MAGIC)
    (hlen,) = struct.unpack_from("<I", blob, off); off += 4
    hdr = json.loads(blob[off:off + hlen].decode("utf-8")); off += hlen
    P = int(hdr["n_pairs"])
    if int(hdr.get("pcar_store_nothing_v", 0)) == 2:
        # #257 v2: ξ payload -> dequant ξ -> DERIVE H (FREE, rule-118). No stored H, no kf_of_pair.
        (xlen,) = struct.unpack_from("<I", blob, off); off += 4
        xi_dq = _xip.decode_xi_payload(blob[off:off + xlen]); off += xlen
        (n_kf,) = struct.unpack_from("<I", blob, off); off += 4
        if n_kf != 0:
            raise ValueError(f"store-nothing v2 must have n_kf=0; got {n_kf} (NO-FAKE)")
        H = _xip.homographies_from_xi(xi_dq, float(hdr["pitch"]))
        if off != len(blob):  # #402 EXACT CONSUMPTION (mirrors inflate._pcar_parse v2).
            raise ValueError(f"PCAR v2 block has {len(blob) - off} unconsumed trailing byte(s)")
        return {"hdr": hdr, "H": H, "xi": xi_dq, "keyframes": [], "kf_of_pair": [0] * P}
    # legacy: stored fp64 H block (warp_real_luma; and any pre-#257 store-nothing archive).
    H = np.frombuffer(blob[off:off + P * 9 * 8], dtype=np.float64).reshape(P, 3, 3).copy(); off += P * 9 * 8
    xi = np.frombuffer(blob[off:off + P * 6 * 2], dtype=np.float16).reshape(P, 6).astype(np.float64); off += P * 6 * 2
    (n_kf,) = struct.unpack_from("<I", blob, off); off += 4
    kh, kw = int(hdr["kf_store_h"]), int(hdr["kf_store_w"])
    keyframes = []
    for _ in range(n_kf):
        (blen,) = struct.unpack_from("<I", blob, off); off += 4
        raw = brotli.decompress(blob[off:off + blen]); off += blen
        keyframes.append(np.frombuffer(raw, dtype=np.uint8).reshape(kh, kw, 3).copy())
    if off != len(blob):  # #402 EXACT CONSUMPTION (mirrors inflate._pcar_parse legacy).
        raise ValueError(f"PCAR legacy block has {len(blob) - off} unconsumed trailing byte(s)")
    return {"hdr": hdr, "H": H, "xi": xi, "keyframes": keyframes, "kf_of_pair": hdr["kf_of_pair"]}


def _pcar_upsample_to_native(kf: np.ndarray, native_hw: tuple[int, int]) -> np.ndarray:
    """Bilinear-upsample a stored keyframe (kh,kw,3) uint8 -> native fp64. Deterministic torch CPU
    (bit-identical run-to-run + between inflate/oracle). Identity when already native."""
    import torch

    Hh, Ww = int(native_hw[0]), int(native_hw[1])
    if kf.shape[:2] == (Hh, Ww):
        return np.asarray(kf, dtype=np.float64)
    x = torch.from_numpy(np.ascontiguousarray(kf)).permute(2, 0, 1)[None].float()
    with torch.inference_mode():
        up = torch.nn.functional.interpolate(x, size=(Hh, Ww), mode="bilinear", align_corners=False)
    return up[0].permute(1, 2, 0).contiguous().numpy().astype(np.float64)


def pose_carrier_frame0(pc: dict[str, Any], pi: int) -> np.ndarray:
    """Decode frame0 for pair ``pi`` = warp(stored keyframe, stored H[pi]) at native res -> uint8.
    The tool-side authority (warp_real_luma mode); the shipped inflate inlines a verbatim copy."""
    native_hw = (int(pc["hdr"]["native_h"]), int(pc["hdr"]["native_w"]))
    kf = pc["keyframes"][int(pc["kf_of_pair"][pi])]
    src = _pcar_upsample_to_native(kf, native_hw)
    return _pcar_warp_frame0_from_H(src, pc["H"][pi], native_hw)


def pose_carrier_mode(pc: dict[str, Any]) -> str:
    """The pose-carrier decode mode from the PCAR hdr. ``warp_real_luma`` (default; stored real
    keyframe) or ``store_nothing`` (store ONLY xi/H; frame0 = warp(GENERATED witness render, H))."""
    return str(pc["hdr"].get("pose_carrier_mode", "warp_real_luma"))


def pose_carrier_frame0_from_source(pc: dict[str, Any], pi: int, src_native: np.ndarray) -> np.ndarray:
    """STORE-NOTHING frame0 = warp(a GENERATED camera-native ``src_native``, stored H[pi]) -> uint8.

    ``src_native`` is the witness's OWN frame0 RGB render at camera resolution (NOT a stored real
    keyframe -- store_nothing stores ONLY xi/H, ~0 marginal bytes). Op-for-op the SAME warp as
    ``_pcar_warp_frame0_from_H`` so the shipped inflate's inlined store-nothing warp == this oracle
    bit-for-bit (the general bit-exact gate proves it over frame0 too)."""
    native_hw = (int(pc["hdr"]["native_h"]), int(pc["hdr"]["native_w"]))
    return _pcar_warp_frame0_from_H(np.asarray(src_native, dtype=np.float64), pc["H"][pi], native_hw)


def _downscale_keyframe(gt_f0_native: np.ndarray, store_hw: tuple[int, int]) -> np.ndarray:
    """Store-res keyframe: bilinear-downscale native gt_f0 -> (store_h,store_w,3) uint8 (torch CPU,
    deterministic). Identity when store_hw == native. Round-trips through the decode-time upsample."""
    import torch

    sh, sw = int(store_hw[0]), int(store_hw[1])
    if (sh, sw) == (gt_f0_native.shape[0], gt_f0_native.shape[1]):
        return np.ascontiguousarray(gt_f0_native, dtype=np.uint8)
    x = torch.from_numpy(np.ascontiguousarray(gt_f0_native)).permute(2, 0, 1)[None].float()
    with torch.inference_mode():
        dn = torch.nn.functional.interpolate(x, size=(sh, sw), mode="bilinear", align_corners=False)
        dn = torch.clamp(torch.round(dn), 0.0, 255.0)
    return dn[0].permute(1, 2, 0).contiguous().numpy().astype(np.uint8)


def build_pose_carrier_section(
    gt_cache: str | None, n_pairs: int, pc: dict[str, Any],
    xi_override: np.ndarray | None = None,
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    """Build the #205 pose-carrier section from the GT cache (compress-time: the ego pose
    ``gt_poses`` [+ real keyframe luma ``gt_f0`` for warp_real_luma] are fully available). Returns
    (pose_carrier_bytes, manifest_cfg, report). NO-FAKE: missing cache raises.

    Two selectable modes (``pc["mode"]``, A/B-able against each other at #205; NEITHER removed):

    * ``warp_real_luma`` (default) -- store the real keyframe luma; frame0 = warp(stored keyframe,
      per-pair H). Keyframes stored at ``store_hw`` (native=lossless; downscaled=cheap-rate lever).
      Rate = keyframe brotli + H(72 B/pair) + xi(12 B/pair), COUNTED in archive.zip.
    * ``store_nothing`` (Track B ``18927a1ae`` / ``keyframe_rate_minimization_builds``) -- store ONLY
      xi (+ H for decode simplicity), NO real keyframe. frame0 = warp(the witness's OWN frame0 RGB
      render [generated FREE by the INR from the counted weights+code -- rule-118], per-pair H). The
      keyframe payload collapses to ~0 marginal bytes (rate ~= xi/H only). MEASURED n600 (Track B,
      classmean proxy): d_pose 4.97 pre-residual (witness render is richer -> <= 4.97); the trained
      rank-6 dxi residual (#205, w_pose>0) closes the fixed offset toward ~3.4e-5 (UNMEASURED here).

    For each pair p: the per-pair ego twist ``xi[p] = xi_from_pose_calibration(gt_poses[p], s_t,
    s_r, pitch)``; ``H[p] = homography_from_xi_numpy(xi[p], geom)`` (fp64, the exact warp). ``stride``
    keyframes (warp_real_luma only): pair p uses keyframe ``(p//stride)*stride`` (stride=1 -> per-pair
    exact upper bound; stride>1 -> shared, geometrically-approximate, LOUDLY flagged)."""
    mode = str(pc.get("mode", "warp_real_luma"))
    if mode not in ("warp_real_luma", "store_nothing"):
        raise ValueError(f"pose-carrier mode {mode!r} unknown (warp_real_luma|store_nothing).")
    store_nothing = mode == "store_nothing"
    if not gt_cache:
        raise ValueError("--pose-carrier requires --gt-cache (gt_poses [+ gt_f0 for warp_real_luma]). "
                         "NO-FAKE: refusing to fabricate the pose payload.")
    cp = Path(gt_cache)
    if not cp.exists():
        raise FileNotFoundError(f"--gt-cache {cp} not found (pose-carrier needs gt_poses [+ gt_f0]).")
    z = np.load(cp, allow_pickle=False)
    # store_nothing needs ONLY the ego poses (no real keyframe luma -> the pure "store nothing but xi").
    required = ("gt_poses",) if store_nothing else ("gt_f0", "gt_poses")
    for req in required:
        if req not in z.files:
            raise ValueError(f"gt cache {cp} lacks {req!r} (pose-carrier mode={mode}).")
    gt_poses = np.asarray(z["gt_poses"], dtype=np.float64)
    P = int(min(int(n_pairs), int(gt_poses.shape[0])))
    if not store_nothing:
        gt_f0 = z["gt_f0"]
        P = min(P, int(gt_f0.shape[0]))
    s_t, s_r, pitch = float(pc["s_t"]), float(pc["s_r"]), float(pc["pitch"])
    stride = max(1, int(pc["stride"]))
    ds = max(1, int(pc.get("downscale", 1)))
    xi_coder = str(pc.get("xi_coder", "delta_ar"))     # #257: 'delta_ar' (default) | 'none' (raw fallback)
    xi_q_levels = int(pc.get("xi_q_levels", 4096))     # #257: ξ quantization precision (store_nothing only)
    store_hw = (CAMERA_H, CAMERA_W) if (store_nothing or ds == 1) else (CAMERA_H // ds, CAMERA_W // ds)
    geom = _wrl.GroundHomographyGeom.eon(pitch=pitch)

    # per-pair ego twist ξ (fp64 authority) -- both modes. H is DERIVED FREE at decode for
    # store_nothing (#257), STORED only for warp_real_luma (the legacy stored-keyframe path).
    # #238 CONNECTOR: when ``xi_override`` is supplied (the TRAINED ξ_eff = xi_stored + dxi loaded
    # from the run checkpoint), ship THAT instead of the deterministic calibration recompute -- this
    # is what carries R1's trained d_pose descent into a shippable archive. The override is the FULL
    # trained per-pair twist; the calibration s_t/s_r/pitch are then provenance-only (H is still
    # derived from the SHIPPED ξ with the same ``pitch`` the trainer used).
    xi_from_ckpt = xi_override is not None
    if xi_from_ckpt:
        xo = np.asarray(xi_override, dtype=np.float64)
        if xo.ndim != 2 or xo.shape[1] != 6:
            raise ValueError(f"xi_override must be (P,6); got {xo.shape} (NO-FAKE).")
        if xo.shape[0] < P:
            raise ValueError(f"xi_override has {xo.shape[0]} pairs < needed {P} (NO-FAKE).")
        xi_stack = xo[:P].copy()
    else:
        xi_stack = np.zeros((P, 6), dtype=np.float64)
        for p in range(P):
            xi_stack[p] = _wrl.xi_from_pose_calibration(gt_poses[p], s_t, s_r, pitch)

    quant_report: dict[str, Any] = {}
    if store_nothing:
        # #257 STORE-NOTHING v2: store ONLY the (quantized+coded) ξ. NO stored H (43,200 B redundancy
        # DROPPED -> derived FREE at decode), NO kf_of_pair junk list. frame0 = warp(witness's OWN
        # frame0 render, DERIVED H). The whole keyframe payload is GONE (rule-118: render is FREE).
        keyframes: list[np.ndarray] = []
        hdr_extra = {
            "pose_carrier_mode": mode,
            "s_t": s_t, "s_r": s_r, "pitch": pitch, "stride": stride,
            "kf_store_h": int(store_hw[0]), "kf_store_w": int(store_hw[1]),
            "keyframe_lossless_native": True,
            "generator": "witness_render_frame0",
            "calibration_note": ("STORE-NOTHING v2 (#257): store ONLY the per-pair ego twist ξ "
                                 "(quantized+coded); H = homographies_from_xi(ξ, pitch) DERIVED FREE at "
                                 "decode (rule-118). frame0 = warp(witness's OWN frame0 INR render, H)."),
        }
        blob, quant_report = serialize_pose_carrier_store_nothing(
            xi_stack, hdr_extra, coder=xi_coder, q_levels=xi_q_levels)
        kf_of_pair: list[int] = [0] * P
    else:
        # warp_real_luma (legacy): store fp64 H + keyframe luma (BYTE-IDENTICAL to pre-#257).
        H_stack = np.zeros((P, 3, 3), dtype=np.float64)
        for p in range(P):
            H_stack[p] = _wrl.homography_from_xi_numpy(xi_stack[p], geom)
        kf_indices = sorted({min(P - 1, (p // stride) * stride) for p in range(P)})
        kf_pos = {k: i for i, k in enumerate(kf_indices)}
        kf_of_pair = [kf_pos[min(P - 1, (p // stride) * stride)] for p in range(P)]
        keyframes = [_downscale_keyframe(np.asarray(gt_f0[k]), store_hw) for k in kf_indices]
        hdr_extra = {
            "pose_carrier_mode": mode,
            "s_t": s_t, "s_r": s_r, "pitch": pitch, "stride": stride,
            "kf_store_h": int(store_hw[0]), "kf_store_w": int(store_hw[1]),
            "keyframe_lossless_native": bool(ds == 1),
            "generator": "stored_real_keyframe",
            "calibration_note": ("per-pair xi = xi_from_pose_calibration(gt_poses, s_t, s_r, pitch); "
                                 "H = homography_from_xi_numpy(xi). warp-real-luma frame0 semantics "
                                 "(tac.boundary_math.warp_real_luma_frame0)."),
        }
        blob = serialize_pose_carrier(H_stack, xi_stack, keyframes, kf_of_pair, hdr_extra)

    kf_bytes_total = sum(int(len(s)) for s in _pcar_keyframe_blob_sizes(keyframes))  # 0 for store_nothing
    manifest = {
        "mode": mode,
        "s_t": s_t, "s_r": s_r, "pitch": pitch, "stride": stride,
        "kf_store_h": int(store_hw[0]), "kf_store_w": int(store_hw[1]),
        "n_keyframes": len(keyframes), "n_pairs": P,
        "keyframe_lossless_native": bool(store_nothing or ds == 1),
    }
    report = {
        "active": True,
        "mode": mode,
        "xi_source": ("ckpt_xi_eff (xi_stored+dxi, #238 trained twist)" if xi_from_ckpt
                      else "calibration (xi_from_pose_calibration recompute)"),
        "generator": ("witness_render_frame0" if store_nothing else "stored_real_keyframe"),
        "source_gt_cache": str(cp),
        "n_pairs": P,
        "n_keyframes": len(keyframes),
        "stride": stride,
        "keyframe_store_hw": [int(store_hw[0]), int(store_hw[1])],
        "keyframe_lossless_native": bool(store_nothing or ds == 1),
        "pose_carrier_section_bytes": len(blob),
        "keyframe_blob_bytes_total": kf_bytes_total,
        # #257: store_nothing v2 stores NO H (derived FREE) -> H_bytes=0; xi_bytes = the coded/raw
        # ξ payload actually stored. warp_real_luma keeps the legacy stored fp64-H (72 B/pair).
        "H_bytes": 0 if store_nothing else int(P * 9 * 8),
        "xi_bytes": int(quant_report.get("xi_payload_bytes", P * 6 * 2)) if store_nothing else int(P * 6 * 2),
        "xi_coder": quant_report.get("xi_coder") if store_nothing else None,
        "xi_q_levels": quant_report.get("xi_q_levels") if store_nothing else None,
        "xi_raw_bytes_ref": quant_report.get("xi_raw_bytes_ref") if store_nothing else None,
        "calibration": {"s_t": s_t, "s_r": s_r, "pitch": pitch},
        "stride_note": (
            "store_nothing v2 (#257): no stored keyframe AND no stored H -> frame0 = warp(witness's OWN "
            "frame0 render, DERIVED per-pair H); stride unused (only the coded ξ is stored)."
            if store_nothing else
            ("stride>1 shares one stored keyframe across a window but warps each pair by ITS OWN intra-pair "
             "xi (no kf->pair displacement) -> geometrically approximate for shared keyframes; stride=1 "
             "(per-pair) is the EXACT warp-real-luma upper bound." if stride > 1 else
             "stride=1: per-pair keyframe, exact warp-real-luma.")),
        "rule_118": {
            "COUNTED (archive.zip)": (
                "ONLY the per-pair ego twist ξ (quantized + entropy-coded) + 6 per-channel scales + the "
                "scalar pitch; NO stored H (43,200 B redundancy DROPPED -> derived FREE), NO keyframe luma"
                if store_nothing else
                "stored real keyframe luma (video-derived) + per-pair homography H; warped at decode by "
                "the per-pair ego twist to synthesize frame0"),
            "FREE (inflate.py)": (
                "the witness INR frame0 render + exp_se3 + H=K(R-t nT/d)K^-1 (DERIVE-H) + the ξ arithmetic "
                "decoder + inverse-warp bilinear + R -- all generic algorithm, 0 archive bytes (rule-118)"
                if store_nothing else
                "inverse-warp bilinear + R (generic algorithm); the dual-use twist xi is the pose sidecar "
                "(~0 marginal bytes in the byte-optimal design)"),
            "no_gt_no_scorer": "no GT mask, no SegNet/PoseNet weights, no per-pixel table ship",
        },
        "byte_optimal_note": (
            "STORE-NOTHING v2 (#257): stores ONLY the coded ξ ({} B, coder={}, q_levels={}; raw-ξ ref {} B). "
            "The redundant per-pair fp64 H (43,200 B = 83% of the pre-#257 section, FINDING-1) is DROPPED "
            "and DERIVED FREE at decode (exp_se3 + plane homography, rule-118); the kf_of_pair junk list "
            "is DROPPED. ZERO keyframe bytes. d_pose is measured through the REAL byte-closed decode (the "
            "derived H reproduces the fp64-H warp; the trained rank-6 dξ residual (#205 w_pose>0) closes "
            "d_pose -- UNMEASURED here; NO borrowed number).".format(
                quant_report.get("xi_payload_bytes", "?"), quant_report.get("xi_coder", "?"),
                quant_report.get("xi_q_levels", "?"), quant_report.get("xi_raw_bytes_ref", "?"))
            if store_nothing else
            "H (fp64, 72 B/pair) is stored for BIT-EXACT decode simplicity (no exp_se3 in the decode path); "
            "the byte-optimal design stores ONLY xi (fp16, 12 B/pair, dual-use with the pose sidecar) and "
            "derives H FREE via exp_se3. Keyframe SPARSITY (stride>1 with kf->pair relative warp) + lossy "
            "keyframe codec are the OPEN rate levers -- NOT applied/borrowed here (measured rate is of what "
            "is ACTUALLY stored)."),
    }
    return blob, manifest, report


def _pcar_keyframe_blob_sizes(keyframes: list[np.ndarray]) -> list[bytes]:
    """Recompute the per-keyframe brotli blobs (for the byte report). Deterministic == the serializer."""
    import brotli

    return [brotli.compress(np.ascontiguousarray(kf, dtype=np.uint8).tobytes(), quality=11) for kf in keyframes]


def _cap_pose_carrier(pcar_bytes: bytes, eval_pairs: int) -> bytes:
    """Slice a pose-carrier section to the first ``eval_pairs`` pairs (for the capped/gate inflate):
    keep H/xi[:eval_pairs], prune to the referenced keyframes, remap kf_of_pair. Re-serialized ==
    the full serializer, so the shipped inflate + oracle stay bit-exact on the capped set."""
    pc = parse_pose_carrier(pcar_bytes)
    P = min(int(eval_pairs), int(pc["hdr"]["n_pairs"]))
    hdr = pc["hdr"]
    hdr_extra = {k: hdr[k] for k in ("pose_carrier_mode", "generator", "s_t", "s_r", "pitch", "stride",
                                     "kf_store_h", "kf_store_w", "keyframe_lossless_native",
                                     "calibration_note") if k in hdr}
    if int(hdr.get("pcar_store_nothing_v", 0)) == 2:  # #257 v2: re-serialize the sliced coded ξ (derive-H)
        capped, _qr = serialize_pose_carrier_store_nothing(
            np.asarray(pc["xi"][:P], dtype=np.float64), hdr_extra,
            coder=str(hdr.get("xi_coder", "delta_ar")), q_levels=int(hdr.get("xi_q_levels", 4096)))
        return capped
    if len(pc["keyframes"]) == 0:  # legacy store_nothing: no keyframes; kf_of_pair unused placeholder
        return serialize_pose_carrier(pc["H"][:P], pc["xi"][:P], [], [0] * P, hdr_extra)
    kf_of_pair = [int(k) for k in pc["kf_of_pair"][:P]]
    used = sorted(set(kf_of_pair))
    remap = {old: i for i, old in enumerate(used)}
    keyframes = [pc["keyframes"][old] for old in used]
    kf_of_pair_new = [remap[k] for k in kf_of_pair]
    return serialize_pose_carrier(pc["H"][:P], pc["xi"][:P], keyframes, kf_of_pair_new, hdr_extra)


# ---------------------------------------------------------------------------
# the self-contained inflate.py (numpy fwd + torch R [+ scipy iff self-orient]).
# ---------------------------------------------------------------------------
# LOC-UNRESTRICTED: inflate.py inlines THREE rule-118 FREE levers whose
# archive-counted statistics require receiver consumption -- (1) the
# SELF-ORIENT fixed-point (curvelet bank regen + decoder-own-argmax tangent + directional feats), the
# byte-closeable directional lever the mod-32 target uses; (2) the #224 Wave E analytic-lane
# RENDER-BAND decode reproduction (_lane_parse + _lane_coverage AA-SDF rasterizer + _lane_composite),
# which EXPANDS the counted per-pair lane manifold coords into the (H,W) coverage + composites the
# band over the render FREE (0 archive bytes); (3) the #257 STORE-NOTHING DERIVE-H + ξ ARITHMETIC
# DECODER (_ar_decode + _xip_parse + _xip_H_from_xi), which EXPANDS the counted per-pair ego twist ξ
# into the per-pair plane-induced homography H (exp_se3 + K(R-t nT/d)K^-1) FREE at decode -- dropping
# the 43,200 B redundant stored fp64 H (FINDING-1). All THREE are op-for-op mirrors of the canonical
# numpy authority (levelset_rgb_forward_numpy / levelset_band_forward_numpy / rasterize_lane_coverage_
# range_dependent / composite_band_on_render / tac.boundary_math.xi_pose_coder / tac.lossless.range_
# coder.decode_static_symbols) and are BIT-EXACT-gate proven vs it.
# FEED-eg (2026-06-27): n600 inflate is float64-activation-bound (~50-60 min on a 4-core CPU for
# 600 pairs x 6 forwards x 5 tanh(sin) layers @ 384x512 -- OVER the 30-min budget). The forward is
# split into _in_proj_h0 (feats-only) + _outputs_from_h0 (code-dependent, want_rgb flag) to enable
# THREE BIT-IDENTICAL (sha256-proven) decode speedups, archive bytes UNCHANGED (inflate code = FREE,
# rule 118): (1) self-orient EARLY-STOP at the argmax fixed point (the big lever on a CONVERGED
# decoder -- up to ~2x); (2) self-orient forwards skip the rgb head (argmax(phi) needs no rgb);
# (3) the in_proj h0 is shared by a pair's 2 final frames. Guaranteed ~3% (skip-rgb+share-h0);
# early-stop is data-dependent (measure on the first real ckpt). Even so, contest-LEGALITY needs a
# trainer-side config cut (so_iters/render-res/n_hidden), which CHANGES the witness -> re-measure.
_INFLATE_PY = r'''#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# LEVEL-SET witness inflate -- numpy fwd + torch R, self-contained. Emits the FULL
# (2*n_pairs, camera_h, camera_w, 3) uint8 .raw (frame0+frame1 per pair). The curvelet bank +
# (when used) the self-orient directional feats are REGENERATED for FREE (rule 118); only the
# learned int8+brotli payload comes from the archive.
# FEED-eg bit-identical decode speedups (inflate.py code only -> FREE, archive bytes UNCHANGED):
#   (1) self-orient EARLY-STOP at the argmax fixed point (consecutive-equal argmax => dirf frozen
#       => remaining iters are no-ops; bit-identical, the big lever on a converged decoder);
#   (2) self-orient forwards compute ONLY phi (argmax needs no rgb head -- out_tex/palette/softmax/
#       sigmoid skipped, they do not feed phi);
#   (3) the in_proj activation h0 is computed ONCE per pair and shared by the pair's 2 final frames
#       (identical feats => identical h0). All three preserve the EXACT float64/float32 op order of
#       levelset_rgb_forward_numpy -> bit-identical .raw (proven by sha256 parity).
# FEED-eh PARALLEL decode speedup (inflate.py code only -> FREE, archive bytes UNCHANGED, BIT-IDENTICAL):
#   the n_pairs are INDEPENDENT -> render them across a process Pool (1-thread BLAS per worker so N
#   workers scale cleanly instead of oversubscribing the already-threaded GEMM), each worker writing its
#   2 frames to DISJOINT offsets of the preallocated .raw. Proven 10.8x on M5 Max (15w),
#   max_abs_uint8_diff=0 vs the serial output. Contest: Linux fork (workers inherit setup); keeps the
#   inflate.py cost well inside the 30-min upstream/evaluate.py budget (inflate + scoring).
# FEED-ei bit-identical _act fusion (inflate.py code only -> FREE, archive bytes UNCHANGED): the hosc
#   activation tanh(beta*sin(omega*u)) is the #1 hot path (~66% of decode; ~29 acts/pair on the (H*W,
#   hidden) float64 grid = ~151MB/array). (1) SKIP the identity full-array multiplies when omega==1 /
#   beta==1 (1.0*x === x in IEEE754, but numpy still executes the whole ~151MB DRAM round-trip); (2)
#   compute sin -> (beta) -> tanh IN-PLACE in one buffer (the 'pre' input is dead after _act) -> fewer
#   151MB allocs. In-place ufunc == out-of-place ufunc elementwise, so the exact float64 op values are
#   preserved -> BIT-IDENTICAL .raw (proven: cmp -s == identical vs the pre-fusion forward, n40/4w). The
#   DRAM-traffic cut helps MOST under the parallel Pool (workers share memory bandwidth): measured 1.24x
#   4-worker wall-clock (contest-4-core full-600 inflate ~20 min -> ~16 min).
# FEED-ej bit-identical self-orient CONVERGED-h0 reuse (inflate.py code only -> FREE, archive UNCHANGED):
#   the self-orient fixed point (task #281) EARLY-STOPS when the decoder's argmax stops changing (FEED-eg);
#   on the converging iteration the loop's `feats` are already the FINAL feats, so its in_proj activation
#   h0 IS the post-loop h0 -> cache + reuse it instead of recomputing _in_proj_h0 once more. BIT-EXACT: the
#   reuse only fires on the early-break (feats frozen); when the loop runs to completion (no convergence)
#   h0 is recomputed exactly as before. Proven byte-identical vs baseline on the #205 config (n24) AND by a
#   determinism micro-proof on the real weights (dir_feats/in_proj_h0 are pure functions of their inputs).
#   Like the early-stop it is a NO-OP on a mid-training archive (the argmax does not converge even at
#   so_iters=12) and pays only on a well-trained archive whose argmax stabilizes within so_iters.
# FEED-ek OPT-IN fp32 forward (task #281; INFLATE_FP32=1, DEFAULT OFF -> the shipped path stays fp64, the
#   bit-exact cross-host authority). fp32 runs the INR forward in float32: ~1.7-1.9x faster (sin/tanh 3.7x,
#   GEMM 4.3x). It is LOCAL FAST-DECODE ONLY and MUST NOT ship in a contest archive: fp32 last-bit rounding
#   can move an argmax-boundary pixel by up to ~22 uint8 levels, and a different BLAS/SIMD/FMA host produces
#   DIFFERENT boundary flips -> it BREAKS cross-host bit-identity (the deterministic-reproducibility hard
#   limit: same archive.zip -> bit-identical inflate output on EVERY host). Measured score-preservation is
#   only WITHIN ~1e-4 on OUR host (#205 mid-train n24: 100*Δd_seg=-8.5e-5, Δsqrt(10*d_pose)=+1.6e-7); that
#   is same-host-deterministic (run1==run2 byte-identical) but NOT contest-portable. fp64 already meets the
#   30-min budget via the parallel Pool, so the default never needs fp32; the flag exists for dev-loop speed.
import os
for _tv in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")  # 1-thread BLAS/worker (set BEFORE numpy import); user-overridable
import sys, json, struct, math, hashlib, multiprocessing as mp
from bisect import bisect_right as _bisect_right  # #257 ξ arithmetic decoder (pure stdlib)
import numpy as np
import brotli
import torch

# FEED-ek OPT-IN fp32 forward flag (default OFF). Read ONCE at process start (deterministic within a run).
# _FDT is the forward's working dtype: float64 (default, bit-exact authority) or float32 (local fast mode).
_FP32 = os.environ.get("INFLATE_FP32", "0") == "1"
_FDT = np.float32 if _FP32 else np.float64

MAGIC = b"LVLS1\x00"
PCAR_MAGIC = b"PCAR1\x00"  # #205 warp-real-luma pose carrier (stored keyframe luma + per-pair homography)
XIP_MAGIC = b"XIP2"        # #257 store-nothing ξ payload (quantized + coded ego twist)
EKPR_MAGIC = b"EKPR1\x00"
EKPR_VERSION = 1
EKPR_APPLICATION = "frame1_phi_argmax_pre_aa_lane_r_add_clip_uint8_domain"
EKPR_HEADER = struct.Struct("<6sBIHBBI")


def _take(raw, off, n):
    # #402 fail-closed exact slice: a short buffer RAISES (never silently returns fewer bytes than
    # declared -> a downstream reshape can't quietly consume a truncated stream). advisory PR128 §8.2.
    end = off + n
    if n < 0 or end > len(raw):
        raise ValueError("LVLS1 truncated: section needs %d B at off=%d but blob is %d B" % (n, off, len(raw)))
    return raw[off:end], end


def _ekpr_parse(blob, n_pairs, n_classes, manifest_entry):
    if len(blob) < EKPR_HEADER.size:
        raise ValueError("EKPR1 truncated header")
    magic, version, npairs, nclasses, channels, dtype_code, payload_len = EKPR_HEADER.unpack_from(blob)
    if magic != EKPR_MAGIC:
        raise ValueError("bad EKPR1 magic")
    if version != EKPR_VERSION:
        raise ValueError("unsupported EKPR1 version")
    if npairs != n_pairs or nclasses != n_classes or channels != 3:
        raise ValueError("EKPR1 shape does not match LVLS1")
    if dtype_code != 1:
        raise ValueError("EKPR1 dtype is not signed-int8")
    expected_payload = int(npairs) * int(nclasses) * 3
    if payload_len != expected_payload:
        raise ValueError("EKPR1 payload length field does not match shape")
    expected_total = EKPR_HEADER.size + expected_payload
    if len(blob) < expected_total:
        raise ValueError("EKPR1 truncated payload")
    if len(blob) > expected_total:
        raise ValueError("EKPR1 has unconsumed trailing bytes")
    expected_manifest = {
        "codec": "EKPR1",
        "version": EKPR_VERSION,
        "shape": [int(npairs), int(nclasses), 3],
        "dtype": "int8",
        "application": EKPR_APPLICATION,
    }
    if manifest_entry != expected_manifest:
        raise ValueError("LVLS1 palette_residual manifest/section mismatch")
    return np.frombuffer(blob, dtype=np.int8, offset=EKPR_HEADER.size).reshape(npairs, nclasses, 3)


def _ekpr_apply(rgb, phi, residuals, pair_index):
    labels = phi.argmax(-1)
    delta = residuals[pair_index, labels].astype(rgb.dtype, copy=False)
    return np.clip(rgb + delta, 0.0, 255.0).astype(rgb.dtype, copy=False)


def _read_blob_full(path):
    raw = open(path, "rb").read()
    assert raw[:len(MAGIC)] == MAGIC, "bad level-set magic"
    off = len(MAGIC); out = []
    for _ in range(4):
        (n,) = struct.unpack_from("<I", raw, off); off += 4
        blk, off = _take(raw, off, n); out.append(blk)
    m = json.loads(out[0].decode("utf-8"))
    # Optional trailing blocks are gated by the MANIFEST flags (NOT bare off<len): 5th = lane band
    # (#224 Wave E), 6th = pose carrier (#205), 7th = chart payload (#497 gap-a: quantized int16
    # (P,6) startup pose table + fp32 (6,) scales -- the COUNTED table the ground chart derives from).
    lane_b = pcar_b = chart_b = palette_b = None
    if m.get("lane_render_band") is not None:
        (n,) = struct.unpack_from("<I", raw, off); off += 4
        lane_b, off = _take(raw, off, n)
    if m.get("pose_carrier") is not None:
        (n,) = struct.unpack_from("<I", raw, off); off += 4
        pcar_b, off = _take(raw, off, n)
    if m.get("chart_payload") is not None:
        (n,) = struct.unpack_from("<I", raw, off); off += 4
        chart_b, off = _take(raw, off, n)
    if m.get("palette_residual") is not None:
        if off + 4 > len(raw):
            raise ValueError("palette_residual manifest is present without EKPR1 section")
        (n,) = struct.unpack_from("<I", raw, off); off += 4
        palette_b, off = _take(raw, off, n)
        _ekpr_parse(palette_b, int(m["n_pairs"]), int(m["n_classes"]), m["palette_residual"])
    # #402 EXACT CONSUMPTION: the LVLS1 grammar must consume the WHOLE blob. Trailing bytes are a
    # truncation/format-mismatch/tamper signal -- fail closed, never silently ignore (advisory PR128
    # §8.2: "Section range streams are not required to be consumed exactly. Trailing words can be ignored.").
    if off != len(raw):
        raise ValueError("LVLS1 blob has %d unconsumed trailing byte(s) (off=%d len=%d) -- refusing to "
                         "decode a non-exact stream (NO-FAKE fail-closed)." % (len(raw) - off, off, len(raw)))
    return m, out[1], out[2], out[3], lane_b, pcar_b, chart_b, palette_b


def _read_blob(path):
    # Compatibility wrapper for existing parser callers. The full parser still validates and
    # exactly consumes EKPR1; receiver setup uses _read_blob_full to retain its bytes.
    m, base_b, code_b, pose_b, lane_b, pcar_b, chart_b, _palette_b = _read_blob_full(path)
    return m, base_b, code_b, pose_b, lane_b, pcar_b, chart_b


def _dequant(blob, order, shapes, scales, xcodec=None):
    out, off = {}, 0
    flat = np.frombuffer(blob, dtype=np.int8)
    perm_indices = set(int(i) for i in (xcodec or {}).get("p", []))
    if any(i < 0 or i >= len(order) for i in perm_indices):
        raise ValueError("LVLS1 xcodec has invalid base permutation index")
    for index, name in enumerate(order):
        shp = tuple(shapes[name]); n = int(np.prod(shp))
        chunk = flat[off:off + n]
        if chunk.size != n:  # #402: base blob short for this tensor -> fail closed, never partial-decode.
            raise ValueError("LVLS1 base blob short for %r: need %d int8 got %d" % (name, n, int(chunk.size)))
        if index in perm_indices:
            if len(shp) != 2:
                raise ValueError("LVLS1 xcodec transpose targets non-2-D tensor %r" % name)
            q = chunk.reshape(shp[1], shp[0]).T
        else:
            q = chunk.reshape(shp)
        out[name] = q.astype(np.float32) * float(scales[name]); off += n
    if off != flat.size:  # #402 EXACT CONSUMPTION: trailing int8 in the base blob = format mismatch.
        raise ValueError("LVLS1 base blob has %d unconsumed int8 byte(s) (off=%d size=%d) -- fail closed."
                         % (int(flat.size) - off, off, int(flat.size)))
    return out


def _decode_code_q(blob, shape, xcodec=None):
    rows, dims = int(shape[0]), int(shape[1])
    raw = np.frombuffer(blob, dtype=np.uint8)
    if raw.size != rows * dims:
        raise ValueError("LVLS1 code stream has %d B, expected %d B" % (raw.size, rows * dims))
    mode = int((xcodec or {}).get("c", 0))
    if mode == 0:
        return raw.view(np.int8).reshape(rows, dims).copy()
    if mode != 1 or rows % 2:
        raise ValueError("LVLS1 xcodec has invalid code transform id/shape")
    pairs = rows // 2; half = pairs * dims
    d0 = raw[:half].reshape(pairs, dims); d1 = raw[half:].reshape(pairs, dims)
    q0 = (np.cumsum(d0.astype(np.uint64), axis=0) & 255).astype(np.uint8)
    q1 = (np.cumsum(d1.astype(np.uint64), axis=0) & 255).astype(np.uint8)
    return np.stack([q0, q1], axis=1).reshape(rows, dims).view(np.int8).copy()


def _coords(h, w):
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32); xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)


def _curvelet_B(ns, no0, f0, base, niso, max_freq):
    cols = []
    for j in range(int(ns)):
        f_j = float(f0) * (float(base) ** j); l_j = int(no0) * (2 ** (j // 2))
        for l in range(l_j):
            th = np.pi * l / l_j; cols.append([f_j * np.cos(th), f_j * np.sin(th)])
    for i in range(int(niso)):
        th = np.pi * i / max(int(niso), 1); fl = float(f0) * 0.5; cols.append([fl * np.cos(th), fl * np.sin(th)])
    B = np.asarray(cols, np.float32).T  # (2, n)
    if max_freq is not None:
        nrm = np.sqrt((B.astype(np.float64) ** 2).sum(0)); keep = nrm <= float(max_freq) + 1e-6
        if not keep.any(): keep = nrm <= float(nrm.min()) + 1e-6
        B = B[:, keep]
    return B


def _curvelet_feats(coords, B):
    # _FDT = float64 (default, bit-exact) or float32 (FEED-ek opt-in fast mode). Default path unchanged.
    proj = (2.0 * np.pi) * (np.asarray(coords, _FDT) @ np.asarray(B, _FDT))
    return np.concatenate([np.sin(proj), np.cos(proj)], axis=-1).astype(np.float32)


def _windowed_curvelet_feats(coords, cfg):
    # Op-for-op compiler twin of
    # tac.boundary_math.windowed_curvelet_frame.windowed_curvelet_feats.  The config is generic,
    # seed-free receiver program state; learned/video-derived coefficients remain counted in P.
    required = {
        "n_scales", "n_orient0", "f0", "base", "w0", "width_ratio", "n_trans",
        "coord_margin", "min_sigma", "aniso",
    }
    if set(cfg) != required:
        raise ValueError("windowed_curvelet_config keys do not match the sealed receiver schema")
    ns, no0, nt = int(cfg["n_scales"]), int(cfg["n_orient0"]), int(cfg["n_trans"])
    if min(ns, no0, nt) <= 0:
        raise ValueError("windowed curvelet integer config must be positive")
    f0, base = float(cfg["f0"]), float(cfg["base"])
    w0, ratio = float(cfg["w0"]), float(cfg["width_ratio"])
    margin, min_sigma, aniso = (
        float(cfg["coord_margin"]), float(cfg["min_sigma"]), float(cfg["aniso"])
    )
    if not all(math.isfinite(v) for v in (f0, base, w0, ratio, margin, min_sigma, aniso)):
        raise ValueError("windowed curvelet float config must be finite")
    if min(f0, w0, min_sigma, aniso) <= 0.0 or base <= 1.0 or ratio <= 1.0:
        raise ValueError("invalid windowed curvelet scale/frequency config")
    if margin < 0.0 or aniso < 1.0:
        raise ValueError("invalid windowed curvelet margin/anisotropy config")
    if nt == 1:
        centers = [0.0]
    else:
        centers = list(np.linspace(-1.0 + margin, 1.0 - margin, nt))
    x = np.asarray(coords)[:, 0]
    y = np.asarray(coords)[:, 1]
    real_cols, imag_cols = [], []
    for j in range(ns):
        freq = f0 * (base ** j)
        n_orient = no0 * (2 ** (j // 2))
        sigma_n = max(w0 * ratio ** (-j), min_sigma)
        sigma_t = max(aniso * w0 * ratio ** (-0.5 * j), min_sigma)
        for orient in range(n_orient):
            theta = math.pi * orient / n_orient
            ct, st = math.cos(theta), math.sin(theta)
            for cy in centers:
                for cx in centers:
                    dx, dy = x - float(cx), y - float(cy)
                    u_n = dx * ct + dy * st
                    u_t = -dx * st + dy * ct
                    env = np.exp(-0.5 * ((u_n / sigma_n) ** 2 + (u_t / sigma_t) ** 2))
                    phase = (2.0 * math.pi) * freq * u_n
                    real_cols.append(env * np.cos(phase))
                    imag_cols.append(env * np.sin(phase))
    return np.concatenate(
        [np.stack(real_cols, axis=-1), np.stack(imag_cols, axis=-1)], axis=-1
    ).astype(np.float32)


def _compact_shearlet_feats(coords, cfg):
    # Op-for-op compiler twin of
    # tac.boundary_math.compact_shearlet_frame.compact_shearlet_feats.  The atom table is
    # deterministic, video-free rule-118 program state; learned coefficients stay counted.
    required = {
        "n_scales", "n_shear", "two_cones", "shear_step", "f0", "base", "w0",
        "width_ratio", "n_trans", "coord_margin", "min_sigma", "aniso",
    }
    if set(cfg) != required:
        raise ValueError("compact_shearlet_config keys do not match the sealed receiver schema")
    ns, nsh, nt = int(cfg["n_scales"]), int(cfg["n_shear"]), int(cfg["n_trans"])
    if ns <= 0 or nsh < 0 or nt <= 0 or not isinstance(cfg["two_cones"], bool):
        raise ValueError("invalid compact shearlet integer/bool config")
    shear_step, f0, base = float(cfg["shear_step"]), float(cfg["f0"]), float(cfg["base"])
    w0, ratio = float(cfg["w0"]), float(cfg["width_ratio"])
    margin, min_sigma, aniso = (
        float(cfg["coord_margin"]), float(cfg["min_sigma"]), float(cfg["aniso"])
    )
    vals = (shear_step, f0, base, w0, ratio, margin, min_sigma, aniso)
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("compact shearlet float config must be finite")
    if min(shear_step, f0, w0, min_sigma, aniso) <= 0.0 or base <= 1.0 or ratio <= 1.0:
        raise ValueError("invalid compact shearlet scale/frequency config")
    if margin < 0.0 or aniso < 1.0:
        raise ValueError("invalid compact shearlet margin/anisotropy config")
    centers = [0.0] if nt == 1 else list(np.linspace(-1.0 + margin, 1.0 - margin, nt))
    shears = [shear_step * s for s in range(-nsh, nsh + 1)]
    x = np.asarray(coords)[:, 0]
    y = np.asarray(coords)[:, 1]
    real_cols, imag_cols = [], []
    for cone in range(2 if cfg["two_cones"] else 1):
        for j in range(ns):
            freq = f0 * (base ** j)
            sigma_n = max(w0 * ratio ** (-j), min_sigma)
            sigma_t = max(aniso * w0 * ratio ** (-0.5 * j), min_sigma)
            for shear_k in shears:
                for cy in centers:
                    for cx in centers:
                        dx, dy = x - float(cx), y - float(cy)
                        if cone == 0:
                            xi, eta = dx + shear_k * dy, dy
                        else:
                            xi, eta = dy + shear_k * dx, dx
                        env = np.exp(-0.5 * ((xi / sigma_n) ** 2 + (eta / sigma_t) ** 2))
                        phase = (2.0 * math.pi) * freq * xi
                        real_cols.append(env * np.cos(phase))
                        imag_cols.append(env * np.sin(phase))
    return np.concatenate(
        [np.stack(real_cols, axis=-1), np.stack(imag_cols, axis=-1)], axis=-1
    ).astype(np.float32)


def _basis_feats(coords, m):
    family = m.get("basis_family", "polar_fourier")
    if family == "polar_fourier":
        B = _curvelet_B(
            m["bank_n_scales"], m["bank_n_orient0"], m["bank_f0"], m["bank_base"],
            m["bank_n_iso"], m["max_bank_freq"],
        )
        return _curvelet_feats(coords, B)
    if family == "windowed_curvelet":
        return _windowed_curvelet_feats(coords, m["windowed_curvelet_config"])
    if family == "compact_shearlet":
        return _compact_shearlet_feats(coords, m["compact_shearlet_config"])
    if family == "literal_polar_curvelet":
        program = BasisProgramConfig.from_dict(m["basis_program"])
        if program.canonical_sha256() != m.get("basis_program_sha256"):
            raise ValueError("literal basis-program manifest SHA-256 mismatch")
        return basis_features_numpy(coords)
    raise ValueError("unknown basis_family %r" % (family,))


def _act(u, kind, w0, s0, beta, omega):
    u = np.asarray(u, _FDT)  # _FDT = float64 (default, bit-exact) or float32 (FEED-ek opt-in fast mode)
    if kind == "hosc":
        # tanh(beta*sin(omega*u)). BIT-IDENTICAL fused form: (1) omega==1 / beta==1 skip the identity
        # full-array float64 multiply (1.0*x === x in IEEE754 -- these are ~151MB DRAM round-trips per
        # act, ~29x/pair); (2) in-place np.multiply/np.tanh reuse the sin buffer (u='pre' is not reused
        # after _act) -> fewer 151MB allocs. In-place ufunc == out-of-place ufunc elementwise -> the
        # exact float64 op values are preserved (proven by sha256 .raw parity vs the pre-fusion forward).
        t = np.sin(u) if omega == 1.0 else np.sin(omega * u)
        if beta != 1.0: np.multiply(t, beta, out=t)
        np.tanh(t, out=t)
        return t.astype(np.float32)
    if kind == "wire": return (np.cos(w0 * u) * np.exp(-((s0 * u) ** 2))).astype(np.float32)
    return np.maximum(u, 0.0).astype(np.float32)


def _in_proj_h0(P, feats, m):
    # h0 = act(feats @ in_proj.weight.T + b) -- depends ONLY on feats (NOT code) -> shared by a
    # pair's two final frames (identical feats). Returns float32 (== _act), exactly as the monolithic
    # forward's first layer. P is the float64 param dict (converted once in main).
    kw = (m["activation"], m["wire_w0"], m["wire_s0"], m["hosc_beta"], m["hosc_omega"])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return _act(np.asarray(feats, np.float64) @ P["in_proj.weight"].T + P["in_proj.bias"], *kw)


# --- #417 receiver-consumption bijection: v7.5.3 tex_trunk / out_tex_h + v8 decoupled_head. --------
# AUTO-DETECTED OPTIONAL branches (the film_pl/concat_pl idiom): each fires ONLY when the group's param
# keys are present in P. A shared-head witness (none present) => the forward is BYTE-IDENTICAL (the new
# branches are skipped). Each mirrors the trainer's MLX submodule OP-FOR-OP (parity-gated in
# tools/tests/test_receiver_bijection_v753_v8_parity.py): out_tex_h = the #395 A2 widened texture head
# (hidden->N->3 ReLU MLP); tex_trunk = the #395 band-designed Gabor texture trunk (bank REGENERATED
# free at decode, rule 118; only w_tex/bias are counted); decoupled_head = the v8 B1 K independent
# per-class DECOUPLED partition fields (replace the shared out_sdf phi). Without these branches those
# COUNTED groups paid rate but were INERT through R (a FAKE lever verdict, NO-FAKE #8) -- #417 fix half.
_TEX_BANK = {}  # cache the FREE Gabor bank per (render_h, render_w) -- regenerated, never stored.


def _tex_trunk_bank(rh, rw):
    # op-for-op mirror of tac.boundary_math.texture_trunk.build_gabor_bank_numpy with the DEFAULT
    # TextureBandSpec: periods {4,6,8} render-px x orientations {0,45,90,135}deg x cos/sin quadrature
    # => F=24. band_hi does NOT change bank content (periods are the fixed dataclass default), so ONLY
    # (H,W) parametrize the bank -> regenerable free (rule 118). Pixel coords row-major (x fastest),
    # matching _coords -> the same per-pixel ordering the softmax masks use.
    periods = (4.0, 6.0, 8.0); orients = (0.0, 45.0, 90.0, 135.0); phases = (0.0, 0.5 * np.pi)
    yy, xx = np.mgrid[0:int(rh), 0:int(rw)]
    xf = xx.astype(np.float64).ravel(); yf = yy.astype(np.float64).ravel()
    cols = []
    for p in periods:
        for o in orients:
            th = np.radians(o); proj = xf * np.cos(th) + yf * np.sin(th)
            base = 2.0 * np.pi * proj / p
            for ph in phases:
                cols.append(np.cos(base - ph))
    return np.stack(cols, axis=-1).astype(np.float32)  # (P, F=24)


def _tex_trunk_forward(bank, w_tex, bias, soft, n_classes, annulus_power):
    # op-for-op mirror of texture_trunk_numpy_forward: per-class band texture PLACED by the softmax
    # masks. bank (P,F); w_tex (F,K,3); bias (K,3); soft (P,K) -> (P,3) additive PRE-sigmoid term.
    bank = np.asarray(bank, np.float64); w_tex = np.asarray(w_tex, np.float64)
    soft64 = np.asarray(soft, np.float64)
    class_tex = np.einsum("pf,fkc->pkc", bank, w_tex) + np.asarray(bias, np.float64)[None]  # (P,K,3)
    tex = np.einsum("pk,pkc->pc", soft64, class_tex)  # (P,3) placement-aware by construction
    if annulus_power > 0.0:
        peak = soft64.max(axis=-1); frac = 1.0 / float(n_classes)
        g = np.clip((peak - frac) / (1.0 - frac), 0.0, 1.0) ** float(annulus_power)
        tex = tex * g[..., None]
    return tex


def _decoupled_phi(P, feats, code_row):
    # op-for-op mirror of tac.boundary_math.decoupled_field.decoupled_field_numpy_forward (single-pair)
    # = the v8 B1 K INDEPENDENT per-class fields (relu; the trainer hardcodes activation="relu").
    # Replaces the shared out_sdf(h) partition phi; reads the SAME coord feats the trunk in_proj reads
    # (spec in_feat == in_proj in-dim). Shapes (K,I,H / K,M,2LH / K,L,H,H / ...) come from the loaded P.
    w_in = P["decoupled_head.w_in"]; b_in = P["decoupled_head.b_in"]          # (K,I,H),(K,H)
    w_film = P["decoupled_head.w_film"]; w_hid = P["decoupled_head.w_hid"]    # (K,M,2LH),(K,L,H,H)
    b_hid = P["decoupled_head.b_hid"]; w_out = P["decoupled_head.w_out"]; b_out = P["decoupled_head.b_out"]
    kk = int(w_hid.shape[0]); _ell = int(w_hid.shape[1]); hh = int(w_hid.shape[2])
    coord = np.asarray(feats, np.float64); code = np.asarray(code_row, np.float64)
    h0 = np.maximum(np.einsum("pi,kih->pkh", coord, w_in) + b_in[None], 0.0)  # (P,K,H) relu
    film = np.einsum("m,kmf->kf", code, np.asarray(w_film, np.float64)).reshape(kk, _ell, 2, hh)
    hcur = h0  # (P,K,H)
    for li in range(_ell):
        pre = np.einsum("pkh,khj->pkj", hcur, w_hid[:, li]) + b_hid[:, li][None]
        scale = 1.0 + film[None, :, li, 0, :]; shift = film[None, :, li, 1, :]
        hcur = np.maximum(pre * scale + shift, 0.0)  # relu
    return np.einsum("pkh,kh->pk", hcur, w_out) + b_out[None]  # (P,K)


def _outputs_from_h0(P, h0, code_row, m, want_rgb, want_lane=False, feats=None):
    # op-for-op mirror of levelset_rgb_forward_numpy AFTER in_proj. h0 = the float32 in_proj act.
    # want_rgb=False -> return (phi, None) skipping the rgb head (out_tex/palette/softmax/sigmoid do
    # NOT feed phi, so argmax(phi) is identical) -- used by the self-orient fixed point.
    # want_lane=True (#224 Wave E) -> ALSO return the witness's OWN lane color sigmoid(palette[lane]+
    # tex)*255 + the softmax decision margin (top1-top2) for the decode-consistent render-band; mirror
    # of tac.boundary_math.lever_b_levelset_generator.levelset_band_forward_numpy. want_lane=False (all
    # non-band callers) -> BYTE-IDENTICAL 2-tuple return as before (default-off guarantee).
    kw = (m["activation"], m["wire_w0"], m["wire_s0"], m["hosc_beta"], m["hosc_omega"])
    cr = np.asarray(code_row, np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        film = (cr @ P["film.weight"].T + P["film.bias"]).reshape(m["n_hidden"], 2, m["hidden_dim"])
        # LEVER-A (FiLM-rank-fix) byte-close forward: AUTO-DETECT the OPTIONAL per-layer residual FiLM
        # (film_pl.*) + additive code-concat (concat_pl.*) routes from the loaded param keys so the
        # byte-close render reflects the ACTUAL trained witness (NO-FAKE). ABSENT keys (default-off
        # witness) => both branches skipped => BYTE-IDENTICAL to the pre-LEVER-A byte-close forward.
        _has_film_pl = any(str(k).startswith("film_pl.") for k in P)
        _has_concat = any(str(k).startswith("concat_pl.") for k in P)
        # #417 AUTO-DETECT (absent => shared-head byte-identical): the widened texture head, the band
        # texture trunk, the v8 decoupled partition fields. Each key is a LITERAL P[...] read below, so
        # the receiver-consumption bijection gate sees them as CONSUMED (zero orphans for those groups).
        _has_out_tex_h = "out_tex_h.weight" in P
        _has_tex_trunk = "tex_trunk.w_tex" in P
        _has_decoupled = ("decoupled_head.w_in" in P) and feats is not None
        h = h0
        for li in range(m["n_hidden"]):
            scale = 1.0 + film[li, 0]
            shift = film[li, 1]
            if _has_film_pl:
                pl = (cr @ P["film_pl.%d.weight" % li].T + P["film_pl.%d.bias" % li]).reshape(2, m["hidden_dim"])
                scale = scale + pl[0]
                shift = shift + pl[1]
            pre = (h @ P["hidden.%d.weight" % li].T + P["hidden.%d.bias" % li]) * scale + shift
            if _has_concat:
                pre = pre + (cr @ P["concat_pl.%d.weight" % li].T + P["concat_pl.%d.bias" % li])
            h = _act(pre, *kw)
        # PARTITION phi: v8 decoupled per-class fields when present (+ feats supplied), else the shared
        # out_sdf(h). The shared trunk h is STILL computed above (out_tex reads it) -- only phi changes.
        if _has_decoupled:
            phi = _decoupled_phi(P, feats, code_row)
        else:
            phi = h @ P["out_sdf.weight"].T + P["out_sdf.bias"]
        if not want_rgb:
            return phi.astype(np.float32), None
        # TEXTURE head: #395 A2 widened (hidden->N->3 ReLU MLP) when out_tex_h present, else the linear
        # out_tex. Mirror of the trainer's single _tex(h) site (both rgb + lane color route through it).
        if _has_out_tex_h:
            _th = np.maximum(h @ P["out_tex_h.weight"].T + P["out_tex_h.bias"], 0.0)  # relu
            tex = _th @ P["out_tex.weight"].T + P["out_tex.bias"]
        else:
            tex = h @ P["out_tex.weight"].T + P["out_tex.bias"]
        z = phi / float(m["softmax_temp"]); z = z - z.max(-1, keepdims=True)
        soft = np.exp(z); soft = soft / soft.sum(-1, keepdims=True)
        # TEXTURE TRUNK (#395): ADD the band texture PLACED by the SAME softmax masks, into the SAME
        # pre-sigmoid term as out_tex (trainer _compose_rgb). Bank regenerated FREE (rule 118). Guarded
        # so the shared-head pre-sigmoid stays EXACTLY `soft@palette + tex` (no `+0.0`) => byte-identical.
        _pre = soft @ P["palette"] + tex
        if _has_tex_trunk:
            _rh, _rw = int(m["render_h"]), int(m["render_w"])
            _bank = _TEX_BANK.get((_rh, _rw))
            if _bank is None:
                _bank = _tex_trunk_bank(_rh, _rw); _TEX_BANK[(_rh, _rw)] = _bank
            _pre = _pre + _tex_trunk_forward(
                _bank, P["tex_trunk.w_tex"], P["tex_trunk.bias"], soft, int(soft.shape[-1]),
                float(m.get("texture_trunk_annulus_power", 0.0)))
        rgb = (1.0 / (1.0 + np.exp(-_pre))) * 255.0
        lane_rgb = margin = None
        if want_lane:
            lc = int(m.get("lane_render_band", {}).get("lane_cls", 1))
            lane_rgb = (1.0 / (1.0 + np.exp(-(P["palette"][lc][None, :] + tex)))) * 255.0
            ss = np.sort(soft, axis=-1); margin = (ss[:, -1] - ss[:, -2])
        if not m["chroma"]:
            luma = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
            rgb = np.concatenate([luma, luma, luma], axis=-1)
            if want_lane:
                ll = 0.299 * lane_rgb[:, 0:1] + 0.587 * lane_rgb[:, 1:2] + 0.114 * lane_rgb[:, 2:3]
                lane_rgb = np.concatenate([ll, ll, ll], axis=-1)
    if want_lane:
        return (phi.astype(np.float32), rgb.astype(np.float32),
                lane_rgb.astype(np.float32), margin.astype(np.float32))
    return phi.astype(np.float32), rgb.astype(np.float32)


def _dir_feats(coords, argmax_hw, n_freqs, fa, fc, tau):
    # self-orient: tangent of the decoder's OWN all-class argmax boundary -> oriented Fourier feats.
    from scipy import ndimage
    a = np.asarray(argmax_hw); h, w = a.shape
    b = np.zeros(a.shape, bool)
    b[:-1, :] |= a[:-1, :] != a[1:, :]; b[1:, :] |= a[:-1, :] != a[1:, :]
    b[:, :-1] |= a[:, :-1] != a[:, 1:]; b[:, 1:] |= a[:, :-1] != a[:, 1:]
    if not b.any():
        tx = np.ones(h * w, np.float32); ty = np.zeros(h * w, np.float32)
    else:
        dist = ndimage.distance_transform_edt(~b).astype(np.float32)
        gy = np.zeros_like(dist); gx = np.zeros_like(dist)
        gy[1:-1, :] = (dist[2:, :] - dist[:-2, :]) * 0.5; gx[:, 1:-1] = (dist[:, 2:] - dist[:, :-2]) * 0.5
        tx = -gy.ravel(); ty = gx.ravel(); nrm = np.sqrt(tx * tx + ty * ty)
        flat = nrm < 1e-6; nrm = np.maximum(nrm, 1e-8); tx = tx / nrm; ty = ty / nrm
        tx[flat] = 1.0; ty[flat] = 0.0
    cx = coords[:, 0]; cy = coords[:, 1]; nx = ty; ny = -tx
    u_t = cx * tx + cy * ty; u_n = cx * nx + cy * ny; tp = 2.0 * np.pi
    feats = []
    for k in range(int(n_freqs)):
        a_f = float(fa) * (2.0 ** k); c_f = float(fc) * (2.0 ** k)
        feats += [np.sin(tp * a_f * u_t), np.cos(tp * a_f * u_t), np.sin(tp * c_f * u_n), np.cos(tp * c_f * u_n)]
    return np.stack(feats, axis=-1).astype(np.float32)


# --- #224 Wave E: decode-consistent analytic-lane RENDER-BAND (FREE rasterizer, rule 118) ------------
# The lane MANIFOLD COORDS (per-pair LaneLine coeffs) are the COUNTED 5th block; these fns REGENERATE
# the (H,W) coverage + composite the band over the witness render for FREE (0 archive bytes) -- op-for-
# op mirrors of tac.boundary_math.{lane_sdf_component,analytic_lane_render_band} + composite_band_on_render.
LANE_MAGIC = b"LBND1\x00"


def _lane_parse(blob):
    # bit-exact inverse of analytic_lane_render_band.serialize_lane_band. Returns (pairs_lines, header);
    # each line = (centerline f64[], halfwidth f64[], dash_period, dash_phase, dash_duty, fwd0, fwd1).
    assert blob[:len(LANE_MAGIC)] == LANE_MAGIC, "bad lane-band magic"
    off = len(LANE_MAGIC); (hlen,) = struct.unpack_from("<I", blob, off); off += 4
    hdr = json.loads(blob[off:off + hlen].decode("utf-8")); off += hlen
    vals = np.frombuffer(blob[off:], dtype=np.float64)
    pairs = []; vi = 0
    for line_metas in hdr["pairs"]:
        lines = []
        for meta in line_metas:
            nc = int(meta["nc"]); nh = int(meta["nh"]); hd = bool(meta["has_dash"])
            cc = np.asarray(vals[vi:vi + nc], np.float64); vi += nc
            hc = np.asarray(vals[vi:vi + nh], np.float64); vi += nh
            dp = dph = 0.0; dd = 0.5
            if hd:
                dp = float(vals[vi]); dph = float(vals[vi + 1]); dd = float(vals[vi + 2]); vi += 3
            fr0 = float(vals[vi]); fr1 = float(vals[vi + 1]); vi += 2
            lines.append((cc, hc, dp, dph, dd, fr0, fr1))
        pairs.append(lines)
    if vi != len(vals):  # #402 EXACT CONSUMPTION: trailing lane-coeff float64 words = format mismatch.
        raise ValueError("LBND1 lane block has %d unconsumed float64 word(s)" % (len(vals) - vi))
    return pairs, hdr


# Wave-F: LBND2 optimal RD parse -- bit-exact inverse of analytic_lane_render_band.
# serialize_lane_band_rd. PURE numpy/struct/json (NO constriction/mlx dep): quantize ->
# temporal-delta -> zigzag-uint32 stream + brotli (outer) is the entropy backend. Returns
# the SAME (cc, hc, dp, dph, dd, fr0, fr1) tuple format _lane_coverage consumes.
LANE_MAGIC_RD = b"LBND2\x00"
_D_SLOT_RD = 11


def _lane_parse_rd(blob):
    assert blob[:len(LANE_MAGIC_RD)] == LANE_MAGIC_RD, "bad LBND2 magic"
    off = len(LANE_MAGIC_RD); (hlen,) = struct.unpack_from("<I", blob, off); off += 4
    hdr = json.loads(blob[off:off + hlen].decode("utf-8")); off += hlen
    (plen,) = struct.unpack_from("<I", blob, off); off += 4
    pbytes = blob[off:off + plen]; off += plen
    rd = hdr["rd"]; K = int(rd["K"]); P = int(rd["n_pairs"]); ds = int(rd["d_slot"])
    steps = np.asarray(rd["base_steps"], np.float64); D = K * ds
    pairs = []
    if K:
        presence = np.unpackbits(np.frombuffer(pbytes, dtype=np.uint8))[:P * K].reshape(P, K).astype(bool)
        zz = np.frombuffer(blob[off:], dtype=np.uint32).reshape(P, D).astype(np.int64)
        dq = (zz >> 1) ^ -(zz & 1)                     # zigzag -> signed int64
        Q = np.cumsum(dq, axis=0)                      # undo temporal delta (row0=seed)
        M = Q.astype(np.float64) * np.tile(steps, K)   # dequant (float64, bit-exact vs module)
        for p in range(P):
            lines = []
            for slot in range(K):
                if presence[p, slot]:
                    v = M[p, slot * ds:(slot + 1) * ds]
                    lines.append((np.asarray(v[0:4], np.float64), np.asarray(v[4:6], np.float64),
                                  float(v[6]), float(v[7]), float(v[8]), float(v[9]), float(v[10])))
            pairs.append(lines)
    else:
        pairs = [[] for _ in range(P)]
    return pairs, hdr


def _lane_parse_any(blob):
    # magic dispatch: LBND2 (Wave-F RD) else LBND1 (Wave-E naive).
    if blob[:len(LANE_MAGIC_RD)] == LANE_MAGIC_RD:
        return _lane_parse_rd(blob)
    return _lane_parse(blob)


def _lane_coverage(lines, rh, rw, hdr):
    # AA-SDF range-dependent coverage raster (float64 -> float32), mirror of
    # rasterize_lane_coverage_range_dependent + _line_row_params + _forward_of_rows. Returns (rh,rw).
    g = hdr["geom"]; cam_h = float(g["cam_h"]); fx = float(g["fx"]); fy = float(g["fy"])
    v_h = float(hdr["v_h"]); soft = max(float(hdr["softness"]), 1e-6)
    dash_gate = bool(hdr["dash_gate"]); dfm = float(hdr["dash_forward_max_m"])
    cxx = float(rw / 2.0) if hdr.get("cx") is None else float(hdr["cx"])
    cov = np.zeros((rh, rw), np.float32)
    if not lines:
        return cov
    rows = np.arange(rh, dtype=np.float64); below = rows > (v_h + 1.0)
    if not below.any():
        return cov
    vr = rows[below]; col = np.arange(rw, dtype=np.float64)[None, :]
    forward = cam_h * fy / np.maximum(vr - v_h, 1e-3)
    acc = np.zeros((int(below.sum()), rw), np.float64)
    for (cc, hc, dp, dph, dd, fr0, fr1) in lines:
        lateral = np.polyval(cc, forward)
        u_c = cxx - lateral * fx / forward
        hw = np.maximum(np.polyval(hc, vr), 0.5)
        in_range = (forward >= fr0 - 1.0) & (forward <= fr1 + 5.0)
        on = np.ones_like(vr, bool)
        if dash_gate and dp > 0.0:
            near = forward < dfm
            phase = np.mod(forward - dph, dp) / dp
            on = np.where(near, phase < dd, True)
        gate = (on & in_range).astype(np.float64)
        s = hw[:, None] - np.abs(col - u_c[:, None])
        acc = np.maximum(acc, np.clip(s / soft + 0.5, 0.0, 1.0) * gate[:, None])
    cov[below] = acc.astype(np.float32)
    return cov


def _lane_uncert(margin, tau, eps):
    # mirror of analytic_lane_render_band.witness_uncertainty_mask.
    return np.clip((float(tau) - np.asarray(margin, np.float32)) / max(float(eps), 1e-6) + 0.5, 0.0, 1.0).astype(np.float32)


def _lane_composite(rgb, lane_rgb, cov_flat, margin, hdr):
    # comp = rgb*(1-a) + lane_rgb*a ; a = (cov*weight)*u_mask  (mirror composite_band_on_render/band_alpha).
    a = np.asarray(cov_flat, np.float32) * np.float32(hdr["weight"])
    um = hdr.get("u_mask")
    if um is not None and margin is not None:
        a = a * _lane_uncert(margin, um["tau"], um["eps"])
    a = a[:, None]
    return (np.asarray(rgb, np.float32) * (1.0 - a) + np.asarray(lane_rgb, np.float32) * a).astype(np.float32)


def _R(rgb, rh, rw, ch, cw):
    x = torch.from_numpy(np.ascontiguousarray(rgb.reshape(rh, rw, 3))).permute(2, 0, 1)[None].float()
    with torch.inference_mode():
        up = torch.nn.functional.interpolate(x, size=(ch, cw), mode="bicubic", align_corners=False)
        up = torch.clamp(torch.round(up), 0.0, 255.0)
    return up[0].permute(1, 2, 0).contiguous().numpy().astype(np.uint8)


# --- #205 warp-real-luma pose carrier (FREE inverse-warp; the keyframe luma + H are COUNTED) --------
# frame0 = warp(stored keyframe, stored per-pair homography H) at native res (seg-free: SegNet reads
# only frame1). VERBATIM op-for-op mirror of the tool-side _pcar_warp_frame0_from_H / pose_carrier_frame0
# (== tac.boundary_math.warp_real_luma_frame0.warp_frame0_uint8_numpy when H=homography_from_xi_numpy(xi)),
# so the shipped inflate == the numpy oracle bit-for-bit (proven by the bit-exact gate on frame0 too).
# --- #257 store-nothing ξ arithmetic decoder + DERIVE-H (rule-118 FREE: 0 archive bytes). VERBATIM
# op-for-op copies of tac.lossless.range_coder.decode_static_symbols + tac.boundary_math.xi_pose_coder
# (parse_xi_payload + homographies_from_xi) -> the shipped inflate derives the SAME H the tool oracle
# derives, bit-for-bit (bit-exact-gate proven). The 43,200 B redundant stored fp64 H is GONE.
_ST_BITS = 32; _FULL = 1 << _ST_BITS; _HALF = _FULL >> 1; _QTR = _HALF >> 1; _TQTR = _QTR * 3
# (#328 clip_profile Phase-2) sourced from the module-level profile resolution above
# (MEASURED-ANCHOR > literal fallback; byte-identical on 0.mkv — 910/582/437/1.22).
# __EMIT_HASH_BOUND_XI_CAMERA_CONTRACT__
_XI_FX = _CP_XI_FX; _XI_CX = _CP_XI_CX; _XI_CY = _CP_XI_CY; _XI_D = _CP_XI_D; _XI_EPS = 1e-6


def _ar_decode(encoded, count, freqs):
    # op-for-op range_coder.decode_static_symbols (static-frequency arithmetic decode, pure stdlib).
    if count <= 0:
        return []
    cum = [0]
    for f in freqs:
        cum.append(cum[-1] + int(f))
    total = cum[-1]
    nby = len(encoded); pos = [0, 0]  # [byte_index, bit_index]
    def _bit():
        if pos[0] >= nby:
            return 0
        b = (encoded[pos[0]] >> (7 - pos[1])) & 1
        pos[1] += 1
        if pos[1] == 8:
            pos[1] = 0; pos[0] += 1
        return b
    low = 0; high = _FULL - 1; code = 0
    for _ in range(_ST_BITS):
        code = (code << 1) | _bit()
    out = []
    for _ in range(count):
        rng = high - low + 1
        scaled = ((code - low + 1) * total - 1) // rng
        sym = _bisect_right(cum, scaled) - 1
        out.append(sym)
        high = low + (rng * cum[sym + 1] // total) - 1
        low = low + (rng * cum[sym] // total)
        while True:
            if high < _HALF:
                pass
            elif low >= _HALF:
                low -= _HALF; high -= _HALF; code -= _HALF
            elif low >= _QTR and high < _TQTR:
                low -= _QTR; high -= _QTR; code -= _QTR
            else:
                break
            low <<= 1; high = (high << 1) | 1; code = (code << 1) | _bit()
    return out


def _xip_parse(blob):
    # op-for-op xi_pose_coder.parse_xi_payload -> (q int16 (P,D), scales fp32 (D,)).
    assert blob[:4] == XIP_MAGIC, "bad xi-payload magic"
    off = 4
    cid, P, D = struct.unpack_from("<BHB", blob, off); off += 4
    scales = np.frombuffer(blob[off:off + D * 4], dtype=np.float32).copy(); off += D * 4
    if cid == 0:
        q = np.frombuffer(blob[off:off + P * D * 2], dtype=np.int16).reshape(P, D).copy()
        return q, scales
    cols = []
    for _k in range(D):
        seed, lo, hi = struct.unpack_from("<iii", blob, off); off += 12
        (mlen,) = struct.unpack_from("<I", blob, off); off += 4
        counts = np.frombuffer(brotli.decompress(blob[off:off + mlen]), dtype=np.uint32) if mlen else None
        off += mlen
        (slen,) = struct.unpack_from("<I", blob, off); off += 4
        stream = blob[off:off + slen]
        if len(stream) != slen:  # #402: xi arithmetic stream short -> fail closed.
            raise ValueError("XIP2 stream short: need %d B got %d" % (slen, len(stream)))
        off += slen
        col = np.empty(P, dtype=np.int64)
        if P:
            col[0] = seed
        if P > 1:
            if hi > lo and counts is not None:
                d = np.asarray(_ar_decode(stream, P - 1, counts.tolist()), dtype=np.int64) + lo
            else:
                d = np.full(P - 1, lo, dtype=np.int64)
            col[1:] = seed + np.cumsum(d)
        cols.append(col)
    if off != len(blob):  # #402 EXACT CONSUMPTION of the xi payload block.
        raise ValueError("XIP2 payload has %d unconsumed trailing byte(s)" % (len(blob) - off))
    return np.stack(cols, axis=1).astype(np.int16), scales


def _xip_H_from_xi(xi, pitch):
    # op-for-op xi_pose_coder.homographies_from_xi (exp_se3 -> plane-induced homography, batched).
    xi = np.asarray(xi, np.float64); rho = xi[:, :3]; om = xi[:, 3:]
    a, b, c = om[:, 0], om[:, 1], om[:, 2]; z = np.zeros_like(a)
    Ks = np.stack([np.stack([z, -c, b], -1), np.stack([c, z, -a], -1), np.stack([-b, a, z], -1)], -2)
    Ks2 = Ks @ Ks
    th2 = (om * om).sum(-1); th = np.sqrt(np.maximum(th2, 0.0))
    sm = th < _XI_EPS
    ths = np.maximum(th, _XI_EPS); th2s = np.maximum(th2, _XI_EPS * _XI_EPS); th3s = np.maximum(th ** 3, _XI_EPS ** 3)
    A = np.where(sm, 1.0 - th2 / 6.0 + th2 * th2 / 120.0, np.sin(th) / ths)
    B = np.where(sm, 0.5 - th2 / 24.0 + th2 * th2 / 720.0, (1.0 - np.cos(th)) / th2s)
    C = np.where(sm, 1.0 / 6.0 - th2 / 120.0 + th2 * th2 / 5040.0, (th - np.sin(th)) / th3s)
    eye = np.broadcast_to(np.eye(3, dtype=np.float64), Ks.shape).copy()
    R = eye + A[..., None, None] * Ks + B[..., None, None] * Ks2
    V = eye + B[..., None, None] * Ks + C[..., None, None] * Ks2
    t = (V @ rho[..., None])[..., 0]
    K = np.array([[_XI_FX, 0.0, _XI_CX], [0.0, _XI_FX, _XI_CY], [0.0, 0.0, 1.0]], np.float64)
    Kinv = np.linalg.inv(K)
    n = np.array([0.0, -np.cos(float(pitch)), -np.sin(float(pitch))], np.float64)
    M = R - (t[..., :, None] * n[None, None, :]) / _XI_D
    return K[None] @ M @ Kinv[None]


def _pcar_parse(blob):
    assert blob[:len(PCAR_MAGIC)] == PCAR_MAGIC, "bad pose-carrier magic"
    off = len(PCAR_MAGIC)
    (hlen,) = struct.unpack_from("<I", blob, off); off += 4
    hdr = json.loads(blob[off:off + hlen].decode("utf-8")); off += hlen
    P = int(hdr["n_pairs"])
    if int(hdr.get("pcar_store_nothing_v", 0)) == 2:
        # #257 store-nothing v2: decode ξ -> DERIVE H FREE (rule-118). No stored H, no kf_of_pair.
        (xlen,) = struct.unpack_from("<I", blob, off); off += 4
        q, scales = _xip_parse(blob[off:off + xlen]); off += xlen
        # #402: the v2 serializer writes a trailing u32 n_kf(=0). CONSUME + validate it (the tool-side
        # parse_pose_carrier already did; the shipped inflate previously ignored these 4 trailing bytes).
        (n_kf,) = struct.unpack_from("<I", blob, off); off += 4
        if n_kf != 0:
            raise ValueError("PCAR store-nothing v2 must have n_kf=0; got %d (NO-FAKE)" % n_kf)
        xi = q.astype(np.float64) * scales.astype(np.float64)
        H = _xip_H_from_xi(xi, float(hdr["pitch"]))
        if off != len(blob):  # #402 EXACT CONSUMPTION of the pose-carrier block.
            raise ValueError("PCAR v2 block has %d unconsumed trailing byte(s)" % (len(blob) - off))
        return {"hdr": hdr, "H": H, "keyframes": [], "kf_of_pair": [0] * P}
    H = np.frombuffer(blob[off:off + P * 9 * 8], dtype=np.float64).reshape(P, 3, 3).copy(); off += P * 9 * 8
    off += P * 6 * 2  # xi (fp16, provenance-only; the legacy decode uses the stored H)
    (n_kf,) = struct.unpack_from("<I", blob, off); off += 4
    kh, kw = int(hdr["kf_store_h"]), int(hdr["kf_store_w"])
    kfs = []
    for _ in range(n_kf):
        (blen,) = struct.unpack_from("<I", blob, off); off += 4
        raw = brotli.decompress(blob[off:off + blen]); off += blen
        kfs.append(np.frombuffer(raw, dtype=np.uint8).reshape(kh, kw, 3).copy())
    if off != len(blob):  # #402 EXACT CONSUMPTION of the legacy warp-real-luma pose-carrier block.
        raise ValueError("PCAR legacy block has %d unconsumed trailing byte(s)" % (len(blob) - off))
    return {"hdr": hdr, "H": H, "keyframes": kfs, "kf_of_pair": hdr["kf_of_pair"]}


def _pcar_upsample(kf, ch, cw):
    if kf.shape[0] == ch and kf.shape[1] == cw:
        return np.asarray(kf, dtype=np.float64)
    x = torch.from_numpy(np.ascontiguousarray(kf)).permute(2, 0, 1)[None].float()
    with torch.inference_mode():
        up = torch.nn.functional.interpolate(x, size=(ch, cw), mode="bilinear", align_corners=False)
    return up[0].permute(1, 2, 0).contiguous().numpy().astype(np.float64)


def _pcar_warp_f0(src, H, ch, cw):
    src = np.asarray(src, dtype=np.float64)
    Hh, Ww, C = int(ch), int(cw), src.shape[2]
    flat = src.reshape(-1, C)
    us, vs = np.meshgrid(np.arange(Ww), np.arange(Hh))
    grid = np.stack([us.ravel(), vs.ravel(), np.ones(Hh * Ww)], 0).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        Hinv = np.linalg.inv(np.asarray(H, np.float64))
        src_h = Hinv @ grid
        z = src_h[2]; su = src_h[0] / z; sv = src_h[1] / z
    valid = (np.isfinite(su) & np.isfinite(sv) & (z > 0)
             & (su >= 0) & (su <= Ww - 1) & (sv >= 0) & (sv <= Hh - 1))
    su_c = np.clip(su, 0.0, Ww - 1); sv_c = np.clip(sv, 0.0, Hh - 1)
    x0 = np.floor(su_c).astype(np.int64); y0 = np.floor(sv_c).astype(np.int64)
    x1 = np.minimum(x0 + 1, Ww - 1); y1 = np.minimum(y0 + 1, Hh - 1)
    wx = (su_c - x0)[:, None]; wy = (sv_c - y0)[:, None]
    Ia = flat[y0 * Ww + x0]; Ib = flat[y0 * Ww + x1]
    Ic = flat[y1 * Ww + x0]; Id = flat[y1 * Ww + x1]
    top = Ia * (1.0 - wx) + Ib * wx; bot = Ic * (1.0 - wx) + Id * wx
    sampled = top * (1.0 - wy) + bot * wy
    out = np.where(valid[:, None], sampled, flat).reshape(Hh, Ww, C)
    return np.clip(np.round(out), 0.0, 255.0).astype(np.uint8)


def _pcar_frame0(pc, pi, ch, cw):
    kf = pc["keyframes"][int(pc["kf_of_pair"][pi])]
    src = _pcar_upsample(kf, ch, cw)
    return _pcar_warp_f0(src, pc["H"][pi], ch, cw)


def _blind_retained_axis(n_in, n_out):
    # Exact align_corners=False bilinear support geometry.  This derives only which input
    # coordinates are read; no video-derived values are embedded in receiver code.
    used = np.zeros(int(n_in), dtype=bool)
    for oi in range(int(n_out)):
        source = (oi + 0.5) * float(n_in) / float(n_out) - 0.5
        lo = max(0, min(int(n_in) - 1, int(np.floor(source))))
        hi = max(0, min(int(n_in) - 1, lo + 1))
        used[lo] = True
        used[hi] = True
    return np.where(used)[0]


def _blind_coordinate_fill(frame):
    # Op twin of tac.through_r.blind_coordinate.apply_blind_fill(..., fill=None): retain the
    # scorer-visible 768x1024 product grid and fill the blind complement by separable interpolation.
    a = np.asarray(frame)
    ch, cw = a.shape[:2]
    rr = _blind_retained_axis(ch, 384)
    cc = _blind_retained_axis(cw, 512)
    sub = a[np.ix_(rr, cc)].astype(np.float64)
    full_w = np.empty((len(rr), cw, 3), dtype=np.float64)
    all_cols = np.arange(cw)
    for k in range(3):
        for ri in range(len(rr)):
            full_w[ri, :, k] = np.interp(all_cols, cc, sub[ri, :, k])
    out = np.empty((ch, cw, 3), dtype=np.float64)
    all_rows = np.arange(ch)
    for k in range(3):
        for ci in range(cw):
            out[:, ci, k] = np.interp(all_rows, rr, full_w[:, ci, k])
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _maybe_blind_coordinate_fill(frame, manifest):
    if manifest.get("blind_coordinate_fill") is None:
        return np.asarray(frame)
    return _blind_coordinate_fill(frame)


# --- #497 gap-a: COUNTED-CHART-PAYLOAD ground chart (rule-118 FREE math, counted q/scales). ---
# VERBATIM op-for-op inline of tac.boundary_math.ground_frame_chart.{_expmap_so3, plane_motion_step,
# GroundFrameChart.build} with the canonical camera constants inlined as literals (parity-pinned
# BIT-EXACT vs GroundFrameChart.build_from_xi by test_literal_chart_byte_close.py -- any constant
# or op-order drift there is a train/decode semantic fork and must fail that test, never ship).
# The per-pair charted features then come from the SEALED charted_grid_bilinear_v1 evaluator
# (charted_pair_feats_numpy -- embedded module source), the fast receiver program that replaces
# the n600-prohibitive direct sparse transform.
_CH_FX = 910.0; _CH_FY = 910.0; _CH_CX = 582.0; _CH_CY = 437.0
_CH_NW = 1164.0; _CH_NH = 874.0; _CH_D = 1.22


def _ch_expmap(omega):
    # op-for-op ground_frame_chart._expmap_so3 (Rodrigues axis-angle -> rotation, fp64).
    theta = float(np.linalg.norm(omega))
    K = np.array([[0.0, -omega[2], omega[1]],
                  [omega[2], 0.0, -omega[0]],
                  [-omega[1], omega[0], 0.0]], dtype=np.float64)
    if theta < 1e-12:
        return np.eye(3) + K
    return (np.eye(3)
            + (np.sin(theta) / theta) * K
            + ((1.0 - np.cos(theta)) / (theta * theta)) * (K @ K))


def _ch_step(pose6, s_t, s_r, pitch, regime):
    # op-for-op ground_frame_chart.plane_motion_step (ground: M = R - t n^T / d).
    if regime == "identity":
        return np.eye(3, dtype=np.float64)
    pose6 = np.asarray(pose6, dtype=np.float64).reshape(-1)
    R = _ch_expmap(s_r * np.array([pose6[3], pose6[4], pose6[5]], dtype=np.float64))
    if regime == "rotonly":
        return R
    t = s_t * np.array([pose6[2], pose6[1], pose6[0]], dtype=np.float64)
    n = np.array([0.0, -np.cos(pitch), -np.sin(pitch)], dtype=np.float64)
    return R - np.outer(t, n) / _CH_D


def _chart_H_norm(xi, ref, s_t, s_r, pitch, regime, h, w):
    # op-for-op GroundFrameChart.build: incremental ref->t composition (fp64), exact identity at ref.
    xi = np.asarray(xi, dtype=np.float64)
    P = int(xi.shape[0]); ref = int(ref)
    sx = float(w) / _CH_NW; sy = float(h) / _CH_NH
    Kp = np.array([[_CH_FX * sx, 0.0, _CH_CX * sx],
                   [0.0, _CH_FY * sy, _CH_CY * sy],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    A = np.array([[(w - 1) / 2.0, 0.0, (w - 1) / 2.0],
                  [0.0, (h - 1) / 2.0, (h - 1) / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    Ainv = np.linalg.inv(A)
    Kinv = np.linalg.inv(Kp)
    H_fwd = np.empty((P, 3, 3), dtype=np.float64)
    H_chart = np.empty((P, 3, 3), dtype=np.float64)
    M_up = np.eye(3, dtype=np.float64)
    for t in range(ref, P):
        if t > ref:
            M_up = _ch_step(xi[t], s_t, s_r, pitch, regime) @ M_up
        H_fwd[t] = np.eye(3) if t == ref else Kp @ M_up @ Kinv
    M_dn = np.eye(3, dtype=np.float64)
    for t in range(ref - 1, -1, -1):
        M_dn = M_dn @ _ch_step(xi[t + 1], s_t, s_r, pitch, regime)
        H_fwd[t] = Kp @ np.linalg.inv(M_dn) @ Kinv
    for t in range(P):
        if t == ref:
            H_chart[t] = np.eye(3, dtype=np.float64)
        else:
            H_chart[t] = Ainv @ np.linalg.inv(H_fwd[t]) @ A
    return H_chart


_G = {}


def _setup(src):
    # per-worker (spawn) / inherited-then-reset (fork) setup: dequant params + regen the FREE curvelet
    # bank + coords. Deterministic + identical across workers -> bit-exact. ~150ms, amortized over the
    # worker's pairs. Same op order as the serial main -> identical output.
    m, base_b, code_b, _pose, lane_b, pcar_b, chart_b, palette_b = _read_blob_full(src)
    params = _dequant(brotli.decompress(base_b), m["base_param_order"], m["base_shapes"], m["base_scales"], m.get("xcodec"))
    code = _decode_code_q(brotli.decompress(code_b), m["code_shape"], m.get("xcodec")).astype(np.float32) * float(m["code_scale"])
    rh, rw, ch, cw = int(m["render_h"]), int(m["render_w"]), int(m["camera_h"]), int(m["camera_w"])
    # (#497 gap-b) literal post-render supersample A_s -- SEALED semantics Y = R[A_s G(Phi(X_s))]:
    # the WHOLE feature program + nonlinear render run on the FINE (ss*rh, ss*rw) grid; A_s = exact
    # ss x ss box average AFTER the renderer, BEFORE lane compositing and BEFORE R (compose-at-base).
    # ss=1 => bit-identical to the point-sampled path (same coords/curv; _aa_down is identity).
    aa_ss = 1
    if m.get("basis_family") == "literal_polar_curvelet":
        _bp = m.get("basis_program") or {}
        if _bp.get("aa_mode") == "supersample":
            aa_ss = int(_bp.get("aa_factor", 1))
    gh, gw = rh * aa_ss, rw * aa_ss
    coords = _coords(gh, gw)  # fine grid == build_supersampled_coords(rh, rw, ss) bit-exact
    curv = _basis_feats(coords, m)
    P = {k: np.asarray(v, _FDT) for k, v in params.items()}  # convert once; _FDT=float64 default (bit-identical), float32 = FEED-ek opt-in fast mode
    # #224 Wave E: parse the OPTIONAL lane render-band (per-pair coords + hdr) -> per-pair coverage
    # rasters (FREE regen; computed ONCE per pair, cached). Absent -> lane_pairs=None -> band skipped.
    lane_pairs = lane_hdr = None
    if m.get("lane_render_band") is not None and lane_b is not None:
        lane_pairs, lane_hdr = _lane_parse_any(brotli.decompress(lane_b))  # Wave-F: LBND1/LBND2 dispatch
    # #205 warp-real-luma pose carrier: parse the OPTIONAL keyframe luma + per-pair H (frame0 warp).
    pcar = None
    if m.get("pose_carrier") is not None and pcar_b is not None:
        pcar = _pcar_parse(pcar_b)
    # #497 gap-a: parse the OPTIONAL counted chart payload -> rebuild the ground chart ONCE
    # (fp64 homography chain from the DEQUANTIZED table, op-for-op the trainer's build_from_xi)
    # + the ONCE-computed fine feats table for the sealed charted_grid_bilinear_v1 evaluator.
    chart_H = chart_fine = chart_prog = None
    if m.get("chart_payload") is not None and chart_b is not None:
        _cn = int(m["chart_payload"]["n_pairs"])
        _exp = _cn * 12 + 24
        if len(chart_b) != _exp:
            raise ValueError("chart payload is %d B != expected %d B (int16 (P,6) + fp32 (6,)) "
                             "-- fail closed (NO-FAKE)." % (len(chart_b), _exp))
        _cq = np.frombuffer(chart_b[:_cn * 12], dtype="<i2").reshape(_cn, 6)
        _cs = np.frombuffer(chart_b[_cn * 12:], dtype="<f4")
        # op-for-op xi_pose_coder.dequantize_xi: q(f8) * scales(f8).
        _xi_dq = np.asarray(_cq, dtype=np.float64) * np.asarray(_cs, dtype=np.float64)
        chart_prog = BasisProgramConfig.from_dict(m["basis_program"])
        chart_H = _chart_H_norm(
            _xi_dq, chart_prog.chart_ref_pair, chart_prog.chart_s_t, chart_prog.chart_s_r,
            chart_prog.chart_pitch, chart_prog.chart_regime, rh, rw)
        chart_fine = charted_fine_feats_cache_numpy(chart_prog, rh, rw)
    palette_residual = None
    if m.get("palette_residual") is not None and palette_b is not None:
        palette_residual = _ekpr_parse(
            palette_b, int(m["n_pairs"]), int(m["n_classes"]), m["palette_residual"]
        )
    _G.update(m=m, code=code, coords=coords, curv=curv, P=P, rh=rh, rw=rw, ch=ch, cw=cw,
              gh=gh, gw=gw, aa_ss=aa_ss,
              chart_H=chart_H, chart_fine=chart_fine, chart_prog=chart_prog,
              framebytes=ch * cw * 3, dst=None, lane_pairs=lane_pairs, lane_hdr=lane_hdr, pcar=pcar,
              palette_residual=palette_residual)


def _aa_down(x_flat, rh, rw, ss):
    # (#497) A_s: exact ss x ss box average, op-for-op the oracle's box_downsample_np call (the
    # reshape below is a VIEW composing to the same 6-D strides; mean over (2,4) is the identical
    # ufunc on the identical buffer -> bit-identical oracle/receiver). ss=1 => identity (same object).
    if ss == 1:
        return x_flat
    x2 = x_flat if x_flat.ndim == 2 else x_flat[:, None]
    c = x2.shape[-1]
    flat = x2.reshape(1, rh, ss, rw, ss, c).mean(axis=(2, 4)).reshape(rh * rw, c)
    return flat if x_flat.ndim == 2 else flat.reshape(rh * rw)


def _render_pair(pi):
    # op-for-op the serial per-pair body; each pair is INDEPENDENT so parallel == serial (bit-identical).
    # Writes the pair's 2 frames to disjoint offsets of the preallocated .raw (POSIX-safe concurrent write).
    m, code, coords, curv, P = _G["m"], _G["code"], _G["coords"], _G["curv"], _G["P"]
    rh, rw, ch, cw = _G["rh"], _G["rw"], _G["ch"], _G["cw"]
    gh, gw, aa_ss = _G["gh"], _G["gw"], _G["aa_ss"]  # #497: fine grid dims (== rh, rw at ss=1)
    _literal_program = None
    if m.get("basis_family") == "literal_polar_curvelet":
        _literal_program = BasisProgramConfig.from_dict(m["basis_program"])
    # (#497 gap-a) per-pair base features: charted (sealed charted_grid_bilinear_v1; identity ref
    # pair = exact uncharted grid) when the chart payload is active, else the shared static bank.
    _chart_H = _G.get("chart_H")
    if _chart_H is not None:
        base = charted_pair_feats_numpy(
            _G["chart_prog"], rh, rw, _chart_H[pi], _G["chart_fine"]
        )
    else:
        base = curv
    if _literal_program is not None and _literal_program.native_orientation_enabled:
        _scale_ids, _angles = orientation_metadata_from_atom_specs(ATOM_SPECS)

        def _decode_native(_features):
            _h0_native = _in_proj_h0(P, _features, m)
            _phi_native, _ = _outputs_from_h0(
                P, _h0_native, code[2 * pi + 1], m, False, feats=_features
            )
            return _phi_native.argmax(-1).reshape(gh, gw).astype(np.int64)

        # (#497 gap-a) chart x native: normal COVECTORS transform through J^-T (embedded
        # curvelet_placement ops) -- op-for-op the trainer + tool oracle chart branch.
        _normal_transform = None
        if _chart_H is not None:
            _jac = projective_jacobian_numpy(coords, _chart_H[pi])

            def _normal_transform(_normals):
                return transform_normal_covector_numpy(_normals, _jac)

        feats, _native_gates, _native_receipt = native_orientation_fixed_point_numpy(
            base,
            _decode_native,
            _scale_ids,
            _angles,
            kappa=_literal_program.native_orientation_kappa,
            iteration_cap=_literal_program.fixed_point_iteration_cap,
            normal_transform=_normal_transform,
        )
        h0 = _in_proj_h0(P, feats, m)
    elif m["self_orient"]:
        # fixed-point: dir feats from the decoder's OWN frame1 argmax (GT-free, 0 bytes).
        dirf = np.zeros((base.shape[0], 4 * int(m["n_dir_freqs"])), np.float32)
        prev_am = None; _h0_conv = None  # FEED-ej: cache the converged iter's h0 (== the post-loop h0)
        for _ in range(int(m["so_iters"])):
            feats = np.concatenate([base, dirf], axis=-1)
            _h0_it = _in_proj_h0(P, feats, m)
            phi, _ = _outputs_from_h0(P, _h0_it, code[2 * pi + 1], m, False, feats=feats)
            am = phi.argmax(-1).reshape(gh, gw).astype(np.int64)  # #497: grid dims (fine when ss>1)
            if prev_am is not None and np.array_equal(am, prev_am):
                _h0_conv = _h0_it  # converged: feats frozen from here -> this h0 IS the final h0 (bit-exact reuse)
                break  # argmax fixed point: dirf would not change -> remaining iters are no-ops
            dirf = _dir_feats(coords, am, m["n_dir_freqs"], m["so_freq_along"], m["so_freq_across"], m["so_tau"])
            prev_am = am
        feats = np.concatenate([base, dirf], axis=-1)
        # FEED-ej: on the early-break `feats` is unchanged from the converged iter -> reuse its h0
        # (bit-exact: identical feats -> identical _in_proj_h0). No convergence -> recompute (as before).
        h0 = _h0_conv if _h0_conv is not None else _in_proj_h0(P, feats, m)
    else:
        feats = base
        h0 = _in_proj_h0(P, feats, m)  # shared across the pair's 2 frames (identical feats)
    lane_pairs, lane_hdr = _G.get("lane_pairs"), _G.get("lane_hdr")
    band = lane_pairs is not None and pi < len(lane_pairs)
    pcar = _G.get("pcar")
    cov_flat = None
    if band:
        # coverage depends ONLY on the pair's lines (per-pair) -> rasterize ONCE, share across f0/f1.
        cov_flat = _lane_coverage(lane_pairs[pi], rh, rw, lane_hdr).reshape(-1)
    frames = []
    for fk in range(2):
        # #205: frame0 (fk==0) is the pose carrier (seg-free: SegNet reads only frame1). frame1
        # (fk==1) stays the witness render (d_seg). Absent pose carrier -> frame0 is the INR render.
        #  * warp_real_luma: frame0 = warp(stored keyframe, stored per-pair H).
        #  * store_nothing:  frame0 = warp(the witness's OWN frame0 INR render, stored per-pair H)
        #                    -- NO stored keyframe (rule-118: the render is FREE, only xi/H COUNTED).
        if fk == 0 and pcar is not None and pi < int(pcar["hdr"]["n_pairs"]):
            if pcar["hdr"].get("pose_carrier_mode") == "store_nothing":
                _phi0, rgb0 = _outputs_from_h0(P, h0, code[2 * pi + 0], m, True, feats=feats)
                # #497: A_s (fine render -> base) BEFORE R; identity at ss=1.
                src0 = _R(_aa_down(rgb0, rh, rw, aa_ss), rh, rw, ch, cw).astype(np.float64)
                frame = _pcar_warp_f0(src0, pcar["H"][pi], ch, cw)
            else:
                frame = _pcar_frame0(pcar, pi, ch, cw)
            frames.append(_maybe_blind_coordinate_fill(frame, m).tobytes())
            continue
        if band:
            _phi, rgb, lane_rgb, margin = _outputs_from_h0(P, h0, code[2 * pi + fk], m, True, True, feats=feats)
            if fk == 1 and _G.get("palette_residual") is not None:
                rgb = _ekpr_apply(rgb, _phi, _G["palette_residual"], pi)
            # #497 sealed order: A_s integrates rgb/lane_rgb/margin to BASE first, then the
            # uncertainty mask + composite run at base (cov is base-rasterized). Identity at ss=1.
            rgb = _aa_down(rgb, rh, rw, aa_ss)
            lane_rgb = _aa_down(lane_rgb, rh, rw, aa_ss)
            margin = _aa_down(margin, rh, rw, aa_ss)
            rgb = _lane_composite(rgb, lane_rgb, cov_flat, margin, lane_hdr)
        else:
            _phi, rgb = _outputs_from_h0(P, h0, code[2 * pi + fk], m, True, feats=feats)
            if fk == 1 and _G.get("palette_residual") is not None:
                rgb = _ekpr_apply(rgb, _phi, _G["palette_residual"], pi)
            rgb = _aa_down(rgb, rh, rw, aa_ss)  # #497: A_s after the renderer; identity at ss=1.
        frame = _R(rgb, rh, rw, ch, cw)
        frames.append(_maybe_blind_coordinate_fill(frame, m).tobytes())
    fb = _G["framebytes"]
    with open(_G["dst"], "r+b") as f:
        f.seek(pi * 2 * fb); f.write(b"".join(frames))
    return pi


def _init_worker(src, dst):
    _setup(src); _G["dst"] = dst


def main():
    src, dst = sys.argv[1], sys.argv[2]
    _setup(src)  # main process: get n_pairs + dims (workers re-setup via _init_worker)
    m = _G["m"]; n_pairs = int(m["n_pairs"]); ch, cw = _G["ch"], _G["cw"]
    _cap = int(os.environ.get("INFLATE_MAX_PAIRS", "0"))  # 0 => all pairs (contest default); >0 = debug/CI bounded inflate
    if _cap > 0:
        n_pairs = min(n_pairs, _cap)
    # #402 ATOMIC WRITE: workers write to a ``.partial`` sibling; only after the FULL expected byte
    # count is verified do we os.replace() it onto ``dst``. A crash/OOM before the rename leaves the
    # ``.partial`` (never a full-size, scoreable-LOOKING ``dst``) -> the evaluator cannot score a
    # truncated output (advisory PR128 §8.2: "output writes are non-atomic and non-resumable").
    tmp = dst + ".partial"
    expected_raw = 2 * n_pairs * _G["framebytes"]
    with open(tmp, "wb") as f:  # preallocate the full .raw so workers write disjoint offsets
        f.truncate(expected_raw)
    nworkers = max(1, min(n_pairs, int(os.environ.get("INFLATE_WORKERS", "0")) or (os.cpu_count() or 1)))
    if nworkers == 1:  # serial fallback (bit-identical) -- e.g. INFLATE_WORKERS=1 for debugging
        _G["dst"] = tmp
        for pi in range(n_pairs):
            _render_pair(pi)
    else:
        methods = mp.get_all_start_methods()
        ctx = mp.get_context("fork" if "fork" in methods else "spawn")  # Linux fork / macOS spawn
        with ctx.Pool(nworkers, initializer=_init_worker, initargs=(src, tmp)) as pool:
            for _ in pool.imap_unordered(_render_pair, range(n_pairs), chunksize=1):
                pass
    # #402 FINAL RAW ASSERTION: the output MUST be exactly the expected byte count BEFORE it is
    # promoted to ``dst`` (== before scoring). A short raw = evaluator truncation = NO-FAKE failure.
    actual_raw = os.path.getsize(tmp)
    if actual_raw != expected_raw:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise SystemExit("inflate output SHORT: %d B != expected %d B (%d frames @ %dx%dx3) -- refusing "
                         "to promote a truncated .raw to a scoreable path (NO-FAKE)."
                         % (actual_raw, expected_raw, 2 * n_pairs, ch, cw))
    os.replace(tmp, dst)  # atomic rename onto the scoreable path (POSIX same-filesystem guarantee).
    print("inflated %d frames (%d pairs) -> %s [%dx%dx%dx3 uint8] (%d workers)" % (2 * n_pairs, n_pairs, dst, 2 * n_pairs, ch, cw, nworkers), flush=True)


if __name__ == "__main__":
    main()
'''
_INFLATE_PY = _bind_xi_camera_contract(_INFLATE_PY)

_INFLATE_SH = """#!/usr/bin/env bash
# Level-set inflate launcher. Produces <OUTPUT_DIR>/<base>.raw = flat uint8 (N,874,1164,3).
# Interpreter resolution: honor ${PYTHON} (canonical contest_auth_eval._run_inflate sets it to the
# deps-complete eval interpreter), else fall back to python3 (python3 is present on macOS + most
# Linux x86_64 CPU hosts where bare `python` is NOT -- the runtime-closure bug class that left
# inflate.sh dead with `python: command not found`). Generic launcher code, rule-118 clean.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYBIN="${PYTHON:-python3}"
DATA_DIR="$1"; OUTPUT_DIR="$2"; FILE_LIST="$3"
mkdir -p "$OUTPUT_DIR"
while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  SRC="${DATA_DIR}/${BASE}.bin"
  DST="${OUTPUT_DIR}/${BASE}.raw"
  [ ! -f "$SRC" ] && echo "ERROR: ${SRC} not found" >&2 && exit 1
  printf "Inflating %s ... " "$line"
  "$PYBIN" "${HERE}/inflate.py" "$SRC" "$DST"
done < "$FILE_LIST"
"""


# ---------------------------------------------------------------------------
# packet assembly: archive.zip (0.bin) + inflate.py + inflate.sh
# ---------------------------------------------------------------------------
def _inflate_source_for_manifest(manifest: dict[str, Any]) -> str:
    receiver_source = _INFLATE_PY
    if manifest.get("palette_residual") is not None:
        # EKPR1 is the first optional receiver path whose focused oracle executes the
        # complete shipped source on a tiny fixture. Bind the already-canonical clip
        # profile into that fresh interpreter. The absent LVLS1 payload remains
        # byte-identical to the pre-EKPR payload; receiver source bytes intentionally
        # grow to implement the optional parser and consumer.
        receiver_source = receiver_source.replace(
            "# SPDX-License-Identifier: MIT\n",
            "# SPDX-License-Identifier: MIT\n"
            f"_CP_XI_FX = {float(_CP_XI_FX)!r}\n"
            f"_CP_XI_CX = {float(_CP_XI_CX)!r}\n"
            f"_CP_XI_CY = {float(_CP_XI_CY)!r}\n"
            f"_CP_XI_D = {float(_CP_XI_D)!r}\n",
            1,
        )
    if manifest.get("basis_family") != LITERAL_POLAR_CURVELET:
        return receiver_source
    else:
        # Rule-118 FREE generic source, content-bound by BasisProgramConfig.  The
        # literal atom program and placement fixed point are prepended so the
        # contest packet remains package-independent.
        placement_source = (
            _REPO / "src" / "tac" / "boundary_math" / "curvelet_placement.py"
        ).read_text(encoding="utf-8")
        placement_source = placement_source.replace("from __future__ import annotations\n", "")
        return (
            literal_curvelet_generated_source()
            + "\n"
            + placement_source
            + "\n"
            + receiver_source
        )


def assemble_packet(blob: bytes, packet_dir: Path) -> tuple[Path, int]:
    packet_dir.mkdir(parents=True, exist_ok=True)
    zip_path = packet_dir / "archive.zip"
    info = zipfile.ZipInfo(filename="0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(info, blob)
    # Parse with the host-side mirror. ``_io_unpack`` exists only inside the
    # generated receiver string and is intentionally not a module global.
    manifest = _read_blob_bytes(blob)[0]
    inflate_source = _inflate_source_for_manifest(manifest)
    (packet_dir / "inflate.py").write_text(inflate_source)
    sh = packet_dir / "inflate.sh"
    sh.write_text(_INFLATE_SH)
    sh.chmod(0o755)
    return zip_path, int(zip_path.stat().st_size)


# ---------------------------------------------------------------------------
# run inflate (subprocess, exactly as the contest evaluate.sh does)
# ---------------------------------------------------------------------------
def _raw_storage_preflight(out_dir: Path, expected_bytes: int, *, safety: float = 1.05) -> dict[str, Any]:
    """#402 STORAGE PREFLIGHT: refuse to start a ~3.66 GB inflate that would fill the output volume.

    Contest 1,200-frame output = 1,200*874*1164*3 = 3,662,409,600 B (~3.41 GiB). Per CLAUDE.md's
    storage-waterfall non-negotiable, a large-artifact producer MUST fail CLOSED when no tier has
    room -- silently filling the disk mid-inflate corrupts the run AND every sibling job on the
    volume. ``shutil.disk_usage`` on the actual output directory's filesystem is the ground truth."""
    import shutil

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    free = int(shutil.disk_usage(out_dir).free)
    required = int(expected_bytes * float(safety))
    info = {
        "path": str(out_dir), "free_bytes": free, "expected_raw_bytes": int(expected_bytes),
        "required_bytes": required, "safety_margin": float(safety), "ok": bool(free >= required),
    }
    if free < required:
        raise RuntimeError(
            f"storage preflight FAILED: {out_dir} has {free} B free < {required} B required "
            f"({expected_bytes / 1e9:.2f} GB raw output + {(safety - 1) * 100:.0f}% margin) -- refusing "
            f"to start the inflate (CLAUDE.md storage waterfall / NO-FAKE fail-closed).")
    return info


def run_inflate(packet_dir: Path, n_pairs_total: int, max_pairs: int | None) -> dict[str, Any]:
    """Unzip archive.zip -> 0.bin, run inflate.py -> 0.raw, validate the FULL output shape.

    ``max_pairs`` (test/speed): when set < n_pairs, a CAPPED 0.bin is written so inflate emits
    only the first ``max_pairs`` pairs (still frame0+frame1 each). The full archive.zip always
    encodes ALL codes (the rate term is the full blob)."""
    archive_dir = packet_dir / "archive"
    inflated_dir = packet_dir / "inflated"
    archive_dir.mkdir(exist_ok=True)
    inflated_dir.mkdir(exist_ok=True)
    # RUNTIME QUARANTINE GATE (operator 2026-07-21): refuse to decode a retired-vehicle
    # archive's BYTES here — this fires in main, every worktree, and every subagent.
    from tac.artifact_quarantine import assert_not_quarantined_archive
    assert_not_quarantined_archive(packet_dir / "archive.zip", context="levelset_byte_close decode")
    import shutil

    from tac.submission_archive import safe_extract_zip

    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    safe_extract_zip(packet_dir / "archive.zip", archive_dir)
    src_bin = archive_dir / "0.bin"

    eval_pairs = n_pairs_total if max_pairs is None else min(int(max_pairs), n_pairs_total)
    if eval_pairs < n_pairs_total:
        import brotli

        man, base_b, code_b, pose_b, lane_b, pcar_b, chart_b, palette_b = _read_blob_bytes_full(
            src_bin.read_bytes()
        )
        code = _decode_code(man, code_b)
        code_cap = code[: 2 * eval_pairs]
        qc, sc = _int8_symmetric(code_cap)
        man["n_pairs"] = eval_pairs
        man["code_shape"] = list(code_cap.shape)
        man["code_scale"] = float(sc)
        # #224 Wave E: cap the lane render-band to eval_pairs too (slice + re-serialize).
        lane_cap = None
        if lane_b is not None and man.get("lane_render_band") is not None:
            all_pairs, lane_hdr = deserialize_lane_band_any(brotli.decompress(lane_b))  # Wave-F dispatch
            lane_cfg_cap = render_config_from_header({**man["lane_render_band"], "pairs": []})
            lane_cap = brotli.compress(
                serialize_lane_band_any(all_pairs[:eval_pairs], lane_cfg_cap, lane_hdr), quality=11)
        # #205: cap the pose carrier to eval_pairs (slice H/xi + prune to referenced keyframes).
        pcar_cap = None
        if pcar_b is not None and man.get("pose_carrier") is not None:
            pcar_cap = _cap_pose_carrier(pcar_b, eval_pairs)
            man["pose_carrier"] = {**man["pose_carrier"], "n_pairs": eval_pairs}
        # #497 gap-a: cap the chart payload to eval_pairs. Valid because the chart's incremental
        # composition H_fwd[t] depends only on poses[ref+1..t]: slicing the FIRST eval_pairs rows
        # preserves every kept pair's homography EXACTLY (scales are per-channel over the FULL
        # table and ship unchanged -> kept rows dequantize to identical values). ref must survive.
        chart_cap = None
        if chart_b is not None and man.get("chart_payload") is not None:
            _cp = _parse_chart_payload(chart_b, int(man["chart_payload"]["n_pairs"]))
            _ref = int((man.get("basis_program") or {}).get("chart_ref_pair", 0))
            if _ref >= eval_pairs:
                raise ValueError(
                    f"chart cap: ref_pair {_ref} >= eval_pairs {eval_pairs} -- the capped decode "
                    "would lose the identity reference pair (fail closed)."
                )
            chart_cap = _chart_payload_bytes(_cp["q"][:eval_pairs], _cp["scales"])
            man["chart_payload"] = {**man["chart_payload"], "n_pairs": eval_pairs}
        palette_cap = None
        if palette_b is not None and man.get("palette_residual") is not None:
            palette_cap = cap_palette_residual(palette_b, eval_pairs)
            man["palette_residual"] = decode_palette_residual(palette_cap).manifest
        mj = json.dumps(man, separators=(",", ":")).encode()
        capped = _io_pack(mj, base_b, _encode_code_brotli(qc, man),
                          pose_b or None, lane_cap, pcar_cap, chart_cap, palette_cap)
        src_bin.write_bytes(capped)

    dst_raw = inflated_dir / "0.raw"
    n_frames_expected = 2 * eval_pairs
    expected_bytes = n_frames_expected * CAMERA_H * CAMERA_W * 3
    # #402 STORAGE PREFLIGHT: fail closed if the output volume cannot hold the ~3.66 GB .raw.
    storage = _raw_storage_preflight(inflated_dir, expected_bytes)
    cmd = [sys.executable, str(packet_dir / "inflate.py"), str(src_bin), str(dst_raw)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(packet_dir))
    if proc.returncode != 0:
        raise RuntimeError(f"inflate.py FAILED rc={proc.returncode}\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}")
    actual_bytes = dst_raw.stat().st_size
    # #402 FINAL RAW ASSERTION (fail closed): a short .raw is evaluator truncation -- RAISE, do not
    # merely record a bool that a downstream caller might ignore and score anyway.
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"run_inflate: .raw is {actual_bytes} B != expected {expected_bytes} B "
            f"({n_frames_expected} frames @ {CAMERA_H}x{CAMERA_W}x3) -- a short raw is evaluator "
            f"truncation; refusing to hand a truncated output to scoring (NO-FAKE).")
    return {
        "inflate_stdout": proc.stdout.strip(),
        "eval_pairs": eval_pairs,
        "n_frames_emitted": n_frames_expected,
        "raw_path": str(dst_raw),
        "raw_bytes": actual_bytes,
        "expected_bytes": expected_bytes,
        "full_output_shape_ok": bool(actual_bytes == expected_bytes),
        "storage_preflight": storage,
        "frame_layout": f"({n_frames_expected}, {CAMERA_H}, {CAMERA_W}, 3) uint8 [f0,f1 per pair]",
    }


# ---------------------------------------------------------------------------
# d_seg / d_pose parity ON THE INFLATED FRAMES (frozen CPU-torch, vs GT)
# ---------------------------------------------------------------------------
def parity_on_inflated(raw_path: Path, eval_pairs: int, gt_cache: str | None, num_pairs: int) -> dict[str, Any]:
    """Read the .raw back, run the FROZEN CPU-torch SegNet/PoseNet on the INFLATED frames over all
    eval pairs, return realized d_seg/d_pose vs GT (the contest-faithful realized numbers)."""
    if gt_cache:
        gt, seg_cpu, posenet_cpu = twr.load_gt_from_cache(Path(gt_cache), num_pairs)
    else:
        gt, seg_cpu, posenet_cpu = twr.precompute_gt(num_pairs)
    P = min(eval_pairs, gt.n_pairs)
    frame_bytes = CAMERA_H * CAMERA_W * 3
    f0s, f1s = [], []
    with open(raw_path, "rb") as f:
        for _pi in range(P):
            f0s.append(np.frombuffer(f.read(frame_bytes), dtype=np.uint8).reshape(CAMERA_H, CAMERA_W, 3))
            f1s.append(np.frombuffer(f.read(frame_bytes), dtype=np.uint8).reshape(CAMERA_H, CAMERA_W, 3))
    d_segs = twr.cpu_verdict_d_seg_batch(seg_cpu, f1s, [gt.lstars[pi] for pi in range(P)])
    d_poses = twr.cpu_verdict_d_pose_batch(posenet_cpu, f0s, f1s, [gt.gt_poses[pi] for pi in range(P)])
    return {
        "pairs_scored": P,
        "d_seg_realized_on_inflated": float(np.mean(d_segs)),
        "d_pose_realized_on_inflated": float(np.mean(d_poses)),
    }


def _dequant_blob(blob: bytes) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray,
                                        list[list[Any]] | None, dict[str, Any] | None,
                                        dict[str, np.ndarray] | None]:
    """Dequant (int8 -> fp32*scale) the base params + code from a LVLS1 blob, EXACTLY as the shipped
    inflate does (the numpy-fp32 oracle authority uses the SAME dequantized weights). Returns
    (manifest, params, code, lane_pairs|None, pose_carrier_parsed|None, chart_payload|None); the
    chart payload is the #497 gap-a counted table {"q": int16 (P,6), "scales": fp32 (6,)}. Used by
    store_nothing's ``pose_carrier_confirm`` to regenerate the witness-render frame0 authority."""
    import brotli

    m, base_b, code_b, _pose, lane_b, pcar_b, chart_b = _read_blob_bytes(blob)
    params = _decode_base_params(m, base_b)
    code = _decode_code(m, code_b)
    lane_pairs: list[list[Any]] | None = None
    if lane_b is not None and m.get("lane_render_band") is not None:
        lane_pairs, _hdr = deserialize_lane_band_any(brotli.decompress(lane_b))
    pcar_parsed = parse_pose_carrier(pcar_b) if (pcar_b is not None and m.get("pose_carrier") is not None) else None
    chart_parsed = (
        _parse_chart_payload(chart_b, int(m["chart_payload"]["n_pairs"]))
        if (chart_b is not None and m.get("chart_payload") is not None) else None
    )
    return m, params, code, lane_pairs, pcar_parsed, chart_parsed


def pose_carrier_confirm(
    raw_path: Path, eval_pairs: int, gt_cache: str | None, num_pairs: int, pose_carrier_bytes: bytes,
    blob: bytes | None = None,
) -> dict[str, Any]:
    """CONFIRM the pose-carrier decode through the REAL byte-closed inflate (the #205 launch gate).

    (1) frame0 DECODE-REPRODUCTION: the inflated .raw frame0 == the numpy AUTHORITY frame0 BIT-FOR-BIT
        (proves the shipped inflate warps by the stored H exactly -> the training-side warp d_pose IS
        the real-decode d_pose, no gap by construction). The authority frame0 is mode-dependent:
          * warp_real_luma: ``pose_carrier_frame0(pc, pi)`` = warp(stored keyframe, H).
          * store_nothing:  the numpy-fp32 ORACLE frame0 = warp(the witness's OWN f0 render, H) --
            regenerated from ``blob`` via ``numpy_oracle_reference_frames`` (needs ``blob``; NO-FAKE).
    (2) CONTEXT d_pose triad (frozen CPU-torch PoseNet, the authority; NEVER MPS):
        * carrier (raw f0 = warp, raw f1 = witness) -- the REAL row d_pose (what evaluate.py scores);
        * unwarped (gt_f0 real, raw f1 = witness) -- isolates the warp's marginal effect;
        * CEILING (gt_f0 real, warp(gt_f0, H)) -- the (real, warped-real) pair the reference tool
          ``measure_warp_dpose_through_R`` measures. The carrier-vs-ceiling GAP is the OPEN
          witness-f1 / trained-residual work.
    NON-PROMOTABLE (macOS-CPU advisory)."""
    pc = parse_pose_carrier(pose_carrier_bytes)
    mode = pose_carrier_mode(pc)
    if gt_cache:
        gt, _seg, posenet_cpu = twr.load_gt_from_cache(Path(gt_cache), num_pairs)
    else:
        gt, _seg, posenet_cpu = twr.precompute_gt(num_pairs)
    P = min(int(eval_pairs), gt.n_pairs, int(pc["hdr"]["n_pairs"]))
    fb = CAMERA_H * CAMERA_W * 3
    raw_f0, raw_f1 = [], []
    with open(raw_path, "rb") as f:
        for _pi in range(P):
            raw_f0.append(np.frombuffer(f.read(fb), dtype=np.uint8).reshape(CAMERA_H, CAMERA_W, 3))
            raw_f1.append(np.frombuffer(f.read(fb), dtype=np.uint8).reshape(CAMERA_H, CAMERA_W, 3))
    gt_f0 = [np.asarray(gt.gt_f0[pi]) for pi in range(P)]
    poses = [gt.gt_poses[pi] for pi in range(P)]
    # (1) frame0 decode-reproduction (numpy authority == the shipped inflate .raw).
    if mode == "store_nothing":
        if blob is None:
            raise ValueError(
                "store_nothing pose_carrier_confirm needs the LVLS1 blob to regenerate the "
                "witness-render frame0 authority (the frame0 source is the FREE INR render, not a "
                "stored keyframe). NO-FAKE: refusing to fabricate the authority.")
        m, params, code, lane_pairs, pc_oracle, chart_oracle = _dequant_blob(blob)
        oracle_frames, _am = numpy_oracle_reference_frames(
            params, code, m, P, lane_pairs, pose_carrier=pc_oracle, chart_payload=chart_oracle)
        auth_f0 = [np.asarray(oracle_frames[2 * pi]) for pi in range(P)]
        # store_nothing ceiling = (real f0, warp(real f0, H)) -- the keyframe-INDEPENDENT reference
        # (matches measure_warp_dpose_through_R), NOT warp(witness render) which is non-photoreal.
        ceiling_f1 = [pose_carrier_frame0_from_source(pc, pi, gt_f0[pi]) for pi in range(P)]
    else:
        auth_f0 = [pose_carrier_frame0(pc, pi) for pi in range(P)]
        ceiling_f1 = auth_f0  # warp_real_luma: stored keyframe ~= gt_f0, warp(keyframe, H) is the ceiling
    max_abs = 0
    n_diff = 0
    for pi in range(P):
        d = int(np.abs(raw_f0[pi].astype(np.int32) - auth_f0[pi].astype(np.int32)).max())
        if d != 0:
            n_diff += 1
        max_abs = max(max_abs, d)
    # (2) d_pose triad (authority PoseNet).
    dp_carrier = twr.cpu_verdict_d_pose_batch(posenet_cpu, raw_f0, raw_f1, poses)
    dp_unwarped = twr.cpu_verdict_d_pose_batch(posenet_cpu, gt_f0, raw_f1, poses)
    dp_ceiling = twr.cpu_verdict_d_pose_batch(posenet_cpu, gt_f0, ceiling_f1, poses)
    return {
        "pairs": P,
        "pose_carrier_mode": mode,
        "frame0_decode_bit_exact": bool(max_abs == 0),
        "frame0_max_abs_uint8_diff": int(max_abs),
        "frame0_n_frames_differing": int(n_diff),
        "d_pose_carrier_warp_f0_witness_f1": float(np.mean(dp_carrier)),
        "d_pose_unwarped_gtf0_witness_f1": float(np.mean(dp_unwarped)),
        "d_pose_ceiling_gtf0_warpf0": float(np.mean(dp_ceiling)),
        "d_pose_null_pose_blind_ref": 189.62,
        "training_side_vs_real_decode_parity": (
            ("IDENTICAL by construction: raw frame0 == numpy-authority "
             + ("store-nothing witness-render warp" if mode == "store_nothing" else "stored-keyframe warp")
             + " frame0 bit-for-bit (above) AND raw frame1 == witness render (bit-exact gate) -> the "
               "training-side warp d_pose EQUALS the real-decode d_pose (no surrogate gap).")
            if max_abs == 0 else
            f"MISMATCH: raw frame0 != authority (max_abs={max_abs}) -> the byte-close does NOT "
            "faithfully reproduce the pose decode (NO-FAKE: investigate)."),
        "gap_note": (
            "carrier (warp-f0 + WITNESS-f1) vs ceiling (real-f0 + warped-real-f1): the gap is the OPEN "
            "witness-composition work -- the witness f1 is non-photoreal + the trained dxi residual "
            "(w_pose>0) is UNMEASURED. The ceiling is the reference (real,warped-real) pair, NOT the "
            "contest-legal witness pair. NO borrowed 3.4e-5 (ancestor-RGB, never validated on the witness)."
            + (" store_nothing stores ONLY xi/H (~0 marginal bytes) -- the frame0 source is the FREE "
               "witness render, so the WHOLE keyframe payload (the warp_real_luma rate) is GONE." if mode == "store_nothing" else "")),
        "authority": _AUTHORITY,
        "promotion_claim": False,
    }


def _pose_carrier_confirmation_payload(packet_dir: Path) -> tuple[bytes, bytes]:
    """Return the exact blob and pose section that ``run_inflate`` decoded.

    A capped run rewrites ``archive/0.bin`` with sliced and re-quantized code plus a
    correspondingly capped pose-carrier section.  Confirmation must consume those
    exact bytes, not the original full blob used to assemble ``archive.zip``.
    """

    scored_blob_path = packet_dir / "archive" / "0.bin"
    if not scored_blob_path.is_file():
        raise RuntimeError(
            f"pose-carrier confirmation missing decoded blob: {scored_blob_path}"
        )
    scored_blob = scored_blob_path.read_bytes()
    scored_pose_carrier = _read_blob_bytes(scored_blob)[5]
    if scored_pose_carrier is None:
        raise RuntimeError(
            "pose-carrier confirmation requested but the exact decoded blob has no pose section"
        )
    return scored_blob, scored_pose_carrier


# ---------------------------------------------------------------------------
# BIT-EXACT ROUND-TRIP GATE (the correctness proof):
#   shipped inflate.py(archive) output  ==  the canonical numpy-fp32 ORACLE forward
#   (tac.boundary_math.lever_b_levelset_generator.levelset_rgb_forward_numpy) of the SAME
#   int8-DEQUANTIZED checkpoint, over the SAME regenerated feats (curvelet bank + self-orient
#   fixed point), bit-for-bit on the uint8 .raw frames (np.array_equal), which subsumes the SegNet
#   argmax the score reads. This proves the shipped decoder faithfully implements the generator
#   (it does NOT measure quant error -- both sides use the SAME dequantized weights; quant error is
#   the separate realized-vs-fp32-EMA quantity the trainer verdict tracks).
# ---------------------------------------------------------------------------
def _torch_R_reference(rgb: np.ndarray, rh: int, rw: int, ch: int, cw: int) -> np.ndarray:
    """Contest R, byte-identical to the shipped inflate.py ``_R``: bicubic up render->camera, round,
    clamp -> uint8. ALWAYS fp32 (matches the shipped ``.float()`` cast)."""
    import torch

    x = torch.from_numpy(np.ascontiguousarray(rgb.reshape(rh, rw, 3))).permute(2, 0, 1)[None].float()
    with torch.inference_mode():
        up = torch.nn.functional.interpolate(x, size=(ch, cw), mode="bicubic", align_corners=False)
        up = torch.clamp(torch.round(up), 0.0, 255.0)
    return up[0].permute(1, 2, 0).contiguous().numpy().astype(np.uint8)


def _blind_coordinate_reference(
    frame: np.ndarray, manifest: dict[str, Any]
) -> np.ndarray:
    """Canonical helper consumer paired with the generated receiver op twin."""

    if manifest.get("blind_coordinate_fill") is None:
        return np.asarray(frame)
    return apply_blind_fill(np.asarray(frame), fill=None)


def numpy_oracle_reference_frames(
    params: dict[str, np.ndarray], code: np.ndarray, manifest: dict[str, Any], n_pairs: int,
    lane_pairs: list[list[Any]] | None = None, pose_carrier: dict[str, Any] | None = None,
    chart_payload: dict[str, np.ndarray] | None = None,
    palette_residual: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Regenerate the FULL uint8 camera frames for the first ``n_pairs`` pairs via the CANONICAL
    numpy-fp32 oracle (``levelset_rgb_forward_numpy`` / ``levelset_band_forward_numpy``) + the
    canonical FREE-table regen (``torch_levelset_inflate`` numpy helpers) + the reference R. This is
    the independent authority the shipped inflate.py must match bit-for-bit. Returns (frames
    [2*n_pairs uint8 arrays, f0,f1 per pair], final_frame_argmax [grid-res int argmax per pair —
    the FINE (ss*rh, ss*rw) grid when literal aa supersample is active, else render-res]).

    The self-orient fixed point mirrors the shipped inflate EXACTLY. When ``manifest`` carries the
    ``lane_render_band`` cfg AND ``lane_pairs`` (the deserialized per-pair ``LaneLine`` lists), the
    CANONICAL render-band (``rasterize_lane_coverage_range_dependent`` + ``composite_band_on_render``
    + ``witness_uncertainty_mask``) is composited over each frame BEFORE R -- so the bit-exact gate
    proves the shipped inflate's inline band == this canonical band. ``params``/``code`` are the
    int8-DEQUANTIZED values read back from the byte-closed blob (both sides render the SAME weights).

    (#497 gap-b) literal post-render supersample A_s — SEALED semantics ``Y = R[A_s G(Phi(X_s))]``:
    the WHOLE feature program (curvelet feats + self-orient dir feats / native orientation fixed
    point, argmax included) and the nonlinear render run on the FINE ``(ss*rh, ss*rw)`` grid; A_s is
    the exact ``ss x ss`` box average applied AFTER the renderer to rgb / lane_rgb / margin, BEFORE
    lane compositing + uncertainty masking (compose-at-base, #220) and BEFORE R. Lane coverage stays
    rasterized at the BASE grid. ``A_1 = identity`` (ss=1 is bit-identical to the point-sampled
    path: same coords, same feats, ``_aa_down`` returns its input unchanged)."""
    rh, rw = int(manifest["render_h"]), int(manifest["render_w"])
    ch, cw = int(manifest["camera_h"]), int(manifest["camera_w"])
    basis_family = normalize_basis_family(manifest.get("basis_family", LEGACY_FOURIER_AB_CONTROL))
    aa_ss = 1
    if basis_family == LITERAL_POLAR_CURVELET:
        _bp_dict = manifest.get("basis_program") or {}
        if _bp_dict.get("aa_mode") == "supersample":
            aa_ss = int(_bp_dict.get("aa_factor", 1))
    gh, gw = rh * aa_ss, rw * aa_ss
    coords = _canon_coords_grid(gh, gw)

    def _aa_down(x_flat: np.ndarray) -> np.ndarray:
        # A_s box average, op-for-op ``aa_sdf_observation_render.box_downsample_np`` (the reshapes
        # are views of the same buffer; the mean over axes (2,4) is the identical ufunc call the
        # shipped inflate's inline ``_aa_down`` performs -> bit-identical between oracle + receiver).
        if aa_ss == 1:
            return x_flat
        from tac.boundary_math.aa_sdf_observation_render import box_downsample_np

        x2 = x_flat if x_flat.ndim == 2 else x_flat[:, None]
        flat = box_downsample_np(x2.reshape(1, gh, gw, x2.shape[-1]), aa_ss).reshape(
            rh * rw, x2.shape[-1]
        )
        return flat if x_flat.ndim == 2 else flat.reshape(rh * rw)

    literal_program: BasisProgramConfig | None = None
    if basis_family == LEGACY_FOURIER_AB_CONTROL:
        B = _canon_curvelet_B(
            manifest["bank_n_scales"], manifest["bank_n_orient0"], manifest["bank_f0"],
            manifest["bank_base"], manifest["bank_n_iso"], manifest["max_bank_freq"],
        )
        curv = _canon_curvelet_feats(coords, B)
    elif basis_family == "windowed_curvelet":
        curv = windowed_curvelet_feats(
            coords, WindowedCurveletConfig(**manifest["windowed_curvelet_config"])
        )
    elif basis_family == COMPACT_SHEARLET:
        curv = compact_shearlet_feats(
            coords, CompactShearletConfig(**manifest["compact_shearlet_config"])
        )
    elif basis_family == LITERAL_POLAR_CURVELET:
        literal_program = BasisProgramConfig.from_dict(manifest["basis_program"])
        if literal_program.canonical_sha256() != manifest.get("basis_program_sha256"):
            raise ValueError("literal basis-program manifest SHA-256 mismatch")
        curv = literal_curvelet_feats_numpy(coords)
    else:
        raise ValueError(f"unknown basis_family {basis_family!r} in byte-closed manifest")
    # ── (#497 gap-a) COUNTED-CHART-PAYLOAD ground chart ────────────────────────────────────────
    # The oracle rebuilds the chart EXACTLY as the trainer did: dequantize the shipped q/scales
    # (op-for-op xi_pose_coder.dequantize_xi) -> GroundFrameChart.build_from_xi (fp64) -> per-pair
    # charted feats via the SEALED charted_grid_bilinear_v1 evaluator (identity ref pair = exact
    # uncharted program). The shipped inflate inlines the SAME builder (parity-pinned by test) and
    # the SAME evaluator (embedded module source) -> trainer == oracle == receiver by construction.
    _chart = None
    _chart_fine_cache = None
    if literal_program is not None and literal_program.chart_enabled:
        if chart_payload is None:
            raise ValueError(
                "literal chart_enabled oracle needs the counted chart payload (q/scales) -- "
                "the chart may only derive from counted receiver state (NO-FAKE fail-closed).")
        if aa_ss != 1:
            raise ValueError("chart x supersample is trainer-refused; no such checkpoint exists")
        from tac.boundary_math.ground_frame_chart import ChartCalibration, GroundFrameChart
        from tac.boundary_math.xi_pose_coder import dequantize_xi

        _q = np.asarray(chart_payload["q"], np.int16)
        if _q.shape[0] < n_pairs:
            raise ValueError(
                f"chart payload has {_q.shape[0]} pairs < requested n_pairs {n_pairs}")
        _chart = GroundFrameChart.build_from_xi(
            dequantize_xi(_q, np.asarray(chart_payload["scales"], np.float32)),
            ref_pair=int(literal_program.chart_ref_pair),
            calib=ChartCalibration(
                s_t=float(literal_program.chart_s_t),
                s_r=float(literal_program.chart_s_r),
                pitch=float(literal_program.chart_pitch),
            ),
            grid_hw=(rh, rw),
            regime=str(literal_program.chart_regime),
        )
        _chart_fine_cache = charted_fine_feats_cache_numpy(literal_program, rh, rw)
    lr = manifest.get("lane_render_band")
    band_on = bool(lr is not None and lane_pairs is not None)
    lane_cfg = render_config_from_header({**lr, "pairs": []}) if band_on else None
    fwd_kw = {
        "n_hidden": int(manifest["n_hidden"]), "hidden_dim": int(manifest["hidden_dim"]),
        "n_classes": int(manifest["n_classes"]), "activation": str(manifest["activation"]),
        "softmax_temp": float(manifest["softmax_temp"]), "wire_w0": float(manifest["wire_w0"]),
        "wire_s0": float(manifest["wire_s0"]), "hosc_beta": float(manifest["hosc_beta"]),
        "hosc_omega": float(manifest["hosc_omega"]), "chroma": bool(manifest["chroma"]),
    }
    frames: list[np.ndarray] = []
    argmaxes: list[np.ndarray] = []
    for pi in range(n_pairs):
        # (#497 gap-a) per-pair base features: charted (sealed bilinear program; identity ref pair
        # = exact uncharted grid) when the chart is active, else the shared static bank.
        if _chart is not None:
            base = charted_pair_feats_numpy(
                literal_program, rh, rw, _chart.H_chart_norm[pi], _chart_fine_cache
            )
        else:
            base = curv
        if literal_program is not None and literal_program.native_orientation_enabled:
            scale_ids, angles = orientation_metadata_from_atom_specs(
                LITERAL_CURVELET_ATOM_SPECS
            )

            def _decode_native(features: np.ndarray) -> np.ndarray:
                _rgb, phi = levelset_rgb_forward_numpy(
                    params, features, code[2 * pi + 1], **fwd_kw
                )
                # (#497 gap-b) argmax at the feature GRID dims (fine when aa supersample is on);
                # unreachable while the native+supersample byte-close gate refuses, kept
                # grid-correct so the sealed semantics is one formula for every path.
                return phi.argmax(-1).reshape(gh, gw).astype(np.int64)

            # (#497 gap-a) chart x native: normal COVECTORS transform through J^-T (SPEC
            # tangent-vector/normal-covector covariance) -- op-for-op the trainer's
            # recompute_native_orient chart branch.
            _normal_transform = None
            if _chart is not None:
                _jac = projective_jacobian_numpy(coords, _chart.H_chart_norm[pi])

                def _normal_transform(normals: np.ndarray) -> np.ndarray:
                    return transform_normal_covector_numpy(normals, _jac)

            feats, _gates, _receipt = native_orientation_fixed_point_numpy(
                base,
                _decode_native,
                scale_ids,
                angles,
                kappa=literal_program.native_orientation_kappa,
                iteration_cap=literal_program.fixed_point_iteration_cap,
                normal_transform=_normal_transform,
            )
        elif bool(manifest["self_orient"]):
            ndf = int(manifest["n_dir_freqs"])
            dirf = np.zeros((base.shape[0], 4 * ndf), np.float32)
            prev_am = None
            for _ in range(int(manifest["so_iters"])):
                feats = np.concatenate([base, dirf], axis=-1)
                _rgb, phi = levelset_rgb_forward_numpy(params, feats, code[2 * pi + 1], **fwd_kw)
                am = phi.argmax(-1).reshape(gh, gw).astype(np.int64)  # #497: grid dims (fine when ss>1)
                if prev_am is not None and np.array_equal(am, prev_am):
                    break  # argmax fixed point -> dirf frozen -> remaining iters no-ops (== shipped)
                dirf = _canon_dir_feats(
                    coords, am, ndf, float(manifest["so_freq_along"]),
                    float(manifest["so_freq_across"]), float(manifest["so_tau"]),
                )
                prev_am = am
            feats = np.concatenate([base, dirf], axis=-1)
        else:
            feats = base
        cov = None
        if band_on and pi < len(lane_pairs):
            cov = rasterize_lane_coverage_range_dependent(
                lane_pairs[pi], h=rh, w=rw, softness=lane_cfg.softness,
                dash_gate=lane_cfg.dash_gate, dash_forward_max_m=lane_cfg.dash_forward_max_m,
                v_h=lane_cfg.v_h, cx=lane_cfg.cx,
            ).reshape(-1)
        for fk in range(2):
            # #205 pose carrier: frame0 (fk==0) is the carrier (seg-free), NOT the plain INR frame0.
            # frame1 (fk==1) stays the witness render (it drives d_seg + the argmax). argmax is taken
            # at fk==1 -> the pose carrier is argmax-free. Mode-aware (bit-mirrors the shipped inflate):
            #  * warp_real_luma: warp(stored keyframe, H); store_nothing: warp(witness's OWN f0 render, H).
            if fk == 0 and pose_carrier is not None:
                if pose_carrier_mode(pose_carrier) == "store_nothing":
                    rgb0, _phi0 = levelset_rgb_forward_numpy(params, feats, code[2 * pi + 0], **fwd_kw)
                    # #497: A_s (fine render -> base) BEFORE R; identity at ss=1.
                    src0 = _torch_R_reference(_aa_down(rgb0), rh, rw, ch, cw).astype(np.float64)
                    frame = pose_carrier_frame0_from_source(pose_carrier, pi, src0)
                else:
                    frame = pose_carrier_frame0(pose_carrier, pi)
                frames.append(_blind_coordinate_reference(frame, manifest))
                continue
            if cov is not None:
                rgb, phi, lane_rgb, margin = levelset_band_forward_numpy(
                    params, feats, code[2 * pi + fk], lane_cls=lane_cfg.lane_cls, **fwd_kw)
                if fk == 1 and palette_residual is not None:
                    rgb = apply_palette_residual(
                        rgb, phi, palette_residual, pair_index=pi
                    )
                # #497 sealed order: A_s footprint-integrates rgb/lane_rgb/margin to the BASE grid
                # FIRST, then the uncertainty mask + composite run at base (compose-at-base, #220).
                # Coverage ``cov`` is rasterized at base already. Identity at ss=1.
                rgb, lane_rgb, margin = _aa_down(rgb), _aa_down(lane_rgb), _aa_down(margin)
                um = (witness_uncertainty_mask(margin, tau=lane_cfg.u_mask_tau, eps=lane_cfg.u_mask_eps)
                      if lane_cfg.u_mask_enabled else None)
                rgb = composite_band_on_render(rgb, lane_rgb, cov, um, lane_cfg.weight)
            else:
                rgb, phi = levelset_rgb_forward_numpy(params, feats, code[2 * pi + fk], **fwd_kw)
                if fk == 1 and palette_residual is not None:
                    rgb = apply_palette_residual(
                        rgb, phi, palette_residual, pair_index=pi
                    )
                rgb = _aa_down(rgb)  # #497: A_s after the nonlinear renderer; identity at ss=1.
            frame = _torch_R_reference(rgb, rh, rw, ch, cw)
            frames.append(_blind_coordinate_reference(frame, manifest))
            if fk == 1:
                argmaxes.append(phi.argmax(-1).reshape(gh, gw).astype(np.int64))
    return frames, argmaxes


def bit_exact_roundtrip_gate(
    packet_dir: Path, blob: bytes, gate_pairs: int, strict: bool
) -> dict[str, Any]:
    """THE correctness proof. Inflate a ``gate_pairs``-capped 0.bin with the SHIPPED inflate.py,
    read back the uint8 .raw, and assert it is BIT-IDENTICAL (np.array_equal) to the canonical
    numpy-fp32 oracle forward of the SAME dequantized checkpoint. Returns a report dict; raises in
    ``strict`` mode on any mismatch."""
    import brotli

    # dequant the blob EXACTLY as the shipped inflate does (int8 -> fp32 * scale).
    m, base_b, code_b, _pose, lane_b, pcar_b, chart_b, palette_b = _read_blob_bytes_full(blob)
    params = _decode_base_params(m, base_b)
    code = _decode_code(m, code_b)
    n_pairs_total = int(m["n_pairs"])
    gp = max(1, min(int(gate_pairs), n_pairs_total))

    # SHIPPED inflate.py on a gp-capped 0.bin -> .raw frames.
    gate_root = packet_dir / "_bitexact_gate"
    gate_root.mkdir(parents=True, exist_ok=True)
    code_cap = code[: 2 * gp]
    qc, sc = _int8_symmetric(code_cap)
    man = dict(m)
    man["n_pairs"] = gp
    man["code_shape"] = list(code_cap.shape)
    man["code_scale"] = float(sc)
    # #224 Wave E: carry the lane render-band through the gp-capped repack (slice pairs to gp) so the
    # shipped inflate composites the SAME band the oracle does over the SAME gp pairs.
    lane_pairs_cap = None
    lane_b_cap = None
    if lane_b is not None and m.get("lane_render_band") is not None:
        all_pairs, lane_hdr = deserialize_lane_band_any(brotli.decompress(lane_b))  # Wave-F dispatch
        lane_pairs_cap = all_pairs[:gp]
        lane_cfg_cap = render_config_from_header({**m["lane_render_band"], "pairs": []})
        lane_b_cap = brotli.compress(
            serialize_lane_band_any(lane_pairs_cap, lane_cfg_cap, lane_hdr), quality=11)
    # #205: carry the pose carrier through the gp-capped repack (slice to gp) so the shipped inflate
    # warps the SAME frame0's the oracle does over the SAME gp pairs (bit-exact frame0 too).
    pcar_cap = None
    pose_carrier_oracle = None
    if pcar_b is not None and m.get("pose_carrier") is not None:
        pcar_cap = _cap_pose_carrier(pcar_b, gp)
        pose_carrier_oracle = parse_pose_carrier(pcar_cap)
        man["pose_carrier"] = {**m["pose_carrier"], "n_pairs": gp}
    # #497 gap-a: carry the chart payload through the gp-capped repack (slice rows; H for kept
    # pairs is EXACT because the incremental composition only reads poses[ref+1..t]).
    chart_cap = None
    chart_oracle = None
    if chart_b is not None and m.get("chart_payload") is not None:
        _cp = _parse_chart_payload(chart_b, int(m["chart_payload"]["n_pairs"]))
        _ref = int((m.get("basis_program") or {}).get("chart_ref_pair", 0))
        if _ref >= gp:
            raise ValueError(
                f"bit-exact gate: chart ref_pair {_ref} >= gate_pairs {gp} -- capped decode would "
                "lose the identity reference pair (raise gate_pairs or lower ref_pair).")
        chart_cap = _chart_payload_bytes(_cp["q"][:gp], _cp["scales"])
        chart_oracle = {"q": _cp["q"][:gp], "scales": _cp["scales"]}
        man["chart_payload"] = {**m["chart_payload"], "n_pairs": gp}
    palette_cap = None
    palette_oracle = None
    if palette_b is not None and m.get("palette_residual") is not None:
        palette_cap = cap_palette_residual(palette_b, gp)
        palette_oracle = decode_palette_residual(palette_cap).residuals
        man["palette_residual"] = decode_palette_residual(palette_cap).manifest
    mj = json.dumps(man, separators=(",", ":")).encode()
    capped_bin = gate_root / "gate.bin"
    capped_bin.write_bytes(_io_pack(
        mj, base_b, _encode_code_brotli(qc, man), None, lane_b_cap, pcar_cap, chart_cap,
        palette_cap))
    gate_raw = gate_root / "gate.raw"
    proc = subprocess.run(
        [sys.executable, str(packet_dir / "inflate.py"), str(capped_bin), str(gate_raw)],
        capture_output=True, text=True, cwd=str(packet_dir),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"bit-exact gate: shipped inflate.py FAILED rc={proc.returncode}\n{proc.stderr}")
    ch, cw = int(m["camera_h"]), int(m["camera_w"])
    fb = ch * cw * 3
    shipped: list[np.ndarray] = []
    with open(gate_raw, "rb") as f:
        for _ in range(2 * gp):
            shipped.append(np.frombuffer(f.read(fb), dtype=np.uint8).reshape(ch, cw, 3))

    # canonical numpy-fp32 ORACLE reference (dequant capped code so both sides see identical values).
    ref_params: dict[str, np.ndarray] = {}
    for name in m["base_param_order"]:  # re-dequant from the SAME capped blob for byte-identical inputs
        ref_params[name] = params[name]
    ref_code = (qc.astype(np.float32) * sc).reshape(code_cap.shape)
    ref_frames, ref_argmax = numpy_oracle_reference_frames(
        ref_params, ref_code, man, gp, lane_pairs_cap, pose_carrier=pose_carrier_oracle,
        chart_payload=chart_oracle, palette_residual=palette_oracle)

    max_abs = 0
    all_equal = True
    n_diff_frames = 0
    for a, b in zip(shipped, ref_frames, strict=True):
        if not np.array_equal(a, b):
            all_equal = False
            n_diff_frames += 1
            max_abs = max(max_abs, int(np.abs(a.astype(np.int32) - b.astype(np.int32)).max()))
    import shutil
    shutil.rmtree(gate_root, ignore_errors=True)  # disk-hygiene (rebuildable from archive.zip)

    result = {
        "bit_exact": bool(all_equal),
        "gate_pairs": gp,
        "frames_compared": 2 * gp,
        "n_frames_differing": n_diff_frames,
        "max_abs_uint8_diff": int(max_abs),
        "reference": "tac.boundary_math.lever_b_levelset_generator.levelset_rgb_forward_numpy "
                     "(numpy-fp32 oracle) + canonical FREE-table regen + reference R",
        "proves": "shipped inflate.py bit-identically implements the canonical generator on the "
                  "int8-dequantized checkpoint (argmax the score reads is byte-determined)",
    }
    verdict = "BIT-EXACT" if all_equal else f"MISMATCH ({n_diff_frames} frames, max_abs={max_abs})"
    print(f"[bit-exact gate] {verdict} over {2 * gp} frames vs numpy-fp32 oracle  {_AUTHORITY}", flush=True)
    if not all_equal and strict:
        raise RuntimeError(
            f"BIT-EXACT GATE FAILED: shipped inflate.py != numpy-fp32 oracle on {n_diff_frames} of "
            f"{2 * gp} frames (max_abs={max_abs}). The shipped decoder diverges from the canonical "
            "generator -> the byte-close is NOT a faithful witness (NO-FAKE).")
    return result


def _read_blob_bytes_full(
    blob: bytes,
) -> tuple[
    dict[str, Any],
    bytes,
    bytes,
    bytes,
    bytes | None,
    bytes | None,
    bytes | None,
    bytes | None,
]:
    """Parse an in-memory LVLS1 blob (same grammar as the shipped inflate._read_blob). Returns
    (manifest, base, code, pose, lane_band|None, pose_carrier|None, chart|None,
    palette_residual|None). Trailing optional blocks are gated by the MANIFEST flags
    (lane_render_band / pose_carrier / chart_payload / palette_residual), NOT a bare off<len -- so a
    lone pose_carrier block is not misread as lane. Default-off -> all None."""
    assert blob[: len(_MAGIC)] == _MAGIC, "bad level-set magic"

    def _take(o: int, n: int) -> tuple[bytes, int]:  # #402 fail-closed exact slice (short => raise)
        end = o + n
        if n < 0 or end > len(blob):
            raise ValueError(f"LVLS1 truncated: section needs {n} B at off={o} but blob is {len(blob)} B")
        return blob[o:end], end

    off = len(_MAGIC)
    out: list[bytes] = []
    for _ in range(4):
        (n,) = struct.unpack_from("<I", blob, off)
        off += 4
        blk, off = _take(off, n)
        out.append(blk)
    manifest = json.loads(out[0].decode("utf-8"))
    lane_band: bytes | None = None
    pose_carrier: bytes | None = None
    if manifest.get("lane_render_band") is not None:
        (n,) = struct.unpack_from("<I", blob, off); off += 4
        lane_band, off = _take(off, n)
    if manifest.get("pose_carrier") is not None:
        (n,) = struct.unpack_from("<I", blob, off); off += 4
        pose_carrier, off = _take(off, n)
    chart: bytes | None = None  # #497 gap-a: 7th optional block, gated by manifest["chart_payload"]
    if manifest.get("chart_payload") is not None:
        (n,) = struct.unpack_from("<I", blob, off); off += 4
        chart, off = _take(off, n)
    palette_residual: bytes | None = None
    if manifest.get("palette_residual") is not None:
        if off + 4 > len(blob):
            raise ValueError("palette_residual manifest is present without EKPR1 section")
        (n,) = struct.unpack_from("<I", blob, off)
        off += 4
        palette_residual, off = _take(off, n)
        validate_palette_residual_binding(
            manifest["palette_residual"],
            palette_residual,
            expected_n_pairs=int(manifest["n_pairs"]),
            expected_n_classes=int(manifest["n_classes"]),
        )
    # #402 EXACT CONSUMPTION (mirrors the shipped inflate._read_blob): trailing bytes fail closed.
    if off != len(blob):
        raise ValueError(f"LVLS1 blob has {len(blob) - off} unconsumed trailing byte(s) "
                         f"(off={off} len={len(blob)}) -- refusing a non-exact stream (NO-FAKE).")
    return manifest, out[1], out[2], out[3], lane_band, pose_carrier, chart, palette_residual


def _read_blob_bytes(
    blob: bytes,
) -> tuple[dict[str, Any], bytes, bytes, bytes, bytes | None, bytes | None, bytes | None]:
    """Compatibility view of :func:`_read_blob_bytes_full`.

    EKPR1 is still parsed, bound, and exactly consumed; legacy callers that do
    not render it retain the historical seven-element tuple.
    """

    manifest, base, code, pose, lane, pcar, chart, _palette = _read_blob_bytes_full(blob)
    return manifest, base, code, pose, lane, pcar, chart


# ---------------------------------------------------------------------------
# REAL upstream/evaluate.py wrapper (CPU only; NEVER MPS) on the EXACT archive bytes.
#   archive.zip (0.bin) + inflated/0.raw  ->  upstream/evaluate.py --device cpu  ->  real
#   d_seg + d_pose + rate + S, cross-checked against tac.contest_score.compute_contest_score.
# This is the exact-eval path that turns the advisory realized-parity into a REAL evaluate.py row.
# ---------------------------------------------------------------------------
# CONSOLIDATED 2026-08-04 (ddm_sub1). The report-pattern table and the parse
# lived here as ONE OF >=6 live private twins of the same authority-path step
# (levelset, contest_eval, contest_auth_eval, mask_rate_sweep, proxy_eval, ...).
# Six regex tables for one report format means six ways to silently disagree
# about what a score IS. The patterns now live once, in
# tac.submission_chain._EVAL_REPORT_PATTERNS; this is a thin adapter that
# preserves THIS tool's exception contract (ValueError), which the canonical
# module raises as SubmissionChainError. Pinned by
# tools/tests/test_levelset_evaluate_wrapper_consolidation.py, written
# characterization-first against the pre-consolidation behaviour.
from tac.submission_chain import (  # noqa: E402
    parse_evaluate_report as _canonical_parse_evaluate_report,
)


def _parse_evaluate_report(text: str) -> dict[str, Any]:
    """Parse upstream/evaluate.py's report block (printed lines 93-101) into a structured dict.
    NO-FAKE: a missing required field raises (never fabricate a score).

    Delegates to the canonical parser; re-raises as ``ValueError`` so this tool's
    long-standing exception contract is unchanged by the consolidation.
    """
    from tac.submission_chain import SubmissionChainError

    try:
        return _canonical_parse_evaluate_report(text)
    except SubmissionChainError as exc:
        raise ValueError(str(exc)) from exc


def _require_full_600_samples(n_samples: Any, report_path: Path) -> None:
    """n600 or it is NOT evidence (CLAUDE.md): an exact row with a partial sample count must fail
    CLOSED, never land silently. A report format that omits the "Evaluation results over N samples"
    line parses ``n_samples=None`` and proceeds (documented; None is absence-of-field, not a
    partial-sample claim)."""
    if n_samples is not None and int(n_samples) != 600:
        raise RuntimeError(
            f"run_upstream_evaluate: evaluate.py report says n_samples={int(n_samples)} != 600 -- "
            "refusing to record a partial-sample count as an exact row (NO-FAKE / n600-or-not-"
            f"evidence). Report: {report_path}")


def run_upstream_evaluate(
    packet_dir: Path, *, device: str, uncompressed_dir: Path, video_names_file: Path,
    archive_bytes: int, timeout: int,
) -> dict[str, Any]:
    """Run the REAL contest scorer (``upstream/evaluate.py --device <cpu|cuda>``) on the exact
    packet bytes (``packet_dir/archive.zip`` + ``packet_dir/inflated/0.raw``, both already produced
    by run_inflate) and return the real d_seg/d_pose/rate + evaluate.py's own Final score, plus our
    recomputed S via ``compute_contest_score`` (cross-check). CPU by default; MPS is REFUSED."""
    if device == "mps":
        raise ValueError("MPS is NEVER a score authority (CLAUDE.md). Use --eval-device cpu (or cuda).")
    from tac.contest_score import compute_contest_score

    # Axis + authority from the ACTUAL device FIRST (CPU/CUDA = separate evidence spaces;
    # the host-platform-only _AUTHORITY mislabeled real CUDA rows as [contest-CPU]).
    axis, authority = _axis_and_authority(device)

    submission_dir = packet_dir  # has archive.zip AND inflated/0.raw (== upstream/evaluate.py layout)
    inflated = submission_dir / "inflated"
    if not (submission_dir / "archive.zip").exists():
        raise FileNotFoundError(f"run_upstream_evaluate: missing {submission_dir/'archive.zip'} (NO-FAKE).")
    raws = list(inflated.glob("*.raw"))
    if not raws:
        raise FileNotFoundError(
            f"run_upstream_evaluate: no inflated .raw in {inflated} -- the full inflate must run "
            "FIRST (do NOT cap --max-pairs for the exact row; evaluate.py needs all 600 pairs).")
    uncompressed_dir = Path(uncompressed_dir).resolve()
    video_names_file = Path(video_names_file).resolve()
    if not uncompressed_dir.exists():
        raise FileNotFoundError(f"--uncompressed-dir missing: {uncompressed_dir} (needs the GT 0.mkv).")
    if not video_names_file.exists():
        raise FileNotFoundError(f"--video-names-file missing: {video_names_file}.")

    report_path = submission_dir / "report.txt"
    evaluate_py = _REPO / "upstream" / "evaluate.py"
    cmd = [
        sys.executable, str(evaluate_py),
        "--submission-dir", str(submission_dir.resolve()),
        "--uncompressed-dir", str(uncompressed_dir),
        "--video-names-file", str(video_names_file),
        "--device", device,
        "--report", str(report_path.resolve()),
        "--batch-size", "8",
    ]
    env = dict(os.environ)
    up = str((_REPO / "upstream").resolve())
    env["PYTHONPATH"] = up + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""  # force CPU (defensive: never let it pick an MPS/CUDA path)
    print(f"[exact-eval] running upstream/evaluate.py --device {device} on {submission_dir.name} "
          f"(this is the REAL contest scorer; CPU 600-pair ~1-2h)  {authority}", flush=True)
    proc = subprocess.run(cmd, cwd=up, env=env, capture_output=True, text=True, timeout=timeout)
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"upstream/evaluate.py FAILED rc={proc.returncode}\n{combined[-4000:]}")

    src_text = report_path.read_text() if report_path.exists() else combined
    parsed = _parse_evaluate_report(src_text)
    n_samples = parsed.get("n_samples")
    _require_full_600_samples(n_samples, report_path)
    d_seg = float(parsed["d_seg"])
    d_pose = float(parsed["d_pose"])
    recomputed_S = compute_contest_score(d_seg, d_pose, archive_bytes)
    return {
        "ran": True,
        "device": device,
        "evaluate_py_final_score": float(parsed["final_score"]),
        "d_seg": d_seg,
        "d_pose": d_pose,
        "rate_from_evaluate": float(parsed["rate"]),
        "n_samples": n_samples,
        "recomputed_S_compute_contest_score": recomputed_S,
        "recomputed_vs_evaluate_delta": abs(recomputed_S - float(parsed["final_score"])),
        "archive_bytes_scored": int(archive_bytes),
        "report_path": str(report_path),
        "score_axis": axis,
        "authority": authority,
        "promotion_claim": False,
    }


# ---------------------------------------------------------------------------
# #224 Wave E: build the decode-consistent analytic-lane render-band section.
# ---------------------------------------------------------------------------
def _lane_manifest_from_cfg(cfg: LaneBandRenderConfig, fit_stats: dict[str, Any]) -> dict[str, Any]:
    """The manifest ``lane_render_band`` cfg (scalars ``render_config_from_header`` reads + geometry +
    fit observability). The lane MANIFOLD COORDS ride the counted 5th block, NOT this."""
    from tac.boundary_math.lane_sdf_component import _CAM_H, _FX, _FY, _SEG_H, _SEG_W

    return {
        "softness": float(cfg.softness), "dash_gate": bool(cfg.dash_gate),
        "dash_forward_max_m": float(cfg.dash_forward_max_m), "v_h": float(cfg.v_h),
        "cx": (None if cfg.cx is None else float(cfg.cx)), "weight": float(cfg.weight),
        "lane_cls": int(cfg.lane_cls), "lane_rgb_mode": str(cfg.lane_rgb_mode),
        "u_mask": ({"source": "witness_margin", "tau": float(cfg.u_mask_tau), "eps": float(cfg.u_mask_eps)}
                   if cfg.u_mask_enabled else None),
        "geom": {"cam_h": _CAM_H, "fx": _FX, "fy": _FY, "seg_h": _SEG_H, "seg_w": _SEG_W},
        "fit_stats": fit_stats,
    }


def build_lane_band_section(
    gt_cache: str | None, n_pairs: int, cfg: LaneBandRenderConfig, *, rd: bool = True,
    res: bool = False,
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    """Fit the per-pair lane manifold coords from the GT SegNet argmax cache (compress-time; the
    source is fully available), serialize + brotli them (COUNTED), and build the manifest cfg.
    Returns (brotli_lane_bytes, lane_manifest, report). NO-FAKE: missing GT cache raises.

    ``rd=True`` (Wave-F default) uses the OPTIMAL LBND2 rate-distortion codec
    (quantize->temporal-delta->L4-slots->zigzag, brotli entropy backend). ``rd=False`` uses
    the naive LBND1 float64 serializer (kept for the default-off byte-identical gate + the
    naive-vs-RD rate comparison). ``res=True`` (--lane-band-res, DEFAULT OFF per the
    sealed-config discipline) selects LBND4: the SAME LBND2 quantization grid (dequantized
    statistic bit-identical, asserted at encode) with the ξ delta/context residual entropy
    stage (best-of-three {varint,zlib9,rice}, MEASURED n600 −10,634 B / −25.6% vs LBND2;
    experiments/results/lane_band_res_coder_20260707/). NOTE: the inline _INFLATE_PY does
    NOT yet inline the LBND4 decode — a shipped LBND4 packet FAILS CLOSED at the parity
    gate (unknown magic) until that decode half is inlined; measurement lever first.
    The report carries the MEASURED per-lever byte accounting."""
    import brotli

    if not gt_cache:
        raise ValueError("--lane-render-band requires --gt-cache (the frozen SegNet argmax lstars) to "
                         "fit the lane lines at compress time. NO-FAKE: refusing to fabricate.")
    p = Path(gt_cache)
    if not p.exists():
        raise FileNotFoundError(f"--gt-cache {p} not found (needed to fit the lane band lines).")
    z = np.load(p, allow_pickle=False)
    if "lstars" not in z.files:
        raise ValueError(f"gt cache {p} lacks 'lstars' (the frozen SegNet argmax) -- cannot fit lines.")
    lstars = z["lstars"]
    ncap = min(int(n_pairs), int(len(lstars)))
    lst_list = [np.asarray(lstars[i], np.int64) for i in range(ncap)]
    pairs_lines, fit_stats = build_lane_band_pairs_from_lstars(lst_list, cfg)
    # Wave-F: measured per-lever rate accounting (naive LBND1 vs optimal LBND2 RD).
    rate_report = lane_band_rd_rate_report(pairs_lines, cfg)
    if res and not rd:
        raise ValueError("--lane-band-res and --lane-band-naive are mutually exclusive "
                         "(LBND4 is the RD grid + a different entropy stage; NO-FAKE: refusing "
                         "an ambiguous codec selection).")
    if res:
        raw = serialize_lane_band_res(pairs_lines, cfg)
    else:
        raw = serialize_lane_band_rd(pairs_lines, cfg) if rd else serialize_lane_band(pairs_lines, cfg)
    lane_bytes = brotli.compress(raw, quality=11)
    lane_manifest = _lane_manifest_from_cfg(cfg, fit_stats)
    report = {
        "active": True,
        "codec": ("LBND4_res" if res else ("LBND2_rd" if rd else "LBND1_naive")),
        "source_gt_cache": str(p),
        "n_pairs_fit": ncap,
        "serialized_raw_bytes": len(raw),
        "counted_brotli_bytes": len(lane_bytes),
        "counted_rate_term_contribution": 25.0 * len(lane_bytes) / RATE_DENOM,
        "fit_stats": fit_stats,
        "rate_report": rate_report,  # Wave-F: naive-vs-RD + Shannon floor + PTC1 + induced lat RMS
        "u_mask_enabled": bool(cfg.u_mask_enabled),
        "rule_118": {
            "COUNTED (archive.zip)": ("per-pair LaneLine manifold coords, QUANTIZED (geometric-tolerance "
                                      "steps) + temporal-delta + zigzag -- the video-derived statistic"
                                      if rd else
                                      "per-pair LaneLine manifold coords (float64) -- the video-derived statistic"),
            "FREE (inflate.py)": "quantize/dequantize + rasterize_lane_coverage_range_dependent (the AA-SDF "
                                 "coverage raster) + the composite -- generic deterministic algorithm, 0 archive bytes",
            "no_gt_no_scorer": "no GT mask, no SegNet/PoseNet weights, no per-pixel table ship",
        },
    }
    return lane_bytes, lane_manifest, report


# ---------------------------------------------------------------------------
# #359 PHASE-RESIDUAL CARRIER (store-half of the appearance-phase d_seg reframe).
#   The witness converges to the temporal-majority oracle floor (d_seg ~= 0.005318 = the GT stride-2
#   spike rate); sub-0.15 needs 0.0008-0.0012 = 4.5-7x BELOW it, pierceable ONLY by APPEARANCE-PHASE
#   faithfulness. The train-side (T1/#360) descends the xi-COHERENT phase; the PHASE CARRIER STORES
#   the per-pair RESIDUAL r = t_wit_actual - t_wit_xi_predicted the witness cannot predict
#   (xi-transport amortized), entropy-coded. rule-118: COUNTED = the residual (video-derived); FREE =
#   the xi-transport warp + tie-coordinate extraction + the DECODER-DERIVABLE straddle mask.
#
#   SELECTABLE CARRIER MODE (build-time codec parallel to the pose codec, #140 disposition): the
#   through-R n600 d_seg this recovers is OWED-gated (needs a scorer forward the live #205 run
#   forbids concurrently), so this mode BUILDS + MEASURES the section (real counted bytes -> a
#   per-lever ΔS/byte candidate ROW for the #406 apply-pass) WITHOUT yet mutating the shipped
#   _io_pack/inflate grammar -- the byte-identical-when-off guarantee is absolute (no grammar change
#   at all) and the inflate-consumption wire lands with the through-R A/B that proves the d_seg
#   (the same NO-FAKE staging every carrier goes through before its first exact row).
# ---------------------------------------------------------------------------
def build_phase_carrier_section(
    gt_cache: str | None, n_pairs: int, cfg: dict[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    """Build the phase-residual carrier section from the GT cache (lstars/margins/gt_poses).

    Returns ``(section_bytes, report)``. NO-FAKE: a missing cache/keys raises. ``cfg`` keys:
    ``q_step`` (default 1/64), ``classes`` (GROUND {0,1,2}), ``residual_scheme`` ("auto"),
    ``annulus_band`` (1.0), ``gap_xi`` ("interp"), ``pitch`` (0.0)."""
    from tac.boundary_math.phase_residual_carrier import PhaseCarrierConfig, phase_carrier_report

    if not gt_cache:
        raise ValueError("--phase-carrier requires --gt-cache (lstars/margins/gt_poses). NO-FAKE: "
                         "refusing to fabricate the phase payload.")
    cp = Path(gt_cache)
    if not cp.exists():
        raise FileNotFoundError(f"--gt-cache {cp} not found (phase-carrier needs lstars/margins/gt_poses).")
    z = np.load(cp, allow_pickle=False)
    for req in ("lstars", "margins", "gt_poses"):
        if req not in z.files:
            raise ValueError(f"gt cache {cp} lacks {req!r} (phase-carrier needs cached argmax+margins+poses).")
    P = int(min(int(n_pairs), int(z["lstars"].shape[0])))
    pcfg = PhaseCarrierConfig(
        q_step=float(cfg.get("q_step", 1.0 / 64.0)),
        residual_scheme=str(cfg.get("residual_scheme", "auto")),
        classes=tuple(int(c) for c in cfg.get("classes", (0, 1, 2))),
        annulus_band=float(cfg.get("annulus_band", 1.0)),
        gap_xi=str(cfg.get("gap_xi", "interp")),
        pitch=float(cfg.get("pitch", 0.0)),
    )
    section, rep = phase_carrier_report(
        np.asarray(z["lstars"])[:P], np.asarray(z["margins"])[:P], np.asarray(z["gt_poses"])[:P], pcfg
    )
    rate_term_contribution = _cscore.rate_term(len(section))
    report = {
        "active": True,
        "n_frames": rep.n_frames,
        "classes": list(rep.classes),
        "q_step": rep.q_step,
        "residual_scheme": rep.residual_scheme,
        "total_residual_count": rep.total_residual_count,
        "per_frame_class_counts": rep.per_frame_class_counts,
        "section_bytes": rep.section_bytes,
        "xi_amortized_residual_bytes": rep.xi_amortized_residual_bytes,
        "raw_tie_residual_bytes": rep.raw_tie_residual_bytes,
        "amortization_ratio": rep.amortization_ratio,
        "mean_abs_residual_q": rep.mean_abs_residual_q,
        "max_abs_residual_q": rep.max_abs_residual_q,
        "tie_recon_rmse_px": rep.tie_recon_rmse_px,
        "reconstruction_bit_identical": rep.reconstruction_bit_identical,
        "counted_rate_term_contribution": rate_term_contribution,
        "recovered_d_seg": None,  # OWED-gated: needs a through-R n600 scorer forward (memory-blocked by live #205)
        "recovered_d_seg_status": "OWED_through_R_n600_AB (intrinsic bytes MEASURED; d_seg NOT claimed, NO-FAKE)",
        "source_gt_cache": str(cp),
        "rule_118_boundary": {
            "COUNTED (archive.zip)": "the quantized per-pair phase RESIDUAL (video-derived; the part "
                                     "xi-transport cannot predict)",
            "FREE (inflate.py)": "the xi-transport warp A_xi + tie-coordinate extraction (generic algorithm) "
                                 "+ the DECODER-DERIVABLE straddle mask (the witness's own rendered partition)",
            "no_gt_no_scorer_shipped": "no GT mask, no SegNet/PoseNet weights, no per-pixel table ship",
        },
    }
    return section, report


def build_dash_phase_carrier_section(
    gt_cache: str | None, n_pairs: int, cfg: dict[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    """Build the #425 DASH-phase carrier section (curve-domain per-dash δ(s) codec).

    The store-side complement of the raster #359 carrier: lane dash TRACKS (ξ-advected,
    world-frame dormant-pool rebirth) + per-dash curve-relative (δs, δn) residuals coded
    with the prior-derived canonical Huffman code (measured jitter prior 40/72/80/20).
    Returns ``(section_bytes, report)``. NO-FAKE: a missing cache/keys raises; the encoder
    runs the full decoder and refuses on any mismatch. ``cfg`` keys mirror
    ``tac.boundary_math.dash_phase_carrier.DashPhaseConfig``."""
    from tac.boundary_math.dash_phase_carrier import DashPhaseConfig, dash_phase_carrier_report

    if not gt_cache:
        raise ValueError("--dash-phase-carrier requires --gt-cache (lstars/gt_poses). NO-FAKE: "
                         "refusing to fabricate the dash payload.")
    cp = Path(gt_cache)
    if not cp.exists():
        raise FileNotFoundError(f"--gt-cache {cp} not found (dash-phase-carrier needs lstars/gt_poses).")
    z = np.load(cp, allow_pickle=False)
    for req in ("lstars", "gt_poses"):
        if req not in z.files:
            raise ValueError(f"gt cache {cp} lacks {req!r} (dash-phase-carrier needs cached argmax+poses).")
    P = int(min(int(n_pairs), int(z["lstars"].shape[0])))
    dcfg = DashPhaseConfig(
        lane_class=int(cfg.get("lane_class", 1)),
        min_area=int(cfg.get("min_area", 3)),
        border_px=int(cfg.get("border_px", 6)),
        match_radius_px=float(cfg.get("match_radius_px", 6.0)),
        q_px=float(cfg.get("q_px", 1.0)),
        dormant_max_frames=int(cfg.get("dormant_max_frames", 30)),
        gap_xi=str(cfg.get("gap_xi", "interp")),
        # MEASURED advection-memo pose->xi calibration (s_t=-0.00322, s_r=0, pitch=-0.01);
        # the raw s_t=1 scale mis-advects (n20 smoke: coverage 52% vs 82%).
        s_t=float(cfg.get("s_t", -0.00322)),
        s_r=float(cfg.get("s_r", 0.0)),
        pitch=float(cfg.get("pitch", -0.01)),
        include_xi=bool(cfg.get("include_xi", True)),
    )
    section, rep = dash_phase_carrier_report(np.asarray(z["lstars"])[:P], np.asarray(z["gt_poses"])[:P], dcfg)
    rate_term_contribution = _cscore.rate_term(len(section))
    report = {
        "active": True,
        "n_frames": rep.n_frames,
        "section_bytes": rep.section_bytes,
        "section_bytes_excl_xi": rep.section_bytes_excl_xi,
        "xi_bytes_in_section": rep.xi_bytes,
        "n_tracks_total": rep.n_tracks_total,
        "n_matched": rep.n_matched,
        "n_births": rep.n_births,
        "n_rebirths": rep.n_rebirths,
        "n_deaths": rep.n_deaths,
        "blink_back_fraction": rep.blink_back_fraction,
        "esc_rate": rep.esc_rate,
        "expected_bits_per_dash_prior": rep.expected_bits_per_dash_prior,
        "measured_bits_per_matched_dash": rep.measured_bits_per_matched_dash,
        "symbol_histogram": rep.symbol_histogram,
        "mean_abs_delta_px": rep.mean_abs_delta_px,
        "prior_code_delta_bytes": rep.prior_code_delta_bytes,
        "zlib9_delta_stream_bytes": rep.zlib9_delta_stream_bytes,
        "reconstruction_bit_identical": rep.reconstruction_bit_identical,
        "counted_rate_term_contribution": rate_term_contribution,
        "recovered_d_seg": None,
        "recovered_d_seg_status": ("label_space_lane_layer_via_tools/measure_dash_phase_carrier_n600.py; "
                                   "through-R d_seg OWED (NO-FAKE — bytes measured, d_seg not claimed)"),
        "source_gt_cache": str(cp),
        "rule_118_boundary": {
            "COUNTED (archive.zip)": "dash anchors + alive/event bits + (δs, δn) residual symbols + the "
                                     "header code-lengths table (+ fp16 ξ unless composed with the L68 dxi)",
            "FREE (inflate.py)": "the ξ point-advection homography + the canonical Huffman decoder (generic "
                                 "given header lengths) + the downstream dash rasterizer",
            "no_gt_no_scorer_shipped": "no GT mask, no SegNet/PoseNet weights, no per-pixel table ship",
        },
    }
    return section, report


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run(
    ckpt_dir: Path,
    *,
    npz_name: str | None,
    max_pairs: int | None,
    fold_pose_sidecar: bool,
    pose_sidecar_path: Path | None,
    gt_cache: str | None,
    keep_packet: bool,
    packet_dir: Path | None,
    skip_parity: bool,
    so_overrides: dict[str, Any],
    lane_render_band: bool = False,
    lane_band_cfg: LaneBandRenderConfig | None = None,
    lane_rd: bool = True,  # Wave-F: LBND2 optimal RD codec (default); False -> naive LBND1
    lane_res: bool = False,  # LBND4: RD grid + ξ residual entropy stage (default OFF, measurement lever)
    pose_carrier: bool = False,  # #205 warp-real-luma frame0 pose carrier
    pose_carrier_cfg: dict[str, Any] | None = None,
    pose_carrier_xi_override: np.ndarray | None = None,  # #238: ship the TRAINED ξ_eff (xi_stored+dxi)
    phase_carrier: bool = False,  # #359 phase-residual carrier (store-half of the flicker reframe)
    phase_carrier_cfg: dict[str, Any] | None = None,
    dash_phase_carrier: bool = False,  # #425 curve-domain per-dash δ(s) phase codec (STORE leg)
    dash_phase_carrier_cfg: dict[str, Any] | None = None,
    cross_tensor_codec: bool = False,
    blind_coordinate_fill: bool = False,
    blind_coordinate_receipt: Path | None = None,
    palette_residual_path: Path | None = None,
    verify_bit_exact: bool = False,
    bit_exact_pairs: int = 2,
    bit_exact_strict: bool = True,
    run_exact_eval: bool = False,
    eval_device: str = "cpu",
    uncompressed_dir: Path | None = None,
    video_names_file: Path | None = None,
    eval_timeout: int = 18000,
) -> dict[str, Any]:
    blind_coordinate_report: dict[str, Any] = {"active": False}
    if blind_coordinate_fill:
        if blind_coordinate_receipt is None:
            raise ValueError(
                "--blind-coordinate-fill requires --blind-coordinate-receipt with n600 zero-delta custody"
            )
        # Receipt validation precedes checkpoint loading, packet construction, raw generation, and
        # weights-arm selection.  A malformed proof cannot mutate or rank a candidate.
        blind_coordinate_report = validate_blind_coordinate_n600_receipt(
            blind_coordinate_receipt
        )
    params, cfg = _load_levelset_ckpt(ckpt_dir, npz_name)
    basis_deploy_report: dict[str, Any] = {"active": False}
    if cfg["basis"] == LITERAL_POLAR_CURVELET:
        program: BasisProgramConfig = cfg["basis_program"]
        # (#497 gap-a CLOSED for counted_chart_payload) the literal ground chart is now a COUNTED
        # RECEIVER PROGRAM: the quantized startup pose table + scales ship as the 7th blob section;
        # trainer, oracle, and shipped inflate all rebuild the chart via build_from_xi on the SAME
        # dequantized values and evaluate per-pair feats via the SEALED charted_grid_bilinear_v1
        # evaluator (the fast receiver path that replaces the n600-prohibitive direct sparse
        # transform). Proof authority: bit_exact_roundtrip_gate with a NONTRIVIAL chart.
        if program.chart_enabled and program.chart_pose_dependency != "counted_chart_payload":
            raise ValueError(
                "literal ground-chart byte-close supports ONLY chart_pose_dependency="
                "'counted_chart_payload' (the chart's OWN quantized pose-table section). The "
                "'counted_pose_carrier_xi' dependency is structurally unsatisfiable for a "
                "startup-static chart: the carrier's xi_eff = xi_stored + TRAINED dxi does not "
                "exist at ep0. Composition scope, not a curvelet-family verdict"
            )
        if program.chart_enabled and program.aa_mode != "none":
            raise ValueError(
                "literal ground chart + post-render supersample byte-close is fail-closed: the "
                "trainer refuses --ground-frame-chart with --render-aa != none, so no checkpoint "
                "exists to prove train/decode identity. Composition implementation gap"
            )
        # (#497 gap-b CLOSED) literal post-render supersample IS receiver-sealed: the shipped
        # inflate + the numpy oracle both run the whole feature program + nonlinear render on the
        # FINE (ss*rh, ss*rw) grid and apply A_s (exact ss x ss box average) AFTER the renderer,
        # BEFORE lane compositing and BEFORE R (compose-at-base, #220 sealed semantics; A_1 =
        # identity by construction). Proof authority: bit_exact_roundtrip_gate at aa_factor=2.
        if program.native_orientation_enabled and program.aa_mode != "none":
            raise ValueError(
                "literal native orientation + post-render supersample byte-close is fail-closed: "
                "the trainer refuses this combination (fine-grid native gates are not "
                "trainer-sealed yet), so no checkpoint exists to prove train/decode identity. "
                "Composition implementation gap, not a curvelet-family verdict"
            )
        if program.aa_mode != "none" and any(str(k).startswith("tex_trunk.") for k in params):
            # The #395 texture-trunk Gabor bank is regenerated at the BASE (render_h, render_w) grid
            # inside the receiver forward, but supersample renders at the FINE grid -> the bank/soft
            # shapes would mismatch. Narrow honest gap (bank-at-fine-grid not receiver-sealed), not a
            # curvelet-family verdict.
            raise ValueError(
                "literal post-render supersample + texture trunk (tex_trunk.*) byte-close is "
                "fail-closed: the receiver regenerates the trunk bank at the BASE grid while "
                "supersample renders at the FINE grid. Composition implementation gap"
            )
        fold_receipt = None
        if program.taper_enabled:
            folded_weight, fold_receipt = fold_taper_into_in_proj_numpy(
                params["in_proj.weight"], cfg["basis_taper_unfolded"]
            )
            params = dict(params)
            params["in_proj.weight"] = folded_weight
            program = replace(
                program, deploy_fold_receipt_sha256=fold_receipt.receipt_sha256
            )
        cfg["basis_program_deploy"] = program
        basis_deploy_report = {
            "active": True,
            "family": program.family,
            "basis_version": program.basis_version,
            "atom_spec_sha256": program.atom_spec_sha256,
            "train_program_sha256": cfg["basis_program_train_sha256"],
            "deploy_program_sha256": program.canonical_sha256(),
            "taper_fold_receipt": (fold_receipt.to_dict() if fold_receipt is not None else None),
            "native_orientation": program.native_orientation_enabled,
            "chart_enabled": program.chart_enabled,
            "chart_pose_dependency": program.chart_pose_dependency,
            "chart_eval_semantics": program.chart_eval_semantics,
            "chart_fine_factor": program.chart_fine_factor,
            "chart_payload_pairs": (
                int(np.asarray(cfg["chart_pose_q"]).shape[0]) if program.chart_enabled else 0
            ),
            "aa_mode": program.aa_mode,
            "aa_factor": program.aa_factor,
        }
    n_pairs = int(cfg["n_pairs"])
    palette_residual_bytes: bytes | None = None
    palette_residual_report: dict[str, Any] = {"active": False}
    if palette_residual_path is not None:
        section_path = Path(palette_residual_path)
        if not section_path.is_file():
            raise FileNotFoundError(f"--palette-residual-section does not exist: {section_path}")
        palette_residual_bytes = section_path.read_bytes()
        parsed_palette = decode_palette_residual(
            palette_residual_bytes,
            expected_n_pairs=n_pairs,
            expected_n_classes=int(cfg["n_classes"]),
        )
        palette_residual_report = {
            "active": True,
            "codec": EKPR1_CODEC,
            "source_path": str(section_path),
            "section_bytes": len(palette_residual_bytes),
            "sha256": hashlib.sha256(palette_residual_bytes).hexdigest(),
            "shape": list(parsed_palette.residuals.shape),
            "application": EKPR1_APPLICATION,
        }
    so = detect_self_orient(cfg, so_overrides)
    # The exact-eval row needs the FULL packet + ALL 600 pairs inflated on disk.
    if run_exact_eval:
        keep_packet = True
        if max_pairs is not None:
            print(f"[exact-eval] ignoring --max-pairs={max_pairs}: the real evaluate.py row needs ALL "
                  f"{n_pairs} pairs (evaluate.py zips full GT with the inflated frames).", flush=True)
            max_pairs = None

    pose_bytes: bytes | None = None
    pose_note = ("off (level-set witness carries pose in per-(pair,frame) codes/texture; train "
                 "--w-pose>0. A stored sidecar adds COUNTED bytes the scorer never reads -> does "
                 "NOT lower realized d_pose).")
    if fold_pose_sidecar:
        if pose_sidecar_path and Path(pose_sidecar_path).exists():
            pose_bytes = Path(pose_sidecar_path).read_bytes()
            pose_note = (f"folded {len(pose_bytes)} COUNTED B from {pose_sidecar_path}; WARNING: the "
                         "inflate render does NOT read it -> realized d_pose is UNCHANGED by the "
                         "sidecar (dead bytes on a code/texture-pose witness; NO-FAKE honesty).")
        else:
            raise FileNotFoundError(
                "--fold-pose-sidecar requires --pose-sidecar-path <posenet_targets.bin>. NO-FAKE: "
                "refusing to fabricate.")

    lane_band_bytes: bytes | None = None
    lane_manifest: dict[str, Any] | None = None
    lane_report: dict[str, Any] = {"active": False}
    if lane_render_band:
        lane_cfg = lane_band_cfg or LaneBandRenderConfig()
        lane_band_bytes, lane_manifest, lane_report = build_lane_band_section(
            gt_cache, n_pairs, lane_cfg, rd=lane_rd, res=lane_res)
        _rr = lane_report.get("rate_report", {})
        print(f"[lane-band] active ({lane_report['codec']}): {lane_report['n_pairs_fit']} pairs fit, "
              f"COUNTED brotli={lane_report['counted_brotli_bytes']} B "
              f"(rate_term += {lane_report['counted_rate_term_contribution']:.5f}); "
              f"naive-LBND1 brotli={_rr.get('naive_lbnd1_brotli_bytes','?')} B "
              f"(rd/naive={_rr.get('rd_vs_naive_ratio', float('nan')):.3f}); "
              f"u_mask={lane_report['u_mask_enabled']}; quantize+raster = FREE (rule 118)  {_AUTHORITY}",
              flush=True)

    # #205 warp-real-luma pose carrier: build the COUNTED keyframe luma + per-pair homography section.
    pose_carrier_bytes: bytes | None = None
    pose_carrier_manifest: dict[str, Any] | None = None
    pose_carrier_report: dict[str, Any] = {"active": False}
    keyframe_accounting: dict[str, Any] | None = None
    if pose_carrier:
        pc_cfg = pose_carrier_cfg or {"s_t": 0.16, "s_r": 1.0, "pitch": 0.02, "stride": 1,
                                      "downscale": 1, "mode": "warp_real_luma"}
        pc_mode = str(pc_cfg.get("mode", "warp_real_luma"))
        pose_carrier_bytes, pose_carrier_manifest, pose_carrier_report = build_pose_carrier_section(
            gt_cache, n_pairs, pc_cfg, xi_override=pose_carrier_xi_override)
        # REUSE the canonical keyframe_payload_accounting (tools.compose_witness_archive, f7c6abdea) for
        # the rule-118 line item -- fed the MEASURED keyframe blob bytes actually stored (0 for store_nothing).
        from tools.compose_witness_archive import keyframe_payload_accounting  # noqa: PLC0415
        kf_measured = int(pose_carrier_report["keyframe_blob_bytes_total"])
        keyframe_accounting = keyframe_payload_accounting(
            argparse.Namespace(keyframe_payload_path=None, keyframe_payload_bytes=kf_measured,
                               keyframe_count=int(pose_carrier_report["n_keyframes"])))
        _pc_section = int(pose_carrier_report["pose_carrier_section_bytes"])
        if pc_mode == "store_nothing":
            pose_note = (f"STORE-NOTHING (xi-only) carrier ACTIVE: NO stored keyframe (0 B) -- frame0 = "
                         f"warp(the witness's OWN frame0 render, per-pair H). Section={_pc_section} B "
                         f"(xi/H only; ~0 marginal keyframe rate). Track B store-nothing-but-xi (18927a1ae).")
            print(f"[pose-carrier] STORE-NOTHING (xi-only) ACTIVE: keyframe_bytes=0 "
                  f"section={_pc_section} B (H+xi only; witness render warped by xi = FREE frame0)  "
                  f"{_AUTHORITY}", flush=True)
        else:
            pose_note = (f"WARP-REAL-LUMA frame0 carrier ACTIVE: {pose_carrier_report['n_keyframes']} stored "
                         f"keyframe(s) (store_hw={pose_carrier_report['keyframe_store_hw']}, "
                         f"lossless_native={pose_carrier_report['keyframe_lossless_native']}) COUNTED "
                         f"{kf_measured} B; frame0 = warp(keyframe, per-pair H) -> PoseNet reads real warped "
                         f"motion. Section={_pc_section} B.")
            print(f"[pose-carrier] warp-real-luma frame0 ACTIVE: n_kf={pose_carrier_report['n_keyframes']} "
                  f"stride={pose_carrier_report['stride']} store_hw={pose_carrier_report['keyframe_store_hw']} "
                  f"keyframe_bytes={kf_measured} section={_pc_section} B "
                  f"(rate += {keyframe_accounting['keyframe_blob_rate']:.5f})  {_AUTHORITY}", flush=True)

    # #359 PHASE-RESIDUAL CARRIER (store-half of the flicker reframe). SELECTABLE MODE: BUILD + MEASURE
    # the section (real counted bytes -> per-lever ΔS/byte candidate row for the #406 apply-pass) WITHOUT
    # mutating the shipped _io_pack/inflate grammar (byte-identical when off; inflate-consumption OWED with
    # the through-R A/B that proves the d_seg -- NO-FAKE staging). The through-R n600 d_seg is memory-gated
    # by the live #205 run, so d_seg is reported OWED, never claimed.
    phase_carrier_report_out: dict[str, Any] = {"active": False}
    if phase_carrier:
        _phase_section, phase_carrier_report_out = build_phase_carrier_section(
            gt_cache, n_pairs, phase_carrier_cfg or {})
        print(f"[phase-carrier] ACTIVE (GROUND classes {phase_carrier_report_out['classes']}): "
              f"section={phase_carrier_report_out['section_bytes']} B "
              f"({phase_carrier_report_out['total_residual_count']} residuals, "
              f"scheme={phase_carrier_report_out['residual_scheme']}, "
              f"rate_term += {phase_carrier_report_out['counted_rate_term_contribution']:.6f}); "
              f"xi_amort={phase_carrier_report_out['amortization_ratio']:.3f} "
              f"rmse_px={phase_carrier_report_out['tie_recon_rmse_px']:.5f} "
              f"bit_identical={phase_carrier_report_out['reconstruction_bit_identical']}; "
              f"recovered_d_seg={phase_carrier_report_out['recovered_d_seg_status']}  {_AUTHORITY}", flush=True)

    # #425 DASH-PHASE CARRIER (curve-domain per-dash δ(s) codec — the STORE leg of lane-crux-3).
    # Same NO-FAKE staging as #359: SELECTABLE build+measure, byte-identical when off; the label-space
    # lane-layer recovery lives in tools/measure_dash_phase_carrier_n600.py; through-R d_seg OWED.
    dash_phase_carrier_report_out: dict[str, Any] = {"active": False}
    if dash_phase_carrier:
        _dash_section, dash_phase_carrier_report_out = build_dash_phase_carrier_section(
            gt_cache, n_pairs, dash_phase_carrier_cfg or {})
        print(f"[dash-phase-carrier] ACTIVE (#425 curve-domain δ(s)): "
              f"section={dash_phase_carrier_report_out['section_bytes']} B "
              f"(excl-ξ {dash_phase_carrier_report_out['section_bytes_excl_xi']} B, "
              f"tracks={dash_phase_carrier_report_out['n_tracks_total']}, "
              f"matched={dash_phase_carrier_report_out['n_matched']}, "
              f"births={dash_phase_carrier_report_out['n_births']}, "
              f"rebirths={dash_phase_carrier_report_out['n_rebirths']}, "
              f"blink_back={dash_phase_carrier_report_out['blink_back_fraction']:.3f}, "
              f"bits/dash={dash_phase_carrier_report_out['measured_bits_per_matched_dash']:.2f} "
              f"vs prior {dash_phase_carrier_report_out['expected_bits_per_dash_prior']:.2f}, "
              f"rate_term += {dash_phase_carrier_report_out['counted_rate_term_contribution']:.6f}); "
              f"bit_identical={dash_phase_carrier_report_out['reconstruction_bit_identical']}; "
              f"recovered_d_seg={dash_phase_carrier_report_out['recovered_d_seg_status']}  {_AUTHORITY}",
              flush=True)

    blob, breakdown = build_levelset_blob(params, cfg, so, pose_bytes, lane_band_bytes, lane_manifest,
                                          pose_carrier_bytes, pose_carrier_manifest,
                                          cross_tensor_codec=cross_tensor_codec,
                                          blind_coordinate_fill=blind_coordinate_fill,
                                          palette_residual_bytes=palette_residual_bytes)
    if not cross_tensor_codec and not breakdown["accounting_matches_canonical"]:
        print(f"[WARN] byte-close accounting (base={breakdown['base_int8_brotli_bytes']}, "
              f"code={breakdown['code_int8_brotli_bytes']}) != canonical quantize_levelset_blob "
              f"({breakdown['canonical_quantize_blob_bytes']}) -- investigate (should match exactly).", flush=True)

    packet_dir = packet_dir or (
        _REPO / "experiments" / "results"
        / f"levelset_packet_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    _refuse_tmp(packet_dir, "packet_dir")
    zip_path, zip_bytes = assemble_packet(blob, packet_dir)
    rate = zip_bytes / RATE_DENOM
    rate_term = _cscore.rate_term(zip_bytes)  # canonical 25 * bytes / 37_545_489 (tac.contest_score)

    print(f"[ckpt] {ckpt_dir}/{cfg['npz_name']}  n_pairs={n_pairs}  params={breakdown['n_params']}  "
          f"self_orient={so['self_orient']}  {_AUTHORITY}", flush=True)
    print(f"[byte-close] 0.bin={breakdown['total_0bin_bytes']} B  archive.zip={zip_bytes} B  "
          f"rate={rate:.6f} rate_term={rate_term:.4f}  bank=FREE(rule118)  pose={pose_note}", flush=True)

    bit_exact: dict[str, Any] = {"checked": False}
    if verify_bit_exact:
        bit_exact = bit_exact_roundtrip_gate(packet_dir, blob, bit_exact_pairs, bit_exact_strict)
        bit_exact["checked"] = True

    inflate_info = run_inflate(packet_dir, n_pairs, max_pairs)
    print(f"[inflate] {inflate_info['frame_layout']}  full_output_ok={inflate_info['full_output_shape_ok']}  "
          f"raw_bytes={inflate_info['raw_bytes']}", flush=True)

    parity: dict[str, Any] = {"skipped": True}
    if not skip_parity:
        # GT-load count is the EVAL subset (eval_pairs), NOT the full checkpoint n_pairs: parity
        # only scores P = min(eval_pairs, gt.n_pairs) pairs, and a capped --max-pairs smoke must not
        # demand an n_pairs-sized GT cache (else a 32-pair smoke fails against gt_n96). For a full
        # eval (max_pairs=None) eval_pairs == n_pairs, so this is behaviour-identical there.
        parity = parity_on_inflated(Path(inflate_info["raw_path"]), inflate_info["eval_pairs"], gt_cache, inflate_info["eval_pairs"])
        d_seg = parity["d_seg_realized_on_inflated"]
        d_pose = parity["d_pose_realized_on_inflated"]
        # canonical tac.contest_score helpers (no hand-rolled formula; the old path carried a
        # +1e-12 epsilon inside sqrt that the canonical pose_term does not -- authority wins).
        seg_term = _cscore.seg_term(d_seg)
        pose_term = _cscore.pose_term(d_pose)
        score = _cscore.compute_contest_score(d_seg, d_pose, zip_bytes)
        parity.update({"seg_term": seg_term, "pose_term": pose_term, "rate_term": rate_term,
                       "implied_S_advisory": score})
        pose_blind = d_pose > 1.0
        print(f"[parity] d_seg={d_seg:.6f} d_pose={d_pose:.6f} (realized on INFLATED frames, "
              f"{parity['pairs_scored']} pairs) | S_advisory={score:.4f}  {_AUTHORITY}", flush=True)
        if pose_blind:
            print(f"[POSE-BLIND] realized d_pose={d_pose:.3f} >> 0 -> the render is POSE-BLIND "
                  "(w_pose=0). The S pose term dominates and is GARBAGE. A pose-TRAINED render "
                  "(--w-pose>0) is REQUIRED for the level-set ROW; a stored sidecar does NOT fix "
                  "it (the scorer runs PoseNet on the FRAMES).", flush=True)
        parity["pose_blind"] = bool(pose_blind)

    # #205: CONFIRM the pose-carrier decode through the REAL byte-closed inflate (frame0 bit-exact
    # reproduction + the d_pose triad). Only when the carrier is active AND we have GT to score.
    pose_carrier_confirmation: dict[str, Any] = {"checked": False}
    if pose_carrier and pose_carrier_bytes is not None and not skip_parity:
        confirmation_blob, confirmation_pose_carrier = _pose_carrier_confirmation_payload(
            packet_dir
        )
        pose_carrier_confirmation = pose_carrier_confirm(
            Path(inflate_info["raw_path"]), inflate_info["eval_pairs"], gt_cache, inflate_info["eval_pairs"],
            confirmation_pose_carrier, blob=confirmation_blob)
        pose_carrier_confirmation["checked"] = True
        pc_c = pose_carrier_confirmation
        print(f"[pose-carrier CONFIRM] frame0 decode bit-exact={pc_c['frame0_decode_bit_exact']} "
              f"(max_abs={pc_c['frame0_max_abs_uint8_diff']}) | d_pose: carrier(warp-f0+witness-f1)="
              f"{pc_c['d_pose_carrier_warp_f0_witness_f1']:.4f}  unwarped(gtf0+witness-f1)="
              f"{pc_c['d_pose_unwarped_gtf0_witness_f1']:.4f}  ceiling(gtf0+warp-gtf0)="
              f"{pc_c['d_pose_ceiling_gtf0_warpf0']:.4f}  (null~189.62)  {_AUTHORITY}", flush=True)
        if not pc_c["frame0_decode_bit_exact"]:
            print("[pose-carrier CONFIRM] WARNING: frame0 decode NOT bit-exact vs the numpy authority "
                  "-> the byte-close does NOT faithfully reproduce the pose decode (NO-FAKE).", flush=True)

    exact_eval: dict[str, Any] = {"ran": False}
    if run_exact_eval:
        exact_eval = run_upstream_evaluate(
            packet_dir,
            device=eval_device,
            uncompressed_dir=(uncompressed_dir or (_REPO / "upstream" / "videos")),
            video_names_file=(video_names_file or (_REPO / "upstream" / "public_test_video_names.txt")),
            archive_bytes=zip_bytes,
            timeout=eval_timeout,
        )
        print(f"[exact-eval] REAL evaluate.py --device {eval_device}: d_seg={exact_eval['d_seg']:.8f} "
              f"d_pose={exact_eval['d_pose']:.8f} -> S={exact_eval['evaluate_py_final_score']:.5f} "
              f"(recomputed {exact_eval['recomputed_S_compute_contest_score']:.5f}, "
              f"delta {exact_eval['recomputed_vs_evaluate_delta']:.2e}) {exact_eval['score_axis']}", flush=True)

    contest_cmd = (
        f".venv/bin/python experiments/contest_auth_eval.py "
        f"--archive {zip_path} --inflate-sh {packet_dir / 'inflate.sh'} "
        f"--device cpu  # [contest-CPU] authoritative ONLY on Linux x86_64 (Modal CPU); macOS-local = advisory"
    )

    report: dict[str, Any] = {
        "tool": "levelset_byte_close_and_eval",
        "authority": _AUTHORITY,
        "promotion_claim": False,
        "utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ckpt_dir": str(ckpt_dir),
        "npz_name": cfg["npz_name"],
        # (B1 confound-F1) which weights arm this run byte-closed. A single-arm run records its own
        # arm; a --select-arms run overwrites the top-level weights_arm with the WINNER + attaches the
        # full N-way ``arm_selection`` block (see select_best_weights_arm).
        "weights_arm": _arm_label_for_npz(cfg.get("npz_name")),
        "n_pairs_total": n_pairs,
        "self_orient": so,
        "basis_deploy": basis_deploy_report,
        "config": {k: cfg.get(k) for k in (
            "n_classes", "hidden_dim", "n_hidden", "mod_dim", "activation", "softmax_temp",
            "chroma", "render_h", "render_w", "in_feat", "max_bank_freq", "lane_edge_weight")},
        "byte_close": {
            **breakdown,
            "archive_zip_bytes": zip_bytes,
            "zip_container_overhead_bytes": zip_bytes - breakdown["total_0bin_bytes"],
            "rate": rate, "rate_term": rate_term, "rate_denom_bytes": int(RATE_DENOM),
            "archive_zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
            "inflate_py_bytes": (packet_dir / "inflate.py").stat().st_size,
            "inflate_py_sha256": hashlib.sha256((packet_dir / "inflate.py").read_bytes()).hexdigest(),
            "xi_receiver_camera_contract": xi_receiver_camera_contract(),
            "free_vs_counted": {
                "FREE_rule118": "curvelet bank (5 scalars) + self-orient directional feats "
                                "(decoder-own-argmax fixed point) -- generic algorithm, 0 bytes",
                "COUNTED": "in_proj/film/hidden.*/out_sdf/out_tex/palette weights + per-frame code "
                           "(int8+brotli, the learned video-derived payload)",
            },
            "pose_sidecar": pose_note,
            "lane_render_band": lane_report,
            "pose_carrier": pose_carrier_report,
            "phase_carrier": phase_carrier_report_out,
            "palette_residual": palette_residual_report,
            "blind_coordinate_fill": blind_coordinate_report,
            "basis_deploy": basis_deploy_report,
        },
        "lane_render_band": lane_report,
        "pose_carrier": pose_carrier_report,
        "phase_carrier": phase_carrier_report_out,
        "palette_residual": palette_residual_report,
        "dash_phase_carrier": dash_phase_carrier_report_out,
        "blind_coordinate_fill": blind_coordinate_report,
        "pose_carrier_keyframe_accounting": (
            None if keyframe_accounting is None else {
                **keyframe_accounting,
                # VERIFY the counted keyframe bytes against the ACTUAL bytes in the composed archive.
                "verified_against_archive": {
                    "pose_carrier_section_bytes_in_0bin": int(breakdown.get("pose_carrier_counted_bytes", 0)),
                    "section_bytes_match_report": bool(
                        int(breakdown.get("pose_carrier_counted_bytes", 0))
                        == int(pose_carrier_report.get("pose_carrier_section_bytes", -1))),
                    "keyframe_bytes_le_section": bool(
                        int(pose_carrier_report.get("keyframe_blob_bytes_total", 0))
                        <= int(breakdown.get("pose_carrier_counted_bytes", 0))),
                    "note": "keyframe payload is COUNTED inside archive.zip (0.bin 6th block -> the "
                            "measured rate term already includes it); the accounting is cross-checked "
                            "against the real section st_size (not an estimate).",
                },
            }),
        "pose_carrier_confirmation": pose_carrier_confirmation,
        "inflate": inflate_info,
        "bit_exact_roundtrip_gate": bit_exact,
        "parity_on_inflated_frames": parity,
        "exact_eval_upstream_evaluate": exact_eval,
        "contest_cpu_eval_cmd": contest_cmd,
        "packet_dir": str(packet_dir),
        "self_orient_parity_caveat": (
            "self-orient deploy dir feats are a FIXED-POINT on the FINAL weights; the trainer's "
            "implied_S used dir feats accumulated along the training trajectory -> close but not "
            "bit-identical; realized d_seg on the inflated frames is the TRUTH."
            if so["self_orient"] else "n/a (no self-orient)"),
        # checkpoint_trained_n600 = the CHECKPOINT encodes all 600 pair codes (a property of the
        # ckpt, NOT of this invocation). this_run_scored_full_600 = THIS run's realized parity
        # actually scored 600 pairs (False when --skip-parity or --max-pairs capped the decode).
        # The old single field "contest_ready_full_600" conflated the two (2026-07-06 review MED).
        "checkpoint_trained_n600": bool(n_pairs == 600),
        "this_run_scored_full_600": _this_run_scored_full_600(parity),
        "contest_ready_note": (
            (
                "checkpoint n_pairs==600 -> full 1200-frame .raw is producible (checkpoint-level); "
                "this_run_scored_full_600 says whether THIS invocation's parity scored all 600 "
                "pairs (skip-parity/max-pairs runs did NOT)"
            )
            if n_pairs == 600 else
            f"checkpoint n_pairs={n_pairs} != 600 -> inflate emits {2 * n_pairs} frames; a 600-pair "
            "witness is required for the 1200-frame contest .raw (this checkpoint is test-only)"),
    }
    if not keep_packet:
        import shutil
        shutil.rmtree(packet_dir, ignore_errors=True)  # disk-hygiene: the .raw is GBs (certify: rebuildable from archive.zip + inflate.py)
        report["packet_dir"] = "(deleted; pass --keep-packet to retain archive+inflate for the exact-eval row)"
    return report


def _this_run_scored_full_600(parity: Any) -> bool:
    """True iff THIS invocation's realized parity scored all 600 pairs. Fail-safe False when
    parity is absent / non-dict / skipped / capped (--skip-parity and --max-pairs runs are NOT
    full-600 rows). Distinct from ``checkpoint_trained_n600`` which is a property of the
    CHECKPOINT (n_pairs it encodes), not of what this invocation scored (2026-07-06 review MED)."""
    return bool(isinstance(parity, dict) and parity.get("pairs_scored") == 600)


def _extract_arm_metrics(report: dict[str, Any]) -> dict[str, Any]:
    """Pull the ranking-relevant scalars out of a single-arm byte-close report (B1)."""
    bc = report.get("byte_close", {}) or {}
    par = report.get("parity_on_inflated_frames", {}) or {}
    s = par.get("implied_S_advisory")
    return {
        "weights_arm": report.get("weights_arm"),
        "npz_name": report.get("npz_name"),
        "archive_zip_bytes": bc.get("archive_zip_bytes"),
        "rate_term": bc.get("rate_term"),
        "d_seg_realized_on_inflated": par.get("d_seg_realized_on_inflated"),
        "d_pose_realized_on_inflated": par.get("d_pose_realized_on_inflated"),
        "implied_S_advisory": (float(s) if isinstance(s, (int, float)) else None),
        "pose_blind": par.get("pose_blind"),
        "pairs_scored": par.get("pairs_scored"),
        "parity_skipped": bool(par.get("skipped", True)),
    }


def select_best_weights_arm(
    ckpt_dir: Path, *, arms: list[str] | None = None, **run_kwargs: Any
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """(B1 confound-F1 fix) Byte-close + realized-parity-score EVERY available weights arm
    ({ema, live, polyak} whose npz is present) and RECORD the N-way selection — the missing
    "picks the better candidate" consumer.

    Returns ``(winner_report, arm_reports)``. ``winner_report`` is the winning arm's full byte-close
    report with its top-level ``weights_arm`` set to the winner and an ``arm_selection`` block attached
    (per-arm d_seg/d_pose/rate/S, the ranked order, the winner, and the S-margin by which the winner
    beat each loser). Fail-open: a missing arm npz is simply absent from the set (an ema-only older run
    ranks a 1-arm "selection" honestly). Ranks by realized ``implied_S_advisory`` (lower is better);
    an arm with no S (unscored) sorts LAST. NO-FAKE: refuses ``skip_parity`` — you cannot "pick the
    better candidate" without measuring d_seg/d_pose through the real byte-closed inflate."""
    if run_kwargs.get("skip_parity"):
        raise ValueError(
            "select_best_weights_arm needs realized parity to rank arms — refusing --skip-parity "
            "(NO-FAKE: 'the better candidate' is undefined without a MEASURED d_seg/d_pose per arm).")
    available = list(arms) if arms is not None else discover_available_arms(ckpt_dir)
    available = [a for a in available if a in _ARM_NPZ and (ckpt_dir / _ARM_NPZ[a]).exists()]
    if not available:
        raise FileNotFoundError(
            f"no weights-arm npz present in {ckpt_dir} (looked for {list(_ARM_NPZ.values())}); "
            "refusing to fabricate (NO-FAKE).")

    per_arm: dict[str, dict[str, Any]] = {}
    arm_reports: dict[str, dict[str, Any]] = {}
    for arm in available:
        print(f"[select-arms] byte-closing + scoring arm={arm} ({_ARM_NPZ[arm]}) …", flush=True)
        rep = run(ckpt_dir, npz_name=_ARM_NPZ[arm], **run_kwargs)
        arm_reports[arm] = rep
        per_arm[arm] = _extract_arm_metrics(rep)

    def _rank_key(a: str) -> tuple[int, float]:
        s = per_arm[a]["implied_S_advisory"]
        return (0, float(s)) if isinstance(s, (int, float)) else (1, float("inf"))

    ranked = sorted(available, key=_rank_key)
    winner = ranked[0]
    win_s = per_arm[winner]["implied_S_advisory"]
    margins: dict[str, float | None] = {}
    for a in available:
        if a == winner:
            continue
        s = per_arm[a]["implied_S_advisory"]
        margins[a] = (
            float(s) - float(win_s)
            if isinstance(s, (int, float)) and isinstance(win_s, (int, float))
            else None)

    winner_report = arm_reports[winner]
    winner_report["weights_arm"] = winner
    winner_report["arm_selection"] = {
        "selected_by": "select_best_weights_arm",
        "metric": "implied_S_advisory (lower is better; realized on INFLATED frames, macOS-numpy)",
        "available_arms": list(available),
        "polyak_present": ("polyak" in available),
        "polyak_scored": (per_arm.get("polyak", {}).get("implied_S_advisory") is not None),
        "per_arm": per_arm,
        "ranked_arms": ranked,
        "winner": winner,
        "winner_implied_S_advisory": (float(win_s) if isinstance(win_s, (int, float)) else None),
        "margin_vs_winner": margins,   # loser_S - winner_S (>= 0 when both scored); None if unscored
        "authority": _AUTHORITY,
        "note": (
            "ALL available arms (ema/live/polyak) are byte-closed + realized-scored so the shipped "
            "candidate is the MEASURED best (confound F1 fix: polyak is no longer an orphaned "
            "candidate). Advisory S is macOS-numpy on inflated frames; the exact-eval row "
            "(upstream/evaluate.py CPU) remains the authority for a promotion claim."),
    }
    print(f"[select-arms] WINNER={winner} implied_S_advisory="
          f"{winner_report['arm_selection']['winner_implied_S_advisory']} | ranked={ranked} | "
          f"margins(loser_S-winner_S)={margins}  {_AUTHORITY}", flush=True)
    return winner_report, arm_reports


def byte_close_verdict_landed(report: dict) -> bool:
    """True iff a REAL realized-parity verdict landed in ``report`` (the gate for recording a #247
    ``measured`` activation event). The realized d_seg/d_pose lives under ``parity_on_inflated_frames``;
    a ``--skip-parity`` run stores ``{"skipped": True}`` there and NO verdict was computed.

    Fail-safe by construction: a MISSING key defaults to ``skipped=True`` (=> returns False => NO
    measured event). Recording a ``measured`` event when no verdict landed is a NO-FAKE violation that
    silently drains the duty_to_measure queue (the 2026-07-06 review-caught key-name bug — the gate
    read the wrong key ``"parity"`` and fired unconditionally, including on --skip-parity runs)."""
    parity = report.get("parity_on_inflated_frames")
    if not isinstance(parity, dict):
        return False
    return not parity.get("skipped", True)


# ---------------------------------------------------------------------------
# Clause-A geometric-section derivability audit (operator no-duplicate-data binding 2026-07-09 +
# design_philosophies_eightfold P1 "one fact, one store, one key"). SCORE-NEUTRAL, READ-ONLY ->
# defaults ON (CLAUDE.md "'Off' is a tracked queue": observability that only reads is not gate-able).
# Byte-identity: report-only; adds a report key, NEVER mutates archive.zip / the blob / any packet byte.
# ---------------------------------------------------------------------------
_DEDUP_AUDIT_MAX_FRAMES = 4  # cap frames for a $0 read-only audit (per-frame builds the analytic lane prior)


def dedup_audit_section(gt_cache: str | None, *, max_frames: int = _DEDUP_AUDIT_MAX_FRAMES) -> dict | None:
    """Pairwise derivability table over the archive's GEOMETRIC sections (the SegNet-argmax ledger rows
    horizon/lane/movable/hood/separatrix — the geometric homes the structured sections code). A
    shared-pixel pair is a byte one section could reconstruct from another (a double-count with no single
    geometric home) — the clause-A "no duplicate data" audit surfaced per run. Sources ``lstars`` (the GT
    SegNet argmax) from the byte-close ``--gt-cache`` npz, bounded to ``max_frames`` for a $0 read-only
    pass. Fail-open: returns ``None`` (never raises) when no gt-cache / no ``lstars`` key / any error —
    an audit that cannot run must never break the byte-close."""
    if not gt_cache:
        return None
    try:
        from tac.boundary_math.movable_deshare import pairwise_dedup_audit
        cp = Path(gt_cache)
        if not cp.is_file():
            return None
        with np.load(cp, allow_pickle=False) as z:
            if "lstars" not in getattr(z, "files", []):
                return None
            lst = np.asarray(z["lstars"])
        if lst.ndim == 2:
            lst = lst[None]
        if lst.ndim != 3 or lst.shape[0] == 0:
            return None
        nframes = int(min(max_frames, lst.shape[0]))
        audit = pairwise_dedup_audit(lst[:nframes])
        audit["source_gt_cache"] = str(cp)
        audit["n_frames_audited"] = nframes
        audit["note"] = (audit.get("note", "") + " [clause-A per-run derivability audit; report-only, "
                         "score-neutral, no archive mutation — P1 one-fact-one-store-one-key]").strip()
        return audit
    except Exception as exc:  # noqa: BLE001 — observability must NEVER break the byte-close
        return {"unavailable": f"{type(exc).__name__}: {exc}", "gt_cache": str(gt_cache)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", type=Path, required=True,
                    help="level-set run dir with levelset_witness_{ema,live}_mlx.npz")
    ap.add_argument("--npz-name", type=str, default=None,
                    help="explicit npz filename (default: prefer *_ema_mlx.npz then *_live_mlx.npz)")
    # (B1 confound-F1) N-way weights-arm selection: byte-close + score EVERY available arm
    # ({ema,live,polyak} present in the run dir) and record the ranked selection (per-arm scores +
    # winner + margins). OFF by default (single-arm run, byte-identical). The polyak candidate is no
    # longer orphaned. Needs realized parity (NOT --skip-parity).
    ap.add_argument("--select-arms", action="store_true",
                    help="byte-close + realized-score EVERY available weights arm (ema/live/polyak) "
                         "and pick the MEASURED best; records the N-way selection in the report. "
                         "Cannot combine with --skip-parity.")
    ap.add_argument("--arms", type=str, default=None,
                    help="comma-separated arms to select over (subset of ema,live,polyak); default = "
                         "every arm whose npz is present. Only used with --select-arms.")
    ap.add_argument("--max-pairs", type=int, default=None,
                    help="cap inflate+parity pairs for SPEED (default: all). Archive always encodes all codes.")
    ap.add_argument("--gt-cache", type=str, default=None, help="shared GT npz for parity (e.g. gt_n6.npz)")
    ap.add_argument("--skip-parity", action="store_true", help="byte-close + inflate only (no GT decode)")
    ap.add_argument("--keep-packet", action="store_true", help="retain the packet dir for the exact-eval row")
    ap.add_argument("--fold-pose-sidecar", action="store_true",
                    help="append a COUNTED stored-pose section; requires --pose-sidecar-path. OFF by "
                         "default. The inflate render does NOT read it -> does NOT lower realized d_pose.")
    ap.add_argument("--pose-sidecar-path", type=Path, default=None,
                    help="prebuilt posenet_targets.bin (tac.scorer_targets.extract_and_save)")
    # #205 WARP-REAL-LUMA FRAME0 POSE CARRIER. OFF by default -> byte-identical. Needs --gt-cache
    # (real gt_f0 keyframes + gt_poses). Replaces the render's SEG-free frame0 with a stored real
    # keyframe warped by the per-pair ego homography -> PoseNet reads real warped motion; the keyframe
    # payload is COUNTED in archive.zip (rule 118: keyframe COUNTED, warp decoder FREE).
    ap.add_argument("--pose-carrier", action="store_true",
                    help="#205 frame0 pose carrier (needs --gt-cache). frame0 = warp(source, per-pair "
                         "ego H); source per --pose-carrier-mode. OFF by default (archive byte-identical "
                         "when off).")
    ap.add_argument("--pose-carrier-mode", type=str, default="warp_real_luma",
                    choices=["warp_real_luma", "store_nothing"],
                    help="pose-carrier frame0 source. warp_real_luma (default) = warp a STORED real "
                         "keyframe (COUNTS keyframe luma+H). store_nothing = store ONLY xi/H (~0 marginal "
                         "bytes); frame0 = warp(the witness's OWN frame0 INR render, H) -- Track B "
                         "store-nothing-but-xi (18927a1ae). A/B-able against warp_real_luma at #205.")
    ap.add_argument("--pc-s-t", type=float, default=0.16, help="pose-carrier translation calibration scale (s_t).")
    ap.add_argument("--pc-s-r", type=float, default=1.0, help="pose-carrier rotation calibration scale (s_r).")
    ap.add_argument("--pc-pitch", type=float, default=0.02, help="pose-carrier road-plane pitch (rad).")
    ap.add_argument("--pc-keyframe-stride", type=int, default=1,
                    help="pose-carrier keyframe stride (1 = per-pair, the exact warp-real-luma upper "
                         "bound; >1 shares one stored keyframe across a window -> cheaper rate, "
                         "geometrically approximate for shared keyframes -- LOUDLY flagged).")
    ap.add_argument("--pc-keyframe-downscale", type=int, default=1,
                    help="pose-carrier keyframe store downscale factor (1 = native lossless; 2 = "
                         "native/2, upsampled before warp -- the cheap-rate lever, ~matched to PoseNet's "
                         "512x384 working res).")
    # #257 store-nothing ξ coder (store_nothing mode only; H is DERIVED FREE at decode, kf_of_pair DROPPED).
    ap.add_argument("--pc-xi-coder", type=str, default="delta_ar", choices=["delta_ar", "none"],
                    help="store-nothing ξ payload coder: delta_ar (DEFAULT) = per-channel temporal-delta + "
                         "arithmetic coder (~0.002 n600 rate); none = raw int16 ξ (~0.005, the "
                         "GUARANTEED-today fallback + strict-parity reference). Both decode to the IDENTICAL "
                         "quantized ξ. Ignored for warp_real_luma.")
    ap.add_argument("--no-xi-coder", action="store_true",
                    help="store-nothing: store raw int16 ξ (== --pc-xi-coder none). The guaranteed fallback.")
    ap.add_argument("--pc-xi-qlevels", type=int, default=4096,
                    help="store-nothing ξ quantization levels per channel (default 4096 ~ fp16+; finer -> "
                         "more coded bytes, coarser -> looser d_pose. Both coders share it -> strict parity).")
    # #238: ship the TRAINED per-pair ego twist ξ_eff = pose_carrier.xi_stored + pose_carrier.dxi from
    # the run checkpoint (store_nothing mode only) INSTEAD of the deterministic calibration recompute.
    # This is what carries a JOINT pose-descent run's trained d_pose (e.g. R1's 0.0011) into a shippable
    # archive; the dxi is video-derived -> COUNTED in the coded ξ payload. Pure byte-close serialization
    # mode (no trainer change). NOTE: pass --pc-pitch matching the trainer's --pose-carrier-pitch (R1=0.0).
    ap.add_argument("--pose-carrier-xi-from-ckpt", action="store_true",
                    help="#238 store-nothing: ship ξ_eff = pose_carrier.xi_stored + pose_carrier.dxi from "
                         "the SAME resolved checkpoint npz (the TRAINED twist), not the calibration "
                         "recompute. Requires --pose-carrier --pose-carrier-mode store_nothing. "
                         "Incompatible with --select-arms (each arm has its own dxi).")
    ap.add_argument("--pose-carrier-dxi-scale", type=float, default=None,
                    help="#238 (with --pose-carrier-xi-from-ckpt): ξ_eff = xi_stored + scale*dxi. Default = "
                         "the checkpoint's trained residual_scale (1.0). scale=0 = the MATCHED no-dxi "
                         "baseline (SAME fitted xi_stored, dxi off) -> the clean dxi A/B isolate.")
    # #359 PHASE-RESIDUAL CARRIER (store-half of the appearance-phase d_seg reframe). OFF by default
    # -> byte-identical (no _io_pack/inflate grammar change). Needs --gt-cache (lstars/margins/gt_poses).
    # BUILDS + MEASURES the section (real counted bytes -> a per-lever ΔS/byte candidate row for the #406
    # apply-pass); the through-R n600 d_seg it recovers is OWED-gated (memory-blocked by the live #205 run),
    # so d_seg is reported OWED, never claimed (NO-FAKE). Inflate-consumption wire lands with the A/B.
    ap.add_argument("--phase-carrier", action="store_true",
                    help="#359 phase-residual carrier: STORE the per-pair sub-pixel boundary-phase RESIDUAL "
                         "r = t_wit_actual - t_wit_xi_predicted (xi-transport amortized), entropy-coded. Needs "
                         "--gt-cache. Selectable build-time codec mode; OFF by default (byte-identical when off).")
    ap.add_argument("--phase-carrier-q-step", type=float, default=1.0 / 64.0,
                    help="phase-carrier sub-pixel tie quantization step in [0,1] (default 1/64).")
    ap.add_argument("--phase-carrier-classes", type=str, default="0,1,2",
                    help="phase-carrier GROUND class channels (default 0,1,2 = Road,Lane,Undrivable; the "
                         "homography is valid on the ground plane, WRONG on Movable/MyCar -> DEFERRED).")
    ap.add_argument("--phase-carrier-scheme", type=str, default="auto",
                    choices=["auto", "varint", "zlib9", "rice"],
                    help="phase-carrier residual entropy scheme (auto = best-of the three).")
    ap.add_argument("--phase-carrier-band", type=float, default=1.0,
                    help="phase-carrier annulus band |margin| < band (the flip-prone straddle set).")
    ap.add_argument("--phase-carrier-pitch", type=float, default=0.0,
                    help="phase-carrier ground-plane pitch for the xi-transport warp (rad).")
    # #425 DASH-PHASE CARRIER (curve-domain per-dash δ(s) codec). OFF by default -> byte-identical.
    ap.add_argument("--dash-phase-carrier", action="store_true",
                    help="#425 dash-phase carrier: track lane dashes by ξ-transport and STORE per-dash "
                         "curve-relative (δs, δn) residuals (prior-derived Huffman) + birth/death/rebirth "
                         "event codes. Needs --gt-cache. OFF by default (byte-identical when off).")
    ap.add_argument("--dash-phase-match-radius", type=float, default=6.0,
                    help="dash-phase track match radius in px (default 6).")
    ap.add_argument("--dash-phase-q-px", type=float, default=1.0,
                    help="dash-phase centroid/residual quantization step in px (default 1).")
    ap.add_argument("--dash-phase-dormant-max", type=int, default=30,
                    help="dash-phase dormant-pool horizon in frames for world-frame rebirth (default 30).")
    ap.add_argument("--dash-phase-no-xi", action="store_true",
                    help="dash-phase: omit the fp16 ξ block from the section (compose with the already-"
                         "banked L68 dxi; decoder then takes ξ externally).")
    ap.add_argument("--dash-phase-pitch", type=float, default=-0.01,
                    help="dash-phase ground-plane pitch for the ξ point-advection (rad; MEASURED "
                         "advection-memo calibration default -0.01).")
    ap.add_argument("--cross-tensor-codec", type=str, default="off",
                    choices=["off", "auto_lossless"],
                    help="lossless witness joint coder. auto_lossless exhaustively derives 2-D axis "
                         "storage permutations from exact Brotli bytes and selects raw-vs-frame-split "
                         "temporal-delta coding for the per-frame FiLM table. It acts after the canonical "
                         "int8 grid, so decoded quantized state is exact. DSL owner: "
                         "WitnessCrossTensorCoderGauge.AUTO_LOSSLESS.")
    ap.add_argument(
        "--blind-coordinate-fill",
        action="store_true",
        help=(
            "D21a/#401 rule-118 FREE fill of the 230,904 camera pixels/frame that are "
            "structurally invisible through R. Requires a measured n600 zero-delta receipt."
        ),
    )
    ap.add_argument(
        "--blind-coordinate-receipt",
        type=Path,
        default=None,
        help=(
            "blind_coordinate_proof.v1 JSON with n_pairs=600, bit-identical Pose/Seg inputs, "
            "and exactly zero max differences; mandatory when --blind-coordinate-fill is active"
        ),
    )
    ap.add_argument(
        "--palette-residual-section",
        type=Path,
        default=None,
        help=(
            "optional strict EKPR1 signed-int8 [n_pairs,n_classes,RGB] section. Applies only to "
            "frame1 after generator RGB via phi.argmax, before AA/lane/R. Section pair count must "
            "exactly match the checkpoint; a partial n24 section cannot attach to n600."
        ),
    )
    # #224 Wave E: decode-consistent analytic-lane RENDER-BAND (fork B). OFF by default -> byte-identical.
    # Fits per-pair lane manifold coords from --gt-cache (COUNTED), reproduces the coverage+composite
    # in inflate (FREE, rule 118) so the band is a REAL (non-phantom) score. Closes R5_BLOCK.
    ap.add_argument("--lane-render-band", action="store_true",
                    help="composite the decode-consistent analytic-lane render-band (needs --gt-cache). "
                         "COUNTS the per-pair lane coords in archive.zip; reproduces the coverage raster "
                         "FREE in inflate.py. OFF by default (archive byte-identical when off).")
    ap.add_argument("--lane-band-softness", type=float, default=1.0, help="AA-SDF coverage edge softness (px).")
    ap.add_argument("--lane-band-dash-forward-max", type=float, default=DEFAULT_DASH_FORWARD_MAX_M,
                    help="range-dependent dash-gate horizon (m); beyond it dashes read continuous (#215).")
    ap.add_argument("--lane-band-weight", type=float, default=1.0, help="global band strength in [0,1].")
    ap.add_argument("--lane-band-lane-cls", type=int, default=1, help="lane class index (comma10k canonical=1).")
    ap.add_argument("--lane-band-umask", action="store_true",
                    help="gate the band by the WITNESS's own softmax margin (top1-top2) -- the "
                         "decode-consistent FP killer (c_full_wit). OFF -> coverage-only band (c_range).")
    ap.add_argument("--lane-band-tau", type=float, default=0.85, help="witness-margin uncertainty threshold (prob).")
    ap.add_argument("--lane-band-eps", type=float, default=0.35, help="witness-margin uncertainty ramp.")
    ap.add_argument("--lane-band-naive", action="store_true",
                    help="Wave-F: use the NAIVE LBND1 float64 serializer instead of the OPTIMAL LBND2 RD "
                         "codec (quantize+temporal-delta+zigzag). Default = LBND2 RD (rate-viable). The "
                         "naive path is kept for the naive-vs-RD rate comparison + the default-off gate.")
    ap.add_argument("--lane-band-res", action="store_true",
                    help="LBND4 (DEFAULT OFF, sealed-config discipline): serialize the lane-coeff payload "
                         "on the SAME LBND2 quantization grid but with the ξ delta/context residual "
                         "entropy stage (best-of-three varint/zlib9/rice; MEASURED n600 −10,634 B / "
                         "−25.6%% vs LBND2, decode-reencode bit-identical — "
                         "experiments/results/lane_band_res_coder_20260707/). Mutually exclusive with "
                         "--lane-band-naive. NOTE: the inline _INFLATE_PY does not yet carry the LBND4 "
                         "decode; a SHIPPED LBND4 packet fails CLOSED at the parity gate until that "
                         "decode half is inlined (measurement lever first; NO-FAKE).")
    # self-orient params the trainer does NOT persist (a trainer gap, flagged) -> trainer defaults.
    ap.add_argument("--so-freq-across", type=float, default=32.0, help="self-orient HIGH freq across the edge (trainer default 32).")
    ap.add_argument("--so-freq-along", type=float, default=4.0, help="self-orient LOW freq along the edge (trainer default 4).")
    ap.add_argument("--so-tau", type=float, default=4.0, help="self-orient boundary-proximity tau (trainer default 4).")
    ap.add_argument("--so-iters", type=int, default=4, help="self-orient fixed-point iterations at decode (convergence).")
    # BIT-EXACT round-trip gate (the correctness proof: shipped inflate == numpy-fp32 oracle).
    ap.add_argument("--verify-bit-exact", action="store_true",
                    help="prove inflate(archive) == numpy-fp32 oracle forward, bit-for-bit on the "
                         ".raw frames (the correctness gate). Runs BEFORE the (long) full inflate.")
    ap.add_argument("--bit-exact-pairs", type=int, default=2,
                    help="pairs to compare in the bit-exact gate (per-pixel; 2 proves the forward).")
    ap.add_argument("--no-bit-exact-strict", action="store_true",
                    help="warn instead of raise on a bit-exact mismatch (default: strict/raise).")
    # REAL upstream/evaluate.py exact-eval row (CPU; NEVER MPS). Forces --keep-packet + all pairs.
    ap.add_argument("--run-exact-eval", action="store_true",
                    help="after inflating ALL pairs, run upstream/evaluate.py on the exact archive "
                         "bytes -> real d_seg/d_pose/rate/S. CPU 600-pair ~1-2h. Forces --keep-packet.")
    ap.add_argument("--eval-device", type=str, default=None, choices=["cpu", "cuda"],
                    help="evaluate.py device for --run-exact-eval (default: driven by --memory-tier — "
                         "cpu for decode_cpu_16gb, cuda for decode_t4_16gb; MPS is REFUSED).")
    ap.add_argument("--memory-tier", type=str, default=DEFAULT_TIER_NAME,
                    choices=sorted(DECODE_MEMORY_TIERS),
                    help="#224 Wave D decode (inflate) memory-tier: decode_cpu_16gb (DEFAULT, the "
                         "proven contest CPU path) / decode_t4_16gb (contest T4 CUDA host; SAME "
                         "bit-exact inflate) / production_edge (edge runtime, REFUSED here). Contest "
                         "tiers are bit-exact (fp64/CPU-torch-R/1-thread BLAS); a fp32/CUDA/multithread "
                         "tier is FORBIDDEN on contest tiers.")
    ap.add_argument("--uncompressed-dir", type=Path, default=None,
                    help="GT videos dir for evaluate.py (default upstream/videos; the rate denominator).")
    ap.add_argument("--video-names-file", type=Path, default=None,
                    help="video-names file for evaluate.py (default upstream/public_test_video_names.txt).")
    ap.add_argument("--eval-timeout", type=int, default=18000,
                    help="upstream/evaluate.py timeout seconds (default 5h for a CPU 600-pair run).")
    ap.add_argument("--out", type=Path, default=None, help="JSON report path")
    args = ap.parse_args(argv)

    # #224 Wave D: resolve + validate the decode memory-tier BEFORE anything. Fail-closed: the
    # contest byte-close/eval path REFUSES production_edge (its relaxed fp32/CUDA/multithread numeric
    # path can flip uint8 boundaries and is DEFERRED #228). Apply the tier's inflate env (1-thread
    # BLAS contract + optional worker cap) so the inflate subprocesses inherit it; drive the downstream
    # evaluate.py device from the tier (explicit --eval-device wins).
    tier = require_contest_tier(resolve_tier(args.memory_tier))  # raises DecodeTierError if refused
    _tier_env = tier_inflate_env(tier)
    for _k, _v in _tier_env.items():
        os.environ[_k] = _v
    eval_device = resolve_eval_device(tier, args.eval_device)
    print(f"# decode memory-tier: {tier.name} (contest={tier.contest}, bit_exact={tier.bit_exact_contract}, "
          f"eval_device={eval_device}); inflate env={_tier_env or '{} (inflate defaults)'}", flush=True)

    _run_kwargs: dict[str, Any] = dict(
        max_pairs=args.max_pairs,
        fold_pose_sidecar=args.fold_pose_sidecar,
        pose_sidecar_path=args.pose_sidecar_path,
        gt_cache=args.gt_cache,
        keep_packet=args.keep_packet,
        packet_dir=None,
        skip_parity=args.skip_parity,
        so_overrides={"freq_across": args.so_freq_across, "freq_along": args.so_freq_along,
                      "tau": args.so_tau, "iters": args.so_iters},
        lane_render_band=args.lane_render_band,
        lane_band_cfg=LaneBandRenderConfig(
            softness=args.lane_band_softness, dash_gate=True,
            dash_forward_max_m=args.lane_band_dash_forward_max, weight=args.lane_band_weight,
            lane_cls=args.lane_band_lane_cls, u_mask_enabled=args.lane_band_umask,
            u_mask_tau=args.lane_band_tau, u_mask_eps=args.lane_band_eps,
        ),
        lane_rd=not args.lane_band_naive,
        lane_res=args.lane_band_res,
        pose_carrier=args.pose_carrier,
        pose_carrier_cfg={
            "s_t": args.pc_s_t, "s_r": args.pc_s_r, "pitch": args.pc_pitch,
            "stride": args.pc_keyframe_stride, "downscale": args.pc_keyframe_downscale,
            "mode": args.pose_carrier_mode,
            "xi_coder": ("none" if args.no_xi_coder else args.pc_xi_coder),  # #257
            "xi_q_levels": args.pc_xi_qlevels,
        },
        phase_carrier=args.phase_carrier,  # #359
        phase_carrier_cfg={
            "q_step": args.phase_carrier_q_step,
            "classes": tuple(int(c) for c in str(args.phase_carrier_classes).split(",") if c.strip() != ""),
            "residual_scheme": args.phase_carrier_scheme,
            "annulus_band": args.phase_carrier_band,
            "pitch": args.phase_carrier_pitch,
        },
        dash_phase_carrier=args.dash_phase_carrier,  # #425
        dash_phase_carrier_cfg={
            "match_radius_px": args.dash_phase_match_radius,
            "q_px": args.dash_phase_q_px,
            "dormant_max_frames": args.dash_phase_dormant_max,
            "include_xi": not args.dash_phase_no_xi,
            "pitch": args.dash_phase_pitch,
        },
        cross_tensor_codec=(args.cross_tensor_codec == "auto_lossless"),
        blind_coordinate_fill=args.blind_coordinate_fill,
        blind_coordinate_receipt=args.blind_coordinate_receipt,
        palette_residual_path=args.palette_residual_section,
        verify_bit_exact=args.verify_bit_exact,
        bit_exact_pairs=args.bit_exact_pairs,
        bit_exact_strict=not args.no_bit_exact_strict,
        run_exact_eval=args.run_exact_eval,
        eval_device=eval_device,
        uncompressed_dir=args.uncompressed_dir,
        video_names_file=args.video_names_file,
        eval_timeout=args.eval_timeout,
    )
    # #238 CONNECTOR: load the TRAINED ξ_eff (xi_stored + dxi) from the SAME resolved checkpoint npz
    # so the byte-close SHIPS the trained twist (store-nothing) instead of the calibration recompute.
    if args.pose_carrier_xi_from_ckpt:
        if not (args.pose_carrier and args.pose_carrier_mode == "store_nothing"):
            raise SystemExit("--pose-carrier-xi-from-ckpt requires --pose-carrier "
                             "--pose-carrier-mode store_nothing (NO-FAKE).")
        if args.select_arms:
            raise SystemExit("--pose-carrier-xi-from-ckpt is incompatible with --select-arms "
                             "(each arm has its own trained dxi; pick one arm via --npz-name).")
        _candidates = ([args.npz_name] if args.npz_name
                       else [_wra.EMA_NPZ, _wra.LIVE_NPZ])
        _npz_path = next((args.ckpt_dir / c for c in _candidates if (args.ckpt_dir / c).exists()), None)
        if _npz_path is None:
            raise SystemExit(f"--pose-carrier-xi-from-ckpt: no checkpoint npz in {args.ckpt_dir} "
                             f"(looked for {_candidates}).")
        _z = np.load(_npz_path, allow_pickle=False)
        if "pose_carrier.xi_stored" not in _z.files or "pose_carrier.dxi" not in _z.files:
            raise SystemExit(f"--pose-carrier-xi-from-ckpt: {_npz_path.name} lacks "
                             "pose_carrier.xi_stored/pose_carrier.dxi (not a trained pose-carrier run).")
        _xi_stored = np.asarray(_z["pose_carrier.xi_stored"], dtype=np.float64)
        _dxi = np.asarray(_z["pose_carrier.dxi"], dtype=np.float64)
        # residual_scale defaults to 1.0 in the trainer (--pose-carrier-residual-scale); if a run saved
        # a non-default scale in cfg, honor it (ξ_eff = xi_stored + scale*dxi).
        _rscale = float(_z["__cfg_pose_carrier_residual_scale"].item()) \
            if "__cfg_pose_carrier_residual_scale" in _z.files else 1.0
        if args.pose_carrier_dxi_scale is not None:
            _rscale = float(args.pose_carrier_dxi_scale)  # 0 = matched no-dxi baseline; 1 = trained
        _xi_eff = _xi_stored + _rscale * _dxi
        _run_kwargs["pose_carrier_xi_override"] = _xi_eff
        print(f"# #238 SHIP-DXI: ξ_eff = xi_stored + {_rscale:g}*dxi from {_npz_path.name} "
              f"(shape {_xi_eff.shape}, |dxi|_mean={float(np.abs(_dxi).mean()):.5f}); "
              f"pc_pitch={args.pc_pitch} (must match trainer --pose-carrier-pitch).", flush=True)

    if args.select_arms:
        # (B1) N-way selection over the available weights arms; the winner report carries arm_selection.
        if args.npz_name:
            raise SystemExit(
                "--npz-name conflicts with --select-arms (the selection ranks the canonical arm npzs; "
                "an explicit filename would be silently ignored — pick one).")
        _arms = None
        if args.arms:
            _arms = [a.strip() for a in args.arms.split(",") if a.strip()]
            _bad = [a for a in _arms if a not in _ARM_NPZ]
            if _bad:
                raise SystemExit(f"--arms: unknown arm(s) {_bad}; valid = {sorted(_ARM_NPZ)}")
        report, _arm_reports = select_best_weights_arm(args.ckpt_dir, arms=_arms, **_run_kwargs)
    else:
        report = run(args.ckpt_dir, npz_name=args.npz_name, **_run_kwargs)
    # record the decode-tier contract in the report (observability + provenance).
    report["decode_memory_tier"] = {
        "name": tier.name, "contest": tier.contest, "bit_exact_contract": tier.bit_exact_contract,
        "eval_device": eval_device, "inflate_env": _tier_env, "note": tier.note,
    }
    # #402 receiver env manifest: pin decode-path dep versions + the cross-microarch bit-identity note.
    report["receiver_env"] = receiver_env_manifest()
    # clause-A geometric-section derivability audit (default-ON observability; report-only, byte-identical).
    _dedup = dedup_audit_section(args.gt_cache)
    if _dedup is not None:
        report["dedup_audit"] = _dedup
    out = args.out or (_REPO / "reports" / f"levelset_byte_close_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[report] wrote {out}  {_AUTHORITY}", flush=True)

    # #247 CLOSE: a byte-closed realized verdict landed -> drain the DSL levers this run FIRED from the
    # activation ledger's duty_to_measure (fired -> measured). Fail-safe: the ledger must NEVER break the
    # byte-close. Only when parity actually ran (a real realized d_seg/d_pose verdict, not --skip-parity).
    if byte_close_verdict_landed(report):
        try:
            from tac.witness_dsl.activation_ledger import record_measured_for_run
            rows = record_measured_for_run(
                str(args.ckpt_dir), verdict_ref=str(out),
                reason="byte-closed realized-parity verdict landed", agent="levelset_byte_close",
            )
            if rows:
                print(f"[activation-ledger] recorded measured for {len(rows)} fired lever(s): "
                      f"{[r['lever'] for r in rows]}", flush=True)
        except Exception as _e:  # noqa: BLE001 — advisory ledger, never blocks the verdict
            print(f"[activation-ledger] measured-record skipped (non-fatal): {_e}", flush=True)
        # #247 continual-learning: also record THIS run's identifiable costates into the cross-run
        # posterior (the compounding memory that sharpens the next run's DECIDE). Fail-safe.
        try:
            from tac.witness_control.costate_posterior import record_run_costates
            from tac.witness_control.shadow_controller import build_shadow_report, load_run_inputs
            _crep = build_shadow_report(load_run_inputs(args.ckpt_dir))
            _crows = record_run_costates(_crep.costates, str(args.ckpt_dir))
            if _crows:
                print(f"[costate-posterior] recorded {len(_crows)} identifiable costate(s) from this run "
                      f"into the cross-run posterior", flush=True)
        except Exception as _e:  # noqa: BLE001 — advisory memory, never blocks the verdict
            print(f"[costate-posterior] record skipped (non-fatal): {_e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
