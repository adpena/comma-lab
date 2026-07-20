#!/usr/bin/env python3
"""Measure R1b5 carrier bytes and a fail-closed compact-layout break-even."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import zipfile
import zlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
SRC: Final = REPO / "src"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.boundary_math.r1b4_section_receiver import (  # noqa: E402
    APPLICATION_ORDER,
    RECEIVER_SCHEMA,
    default_receiver_policy,
    encode_replay_payload,
)
from tac.boundary_math.windowed_curvelet_frame import WindowedCurveletConfig  # noqa: E402
from tac.optimization.boundary_coordinate_joint_solve import (  # noqa: E402
    BoundaryCoordinatePacket,
    FrameFamily,
    encode_boundary_packet,
)
from tac.optimization.r1b2_mdl_xi0_compile import (  # noqa: E402
    ARCHIVE_SCHEMA,
    BOUNDARY_NAME,
    CONDITIONAL_CARRIER_LIMIT_BYTES,
    MANIFEST_NAME,
    PAIR_COUNT,
    REPLAY_NAME,
    XI0_NAME,
    _sha256_bytes,
    _write_zip,
    _zip_members,
    canonical_json,
    sha256_file,
)
from tac.optimization.r1b3_producer_preflight import (  # noqa: E402
    decode_xi0_payload,
    encode_xi0_payload,
)

DEFAULT_CONTROL: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/m1_c2_glue_rebuild_20260719/pre_archive.zip"
)
DEFAULT_XI0: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/r1b3_producers_20260720T185300Z/xi0.xi0"
)
DEFAULT_XI0_CALIBRATION: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/r1b5_row_closer_20260720/xi0_calibration/receipt.json"
)
DEFAULT_VJP: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/vjp_custody_20260719/extension_n600_20260720/"
    "campaign_receipt.json"
)
DEFAULT_ARTIFACT_ROOT: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/r1b5_row_closer_20260720/carrier_audit"
)
_BOUNDARY_PREFIX: Final = struct.Struct("<4sIII")
_XI0_PREFIX: Final = struct.Struct("<4sII")
_REPLAY_PREFIX: Final = struct.Struct("<4sII")
_CRC_BYTES: Final = 4


class CarrierAuditError(RuntimeError):
    """Fail-closed carrier audit error."""


def _custody(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise CarrierAuditError(f"receipt overwrite refused: {path}")
    payload = (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    partial = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with partial.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _zero_boundary_payload() -> bytes:
    packet = BoundaryCoordinatePacket(
        family=FrameFamily.WINDOWED_CURVELET,
        frame_config=asdict(WindowedCurveletConfig(n_scales=1, n_orient0=2, n_trans=1)),
        scorer_height=384,
        scorer_width=512,
        atom_indices=np.asarray([0], dtype=np.uint32),
        coefficients=np.zeros((PAIR_COUNT, 1, 3), dtype=np.int8),
        scales=np.ones(PAIR_COUNT, dtype=np.float16),
    )
    return encode_boundary_packet(packet)


def _member_rows(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path, "r") as archive:
        return [
            {
                "name": row.filename,
                "raw_bytes": row.file_size,
                "compressed_bytes": row.compress_size,
                "zip_structure_bytes": 76 + 2 * len(row.filename.encode("utf-8")),
            }
            for row in archive.infolist()
        ]


def _section_envelope(payload: bytes, prefix: struct.Struct) -> dict[str, int]:
    _magic, header_bytes, body_bytes = prefix.unpack_from(payload)
    expected = prefix.size + header_bytes + body_bytes + _CRC_BYTES
    if expected != len(payload):
        raise CarrierAuditError("section envelope length drifted")
    return {
        "prefix_bytes": prefix.size,
        "json_header_bytes": header_bytes,
        "body_bytes": body_bytes,
        "crc_bytes": _CRC_BYTES,
        "total_bytes": expected,
    }


def _build_manifest(
    *,
    control: Path,
    base_members: Sequence[tuple[str, bytes]],
    boundary: bytes,
    replay: bytes,
    xi0: bytes,
    source_manifest_hashes: Mapping[str, str],
) -> dict[str, Any]:
    with zipfile.ZipFile(control, "r") as archive:
        base_infos = archive.infolist()
    base_compressed = sum(row.compress_size for row in base_infos)
    base_overhead = control.stat().st_size - base_compressed
    return {
        "schema": ARCHIVE_SCHEMA,
        "pair_count": PAIR_COUNT,
        "artifact_role": "r1b2_candidate",
        "base_archive_sha256": sha256_file(control),
        "base_archive_bytes": control.stat().st_size,
        "base_zip_compressed_bytes": base_compressed,
        "base_zip_overhead_bytes": base_overhead,
        "base_sections": {
            name: {"bytes": len(payload), "sha256": _sha256_bytes(payload)}
            for name, payload in base_members
        },
        "sections": {
            name: {"bytes": len(payload), "sha256": _sha256_bytes(payload)}
            for name, payload in (
                (BOUNDARY_NAME, boundary),
                (REPLAY_NAME, replay),
                (XI0_NAME, xi0),
            )
        },
        "source_manifest_hashes": dict(source_manifest_hashes),
        "offline_full_kernel_selection": True,
        "receiver_search": False,
        "xi_coordinate_indices": [0],
        "receiver_schema": RECEIVER_SCHEMA,
        "receiver_policy": default_receiver_policy(),
        "application_order": list(APPLICATION_ORDER),
        "final_output_assertion": {
            "status": "unsealed",
            "pair_cap": PAIR_COUNT,
            "decoded_bytes": PAIR_COUNT * 2 * 874 * 1164 * 3,
            "decoded_sha256": None,
        },
        "score_claim": False,
    }


def _measure_case(
    *,
    control: Path,
    boundary: bytes,
    replay: bytes,
    xi0: bytes,
    source_manifest_hashes: Mapping[str, str],
    output: Path,
) -> dict[str, Any]:
    base_members = _zip_members(control)
    manifest = _build_manifest(
        control=control,
        base_members=base_members,
        boundary=boundary,
        replay=replay,
        xi0=xi0,
        source_manifest_hashes=source_manifest_hashes,
    )
    _write_zip(
        output,
        [
            *base_members,
            (MANIFEST_NAME, canonical_json(manifest)),
            (BOUNDARY_NAME, boundary),
            (REPLAY_NAME, replay),
            (XI0_NAME, xi0),
        ],
    )
    rows = _member_rows(output)
    base_names = {name for name, _payload in base_members}
    extension = [row for row in rows if row["name"] not in base_names]
    carrier_delta = output.stat().st_size - control.stat().st_size
    compressed = {row["name"]: row["compressed_bytes"] for row in extension}
    return {
        "archive_bytes": output.stat().st_size,
        "archive_sha256": sha256_file(output),
        "control_bytes": control.stat().st_size,
        "carrier_delta_bytes": carrier_delta,
        "conditional_limit_bytes": CONDITIONAL_CARRIER_LIMIT_BYTES,
        "excess_bytes": carrier_delta - CONDITIONAL_CARRIER_LIMIT_BYTES,
        "pass": carrier_delta <= CONDITIONAL_CARRIER_LIMIT_BYTES,
        "members": extension,
        "fixed_cost_excluding_xi0_compressed_bytes": carrier_delta - compressed[XI0_NAME],
        "minimum_non_xi0_reduction_even_if_xi0_free": max(
            0, carrier_delta - compressed[XI0_NAME] - CONDITIONAL_CARRIER_LIMIT_BYTES
        ),
        "manifest_raw_bytes": len(canonical_json(manifest)),
    }


def _raw_deflate_size(payload: bytes) -> int:
    compressor = zlib.compressobj(level=9, method=zlib.DEFLATED, wbits=-15)
    return len(compressor.compress(payload) + compressor.flush())


def _compact_binary_projection(
    *,
    control: Path,
    boundary: bytes,
    replay: bytes,
    direct_shifts: bytes,
    source_manifest_hashes: Mapping[str, str],
) -> tuple[bytes, bytes]:
    """Return a projected v2 descriptor + direct-shift payload, not a decoder contract."""

    base_members = _zip_members(control)
    with zipfile.ZipFile(control, "r") as archive:
        infos = archive.infolist()
    base_compressed = sum(row.compress_size for row in infos)
    base_overhead = control.stat().st_size - base_compressed
    direct_payload = (
        struct.pack("<4sBH", b"XS2\0", 2, PAIR_COUNT)
        + direct_shifts
        + struct.pack("<I", zlib.crc32(direct_shifts) & 0xFFFFFFFF)
    )
    extension_tree = hashlib.sha256()
    for name, payload in (
        (BOUNDARY_NAME, boundary),
        (REPLAY_NAME, replay),
        (XI0_NAME, direct_payload),
    ):
        encoded = name.encode("utf-8")
        extension_tree.update(struct.pack("<H", len(encoded)))
        extension_tree.update(encoded)
        extension_tree.update(struct.pack("<I", len(payload)))
        extension_tree.update(hashlib.sha256(payload).digest())
    base_tree = hashlib.sha256()
    for name, payload in base_members:
        base_tree.update(name.encode("utf-8") + b"\0" + hashlib.sha256(payload).digest())
    source_tree = hashlib.sha256(canonical_json(dict(source_manifest_hashes))).digest()
    descriptor = struct.pack(
        "<4sBHIIII32s32s32sIII",
        b"R1C2",
        2,
        PAIR_COUNT,
        control.stat().st_size,
        base_compressed,
        base_overhead,
        PAIR_COUNT * 2 * 874 * 1164 * 3,
        base_tree.digest(),
        extension_tree.digest(),
        source_tree,
        len(boundary),
        len(replay),
        len(direct_payload),
    )
    return descriptor, direct_payload


def execute(args: argparse.Namespace) -> int:
    root = args.artifact_root.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise CarrierAuditError(f"artifact root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    control = args.control_archive.expanduser().resolve(strict=True)
    xi0_path = args.xi0.expanduser().resolve(strict=True)
    calibration_path = args.xi0_calibration.expanduser().resolve(strict=True)
    vjp_path = args.vjp_campaign.expanduser().resolve(strict=True)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("schema") != "r1b5_xi0_coordinate_warp_calibration.v1":
        raise CarrierAuditError("xi0 calibration schema drifted")
    selected = calibration.get("selected_prefix_policy")
    policy = calibration.get("policies", {}).get(str(selected), {}).get("policy")
    if not isinstance(policy, dict) or policy.get("kind") != "scalar":
        raise CarrierAuditError("carrier audit requires the selected scalar prefix calibration")

    boundary = _zero_boundary_payload()
    replay = encode_replay_payload(())
    banked_xi0 = xi0_path.read_bytes()
    xi0_values = decode_xi0_payload(banked_xi0).astype(np.float64)
    gain = float(policy["gain"])
    shifts = np.clip(np.rint(gain * xi0_values), -16, 16).astype(np.int8)
    source_hashes = {"vjp_campaign": sha256_file(vjp_path)}

    with tempfile.TemporaryDirectory(prefix="r1b5_carrier_", dir=root) as temp_name:
        temp = Path(temp_name)
        fixture_control = temp / "fixture_control.zip"
        _write_zip(
            fixture_control,
            [("0.bin", b"base" * 256), ("ipe_manifest.json", b"{}")],
        )
        typed_fixture = _measure_case(
            control=fixture_control,
            boundary=boundary,
            replay=replay,
            xi0=encode_xi0_payload(np.full(PAIR_COUNT, 31.0, dtype=np.float32)),
            source_manifest_hashes={"vjp": "a" * 64},
            output=temp / "typed_fixture.zip",
        )
        banked = _measure_case(
            control=control,
            boundary=boundary,
            replay=replay,
            xi0=banked_xi0,
            source_manifest_hashes=source_hashes,
            output=temp / "banked_reference.zip",
        )
        descriptor, direct_payload = _compact_binary_projection(
            control=control,
            boundary=boundary,
            replay=replay,
            direct_shifts=shifts.tobytes(order="C"),
            source_manifest_hashes=source_hashes,
        )
        base_members = _zip_members(control)
        projection_path = temp / "projection.zip"
        _write_zip(
            projection_path,
            [
                *base_members,
                (MANIFEST_NAME, descriptor),
                (BOUNDARY_NAME, boundary),
                (REPLAY_NAME, replay),
                (XI0_NAME, direct_payload),
            ],
        )
        projection_delta = projection_path.stat().st_size - control.stat().st_size
        projection = {
            "archive_bytes": projection_path.stat().st_size,
            "carrier_delta_bytes": projection_delta,
            "conditional_limit_bytes": CONDITIONAL_CARRIER_LIMIT_BYTES,
            "headroom_bytes": CONDITIONAL_CARRIER_LIMIT_BYTES - projection_delta,
            "structural_byte_target_pass": projection_delta <= CONDITIONAL_CARRIER_LIMIT_BYTES,
            "parse_back": False,
            "receiver_bound": False,
            "admissible": False,
            "members": _member_rows(projection_path)[-4:],
            "descriptor_raw_bytes": len(descriptor),
            "direct_shift_payload_raw_bytes": len(direct_payload),
        }

    boundary_envelope = _section_envelope(boundary, _BOUNDARY_PREFIX)
    xi0_envelope = _section_envelope(banked_xi0, _XI0_PREFIX)
    replay_envelope = _section_envelope(replay, _REPLAY_PREFIX)
    result = {
        "schema": "r1b5_carrier_byte_audit.v1",
        "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "verdict": "MEASURED_CARRIER_BREAK_EVEN_PRODUCTION_LAYOUT_BLOCKED",
        "verdict_scope": (
            "exact deterministic ZIP accounting for the settled 2114-byte typed fixture and the "
            "banked xi0 plus zero-boundary/zero-replay reference; compact v2 is a byte projection "
            "only, not parser/receiver-bound and not a candidate archive"
        ),
        "authority": {
            "axis": "[byte accounting; no scorer axis]",
            "score_claim": False,
            "promotion_eligible": False,
            "pointer": "0.19108 [contest-CPU] UNMOVED",
        },
        "inputs": {
            "control_archive": _custody(control),
            "xi0": _custody(xi0_path),
            "xi0_calibration": _custody(calibration_path),
            "vjp_campaign": _custody(vjp_path),
        },
        "typed_fixture_exact": typed_fixture,
        "banked_xi0_zero_boundary_reference": banked,
        "precision_and_duplicate_information": {
            "boundary": {
                "envelope": boundary_envelope,
                "body_fields": {
                    "shared_atom_indices_uint32": 4,
                    "per_pair_rgb_coefficients_int8": 1_800,
                    "per_pair_scales_float16": 1_200,
                },
                "tolerance_status": (
                    "int8 coefficients plus fp16 scales already use the settled mixed-precision shape; "
                    "production coefficient/scale error-to-admission tolerance remains absent"
                ),
            },
            "replay": {
                "envelope": replay_envelope,
                "entry_body_bytes": 0,
                "duplicate_information": (
                    "179 compressed bytes carry an empty replay header; zero selection is derivable "
                    "from the common manifest/section length"
                ),
            },
            "xi0": {
                "envelope": xi0_envelope,
                "float16_quantization_mse": 0.00003707520629783782,
                "float16_quantization_max_abs": 0.01558685302734375,
                "selected_prefix_gain": gain,
                "realized_shift_dtype": "int8",
                "realized_shift_body_bytes": shifts.nbytes,
                "realized_shift_raw_deflate_bytes": _raw_deflate_size(shifts.tobytes()),
                "realized_shift_histogram": {
                    str(value): count for value, count in sorted(Counter(shifts.tolist()).items())
                },
                "losslessness_scope": (
                    "direct int8 shifts reproduce the selected scalar actuator exactly; the scalar "
                    "policy itself is prefix-only and not production-admitted"
                ),
            },
            "manifest": {
                "banked_reference_raw_bytes": banked["manifest_raw_bytes"],
                "banked_reference_compressed_bytes": next(
                    row["compressed_bytes"]
                    for row in banked["members"]
                    if row["name"] == MANIFEST_NAME
                ),
                "duplicate_fields": [
                    "receiver_policy values fixed by receiver schema",
                    "application_order fixed by receiver code",
                    "pair_count repeated in every section",
                    "section byte lengths repeated in section envelopes",
                    "zero replay semantics repeated in its JSON header",
                ],
            },
        },
        "compact_binary_v2_projection": projection,
        "canonical_break_even": {
            "typed_fixture_required_reduction_bytes": typed_fixture["excess_bytes"],
            "banked_reference_required_reduction_bytes": banked["excess_bytes"],
            "xi0_only_cannot_close_typed_fixture": (
                typed_fixture["minimum_non_xi0_reduction_even_if_xi0_free"] > 0
            ),
            "typed_fixture_minimum_non_xi0_reduction_bytes": typed_fixture[
                "minimum_non_xi0_reduction_even_if_xi0_free"
            ],
            "banked_reference_minimum_non_xi0_reduction_bytes": banked[
                "minimum_non_xi0_reduction_even_if_xi0_free"
            ],
            "required_implementation": (
                "legacy-compatible compact manifest/section envelope plus direct int8 shifts, gated "
                "on n600 Seg-conditioned calibration and exact receiver parse-back"
            ),
        },
        "production_blockers": [
            "REAL_BOUNDARY_PACKET_AND_REPLAY_BYTES_ABSENT",
            "COMPACT_V2_CODEC_PARSER_RECEIVER_BINDING_ABSENT",
            "XI0_SELECTED_POLICY_N600_SEG_CONDITIONED_VALIDATION_ABSENT",
            "PER_FIELD_BOUNDARY_QUANTIZATION_TOLERANCE_ABSENT",
        ],
        "source": {
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    _atomic_json(root / "receipt.json", result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-archive", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--xi0", type=Path, default=DEFAULT_XI0)
    parser.add_argument("--xi0-calibration", type=Path, default=DEFAULT_XI0_CALIBRATION)
    parser.add_argument("--vjp-campaign", type=Path, default=DEFAULT_VJP)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    return parser


if __name__ == "__main__":
    raise SystemExit(execute(_parser().parse_args()))
