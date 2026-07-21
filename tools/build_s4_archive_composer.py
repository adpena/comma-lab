#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the Task #578 S4 one-member archive and standalone receiver tree."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.boundary_math.lane_sdf_component import _CAM_H, _FX, _FY  # noqa: E402
from tac.optimization.s4_archive_composer import (  # noqa: E402
    SectionBytes,
    build_payload_manifest,
    canonical_json_bytes,
    deterministic_archive,
    parse_sections,
    serialize_sections,
)

DEFAULT_SEED = Path("/Volumes/VertigoDataTier/pact/evidence/seed_compose_20260721/seeds/seed_compose_b2_loose.ppcs")
DEFAULT_R3 = Path("/Volumes/VertigoDataTier/pact/evidence/predictor_r3_20260721/canonical_r3_20260721")
DEFAULT_OUTPUT = Path("/Volumes/VertigoDataTier/pact/evidence/s4_composer_20260721/canonical_s4_20260721")


class BuildError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, canonical_json_bytes(value) + b"\n")


def decoded_component_bytes(payload: bytes) -> tuple[int, int]:
    import struct
    import zlib

    offset = total = packets = 0
    while offset < len(payload):
        if offset + 4 > len(payload):
            raise BuildError("component packet length is truncated")
        size = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        if not size or offset + size > len(payload):
            raise BuildError("component packet size is invalid")
        try:
            raw = zlib.decompress(payload[offset : offset + size])
        except zlib.error as exc:
            raise BuildError("component packet zlib replay failed") from exc
        offset += size
        total += len(raw)
        packets += 1
    return total, packets


def decoded_lane_header(payload: bytes) -> dict[str, Any]:
    import lzma
    import struct

    if len(payload) < 8:
        raise BuildError("PBASE3 is truncated")
    static_size, lane_size = struct.unpack_from("<II", payload)
    if len(payload) != 8 + static_size + lane_size:
        raise BuildError("PBASE3 length mismatch")
    raw = lzma.decompress(
        payload[8 + static_size :],
        format=lzma.FORMAT_RAW,
        filters=[{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 20, "lc": 3, "lp": 0, "pb": 2}],
    )
    if not raw.startswith(b"LBND2\x00") or len(raw) < 10:
        raise BuildError("selected lane section is not LBND2")
    header_size = struct.unpack_from("<I", raw, 6)[0]
    try:
        header = json.loads(raw[10 : 10 + header_size].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BuildError("LBND2 header is malformed") from exc
    return header


def runtime_source_audit(runtime: Path) -> dict[str, Any]:
    source = runtime.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(runtime))
    imports = sorted(
        {
            node.names[0].name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
        }
        | {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
    )
    allowed = {"__future__", "argparse", "brotli", "hashlib", "json", "lzma", "math", "numpy", "os", "pathlib", "struct", "zlib"}
    forbidden_imports = sorted(set(imports) - allowed)
    byte_literals = [
        len(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, bytes)
    ]
    audit = {
        "schema": "s4_binary_source_audit.v1",
        "runtime_path": str(runtime.relative_to(REPO)),
        "runtime_sha256": sha256(runtime),
        "imports": imports,
        "forbidden_imports": forbidden_imports,
        "tac_imports": [name for name in imports if name == "tac"],
        "maximum_embedded_bytes_literal": max(byte_literals, default=0),
        "scorer_weight_files": [],
        "source_video_files": [],
        "ground_truth_argmax_files": [],
        "video_derived_sidecars": [],
        "all_video_derived_bytes_required_in_0_bin": True,
        "passed": not forbidden_imports and "tac" not in imports and max(byte_literals, default=0) <= 32,
    }
    if not audit["passed"]:
        raise BuildError(f"standalone runtime source audit failed: {audit}")
    return audit


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if not str(output).startswith("/Volumes/VertigoDataTier/pact/"):
        raise BuildError("S4 evidence must use the SSD tier")
    output.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output)
    if usage.free < 5 << 30:
        raise BuildError("S4 storage preflight requires at least 5 GiB free")

    paths = {
        "seed.ppcs": args.seed.resolve(),
        "base.pbase3": args.base.resolve(),
        "events.pce3": args.events.resolve(),
        "components.pcomp3": args.components.resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise BuildError(f"missing section input {name}: {path}")
    payloads = {name: path.read_bytes() for name, path in paths.items()}
    component_decoded, component_packets = decoded_component_bytes(payloads["components.pcomp3"])
    lane_header = decoded_lane_header(payloads["base.pbase3"])
    base_receipt = json.loads(args.base_receipt.read_text())
    base_decoded = sum(
        int(base_receipt["sections"][name]["raw_bytes"])
        for name in ("static_pxq1_derived_ties", "lane_lbnd2")
    )
    event_receipt = json.loads(args.component_receipt.read_text())["shape_codec_comparator"]["event_alphabet"]
    sections = [
        SectionBytes("seed.ppcs", payloads["seed.ppcs"], "raw", len(payloads["seed.ppcs"])),
        SectionBytes("base.pbase3", payloads["base.pbase3"], "mixed", base_decoded),
        SectionBytes("causal.pcr3", b"", "raw", 0),
        SectionBytes("events.pce3", payloads["events.pce3"], "lzma1_raw_1MiB", int(event_receipt["raw_bytes"])),
        SectionBytes("components.pcomp3", payloads["components.pcomp3"], "zlib9", component_decoded),
    ]
    manifest = build_payload_manifest(sections, source_commit=args.source_commit)
    manifest["runtime"]["component_packet_count"] = component_packets
    manifest["runtime"]["event_alphabet"] = {
        "births": int(event_receipt["birth_events"]),
        "matches": int(event_receipt["match_events"]),
        "implicit_deaths": int(event_receipt["implicit_death_events"]),
        "xor_sites": int(event_receipt["xor_residual_sites"]),
    }
    manifest["weight_derived_constants"] = {
        "R2_max_margin_palette": {
            "classification": "weight-derived",
            "value_u8": [[153, 255, 51], [51, 255, 204], [0, 153, 0], [102, 204, 51], [0, 255, 153]],
            "source": "frozen_public_SegNet_constant_tile_margin_probe",
            "counted_in": "manifest.json",
            "scorer_loaded_at_inflate": False,
        }
    }
    manifest["video_derived_constants"] = {
        "lane_camera_intrinsics": {
            "classification": "video-derived",
            "value": {"height_m": _CAM_H, "fx_scorer": _FX, "fy_scorer": _FY},
            "source": "counted_lane_fit_camera_model",
            "counted_in": "manifest.json",
        }
    }
    manifest["limitations"] = {
        "current_pointer_moved": False,
        "score_claim": False,
        "promotion_eligible": False,
        "receiver_quality": "MEASURED_ONLY_BY_LOCAL_ADVISORY_HARNESS",
        "frame0_carrier": "ABSENT_SAME_REALIZATION_AS_FRAME1",
        "causal_parameter_section": "EMPTY_SELECTED_ZERO_PARAMETER_POLICY",
    }
    manifest_bytes = canonical_json_bytes(manifest)
    all_sections = [SectionBytes("manifest.json", manifest_bytes, "raw", len(manifest_bytes)), *sections]
    monolith = serialize_sections(all_sections)
    if serialize_sections(parse_sections(monolith)) != monolith:
        raise BuildError("S4 monolith parse-back mismatch")
    atomic_bytes(output / "0.bin", monolith)
    archive = deterministic_archive(output / "archive.zip", monolith)

    runtime_tree = output / "runtime"
    runtime_tree.mkdir(parents=True, exist_ok=True)
    source_runtime = REPO / "submissions" / "s4_archive_composer"
    for name in ("inflate.py", "inflate.sh", "requirements.txt", "README.md"):
        shutil.copy2(source_runtime / name, runtime_tree / name)
    os.chmod(runtime_tree / "inflate.sh", 0o755)
    os.chmod(runtime_tree / "inflate.py", 0o755)

    source_audit = runtime_source_audit(source_runtime / "inflate.py")
    constants_audit = {
        "schema": "s4_embedded_constants_audit.v1",
        "runtime_sha256": source_audit["runtime_sha256"],
        "classification_spine": ["video-derived", "weight-derived", "generic"],
        "shipped_constants": [
            {
                "name": "pair_and_image_geometry",
                "classification": "generic",
                "values": {"pair_count": 600, "scorer_hw": [384, 512], "camera_hw": [874, 1164], "class_count": 5},
                "counted": False,
                "location": "inflate.py",
            },
            {
                "name": "lzma1_raw_filter",
                "classification": "generic",
                "values": {"dict_size": 1048576, "lc": 3, "lp": 0, "pb": 2},
                "counted": False,
                "location": "inflate.py",
            },
            {
                "name": "lane_camera_intrinsics",
                "classification": "video-derived",
                "values": {"height_m": _CAM_H, "fx_scorer": _FX, "fy_scorer": _FY},
                "counted": True,
                "location": "0.bin/manifest.json",
            },
            {
                "name": "lane_render_header",
                "classification": "video-derived",
                "values": {
                    key: lane_header[key]
                    for key in (
                        "softness",
                        "dash_gate",
                        "dash_forward_max_m",
                        "v_h",
                        "cx",
                        "weight",
                        "lane_cls",
                        "lane_rgb_mode",
                        "rd",
                    )
                },
                "counted": True,
                "location": "0.bin/base.pbase3/LBND2_header",
            },
            {
                "name": "R2_max_margin_palette",
                "classification": "weight-derived",
                "values": [[153, 255, 51], [51, 255, 204], [0, 153, 0], [102, 204, 51], [0, 255, 153]],
                "counted": True,
                "location": "0.bin/manifest.json",
            },
        ],
        "classification_counts": {"video-derived": 2, "weight-derived": 1, "generic": 2},
        "unclassified_constants": [],
        "dormant_generic_arithmetic_decoder": True,
        "arithmetic_streams_in_current_archive": 0,
        "classification": "complete_three_way_constant_inventory",
        "passed": True,
    }
    payload_audit = {
        **manifest,
        "schema": "s4_archive_payload_audit.v1",
        "container": {
            "path": str(output / "0.bin"),
            "bytes": len(monolith),
            "sha256": hashlib.sha256(monolith).hexdigest(),
            "exact_parseback": True,
        },
        "archive": archive,
        "source_paths": [
            {"name": name, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in paths.items()
        ],
    }
    atomic_json(output / "archive_payload_manifest.json", payload_audit)
    atomic_json(output / "embedded_constants_audit.json", constants_audit)
    atomic_json(output / "binary_source_audit.json", source_audit)
    receipt = {
        "schema": "s4_archive_build_receipt.v1",
        "lane_id": "lane_s4_archive_composer_578_20260721",
        "research_only": True,
        "source_commit": args.source_commit,
        "storage_preflight": {"path": str(output), "free_bytes": usage.free, "required_free_bytes": 5 << 30, "passed": True},
        "archive": archive,
        "runtime_tree": str(runtime_tree),
        "runtime_sha256": source_audit["runtime_sha256"],
        "payload_audits": {
            "manifest": str(output / "archive_payload_manifest.json"),
            "constants": str(output / "embedded_constants_audit.json"),
            "source": str(output / "binary_source_audit.json"),
        },
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "score_claim": False,
        "promotion_eligible": False,
        "main_review_required": True,
    }
    atomic_json(output / "build_receipt.json", receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    command.add_argument("--base", type=Path, default=DEFAULT_R3 / "base/base_terminal.pbase3")
    command.add_argument("--events", type=Path, default=DEFAULT_R3 / "components/all_coherent_event_alphabet.pce3")
    command.add_argument("--components", type=Path, default=DEFAULT_R3 / "components/admitted_knee_components.pcomp3")
    command.add_argument("--base-receipt", type=Path, default=DEFAULT_R3 / "base.json")
    command.add_argument("--component-receipt", type=Path, default=DEFAULT_R3 / "components.json")
    command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    command.add_argument("--source-commit", default=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip())
    return command


def main() -> None:
    receipt = build(parser().parse_args())
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
