#!/usr/bin/env python3
"""Build the retained NR1 K32 packet into a fresh shipping DX2 receiver.

The build is scorer-free.  It pins the K32 producer and every inherited byte,
constructs the counted NI1 archive, installs the additive decoder into a copied
DX2 runtime, retains two complete decoded fields, copies the full coder race,
and proves that mutations of every paid surface are refused.  The canonical
local advisory firer is the only downstream scorer launcher.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
import shutil
import struct
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
BASE_RUNTIME = Path("/Volumes/APDataStore/pact/ddm_dx2/r7/candidate_runtime_dx2")
K32_ROOT = Path(
    "/Volumes/APDataStore/pact/ddm_nr1_taskcell_quotient_prebuild/vq8_k32_e8192_v1"
)
K32_RESULT = K32_ROOT / "RESULT.json"
K32_PACKET = K32_ROOT / "retained/coder/nr1_packet.bin"
K32_CODER_MANIFEST = K32_ROOT / "retained/coder/CODER_MANIFEST.json"
K32_RETAINED_PRODUCER = (
    K32_ROOT / "retained/producer_source/ddm_nr1_taskcell_quotient_prebuild.py"
)
NR1_MODULE = REPO / "src/tac/optimization/nr1_taskcell_quotient.py"
NR1_MATERIALIZER = REPO / "experiments/ddm_nr1_taskcell_quotient_prebuild.py"
NI1_RECEIVER_SOURCE = REPO / "experiments/ddm_ni1_runtime_receiver.py"
PER_CLASS_METHOD = REPO / "experiments/ddm_ri1_per_class_seg_retention.py"
DX2_CANDIDATE_SEAL = Path(
    "/Volumes/APDataStore/pact/ddm_dx2/r7/CANDIDATE_SEAL_dx2_fx5_cabac.json"
)

DEFAULT_OUT = Path(
    "/Volumes/APDataStore/pact/ddm_ni1_nr1_k32_receiver_distortion/build_r1"
)
MIN_FREE_BYTES = 2 * 1024**3
STRICT_SUB012_CEILING_BYTES = 137_986
CONSERVATIVE_PROJECTED_BYTES = 135_595

PINS = {
    K32_RESULT: "d3e7d58c286c82813d0356f3681b76e940d3bb206e88eaad8feebe0c68ace623",
    K32_PACKET: "a68765dc683fa8302b560ef3db0d4a1507eeeccc695322fb8b69f684ed6dab28",
    K32_CODER_MANIFEST: "91a54bfa9eac22ca02fb1413e89fe7a515158ff195583bbedb4de68d3d42089c",
    K32_RETAINED_PRODUCER: "44e8ac10d20ca9c6325d572ac44cd3be9b553409ef0c408d41074f5ef9d7847c",
    NR1_MODULE: "66500b813eeafeaf264d57ecb47ef68360956ec1bdb040043456f3d6f101cbb6",
    NR1_MATERIALIZER: "252717246d10484f88da7a04aa43103bea94252afe7a7b5072d7d7053113c93b",
    PER_CLASS_METHOD: "f7088728f68922b55f2d15623f1cc16bb499e658d2f976e821cced392b9e3d2c",
    REPO / ".omx/research/ddm_nr1_taskcell_quotient_prebuild_20260822.md": "e1ae945821f60d0c0fc2de062b6325c2773fde24125dbb1975862bc3c296c64d",
    REPO / ".omx/research/ddm_rc1_rate_crush_20260822.md": "dfb239fcda4a749925326500b8821637d969e204d7eaf9191d64fc7a524e7c8d",
    REPO / ".omx/research/ddm_vf1_evaluator_visible_floor_20260822.md": "f65e641edfc987a127dd2813d4136bbb01ad1c46ef4b211c80176416afcb87b4",
    REPO / ".omx/research/ddm_jx1_joint_exchange_envelope_20260822.md": "9a6a6adcd06cd4faf454c28b5f0175a691a7da07112457535b2a1521ed92f6fd",
    REPO / ".omx/research/ddm_db1_decode_boundary_families_20260822.md": "08fd9c4b5d4e583293c3977a8a98abb0205b0a0fc0443e67bd5247aed2de86af",
    DX2_CANDIDATE_SEAL: "f3e8970cc2168ed904a8944bbffc43823b02a1ea845aaf790642ce1226a1d13d",
}
BASE_RUNTIME_PINS = {
    "archive.zip": "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674",
    "inflate.py": "b9571ee3c7bd1d7c22c42dc06a7da6e2b095803a5c05eec77993972d57d77ed4",
    "inflate.sh": "971eaa12b78e716825741ea86c28f9362eb9be077cc8cb3b873810ca979beb65",
    "runtime/f26_inflate.py": "5d705f93c051b2b540845dad4140f73d7dd61c721e4de2ed33b2ad32170c35c4",
    "runtime/residual_archive.py": "aca361f3e94941f4f2800bacec79f5032335588e317e76ee1a306bbb5ba64530",
    "cpr1/inflate.py": "ff446edd9237148bdc898be2f8f8c4782bf231a50cf3830c4b0b21a4474a736b",
}

RX1_HEADER = struct.Struct("<4sBBBBHHH")
NI1_HEADER = struct.Struct("<4sBBHIIII32s32s32s32s32s")
DX2_MEMBER_BYTES = 180_268
K32_PACKET_BYTES = 69_004
DECODED_TOKEN_BYTES = 117_964_800
DECODED_TOKEN_SHA256 = (
    "d416895a250ce79be7f485188d4f7dfd1690a269a250063c2f6bc5f48cf8b8d8"
)
SECTION_PINS = {
    "hpac": (13_515, "602115b323b0e403d08287af9b273a2d4fb23e026d83c1f6e4609ed77ef98f98"),
    "semantic": (30_856, "39d1be52ba62933498395c48ce4d9482f37db097d504da76c2a321efe3e4a76f"),
    "carrier": (22_010, "932b979f5181b331a9099162c6f392f558860b7998c62a36f38c2c99629c9b12"),
    "residual": (96, "8ab2fe748ab7d69d2102ba2292289e22bd7ea503f8ae29938e0854ec46ca3da1"),
    "tokens": (113_777, "e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5"),
}


class NI1BuildError(RuntimeError):
    """A custody, integration, retention, or mutation invariant failed."""


def sha256_bytes(payload: bytes | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def runtime_digest(runtime: Path) -> dict[str, Any]:
    source_root = str(REPO / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from tac.candidate_seal import measure_runtime_digest

    return measure_runtime_digest(runtime).to_dict()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with source.open("rb") as src, temporary.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=8 << 20)
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(temporary, destination)


def replace_once(text: str, anchor: str, replacement: str, label: str) -> str:
    count = text.count(anchor)
    if count != 1:
        raise NI1BuildError(f"patch anchor {label!r} appears {count} times")
    return text.replace(anchor, replacement, 1)


def archive_with_member(member: bytes) -> bytes:
    sink = io.BytesIO()
    info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o100644 << 16
    with zipfile.ZipFile(sink, "w") as archive:
        archive.writestr(info, member)
    return sink.getvalue()


def verify_materializer_drift() -> dict[str, Any]:
    retained = K32_RETAINED_PRODUCER.read_text(encoding="utf-8")
    expected_current = retained.replace("DX2_TOKEN_SHA256", "DX2_TASK_FIELD_SHA256")
    current = NR1_MATERIALIZER.read_text(encoding="utf-8")
    if retained.count("DX2_TOKEN_SHA256") != 2 or current != expected_current:
        raise NI1BuildError("NR1 materializer drift is not the reviewed two-use symbol rename")
    return {
        "disposition": "SOURCE_DRIFT_ACCEPTED_EXACT_SYMBOL_RENAME_ONLY",
        "retained": file_fact(K32_RETAINED_PRODUCER),
        "workspace": file_fact(NR1_MATERIALIZER),
        "semantic_change": False,
        "exact_change": "DX2_TOKEN_SHA256 -> DX2_TASK_FIELD_SHA256 at two uses",
    }


def pin_inherited_state() -> dict[str, Any]:
    facts: dict[str, Any] = {}
    for path, expected in PINS.items():
        actual = sha256_file(path)
        if actual != expected:
            raise NI1BuildError(f"inherited pin drifted: {path}: {actual} != {expected}")
        facts[str(path)] = file_fact(path)
    for relative, expected in BASE_RUNTIME_PINS.items():
        path = BASE_RUNTIME / relative
        actual = sha256_file(path)
        if actual != expected:
            raise NI1BuildError(f"DX2 runtime drifted: {relative}: {actual} != {expected}")
        facts[str(path)] = file_fact(path)

    result = json.loads(K32_RESULT.read_text(encoding="utf-8"))
    packet = result.get("packet", {})
    if (
        result.get("config", {}).get("codebook_size_requested") != 32
        or result.get("config", {}).get("event_limit") != 8192
        or packet.get("bytes") != K32_PACKET_BYTES
        or packet.get("sha256") != PINS[K32_PACKET]
        or result.get("token_mismatch_count") != 1_558_833
        or result.get("consumption_trace")
        != {"QCTX": 1, "QEVENT": 1, "QPAIR": 1, "QPARAM": 1}
    ):
        raise NI1BuildError("K32 result no longer names the chartered exact-once row")
    coder_manifest = json.loads(K32_CODER_MANIFEST.read_text(encoding="utf-8"))
    expected_attribution = {
        "QPARAM": 239,
        "QCTX": 152,
        "QPAIR": 52_124,
        "QEVENT": 16_489,
    }
    actual_attribution = {
        name: int(value["bytes"])
        for name, value in coder_manifest.get("physical_attribution", {}).items()
    }
    if actual_attribution != expected_attribution:
        raise NI1BuildError("K32 physical attribution differs from the charter")

    dx2_seal = json.loads(DX2_CANDIDATE_SEAL.read_text(encoding="utf-8"))
    measured_runtime = runtime_digest(BASE_RUNTIME)
    sealed_runtime = dict(dx2_seal.get("runtime", {}))
    sealed_runtime.pop("path", None)
    if measured_runtime != sealed_runtime:
        raise NI1BuildError("complete DX2 runtime differs from its candidate seal")
    return {
        "facts": facts,
        "materializer_drift": verify_materializer_drift(),
        "k32_result_row": {
            "token_agreement": result["token_agreement"],
            "token_mismatch_count": result["token_mismatch_count"],
            "physical_attribution": expected_attribution,
        },
    }


def split_dx2_member() -> dict[str, bytes]:
    with zipfile.ZipFile(BASE_RUNTIME / "archive.zip") as archive:
        if archive.namelist() != ["p"]:
            raise NI1BuildError("DX2 archive member census differs")
        outer = archive.read("p")
    if len(outer) != DX2_MEMBER_BYTES:
        raise NI1BuildError("DX2 member byte count differs")
    fields = RX1_HEADER.unpack_from(outer)
    if fields != (b"RX1M", 1, 2, 0, 26, 13_515, 30_856, 22_010):
        raise NI1BuildError("DX2 RX1 header differs")
    cursor = RX1_HEADER.size
    sections: dict[str, bytes] = {}
    for name in ("hpac", "semantic", "carrier", "residual", "tokens"):
        count, expected_sha = SECTION_PINS[name]
        sections[name] = outer[cursor : cursor + count]
        cursor += count
        if len(sections[name]) != count or sha256_bytes(sections[name]) != expected_sha:
            raise NI1BuildError(f"DX2 {name} section differs")
    if cursor != len(outer):
        raise NI1BuildError("DX2 section arithmetic did not consume the member")
    return sections


def build_ni1_archive() -> tuple[bytes, dict[str, Any]]:
    sections = split_dx2_member()
    packet = K32_PACKET.read_bytes()
    member = NI1_HEADER.pack(
        b"NI1A",
        1,
        1,
        0,
        len(sections["semantic"]),
        len(sections["carrier"]),
        len(sections["residual"]),
        len(packet),
        bytes.fromhex(BASE_RUNTIME_PINS["archive.zip"]),
        hashlib.sha256(sections["semantic"]).digest(),
        hashlib.sha256(sections["carrier"]).digest(),
        hashlib.sha256(sections["residual"]).digest(),
        hashlib.sha256(packet).digest(),
    ) + sections["semantic"] + sections["carrier"] + sections["residual"] + packet
    archive = archive_with_member(member)
    report = {
        "member_bytes": len(member),
        "member_sha256": sha256_bytes(member),
        "archive_bytes": len(archive),
        "archive_sha256": sha256_bytes(archive),
        "sections": {
            name: {"bytes": len(payload), "sha256": sha256_bytes(payload)}
            for name, payload in (
                ("semantic", sections["semantic"]),
                ("carrier", sections["carrier"]),
                ("residual", sections["residual"]),
                ("nr1_packet", packet),
            )
        },
        "removed_paid_section": {
            "name": "DX2 HPAC",
            "bytes": len(sections["hpac"]),
            "sha256": sha256_bytes(sections["hpac"]),
            "reason": "QCTX and QPAIR replace the token context and temporal model",
        },
    }
    return archive, report


def patch_runtime(runtime: Path, archive: bytes) -> dict[str, Any]:
    source_runtime_digest = runtime_digest(BASE_RUNTIME)
    shutil.copytree(
        BASE_RUNTIME,
        runtime,
        copy_function=shutil.copyfile,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "._*", ".DS_Store"),
    )
    atomic_copy(NR1_MODULE, runtime / "runtime/nr1_taskcell_quotient.py")
    atomic_copy(NI1_RECEIVER_SOURCE, runtime / "runtime/ni1_nr1_receiver.py")
    atomic_bytes(runtime / "archive.zip", archive)

    archive_sha = sha256_bytes(archive)
    archive_bytes = len(archive)
    f26_path = runtime / "runtime/f26_inflate.py"
    f26 = f26_path.read_text(encoding="utf-8")
    f26 = replace_once(
        f26,
        "from .residual_archive import decode_production_tokens, read_residual_archive",
        "from .ni1_nr1_receiver import decode_ni1_tokens, read_ni1_archive",
        "NI1 receiver import",
    )
    f26 = replace_once(
        f26,
        '        runtime_dir / "residual_archive.py",\n        runtime_dir / "hpac_inference.py",',
        '        runtime_dir / "residual_archive.py",\n        runtime_dir / "nr1_taskcell_quotient.py",\n        runtime_dir / "ni1_nr1_receiver.py",\n        runtime_dir / "hpac_inference.py",',
        "NI1 fingerprint files",
    )
    f26 = replace_once(
        f26,
        '''    parts = read_residual_archive(archive_path)\n    if parts.schema != "fixed_boundary_int6" or parts.token_codec != "rc64":\n        raise InflationError("archive does not use the fixed F26 residual schema")''',
        '''    parts, ni1_decoded, ni1_decoded_digest, ni1_archive_report = read_ni1_archive(archive_path)\n    if parts.schema != "fixed_boundary_int6" or parts.token_codec != "nr1q-k32":\n        raise InflationError("archive does not use the NI1 fixed residual + NR1 token schema")''',
        "NI1 archive parser",
    )
    f26 = replace_once(
        f26,
        '''    token_decoder = os.environ.get("F26_TOKEN_DECODER", "python").strip()\n    if token_decoder not in {"python", "native-hpac"}:\n        raise InflationError("F26_TOKEN_DECODER must be 'python' or 'native-hpac'")\n    if token_decoder != "python":\n        raise InflationError(\n            "this generation wires the ddm_rr2 free probability corrector into the "\n            "python token decoder only; the native-hpac path is unpatched and would "\n            "decode a different field, so it is refused rather than trusted"\n        )\n    if pair_count != int(renderer.N) and token_decoder != "native-hpac":\n        raise InflationError("advisory prefix inflation requires the resumable native token path")''',
        '''    token_decoder = os.environ.get("F26_TOKEN_DECODER", "ni1-nr1").strip()\n    if token_decoder != "ni1-nr1":\n        raise InflationError("NI1 requires F26_TOKEN_DECODER=ni1-nr1")\n    if pair_count != int(renderer.N):\n        raise InflationError("NI1 accepts only the complete n600 receiver field")''',
        "NI1 decoder selection",
    )
    f26 = replace_once(
        f26,
        '''    if loaded is None:\n        if token_decoder == "native-hpac":\n            if checkpoint_dir is None:\n                raise InflationError("native token decode requires checkpoint_dir")\n            from .f26_hpac_native import decode_native_tokens\n\n            native_progress = checkpoint_dir.resolve() / "native_hpac_progress"\n            tokens, token_report = decode_native_tokens(\n                parts,\n                renderer,\n                renderer_dir,\n                device,\n                frame_limit=pair_count,\n                output_path=native_progress / "tokens_partial.u8",\n                checkpoint_dir=native_progress / "checkpoints",\n            )\n        else:\n            tokens, token_report = decode_production_tokens(parts, renderer, renderer_dir, device)''',
        '''    if loaded is None:\n        tokens, token_report = decode_ni1_tokens(ni1_decoded, ni1_decoded_digest)''',
        "NI1 token decode",
    )
    f26 = replace_once(
        f26,
        '        "residual_schema": parts.schema,\n        "compensation": compensation_report,',
        '        "residual_schema": parts.schema,\n        "ni1_archive": ni1_archive_report,\n        "compensation": compensation_report,',
        "NI1 report",
    )
    atomic_bytes(f26_path, f26.encode("utf-8"))

    inflate_path = runtime / "inflate.py"
    inflate = inflate_path.read_text(encoding="utf-8")
    inflate = replace_once(
        inflate,
        'ARCHIVE_SHA256 = "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674"',
        f'ARCHIVE_SHA256 = "{archive_sha}"',
        "archive SHA",
    )
    inflate = replace_once(
        inflate,
        "ARCHIVE_BYTES = 180_368",
        f"ARCHIVE_BYTES = {archive_bytes:_}",
        "archive bytes",
    )
    atomic_bytes(inflate_path, inflate.encode("utf-8"))

    shell_path = runtime / "inflate.sh"
    shell = shell_path.read_text(encoding="utf-8")
    shell = replace_once(
        shell,
        'export F26_TOKEN_DECODER="${F26_TOKEN_DECODER:-python}"',
        '''if [[ -n "${F26_TOKEN_DECODER:-}" && "$F26_TOKEN_DECODER" != "ni1-nr1" ]]; then\n  echo "NI1 refuses a non-NR1 token decoder" >&2\n  exit 69\nfi\nexport F26_TOKEN_DECODER="ni1-nr1"''',
        "NI1 shell decoder",
    )
    atomic_bytes(shell_path, shell.encode("utf-8"))
    shell_path.chmod(shell_path.stat().st_mode | 0o111)

    if file_fact(runtime / "archive.zip") != {
        "path": str((runtime / "archive.zip").resolve()),
        "bytes": archive_bytes,
        "sha256": archive_sha,
    }:
        raise NI1BuildError("staged NI1 archive differs")
    return {
        "schema": "ddm_ni1_runtime_build.v1",
        "base_runtime": str(BASE_RUNTIME),
        "source_runtime_digest": source_runtime_digest,
        "candidate_runtime_digest": runtime_digest(runtime),
        "shipping_renderer": str((runtime / "cpr1/inflate.py").resolve()),
        "shipping_renderer_sha256": sha256_file(runtime / "cpr1/inflate.py"),
        "candidate_files": {
            relative: file_fact(runtime / relative)
            for relative in (
                "archive.zip",
                "inflate.py",
                "inflate.sh",
                "runtime/f26_inflate.py",
                "runtime/residual_archive.py",
                "runtime/nr1_taskcell_quotient.py",
                "runtime/ni1_nr1_receiver.py",
                "cpr1/inflate.py",
            )
        },
        "integration_boundary": (
            "NR1 replaces only the terminal token decoder and its superseded HPAC model; "
            "semantic weights, carrier, frame-0 selector, compensation, "
            "SemanticTokenRenderer, R/uint8 path, and render_video remain copied DX2 code"
        ),
    }


def load_staged_receiver(runtime: Path):
    runtime_text = str(runtime)
    if runtime_text not in sys.path:
        sys.path.insert(0, runtime_text)
    for name in tuple(sys.modules):
        if name == "runtime" or name.startswith("runtime."):
            del sys.modules[name]
    return importlib.import_module("runtime.ni1_nr1_receiver")


def write_token_payload(path: Path, tokens: Any) -> dict[str, Any]:
    array = tokens.detach().cpu().contiguous().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        array.tofile(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fact = file_fact(path)
    if fact["bytes"] != DECODED_TOKEN_BYTES or fact["sha256"] != DECODED_TOKEN_SHA256:
        raise NI1BuildError("retained decoded token payload differs")
    return fact


def retain_parseback_and_repeats(runtime: Path, retained: Path) -> dict[str, Any]:
    receiver = load_staged_receiver(runtime)
    parts, decoded, digest, parse_report = receiver.read_ni1_archive(runtime / "archive.zip")
    tokens, decode_report = receiver.decode_ni1_tokens(decoded, digest)
    first = write_token_payload(retained / "receiver/tokens_full.u8", tokens)
    del tokens
    parts2, decoded2, digest2, parse_report2 = receiver.read_ni1_archive(
        runtime / "archive.zip"
    )
    repeat_tokens, repeat_report = receiver.decode_ni1_tokens(decoded2, digest2)
    repeat = write_token_payload(retained / "receiver/tokens_full.repeat.u8", repeat_tokens)
    del repeat_tokens
    if (
        first["sha256"] != repeat["sha256"]
        or decode_report != repeat_report
        or parse_report != parse_report2
    ):
        raise NI1BuildError("independent NI1 parse/decode repeat differs")
    archive_repeat = retained / "archive.repeat.zip"
    atomic_copy(runtime / "archive.zip", archive_repeat)
    if sha256_file(archive_repeat) != sha256_file(runtime / "archive.zip"):
        raise NI1BuildError("exact archive repeat differs")
    return {
        "schema": "ddm_ni1_parseback_repeat.v1",
        "parse_report": parse_report,
        "decode_report": decode_report,
        "token_payload": first,
        "token_repeat_payload": repeat,
        "archive": file_fact(runtime / "archive.zip"),
        "archive_repeat": file_fact(archive_repeat),
        "parts": {
            "semantic_sha256": sha256_bytes(parts.semantic_blob),
            "carrier_sha256": sha256_bytes(parts.carrier_blob),
            "residual_sha256": sha256_bytes(parts.residual_payload),
            "nr1_packet_sha256": sha256_bytes(parts.token_stream),
            "repeat_nr1_packet_sha256": sha256_bytes(parts2.token_stream),
        },
        "repeat_identity": True,
    }


def retain_coder_race(retained: Path) -> dict[str, Any]:
    manifest = json.loads(K32_CODER_MANIFEST.read_text(encoding="utf-8"))
    destination_root = retained / "inherited_coder_race"
    copies = []
    for expected in manifest.get("retained_files", []):
        source = Path(expected["path"])
        actual = file_fact(source)
        if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
            raise NI1BuildError(f"K32 coder-race payload drifted: {source}")
        relative = source.relative_to(K32_ROOT / "retained/coder")
        destination = destination_root / relative
        atomic_copy(source, destination)
        copy_fact = file_fact(destination)
        if copy_fact["bytes"] != expected["bytes"] or copy_fact["sha256"] != expected["sha256"]:
            raise NI1BuildError(f"K32 coder-race copy differs: {destination}")
        copies.append(copy_fact)
    if len(copies) != 36:
        raise NI1BuildError(f"K32 coder-race census is {len(copies)}, expected 36")
    atomic_copy(K32_CODER_MANIFEST, destination_root / "CODER_MANIFEST.source.json")
    result = {
        "schema": "ddm_ni1_inherited_coder_race.v1",
        "complete": True,
        "source_manifest": file_fact(K32_CODER_MANIFEST),
        "source_manifest_copy": file_fact(destination_root / "CODER_MANIFEST.source.json"),
        "payload_census": len(copies),
        "payloads": copies,
        "all_losers_and_repeats_retained": True,
    }
    atomic_json(destination_root / "COPY_MANIFEST.json", result)
    result["copy_manifest"] = file_fact(destination_root / "COPY_MANIFEST.json")
    return result


def replace_packet_in_member(member: bytes, packet: bytes) -> bytes:
    fields = list(NI1_HEADER.unpack_from(member))
    semantic_bytes, carrier_bytes, residual_bytes, packet_bytes = fields[4:8]
    packet_offset = NI1_HEADER.size + semantic_bytes + carrier_bytes + residual_bytes
    if packet_bytes != len(packet) or packet_offset + packet_bytes != len(member):
        raise NI1BuildError("NI1 mutation packet boundary differs")
    fields[-1] = hashlib.sha256(packet).digest()
    return NI1_HEADER.pack(*fields) + member[NI1_HEADER.size:packet_offset] + packet


def retain_mutation_controls(runtime: Path, retained: Path) -> dict[str, Any]:
    receiver = load_staged_receiver(runtime)
    with zipfile.ZipFile(runtime / "archive.zip") as archive:
        member = archive.read("p")
    fields = NI1_HEADER.unpack_from(member)
    semantic_bytes, carrier_bytes, residual_bytes, packet_bytes = fields[4:8]
    semantic_offset = NI1_HEADER.size
    carrier_offset = semantic_offset + semantic_bytes
    residual_offset = carrier_offset + carrier_bytes
    packet_offset = residual_offset + residual_bytes
    packet = member[packet_offset : packet_offset + packet_bytes]
    attribution = receiver.nr1.physical_attribution(packet)

    mutations: dict[str, tuple[bytes, int, str | None]] = {}
    for name, offset in {
        "semantic": semantic_offset,
        "carrier": carrier_offset,
        "residual": residual_offset,
    }.items():
        mutated = bytearray(member)
        mutated[offset] ^= 0x01
        mutations[name] = (bytes(mutated), offset, None)
    for section, (start, _end) in attribution.items():
        mutated_packet = bytearray(packet)
        mutated_packet[start] ^= 0x01
        inner_refusal = None
        try:
            receiver.nr1.parse_packet(bytes(mutated_packet))
        except Exception as error:
            inner_refusal = f"{type(error).__name__}: {error}"
        if inner_refusal is None:
            raise NI1BuildError(f"{section.value} inner NR1 mutation was accepted")
        mutations[section.value] = (
            replace_packet_in_member(member, bytes(mutated_packet)),
            packet_offset + start,
            inner_refusal,
        )

    controls = {}
    for name, (mutated_member, offset, inner_refusal) in mutations.items():
        path = retained / "mutation_controls" / f"{name}_bitflip.archive.zip"
        atomic_bytes(path, archive_with_member(mutated_member))
        refusal = None
        try:
            receiver.read_ni1_archive(path)
        except Exception as error:  # Exact retained refusal text is part of the receipt.
            refusal = f"{type(error).__name__}: {error}"
        if refusal is None:
            raise NI1BuildError(f"{name} paid-section mutation was accepted")
        controls[name] = {
            "mutated_offset_in_member": offset,
            "mutation": "xor 0x01",
            "archive": file_fact(path),
            "receiver_disposition": "REFUSED",
            "receiver_error": refusal,
            "inner_nr1_parser_error": inner_refusal,
        }
    if set(controls) != {"semantic", "carrier", "residual", "QPARAM", "QCTX", "QPAIR", "QEVENT"}:
        raise NI1BuildError("paid-section mutation control census differs")
    return {
        "schema": "ddm_ni1_paid_section_mutation_controls.v1",
        "all_paid_sections_refused": True,
        "evidence_scope": "integrity/refusal only; distortion evidence comes from shipped output",
        "controls": controls,
    }


def seal_main_fire_order(out: Path, runtime: Path, build: dict[str, Any]) -> dict[str, Any]:
    attempt_root = Path(
        "/Volumes/VertigoDataTier/pact/ddm_ni1_nr1_k32_receiver_distortion"
    )
    advisory_jobs = []
    for index in (1, 2):
        label = f"ddm_ni1_nr1_k32_receiver_distortion_advisory_r{index}"
        attempt = attempt_root / f"advisory_r{index}"
        advisory_jobs.append(
            {
                "order": 1 if index == 1 else 3,
                "disposition": "QUEUED",
                "owner": "MAIN",
                "consumer_store": str(
                    out / f"harvest/advisory_r{index}/HARVEST.json"
                ),
                "fire_trigger": (
                    "MAIN holds the sole full-n600 scorer slot; every earlier full-n600 "
                    "job is terminal; candidate pins and >=12 GiB Vertigo free space revalidate"
                    if index == 1
                    else "advisory_r1 and per_class_r1 are terminal and MAIN again holds the sole full-n600 scorer slot"
                ),
                "attempt_dir": str(attempt),
                "expected_result": str(attempt / "contest_auth_eval.json"),
                "argv": [
                    str(REPO / ".venv/bin/python"),
                    "tools/fire_local_advisory.py",
                    "--runtime-dir",
                    str(runtime),
                    "--attempt-dir",
                    str(attempt),
                    "--label",
                    label,
                    "--rss-mb",
                    "32768",
                    "--timeout",
                    "21600",
                    "--projected-gib",
                    "12",
                    "--inflate-timeout",
                    "5400",
                    "--evaluate-timeout",
                    "14400",
                ],
            }
        )
    order = {
        "schema": "ddm_ni1_nr1_k32_main_fire_order.v1",
        "sealed": True,
        "disposition": "QUEUED_WITH_A_FIRE_ORDER",
        "scorer_slot_owned_by_ni1": False,
        "why_not_fired": (
            "the charter assigns every scorer launch to MAIN and explicitly says do not fire; "
            "RI1 also occupied the sole fleet n600 slot when this order was sealed"
        ),
        "candidate": file_fact(runtime / "archive.zip"),
        "runtime_dir": str(runtime),
        "runtime_digest": build["candidate_runtime_digest"],
        "ordered_jobs": [
            advisory_jobs[0],
            {
                "order": 2,
                "disposition": "QUEUED",
                "owner": "MAIN",
                "consumer_store": str(out / "harvest/per_class_r1/RESULT.json"),
                "fire_trigger": (
                    "advisory_r1 is terminal with a retained 3,662,409,600-byte raw, "
                    "its reported d_seg is harvested, and MAIN holds the sole full-n600 scorer slot"
                ),
                "method": file_fact(PER_CLASS_METHOD),
                "argv_template": [
                    str(REPO / ".venv/bin/python"),
                    "experiments/ddm_ri1_per_class_seg_retention.py",
                    "--submission-dir",
                    str(attempt_root / "advisory_r1/work"),
                    "--upstream-dir",
                    "/Volumes/APDataStore/pact/upstream_eval_mirror_20260815",
                    "--video-names-file",
                    "/Volumes/APDataStore/pact/upstream_eval_mirror_20260815/public_test_video_names.txt",
                    "--reported-d-seg",
                    "<exact advisory_r1 d_seg>",
                    "--out-dir",
                    str(out / "harvest/per_class_r1"),
                    "--batch-pairs",
                    "16",
                ],
                "method_provenance": (
                    "RI1 landed first; NI1 consumes its exact frozen-SegNet per-class method "
                    "for comparable Road/Lane/Undrivable/Movable/MyCar rows"
                ),
            },
            advisory_jobs[1],
        ],
        "harvest_requirements": [
            "recompute S from exact d_seg, d_pose, and 122250 archive bytes",
            "report Lane class 1 on its own row",
            "rederive the admissible d_seg ceiling from realized d_pose",
            "compare advisory_r1 and advisory_r2 component and raw SHA-256 identity",
            "retain every raw, scorer report, launch manifest, status receipt, and per-class argmax chunk",
        ],
        "promotion_boundary": (
            "local CPU rows are advisory only; even a sub-0.12 row requires a separately "
            "authorized exact contest-axis evaluation before pointer movement"
        ),
    }
    path = out / "SEALED_FIRE_ORDER.json"
    atomic_json(path, order)
    order["receipt"] = file_fact(path)
    return order


def verify_complete_result(result: dict[str, Any]) -> None:
    facts: list[dict[str, Any]] = []
    facts.extend(result.get("build", {}).get("candidate_files", {}).values())
    repeats = result.get("parseback_repeats", {})
    facts.extend(
        repeats.get(name, {})
        for name in ("token_payload", "token_repeat_payload", "archive", "archive_repeat")
    )
    facts.extend(
        control.get("archive", {})
        for control in result.get("mutation_controls", {}).get("controls", {}).values()
    )
    race = result.get("coder_race", {})
    facts.extend(race.get("payloads", []))
    facts.extend((race.get("source_manifest_copy", {}), race.get("copy_manifest", {})))
    facts.append(result.get("fire_order", {}).get("receipt", {}))
    for expected in facts:
        path_text = expected.get("path")
        if not path_text:
            raise NI1BuildError("complete result contains an incomplete file fact")
        actual = file_fact(Path(path_text))
        if actual["bytes"] != expected.get("bytes") or actual["sha256"] != expected.get("sha256"):
            raise NI1BuildError(f"complete result payload drifted: {path_text}")
    runtime_path = Path(result.get("runtime_dir", ""))
    if runtime_digest(runtime_path) != result.get("build", {}).get("candidate_runtime_digest"):
        raise NI1BuildError("complete NI1 runtime tree drifted")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = (args.resume_from or args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "RESULT.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("complete") is not True:
            raise NI1BuildError("existing result is not a complete resumable terminal state")
        pin_inherited_state()
        verify_complete_result(result)
        return result
    if any((out / name).exists() for name in ("runtime", "retained", "checkpoints")):
        raise NI1BuildError("incomplete output already exists; refusing to overwrite retained bytes")

    atomic_json(
        out / "RUN_STATUS.json",
        {
            "schema": "ddm_ni1_build_run_status.v1",
            "status": "RUNNING",
            "started_unix": time.time(),
            "out_dir": str(out),
        },
    )
    free = shutil.disk_usage(out).free
    storage = {
        "schema": "ddm_ni1_storage_preflight.v1",
        "path": str(out),
        "free_bytes": free,
        "required_free_bytes": MIN_FREE_BYTES,
        "status": "PASS" if free >= MIN_FREE_BYTES else "BLOCK",
        "reason": "fresh runtime + two decoded fields + coder race + mutation controls",
    }
    atomic_json(out / "STORAGE_PREFLIGHT.json", storage)
    if free < MIN_FREE_BYTES:
        raise NI1BuildError("APDataStore free space is below the NI1 fail-closed floor")

    inherited = pin_inherited_state()
    atomic_json(out / "checkpoints/01_inherited_custody_complete.json", inherited)
    archive, archive_build = build_ni1_archive()
    atomic_json(out / "checkpoints/02_archive_build_complete.json", archive_build)
    runtime = out / "runtime"
    build = patch_runtime(runtime, archive)
    atomic_json(out / "checkpoints/03_runtime_build_complete.json", build)
    retained = out / "retained"
    repeats = retain_parseback_and_repeats(runtime, retained)
    atomic_json(out / "checkpoints/04_parseback_repeats_complete.json", repeats)
    coder_race = retain_coder_race(retained)
    atomic_json(out / "checkpoints/05_coder_race_copy_complete.json", coder_race)
    controls = retain_mutation_controls(runtime, retained)
    atomic_json(out / "checkpoints/06_mutation_controls_complete.json", controls)
    fire_order = seal_main_fire_order(out, runtime, build)
    atomic_json(out / "checkpoints/07_fire_order_sealed.json", fire_order)

    archive_fact = file_fact(runtime / "archive.zip")
    result = {
        "schema": "ddm_ni1_nr1_k32_receiver_build.v1",
        "complete": True,
        "axis": "[byte-closed shipping receiver build; scorer-free]",
        "score_claim": False,
        "promotable": False,
        "created_unix": time.time(),
        "out_dir": str(out),
        "runtime_dir": str(runtime),
        "archive": archive_fact,
        "archive_build": archive_build,
        "archive_headroom_below_strict_sub012_ceiling_bytes": (
            STRICT_SUB012_CEILING_BYTES - archive_fact["bytes"]
        ),
        "archive_delta_vs_charter_conservative_projection_bytes": (
            archive_fact["bytes"] - CONSERVATIVE_PROJECTED_BYTES
        ),
        "inherited": inherited,
        "build": build,
        "parseback_repeats": repeats,
        "coder_race": coder_race,
        "mutation_controls": controls,
        "fire_order": fire_order,
        "scorer_status": "NOT_RUN_BY_BUILD",
        "next_consumer": "canonical local advisory firer on these exact runtime/archive bytes",
    }
    atomic_json(result_path, result)
    atomic_json(
        out / "RUN_STATUS.json",
        {
            "schema": "ddm_ni1_build_run_status.v1",
            "status": "COMPLETE",
            "completed_unix": time.time(),
            "result": file_fact(result_path),
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resume-from", type=Path, default=None)
    args = parser.parse_args()
    out = (args.resume_from or args.out_dir).resolve()
    try:
        result = run(args)
    except Exception as error:
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(
            out / "RUN_STATUS.json",
            {
                "schema": "ddm_ni1_build_run_status.v1",
                "status": "FAILED",
                "failed_unix": time.time(),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
