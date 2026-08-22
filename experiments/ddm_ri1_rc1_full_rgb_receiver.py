#!/usr/bin/env python3
"""Materialize the RC1 terminal program in a fresh shipping DX2 RGB receiver.

The build is scorer-free.  It pins every inherited input, copies the exact DX2
shipping runtime, installs the additive RC1 decoder, verifies two independent
full token expansions, and retains five paid-section mutation controls.  No
source custody tree is edited in place.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
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
RC1_ROOT = Path("/Volumes/APDataStore/pact/ddm_rc1_rate_crush/measurement_v4")
RC1_RESULT = RC1_ROOT / "RESULT.json"
RC1_FIRE_ORDER = RC1_ROOT / "SEALED_FIRE_ORDER.json"
RC1_PAYLOAD = RC1_ROOT / "retained/candidates/k2048_i3/receiver/tokens.rc1v"
RC1_SHADOW_ARCHIVE = RC1_ROOT / "retained/candidates/k2048_i3/shadow/archive.zip"
RC1_MODULE = REPO / "src/tac/optimization/rc1_terminal_program_vq.py"
RC1_MATERIALIZER = REPO / "experiments/ddm_rc1_rate_crush.py"
RI1_RECEIVER_SOURCE = REPO / "experiments/ddm_ri1_runtime_receiver.py"
DX2_CANDIDATE_SEAL = Path(
    "/Volumes/APDataStore/pact/ddm_dx2/r7/CANDIDATE_SEAL_dx2_fx5_cabac.json"
)

DEFAULT_OUT = Path("/Volumes/APDataStore/pact/ddm_ri1_rc1_full_rgb_receiver/build_r1")
MIN_FREE_BYTES = 8 * 1024**3

PINS = {
    RC1_RESULT: "d51e92a37bddca462a381eec66f4dbc37ff4a38f941fe2e033fc51c3c31e119c",
    RC1_FIRE_ORDER: "0d683cd3ee46dce4ed4d5b5b14d49ef608365537fcea29754619e833907eae56",
    RC1_PAYLOAD: "eab66bad9d113ed79475a810f4002ec821deb335c3e87fc1b1e90ef2b8e61164",
    RC1_SHADOW_ARCHIVE: "6756ae8f39116907828ee27b8f9686b9935eaae94c61f68c3eb02de16d45e87a",
    RC1_MODULE: "6c2ea6f324ea32b21d8cc079bb327c6af97e283cc963ec610859f1f2b0cbfbc9",
    RC1_MATERIALIZER: "19a3f378cce0eebe47d4a68c029bf6975da0c0f74902975ccdcdac68c1717c54",
    REPO / ".omx/research/ddm_rc1_rate_crush_20260822.md": "dfb239fcda4a749925326500b8821637d969e204d7eaf9191d64fc7a524e7c8d",
    REPO / ".omx/research/ddm_jx1_joint_exchange_envelope_20260822.md": "9a6a6adcd06cd4faf454c28b5f0175a691a7da07112457535b2a1521ed92f6fd",
    REPO / ".omx/research/ddm_vf1_evaluator_visible_floor_20260822.md": "f65e641edfc987a127dd2813d4136bbb01ad1c46ef4b211c80176416afcb87b4",
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
RC1_ARCHIVE_BYTES = 113_006
RC1_ARCHIVE_SHA256 = "6756ae8f39116907828ee27b8f9686b9935eaae94c61f68c3eb02de16d45e87a"
RC1_PAYLOAD_BYTES = 59_884
RC1_DECODED_TOKEN_BYTES = 117_964_800
RC1_DECODED_TOKEN_SHA256 = "2c85d29698782b2b12f75a897665f80c59a40a9549f0697e18db16feaca93168"  # gitleaks:allow -- public content digest
STRICT_SUB012_CEILING_BYTES = 137_986

SHADOW_HEADER = struct.Struct("<4sBBHHHIIII32s")
PAYLOAD_HEADER = struct.Struct("<4sBBBBHHHHIIII32s")


class RI1BuildError(RuntimeError):
    """A custody, integration, or retained-control invariant failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
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
        raise RI1BuildError(f"patch anchor {label!r} appears {count} times")
    return text.replace(anchor, replacement, 1)


def pin_inherited_state() -> dict[str, dict[str, Any]]:
    facts: dict[str, dict[str, Any]] = {}
    for path, expected in PINS.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RI1BuildError(f"inherited pin drifted: {path}: {actual} != {expected}")
        facts[str(path)] = file_fact(path)
    for relative, expected in BASE_RUNTIME_PINS.items():
        path = BASE_RUNTIME / relative
        actual = sha256_file(path)
        if actual != expected:
            raise RI1BuildError(f"DX2 shipping runtime drifted: {relative}: {actual} != {expected}")
        facts[str(path)] = file_fact(path)

    order = json.loads(RC1_FIRE_ORDER.read_text(encoding="utf-8"))
    selected = order.get("selected_rc1_payload", {})
    shadow = order.get("selected_shadow_archive", {})
    if (
        order.get("selected_candidate") != "k2048_i3"
        or selected.get("bytes") != RC1_PAYLOAD_BYTES
        or selected.get("sha256") != PINS[RC1_PAYLOAD]
        or shadow.get("bytes") != RC1_ARCHIVE_BYTES
        or shadow.get("sha256") != RC1_ARCHIVE_SHA256
    ):
        raise RI1BuildError("RC1 sealed fire order no longer selects the pinned K=2048 row")
    dx2_seal = json.loads(DX2_CANDIDATE_SEAL.read_text(encoding="utf-8"))
    measured_runtime = runtime_digest(BASE_RUNTIME)
    sealed_runtime = dict(dx2_seal.get("runtime", {}))
    sealed_runtime.pop("path", None)
    if measured_runtime != sealed_runtime:
        raise RI1BuildError("complete DX2 shipping runtime differs from its candidate seal")
    return facts


def patch_runtime(runtime: Path) -> dict[str, Any]:
    source_runtime_digest = runtime_digest(BASE_RUNTIME)
    shutil.copytree(
        BASE_RUNTIME,
        runtime,
        dirs_exist_ok=True,
        copy_function=shutil.copyfile,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "._*", ".DS_Store"),
    )
    atomic_copy(RC1_MODULE, runtime / "runtime/rc1_terminal_program_vq.py")
    atomic_copy(RI1_RECEIVER_SOURCE, runtime / "runtime/ri1_rc1_receiver.py")
    atomic_copy(RC1_SHADOW_ARCHIVE, runtime / "archive.zip")

    f26_path = runtime / "runtime/f26_inflate.py"
    f26 = f26_path.read_text(encoding="utf-8")
    f26 = replace_once(
        f26,
        "from .residual_archive import decode_production_tokens, read_residual_archive",
        "from .ri1_rc1_receiver import decode_ri1_tokens, read_ri1_archive",
        "RI1 receiver import",
    )
    f26 = replace_once(
        f26,
        '        runtime_dir / "residual_archive.py",\n        runtime_dir / "hpac_inference.py",',
        '        runtime_dir / "residual_archive.py",\n        runtime_dir / "rc1_terminal_program_vq.py",\n        runtime_dir / "ri1_rc1_receiver.py",\n        runtime_dir / "hpac_inference.py",',
        "RI1 fingerprint files",
    )
    f26 = replace_once(
        f26,
        '''    parts = read_residual_archive(archive_path)\n    if parts.schema != "fixed_boundary_int6" or parts.token_codec != "rc64":\n        raise InflationError("archive does not use the fixed F26 residual schema")''',
        '''    parts, ri1_model, ri1_decoded_digest, ri1_archive_report = read_ri1_archive(archive_path)\n    if parts.schema != "fixed_boundary_int6" or parts.token_codec != "rc1v":\n        raise InflationError("archive does not use the RI1 fixed residual + RC1 token schema")''',
        "RI1 archive parser",
    )
    f26 = replace_once(
        f26,
        '''    token_decoder = os.environ.get("F26_TOKEN_DECODER", "python").strip()\n    if token_decoder not in {"python", "native-hpac"}:\n        raise InflationError("F26_TOKEN_DECODER must be 'python' or 'native-hpac'")\n    if token_decoder != "python":\n        raise InflationError(\n            "this generation wires the ddm_rr2 free probability corrector into the "\n            "python token decoder only; the native-hpac path is unpatched and would "\n            "decode a different field, so it is refused rather than trusted"\n        )\n    if pair_count != int(renderer.N) and token_decoder != "native-hpac":\n        raise InflationError("advisory prefix inflation requires the resumable native token path")''',
        '''    token_decoder = os.environ.get("F26_TOKEN_DECODER", "ri1-rc1").strip()\n    if token_decoder != "ri1-rc1":\n        raise InflationError("RI1 requires F26_TOKEN_DECODER=ri1-rc1")\n    if pair_count != int(renderer.N):\n        raise InflationError("RI1 accepts only the complete n600 receiver field")''',
        "RI1 decoder selection",
    )
    f26 = replace_once(
        f26,
        '''    if loaded is None:\n        if token_decoder == "native-hpac":\n            if checkpoint_dir is None:\n                raise InflationError("native token decode requires checkpoint_dir")\n            from .f26_hpac_native import decode_native_tokens\n\n            native_progress = checkpoint_dir.resolve() / "native_hpac_progress"\n            tokens, token_report = decode_native_tokens(\n                parts,\n                renderer,\n                renderer_dir,\n                device,\n                frame_limit=pair_count,\n                output_path=native_progress / "tokens_partial.u8",\n                checkpoint_dir=native_progress / "checkpoints",\n            )\n        else:\n            tokens, token_report = decode_production_tokens(parts, renderer, renderer_dir, device)''',
        '''    if loaded is None:\n        tokens, token_report = decode_ri1_tokens(ri1_model, ri1_decoded_digest)''',
        "RI1 token decode",
    )
    f26 = replace_once(
        f26,
        '        "residual_schema": parts.schema,\n        "compensation": compensation_report,',
        '        "residual_schema": parts.schema,\n        "ri1_archive": ri1_archive_report,\n        "compensation": compensation_report,',
        "RI1 report",
    )
    atomic_bytes(f26_path, f26.encode("utf-8"))

    inflate_path = runtime / "inflate.py"
    inflate = inflate_path.read_text(encoding="utf-8")
    inflate = replace_once(
        inflate,
        'ARCHIVE_SHA256 = "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674"',
        f'ARCHIVE_SHA256 = "{RC1_ARCHIVE_SHA256}"',
        "archive SHA",
    )
    inflate = replace_once(
        inflate,
        "ARCHIVE_BYTES = 180_368",
        f"ARCHIVE_BYTES = {RC1_ARCHIVE_BYTES:_}",
        "archive bytes",
    )
    atomic_bytes(inflate_path, inflate.encode("utf-8"))

    shell_path = runtime / "inflate.sh"
    shell = shell_path.read_text(encoding="utf-8")
    shell = replace_once(
        shell,
        'export F26_TOKEN_DECODER="${F26_TOKEN_DECODER:-python}"',
        '''if [[ -n "${F26_TOKEN_DECODER:-}" && "$F26_TOKEN_DECODER" != "ri1-rc1" ]]; then\n  echo "RI1 refuses a non-RC1 token decoder" >&2\n  exit 69\nfi\nexport F26_TOKEN_DECODER="ri1-rc1"''',
        "RI1 shell decoder",
    )
    atomic_bytes(shell_path, shell.encode("utf-8"))
    shell_path.chmod(shell_path.stat().st_mode | 0o111)

    if (
        runtime.joinpath("archive.zip").stat().st_size != RC1_ARCHIVE_BYTES
        or sha256_file(runtime / "archive.zip") != RC1_ARCHIVE_SHA256
    ):
        raise RI1BuildError("staged RI1 archive differs from the retained RC1 shadow archive")
    return {
        "schema": "ddm_ri1_runtime_build.v1",
        "base_runtime": str(BASE_RUNTIME),
        "base_runtime_pins": BASE_RUNTIME_PINS,
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
                "runtime/rc1_terminal_program_vq.py",
                "runtime/ri1_rc1_receiver.py",
                "cpr1/inflate.py",
            )
        },
        "integration_boundary": (
            "RC1 replaces only the terminal token decoder; semantic weights, carrier, "
            "frame-0 selector, compensation, SemanticTokenRenderer, and render_video "
            "remain the copied DX2 shipping implementation"
        ),
    }


def load_staged_receiver(runtime: Path):
    runtime_text = str(runtime)
    if runtime_text not in sys.path:
        sys.path.insert(0, runtime_text)
    for name in tuple(sys.modules):
        if name == "runtime" or name.startswith("runtime."):
            del sys.modules[name]
    return importlib.import_module("runtime.ri1_rc1_receiver")


def write_token_payload(path: Path, tokens: Any) -> dict[str, Any]:
    array = tokens.detach().cpu().contiguous().numpy()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as stream:
        array.tofile(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fact = file_fact(path)
    if fact["bytes"] != RC1_DECODED_TOKEN_BYTES or fact["sha256"] != RC1_DECODED_TOKEN_SHA256:
        raise RI1BuildError("retained decoded token payload differs from the RC1 digest")
    return fact


def retain_parseback_and_repeats(runtime: Path, retained: Path) -> dict[str, Any]:
    receiver = load_staged_receiver(runtime)
    parts, model, digest, parse_report = receiver.read_ri1_archive(runtime / "archive.zip")
    tokens, decode_report = receiver.decode_ri1_tokens(model, digest)
    first = write_token_payload(retained / "receiver/tokens_full.u8", tokens)
    del tokens
    repeat_tokens, repeat_report = receiver.decode_ri1_tokens(model, digest)
    repeat = write_token_payload(retained / "receiver/tokens_full.repeat.u8", repeat_tokens)
    del repeat_tokens
    if first["sha256"] != repeat["sha256"] or decode_report != repeat_report:
        raise RI1BuildError("independent RI1 token decode repeat differs")
    archive_repeat = retained / "archive.repeat.zip"
    atomic_copy(runtime / "archive.zip", archive_repeat)
    if sha256_file(archive_repeat) != RC1_ARCHIVE_SHA256:
        raise RI1BuildError("exact archive repeat differs")
    return {
        "schema": "ddm_ri1_parseback_repeat.v1",
        "parse_report": parse_report,
        "decode_report": decode_report,
        "token_payload": first,
        "token_repeat_payload": repeat,
        "archive": file_fact(runtime / "archive.zip"),
        "archive_repeat": file_fact(archive_repeat),
        "parts": {
            "semantic_sha256": hashlib.sha256(parts.semantic_blob).hexdigest(),
            "carrier_sha256": hashlib.sha256(parts.carrier_blob).hexdigest(),
            "residual_sha256": hashlib.sha256(parts.residual_payload).hexdigest(),
            "rc1_payload_sha256": hashlib.sha256(parts.token_stream).hexdigest(),
        },
        "repeat_identity": True,
    }


def archive_with_member(member: bytes) -> bytes:
    import io

    sink = io.BytesIO()
    info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o100644 << 16
    with zipfile.ZipFile(sink, "w") as archive:
        archive.writestr(info, member)
    return sink.getvalue()


def retain_mutation_controls(runtime: Path, retained: Path) -> dict[str, Any]:
    receiver = load_staged_receiver(runtime)
    with zipfile.ZipFile(runtime / "archive.zip") as archive:
        outer = archive.read("p")
    shadow = SHADOW_HEADER.unpack_from(outer)
    semantic_bytes, carrier_bytes, residual_bytes = shadow[3:6]
    rc1_bytes = shadow[6]
    semantic_offset = SHADOW_HEADER.size
    carrier_offset = semantic_offset + semantic_bytes
    residual_offset = carrier_offset + carrier_bytes
    payload_offset = residual_offset + residual_bytes
    payload = outer[payload_offset : payload_offset + rc1_bytes]
    payload_fields = PAYLOAD_HEADER.unpack_from(payload)
    assignment_bytes = payload_fields[9]
    assignment_offset = payload_offset + PAYLOAD_HEADER.size
    codebook_offset = assignment_offset + assignment_bytes
    offsets = {
        "semantic": semantic_offset,
        "carrier": carrier_offset,
        "residual": residual_offset,
        "assignment": assignment_offset,
        "codebook": codebook_offset,
    }
    controls: dict[str, Any] = {}
    for name, offset in offsets.items():
        mutated = bytearray(outer)
        mutated[offset] ^= 0x01
        archive_payload = archive_with_member(bytes(mutated))
        path = retained / "mutation_controls" / f"{name}_bitflip.archive.zip"
        atomic_bytes(path, archive_payload)
        refusal = None
        try:
            receiver.read_ri1_archive(path)
        except Exception as error:  # The exact refusal type/message is retained.
            refusal = f"{type(error).__name__}: {error}"
        if refusal is None:
            raise RI1BuildError(f"{name} paid-section mutation was accepted")
        controls[name] = {
            "mutated_offset_in_member": offset,
            "mutation": "xor 0x01",
            "archive": file_fact(path),
            "receiver_disposition": "REFUSED",
            "receiver_error": refusal,
        }
    return {
        "schema": "ddm_ri1_paid_section_mutation_controls.v1",
        "all_paid_sections_refused": True,
        "controls": controls,
    }


def verify_complete_result(result: dict[str, Any]) -> None:
    """Revalidate every retained payload before accepting a resume terminal."""
    fact_groups = [
        result.get("build", {}).get("candidate_files", {}).values(),
        (
            result.get("parseback_repeats", {}).get("token_payload", {}),
            result.get("parseback_repeats", {}).get("token_repeat_payload", {}),
            result.get("parseback_repeats", {}).get("archive", {}),
            result.get("parseback_repeats", {}).get("archive_repeat", {}),
        ),
        (
            control.get("archive", {})
            for control in result.get("mutation_controls", {})
            .get("controls", {})
            .values()
        ),
    ]
    checked = 0
    for facts in fact_groups:
        for expected in facts:
            path_text = expected.get("path")
            if not path_text:
                raise RI1BuildError("complete result contains an incomplete payload fact")
            path = Path(path_text)
            actual = file_fact(path)
            if (
                actual["bytes"] != expected.get("bytes")
                or actual["sha256"] != expected.get("sha256")
            ):
                raise RI1BuildError(f"complete result payload drifted: {path}")
            checked += 1
    if checked != 17:
        raise RI1BuildError(f"complete result custody census is {checked}, expected 17")
    runtime_path = Path(result.get("runtime_dir", ""))
    expected_digest = result.get("build", {}).get("candidate_runtime_digest")
    if runtime_digest(runtime_path) != expected_digest:
        raise RI1BuildError("complete RI1 shipping runtime tree drifted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resume-from", type=Path, default=None)
    args = parser.parse_args()
    out = (args.resume_from or args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "RESULT.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("complete") is not True:
            raise RI1BuildError("existing result is not a complete resumable terminal state")
        pin_inherited_state()
        verify_complete_result(result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    free = shutil.disk_usage(out).free
    storage = {
        "schema": "ddm_ri1_storage_preflight.v1",
        "path": str(out),
        "free_bytes": free,
        "required_free_bytes": MIN_FREE_BYTES,
        "status": "PASS" if free >= MIN_FREE_BYTES else "BLOCK",
        "reason": "runtime + exact token repeats + controls + one retained full-RGB advisory raw",
    }
    atomic_json(out / "STORAGE_PREFLIGHT.json", storage)
    if free < MIN_FREE_BYTES:
        raise RI1BuildError("APDataStore free space is below the RI1 fail-closed floor")

    inherited = pin_inherited_state()
    atomic_json(
        out / "checkpoints/01_inherited_custody_complete.json",
        {"complete": True, "inherited": inherited},
    )
    runtime = out / "runtime"
    build = patch_runtime(runtime)
    atomic_json(out / "checkpoints/02_runtime_build_complete.json", build)
    retained = out / "retained"
    repeats = retain_parseback_and_repeats(runtime, retained)
    atomic_json(out / "checkpoints/03_parseback_repeats_complete.json", repeats)
    controls = retain_mutation_controls(runtime, retained)
    atomic_json(out / "checkpoints/04_mutation_controls_complete.json", controls)

    result = {
        "schema": "ddm_ri1_rc1_full_rgb_receiver_build.v1",
        "complete": True,
        "axis": "[byte-closed receiver build; scorer-free]",
        "score_claim": False,
        "promotable": False,
        "created_unix": time.time(),
        "out_dir": str(out),
        "runtime_dir": str(runtime),
        "archive": file_fact(runtime / "archive.zip"),
        "archive_headroom_below_strict_sub012_ceiling_bytes": (
            STRICT_SUB012_CEILING_BYTES - RC1_ARCHIVE_BYTES
        ),
        "inherited": inherited,
        "build": build,
        "parseback_repeats": repeats,
        "mutation_controls": controls,
        "scorer_status": "NOT_RUN_BY_BUILD",
        "next_consumer": "canonical local advisory firer on the exact runtime/archive bytes",
    }
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
