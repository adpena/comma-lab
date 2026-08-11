#!/usr/bin/env python3
"""Build the scorer-free direct PGQ1 -> LC2 receiver rung.

PGQ1 stores a compact six-dimensional output gauge, not pixels.  This runner
keeps LC2's counted spatial basis and fits a counted fixed-point map from the
literal decoded PGQ1 pose object to LC2 coefficient codes.  It deliberately omits PZ3's
exact residual, so the resulting receiver changes frames and can expose the
surrogate-to-realization gap through a later real scorer pass.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import io
import json
import os
import platform
import struct
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "experiments"
PZ4_RUNTIME_SOURCE = EXPERIMENTS / "ddm_pz4r_runtime"
PZ3_RUNTIME_SOURCE = REPO / "src/tac/pr130_runtime/fx1_runtime_tree"
LC2_RUNTIME_SOURCE = Path("/Volumes/VertigoDataTier/pact/ddm_lc2_20260810/submission")
LC2_ARCHIVE = LC2_RUNTIME_SOURCE / "archive.zip"
LC2_CARRIER = Path("/Volumes/VertigoDataTier/pact/ddm_lc2_20260810/retained/inputs/carrier.raw")
PGQ1 = Path("/Volumes/VertigoDataTier/pact/ddm_pz4p_20260811/preproof_v3/candidates/r6_b12_global/gauge.pgq1")
PGQ1_DECODED = PGQ1.with_name("decoded_outputs.float32.npy")
DEFAULT_OUTPUT = Path("/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/direct_v6")

EXPECTED_LC2_ARCHIVE_SHA256 = "f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45"
EXPECTED_LC2_ARCHIVE_BYTES = 187_226
EXPECTED_LC2_CARRIER_SHA256 = "a05d0985ca5a8d5110bd5bf5be39f238c6f89640b8a8bb888a3e1269bdf636e4"
EXPECTED_PGQ1_SHA256 = "b903c7f0e6100e3602e414fbc261725aa5026fa6e1c6af8fba104ded867b9cac"
EXPECTED_PGQ1_BYTES = 5_588
EXPECTED_PGQ1_DECODED_SHA256 = "fe460e79d6d1a4b95e1e9b1171c3fbacee36110bc4d9871d05308997b2740830"
EXPECTED_RATE_ENVELOPE_BYTES = 168_005
EXPECTED_RATE_ENVELOPE_SHA256 = "66d142c7db35f3762be4d810b7549bd94bebe77120fa5dd53173937d5c6d2620"
N = 600
DIM = 12
BASIS_SHAPE = (DIM, 3, 24, 32)
SEED = 20_260_811
AXIS = "[macOS-CPU scorer-free receiver build]"
RESUME_SCHEMA = "ddm_pz4r_resume.v2"
MATERIALIZE_SCHEMA = "ddm_pz4r_materialize.v2"
VERIFY_SCHEMA = "ddm_pz4r_verify.v2"
RESULT_SCHEMA = "ddm_pz4r_pgq1_receiver.v2"
RUNTIME_MANIFEST_SCHEMA = "comma_lab.pz4r_runtime_dependencies.v1"
RUNTIME_PYTHON_FILES = (
    "carrier_codec.py",
    "hpac_integer.py",
    "hpac_integer_sparse.py",
    "inflate.py",
    "integer_model_io.py",
    "pose_gauge_receiver.py",
    "pose_target_receiver.py",
    "receiver.py",
)
RUNTIME_SOURCE_FILES = (*RUNTIME_PYTHON_FILES, "inflate.sh")


class PZ4RBuildError(RuntimeError):
    """Raised when a custody, receiver, or resumability invariant fails."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_bytes(payload)
    os.replace(pending, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(pending, path)


def file_record(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def validate_file_record(
    record: dict[str, object],
    *,
    label: str,
    expected_path: Path | None = None,
) -> Path:
    """Re-hash a retained artifact instead of trusting checkpoint booleans."""

    if not isinstance(record, dict) or not {"path", "bytes", "sha256"} <= record.keys():
        raise PZ4RBuildError(f"{label}: malformed file record")
    path = Path(str(record["path"])).resolve()
    if expected_path is not None and path != expected_path.resolve():
        raise PZ4RBuildError(f"{label}: recorded path differs from canonical path")
    if not path.is_file():
        raise PZ4RBuildError(f"{label}: retained file is absent: {path}")
    if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
        raise PZ4RBuildError(f"{label}: retained file bytes drifted: {path}")
    return path


def source_binding_records() -> dict[str, dict[str, object]]:
    """Return every source input whose bytes can alter a PZ4R build."""

    paths = {
        "builder": Path(__file__),
        "pose_gauge_receiver": PZ4_RUNTIME_SOURCE / "pose_gauge_receiver.py",
        "fixed_point_primitives": PZ3_RUNTIME_SOURCE / "pose_target_receiver.py",
        "pz3_builder_primitives": EXPERIMENTS / "ddm_pz3_pose_receiver_realization.py",
        "pgq1_codec_and_source": EXPERIMENTS / "ddm_pz4p_pose_gauge_preproof.py",
        "lc2_archive_builder": EXPERIMENTS / "ddm_ps135_pose_resolve.py",
    }
    paths.update(
        {
            f"lc2_runtime_{name}": LC2_RUNTIME_SOURCE / name
            for name in (
                "carrier_codec.py",
                "hpac_integer.py",
                "hpac_integer_sparse.py",
                "inflate.py",
                "inflate.sh",
                "integer_model_io.py",
                "receiver.py",
                "runtime-dependencies.json",
            )
        }
    )
    return {name: file_record(path) for name, path in paths.items()}


def validate_source_bindings(stored: dict[str, object]) -> None:
    current = source_binding_records()
    if set(stored) != set(current):
        raise PZ4RBuildError("resume source-binding denominator differs")
    for name, current_record in current.items():
        if stored[name] != current_record:
            raise PZ4RBuildError(f"resume source binding drifted: {name}")


def load_recorded_json(
    path: Path,
    expected_sha256: object,
    *,
    schema: str,
    label: str,
) -> dict[str, object]:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise PZ4RBuildError(f"{label}: receipt is absent or hash-drifted")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise PZ4RBuildError(f"{label}: unsupported receipt schema")
    return value


def require_pin(path: Path, *, size: int | None, digest: str) -> dict[str, object]:
    record = file_record(path)
    if (size is not None and record["bytes"] != size) or record["sha256"] != digest:
        raise PZ4RBuildError(f"pinned artifact drifted: {path}")
    return record


def validate_output(path: Path) -> Path:
    root = Path("/Volumes/VertigoDataTier/pact").resolve()
    resolved = path.resolve()
    if resolved == root or root not in resolved.parents:
        raise PZ4RBuildError("output must be an arm-specific VertigoDataTier path")
    if Path("/tmp") in resolved.parents or resolved == Path("/tmp"):
        raise PZ4RBuildError("persisted evidence may not use /tmp")
    return resolved


def storage_preflight(path: Path, required_free_bytes: int) -> dict[str, object]:
    path.mkdir(parents=True, exist_ok=True)
    stat = os.statvfs(path)
    free_bytes = int(stat.f_bavail * stat.f_frsize)
    result = {
        "path": str(path),
        "required_free_bytes": required_free_bytes,
        "free_bytes": free_bytes,
        "passed": free_bytes >= required_free_bytes,
        "measured_at_utc": utc_now(),
    }
    if not result["passed"]:
        raise PZ4RBuildError(f"storage preflight failed: {free_bytes} < {required_free_bytes}")
    return result


def import_surfaces() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    for path in (EXPERIMENTS, PZ3_RUNTIME_SOURCE, PZ4_RUNTIME_SOURCE):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    pz3 = importlib.import_module("ddm_pz3_pose_receiver_realization")
    pz4p = importlib.import_module("ddm_pz4p_pose_gauge_preproof")
    ps135 = importlib.import_module("ddm_ps135_pose_resolve")
    codec, wire_receiver, inflate = ps135.import_runtime_modules()
    predictor_receiver = importlib.import_module("pose_target_receiver")
    gauge_receiver = importlib.import_module("pose_gauge_receiver")
    return pz3, pz4p, ps135, codec, wire_receiver, inflate, predictor_receiver, gauge_receiver


def build_archive(
    carrier: bytes,
    *,
    ps135: Any,
    source: Any,
    semantic_stream: bytes,
    hpac_stream: bytes,
    wire_receiver: Any,
) -> tuple[bytes, bytes, bytes, bytes]:
    carrier_stream = ps135.brotli_compress(carrier, quality=9)
    model_pack = ps135.split_pack((semantic_stream, carrier_stream, hpac_stream))
    member = wire_receiver.pack_payload(
        model_pack,
        source.tokens,
        token_codec="ans",
        model_codec="split_brotli_cx2",
    )
    archive = ps135.deterministic_stored_zip(member)
    return carrier_stream, model_pack, member, archive


def parse_wire_archive(
    archive_blob: bytes,
    *,
    expected_carrier: bytes,
    source: Any,
    wire_receiver: Any,
) -> dict[str, object]:
    with zipfile.ZipFile(io.BytesIO(archive_blob), "r") as archive:
        entries = archive.infolist()
        if len(entries) != 1 or entries[0].filename != "p":
            raise PZ4RBuildError("candidate ZIP grammar differs from LC2")
        if entries[0].compress_type != zipfile.ZIP_STORED or archive.testzip() is not None:
            raise PZ4RBuildError("candidate ZIP is not a valid stored archive")
        member = archive.read("p")
    parts = wire_receiver.split_payload(member)
    if parts.model_codec != "split_brotli_cx2" or parts.token_codec != "ans":
        raise PZ4RBuildError("candidate wire selectors differ from LC2")
    decoded = wire_receiver.decode_models(parts.models, model_codec=parts.model_codec)
    models_raw, temporal = wire_receiver.split_optional_temporal_reversion(decoded.raw)
    if temporal is None or temporal.packed != source.temporal_packed:
        raise PZ4RBuildError("candidate temporal payload changed")
    if parts.tokens != source.tokens:
        raise PZ4RBuildError("candidate token payload changed")
    semantic_bytes, carrier_bytes = struct.unpack_from("<II", models_raw)
    semantic_end = 8 + semantic_bytes
    carrier_end = semantic_end + carrier_bytes
    if carrier_end > len(models_raw):
        raise PZ4RBuildError("candidate model lengths exceed the decoded bundle")
    semantic = models_raw[8:semantic_end]
    carrier = models_raw[semantic_end:carrier_end]
    hpac = models_raw[carrier_end:]
    if semantic != source.semantic or hpac != source.hpac_base:
        raise PZ4RBuildError("candidate changed semantic or HPAC bytes")
    if carrier != expected_carrier:
        raise PZ4RBuildError("candidate carrier parse-back differs")
    return {
        "member_sha256": sha256_bytes(member),
        "semantic_sha256": sha256_bytes(semantic),
        "hpac_sha256": sha256_bytes(hpac),
        "tokens_sha256": sha256_bytes(parts.tokens),
        "temporal_sha256": sha256_bytes(temporal.packed),
        "carrier_sha256": sha256_bytes(carrier),
    }


def coefficient_metrics(predicted: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    # Coefficient *storage* is signed 12-bit, but the rendered amplitude is not
    # circular.  In particular, 2047 and -2048 are 4095 units apart to LC2's
    # renderer, not one unit apart.  Selection therefore uses ordinary error.
    signed = predicted.astype(np.int64) - reference.astype(np.int64)
    centered = reference.astype(np.float64) - reference.mean(axis=0, dtype=np.float64)
    denominator = float(np.sum(centered * centered, dtype=np.float64))
    squared = signed.astype(np.float64) ** 2
    return {
        "coefficient_code_r2_variance_weighted": 1.0 - float(squared.sum()) / denominator,
        "coefficient_code_mae": float(np.mean(np.abs(signed), dtype=np.float64)),
        "coefficient_code_rmse": float(np.sqrt(np.mean(squared, dtype=np.float64))),
        "coefficient_code_exact": int(np.count_nonzero(signed == 0)),
        "coefficient_code_total": int(signed.size),
        "coefficient_error_geometry": "ordinary_signed_difference_non_circular",
        "endpoint_wrap_crossings": int(np.count_nonzero(np.abs(signed) > 2048)),
        "coefficient_parseback_exact_to_prediction": True,
        "coefficient_exact_to_lc2": bool(np.array_equal(predicted, reference)),
    }


def patch_inflate_source(source: str) -> str:
    import_anchor = (
        "from carrier_codec import MAGIC as COMPACT_CARRIER_MAGIC\nfrom carrier_codec import decode_compact_carrier\n"
    )
    import_replacement = import_anchor + (
        "from pose_gauge_receiver import MAGIC as POSE_GAUGE_CARRIER_MAGIC\n"
        "from pose_gauge_receiver import decode_pose_gauge_carrier\n"
    )
    branch_anchor = "    if carrier_blob[:4] == COMPACT_CARRIER_MAGIC:\n"
    branch_replacement = (
        "    if carrier_blob[:4] == POSE_GAUGE_CARRIER_MAGIC:\n"
        "        basis_array, coefficient_array = decode_pose_gauge_carrier(carrier_blob)\n"
        "        basis = torch.from_numpy(basis_array)\n"
        "        coeff = torch.from_numpy(coefficient_array)\n"
        "        return semantic, basis, coeff\n"
        "    elif carrier_blob[:4] == COMPACT_CARRIER_MAGIC:\n"
    )
    if source.count(import_anchor) != 1 or source.count(branch_anchor) != 1:
        raise PZ4RBuildError("LC2 inflate patch anchors changed")
    patched = source.replace(import_anchor, import_replacement, 1)
    patched = patched.replace(branch_anchor, branch_replacement, 1)
    compile(patched, "inflate.py", "exec")
    return patched


def runtime_import_closure(runtime_dir: Path) -> dict[str, list[str]]:
    """Enumerate third-party top-level imports for the shipped Python modules."""

    local_modules = {Path(name).stem for name in RUNTIME_PYTHON_FILES}
    closure: dict[str, list[str]] = {}
    for name in RUNTIME_PYTHON_FILES:
        tree = ast.parse((runtime_dir / name).read_text(encoding="utf-8"), filename=name)
        roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".", 1)[0])
        for root in sorted(roots - local_modules - set(sys.stdlib_module_names)):
            closure.setdefault(root, []).append(name)
    return {name: sorted(importers) for name, importers in sorted(closure.items())}


def materialize_runtime_manifest(runtime_dir: Path) -> dict[str, object]:
    """Regenerate dependency custody for the actual patched PZ4R tree."""

    base = json.loads((LC2_RUNTIME_SOURCE / "runtime-dependencies.json").read_text(encoding="utf-8"))
    imports = runtime_import_closure(runtime_dir)
    dependencies = {row["name"]: row for row in base["dependencies"]}
    if set(imports) != set(dependencies):
        raise PZ4RBuildError(f"runtime dependency closure changed: {sorted(imports)} != {sorted(dependencies)}")
    for name, row in dependencies.items():
        row["imported_by"] = imports[name]
    source_hashes = {name: sha256_file(runtime_dir / name) for name in RUNTIME_SOURCE_FILES}
    closure_provenance = dict(base["closure_provenance"])
    closure_provenance.pop("denominator", None)
    dependency_precedent = closure_provenance.pop("receipt", None)
    closure_provenance.update(
        {
            "enumerated_from": (
                "AST top-level imports of all 8 shipped Python modules; inflate.sh is the separately counted entrypoint"
            ),
            "python_module_denominator": len(RUNTIME_PYTHON_FILES),
            "entrypoint_denominator": 1,
            "runtime_source_file_denominator": len(RUNTIME_SOURCE_FILES),
            "third_party_packages": len(imports),
            "pz4r_closure_status": "scorer_free_parseback_measured; Linux retest queued",
            "linux_retest_status": ("QUEUED: this PZ4R 8-module plus entrypoint closure has not run on Linux"),
            "dependency_precedent_receipt": dependency_precedent,
            "receipt": "../PZ4R_RESULT.json",
        }
    )
    base.update(
        {
            "schema": RUNTIME_MANIFEST_SCHEMA,
            "dependencies": [dependencies[row["name"]] for row in base["dependencies"]],
            "closure_provenance": closure_provenance,
            "receiver": ("inflate.py+receiver.py+pose_gauge_receiver.py+pose_target_receiver.py"),
            "borrowed_substrate_accounting": {
                **base["borrowed_substrate_accounting"],
                "ddm_pz4r_modified_files": {
                    "inflate.py": (
                        "adds strict PZ4R dispatch and returns its decoded LC2 basis and "
                        "residual-free predicted coefficients"
                    ),
                    "pose_gauge_receiver.py": ("new strict PGQ1 parser and checked fixed-point PZ4R receiver"),
                    "pose_target_receiver.py": ("verbatim PZ3 fixed-point predictor and LC2 basis primitives"),
                },
            },
            "source": {
                **base["source"],
                "copied_files_semantics": (
                    "exact current shipping-tree custody hashes after the PZ4R patch; "
                    "archive.zip is counted separately and is not hidden in source"
                ),
                "copied_files": source_hashes,
            },
            "counted_archive": file_record(runtime_dir / "archive.zip"),
        }
    )
    manifest_path = runtime_dir / "runtime-dependencies.json"
    atomic_json(manifest_path, base)
    return base


def validate_runtime_manifest(runtime_dir: Path) -> dict[str, object]:
    manifest_path = runtime_dir / "runtime-dependencies.json"
    if not manifest_path.is_file():
        raise PZ4RBuildError("runtime dependency manifest is absent")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != RUNTIME_MANIFEST_SCHEMA:
        raise PZ4RBuildError("runtime dependency manifest schema differs")
    expected_hashes = {name: sha256_file(runtime_dir / name) for name in RUNTIME_SOURCE_FILES}
    if manifest.get("source", {}).get("copied_files") != expected_hashes:
        raise PZ4RBuildError("runtime dependency manifest source hashes are stale")
    closure = manifest.get("closure_provenance", {})
    if (
        "denominator" in closure
        or closure.get("python_module_denominator") != len(RUNTIME_PYTHON_FILES)
        or closure.get("entrypoint_denominator") != 1
        or closure.get("runtime_source_file_denominator") != len(RUNTIME_SOURCE_FILES)
    ):
        raise PZ4RBuildError("runtime dependency manifest denominator differs")
    imported_by = {row["name"]: row.get("imported_by") for row in manifest.get("dependencies", [])}
    if imported_by != runtime_import_closure(runtime_dir):
        raise PZ4RBuildError("runtime dependency manifest import closure is stale")
    validate_file_record(
        manifest.get("counted_archive", {}),
        label="runtime counted archive",
        expected_path=runtime_dir / "archive.zip",
    )
    return manifest


def materialize_runtime(output: Path, selected_archive: Path) -> dict[str, object]:
    runtime_dir = output / "submission"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    source_records: dict[str, object] = {}
    for name in (
        "carrier_codec.py",
        "hpac_integer.py",
        "hpac_integer_sparse.py",
        "inflate.sh",
        "integer_model_io.py",
        "receiver.py",
    ):
        source = LC2_RUNTIME_SOURCE / name
        destination = runtime_dir / name
        atomic_bytes(destination, source.read_bytes())
        source_records[name] = file_record(destination)
    patched = patch_inflate_source((LC2_RUNTIME_SOURCE / "inflate.py").read_text(encoding="utf-8"))
    atomic_bytes(runtime_dir / "inflate.py", patched.encode("utf-8"))
    for name, source in (
        ("pose_target_receiver.py", PZ3_RUNTIME_SOURCE / "pose_target_receiver.py"),
        ("pose_gauge_receiver.py", PZ4_RUNTIME_SOURCE / "pose_gauge_receiver.py"),
    ):
        atomic_bytes(runtime_dir / name, source.read_bytes())
    atomic_bytes(runtime_dir / "archive.zip", selected_archive.read_bytes())
    materialize_runtime_manifest(runtime_dir)
    for name in (
        "inflate.py",
        "pose_target_receiver.py",
        "pose_gauge_receiver.py",
        "archive.zip",
        "runtime-dependencies.json",
    ):
        source_records[name] = file_record(runtime_dir / name)
    validate_runtime_manifest(runtime_dir)
    return {"path": str(runtime_dir), "files": source_records}


def public_runtime_parse(
    runtime: Path,
    *,
    ps135: Any,
) -> dict[str, object]:
    script = r"""import hashlib, json, os, sys, zipfile
from pathlib import Path
import receiver, inflate
root = Path(sys.argv[1])
with zipfile.ZipFile(root / "archive.zip", "r") as zf:
    member = zf.read("p")
parts = receiver.split_payload(member)
decoded = receiver.decode_models(parts.models, model_codec=parts.model_codec)
models, _ = receiver.split_optional_temporal_reversion(decoded.raw)
semantic_bytes = int.from_bytes(models[:4], "little")
carrier_bytes = int.from_bytes(models[4:8], "little")
end = 8 + semantic_bytes + carrier_bytes
_, basis, coeff = inflate.unpack_semantic_pose(models[:end])
def h(t): return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
print(json.dumps({"basis_shape": list(basis.shape), "coeff_shape": list(coeff.shape),
                  "basis_sha256": h(basis), "coeff_sha256": h(coeff)}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(runtime)
    environment["PR130_BROTLI_CLI"] = str(ps135.BROTLI_CLI)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script, str(runtime)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def validate_materialized_receipt(materialized: dict[str, object], output: Path) -> dict[str, object]:
    retained = output / "retained"
    shared_names = {
        "base_carrier": "base_carrier.cpr1",
        "basis_component": "basis_component.bin",
        "pgq1": "gauge.pgq1",
        "source_arrays": "source_arrays.npz",
    }
    shared = materialized.get("shared")
    if not isinstance(shared, dict) or set(shared) != set(shared_names):
        raise PZ4RBuildError("materialize receipt shared denominator differs")
    for name, filename in shared_names.items():
        validate_file_record(
            shared[name],
            label=f"materialize shared {name}",
            expected_path=retained / "shared" / filename,
        )

    candidates = materialized.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 12:
        raise PZ4RBuildError("materialize candidate denominator differs from sealed sweep")
    candidate_files = {
        "carrier": "carrier.pz4r",
        "carrier_repeat": "carrier.repeat.pz4r",
        "carrier_q9": "carrier.q9.br",
        "carrier_q9_repeat": "carrier.repeat.q9.br",
        "model_pack": "models.split",
        "model_pack_repeat": "models.repeat.split",
        "member": "p",
        "member_repeat": "p.repeat",
        "archive": "archive.zip",
        "archive_repeat": "archive.repeat.zip",
        "parsed_arrays": "parsed_arrays.npz",
    }
    with np.load(Path(shared["source_arrays"]["path"])) as arrays:
        reference = arrays["absolute_coefficients"].copy()
    seen: set[str] = set()
    for row in candidates:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise PZ4RBuildError("materialize candidate row is malformed")
        name = row["name"]
        if name in seen:
            raise PZ4RBuildError(f"duplicate materialize candidate: {name}")
        seen.add(name)
        records = row.get("records")
        if not isinstance(records, dict) or set(records) != set(candidate_files):
            raise PZ4RBuildError(f"{name}: retained payload denominator differs")
        for key, filename in candidate_files.items():
            validate_file_record(
                records[key],
                label=f"{name} {key}",
                expected_path=retained / "candidates" / name / filename,
            )
        for base, repeat in (
            ("carrier", "carrier_repeat"),
            ("carrier_q9", "carrier_q9_repeat"),
            ("model_pack", "model_pack_repeat"),
            ("member", "member_repeat"),
            ("archive", "archive_repeat"),
        ):
            if (
                records[base]["bytes"] != records[repeat]["bytes"]
                or records[base]["sha256"] != records[repeat]["sha256"]
            ):
                raise PZ4RBuildError(f"{name}: {base} repeat differs")
        with np.load(Path(records["parsed_arrays"]["path"])) as arrays:
            metrics = coefficient_metrics(arrays["predicted_codes"], reference)
        if row.get("metrics") != metrics:
            raise PZ4RBuildError(f"{name}: coefficient metrics are stale")

    eligible = [row for row in candidates if row["records"]["archive"]["bytes"] < EXPECTED_LC2_ARCHIVE_BYTES]
    if not eligible:
        raise PZ4RBuildError("resume has no rate-eligible candidate")
    selected = max(
        eligible,
        key=lambda row: (
            row["metrics"]["coefficient_code_r2_variance_weighted"],
            -row["records"]["archive"]["bytes"],
        ),
    )
    if materialized.get("selected") != selected["name"]:
        raise PZ4RBuildError("materialize winner differs under the declared ordinary-error rule")
    return selected


def validate_runtime_receipt(runtime: dict[str, object], output: Path, selected: dict[str, object]) -> None:
    runtime_dir = (output / "submission").resolve()
    if runtime.get("path") != str(runtime_dir):
        raise PZ4RBuildError("runtime receipt path differs from canonical submission path")
    records = runtime.get("files")
    expected_names = {*RUNTIME_SOURCE_FILES, "runtime-dependencies.json", "archive.zip"}
    if not isinstance(records, dict) or set(records) != expected_names:
        raise PZ4RBuildError("runtime receipt file denominator differs")
    for name in expected_names:
        validate_file_record(
            records[name],
            label=f"runtime {name}",
            expected_path=runtime_dir / name,
        )
    archive = records["archive.zip"]
    selected_archive = selected["records"]["archive"]
    if (archive["bytes"], archive["sha256"]) != (
        selected_archive["bytes"],
        selected_archive["sha256"],
    ):
        raise PZ4RBuildError("runtime archive differs from selected candidate archive")
    validate_runtime_manifest(runtime_dir)


def validate_resume_state(
    state: dict[str, object],
    output: Path,
    *,
    ps135: Any | None = None,
) -> None:
    """Validate every completed stage and its live bytes before any resume skips it."""

    if state.get("schema") != RESUME_SCHEMA or state.get("output") != str(output):
        raise PZ4RBuildError("unsupported or cross-output resume checkpoint")
    stages = ("preflight_complete", "materialize_complete", "verify_complete", "finalize_complete")
    for stage in stages:
        if stage in state and type(state[stage]) is not bool:
            raise PZ4RBuildError(f"resume stage flag is not boolean: {stage}")
    values = [state.get(stage, False) for stage in stages]
    if any(values[index] and not values[index - 1] for index in range(1, len(values))):
        raise PZ4RBuildError("resume checkpoint has a non-monotone stage state")
    if not values[0]:
        return

    validate_source_bindings(state.get("source_files", {}))
    pins = state.get("pins")
    expected_pins = {
        "lc2_archive": require_pin(
            LC2_ARCHIVE,
            size=EXPECTED_LC2_ARCHIVE_BYTES,
            digest=EXPECTED_LC2_ARCHIVE_SHA256,
        ),
        "lc2_carrier": require_pin(
            LC2_CARRIER,
            size=23_054,
            digest=EXPECTED_LC2_CARRIER_SHA256,
        ),
        "pgq1": require_pin(PGQ1, size=EXPECTED_PGQ1_BYTES, digest=EXPECTED_PGQ1_SHA256),
        "pgq1_decoded": require_pin(
            PGQ1_DECODED,
            size=14_528,
            digest=EXPECTED_PGQ1_DECODED_SHA256,
        ),
    }
    if pins != expected_pins:
        raise PZ4RBuildError("resume pinned-input records drifted")
    if not values[1]:
        return

    materialized_path = Path(str(state.get("materialize_receipt", "")))
    if materialized_path.resolve() != (output / "materialize_receipt.json").resolve():
        raise PZ4RBuildError("resume materialize receipt path differs")
    materialized = load_recorded_json(
        materialized_path,
        state.get("materialize_receipt_sha256"),
        schema=MATERIALIZE_SCHEMA,
        label="materialize",
    )
    selected = validate_materialized_receipt(materialized, output)
    if state.get("selected") != selected["name"]:
        raise PZ4RBuildError("resume selected candidate differs from receipt")
    if not values[2]:
        return

    verify_path = Path(str(state.get("verify_receipt", "")))
    if verify_path.resolve() != (output / "verify_receipt.json").resolve():
        raise PZ4RBuildError("resume verify receipt path differs")
    verify = load_recorded_json(
        verify_path,
        state.get("verify_receipt_sha256"),
        schema=VERIFY_SCHEMA,
        label="verify",
    )
    if verify.get("selected") != selected:
        raise PZ4RBuildError("verify receipt selected row differs from materialization")
    runtime = verify.get("public_runtime")
    if not isinstance(runtime, dict):
        raise PZ4RBuildError("verify runtime receipt is malformed")
    validate_runtime_receipt(runtime, output, selected)
    consumption = verify.get("pgq_consumption")
    if not isinstance(consumption, dict):
        raise PZ4RBuildError("verify PGQ-consumption receipt is malformed")
    mutation_files = {
        "mutated_gauge": "gauge.mutated.pgq1",
        "mutated_carrier": "carrier.mutated.pz4r",
        "selected_frame": "selected_frame0_pair0.npy",
        "mutated_frame": "mutated_frame0_pair0.npy",
    }
    for name, filename in mutation_files.items():
        validate_file_record(
            consumption.get(name, {}),
            label=f"mutation proof {name}",
            expected_path=output / "retained" / "mutation_proof" / filename,
        )
    if not consumption.get("coefficients_changed") or not consumption.get("rendered_frame_bytes_changed"):
        raise PZ4RBuildError("verify receipt lacks causal coefficient/frame mutation")
    if ps135 is not None:
        live_parse = public_runtime_parse(output / "submission", ps135=ps135)
        stored_parse = verify.get("public_runtime_parse", {})
        for key in ("basis_shape", "coeff_shape", "basis_sha256", "coeff_sha256"):
            if stored_parse.get(key) != live_parse.get(key):
                raise PZ4RBuildError(f"public runtime parse drifted: {key}")
    if not values[3]:
        return

    final_path = Path(str(state.get("final_result", "")))
    if final_path.resolve() != (output / "PZ4R_RESULT.json").resolve():
        raise PZ4RBuildError("resume final result path differs")
    final = load_recorded_json(
        final_path,
        state.get("final_result_sha256"),
        schema=RESULT_SCHEMA,
        label="final result",
    )
    if (
        final.get("selected") != selected
        or final.get("public_runtime") != runtime
        or final.get("public_runtime_parse") != verify.get("public_runtime_parse")
        or final.get("pgq_consumption") != consumption
    ):
        raise PZ4RBuildError("final result differs from its validated stage receipts")
    for index, record in enumerate(final.get("retained_payloads", [])):
        validate_file_record(record, label=f"final retained payload {index}")
    if final.get("retained_payload_count") != len(final.get("retained_payloads", [])):
        raise PZ4RBuildError("final retained-payload denominator differs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--required-free-bytes", type=int, default=2 << 30)
    args = parser.parse_args()

    output = validate_output(args.output)
    retained = output / "retained"
    checkpoints = output / "checkpoints"
    state_path = args.resume_from or checkpoints / "state.json"
    if state_path.resolve() != (checkpoints / "state.json").resolve():
        raise PZ4RBuildError("--resume-from must name the canonical state checkpoint")
    retained.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {
            "schema": RESUME_SCHEMA,
            "created_at_utc": utc_now(),
            "output": str(output),
        }
    )
    pz3, pz4p, ps135, codec, wire_receiver, inflate, predictor_receiver, gauge_receiver = import_surfaces()
    validate_resume_state(state, output, ps135=ps135)

    if not state.get("preflight_complete"):
        pins = {
            "lc2_archive": require_pin(
                LC2_ARCHIVE,
                size=EXPECTED_LC2_ARCHIVE_BYTES,
                digest=EXPECTED_LC2_ARCHIVE_SHA256,
            ),
            "lc2_carrier": require_pin(
                LC2_CARRIER,
                size=23_054,
                digest=EXPECTED_LC2_CARRIER_SHA256,
            ),
            "pgq1": require_pin(PGQ1, size=EXPECTED_PGQ1_BYTES, digest=EXPECTED_PGQ1_SHA256),
            "pgq1_decoded": require_pin(
                PGQ1_DECODED,
                size=14_528,
                digest=EXPECTED_PGQ1_DECODED_SHA256,
            ),
        }
        source, semantic_stream, hpac_stream, _, rate_proof = pz4p.verify_lc2_rate_path(ps135)
        if rate_proof["rebuilt_archive"]["sha256"] != EXPECTED_LC2_ARCHIVE_SHA256:
            raise PZ4RBuildError("LC2 exact rebuild proof drifted")
        state.update(
            {
                "preflight_complete": True,
                "storage": storage_preflight(output, args.required_free_bytes),
                "pins": pins,
                "rate_path_proof": rate_proof,
                "git_head": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=REPO,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                "source_files": source_binding_records(),
                "platform": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "torch": torch.__version__,
                },
            }
        )
        atomic_json(state_path, state)
        atomic_json(checkpoints / "stage_preflight_complete.json", state)

    if not state.get("materialize_complete"):
        source = ps135.load_lc2_source()
        semantic_stream, hpac_stream, _ = ps135.selected_lc2_streams()
        basis_component, basis, coefficients, coefficient_scales, absolute = pz3.split_cpr1(source.carrier, codec)
        config, decoded_targets, pgq_state = pz4p.decode_gauge(PGQ1.read_bytes())
        retained_decoded = np.load(PGQ1_DECODED)
        if not np.array_equal(decoded_targets, retained_decoded):
            raise PZ4RBuildError("literal PGQ1 decode differs from retained output")
        shared = retained / "shared"
        atomic_bytes(shared / "base_carrier.cpr1", source.carrier)
        atomic_bytes(shared / "basis_component.bin", basis_component)
        atomic_bytes(shared / "gauge.pgq1", PGQ1.read_bytes())
        atomic_npz(
            shared / "source_arrays.npz",
            basis=basis,
            coefficients=coefficients,
            coefficient_scales=coefficient_scales,
            absolute_coefficients=absolute,
            pgq_latent_codes=pgq_state["codes"].astype(np.int32),
            pgq_outputs=decoded_targets,
        )
        modes = {
            "target": predictor_receiver.FEATURE_TARGET,
            "target_quadratic": predictor_receiver.FEATURE_TARGET_QUADRATIC,
            "target_previous": predictor_receiver.FEATURE_TARGET_PREVIOUS,
            "target_quadratic_previous": predictor_receiver.FEATURE_TARGET_QUADRATIC_PREVIOUS,
        }
        candidates: list[dict[str, object]] = []
        for mode_name, feature_mode in modes.items():
            fraction_bits_grid = (
                (8, 10)
                if feature_mode
                in (
                    predictor_receiver.FEATURE_TARGET_QUADRATIC,
                    predictor_receiver.FEATURE_TARGET_QUADRATIC_PREVIOUS,
                )
                else (8, 10, 12, 16)
            )
            for target_fraction_bits in fraction_bits_grid:
                target_codes = gauge_receiver.quantize_pose_object(
                    decoded_targets,
                    target_fraction_bits,
                )
                # Q20 is the highest tested stable fixed-point predictor rung.
                # The fractional-bit grid is the feature-scaling sweep; every
                # materialized cell below is retained before comparison.
                shift = 20
                name = f"{mode_name}_f{target_fraction_bits}_q{shift}"
                candidate_dir = retained / "candidates" / name
                predictor = pz3.fit_predictor(
                    predictor_receiver,
                    target_codes,
                    absolute,
                    feature_mode,
                    shift,
                )
                carrier = gauge_receiver.encode_pose_gauge_carrier(
                    basis_component=basis_component,
                    gauge=PGQ1.read_bytes(),
                    predictor=predictor,
                    coefficient_scales=coefficient_scales,
                    target_fraction_bits=target_fraction_bits,
                )
                repeat_carrier = gauge_receiver.encode_pose_gauge_carrier(
                    basis_component=basis_component,
                    gauge=PGQ1.read_bytes(),
                    predictor=predictor,
                    coefficient_scales=coefficient_scales,
                    target_fraction_bits=target_fraction_bits,
                )
                if carrier != repeat_carrier:
                    raise PZ4RBuildError(f"{name}: carrier repeat differs")
                carrier_stream, model_pack, member, archive = build_archive(
                    carrier,
                    ps135=ps135,
                    source=source,
                    semantic_stream=semantic_stream,
                    hpac_stream=hpac_stream,
                    wire_receiver=wire_receiver,
                )
                repeat_stream, repeat_pack, repeat_member, repeat_archive = build_archive(
                    repeat_carrier,
                    ps135=ps135,
                    source=source,
                    semantic_stream=semantic_stream,
                    hpac_stream=hpac_stream,
                    wire_receiver=wire_receiver,
                )
                if (carrier_stream, model_pack, member, archive) != (
                    repeat_stream,
                    repeat_pack,
                    repeat_member,
                    repeat_archive,
                ):
                    raise PZ4RBuildError(f"{name}: archive repeat differs")
                parsed_basis, parsed_coefficients = gauge_receiver.decode_pose_gauge_carrier(carrier)
                if not np.array_equal(parsed_basis, basis):
                    raise PZ4RBuildError(f"{name}: basis parse-back differs")
                predicted_codes = np.rint(parsed_coefficients / coefficient_scales[None]).astype(np.int32)
                parsed_repeat = gauge_receiver.decode_pose_gauge_carrier(repeat_carrier)
                if not np.array_equal(parsed_coefficients, parsed_repeat[1]):
                    raise PZ4RBuildError(f"{name}: coefficient repeat differs")
                wire = parse_wire_archive(
                    archive,
                    expected_carrier=carrier,
                    source=source,
                    wire_receiver=wire_receiver,
                )
                payloads = {
                    "carrier": ("carrier.pz4r", carrier),
                    "carrier_repeat": ("carrier.repeat.pz4r", repeat_carrier),
                    "carrier_q9": ("carrier.q9.br", carrier_stream),
                    "carrier_q9_repeat": ("carrier.repeat.q9.br", repeat_stream),
                    "model_pack": ("models.split", model_pack),
                    "model_pack_repeat": ("models.repeat.split", repeat_pack),
                    "member": ("p", member),
                    "member_repeat": ("p.repeat", repeat_member),
                    "archive": ("archive.zip", archive),
                    "archive_repeat": ("archive.repeat.zip", repeat_archive),
                }
                records: dict[str, object] = {}
                for key, (filename, payload) in payloads.items():
                    path = candidate_dir / filename
                    atomic_bytes(path, payload)
                    records[key] = file_record(path)
                arrays_path = candidate_dir / "parsed_arrays.npz"
                atomic_npz(
                    arrays_path,
                    basis=parsed_basis,
                    coefficients=parsed_coefficients,
                    predicted_codes=predicted_codes,
                )
                records["parsed_arrays"] = file_record(arrays_path)
                candidates.append(
                    {
                        "name": name,
                        "feature_mode": feature_mode,
                        "target_fraction_bits": target_fraction_bits,
                        "shift": shift,
                        "records": records,
                        "wire_parseback": wire,
                        "metrics": coefficient_metrics(predicted_codes, absolute),
                    }
                )
                print(
                    f"{name}: archive={len(archive)} carrier={len(carrier)} "
                    f"r2={candidates[-1]['metrics']['coefficient_code_r2_variance_weighted']:.6f}",
                    flush=True,
                )
        eligible = [row for row in candidates if row["records"]["archive"]["bytes"] < EXPECTED_LC2_ARCHIVE_BYTES]
        if not eligible:
            raise PZ4RBuildError("no direct receiver candidate beats the LC2 archive ceiling")
        selected = max(
            eligible,
            key=lambda row: (
                row["metrics"]["coefficient_code_r2_variance_weighted"],
                -row["records"]["archive"]["bytes"],
            ),
        )
        receipt = {
            "schema": MATERIALIZE_SCHEMA,
            "created_at_utc": utc_now(),
            "axis": AXIS,
            "score_claim": False,
            "selection_rule": (
                "highest full-n600 ordinary signed (non-circular) decoded-coefficient-code "
                "variance-weighted R2 among byte-closed archives below the LC2 archive "
                "ceiling; archive bytes break ties"
            ),
            "shared": {
                name: file_record(shared / filename)
                for name, filename in (
                    ("base_carrier", "base_carrier.cpr1"),
                    ("basis_component", "basis_component.bin"),
                    ("pgq1", "gauge.pgq1"),
                    ("source_arrays", "source_arrays.npz"),
                )
            },
            "candidates": candidates,
            "selected": selected["name"],
            "rate_envelope_reconciliation": {
                "preproof_envelope_bytes": EXPECTED_RATE_ENVELOPE_BYTES,
                "selected_archive_bytes": selected["records"]["archive"]["bytes"],
                "delta_bytes": selected["records"]["archive"]["bytes"] - EXPECTED_RATE_ENVELOPE_BYTES,
                "reason": "the direct rung retains the counted 13101-byte LC2 spatial basis plus a counted fixed-point map; the 168005-byte envelope assumed both basis and coefficient descriptions disappeared",
            },
        }
        receipt_path = output / "materialize_receipt.json"
        atomic_json(receipt_path, receipt)
        state.update(
            {
                "materialize_complete": True,
                "materialize_receipt": str(receipt_path),
                "materialize_receipt_sha256": sha256_file(receipt_path),
                "selected": selected["name"],
            }
        )
        atomic_json(state_path, state)
        atomic_json(checkpoints / "stage_materialize_complete.json", state)

    if not state.get("verify_complete"):
        materialized = json.loads(Path(state["materialize_receipt"]).read_text(encoding="utf-8"))
        selected = next(row for row in materialized["candidates"] if row["name"] == state["selected"])
        source = ps135.load_lc2_source()
        _, basis, _, coefficient_scales, _ = pz3.split_cpr1(source.carrier, codec)
        carrier_path = Path(selected["records"]["carrier"]["path"])
        selected_basis, selected_coefficients = gauge_receiver.decode_pose_gauge_carrier(carrier_path.read_bytes())
        config, _, pgq_state = pz4p.decode_gauge(PGQ1.read_bytes())
        mutated_codes = pgq_state["codes"].copy()
        qmax = (1 << (config.depth - 1)) - 1
        mutated_codes[0, 0] += -1 if mutated_codes[0, 0] >= qmax else 1
        mutated_gauge = pz4p.encode_gauge(
            config,
            mutated_codes,
            pgq_state["scales"],
            pgq_state["compensation"],
        )
        fields = gauge_receiver.HEADER.unpack_from(carrier_path.read_bytes())
        target_fraction_bits = fields[4]
        basis_bytes, gauge_bytes, model_bytes = fields[5:8]
        model_start = gauge_receiver.HEADER.size + basis_bytes + gauge_bytes
        predictor, _ = predictor_receiver.deserialize_predictor(
            carrier_path.read_bytes()[model_start : model_start + model_bytes]
        )
        mutated_carrier = gauge_receiver.encode_pose_gauge_carrier(
            basis_component=Path(materialized["shared"]["basis_component"]["path"]).read_bytes(),
            gauge=mutated_gauge,
            predictor=predictor,
            coefficient_scales=coefficient_scales,
            target_fraction_bits=target_fraction_bits,
        )
        mutated_basis, mutated_coefficients = gauge_receiver.decode_pose_gauge_carrier(mutated_carrier)
        coefficients_changed = bool(np.any(selected_coefficients != mutated_coefficients))
        if not coefficients_changed or not np.array_equal(mutated_basis, selected_basis):
            raise PZ4RBuildError("PGQ mutation did not change only predicted coefficients")
        mutation_dir = retained / "mutation_proof"
        atomic_bytes(mutation_dir / "gauge.mutated.pgq1", mutated_gauge)
        atomic_bytes(mutation_dir / "carrier.mutated.pz4r", mutated_carrier)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        base_frame = mutation_dir / "selected_frame0_pair0.npy"
        mutated_frame = mutation_dir / "mutated_frame0_pair0.npy"
        base_render = pz3.render_selected_slaves(
            base_frame,
            inflate,
            torch.from_numpy(selected_basis),
            torch.from_numpy(selected_coefficients),
            [0],
        )
        mutated_render = pz3.render_selected_slaves(
            mutated_frame,
            inflate,
            torch.from_numpy(mutated_basis),
            torch.from_numpy(mutated_coefficients),
            [0],
        )
        frames_changed = base_render["tensor_sha256"] != mutated_render["tensor_sha256"]
        if not frames_changed:
            raise PZ4RBuildError("PGQ mutation changed coefficients but not rendered frame bytes")
        selected_archive = Path(selected["records"]["archive"]["path"])
        runtime = materialize_runtime(output, selected_archive)
        runtime_parse = public_runtime_parse(Path(runtime["path"]), ps135=ps135)
        direct_basis_sha256 = sha256_bytes(np.ascontiguousarray(selected_basis).tobytes())
        direct_coeff_sha256 = sha256_bytes(np.ascontiguousarray(selected_coefficients).tobytes())
        if runtime_parse["basis_sha256"] != direct_basis_sha256 or runtime_parse["coeff_sha256"] != direct_coeff_sha256:
            raise PZ4RBuildError("public runtime arrays differ from direct PZ4R parse")
        runtime_parse["byte_identical_to_direct_parse"] = True
        verify = {
            "schema": VERIFY_SCHEMA,
            "created_at_utc": utc_now(),
            "axis": AXIS,
            "score_claim": False,
            "selected": selected,
            "public_runtime": runtime,
            "public_runtime_parse": runtime_parse,
            "pgq_consumption": {
                "mutated_gauge": file_record(mutation_dir / "gauge.mutated.pgq1"),
                "mutated_carrier": file_record(mutation_dir / "carrier.mutated.pz4r"),
                "coefficients_changed": coefficients_changed,
                "rendered_frame_bytes_changed": frames_changed,
                "selected_frame": {**file_record(base_frame), **base_render},
                "mutated_frame": {**file_record(mutated_frame), **mutated_render},
            },
            "receiver_boundary": {
                "cpr1_packet_present": False,
                "exact_lc2_basis_retained_and_counted": True,
                "exact_coefficient_residual_present": False,
                "scorer_imported_by_receiver": False,
                "semantic_hpac_tokens_unchanged": True,
                "basis_parseback_exact": True,
                "coefficient_parseback_exact_to_prediction": True,
                "coefficient_exact_to_lc2": False,
            },
        }
        verify_path = output / "verify_receipt.json"
        atomic_json(verify_path, verify)
        state.update(
            {
                "verify_complete": True,
                "verify_receipt": str(verify_path),
                "verify_receipt_sha256": sha256_file(verify_path),
            }
        )
        atomic_json(state_path, state)
        atomic_json(checkpoints / "stage_verify_complete.json", state)

    if not state.get("finalize_complete"):
        materialized = json.loads(Path(state["materialize_receipt"]).read_text(encoding="utf-8"))
        verify = json.loads(Path(state["verify_receipt"]).read_text(encoding="utf-8"))
        payloads: list[dict[str, object]] = list(materialized["shared"].values())
        for candidate in materialized["candidates"]:
            payloads.extend(candidate["records"].values())
        payloads.extend(
            [
                verify["pgq_consumption"]["mutated_gauge"],
                verify["pgq_consumption"]["mutated_carrier"],
                verify["pgq_consumption"]["selected_frame"],
                verify["pgq_consumption"]["mutated_frame"],
            ]
        )
        for record in payloads:
            path = Path(record["path"])
            if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
                raise PZ4RBuildError(f"retained payload drift: {path}")
        selected = verify["selected"]
        final = {
            "schema": RESULT_SCHEMA,
            "created_at_utc": utc_now(),
            "axis": AXIS,
            "score_claim": False,
            "verdict": "DIRECT-RECEIVER-BYTE-CLOSED; REALIZED-SCORER-PENDING",
            "selected": selected,
            "rate_envelope_reconciliation": materialized["rate_envelope_reconciliation"],
            "public_runtime": verify["public_runtime"],
            "public_runtime_parse": verify["public_runtime_parse"],
            "pgq_consumption": verify["pgq_consumption"],
            "receiver_boundary": verify["receiver_boundary"],
            "retained_payloads": payloads,
            "retained_payload_count": len(payloads),
            "resumability": {
                "resume_from": str(checkpoints / "state.json"),
                "stage_checkpoints": [
                    str(checkpoints / "stage_preflight_complete.json"),
                    str(checkpoints / "stage_materialize_complete.json"),
                    str(checkpoints / "stage_verify_complete.json"),
                    str(checkpoints / "stage_finalize_complete.json"),
                ],
            },
            "boundaries": {
                "full_n600_decode": False,
                "d_seg_measured": False,
                "d_pose_measured": False,
                "scorer_lane_claimed": False,
                "modal_or_paid_dispatch": False,
                "contest_cpu_cuda_eval": False,
                "upstream_modified": False,
            },
        }
        final_path = output / "PZ4R_RESULT.json"
        atomic_json(final_path, final)
        state.update(
            {
                "finalize_complete": True,
                "final_result": str(final_path),
                "final_result_sha256": sha256_file(final_path),
            }
        )
        atomic_json(state_path, state)
        atomic_json(checkpoints / "stage_finalize_complete.json", state)

    # A completed invocation is accepted only after the same validation used by
    # a resumed invocation re-hashes sources, receipts, payloads, and runtime.
    validate_resume_state(state, output, ps135=ps135)

    print(
        json.dumps(
            {
                "result": state["final_result"],
                "sha256": state["final_result_sha256"],
                "selected": state["selected"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
