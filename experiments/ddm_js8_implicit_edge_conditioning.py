#!/usr/bin/env python3
"""Build JS8's decoder-derived Road-hub edge gate on the exact MC36 receiver.

The gate is a counted 5x5 edge-family table, not a pixel mask.  At decode the
receiver derives its spatial support from already-decoded semantic tokens and
uses it to gate the retained EC2 adapter at EC2's pre-TokenBlock injection
site.  This program retains the active and inactive payloads, deterministic
archives, parse-back state, adapted runtimes, and an exact real-semantic-path
inactive identity probe.  It does not claim a scorer verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import numpy as np
import torch

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.ddm_ec1_runtime import js8_edge_state_conditioner as js8_runtime

LOGICAL_ROOT: Final = Path("/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/edge_conditioned/js8")
BULK_ROOT: Final = Path("/Volumes/APDataStore/pact/pr135_joint_solve_20260810/edge_conditioned/js8")
JS1C_ROOT: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/contest_cuda/ddm_js1c_20260814"
)
STAGE0: Final = JS1C_ROOT / "STAGE0_RESULT.json"
TRIGGER: Final = JS1C_ROOT / "TASK_1043_TRIGGER_RECEIPT.json"
TOKENS: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/"
    "candidates/cp135_base/retained/decoded_tokens_n600.npy"
)
BASE_ARCHIVE: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105/micro35_candidate/archive.zip"
)
BASE_RUNTIME: Final = Path("/Volumes/VertigoDataTier/pact/ddm_f26p_runtime_cpu_lift_20260814/lifted_submission_cpu")
EC2_MODULE: Final = Path("/Volumes/VertigoDataTier/pact/ddm_bg1_20260814/retained/ec2_oriented/ec1_latent.int8.br")
EC1_RUNTIME_SOURCE: Final = REPO / "experiments/ddm_ec1_runtime/ec1_latent_conditioner.py"
JS8_RUNTIME_SOURCE: Final = REPO / "experiments/ddm_ec1_runtime/js8_edge_state_conditioner.py"

STAGE0_SHA256: Final = "472fc816f6656ec0cdd37bd475598e8e9683260dc97adeb4163ead5ae90b3e67"
TRIGGER_SHA256: Final = "ad9da227d6329efcbf510f084f748c5f53e28522bea8bb30e694d004bb4ce8e0"
BASE_ARCHIVE_SHA256: Final = "f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de"
BASE_ARCHIVE_BYTES: Final = 186_269
EC2_MODULE_SHA256: Final = "9559c2ab5128f193c8b0c754c5d61851b7784070fa049e04cf48cfd157eead82"
TOKENS_SHA256: Final = "03f5379d70e4bbd88e125cfbfb785cf5473315c70a5b78661fa426bb3e96e0f4"
NUM_CLASSES: Final = 5
CLASS_NAMES: Final = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
SEED: Final = 20_260_814


class JS8BuildError(RuntimeError):
    """A pinned input, archive, parse-back, or receiver invariant failed."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_npy(path: Path, value: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    atomic_bytes(path, buffer.getvalue())


def require_file(path: Path, *, size: int | None = None, digest: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise JS8BuildError(f"required file missing: {path}")
    record = file_record(path)
    if size is not None and record["bytes"] != size:
        raise JS8BuildError(f"required file byte count differs: {path}")
    if digest is not None and record["sha256"] != digest:
        raise JS8BuildError(f"required file SHA-256 differs: {path}")
    return record


def storage_preflight() -> dict[str, Any]:
    LOGICAL_ROOT.mkdir(parents=True, exist_ok=True)
    BULK_ROOT.mkdir(parents=True, exist_ok=True)
    logical = shutil.disk_usage(LOGICAL_ROOT)
    bulk = shutil.disk_usage(BULK_ROOT)
    report = {
        "schema": "ddm_js8_storage_routing.v1",
        "logical_consumer_store": str(LOGICAL_ROOT),
        "bulk_store": str(BULK_ROOT),
        "logical_free_bytes": logical.free,
        "bulk_free_bytes": bulk.free,
        "bulk_minimum_reserve_bytes": 20 * 1024**3,
        "bulk_admitted": bulk.free >= 20 * 1024**3,
        "policy": "small routing/receipts on Vertigo; all materialized payloads on APDataStore; certify-or-block",
    }
    atomic_json(LOGICAL_ROOT / "STORAGE_ROUTING.json", report)
    if not report["bulk_admitted"]:
        raise JS8BuildError("APDataStore lacks the JS8 bulk reserve")
    return report


def derive_edge_table(stage0: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Turn the measured CP135 Road-hub edge mass into a decoded-token gate."""
    rows = stage0["road_hub_map"]["cp135_base"]["undirected_interfaces"]
    by_edge = {str(row["edge"]): int(row["flips"]) for row in rows}
    anchor = by_edge["Road<->Lane"]
    if anchor <= 0:
        raise JS8BuildError("Road-Lane decomposition anchor is non-positive")
    weights = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float32)
    used = []
    for other_index, other_name in enumerate(CLASS_NAMES[1:], start=1):
        edge = f"Road<->{other_name}"
        measured_flips = by_edge[edge]
        weight = measured_flips / anchor
        weights[0, other_index] = weights[other_index, 0] = weight
        used.append({"edge": edge, "measured_flips": measured_flips, "weight_vs_road_lane": weight})
    derivation = {
        "schema": "ddm_js8_edge_table_derivation.v1",
        "source_stage0": file_record(STAGE0),
        "source_object": "road_hub_map.cp135_base.undirected_interfaces",
        "normalization": "each Road-hub edge's measured flips divided by Road<->Lane flips",
        "rows": used,
        "road_incident_flips": int(stage0["road_hub_map"]["cp135_base"]["incident_flips"]),
        "road_incident_share": float(stage0["road_hub_map"]["cp135_base"]["incident_share"]),
        "no_explicit_mask": True,
    }
    return weights, derivation


def deterministic_archive(ec1_blob: bytes, gate_blob: bytes) -> bytes:
    with zipfile.ZipFile(BASE_ARCHIVE) as archive:
        if archive.namelist() != ["p"]:
            raise JS8BuildError("MC36 archive grammar differs")
        payload = archive.read("p")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, value in (("p", payload), ("ec1_latent.br", ec1_blob), ("js8_edge_gate.br", gate_blob)):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, value)
    return stream.getvalue()


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    if source.count(old) != 1:
        raise JS8BuildError(f"expected one runtime patch site in {path}: {old[:80]!r}")
    atomic_bytes(path, source.replace(old, new).encode())


def adapt_runtime(root: Path, archive_record: dict[str, Any]) -> Path:
    destination = root / "adapted_runtime"
    if destination.exists():
        raise JS8BuildError(f"refusing to overwrite adapted runtime: {destination}")
    staging = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    if staging.exists():
        raise JS8BuildError(f"preserve prior partial adapted runtime: {staging}")
    shutil.copytree(
        BASE_RUNTIME,
        staging,
        ignore=shutil.ignore_patterns("archive.zip", "__pycache__", "*.pyc", "._*", ".DS_Store"),
    )
    atomic_bytes(staging / "archive.zip", Path(archive_record["path"]).read_bytes())
    shutil.copy2(EC1_RUNTIME_SOURCE, staging / "runtime/ec1_latent_conditioner.py")
    shutil.copy2(JS8_RUNTIME_SOURCE, staging / "runtime/js8_edge_state_conditioner.py")

    residual = staging / "runtime/residual_archive.py"
    replace_once(
        residual,
        'if archive.namelist() != ["p"]:\n            raise ResidualArchiveError("archive must contain exactly member p")',
        'if archive.namelist() != ["p", "ec1_latent.br", "js8_edge_gate.br"]:\n'
        '            raise ResidualArchiveError("JS8 archive must contain p plus counted EC2 and edge-gate payloads")',
    )
    f26 = staging / "runtime/f26_inflate.py"
    replace_once(f26, "import time\n", "import time\nimport zipfile\n")
    replace_once(
        f26,
        "    parts = read_residual_archive(archive_path)\n",
        "    with zipfile.ZipFile(archive_path) as archive:\n"
        '        if archive.namelist() != ["p", "ec1_latent.br", "js8_edge_gate.br"]:\n'
        '            raise InflationError("JS8 archive members differ")\n'
        '        ec1_blob = archive.read("ec1_latent.br")\n'
        '        js8_gate_blob = archive.read("js8_edge_gate.br")\n'
        "    parts = read_residual_archive(archive_path)\n",
    )
    replace_once(
        f26,
        "    renderer.render_video(semantic, basis, coefficients, tokens, partial, device)\n",
        "    renderer.render_video(semantic, basis, coefficients, tokens, partial, device, "
        "ec1_blob=ec1_blob, js8_gate_blob=js8_gate_blob)\n",
    )
    renderer = staging / "cpr1/inflate.py"
    replace_once(
        renderer,
        "from torch.nn import functional\n",
        "from torch.nn import functional\nfrom runtime.js8_edge_state_conditioner import conditioned_semantic_forward\n",
    )
    replace_once(
        renderer,
        "def render_video(semantic, basis, coefficients, tokens, destination: Path, device):\n",
        "def render_video(semantic, basis, coefficients, tokens, destination: Path, device, *, ec1_blob, js8_gate_blob):\n",
    )
    replace_once(
        renderer,
        "                semantic(tokens[start:end].long().to(device), indices),\n",
        "                conditioned_semantic_forward(semantic, tokens[start:end].long().to(device), "
        "indices, ec1_blob, js8_gate_blob),\n",
    )
    outer = staging / "inflate.py"
    replace_once(outer, f'ARCHIVE_SHA256 = "{BASE_ARCHIVE_SHA256}"', f'ARCHIVE_SHA256 = "{archive_record["sha256"]}"')
    replace_once(outer, "ARCHIVE_BYTES = 186_269", f"ARCHIVE_BYTES = {archive_record['bytes']:_}")
    replace_once(
        outer,
        'if archive.namelist() != ["p"]:\n            raise ValueError("archive.zip must contain exactly the payload file p")',
        'if archive.namelist() != ["p", "ec1_latent.br", "js8_edge_gate.br"]:\n'
        '            raise ValueError("archive.zip must contain p plus counted EC2 and JS8 gate payloads")',
    )
    os.replace(staging, destination)
    return destination


def _load_renderer(renderer_dir: Path) -> ModuleType:
    path = renderer_dir / "inflate.py"
    name = f"_ddm_js8_renderer_{sha256_file(path)[:12]}"
    sys.path.insert(0, str(renderer_dir))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise JS8BuildError(f"cannot load renderer: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def load_exact_semantic() -> torch.nn.Module:
    sys.path.insert(0, str(BASE_RUNTIME))
    try:
        from runtime.entropy.renderer_weight_codec import decode_wans1
        from runtime.residual_archive import read_residual_archive
    finally:
        sys.path.pop(0)
    renderer = _load_renderer(BASE_RUNTIME / "cpr1")
    parts = read_residual_archive(BASE_ARCHIVE)
    semantic = renderer.SemanticTokenRenderer(96)
    state = {
        record.schema.name: torch.from_numpy(np.ascontiguousarray(record.values, dtype=np.float32))
        for record in decode_wans1(parts.semantic_blob)
    }
    semantic.load_state_dict(state, strict=True)
    return semantic.eval()


def identity_probe(ec1_blob: bytes, inactive_blob: bytes, output: Path) -> dict[str, Any]:
    """Exercise the real MC36 semantic object on seeded pairs with the gate inactive."""
    tokens = np.load(TOKENS, mmap_mode="r", allow_pickle=False)
    if tokens.shape != (600, 384, 512) or tokens.dtype != np.uint8:
        raise JS8BuildError("decoded token field geometry differs")
    rng = np.random.default_rng(SEED)
    pair_ids = np.sort(rng.choice(600, size=32, replace=False)).astype(np.int64)
    semantic = load_exact_semantic()
    base_outputs = []
    inactive_outputs = []
    with torch.inference_mode():
        for pair in pair_ids:
            token = torch.from_numpy(np.asarray(tokens[int(pair)]).copy())[None].long()
            index = torch.tensor([int(pair)], dtype=torch.long)
            base_outputs.append(semantic(token, index)[0].cpu().numpy().astype(np.float32, copy=False))
            inactive_outputs.append(
                js8_runtime.conditioned_semantic_forward(semantic, token, index, ec1_blob, inactive_blob)[0]
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
    base = np.stack(base_outputs)
    inactive = np.stack(inactive_outputs)
    atomic_npy(output / "retained/identity_probe/pair_ids.int64.npy", pair_ids)
    atomic_npy(output / "retained/identity_probe/base_pre_r.float32.npy", base)
    atomic_npy(output / "retained/identity_probe/inactive_pre_r.float32.npy", inactive)
    if not np.array_equal(base, inactive):
        raise JS8BuildError("inactive JS8 gate is not byte-identical on the real MC36 semantic path")
    return {
        "schema": "ddm_js8_inactive_identity_probe.v1",
        "selection_mode": "seeded random n32; identity invariant only, not a score verdict",
        "seed": SEED,
        "pairs": pair_ids.tolist(),
        "array_equal": True,
        "max_abs_difference": 0.0,
        "payloads": {
            name: file_record(output / f"retained/identity_probe/{name}")
            for name in ("pair_ids.int64.npy", "base_pre_r.float32.npy", "inactive_pre_r.float32.npy")
        },
    }


def build() -> dict[str, Any]:
    storage = storage_preflight()
    pins = {
        "stage0": require_file(STAGE0, digest=STAGE0_SHA256),
        "trigger": require_file(TRIGGER, digest=TRIGGER_SHA256),
        "mc36_archive": require_file(BASE_ARCHIVE, size=BASE_ARCHIVE_BYTES, digest=BASE_ARCHIVE_SHA256),
        "ec2_adapter": require_file(EC2_MODULE, digest=EC2_MODULE_SHA256),
        "decoded_tokens": require_file(TOKENS, size=117_964_928, digest=TOKENS_SHA256),
    }
    stage0 = json.loads(STAGE0.read_text())
    weights, derivation = derive_edge_table(stage0)
    output = BULK_ROOT / "build_v1"
    if output.exists():
        raise JS8BuildError(f"immutable JS8 output already exists: {output}")
    output.mkdir(parents=True)
    atomic_json(output / "EDGE_TABLE_DERIVATION.json", derivation)
    atomic_npy(output / "retained/conditioning/edge_weights.float32.npy", weights)
    ec1_blob = EC2_MODULE.read_bytes()
    active_blob = js8_runtime.serialize_gate(weights, adapter_scale=1.0)
    inactive_blob = js8_runtime.serialize_gate(np.zeros_like(weights), adapter_scale=0.0)
    records = {}
    for label, gate_blob in (("active", active_blob), ("inactive_identity", inactive_blob)):
        root = output / label
        atomic_bytes(root / "retained/ec1_latent.int8.br", ec1_blob)
        atomic_bytes(root / "retained/js8_edge_gate.br", gate_blob)
        decoded, scale, header = js8_runtime.parse_gate(gate_blob)
        atomic_npy(root / "retained/js8_edge_gate.parseback.float32.npy", decoded)
        archive = deterministic_archive(ec1_blob, gate_blob)
        repeat = deterministic_archive(ec1_blob, gate_blob)
        if archive != repeat:
            raise JS8BuildError(f"{label} archive repeat differs")
        atomic_bytes(root / "retained/archive.zip", archive)
        atomic_bytes(root / "retained/archive.repeat.zip", repeat)
        archive_record = file_record(root / "retained/archive.zip")
        runtime = adapt_runtime(root, archive_record)
        records[label] = {
            "gate": file_record(root / "retained/js8_edge_gate.br"),
            "gate_parseback": file_record(root / "retained/js8_edge_gate.parseback.float32.npy"),
            "gate_header": header,
            "adapter_scale": scale,
            "archive": archive_record,
            "archive_repeat": file_record(root / "retained/archive.repeat.zip"),
            "archive_delta_bytes_vs_mc36": archive_record["bytes"] - BASE_ARCHIVE_BYTES,
            "runtime": str(runtime),
        }
    identity = identity_probe(ec1_blob, inactive_blob, output)
    result = {
        "schema": "ddm_js8_build_result.v1",
        "status": "RECEIVER_CLOSED_ADMISSION_BUILT_SCORER_OWED",
        "axis": "[macOS-CPU torch receiver-component identity, seeded-random n32] NO SCORE",
        "score_claim": False,
        "pointer_moved": False,
        "pins": pins,
        "storage": storage,
        "design": derivation,
        "payloads": records,
        "inactive_identity": identity,
        "measured": "exact bytes, deterministic archive repeat, parse-back, and inactive equality on the real MC36 semantic receiver path",
        "not_measured": "active candidate SegNet, PoseNet, realized joint delta-S, full-n600 real-decode output, or contest axis",
        "next_fire": {
            "disposition": "QUEUED-WITH-A-FIRE-ORDER",
            "owner": "JS8 scorer-slot owner / MAIN",
            "consumer_store": str(BULK_ROOT / "full_n600_v1"),
            "fire_trigger": "one full-n600 scorer slot is explicitly owned; first compile qs5-pattern frame-0 compensation for active frame-1 edits, then decode and score active versus MC36 in chunks <=120",
        },
    }
    atomic_json(output / "BUILD_RESULT.json", result)
    atomic_json(LOGICAL_ROOT / "BUILD_POINTER.json", {"result": file_record(output / "BUILD_RESULT.json")})
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("build",))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "build":
        print(json.dumps(build(), indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
