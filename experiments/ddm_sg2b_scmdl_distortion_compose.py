#!/usr/bin/env python3
"""Build scorer-free, receiver-closed SCMDL advisory candidates on AFR1.

This module never imports or runs a scorer.  It verifies SFP1's retained dense
fields, derives their changed-site counts against the null field, materializes
pair-plane overlays for the proven JG2 inverse coder, and stages pin-consistent
runtime/archive trees consumed by ``tools/fire_local_advisory.py``.

The long ``batch`` stage is restartable through JG2's immutable RC64 payloads
and 25-frame checkpoints.  It must be launched detached: set
``SG2B_DETACHED_BATCH=1`` only in the durable nohup launcher recorded by
``prepare``.  Every materialized stream, archive, overlay, receipt, and runtime
is retained under the SG2B APDataStore root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tac.candidate_seal import CONSISTENT, check_pin_consistency, repin_receiver

AP_ROOT = Path("/Volumes/APDataStore/pact")
VERTIGO_ROOT = Path("/Volumes/VertigoDataTier/pact")
STORE = AP_ROOT / "ddm_sg2b_scmdl_distortion_leg_build"
FIRE_ROOT = VERTIGO_ROOT / "ddm_sg2b_scmdl_distortion_leg_build" / "fire_main"
SFP1_ROOT = AP_ROOT / "ddm_jc1" / "scmdl_projection"
SOURCE_RUNTIME = AP_ROOT / "ddm_pq12" / "generation_7_afr1"
SOURCE_ARCHIVE = SOURCE_RUNTIME / "archive.zip"
BASE_FIELD = SFP1_ROOT / "controls" / "null_empty_proposal.u8"
REENCODE_STORE = STORE / "reencode"
BASE_RUNTIME = STORE / "runtimes" / "p00_null"
JG2 = REPO / "experiments" / "ddm_jg2_tail_reencode.py"
FIRE_TOOL = REPO / "tools" / "fire_local_advisory.py"

N, H, W = 600, 384, 512
FIELD_BYTES = N * H * W
AFR1_ARCHIVE_BYTES = 180_002
AFR1_ARCHIVE_SHA256 = "cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25"
BASE_FIELD_SHA256 = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
AFR1_D_SEG = 0.00020139
AFR1_D_POSE = 6.37e-06
MINIMUM_FREE_BYTES = 512 << 20
RX1_HEADER = struct.Struct("<4sBBBBHHH")
RX1_EXPECTED = {
    "magic": "RX1M",
    "version": 1,
    "codec": 2,
    "table_mode": 0,
    "reserved": 0b11010,
    "hpac_bytes": 13_515,
    "semantic_bytes": 30_856,
    "carrier_bytes": 22_010,
}

PINNED_SOURCE_FILES = {
    "runtime/f26_inflate.py": "5d705f93c051b2b540845dad4140f73d7dd61c721e4de2ed33b2ad32170c35c4",
    "runtime/residual_archive.py": "aca361f3e94941f4f2800bacec79f5032335588e317e76ee1a306bbb5ba64530",
    "cpr1/inflate.py": "ff446edd9237148bdc898be2f8f8c4782bf231a50cf3830c4b0b21a4474a736b",
    "inflate.py": "a499942a9993737755f771a95a81b8a12fc4a3b2e1b5ba5cd9d9cbfc738ea958",
    "inflate.sh": "971eaa12b78e716825741ea86c28f9362eb9be077cc8cb3b873810ca979beb65",
}
RUNTIME_INVARIANT_SOURCE_FILES = tuple(
    relative for relative in PINNED_SOURCE_FILES if relative != "inflate.py"
)


@dataclass(frozen=True)
class Proposal:
    key: str
    proposal_id: str
    path: Path
    sha256: str
    changed_sites: int


PROPOSALS = (
    Proposal("p00", "p00_null", BASE_FIELD, BASE_FIELD_SHA256, 0),
    Proposal(
        "p01",
        "sfp1_p01_atlas24_boundary1",
        SFP1_ROOT / "candidates" / "sfp1_p01_atlas24_boundary1.u8",
        "75fe37daf8c3f615cd943a76697e9c6e8eabc56cb1c23d55a6b4251fc4553690",
        1_084,
    ),
    Proposal(
        "p02",
        "sfp1_p02_atlas64_boundary1",
        SFP1_ROOT / "candidates" / "sfp1_p02_atlas64_boundary1.u8",
        "656bd0c5c102109c3327eccd0c6e3a606aac44cbce7d9144396f8c171e24b76e",
        2_831,
    ),
    Proposal(
        "p03",
        "sfp1_p03_mi1_patch12_boundary1",
        SFP1_ROOT / "candidates" / "sfp1_p03_mi1_patch12_boundary1.u8",
        "fe6a9dd8ce770e308c7c3d1903ea1e40bee44938cc836188e486eefd408f527a",
        9_723,
    ),
)
BY_KEY = {proposal.key: proposal for proposal in PROPOSALS}


class Sg2bError(RuntimeError):
    """A custody, composition, receiver, or routing gate refused."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def require_file(path: Path, expected_bytes: int, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise Sg2bError(f"missing pinned input: {path}")
    observed = file_fact(path)
    if observed["bytes"] != expected_bytes or observed["sha256"] != expected_sha256:
        raise Sg2bError(
            f"pin mismatch for {path}: {observed['bytes']}/{observed['sha256']} != "
            f"{expected_bytes}/{expected_sha256}"
        )
    return observed


def derive_changed_sites(base_path: Path, candidate_path: Path, *, expected_bytes: int) -> int:
    """Count byte positions that differ without materializing either full field."""
    if base_path.stat().st_size != expected_bytes or candidate_path.stat().st_size != expected_bytes:
        raise Sg2bError("changed-site inputs do not match the declared dense-field denominator")
    changed = 0
    with base_path.open("rb") as base, candidate_path.open("rb") as candidate:
        while True:
            left = base.read(8 << 20)
            right = candidate.read(8 << 20)
            if not left and not right:
                break
            if len(left) != len(right):
                raise Sg2bError("changed-site inputs ended at different offsets")
            changed += int(np.count_nonzero(np.frombuffer(left, dtype=np.uint8) != np.frombuffer(right, dtype=np.uint8)))
    return changed


def materialize_overlay(
    base_path: Path,
    candidate_path: Path,
    destination: Path,
    *,
    shape: tuple[int, int, int] = (N, H, W),
) -> dict[str, Any]:
    """Retain JG2's exact pair-plane overlay, refusing overwrite or silent loss."""
    frames, height, width = shape
    expected_bytes = frames * height * width
    if base_path.stat().st_size != expected_bytes or candidate_path.stat().st_size != expected_bytes:
        raise Sg2bError("overlay inputs do not match the declared shape")
    base = np.memmap(base_path, mode="r", dtype=np.uint8, shape=shape)
    candidate = np.memmap(candidate_path, mode="r", dtype=np.uint8, shape=shape)
    active = [index for index in range(frames) if np.any(base[index] != candidate[index])]
    payload = {str(index): np.asarray(candidate[index]).copy() for index in active}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    observed = file_fact(temporary)
    if destination.exists():
        if file_fact(destination)["sha256"] != observed["sha256"]:
            raise Sg2bError(f"refusing to overwrite a different retained overlay: {destination}")
        temporary.unlink()
    else:
        os.replace(temporary, destination)
    with np.load(destination, allow_pickle=False) as blob:
        restored = np.array(base, copy=True)
        for key in blob.files:
            restored[int(key)] = np.asarray(blob[key], dtype=np.uint8)
    restored_sha = hashlib.sha256(restored.tobytes()).hexdigest()
    candidate_sha = sha256_file(candidate_path)
    if restored_sha != candidate_sha:
        raise Sg2bError(f"overlay parse-back differs from dense proposal {candidate_path}")
    return {
        "overlay": file_fact(destination),
        "active_pairs": active,
        "active_pair_count": len(active),
        "parseback_field_sha256": restored_sha,
        "parseback_matches_dense_field": True,
    }


def read_rx1_header(archive_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != ["p"]:
            raise Sg2bError("AFR1 archive must contain exactly member p")
        member = archive.read("p")
    if len(member) < RX1_HEADER.size:
        raise Sg2bError("AFR1 member is shorter than the RX1M header")
    magic, version, codec, table_mode, reserved, hpac, semantic, carrier = RX1_HEADER.unpack_from(member)
    observed = {
        "magic": magic.decode("ascii", errors="replace"),
        "version": version,
        "codec": codec,
        "table_mode": table_mode,
        "reserved": reserved,
        "hpac_bytes": hpac,
        "semantic_bytes": semantic,
        "carrier_bytes": carrier,
    }
    if observed != RX1_EXPECTED:
        raise Sg2bError(f"AFR1 RX1M header drift: {observed} != {RX1_EXPECTED}")
    return observed


def verify_runtime_surface(runtime: Path) -> dict[str, dict[str, Any]]:
    """Verify every load-bearing copied file that receiver repinning cannot change."""
    verified = {}
    for relative in RUNTIME_INVARIANT_SOURCE_FILES:
        path = runtime / relative
        if not path.is_file():
            raise Sg2bError(f"staged runtime is missing pinned source: {path}")
        verified[relative] = require_file(
            path,
            path.stat().st_size,
            PINNED_SOURCE_FILES[relative],
        )
    return verified


def stage_runtime(source: Path, archive: Path, destination: Path) -> dict[str, Any]:
    """Copy a runtime, stage one archive, and derive its receiver pin from disk."""
    if destination.exists():
        verdict = check_pin_consistency(destination)
        if verdict.verdict != CONSISTENT:
            raise Sg2bError(f"existing runtime is pin-inconsistent: {verdict.summary()}")
        if sha256_file(destination / "archive.zip") != sha256_file(archive):
            raise Sg2bError(f"existing runtime names different archive bytes: {destination}")
        return {
            "runtime": str(destination),
            "archive": file_fact(destination / "archive.zip"),
            "pin_consistency": verdict.verdict,
            "pinned_surface": verify_runtime_surface(destination),
            "resumed": True,
        }
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        raise Sg2bError(f"partial runtime exists; inspect before retry: {temporary}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, temporary, copy_function=shutil.copy2)
    shutil.copy2(archive, temporary / "archive.zip")
    repin = repin_receiver(temporary)
    verdict = check_pin_consistency(temporary)
    if verdict.verdict != CONSISTENT:
        raise Sg2bError(f"repinned runtime refused: {verdict.summary()}")
    os.replace(temporary, destination)
    return {
        "runtime": str(destination),
        "archive": file_fact(destination / "archive.zip"),
        "pin_consistency": verdict.verdict,
        "pinned_surface": verify_runtime_surface(destination),
        "repin_changed": repin.changed,
        "resumed": False,
    }


def jg2_command(stage: str, proposal: Proposal | None = None) -> list[str]:
    command = [
        str(REPO / ".venv" / "bin" / "python"),
        str(JG2),
        "--stage",
        stage,
        "--store",
        str(REENCODE_STORE),
        "--runtime-root",
        str(BASE_RUNTIME),
        "--pointer-archive",
        str(SOURCE_ARCHIVE),
        "--expect-pointer-sha256",
        AFR1_ARCHIVE_SHA256,
        "--tokens",
        str(BASE_FIELD),
        "--frames",
        str(N),
        "--checkpoint-every",
        "25",
        "--resume",
    ]
    if proposal is not None:
        command.extend(
            [
                "--edits",
                str(STORE / "retained" / "overlays" / f"{proposal.key}.npz"),
                "--tag",
                f"sg2b_{proposal.key}",
            ]
        )
    return command


def run_checked(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=REPO, check=False)
    if completed.returncode:
        raise Sg2bError(f"command failed rc={completed.returncode}: {command}")


def stage_prepare() -> dict[str, Any]:
    STORE.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(STORE).free
    if free < MINIMUM_FREE_BYTES:
        raise Sg2bError(f"storage preflight: {free} B free < {MINIMUM_FREE_BYTES} B")
    archive = require_file(SOURCE_ARCHIVE, AFR1_ARCHIVE_BYTES, AFR1_ARCHIVE_SHA256)
    pin = check_pin_consistency(SOURCE_RUNTIME)
    if pin.verdict != CONSISTENT:
        raise Sg2bError(f"AFR1 source runtime is pin-inconsistent: {pin.summary()}")
    source_files = {}
    for relative, digest in PINNED_SOURCE_FILES.items():
        path = SOURCE_RUNTIME / relative
        if not path.is_file():
            raise Sg2bError(f"missing pinned AFR1 source file: {path}")
        source_files[relative] = require_file(path, path.stat().st_size, digest)
    if sha256_file(JG2) != "e762bead28ab981980aa64161e9104bf1ef5e61c450888edf7777a550c3ac70d":
        raise Sg2bError("pinned JG2 inverse-coder source drifted")
    if sha256_file(FIRE_TOOL) != "afe2f05be10daa44537126cf238256d80e5f565f337498d65593646ad0d963ee":
        raise Sg2bError("canonical fire_local_advisory source drifted")
    rows = []
    for proposal in PROPOSALS:
        field = require_file(proposal.path, FIELD_BYTES, proposal.sha256)
        changed = derive_changed_sites(BASE_FIELD, proposal.path, expected_bytes=FIELD_BYTES)
        if changed != proposal.changed_sites:
            raise Sg2bError(
                f"{proposal.key} changed-site count drift: {changed} != {proposal.changed_sites}"
            )
        overlay = None
        if proposal.key != "p00":
            overlay = materialize_overlay(
                BASE_FIELD,
                proposal.path,
                STORE / "retained" / "overlays" / f"{proposal.key}.npz",
            )
        rows.append(
            {
                "key": proposal.key,
                "proposal_id": proposal.proposal_id,
                "field": field,
                "changed_sites_derived": changed,
                "denominator_sites": FIELD_BYTES,
                "overlay": overlay,
            }
        )
    runtime = stage_runtime(SOURCE_RUNTIME, SOURCE_ARCHIVE, BASE_RUNTIME)
    native_identity = (
        VERTIGO_ROOT
        / "ddm_afr1_tile48_receiver_identity"
        / "measurement_v1"
        / "NATIVE_IDENTITY.json"
    )
    identity = json.loads(native_identity.read_text())
    checks = identity.get("checks", {})
    if (
        identity.get("status") != "PASS"
        or identity.get("candidate_archive", {}).get("sha256") != AFR1_ARCHIVE_SHA256
        or identity.get("result", {}).get("decoded_token_sha256") != BASE_FIELD_SHA256
        or not all(checks.get(key) for key in ("full_n600", "raw_size", "token_field_identity"))
    ):
        raise Sg2bError("retained AFR1 full-receiver identity no longer proves the p00 object")
    null_receipt = {
        "schema": "ddm_sg2b_null_identity.v1",
        "axis": "[reused measured contest-CUDA components plus byte-identical receiver proof]",
        "score_claim": False,
        "fresh_scorer_run": False,
        "field": rows[0]["field"],
        "archive": archive,
        "runtime": runtime,
        "retained_full_receiver_identity": file_fact(native_identity),
        "receiver_denominator": {
            "pairs": 600,
            "tokens": FIELD_BYTES,
            "raw_bytes": 3_662_409_600,
            "raw_bytes_differing": 0,
        },
        "expected_advisory_identity_gate": {
            "d_seg": AFR1_D_SEG,
            "d_pose": AFR1_D_POSE,
            "comparison": "exact numeric equality",
            "failure_disposition": "STOP_INSTRUMENT_DEFECT",
        },
        "status": "BYTE_IDENTITY_PROVED__FRESH_ADVISORY_GATE_QUEUED",
    }
    atomic_json(STORE / "NULL_IDENTITY.json", null_receipt)
    detached = [
        "nohup",
        "env",
        "SG2B_DETACHED_BATCH=1",
        str(REPO / ".venv" / "bin" / "python"),
        str(Path(__file__).resolve()),
        "--stage",
        "batch",
    ]
    payload = {
        "schema": "ddm_sg2b_prepare.v1",
        "axis": "[macOS-CPU scorer-free build / no score claim]",
        "score_claim": False,
        "storage": {
            "root": str(STORE),
            "observed_free_bytes": free,
            "minimum_free_bytes": MINIMUM_FREE_BYTES,
            "status": "PASS",
            "advisory_attempt_root": str(FIRE_ROOT),
            "advisory_route_reason": "APDataStore lacks one retained 3.66 GB raw plus reserve; full advisory payloads route to Vertigo",
        },
        "archive": archive,
        "rx1m_header": read_rx1_header(SOURCE_ARCHIVE),
        "exact_render_path": {
            "container_parser": source_files["runtime/residual_archive.py"],
            "receiver_orchestrator": source_files["runtime/f26_inflate.py"],
            "renderer": source_files["cpr1/inflate.py"],
            "field_consumer": "cpr1/inflate.py::render_video tokens argument at lines 300-328",
            "route": "RX1M read_residual_archive -> decode_production_tokens -> render_video -> raw uint8",
            "not_a_substitute": True,
        },
        "instruments": {"jg2": file_fact(JG2), "fire_local_advisory": file_fact(FIRE_TOOL)},
        "proposals": rows,
        "p00_runtime": runtime,
        "null_identity_receipt": file_fact(STORE / "NULL_IDENTITY.json"),
        "commands": {
            "control": jg2_command("control"),
            "candidate": {
                proposal.key: jg2_command("encode", proposal)
                for proposal in PROPOSALS
                if proposal.key != "p00"
            },
            "detached_batch_argv_prefix": detached,
            "detached_requirement": "launch with shell redirection, pidfile, nohup, and disown; batch writes BATCH_PROGRESS.json and BATCH_DONE.json",
        },
    }
    atomic_json(STORE / "PREPARE.json", payload)
    return payload


def stage_control() -> dict[str, Any]:
    if not (STORE / "PREPARE.json").is_file():
        raise Sg2bError("prepare must pass before the p00 inverse-coder control")
    run_checked(jg2_command("control"))
    receipt_path = REENCODE_STORE / "retained" / "S1_control_600.json"
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("frames") != N
        or not receipt.get("byte_identical")
        or receipt.get("emitted_sha256") != "5601d6fd792c60c176e7cb7478e6033c4ed9a7e87404582340ed3f50ed60cfe3"
        or receipt.get("emitted_bytes") != 113_411
    ):
        raise Sg2bError("p00 inverse-coder control did not reproduce AFR1 byte-identically")
    payload = {
        "schema": "ddm_sg2b_p00_control.v1",
        "axis": "[macOS-CPU scorer-free exact RC64 control]",
        "score_claim": False,
        "receipt": file_fact(receipt_path),
        "emitted_stream": receipt["stream"],
        "frames": N,
        "tokens": FIELD_BYTES,
        "byte_identical": True,
        "status": "PASS",
    }
    atomic_json(STORE / "P00_CONTROL.json", payload)
    return payload


def stage_compose(proposal: Proposal) -> dict[str, Any]:
    if proposal.key == "p00":
        return stage_control()
    control_path = STORE / "P00_CONTROL.json"
    if not control_path.is_file() or json.loads(control_path.read_text()).get("status") != "PASS":
        raise Sg2bError("p00 control must PASS before composing any changed field")
    run_checked(jg2_command("encode", proposal))
    receipt_path = REENCODE_STORE / "retained" / f"S1_encode_sg2b_{proposal.key}.json"
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("frames") != N
        or receipt.get("tokens_changed") != proposal.changed_sites
        or not receipt.get("delta_trustworthy")
    ):
        raise Sg2bError(f"{proposal.key} JG2 receipt is not a trustworthy n600 composition")
    archive = Path(receipt["candidate_archive"]["path"])
    if file_fact(archive) != receipt["candidate_archive"]:
        raise Sg2bError(f"{proposal.key} candidate archive drifted after JG2 receipt")
    runtime = stage_runtime(BASE_RUNTIME, archive, STORE / "runtimes" / proposal.key)
    payload = {
        "schema": "ddm_sg2b_compose.v1",
        "axis": "[macOS-CPU scorer-free exact field re-encode]",
        "score_claim": False,
        "proposal": proposal.key,
        "field": file_fact(proposal.path),
        "changed_sites": proposal.changed_sites,
        "denominator_sites": FIELD_BYTES,
        "jg2_receipt": file_fact(receipt_path),
        "candidate_archive": receipt["candidate_archive"],
        "runtime": runtime,
        "decoded_equals_proposal_basis": (
            "JG2 p00 control is byte-identical and candidate encode follows the same "
            "receiver trajectory while asserting every emitted frame equals the target"
        ),
        "status": "ADVISORY_READY_RECEIVER_PIN_CONSISTENT",
    }
    atomic_json(STORE / f"COMPOSE_{proposal.key.upper()}.json", payload)
    return payload


def runtime_path(proposal: Proposal) -> Path:
    """Return the staged runtime name; p00 keeps its explicit null-control suffix."""
    return BASE_RUNTIME if proposal.key == "p00" else STORE / "runtimes" / proposal.key


def fire_command(proposal: Proposal) -> list[str]:
    runtime = runtime_path(proposal)
    attempt = FIRE_ROOT / proposal.key
    return [
        str(REPO / ".venv" / "bin" / "python"),
        str(FIRE_TOOL),
        "--runtime-dir",
        str(runtime),
        "--archive",
        str(runtime / "archive.zip"),
        "--attempt-dir",
        str(attempt),
        "--label",
        f"ddm_sg2b_{proposal.key}_n600",
        "--projected-gib",
        "12",
        "--timeout",
        "21600",
    ]


def stage_finalize() -> dict[str, Any]:
    control = STORE / "P00_CONTROL.json"
    if not control.is_file() or json.loads(control.read_text()).get("status") != "PASS":
        raise Sg2bError("finalize requires the passing p00 control")
    rows = []
    for proposal in PROPOSALS:
        runtime = runtime_path(proposal)
        if proposal.key != "p00":
            compose = STORE / f"COMPOSE_{proposal.key.upper()}.json"
            if not compose.is_file():
                raise Sg2bError(f"missing composition receipt: {compose}")
        verdict = check_pin_consistency(runtime)
        if verdict.verdict != CONSISTENT:
            raise Sg2bError(f"{proposal.key} runtime is not advisory-ready: {verdict.summary()}")
        pinned_surface = verify_runtime_surface(runtime)
        command = fire_command(proposal)
        dry = subprocess.run([*command, "--dry-run"], cwd=REPO, text=True, capture_output=True)
        if dry.returncode:
            raise Sg2bError(f"{proposal.key} canonical advisory dry-run refused: {dry.stderr}")
        try:
            dry_manifest = json.loads(dry.stdout)
        except json.JSONDecodeError as error:
            raise Sg2bError(f"{proposal.key} advisory dry-run was not JSON") from error
        dry_path = STORE / "retained" / "dry_runs" / f"{proposal.key}.json"
        atomic_json(dry_path, dry_manifest)
        rows.append(
            {
                "proposal": proposal.key,
                "proposal_id": proposal.proposal_id,
                "field": file_fact(proposal.path),
                "changed_sites": proposal.changed_sites,
                "denominator_sites": FIELD_BYTES,
                "compose_route": (
                    "byte-identical AFR1 runtime/archive plus full-n600 JG2 inverse-coder control"
                    if proposal.key == "p00"
                    else "receiver-closed archive via JG2 lossless re-encode and derived receiver repin"
                ),
                "runtime": str(runtime),
                "archive": file_fact(runtime / "archive.zip"),
                "pin_consistency": verdict.verdict,
                "pinned_runtime_surface": pinned_surface,
                "canonical_advisory_dry_run": file_fact(dry_path),
                "disposition": "GATE_NULL_IDENTITY" if proposal.key == "p00" else "QUEUED_MAIN_ADVISORY",
                "owner": "MAIN sole scorer-lane router",
                "consumer_store": str(FIRE_ROOT / proposal.key),
                "fire_trigger": (
                    "MAIN free scorer slot, no duplicate active lane, and no prior SG2B row active; "
                    "p00 first; p01/p02/p03 require p00 exact d_seg/d_pose identity"
                ),
                "argv": command,
            }
        )
    order = {
        "schema": "ddm_sg2b_main_fire_order.v1",
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "owner": "MAIN sole scorer-lane router",
        "ordering": ["p00", "p01", "p02", "p03"],
        "identity_gate": {
            "proposal": "p00",
            "required_d_seg": AFR1_D_SEG,
            "required_d_pose": AFR1_D_POSE,
            "comparison": "exact numeric equality",
            "on_failure": "STOP_INSTRUMENT_DEFECT; do not fire p01/p02/p03",
        },
        "joint_consumer": {
            "store": str(SFP1_ROOT / "HANDOFF.json"),
            "operation": "MAIN joins advisory delta d_seg/d_pose with RXC1 exact_delta bytes after gate 1",
            "promotion": "none; advisory rows are non-authority",
        },
        "rows": rows,
    }
    atomic_json(STORE / "MAIN_FIRE_ORDER.json", order)
    build_done = {
        "schema": "ddm_sg2b_build_done.v1",
        "status": "PASS",
        "score_claim": False,
        "scorer_ran": False,
        "modal_ran": False,
        "proposal_order": ["p00", "p01", "p02", "p03"],
        "controls": {
            "p00_inverse_coder_byte_identity": True,
            "field_sha_and_changed_site_denominators": True,
            "canonical_advisory_dry_runs": 4,
        },
        "fire_order": file_fact(STORE / "MAIN_FIRE_ORDER.json"),
        "next_owner": "MAIN sole scorer-lane router",
    }
    atomic_json(STORE / "BUILD_DONE.json", build_done)
    manifest_entries = []
    manifest_path = STORE / "MANIFEST.json"
    for path in sorted(STORE.rglob("*")):
        if path.is_file() and path != manifest_path and not path.name.endswith(".partial"):
            manifest_entries.append(
                {
                    "path": str(path.relative_to(STORE)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest = {
        "schema": "ddm_sg2b_manifest.v1",
        "root": str(STORE),
        "entries": manifest_entries,
        "total_bytes": sum(int(item["bytes"]) for item in manifest_entries),
    }
    atomic_json(manifest_path, manifest)
    return order


def stage_batch() -> dict[str, Any]:
    if os.environ.get("SG2B_DETACHED_BATCH") != "1":
        raise Sg2bError("batch is projected >30 minutes and must use the retained detached launcher")
    stages: list[str] = []
    if not (STORE / "PREPARE.json").is_file():
        stage_prepare()
    if not (STORE / "P00_CONTROL.json").is_file():
        stage_control()
    stages.append("p00")
    atomic_json(STORE / "BATCH_PROGRESS.json", {"complete": stages, "status": "RUNNING"})
    for proposal in PROPOSALS[1:]:
        receipt = STORE / f"COMPOSE_{proposal.key.upper()}.json"
        if not receipt.is_file():
            stage_compose(proposal)
        stages.append(proposal.key)
        atomic_json(STORE / "BATCH_PROGRESS.json", {"complete": stages, "status": "RUNNING"})
    order = stage_finalize()
    done = {
        "schema": "ddm_sg2b_batch_done.v1",
        "status": "PASS",
        "complete": stages,
        "fire_order": file_fact(STORE / "MAIN_FIRE_ORDER.json"),
        "manifest": file_fact(STORE / "MANIFEST.json"),
        "scorer_ran": False,
    }
    atomic_json(STORE / "BATCH_DONE.json", done)
    return {**done, "order": order["ordering"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage",
        required=True,
        choices=("prepare", "control", "compose", "finalize", "batch"),
    )
    parser.add_argument("--proposal", choices=tuple(BY_KEY), help="compose-stage proposal")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "compose":
        if args.proposal is None:
            raise SystemExit("--stage compose requires --proposal")
        payload = stage_compose(BY_KEY[args.proposal])
    elif args.proposal is not None:
        raise SystemExit("--proposal is valid only with --stage compose")
    else:
        payload = {
            "prepare": stage_prepare,
            "control": stage_control,
            "finalize": stage_finalize,
            "batch": stage_batch,
        }[args.stage]()
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
