#!/usr/bin/env python3
"""Compare public PR79 inflate custody with robust_current on identical bytes.

This is a local forensic tool. It does not run the contest scorer and it does
not dispatch remote jobs. The default path compares ZIP/member custody,
single-payload slicing, decoded runtime-member hashes, action-record hashes,
pose hashes, and runtime source manifests. Optional raw-output comparison can
run both inflate scripts against a caller-provided file list.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PR79_CHECKOUT = Path("/tmp/pact_pr79_inspect")
DEFAULT_OUTPUT_JSON = (
    REPO_ROOT
    / "experiments/results/pr79_runtime_parity_20260503_worker/pr79_runtime_parity.json"
)
DEFAULT_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr79_action_dictionary_repack_v2_20260503_codex/"
    / "pr79_s2_fixed_adaptive_actions/archive.zip"
)
FALLBACK_ARCHIVE = (
    REPO_ROOT / "experiments/results/top_submission_reverse_engineering_20260503_pr79/archive.zip"
)
PR79_REPO_URL = "https://github.com/commaai/comma_video_compression_challenge.git"
PR79_SUBMISSION = "submissions/qpose14_r55_segactions_minp"
PUBLIC_MASK_LEN = 219_472
PUBLIC_MODEL_LEN = 55_756
PUBLIC_POSE_BR_LEN = 898
PUBLIC_REPORTED_BODY_SCORE = 0.31372571308675656
ROBUST_S2_EXACT_T4_SCORE = 0.31453355357318635


class ParityError(ValueError):
    """Raised when byte custody or runtime parsing fails closed."""


@dataclass(frozen=True)
class RuntimeProfile:
    label: str
    ok: bool
    payload_format: str | None
    decoded_members: dict[str, bytes]
    charged_members: dict[str, bytes]
    member_meta: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    error: str | None = None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(payload))


def safe_zip_member(name: str) -> str:
    parts = Path(name).parts
    if (
        not name
        or name.startswith("/")
        or "\\" in name
        or "\x00" in name
        or name.startswith(".")
        or len(parts) != 1
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ParityError(f"unsafe ZIP member path: {name!r}")
    return name


def archive_inventory(archive: Path) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    seen: set[str] = set()
    inventory: list[dict[str, Any]] = []
    members: dict[str, bytes] = {}
    with zipfile.ZipFile(archive, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = safe_zip_member(info.filename)
            if name in seen:
                raise ParityError(f"duplicate archive member: {name}")
            seen.add(name)
            data = zf.read(info)
            if len(data) != info.file_size:
                raise ParityError(f"ZIP member size drift for {name}")
            members[name] = data
            inventory.append(
                {
                    "name": name,
                    "file_size": int(info.file_size),
                    "compress_size": int(info.compress_size),
                    "compress_type": int(info.compress_type),
                    "sha256": sha256_bytes(data),
                    "crc": f"{info.CRC:08x}",
                }
            )
    if not inventory:
        raise ParityError(f"archive has no file members: {archive}")
    return inventory, members


def extract_archive(archive: Path, out_dir: Path) -> None:
    inventory, _members = archive_inventory(archive)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        for item in inventory:
            target = out_dir / item["name"]
            target.write_bytes(zf.read(item["name"]))


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ParityError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_cmd(cmd: Sequence[str], *, cwd: Path | None = None, timeout: int = 300) -> str:
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed rc={proc.returncode}: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout.strip()


def ensure_pr79_checkout(path: Path, *, repo_url: str = PR79_REPO_URL) -> Path:
    """Use an existing PR79 checkout or fetch PR #79 into /tmp."""

    if path.exists():
        inflate = path / PR79_SUBMISSION / "inflate.py"
        if not inflate.exists():
            raise ParityError(f"PR79 checkout exists but missing {inflate}")
        return path
    if path.parent != Path("/tmp"):
        raise ParityError(f"refusing to clone PR79 outside /tmp: {path}")
    path.mkdir(parents=True, exist_ok=False)
    run_cmd(["git", "init", "-q"], cwd=path)
    run_cmd(["git", "remote", "add", "origin", repo_url], cwd=path)
    run_cmd(["git", "fetch", "--depth=1", "origin", "refs/pull/79/head"], cwd=path)
    run_cmd(["git", "checkout", "-q", "--detach", "FETCH_HEAD"], cwd=path)
    return path


def git_commit(path: Path) -> str | None:
    try:
        return run_cmd(["git", "rev-parse", "HEAD"], cwd=path)
    except Exception:
        return None


def source_file_manifest(paths: Mapping[str, Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label, path in paths.items():
        if path.exists():
            out.append(
                {
                    "label": label,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_path(path),
                }
            )
        else:
            out.append({"label": label, "path": str(path), "missing": True})
    return out


def read_payload_from_members(members: Mapping[str, bytes]) -> tuple[str, bytes]:
    present = [name for name in ("p", "renderer_payload.bin", "renderer_payload.bin.br") if name in members]
    if len(present) != 1:
        raise ParityError(
            "expected exactly one renderer payload container among p, "
            f"renderer_payload.bin, renderer_payload.bin.br; got {present}"
        )
    name = present[0]
    payload = members[name]
    if name.endswith(".br"):
        import brotli

        payload = brotli.decompress(payload)
    return name, payload


def uvarint(value: int) -> bytes:
    if value < 0:
        raise ValueError("uvarint requires a nonnegative value")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def read_uvarint(data: bytes, cursor: int) -> tuple[int, int]:
    shift = 0
    value = 0
    start = cursor
    while cursor < len(data):
        byte = data[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, cursor
        shift += 7
        if shift > 63:
            break
    raise ParityError(f"truncated uvarint at byte {start}")


def decode_qp1_float32(payload: bytes) -> bytes:
    import numpy as np

    if len(payload) < 5 or not payload.startswith(b"QP1"):
        raise ParityError(f"bad QP1 payload head: {payload[:8]!r}")
    first = struct.unpack_from("<H", payload, 3)[0]
    vals = [int(first)]
    cursor = 5
    while cursor < len(payload):
        acc, cursor = read_uvarint(payload, cursor)
        delta = (acc >> 1) ^ -(acc & 1)
        vals.append(vals[-1] + delta)
    q_pose = np.zeros((len(vals), 6), dtype=np.uint16)
    q_pose[:, 0] = np.asarray(vals, dtype=np.uint16)
    pose = np.empty(q_pose.shape, dtype=np.float32)
    pose[:, 0] = q_pose[:, 0].astype(np.float32) / 512.0 + 20.0
    pose[:, 1:] = q_pose[:, 1:].view(np.int16).astype(np.float32) / 2048.0
    return pose.astype("<f4", copy=False).tobytes()


def action_record_stats(raw: bytes) -> dict[str, Any]:
    payload = raw
    grid_header: dict[str, Any] | None = None
    if payload.startswith(b"TG1"):
        if len(payload) < 5:
            raise ParityError("TG1 action header is truncated")
        grid_header = {"magic": "TG1", "tile_size": int.from_bytes(payload[3:5], "little")}
        payload = payload[5:]
    record_size = 5 if payload.startswith(b"TA5") else None
    if payload.startswith(b"TA5"):
        payload = payload[3:]
    if record_size is None:
        if len(payload) % 4 == 0:
            record_size = 4
        elif len(payload) % 5 == 0:
            record_size = 5
        else:
            raise ParityError(f"action records are not raw4/raw5 aligned: {len(payload)}")
    if len(payload) % record_size != 0:
        raise ParityError("action record length mismatch")
    records = []
    for offset in range(0, len(payload), record_size):
        pair = int.from_bytes(payload[offset : offset + 2], "little")
        if record_size == 4:
            tile = payload[offset + 2]
            action = payload[offset + 3]
        else:
            tile = int.from_bytes(payload[offset + 2 : offset + 4], "little")
            action = payload[offset + 4]
        records.append((pair, tile, action))
    pairs = [record[0] for record in records]
    tiles = [record[1] for record in records]
    actions = [record[2] for record in records]
    return {
        "record_size": record_size,
        "record_count": len(records),
        "raw_sha256": sha256_bytes(raw),
        "canonical_records_sha256": sha256_bytes(payload),
        "first_records": records[:8],
        "pair_min": min(pairs) if pairs else None,
        "pair_max": max(pairs) if pairs else None,
        "unique_pairs": len(set(pairs)),
        "unique_tiles": len(set(tiles)),
        "unique_actions": len(set(actions)),
        "grid_header": grid_header,
    }


def decode_public_pr79_action_records(data: bytes) -> bytes:
    import brotli

    raw = brotli.decompress(data)
    records: list[tuple[int, int, int]] = []
    if raw.startswith(b"SG2") or (len(raw) % 4 != 0 and len(raw) % 5 != 0):
        cursor = 3 if raw.startswith(b"SG2") else 0
        while cursor < len(raw):
            tile, cursor = read_uvarint(raw, cursor)
            count, cursor = read_uvarint(raw, cursor)
            frame = 0
            for idx in range(count):
                delta, cursor = read_uvarint(raw, cursor)
                frame = delta if idx == 0 else frame + delta
                if cursor >= len(raw):
                    raise ParityError("public action payload ended inside record")
                action = raw[cursor]
                cursor += 1
                records.append((frame, tile, action))
    elif len(raw) % 4 == 0:
        for offset in range(0, len(raw), 4):
            records.append(
                (
                    int.from_bytes(raw[offset : offset + 2], "little"),
                    raw[offset + 2],
                    raw[offset + 3],
                )
            )
    elif len(raw) % 5 == 0:
        for offset in range(0, len(raw), 5):
            records.append(
                (
                    int.from_bytes(raw[offset : offset + 2], "little"),
                    int.from_bytes(raw[offset + 2 : offset + 4], "little"),
                    raw[offset + 4],
                )
            )
    else:
        raise ParityError(f"unsupported public action payload length: {len(raw)}")
    use_raw5 = any(tile > 255 for _pair, tile, _action in records)
    out = bytearray()
    for pair, tile, action in records:
        out += int(pair).to_bytes(2, "little")
        if use_raw5:
            out += int(tile).to_bytes(2, "little")
        else:
            if tile > 255:
                raise ParityError(f"tile id outside raw4 range: {tile}")
            out.append(int(tile))
        out.append(int(action))
    return bytes(out)


def public_pr79_profile(payload: bytes) -> RuntimeProfile:
    import brotli

    diagnostics: dict[str, Any] = {"payload_bytes": len(payload), "payload_sha256": sha256_bytes(payload)}
    try:
        raw_slices: dict[str, bytes]
        payload_format: str
        if len(payload) == 276_641:
            payload_format = "public_pr79_legacy_fixed_276641"
            raw_slices = {
                "masks.mkv": payload[:PUBLIC_MASK_LEN],
                "renderer.bin": payload[PUBLIC_MASK_LEN:275_506],
                "seg_tile_actions.bin": payload[275_506:275_742],
                "optimized_poses.qp1": payload[275_742:],
            }
        elif payload.startswith(b"P3"):
            if len(payload) < 10:
                raise ParityError("P3 payload is too short")
            mask_len, model_len, actions_len = struct.unpack_from("<IHH", payload, 2)
            cursor = 10
            payload_format = "public_pr79_p3"
            raw_slices = {
                "masks.mkv": payload[cursor : cursor + mask_len],
                "renderer.bin": payload[cursor + mask_len : cursor + mask_len + model_len],
                "seg_tile_actions.bin": payload[
                    cursor + mask_len + model_len : cursor + mask_len + model_len + actions_len
                ],
                "optimized_poses.qp1": payload[cursor + mask_len + model_len + actions_len :],
            }
        elif payload.startswith(b"P2"):
            if len(payload) < 8:
                raise ParityError("P2 payload is too short")
            mask_len, model_len = struct.unpack_from("<IH", payload, 2)
            cursor = 8
            payload_format = "public_pr79_p2_no_actions"
            raw_slices = {
                "masks.mkv": payload[cursor : cursor + mask_len],
                "renderer.bin": payload[cursor + mask_len : cursor + mask_len + model_len],
                "optimized_poses.qp1": payload[cursor + mask_len + model_len :],
            }
        elif len(payload) in (276_574, 276_749) or 276_900 <= len(payload) <= 278_000:
            actions_len = len(payload) - PUBLIC_MASK_LEN - PUBLIC_MODEL_LEN - PUBLIC_POSE_BR_LEN
            payload_format = "public_pr79_fixed_minp_window"
            raw_slices = {
                "masks.mkv": payload[:PUBLIC_MASK_LEN],
                "renderer.bin": payload[PUBLIC_MASK_LEN : PUBLIC_MASK_LEN + PUBLIC_MODEL_LEN],
                "seg_tile_actions.bin": payload[
                    PUBLIC_MASK_LEN
                    + PUBLIC_MODEL_LEN : PUBLIC_MASK_LEN
                    + PUBLIC_MODEL_LEN
                    + actions_len
                ],
                "optimized_poses.qp1": payload[-PUBLIC_POSE_BR_LEN:],
            }
        else:
            mask_len = PUBLIC_MASK_LEN
            if 276_430 <= len(payload) <= 276_470:
                model_len = 56_093
            elif 276_550 <= len(payload) <= 276_610:
                model_len = 56_221
            elif 278_100 <= len(payload) <= 278_130:
                model_len = 57_757
            elif 277_400 <= len(payload) <= 277_430:
                model_len = 57_053
            elif 277_350 <= len(payload) <= 277_399:
                model_len = 57_031
            elif len(payload) == 281_240:
                model_len = 60_880
            else:
                model_len = 61_147
            payload_format = "public_pr79_fallback_fixed_guess"
            raw_slices = {
                "masks.mkv": payload[:mask_len],
                "renderer.bin": payload[mask_len : mask_len + model_len],
                "optimized_poses.qp1": payload[mask_len + model_len :],
            }

        members = {
            "masks.mkv": brotli.decompress(raw_slices["masks.mkv"]),
            "renderer.bin": brotli.decompress(raw_slices["renderer.bin"]),
        }
        if "optimized_poses.qp1" in raw_slices:
            members["optimized_poses.qp1"] = brotli.decompress(raw_slices["optimized_poses.qp1"])
            diagnostics["pose_float32_sha256"] = sha256_bytes(
                decode_qp1_float32(members["optimized_poses.qp1"])
            )
        if "seg_tile_actions.bin" in raw_slices:
            members["seg_tile_actions.bin"] = decode_public_pr79_action_records(
                raw_slices["seg_tile_actions.bin"]
            )
            diagnostics["seg_tile_action_stats"] = action_record_stats(
                members["seg_tile_actions.bin"]
            )
        member_meta = []
        for name, charged in raw_slices.items():
            decoded_name = "optimized_poses.qp1" if name == "optimized_poses.bin" else name
            decoded = members.get(decoded_name)
            member_meta.append(
                {
                    "name": decoded_name,
                    "charged_bytes": len(charged),
                    "charged_sha256": sha256_bytes(charged),
                    "decoded_bytes": len(decoded) if decoded is not None else None,
                    "decoded_sha256": sha256_bytes(decoded) if decoded is not None else None,
                }
            )
        return RuntimeProfile(
            label="public_pr79",
            ok=True,
            payload_format=payload_format,
            decoded_members=members,
            charged_members=raw_slices,
            member_meta=member_meta,
            diagnostics=diagnostics,
        )
    except Exception as exc:
        diagnostics["failure_class"] = exc.__class__.__name__
        return RuntimeProfile(
            label="public_pr79",
            ok=False,
            payload_format=diagnostics.get("payload_format"),
            decoded_members={},
            charged_members={},
            member_meta=[],
            diagnostics=diagnostics,
            error=str(exc),
        )


def robust_current_profile(extracted_archive_dir: Path) -> RuntimeProfile:
    module_path = REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py"
    module = load_module(module_path, "robust_unpack_renderer_payload_for_pr79_parity")
    diagnostics: dict[str, Any] = {}
    try:
        payload = module._read_payload_bytes(extracted_archive_dir)
        header, members = module._parse_payload(payload)
        diagnostics["payload_bytes"] = len(payload)
        diagnostics["payload_sha256"] = sha256_bytes(payload)
        diagnostics["header"] = header
        decoded_members = {str(name): data for name, data in members.items()}
        if "optimized_poses.qp1" in decoded_members:
            diagnostics["pose_float32_sha256"] = sha256_bytes(
                decode_qp1_float32(decoded_members["optimized_poses.qp1"])
            )
        if "seg_tile_actions.bin" in decoded_members:
            diagnostics["seg_tile_action_stats"] = action_record_stats(
                decoded_members["seg_tile_actions.bin"]
            )
        charged_members: dict[str, bytes] = {}
        for meta in header.get("members", []):
            name = str(meta.get("name"))
            charged_members[name] = b""
        return RuntimeProfile(
            label="robust_current",
            ok=True,
            payload_format=header.get("payload_format"),
            decoded_members=decoded_members,
            charged_members=charged_members,
            member_meta=list(header.get("members", [])),
            diagnostics=diagnostics,
        )
    except Exception as exc:
        diagnostics["failure_class"] = exc.__class__.__name__
        return RuntimeProfile(
            label="robust_current",
            ok=False,
            payload_format=None,
            decoded_members={},
            charged_members={},
            member_meta=[],
            diagnostics=diagnostics,
            error=str(exc),
        )


def profile_to_json(profile: RuntimeProfile) -> dict[str, Any]:
    return {
        "label": profile.label,
        "ok": profile.ok,
        "payload_format": profile.payload_format,
        "error": profile.error,
        "member_meta": profile.member_meta,
        "decoded_members": {
            name: {"bytes": len(data), "sha256": sha256_bytes(data)}
            for name, data in sorted(profile.decoded_members.items())
        },
        "diagnostics": profile.diagnostics,
    }


def compare_profiles(public: RuntimeProfile, robust: RuntimeProfile) -> dict[str, Any]:
    names = sorted(set(public.decoded_members) | set(robust.decoded_members))
    member_comparisons = []
    parity_gap_classes: list[str] = []
    if not public.ok:
        parity_gap_classes.append("public_pr79_parse_or_decode_failed")
    if not robust.ok:
        parity_gap_classes.append("robust_current_parse_or_decode_failed")
    for name in names:
        lhs = public.decoded_members.get(name)
        rhs = robust.decoded_members.get(name)
        item: dict[str, Any] = {"name": name, "public_present": lhs is not None, "robust_present": rhs is not None}
        if lhs is not None:
            item["public_bytes"] = len(lhs)
            item["public_sha256"] = sha256_bytes(lhs)
        if rhs is not None:
            item["robust_bytes"] = len(rhs)
            item["robust_sha256"] = sha256_bytes(rhs)
        if lhs is not None and rhs is not None:
            item["exact_equal"] = lhs == rhs
            if lhs != rhs:
                parity_gap_classes.append(f"decoded_member_mismatch:{name}")
        else:
            parity_gap_classes.append(f"decoded_member_presence_mismatch:{name}")
        member_comparisons.append(item)

    public_pose = public.diagnostics.get("pose_float32_sha256")
    robust_pose = robust.diagnostics.get("pose_float32_sha256")
    pose_float32_equal = public_pose is not None and public_pose == robust_pose
    if public_pose is not None or robust_pose is not None:
        if not pose_float32_equal:
            parity_gap_classes.append("pose_float32_decode_hash_mismatch")

    public_actions = public.diagnostics.get("seg_tile_action_stats", {})
    robust_actions = robust.diagnostics.get("seg_tile_action_stats", {})
    action_records_equal = (
        public_actions.get("canonical_records_sha256") is not None
        and public_actions.get("canonical_records_sha256")
        == robust_actions.get("canonical_records_sha256")
    )
    if public_actions or robust_actions:
        if not action_records_equal:
            parity_gap_classes.append("seg_tile_action_record_hash_mismatch")

    return {
        "decoded_member_comparisons": member_comparisons,
        "pose_float32_sha256": {
            "public": public_pose,
            "robust": robust_pose,
            "exact_equal": pose_float32_equal,
        },
        "seg_tile_actions": {
            "public": public_actions,
            "robust": robust_actions,
            "canonical_records_exact_equal": action_records_equal,
        },
        "parity_gap_classes": sorted(set(parity_gap_classes)),
        "decoded_member_hashes_all_equal": all(
            item.get("exact_equal") is True for item in member_comparisons
        )
        if member_comparisons
        else False,
    }


def raw_output_hashes(out_dir: Path) -> dict[str, dict[str, Any]]:
    hashes = {}
    for path in sorted(out_dir.glob("*.raw")):
        hashes[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_path(path)}
    return hashes


def run_raw_parity(
    *,
    archive: Path,
    pr79_checkout: Path,
    file_list: Path,
    timeout: int,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="pr79_raw_parity_") as tmp:
        root = Path(tmp)
        public_archive = root / "public_archive"
        robust_archive = root / "robust_archive"
        public_out = root / "public_out"
        robust_out = root / "robust_out"
        extract_archive(archive, public_archive)
        extract_archive(archive, robust_archive)
        public_script = pr79_checkout / PR79_SUBMISSION / "inflate.py"
        robust_script = REPO_ROOT / "submissions/robust_current/inflate.sh"
        public_cmd = [sys.executable, str(public_script), str(public_archive), str(public_out), str(file_list)]
        robust_cmd = ["bash", str(robust_script), str(robust_archive), str(robust_out), str(file_list)]
        public_stdout = run_cmd(public_cmd, cwd=pr79_checkout, timeout=timeout)
        env = os.environ.copy()
        env.setdefault("PYTHON_INFLATE", "renderer")
        env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
        proc = subprocess.run(
            robust_cmd,
            cwd=str(REPO_ROOT),
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"robust inflate failed rc={proc.returncode}: {' '.join(robust_cmd)}\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        public_hashes = raw_output_hashes(public_out)
        robust_hashes = raw_output_hashes(robust_out)
        names = sorted(set(public_hashes) | set(robust_hashes))
        comparisons = [
            {
                "name": name,
                "public": public_hashes.get(name),
                "robust": robust_hashes.get(name),
                "exact_equal": public_hashes.get(name) == robust_hashes.get(name),
            }
            for name in names
        ]
        return {
            "commands": {
                "public": public_cmd,
                "robust": robust_cmd,
            },
            "public_stdout_tail": public_stdout[-4000:],
            "robust_stdout_tail": proc.stdout[-4000:],
            "robust_stderr_tail": proc.stderr[-4000:],
            "comparisons": comparisons,
            "all_exact_equal": all(item["exact_equal"] for item in comparisons),
        }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    archive = Path(args.archive)
    if not archive.exists():
        raise FileNotFoundError(f"archive not found: {archive}")
    pr79_checkout = ensure_pr79_checkout(Path(args.pr79_checkout))
    public_runtime_files = {
        "inflate.py": pr79_checkout / PR79_SUBMISSION / "inflate.py",
        "inflate.sh": pr79_checkout / PR79_SUBMISSION / "inflate.sh",
    }
    robust_runtime_files = {
        "inflate.sh": REPO_ROOT / "submissions/robust_current/inflate.sh",
        "inflate_renderer.py": REPO_ROOT / "submissions/robust_current/inflate_renderer.py",
        "unpack_renderer_payload.py": REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py",
    }
    inventory, members = archive_inventory(archive)
    container_name, payload = read_payload_from_members(members)
    public_profile = public_pr79_profile(payload)
    with tempfile.TemporaryDirectory(prefix="pr79_runtime_parity_") as tmp:
        extracted = Path(tmp) / "archive"
        extract_archive(archive, extracted)
        robust_profile = robust_current_profile(extracted)

    comparison = compare_profiles(public_profile, robust_profile)
    raw_parity = None
    if args.run_raw_parity:
        if args.file_list is None:
            raise ParityError("--run-raw-parity requires --file-list")
        raw_parity = run_raw_parity(
            archive=archive,
            pr79_checkout=pr79_checkout,
            file_list=Path(args.file_list),
            timeout=int(args.raw_timeout),
        )

    return {
        "schema": "pr79_runtime_parity_forensics_v1",
        "tool": "experiments/compare_pr79_runtime_parity.py",
        "score_context_not_recomputed": {
            "public_pr79_body_formula_score": PUBLIC_REPORTED_BODY_SCORE,
            "robust_current_s2_exact_t4_score": ROBUST_S2_EXACT_T4_SCORE,
            "delta_robust_minus_public_body": ROBUST_S2_EXACT_T4_SCORE
            - PUBLIC_REPORTED_BODY_SCORE,
            "note": "This tool does not run upstream/evaluate.py; exact CUDA auth eval remains score truth.",
        },
        "archive": {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": sha256_path(archive),
            "payload_container": container_name,
            "payload_bytes": len(payload),
            "payload_sha256": sha256_bytes(payload),
            "inventory": inventory,
        },
        "runtime_manifests": {
            "public_pr79": {
                "checkout": str(pr79_checkout),
                "commit": git_commit(pr79_checkout),
                "files": source_file_manifest(public_runtime_files),
            },
            "robust_current": {
                "repo_root": str(REPO_ROOT),
                "commit": git_commit(REPO_ROOT),
                "files": source_file_manifest(robust_runtime_files),
            },
        },
        "profiles": {
            "public_pr79": profile_to_json(public_profile),
            "robust_current": profile_to_json(robust_profile),
        },
        "comparison": comparison,
        "raw_output_parity": raw_parity,
        "forensic_interpretation": {
            "public_can_parse_archive_bytes": public_profile.ok,
            "robust_can_parse_archive_bytes": robust_profile.ok,
            "decoded_runtime_members_equal": comparison["decoded_member_hashes_all_equal"],
            "raw_output_parity_run": raw_parity is not None,
            "known_limits": [
                "No scorer invocation and no remote GPU dispatch.",
                "Member/action/pose parity cannot prove SegNet/PoseNet score parity.",
                "If raw_output_parity is omitted, rendered-byte parity remains unmeasured by this run.",
            ],
        },
    }


def default_archive() -> Path:
    return DEFAULT_ARCHIVE if DEFAULT_ARCHIVE.exists() else FALLBACK_ARCHIVE


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=default_archive())
    parser.add_argument("--pr79-checkout", type=Path, default=DEFAULT_PR79_CHECKOUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--run-raw-parity", action="store_true")
    parser.add_argument("--file-list", type=Path, default=None)
    parser.add_argument("--raw-timeout", type=int, default=1800)
    parser.add_argument("--stdout", action="store_true", help="Print JSON instead of writing only to --output-json.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = build_report(args)
    if args.output_json:
        write_json(Path(args.output_json), report)
    if args.stdout or not args.output_json:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
