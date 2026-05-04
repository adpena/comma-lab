#!/usr/bin/env python3
"""Bounded Q-FAITHFUL checkpoint snapshot export and H100 screen scaffold.

This script is intentionally separate from the long Q-FAITHFUL launcher. It can
be run beside an active training job to turn stable checkpoints into deterministic
archive snapshots, then optionally screen those exact bytes through the existing
archive-only CUDA eval wrapper.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any


EXPECTED_MASK_FRAMES = 1200
HALF_FRAME_MASK_FRAMES = EXPECTED_MASK_FRAMES // 2
SNAPSHOT_RUNTIME_CONTRACT_VERSION = "qfaithful_snapshot_runtime_contract_v2"
ZOOM_WARP_ARCHIVE_MEMBER = "zoom_scalars.bin"
QFAITHFUL_RENDERER_ARCHITECTURE = "quantizr_faithful_joint_frame_generator"
QFAITHFUL_EXPORT_CONTRACT_REQUIRED_KEYS = (
    "runtime_contract_version",
    "mask_frame_contract",
    "renderer_zoom_contract",
    "eval_roundtrip_required",
    "profile",
    "promotable_exact_screen",
    "pose_tensor_contract",
    "training_pose_contract",
    "packed_from_ema_shadow",
)

REPO_RUNTIME_SHA_PATHS = (
    "scripts/q_faithful_snapshot_loop.py",
    "experiments/repack_quantizr_faithful_qzs3_archive.py",
    "scripts/remote_archive_only_eval.sh",
    "submissions/robust_current/inflate.sh",
    "submissions/robust_current/inflate_renderer.py",
    "submissions/robust_current/unpack_renderer_payload.py",
    "submissions/robust_current/apply_qzs3_postprocess.py",
    "src/tac/quantizr_qzs3_codec.py",
    "src/tac/quantizr_faithful_export.py",
    "src/tac/quantizr_faithful_renderer.py",
    "src/tac/profiles.py",
)
class SnapshotError(RuntimeError):
    """Classified snapshot failure."""

    def __init__(self, failure_class: str, message: str):
        super().__init__(message)
        self.failure_class = failure_class


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stat_size(path: Path) -> int:
    return path.stat().st_size


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _repo_path(workspace: Path, rel: str) -> Path:
    if rel.startswith("/") or ".." in Path(rel).parts:
        raise SnapshotError("unsafe_runtime_path", f"unsafe runtime path: {rel}")
    return workspace / rel


def source_runtime_shas(workspace: Path) -> dict[str, str]:
    shas: dict[str, str] = {}
    for rel in REPO_RUNTIME_SHA_PATHS:
        path = _repo_path(workspace, rel)
        if not path.is_file():
            raise SnapshotError("missing_runtime_source", f"missing runtime source: {rel}")
        shas[rel] = sha256_file(path)
    return shas


def required_source_sha_env(shas: dict[str, str]) -> str:
    return "\n".join(f"{rel}={sha}" for rel, sha in sorted(shas.items()))


def verify_eval_roundtrip_profile(workspace: Path, profile: str) -> dict[str, Any]:
    sys.path.insert(0, str(workspace / "src"))
    try:
        from tac.profiles import get_profile
    finally:
        try:
            sys.path.remove(str(workspace / "src"))
        except ValueError:
            pass
    cfg = dict(get_profile(profile))
    if cfg.get("eval_roundtrip") is not True:
        raise SnapshotError(
            "eval_roundtrip_not_proven",
            f"profile {profile!r} does not declare eval_roundtrip=True",
        )
    return {
        "profile": profile,
        "eval_roundtrip": True,
        "source": "tac.profiles.get_profile",
    }


def checkpoint_profile_hint(checkpoint: Path) -> str | None:
    """Best-effort profile lookup without requiring CUDA.

    Returns None for legacy/raw state_dict checkpoints. The profile argument is
    still used as the authoritative roundtrip proof in that case.
    """

    try:
        import torch

        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(state, dict):
        return None
    meta = state.get("__meta__") or state.get("arch_meta") or {}
    if isinstance(meta, dict) and isinstance(meta.get("profile"), str):
        return meta["profile"]
    return None


def _checkpoint_meta_candidates(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    candidates: list[dict[str, Any]] = []
    for key in (
        "qfaithful_training_pose_contract",
        "training_pose_contract",
        "snapshot_training_pose_contract",
        "arch_meta",
        "__meta__",
        "meta",
    ):
        value = payload.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    for container_key in ("arch_meta", "__meta__", "meta"):
        container = payload.get(container_key)
        if not isinstance(container, dict):
            continue
        for key in (
            "qfaithful_training_pose_contract",
            "training_pose_contract",
            "snapshot_training_pose_contract",
        ):
            value = container.get(key)
            if isinstance(value, dict):
                candidates.append(value)
    return candidates


def _contract_bool(contract: dict[str, Any], *keys: str) -> bool:
    return any(contract.get(key) is True for key in keys)


def inspect_checkpoint_training_pose_contract(
    checkpoint: Path,
    *,
    deployed_pose_contract: dict[str, Any] | None,
    profile: str,
) -> dict[str, Any]:
    """Require proof that Q-FAITHFUL training used the deployed pose stream."""
    base: dict[str, Any] = {
        "checkpoint_path": str(checkpoint),
        "profile": profile,
        "required_for_qfaithful_successor": True,
        "required_pose_dim": 6,
        "zero_pose_fallback_allowed": False,
        "training_pose_contract_promotable": False,
    }
    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {
            **base,
            "failure_class": "checkpoint_training_pose_contract_unreadable",
            "failure": str(exc),
        }

    deployed_sha = None
    if deployed_pose_contract is not None:
        deployed_sha = deployed_pose_contract.get("sha256")
    for candidate in _checkpoint_meta_candidates(payload):
        contract = candidate
        nested = candidate.get("training_pose_contract")
        if isinstance(nested, dict):
            contract = nested
        pose_dim = contract.get("pose_dim", contract.get("required_pose_dim"))
        pose_sha = contract.get("pose_sha256") or contract.get("pose_source_sha256")
        if not _contract_bool(
            contract,
            "training_uses_nonzero_pose_stream",
            "uses_nonzero_pose_stream",
            "training_uses_deployed_pose_stream",
        ):
            continue
        if contract.get("zero_pose_fallback_allowed") is not False:
            continue
        try:
            pose_dim_ok = int(pose_dim) == 6
        except (TypeError, ValueError):
            pose_dim_ok = False
        if not pose_dim_ok:
            continue
        if not pose_sha:
            continue
        if deployed_sha and pose_sha != deployed_sha:
            return {
                **base,
                "contract": contract,
                "deployed_pose_sha256": deployed_sha,
                "training_pose_sha256": pose_sha,
                "failure_class": "training_pose_sha_mismatch",
            }
        return {
            **base,
            "contract": contract,
            "deployed_pose_sha256": deployed_sha,
            "training_pose_sha256": pose_sha,
            "training_pose_contract_promotable": True,
            "failure_class": None,
        }

    return {
        **base,
        "failure_class": "qfaithful_training_pose_contract_missing",
        "unblock_requirement": (
            "checkpoint metadata must prove pose_dim=6 training consumed the "
            "same nonzero deployed pose stream, include its SHA-256, and set "
            "zero_pose_fallback_allowed=false"
        ),
    }


def select_state_dict(payload: Any, state_source: str) -> tuple[dict[str, Any], str]:
    if not isinstance(payload, dict):
        if state_source not in {"auto", "raw"}:
            raise SnapshotError("checkpoint_state_missing", "checkpoint is not a dict")
        return payload, "raw"  # type: ignore[return-value]

    priority = (
        ("ema_shadow", "ema_shadow"),
        ("model_state_dict", "model_state_dict"),
        ("model", "model"),
        ("state_dict", "state_dict"),
    )
    if state_source != "auto":
        priority = ((state_source, state_source),)
    for key, label in priority:
        value = payload.get(key)
        if isinstance(value, dict):
            return value, label
    if state_source == "auto" and payload and all(hasattr(v, "shape") for v in payload.values()):
        return payload, "raw"
    raise SnapshotError(
        "checkpoint_state_missing",
        f"checkpoint lacks requested state source {state_source!r}",
    )


def export_qfai_renderer(
    checkpoint: Path,
    renderer_bin: Path,
    *,
    state_source: str,
    brotli_quality: int,
    extra_meta: dict[str, Any],
) -> dict[str, Any]:
    import brotli
    import torch

    from tac.quantizr_faithful_export import save_qfai
    from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict, selected_source = select_state_dict(payload, state_source)
    stripped = {
        (key[len("gen.") :] if isinstance(key, str) and key.startswith("gen.") else key): value
        for key, value in state_dict.items()
    }

    model = build_quantizr_faithful_renderer()
    model.load_state_dict(stripped, strict=True)
    model.eval()

    raw_qfai = renderer_bin.with_suffix(".qfai.bin")
    raw_qfai.parent.mkdir(parents=True, exist_ok=True)
    export_meta = {
        **extra_meta,
        "checkpoint_state_source": selected_source,
        "packed_from_ema_shadow": selected_source == "ema_shadow",
    }
    raw_bytes = save_qfai(model, raw_qfai, extra_meta=export_meta)
    raw_payload = raw_qfai.read_bytes()
    renderer_bin.write_bytes(raw_payload)
    compressed_qfai = raw_qfai.with_suffix(raw_qfai.suffix + ".br")
    compressed_qfai.write_bytes(brotli.compress(raw_payload, quality=brotli_quality))
    return {
        "checkpoint_state_source": selected_source,
        "packed_from_ema_shadow": selected_source == "ema_shadow",
        "ema_shadow_required_for_promotable_eval": True,
        "raw_qfai_path": str(raw_qfai),
        "raw_qfai_bytes": raw_bytes,
        "raw_qfai_sha256": sha256_file(raw_qfai),
        "compressed_qfai_sidecar_path": str(compressed_qfai),
        "compressed_qfai_sidecar_bytes": _stat_size(compressed_qfai),
        "compressed_qfai_sidecar_sha256": sha256_file(compressed_qfai),
        "renderer_bin_path": str(renderer_bin),
        "renderer_bin_bytes": _stat_size(renderer_bin),
        "renderer_bin_sha256": sha256_file(renderer_bin),
        "renderer_bin_wire_format": "QFAI",
        "renderer_bin_brotli_compressed": False,
        "brotli_quality": brotli_quality,
    }


def enforce_ema_export_contract(qfai_meta: dict[str, Any], *, allow_live_weight_export: bool) -> None:
    if qfai_meta.get("packed_from_ema_shadow") is True:
        return
    if allow_live_weight_export:
        return
    raise SnapshotError(
        "ema_shadow_export_missing",
        "refusing Q-FAITHFUL eval because export did not pack ema_shadow; "
        f"selected={qfai_meta.get('checkpoint_state_source')!r}",
    )


def build_raw_archive(
    *,
    renderer_bin: Path,
    masks_mkv: Path,
    poses_pt: Path,
    output_archive: Path,
    zoom_warp_path: Path | None = None,
) -> dict[str, Any]:
    for path, failure in (
        (renderer_bin, "missing_renderer_bin"),
        (masks_mkv, "missing_masks_mkv"),
        (poses_pt, "missing_poses_pt"),
    ):
        if not path.is_file():
            raise SnapshotError(failure, f"missing required archive input: {path}")
    output_archive.parent.mkdir(parents=True, exist_ok=True)
    sources = {
        "renderer.bin": renderer_bin,
        "masks.mkv": masks_mkv,
        "optimized_poses.bin": poses_pt,
    }
    if zoom_warp_path is not None:
        if not zoom_warp_path.is_file():
            raise SnapshotError(
                "missing_zoom_warp_geometry",
                f"missing zoom/warp geometry input: {zoom_warp_path}",
            )
        sources[ZOOM_WARP_ARCHIVE_MEMBER] = zoom_warp_path
    with zipfile.ZipFile(output_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for member in ("renderer.bin", "masks.mkv", "optimized_poses.bin", ZOOM_WARP_ARCHIVE_MEMBER):
            if member not in sources:
                continue
            info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            info.create_system = 3
            zf.writestr(
                info,
                sources[member].read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    meta = archive_metadata(output_archive)
    meta["member_order"] = list(sources)
    meta["members"] = {
        member: {
            "member_name": member,
            "source_path": str(path),
            "bytes": _stat_size(path),
            "sha256": sha256_file(path),
        }
        for member, path in sources.items()
    }
    return meta


def archive_metadata(path: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "path": str(path),
        "bytes": _stat_size(path),
        "sha256": sha256_file(path),
    }
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, "r") as zf:
            infos = [info for info in zf.infolist() if not info.is_dir()]
            meta["member_names"] = [info.filename for info in infos]
            meta["members"] = {
                info.filename: {
                    "member_name": info.filename,
                    "bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "sha256": hashlib.sha256(zf.read(info)).hexdigest(),
                }
                for info in infos
            }
    return meta


def validate_repacked_geometry_contract(
    *,
    screen_contract: dict[str, Any],
    repacked_archive_meta: dict[str, Any],
) -> None:
    zoom_contract = screen_contract.get("zoom_warp_geometry") or {}
    if not zoom_contract.get("required_for_half_frame"):
        return
    source = zoom_contract.get("source") or {}
    if not source.get("present"):
        return
    member_name = zoom_contract.get("archive_member_name") or ZOOM_WARP_ARCHIVE_MEMBER
    members = repacked_archive_meta.get("members") or {}
    member = members.get(member_name)
    if not member:
        raise SnapshotError(
            "zoom_warp_geometry_not_preserved_by_repack_contract",
            f"repacked archive missing required zoom/warp geometry member {member_name!r}",
        )
    if member.get("sha256") != source.get("sha256"):
        raise SnapshotError(
            "zoom_warp_geometry_sha_mismatch",
            f"repacked archive member {member_name!r} does not match supplied zoom/warp geometry",
        )


def _optional_path_metadata(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "present": False}
    meta: dict[str, Any] = {"path": str(path), "present": path.is_file()}
    if path.is_file():
        meta.update(
            {
                "bytes": _stat_size(path),
                "sha256": sha256_file(path),
            }
        )
    return meta


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _tensor_summary(value: Any) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is present in normal repo tests
        raise SnapshotError("torch_unavailable_for_pose_contract", str(exc)) from exc
    if not torch.is_tensor(value):
        raise SnapshotError("pose_tensor_missing", "selected pose payload is not a tensor")
    detached = value.detach().cpu()
    finite = bool(torch.isfinite(detached).all()) if detached.numel() else False
    nonzero = int(torch.count_nonzero(detached).item()) if detached.numel() else 0
    shape = [int(dim) for dim in detached.shape]
    pose_dim = int(shape[-1]) if shape else 0
    if detached.numel() == 0:
        failure = "pose_tensor_empty"
    elif not finite:
        failure = "pose_tensor_nonfinite"
    elif nonzero == 0:
        failure = "pose_tensor_all_zero"
    elif pose_dim != 6:
        failure = "pose_tensor_dim_not_6"
    else:
        failure = None
    return {
        "parse_kind": "torch_tensor",
        "shape": shape,
        "dtype": str(detached.dtype),
        "numel": int(detached.numel()),
        "pose_dim": pose_dim,
        "finite": finite,
        "nonzero_elements": nonzero,
        "all_zero": nonzero == 0,
        "failure_class": failure,
        "promotable_pose_contract": failure is None,
    }


def _select_pose_tensor(payload: Any) -> tuple[Any, str]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise SnapshotError("torch_unavailable_for_pose_contract", str(exc)) from exc
    if torch.is_tensor(payload):
        return payload, "raw_tensor"
    if isinstance(payload, dict):
        for key in ("poses", "optimized_poses", "pose_tensor", "pose", "optimized_poses_bin"):
            value = payload.get(key)
            if torch.is_tensor(value):
                return value, key
        for key, value in payload.items():
            if torch.is_tensor(value):
                return value, str(key)
    raise SnapshotError("pose_tensor_missing", "pose file did not contain a tensor")


def inspect_pose_tensor_contract(path: Path) -> dict[str, Any]:
    """Inspect deployed pose tensors and reject zero-pose/silent-fallback inputs."""
    meta = _optional_path_metadata(path)
    base: dict[str, Any] = {
        **meta,
        "required_pose_dim": 6,
        "silent_zero_fallback_allowed": False,
        "promotable_pose_contract": False,
    }
    if not path.is_file():
        return {**base, "failure_class": "missing_pose_tensor"}
    if path.stat().st_size == 0:
        return {**base, "failure_class": "pose_tensor_empty"}

    if path.suffix in {".pt", ".pth"}:
        try:
            import torch

            payload = torch.load(path, map_location="cpu", weights_only=False)
            tensor, source_key = _select_pose_tensor(payload)
            return {
                **base,
                **_tensor_summary(tensor),
                "source_key": source_key,
                "source_format": "torch_load",
            }
        except Exception as exc:
            return {
                **base,
                "source_format": "torch_load",
                "failure_class": "pose_tensor_unreadable",
                "failure": str(exc),
            }

    data = path.read_bytes()
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        return {
            **base,
            "source_format": "raw_binary",
            "failure_class": "numpy_unavailable_for_pose_contract",
            "failure": str(exc),
        }
    candidates: list[tuple[str, Any]] = []
    if len(data) % (2 * 6) == 0:
        candidates.append(("float16_raw_6d", np.frombuffer(data, dtype="<f2").reshape(-1, 6)))
    if len(data) % (4 * 6) == 0:
        candidates.append(("float32_raw_6d", np.frombuffer(data, dtype="<f4").reshape(-1, 6)))
    if not candidates:
        return {
            **base,
            "source_format": "raw_binary",
            "failure_class": "pose_tensor_unreadable",
            "failure": "raw pose bytes are not divisible by 6 float16 or float32 values",
        }
    kind, arr = candidates[0]
    finite = bool(np.isfinite(arr).all()) if arr.size else False
    nonzero = int(np.count_nonzero(arr)) if arr.size else 0
    if arr.size == 0:
        failure = "pose_tensor_empty"
    elif not finite:
        failure = "pose_tensor_nonfinite"
    elif nonzero == 0:
        failure = "pose_tensor_all_zero"
    else:
        failure = None
    return {
        **base,
        "source_format": "raw_binary",
        "parse_kind": kind,
        "shape": [int(dim) for dim in arr.shape],
        "dtype": str(arr.dtype),
        "numel": int(arr.size),
        "pose_dim": 6,
        "finite": finite,
        "nonzero_elements": nonzero,
        "all_zero": nonzero == 0,
        "failure_class": failure,
        "promotable_pose_contract": failure is None,
    }


def enforce_pose_tensor_contract(contract: dict[str, Any], *, allow_unproven: bool) -> None:
    if contract.get("promotable_pose_contract") is True:
        return
    if allow_unproven and contract.get("failure_class") not in {
        "missing_pose_tensor",
        "pose_tensor_empty",
        "pose_tensor_all_zero",
    }:
        return
    raise SnapshotError(
        contract.get("failure_class") or "pose_tensor_contract_missing",
        f"refusing Q-FAITHFUL eval with unproven pose tensor contract: {contract}",
    )


def enforce_checkpoint_training_pose_contract(contract: dict[str, Any]) -> None:
    if contract.get("training_pose_contract_promotable") is True:
        return
    raise SnapshotError(
        contract.get("failure_class") or "qfaithful_training_pose_contract_missing",
        f"refusing Q-FAITHFUL eval with unproven training pose contract: {contract}",
    )


def _ffprobe_frame_count(path: Path) -> tuple[int | None, dict[str, Any]]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None, {
            "method": "ffprobe",
            "available": False,
            "failure": "ffprobe_not_found",
        }
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    evidence: dict[str, Any] = {
        "method": "ffprobe",
        "available": True,
        "command": cmd,
        "returncode": proc.returncode,
    }
    if proc.returncode != 0:
        evidence["stderr"] = proc.stderr.strip()
        return None, evidence
    try:
        payload = json.loads(proc.stdout)
        stream = (payload.get("streams") or [{}])[0]
    except (json.JSONDecodeError, IndexError, TypeError):
        evidence["failure"] = "ffprobe_json_parse_failed"
        return None, evidence
    for key in ("nb_read_frames", "nb_frames"):
        raw = stream.get(key)
        if raw is None or raw == "N/A":
            continue
        try:
            return int(raw), {**evidence, "selected_field": key, "raw_value": raw}
        except (TypeError, ValueError):
            continue
    evidence["failure"] = "frame_count_unavailable"
    evidence["stream"] = stream
    return None, evidence


def inspect_mask_frame_contract(
    masks_mkv: Path,
    *,
    declared_contract: str,
) -> dict[str, Any]:
    if declared_contract not in {"auto", "full", "half"}:
        raise SnapshotError(
            "invalid_mask_frame_contract",
            f"unsupported mask frame contract: {declared_contract!r}",
        )
    if declared_contract != "auto":
        frames = EXPECTED_MASK_FRAMES if declared_contract == "full" else HALF_FRAME_MASK_FRAMES
        return {
            "contract": declared_contract,
            "frame_count": frames,
            "source": "operator_declared",
            "masks_mkv": _optional_path_metadata(masks_mkv),
            "expected_full_frame_count": EXPECTED_MASK_FRAMES,
            "expected_half_frame_count": HALF_FRAME_MASK_FRAMES,
        }

    frames, evidence = _ffprobe_frame_count(masks_mkv)
    if frames == EXPECTED_MASK_FRAMES:
        contract = "full"
    elif frames == HALF_FRAME_MASK_FRAMES:
        contract = "half"
    else:
        contract = "unknown"
    return {
        "contract": contract,
        "frame_count": frames,
        "source": "ffprobe",
        "ffprobe": evidence,
        "masks_mkv": _optional_path_metadata(masks_mkv),
        "expected_full_frame_count": EXPECTED_MASK_FRAMES,
        "expected_half_frame_count": HALF_FRAME_MASK_FRAMES,
    }


def qfaithful_renderer_zoom_contract() -> dict[str, Any]:
    """Return the static zoom-consumption contract for this snapshot exporter.

    The Q-FAITHFUL snapshot path currently serializes
    ``tac.quantizr_faithful_renderer.JointFrameGenerator`` through QFAI/QZS3.
    That public-floor architecture has no ``use_zoom_flow``/``ego_flow`` input,
    but the contest inflate runtime can still consume charged
    ``zoom_scalars.bin`` for half-frame mask expansion before invoking the
    renderer.  This distinction matters: zoom geometry need not be a renderer
    input to be score-affecting and contest-compliant.
    """

    return {
        "architecture": QFAITHFUL_RENDERER_ARCHITECTURE,
        "detection": "static_export_builder",
        "consumes_zoom_warp": True,
        "renderer_consumes_ego_flow": False,
        "runtime_consumes_zoom_warp_for_mask_expansion": True,
        "consumption_proof": (
            "submissions/robust_current/inflate_renderer.py loads charged "
            "zoom_scalars.bin whenever half-frame masks are present, even if "
            "renderer.use_zoom_flow is false"
        ),
        "failure_class_if_required": "zoom_warp_geometry_not_consumed_by_runtime",
        "unblock_requirement": (
            "half-frame masks with charged zoom geometry require an archive "
            "runtime that preserves zoom_scalars.bin and uses it for half-frame "
            "mask expansion before renderer invocation"
        ),
    }


def build_snapshot_screen_contract(args: argparse.Namespace) -> dict[str, Any]:
    mask_contract = inspect_mask_frame_contract(
        args.masks_mkv,
        declared_contract=args.mask_frame_contract,
    )
    zoom_warp_path = args.zoom_warp_path.resolve() if args.zoom_warp_path else None
    zoom_meta = _optional_path_metadata(zoom_warp_path)
    renderer_zoom_contract = qfaithful_renderer_zoom_contract()
    poses_path = getattr(args, "poses_pt", None)
    pose_contract = inspect_pose_tensor_contract(poses_path) if poses_path is not None else None
    allow_unproven_pose = bool(getattr(args, "allow_unproven_pose_custody", False))
    mask_mode = mask_contract["contract"]
    non_promotable_reasons: list[str] = []

    if mask_mode == "unknown":
        non_promotable_reasons.append("mask_frame_contract_unknown")
    if mask_mode == "half" and not zoom_meta["present"]:
        non_promotable_reasons.append("half_frame_masks_without_zoom_warp_geometry")

    current_repack_preserves_zoom_warp = bool(zoom_meta["present"])
    if mask_mode == "half" and zoom_meta["present"] and not current_repack_preserves_zoom_warp:
        non_promotable_reasons.append("zoom_warp_geometry_not_preserved_by_repack_contract")
    if (
        mask_mode == "half"
        and zoom_meta["present"]
        and not renderer_zoom_contract["consumes_zoom_warp"]
    ):
        non_promotable_reasons.append(
            renderer_zoom_contract["failure_class_if_required"]
        )
    if (
        pose_contract is not None
        and pose_contract.get("promotable_pose_contract") is not True
        and not allow_unproven_pose
    ):
        non_promotable_reasons.append(
            pose_contract.get("failure_class") or "pose_tensor_contract_missing"
        )

    return {
        "schema_version": 1,
        "runtime_contract_version": SNAPSHOT_RUNTIME_CONTRACT_VERSION,
        "mask_frame_contract": mask_contract,
        "renderer_zoom_contract": renderer_zoom_contract,
        "export_runtime_contract_metadata_required": True,
        "export_contract_required_keys": list(QFAITHFUL_EXPORT_CONTRACT_REQUIRED_KEYS),
        "pose_tensor_contract": pose_contract,
        "allow_unproven_pose_custody": allow_unproven_pose,
        "zoom_warp_geometry": {
            "required_for_half_frame": mask_mode == "half",
            "required_renderer_consumption": False,
            "required_runtime_mask_expansion_consumption": mask_mode == "half" and zoom_meta["present"],
            "source": zoom_meta,
            "archive_member_name": ZOOM_WARP_ARCHIVE_MEMBER if zoom_meta["present"] else None,
            "packed_in_repacked_archive": current_repack_preserves_zoom_warp,
            "repack_preservation_note": (
                "experiments/repack_quantizr_faithful_qzs3_archive.py preserves "
                "zoom/warp geometry as charged zoom_scalars.bin when present; "
                "half-frame exact screens are promotable only when that member "
                "is present in the raw and repacked runtime archive contract "
                "and the inflate runtime proves it consumes the member during "
                "half-frame mask expansion"
            ),
        },
        "full_frame_archives_preserve_legacy_behavior": mask_mode == "full",
        "promotable_exact_screen": not non_promotable_reasons,
        "score_claim": False,
        "non_promotable_reasons": non_promotable_reasons,
    }


def enforce_exact_screen_contract(contract: dict[str, Any], *, eval_mode: str) -> None:
    if eval_mode != "run":
        return
    if contract.get("promotable_exact_screen") is True:
        return
    reasons = ", ".join(contract.get("non_promotable_reasons") or ["unknown_contract_failure"])
    raise SnapshotError(
        "non_promotable_runtime_contract",
        f"refusing exact screen run for non-promotable Q-FAITHFUL snapshot contract: {reasons}",
    )


def qfai_export_contract_metadata(
    *,
    checkpoint: Path,
    checkpoint_sha: str,
    profile: str,
    screen_contract: dict[str, Any],
    training_pose_contract: dict[str, Any],
    packed_from_ema_shadow: bool = False,
) -> dict[str, Any]:
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "runtime_contract_version": SNAPSHOT_RUNTIME_CONTRACT_VERSION,
        "mask_frame_contract": screen_contract["mask_frame_contract"]["contract"],
        "renderer_zoom_contract": screen_contract["renderer_zoom_contract"],
        "eval_roundtrip_required": True,
        "profile": profile,
        "promotable_exact_screen": bool(screen_contract["promotable_exact_screen"]),
        "non_promotable_reasons": list(screen_contract["non_promotable_reasons"]),
        "pose_tensor_contract": screen_contract.get("pose_tensor_contract"),
        "training_pose_contract": training_pose_contract,
        "packed_from_ema_shadow": bool(packed_from_ema_shadow),
    }


def build_repack_command(
    *,
    python_bin: str,
    workspace: Path,
    source_archive: Path,
    output_dir: Path,
    output_archive: Path,
    renderer_codec: str,
    qzs3_block_size: int,
    submission_layout: str,
    pose_codec: str,
    pose_residual_topk: int,
    brotli_quality: int,
) -> list[str]:
    return [
        python_bin,
        str(workspace / "experiments" / "repack_quantizr_faithful_qzs3_archive.py"),
        "--source-archive",
        str(source_archive),
        "--output-dir",
        str(output_dir),
        "--output-archive",
        str(output_archive),
        "--renderer-codec",
        renderer_codec,
        "--qzs3-block-size",
        str(qzs3_block_size),
        "--submission-layout",
        submission_layout,
        "--pose-codec",
        pose_codec,
        "--pose-residual-topk",
        str(pose_residual_topk),
        "--brotli-quality",
        str(brotli_quality),
    ]


def build_eval_invocation(
    *,
    workspace: Path,
    archive_path: Path,
    archive_label: str,
    log_dir: Path,
    predicted_low: float,
    predicted_high: float,
    controlled_baseline: str,
    source_shas: dict[str, str],
    eval_script: Path,
) -> tuple[list[str], dict[str, str]]:
    env = {
        "WORKSPACE": str(workspace),
        "ARCHIVE_PATH": str(archive_path),
        "ARCHIVE_LABEL": archive_label,
        "LOG_DIR": str(log_dir),
        "PREDICTED_LOW": str(predicted_low),
        "PREDICTED_HIGH": str(predicted_high),
        "CONTROLLED_BASELINE": controlled_baseline,
        "REQUIRED_SOURCE_SHA256S": required_source_sha_env(source_shas),
    }
    return ["bash", str(eval_script)], env


def build_claim_command(
    *,
    workspace: Path,
    lane_id: str,
    platform: str,
    instance_job_id: str,
    agent: str,
    predicted_eta_utc: str,
    child_of: str | None,
    parallel_reason: str | None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(workspace / "tools" / "claim_lane_dispatch.py"),
        "claim",
        "--lane-id",
        lane_id,
        "--platform",
        platform,
        "--instance-job-id",
        instance_job_id,
        "--agent",
        agent,
        "--predicted-eta-utc",
        predicted_eta_utc,
        "--status",
        "eval",
        "--notes",
        "bounded_q_faithful_snapshot_h100_screen",
    ]
    if child_of or parallel_reason:
        if not (child_of and parallel_reason):
            raise SnapshotError(
                "dispatch_claim_incomplete",
                "--dispatch-child-of and --dispatch-parallel-reason must be paired",
            )
        cmd.extend(
            [
                "--allow-parallel",
                "--child-of",
                child_of,
                "--parallel-reason",
                parallel_reason,
            ]
        )
    return cmd


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    base = {
        "schema_version": 1,
        "tool": "scripts/q_faithful_snapshot_loop.py",
        "recorded_at_utc": utc_now(),
        "score_claim": False,
        "score_claim_reason": "snapshot screen is non-claiming; exact CUDA JSON is required before any claim",
        "exact_cuda_json_required": True,
        "all_score_affecting_bits_inside_archive": True,
        "sidecars_score_affecting": False,
    }
    base.update(payload)
    _json_dump(path, base)


def stable_checkpoints(
    checkpoint_dir: Path,
    *,
    glob_pattern: str,
    min_age_seconds: float,
    processed_shas: set[str],
) -> list[Path]:
    now = time.time()
    candidates: list[Path] = []
    for path in sorted(checkpoint_dir.glob(glob_pattern), key=lambda p: p.stat().st_mtime):
        if not path.is_file() or path.suffix == ".tmp":
            continue
        if now - path.stat().st_mtime < min_age_seconds:
            continue
        sha = sha256_file(path)
        if sha in processed_shas:
            continue
        candidates.append(path)
    return candidates


def validate_static_inputs(args: argparse.Namespace) -> None:
    for attr, failure in (
        ("checkpoint_dir", "missing_checkpoint_dir"),
        ("masks_mkv", "missing_masks_mkv"),
        ("poses_pt", "missing_poses_pt"),
    ):
        path = Path(getattr(args, attr))
        if attr == "checkpoint_dir":
            ok = path.is_dir()
        else:
            ok = path.is_file()
        if not ok:
            raise SnapshotError(failure, f"{attr.replace('_', '-')} not found: {path}")
    if args.eval_mode == "run" and args.dispatch_claim_mode == "none":
        raise SnapshotError(
            "dispatch_claim_required",
            "--eval-mode run requires a dispatch claim; use claim or already-claimed",
        )
    if args.dispatch_claim_mode == "already-claimed" and not args.existing_dispatch_claim_id:
        raise SnapshotError(
            "dispatch_claim_required",
            "--dispatch-claim-mode already-claimed requires --existing-dispatch-claim-id",
        )


def _run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=merged, check=True)


def process_checkpoint(args: argparse.Namespace, checkpoint: Path, source_shas: dict[str, str]) -> Path:
    checkpoint_sha = sha256_file(checkpoint)
    snapshot_id = f"{checkpoint.stem}-{checkpoint_sha[:12]}"
    snapshot_dir = args.output_root / snapshot_id
    export_dir = snapshot_dir / "export"
    raw_archive = snapshot_dir / "raw_qfai" / "archive.zip"
    repack_dir = snapshot_dir / "qzs3"
    qzs3_archive = repack_dir / "archive.zip"
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    eval_log_dir = snapshot_dir / "h100_exact_screen"
    eval_label = f"{args.archive_label_prefix}_{snapshot_id}"

    export_command = [
        args.python_bin,
        str(Path("scripts") / "q_faithful_snapshot_loop.py"),
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--masks-mkv",
        str(args.masks_mkv),
        "--poses-pt",
        str(args.poses_pt),
        "--output-root",
        str(args.output_root),
        "--profile",
        args.profile,
        "--max-snapshots",
        "1",
    ]
    repack_command = build_repack_command(
        python_bin=args.python_bin,
        workspace=args.workspace,
        source_archive=raw_archive,
        output_dir=repack_dir,
        output_archive=qzs3_archive,
        renderer_codec=args.renderer_codec,
        qzs3_block_size=args.qzs3_block_size,
        submission_layout=args.submission_layout,
        pose_codec=args.pose_codec,
        pose_residual_topk=args.pose_residual_topk,
        brotli_quality=args.brotli_quality,
    )
    eval_command, eval_env = build_eval_invocation(
        workspace=args.workspace,
        archive_path=qzs3_archive,
        archive_label=eval_label,
        log_dir=eval_log_dir,
        predicted_low=args.predicted_low,
        predicted_high=args.predicted_high,
        controlled_baseline=args.controlled_baseline,
        source_shas=source_shas,
        eval_script=args.eval_script,
    )

    screen_contract = build_snapshot_screen_contract(args)
    training_pose_contract = inspect_checkpoint_training_pose_contract(
        checkpoint,
        deployed_pose_contract=screen_contract.get("pose_tensor_contract"),
        profile=args.profile,
    )
    manifest_base = {
        "status": "started",
        "failure_class": None,
        "snapshot_id": snapshot_id,
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": _stat_size(checkpoint),
            "sha256": checkpoint_sha,
            "profile_hint": checkpoint_profile_hint(checkpoint),
        },
        "eval_roundtrip_proof": verify_eval_roundtrip_profile(args.workspace, args.profile),
        "export_command": export_command,
        "repack_command": repack_command,
        "eval_command": eval_command,
        "eval_environment": {
            key: value
            for key, value in eval_env.items()
            if key != "REQUIRED_SOURCE_SHA256S"
        },
        "required_source_sha256s": source_shas,
        "source_runtime_sha256s": source_shas,
        "snapshot_screen_contract": screen_contract,
        "checkpoint_training_pose_contract": training_pose_contract,
    }
    write_manifest(manifest_path, manifest_base)

    try:
        if args.eval_mode == "run":
            pose_contract = manifest_base["snapshot_screen_contract"].get("pose_tensor_contract") or {}
            enforce_pose_tensor_contract(
                pose_contract,
                allow_unproven=args.allow_unproven_pose_custody,
            )
            enforce_checkpoint_training_pose_contract(training_pose_contract)
        enforce_exact_screen_contract(
            manifest_base["snapshot_screen_contract"],
            eval_mode=args.eval_mode,
        )
        if args.dry_run:
            write_manifest(
                manifest_path,
                {
                    **manifest_base,
                    "status": "dry_run",
                    "failure_class": None,
                    "dry_run": True,
                },
            )
            return manifest_path

        export_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.masks_mkv, export_dir / "masks.mkv")
        shutil.copy2(args.poses_pt, export_dir / "optimized_poses.bin")
        qfai_meta = export_qfai_renderer(
            checkpoint,
            export_dir / "renderer.bin",
            state_source=args.state_source,
            brotli_quality=args.brotli_quality,
            extra_meta=qfai_export_contract_metadata(
                checkpoint=checkpoint,
                checkpoint_sha=checkpoint_sha,
                profile=args.profile,
                screen_contract=manifest_base["snapshot_screen_contract"],
                training_pose_contract=training_pose_contract,
            ),
        )
        if args.eval_mode == "run":
            enforce_ema_export_contract(
                qfai_meta,
                allow_live_weight_export=args.allow_live_weight_export,
            )
        raw_meta = build_raw_archive(
            renderer_bin=export_dir / "renderer.bin",
            masks_mkv=export_dir / "masks.mkv",
            poses_pt=export_dir / "optimized_poses.bin",
            output_archive=raw_archive,
            zoom_warp_path=args.zoom_warp_path,
        )
        _run(repack_command, cwd=args.workspace)
        if not qzs3_archive.is_file():
            raise SnapshotError("repack_output_missing", f"missing repacked archive: {qzs3_archive}")
        qzs3_meta = archive_metadata(qzs3_archive)
        validate_repacked_geometry_contract(
            screen_contract=manifest_base["snapshot_screen_contract"],
            repacked_archive_meta=qzs3_meta,
        )

        eval_meta: dict[str, Any] = {
            "mode": args.eval_mode,
            "log_dir": str(eval_log_dir),
            "contest_auth_eval_json": None,
        }
        if args.eval_mode in {"command", "run"}:
            eval_meta["command_ready"] = bool(
                manifest_base["snapshot_screen_contract"]["promotable_exact_screen"]
                and training_pose_contract.get("training_pose_contract_promotable") is True
            )
            if not eval_meta["command_ready"]:
                reasons = list(
                    manifest_base["snapshot_screen_contract"]["non_promotable_reasons"]
                )
                if training_pose_contract.get("training_pose_contract_promotable") is not True:
                    reasons.append(
                        training_pose_contract.get("failure_class")
                        or "qfaithful_training_pose_contract_missing"
                    )
                eval_meta["non_promotable_reasons"] = reasons
        if args.eval_mode == "run":
            if args.dispatch_claim_mode == "claim":
                claim_cmd = build_claim_command(
                    workspace=args.workspace,
                    lane_id=args.dispatch_lane_id,
                    platform=args.dispatch_platform,
                    instance_job_id=eval_label,
                    agent=args.dispatch_agent,
                    predicted_eta_utc=args.dispatch_predicted_eta_utc,
                    child_of=args.dispatch_child_of,
                    parallel_reason=args.dispatch_parallel_reason,
                )
                _run(claim_cmd, cwd=args.workspace)
                eval_meta["dispatch_claim_command"] = claim_cmd
            else:
                eval_meta["existing_dispatch_claim_id"] = args.existing_dispatch_claim_id
            _run(eval_command, env=eval_env, cwd=args.workspace)
            eval_json = eval_log_dir / "contest_auth_eval.json"
            if not eval_json.is_file():
                raise SnapshotError("exact_cuda_json_missing", f"missing exact CUDA JSON: {eval_json}")
            eval_meta["contest_auth_eval_json"] = str(eval_json)

        write_manifest(
            manifest_path,
            {
                **manifest_base,
                "status": "completed",
                "failure_class": None,
                "qfai_export": qfai_meta,
                "raw_archive": raw_meta,
                "repacked_archive": qzs3_meta,
                "h100_exact_screen": eval_meta,
            },
        )
        return manifest_path
    except subprocess.CalledProcessError as exc:
        write_manifest(
            manifest_path,
            {
                **manifest_base,
                "status": "failed",
                "failure_class": "subprocess_failed",
                "failure": {"returncode": exc.returncode, "cmd": exc.cmd},
            },
        )
        raise
    except SnapshotError as exc:
        write_manifest(
            manifest_path,
            {
                **manifest_base,
                "status": "failed",
                "failure_class": exc.failure_class,
                "failure": {"message": str(exc)},
            },
        )
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-glob", default="*.pt")
    parser.add_argument("--min-checkpoint-age-seconds", type=float, default=60.0)
    parser.add_argument("--masks-mkv", type=Path, required=True)
    parser.add_argument(
        "--mask-frame-contract",
        choices=("auto", "full", "half"),
        default="auto",
        help=(
            "Expected masks.mkv frame contract. auto uses ffprobe; exact "
            "screens fail closed if the contract is unknown or half-frame "
            "without charged zoom/warp geometry."
        ),
    )
    parser.add_argument(
        "--zoom-warp-path",
        type=Path,
        default=None,
        help=(
            "Optional zoom/warp geometry source for half-frame masks. Half-frame "
            "snapshots remain non-promotable unless the runtime contract proves "
            "the renderer consumes this charged geometry member."
        ),
    )
    parser.add_argument("--poses-pt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile", default="q_faithful_dilated_88k")
    parser.add_argument(
        "--state-source",
        choices=("auto", "model_state_dict", "ema_shadow", "model", "state_dict", "raw"),
        default="auto",
    )
    parser.add_argument(
        "--allow-live-weight-export",
        action="store_true",
        default=_env_truthy("QFAITHFUL_ALLOW_LIVE_WEIGHT_EXPORT"),
        help=(
            "Explicit diagnostic escape hatch for non-EMA exports. Exact eval "
            "runs fail closed unless the checkpoint export packed ema_shadow."
        ),
    )
    parser.add_argument(
        "--allow-unproven-pose-custody",
        action="store_true",
        default=_env_truthy("QFAITHFUL_ALLOW_UNPROVEN_POSE_CUSTODY"),
        help=(
            "Explicit diagnostic escape hatch for unreadable/legacy pose "
            "custody. Missing, empty, and all-zero pose tensors still fail."
        ),
    )
    parser.add_argument("--renderer-codec", choices=("qzs3", "qzs4", "torch_fp4"), default="qzs3")
    parser.add_argument("--qzs3-block-size", type=int, default=32)
    parser.add_argument(
        "--submission-layout",
        choices=("multi_member", "rpk1_single_blob", "pr64_single_blob", "pr64_mask_first_single_blob"),
        default="multi_member",
    )
    parser.add_argument(
        "--pose-codec",
        choices=(
            "raw",
            "pose_fp16_col_delta_v1",
            "pose_qpose14_col_delta_v1",
            "pose_qp1_v1",
            "pose_fp16_velocity_only_v1",
            "pose_fp16_velocity_residual_topk_v1",
        ),
        default="raw",
    )
    parser.add_argument("--pose-residual-topk", type=int, default=0)
    parser.add_argument("--brotli-quality", type=int, default=11)
    parser.add_argument("--max-snapshots", type=int, default=1)
    parser.add_argument("--max-idle-polls", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=900.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-mode", choices=("none", "command", "run"), default="none")
    parser.add_argument("--eval-script", type=Path, default=Path("scripts/remote_archive_only_eval.sh"))
    parser.add_argument("--archive-label-prefix", default="qfaithful_snapshot")
    parser.add_argument("--predicted-low", type=float, default=0.0)
    parser.add_argument("--predicted-high", type=float, default=9.99)
    parser.add_argument(
        "--controlled-baseline",
        default="Q-FAITHFUL long-run checkpoint snapshot; non-claiming until exact CUDA JSON",
    )
    parser.add_argument(
        "--dispatch-claim-mode",
        choices=("claim", "already-claimed", "none"),
        default="claim",
    )
    parser.add_argument("--existing-dispatch-claim-id")
    parser.add_argument("--dispatch-lane-id", default="q_faithful_snapshot")
    parser.add_argument("--dispatch-platform", default="h100")
    parser.add_argument("--dispatch-agent", default="codex:q_faithful_snapshot_loop")
    parser.add_argument("--dispatch-predicted-eta-utc", default=utc_now())
    parser.add_argument("--dispatch-child-of")
    parser.add_argument("--dispatch-parallel-reason")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.workspace = args.workspace.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.masks_mkv = args.masks_mkv.resolve()
    if args.zoom_warp_path is not None:
        args.zoom_warp_path = args.zoom_warp_path.resolve()
    args.poses_pt = args.poses_pt.resolve()
    args.output_root = args.output_root.resolve()
    if not args.eval_script.is_absolute():
        args.eval_script = args.workspace / args.eval_script
    validate_static_inputs(args)
    source_shas = source_runtime_shas(args.workspace)

    processed: set[str] = set()
    idle_polls = 0
    completed = 0
    args.output_root.mkdir(parents=True, exist_ok=True)
    while completed < args.max_snapshots:
        candidates = stable_checkpoints(
            args.checkpoint_dir,
            glob_pattern=args.checkpoint_glob,
            min_age_seconds=args.min_checkpoint_age_seconds,
            processed_shas=processed,
        )
        if not candidates:
            if idle_polls >= args.max_idle_polls:
                if completed == 0:
                    raise SnapshotError("no_stable_checkpoints", "no stable checkpoints found")
                break
            idle_polls += 1
            time.sleep(args.poll_seconds)
            continue
        idle_polls = 0
        for checkpoint in candidates:
            manifest = process_checkpoint(args, checkpoint, source_shas)
            print(f"SNAPSHOT_MANIFEST {manifest}", flush=True)
            processed.add(sha256_file(checkpoint))
            completed += 1
            if completed >= args.max_snapshots:
                break
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SnapshotError as exc:
        print(f"FATAL[{exc.failure_class}]: {exc}", file=sys.stderr)
        raise SystemExit(2)
