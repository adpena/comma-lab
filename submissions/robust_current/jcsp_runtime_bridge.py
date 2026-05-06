#!/usr/bin/env python3
"""Fail-closed JCSP runtime probe for ``submissions/robust_current``.

This module is intentionally stdlib-only.  It runs from ``inflate.sh`` before
any rendering branch so an archive carrying ``jcsp.bin`` cannot silently fall
through to an unrelated runtime path.  The bridge currently probes and proves
the member shape; it does not decode streams or emit frames.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

JCSP_RUNTIME_BRIDGE_PROBE_SCHEMA = "jcsp_submission_runtime_bridge_probe_v1"
JCSP_ARCHIVE_MEMBER_NAME = "jcsp.bin"
JCSP_REQUIRED_SUBMISSION_RUNTIME = "submissions/robust_current"
JCSP_RUNTIME_BRIDGE_PATH = "submissions/robust_current/jcsp_runtime_bridge.py"
JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER = (
    "submissions_robust_current_jcsp_bin_consumption_missing"
)
JCSP_LOCAL_SKELETON_RUNTIME_BLOCKER = (
    "jcsp_local_skeleton_not_submission_runtime_container"
)
JCSP_MAGIC = b"JCSP"
JCSK_MAGIC = b"JCSK"
JCSP_VERSION = 1
JCSK_VERSION = 1
KIND_ARITHMETIC_STATIC = 0
KIND_BALLE_HYPERPRIOR = 1
KIND_RAW_PASSTHROUGH = 2
EXIT_JCSP_MEMBER_REFUSED = 44

_PAYLOAD_MAGICS_BY_CODEC_KIND: dict[int, tuple[bytes, ...]] = {
    KIND_ARITHMETIC_STATIC: (b"AQv1", b"AQc1"),
    KIND_BALLE_HYPERPRIOR: (b"BHv1",),
}


def _reject_duplicate_json_object_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _with_manifest_sha256(payload: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out.pop("manifest_sha256", None)
    out["manifest_sha256"] = _sha256_bytes(_canonical_json_bytes(out))
    return out


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(manifest) + b"\n")


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in items))


def _require_available(blob: bytes, cursor: int, n_bytes: int, context: str) -> None:
    if n_bytes < 0 or cursor < 0 or cursor + n_bytes > len(blob):
        raise ValueError(
            f"truncated {context} at offset {cursor}; need {n_bytes} bytes, "
            f"blob len={len(blob)}"
        )


def _payload_magic_for_kind(
    *,
    codec_kind: int,
    payload: bytes,
    stream_name: str,
) -> str:
    if codec_kind == KIND_RAW_PASSTHROUGH:
        if not payload:
            raise ValueError(f"stream {stream_name!r} raw payload is empty")
        return payload[:4].decode("ascii", errors="replace")
    allowed = _PAYLOAD_MAGICS_BY_CODEC_KIND.get(int(codec_kind))
    if allowed is None:
        raise ValueError(f"stream {stream_name!r} has invalid codec_kind {codec_kind}")
    if len(payload) < 4:
        raise ValueError(
            f"stream {stream_name!r} payload is too small for codec magic"
        )
    magic = payload[:4]
    if magic not in allowed:
        allowed_text = ", ".join(repr(item) for item in allowed)
        raise ValueError(
            f"stream {stream_name!r} payload magic {magic!r} is incompatible "
            f"with codec_kind {codec_kind}; expected one of {allowed_text}"
        )
    return magic.decode("ascii", errors="replace")


def _parse_real_jcsp_container(blob: bytes) -> dict[str, Any]:
    _require_available(blob, 0, 7, "JCSP header")
    if blob[:4] != JCSP_MAGIC:
        raise ValueError(f"bad JCSP magic {blob[:4]!r}")
    cursor = 4
    (version,) = struct.unpack_from("<H", blob, cursor)
    cursor += 2
    if version != JCSP_VERSION:
        raise ValueError(f"unsupported JCSP version {version}")
    (stream_count,) = struct.unpack_from("<B", blob, cursor)
    cursor += 1

    streams: list[dict[str, Any]] = []
    names: list[str] = []
    for index in range(int(stream_count)):
        _require_available(blob, cursor, 1, f"stream {index} name length")
        (name_len,) = struct.unpack_from("<B", blob, cursor)
        cursor += 1
        if name_len <= 0:
            raise ValueError(f"stream {index} has empty name")
        _require_available(blob, cursor, name_len, f"stream {index} name")
        name = blob[cursor : cursor + name_len].decode("utf-8", errors="strict")
        cursor += name_len
        if "\x00" in name:
            raise ValueError(f"stream {index} name contains NUL")
        if name in names:
            raise ValueError(f"duplicate stream name {name!r}")
        names.append(name)

        _require_available(blob, cursor, 1, f"stream {name!r} codec kind")
        (codec_kind,) = struct.unpack_from("<B", blob, cursor)
        cursor += 1
        if codec_kind not in (
            KIND_ARITHMETIC_STATIC,
            KIND_BALLE_HYPERPRIOR,
            KIND_RAW_PASSTHROUGH,
        ):
            raise ValueError(f"stream {name!r} has invalid codec_kind {codec_kind}")

        _require_available(blob, cursor, 4, f"stream {name!r} ADMM target")
        (admm_bytes_target,) = struct.unpack_from("<I", blob, cursor)
        cursor += 4
        _require_available(blob, cursor, 4, f"stream {name!r} actual bytes")
        (actual_bytes,) = struct.unpack_from("<I", blob, cursor)
        cursor += 4
        _require_available(blob, cursor, 4, f"stream {name!r} score delta")
        (score_delta_milli,) = struct.unpack_from("<i", blob, cursor)
        cursor += 4
        _require_available(blob, cursor, 4, f"stream {name!r} marginal")
        (marginal_milli,) = struct.unpack_from("<i", blob, cursor)
        cursor += 4
        _require_available(blob, cursor, 4, f"stream {name!r} payload length")
        (payload_len,) = struct.unpack_from("<I", blob, cursor)
        cursor += 4
        if int(actual_bytes) != int(payload_len):
            raise ValueError(
                f"stream {name!r} actual_bytes={actual_bytes} does not match "
                f"payload_len={payload_len}"
            )
        _require_available(blob, cursor, payload_len, f"stream {name!r} payload")
        payload = blob[cursor : cursor + payload_len]
        cursor += payload_len
        payload_magic = _payload_magic_for_kind(
            codec_kind=int(codec_kind),
            payload=payload,
            stream_name=name,
        )
        streams.append(
            {
                "index": index,
                "name": name,
                "codec_kind": int(codec_kind),
                "admm_bytes_target": int(admm_bytes_target),
                "actual_bytes": int(actual_bytes),
                "score_delta_milli": int(score_delta_milli),
                "marginal_milli": int(marginal_milli),
                "payload_magic": payload_magic,
                "payload_sha256": _sha256_bytes(payload),
            }
        )

    _require_available(blob, cursor, 4, "JCSP KKT residual")
    (kkt_residual_milli,) = struct.unpack_from("<I", blob, cursor)
    cursor += 4
    _require_available(blob, cursor, 4, "JCSP iteration count")
    (iters,) = struct.unpack_from("<I", blob, cursor)
    cursor += 4
    _require_available(blob, cursor, 1, "JCSP converged flag")
    (converged_raw,) = struct.unpack_from("<B", blob, cursor)
    cursor += 1
    if converged_raw not in (0, 1):
        raise ValueError(f"invalid JCSP converged flag {converged_raw}")
    if cursor != len(blob):
        raise ValueError(
            f"trailing bytes after JCSP container: cursor={cursor}, len={len(blob)}"
        )
    return {
        "container_magic": "JCSP",
        "container_version": int(version),
        "stream_count": int(stream_count),
        "streams": streams,
        "waterline_kkt_residual_milli": int(kkt_residual_milli),
        "iters": int(iters),
        "converged": bool(converged_raw),
        "noop_fixture": int(stream_count) == 0,
    }


def _probe_local_skeleton_container(blob: bytes) -> dict[str, Any]:
    details: dict[str, Any] = {
        "container_magic": "JCSK",
        "refused_preview_member": True,
    }
    if len(blob) < 10:
        details["preview_parse_error"] = "truncated JCSK header"
        return details
    (version,) = struct.unpack_from("<H", blob, 4)
    (body_len,) = struct.unpack_from("<I", blob, 6)
    body_start = 10
    body_end = body_start + int(body_len)
    details["container_version"] = int(version)
    details["declared_body_bytes"] = int(body_len)
    if version != JCSK_VERSION:
        details["preview_parse_error"] = f"unsupported JCSK version {version}"
        return details
    if body_end != len(blob):
        details["preview_parse_error"] = (
            f"JCSK body length mismatch declared={body_len} "
            f"actual={len(blob) - body_start}"
        )
        return details
    try:
        manifest = json.loads(
            blob[body_start:body_end].decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_object_pairs,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        details["preview_parse_error"] = str(exc)
        return details
    if not isinstance(manifest, Mapping):
        details["preview_parse_error"] = "JCSK manifest is not a mapping"
        return details
    details["preview_manifest_schema"] = str(manifest.get("schema", ""))
    details["preview_manifest_sha256"] = str(manifest.get("manifest_sha256", ""))
    try:
        details["preview_stream_count"] = int(manifest.get("stream_count", -1))
    except (TypeError, ValueError):
        details["preview_parse_error"] = "JCSK manifest stream_count is invalid"
    return details


def probe_jcsp_runtime_bridge(
    archive_dir: str | Path,
    *,
    member_name: str = JCSP_ARCHIVE_MEMBER_NAME,
    manifest_json: str | Path | None = None,
) -> dict[str, Any]:
    """Probe ``archive_dir/member_name`` and return a deterministic contract.

    A present JCSP member is never treated as dispatch-ready by this tranche.
    Real ``JCSP`` bytes are parsed and then refused because no stream consumer
    is wired.  Local ``JCSK`` preview bytes are refused before runtime-loader
    readiness.
    """

    archive_root = Path(archive_dir)
    member_path = archive_root / member_name
    member_exists = member_path.exists()
    base: dict[str, Any] = {
        "schema": JCSP_RUNTIME_BRIDGE_PROBE_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "required_submission_runtime": JCSP_REQUIRED_SUBMISSION_RUNTIME,
        "runtime_bridge_path": JCSP_RUNTIME_BRIDGE_PATH,
        "member_name": member_name,
        "member_present": member_exists,
        "detects_required_member": member_exists,
        "detected_real_jcsp_member": False,
        "refused_preview_member": False,
        "ready_for_runtime_probe": True,
        "ready_for_runtime_loader": False,
        "consumes_required_member": False,
        "ready_for_submission_runtime_consumption": False,
        "ready_for_exact_eval_dispatch": False,
        "runtime_action": "no_jcsp_member_present",
        "dispatch_blockers": [],
    }
    if not member_path.exists():
        manifest = _with_manifest_sha256(base)
        if manifest_json is not None:
            _write_manifest(Path(manifest_json), manifest)
        return manifest
    if not member_path.is_file():
        base.update(
            {
                "runtime_action": "refuse_non_file_jcsp_member_path",
                "refusal_reason": "jcsp member path is not a regular file",
                "dispatch_blockers": [
                    "jcsp_member_path_not_regular_file",
                    JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
                    "exact_cuda_auth_eval_missing",
                ],
            }
        )
        manifest = _with_manifest_sha256(base)
        if manifest_json is not None:
            _write_manifest(Path(manifest_json), manifest)
        return manifest

    blob = member_path.read_bytes()
    base.update(
        {
            "member_bytes": len(blob),
            "member_sha256": _sha256_bytes(blob),
            "member_prefix_hex": blob[:16].hex(),
        }
    )
    if len(blob) < 4:
        base.update(
            {
                "runtime_action": "refuse_invalid_jcsp_member",
                "refusal_reason": "jcsp member is too small for magic",
                "dispatch_blockers": [
                    "jcsp_member_too_small_for_magic",
                    JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
                    "exact_cuda_auth_eval_missing",
                ],
            }
        )
    elif blob[:4] == JCSK_MAGIC:
        details = _probe_local_skeleton_container(blob)
        base.update(details)
        base.update(
            {
                "runtime_action": "refuse_jcsk_preview_member",
                "refusal_reason": (
                    "jcsp.bin contains local JCSK preview bytes, not the "
                    "runtime JCSP container"
                ),
                "dispatch_blockers": [
                    JCSP_LOCAL_SKELETON_RUNTIME_BLOCKER,
                    JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
                    "strict_preflight_proof_missing",
                    "exact_cuda_auth_eval_missing",
                ],
            }
        )
    elif blob[:4] == JCSP_MAGIC:
        try:
            parsed = _parse_real_jcsp_container(blob)
        except ValueError as exc:
            base.update(
                {
                    "container_magic": "JCSP",
                    "runtime_action": "refuse_invalid_jcsp_container",
                    "refusal_reason": str(exc),
                    "dispatch_blockers": [
                        "jcsp_runtime_probe_parse_failed",
                        JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
                        "exact_cuda_auth_eval_missing",
                    ],
                }
            )
        else:
            base.update(parsed)
            base.update(
                {
                    "detected_real_jcsp_member": True,
                    "ready_for_runtime_loader": True,
                    "runtime_action": "refuse_until_jcsp_stream_consumer_implemented",
                    "refusal_reason": (
                        "real JCSP container parsed, but robust_current does "
                        "not decode JCSP streams or emit frames from them yet"
                    ),
                    "dispatch_blockers": [
                        JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
                        "not_integrated_into_submission_inflate_path",
                        "jcsp_stream_decode_emit_frames_missing",
                        "exact_cuda_auth_eval_missing",
                    ],
                }
            )
    else:
        base.update(
            {
                "container_magic": blob[:4].decode("ascii", errors="replace"),
                "runtime_action": "refuse_unknown_jcsp_member_magic",
                "refusal_reason": (
                    f"unknown jcsp.bin magic {blob[:4]!r}; expected "
                    f"{JCSP_MAGIC!r} or refused preview {JCSK_MAGIC!r}"
                ),
                "dispatch_blockers": [
                    "jcsp_unknown_member_magic",
                    JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
                    "exact_cuda_auth_eval_missing",
                ],
            }
        )

    base["dispatch_blockers"] = _dedupe(base["dispatch_blockers"])
    manifest = _with_manifest_sha256(base)
    if manifest_json is not None:
        _write_manifest(Path(manifest_json), manifest)
    return manifest


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe robust_current JCSP runtime bridge state"
    )
    parser.add_argument("archive_dir", help="inflater archive directory")
    parser.add_argument(
        "--member-name",
        default=JCSP_ARCHIVE_MEMBER_NAME,
        help="JCSP member filename inside archive_dir",
    )
    parser.add_argument(
        "--manifest-json",
        required=True,
        help="path to write deterministic probe manifest JSON",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    manifest = probe_jcsp_runtime_bridge(
        args.archive_dir,
        member_name=args.member_name,
        manifest_json=args.manifest_json,
    )
    if not manifest["member_present"]:
        return 0
    print(
        "[jcsp-runtime-bridge] wrote deterministic probe manifest: "
        f"{args.manifest_json}",
        file=sys.stderr,
    )
    print(
        "[jcsp-runtime-bridge] FATAL: "
        f"{manifest.get('refusal_reason', 'jcsp member refused')}",
        file=sys.stderr,
    )
    return EXIT_JCSP_MEMBER_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
