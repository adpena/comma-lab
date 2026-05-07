"""Charged runtime consumer for categorical payload candidates.

This module is packaged inside local candidate archives. It is still not a
contest decoder and never claims score readiness, but it does real inflate-time
work from charged members only: verifies the label/codebook manifests, parses
the categorical payload, and structurally consumes PR91/HPM1 payloads including
the embedded PPMd torch state when the runtime dependencies are present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

REQUIRED_MEMBERS = (
    "categorical_payload.bin",
    "class_codebook.json",
    "label_prior_payload_manifest.json",
    "runtime_consumer_proof_skeleton.json",
)
LABEL_PRIOR_PAYLOAD_MANIFEST_CONTRACT = "categorical_label_prior_payload_manifest_v1"
RUNTIME_LABEL_CONTRACT = "contest_zero_based_comma10k_order"
CLASS_CODEBOOK_CONTRACT = "categorical_class_codebook_v1"
CONTEST_SEGNET_CLASS_ORDER = (
    "road",
    "lane_markings",
    "undrivable",
    "movable",
    "my_car",
)
HPM1_MAGIC = b"HPM1"
HPM1_HEADER_BYTES = 48


class RuntimeConsumerError(RuntimeError):
    """Raised when the charged runtime consumer contract is not satisfied."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeConsumerError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeConsumerError(f"{label} must be a JSON object")
    return payload


def _verify_class_codebook(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path, label="class codebook")
    if payload.get("class_codebook_contract") != CLASS_CODEBOOK_CONTRACT:
        raise RuntimeConsumerError("class codebook contract mismatch")
    if payload.get("class_id_contract") != RUNTIME_LABEL_CONTRACT:
        raise RuntimeConsumerError("class codebook label contract mismatch")
    classes = payload.get("classes")
    if not isinstance(classes, list):
        raise RuntimeConsumerError("class codebook classes must be a list")
    names = [row.get("name") for row in classes if isinstance(row, dict)]
    if names != list(CONTEST_SEGNET_CLASS_ORDER):
        raise RuntimeConsumerError("class codebook class order mismatch")
    class_ids = [row.get("class_id") for row in classes if isinstance(row, dict)]
    if class_ids != list(range(len(CONTEST_SEGNET_CLASS_ORDER))):
        raise RuntimeConsumerError("class codebook class ids mismatch")
    return {
        "contract": payload.get("class_codebook_contract", ""),
        "label_contract": payload.get("class_id_contract", ""),
        "class_count": len(classes),
        "class_order": names,
    }


def _verify_label_prior_payload_manifest(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path, label="label prior payload manifest")
    if payload.get("schema_version") != 1:
        raise RuntimeConsumerError("label prior payload manifest schema_version mismatch")
    if payload.get("kind") != "categorical_label_prior_payload_manifest":
        raise RuntimeConsumerError("label prior payload manifest kind mismatch")
    if payload.get("label_prior_payload_manifest_contract") != LABEL_PRIOR_PAYLOAD_MANIFEST_CONTRACT:
        raise RuntimeConsumerError("label prior payload manifest contract mismatch")
    if payload.get("score_claim") is not False or payload.get("dispatch_attempted") is not False:
        raise RuntimeConsumerError("label prior payload manifest must not claim score or dispatch")
    if payload.get("ready_for_exact_eval_dispatch") is not False:
        raise RuntimeConsumerError("label prior payload manifest cannot claim exact-eval readiness")
    if payload.get("label_contract") != RUNTIME_LABEL_CONTRACT:
        raise RuntimeConsumerError("label prior payload manifest label contract mismatch")
    if payload.get("semantic_class_order") != list(CONTEST_SEGNET_CLASS_ORDER):
        raise RuntimeConsumerError("label prior payload manifest class order mismatch")
    return {
        "contract": payload.get("label_prior_payload_manifest_contract", ""),
        "label_contract": payload.get("label_contract", ""),
        "conditioning_prior_count": len(payload.get("conditioning_priors", []))
        if isinstance(payload.get("conditioning_priors"), list)
        else 0,
    }


def _load_runtime_proof_skeleton(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path, label="runtime consumer proof skeleton")
    if payload.get("schema_version") != 1:
        raise RuntimeConsumerError("runtime proof skeleton schema_version mismatch")
    if payload.get("kind") != "categorical_runtime_consumer_proof_skeleton":
        raise RuntimeConsumerError("runtime proof skeleton kind mismatch")
    if payload.get("score_claim") is not False or payload.get("dispatch_attempted") is not False:
        raise RuntimeConsumerError("runtime proof skeleton must not claim score or dispatch")
    return {
        "kind": payload.get("kind", ""),
        "ready_for_exact_eval_dispatch": payload.get("ready_for_exact_eval_dispatch") is True,
        "declared_charged_member_count": len(payload.get("charged_member_names", []))
        if isinstance(payload.get("charged_member_names"), list)
        else 0,
    }


def _parse_hpm1_payload(payload: bytes) -> dict[str, Any]:
    if len(payload) < HPM1_HEADER_BYTES:
        raise RuntimeConsumerError("truncated HPM1 payload")
    if payload[:4] != HPM1_MAGIC:
        raise RuntimeConsumerError("payload is not HPM1")
    fields = struct.unpack_from("<IIIIIIIIIII", payload, len(HPM1_MAGIC))
    (
        n_frames,
        height,
        width,
        predictor_count,
        delta,
        channels,
        use_spm,
        hpac_d_film,
        tokens_len,
        hpac_len,
        ppmd_order,
    ) = map(int, fields)
    token_start = HPM1_HEADER_BYTES
    token_end = token_start + tokens_len
    hpac_end = token_end + hpac_len
    if token_end > len(payload) or hpac_end > len(payload):
        raise RuntimeConsumerError("truncated HPM1 token or HPAC section")
    tokens = payload[token_start:token_end]
    hpac = payload[token_end:hpac_end]
    if tokens_len <= 0 or hpac_len <= 0:
        raise RuntimeConsumerError("HPM1 tokens and HPAC model must be nonempty")
    if tokens_len % 4:
        raise RuntimeConsumerError("HPM1 token stream is not uint32 aligned")
    if min(n_frames, height, width, predictor_count, channels, hpac_d_film) <= 0:
        raise RuntimeConsumerError("HPM1 geometry/model fields must be positive")
    if use_spm not in (0, 1):
        raise RuntimeConsumerError("HPM1 use_spm field must be boolean")
    reencoded = (
        HPM1_MAGIC
        + struct.pack(
            "<IIIIIIIIIII",
            n_frames,
            height,
            width,
            predictor_count,
            delta,
            channels,
            use_spm,
            hpac_d_film,
            len(tokens),
            len(hpac),
            ppmd_order,
        )
        + tokens
        + hpac
    )
    report: dict[str, Any] = {
        "codec": "HPM1",
        "parsed": True,
        "segment_bytes": len(payload),
        "segment_sha256": _sha256_bytes(payload),
        "header": {
            "n_frames": n_frames,
            "height": height,
            "width": width,
            "predictor_count": predictor_count,
            "delta": delta,
            "channels": channels,
            "use_spm": use_spm,
            "hpac_d_film": hpac_d_film,
            "tokens_len": tokens_len,
            "hpac_len": hpac_len,
            "ppmd_order": ppmd_order,
        },
        "tokens": {
            "bytes": len(tokens),
            "sha256": _sha256_bytes(tokens),
            "uint32_word_count": len(tokens) // 4,
        },
        "hpac_model": {
            "bytes": len(hpac),
            "sha256": _sha256_bytes(hpac),
            "ppmd_order": ppmd_order,
        },
        "structural_reencode": {
            "passed": reencoded == payload,
            "sha256": _sha256_bytes(reencoded),
        },
        "decoded_geometry": {
            "frame_count": n_frames,
            "shape_nhw": [n_frames, height, width],
            "symbol_count": n_frames * height * width,
            "symbol_domain": [0, 4],
        },
        "tail_bytes": len(payload) - hpac_end,
    }
    try:
        import pyppmd
        import torch

        raw_state = pyppmd.decompress(
            hpac,
            max_order=ppmd_order,
            mem_size=16 << 20,
        )
        state = torch.load(BytesIO(raw_state), map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            raise RuntimeConsumerError("HPM1 HPAC torch state is not a mapping")
        tensor_count = 0
        keys: list[str] = []
        for key, value in state.items():
            keys.append(str(key))
            if torch.is_tensor(value):
                tensor_count += 1
        report["hpac_model_load"] = {
            "attempted": True,
            "passed": True,
            "decompressed_bytes": len(raw_state),
            "decompressed_sha256": _sha256_bytes(raw_state),
            "state_key_count": len(keys),
            "state_keys_sha256": _sha256_bytes("\n".join(sorted(keys)).encode("utf-8")),
            "tensor_count": tensor_count,
        }
    except ImportError as exc:
        report["hpac_model_load"] = {
            "attempted": True,
            "passed": False,
            "failure_class": "missing_dependency",
            "failure": type(exc).__name__,
        }
        raise RuntimeConsumerError(f"HPM1 HPAC model dependency missing: {exc}") from exc
    except Exception as exc:
        report["hpac_model_load"] = {
            "attempted": True,
            "passed": False,
            "failure_class": type(exc).__name__,
        }
        raise RuntimeConsumerError(f"HPM1 HPAC model load failed: {exc}") from exc
    return report


def _consume_payload(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    if payload.startswith(HPM1_MAGIC):
        return _parse_hpm1_payload(payload)
    return {
        "codec": "opaque_categorical_payload",
        "parsed": True,
        "payload_bytes": len(payload),
        "payload_sha256": _sha256_bytes(payload),
        "note": "non-HPM1 payload consumed as opaque charged categorical bytes",
    }


def verify_charged_members(archive_root: str | Path) -> dict[str, Any]:
    """Consume required charged members without loading any uncharged sidecars."""

    root = Path(archive_root)
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in REQUIRED_MEMBERS:
        path = root / name
        if not path.is_file():
            missing.append(name)
            continue
        records.append(
            {
                "name": name,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if missing:
        raise RuntimeConsumerError("missing charged runtime member(s): " + ", ".join(missing))
    class_codebook = _verify_class_codebook(root / "class_codebook.json")
    label_prior_payload_manifest = _verify_label_prior_payload_manifest(
        root / "label_prior_payload_manifest.json"
    )
    runtime_proof_skeleton = _load_runtime_proof_skeleton(
        root / "runtime_consumer_proof_skeleton.json"
    )
    categorical_payload = _consume_payload(root / "categorical_payload.bin")
    consumed_names = [record["name"] for record in records]
    output_core = {
        "charged_members": records,
        "class_codebook": class_codebook,
        "label_prior_payload_manifest": label_prior_payload_manifest,
        "runtime_proof_skeleton": runtime_proof_skeleton,
        "categorical_payload": categorical_payload,
    }
    return {
        "schema_version": 1,
        "kind": "categorical_charged_runtime_consumer_report",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "runtime_executed": True,
        "sidecar_free": True,
        "fallback_used": False,
        "consumed_charged_members": consumed_names,
        "charged_members_verified": records,
        "class_codebook": class_codebook,
        "label_prior_payload_manifest": label_prior_payload_manifest,
        "runtime_proof_skeleton": runtime_proof_skeleton,
        "categorical_payload": categorical_payload,
        "runtime_output_sha256": _sha256_bytes(_json_bytes(output_core)),
        "runtime_output_scope": "charged_member_contract_parse_and_hpm1_model_load",
        "runtime_contract_consumption_proven": categorical_payload.get("codec") == "HPM1"
        and categorical_payload.get("structural_reencode", {}).get("passed") is True
        and categorical_payload.get("hpac_model_load", {}).get("passed") is True,
        "runtime_output_parity_proven": False,
        "full_decode_reencode_parity_proven": False,
        "dispatch_blockers": [
            "decode_reencode_parity_missing",
            "semantic_runtime_output_parity_missing",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path("."))
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    try:
        payload = verify_charged_members(args.archive_root)
    except RuntimeConsumerError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
