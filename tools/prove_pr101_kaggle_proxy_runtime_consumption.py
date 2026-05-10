#!/usr/bin/env python3
"""Prove local runtime consumption for the PR101 Kaggle proxy packet.

This is a static, local-only proof. It reads the runtime packet manifest emitted
by ``tools/build_pr101_kaggle_proxy_runtime_packet.py``, verifies the unchanged
archive custody, verifies that ``inflate.py`` contains only the three supported
bias-param replacements, and writes ``runtime_consumption_proof.json``.

It does not run inflate, invoke scorers, dispatch jobs, or claim score.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import json_text, read_json, repo_relative, sha256_file, write_json  # noqa: E402

TOOL_NAME = "tools/prove_pr101_kaggle_proxy_runtime_consumption.py"
PACKET_SCHEMA = "pr101_kaggle_proxy_runtime_packet_v1"
PROOF_SCHEMA = "pr101_kaggle_proxy_runtime_consumption_proof_v1"
DEFAULT_MANIFEST = Path(
    "experiments/results/kaggle_pr101_proxy_sweep_20260510_codex/"
    "pr101_proxy_sweep/proxy_runtime_packet/runtime_packet_manifest.json"
)
PROOF_NAME = "runtime_consumption_proof.json"
SUPPORTED_BIAS_SLOTS = {
    "bias_r": "up[:, 0, 0]",
    "bias_b": "up[:, 0, 2]",
    "bias_g": "up[:, 1, 1]",
}
OLD_PR101_LINES = {
    "bias_r": "up[:, 0, 0].sub_(1.0)",
    "bias_b": "up[:, 0, 2].sub_(1.0)",
    "bias_g": "up[:, 1, 1].sub_(1.0)",
}
UNSUPPORTED_PARAMS = ("delta_scale", "latent_delta_scale", "smooth_weight")
FALSE_AUTHORITY_FIELDS = {
    "score_claim": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
}


class RuntimeConsumptionProofError(ValueError):
    """Raised when the local runtime-consumption proof must fail closed."""


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_rel(path: Path) -> str:
    return repo_relative(_repo_path(path), REPO_ROOT)


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeConsumptionProofError(f"{field} must be a JSON object")
    return value


def _require_false(payload: Mapping[str, Any], field: str) -> None:
    if payload.get(field) is not False:
        raise RuntimeConsumptionProofError(f"{field} must be false")


def _canonical_json_sha256(payload: Any) -> str:
    import hashlib

    return hashlib.sha256(json_text(payload).encode("utf-8")).hexdigest()


def _verify_manifest_self_hash(manifest: Mapping[str, Any]) -> None:
    expected = manifest.get("manifest_sha256_excluding_self")
    if not isinstance(expected, str):
        raise RuntimeConsumptionProofError("manifest_sha256_excluding_self must be present")
    basis = dict(manifest)
    basis.pop("manifest_sha256_excluding_self", None)
    actual = _canonical_json_sha256(basis)
    if actual != expected:
        raise RuntimeConsumptionProofError("runtime packet manifest self-hash mismatch")


def _resolve_packet_file(packet_dir: Path, record: Mapping[str, Any], field: str) -> Path:
    relpath = record.get("relpath")
    if not isinstance(relpath, str) or not relpath:
        raise RuntimeConsumptionProofError(f"{field}.relpath must be a non-empty string")
    path = Path(relpath)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeConsumptionProofError(f"{field}.relpath must stay inside packet_dir")
    return packet_dir / path


def _verify_archive_unchanged(manifest: Mapping[str, Any], packet_dir: Path) -> dict[str, Any]:
    if manifest.get("archive_changed") is not False:
        raise RuntimeConsumptionProofError("archive_changed must be false")
    source = _require_mapping(manifest.get("source_archive"), "source_archive")
    packet = _require_mapping(manifest.get("packet_archive"), "packet_archive")
    source_sha = source.get("sha256")
    packet_sha = packet.get("sha256")
    unchanged_sha = manifest.get("archive_unchanged_sha256")
    if not all(isinstance(value, str) and len(value) == 64 for value in (source_sha, packet_sha, unchanged_sha)):
        raise RuntimeConsumptionProofError("archive SHA fields must be 64-char hex strings")
    if source_sha != packet_sha or packet_sha != unchanged_sha:
        raise RuntimeConsumptionProofError("source, packet, and unchanged archive SHA fields must match")
    archive_path = _resolve_packet_file(packet_dir, packet, "packet_archive")
    if not archive_path.is_file():
        raise RuntimeConsumptionProofError(f"packet archive missing: {archive_path}")
    actual_sha = sha256_file(archive_path)
    if actual_sha != packet_sha:
        raise RuntimeConsumptionProofError("packet archive file SHA does not match manifest")
    return {
        "archive_path": _repo_rel(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": actual_sha,
        "manifest_archive_sha256": packet_sha,
    }


def _runtime_file_record(manifest: Mapping[str, Any], relpath: str) -> Mapping[str, Any]:
    custody = _require_mapping(manifest.get("runtime_custody"), "runtime_custody")
    files = custody.get("runtime_files")
    if not isinstance(files, list):
        raise RuntimeConsumptionProofError("runtime_custody.runtime_files must be a list")
    matches = [
        _require_mapping(row, "runtime_custody.runtime_files[]")
        for row in files
        if isinstance(row, Mapping) and row.get("relpath") == relpath
    ]
    if len(matches) != 1:
        raise RuntimeConsumptionProofError(f"expected exactly one runtime file record for {relpath}")
    return matches[0]


def _verify_inflate_file(manifest: Mapping[str, Any], packet_dir: Path) -> dict[str, Any]:
    patch = _require_mapping(manifest.get("runtime_patch"), "runtime_patch")
    if patch.get("patched_file") != "inflate.py":
        raise RuntimeConsumptionProofError("runtime_patch.patched_file must be inflate.py")
    inflate_record = _runtime_file_record(manifest, "inflate.py")
    inflate_path = packet_dir / "inflate.py"
    if not inflate_path.is_file():
        raise RuntimeConsumptionProofError(f"inflate.py missing: {inflate_path}")
    actual_sha = sha256_file(inflate_path)
    if inflate_record.get("sha256") != actual_sha:
        raise RuntimeConsumptionProofError("inflate.py SHA does not match runtime custody manifest")

    text = inflate_path.read_text(encoding="utf-8")
    rows_raw = patch.get("runtime_consumed_params")
    if not isinstance(rows_raw, list):
        raise RuntimeConsumptionProofError("runtime_patch.runtime_consumed_params must be a list")
    rows = [_require_mapping(row, "runtime_patch.runtime_consumed_params[]") for row in rows_raw]
    params_seen: set[str] = set()
    consumed_rows: list[dict[str, Any]] = []
    for row in rows:
        param = row.get("param")
        replacement = row.get("replacement")
        slot = row.get("slot")
        if not isinstance(param, str) or param not in SUPPORTED_BIAS_SLOTS:
            raise RuntimeConsumptionProofError(f"unsupported runtime-consumed param in manifest: {param!r}")
        if param in params_seen:
            raise RuntimeConsumptionProofError(f"duplicate runtime-consumed param in manifest: {param}")
        params_seen.add(param)
        if slot != SUPPORTED_BIAS_SLOTS[param]:
            raise RuntimeConsumptionProofError(f"slot mismatch for {param}")
        if not isinstance(replacement, str) or not replacement:
            raise RuntimeConsumptionProofError(f"replacement missing for {param}")
        if text.count(replacement) != 1:
            raise RuntimeConsumptionProofError(f"inflate.py must contain exactly one manifest replacement for {param}")
        consumed_rows.append(
            {
                "param": param,
                "slot": slot,
                "replacement": replacement,
                "value": row.get("value"),
                "occurrences": 1,
            }
        )
    if params_seen != set(SUPPORTED_BIAS_SLOTS):
        missing = sorted(set(SUPPORTED_BIAS_SLOTS) - params_seen)
        raise RuntimeConsumptionProofError(f"missing supported bias replacements: {missing}")

    for param, old_line in OLD_PR101_LINES.items():
        if old_line in text:
            raise RuntimeConsumptionProofError(f"old PR101 bias line remains in inflate.py for {param}")

    for param in UNSUPPORTED_PARAMS:
        if param in text:
            raise RuntimeConsumptionProofError(f"unsupported param name appears in inflate.py: {param}")

    return {
        "inflate_path": _repo_rel(inflate_path),
        "inflate_sha256": actual_sha,
        "supported_bias_params_consumed": sorted(consumed_rows, key=lambda row: row["param"]),
        "old_pr101_sub_lines_absent": sorted(OLD_PR101_LINES.values()),
        "unsupported_param_names_absent_from_inflate_py": list(UNSUPPORTED_PARAMS),
    }


def _verify_unsupported_params_remain_blocked(manifest: Mapping[str, Any]) -> dict[str, Any]:
    unsupported = _require_mapping(manifest.get("unsupported_params"), "unsupported_params")
    blockers = manifest.get("blockers")
    if not isinstance(blockers, list) or not all(isinstance(row, str) for row in blockers):
        raise RuntimeConsumptionProofError("blockers must be a list of strings")
    consumed = _require_mapping(manifest.get("runtime_consumed_params"), "runtime_consumed_params")
    if set(consumed) != set(SUPPORTED_BIAS_SLOTS):
        raise RuntimeConsumptionProofError("runtime_consumed_params must contain only supported bias params")

    rows: dict[str, Any] = {}
    for param in UNSUPPORTED_PARAMS:
        row = _require_mapping(unsupported.get(param), f"unsupported_params.{param}")
        blocker = f"{param}_not_runtime_consumed"
        if row.get("runtime_consumed") is not False:
            raise RuntimeConsumptionProofError(f"unsupported_params.{param}.runtime_consumed must be false")
        if row.get("blocker") != blocker:
            raise RuntimeConsumptionProofError(f"unsupported_params.{param}.blocker must be {blocker}")
        if blocker not in blockers:
            raise RuntimeConsumptionProofError(f"missing blocker: {blocker}")
        if param in consumed:
            raise RuntimeConsumptionProofError(f"unsupported param consumed by runtime: {param}")
        rows[param] = dict(row)
    if "unsupported_proxy_params_not_runtime_consumed" not in blockers:
        raise RuntimeConsumptionProofError("missing unsupported_proxy_params_not_runtime_consumed blocker")
    return {
        "unsupported_params": rows,
        "required_blockers_present": [
            "unsupported_proxy_params_not_runtime_consumed",
            *[f"{param}_not_runtime_consumed" for param in UNSUPPORTED_PARAMS],
        ],
    }


def build_runtime_consumption_proof(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    proof_path: Path | None = None,
) -> dict[str, Any]:
    """Build and write a local proof for a PR101 Kaggle proxy runtime packet."""

    manifest_path = _repo_path(manifest_path)
    manifest = _require_mapping(read_json(manifest_path), "runtime_packet_manifest")
    if manifest.get("schema") != PACKET_SCHEMA:
        raise RuntimeConsumptionProofError(f"manifest schema must be {PACKET_SCHEMA!r}")
    _verify_manifest_self_hash(manifest)
    for field in (
        "score_claim",
        "score_claim_valid",
        "ready_for_exact_eval_dispatch",
        "dispatch_attempted",
        "exact_auth_eval_performed",
        "contest_cuda_auth_eval",
        "scorers_invoked",
    ):
        _require_false(manifest, field)

    packet_dir_value = manifest.get("packet_dir")
    if not isinstance(packet_dir_value, str) or not packet_dir_value:
        raise RuntimeConsumptionProofError("packet_dir must be a non-empty string")
    packet_dir = _repo_path(Path(packet_dir_value))
    if not packet_dir.is_dir():
        raise RuntimeConsumptionProofError(f"packet_dir missing: {packet_dir}")

    archive_proof = _verify_archive_unchanged(manifest, packet_dir)
    inflate_proof = _verify_inflate_file(manifest, packet_dir)
    unsupported_proof = _verify_unsupported_params_remain_blocked(manifest)

    if proof_path is None:
        proof_path = packet_dir / PROOF_NAME
    else:
        proof_path = _repo_path(proof_path)

    proof: dict[str, Any] = {
        "schema": PROOF_SCHEMA,
        "tool": TOOL_NAME,
        "candidate_id": manifest.get("candidate_id", ""),
        "manifest_path": _repo_rel(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "packet_dir": _repo_rel(packet_dir),
        "proof_path": _repo_rel(proof_path),
        "archive_unchanged_proof": archive_proof,
        "inflate_static_consumption_proof": inflate_proof,
        "unsupported_params_blocker_proof": unsupported_proof,
        "runtime_consumption_proven_for_supported_bias_params": True,
        "unsupported_proxy_params_runtime_consumed": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "dispatch_blockers": [
            "proxy_substrate_not_contest_exact_eval",
            "no_contest_cuda_auth_eval",
            "unsupported_proxy_params_not_runtime_consumed",
            "active_level2_lane_dispatch_claim_required_before_exact_eval",
        ],
    }
    proof["proof_sha256_excluding_self"] = _canonical_json_sha256(proof)
    write_json(proof_path, proof)
    return proof


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--proof-path", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    proof = build_runtime_consumption_proof(
        manifest_path=args.manifest,
        proof_path=args.proof_path,
    )
    print(json_text({
        "schema": "pr101_kaggle_proxy_runtime_consumption_proof_stdout_v1",
        "proof": proof["proof_path"],
        "candidate_id": proof["candidate_id"],
        **FALSE_AUTHORITY_FIELDS,
        "proof_sha256_excluding_self": proof["proof_sha256_excluding_self"],
        "dispatch_blockers": proof["dispatch_blockers"],
    }), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
