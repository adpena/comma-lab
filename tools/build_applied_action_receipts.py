#!/usr/bin/env python3
"""Materialize fail-closed J8F/PF3/J12 applied-action adaptation results."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.analysis.applied_action_adapters import (
    SourceArtifactIdentity,
    adapt_j8f_checkpoints,
    adapt_j12_receipt,
    adapt_pf3_checkpoints,
    build_adapter_manifest,
    load_json_artifact,
    verify_source_artifact,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_J8F_ROOT = Path(
    "/Volumes/VertigoDataTier/pact/experiments/results/ddm_j8f_counted_application_20260724T181414Z"
)
DEFAULT_J8F_CONFIG = REPO_ROOT / ".omx/research/configs/ddm_j8f_counted_application_20260724.json"
DEFAULT_PF3_RECEIPT = REPO_ROOT / ".omx/research/ddm_pf3_finite_price_materialization_20260725T193409Z/receipt.json"
DEFAULT_J12_RECEIPT = REPO_ROOT / ".omx/research/ddm_j12_366_receiver_coordinate_custody_receipt_20260725.json"


def _checkpoint_paths(root: Path) -> list[Path]:
    paths = sorted((root / "checkpoints").glob("application_step_*.json"))
    if not paths:
        raise FileNotFoundError(f"no J8F application checkpoints under {root}")
    return paths


def _physical_path(display_path: str) -> Path:
    path = Path(display_path)
    return path if path.is_absolute() else REPO_ROOT / path


def _declared_refs(value: Any, *, context: str) -> list[tuple[str, int | None, str, str]]:
    """Find path/SHA declarations, including prefix_path/prefix_sha256 forms."""

    found: list[tuple[str, int | None, str, str]] = []
    if isinstance(value, Mapping):
        endpoint = value.get("endpoint")
        if isinstance(value.get("archive_path"), str) and isinstance(endpoint, Mapping):
            endpoint_sha = endpoint.get("archive_sha256")
            endpoint_bytes = endpoint.get("archive_bytes")
            if isinstance(endpoint_sha, str):
                found.append(
                    (
                        value["archive_path"],
                        endpoint_bytes if isinstance(endpoint_bytes, int) else None,
                        endpoint_sha,
                        f"{context}.archive_endpoint",
                    )
                )
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            found.append(
                (
                    value["path"],
                    value.get("bytes") if isinstance(value.get("bytes"), int) else None,
                    value["sha256"],
                    str(value.get("role") or context),
                )
            )
        for key, path in value.items():
            if not key.endswith("_path") or not isinstance(path, str):
                continue
            prefix = key.removesuffix("_path")
            sha = value.get(f"{prefix}_sha256")
            if isinstance(sha, str):
                size = value.get(f"{prefix}_bytes")
                found.append((path, size if isinstance(size, int) else None, sha, f"{context}.{prefix}"))
        for key, child in value.items():
            found.extend(_declared_refs(child, context=f"{context}.{key}"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            found.extend(_declared_refs(child, context=f"{context}[{index}]"))
    return found


def _verify_refs(
    refs: Sequence[tuple[str, int | None, str, str]],
    *,
    skip_paths: Sequence[str] = (),
) -> tuple[SourceArtifactIdentity, ...]:
    skipped = set(skip_paths)
    by_path: dict[str, SourceArtifactIdentity] = {}
    for display, size, sha, role in refs:
        if display in skipped:
            continue
        identity = verify_source_artifact(
            _physical_path(display),
            role=role,
            expected_bytes=size,
            expected_sha256=sha,
            display_path=display,
        )
        previous = by_path.get(display)
        if previous is not None and (previous.bytes, previous.sha256) != (
            identity.bytes,
            identity.sha256,
        ):
            raise ValueError(f"conflicting custody declarations for {display}")
        by_path.setdefault(display, identity)
    return tuple(sorted(by_path.values(), key=lambda item: (item.path, item.role)))


def _j8f_inputs(
    root: Path, config_path: Path
) -> tuple[
    Mapping[str, Any],
    list[Mapping[str, Any]],
    Mapping[str, Any],
    SourceArtifactIdentity,
    list[SourceArtifactIdentity],
    SourceArtifactIdentity,
    tuple[SourceArtifactIdentity, ...],
]:
    receipt_path = root / "ddm_j8f_counted_application_receipt.json"
    smoke, smoke_artifact = load_json_artifact(receipt_path, role="J8F primary receipt")
    config, config_artifact = load_json_artifact(config_path, role="J8F typed config")
    checkpoints: list[Mapping[str, Any]] = []
    checkpoint_artifacts: list[SourceArtifactIdentity] = []
    for path in _checkpoint_paths(root):
        payload, artifact = load_json_artifact(path, role="J8F cumulative checkpoint")
        checkpoints.append(payload)
        checkpoint_artifacts.append(artifact)

    refs = _declared_refs(config.get("source_bindings"), context="J8F.config.source_bindings")
    refs.extend(_declared_refs(smoke.get("preflight"), context="J8F.preflight"))
    refs.extend(_declared_refs(smoke.get("range_gauge_projected_arm", {}).get("archive"), context="J8F.final"))
    custody = list(_verify_refs(refs))

    verdict_path = root / "verdicts/range_gauge_projected_n600.json"
    verdict, verdict_artifact = load_json_artifact(verdict_path, role="J8F final n600 verdict")
    if verdict != smoke.get("range_gauge_projected_arm", {}).get("verdict"):
        raise ValueError("J8F embedded and file-backed final verdicts differ")
    custody.append(verdict_artifact)
    custody = sorted(custody, key=lambda item: (item.path, item.role))
    return (
        smoke,
        checkpoints,
        config,
        smoke_artifact,
        checkpoint_artifacts,
        config_artifact,
        tuple(custody),
    )


def _pf3_inputs(
    receipt_path: Path,
) -> tuple[
    Mapping[str, Any],
    list[Mapping[str, Any]],
    SourceArtifactIdentity,
    list[SourceArtifactIdentity],
    tuple[SourceArtifactIdentity, ...],
]:
    receipt, receipt_artifact = load_json_artifact(receipt_path, role="PF3 primary receipt")
    declared = receipt.get("inventory", {}).get("candidate_checkpoint_custody", {}).get("artifacts")
    if not isinstance(declared, list):
        raise ValueError("PF3 receipt does not expose checkpoint custody")
    checkpoints: list[Mapping[str, Any]] = []
    checkpoint_artifacts: list[SourceArtifactIdentity] = []
    for index, row in enumerate(declared):
        if not isinstance(row, Mapping):
            raise ValueError("PF3 checkpoint custody row is not an object")
        display = str(row["path"])
        payload, artifact = load_json_artifact(
            _physical_path(display),
            role=f"PF3 measured checkpoint {index + 1:02d}",
            expected_bytes=int(row["bytes"]),
            expected_sha256=str(row["sha256"]),
            display_path=display,
        )
        checkpoints.append(payload)
        checkpoint_artifacts.append(artifact)
    refs = _declared_refs(receipt.get("source_custody"), context="PF3.source_custody")
    for index, checkpoint in enumerate(checkpoints):
        refs.extend(_declared_refs(checkpoint, context=f"PF3.checkpoint[{index}]"))
    custody = _verify_refs(refs, skip_paths=[artifact.path for artifact in checkpoint_artifacts])
    return receipt, checkpoints, receipt_artifact, checkpoint_artifacts, custody


def _j12_inputs(
    receipt_path: Path,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    SourceArtifactIdentity,
    SourceArtifactIdentity,
    tuple[SourceArtifactIdentity, ...],
]:
    receipt, receipt_artifact = load_json_artifact(receipt_path, role="J12 compact receipt")
    pricing_ref = receipt.get("pricing")
    if not isinstance(pricing_ref, Mapping):
        raise ValueError("J12 compact receipt lacks pricing custody")
    pricing_display = str(pricing_ref["receipt_path"])
    pricing, pricing_artifact = load_json_artifact(
        _physical_path(pricing_display),
        role="J12 decomposition pricing receipt",
        expected_sha256=str(pricing_ref["receipt_sha256"]),
        display_path=pricing_display,
    )

    refs = _declared_refs(receipt, context="J12.compact")
    refs.extend(_declared_refs(pricing, context="J12.pricing"))
    step8 = receipt.get("pc1_adapter", {}).get("step8")
    if not isinstance(step8, Mapping):
        raise ValueError("J12 compact receipt lacks step8 endpoint")
    step8_path = (
        _physical_path(pricing_display).parents[1]
        / "03_pc1_rehome/W_joint_step50_live__pc1_accepted_008/archive.zip.receipt-bytes"
    )
    refs.append(
        (
            str(step8_path),
            int(step8["archive_bytes"]),
            str(step8["archive_sha256"]),
            "J12.pc1_adapter.step8.archive_endpoint",
        )
    )
    full_ref = receipt.get("full_receipt")
    if not isinstance(full_ref, Mapping):
        raise ValueError("J12 compact receipt lacks full receipt custody")
    full_display = str(full_ref["path"])
    full, _ = load_json_artifact(
        _physical_path(full_display),
        role="J12 full receipt",
        expected_sha256=str(full_ref["sha256"]),
        display_path=full_display,
    )
    refs.extend(_declared_refs(full, context="J12.full"))
    custody = _verify_refs(refs, skip_paths=(pricing_display,))
    return receipt, pricing, receipt_artifact, pricing_artifact, custody


def materialize(
    *,
    j8f_root: Path,
    j8f_config_path: Path,
    pf3_receipt_path: Path,
    j12_receipt_path: Path,
) -> dict[str, object]:
    """Adapt every exact primary artifact after byte-level custody verification."""

    smoke, checkpoints, config, smoke_id, checkpoint_ids, config_id, j8_custody = _j8f_inputs(j8f_root, j8f_config_path)
    j8f = adapt_j8f_checkpoints(
        smoke,
        checkpoints,
        config,
        smoke_artifact=smoke_id,
        checkpoint_artifacts=checkpoint_ids,
        config_artifact=config_id,
        custody_artifacts=j8_custody,
    )
    pf_receipt, pf_checkpoints, pf_id, pf_checkpoint_ids, pf_custody = _pf3_inputs(pf3_receipt_path)
    pf3 = adapt_pf3_checkpoints(
        pf_receipt,
        pf_checkpoints,
        receipt_artifact=pf_id,
        checkpoint_artifacts=pf_checkpoint_ids,
        custody_artifacts=pf_custody,
    )
    j12_receipt, pricing, j12_id, pricing_id, j12_custody = _j12_inputs(j12_receipt_path)
    j12 = adapt_j12_receipt(
        j12_receipt,
        pricing,
        receipt_artifact=j12_id,
        pricing_artifact=pricing_id,
        custody_artifacts=j12_custody,
    )
    return build_adapter_manifest((*j8f, pf3, j12))


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--j8f-root", type=Path, default=DEFAULT_J8F_ROOT)
    parser.add_argument("--j8f-config", type=Path, default=DEFAULT_J8F_CONFIG)
    parser.add_argument("--pf3-receipt", type=Path, default=DEFAULT_PF3_RECEIPT)
    parser.add_argument("--j12-receipt", type=Path, default=DEFAULT_J12_RECEIPT)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = materialize(
        j8f_root=args.j8f_root,
        j8f_config_path=args.j8f_config,
        pf3_receipt_path=args.pf3_receipt,
        j12_receipt_path=args.j12_receipt,
    )
    _atomic_write_json(args.output, manifest)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "content_sha256": manifest["content_sha256"],
                "receipt_count": manifest["receipt_count"],
                "blocked_source_count": manifest["blocked_source_count"],
                "source_artifact_count": manifest["source_artifact_count"],
                "research_only": True,
                "score_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
