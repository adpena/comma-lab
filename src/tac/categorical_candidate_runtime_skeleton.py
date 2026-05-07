"""Fail-closed runtime skeleton for categorical payload candidates.

This module is intentionally not a contest decoder. It is packaged as a
charged archive member by the categorical payload builder so local manifests can
prove that inflate-time state is archive-contained while decode/re-encode and
runtime-output parity remain explicit blockers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
CONTEST_SEGNET_CLASS_ORDER = (
    "road",
    "lane_markings",
    "undrivable",
    "movable",
    "my_car",
)


class RuntimeSkeletonError(RuntimeError):
    """Raised when the charged-member skeleton contract is not satisfied."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_label_prior_payload_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeSkeletonError(f"label prior payload manifest is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeSkeletonError("label prior payload manifest must be a JSON object")
    if payload.get("schema_version") != 1:
        raise RuntimeSkeletonError("label prior payload manifest schema_version mismatch")
    if payload.get("kind") != "categorical_label_prior_payload_manifest":
        raise RuntimeSkeletonError("label prior payload manifest kind mismatch")
    if payload.get("label_prior_payload_manifest_contract") != LABEL_PRIOR_PAYLOAD_MANIFEST_CONTRACT:
        raise RuntimeSkeletonError("label prior payload manifest contract mismatch")
    if payload.get("score_claim") is not False or payload.get("dispatch_attempted") is not False:
        raise RuntimeSkeletonError("label prior payload manifest must not claim score or dispatch")
    if payload.get("ready_for_exact_eval_dispatch") is not False:
        raise RuntimeSkeletonError("label prior payload manifest cannot claim exact-eval readiness")
    if payload.get("label_contract") != RUNTIME_LABEL_CONTRACT:
        raise RuntimeSkeletonError("label prior payload manifest label contract mismatch")
    if payload.get("semantic_class_order") != list(CONTEST_SEGNET_CLASS_ORDER):
        raise RuntimeSkeletonError("label prior payload manifest class order mismatch")
    return {
        "contract": payload.get("label_prior_payload_manifest_contract", ""),
        "label_contract": payload.get("label_contract", ""),
        "conditioning_prior_count": len(payload.get("conditioning_priors", []))
        if isinstance(payload.get("conditioning_priors"), list)
        else 0,
    }


def verify_charged_members(archive_root: str | Path) -> dict[str, Any]:
    """Verify required charged members without loading any uncharged sidecars."""

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
        raise RuntimeSkeletonError("missing charged runtime member(s): " + ", ".join(missing))
    label_prior_payload_manifest = _verify_label_prior_payload_manifest(
        root / "label_prior_payload_manifest.json"
    )
    return {
        "schema_version": 1,
        "kind": "categorical_runtime_skeleton_member_check",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "charged_members_verified": records,
        "label_prior_payload_manifest": label_prior_payload_manifest,
        "runtime_output_parity_proven": False,
        "dispatch_blockers": [
            "categorical_runtime_skeleton_not_a_decoder",
            "decode_reencode_parity_missing",
            "runtime_output_parity_missing",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    try:
        payload = verify_charged_members(args.archive_root)
    except RuntimeSkeletonError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True), end="\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
