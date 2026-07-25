#!/usr/bin/env python3
"""Materialize fail-closed J8F/PF3/J12 applied-action adaptation results."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from tac.analysis.applied_action_adapters import (
    adapt_j8f_checkpoints,
    adapt_j12_receipt,
    adapt_pf3_checkpoint,
    build_adapter_manifest,
    load_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_J8F_ROOT = Path(
    "/Volumes/VertigoDataTier/pact/experiments/results/"
    "ddm_j8f_counted_application_20260724T181414Z"
)
DEFAULT_J8F_CONFIG = REPO_ROOT / ".omx/research/configs/ddm_j8f_counted_application_20260724.json"
DEFAULT_PF3_RECEIPT = (
    REPO_ROOT / ".omx/research/ddm_pf3_finite_price_materialization_20260725T193409Z/receipt.json"
)
DEFAULT_J12_RECEIPT = (
    REPO_ROOT / ".omx/research/ddm_j12_366_receiver_coordinate_custody_receipt_20260725.json"
)


def _checkpoint_paths(root: Path) -> list[Path]:
    paths = sorted((root / "checkpoints").glob("application_step_*.json"))
    if not paths:
        raise FileNotFoundError(f"no J8F application checkpoints under {root}")
    return paths


def materialize(
    *,
    j8f_root: Path,
    j8f_config_path: Path,
    pf3_receipt_path: Path,
    j12_receipt_path: Path,
) -> dict[str, object]:
    """Adapt the available primary artifacts and parse every emitted receipt."""

    j8f_receipt_path = j8f_root / "ddm_j8f_counted_application_receipt.json"
    checkpoint_paths = _checkpoint_paths(j8f_root)
    j8f = adapt_j8f_checkpoints(
        load_json(j8f_receipt_path),
        [load_json(path) for path in checkpoint_paths],
        load_json(j8f_config_path),
    )

    pf3_receipt = load_json(pf3_receipt_path)
    checkpoint_artifacts = (
        pf3_receipt.get("inventory", {})
        .get("candidate_checkpoint_custody", {})
        .get("artifacts", [])
    )
    if not isinstance(checkpoint_artifacts, list) or not checkpoint_artifacts:
        raise ValueError("PF3 receipt does not expose candidate checkpoint custody")
    pf3_checkpoint_path = Path(str(checkpoint_artifacts[0]["path"]))
    pf3 = adapt_pf3_checkpoint(
        pf3_receipt,
        load_json(pf3_checkpoint_path),
        source_id=str(pf3_checkpoint_path),
    )
    j12 = adapt_j12_receipt(
        load_json(j12_receipt_path),
        source_id=str(j12_receipt_path),
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
    print(json.dumps({
        "output": str(args.output),
        "content_sha256": manifest["content_sha256"],
        "receipt_count": manifest["receipt_count"],
        "blocked_source_count": manifest["blocked_source_count"],
        "research_only": True,
        "score_claim": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
