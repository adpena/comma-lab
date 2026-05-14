#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize a reversed-base CDO1 economics row as a byte-closed archive.

The economics planner is intentionally local-only: it prices candidate base
mask payloads plus CDO1 overlays but does not emit archives.  This bridge keeps
that boundary explicit.  It rebuilds the selected CDO1 payload from the
economics row, verifies the raw SHA against the planner output, then delegates
archive construction to the reviewed CDO1 overlay builder.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.build_c067_decoded_delta_overlay_candidate import (  # noqa: E402
    build_candidate,
)
from experiments.plan_c067_decoded_delta_overlay_mask_topology import (  # noqa: E402
    RUN_STRUCT_NAME,
    _encode_overlay_payload,
    _load_mask_array,
    _mask_tensor_sha256,
    _resolve_pair_index_basis,
    _runs_from_selected,
    _selected_pair_indices_from_frame_indices,
)
from experiments.plan_c067_reversed_base_cdo1_overlay_economics import (  # noqa: E402
    DEFAULT_OUTPUT_JSON,
    SCHEMA as ECONOMICS_SCHEMA,
    TOOL as ECONOMICS_TOOL,
    _display_path,
    _selected_pixels_for_policy,
    _sha256_bytes,
)


SCHEMA = "c067_reversed_base_cdo1_candidate_v1"
TOOL = "experiments/build_c067_reversed_base_cdo1_candidate.py"
DEFAULT_BASE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_c067_cmg3_nonzero_top1_t4_20260502T1100Z/archive.zip"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/c067_reversed_base_cdo1_candidate_20260503"
DEFAULT_OUTPUT_ARCHIVE = DEFAULT_OUTPUT_DIR / "archive.zip"
DEFAULT_MANIFEST_JSON = DEFAULT_OUTPUT_DIR / "c067_reversed_base_cdo1_candidate_manifest.json"


class ReversedBaseBuildError(ValueError):
    """Raised when an economics row cannot be materialized safely."""


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReversedBaseBuildError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ReversedBaseBuildError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _resolve_repo_path(repo_root: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return repo_root / path


def _candidate_rows(economics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = economics.get("all_candidates")
    if not isinstance(rows, list):
        raise ReversedBaseBuildError("economics JSON lacks all_candidates")
    return [row for row in rows if isinstance(row, dict)]


def _select_candidate(
    economics: dict[str, Any],
    *,
    candidate_id: str | None,
) -> dict[str, Any]:
    rows = _candidate_rows(economics)
    if candidate_id:
        for row in rows:
            if row.get("candidate_id") == candidate_id:
                return row
        raise ReversedBaseBuildError(f"candidate_id not found: {candidate_id}")
    if not rows:
        raise ReversedBaseBuildError("economics JSON contains no candidates")
    return rows[0]


def _selected_policy_summary(
    *,
    candidate: dict[str, Any],
    rebuild: dict[str, Any],
) -> dict[str, Any]:
    policy = candidate.get("policy") if isinstance(candidate.get("policy"), dict) else {}
    payload = (
        candidate.get("cdo1_payload")
        if isinstance(candidate.get("cdo1_payload"), dict)
        else {}
    )
    compressed = (
        payload.get("compressed_payloads")
        if isinstance(payload.get("compressed_payloads"), dict)
        else {}
    )
    selected_atoms = policy.get("selected_atoms")
    if not isinstance(selected_atoms, list):
        selected_atoms = []
    atom_summaries: list[dict[str, Any]] = []
    for atom in selected_atoms:
        if not isinstance(atom, dict):
            continue
        atom_summaries.append(
            {
                "atom_id": atom.get("atom_id"),
                "class_id": atom.get("class_id"),
                "pair_indices": atom.get("pair_indices"),
                "changed_pixels": atom.get("changed_pixels"),
                "expected_component_score_improvement_first_order": atom.get(
                    "expected_component_score_improvement_first_order"
                ),
            }
        )
    return {
        "candidate_id": candidate.get("candidate_id"),
        "policy_id": policy.get("policy_id"),
        "selection": policy.get("selection"),
        "source": policy.get("source"),
        "pair_index_basis": rebuild.get("pair_index_basis"),
        "base_vs_target_pair_indices": rebuild.get("base_vs_target_pair_indices"),
        "selected_pair_indices": rebuild["payload_header"].get(
            "selected_pair_indices"
        ),
        "residual_vs_target_pair_indices_after_overlay": rebuild.get(
            "residual_vs_target_pair_indices_after_overlay"
        ),
        "selected_class_ids": policy.get("class_ids"),
        "selected_atom_count": len(atom_summaries),
        "selected_atoms": atom_summaries,
        "rebuilt_run_count": rebuild.get("rebuilt_run_count"),
        "rebuilt_selected_pixel_count": rebuild.get(
            "rebuilt_selected_pixel_count"
        ),
        "rebuilt_raw_bytes": rebuild.get("rebuilt_raw_bytes"),
        "rebuilt_raw_sha256": rebuild.get("rebuilt_raw_sha256"),
        "expected_raw_sha256": rebuild.get("expected_raw_sha256"),
        "planner_cdo1_payload": {
            "raw_bytes": payload.get("raw_bytes"),
            "raw_sha256": payload.get("raw_sha256"),
            "recommended_compressor": payload.get("recommended_compressor"),
            "run_count": payload.get("run_count"),
            "compressed_payloads": compressed,
        },
    }


def _rebuild_payload(
    *,
    economics: dict[str, Any],
    candidate: dict[str, Any],
    repo_root: Path,
) -> tuple[bytes, dict[str, Any]]:
    target_info = economics.get("target_decoded_mask")
    if not isinstance(target_info, dict) or not isinstance(target_info.get("path"), str):
        raise ReversedBaseBuildError("economics JSON lacks target_decoded_mask.path")
    base_info = candidate.get("base")
    if not isinstance(base_info, dict) or not isinstance(
        base_info.get("decoded_mask_array"), str
    ):
        raise ReversedBaseBuildError("candidate lacks base.decoded_mask_array")
    policy = candidate.get("policy")
    if not isinstance(policy, dict):
        raise ReversedBaseBuildError("candidate lacks policy object")

    target_path = _resolve_repo_path(repo_root, target_info["path"])
    base_path = _resolve_repo_path(repo_root, base_info["decoded_mask_array"])
    target = _load_mask_array(target_path)
    base = _load_mask_array(base_path)
    if base.shape != target.shape:
        raise ReversedBaseBuildError(
            f"base shape {base.shape} differs from target shape {target.shape}"
        )

    gates = economics.get("gates") if isinstance(economics.get("gates"), dict) else {}
    pair_index_basis = str(gates.get("pair_index_basis") or "auto")
    resolved_pair_index_basis = _resolve_pair_index_basis(
        int(target.shape[0]),
        pair_index_basis,
    )
    candidate_basis = None
    mask_disagreement = candidate.get("mask_disagreement")
    if isinstance(mask_disagreement, dict):
        candidate_basis = mask_disagreement.get("pair_index_basis")
    if candidate_basis is not None and str(candidate_basis) != resolved_pair_index_basis:
        raise ReversedBaseBuildError(
            "candidate mask_disagreement pair_index_basis "
            f"{candidate_basis!r} does not match economics gate "
            f"{resolved_pair_index_basis!r}"
        )
    max_residual = float(
        gates.get(
            "max_residual_disagreement_fraction",
            (candidate.get("mask_disagreement") or {}).get(
                "max_residual_disagreement_fraction",
                0.001,
            ),
        )
    )
    selected = _selected_pixels_for_policy(
        base,
        target,
        policy,
        max_residual_disagreement_fraction=max_residual,
        pair_index_basis=resolved_pair_index_basis,
    )
    final = base.copy()
    final[selected] = target[selected]
    diff = base != target
    residual = final != target
    runs = _runs_from_selected(base, target, selected)
    header = {
        "schema": "c067_decoded_delta_overlay_payload_v1",
        "producer": ECONOMICS_TOOL,
        "score_claim": False,
        "base_mask_tensor_sha256": _mask_tensor_sha256(base),
        "target_mask_tensor_sha256": _mask_tensor_sha256(target),
        "reconstructed_mask_u8_sha256": _mask_tensor_sha256(final),
        "shape": [int(value) for value in target.shape],
        "pair_index_basis": resolved_pair_index_basis,
        "run_struct": RUN_STRUCT_NAME,
        "run_count": len(runs),
        "selected_pixel_count": int(selected.sum()),
        "selected_pair_indices": _selected_pair_indices_from_frame_indices(
            np.nonzero(selected)[0],
            frame_count=int(target.shape[0]),
            pair_index_basis=resolved_pair_index_basis,
        ),
        "policy_id": policy.get("policy_id"),
    }
    raw = _encode_overlay_payload(runs=runs, header=header)
    expected = (candidate.get("cdo1_payload") or {}).get("raw_sha256")
    actual = _sha256_bytes(raw)
    if expected and expected != actual:
        raise ReversedBaseBuildError(
            f"rebuilt CDO1 raw SHA mismatch: expected={expected} actual={actual}"
        )
    return raw, {
        "target_mask_array": _display_path(target_path, repo_root),
        "base_mask_array": _display_path(base_path, repo_root),
        "pair_index_basis": resolved_pair_index_basis,
        "max_residual_disagreement_fraction": max_residual,
        "base_vs_target_pair_indices": _selected_pair_indices_from_frame_indices(
            np.nonzero(diff)[0],
            frame_count=int(target.shape[0]),
            pair_index_basis=resolved_pair_index_basis,
        ),
        "selected_pair_indices": header["selected_pair_indices"],
        "residual_vs_target_pair_indices_after_overlay": _selected_pair_indices_from_frame_indices(
            np.nonzero(residual)[0],
            frame_count=int(target.shape[0]),
            pair_index_basis=resolved_pair_index_basis,
        ),
        "rebuilt_raw_bytes": len(raw),
        "rebuilt_raw_sha256": actual,
        "rebuilt_run_count": len(runs),
        "rebuilt_selected_pixel_count": int(selected.sum()),
        "expected_raw_sha256": expected,
        "payload_header": header,
    }


def build_reversed_base_candidate(
    *,
    economics_json: Path = DEFAULT_OUTPUT_JSON,
    candidate_id: str | None = None,
    base_archive: Path = DEFAULT_BASE_ARCHIVE,
    output_archive: Path = DEFAULT_OUTPUT_ARCHIVE,
    manifest_json: Path = DEFAULT_MANIFEST_JSON,
    overlay_compressor: str | None = None,
    pack_output_payload: bool = True,
    packed_payload_member_name: str = "p",
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    economics = _read_json(economics_json)
    schema = economics.get("schema")
    if schema != ECONOMICS_SCHEMA:
        raise ReversedBaseBuildError(
            f"unexpected economics schema: {schema!r}; expected {ECONOMICS_SCHEMA!r}"
        )
    candidate = _select_candidate(economics, candidate_id=candidate_id)
    raw, rebuild = _rebuild_payload(
        economics=economics,
        candidate=candidate,
        repo_root=repo_root,
    )

    with tempfile.TemporaryDirectory(prefix="reversed_base_cdo1_") as tmp:
        raw_path = Path(tmp) / "overlay.cdo1"
        raw_path.write_bytes(raw)
        archive_report = build_candidate(
            base_archive=base_archive,
            overlay_payload=raw_path,
            output_archive=output_archive,
            manifest_json=None,
            overlay_compressor=overlay_compressor,
            pack_output_payload=pack_output_payload,
            packed_payload_member_name=packed_payload_member_name,
            repo_root=repo_root,
        )

    report = {
        "schema": SCHEMA,
        "producer": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "auth_eval_required": True,
        "economics_json": _display_path(economics_json, repo_root),
        "selected_economics_candidate": candidate,
        "selected_policy_summary": _selected_policy_summary(
            candidate=candidate,
            rebuild=rebuild,
        ),
        "rebuilt_cdo1_payload": rebuild,
        "base_archive_role": "reversed_base_mask_payload_archive",
        "archive_report": archive_report,
    }
    import hashlib

    report["economics_json_sha256"] = hashlib.sha256(
        economics_json.read_bytes()
    ).hexdigest()
    _write_json(manifest_json, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--economics-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--candidate-id")
    parser.add_argument("--base-archive", type=Path, default=DEFAULT_BASE_ARCHIVE)
    parser.add_argument("--output-archive", type=Path, default=DEFAULT_OUTPUT_ARCHIVE)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    parser.add_argument(
        "--overlay-compressor",
        choices=("brotli", "lzma_xz", "raw", "zlib"),
        default=None,
    )
    parser.add_argument("--no-pack-output-payload", action="store_true")
    parser.add_argument(
        "--packed-payload-member-name",
        choices=("p", "renderer_payload.bin.br"),
        default="p",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_reversed_base_candidate(
        economics_json=args.economics_json,
        candidate_id=args.candidate_id,
        base_archive=args.base_archive,
        output_archive=args.output_archive,
        manifest_json=args.manifest_json,
        overlay_compressor=args.overlay_compressor,
        pack_output_payload=not args.no_pack_output_payload,
        packed_payload_member_name=args.packed_payload_member_name,
        repo_root=args.repo_root,
    )
    archive = report["archive_report"]["output_archive"]
    print(
        json.dumps(
            {
                "archive": archive,
                "candidate_id": report["selected_economics_candidate"].get(
                    "candidate_id"
                ),
                "score_claim": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
