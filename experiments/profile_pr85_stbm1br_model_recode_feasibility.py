#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile model-only PR85/STBM1BR recode feasibility.

This is a local guard for the PR85_STBM1BR frontier archive. It reuses the
reviewed QH0/QM0 and QFQ4 model serializer screens, then fails closed unless a
candidate archive is byte-positive, decoded-model-parity clean, runtime
compatible, and proven to change only the PR85 ``model`` segment.

No scorer, CUDA eval, lane claim, or remote dispatch is used here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.analyze_or_build_pr85_qfq4_model_serializer_candidate import (  # noqa: E402
    build_or_block_probe as build_qfq4_probe,
)
from experiments.build_pr85_qh0_serializer_candidates import (  # noqa: E402
    build_candidates as build_qh0_candidates,
)
from tac.pr85_bundle import SEGMENT_ORDER, parse_pr85_bundle, validate_pr85_member_name  # noqa: E402
from tac.qh0_record_serializer import sha256_bytes  # noqa: E402


TOOL = "experiments/profile_pr85_stbm1br_model_recode_feasibility.py"
SCHEMA = "pr85_stbm1br_model_recode_feasibility_v1"
BLOCKER_SCHEMA = "pr85_stbm1br_model_recode_dispatch_blocker_v1"
DEFAULT_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr85_stbm1br_mask_recode_20260504_worker/"
    "pr90_stbm1br_lossless_pr85_mask_recode/archive.zip"
)
DEFAULT_REPLAY_INFLATE = (
    REPO_ROOT
    / "experiments/results/pr85_stbm1br_mask_recode_20260504_worker/"
    "replay_submission_stbm/inflate.py"
)
DEFAULT_ROBUST_CURRENT = REPO_ROOT / "submissions/robust_current"
DEFAULT_OUT_DIR = (
    REPO_ROOT / "experiments/results/pr85_stbm1br_model_recode_feasibility_20260504_codex"
)
ORIGINAL_VIDEO_BYTES = 37_545_489


class STBMModelRecodeError(RuntimeError):
    """Raised when the local model-recode feasibility profile cannot be built."""


def _repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_single_x_archive(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise STBMModelRecodeError(f"archive not found: {_repo_rel(path)}")
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != ["x"]:
            raise STBMModelRecodeError(
                f"PR85/STBM archive must contain exactly one member 'x'; got {names!r}"
            )
        info = infos[0]
        validate_pr85_member_name(info.filename)
        payload = zf.read(info)
    return (
        {
            "path": _repo_rel(path),
            "archive_bytes": int(path.stat().st_size),
            "archive_sha256": _sha256_file(path),
            "member_name": "x",
            "member_bytes": len(payload),
            "member_sha256": sha256_bytes(payload),
            "zip_overhead_bytes": int(path.stat().st_size) - len(payload),
            "zip_compress_type": int(info.compress_type),
        },
        payload,
    )


def _segment_fingerprints(segments: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "bytes": len(segments[name]),
            "sha256": sha256_bytes(segments[name]),
            "magic_hex": segments[name][:8].hex(),
        }
        for name in SEGMENT_ORDER
    }


def _candidate_archive_audit(
    *,
    candidate_archive: Path,
    source_segments: Mapping[str, bytes],
) -> dict[str, Any]:
    meta, payload = _read_single_x_archive(candidate_archive)
    bundle = parse_pr85_bundle(payload)
    changed = [
        name
        for name in SEGMENT_ORDER
        if bytes(bundle.segments[name]) != bytes(source_segments[name])
    ]
    forbidden_changed = [name for name in changed if name != "model"]
    return {
        "candidate_archive": meta,
        "changed_segments": changed,
        "forbidden_changed_segments": forbidden_changed,
        "model_changed": "model" in changed,
        "only_model_changed": changed == ["model"],
        "non_model_segments_preserved": not forbidden_changed,
        "segment_fingerprints": _segment_fingerprints(bundle.segments),
    }


def _audit_child_candidates(
    *,
    child_summary: Mapping[str, Any],
    source_segments: Mapping[str, bytes],
) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    manifests = child_summary.get("candidate_manifests", [])
    if not isinstance(manifests, list):
        return audits
    for manifest in manifests:
        if not isinstance(manifest, Mapping):
            continue
        archive_path_raw = manifest.get("archive_path")
        if not archive_path_raw:
            continue
        archive_path = Path(str(archive_path_raw))
        if not archive_path.is_absolute():
            archive_path = REPO_ROOT / archive_path
        audits.append(
            _candidate_archive_audit(
                candidate_archive=archive_path,
                source_segments=source_segments,
            )
        )
    return audits


def _best_qfq4_byte_screen(qfq4_summary: Mapping[str, Any], source_archive_bytes: int) -> dict[str, Any] | None:
    row = qfq4_summary.get("best_screened_candidate")
    if not isinstance(row, Mapping):
        return None
    delta = int(row.get("outer_pr85_model_delta_bytes_vs_source", 0))
    return {
        "candidate_id": row.get("candidate_id"),
        "model_delta_bytes_vs_source": delta,
        "archive_delta_bytes_vs_source_formula": delta,
        "projected_archive_bytes_if_components_identical_formula_only": int(source_archive_bytes)
        + delta,
        "rate_score_delta_if_components_identical_formula_only": delta
        * 25.0
        / ORIGINAL_VIDEO_BYTES,
        "decoded_tensor_parity": row.get("decoded_tensor_parity"),
        "runtime_compatibility": row.get("runtime_compatibility"),
        "build_blockers": row.get("build_blockers"),
    }


def _best_qh0_byte_screen(qh0_summary: Mapping[str, Any], source_archive_bytes: int) -> dict[str, Any] | None:
    row = qh0_summary.get("best_screened_candidate")
    if not isinstance(row, Mapping):
        return None
    delta = int(row.get("candidate_model_delta_bytes_vs_source", 0))
    return {
        "candidate_id": row.get("candidate_id"),
        "model_delta_bytes_vs_source": delta,
        "archive_delta_bytes_vs_source_formula": delta,
        "projected_archive_bytes_if_components_identical_formula_only": int(source_archive_bytes)
        + delta,
        "rate_score_delta_if_components_identical_formula_only": delta
        * 25.0
        / ORIGINAL_VIDEO_BYTES,
        "decoded_tensor_parity": row.get("decoded_tensor_parity"),
        "runtime_compatibility": row.get("runtime_compatibility"),
    }


def _build_dispatch_blocker(
    *,
    source_archive: Mapping[str, Any],
    source_bundle: Mapping[str, Any],
    qh0_summary: Mapping[str, Any],
    qfq4_summary: Mapping[str, Any],
    blockers: Sequence[str],
    candidate_audits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the fail-closed PR85/STBM model-recode dispatch contract."""

    forbidden_candidate_audits = [
        audit for audit in candidate_audits if audit.get("forbidden_changed_segments")
    ]
    qfq4_best = qfq4_summary.get("best_screened_candidate")
    qh0_best = qh0_summary.get("best_screened_candidate")
    qfq4_parity = (
        qfq4_best.get("decoded_tensor_parity")
        if isinstance(qfq4_best, Mapping)
        else None
    )
    qfq4_runtime = (
        qfq4_best.get("runtime_compatibility")
        if isinstance(qfq4_best, Mapping)
        else qfq4_summary.get("runtime_compatibility")
    )
    return {
        "schema": BLOCKER_SCHEMA,
        "tool": TOOL,
        "dispatch": False,
        "score_claim": False,
        "remote_gpu_dispatch_performed": False,
        "blocker_class": "no_byte_closed_model_recode_candidate",
        "blocker": "; ".join(blockers),
        "source_archive": source_archive,
        "source_bundle": source_bundle,
        "owned_surface": "model",
        "forbidden_surfaces": [name for name in SEGMENT_ORDER if name != "model"],
        "candidate_archive_emitted": bool(candidate_audits),
        "candidate_archive_audits": list(candidate_audits),
        "blockers": list(blockers),
        "qh0_qm0_contract": {
            "blocker_class": qh0_summary.get("blocker_class"),
            "blocker": qh0_summary.get("blocker"),
            "best_screened_candidate": qh0_best,
            "decision": "reject_as_byte_neutral" if not qh0_summary.get("built_candidate_count") else "candidate_built",
        },
        "qfq4_contract": {
            "blocker_class": qfq4_summary.get("blocker_class"),
            "blocker": qfq4_summary.get("blocker"),
            "best_screened_candidate": qfq4_best,
            "decoded_tensor_parity_gate": qfq4_parity,
            "runtime_compatibility_gate": qfq4_runtime,
            "decision": "fail_closed_no_archive_candidate",
        },
        "runtime_output_parity_gate": {
            "passed": False,
            "status": "not_run_qfq4_tensor_parity_and_runtime_loader_blocked",
            "required": True,
            "requirement": (
                "Any QFQ4 drift-tolerant design must first add scored-runtime decode "
                "support and prove source-vs-candidate renderer output parity locally."
            ),
        },
        "model_only_segment_gate": {
            "passed": not forbidden_candidate_audits,
            "violation_count": len(forbidden_candidate_audits),
            "requirement": (
                "An emitted candidate may change only the PR85 model segment; mask, pose, "
                "post, shift, frac, frac2, frac3, bias, region, and randmulti must be byte-identical."
            ),
        },
        "required_before_dispatch": [
            "byte_positive_model_recode_archive",
            "decoded_model_tensor_parity_or_reviewed_runtime_output_parity",
            "robust_current_compatible_QFQ4_or_QH0_QM0_loader",
            "local_renderer_output_parity_on_source_vs_candidate",
            "model_only_archive_audit",
            "fresh_lane_claim_before_any_remote_exact_eval",
        ],
        "adversarial_decision": (
            "Do not emit PR85_STBM1BR QFQ4 from the current formula-only screen. "
            "The only byte-positive QFQ4 row changes decoded model tensors and the "
            "scored runtime has no QFQ4 loader, so dispatch is false."
        ),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_feasibility_profile(
    *,
    archive: Path = DEFAULT_ARCHIVE,
    out_dir: Path = DEFAULT_OUT_DIR,
    replay_inflate_py: Path = DEFAULT_REPLAY_INFLATE,
    robust_current_dir: Path = DEFAULT_ROBUST_CURRENT,
    qh0_qualities: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    qh0_lgwins: Sequence[int] = (18, 20, 22, 24),
    qfq4_qrow_policies: Sequence[str] = ("shifted_int8_rows", "all_fp16_rows"),
) -> dict[str, Any]:
    """Run local model-only screens and emit a fail-closed STBM feasibility profile."""

    source_archive, x_payload = _read_single_x_archive(archive)
    bundle = parse_pr85_bundle(x_payload)
    source_segments = {name: bytes(bundle.segments[name]) for name in SEGMENT_ORDER}

    qh0_dir = out_dir / "qh0_serializer"
    qfq4_dir = out_dir / "qfq4_serializer"
    qh0_summary = build_qh0_candidates(
        archive,
        qh0_dir,
        qualities=qh0_qualities,
        lgwins=qh0_lgwins,
        replay_inflate_py=replay_inflate_py,
        robust_current_dir=robust_current_dir,
    )
    qfq4_summary = build_qfq4_probe(
        archive=archive,
        out_dir=qfq4_dir,
        replay_inflate_py=replay_inflate_py,
        robust_current_dir=robust_current_dir,
        qrow_policies=qfq4_qrow_policies,
    )

    candidate_audits = []
    candidate_audits.extend(
        {"source_screen": "qh0_qm0", **audit}
        for audit in _audit_child_candidates(
            child_summary=qh0_summary,
            source_segments=source_segments,
        )
    )
    candidate_audits.extend(
        {"source_screen": "qfq4", **audit}
        for audit in _audit_child_candidates(
            child_summary=qfq4_summary,
            source_segments=source_segments,
        )
    )
    forbidden_violations = [
        audit
        for audit in candidate_audits
        if audit.get("forbidden_changed_segments")
    ]
    model_only_guard_passed = not forbidden_violations

    built_candidates = int(qh0_summary.get("built_candidate_count", 0)) + int(
        qfq4_summary.get("built_candidate_count", 0)
    )
    exact_eval_ready = bool(built_candidates) and model_only_guard_passed
    blockers: list[str] = []
    if not built_candidates:
        blockers.append("no_byte_closed_model_recode_candidate_built")
    if not model_only_guard_passed:
        blockers.append("candidate_changed_forbidden_non_model_segments")
    qh0_blocker = qh0_summary.get("blocker_class")
    if qh0_blocker:
        blockers.append(f"qh0_qm0:{qh0_blocker}")
    qfq4_blocker = qfq4_summary.get("blocker_class")
    if qfq4_blocker:
        blockers.append(f"qfq4:{qfq4_blocker}")

    source_bundle = {
        "format": bundle.format,
        "header_bytes": bundle.header_bytes,
        "segment_lengths": bundle.segment_lengths,
        "segment_fingerprints": _segment_fingerprints(source_segments),
    }
    dispatch_blocker = (
        None
        if exact_eval_ready
        else _build_dispatch_blocker(
            source_archive=source_archive,
            source_bundle=source_bundle,
            qh0_summary=qh0_summary,
            qfq4_summary=qfq4_summary,
            blockers=blockers,
            candidate_audits=candidate_audits,
        )
    )

    profile = {
        "schema": SCHEMA,
        "tool": TOOL,
        "dispatch": False,
        "planning_only": not exact_eval_ready,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_gpu_dispatch_performed": False,
        "source_archive": source_archive,
        "source_bundle": source_bundle,
        "owned_surface": "model",
        "forbidden_surfaces": [name for name in SEGMENT_ORDER if name != "model"],
        "qh0_qm0_screen": {
            "summary_path": _repo_rel(qh0_dir / "candidate_summary.json"),
            "built_candidate_count": qh0_summary.get("built_candidate_count"),
            "dispatch_unlocked": qh0_summary.get("dispatch_unlocked"),
            "blocker_class": qh0_summary.get("blocker_class"),
            "blocker": qh0_summary.get("blocker"),
            "best_byte_screen": _best_qh0_byte_screen(
                qh0_summary,
                int(source_archive["archive_bytes"]),
            ),
        },
        "qfq4_screen": {
            "summary_path": _repo_rel(qfq4_dir / "candidate_summary.json"),
            "built_candidate_count": qfq4_summary.get("built_candidate_count"),
            "dispatch_unlocked": qfq4_summary.get("dispatch_unlocked"),
            "blocker_class": qfq4_summary.get("blocker_class"),
            "blocker": qfq4_summary.get("blocker"),
            "best_byte_screen": _best_qfq4_byte_screen(
                qfq4_summary,
                int(source_archive["archive_bytes"]),
            ),
        },
        "model_only_candidate_guard": {
            "passed": model_only_guard_passed,
            "candidate_archive_audits": candidate_audits,
            "violation_count": len(forbidden_violations),
            "requirement": "Any emitted archive must preserve mask, pose, post, shift, frac, frac2, frac3, bias, region, and randmulti bytes exactly.",
        },
        "exact_eval_readiness": {
            "ready_after_lane_claim": exact_eval_ready,
            "remote_gpu_dispatch_performed": False,
            "requires_lane_claim_before_remote_eval": True,
            "blockers": blockers,
        },
        "structured_blocker_json": _repo_rel(out_dir / "dispatch_blocker.json")
        if dispatch_blocker is not None
        else None,
        "structured_blocker": dispatch_blocker,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidate_summary.json").write_bytes(_json_bytes(profile))
    if dispatch_blocker is not None:
        (out_dir / "dispatch_blocker.json").write_bytes(_json_bytes(dispatch_blocker))
    # Re-read to ensure the emitted artifact is strict JSON and matches callers.
    return _read_json(out_dir / "candidate_summary.json")


def _parse_int_csv(text: str, *, label: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in text.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        try:
            values.append(int(stripped))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{label} must be comma-separated ints") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{label} must not be empty")
    return tuple(values)


def _parse_str_csv(text: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("must contain at least one value")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--replay-inflate-py", type=Path, default=DEFAULT_REPLAY_INFLATE)
    parser.add_argument("--robust-current-dir", type=Path, default=DEFAULT_ROBUST_CURRENT)
    parser.add_argument("--qh0-qualities", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--qh0-lgwins", default="18,20,22,24")
    parser.add_argument("--qfq4-qrow-policies", type=_parse_str_csv, default=("shifted_int8_rows", "all_fp16_rows"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    profile = build_feasibility_profile(
        archive=args.archive,
        out_dir=args.out_dir,
        replay_inflate_py=args.replay_inflate_py,
        robust_current_dir=args.robust_current_dir,
        qh0_qualities=_parse_int_csv(args.qh0_qualities, label="qh0 qualities"),
        qh0_lgwins=_parse_int_csv(args.qh0_lgwins, label="qh0 lgwins"),
        qfq4_qrow_policies=args.qfq4_qrow_policies,
    )
    print(
        json.dumps(
            {
                "summary_path": _repo_rel(args.out_dir / "candidate_summary.json"),
                "source_archive_bytes": profile["source_archive"]["archive_bytes"],
                "qh0_qm0_best": profile["qh0_qm0_screen"]["best_byte_screen"],
                "qfq4_best": profile["qfq4_screen"]["best_byte_screen"],
                "model_only_guard_passed": profile["model_only_candidate_guard"]["passed"],
                "exact_eval_readiness": profile["exact_eval_readiness"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
