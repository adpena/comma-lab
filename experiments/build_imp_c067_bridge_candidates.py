#!/usr/bin/env python3
"""Build C067-anchored IMP bridge archive candidates.

This is a bridge between the old Lane J IMP work and the current C067 public
floor archive. It is deliberately build-only: it applies deterministic
global-magnitude pruning to the decoded C067 JointFrameGenerator state, repacks
the renderer through the current QZS3/runtime single-blob container, and writes
byte-screen manifests. It does not claim score.

The intended use is to answer the wall-clock question before spending a long
H100 run: does C067/JFG/QZS3 have a byte path that could make IMP worth a
full train-and-rewind cycle, or is IMP only useful as a sparsity prior for a
different self-compression export?
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import build_blockfp_c067_archive as blockfp_builder  # noqa: E402
from tac.iterative_magnitude_pruning import (  # noqa: E402
    apply_mask_to_model,
    compute_actual_sparsity,
    iter_prunable_parameters,
    prune_lowest_magnitude,
)
from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer  # noqa: E402
from tac.quantizr_qzs3_codec import decode_qzs3_state_dict, encode_qzs3_state_dict  # noqa: E402

SCHEMA = "imp_c067_bridge_candidate_builder_v1"
SUMMARY_SCHEMA = "imp_c067_bridge_summary_v1"
ORIGINAL_VIDEO_BYTES = 37_545_489
DEFAULT_CYCLE_COUNTS = (1, 2, 5, 10)
DEFAULT_QZS3_BLOCK_SIZES = (16, 24, 32, 48, 64, 96, 128)
RENDERER_MEMBER_NAME = "renderer.bin"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def parse_cycle_counts(value: str) -> tuple[int, ...]:
    """Parse comma-separated IMP cycle counts for argparse/tests."""

    raw = [item.strip() for item in value.split(",") if item.strip()]
    if not raw:
        raise argparse.ArgumentTypeError("cycle-count list must not be empty")
    out: list[int] = []
    for item in raw:
        try:
            count = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid cycle count {item!r} in {value!r}"
            ) from exc
        if count <= 0 or count > 32:
            raise argparse.ArgumentTypeError(
                f"cycle count must be in [1, 32], got {count}"
            )
        if count not in out:
            out.append(count)
    return tuple(out)


def _load_c067_joint_frame_generator(
    renderer_bytes: bytes,
) -> tuple[Any, dict[str, Any]]:
    """Decode a current C067-compatible JFG renderer into a torch module."""

    magic = renderer_bytes[:4]
    if magic != b"QZS3":
        raise ValueError(
            "IMP-C067 bridge currently requires a QZS3 JointFrameGenerator "
            f"renderer; got magic {magic!r}. Do not route old Lane G/ASYM "
            "archives through this current-anchor bridge."
        )
    state = decode_qzs3_state_dict(renderer_bytes, device="cpu")
    model = build_quantizr_faithful_renderer().eval()
    model.load_state_dict(state, strict=True)
    block_size = int.from_bytes(renderer_bytes[4:6], "little")
    prunable = iter_prunable_parameters(model)
    prunable_values = sum(int(param.numel()) for _, param in prunable)
    return model, {
        "renderer_wire_format": "QZS3",
        "source_qzs3_block_size": block_size,
        "state_tensor_count": len(state),
        "prunable_tensor_count": len(prunable),
        "prunable_value_count": prunable_values,
    }


def _mask_stats(mask: dict[str, Any]) -> dict[str, Any]:
    tensors: list[dict[str, Any]] = []
    total = 0
    kept = 0
    for name in sorted(mask):
        m = mask[name].detach().cpu().bool()
        count = int(m.numel())
        kept_count = int(m.sum().item())
        total += count
        kept += kept_count
        tensors.append(
            {
                "name": name,
                "value_count": count,
                "kept": kept_count,
                "pruned": count - kept_count,
                "sparsity": (count - kept_count) / count if count else 0.0,
            }
        )
    return {
        "tensor_count": len(tensors),
        "value_count": total,
        "kept": kept,
        "pruned": total - kept,
        "sparsity": (total - kept) / total if total else 0.0,
        "tensors": tensors,
    }


def _build_candidate_archive(
    *,
    runtime_members: dict[str, bytes],
    transformed_renderer: bytes,
    source_archive_sha256: str,
    candidate_dir: Path,
    payload_format: str,
    payload_member_name: str,
    brotli_quality: int,
) -> dict[str, Any]:
    """Build one deterministic packed archive and verify runtime unpack."""

    import brotli

    members = dict(runtime_members)
    members[RENDERER_MEMBER_NAME] = transformed_renderer
    ordered = blockfp_builder._ordered_three_member_payload(members)
    payload, payload_header = blockfp_builder._build_payload(
        ordered,
        source_archive_sha256=source_archive_sha256,
        payload_format=payload_format,
    )
    compressed = brotli.compress(payload, quality=brotli_quality, lgwin=24)
    if brotli.decompress(compressed) != payload:
        raise RuntimeError("Brotli round-trip mismatch for IMP-C067 payload")

    candidate_dir.mkdir(parents=True, exist_ok=True)
    archive_path = candidate_dir / "archive.zip"
    blockfp_builder.PACKER.write_deterministic_payload_archive(
        archive_path,
        compressed,
        payload_member_name=payload_member_name,
    )
    runtime_unpack_check = blockfp_builder._verify_output_archive(
        archive_path,
        payload_member_name=payload_member_name,
        expected_renderer=transformed_renderer,
    )
    return {
        "archive_path": archive_path,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": _sha256_file(archive_path),
        "payload_raw_bytes": len(payload),
        "payload_compressed_bytes": len(compressed),
        "payload_header": payload_header,
        "runtime_unpack_check": runtime_unpack_check,
    }


def build_imp_c067_bridge_candidates(
    *,
    source_archive: Path,
    output_dir: Path,
    cycle_counts: tuple[int, ...] = DEFAULT_CYCLE_COUNTS,
    sparsity_increment: float = 0.20,
    qzs3_block_sizes: tuple[int, ...] = DEFAULT_QZS3_BLOCK_SIZES,
    payload_member_name: str = blockfp_builder.PACKER.SHORT_PAYLOAD_MEMBER_NAME,
    payload_format: str = blockfp_builder.PACKER.PAYLOAD_FORMAT_PUBLIC_PR64_MASK_FIRST_LEN_TABLE,
    brotli_quality: int = 11,
    force: bool = False,
) -> dict[str, Any]:
    """Build deterministic no-train IMP bridge byte-screen candidates."""

    if not (0.0 < sparsity_increment < 1.0):
        raise ValueError(
            f"sparsity_increment must be in (0, 1), got {sparsity_increment}"
        )
    if not 0 <= brotli_quality <= 11:
        raise ValueError(f"brotli_quality must be in [0, 11], got {brotli_quality}")
    cycle_counts = parse_cycle_counts(",".join(str(x) for x in cycle_counts))
    qzs3_block_sizes = blockfp_builder.parse_block_sizes(
        ",".join(str(x) for x in qzs3_block_sizes)
    )

    source_archive = source_archive.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(
            f"output directory is non-empty; pass --force to overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_bytes = source_archive.read_bytes()
    source_sha = _sha256_bytes(source_bytes)
    runtime_members, source_packaging = blockfp_builder.extract_runtime_members(
        source_archive
    )
    if RENDERER_MEMBER_NAME not in runtime_members:
        raise ValueError("source archive does not contain logical renderer.bin")
    source_renderer = runtime_members[RENDERER_MEMBER_NAME]
    source_renderer_sha = _sha256_bytes(source_renderer)
    model, renderer_contract = _load_c067_joint_frame_generator(source_renderer)

    max_cycles = max(cycle_counts)
    requested = set(cycle_counts)
    current_mask = None
    cycle_manifests: list[dict[str, Any]] = []
    best_candidates: list[dict[str, Any]] = []

    for cycle in range(1, max_cycles + 1):
        current_mask = prune_lowest_magnitude(
            model,
            sparsity_increment=sparsity_increment,
            current_mask=current_mask,
        )
        apply_mask_to_model(model, current_mask)
        if cycle not in requested:
            continue

        sparsity = compute_actual_sparsity(model, current_mask)
        mask_stats = _mask_stats(current_mask)
        block_candidates: list[dict[str, Any]] = []
        for block_size in qzs3_block_sizes:
            transformed_renderer = encode_qzs3_state_dict(
                model.state_dict(),
                block_size=block_size,
            )
            # Strict decode/load parity before writing archive bytes.
            decoded = decode_qzs3_state_dict(transformed_renderer, device="cpu")
            parity_model = build_quantizr_faithful_renderer()
            parity_model.load_state_dict(decoded, strict=True)

            candidate_id = f"imp_c{cycle:02d}_qzs3_b{block_size:04d}"
            candidate_dir = output_dir / candidate_id
            built = _build_candidate_archive(
                runtime_members=runtime_members,
                transformed_renderer=transformed_renderer,
                source_archive_sha256=source_sha,
                candidate_dir=candidate_dir,
                payload_format=payload_format,
                payload_member_name=payload_member_name,
                brotli_quality=brotli_quality,
            )
            archive_delta = built["archive_bytes"] - len(source_bytes)
            renderer_delta = len(transformed_renderer) - len(source_renderer)
            manifest = {
                "schema": SCHEMA,
                "candidate_id": candidate_id,
                "score_claim": False,
                "promotion_eligible": False,
                "evidence_grade": "empirical_archive_candidate_until_exact_cuda",
                "bridge_mode": "no_train_global_magnitude_prune",
                "canonical_score_source_required": (
                    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
                    "experiments/contest_auth_eval.py --device cuda"
                ),
                "source_archive": {
                    "path": str(source_archive),
                    "bytes": len(source_bytes),
                    "sha256": source_sha,
                    **source_packaging,
                },
                "renderer_source_member": {
                    "member_name": RENDERER_MEMBER_NAME,
                    "bytes": len(source_renderer),
                    "sha256": source_renderer_sha,
                    **renderer_contract,
                },
                "imp_pruning": {
                    "cycle": cycle,
                    "sparsity_increment": sparsity_increment,
                    "expected_sparsity": 1.0 - (1.0 - sparsity_increment) ** cycle,
                    "actual_sparsity": sparsity,
                    "mask_stats": mask_stats,
                    "training_applied": False,
                    "score_claim": False,
                    "full_imp_training_required_before_promotion": True,
                },
                "transformed_renderer_payload": {
                    "member_name": RENDERER_MEMBER_NAME,
                    "wire_format": "QZS3",
                    "transform": "c067_qzs3_decoded_jfg_no_train_imp_prune_reencoded_qzs3",
                    "qzs3_block_size": block_size,
                    "bytes": len(transformed_renderer),
                    "sha256": _sha256_bytes(transformed_renderer),
                    "delta_bytes_vs_source_renderer": renderer_delta,
                },
                "output_archive": {
                    "path": str(built["archive_path"]),
                    "bytes": built["archive_bytes"],
                    "sha256": built["archive_sha256"],
                    "delta_bytes_vs_source_archive": archive_delta,
                    "formula_only_rate_delta_vs_source_archive": (
                        25.0 * archive_delta / ORIGINAL_VIDEO_BYTES
                    ),
                },
                "packed_payload": {
                    "payload_format": payload_format,
                    "payload_member": payload_member_name,
                    "payload_raw_bytes": built["payload_raw_bytes"],
                    "payload_compressed_bytes": built["payload_compressed_bytes"],
                    "brotli_quality": brotli_quality,
                    "header": built["payload_header"],
                    **built["runtime_unpack_check"],
                },
                "decision_support": {
                    "local_archive_byte_win": archive_delta < 0,
                    "local_renderer_byte_win": renderer_delta < 0,
                    "exact_cuda_auth_eval_required_for_score": True,
                    "safe_to_promote_from_this_manifest": False,
                    "recommended_use": (
                        "If byte-positive and exact CUDA components survive, "
                        "launch a real trained IMP-C067 H100 run; otherwise "
                        "use the mask as a sparsity prior for Block-FP/self-compression."
                    ),
                },
            }
            manifest_path = candidate_dir / "build_manifest.json"
            manifest_path.write_bytes(_json_bytes(manifest))
            block_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "cycle": cycle,
                    "qzs3_block_size": block_size,
                    "archive_path": str(built["archive_path"]),
                    "archive_bytes": built["archive_bytes"],
                    "archive_sha256": built["archive_sha256"],
                    "delta_bytes_vs_source_archive": archive_delta,
                    "formula_only_rate_delta_vs_source_archive": (
                        25.0 * archive_delta / ORIGINAL_VIDEO_BYTES
                    ),
                    "renderer_bytes": len(transformed_renderer),
                    "renderer_sha256": _sha256_bytes(transformed_renderer),
                    "delta_bytes_vs_source_renderer": renderer_delta,
                    "actual_sparsity": sparsity,
                    "manifest_path": str(manifest_path),
                    "local_archive_byte_win": archive_delta < 0,
                }
            )

        best_for_cycle = min(
            block_candidates,
            key=lambda item: (
                item["archive_bytes"],
                item["renderer_bytes"],
                item["qzs3_block_size"],
            ),
        )
        cycle_manifests.append(
            {
                "cycle": cycle,
                "expected_sparsity": 1.0 - (1.0 - sparsity_increment) ** cycle,
                "actual_sparsity": sparsity,
                "best_candidate_id": best_for_cycle["candidate_id"],
                "best_archive_bytes": best_for_cycle["archive_bytes"],
                "best_delta_bytes_vs_source_archive": best_for_cycle[
                    "delta_bytes_vs_source_archive"
                ],
                "candidates": block_candidates,
            }
        )
        best_candidates.append(best_for_cycle)

    best = min(
        best_candidates,
        key=lambda item: (
            item["archive_bytes"],
            item["delta_bytes_vs_source_archive"],
            item["cycle"],
            item["qzs3_block_size"],
        ),
    )
    unchanged_distortion_score_delta = (
        25.0 * best["delta_bytes_vs_source_archive"] / ORIGINAL_VIDEO_BYTES
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "source_archive": {
            "path": str(source_archive),
            "bytes": len(source_bytes),
            "sha256": source_sha,
        },
        "source_renderer": {
            "bytes": len(source_renderer),
            "sha256": source_renderer_sha,
            **renderer_contract,
        },
        "bridge_mode": "no_train_global_magnitude_prune",
        "sparsity_increment": sparsity_increment,
        "cycle_counts": list(cycle_counts),
        "qzs3_block_sizes": list(qzs3_block_sizes),
        "best_by_output_archive_bytes": {
            **best,
            "unchanged_distortion_score_delta": unchanged_distortion_score_delta,
        },
        "cycles": cycle_manifests,
        "decision": {
            "byte_screen_positive": best["delta_bytes_vs_source_archive"] < 0,
            "exact_cuda_dispatch_recommended": best["delta_bytes_vs_source_archive"] < 0,
            "full_h100_imp_training_recommended": (
                best["delta_bytes_vs_source_archive"] < 0
                and math.isfinite(unchanged_distortion_score_delta)
            ),
            "caveat": (
                "This is an untrained bridge. A positive byte screen only "
                "justifies exact diagnostic or full H100 training; it is not "
                "evidence that no-train IMP preserves scorer components."
            ),
        },
    }
    (output_dir / "imp_c067_bridge_summary.json").write_bytes(_json_bytes(summary))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cycle-counts", type=parse_cycle_counts, default=DEFAULT_CYCLE_COUNTS)
    parser.add_argument("--sparsity-increment", type=float, default=0.20)
    parser.add_argument(
        "--qzs3-block-sizes",
        type=blockfp_builder.parse_block_sizes,
        default=DEFAULT_QZS3_BLOCK_SIZES,
    )
    parser.add_argument(
        "--payload-member-name",
        choices=blockfp_builder.PACKER.ALLOWED_PAYLOAD_MEMBER_NAMES,
        default=blockfp_builder.PACKER.SHORT_PAYLOAD_MEMBER_NAME,
    )
    parser.add_argument(
        "--payload-format",
        choices=blockfp_builder.PACKER.ALLOWED_PAYLOAD_FORMATS,
        default=blockfp_builder.PACKER.PAYLOAD_FORMAT_PUBLIC_PR64_MASK_FIRST_LEN_TABLE,
    )
    parser.add_argument("--brotli-quality", type=int, default=11)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_imp_c067_bridge_candidates(
        source_archive=args.source_archive,
        output_dir=args.output_dir,
        cycle_counts=args.cycle_counts,
        sparsity_increment=args.sparsity_increment,
        qzs3_block_sizes=args.qzs3_block_sizes,
        payload_member_name=args.payload_member_name,
        payload_format=args.payload_format,
        brotli_quality=args.brotli_quality,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
