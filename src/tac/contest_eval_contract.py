# SPDX-License-Identifier: MIT
"""Ground-truth contract for the pinned upstream contest evaluator.

This module is intentionally scorer-neutral: it reads source custody and
records the implementation semantics that our allocators must obey, but it
does not run the evaluator or turn local/proxy output into score authority.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from tac.archive_byte_profile import CONTEST_ORIGINAL_BYTES, contest_rate_term

UPSTREAM_EVAL_CONTRACT_SCHEMA = "upstream_contest_eval_contract.v1"
UPSTREAM_SCORE_ALLOCATION_CONTRACT_SCHEMA = "upstream_score_allocation_contract.v1"
UPSTREAM_SALIENCY_VERIFICATION_CONTRACT_SCHEMA = (
    "upstream_saliency_verification_contract.v1"
)

SEQ_LEN = 2
CAMERA_SIZE_WH = (1164, 874)
SCORER_INPUT_SIZE_WH = (512, 384)
RGB_CHANNELS = 3
PUBLIC_TEST_FRAME_COUNT = 1200
PUBLIC_TEST_PAIR_COUNT = PUBLIC_TEST_FRAME_COUNT // SEQ_LEN
PUBLIC_TEST_RAW_OUTPUT_BYTES = (
    PUBLIC_TEST_FRAME_COUNT * CAMERA_SIZE_WH[0] * CAMERA_SIZE_WH[1] * RGB_CHANNELS
)
RATE_TERM_COEFFICIENT = 25.0
SEGNET_TERM_COEFFICIENT = 100.0
POSENET_TERM_DERIVATIVE_FORMULA = "5/sqrt(10*d_pose)"

_SOURCE_FILES = (
    "evaluate.py",
    "evaluate.sh",
    "modules.py",
    "frame_utils.py",
    ".github/workflows/eval.yml",
)
_MODEL_FILES = (
    "models/posenet.safetensors",
    "models/segnet.safetensors",
)
_EXPECTED_MODEL_SHA256: dict[str, str] = {
    "models/posenet.safetensors": "0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576",
    "models/segnet.safetensors": "68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6",
}

_REQUIRED_SNIPPETS: dict[str, tuple[tuple[str, str], ...]] = {
    "evaluate.py": (
        ("archive_zip_bytes_authority", "(args.submission_dir / 'archive.zip').stat().st_size"),
        ("rate_denominator_source_tree", "sum(file.stat().st_size for file in args.uncompressed_dir.rglob"),
        ("score_formula", "100 * segnet_dist +  math.sqrt(posenet_dist * 10)  + 25 * rate"),
        ("candidate_raw_dataset", "TensorVideoDataset(test_video_names"),
        ("posenet_full_video_pair_sum", "posenet_dists += posenet_dist.sum()"),
        ("segnet_full_video_pair_sum", "segnet_dists += segnet_dist.sum()"),
        ("batch_size_weighted_reduction", "batch_sizes += batch_gt.shape[0]"),
        ("posenet_full_video_mean", "posenet_dist = (posenet_dists / batch_sizes).item()"),
        ("segnet_full_video_mean", "segnet_dist = (segnet_dists / batch_sizes).item()"),
    ),
    "evaluate.sh": (
        ("archive_unzip_before_inflate", 'unzip -o "$ARCHIVE_ZIP" -d "$ARCHIVE_DIR"'),
        ("inflate_contract", 'bash "${SUBMISSION_DIR}/inflate.sh" "$ARCHIVE_DIR" "$INFLATED_DIR" "$VIDEO_NAMES_FILE"'),
        ("raw_existence_gate", 'RAW_PATH="${INFLATED_DIR}/${BASE}.raw"'),
    ),
    "modules.py": (
        ("segnet_last_frame_only", "x = x[:, -1, ...] # Use only last frame"),
        ("segnet_argmax_flip", "out1.argmax(dim=1) != out2.argmax(dim=1)"),
        ("posenet_first_six_dims", "out1[h.name][..., : h.out // 2] - out2[h.name][..., : h.out // 2]"),
        ("posenet_yuv6_pair_input", "return einops.rearrange(rgb_to_yuv6(x), '(b t) c h w -> b (t c) h w'"),
    ),
    "frame_utils.py": (
        ("sequence_length_two", "seq_len = 2"),
        ("camera_size", "camera_size = (1164, 874)"),
        ("scorer_input_size", "segnet_model_input_size = (512, 384)"),
        ("inplace_yuv_clamp", "clamp_(0.0, 255.0)"),
        ("tensor_raw_shape_loader", "shape=(N, H, W, C)"),
    ),
    ".github/workflows/eval.yml": (
        ("contest_eval_timeout_minutes", "timeout-minutes: 30"),
        ("cuda_runner_axis", "linux-nvidia-t4"),
        ("cpu_runner_axis", "ubuntu-latest"),
        ("uv_group_axis_select", "UV_GROUP:"),
    ),
}


def build_score_allocation_contract() -> dict[str, Any]:
    """Return the evaluator semantics that score-aware allocators must use."""

    return {
        "schema": UPSTREAM_SCORE_ALLOCATION_CONTRACT_SCHEMA,
        "score_formula": (
            "100*d_seg(full_video)+sqrt(10*d_pose(full_video))"
            "+25*archive_zip_bytes/source_video_bytes"
        ),
        "rate": {
            "archive_authority": "submission_dir/archive.zip.stat().st_size",
            "denominator_source": (
                "sum(file.stat().st_size for file in upstream/videos if file.is_file())"
            ),
            "canonical_denominator_bytes": CONTEST_ORIGINAL_BYTES,
            "rate_price_per_archive_byte": contest_rate_term(1),
            "raw_output_shape_bytes_are_not_rate_denominator": PUBLIC_TEST_RAW_OUTPUT_BYTES,
        },
        "receiver_equivalence_floor": {
            "planning_semantics": (
                "at fixed archive bytes, exact SegNet/PoseNet receiver equivalence "
                "has score equal to the rate term only"
            ),
            "zero_distortion_score_formula": "25*archive_zip_bytes/source_video_bytes",
            "helper": "tac.score_geometry.receiver_equivalence_floor_audit",
            "receiver_equivalence_witness_required_for_promotion": True,
            "score_claim": False,
            "promotion_eligible": False,
        },
        "distortion_reduction": {
            "authority": "upstream/evaluate.py full-video pair-sum reduction",
            "pair_weighting": "sum per-pair distortions over all evaluated pairs, then divide by total pair count",
            "update_before_full_reduction_allowed": False,
            "gradient_acquisition_rule": (
                "chunking is allowed only as exact accumulation; optimizer or "
                "allocator updates must wait until every intended pair shard "
                "has reduced into the full-video scalar"
            ),
        },
        "pair_geometry": {
            "seq_len": SEQ_LEN,
            "public_test_frame_count": PUBLIC_TEST_FRAME_COUNT,
            "public_test_pair_count": PUBLIC_TEST_PAIR_COUNT,
            "camera_size_wh": list(CAMERA_SIZE_WH),
            "candidate_raw_shape": [
                PUBLIC_TEST_FRAME_COUNT,
                CAMERA_SIZE_WH[1],
                CAMERA_SIZE_WH[0],
                RGB_CHANNELS,
            ],
            "candidate_raw_bytes": PUBLIC_TEST_RAW_OUTPUT_BYTES,
        },
        "segnet": {
            "contest_term": "100*d_seg",
            "coefficient": SEGNET_TERM_COEFFICIENT,
            "frame_scope": "last_frame_only",
            "scored_frame_index_within_pair": 1,
            "unscored_frame_index_within_pair": 0,
            "input_domain": "RGB frame -> bilinear resize to 384x512 -> five-class logits",
            "input_size_wh": list(SCORER_INPUT_SIZE_WH),
            "distortion": "mean(argmax(gt_logits)!=argmax(candidate_logits))",
            "allocation_guard": (
                "frame_0 atoms get zero direct SegNet saliency; frame_1 atoms "
                "carry SegNet plus PoseNet incidence"
            ),
        },
        "posenet": {
            "contest_term": "sqrt(10*d_pose)",
            "derivative_wrt_d_pose": POSENET_TERM_DERIVATIVE_FORMULA,
            "frame_scope": "both_frames_in_pair",
            "scored_frame_indices_within_pair": [0, 1],
            "input_domain": "two RGB frames -> bilinear resize -> YUV6 -> 12-channel tensor",
            "input_size_wh": list(SCORER_INPUT_SIZE_WH),
            "yuv6_shape_per_frame_chw": [6, SCORER_INPUT_SIZE_WH[1] // 2, SCORER_INPUT_SIZE_WH[0] // 2],
            "distortion": "MSE on first six dimensions of the 12-output pose head",
            "true_p19_requires": (
                "six-axis PoseNet VJP/JVP or finite-difference null probe under "
                "contest inverse-variance/Mahalanobis weighting"
            ),
        },
        "autograd_guard": {
            "upstream_rgb_to_yuv6_uses_inplace_clamp": True,
            "pose_gradient_requires_differentiable_replacement": (
                "use tac.differentiable_eval_roundtrip or an equivalent patched "
                "rgb_to_yuv6 path for VJP; upstream frame_utils.rgb_to_yuv6 is "
                "the eval authority but not the gradient path"
            ),
        },
        "authority": {
            "mlx_or_proxy_rows": "proposal_only",
            "score_authority_requires": "contest CPU/CUDA upstream evaluate.py axis",
            "receiver_contract": "inflate.sh archive_dir inflated_dir video_names_file",
        },
        "saliency_verification": build_saliency_verification_contract(),
    }


def build_saliency_verification_contract() -> dict[str, Any]:
    """Return the no-fake proof gates for P18/P19 saliency production."""

    return {
        "schema": UPSTREAM_SALIENCY_VERIFICATION_CONTRACT_SCHEMA,
        "contest_compliance": {
            "receiver_must_not": [
                "load PoseNet/SegNet",
                "inspect evaluator score state",
                "fetch sidecars",
                "adapt at eval time",
            ],
            "compress_side_may": [
                "use frozen upstream scorers for offline analysis",
                "derive saliency/allocation/latents before archive emission",
                "overfit the declared contest video in contest mode",
            ],
            "promotion_boundary": (
                "only emitted archive.zip plus deterministic inflate runtime cross "
                "the receiver boundary"
            ),
        },
        "required_numerical_proofs": [
            {
                "name": "yuv6_forward_parity",
                "authority_path": "upstream/frame_utils.py::rgb_to_yuv6",
                "mirror_path": "tac.differentiable_eval_roundtrip.differentiable_rgb_to_yuv6",
                "requirement": "max_abs_error==0 on deterministic random and real-video tensors",
                "purpose": "PoseNet VJP/JVP uses mirror; upstream remains eval authority",
            },
            {
                "name": "yuv6_gradient_nonzero",
                "authority_path": "upstream implementation is no_grad/inplace-clamp",
                "mirror_path": "differentiable mirror",
                "requirement": "finite nonzero input gradient in unclamped region",
                "purpose": "prevents fake P19 surfaces with severed pose gradients",
            },
            {
                "name": "segnet_last_frame_asymmetry",
                "authority_path": "upstream/modules.py::SegNet.preprocess_input",
                "requirement": (
                    "mutating pair frame_0 alone leaves SegNet input unchanged; "
                    "mutating pair frame_1 can change SegNet logits"
                ),
                "purpose": "prevents spending SegNet budget on frame_0 atoms",
            },
            {
                "name": "segnet_argmax_flip_exactness",
                "authority_path": "upstream/modules.py::SegNet.compute_distortion",
                "requirement": "local d_seg equals mean(argmax(gt)!=argmax(candidate))",
                "purpose": "ties P18 hinge/margin proxy back to hard contest metric",
            },
            {
                "name": "posenet_pair_six_axis_exactness",
                "authority_path": "upstream/modules.py::PoseNet.compute_distortion",
                "requirement": "local d_pose equals MSE over first six pose-head dims",
                "purpose": "ties P19 Mahalanobis/Jacobian weighting back to contest MSE",
            },
            {
                "name": "rate_byte_price_exactness",
                "authority_path": "upstream/evaluate.py",
                "requirement": (
                    "one archive byte costs exactly 25/source_video_bytes score; "
                    "source_video_bytes must not be confused with inflated raw bytes"
                ),
                "purpose": "sets the reverse-waterfill spend threshold",
            },
        ],
        "budget_spend_requires": [
            "source custody contract_valid=true",
            "saliency row names upstream source SHA-256s",
            "saliency row names mirror implementation SHA-256s",
            "real-video or deterministic tensor numerical proof artifact",
            "score_claim=false until exact CPU/CUDA eval",
        ],
    }


def build_upstream_eval_contract(
    *,
    repo_root: str | Path = ".",
    upstream_dir: str | Path = "upstream",
) -> dict[str, Any]:
    """Return source-custody evidence for the actual upstream evaluator."""

    root = Path(repo_root)
    upstream = root / upstream_dir
    source_records = [
        _path_record(
            _resolve_upstream_path(root=root, upstream=upstream, rel_path=rel_path),
            label=rel_path,
        )
        for rel_path in _SOURCE_FILES
    ]
    model_records = [
        _path_record(upstream / rel_path, label=rel_path) for rel_path in _MODEL_FILES
    ]
    for record in model_records:
        expected_sha256 = _EXPECTED_MODEL_SHA256.get(str(record["relative_path"]))
        record["expected_sha256"] = expected_sha256
        record["sha256_matches_expected"] = bool(
            record["exists"]
            and expected_sha256 is not None
            and record["sha256"] == expected_sha256
        )
    snippet_checks = _snippet_checks(root=root, upstream=upstream)
    blockers = [
        f"missing_source:{record['relative_path']}"
        for record in source_records
        if not record["exists"]
    ]
    blockers.extend(
        f"missing_model:{record['relative_path']}"
        for record in model_records
        if not record["exists"]
    )
    blockers.extend(
        f"model_sha256_mismatch:{record['relative_path']}"
        for record in model_records
        if record["exists"] and not record["sha256_matches_expected"]
    )
    blockers.extend(
        f"required_snippet_missing:{item['relative_path']}:{item['name']}"
        for item in snippet_checks
        if not item["present"]
    )
    live_denominator = _live_rate_denominator(upstream / "videos")
    if live_denominator is not None and live_denominator != CONTEST_ORIGINAL_BYTES:
        blockers.append("live_rate_denominator_differs_from_canonical_constant")

    return {
        "schema": UPSTREAM_EVAL_CONTRACT_SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "source_custody": source_records,
        "model_custody": model_records,
        "implementation_snippet_checks": snippet_checks,
        "score_allocation_contract": build_score_allocation_contract(),
        "live_rate_denominator_bytes": live_denominator,
        "canonical_rate_denominator_bytes": CONTEST_ORIGINAL_BYTES,
        "rate_denominator_note": (
            "upstream/evaluate.py names this uncompressed_size, but computes it "
            "from upstream/videos source-file byte sizes; inflated raw bytes are "
            "a receiver shape requirement, not the rate denominator"
        ),
        "blockers": blockers,
        "contract_valid": not blockers,
    }


def _resolve_upstream_path(*, root: Path, upstream: Path, rel_path: str) -> Path:
    if rel_path.startswith(".github/"):
        upstream_workflow = upstream / rel_path
        return upstream_workflow if upstream_workflow.exists() else root / rel_path
    return upstream / rel_path


def _snippet_checks(*, root: Path, upstream: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for rel_path, snippets in _REQUIRED_SNIPPETS.items():
        path = _resolve_upstream_path(root=root, upstream=upstream, rel_path=rel_path)
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        for name, snippet in snippets:
            checks.append(
                {
                    "relative_path": rel_path,
                    "name": name,
                    "present": snippet in text,
                }
            )
    return checks


def _path_record(path: Path, *, label: str) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "relative_path": label,
        "resolved_path": path.as_posix(),
        "exists": exists,
        "bytes": path.stat().st_size if exists else None,
        "sha256": _sha256_file(path) if exists else None,
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _live_rate_denominator(video_dir: Path) -> int | None:
    if not video_dir.is_dir():
        return None
    files = [path for path in video_dir.rglob("*") if path.is_file()]
    if not files:
        return None
    return sum(path.stat().st_size for path in files)


__all__ = [
    "CAMERA_SIZE_WH",
    "CONTEST_ORIGINAL_BYTES",
    "PUBLIC_TEST_FRAME_COUNT",
    "PUBLIC_TEST_PAIR_COUNT",
    "PUBLIC_TEST_RAW_OUTPUT_BYTES",
    "SCORER_INPUT_SIZE_WH",
    "SEQ_LEN",
    "UPSTREAM_EVAL_CONTRACT_SCHEMA",
    "UPSTREAM_SALIENCY_VERIFICATION_CONTRACT_SCHEMA",
    "UPSTREAM_SCORE_ALLOCATION_CONTRACT_SCHEMA",
    "build_saliency_verification_contract",
    "build_score_allocation_contract",
    "build_upstream_eval_contract",
]
