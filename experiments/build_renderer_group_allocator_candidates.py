#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build local renderer tensor/group allocation candidates for PR75 archives.

This is a local-only archive builder and policy profiler. It preserves every
non-renderer logical payload member from a source archive, rewrites only the
JointFrameGenerator renderer bytes with charged MQZ1 per-tensor block-size
metadata, and optionally runs the local renderer-transplant pose-safety gate.

The output is never score evidence. A candidate becomes exact-eval-ready only
when it is byte-sufficient, passes the local pose-safety preflight, and still
requires a dispatch claim before any remote CUDA auth eval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments import build_blockfp_c067_archive as blockfp  # noqa: E402
from experiments import build_mixed_qzs_block_candidate as mixed_qzs  # noqa: E402
from experiments import build_renderer_shrink_candidate as pr75_shrink  # noqa: E402
from experiments import preflight_renderer_transplant_pose_safety as pose_safety  # noqa: E402
from tac.quantizr_qzs3_codec import (  # noqa: E402
    _is_fp4_weight_name,
    _quantize_fp4_blocks,
    decode_mixed_qzs_block_state_dict,
    decode_qzs3_state_dict,
)


SCHEMA = "renderer_group_allocator_candidates_v1"
ORIGINAL_VIDEO_BYTES = 37_545_489
FRONTIER_SCORE = 0.3154707273953505  # [external: PR-65 contest-CUDA T4 frontier]
TARGET_SCORE = 0.314  # [heuristic: aspirational floor below PR-65 frontier 0.3155]
TARGET_ARCHIVE_BYTES_UNCHANGED_COMPONENTS = 274_133
DEFAULT_SOURCE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip"
)
DEFAULT_SOURCE_EVIDENCE = DEFAULT_SOURCE_ARCHIVE.with_name("contest_auth_eval.json")
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "experiments/results/renderer_group_allocator_worker_20260503"
)

# Exact A-negative renderer-shrink source archives must not become parents for
# byte-only child work. This list is intentionally narrow and records archives
# whose decoded renderer stream is already known to collapse PoseNet.
KNOWN_A_NEGATIVE_PARENT_ARCHIVE_SHAS = {
    "bb8d1835303478183465e15b28fd9af8776b4ea1c07cdd6cfe0895032a267b64": (
        "renderer_zero_fp4_frame1_head_010 exact T4 PoseNet collapse"
    ),
    "002b1d0681a895aac6dbf2eb1194c9d765debdee5e3b3101e034173e21295bac": (
        "lossless child of renderer_zero_fp4_frame1_head_010 A-negative parent"
    ),
}

LOW_RISK_PREFIXES = (
    "frame2_head.pre",
    "frame2_head.block2",
    "frame2_head.block1",
    "frame2_head",
)
HIGH_RISK_PREFIXES = (
    "shared_trunk",
    "frame1_head",
    "pose_mlp",
)
DEFAULT_BLOCK_SIZES = (48, 64, 96, 128)


@dataclass(frozen=True)
class AllocationPolicy:
    """One deterministic renderer block allocation policy."""

    name: str
    spec: str
    default_block_size: int
    overrides: tuple[tuple[str, int], ...]
    source: str
    rationale: str

    def block_size_for(self, tensor_name: str) -> int:
        for prefix, block_size in self.overrides:
            if tensor_name == prefix or tensor_name.startswith(prefix + "."):
                return block_size
        return self.default_block_size

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "spec": self.spec,
            "default_block_size": self.default_block_size,
            "overrides": [
                {"prefix": prefix, "block_size": block_size}
                for prefix, block_size in self.overrides
            ],
            "source": self.source,
            "rationale": self.rationale,
            "risk_class": policy_risk_class(self),
        }


class _MixedPolicyAdapter:
    """Adapter for ``build_mixed_qzs_block_candidate`` encoding."""

    def __init__(self, policy: AllocationPolicy) -> None:
        self._policy = policy
        self.name = policy.name
        self.spec = policy.spec
        self.default_block_size = policy.default_block_size
        self.prefix_overrides = policy.overrides
        self.exact_evaluable_archive = True
        self.component_awareness = {
            "schema": "renderer_group_allocator_policy_v1",
            "risk_class": policy_risk_class(policy),
            "rationale": policy.rationale,
        }

    def block_size_for(self, tensor_name: str) -> int:
        return self._policy.block_size_for(tensor_name)

    def as_json(self) -> dict[str, Any]:
        payload = self._policy.as_json()
        payload["exact_evaluable_archive"] = True
        return payload


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


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return slug or "policy"


def _normalise_block_size(value: int) -> int:
    block_size = int(value)
    if block_size <= 0 or block_size > 4096:
        raise ValueError(f"block size must be in [1, 4096], got {block_size}")
    return block_size


def _safe_prefix(prefix: str) -> str:
    if not prefix or "/" in prefix or "\\" in prefix or prefix.startswith("."):
        raise ValueError(f"unsafe or empty tensor prefix: {prefix!r}")
    return prefix


def _matches_prefix(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def policy_risk_class(policy: AllocationPolicy) -> str:
    prefixes = [prefix for prefix, _ in policy.overrides]
    if not prefixes:
        return "no_op_or_global"
    if any(
        _matches_prefix(prefix, high) or _matches_prefix(high, prefix)
        for prefix in prefixes
        for high in HIGH_RISK_PREFIXES
    ):
        return "high_risk_pose_or_shared"
    if all(
        any(_matches_prefix(prefix, low) or _matches_prefix(low, prefix) for low in LOW_RISK_PREFIXES)
        for prefix in prefixes
    ):
        return "low_risk_frame2_only"
    return "unknown_risk"


def parse_policy_spec(raw: str, *, source_block_size: int = 32) -> AllocationPolicy:
    """Parse a deterministic allocation policy spec.

    Supported grammar:
    - ``mixed:<default>:prefix=block,prefix=block``
    - ``group:<prefix>:<block>``
    """

    spec = raw.strip()
    if not spec:
        raise ValueError("empty policy spec")
    if spec.startswith("group:"):
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"group policy must be group:<prefix>:<block>; got {raw!r}")
        prefix = _safe_prefix(parts[1])
        block_size = _normalise_block_size(int(parts[2]))
        return AllocationPolicy(
            name=_slug(f"group_{prefix}_b{block_size}"),
            spec=spec,
            default_block_size=int(source_block_size),
            overrides=((prefix, block_size),),
            source="cli",
            rationale="CLI-specified tensor-group block allocation",
        )
    if not spec.startswith("mixed:"):
        raise ValueError(
            f"unsupported policy {raw!r}; expected group:<prefix>:<block> or "
            "mixed:<default>:prefix=block,..."
        )
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"mixed policy must be mixed:<default>:prefix=block,...; got {raw!r}")
    default_block_size = _normalise_block_size(int(parts[1]))
    overrides: list[tuple[str, int]] = []
    for item in parts[2].split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"mixed override must be prefix=block; got {item!r}")
        prefix, value = item.split("=", 1)
        overrides.append((_safe_prefix(prefix.strip()), _normalise_block_size(int(value))))
    if not overrides:
        raise ValueError(f"mixed policy requires at least one override: {raw!r}")
    return AllocationPolicy(
        name=_slug(spec.replace(":", "_").replace(",", "_")),
        spec=spec,
        default_block_size=default_block_size,
        overrides=tuple(overrides),
        source="cli",
        rationale="CLI-specified tensor-group block allocation",
    )


def _source_context(source_archive: Path) -> dict[str, Any]:
    source_archive = source_archive.resolve()
    source_bytes = source_archive.read_bytes()
    members, packaging = blockfp.extract_runtime_members(source_archive)
    if pr75_shrink.RENDERER_MEMBER not in members:
        raise ValueError("source archive missing renderer.bin")
    renderer = members[pr75_shrink.RENDERER_MEMBER]
    if not renderer.startswith(b"QZS3"):
        raise ValueError(f"renderer group allocator requires QZS3 source, got {renderer[:4]!r}")
    source_payload = pr75_shrink._read_single_payload(source_archive)
    pr75_slices = (
        pr75_shrink._parse_pr75_slices(source_payload)
        if source_payload is not None
        else None
    )
    if pr75_slices is None:
        raise ValueError("renderer group allocator requires a PR75 single-member p archive")
    non_renderer = {
        name: payload for name, payload in members.items() if name != pr75_shrink.RENDERER_MEMBER
    }
    return {
        "source_archive": source_archive,
        "source_bytes": source_bytes,
        "source_sha256": _sha256_bytes(source_bytes),
        "members": members,
        "packaging": packaging,
        "renderer": renderer,
        "renderer_sha256": _sha256_bytes(renderer),
        "source_block_size": int.from_bytes(renderer[4:6], "little"),
        "state": decode_qzs3_state_dict(renderer, device="cpu"),
        "pr75_slices": pr75_slices,
        "non_renderer_members": non_renderer,
    }


def _file_meta(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "exists": False}
    meta: dict[str, Any] = {
        "path": str(resolved),
        "exists": True,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }
    if resolved.suffix == ".json":
        try:
            payload = json.loads(resolved.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            meta["json_parseable"] = False
        else:
            meta["json_parseable"] = True
            meta["summary"] = {
                key: payload[key]
                for key in (
                    "canonical_score",
                    "score_recomputed_from_components",
                    "avg_posenet_dist",
                    "avg_segnet_dist",
                    "archive_size_bytes",
                    "n_samples",
                )
                if key in payload
            }
            provenance = payload.get("provenance")
            if isinstance(provenance, dict):
                for key in ("archive_sha256", "archive_size_bytes", "device", "gpu_model"):
                    if key in provenance:
                        meta["summary"][f"provenance_{key}"] = provenance[key]
    return meta


def _source_evidence_guard(
    *,
    context: dict[str, Any],
    source_evidence_path: Path | None,
) -> dict[str, Any]:
    source_sha = context["source_sha256"]
    if source_sha in KNOWN_A_NEGATIVE_PARENT_ARCHIVE_SHAS:
        return {
            "ok": False,
            "failure_class": "known_a_negative_parent_archive",
            "reason": KNOWN_A_NEGATIVE_PARENT_ARCHIVE_SHAS[source_sha],
            "source_archive_sha256": source_sha,
        }
    if source_evidence_path is None:
        return {
            "ok": True,
            "warning": "source evidence path not supplied; outputs cannot be exact-eval-ready",
            "source_archive_sha256": source_sha,
        }
    meta = _file_meta(source_evidence_path)
    if not meta or not meta.get("exists") or not meta.get("json_parseable"):
        return {
            "ok": False,
            "failure_class": "source_evidence_missing_or_unparseable",
            "source_evidence": meta,
        }
    payload = json.loads(Path(source_evidence_path).read_text())
    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
    failures: list[str] = []
    if provenance.get("archive_sha256") != source_sha:
        failures.append("source_evidence_archive_sha_mismatch")
    if int(payload.get("archive_size_bytes", -1)) != len(context["source_bytes"]):
        failures.append("source_evidence_archive_size_mismatch")
    if int(payload.get("n_samples", -1)) != 600:
        failures.append("source_evidence_not_full_sample_count")
    if provenance.get("device") != "cuda":
        failures.append("source_evidence_not_cuda")
    if float(payload.get("avg_posenet_dist", 99.0)) > 0.002:
        failures.append("source_pose_component_outside_frontier_trust_region")
    if float(payload.get("avg_segnet_dist", 99.0)) > 0.002:
        failures.append("source_seg_component_outside_frontier_trust_region")
    return {
        "ok": not failures,
        "failure_class": None if not failures else "source_evidence_not_frontier_safe",
        "failures": failures,
        "source_evidence": meta,
    }


def _fp4_tensor_group(name: str) -> str:
    parts = name.split(".")
    if name.startswith("frame2_head.") and len(parts) >= 2:
        return ".".join(parts[:2])
    if name.startswith("frame1_head.") and len(parts) >= 2:
        return ".".join(parts[:2])
    if name.startswith("shared_trunk.") and len(parts) >= 2:
        return ".".join(parts[:2])
    return parts[0]


def build_tensor_profiles(
    state: dict[str, torch.Tensor],
    *,
    source_block_size: int,
    candidate_block_sizes: tuple[int, ...] = DEFAULT_BLOCK_SIZES,
) -> list[dict[str, Any]]:
    """Profile raw FP4 byte deltas for each tensor and candidate block."""

    profiles: list[dict[str, Any]] = []
    for name, tensor in state.items():
        if not _is_fp4_weight_name(name):
            continue
        src_packed, src_scales = _quantize_fp4_blocks(tensor, source_block_size)
        candidates = []
        for block_size in candidate_block_sizes:
            packed, scales = _quantize_fp4_blocks(tensor, block_size)
            candidates.append(
                {
                    "block_size": block_size,
                    "packed_bytes": len(packed),
                    "scale_bytes": len(scales),
                    "raw_bytes": len(packed) + len(scales),
                    "raw_delta_vs_source": len(packed) + len(scales) - len(src_packed) - len(src_scales),
                }
            )
        profiles.append(
            {
                "name": name,
                "group": _fp4_tensor_group(name),
                "numel": int(tensor.numel()),
                "risk_class": (
                    "low_risk_frame2_only"
                    if name.startswith("frame2_head.")
                    else "high_risk_pose_or_shared"
                    if name.startswith(("shared_trunk.", "frame1_head."))
                    else "unknown_risk"
                ),
                "source": {
                    "block_size": source_block_size,
                    "packed_bytes": len(src_packed),
                    "scale_bytes": len(src_scales),
                    "raw_bytes": len(src_packed) + len(src_scales),
                },
                "candidates": candidates,
            }
        )
    return profiles


def _policy_raw_delta(policy: AllocationPolicy, profiles: list[dict[str, Any]]) -> int:
    delta = 0
    for profile in profiles:
        block_size = policy.block_size_for(str(profile["name"]))
        if block_size == int(profile["source"]["block_size"]):
            continue
        candidate = next(
            (
                item
                for item in profile["candidates"]
                if int(item["block_size"]) == block_size
            ),
            None,
        )
        if candidate is None:
            continue
        delta += int(candidate["raw_delta_vs_source"])
    return delta


def _normalised_overrides(overrides: tuple[tuple[str, int], ...]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((str(prefix), int(block)) for prefix, block in overrides))


def generate_default_policies(
    profiles: list[dict[str, Any]],
    *,
    source_block_size: int,
    max_generated: int = 24,
) -> tuple[AllocationPolicy, ...]:
    """Generate deterministic low-risk group and top-tensor policies."""

    policies: list[AllocationPolicy] = []
    for prefix in ("frame2_head.pre", "frame2_head.block2", "frame2_head.block1", "frame2_head"):
        for block_size in DEFAULT_BLOCK_SIZES:
            policies.append(
                AllocationPolicy(
                    name=_slug(f"group_{prefix}_b{block_size}"),
                    spec=f"group:{prefix}:{block_size}",
                    default_block_size=source_block_size,
                    overrides=((prefix, block_size),),
                    source="generated_low_risk_group",
                    rationale=(
                        "frame2-only renderer tensor group; avoids shared trunk, "
                        "pose MLP, and frame1 pose-critical groups"
                    ),
                )
            )

    low_risk_tensor_gains: list[tuple[int, str, int]] = []
    for profile in profiles:
        if profile.get("risk_class") != "low_risk_frame2_only":
            continue
        best = min(profile["candidates"], key=lambda item: int(item["raw_delta_vs_source"]))
        if int(best["raw_delta_vs_source"]) < 0:
            low_risk_tensor_gains.append(
                (int(best["raw_delta_vs_source"]), str(profile["name"]), int(best["block_size"]))
            )
    low_risk_tensor_gains.sort(key=lambda item: (item[0], item[1], item[2]))
    for count in (1, 2, 4, 8, 12):
        selected = low_risk_tensor_gains[:count]
        if not selected:
            continue
        overrides = tuple((name, block_size) for _, name, block_size in selected)
        policies.append(
            AllocationPolicy(
                name=_slug(f"top{count}_lowrisk_tensor_raw_gain"),
                spec="generated:top_lowrisk_tensor_raw_gain:" + str(count),
                default_block_size=source_block_size,
                overrides=overrides,
                source="generated_top_tensor_raw_gain",
                rationale="best low-risk frame2 tensor raw-byte deltas under MQZ1",
            )
        )

    seen: set[tuple[tuple[str, int], ...]] = set()
    filtered: list[AllocationPolicy] = []
    for policy in policies:
        key = _normalised_overrides(policy.overrides)
        if key in seen:
            continue
        seen.add(key)
        if all(block == source_block_size for _, block in policy.overrides):
            continue
        filtered.append(policy)
    filtered.sort(
        key=lambda policy: (
            _policy_raw_delta(policy, profiles),
            policy_risk_class(policy) != "low_risk_frame2_only",
            policy.name,
        )
    )
    return tuple(filtered[:max_generated])


def _policy_delta_table(
    policy: AllocationPolicy,
    profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        block_size = policy.block_size_for(str(profile["name"]))
        if block_size == int(profile["source"]["block_size"]):
            continue
        candidate = next(
            (item for item in profile["candidates"] if int(item["block_size"]) == block_size),
            None,
        )
        if candidate is None:
            continue
        rows.append(
            {
                "name": profile["name"],
                "group": profile["group"],
                "risk_class": profile["risk_class"],
                "source_block_size": profile["source"]["block_size"],
                "output_block_size": block_size,
                "source_raw_bytes": profile["source"]["raw_bytes"],
                "output_raw_bytes": candidate["raw_bytes"],
                "raw_delta_vs_source": candidate["raw_delta_vs_source"],
            }
        )
    return rows


def _validate_policy(
    policy: AllocationPolicy,
    *,
    source_block_size: int,
    allow_high_risk: bool,
) -> dict[str, Any]:
    failures: list[str] = []
    if not policy.overrides:
        failures.append("empty_policy")
    if all(block == source_block_size for _, block in policy.overrides):
        failures.append("no_effective_block_change")
    risk_class = policy_risk_class(policy)
    if risk_class != "low_risk_frame2_only" and not allow_high_risk:
        failures.append("policy_not_low_risk_frame2_only")
    return {
        "ok": not failures,
        "risk_class": risk_class,
        "failures": failures,
    }


def _build_candidate(
    *,
    context: dict[str, Any],
    policy: AllocationPolicy,
    policy_delta_table: list[dict[str, Any]],
    output_dir: Path,
    source_evidence_path: Path | None,
    brotli_quality: int,
) -> dict[str, Any]:
    adapter = _MixedPolicyAdapter(policy)
    renderer, block_meta = mixed_qzs.encode_mixed_qzs_block_payload(
        context["state"],
        adapter,
    )
    decode_mixed_qzs_block_state_dict(renderer, device="cpu")
    if renderer == context["renderer"]:
        raise ValueError("no-op transform: output renderer bytes equal source renderer")

    payload, payload_meta = pr75_shrink._build_pr75_payload(
        context["pr75_slices"],
        renderer_bytes=renderer,
        brotli_quality=brotli_quality,
    )
    candidate_dir = output_dir / policy.name
    candidate_dir.mkdir(parents=True, exist_ok=True)
    archive_path = candidate_dir / "archive.zip"
    pr75_shrink._write_single_member_archive(archive_path, payload)
    runtime_unpack = pr75_shrink._verify_archive(
        archive_path,
        expected_renderer=renderer,
        expected_non_renderer_members=context["non_renderer_members"],
    )
    archive_bytes = archive_path.stat().st_size
    delta = archive_bytes - len(context["source_bytes"])
    manifest = {
        "schema": SCHEMA,
        "tool": "experiments/build_renderer_group_allocator_candidates.py",
        "candidate_id": policy.name,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "empirical_archive_candidate_until_pose_safety_and_exact_cuda",
        "remote_gpu_dispatch_performed": False,
        "source_archive": {
            "path": str(context["source_archive"]),
            "bytes": len(context["source_bytes"]),
            "sha256": context["source_sha256"],
            **context["packaging"],
        },
        "source_evidence": _file_meta(source_evidence_path),
        "output_archive": {
            "path": str(archive_path),
            "bytes": archive_bytes,
            "sha256": _sha256_file(archive_path),
            "delta_bytes_vs_source_archive": delta,
            "formula_only_rate_delta_vs_source_archive": (
                25.0 * delta / ORIGINAL_VIDEO_BYTES
            ),
            "frontier_score_if_only_bytes_change": (
                FRONTIER_SCORE + 25.0 * delta / ORIGINAL_VIDEO_BYTES
            ),
            "target_score": TARGET_SCORE,
            "byte_target_for_unchanged_components": TARGET_ARCHIVE_BYTES_UNCHANGED_COMPONENTS,
            "byte_sufficient_for_sub314_if_components_unchanged": (
                archive_bytes <= TARGET_ARCHIVE_BYTES_UNCHANGED_COMPONENTS
            ),
        },
        "policy": policy.as_json(),
        "policy_delta_table": policy_delta_table,
        "renderer_transform": {
            "source_format": "QZS3",
            "output_format": "MQZ1",
            "source_bytes": len(context["renderer"]),
            "source_sha256": context["renderer_sha256"],
            "output_bytes": len(renderer),
            "output_sha256": _sha256_bytes(renderer),
            "renderer_delta_bytes": len(renderer) - len(context["renderer"]),
            "mixed_qzs_meta": block_meta,
        },
        "payload": {
            "member_name": pr75_shrink.PR75_PAYLOAD_MEMBER,
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
            **payload_meta,
        },
        "non_renderer_preservation": {
            "all_non_renderer_members_preserved": True,
            "members": pr75_shrink._member_meta(context["non_renderer_members"]),
        },
        "runtime_contract": {
            "byte_closed": True,
            "single_payload_member": True,
            "runtime_unpack_verified": True,
            "runtime_unpack_summary": runtime_unpack,
            "renderer_only_transplant": True,
            "pose_safety_preflight_required_before_dispatch": True,
            "canonical_score_source_required": (
                "archive.zip -> inflate.sh -> upstream/evaluate.py via "
                "experiments/contest_auth_eval.py --device cuda"
            ),
        },
    }
    (candidate_dir / "build_manifest.json").write_bytes(_json_bytes(manifest))
    return manifest


def _preflight_candidate(
    *,
    source_archive: Path,
    manifest: dict[str, Any],
    max_pairs: int,
    max_mean_abs_delta: float,
    max_rms_delta: float,
    max_max_abs_delta: float,
) -> dict[str, Any]:
    archive_path = Path(manifest["output_archive"]["path"])
    output_json = archive_path.with_name("pose_safety_preflight.json")
    try:
        report = pose_safety.build_pose_safety_preflight(
            source_archive=source_archive,
            candidate_archive=archive_path,
            output_json=output_json,
            max_pairs=max_pairs,
            max_mean_abs_delta=max_mean_abs_delta,
            max_rms_delta=max_rms_delta,
            max_max_abs_delta=max_max_abs_delta,
        )
    except Exception as exc:
        report = {
            "schema": pose_safety.SCHEMA,
            "score_claim": False,
            "promotion_eligible": False,
            "remote_gpu_dispatch_performed": False,
            "safe_for_exact_eval_dispatch": False,
            "failure_class": "renderer_transplant_pose_safety_exception",
            "fail_closed_reasons": [type(exc).__name__],
            "exception": str(exc),
        }
        output_json.write_bytes(_json_bytes(report))
    return {
        "path": str(output_json),
        "safe_for_exact_eval_dispatch": bool(report.get("safe_for_exact_eval_dispatch")),
        "failure_class": report.get("failure_class"),
        "fail_closed_reasons": report.get("fail_closed_reasons", []),
        "output_parity": report.get("output_parity"),
    }


def _candidate_row(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": manifest["candidate_id"],
        "policy": manifest["policy"],
        "archive": manifest["output_archive"]["path"],
        "archive_bytes": manifest["output_archive"]["bytes"],
        "archive_sha256": manifest["output_archive"]["sha256"],
        "delta_bytes_vs_source_archive": manifest["output_archive"]["delta_bytes_vs_source_archive"],
        "frontier_score_if_only_bytes_change": manifest["output_archive"]["frontier_score_if_only_bytes_change"],
        "byte_sufficient_for_sub314_if_components_unchanged": manifest["output_archive"][
            "byte_sufficient_for_sub314_if_components_unchanged"
        ],
        "renderer_delta_bytes": manifest["renderer_transform"]["renderer_delta_bytes"],
        "manifest": str(Path(manifest["output_archive"]["path"]).with_name("build_manifest.json")),
        "score_claim": False,
        "promotion_eligible": False,
    }


def _recommend(rows: list[dict[str, Any]], source_guard: dict[str, Any]) -> dict[str, Any]:
    if not source_guard.get("ok"):
        return {
            "recommendation": "do_not_dispatch",
            "reason": "source archive failed A-negative/frontier-safety guard",
            "source_guard": source_guard,
            "claim_required_before_dispatch": True,
            "remote_gpu_dispatch_performed": False,
        }
    safe_target = [
        row
        for row in rows
        if row.get("pose_safety", {}).get("safe_for_exact_eval_dispatch")
        and row["byte_sufficient_for_sub314_if_components_unchanged"]
    ]
    if safe_target:
        best = min(safe_target, key=lambda row: (row["archive_bytes"], row["candidate_id"]))
        return {
            "recommendation": "claim_lane_then_remote_exact_eval",
            "reason": "byte-sufficient candidate passed local pose-safety; exact CUDA still required",
            "candidate": best,
            "claim_required_before_dispatch": True,
            "claim_tool": "tools/claim_lane_dispatch.py claim ...",
            "remote_gpu_dispatch_performed": False,
        }
    safe = [
        row
        for row in rows
        if row.get("pose_safety", {}).get("safe_for_exact_eval_dispatch")
    ]
    if safe:
        best = min(safe, key=lambda row: (row["archive_bytes"], row["candidate_id"]))
        return {
            "recommendation": "do_not_dispatch_yet_safe_but_too_small",
            "reason": "local-safe renderer allocation did not save the 2209-byte target",
            "candidate": best,
            "claim_required_before_dispatch": True,
            "remote_gpu_dispatch_performed": False,
        }
    return {
        "recommendation": "do_not_dispatch",
        "reason": "no byte-saving renderer allocation passed local pose-safety",
        "claim_required_before_dispatch": True,
        "remote_gpu_dispatch_performed": False,
    }


def run_allocator(
    *,
    source_archive: Path = DEFAULT_SOURCE_ARCHIVE,
    source_evidence_path: Path | None = DEFAULT_SOURCE_EVIDENCE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    policy_specs: tuple[str, ...] = (),
    max_generated_policies: int = 24,
    max_build_candidates: int = 16,
    max_preflight_candidates: int = 8,
    preflight_max_pairs: int = pose_safety.DEFAULT_MAX_PAIRS,
    max_mean_abs_delta: float = pose_safety.DEFAULT_MAX_MEAN_ABS_DELTA,
    max_rms_delta: float = pose_safety.DEFAULT_MAX_RMS_DELTA,
    max_max_abs_delta: float = pose_safety.DEFAULT_MAX_MAX_ABS_DELTA,
    brotli_quality: int = 11,
    allow_high_risk: bool = False,
    skip_preflight: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Build deterministic local candidates and dispatch recommendations."""

    if not 0 <= brotli_quality <= 11:
        raise ValueError(f"brotli_quality must be in [0, 11], got {brotli_quality}")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"output directory is non-empty; pass --force: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    context = _source_context(source_archive)
    source_guard = _source_evidence_guard(
        context=context,
        source_evidence_path=source_evidence_path,
    )
    profiles = build_tensor_profiles(
        context["state"],
        source_block_size=context["source_block_size"],
        candidate_block_sizes=DEFAULT_BLOCK_SIZES,
    )
    (output_dir / "tensor_group_profile.json").write_bytes(
        _json_bytes(
            {
                "schema": "renderer_group_allocator_tensor_profile_v1",
                "source_archive": {
                    "path": str(context["source_archive"]),
                    "bytes": len(context["source_bytes"]),
                    "sha256": context["source_sha256"],
                },
                "source_renderer": {
                    "bytes": len(context["renderer"]),
                    "sha256": context["renderer_sha256"],
                    "block_size": context["source_block_size"],
                },
                "profiles": profiles,
            }
        )
    )

    if policy_specs:
        policies = tuple(
            parse_policy_spec(spec, source_block_size=context["source_block_size"])
            for spec in policy_specs
        )
    else:
        policies = generate_default_policies(
            profiles,
            source_block_size=context["source_block_size"],
            max_generated=max_generated_policies,
        )

    policy_rows = []
    build_queue: list[tuple[AllocationPolicy, int, list[dict[str, Any]]]] = []
    for policy in policies:
        validation = _validate_policy(
            policy,
            source_block_size=context["source_block_size"],
            allow_high_risk=allow_high_risk,
        )
        delta_table = _policy_delta_table(policy, profiles)
        raw_delta = sum(int(row["raw_delta_vs_source"]) for row in delta_table)
        policy_row = {
            "policy": policy.as_json(),
            "validation": validation,
            "estimated_renderer_raw_delta": raw_delta,
            "changed_tensor_count": len(delta_table),
            "delta_table": delta_table,
            "build_selected": False,
        }
        if source_guard.get("ok") and validation["ok"] and raw_delta < 0:
            build_queue.append((policy, raw_delta, delta_table))
        policy_rows.append(policy_row)
    build_queue.sort(key=lambda item: (item[1], item[0].name))
    selected_policy_names = {item[0].name for item in build_queue[:max_build_candidates]}
    for row in policy_rows:
        row["build_selected"] = row["policy"]["name"] in selected_policy_names

    manifests: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    if source_guard.get("ok"):
        for policy, _, delta_table in build_queue[:max_build_candidates]:
            candidate_dir = output_dir / policy.name
            candidate_dir.mkdir(parents=True, exist_ok=True)
            try:
                manifest = _build_candidate(
                    context=context,
                    policy=policy,
                    policy_delta_table=delta_table,
                    output_dir=output_dir,
                    source_evidence_path=source_evidence_path,
                    brotli_quality=brotli_quality,
                )
            except Exception as exc:
                rejection = {
                    "schema": SCHEMA,
                    "candidate_id": policy.name,
                    "policy": policy.as_json(),
                    "score_claim": False,
                    "promotion_eligible": False,
                    "remote_gpu_dispatch_performed": False,
                    "failure_class": "candidate_build_or_runtime_unpack_failed",
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                }
                (candidate_dir / "build_rejected.json").write_bytes(_json_bytes(rejection))
                rejected.append(rejection)
                continue
            manifests.append(manifest)

    rows = [_candidate_row(manifest) for manifest in manifests]
    rows.sort(key=lambda row: (row["archive_bytes"], row["candidate_id"]))
    preflight_targets = [] if skip_preflight else rows[:max_preflight_candidates]
    target_ids = {row["candidate_id"] for row in preflight_targets}
    manifest_by_id = {manifest["candidate_id"]: manifest for manifest in manifests}
    for row in rows:
        if row["candidate_id"] not in target_ids:
            row["pose_safety"] = {
                "preflight_ran": False,
                "safe_for_exact_eval_dispatch": False,
                "failure_class": "pose_safety_preflight_not_run",
            }
            continue
        row["pose_safety"] = {
            "preflight_ran": True,
            **_preflight_candidate(
                source_archive=context["source_archive"],
                manifest=manifest_by_id[row["candidate_id"]],
                max_pairs=preflight_max_pairs,
                max_mean_abs_delta=max_mean_abs_delta,
                max_rms_delta=max_rms_delta,
                max_max_abs_delta=max_max_abs_delta,
            ),
        }

    recommendation = _recommend(rows, source_guard)
    summary = {
        "schema": SCHEMA,
        "tool": "experiments/build_renderer_group_allocator_candidates.py",
        "score_claim": False,
        "promotion_eligible": False,
        "remote_gpu_dispatch_performed": False,
        "output_dir": str(output_dir),
        "source_archive": {
            "path": str(context["source_archive"]),
            "bytes": len(context["source_bytes"]),
            "sha256": context["source_sha256"],
        },
        "source_evidence_path": str(source_evidence_path) if source_evidence_path else None,
        "source_guard": source_guard,
        "frontier_score": FRONTIER_SCORE,
        "target_score": TARGET_SCORE,
        "target_archive_bytes_if_components_unchanged": TARGET_ARCHIVE_BYTES_UNCHANGED_COMPONENTS,
        "byte_delta_needed_if_components_unchanged": (
            TARGET_ARCHIVE_BYTES_UNCHANGED_COMPONENTS - len(context["source_bytes"])
        ),
        "policy_count": len(policy_rows),
        "policy_screen": policy_rows,
        "candidate_count": len(rows),
        "rejected_candidate_count": len(rejected),
        "rejected_candidates": rejected,
        "max_build_candidates": max_build_candidates,
        "max_preflight_candidates": max_preflight_candidates,
        "preflight_thresholds": {
            "max_pairs": preflight_max_pairs,
            "max_mean_abs_delta": max_mean_abs_delta,
            "max_rms_delta": max_rms_delta,
            "max_max_abs_delta": max_max_abs_delta,
        },
        "candidates": rows,
        "best_by_archive_bytes": rows[0] if rows else None,
        "dispatch_recommendation": recommendation,
    }
    (output_dir / "summary.json").write_bytes(_json_bytes(summary))
    (output_dir / "dispatch_recommendation.json").write_bytes(_json_bytes(recommendation))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--source-evidence-path", type=Path, default=DEFAULT_SOURCE_EVIDENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--policy",
        action="append",
        default=None,
        help=(
            "Repeatable policy. Supported: group:<prefix>:<block> or "
            "mixed:<default>:prefix=block,... . Defaults to generated low-risk "
            "frame2-only allocator policies."
        ),
    )
    parser.add_argument("--max-generated-policies", type=int, default=24)
    parser.add_argument("--max-build-candidates", type=int, default=16)
    parser.add_argument("--max-preflight-candidates", type=int, default=8)
    parser.add_argument(
        "--preflight-max-pairs",
        type=int,
        default=pose_safety.DEFAULT_MAX_PAIRS,
    )
    parser.add_argument(
        "--max-mean-abs-delta",
        type=float,
        default=pose_safety.DEFAULT_MAX_MEAN_ABS_DELTA,
    )
    parser.add_argument(
        "--max-rms-delta",
        type=float,
        default=pose_safety.DEFAULT_MAX_RMS_DELTA,
    )
    parser.add_argument(
        "--max-max-abs-delta",
        type=float,
        default=pose_safety.DEFAULT_MAX_MAX_ABS_DELTA,
    )
    parser.add_argument("--brotli-quality", type=int, default=11)
    parser.add_argument("--allow-high-risk", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_allocator(
        source_archive=args.source_archive,
        source_evidence_path=args.source_evidence_path,
        output_dir=args.output_dir,
        policy_specs=tuple(args.policy or ()),
        max_generated_policies=args.max_generated_policies,
        max_build_candidates=args.max_build_candidates,
        max_preflight_candidates=args.max_preflight_candidates,
        preflight_max_pairs=args.preflight_max_pairs,
        max_mean_abs_delta=args.max_mean_abs_delta,
        max_rms_delta=args.max_rms_delta,
        max_max_abs_delta=args.max_max_abs_delta,
        brotli_quality=args.brotli_quality,
        allow_high_risk=args.allow_high_risk,
        skip_preflight=args.skip_preflight,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
