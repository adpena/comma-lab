#!/usr/bin/env python3
"""Plan and build PR95-family latent residual atoms.

This is a local-only bridge from the PR95 HNeRV archive anatomy to concrete
charged atom candidates. The default mode emits a diagnostic opportunity
ledger from the archive's latent stream and optional per-pair component trace.
It does not claim score. Candidate building is intentionally stricter: callers
must provide an explicit atom plan, every atom must change a charged latent
value inside ``0.bin``, and the output archive must remain a single-member
deterministic ZIP that uses the existing PR95 runtime contract.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import zipfile
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import brotli

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.profile_pr95_hnerv_muon_packing import (
    LatentPayload,
    encode_top_blob,
    parse_latents_raw,
    parse_top_blob,
    read_single_member_zip,
    sha256_bytes,
    write_stored_zip,
)


CONTEST_ORIGINAL_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / CONTEST_ORIGINAL_BYTES
DEFAULT_PR95_REPACK_EXACT_JSON = Path(
    "experiments/results/lightning_batch/"
    "exact_eval_pr95_hnerv_muon_repacked_t4_fix2_20260504T0848Z/"
    "contest_auth_eval.adjudicated.json"
)
DEFAULT_PR95_RUNTIME_INFLATE = Path(
    "experiments/results/public_pr95_intake_20260504_codex/pr95_src/"
    "submissions/hnerv_muon/inflate.sh"
)
SIGNED_POLICY_FILENAME = "pr95_hnerv_residual_atom_plan.signed.json"


class PR95AtomPlanError(ValueError):
    """Raised when a PR95 atom plan is malformed or unsafe."""


@dataclasses.dataclass(frozen=True)
class PairLatentProfile:
    pair_index: int
    latent_l1_from_prev: int
    latent_l2_from_prev: float
    latent_linf_from_prev: int
    active_delta_dims: int
    largest_delta_dims: tuple[int, ...]
    latent_value_sum: int
    latent_value_l2: float
    proxy_rank_signal: float
    estimated_min_patch_bytes: int
    break_even_score_cost: float
    break_even_seg_dist_reduction: float
    break_even_pose_dist_reduction: float | None
    component_trace: dict[str, Any] | None = None
    initial_anchor_pair: bool = False

    def asdict(self) -> dict[str, Any]:
        return {
            "pair_index": self.pair_index,
            "latent_l1_from_prev": self.latent_l1_from_prev,
            "latent_l2_from_prev": self.latent_l2_from_prev,
            "latent_linf_from_prev": self.latent_linf_from_prev,
            "active_delta_dims": self.active_delta_dims,
            "largest_delta_dims": list(self.largest_delta_dims),
            "latent_value_sum": self.latent_value_sum,
            "latent_value_l2": self.latent_value_l2,
            "proxy_rank_signal": self.proxy_rank_signal,
            "estimated_min_patch_bytes": self.estimated_min_patch_bytes,
            "break_even_score_cost": self.break_even_score_cost,
            "break_even_seg_dist_reduction": self.break_even_seg_dist_reduction,
            "break_even_pose_dist_reduction": self.break_even_pose_dist_reduction,
            "component_trace": self.component_trace,
            "initial_anchor_pair": self.initial_anchor_pair,
        }


@dataclasses.dataclass(frozen=True)
class ComponentTraceProfile:
    samples_by_pair: dict[int, dict[str, Any]]
    source_json: str | None = None
    source_json_sha256: str | None = None
    schema_version: int | None = None
    n_samples: int | None = None
    archive_sha256: str | None = None
    archive_size_bytes: int | None = None
    avg_posenet_dist: float | None = None
    avg_segnet_dist: float | None = None
    score_recomputed_from_components: float | None = None
    cross_check_all_match: bool | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PR95AtomPlanError(f"{path} must contain a JSON object")
    return payload


def _finite_float(payload: Mapping[str, Any], key: str) -> float:
    value = float(payload[key])
    if not math.isfinite(value):
        raise PR95AtomPlanError(f"{key} must be finite, got {payload[key]!r}")
    return value


def _json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_exact_baseline(path: Path | str) -> dict[str, Any]:
    payload = load_json(path)
    required = ("archive_size_bytes", "avg_posenet_dist", "avg_segnet_dist", "score_recomputed_from_components")
    missing = [key for key in required if key not in payload]
    if missing:
        raise PR95AtomPlanError(f"{path} missing exact-eval field(s): {missing}")
    archive_sha = payload.get("provenance", {}).get("archive_sha256")
    return {
        "source_json": str(path),
        "source_json_sha256": sha256_file(Path(path)),
        "archive_size_bytes": int(payload["archive_size_bytes"]),
        "archive_sha256": archive_sha,
        "avg_posenet_dist": _finite_float(payload, "avg_posenet_dist"),
        "avg_segnet_dist": _finite_float(payload, "avg_segnet_dist"),
        "score_recomputed_from_components": _finite_float(payload, "score_recomputed_from_components"),
        "score_pose_contribution": _finite_float(payload, "score_pose_contribution"),
        "score_seg_contribution": _finite_float(payload, "score_seg_contribution"),
        "score_rate_contribution": _finite_float(payload, "score_rate_contribution"),
        "n_samples": int(payload.get("n_samples", 0)),
        "hardware": payload.get("provenance", {}).get("hardware"),
        "runtime_tree_sha256": payload.get("provenance", {})
        .get("inflate_runtime_manifest", {})
        .get("runtime_tree_sha256"),
    }


def _optional_finite_float(payload: Mapping[str, Any], key: str) -> float | None:
    if key not in payload or payload[key] is None:
        return None
    return _finite_float(payload, key)


def _unwrap_component_trace_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("component_trace"), dict) and "samples" not in payload:
        return dict(payload["component_trace"])
    if isinstance(payload.get("trace"), dict) and "samples" not in payload:
        return dict(payload["trace"])
    return payload


def _close_enough(actual: float | None, expected: float | None, *, tolerance: float) -> bool:
    if actual is None or expected is None:
        return True
    return abs(actual - expected) <= tolerance


def _sample_optional_float(sample: Mapping[str, Any], key: str) -> float | None:
    if key not in sample or sample[key] is None:
        return None
    value = float(sample[key])
    if not math.isfinite(value):
        raise PR95AtomPlanError(f"component trace {key} must be finite, got {sample[key]!r}")
    return value


def load_component_trace(
    path: Path | str | None,
    *,
    expected_pairs: int,
    expected_archive_sha256: str | None = None,
    expected_archive_size_bytes: int | None = None,
    exact_baseline: Mapping[str, Any] | None = None,
) -> ComponentTraceProfile:
    if path is None:
        return ComponentTraceProfile(samples_by_pair={})
    raw_payload = load_json(path)
    payload = _unwrap_component_trace_payload(raw_payload)
    if payload.get("score_claim") is not False:
        raise PR95AtomPlanError("component trace must be diagnostic with score_claim=false")
    if payload.get("evidence_grade") != "diagnostic_component_trace":
        raise PR95AtomPlanError("component trace must have evidence_grade=diagnostic_component_trace")
    schema_version = int(payload.get("schema_version", 0))
    if schema_version != 1:
        raise PR95AtomPlanError(f"component trace schema_version must be 1, got {schema_version!r}")
    n_samples = payload.get("n_samples")
    if n_samples is not None and int(n_samples) != expected_pairs:
        raise PR95AtomPlanError(f"component trace n_samples must be {expected_pairs}, got {n_samples}")
    expected_contest_samples = payload.get("expected_contest_samples")
    if expected_contest_samples is not None and int(expected_contest_samples) != expected_pairs:
        raise PR95AtomPlanError(
            f"component trace expected_contest_samples must be {expected_pairs}, got {expected_contest_samples}"
        )
    archive_size_bytes = payload.get("archive_size_bytes")
    if (
        expected_archive_size_bytes is not None
        and archive_size_bytes is not None
        and int(archive_size_bytes) != expected_archive_size_bytes
    ):
        raise PR95AtomPlanError(
            "component trace archive_size_bytes mismatch: "
            f"{archive_size_bytes} != {expected_archive_size_bytes}"
        )
    trace_inputs = payload.get("trace_inputs") if isinstance(payload.get("trace_inputs"), dict) else {}
    trace_archive_sha = trace_inputs.get("archive_sha256")
    if expected_archive_sha256 and trace_archive_sha and trace_archive_sha != expected_archive_sha256:
        raise PR95AtomPlanError(
            "component trace archive SHA mismatch: "
            f"{trace_archive_sha} != {expected_archive_sha256}"
        )
    cross_check = (
        payload.get("contest_auth_eval_cross_check")
        if isinstance(payload.get("contest_auth_eval_cross_check"), dict)
        else None
    )
    cross_check_all_match = None
    if cross_check is not None:
        if cross_check.get("all_match") is not True:
            raise PR95AtomPlanError("component trace contest_auth_eval_cross_check.all_match must be true")
        cross_check_all_match = True
    avg_pose = _optional_finite_float(payload, "avg_posenet_dist")
    avg_seg = _optional_finite_float(payload, "avg_segnet_dist")
    score = _optional_finite_float(payload, "score_recomputed_from_components")
    if exact_baseline is not None:
        if not _close_enough(avg_pose, float(exact_baseline["avg_posenet_dist"]), tolerance=1e-5):
            raise PR95AtomPlanError("component trace avg_posenet_dist does not match exact baseline")
        if not _close_enough(avg_seg, float(exact_baseline["avg_segnet_dist"]), tolerance=1e-5):
            raise PR95AtomPlanError("component trace avg_segnet_dist does not match exact baseline")
        if not _close_enough(score, float(exact_baseline["score_recomputed_from_components"]), tolerance=1e-5):
            raise PR95AtomPlanError(
                "component trace score_recomputed_from_components does not match exact baseline"
            )
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != expected_pairs:
        raise PR95AtomPlanError(f"component trace must contain {expected_pairs} samples")
    out: dict[int, dict[str, Any]] = {}
    for sample in samples:
        pair_index = int(sample["pair_index"])
        if pair_index in out:
            raise PR95AtomPlanError(f"duplicate component trace pair {pair_index}")
        posenet_dist = float(sample["posenet_dist"])
        segnet_dist = float(sample["segnet_dist"])
        if not math.isfinite(posenet_dist) or not math.isfinite(segnet_dist):
            raise PR95AtomPlanError(f"component trace pair {pair_index} has non-finite component distance")
        score_seg = _sample_optional_float(sample, "score_seg_contribution_exact")
        score_pose = _sample_optional_float(sample, "score_pose_contribution_first_order")
        score_combined = _sample_optional_float(sample, "score_combined_contribution_first_order")
        if score_combined is None:
            if score_seg is None:
                score_seg = 100.0 * segnet_dist / expected_pairs
            if score_pose is None and avg_pose is not None and avg_pose > 0.0:
                score_pose = (5.0 / math.sqrt(10.0 * avg_pose)) * (posenet_dist / expected_pairs)
            if score_pose is None:
                raise PR95AtomPlanError(
                    f"component trace pair {pair_index} missing first-order pose contribution"
                )
            score_combined = score_seg + score_pose
        out[pair_index] = {
            "posenet_dist": posenet_dist,
            "segnet_dist": segnet_dist,
            "score_seg_contribution_exact": score_seg,
            "score_pose_contribution_first_order": score_pose,
            "score_combined_contribution_first_order": score_combined,
        }
    missing = sorted(set(range(expected_pairs)) - set(out))
    if missing:
        raise PR95AtomPlanError(f"component trace missing pairs: {missing[:8]}")
    return ComponentTraceProfile(
        samples_by_pair=out,
        source_json=str(path),
        source_json_sha256=sha256_file(Path(path)),
        schema_version=schema_version,
        n_samples=int(n_samples) if n_samples is not None else len(samples),
        archive_sha256=trace_archive_sha,
        archive_size_bytes=int(archive_size_bytes) if archive_size_bytes is not None else None,
        avg_posenet_dist=avg_pose,
        avg_segnet_dist=avg_seg,
        score_recomputed_from_components=score,
        cross_check_all_match=cross_check_all_match,
    )


def latent_rows(payload: LatentPayload) -> list[list[int]]:
    return [list(row) for row in payload.quantized]


def latent_payload_from_rows(source: LatentPayload, rows: Sequence[Sequence[int]]) -> LatentPayload:
    if len(rows) != source.n_pairs:
        raise PR95AtomPlanError(f"expected {source.n_pairs} latent rows, got {len(rows)}")
    checked: list[tuple[int, ...]] = []
    for pair_index, row in enumerate(rows):
        if len(row) != source.latent_dim:
            raise PR95AtomPlanError(
                f"pair {pair_index} expected latent_dim={source.latent_dim}, got {len(row)}"
            )
        out_row: list[int] = []
        for dim_index, value in enumerate(row):
            ivalue = int(value)
            if not 0 <= ivalue <= 255:
                raise PR95AtomPlanError(
                    f"latent value out of uint8 range at pair {pair_index}, dim {dim_index}: {ivalue}"
                )
            out_row.append(ivalue)
        checked.append(tuple(out_row))
    return LatentPayload(
        n_pairs=source.n_pairs,
        latent_dim=source.latent_dim,
        mins_f16=source.mins_f16,
        scales_f16=source.scales_f16,
        quantized=tuple(checked),
    )


def estimate_min_patch_bytes(active_dims: int) -> int:
    # One byte atom tag + varint-ish pair id + count + dim/delta tuples.
    # This is conservative for tiny pair-local latent repairs and keeps break-even
    # math honest before a final entropy coder exists.
    return 4 + max(1, int(active_dims)) * 3


def pose_dist_break_even(rate_score_cost: float, avg_posenet_dist: float) -> float | None:
    if avg_posenet_dist <= 0:
        return None
    derivative = 5.0 / math.sqrt(10.0 * avg_posenet_dist)
    return rate_score_cost / derivative


def build_pair_profiles(
    latents: LatentPayload,
    *,
    baseline: Mapping[str, Any],
    component_trace: Mapping[int, Mapping[str, Any]],
) -> list[PairLatentProfile]:
    rows = latent_rows(latents)
    profiles: list[PairLatentProfile] = []
    avg_pose = float(baseline["avg_posenet_dist"])
    for pair_index, row in enumerate(rows):
        prev = rows[pair_index - 1] if pair_index else [0] * latents.latent_dim
        deltas = [int(value) - int(prev_value) for value, prev_value in zip(row, prev)]
        abs_deltas = [abs(value) for value in deltas]
        largest = tuple(
            dim
            for dim, _ in sorted(
                enumerate(abs_deltas),
                key=lambda item: (-item[1], item[0]),
            )[: min(4, latents.latent_dim)]
        )
        active = sum(1 for value in deltas if value != 0)
        estimated_bytes = estimate_min_patch_bytes(min(active, 4))
        rate_cost = estimated_bytes * RATE_SCORE_PER_BYTE
        latent_l2 = math.sqrt(sum(value * value for value in abs_deltas))
        value_l2 = math.sqrt(sum(int(value) * int(value) for value in row))
        trace = dict(component_trace[pair_index]) if pair_index in component_trace else None
        if trace and trace.get("score_combined_contribution_first_order") is not None:
            proxy = float(trace["score_combined_contribution_first_order"])
        elif pair_index == 0:
            proxy = 0.0
        else:
            proxy = latent_l2 + 0.05 * sum(abs_deltas) + 0.25 * active
        profiles.append(
            PairLatentProfile(
                pair_index=pair_index,
                latent_l1_from_prev=sum(abs_deltas),
                latent_l2_from_prev=latent_l2,
                latent_linf_from_prev=max(abs_deltas) if abs_deltas else 0,
                active_delta_dims=active,
                largest_delta_dims=largest,
                latent_value_sum=sum(int(v) for v in row),
                latent_value_l2=value_l2,
                proxy_rank_signal=proxy,
                estimated_min_patch_bytes=estimated_bytes,
                break_even_score_cost=rate_cost,
                break_even_seg_dist_reduction=rate_cost / 100.0,
                break_even_pose_dist_reduction=pose_dist_break_even(rate_cost, avg_pose),
                component_trace=trace,
                initial_anchor_pair=pair_index == 0,
            )
        )
    return profiles


def validate_single_member_archive(path: Path) -> tuple[str, bytes, dict[str, Any]]:
    member, blob, zip_meta = read_single_member_zip(path)
    if member != "0.bin":
        raise PR95AtomPlanError(f"PR95-family archive must contain exactly 0.bin, got {member!r}")
    with zipfile.ZipFile(path, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise PR95AtomPlanError(f"zip CRC validation failed at {bad}")
        names = [info.filename for info in zf.infolist()]
    forbidden = [name for name in names if name.startswith("/") or ".." in Path(name).parts]
    if forbidden:
        raise PR95AtomPlanError(f"zip-slip unsafe member(s): {forbidden}")
    return member, blob, zip_meta


def atom_rows_from_plan_payload(
    plan: Mapping[str, Any],
    *,
    source_archive_sha256: str,
    source_member_sha256: str,
    latents: LatentPayload,
    plan_label: str,
) -> tuple[list[list[int]], dict[str, Any]]:
    if plan.get("source_archive_sha256") != source_archive_sha256:
        raise PR95AtomPlanError(
            "atom plan source_archive_sha256 mismatch: "
            f"{plan.get('source_archive_sha256')!r} != {source_archive_sha256}"
        )
    if plan.get("source_member_sha256") != source_member_sha256:
        raise PR95AtomPlanError("atom plan source_member_sha256 mismatch")
    if plan.get("forbid_sidecars", True) is not True:
        raise PR95AtomPlanError("atom plan must forbid sidecars")
    atoms = plan.get("atoms")
    if not isinstance(atoms, list) or not atoms:
        raise PR95AtomPlanError("atom plan must contain at least one atom")
    original_rows = latent_rows(latents)
    rows = [list(row) for row in original_rows]
    changes: list[dict[str, Any]] = []
    seen_targets: set[tuple[int, int]] = set()
    for atom_index, atom in enumerate(atoms):
        if not isinstance(atom, dict):
            raise PR95AtomPlanError(f"atom {atom_index} must be an object")
        kind = atom.get("kind")
        pair_index = int(atom.get("pair_index"))
        dim_index = int(atom.get("dim_index"))
        if not 0 <= pair_index < latents.n_pairs:
            raise PR95AtomPlanError(f"atom {atom_index} pair_index out of range: {pair_index}")
        if not 0 <= dim_index < latents.latent_dim:
            raise PR95AtomPlanError(f"atom {atom_index} dim_index out of range: {dim_index}")
        target = (pair_index, dim_index)
        if target in seen_targets:
            raise PR95AtomPlanError(
                f"atom {atom_index} rewrites duplicate target pair={pair_index}, dim={dim_index}"
            )
        seen_targets.add(target)
        old_value = rows[pair_index][dim_index]
        expected_old = atom.get("expected_old_value")
        if expected_old is not None and int(expected_old) != old_value:
            raise PR95AtomPlanError(
                f"atom {atom_index} expected_old_value mismatch: {expected_old} != {old_value}"
            )
        if kind == "latent_uint8_delta":
            new_value = old_value + int(atom.get("delta"))
        elif kind == "latent_uint8_set":
            new_value = int(atom.get("value"))
        else:
            raise PR95AtomPlanError(f"atom {atom_index} unsupported kind: {kind!r}")
        if not 0 <= new_value <= 255:
            raise PR95AtomPlanError(f"atom {atom_index} new value out of uint8 range: {new_value}")
        if new_value == old_value:
            raise PR95AtomPlanError(f"atom {atom_index} is a no-op")
        rows[pair_index][dim_index] = new_value
        if rows[pair_index][dim_index] == original_rows[pair_index][dim_index]:
            raise PR95AtomPlanError(
                f"atom {atom_index} is source-preserving at pair={pair_index}, dim={dim_index}"
            )
        changes.append(
            {
                "atom_index": atom_index,
                "kind": kind,
                "pair_index": pair_index,
                "dim_index": dim_index,
                "old_value": old_value,
                "new_value": new_value,
                "delta": new_value - old_value,
            }
        )
    if rows == original_rows:
        raise PR95AtomPlanError(f"{plan_label} is source-preserving")
    return rows, {"plan": dict(plan), "changes": changes}


def atom_rows_from_plan(
    plan_path: Path,
    *,
    source_archive: Path,
    source_member_sha256: str,
    latents: LatentPayload,
) -> tuple[list[list[int]], dict[str, Any]]:
    plan = load_json(plan_path)
    return atom_rows_from_plan_payload(
        plan,
        source_archive_sha256=sha256_file(source_archive),
        source_member_sha256=source_member_sha256,
        latents=latents,
        plan_label=str(plan_path),
    )


def _signed_delta_toward_previous(
    rows: Sequence[Sequence[int]],
    pair_index: int,
    dim_index: int,
) -> int | None:
    if pair_index <= 0:
        return None
    current = int(rows[pair_index][dim_index])
    previous = int(rows[pair_index - 1][dim_index])
    if current > previous:
        return -1
    if current < previous:
        return 1
    return None


def build_signed_atom_policy(
    *,
    source_archive_sha256: str,
    source_member_sha256: str,
    component_trace: ComponentTraceProfile,
    latents: LatentPayload,
    ranked_profiles: Sequence[PairLatentProfile],
    signed_policy_pairs: int,
    signed_policy_dims_per_pair: int,
) -> dict[str, Any]:
    if not component_trace.samples_by_pair:
        raise PR95AtomPlanError("signed policy requires a component trace")
    if signed_policy_pairs <= 0:
        raise PR95AtomPlanError("signed_policy_pairs must be positive")
    if signed_policy_dims_per_pair <= 0:
        raise PR95AtomPlanError("signed_policy_dims_per_pair must be positive")

    rows = latent_rows(latents)
    atoms: list[dict[str, Any]] = []
    selected_pairs: list[int] = []
    for profile in ranked_profiles:
        if len(selected_pairs) >= signed_policy_pairs:
            break
        if profile.pair_index == 0:
            continue
        trace = profile.component_trace
        if trace is None:
            continue
        if float(profile.proxy_rank_signal) <= 0.0:
            continue
        pair_atoms: list[dict[str, Any]] = []
        for dim_index in profile.largest_delta_dims:
            delta = _signed_delta_toward_previous(rows, profile.pair_index, dim_index)
            if delta is None:
                continue
            current = int(rows[profile.pair_index][dim_index])
            previous = int(rows[profile.pair_index - 1][dim_index])
            pair_atoms.append(
                {
                    "kind": "latent_uint8_delta",
                    "pair_index": profile.pair_index,
                    "dim_index": int(dim_index),
                    "expected_old_value": current,
                    "delta": delta,
                    "previous_pair_value": previous,
                    "source_delta_from_previous_pair": current - previous,
                    "sign_rule": "shrink_pair_delta_toward_previous_latent_by_one_uint8",
                    "component_trace": {
                        "score_combined_contribution_first_order": trace.get(
                            "score_combined_contribution_first_order"
                        ),
                        "score_pose_contribution_first_order": trace.get(
                            "score_pose_contribution_first_order"
                        ),
                        "score_seg_contribution_exact": trace.get("score_seg_contribution_exact"),
                    },
                    "dispatchable_after_lane_claim": True,
                    "requires_exact_cuda_eval": True,
                }
            )
            if len(pair_atoms) >= signed_policy_dims_per_pair:
                break
        if pair_atoms:
            selected_pairs.append(profile.pair_index)
            atoms.extend(pair_atoms)

    if not atoms:
        raise PR95AtomPlanError("component trace produced no non-noop signed latent atoms")

    policy_core: dict[str, Any] = {
        "schema_version": 1,
        "policy_schema": "pr95_hnerv_residual_atom_signed_policy_v1",
        "tool": "build_pr95_hnerv_residual_atom_plan.py",
        "source_archive_sha256": source_archive_sha256,
        "source_member_sha256": source_member_sha256,
        "source_component_trace_json": component_trace.source_json,
        "source_component_trace_json_sha256": component_trace.source_json_sha256,
        "forbid_sidecars": True,
        "exact_eval_ready": True,
        "score_claim": False,
        "evidence_grade": "component_trace_signed_policy_until_exact_cuda_eval",
        "selection": {
            "signed_policy_pairs": signed_policy_pairs,
            "signed_policy_dims_per_pair": signed_policy_dims_per_pair,
            "selected_pairs": selected_pairs,
            "excluded_initial_anchor_pair": True,
            "ranking_mode": "component_trace_first_order_score",
            "sign_rule": "shrink_pair_delta_toward_previous_latent_by_one_uint8",
        },
        "safety": {
            "no_op_atoms_rejected": True,
            "source_preserving_atoms_rejected": True,
            "duplicate_pair_dim_rewrites_rejected": True,
            "sidecars_allowed": False,
            "requires_exact_cuda_eval": True,
            "requires_lane_claim_before_remote_eval": True,
        },
        "atoms": atoms,
        "notes": [
            "Signed here means deterministic atom direction, not score evidence.",
            "This policy is byte-closed input for local candidate building; promotion still requires exact CUDA auth eval.",
        ],
    }
    policy = {
        "policy_id": f"pr95_hnerv_signed_shrink_v1_{_json_sha256(policy_core)[:16]}",
        **policy_core,
    }
    atom_rows_from_plan_payload(
        policy,
        source_archive_sha256=source_archive_sha256,
        source_member_sha256=source_member_sha256,
        latents=latents,
        plan_label="generated signed policy",
    )
    return policy


def build_candidate_archive(
    *,
    source_archive: Path,
    output_archive: Path,
    atom_plan_json: Path,
    brotli_quality: int,
) -> dict[str, Any]:
    member, blob, _zip_meta = validate_single_member_archive(source_archive)
    parts = parse_top_blob(blob)
    latents = parse_latents_raw(parts["latents_raw"])
    rows, plan_meta = atom_rows_from_plan(
        atom_plan_json,
        source_archive=source_archive,
        source_member_sha256=sha256_bytes(blob),
        latents=latents,
    )
    new_latents = latent_payload_from_rows(latents, rows).to_bytes()
    if new_latents == parts["latents_raw"]:
        raise PR95AtomPlanError("atom plan produced unchanged latent raw stream")
    latents_brotli = brotli.compress(new_latents, quality=brotli_quality)
    candidate_blob = encode_top_blob(parts["meta_brotli"], parts["decoder_brotli"], latents_brotli)
    if candidate_blob == blob:
        raise PR95AtomPlanError("atom plan produced unchanged PR95 member blob")
    write_stored_zip(output_archive, member, candidate_blob)
    if sha256_file(output_archive) == sha256_file(source_archive):
        raise PR95AtomPlanError("candidate archive SHA equals source archive SHA")
    return {
        "atom_plan_json": str(atom_plan_json),
        "source_archive": str(source_archive),
        "source_archive_bytes": source_archive.stat().st_size,
        "source_archive_sha256": sha256_file(source_archive),
        "source_member_sha256": sha256_bytes(blob),
        "candidate_archive": str(output_archive),
        "candidate_archive_bytes": output_archive.stat().st_size,
        "candidate_archive_sha256": sha256_file(output_archive),
        "candidate_member_bytes": len(candidate_blob),
        "candidate_member_sha256": sha256_bytes(candidate_blob),
        "archive_byte_delta": output_archive.stat().st_size - source_archive.stat().st_size,
        "member_byte_delta": len(candidate_blob) - len(blob),
        "changed_atoms": plan_meta["changes"],
        "score_claim": False,
        "evidence_grade": "candidate_archive_until_exact_cuda_eval",
        "safety": {
            "single_member_zip": True,
            "member_name": member,
            "sidecars_allowed": False,
            "no_op_atoms_rejected": True,
            "source_sha_checked": True,
            "uses_existing_pr95_runtime_contract": True,
            "requires_exact_cuda_eval": True,
        },
    }


def emit_plan(
    *,
    source_archive: Path,
    exact_json: Path,
    output_dir: Path,
    component_trace_json: Path | None,
    top_k: int,
    build_plan_json: Path | None,
    signed_policy_pairs: int = 10,
    signed_policy_dims_per_pair: int = 2,
    build_generated_signed_policy: bool = False,
) -> dict[str, Any]:
    member, blob, zip_meta = validate_single_member_archive(source_archive)
    source_archive_sha = sha256_file(source_archive)
    source_member_sha = sha256_bytes(blob)
    parts = parse_top_blob(blob)
    latents = parse_latents_raw(parts["latents_raw"])
    baseline = load_exact_baseline(exact_json)
    if baseline["archive_size_bytes"] != source_archive.stat().st_size:
        raise PR95AtomPlanError(
            "exact baseline archive_size_bytes mismatch: "
            f"{baseline['archive_size_bytes']} != {source_archive.stat().st_size}"
        )
    if baseline.get("archive_sha256") and baseline["archive_sha256"] != source_archive_sha:
        raise PR95AtomPlanError(
            f"exact baseline archive SHA mismatch: {baseline['archive_sha256']} != {source_archive_sha}"
        )
    component_trace = load_component_trace(
        component_trace_json,
        expected_pairs=latents.n_pairs,
        expected_archive_sha256=source_archive_sha,
        expected_archive_size_bytes=source_archive.stat().st_size,
        exact_baseline=baseline,
    )
    profiles = build_pair_profiles(
        latents,
        baseline=baseline,
        component_trace=component_trace.samples_by_pair,
    )
    ranked = sorted(profiles, key=lambda row: (-row.proxy_rank_signal, row.pair_index))
    output_dir.mkdir(parents=True, exist_ok=True)

    top_rows = [row.asdict() for row in ranked[:top_k]]
    generated_atoms: list[dict[str, Any]] = []
    for row in ranked[:top_k]:
        for dim_index in row.largest_delta_dims[:2]:
            current = latents.quantized[row.pair_index][dim_index]
            generated_atoms.append(
                {
                    "kind": "latent_uint8_delta",
                    "pair_index": row.pair_index,
                    "dim_index": dim_index,
                    "expected_old_value": current,
                    "delta": 1 if current < 255 else -1,
                    "dispatchable": False,
                    "reason": "proxy-ranked perturbation only; requires component-trace or optimizer sign evidence",
                }
            )

    planning_atom_path = output_dir / "pr95_hnerv_residual_atom_plan.proxy.json"
    planning_atom_payload = {
        "schema_version": 1,
        "tool": "build_pr95_hnerv_residual_atom_plan.py",
        "source_archive_sha256": source_archive_sha,
        "source_member_sha256": source_member_sha,
        "forbid_sidecars": True,
        "exact_eval_ready": False,
        "atoms": generated_atoms,
        "notes": [
            "Proxy atom signs are not dispatchable. Use this as an optimizer template, not as score evidence.",
            "Set exact_eval_ready only after component-trace or optimizer evidence chooses atom values.",
        ],
    }
    planning_atom_path.write_text(json.dumps(planning_atom_payload, indent=2, sort_keys=True) + "\n")

    signed_policy_path: Path | None = None
    signed_policy_payload: dict[str, Any] | None = None
    if component_trace.samples_by_pair:
        signed_policy_payload = build_signed_atom_policy(
            source_archive_sha256=source_archive_sha,
            source_member_sha256=source_member_sha,
            component_trace=component_trace,
            latents=latents,
            ranked_profiles=ranked,
            signed_policy_pairs=signed_policy_pairs,
            signed_policy_dims_per_pair=signed_policy_dims_per_pair,
        )
        signed_policy_path = output_dir / SIGNED_POLICY_FILENAME
        signed_policy_path.write_text(
            json.dumps(signed_policy_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if build_generated_signed_policy and build_plan_json is not None:
        raise PR95AtomPlanError("--build-generated-signed-policy cannot be combined with --build-plan-json")
    selected_build_plan_json = build_plan_json
    if build_generated_signed_policy:
        if signed_policy_path is None:
            raise PR95AtomPlanError("--build-generated-signed-policy requires --component-trace-json")
        selected_build_plan_json = signed_policy_path

    build_result: dict[str, Any] | None = None
    if selected_build_plan_json is not None:
        build_result = build_candidate_archive(
            source_archive=source_archive,
            output_archive=output_dir / "archive.pr95_hnerv_residual_atoms.zip",
            atom_plan_json=selected_build_plan_json,
            brotli_quality=11,
        )

    manifest = {
        "schema_version": 1,
        "tool": "build_pr95_hnerv_residual_atom_plan.py",
        "source_archive": str(source_archive),
        "source_archive_bytes": source_archive.stat().st_size,
        "source_archive_sha256": source_archive_sha,
        "source_member": member,
        "source_member_bytes": len(blob),
        "source_member_sha256": source_member_sha,
        "source_zip_member_meta": zip_meta,
        "exact_baseline": baseline,
        "pr95_blob_streams": {
            "meta_brotli_bytes": len(parts["meta_brotli"]),
            "meta_raw_bytes": len(parts["meta_raw"]),
            "decoder_brotli_bytes": len(parts["decoder_brotli"]),
            "decoder_raw_bytes": len(parts["decoder_raw"]),
            "latents_brotli_bytes": len(parts["latents_brotli"]),
            "latents_raw_bytes": len(parts["latents_raw"]),
            "latent_pairs": latents.n_pairs,
            "latent_dim": latents.latent_dim,
        },
        "component_trace": {
            "available": bool(component_trace.samples_by_pair),
            "source_json": component_trace.source_json,
            "source_json_sha256": component_trace.source_json_sha256,
            "schema_version": component_trace.schema_version,
            "n_samples": component_trace.n_samples,
            "archive_sha256": component_trace.archive_sha256,
            "archive_size_bytes": component_trace.archive_size_bytes,
            "cross_check_all_match": component_trace.cross_check_all_match,
            "ranking_mode": "component_trace_first_order_score"
            if component_trace.samples_by_pair
            else "latent_proxy_only",
        },
        "opportunity_ledger": {
            "top_k": top_k,
            "ranked_pairs": top_rows,
            "all_pairs_json": "pr95_hnerv_pair_opportunity_ledger.json",
            "break_even_math": {
                "rate_score_per_byte": RATE_SCORE_PER_BYTE,
                "seg_dist_reduction_per_added_byte": RATE_SCORE_PER_BYTE / 100.0,
                "pose_dist_reduction_formula": "rate_score_cost / (5 / sqrt(10 * avg_posenet_dist))",
            },
        },
        "planning_atom_template": str(planning_atom_path),
        "signed_atom_policy": None
        if signed_policy_path is None
        else {
            "path": str(signed_policy_path),
            "sha256": sha256_file(signed_policy_path),
            "policy_id": signed_policy_payload["policy_id"] if signed_policy_payload else None,
            "atom_count": len(signed_policy_payload["atoms"]) if signed_policy_payload else 0,
            "selected_pairs": signed_policy_payload["selection"]["selected_pairs"]
            if signed_policy_payload
            else [],
            "build_generated_signed_policy": build_generated_signed_policy,
        },
        "candidate_build": build_result,
        "score_claim": False,
        "evidence_grade": "diagnostic_planning_or_candidate_until_exact_cuda_eval",
        "dispatch_readiness": {
            "ready_for_exact_eval": bool(build_result),
            "reason": (
                "candidate archive built from explicit atom plan"
                if build_result
                else (
                    "component-trace signed atom policy emitted; build a local candidate before exact eval"
                    if signed_policy_path is not None
                    else "no candidate archive built; proxy atom template is not dispatchable"
                )
            ),
            "claim_required_before_remote_eval": True,
            "example_exact_eval_command": None
            if not build_result
            else (
                ".venv/bin/python scripts/launch_lightning_batch_job.py exact-eval "
                "--job-name exact_eval_pr95_residual_atoms_t4_${UTC_STAMP} "
                "--accelerator 'gpu-t4' --machine-name g4dn.xlarge "
                f"--archive {build_result['candidate_archive']} "
                f"--inflate-sh {DEFAULT_PR95_RUNTIME_INFLATE} "
                "--require-adjudication"
            ),
        },
    }
    (output_dir / "pr95_hnerv_residual_atom_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "pr95_hnerv_pair_opportunity_ledger.json").write_text(
        json.dumps([row.asdict() for row in ranked], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--exact-json", type=Path, default=DEFAULT_PR95_REPACK_EXACT_JSON)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--component-trace-json", type=Path)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--build-plan-json", type=Path)
    parser.add_argument("--signed-policy-pairs", type=int, default=10)
    parser.add_argument("--signed-policy-dims-per-pair", type=int, default=2)
    parser.add_argument(
        "--build-generated-signed-policy",
        action="store_true",
        help="When --component-trace-json is present, build the candidate from the generated signed policy.",
    )
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.signed_policy_pairs <= 0:
        parser.error("--signed-policy-pairs must be positive")
    if args.signed_policy_dims_per_pair <= 0:
        parser.error("--signed-policy-dims-per-pair must be positive")
    if args.build_generated_signed_policy and args.build_plan_json is not None:
        parser.error("--build-generated-signed-policy cannot be combined with --build-plan-json")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = emit_plan(
        source_archive=args.archive,
        exact_json=args.exact_json,
        output_dir=args.output_dir,
        component_trace_json=args.component_trace_json,
        top_k=args.top_k,
        build_plan_json=args.build_plan_json,
        signed_policy_pairs=args.signed_policy_pairs,
        signed_policy_dims_per_pair=args.signed_policy_dims_per_pair,
        build_generated_signed_policy=args.build_generated_signed_policy,
    )
    if args.stdout:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
