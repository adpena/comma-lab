# SPDX-License-Identifier: MIT
"""Planning-only residual atom tools for PR95-family HNeRV latent streams."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import brotli

from tac.pr95_hnerv import (
    LatentPayload,
    encode_top_blob,
    latent_payload_from_rows,
    latent_rows,
    parse_latents_raw,
    parse_top_blob,
    read_single_member_zip,
    sha256_bytes,
    sha256_file,
    write_stored_zip,
)

CONTEST_ORIGINAL_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25 / CONTEST_ORIGINAL_BYTES
SIGNED_POLICY_FILENAME = "pr95_hnerv_residual_atom_plan.signed.json"


class PR95AtomPlanError(ValueError):
    """Raised when a PR95 atom plan is malformed or unsafe."""


@dataclasses.dataclass(frozen=True)
class ExactBaseline:
    archive_size_bytes: int
    archive_sha256: str | None
    avg_posenet_dist: float
    avg_segnet_dist: float
    score_recomputed_from_components: float
    n_samples: int | None


@dataclasses.dataclass(frozen=True)
class ComponentTrace:
    samples_by_pair: dict[int, dict[str, float]]


@dataclasses.dataclass(frozen=True)
class PairProfile:
    pair_index: int
    top_latent_dims: list[int]
    latent_delta_l1: int
    proxy_rank_signal: float
    estimated_min_atom_bytes: int
    rate_score_cost: float
    pose_dist_break_even: float | None
    component_trace: dict[str, float] | None


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PR95AtomPlanError(f"{path} must contain a JSON object")
    return payload


def _finite_float(payload: Mapping[str, Any], key: str) -> float:
    try:
        value = float(payload[key])
    except KeyError as exc:
        raise PR95AtomPlanError(f"missing required key: {key}") from exc
    if not math.isfinite(value):
        raise PR95AtomPlanError(f"{key} must be finite, got {payload[key]!r}")
    return value


def _optional_finite_float(payload: Mapping[str, Any], key: str) -> float | None:
    if key not in payload or payload[key] is None:
        return None
    value = float(payload[key])
    if not math.isfinite(value):
        raise PR95AtomPlanError(f"{key} must be finite, got {payload[key]!r}")
    return value


def load_exact_baseline(path: Path) -> ExactBaseline:
    payload = _load_json(path)
    required = (
        "archive_size_bytes",
        "avg_posenet_dist",
        "avg_segnet_dist",
        "score_recomputed_from_components",
    )
    for key in required:
        if key not in payload:
            raise PR95AtomPlanError(f"exact baseline missing {key}")
    archive_sha = payload.get("archive_sha256")
    provenance = payload.get("provenance")
    if archive_sha is None and isinstance(provenance, dict):
        archive_sha = provenance.get("archive_sha256")
    n_samples = payload.get("n_samples")
    return ExactBaseline(
        archive_size_bytes=int(payload["archive_size_bytes"]),
        archive_sha256=None if archive_sha is None else str(archive_sha),
        avg_posenet_dist=_finite_float(payload, "avg_posenet_dist"),
        avg_segnet_dist=_finite_float(payload, "avg_segnet_dist"),
        score_recomputed_from_components=_finite_float(payload, "score_recomputed_from_components"),
        n_samples=None if n_samples is None else int(n_samples),
    )


def _unwrap_component_trace_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("component_trace"), dict) and "samples" not in payload:
        return dict(payload["component_trace"])
    if isinstance(payload.get("trace"), dict) and "samples" not in payload:
        return dict(payload["trace"])
    return payload


def _close_enough(actual: float, expected: float, *, tolerance: float = 1e-5) -> bool:
    return abs(float(actual) - float(expected)) <= tolerance


def load_component_trace(
    path: Path | None,
    *,
    expected_pairs: int,
    expected_archive_sha256: str,
    expected_archive_size_bytes: int,
    exact_baseline: ExactBaseline,
) -> ComponentTrace:
    if path is None:
        return ComponentTrace(samples_by_pair={})
    payload = _unwrap_component_trace_payload(_load_json(path))
    if int(payload.get("schema_version", -1)) != 1:
        raise PR95AtomPlanError("component trace schema_version must be 1")
    if payload.get("score_claim") is not False:
        raise PR95AtomPlanError("component trace must be diagnostic and score_claim=false")
    if payload.get("evidence_grade") != "diagnostic_component_trace":
        raise PR95AtomPlanError("component trace evidence_grade must be diagnostic_component_trace")
    if int(payload.get("archive_size_bytes", -1)) != expected_archive_size_bytes:
        raise PR95AtomPlanError("component trace archive_size_bytes mismatch")
    trace_inputs = payload.get("trace_inputs")
    if not isinstance(trace_inputs, dict) or trace_inputs.get("archive_sha256") != expected_archive_sha256:
        raise PR95AtomPlanError("component trace archive_sha256 mismatch")
    cross = payload.get("contest_auth_eval_cross_check")
    if isinstance(cross, dict) and cross.get("all_match") is not True:
        raise PR95AtomPlanError("component trace contest_auth_eval_cross_check failed")
    if not _close_enough(_finite_float(payload, "avg_posenet_dist"), exact_baseline.avg_posenet_dist):
        raise PR95AtomPlanError("component trace avg_posenet_dist does not match exact baseline")
    if not _close_enough(_finite_float(payload, "avg_segnet_dist"), exact_baseline.avg_segnet_dist):
        raise PR95AtomPlanError("component trace avg_segnet_dist does not match exact baseline")
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise PR95AtomPlanError("component trace samples must be a list")
    out: dict[int, dict[str, float]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            raise PR95AtomPlanError("component trace sample must be an object")
        pair_index = int(sample["pair_index"])
        if not 0 <= pair_index < expected_pairs:
            raise PR95AtomPlanError(f"component trace pair_index out of range: {pair_index}")
        if pair_index in out:
            raise PR95AtomPlanError(f"duplicate component trace pair_index: {pair_index}")
        record = {
            "posenet_dist": _finite_float(sample, "posenet_dist"),
            "segnet_dist": _finite_float(sample, "segnet_dist"),
            "score_combined_contribution_first_order": (
                _optional_finite_float(sample, "score_combined_contribution_first_order") or 0.0
            ),
        }
        out[pair_index] = record
    if len(out) != expected_pairs:
        raise PR95AtomPlanError(f"component trace expected {expected_pairs} samples, got {len(out)}")
    return ComponentTrace(samples_by_pair=out)


def estimate_min_patch_bytes(active_dims: int) -> int:
    return 4 + max(1, int(active_dims)) * 3


def pose_dist_break_even(rate_score_cost: float, avg_posenet_dist: float) -> float | None:
    if avg_posenet_dist <= 0:
        return None
    derivative = 5 / math.sqrt(10 * avg_posenet_dist)
    return rate_score_cost / derivative


def build_pair_profiles(
    latents: LatentPayload,
    *,
    baseline: ExactBaseline,
    component_trace: Mapping[int, Mapping[str, float]],
    top_dims: int = 4,
) -> list[PairProfile]:
    rows = latent_rows(latents)
    profiles: list[PairProfile] = []
    for pair_index, row in enumerate(rows):
        if pair_index == 0:
            continue
        previous = rows[pair_index - 1]
        deltas = [(dim, abs(int(value) - int(previous[dim]))) for dim, value in enumerate(row)]
        ranked_dims = sorted(deltas, key=lambda item: (-item[1], item[0]))[:top_dims]
        latent_l1 = sum(delta for _, delta in deltas)
        trace = component_trace.get(pair_index)
        trace_signal = (
            float(trace.get("score_combined_contribution_first_order", 0.0))
            if trace is not None
            else 0.0
        )
        proxy_signal = trace_signal * 1000.0 + float(latent_l1)
        atom_bytes = estimate_min_patch_bytes(len([dim for dim, delta in ranked_dims if delta > 0]) or 1)
        rate_cost = atom_bytes * RATE_SCORE_PER_BYTE
        profiles.append(
            PairProfile(
                pair_index=pair_index,
                top_latent_dims=[dim for dim, _delta in ranked_dims],
                latent_delta_l1=latent_l1,
                proxy_rank_signal=proxy_signal,
                estimated_min_atom_bytes=atom_bytes,
                rate_score_cost=rate_cost,
                pose_dist_break_even=pose_dist_break_even(rate_cost, baseline.avg_posenet_dist),
                component_trace=None if trace is None else dict(trace),
            )
        )
    return profiles


def atom_rows_from_plan_payload(
    plan: Mapping[str, Any],
    *,
    source_archive_sha256: str,
    source_member_sha256: str,
    latents: LatentPayload,
    plan_label: str,
) -> tuple[list[list[int]], dict[str, Any]]:
    if plan.get("source_archive_sha256") != source_archive_sha256:
        raise PR95AtomPlanError("atom plan source_archive_sha256 mismatch")
    if plan.get("source_member_sha256") != source_member_sha256:
        raise PR95AtomPlanError("atom plan source_member_sha256 mismatch")
    if plan.get("forbid_sidecars", True) is not True:
        raise PR95AtomPlanError("atom plan must forbid sidecars")
    atoms = plan.get("atoms")
    if not isinstance(atoms, list) or not atoms:
        raise PR95AtomPlanError("atom plan must contain at least one atom")
    rows = latent_rows(latents)
    seen: set[tuple[int, int]] = set()
    changed = 0
    for atom in atoms:
        if not isinstance(atom, dict):
            raise PR95AtomPlanError("atom must be an object")
        kind = atom.get("kind")
        pair_index = int(atom["pair_index"])
        dim_index = int(atom["dim_index"])
        if not 0 <= pair_index < latents.n_pairs:
            raise PR95AtomPlanError(f"atom pair_index out of range: {pair_index}")
        if not 0 <= dim_index < latents.latent_dim:
            raise PR95AtomPlanError(f"atom dim_index out of range: {dim_index}")
        key = (pair_index, dim_index)
        if key in seen:
            raise PR95AtomPlanError(f"duplicate target latent atom: pair={pair_index}, dim={dim_index}")
        seen.add(key)
        old_value = int(rows[pair_index][dim_index])
        if int(atom.get("expected_old_value", old_value)) != old_value:
            raise PR95AtomPlanError(
                f"atom expected_old_value mismatch at pair {pair_index}, dim {dim_index}: "
                f"{atom.get('expected_old_value')} != {old_value}"
            )
        if kind == "latent_uint8_delta":
            new_value = old_value + int(atom["delta"])
        elif kind == "latent_uint8_set":
            new_value = int(atom["value"])
        else:
            raise PR95AtomPlanError(f"unsupported atom kind: {kind!r}")
        if not 0 <= new_value <= 255:
            raise PR95AtomPlanError(f"atom value out of uint8 range: {new_value}")
        if new_value == old_value:
            raise PR95AtomPlanError("no-op latent atom")
        rows[pair_index][dim_index] = new_value
        changed += 1
    return rows, {"plan_label": plan_label, "atom_count": changed}


def atom_rows_from_plan(
    plan_path: Path,
    *,
    source_archive: Path,
    source_member_sha256: str,
    latents: LatentPayload,
) -> tuple[list[list[int]], dict[str, Any]]:
    plan = _load_json(plan_path)
    return atom_rows_from_plan_payload(
        plan,
        source_archive_sha256=sha256_file(source_archive),
        source_member_sha256=source_member_sha256,
        latents=latents,
        plan_label=str(plan_path),
    )


def _signed_delta_toward_previous(rows: Sequence[Sequence[int]], pair_index: int, dim_index: int) -> int | None:
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
    component_trace: ComponentTrace,
    latents: LatentPayload,
    ranked_profiles: Sequence[PairProfile],
    signed_policy_pairs: int,
    signed_policy_dims_per_pair: int,
) -> dict[str, Any]:
    if not component_trace.samples_by_pair:
        raise PR95AtomPlanError("signed policy requires a component trace")
    rows = latent_rows(latents)
    atoms: list[dict[str, int | str]] = []
    selected_pairs: list[int] = []
    for profile in ranked_profiles:
        if len(selected_pairs) >= signed_policy_pairs:
            break
        added_for_pair = 0
        for dim_index in profile.top_latent_dims:
            delta = _signed_delta_toward_previous(rows, profile.pair_index, dim_index)
            if delta is None:
                continue
            atoms.append(
                {
                    "kind": "latent_uint8_delta",
                    "pair_index": profile.pair_index,
                    "dim_index": dim_index,
                    "expected_old_value": int(rows[profile.pair_index][dim_index]),
                    "delta": int(delta),
                }
            )
            added_for_pair += 1
            if added_for_pair >= signed_policy_dims_per_pair:
                break
        if added_for_pair:
            selected_pairs.append(profile.pair_index)
    if not atoms:
        raise PR95AtomPlanError("signed policy produced no atoms")
    return {
        "schema_version": 1,
        "source_archive_sha256": source_archive_sha256,
        "source_member_sha256": source_member_sha256,
        "forbid_sidecars": True,
        "policy": "component_trace_signed_toward_previous_pair",
        "selected_pairs": selected_pairs,
        "dispatchable": False,
        "exact_eval_ready": False,
        "score_claim": False,
        "atoms": atoms,
    }


def build_candidate_archive(
    *,
    source_archive: Path,
    output_archive: Path,
    atom_plan_json: Path,
    brotli_quality: int = 11,
) -> dict[str, Any]:
    member, blob, _zip_meta = read_single_member_zip(source_archive)
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
    latents_brotli = brotli.compress(new_latents, quality=int(brotli_quality))
    candidate_blob = encode_top_blob(parts["meta_brotli"], parts["decoder_brotli"], latents_brotli)
    if candidate_blob == blob:
        raise PR95AtomPlanError("atom plan produced unchanged PR95 member blob")
    write_stored_zip(output_archive, member, candidate_blob)
    if sha256_file(output_archive) == sha256_file(source_archive):
        raise PR95AtomPlanError("candidate archive SHA equals source archive SHA")
    return {
        "archive": str(output_archive),
        "archive_bytes": output_archive.stat().st_size,
        "archive_sha256": sha256_file(output_archive),
        "member_sha256": sha256_bytes(candidate_blob),
        "source_archive": str(source_archive),
        "source_archive_sha256": sha256_file(source_archive),
        "plan": plan_meta,
        "score_claim": False,
        "exact_eval_ready": False,
        "evidence_grade": "candidate_archive_requires_exact_cuda_eval",
    }


def emit_plan(
    *,
    source_archive: Path,
    exact_json: Path,
    output_dir: Path,
    component_trace_json: Path | None = None,
    top_k: int = 64,
    build_plan_json: Path | None = None,
    signed_policy_pairs: int = 10,
    signed_policy_dims_per_pair: int = 2,
    build_generated_signed_policy: bool = False,
    brotli_quality: int = 11,
) -> dict[str, Any]:
    member, blob, zip_meta = read_single_member_zip(source_archive)
    source_archive_sha = sha256_file(source_archive)
    source_member_sha = sha256_bytes(blob)
    parts = parse_top_blob(blob)
    latents = parse_latents_raw(parts["latents_raw"])
    baseline = load_exact_baseline(exact_json)
    if baseline.archive_size_bytes != source_archive.stat().st_size:
        raise PR95AtomPlanError("exact baseline archive_size_bytes mismatch")
    if baseline.archive_sha256 and baseline.archive_sha256 != source_archive_sha:
        raise PR95AtomPlanError("exact baseline archive SHA mismatch")
    component_trace = load_component_trace(
        component_trace_json,
        expected_pairs=latents.n_pairs,
        expected_archive_sha256=source_archive_sha,
        expected_archive_size_bytes=source_archive.stat().st_size,
        exact_baseline=baseline,
    )
    profiles = build_pair_profiles(latents, baseline=baseline, component_trace=component_trace.samples_by_pair)
    ranked = sorted(profiles, key=lambda row: (-row.proxy_rank_signal, row.pair_index))[: int(top_k)]
    output_dir.mkdir(parents=True, exist_ok=True)

    ledger_path = output_dir / "pr95_hnerv_pair_opportunity_ledger.json"
    ledger = {
        "schema_version": 1,
        "source_archive": str(source_archive),
        "source_archive_sha256": source_archive_sha,
        "source_member_sha256": source_member_sha,
        "member": member,
        "zip_info": zip_meta,
        "latent_pairs": latents.n_pairs,
        "latent_dim": latents.latent_dim,
        "component_trace_json": None if component_trace_json is None else str(component_trace_json),
        "ranking_basis": "component_trace_plus_latent_l1" if component_trace.samples_by_pair else "latent_proxy_only",
        "score_claim": False,
        "exact_eval_ready": False,
        "evidence_grade": "planning_only",
        "pairs": [dataclasses.asdict(profile) for profile in ranked],
    }
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    proxy_plan_path = output_dir / "pr95_hnerv_residual_atom_plan.proxy.json"
    proxy_plan = {
        "schema_version": 1,
        "source_archive_sha256": source_archive_sha,
        "source_member_sha256": source_member_sha,
        "forbid_sidecars": True,
        "score_claim": False,
        "exact_eval_ready": False,
        "dispatchable": False,
        "atoms": [
            {
                "kind": "latent_uint8_delta",
                "pair_index": profile.pair_index,
                "dim_index": dim,
                "expected_old_value": int(latent_rows(latents)[profile.pair_index][dim]),
                "delta": 0,
                "dispatchable": False,
                "requires": "component-trace or optimizer sign evidence",
            }
            for profile in ranked
            for dim in profile.top_latent_dims[: min(2, len(profile.top_latent_dims))]
        ],
    }
    proxy_plan_path.write_text(json.dumps(proxy_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    signed_policy_path = None
    if component_trace.samples_by_pair:
        signed_policy = build_signed_atom_policy(
            source_archive_sha256=source_archive_sha,
            source_member_sha256=source_member_sha,
            component_trace=component_trace,
            latents=latents,
            ranked_profiles=ranked,
            signed_policy_pairs=int(signed_policy_pairs),
            signed_policy_dims_per_pair=int(signed_policy_dims_per_pair),
        )
        signed_policy_path = output_dir / SIGNED_POLICY_FILENAME
        signed_policy_path.write_text(json.dumps(signed_policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    candidate_build = None
    selected_plan = build_plan_json
    if build_generated_signed_policy:
        if signed_policy_path is None:
            raise PR95AtomPlanError("--build-generated-signed-policy requires --component-trace-json")
        selected_plan = signed_policy_path
    if selected_plan is not None:
        candidate_build = build_candidate_archive(
            source_archive=source_archive,
            output_archive=output_dir / "archive.pr95_residual_atoms.zip",
            atom_plan_json=selected_plan,
            brotli_quality=int(brotli_quality),
        )

    manifest = {
        "schema_version": 1,
        "tool": "tac.pr95_residual_atoms",
        "source_archive": str(source_archive),
        "source_archive_bytes": source_archive.stat().st_size,
        "source_archive_sha256": source_archive_sha,
        "source_member_sha256": source_member_sha,
        "exact_json": str(exact_json),
        "exact_baseline": dataclasses.asdict(baseline),
        "ledger_json": str(ledger_path),
        "proxy_plan_json": str(proxy_plan_path),
        "signed_policy_json": None if signed_policy_path is None else str(signed_policy_path),
        "candidate_build": candidate_build,
        "score_claim": False,
        "exact_eval_ready": False,
        "promotion_eligible": False,
        "evidence_grade": "planning_only" if candidate_build is None else "candidate_archive_requires_exact_cuda_eval",
    }
    manifest_path = output_dir / "pr95_hnerv_residual_atom_manifest.json"
    manifest["manifest_json"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--exact-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--component-trace-json", type=Path)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--build-plan-json", type=Path)
    parser.add_argument("--signed-policy-pairs", type=int, default=10)
    parser.add_argument("--signed-policy-dims-per-pair", type=int, default=2)
    parser.add_argument("--build-generated-signed-policy", action="store_true")
    parser.add_argument("--brotli-quality", type=int, default=11)
    parser.add_argument("--stdout", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
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
        brotli_quality=args.brotli_quality,
    )
    if args.stdout:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
