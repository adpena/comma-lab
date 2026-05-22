#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan and materialize fixed-length decoder-q waterbucket candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.fec6_decoder_mutations import (  # noqa: E402
    DecoderQMutation,
    Fec6DecoderMutationError,
    apply_q_mutations,
    extract_fec6_decoder_blob,
    prepare_decoder_blob,
    recompress_prepared_decoder,
    replace_fec6_decoder_blob,
    sha256_bytes,
    write_stored_archive,
)
from tac.optimization.decoder_q_surface_objective import (  # noqa: E402
    build_surface_objective,
    sort_atoms_for_surface_objective,
    summarize_candidate_surface_proxy,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_csv_ints(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise SystemExit("expected at least one integer")
    return out


def _mutation_key(row: dict[str, Any]) -> tuple[str, int, int]:
    mutation = row["mutation"]
    return str(mutation["tensor_name"]), int(mutation["q_offset"]), int(mutation["delta"])


def _offset_key(row: dict[str, Any]) -> tuple[str, int]:
    tensor_name, q_offset, _delta = _mutation_key(row)
    return tensor_name, q_offset


def _target_mass(row: dict[str, Any]) -> float:
    evidence = row.get("op3v3_target_evidence")
    if isinstance(evidence, dict):
        return float(evidence.get("score_impact_abs_sum", 0.0))
    return 0.0


def _axis_mass(row: dict[str, Any]) -> dict[str, float]:
    evidence = row.get("op3v3_target_evidence")
    axis = evidence.get("axis_score_impact_abs_sum") if isinstance(evidence, dict) else None
    if not isinstance(axis, dict):
        axis = {}
    return {
        "seg": float(axis.get("seg", 0.0)),
        "pose": float(axis.get("pose", 0.0)),
        "rate": float(axis.get("rate", 0.0)),
    }


def _fixed_rows(feasibility: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in feasibility.get("fixed_length_runtime_compatible_rows", [])
        if isinstance(row, dict)
        and bool(row.get("fixed_length_runtime_compatible"))
        and int(row.get("length_delta", 999)) == 0
    ]


def _advisory_by_key(summary: dict[str, Any] | None, baseline_score: float | None) -> dict[tuple[str, int, int], dict[str, Any]]:
    if summary is None:
        return {}
    out: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in summary.get("candidates", []):
        if not isinstance(row, dict):
            continue
        manifest = row.get("mutation_manifest")
        advisory = row.get("advisory_eval")
        if not isinstance(manifest, dict) or not isinstance(advisory, dict):
            continue
        mutation_row = manifest.get("mutation_row")
        if not isinstance(mutation_row, dict):
            continue
        score = advisory.get("canonical_score")
        delta_score = row.get("delta_vs_baseline_score")
        if delta_score is None and baseline_score is not None and score is not None:
            delta_score = float(score) - float(baseline_score)
        out[_mutation_key(mutation_row)] = {
            "candidate_id": row.get("candidate_id"),
            "returncode": advisory.get("returncode"),
            "canonical_score": score,
            "delta_vs_baseline_score": delta_score,
            "avg_posenet_dist": advisory.get("avg_posenet_dist"),
            "avg_segnet_dist": advisory.get("avg_segnet_dist"),
        }
    return out


def _atom(row: dict[str, Any], advisory: dict[str, Any] | None) -> dict[str, Any]:
    mutation = row["mutation"]
    return {
        "candidate_id": row["mutation_id"],
        "mutation": mutation,
        "q_before": row.get("q_before"),
        "q_after": row.get("q_after"),
        "target_mass": _target_mass(row),
        "axis_mass": _axis_mass(row),
        "advisory": advisory,
    }


def _rank_atoms(
    rows: list[dict[str, Any]],
    advisory_by_key: dict[tuple[str, int, int], dict[str, Any]],
    *,
    surface_objective: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    atoms = [_atom(row, advisory_by_key.get(_mutation_key(row))) for row in rows]

    def improved(atom: dict[str, Any]) -> bool:
        advisory = atom.get("advisory")
        return (
            isinstance(advisory, dict)
            and advisory.get("returncode") == 0
            and advisory.get("delta_vs_baseline_score") is not None
            and float(advisory["delta_vs_baseline_score"]) < 0.0
        )

    def measured_bad(atom: dict[str, Any]) -> bool:
        advisory = atom.get("advisory")
        return (
            isinstance(advisory, dict)
            and advisory.get("returncode") == 0
            and advisory.get("delta_vs_baseline_score") is not None
            and float(advisory["delta_vs_baseline_score"]) >= 0.0
        )

    by_offset: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for atom in atoms:
        mutation = atom["mutation"]
        by_offset[(str(mutation["tensor_name"]), int(mutation["q_offset"]))].append(atom)

    sign_inverted: list[dict[str, Any]] = []
    for group in by_offset.values():
        bad = [atom for atom in group if measured_bad(atom)]
        if not bad:
            continue
        bad_signs = {1 if int(atom["mutation"]["delta"]) > 0 else -1 for atom in bad}
        for atom in group:
            sign = 1 if int(atom["mutation"]["delta"]) > 0 else -1
            if -sign in bad_signs and atom.get("advisory") is None:
                sign_inverted.append(atom)

    unmeasured = [atom for atom in atoms if atom.get("advisory") is None]
    good = [atom for atom in atoms if improved(atom)]
    bad = [atom for atom in atoms if measured_bad(atom)]
    bias = [atom for atom in atoms if str(atom["mutation"]["tensor_name"]).endswith(".bias")]

    score_sort = lambda atom: (
        -float(atom["target_mass"]),
        abs(int(atom["mutation"]["delta"])),
        str(atom["mutation"]["tensor_name"]),
        int(atom["mutation"]["q_offset"]),
        int(atom["mutation"]["delta"]),
    )
    for bucket in (unmeasured, good, bad, bias, sign_inverted):
        bucket.sort(key=score_sort)

    mixed = []
    seen_offsets = set()
    for atom in unmeasured:
        key = (atom["mutation"]["tensor_name"], atom["mutation"]["q_offset"])
        if key in seen_offsets:
            continue
        seen_offsets.add(key)
        mixed.append(atom)

    negative_control = bad[:]
    if not negative_control:
        negative_control = atoms[-8:]

    ranked = {
        "seg_heavy": good + unmeasured,
        "sign_inverted_from_bad": sign_inverted + unmeasured,
        "mixed_tradeoff": mixed + unmeasured,
        "bias_only": bias,
        "negative_control": negative_control,
    }
    if surface_objective is not None:
        if surface_objective["strategy"] == "suppress_or_invert_regressions_first":
            surface_atoms = sign_inverted + unmeasured + good
            preferred_direction = "suppress"
        else:
            surface_atoms = good + unmeasured + sign_inverted
            preferred_direction = "preserve"
        ranked["response_surface_guided"] = sort_atoms_for_surface_objective(
            surface_atoms,
            surface_objective,
            preferred_direction=preferred_direction,
        )
    return ranked


def _dedupe_atoms(atoms: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for atom in atoms:
        mutation = atom["mutation"]
        key = (str(mutation["tensor_name"]), int(mutation["q_offset"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(atom)
        if len(out) >= budget:
            break
    return out


def _candidate_id(bucket: str, budget: int, mutations: list[DecoderQMutation], decoder_sha: str) -> str:
    seed = {
        "bucket": bucket,
        "budget": budget,
        "mutations": [mutation.as_dict() for mutation in mutations],
        "decoder_sha256": decoder_sha,
    }
    return hashlib.sha256(json.dumps(seed, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _materialize_candidate(
    *,
    output_dir: Path,
    bucket: str,
    budget: int,
    atoms: list[dict[str, Any]],
    member_bytes: bytes,
    prepared: Any,
    brotli_quality: int,
    surface_objective: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mutations = [
        DecoderQMutation(
            tensor_name=str(atom["mutation"]["tensor_name"]),
            q_offset=int(atom["mutation"]["q_offset"]),
            delta=int(atom["mutation"]["delta"]),
        )
        for atom in atoms
    ]
    raw, records = apply_q_mutations(prepared, mutations)
    decoder_blob = recompress_prepared_decoder(prepared, raw, brotli_quality=brotli_quality)
    decoder_len = len(decoder_blob)
    fixed_length = decoder_len == len(prepared.decoder_blob)
    candidate_id = _candidate_id(bucket, budget, mutations, sha256_bytes(decoder_blob))
    row: dict[str, Any] = {
        "candidate_id": candidate_id,
        "bucket": bucket,
        "edit_budget": budget,
        "edit_count": len(mutations),
        "atoms": atoms,
        "mutation_records": records,
        "source_decoder_len": len(prepared.decoder_blob),
        "mutated_decoder_len": decoder_len,
        "length_delta": decoder_len - len(prepared.decoder_blob),
        "mutated_decoder_sha256": sha256_bytes(decoder_blob),
        "fixed_length_runtime_compatible": fixed_length,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    surface_proxy = summarize_candidate_surface_proxy(atoms, surface_objective)
    if surface_proxy is not None:
        row["response_surface_objective"] = surface_proxy
    if not fixed_length:
        row["blockers"] = ["combined_decoder_length_changed"]
        return row
    replacement = replace_fec6_decoder_blob(member_bytes, decoder_blob)
    out_dir = output_dir / candidate_id
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_bin = out_dir / "archive.bin"
    archive_zip = out_dir / "archive.zip"
    archive_bin.write_bytes(replacement)
    write_stored_archive(archive_zip, replacement)
    row.update(
        {
            "archive_bin_path": str(archive_bin.resolve()),
            "archive_zip_path": str(archive_zip.resolve()),
            "archive_bin_bytes": len(replacement),
            "archive_zip_bytes": archive_zip.stat().st_size,
            "archive_bin_sha256": sha256_bytes(replacement),
            "archive_zip_sha256": _sha256_file(archive_zip),
            "blockers": [
                "official_inflate_control_missing",
                "advisory_component_response_not_measured",
                "exact_cuda_auth_eval_missing",
            ],
        }
    )
    _write_json(out_dir / "mutation_manifest.json", row)
    return row


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    feasibility = _read_json(args.feasibility.resolve())
    advisory_summary = _read_json(args.advisory_summary.resolve()) if args.advisory_summary and args.advisory_summary.is_file() else None
    response_surface = _read_json(args.decoder_q_response_surface.resolve()) if args.decoder_q_response_surface and args.decoder_q_response_surface.is_file() else None
    surface_objective = build_surface_objective(response_surface) if response_surface is not None else None
    rows = _fixed_rows(feasibility)
    advisory_by_key = _advisory_by_key(advisory_summary, args.baseline_score)
    ranked = _rank_atoms(rows, advisory_by_key, surface_objective=surface_objective)
    budgets = _parse_csv_ints(args.edit_budgets)
    requested_buckets = [part.strip() for part in args.buckets.split(",") if part.strip()]
    member_bytes = args.archive_bin.resolve().read_bytes()
    member_sha = sha256_bytes(member_bytes)
    if args.expected_member_sha256 and member_sha != args.expected_member_sha256:
        raise SystemExit(
            f"archive member SHA mismatch: expected {args.expected_member_sha256}, got {member_sha}"
        )
    prepared = prepare_decoder_blob(extract_fec6_decoder_blob(member_bytes))

    candidates = []
    output_dir = args.candidate_output_dir.resolve()
    for bucket in requested_buckets:
        atoms = ranked.get(bucket, [])
        if not atoms:
            continue
        for budget in budgets:
            selected = _dedupe_atoms(atoms, int(budget))
            if len(selected) != int(budget):
                continue
            try:
                candidates.append(
                    _materialize_candidate(
                        output_dir=output_dir,
                        bucket=bucket,
                        budget=int(budget),
                        atoms=selected,
                        member_bytes=member_bytes,
                        prepared=prepared,
                        brotli_quality=args.brotli_quality,
                        surface_objective=surface_objective if bucket == "response_surface_guided" else None,
                    )
                )
            except Fec6DecoderMutationError as exc:
                candidates.append(
                    {
                        "candidate_id": None,
                        "bucket": bucket,
                        "edit_budget": int(budget),
                        "error": str(exc),
                        "score_claim": False,
                        "promotion_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    }
                )
    fixed = [row for row in candidates if row.get("fixed_length_runtime_compatible")]
    return {
        "schema": "fec6_decoder_q_signed_waterbucket_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "producer": "tools/plan_decoder_q_signed_waterbucket.py",
        "inputs": {
            "archive_bin": str(args.archive_bin.resolve()),
            "archive_bin_sha256": member_sha,
            "feasibility": str(args.feasibility.resolve()),
            "advisory_summary": str(args.advisory_summary.resolve()) if args.advisory_summary else None,
            "decoder_q_response_surface": str(args.decoder_q_response_surface.resolve()) if args.decoder_q_response_surface else None,
            "baseline_score": args.baseline_score,
            "edit_budgets": budgets,
            "buckets": requested_buckets,
            "candidate_output_dir": str(output_dir),
            "brotli_quality": int(args.brotli_quality),
        },
        "ranked_atom_counts": {bucket: len(atoms) for bucket, atoms in ranked.items()},
        "response_surface_objective": surface_objective,
        "summary": {
            "single_fixed_atom_count": len(rows),
            "advisory_measurement_count": len(advisory_by_key),
            "candidate_count": len(candidates),
            "fixed_length_candidate_count": len(fixed),
            "nonfixed_candidate_count": len(candidates) - len(fixed),
        },
        "waterbucket_candidates": candidates,
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "notes": "Multi-edit q candidates are exact-runtime compatible only when fixed_length_runtime_compatible=true; official inflate/advisory/exact eval still required.",
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-bin", type=Path, required=True)
    parser.add_argument("--feasibility", type=Path, required=True)
    parser.add_argument("--advisory-summary", type=Path)
    parser.add_argument("--decoder-q-response-surface", type=Path)
    parser.add_argument("--baseline-score", type=float)
    parser.add_argument("--expected-member-sha256")
    parser.add_argument("--edit-budgets", default="1,2,4,8,16")
    parser.add_argument(
        "--buckets",
        default="response_surface_guided,seg_heavy,sign_inverted_from_bad,mixed_tradeoff,bias_only,negative_control",
    )
    parser.add_argument("--brotli-quality", type=int, default=11)
    parser.add_argument("--candidate-output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_plan(args)
    _write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "candidate_count": payload["summary"]["candidate_count"],
                "fixed_length_candidate_count": payload["summary"]["fixed_length_candidate_count"],
                "nonfixed_candidate_count": payload["summary"]["nonfixed_candidate_count"],
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
