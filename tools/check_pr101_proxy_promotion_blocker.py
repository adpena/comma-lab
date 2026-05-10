#!/usr/bin/env python3
"""Fail closed when a PR101 proxy packet is mistaken for promotion evidence.

This checker is deliberately local-only. It joins the PR101 proxy runtime
packet manifest, the local runtime-consumption proof, the canonical A1 exact
CUDA anchor, and the inflate op-cost xray. It writes a machine-readable
checklist and exits nonzero unless the packet has real promotion evidence.

No scorers, GPU jobs, remote jobs, or preflight surfaces are invoked.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.continual_learning import ContestResult  # noqa: E402
from tac.repo_io import json_text, read_json, repo_relative, sha256_file, write_json  # noqa: E402

TOOL_NAME = "tools/check_pr101_proxy_promotion_blocker.py"
CHECKLIST_SCHEMA = "pr101_a1_proxy_promotion_blocker_checklist_v1"
MANIFEST_SCHEMA = "pr101_kaggle_proxy_runtime_packet_v1"
PROOF_SCHEMA = "pr101_kaggle_proxy_runtime_consumption_proof_v1"
DEFAULT_MANIFEST = Path(
    "experiments/results/kaggle_pr101_proxy_sweep_20260510_codex/"
    "pr101_proxy_sweep/proxy_runtime_packet/runtime_packet_manifest.json"
)
DEFAULT_PROOF = Path(
    "experiments/results/kaggle_pr101_proxy_sweep_20260510_codex/"
    "pr101_proxy_sweep/proxy_runtime_packet/runtime_consumption_proof.json"
)
DEFAULT_A1_AUTH_EVAL = Path(
    "experiments/results/track1_phase_a1_score_gradient_bestproxy_lr2e6_20260509_codex/"
    "harvested_artifacts/eval_work/contest_auth_eval.json"
)
DEFAULT_XRAY = Path("experiments/results/xray_inflate_op_cost_profiler_20260509T104122Z/op_catalog.json")
FALSE_AUTHORITY_FIELDS = (
    "score_claim",
    "score_claim_valid",
    "ready_for_exact_eval_dispatch",
    "dispatch_attempted",
    "exact_auth_eval_performed",
    "contest_cuda_auth_eval",
)
REMOVED_UNSUPPORTED_BLOCKERS = (
    "unsupported_proxy_params_not_runtime_consumed",
    "delta_scale_not_runtime_consumed",
    "latent_delta_scale_not_runtime_consumed",
    "smooth_weight_not_runtime_consumed",
)


class PromotionBlockerError(ValueError):
    """Raised when the promotion-blocker inputs are malformed."""


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_rel(path: Path) -> str:
    return repo_relative(_repo_path(path), REPO_ROOT)


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PromotionBlockerError(f"{field} must be a JSON object")
    return value


def _load_mapping(path: Path, field: str) -> Mapping[str, Any]:
    path = _repo_path(path)
    if not path.is_file():
        raise PromotionBlockerError(f"{field} missing: {path}")
    return _require_mapping(read_json(path), field)


def _sha(value: Any) -> str | None:
    return value if isinstance(value, str) and len(value) == 64 else None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _check(check_id: str, passed: bool, evidence: str, blocker: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": check_id,
        "passed": bool(passed),
        "evidence": evidence,
    }
    if not passed and blocker:
        row["blocker"] = blocker
    return row


def _authority_checks(manifest: Mapping[str, Any], proof: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for field in FALSE_AUTHORITY_FIELDS:
        manifest_value = manifest.get(field)
        proof_value = proof.get(field)
        checks.append(
            _check(
                f"manifest_false_authority.{field}",
                manifest_value is False,
                f"manifest {field}={manifest_value!r}",
                f"proxy_manifest_authority_flag_true:{field}",
            )
        )
        if field in proof:
            checks.append(
                _check(
                    f"proof_false_authority.{field}",
                    proof_value is False,
                    f"proof {field}={proof_value!r}",
                    f"proxy_proof_authority_flag_true:{field}",
                )
            )
    return checks


def _a1_anchor_check(a1_auth_eval: Mapping[str, Any], a1_path: Path) -> dict[str, Any]:
    score = _float_or_none(a1_auth_eval.get("score_recomputed_from_components"))
    archive_size = a1_auth_eval.get("archive_size_bytes")
    provenance = _require_mapping(a1_auth_eval.get("provenance"), "a1.provenance")
    archive_sha = _sha(provenance.get("archive_sha256"))
    hardware_substrate = (
        "linux_x86_64_t4"
        if provenance.get("device") == "cuda" and provenance.get("gpu_t4_match") is True
        else "unknown"
    )
    custody_ok = False
    custody_reason = "missing_score_or_archive_custody"
    if score is not None and archive_sha is not None and isinstance(archive_size, int):
        result = ContestResult(
            axis="cuda",
            hardware_substrate=hardware_substrate,
            architecture_class="a1_pr101_exact_cuda_anchor",
            score_value=score,
            evidence_tag=str(a1_auth_eval.get("lane_tag") or ""),
            archive_sha256=archive_sha,
            archive_bytes=archive_size,
        )
        custody_ok, custody_reason = result.validate_custody()
    passed = (
        a1_auth_eval.get("evidence_grade") == "A++"
        and a1_auth_eval.get("score_axis") == "contest_cuda"
        and a1_auth_eval.get("score_claim_valid") is True
        and a1_auth_eval.get("promotion_eligible") is True
        and a1_auth_eval.get("n_samples") == 600
        and score is not None
        and archive_sha is not None
        and custody_ok
    )
    return _check(
        "a1_exact_cuda_anchor_available",
        passed,
        (
            f"{_repo_rel(a1_path)} score={score!r} archive_sha256={archive_sha!r} "
            f"custody_ok={custody_ok!r} custody_reason={custody_reason!r}"
        ),
        "a1_exact_cuda_anchor_missing_or_invalid",
    )


def _archive_custody_check(manifest: Mapping[str, Any], proof: Mapping[str, Any]) -> dict[str, Any]:
    source = _require_mapping(manifest.get("source_archive"), "manifest.source_archive")
    packet = _require_mapping(manifest.get("packet_archive"), "manifest.packet_archive")
    proof_archive = _require_mapping(proof.get("archive_unchanged_proof"), "proof.archive_unchanged_proof")
    source_sha = _sha(source.get("sha256"))
    packet_sha = _sha(packet.get("sha256"))
    unchanged_sha = _sha(manifest.get("archive_unchanged_sha256"))
    proof_sha = _sha(proof_archive.get("archive_sha256"))
    passed = source_sha is not None and source_sha == packet_sha == unchanged_sha == proof_sha
    return _check(
        "proxy_archive_custody_consistent",
        passed,
        f"source={source_sha!r} packet={packet_sha!r} unchanged={unchanged_sha!r} proof={proof_sha!r}",
        "proxy_archive_custody_mismatch",
    )


def _xray_check(xray: Mapping[str, Any], xray_path: Path) -> dict[str, Any]:
    files = xray.get("files")
    if not isinstance(files, list):
        return _check("xray_op_cost_catalog_available", False, f"{_repo_rel(xray_path)} missing files[]", "xray_op_cost_catalog_invalid")
    mutation_count = 0
    labels: list[str] = []
    for raw in files:
        if not isinstance(raw, Mapping):
            continue
        labels.append(str(raw.get("label", "")))
        count = raw.get("per_channel_mutation_count")
        if isinstance(count, int):
            mutation_count += count
        else:
            mutations = raw.get("per_channel_mutations")
            if isinstance(mutations, list):
                mutation_count += len(mutations)
    return _check(
        "xray_op_cost_catalog_available",
        mutation_count >= 3,
        f"{_repo_rel(xray_path)} labels={labels} per_channel_mutations={mutation_count}",
        "xray_op_cost_catalog_missing_bias_slots",
    )


def _proxy_contract_checks(manifest: Mapping[str, Any], proof: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    runtime_patch = _require_mapping(manifest.get("runtime_patch"), "manifest.runtime_patch")
    patch_rows = runtime_patch.get("runtime_consumed_params")
    patch_params = sorted(row.get("param") for row in patch_rows if isinstance(row, Mapping)) if isinstance(patch_rows, list) else []
    checks.append(
        _check(
            "supported_bias_static_patch_present",
            patch_params == ["bias_b", "bias_g", "bias_r"],
            f"runtime patch params={patch_params}",
            "supported_bias_static_patch_missing",
        )
    )
    checks.append(
        _check(
            "inflate_wrapper_routes_to_packet_inflate_py",
            proof.get("inflate_sh_routes_to_packet_inflate_py") is True,
            f"inflate_sh_routes_to_packet_inflate_py={proof.get('inflate_sh_routes_to_packet_inflate_py')!r}",
            "inflate_wrapper_route_not_proven",
        )
    )
    checks.append(
        _check(
            "full_runtime_consumption_proven",
            proof.get("runtime_consumption_proven_for_supported_bias_params") is True,
            f"runtime_consumption_proven_for_supported_bias_params={proof.get('runtime_consumption_proven_for_supported_bias_params')!r}",
            "full_runtime_consumption_not_proven",
        )
    )
    checks.append(
        _check(
            "candidate_contest_cuda_auth_eval_present",
            manifest.get("contest_cuda_auth_eval") is True
            and manifest.get("exact_auth_eval_performed") is True
            and manifest.get("score_claim_valid") is True,
            "candidate packet has no exact CUDA auth eval fields set true",
            "no_candidate_contest_cuda_auth_eval",
        )
    )

    blockers = manifest.get("blockers")
    blockers_list = blockers if isinstance(blockers, list) else []
    stale_blockers = sorted(blocker for blocker in REMOVED_UNSUPPORTED_BLOCKERS if blocker in blockers_list)
    stale_unsupported_params = "unsupported_params" in manifest or "unsupported_params_blocker_proof" in proof
    checks.append(
        _check(
            "legacy_proxy_params_not_used_as_promotion_blockers",
            not stale_blockers and not stale_unsupported_params,
            f"stale_blockers={stale_blockers} stale_unsupported_params={stale_unsupported_params}",
            "stale_unsupported_proxy_contract",
        )
    )
    return checks


def build_promotion_blocker_checklist(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    proof_path: Path = DEFAULT_PROOF,
    a1_auth_eval_path: Path = DEFAULT_A1_AUTH_EVAL,
    xray_path: Path = DEFAULT_XRAY,
) -> dict[str, Any]:
    manifest_path = _repo_path(manifest_path)
    proof_path = _repo_path(proof_path)
    a1_auth_eval_path = _repo_path(a1_auth_eval_path)
    xray_path = _repo_path(xray_path)

    manifest = _load_mapping(manifest_path, "manifest")
    proof = _load_mapping(proof_path, "proof")
    a1_auth_eval = _load_mapping(a1_auth_eval_path, "a1_auth_eval")
    xray = _load_mapping(xray_path, "xray")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise PromotionBlockerError(f"manifest schema must be {MANIFEST_SCHEMA!r}")
    if proof.get("schema") != PROOF_SCHEMA:
        raise PromotionBlockerError(f"proof schema must be {PROOF_SCHEMA!r}")

    checks: list[dict[str, Any]] = []
    checks.extend(_authority_checks(manifest, proof))
    checks.append(_archive_custody_check(manifest, proof))
    checks.append(_a1_anchor_check(a1_auth_eval, a1_auth_eval_path))
    checks.append(_xray_check(xray, xray_path))
    checks.extend(_proxy_contract_checks(manifest, proof))

    blockers = [row["blocker"] for row in checks if not row.get("passed") and "blocker" in row]
    promotable = not blockers
    verdict = "PROMOTABLE_EXACT_RUNTIME_PACKET" if promotable else "BLOCKED_PROXY_ONLY_NOT_PROMOTABLE"
    checklist: dict[str, Any] = {
        "schema": CHECKLIST_SCHEMA,
        "tool": TOOL_NAME,
        "verdict": verdict,
        "promotable": promotable,
        "candidate_id": manifest.get("candidate_id", proof.get("candidate_id", "")),
        "score_claim": False,
        "dispatch_attempted": False,
        "manifest_path": _repo_rel(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "proof_path": _repo_rel(proof_path),
        "proof_sha256": sha256_file(proof_path),
        "a1_auth_eval_path": _repo_rel(a1_auth_eval_path),
        "xray_path": _repo_rel(xray_path),
        "checks": checks,
        "blockers": blockers,
        "next_exact_eval_candidate": (
            "none_from_this_proxy_packet"
            if not promotable
            else "candidate_requires_claimed_exact_cuda_eval_before_score_claim"
        ),
        "blocker_rationale": (
            "A static bias patch plus wrapper-route proof is useful runtime evidence, "
            "but it is not full inflate execution and not a contest-CUDA auth eval."
        ),
    }
    return checklist


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--proof", type=Path, default=DEFAULT_PROOF)
    parser.add_argument("--a1-auth-eval", type=Path, default=DEFAULT_A1_AUTH_EVAL)
    parser.add_argument("--xray-op-catalog", type=Path, default=DEFAULT_XRAY)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="return success after writing a blocked checklist; default exits 1 when blocked",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        checklist = build_promotion_blocker_checklist(
            manifest_path=args.manifest,
            proof_path=args.proof,
            a1_auth_eval_path=args.a1_auth_eval,
            xray_path=args.xray_op_catalog,
        )
    except Exception as exc:
        print(json_text({"schema": CHECKLIST_SCHEMA, "tool": TOOL_NAME, "error": str(exc)}), file=sys.stderr, end="")
        return 2
    if args.output is not None:
        write_json(_repo_path(args.output), checklist)
    print(json_text({
        "schema": CHECKLIST_SCHEMA,
        "verdict": checklist["verdict"],
        "promotable": checklist["promotable"],
        "candidate_id": checklist["candidate_id"],
        "blockers": checklist["blockers"],
        "output": _repo_rel(args.output) if args.output is not None else None,
    }), end="")
    if checklist["promotable"] or args.allow_blocked:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
