#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PR101/FEC6 CPU-axis sub-0.192 selector profiler.

This is a deterministic, no-dispatch analysis tool for the PR101/FEC6
near-miss archive.  It separates two different mechanisms:

* rate-only selector compression: decoded selector codes/components unchanged;
* component-moving selector changes: decoded selector codes can change, so the
  component evidence must be treated separately from byte savings.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

RATE_DENOMINATOR_BYTES = 37_545_489
DEFAULT_THRESHOLD = 0.192
DEFAULT_FEC6_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip"
)
DEFAULT_FEC6_MANIFEST = (
    REPO_ROOT
    / "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/packet_manifest.json"
)
DEFAULT_CPU_EVAL = (
    REPO_ROOT / "experiments/results/modal_auth_eval_cpu/archive_6bae0201fb08/contest_auth_eval.json"
)
DEFAULT_CUDA_EVAL = (
    REPO_ROOT / "experiments/results/modal_auth_eval/archive_6bae0201fb08/contest_auth_eval.json"
)
DEFAULT_CANDIDATE_GLOBS = (
    "experiments/results/pr101_frame_exploit_selector*/packet_manifest.json",
)


@dataclass(frozen=True)
class AxisReference:
    """Exact CPU-axis near-miss reference plus matching selector metadata."""

    threshold: float
    cpu_score: float
    cpu_score_axis: str
    cuda_score: float | None
    cuda_score_axis: str | None
    archive_bytes: int
    archive_sha256: str
    avg_segnet_dist: float
    avg_posenet_dist: float
    score_rate_contribution: float
    effective_selector_policy_sha256: str | None
    selector_payload_sha256: str | None
    selector_payload_bytes: int | None
    proxy_charged_score: float | None
    proxy_uncharged_score: float | None


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def score_after_byte_delta(score: float, byte_delta: int) -> float:
    """Return score after changing archive bytes while components stay fixed."""

    return float(score) + 25.0 * int(byte_delta) / RATE_DENOMINATOR_BYTES


def required_saving_bytes_for_strict_gate(score: float, threshold: float) -> int:
    """Smallest integer byte saving that makes ``score`` strictly below gate."""

    if float(score) < float(threshold):
        return 0
    raw = (float(score) - float(threshold)) * RATE_DENOMINATOR_BYTES / 25.0
    needed = max(0, math.floor(raw) + 1)
    while score_after_byte_delta(float(score), -needed) >= float(threshold):
        needed += 1
    return needed


def _score_axis(eval_json: dict[str, Any]) -> str:
    return str(eval_json.get("score_axis") or "unknown")


def _canonical_score(eval_json: dict[str, Any]) -> float:
    value = eval_json.get("canonical_score", eval_json.get("score_recomputed_from_components"))
    if value is None:
        raise ValueError("eval JSON lacks canonical score fields")
    return float(value)


def build_axis_reference(
    *,
    cpu_eval: Path,
    cuda_eval: Path | None,
    fec6_manifest: Path,
    threshold: float = DEFAULT_THRESHOLD,
) -> AxisReference:
    cpu = load_json(cpu_eval)
    manifest = load_json(fec6_manifest)
    cuda = load_json(cuda_eval) if cuda_eval is not None and Path(cuda_eval).is_file() else None
    archive = manifest["archive"]
    selector = manifest.get("selector", {})
    pack = archive.get("selector_pack_manifest", {})
    proxy = manifest.get("proxy", {})
    return AxisReference(
        threshold=float(threshold),
        cpu_score=_canonical_score(cpu),
        cpu_score_axis=_score_axis(cpu),
        cuda_score=_canonical_score(cuda) if cuda is not None else None,
        cuda_score_axis=_score_axis(cuda) if cuda is not None else None,
        archive_bytes=int(archive["bytes"]),
        archive_sha256=str(archive["sha256"]),
        avg_segnet_dist=float(cpu["avg_segnet_dist"]),
        avg_posenet_dist=float(cpu["avg_posenet_dist"]),
        score_rate_contribution=float(cpu["score_rate_contribution"]),
        effective_selector_policy_sha256=selector.get("effective_selector_policy_sha256"),
        selector_payload_sha256=pack.get("selector_payload_sha256"),
        selector_payload_bytes=(
            int(pack["selector_payload_bytes"]) if "selector_payload_bytes" in pack else None
        ),
        proxy_charged_score=(
            float(proxy["selector_score_proxy_charged_formula"])
            if "selector_score_proxy_charged_formula" in proxy
            else None
        ),
        proxy_uncharged_score=(
            float(proxy["selector_score_proxy_uncharged_formula"])
            if "selector_score_proxy_uncharged_formula" in proxy
            else None
        ),
    )


def _candidate_policy_kind(reference: AxisReference, manifest: dict[str, Any]) -> str:
    archive = manifest.get("archive", {})
    selector = manifest.get("selector", {})
    pack = archive.get("selector_pack_manifest", {})
    if str(archive.get("sha256")) == reference.archive_sha256:
        return "reference_or_identical_archive"
    effective = selector.get("effective_selector_policy_sha256")
    if effective and effective == reference.effective_selector_policy_sha256:
        return "rate_only_same_decoded_selector_policy"
    payload_sha = pack.get("selector_payload_sha256")
    if payload_sha and payload_sha == reference.selector_payload_sha256:
        return "rate_only_same_selector_payload"
    return "component_moving_selector_policy"


def classify_candidate_manifest(
    reference: AxisReference,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Classify one packet manifest against the exact PR101/FEC6 CPU gate."""

    archive = manifest.get("archive", {})
    selector = manifest.get("selector", {})
    pack = archive.get("selector_pack_manifest", {})
    proxy = manifest.get("proxy", {})
    wire_format = selector.get("wire_format") or pack.get("selector_wire_format")
    wire_format_text = str(wire_format or "")
    byte_closed_archive_charged = "archive_charged" in wire_format_text
    archive_bytes = int(archive["bytes"])
    archive_delta = archive_bytes - reference.archive_bytes
    rate_only_score = score_after_byte_delta(reference.cpu_score, archive_delta)
    allowance = reference.threshold - rate_only_score
    policy_kind = _candidate_policy_kind(reference, manifest)
    component_moving = policy_kind == "component_moving_selector_policy"
    proxy_charged_score = proxy.get("selector_score_proxy_charged_formula")
    proxy_uncharged_score = proxy.get("selector_score_proxy_uncharged_formula")
    proxy_net_delta_vs_fec6: float | None = None
    proxy_component_delta_vs_fec6: float | None = None
    proxy_estimated_exact_cpu_score: float | None = None
    if proxy_charged_score is not None and reference.proxy_charged_score is not None:
        proxy_net_delta_vs_fec6 = float(proxy_charged_score) - reference.proxy_charged_score
        proxy_estimated_exact_cpu_score = reference.cpu_score + proxy_net_delta_vs_fec6
    if proxy_uncharged_score is not None and reference.proxy_uncharged_score is not None:
        proxy_component_delta_vs_fec6 = (
            float(proxy_uncharged_score) - reference.proxy_uncharged_score
        )

    rate_only_passes = rate_only_score < reference.threshold
    proxy_allows_gate = (
        proxy_estimated_exact_cpu_score is not None
        and proxy_estimated_exact_cpu_score < reference.threshold
    )
    component_delta_within_allowance = (
        proxy_component_delta_vs_fec6 is not None and proxy_component_delta_vs_fec6 <= allowance
    )

    if str(archive.get("sha256")) == reference.archive_sha256:
        verdict = "reference"
    elif not component_moving and rate_only_passes:
        verdict = "rate_only_feasible_if_full_frame_parity_holds"
    elif not component_moving:
        verdict = "rate_only_not_enough_bytes"
    elif rate_only_passes and component_delta_within_allowance:
        verdict = "component_moving_rate_feasible_proxy_allows_gate"
    elif rate_only_passes:
        verdict = "component_moving_rate_feasible_proxy_blocks_gate"
    elif proxy_allows_gate:
        verdict = "component_moving_component_gain_proxy_allows_gate"
    else:
        verdict = "component_moving_no_gate_evidence"

    return {
        "path": repo_relative(manifest_path),
        "archive": {
            "bytes": archive_bytes,
            "sha256": archive.get("sha256"),
            "delta_bytes_vs_fec6": archive_delta,
            "saved_bytes_vs_fec6": -archive_delta,
        },
        "selector": {
            "policy_kind": policy_kind,
            "wire_format": wire_format,
            "byte_closed_archive_charged": byte_closed_archive_charged,
            "payload_bytes": pack.get("selector_payload_bytes"),
            "payload_sha256": pack.get("selector_payload_sha256"),
            "selected_non_none_pairs": selector.get("selected_non_none_pairs"),
            "histogram": selector.get("histogram"),
            "effective_selector_policy_sha256": selector.get("effective_selector_policy_sha256"),
        },
        "rate_only_gate": {
            "score_if_fec6_components_unchanged": rate_only_score,
            "passes_sub0192_if_components_unchanged": rate_only_passes,
            "allowable_component_score_delta_vs_fec6": allowance,
        },
        "component_moving_evidence": {
            "proxy_charged_score": proxy_charged_score,
            "proxy_uncharged_score": proxy_uncharged_score,
            "proxy_net_score_delta_vs_fec6": proxy_net_delta_vs_fec6,
            "proxy_component_delta_uncharged_vs_fec6": proxy_component_delta_vs_fec6,
            "proxy_estimated_exact_cpu_score_from_fec6_anchor": proxy_estimated_exact_cpu_score,
            "proxy_allows_sub0192_gate": proxy_allows_gate,
            "component_delta_within_rate_allowance": component_delta_within_allowance,
            "evidence_grade": proxy.get("evidence_grade"),
        },
        "verdict": verdict,
    }


def load_candidate_manifests(patterns: list[str]) -> list[tuple[Path, dict[str, Any]]]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(path) for path in glob.glob(str(REPO_ROOT / pattern)))
    unique = sorted({path.resolve() for path in paths})
    manifests: list[tuple[Path, dict[str, Any]]] = []
    for path in unique:
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        archive = payload.get("archive")
        selector = payload.get("selector")
        if not isinstance(archive, dict) or not isinstance(selector, dict):
            continue
        if "bytes" not in archive or "sha256" not in archive:
            continue
        manifests.append((path, payload))
    return manifests


def _load_tool_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def profile_rate_only_selector_compression(
    *,
    archive: Path,
    target_saving_bytes: int,
) -> dict[str, Any]:
    """Profile unchanged-selector compression opportunities."""

    fec6_tool = _load_tool_module(
        REPO_ROOT / "tools/pr101_fec6_wrapper_profile.py",
        "pr101_fec6_wrapper_profile_for_sub0192",
    )
    wrapper_profile = fec6_tool.profile_archive(archive, source_archive=None)
    selector = wrapper_profile["wrapper"]["selector_payload"]

    fec7_tool = _load_tool_module(
        REPO_ROOT / "tools/profile_pr101_fec7_selector_entropy.py",
        "profile_pr101_fec7_selector_entropy_for_sub0192",
    )
    fec7_profile = fec7_tool.profile_archive(
        archive,
        target_saving_bytes=int(target_saving_bytes),
    )
    best = fec7_profile["best_charged_candidate"]
    global_floor_saving = int(selector["payload_bytes"]) - int(selector["entropy_floor_bytes"])
    return {
        "fec6_selector": {
            "payload_bytes": int(selector["payload_bytes"]),
            "payload_sha256": selector["payload_sha256"],
            "entropy_floor_bytes": int(selector["entropy_floor_bytes"]),
            "global_entropy_floor_saving_bytes": global_floor_saving,
            "gap_to_entropy_floor_bytes": int(selector["gap_to_entropy_floor_bytes"]),
            "histogram": selector["code_histogram"],
        },
        "fec7_best_charged_candidate": best,
        "target_saving_bytes": int(target_saving_bytes),
        "can_rate_only_compression_meet_target": bool(
            max(global_floor_saving, int(best["saving_vs_fec6_selector_bytes"]))
            >= int(target_saving_bytes)
        ),
        "blocker": {
            "blocked": bool(
                max(global_floor_saving, int(best["saving_vs_fec6_selector_bytes"]))
                < int(target_saving_bytes)
            ),
            "reason": (
                "Decoded-selector-preserving compression cannot save the required bytes: "
                "FEC6 is within a few bytes of its global entropy floor, and charged FEC7 "
                "range/adaptive candidates regress after model bytes are counted."
            ),
        },
    }


def build_profile(
    *,
    fec6_archive: Path = DEFAULT_FEC6_ARCHIVE,
    fec6_manifest: Path = DEFAULT_FEC6_MANIFEST,
    cpu_eval: Path = DEFAULT_CPU_EVAL,
    cuda_eval: Path | None = DEFAULT_CUDA_EVAL,
    candidate_globs: list[str] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    reference = build_axis_reference(
        cpu_eval=cpu_eval,
        cuda_eval=cuda_eval,
        fec6_manifest=fec6_manifest,
        threshold=threshold,
    )
    required_saving = required_saving_bytes_for_strict_gate(
        reference.cpu_score, reference.threshold
    )
    rate_only = profile_rate_only_selector_compression(
        archive=fec6_archive,
        target_saving_bytes=required_saving,
    )
    patterns = list(candidate_globs or DEFAULT_CANDIDATE_GLOBS)
    candidate_rows = [
        classify_candidate_manifest(reference, path, manifest)
        for path, manifest in load_candidate_manifests(patterns)
    ]
    candidate_rows.sort(
        key=lambda row: (
            row["verdict"] != "component_moving_rate_feasible_proxy_allows_gate",
            not row["rate_only_gate"]["passes_sub0192_if_components_unchanged"],
            not row["selector"].get("byte_closed_archive_charged", False),
            row["archive"]["bytes"],
            row["path"],
        )
    )
    rate_feasible_component_moving = [
        row
        for row in candidate_rows
        if row["selector"]["policy_kind"] == "component_moving_selector_policy"
        and row["rate_only_gate"]["passes_sub0192_if_components_unchanged"]
        and row["selector"].get("byte_closed_archive_charged", False)
    ]
    rate_feasible_component_moving_non_byte_closed = [
        row
        for row in candidate_rows
        if row["selector"]["policy_kind"] == "component_moving_selector_policy"
        and row["rate_only_gate"]["passes_sub0192_if_components_unchanged"]
        and not row["selector"].get("byte_closed_archive_charged", False)
    ]
    proxy_allowed = [
        row
        for row in rate_feasible_component_moving
        if row["component_moving_evidence"]["proxy_allows_sub0192_gate"]
    ]
    best_rate_feasible = min(
        rate_feasible_component_moving,
        key=lambda row: (
            row["component_moving_evidence"]["proxy_estimated_exact_cpu_score_from_fec6_anchor"]
            if row["component_moving_evidence"]["proxy_estimated_exact_cpu_score_from_fec6_anchor"]
            is not None
            else float("inf"),
            row["archive"]["bytes"],
        ),
        default=None,
    )
    exact_feasible_rate_only = [
        row
        for row in candidate_rows
        if row["selector"]["policy_kind"].startswith("rate_only")
        and row["rate_only_gate"]["passes_sub0192_if_components_unchanged"]
        and row["selector"].get("byte_closed_archive_charged", False)
    ]
    if exact_feasible_rate_only:
        conclusion = "feasible_rate_only_candidate_path"
    elif proxy_allowed:
        conclusion = "feasible_component_moving_candidate_path_needs_exact_cpu_eval"
    elif best_rate_feasible is not None:
        conclusion = "hard_blocker_under_current_component_rows"
    else:
        conclusion = "hard_blocker_no_rate_feasible_selector_candidate"

    return {
        "schema": "pr101_fec6_sub0192_cpu_component_selector_profile.v1",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "threshold": reference.threshold,
        "reference": {
            "archive_bytes": reference.archive_bytes,
            "archive_sha256": reference.archive_sha256,
            "cpu_score_axis": reference.cpu_score_axis,
            "cpu_score": reference.cpu_score,
            "cuda_score_axis": reference.cuda_score_axis,
            "cuda_score": reference.cuda_score,
            "avg_segnet_dist": reference.avg_segnet_dist,
            "avg_posenet_dist": reference.avg_posenet_dist,
            "score_rate_contribution": reference.score_rate_contribution,
            "effective_selector_policy_sha256": reference.effective_selector_policy_sha256,
            "selector_payload_sha256": reference.selector_payload_sha256,
            "selector_payload_bytes": reference.selector_payload_bytes,
        },
        "strict_rate_target": {
            "required_saving_bytes_at_unchanged_components": required_saving,
            "archive_bytes_limit_at_unchanged_components": reference.archive_bytes
            - required_saving,
            "score_after_required_saving": score_after_byte_delta(
                reference.cpu_score, -required_saving
            ),
            "score_after_one_byte_less_saving": score_after_byte_delta(
                reference.cpu_score, -(required_saving - 1)
            )
            if required_saving > 0
            else reference.cpu_score,
        },
        "rate_only_selector_compression": rate_only,
        "candidate_count": len(candidate_rows),
        "candidates": candidate_rows,
        "best_rate_feasible_component_moving_candidate": best_rate_feasible,
        "rate_feasible_component_moving_non_byte_closed_candidates": (
            rate_feasible_component_moving_non_byte_closed
        ),
        "exact_feasible_rate_only_candidates": exact_feasible_rate_only,
        "proxy_allowed_component_moving_candidates": proxy_allowed,
        "conclusion": {
            "verdict": conclusion,
            "found_feasible_sub0192_candidate_path": bool(
                exact_feasible_rate_only or proxy_allowed
            ),
            "hard_blocker": not bool(exact_feasible_rate_only or proxy_allowed),
            "reason": _conclusion_reason(conclusion, rate_only, best_rate_feasible),
        },
    }


def _conclusion_reason(
    conclusion: str,
    rate_only: dict[str, Any],
    best_rate_feasible: dict[str, Any] | None,
) -> str:
    if conclusion == "feasible_rate_only_candidate_path":
        return "At least one same-selector packet clears the CPU rate gate."
    if conclusion == "feasible_component_moving_candidate_path_needs_exact_cpu_eval":
        return (
            "At least one component-moving packet clears the CPU gate under the "
            "current proxy delta model; exact CPU eval is still required."
        )
    if best_rate_feasible is None:
        return (
            "No scanned selector packet saves enough bytes at unchanged components, "
            "and rate-only selector compression is blocked."
        )
    allowance = best_rate_feasible["rate_only_gate"]["allowable_component_score_delta_vs_fec6"]
    proxy_delta = best_rate_feasible["component_moving_evidence"][
        "proxy_component_delta_uncharged_vs_fec6"
    ]
    return (
        "Rate-only selector compression is blocked; the smallest rate-feasible "
        "byte-closed component-moving packet changes selector codes and its proxy component "
        f"delta ({proxy_delta}) exceeds the strict CPU gate allowance ({allowance})."
    )


def render_markdown(profile: dict[str, Any]) -> str:
    ref = profile["reference"]
    target = profile["strict_rate_target"]
    rate_only = profile["rate_only_selector_compression"]
    best = profile["best_rate_feasible_component_moving_candidate"]
    lines = [
        "# PR101/FEC6 CPU Component Selector Sub-0.192 Profile",
        "",
        "- score_claim: `false`",
        "- dispatch_attempted: `false`",
        "- ready_for_exact_eval_dispatch: `false`",
        f"- reference archive: `{ref['archive_sha256']}`",
        f"- reference bytes: `{ref['archive_bytes']}`",
        f"- exact CPU score: `{ref['cpu_score']}`",
        f"- exact CUDA score: `{ref['cuda_score']}`",
        f"- strict CPU byte target at unchanged components: `{target['required_saving_bytes_at_unchanged_components']}`",
        f"- unchanged-component archive byte limit: `{target['archive_bytes_limit_at_unchanged_components']}`",
        "",
        "## Rate-Only Selector Compression",
        "",
        f"- FEC6 selector payload bytes: `{rate_only['fec6_selector']['payload_bytes']}`",
        f"- FEC6 global entropy floor bytes: `{rate_only['fec6_selector']['entropy_floor_bytes']}`",
        f"- best charged FEC7 saving vs FEC6: `{rate_only['fec7_best_charged_candidate']['saving_vs_fec6_selector_bytes']}`",
        f"- blocked: `{str(rate_only['blocker']['blocked']).lower()}`",
        "",
        "## Component-Moving Selector Scan",
        "",
        "| packet | bytes | saved vs FEC6 | byte-closed | kind | rate-only CPU score | proxy-est CPU score | verdict |",
        "|---|---:|---:|---|---|---:|---:|---|",
    ]
    for row in profile["candidates"][:12]:
        proxy_score = row["component_moving_evidence"][
            "proxy_estimated_exact_cpu_score_from_fec6_anchor"
        ]
        lines.append(
            "| {path} | {bytes} | {saved} | {byte_closed} | {kind} | {rate:.12f} | {proxy} | {verdict} |".format(
                path=row["path"],
                bytes=row["archive"]["bytes"],
                saved=row["archive"]["saved_bytes_vs_fec6"],
                byte_closed=str(row["selector"].get("byte_closed_archive_charged", False)).lower(),
                kind=row["selector"]["policy_kind"],
                rate=row["rate_only_gate"]["score_if_fec6_components_unchanged"],
                proxy=f"{proxy_score:.12f}" if proxy_score is not None else "n/a",
                verdict=row["verdict"],
            )
        )
    lines.extend(
        [
            "",
            "## Best Rate-Feasible Component-Moving Row",
            "",
        ]
    )
    if best is None:
        lines.append("- none")
    else:
        lines.extend(
            [
                f"- packet: `{best['path']}`",
                f"- bytes: `{best['archive']['bytes']}`",
                f"- saved vs FEC6: `{best['archive']['saved_bytes_vs_fec6']}`",
                f"- rate-only CPU score if components unchanged: `{best['rate_only_gate']['score_if_fec6_components_unchanged']}`",
                f"- allowable component delta vs FEC6: `{best['rate_only_gate']['allowable_component_score_delta_vs_fec6']}`",
                f"- proxy component delta vs FEC6: `{best['component_moving_evidence']['proxy_component_delta_uncharged_vs_fec6']}`",
                f"- verdict: `{best['verdict']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- verdict: `{profile['conclusion']['verdict']}`",
            f"- found_feasible_sub0192_candidate_path: `{str(profile['conclusion']['found_feasible_sub0192_candidate_path']).lower()}`",
            f"- hard_blocker: `{str(profile['conclusion']['hard_blocker']).lower()}`",
            f"- reason: {profile['conclusion']['reason']}",
            "",
            "## 6-Hook Wire-In",
            "",
            "1. Sensitivity-map contribution: `N/A - no new empirical score anchor; selector rows preserve score_claim=false`.",
            "2. Pareto constraint: `CPU-axis gate only; CUDA axis recorded separately and remains non-promotional`.",
            "3. Bit-allocator hook: strict byte target and per-candidate archive deltas are machine-readable.",
            "4. Cathedral autopilot dispatch hook: `ready_for_exact_eval_dispatch=false`; no dispatch row emitted.",
            "5. Continual-learning posterior update: blocker/candidate classification is recorded here; no score posterior update.",
            "6. Probe-disambiguator: rate-only and component-moving selector mechanisms are reported as separate verdict spaces.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fec6-archive", type=Path, default=DEFAULT_FEC6_ARCHIVE)
    parser.add_argument("--fec6-manifest", type=Path, default=DEFAULT_FEC6_MANIFEST)
    parser.add_argument("--cpu-eval", type=Path, default=DEFAULT_CPU_EVAL)
    parser.add_argument("--cuda-eval", type=Path, default=DEFAULT_CUDA_EVAL)
    parser.add_argument(
        "--candidate-manifest-glob",
        action="append",
        default=[],
        help="Repo-relative glob for PR101 selector packet manifests.",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    patterns = args.candidate_manifest_glob or list(DEFAULT_CANDIDATE_GLOBS)
    profile = build_profile(
        fec6_archive=args.fec6_archive,
        fec6_manifest=args.fec6_manifest,
        cpu_eval=args.cpu_eval,
        cuda_eval=args.cuda_eval,
        candidate_globs=patterns,
        threshold=float(args.threshold),
    )
    if args.json_out:
        write_json(args.json_out, profile)
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(profile) + "\n", encoding="utf-8")
    if not args.json_out and not args.md_out:
        json.dump(profile, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
