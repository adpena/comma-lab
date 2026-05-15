#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run Tier C curves against the real SegNet+PoseNet scorer on CPU.

This is the $0 GPU follow-up to the PR106/DP1 Tier C extension. It compares
A1, PR106, IBPS1, and DP1 by loading the official scorer once, rendering the
same pair indices for every archive, and feeding each archive through the
existing ``tools/mdl_scorer_conditional_ablation.py`` Tier C implementation.

Default mode is plan-only and writes a fail-closed manifest. Use ``--execute``
for the long CPU run. Outputs are planning evidence only: no contest score,
no promotion, no dispatch readiness.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import platform
import random
import struct
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
N_PAIRS = 600
LANE_ID = "lane_tier_c_real_scorer_fourway_20260514"


@dataclass(frozen=True)
class CandidateSpec:
    """Archive/grammar pair for a Tier C real-scorer run."""

    name: str
    archive_path: Path
    grammar: str
    role: str


DEFAULT_CANDIDATES: tuple[CandidateSpec, ...] = (
    CandidateSpec(
        name="a1",
        archive_path=REPO_ROOT / "submissions" / "a1" / "archive.zip",
        grammar="a1",
        role="within_hnerv_control",
    ),
    CandidateSpec(
        name="pr106_r2",
        archive_path=REPO_ROOT
        / "submissions"
        / "pr106_latent_sidecar_r2"
        / "archive.zip",
        grammar="pr106_latent_sidecar",
        role="public_pr106_control",
    ),
    CandidateSpec(
        name="ibps1_c6_5ep",
        archive_path=REPO_ROOT
        / "experiments"
        / "results"
        / "lane_substrate_c6_e4_mdl_ibps_modal_t4_dispatch_20260514T144949Z__smoke__5ep_modal"
        / "harvested_artifacts"
        / "lane_substrate_c6_e4_mdl_ibps_results"
        / "output"
        / "archive.zip",
        grammar="ibps1",
        role="ib_bottleneck_control_5ep",
    ),
    CandidateSpec(
        name="dp1_smoke",
        archive_path=REPO_ROOT
        / "experiments"
        / "results"
        / "dp1_smoke_v2_hardening"
        / "archive.zip",
        grammar="dp1",
        role="frozen_prior_smoke_control",
    ),
)


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hardware_axis_label() -> str:
    if platform.system() == "Darwin":
        return "[macOS-CPU advisory only]"
    if platform.system() == "Linux" and platform.machine().lower() in {
        "x86_64",
        "amd64",
    }:
        return "[contest-CPU Linux x86_64 pair-sampled]"
    return "[CPU pair-sampled advisory]"


def _load_mdl_module() -> Any:
    script = REPO_ROOT / "tools" / "mdl_scorer_conditional_ablation.py"
    name = "_tier_c_mdl_real_scorer"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {script}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_real_scorer(mdl: Any, upstream_dir: Path) -> Any:
    """Load upstream.modules.DistortionNet once on CPU."""

    if str(upstream_dir) not in sys.path:
        sys.path.insert(0, str(upstream_dir))
    torch = mdl.torch
    if torch is None:
        raise RuntimeError("torch is required for real-scorer Tier C execution")
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # type: ignore

    device = torch.device("cpu")
    distortion_net = DistortionNet().eval().to(device)
    distortion_net.load_state_dicts(str(posenet_sd_path), str(segnet_sd_path), device)
    return distortion_net


def _parse_archive_spec(raw: str, repo_root: Path) -> CandidateSpec:
    """Parse ``NAME=PATH,grammar=GRAMMAR[,role=ROLE]``."""

    if "=" not in raw:
        raise ValueError("--archive must use NAME=PATH,grammar=GRAMMAR syntax")
    name, rest = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("--archive name must not be empty")
    pieces = [piece.strip() for piece in rest.split(",") if piece.strip()]
    if not pieces:
        raise ValueError(f"--archive {raw!r} has no path")
    path = Path(pieces[0])
    grammar = ""
    role = "operator_supplied"
    for piece in pieces[1:]:
        key, sep, value = piece.partition("=")
        if not sep:
            raise ValueError(f"archive option {piece!r} must use key=value")
        key = key.strip()
        value = value.strip()
        if key == "grammar":
            grammar = value
        elif key == "role":
            role = value
        else:
            raise ValueError(f"unknown archive option key {key!r}")
    if not grammar:
        raise ValueError(f"--archive {name} missing grammar=...")
    if not path.is_absolute():
        path = repo_root / path
    return CandidateSpec(name=name, archive_path=path, grammar=grammar, role=role)


def _load_pair_indices_file(path: Path) -> list[int]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"pair indices file is empty: {path}")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = [int(tok, 0) for tok in raw.replace(",", " ").split()]
    if isinstance(parsed, int):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError("pair indices file must contain a JSON list or token stream")
    indices = sorted(dict.fromkeys(int(v) for v in parsed))
    bad = [v for v in indices if v < 0 or v >= N_PAIRS]
    if bad:
        raise ValueError(f"pair indices out of range 0..{N_PAIRS - 1}: {bad}")
    if not indices:
        raise ValueError("pair indices file produced no indices")
    return indices


def _normalize_grammar_light(grammar: str) -> str:
    key = grammar.strip().lower()
    if key in {"pretrained_driving_prior", "driving_prior", "dp1_driving_prior"}:
        return "dp1"
    if key in {"ibps1_mdl_ibps", "c6_e4_mdl_ibps", "mdl_ibps"}:
        return "ibps1"
    if key in {"pr106", "pr106_latent_sidecar_r2"}:
        return "pr106_latent_sidecar"
    if key == "pr101":
        return "a1"
    return key


def _read_inner_member_for_capacity(path: Path, grammar: str) -> bytes:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if grammar in {"dp1", "ibps1"} and "0.bin" in names:
            return zf.read("0.bin")
        if "x" in names:
            return zf.read("x")
        if "0.bin" in names:
            return zf.read("0.bin")
        return zf.read(names[0])


def _candidate_pair_capacity(spec: CandidateSpec) -> int:
    """Best-effort pair capacity without importing the heavy scorer stack."""

    if not spec.archive_path.is_file():
        return N_PAIRS
    grammar = _normalize_grammar_light(spec.grammar)
    try:
        inner = _read_inner_member_for_capacity(spec.archive_path, grammar)
        if grammar == "dp1":
            from tac.substrates.pretrained_driving_prior.archive import (
                DP1_HEADER_FMT,
                DP1_HEADER_SIZE,
                DP1_MAGIC,
            )

            if len(inner) < DP1_HEADER_SIZE:
                return N_PAIRS
            magic, _ver, num_pairs, *_rest = struct.unpack_from(
                DP1_HEADER_FMT, inner, 0
            )
            if magic == DP1_MAGIC:
                return max(1, min(N_PAIRS, int(num_pairs)))
        if grammar == "ibps1" and len(inner) >= 25:
            magic, _ver, _latent_dim, num_pairs, *_rest = struct.unpack_from(
                "<4sBHHIIII", inner, 0
            )
            if magic == b"IBPS":
                return max(1, min(N_PAIRS, int(num_pairs)))
    except Exception:
        return N_PAIRS
    return N_PAIRS


def _pair_universe_for_candidates(candidates: list[CandidateSpec]) -> int:
    capacities = [_candidate_pair_capacity(spec) for spec in candidates]
    return min(capacities) if capacities else N_PAIRS


def _resolve_pair_indices(
    args: argparse.Namespace,
    candidates: list[CandidateSpec],
) -> list[int]:
    pair_universe = _pair_universe_for_candidates(candidates)
    if args.pair_indices_file is not None:
        indices = _load_pair_indices_file(args.pair_indices_file)
        bad = [v for v in indices if v >= pair_universe]
        if bad:
            raise ValueError(
                "pair indices exceed selected archive pair capacity "
                f"{pair_universe}: {bad}"
            )
        return indices
    rng = random.Random(args.seed)
    n = min(args.pair_samples, pair_universe)
    return sorted(rng.sample(range(pair_universe), n))


def _scorer_assets(upstream_dir: Path) -> dict[str, dict[str, Any]]:
    paths = {
        "upstream_modules": upstream_dir / "modules.py",
        "posenet_weights": upstream_dir / "models" / "posenet.safetensors",
        "segnet_weights": upstream_dir / "models" / "segnet.safetensors",
        "contest_video": upstream_dir / "videos" / "0.mkv",
    }
    return {
        key: {
            "path": str(path),
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": _sha256_file(path),
        }
        for key, path in paths.items()
    }


def _candidate_status(spec: CandidateSpec) -> dict[str, Any]:
    exists = spec.archive_path.is_file()
    return {
        "name": spec.name,
        "grammar": spec.grammar,
        "role": spec.role,
        "archive_path": str(spec.archive_path),
        "archive_exists": exists,
        "archive_bytes": spec.archive_path.stat().st_size if exists else None,
        "archive_sha256": _sha256_file(spec.archive_path),
        "pair_capacity": _candidate_pair_capacity(spec) if exists else None,
    }


def _build_manifest_base(
    *,
    args: argparse.Namespace,
    candidates: list[CandidateSpec],
    pair_indices: list[int],
    mode: str,
) -> dict[str, Any]:
    assets = _scorer_assets(args.upstream_dir)
    candidate_status = [_candidate_status(spec) for spec in candidates]
    blockers: list[str] = []
    if any(not item["exists"] for item in assets.values()):
        blockers.append("missing_real_scorer_assets")
    missing_candidates = [
        item["name"] for item in candidate_status if not item["archive_exists"]
    ]
    if missing_candidates:
        blockers.append("missing_candidate_archives:" + ",".join(missing_candidates))

    return {
        "schema": "tac_tier_c_real_scorer_runner_v1",
        "schema_version": 1,
        "tool": "tools/run_tier_c_with_real_scorer.py",
        "lane_id": LANE_ID,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": mode,
        "execute_requested": bool(args.execute),
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "dispatch_attempted": False,
        "gpu_required": False,
        "device": "cpu",
        "evidence_axis": "[real-scorer CPU Tier-C delta curves; pair-sampled; no score claim]",
        "hardware_axis": _hardware_axis_label(),
        "pair_samples": len(pair_indices),
        "pair_samples_requested": args.pair_samples,
        "pair_universe": _pair_universe_for_candidates(candidates),
        "pair_indices": pair_indices,
        "noise_sigmas": [0.001, 0.01, 0.1, 1.0],
        "seed": args.seed,
        "scorer_batch_size": args.scorer_batch_size,
        "upstream_dir": str(args.upstream_dir),
        "scorer_assets": assets,
        "candidate_specs": candidate_status,
        "blockers": blockers,
        "notes": [
            "Default mode is plan-only; pass --execute for the long CPU run.",
            "Loads upstream SegNet+PoseNet once, then reuses the scorer across all candidates.",
            "Tier C deltas exclude the archive rate term and are not contest scores.",
            "No GPU, no remote dispatch, no promotion authority.",
        ],
    }


def _write_manifest(output_dir: Path, payload: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "tier_c_real_scorer_manifest.json"
    path.write_text(_json_text(payload), encoding="utf-8")
    return path


def _write_markdown(output_dir: Path, payload: dict[str, Any]) -> Path:
    lines = [
        "# Tier C Real-Scorer Four-Way",
        "",
        f"- mode: `{payload.get('mode')}`",
        f"- score_claim: `{str(payload.get('score_claim')).lower()}`",
        f"- evidence_axis: `{payload.get('evidence_axis')}`",
        f"- hardware_axis: `{payload.get('hardware_axis')}`",
        f"- pair_samples: `{payload.get('pair_samples')}`",
        "",
        "## Candidates",
        "",
        "| name | grammar | archive | status |",
        "|---|---|---|---|",
    ]
    for item in payload.get("candidate_specs", []):
        status = "present" if item.get("archive_exists") else "missing"
        lines.append(
            f"| {item.get('name')} | {item.get('grammar')} | "
            f"`{item.get('archive_path')}` | {status} |"
        )
    if payload.get("archives"):
        lines.extend(
            [
                "",
                "## Results",
                "",
                "| archive | verdict | density | latent_sigma1 | curve_knee | output |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in payload["archives"]:
            lines.append(
                f"| {row['name']} | {row.get('mdl_tier_c_substrate_class_verdict', '')} | "
                f"{row.get('mdl_tier_c_density_estimate', 0.0):.4f} | "
                f"{row.get('mdl_tier_c_latent_sigma1_delta', 0.0):.6f} | "
                f"{row.get('mdl_tier_c_curve_knee_signal', 0.0):.4f} | "
                f"`{row.get('output_json', '')}` |"
            )
    if payload.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{blocker}`" for blocker in payload["blockers"])
    path = output_dir / "tier_c_real_scorer_manifest.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _run_candidate(
    *,
    mdl: Any,
    spec: CandidateSpec,
    upstream_dir: Path,
    output_dir: Path,
    pair_indices: list[int],
    gt_pairs: Any,
    distortion_net: Any,
    seed: int,
    scorer_batch_size: int,
) -> dict[str, Any]:
    torch = mdl.torch
    device = torch.device("cpu")
    grammar = mdl.normalize_grammar(spec.grammar)
    torch.manual_seed(seed)
    if getattr(mdl, "np", None) is not None:
        mdl.np.random.seed(seed)
    rng = random.Random(seed)

    inner_bytes, sections = mdl.load_archive(spec.archive_path, grammar)
    archive_sha256 = _sha256_file(spec.archive_path) or ""
    archive_bytes = spec.archive_path.stat().st_size
    baseline_pose, baseline_seg = _compute_baseline_pose_seg(
        mdl=mdl,
        inner_bytes=inner_bytes,
        grammar=grammar,
        pair_indices=pair_indices,
        gt_pairs=gt_pairs,
        distortion_net=distortion_net,
        device=device,
        rng=rng,
        scorer_batch_size=scorer_batch_size,
    )
    baseline_score_components = mdl._score_components(baseline_pose, baseline_seg)
    result = mdl.ArchiveAblationResult(
        archive_name=spec.name,
        archive_path=str(spec.archive_path),
        archive_sha256=archive_sha256,
        archive_size_bytes=archive_bytes,
        grammar=grammar,
        device="cpu",
        pair_samples=len(pair_indices),
        baseline_seg=baseline_seg,
        baseline_pose=baseline_pose,
        baseline_score_components=baseline_score_components,
        pair_indices=pair_indices,
        decision_grade=False,
        included_sections=[],
        excluded_sections=[],
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        notes=[
            "[real-scorer CPU Tier-C delta curves; pair-sampled; no score claim]",
            _hardware_axis_label(),
            "Rate term excluded; archive bytes are constant across Tier C perturbations.",
            "This is not contest-CUDA, not full contest-CPU, and not promotion authority.",
            f"Sections parsed: {', '.join(sorted(sections))}",
        ],
    )
    result.tier_c = mdl.run_tier_c(
        inner_bytes,
        grammar,
        pair_indices,
        gt_pairs,
        baseline_seg,
        baseline_pose,
        distortion_net,
        device,
        rng,
        scorer_batch_size=scorer_batch_size,
    )
    result = mdl.aggregate_mdl_estimate(result)

    output = {
        "schema": "tac_tier_c_real_scorer_archive_result_v1",
        "schema_version": 1,
        "lane_id": LANE_ID,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_axis": "[real-scorer CPU Tier-C delta curves; pair-sampled; no score claim]",
        "hardware_axis": _hardware_axis_label(),
        "archive_result": dataclasses.asdict(result),
    }
    out_path = output_dir / f"{spec.name}_tier_c_real_scorer.json"
    out_path.write_text(_json_text(output), encoding="utf-8")

    return {
        "name": spec.name,
        "grammar": grammar,
        "role": spec.role,
        "archive_path": str(spec.archive_path),
        "archive_sha256": archive_sha256,
        "archive_bytes": archive_bytes,
        "baseline_seg": result.baseline_seg,
        "baseline_pose": result.baseline_pose,
        "baseline_score_components": result.baseline_score_components,
        "tier_c": [dataclasses.asdict(row) for row in result.tier_c],
        "mdl_tier_c_density_estimate": result.mdl_tier_c_density_estimate,
        "mdl_tier_c_substrate_class_verdict": (
            result.mdl_tier_c_substrate_class_verdict
        ),
        "mdl_tier_c_curve_knee_signal": result.mdl_tier_c_curve_knee_signal,
        "mdl_tier_c_latent_sigma1_delta": result.mdl_tier_c_latent_sigma1_delta,
        "output_json": str(out_path),
    }


def _compute_baseline_pose_seg(
    *,
    mdl: Any,
    inner_bytes: bytes,
    grammar: str,
    pair_indices: list[int],
    gt_pairs: Any,
    distortion_net: Any,
    device: Any,
    rng: random.Random,
    scorer_batch_size: int,
) -> tuple[float, float]:
    """Compute clean baseline pose/seg.

    Most grammars use the shared ``decode_to_frames`` dispatcher. DP1's Tier C
    implementation landed before that older decoder dispatcher learned DP1, so
    the runner uses a zero-sigma Tier C identity pass as a local compatibility
    bridge. With baseline_seg=baseline_pose=0, the zero-sigma row deltas are
    exactly the clean scorer components.
    """

    try:
        baseline_frames = mdl.decode_to_frames(inner_bytes, grammar, pair_indices, device)
    except NotImplementedError:
        if grammar != "dp1":
            raise
        zero_rows = mdl.run_tier_c(
            inner_bytes,
            grammar,
            pair_indices,
            gt_pairs,
            0.0,
            0.0,
            distortion_net,
            device,
            rng,
            noise_sigmas=[0.0],
            scorer_batch_size=scorer_batch_size,
        )
        if not zero_rows:
            raise RuntimeError("dp1 zero-sigma Tier C baseline returned no rows")
        row = zero_rows[0]
        if row.delta_pose is None or row.delta_seg is None:
            raise RuntimeError("dp1 zero-sigma Tier C baseline produced null deltas")
        return float(row.delta_pose), float(row.delta_seg)
    return mdl._compute_seg_pose_delta(
        distortion_net,
        gt_pairs,
        baseline_frames,
        device,
        batch_size=scorer_batch_size,
    )


def _execute(
    *,
    args: argparse.Namespace,
    candidates: list[CandidateSpec],
    pair_indices: list[int],
    manifest: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    if manifest["blockers"]:
        manifest["mode"] = "failed_closed"
        manifest["error"] = {
            "class": "fail_closed_prerequisite_error",
            "message": "cannot execute until scorer assets and all candidate archives exist",
        }
        return 2, manifest

    mdl = _load_mdl_module()
    torch = mdl.torch
    if torch is None:
        manifest["mode"] = "failed_closed"
        manifest["blockers"].append("torch_import_failed")
        return 2, manifest

    torch.manual_seed(args.seed)
    if getattr(mdl, "np", None) is not None:
        mdl.np.random.seed(args.seed)

    device = torch.device("cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.upstream_dir / "videos" / "0.mkv"
    t0 = time.time()
    gt_pairs = mdl._load_ground_truth_pairs(video_path, pair_indices)
    distortion_net = _load_real_scorer(mdl, args.upstream_dir)

    archives = []
    for idx, spec in enumerate(candidates):
        archives.append(
            _run_candidate(
                mdl=mdl,
                spec=spec,
                upstream_dir=args.upstream_dir,
                output_dir=args.output_dir,
                pair_indices=pair_indices,
                gt_pairs=gt_pairs,
                distortion_net=distortion_net,
                seed=args.seed + idx,
                scorer_batch_size=args.scorer_batch_size,
            )
        )

    manifest["mode"] = "executed"
    manifest["device"] = str(device)
    manifest["archives"] = archives
    manifest["elapsed_seconds"] = time.time() - t0
    manifest["blockers"] = []
    return 0, manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build or execute a CPU-only real-scorer Tier C comparison for "
            "A1/PR106/IBPS1/DP1. Default is plan-only; --execute runs scorer."
        )
    )
    p.add_argument("--execute", action="store_true", help="Run the long CPU scorer pass.")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--upstream-dir", type=Path, default=REPO_ROOT / "upstream")
    p.add_argument("--pair-samples", type=int, default=10)
    p.add_argument("--pair-indices-file", type=Path, default=None)
    p.add_argument("--scorer-batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--archive",
        action="append",
        default=[],
        metavar="NAME=PATH,grammar=GRAMMAR[,role=ROLE]",
        help=(
            "Override default candidates. May be repeated. If omitted, runs "
            "the built-in A1/PR106/IBPS1/DP1 control set."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.pair_samples <= 0 or args.pair_samples > N_PAIRS:
        parser.error(f"--pair-samples must be in 1..{N_PAIRS}")
    if args.scorer_batch_size <= 0:
        parser.error("--scorer-batch-size must be > 0")

    args.upstream_dir = Path(args.upstream_dir).resolve()
    if args.output_dir is None:
        args.output_dir = (
            REPO_ROOT
            / "experiments"
            / "results"
            / f"tier_c_real_scorer_fourway_{_utc_stamp()}"
        )
    else:
        args.output_dir = Path(args.output_dir).resolve()

    try:
        candidates = (
            [_parse_archive_spec(raw, REPO_ROOT) for raw in args.archive]
            if args.archive
            else list(DEFAULT_CANDIDATES)
        )
        pair_indices = _resolve_pair_indices(args, candidates)
    except (OSError, ValueError) as exc:
        payload = {
            "schema": "tac_tier_c_real_scorer_runner_v1",
            "schema_version": 1,
            "tool": "tools/run_tier_c_with_real_scorer.py",
            "mode": "failed_closed",
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "error": {
                "class": "fail_closed_input_error",
                "message": str(exc),
            },
        }
        _write_manifest(args.output_dir, payload)
        _write_markdown(args.output_dir, payload)
        print(f"failed_closed_manifest={args.output_dir / 'tier_c_real_scorer_manifest.json'}", file=sys.stderr)
        print(f"error={exc}", file=sys.stderr)
        return 2

    manifest = _build_manifest_base(
        args=args,
        candidates=candidates,
        pair_indices=pair_indices,
        mode="plan_only",
    )

    rc = 0
    if args.execute:
        try:
            rc, manifest = _execute(
                args=args,
                candidates=candidates,
                pair_indices=pair_indices,
                manifest=manifest,
            )
        except Exception as exc:
            manifest["mode"] = "failed_closed"
            manifest.setdefault("blockers", []).append("execution_exception")
            manifest["error"] = {
                "class": f"{type(exc).__name__}",
                "message": str(exc),
            }
            rc = 2

    manifest_path = _write_manifest(args.output_dir, manifest)
    md_path = _write_markdown(args.output_dir, manifest)
    if rc == 0:
        print(f"wrote_manifest={manifest_path}")
        print(f"wrote_markdown={md_path}")
        if not args.execute:
            print("plan_only=true")
            print("next_step=rerun with --execute to load real SegNet+PoseNet on CPU")
    else:
        print(f"failed_closed_manifest={manifest_path}", file=sys.stderr)
        print(f"failed_closed_markdown={md_path}", file=sys.stderr)
        if manifest.get("error"):
            print(f"error={manifest['error']['message']}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
