#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""One-liner Lightning Studio dispatch for PR106-anchor stack lanes.

Wraps the 6-gate `scripts/launch_lightning_batch_job.py exact-eval` boilerplate
into a single-arg invocation. Encodes the canonical Lightning settings
discovered the hard way 2026-05-05 (see
`reference_lightning_studio_canonical_dispatch_recipe_20260505.md`).

Usage:
    .venv/bin/python tools/lightning_dispatch_pr106_stack.py \
        --lane apogee_int4 \
        --archive experiments/results/apogee_int4_repack_20260504_claude/apogee_int4_archive.zip \
        --predicted-low 0.155 --predicted-high 0.180

Optional:
    --inflate-sh PATH   (default: derived from --lane name)
    --skip-stage        (skip lightning_repro_workspace.py if already staged)
    --ssh-target        (or LIGHTNING_SSH_TARGET; required unless --skip-stage)
    --print-only        (print the resolved invocation without staging,
                         claiming, or running)

Workflow:
    1. Stages workspace via lightning_repro_workspace.py (sync src/, experiments/,
       submissions/, scripts/, upstream/, tools/, pyproject.toml + the archive)
    2. Generates source manifest at experiments/results/lightning_batch/<job>/source_manifest.json
    3. Files dispatch claim with platform=lightning canonical
    4. Submits exact-eval via launch_lightning_batch_job.py with all 6 gates pre-filled
    5. Returns the job-name for harvest

Encodes:
    - platform=lightning (claim must be exact lowercase)
    - INFLATE_TORCH_SPEC=torch==2.5.1+cu124 (driver<580 cu13 trap)
    - --allow-skip-remote-preflight-reason for the launcher path-lowercase bug
    - PR106 baseline anchor (0.20945673 / 186239 bytes)

Failure modes handled:
    - Lightning SSH unreachable → fail loud
    - Already-staged workspace → reuse manifest
    - Existing terminal claim → re-file only with --force-claim
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.deploy.lightning.defaults import (  # noqa: E402
    DEFAULT_LIGHTNING_REMOTE_PACT,
    DEFAULT_LIGHTNING_SSH_TARGET,
    DEFAULT_LIGHTNING_STUDIO,
    DEFAULT_LIGHTNING_TEAMSPACE,
    DEFAULT_LIGHTNING_USER,
)
from tac.repo_io import read_json, sha256_file  # noqa: E402

_sha256_file = sha256_file  # Backward-compatible test/tool API alias.

DEFAULT_SSH_TARGET = DEFAULT_LIGHTNING_SSH_TARGET
DEFAULT_REMOTE_PACT = DEFAULT_LIGHTNING_REMOTE_PACT
PR106_BASELINE_SCORE = 0.20945673
PR106_BASELINE_BYTES = 186239
LIGHTNING_STUDIO = DEFAULT_LIGHTNING_STUDIO
LIGHTNING_TEAMSPACE = DEFAULT_LIGHTNING_TEAMSPACE
LIGHTNING_USER = DEFAULT_LIGHTNING_USER
INFLATE_TORCH_SPEC = "torch==2.5.1+cu124"
UV_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cu124"
UV_INDEX_STRATEGY = "unsafe-best-match"
APOGEE_DISTORTION_GATE_PASSED = {"passed", "pass", "ready", "exact_positive_cuda"}
APOGEE_ALLOWED_EVIDENCE_SEMANTICS = {
    "contest_faithful_distortion_model",
    "scorer_basin_parity_gate",
    "contest_cuda_exact_eval_positive",
}
APOGEE_FORBIDDEN_EVIDENCE_MARKERS = (
    "byte_only",
    "byte-only",
    "prediction_only",
    "predicted_band",
    "invalid_predicted_band",
    "proxy_only",
    "distortion_proxy_local",
    "local_distortion_proxy",
    "[distortion-proxy:local]",
)


def _utc_now_iso() -> str:
    return dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_plus_1h_iso() -> str:
    return (dt.datetime.now(tz=dt.UTC) + dt.timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _job_name(lane: str) -> str:
    ts = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"lane_{lane}_pr106_{ts}"


def validate_apogee_dispatch_gate(
    *,
    lane: str,
    archive: Path,
    gate_json: Path | None,
    allow_forensic_print_only: bool,
    print_only: bool,
) -> None:
    """Fail closed for Apogee intN unless a real distortion gate is present."""
    if not lane.startswith("apogee_int"):
        return
    if allow_forensic_print_only and print_only:
        print("[apogee-gate] forensic print-only override; no GPU dispatch authorized")
        return
    if allow_forensic_print_only and not print_only:
        sys.exit("FATAL: --allow-forensic-apogee-intN requires --print-only; GPU dispatch remains blocked")
    if gate_json is None:
        sys.exit(
            "FATAL: apogee_intN dispatch requires --apogee-distortion-gate-json. "
            "Byte-only Pareto predictions are forensic/noncanonical after the int4 exact negative."
        )
    try:
        payload = read_json(gate_json)
    except (OSError, ValueError) as exc:
        sys.exit(f"FATAL: invalid Apogee distortion gate JSON {gate_json}: {exc}")
    if not isinstance(payload, dict):
        sys.exit(f"FATAL: Apogee distortion gate must be a JSON object: {gate_json}")

    actual_sha = sha256_file(archive)
    recorded_sha = (
        payload.get("candidate_archive_sha256")
        or payload.get("archive_sha256")
        or payload.get("archive", {}).get("sha256")
    )
    blockers: list[str] = []
    if recorded_sha != actual_sha:
        blockers.append(f"candidate_archive_sha256 mismatch gate={recorded_sha!r} actual={actual_sha}")
    if payload.get("ready_for_exact_eval_dispatch") is not True:
        blockers.append("ready_for_exact_eval_dispatch is not true")
    semantics = str(payload.get("evidence_semantics", "")).strip().lower()
    if not semantics:
        blockers.append("missing evidence_semantics")
    if semantics not in APOGEE_ALLOWED_EVIDENCE_SEMANTICS:
        blockers.append(f"unsupported evidence_semantics={semantics!r}")
    payload_text = str(payload).lower()
    for marker in APOGEE_FORBIDDEN_EVIDENCE_MARKERS:
        if marker in semantics or marker in payload_text:
            blockers.append(f"forbidden proxy/prediction evidence marker {marker!r}")
            break
    distortion_status = str(payload.get("distortion_model_status", "")).lower()
    parity_status = str(payload.get("scorer_basin_parity_status", "")).lower()
    exact_positive = payload.get("exact_positive_cuda_evidence") is True
    evidence_grade = str(payload.get("evidence_grade", "")).lower()
    if "negative" in evidence_grade or evidence_grade in {"invalid", "external", "prediction"}:  # CUSTODY_VALIDATOR_OK: blocker-accumulator with sys.exit on failure; sha256 verified at the top of the same function (line ~138)
        blockers.append(f"non-promotable evidence_grade={payload.get('evidence_grade')!r}")
    if semantics == "contest_faithful_distortion_model" and distortion_status not in APOGEE_DISTORTION_GATE_PASSED:
        blockers.append("contest_faithful_distortion_model requires passing distortion_model_status")
    if semantics == "scorer_basin_parity_gate" and parity_status not in APOGEE_DISTORTION_GATE_PASSED:
        blockers.append("scorer_basin_parity_gate requires passing scorer_basin_parity_status")
    if semantics == "contest_cuda_exact_eval_positive" and not exact_positive:
        blockers.append("contest_cuda_exact_eval_positive requires exact_positive_cuda_evidence=true")
    if blockers:
        sys.exit("FATAL: Apogee distortion gate blocked dispatch: " + "; ".join(blockers))


def stage_workspace(*, lane: str, job_name: str, archive: Path,
                    ssh_target: str, remote_pact: str) -> Path:
    manifest_dir = REPO_ROOT / "experiments" / "results" / "lightning_batch" / job_name
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = manifest_dir / "source_manifest.json"
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "lightning_repro_workspace.py"),
        "--remote", ssh_target,
        "--remote-pact", remote_pact,
        "--run-id", job_name,
        "--manifest-out", str(manifest_out),
        "--source", "src",
        "--source", "experiments",
        "--source", "submissions",
        "--source", "scripts",
        "--source", "upstream",
        "--source", "tools",
        "--source", "pyproject.toml",
        "--artifact", str(archive),
        "--requirements-mode", "no-install",
        "--no-install",
        "--ssh-connect-timeout", "30",
    ]
    print(f"[stage] {' '.join(cmd[:6])} ... ({len(cmd)} args)")
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        sys.exit(f"FATAL: lightning_repro_workspace.py failed (rc={result.returncode})")
    return manifest_out


def claim_lane(
    *,
    lane: str,
    job_name: str,
    dispatch_lane_id: str | None,
    dispatch_claims_path: Path,
    force_claim: bool,
    force_claim_reason: str | None,
) -> None:
    notes = f"canonical Lightning dispatch via tools/lightning_dispatch_pr106_stack.py {_utc_now_iso()}"
    if force_claim:
        if not force_claim_reason:
            sys.exit("FATAL: --force-claim requires --force-claim-reason")
        notes = f"{notes}; force-claim: {force_claim_reason}"
    cmd = [
        sys.executable, str(REPO_ROOT / "tools" / "claim_lane_dispatch.py"), "claim",
        "--claims-path", str(dispatch_claims_path),
        "--lane-id", dispatch_lane_id or f"lane_{lane}",
        "--agent", "claude_lab",
        "--platform", "lightning",
        "--instance-job-id", job_name,
        "--predicted-eta-utc", _utc_plus_1h_iso(),
        "--status", "active_dispatching",
        "--notes", notes,
    ]
    if force_claim:
        cmd += ["--force"]
    print(f"[claim] platform=lightning lane={dispatch_lane_id or f'lane_{lane}'} job={job_name}")
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        sys.exit(f"FATAL: claim_lane_dispatch.py failed (rc={result.returncode})")


def submit_dispatch(*, lane: str, job_name: str, archive: Path, manifest: Path,
                    inflate_sh: Path, predicted_low: float, predicted_high: float,
                    ssh_target: str, machine: str, print_only: bool,
                    dispatch_lane_id: str | None = None,
                    dispatch_claims_path: Path = REPO_ROOT
                    / ".omx/state/active_lane_dispatch_claims.md",
                    remote_pact: str = DEFAULT_REMOTE_PACT) -> int:
    # FIX 2026-05-05: --repo-dir / --upstream-dir MUST be the REMOTE Lightning
    # path, not the operator's local mac path. Previously this passed
    # str(REPO_ROOT) which evaluated to /Users/adpena/Projects/pact — a path
    # that does NOT exist on the Lightning Studio runner. Every dispatched job
    # failed at the first `cd` in the generated runner script. 8 of 8 jobs
    # cost ~$1.55 wasted because of this single line. Root cause documented
    # at memory feedback_lightning_dispatch_catastrophe_8of8_failed_20260505.
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "launch_lightning_batch_job.py"), "exact-eval",
        "--job-name", job_name,
        # FIX 2026-05-05: --archive must be REPO-RELATIVE so the launcher's
        # _remote_repo_rel() resolves it against --repo-dir (the REMOTE
        # /teamspace/.../pact). If we pass the absolute mac path, the launcher
        # rejects with "must be inside --repo-dir" because /Users/adpena/...
        # is not a prefix of /teamspace/studios/this_studio/pact.
        "--archive", str(archive.relative_to(REPO_ROOT)),
        "--repo-dir", remote_pact,
        "--upstream-dir", f"{remote_pact}/upstream",
        "--teamspace", LIGHTNING_TEAMSPACE,
        "--studio", LIGHTNING_STUDIO,
        "--user", LIGHTNING_USER,
        "--machine", machine,
        # FIX 2026-05-05: same as --archive, --inflate-sh must be REPO-RELATIVE.
        "--inflate-sh", str(inflate_sh.relative_to(REPO_ROOT)),
        "--predicted-band", str(predicted_low), str(predicted_high),
        "--baseline-score", str(PR106_BASELINE_SCORE),
        "--baseline-archive-bytes", str(PR106_BASELINE_BYTES),
        "--infer-expected-archive",
        "--adjudicate",
        "--regression-threshold", "0.05",
        "--dispatch-lane-id", dispatch_lane_id or f"lane_{lane}",
        "--dispatch-claims-path", str(dispatch_claims_path),
        "--source-manifest", str(manifest),
        # NOTE: do NOT pass --remote-preflight-ssh-target here.
        # The launcher's _run_remote_supply_chain_preflight runs unconditionally
        # when ssh-target is set AND lowercases the local repo path which fails
        # on macOS Mixed-Case repos. The skip-reason short-circuits the SHAPE
        # check but not the actual preflight execution. Cleanest path is to
        # not provide the ssh-target and rely on lightning_repro_workspace
        # having already done the supply-chain verification.
        "--allow-skip-remote-preflight-reason",
            "manually-staged-via-lightning_repro_workspace.py pre-submit "
            "(launcher path-lowercase bug + redundant preflight)",
        "--env", f"INFLATE_TORCH_SPEC={INFLATE_TORCH_SPEC}",
        "--env", f"UV_EXTRA_INDEX_URL={UV_EXTRA_INDEX_URL}",
        "--env", f"UV_INDEX_STRATEGY={UV_INDEX_STRATEGY}",
    ]
    if print_only:
        print("=== resolved invocation (would run) ===")
        print(" \\\n  ".join(cmd))
        return 0
    print(f"[submit] launching Lightning exact-eval for {job_name}...")
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", required=True,
                        help="lane id (e.g. apogee_int4, pr106_latent_sidecar)")
    parser.add_argument("--archive", required=True, type=Path,
                        help="path to archive.zip (must be inside repo root)")
    parser.add_argument("--predicted-low", required=True, type=float)
    parser.add_argument("--predicted-high", required=True, type=float)
    parser.add_argument("--inflate-sh", type=Path, default=None,
                        help="defaults to submissions/<derived-from-lane>/inflate.sh")
    parser.add_argument("--skip-stage", action="store_true",
                        help="skip lightning_repro_workspace.py (workspace already staged)")
    parser.add_argument("--ssh-target", default=DEFAULT_SSH_TARGET)
    parser.add_argument("--remote-pact", default=DEFAULT_REMOTE_PACT)
    parser.add_argument("--machine", default="g4dn.2xlarge",
                        help="Lightning machine class (default g4dn.2xlarge — "
                             "AWS T4 instance class that codex used yesterday for "
                             "exact_eval_pr95_hnerv_muon_repacked etc. NOT 'T4' literal — "
                             "the abbreviation fails with 'accelerator T4 not found for AWS cluster').")
    parser.add_argument("--print-only", action="store_true",
                        help="print resolved invocation without staging, claiming, or running")
    parser.add_argument(
        "--force-claim",
        action="store_true",
        help="Force the dispatch claim only when replacing a known terminal/stale claim.",
    )
    parser.add_argument(
        "--force-claim-reason",
        default=None,
        help="Required rationale when --force-claim is set.",
    )
    parser.add_argument(
        "--apogee-distortion-gate-json",
        type=Path,
        help=(
            "Required for apogee_int* GPU dispatch: JSON proving the changed "
            "archive has a scorer-basin distortion gate or exact positive CUDA evidence."
        ),
    )
    parser.add_argument(
        "--allow-forensic-apogee-intN",
        action="store_true",
        help="Allow apogee_int* command rendering only with --print-only; never submits GPU work.",
    )
    parser.add_argument("--job-name", default=None,
                        help="override auto-generated job name")
    parser.add_argument(
        "--dispatch-lane-id",
        default=None,
        help=(
            "Exact lane id expected in the active dispatch-claim ledger. "
            "Defaults to lane_<--lane> for legacy wrapper calls."
        ),
    )
    parser.add_argument(
        "--dispatch-claims-path",
        type=Path,
        default=REPO_ROOT / ".omx/state/active_lane_dispatch_claims.md",
        help="Markdown active-dispatch claim ledger checked before Studio submit.",
    )
    parser.add_argument(
        "--use-existing-dispatch-claim",
        action="store_true",
        help="Do not create a new claim; require the existing matching claim instead.",
    )
    args = parser.parse_args(argv)

    if not args.archive.is_absolute():
        args.archive = REPO_ROOT / args.archive
    if not args.archive.is_file():
        sys.exit(f"FATAL: archive not found: {args.archive}")
    try:
        args.archive.relative_to(REPO_ROOT)
    except ValueError:
        sys.exit(f"FATAL: archive must be inside repo root: {args.archive}")

    job_name = args.job_name or _job_name(args.lane)
    dispatch_lane_id = args.dispatch_lane_id or f"lane_{args.lane}"

    if args.inflate_sh is None:
        # derive: apogee_int4 → submissions/apogee_intN/inflate.sh
        # pr106_latent_sidecar → submissions/pr106_latent_sidecar/inflate.sh
        inflate_dir = "apogee_intN" if args.lane.startswith("apogee_int") else args.lane
        args.inflate_sh = REPO_ROOT / "submissions" / inflate_dir / "inflate.sh"
    elif not args.inflate_sh.is_absolute():
        args.inflate_sh = REPO_ROOT / args.inflate_sh
    if not args.inflate_sh.is_file():
        sys.exit(f"FATAL: inflate.sh not found: {args.inflate_sh}")
    try:
        args.inflate_sh.relative_to(REPO_ROOT)
    except ValueError:
        sys.exit(f"FATAL: inflate.sh must be inside repo root: {args.inflate_sh}")

    validate_apogee_dispatch_gate(
        lane=args.lane,
        archive=args.archive,
        gate_json=args.apogee_distortion_gate_json,
        allow_forensic_print_only=args.allow_forensic_apogee_intN,
        print_only=args.print_only,
    )

    if args.print_only:
        manifest = (
            REPO_ROOT
            / "experiments"
            / "results"
            / "lightning_batch"
            / job_name
            / "source_manifest.json"
        )
    elif args.skip_stage:
        manifest = REPO_ROOT / "experiments" / "results" / "lightning_batch" / job_name / "source_manifest.json"
        if not manifest.is_file():
            sys.exit(f"FATAL: --skip-stage but manifest not found: {manifest}")
    else:
        if not args.ssh_target:
            sys.exit("FATAL: set --ssh-target or LIGHTNING_SSH_TARGET before staging to Lightning")
        manifest = stage_workspace(
            lane=args.lane,
            job_name=job_name,
            archive=args.archive,
            ssh_target=args.ssh_target,
            remote_pact=args.remote_pact,
        )

    if not args.print_only and not args.use_existing_dispatch_claim:
        claim_lane(
            lane=args.lane,
            job_name=job_name,
            dispatch_lane_id=dispatch_lane_id,
            dispatch_claims_path=args.dispatch_claims_path,
            force_claim=args.force_claim,
            force_claim_reason=args.force_claim_reason,
        )
    elif args.use_existing_dispatch_claim:
        print(
            f"[claim] using existing active dispatch claim "
            f"lane={dispatch_lane_id} job={job_name}"
        )

    return submit_dispatch(
        lane=args.lane,
        job_name=job_name,
        archive=args.archive,
        manifest=manifest,
        inflate_sh=args.inflate_sh,
        predicted_low=args.predicted_low,
        predicted_high=args.predicted_high,
        ssh_target=args.ssh_target,
        machine=args.machine,
        print_only=args.print_only,
        dispatch_lane_id=dispatch_lane_id,
        dispatch_claims_path=args.dispatch_claims_path,
        remote_pact=args.remote_pact,
    )


if __name__ == "__main__":
    raise SystemExit(main())
