#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Harvest Kaggle T1 Ballé sweep kernels — sister of ``tools/harvest_modal_calls.py``.

Per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" this is
the harvest counterpart of the parallel-dispatch actuator
``scripts/operator_authorize_kaggle_t1_balle_sweep.sh``. It iterates the
active Kaggle T1 Ballé sweep kernels, downloads their output on completion,
appends cost-band anchors, and closes the terminal lane-claim.

Concretely:

1. **Discover kernels.** Walk ``experiments/kaggle_kernels/*/kernel-metadata.json``
   for the slug list; alternatively accept ``--slug user/slug`` repeated.
2. **Poll status.** Use ``kaggle kernels status <slug>`` (same pattern as
   ``scripts/kaggle_check.py``).
3. **On ``complete``:** download via ``kaggle kernels output <slug> -p <dir>``
   to ``experiments/results/lane_kaggle_t1_balle_<slug>_<utc>/``.
4. **Read kernel summary** (``kaggle_kernel_summary.json``) emitted by the
   kernel script and forward its metrics to the cost-band posterior via
   ``tools/append_cost_band_anchor.py``.
5. **Close the lane claim** via ``tools/claim_lane_dispatch.py claim --force
   --status completed_kaggle`` (or ``failed_kaggle_<reason>``).
6. **Detect P100 trap.** If the kernel exited with rc=2 and the log mentions
   "P100 trap", the harvester records a ``failed_kaggle_p100_assignment``
   status; the operator wrapper can re-push.

Provider-axis contract:
- Kaggle T4 free-tier results harvested here are
  ``[provider-CUDA:kaggle advisory]``. They stay ``score_claim=false`` and
  ``promotion_eligible=false`` until a future exact target contract wires the
  full lifecycle, runtime closure, claim, harvest, and adjudication path.
- They are NOT ``[contest-CUDA]`` evidence and are NOT a substitute for the GHA
  Linux x86_64 ``[contest-CPU]`` axis.

Usage:

    .venv/bin/python tools/harvest_kaggle_kernels.py
    .venv/bin/python tools/harvest_kaggle_kernels.py \\
        --slug adpena/comma-lab-t1-balle-a
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

_TOOL_FILE = Path(__file__).resolve()
_REPO_ROOT_CANDIDATE = _TOOL_FILE.parents[1]
if str(_REPO_ROOT_CANDIDATE) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_CANDIDATE))
if str(_TOOL_FILE.parent) not in sys.path:
    sys.path.insert(0, str(_TOOL_FILE.parent))

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tools.claim_lane_dispatch import TERMINAL_PREFIXES  # noqa: E402

KAGGLE_PROVIDER_AXIS_LABEL = "[provider-CUDA:kaggle advisory]"


def _kaggle_cmd() -> list[str]:
    """Resolve the kaggle CLI binary; matches scripts/kaggle_check.py logic."""
    venv = REPO_ROOT / ".venv" / "bin" / "kaggle"
    if venv.exists():
        return [str(venv)]
    found = shutil.which("kaggle")
    if found:
        return [found]
    print("[harvest-kaggle] FATAL: kaggle CLI not found. uv pip install kaggle", file=sys.stderr)
    sys.exit(3)


def discover_kernel_slugs(kernel_root: Path) -> list[str]:
    """Walk ``experiments/kaggle_kernels/*/kernel-metadata.json`` for sweep slugs.

    Only returns slugs whose basename starts with ``comma-lab-t1-balle-`` — we
    do not poll unrelated kernels.
    """
    slugs: list[str] = []
    if not kernel_root.is_dir():
        return slugs
    for meta_path in sorted(kernel_root.glob("*/kernel-metadata.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        slug = meta.get("id", "")
        # Slug shape: <user>/<basename>; we filter by basename prefix.
        if "/" in slug and slug.split("/", 1)[1].startswith("comma-lab-t1-balle-"):
            slugs.append(slug)
    return slugs


def get_kernel_status(kaggle_cmd: list[str], slug: str, *, timeout_s: int = 15) -> str | None:
    """Return the kernel status string (or None on lookup failure)."""
    try:
        proc = subprocess.run(
            [*kaggle_cmd, "kernels", "status", slug],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return f"STATUS_TIMEOUT_AFTER_{timeout_s}S"
    if proc.returncode != 0:
        return None
    stdout = proc.stdout.strip()
    if '"' in stdout:
        return stdout.split('"')[1]
    return stdout


def download_kernel_output(
    kaggle_cmd: list[str], slug: str, dest: Path, *, timeout_s: int = 300
) -> bool:
    """Download kernel outputs to ``dest``. Returns True on success."""
    dest.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [*kaggle_cmd, "kernels", "output", slug, "-p", str(dest)],
        capture_output=True, text=True, timeout=timeout_s,
    )
    if proc.returncode != 0:
        print(
            f"[harvest-kaggle] WARN: kernels output failed for {slug}: "
            f"{proc.stderr.strip() or proc.stdout.strip()}",
            file=sys.stderr,
        )
        return False
    return True


def classify_terminal_status(
    *,
    kaggle_status: str | None,
    summary: dict | None,
) -> str:
    """Map (kaggle_status, kernel summary) into a lane-claim terminal status.

    Returns one of:
      - completed_kaggle
      - failed_kaggle_p100_assignment
      - failed_kaggle_no_cuda
      - failed_kaggle_trainer_<rc>
      - failed_kaggle_<status>
      - stale_superseded_kaggle (status unknown but slug deleted)
    """
    if kaggle_status is None:
        return "stale_superseded_kaggle"
    norm = kaggle_status.lower()
    if norm == "complete":
        rc = (summary or {}).get("trainer_returncode")
        if rc == 0:
            return "completed_kaggle"
        if rc == 2:
            return "failed_kaggle_p100_assignment"
        if rc == 99:
            return "failed_kaggle_no_cuda"
        return f"failed_kaggle_trainer_rc{rc}"
    if norm in ("error", "failed", "cancelled", "canceled"):
        # If we have the summary, prefer the kernel-side classification.
        rc = (summary or {}).get("trainer_returncode")
        if rc == 2:
            return "failed_kaggle_p100_assignment"
        if rc == 99:
            return "failed_kaggle_no_cuda"
        return f"failed_kaggle_{norm}"
    # running / queued / scheduled — not terminal yet.
    return f"in_flight_kaggle_{norm}"


def append_cost_band_anchor_from_summary(
    *, summary: dict, dispatch_label: str, anchor_tool: Path
) -> dict:
    """Forward the kernel summary to ``tools/append_cost_band_anchor.py``."""
    if not anchor_tool.is_file():
        return {"appended": False, "reason": "anchor_tool_missing"}
    gpu_name = (summary.get("gpu_info") or {}).get("gpu_name", "T4")
    cmd = [
        sys.executable, str(anchor_tool),
        "--dispatch-label", dispatch_label,
        "--trainer", summary.get("trainer", "experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py"),
        "--platform", "kaggle",
        "--gpu", str(gpu_name).replace(" ", "_"),
        "--epochs", str(summary.get("epochs", 1500)),
        "--batch-size", str(summary.get("batch_size", 32)),
        "--all-flags-on",
        "--actual-wall-clock-sec", f"{summary.get('wall_clock_sec', 0.0):.0f}",
        "--actual-cost-usd", "0.00",
        "--notes",
        (
            f"kaggle_harvest;axis={KAGGLE_PROVIDER_AXIS_LABEL};"
            "score_claim=false;promotion_eligible=false;"
            f"rc={summary.get('trainer_returncode')}"
        ),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return {
        "appended": proc.returncode == 0,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "rc": proc.returncode,
    }


def close_lane_claim(
    *, claim_tool: Path, lane_id: str, instance_job_id: str, status: str, notes: str
) -> dict:
    """Close the active lane-claim with a terminal status."""
    if not claim_tool.is_file():
        return {"closed": False, "reason": "claim_tool_missing"}
    cmd = [
        sys.executable, str(claim_tool), "claim",
        "--lane-id", lane_id,
        "--platform", "kaggle",
        "--instance-job-id", instance_job_id,
        "--agent", "claude:harvest_kaggle_kernels",
        "--status", status,
        "--notes", notes,
        "--force",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return {
        "closed": proc.returncode == 0,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "rc": proc.returncode,
    }


def newest_active_kaggle_claim_instance_job_id(
    *,
    claims_path: Path,
    lane_id: str,
) -> str | None:
    """Return newest active Kaggle claim id for a lane, if the ledger has one."""

    if not claims_path.is_file():
        return None
    for line in claims_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or "timestamp_utc" in line or "---" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 8:
            continue
        row_lane = cells[2]
        platform = cells[3]
        instance_job_id = cells[4]
        status = cells[6]
        if row_lane != lane_id or platform != "kaggle":
            continue
        if any(status.startswith(prefix) for prefix in TERMINAL_PREFIXES):
            return None
        return instance_job_id or None
    return None


def harvest_one_slug(
    *, slug: str, kaggle_cmd: list[str], results_root: Path,
    anchor_tool: Path, claim_tool: Path,
) -> dict:
    """Harvest a single kernel slug and return a structured summary."""
    print(f"\n=== {slug} ===")
    status = get_kernel_status(kaggle_cmd, slug)
    print(f"  kaggle status: {status}")

    summary: dict | None = None
    output_dir: Path | None = None
    download_ok: bool = False

    if status and status.lower() == "complete":
        utc = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        slug_basename = slug.split("/", 1)[1] if "/" in slug else slug
        output_dir = results_root / f"lane_kaggle_t1_balle_{slug_basename}_{utc}"
        download_ok = download_kernel_output(kaggle_cmd, slug, output_dir)
        if download_ok:
            summary_path = output_dir / "kaggle_kernel_summary.json"
            if summary_path.is_file():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    print(
                        f"  trainer rc={summary.get('trainer_returncode')} "
                        f"wall={summary.get('wall_clock_sec', 0.0):.0f}s "
                        f"gpu={(summary.get('gpu_info') or {}).get('gpu_name')}"
                    )
                except json.JSONDecodeError:
                    print("  WARN: kaggle_kernel_summary.json was malformed.")

    terminal_status = classify_terminal_status(kaggle_status=status, summary=summary)
    print(f"  terminal status: {terminal_status}")

    cost_anchor_result: dict | None = None
    if summary is not None and terminal_status == "completed_kaggle":
        dispatch_label = summary.get("dispatch_label", slug)
        cost_anchor_result = append_cost_band_anchor_from_summary(
            summary=summary, dispatch_label=dispatch_label, anchor_tool=anchor_tool,
        )
        if cost_anchor_result.get("appended"):
            print(f"  cost-band anchor appended: {cost_anchor_result.get('stdout', '')}")
        else:
            print(f"  cost-band anchor SKIPPED: {cost_anchor_result.get('reason') or cost_anchor_result.get('stderr')}")

    lane_close_result: dict | None = None
    if not terminal_status.startswith("in_flight_"):
        # Variant suffix encoded in dispatch_label (kaggle_t1_balle_<variant>_<utc>).
        # Lane id is `t1_balle_kaggle_sweep_<variant>` per the wrapper.
        # We do a best-effort lookup; if the label format doesn't match we fall
        # back to a synthetic lane-id keyed by the slug basename.
        slug_basename = slug.split("/", 1)[1] if "/" in slug else slug
        variant_suffix = slug_basename.removeprefix("comma-lab-t1-balle-")
        lane_id = f"t1_balle_kaggle_sweep_{variant_suffix}"
        dispatch_label = (
            str(summary.get("dispatch_label"))
            if summary is not None and summary.get("dispatch_label")
            else newest_active_kaggle_claim_instance_job_id(
                claims_path=REPO_ROOT / ".omx/state/active_lane_dispatch_claims.md",
                lane_id=lane_id,
            )
        )
        if not dispatch_label:
            dispatch_label = slug
        rc_label = summary.get("trainer_returncode") if summary is not None else "summary_missing"
        notes = (
            f"kaggle harvest; axis={KAGGLE_PROVIDER_AXIS_LABEL}; "
            "score_claim=false; promotion_eligible=false; "
            f"terminal_status={terminal_status}; rc={rc_label}"
        )
        lane_close_result = close_lane_claim(
            claim_tool=claim_tool,
            lane_id=lane_id,
            instance_job_id=dispatch_label,
            status=terminal_status,
            notes=notes,
        )
        print(f"  lane-claim closed: {lane_close_result.get('closed')} ({terminal_status})")

    return {
        "slug": slug,
        "kaggle_status": status,
        "terminal_status": terminal_status,
        "download_ok": download_ok,
        "output_dir": str(output_dir) if output_dir else None,
        "summary": summary,
        "cost_anchor": cost_anchor_result,
        "claim_close_result": lane_close_result,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slug",
        action="append",
        default=None,
        help="Kernel slug user/name. Repeat for multiple. If unset, walk experiments/kaggle_kernels/*/.",
    )
    parser.add_argument(
        "--kernel-root",
        type=Path,
        default=REPO_ROOT / "experiments" / "kaggle_kernels",
        help="Directory containing per-kernel metadata subdirs.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "experiments" / "results",
        help="Root under which harvested outputs are written.",
    )
    args = parser.parse_args(argv)

    kaggle_cmd = _kaggle_cmd()
    anchor_tool = REPO_ROOT / "tools" / "append_cost_band_anchor.py"
    claim_tool = REPO_ROOT / "tools" / "claim_lane_dispatch.py"

    slugs = list(args.slug) if args.slug else discover_kernel_slugs(args.kernel_root)
    if not slugs:
        print(
            f"[harvest-kaggle] no T1 Ballé sweep slugs found "
            f"(looked at {args.kernel_root}). Pass --slug explicitly.",
            file=sys.stderr,
        )
        return 0

    results: list[dict] = []
    for slug in slugs:
        results.append(harvest_one_slug(
            slug=slug,
            kaggle_cmd=kaggle_cmd,
            results_root=args.results_root,
            anchor_tool=anchor_tool,
            claim_tool=claim_tool,
        ))

    summary_path = args.results_root / "_kaggle_t1_balle_harvest_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[harvest-kaggle] summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
