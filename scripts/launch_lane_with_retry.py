#!/usr/bin/env python3
"""launch_lane_with_retry.py — phase1+phase2 with auto-retry on Vast.ai failures.

Wraps `launch_lane_on_vastai.py phase1` + the phase2 split (wait+scp+extract+launch).
On any phase2 failure (NVDEC_BAD, SCP_FAIL, extract timeout), destroys the
instance and retries with a fresh host. Up to --max-retries (default 3).

Why: 2026-04-29 dispatches showed ~80% NVDEC_BAD rate on Vast.ai 4090.
Each manual retry cost ~5-10 min of operator time. This wrapper makes
dispatch resilient by design.

Usage:
  .venv/bin/python scripts/launch_lane_with_retry.py \\
    --lane-script scripts/remote_lane_X.sh --label lane_X --max-dph 0.40 \\
    --predicted-band 0.85 1.10 --estimated-cost 4.00 --max-retries 3

Exit codes:
  0 = lane successfully launched (instance running, lane_script tmux'd)
  1 = max retries exhausted
  2 = invalid args / pre-flight failure
  3 = remote launch state unknown; verify instance manually, no blind retry
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = REPO_ROOT / "scripts" / "launch_lane_on_vastai.py"
PYBIN = ".venv/bin/python"
PHASE1_TIMEOUT_SECONDS = 180
PHASE2_WAIT_TIMEOUT_SECONDS = 540
PHASE2_SCP_TIMEOUT_SECONDS = 1200
PHASE2_EXTRACT_TIMEOUT_SECONDS = 300
PHASE2_LAUNCH_TIMEOUT_SECONDS = 420
VASTAI = REPO_ROOT / ".venv/bin/vastai"
LOCK_DIR = REPO_ROOT / ".omx/state/launch_locks"
DISPATCH_HOLDS_PATH = REPO_ROOT / ".omx/state/dispatch_holds.json"
LANE12_L2_CLEARANCE_PATH = REPO_ROOT / ".omx/state/lane12_nerv_l2_clearance.json"
_ACTIVE_STAGE_PROC: subprocess.Popen[str] | None = None

LANE_19_HOLD_KEY = "lane_19_logit_margin_2026-04-30"
LANE_20_HOLD_KEY = "lane_20_balle_2026-04-30"
FORENSIC_HOLD_KEYS = {LANE_19_HOLD_KEY, LANE_20_HOLD_KEY}
LANE12_LOGICAL_PREFIXES = (
    "lane_12_nerv",
    "lane12_nerv",
    "lane_nerv",
)
RETRAINING_SCRIPT_RE = re.compile(
    r"experiments/(?:train_[A-Za-z0-9_]+|optimize_poses)\.py"
)


def _kill_process_group(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    pgid: int | None
    try:
        pgid = os.getpgid(proc.pid)
    except (PermissionError, ProcessLookupError):
        pgid = None
    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
    except (PermissionError, ProcessLookupError):
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGKILL)
            else:
                proc.kill()
        except (PermissionError, ProcessLookupError):
            return
        proc.wait(timeout=5)


def _handle_parent_signal(signum: int, _frame) -> None:
    if _ACTIVE_STAGE_PROC is not None:
        _kill_process_group(_ACTIVE_STAGE_PROC)
    raise SystemExit(128 + signum)


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGINT, _handle_parent_signal)
    signal.signal(signal.SIGTERM, _handle_parent_signal)


def run_stage(args: list[str], timeout: int = 300) -> tuple[int, str]:
    """Run a launcher stage. Returns (returncode, combined output)."""
    global _ACTIVE_STAGE_PROC
    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            args,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        _ACTIVE_STAGE_PROC = proc
        stdout, stderr = proc.communicate(timeout=timeout)
        return proc.returncode, (stdout + stderr)
    except subprocess.TimeoutExpired as e:
        if proc is not None:
            _kill_process_group(proc)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
        else:
            stdout, stderr = "", ""
        partial = ""
        for chunk in (getattr(e, "stdout", None), getattr(e, "stderr", None)):
            if isinstance(chunk, bytes):
                partial += chunk.decode(errors="replace")
            elif isinstance(chunk, str):
                partial += chunk
        return 124, f"{partial}{stdout}{stderr}\nTIMEOUT after {timeout}s"
    finally:
        if proc is not None and _ACTIVE_STAGE_PROC is proc:
            _ACTIVE_STAGE_PROC = None


def parse_instance_id(stdout: str) -> int | None:
    for line in stdout.splitlines():
        if line.startswith("INSTANCE_ID="):
            try:
                return int(line.split("=", 1)[1].strip())
            except ValueError:
                return None
    return None


def destroy(instance_id: int) -> None:
    """Best-effort destroy. Does not raise."""
    try:
        # subprocess-no-check-OK: best-effort destroy; ignore failures
        subprocess.run(
            ["bash", "-c", f"echo y | .venv/bin/vastai destroy instance {instance_id}"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
            check=False,
        )
    except Exception:
        pass


def _label_slug(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._")
    return slug or "lane"


def normalize_attempt_label(label: str) -> str:
    """Strip this launcher's per-attempt suffix from a Vast label."""
    return re.sub(r"_a\d+$", "", str(label or "").strip())


def logical_lane_key(label: str) -> str:
    """Collapse retry timestamp variants to one duplicate-spend key.

    Operators often append queue/run suffixes like ``_q1_20260430T211406Z``
    or ``_q1c_20260430T211553Z`` when relaunching a lane. Those are distinct
    Vast labels but the same logical spend lane, so they must share one lock
    and live-instance duplicate guard.
    """
    key = normalize_attempt_label(label)
    key = re.sub(r"_q[0-9A-Za-z]+(?:_\d{8}T\d{6}Z)?$", "", key)
    key = re.sub(r"_\d{8}T\d{6}Z$", "", key)
    key = re.sub(r"(_\d{4}-\d{2}-\d{2})_[a-z]$", r"\1", key)
    return key


class LaunchLock:
    """Advisory single-flight lock for one logical lane label."""

    def __init__(self, label: str) -> None:
        LOCK_DIR.mkdir(parents=True, exist_ok=True)
        self.logical_key = logical_lane_key(label)
        self.path = LOCK_DIR / f"{_label_slug(self.logical_key)}.lock"
        self._fh = None

    def __enter__(self) -> "LaunchLock":
        self._fh = self.path.open("a+")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError(
                "another launch_lane_with_retry process already holds "
                f"{self.path} for logical lane {self.logical_key!r}"
            ) from e
        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(
            f"pid={os.getpid()} started_at={time.time():.3f} "
            f"logical_key={self.logical_key}\n"
        )
        self._fh.flush()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        if self._fh is not None:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None


def live_instances_with_label_prefix(label: str) -> tuple[list[dict], str | None]:
    """Return active Vast instances that already look like attempts for label."""
    if not VASTAI.exists():
        return [], f"missing Vast.ai CLI at {VASTAI}"
    try:
        proc = subprocess.run(
            [str(VASTAI), "show", "instances", "--raw"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
    except subprocess.SubprocessError as e:
        return [], f"vastai show instances failed before execution: {e}"
    if proc.returncode != 0:
        return [], f"vastai show instances failed: {proc.stderr.strip()[:300]}"
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return [], f"vastai show instances returned non-JSON output: {e}"
    if not isinstance(data, list):
        return [], "vastai show instances returned non-list JSON"

    logical_key = logical_lane_key(label)
    attempts_prefix = f"{label}_a"
    matches: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        item_label = str(item.get("label") or "")
        if (
            item_label == label
            or item_label.startswith(attempts_prefix)
            or logical_lane_key(item_label) == logical_key
        ):
            matches.append(item)
    return matches, None


def _repo_text(rel_path: str, *, repo_root: Path | None = None) -> str:
    path = (repo_root or REPO_ROOT) / rel_path
    try:
        return path.read_text(errors="ignore")
    except OSError:
        return ""


def lane_forensic_clearance_violations(
    logical_key: str,
    *,
    repo_root: Path | None = None,
) -> list[str]:
    """Return unmet clearance requirements for lane-level forensic holds.

    These lanes were placed on hold because prior runs either lacked exact
    archive custody/adjudication (Lane 19) or were static-fallback/no-op
    codec measurements (Lane 20). Clearing the JSON hold is therefore not
    enough; the launch preflight also verifies that the required repair
    markers are present in the lane script and relevant runtime path.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []

    if logical_key == LANE_19_HOLD_KEY:
        script = _repo_text("scripts/remote_lane_19_logit_margin.sh", repo_root=root)
        profiles = _repo_text("src/tac/profiles.py", repo_root=root)
        combined = script + "\n" + profiles
        required_markers = {
            "deterministic archive build": (
                "ZipInfo" in script
                and "date_time=(1980, 1, 1, 0, 0, 0)" in script
                and "writestr" in script
                and "archive_manifest.json" in script
            ),
            "JSON adjudication": (
                "scripts/adjudicate_contest_auth_eval.py" in script
                and "--contest-json" in script
                and "contest_auth_eval.json" in script
                and "score_recomputed_from_components" in script
            ),
            "current frontier score gate": (
                "1.043987524793892" in script
                and "686635" in script
                and "--regression-threshold" in script
            ),
            "current frontier component gates": (
                "0.00346442" in script
                and "0.00400656" in script
                and "--max-posenet-relative" in script
                and "--max-segnet-relative" in script
            ),
            "corrected Lane 19 provenance/comments": (
                "forensic_hold_clearance_requirements" in script
                and "logit_margin_weight': '10.0" in script
                and "kl_distill_weight': '0.0" in script
                and "Lane G v3 PFP16 A++ frontier" in combined
            ),
        }
        for label, ok in required_markers.items():
            if not ok:
                violations.append(f"{logical_key}: missing {label}")
        stale_markers = (
            "KL distill (not as replacement)",
            "logit_margin_weight=0.1",
            "'logit_margin_weight': '0.0 -> 0.1'",
            "anchor baseline: 1.05 [contest-CUDA]",
        )
        for marker in stale_markers:
            if marker in combined:
                violations.append(f"{logical_key}: stale provenance/comment marker {marker!r}")

    elif logical_key == LANE_20_HOLD_KEY:
        script = _repo_text("scripts/remote_lane_20_balle.sh", repo_root=root)
        inflate = _repo_text("submissions/robust_current/inflate_renderer.py", repo_root=root)
        required_markers = {
            "non-static byte precheck": (
                "FATAL_NON_STATIC_BYTE_PRECHECK_FAILED" in script
                and "non_static_byte_precheck.json" in script
                and "BALLE_BEATS_STATIC" in script
                and "best_full_balle_bytes" in script
                and "static_baseline_bytes" in script
            ),
            "real BHv1 archive build": (
                "BHV1_ARCHIVE_INTEGRATION_READY" in script
                and "FATAL_BHV1_ARCHIVE_INFLATE_INTEGRATION_MISSING" not in script
                and "renderer.bhv1" in script
                and "BHv1 archive member" in script
                and "encode_qints_balle_auto" in script
            ),
            "inflate-side BHv1 decode integration": (
                "decode_qints_balle" in inflate
                and "BHv1" in inflate
                and "renderer.bhv1" in inflate
            ),
        }
        for label, ok in required_markers.items():
            if not ok:
                violations.append(f"{logical_key}: missing {label}")
        stale_markers = (
            "Stage 3 runs only when Ballé beats static",
            "currently a placeholder",
            "Lane 20 codec ships ZERO bytes",
            "Skipping Stage 3+4",
        )
        for marker in stale_markers:
            if marker in script:
                violations.append(f"{logical_key}: stale/no-op marker {marker!r}")

    return violations


def dispatch_hold_for_label(label: str) -> dict | None:
    """Return a matching fail-closed dispatch hold for this logical lane."""
    logical_key = logical_lane_key(label)
    if not DISPATCH_HOLDS_PATH.exists():
        clearance_violations = lane_forensic_clearance_violations(logical_key)
        if logical_key in FORENSIC_HOLD_KEYS and clearance_violations:
            return {
                "logical_key": logical_key,
                "reason": (
                    "forensic dispatch hold file is missing and clearance "
                    "requirements are still unmet"
                ),
                "clearance_violations": clearance_violations,
            }
        return None
    try:
        payload = json.loads(DISPATCH_HOLDS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {
            "logical_key": logical_key,
            "reason": f"unreadable dispatch hold file: {DISPATCH_HOLDS_PATH}",
        }
    holds = payload.get("holds") if isinstance(payload, dict) else payload
    if not isinstance(holds, list):
        return {
            "logical_key": logical_key,
            "reason": f"malformed dispatch hold file: {DISPATCH_HOLDS_PATH}",
        }
    for item in holds:
        if not isinstance(item, dict):
            continue
        hold_key = str(item.get("logical_key") or "")
        hold_prefix = str(item.get("logical_prefix") or "")
        if hold_key == logical_key or (hold_prefix and logical_key.startswith(hold_prefix)):
            clearance_violations = lane_forensic_clearance_violations(logical_key)
            if bool(item.get("cleared", False)):
                if clearance_violations:
                    return {
                        "logical_key": logical_key,
                        "reason": (
                            "forensic hold was marked cleared, but lane "
                            "clearance requirements are still unmet"
                        ),
                        "clearance_violations": clearance_violations,
                    }
                return None
            return item
    clearance_violations = lane_forensic_clearance_violations(logical_key)
    if logical_key in FORENSIC_HOLD_KEYS and clearance_violations:
        return {
            "logical_key": logical_key,
            "reason": (
                "forensic dispatch hold entry is missing and clearance "
                "requirements are still unmet"
            ),
            "clearance_violations": clearance_violations,
        }
    return None


def lane_script_has_retraining_payload(
    lane_script: str | Path,
    *,
    repo_root: Path | None = None,
) -> bool:
    root = repo_root or REPO_ROOT
    path = Path(lane_script)
    if not path.is_absolute():
        path = root / path
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return False
    return bool(RETRAINING_SCRIPT_RE.search(text))


def lane12_l2_clearance_violations(
    *,
    clearance_path: Path | None = None,
) -> list[str]:
    path = clearance_path or LANE12_L2_CLEARANCE_PATH
    if not path.exists():
        return [
            f"missing Lane 12 L2 clearance packet: {path}. "
            "Create this only after Lane 12/Alpha records L2 evidence, "
            "geometry gates, and Grand Council clean-pass review."
        ]
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return [f"unreadable Lane 12 L2 clearance packet {path}: {e}"]
    if not isinstance(payload, dict):
        return [f"Lane 12 L2 clearance packet {path} must be a JSON object"]

    violations: list[str] = []
    if payload.get("lane_id") not in {"lane_12_nerv_mask_codec", "lane_12_nerv"}:
        violations.append("lane_id must be lane_12_nerv_mask_codec or lane_12_nerv")
    if payload.get("cleared_for_retraining_unblock") is not True:
        violations.append("cleared_for_retraining_unblock must be true")
    if payload.get("lane12_l2") is not True:
        violations.append("lane12_l2 must be true")
    clean_passes = payload.get("grand_council_clean_passes")
    if not isinstance(clean_passes, int) or clean_passes < 3:
        violations.append("grand_council_clean_passes must be an integer >= 3")
    if not str(payload.get("evidence") or "").strip():
        violations.append("evidence must cite the Lane 12 L2 packet/artifacts")
    if payload.get("geometry_gate_passed") is not True:
        violations.append("geometry_gate_passed must be true")
    return violations


def lane12_retraining_gate_violations(
    *,
    lane_script: str | Path,
    label: str,
    repo_root: Path | None = None,
    clearance_path: Path | None = None,
) -> list[str]:
    logical_key = logical_lane_key(label)
    if logical_key.startswith(LANE12_LOGICAL_PREFIXES):
        return []
    if not lane_script_has_retraining_payload(lane_script, repo_root=repo_root):
        return []
    clearance_violations = lane12_l2_clearance_violations(
        clearance_path=clearance_path
    )
    if not clearance_violations:
        return []
    return [
        (
            f"{logical_key}: retraining dispatch is blocked until Lane 12/Alpha "
            "has explicit L2 clearance"
        ),
        *clearance_violations,
    ]


def _format_live_instance(item: dict) -> str:
    return (
        f"id={item.get('id')} label={item.get('label')} "
        f"status={item.get('actual_status') or item.get('cur_state')} "
        f"ssh={item.get('ssh_host')}:{item.get('ssh_port')} "
        f"$/hr={item.get('dph_total')}"
    )


def attempt_dispatch(args: argparse.Namespace, attempt: int) -> tuple[str, int | None, str]:
    """Single phase1+phase2 attempt.

    Returns (status, instance_id, log), where status is one of:
    "success", "retry", or "unknown". "unknown" means the lane may already
    be running remotely; the caller must not blindly launch a duplicate.
    """
    log: list[str] = []
    log.append(f"=== ATTEMPT {attempt} ===")

    if not getattr(args, "allow_existing_label_prefix", False):
        existing, error = live_instances_with_label_prefix(args.label)
        if error:
            log.append(f"UNKNOWN_EXISTING_LABEL_PREFIX: {error}")
            log.append("No new instance launched; retry only after manual state verification.")
            return "unknown", None, "\n".join(log)
        if existing:
            iid = existing[0].get("id")
            log.append("UNKNOWN_EXISTING_LABEL_PREFIX: live Vast instance(s) already match this lane label:")
            for item in existing:
                log.append(f"  {_format_live_instance(item)}")
            log.append("No duplicate retry launched; recover, verify, or destroy the existing instance first.")
            try:
                return "unknown", int(iid), "\n".join(log)
            except (TypeError, ValueError):
                return "unknown", None, "\n".join(log)

    # phase1 — find offer + create instance
    label = f"{args.label}_a{attempt}"
    p1_args = [
        PYBIN, str(LAUNCHER), "phase1",
        "--lane-script", args.lane_script,
        "--label", label,
        "--max-dph", str(args.max_dph),
    ]
    if args.predicted_band:
        p1_args += ["--predicted-band", str(args.predicted_band[0]), str(args.predicted_band[1])]
    if args.estimated_cost is not None:
        p1_args += ["--estimated-cost", str(args.estimated_cost)]

    rc, out = run_stage(p1_args, timeout=PHASE1_TIMEOUT_SECONDS)
    log.append(f"[phase1] rc={rc}")
    if rc != 0:
        log.append(out[-500:])
        return "retry", None, "\n".join(log)

    iid = parse_instance_id(out)
    if iid is None:
        log.append("FAIL: phase1 succeeded but no INSTANCE_ID parsed")
        log.append(out[-500:])
        return "retry", None, "\n".join(log)
    log.append(f"  instance_id={iid}")

    # phase2-wait — may take 3-5 min
    rc, out = run_stage(
        [PYBIN, str(LAUNCHER), "phase2-wait", "--instance-id", str(iid)],
        timeout=PHASE2_WAIT_TIMEOUT_SECONDS,
    )
    log.append(f"[phase2-wait] rc={rc}")
    if rc != 0:
        log.append(out[-300:])
        destroy(iid)
        return "retry", iid, "\n".join(log)

    # phase2-scp — build tarball + ship to remote
    rc, out = run_stage(
        [PYBIN, str(LAUNCHER), "phase2-scp", "--instance-id", str(iid),
         "--lane-script", args.lane_script],
        timeout=PHASE2_SCP_TIMEOUT_SECONDS,
    )
    log.append(f"[phase2-scp] rc={rc}")
    if rc != 0:
        log.append(out[-300:])
        destroy(iid)
        return "retry", iid, "\n".join(log)

    # phase2-extract — extract on remote + CUDA probe (auto-destroy on fail)
    rc, out = run_stage(
        [PYBIN, str(LAUNCHER), "phase2-extract", "--instance-id", str(iid),
         "--lane-script", args.lane_script],
        timeout=PHASE2_EXTRACT_TIMEOUT_SECONDS,
    )
    log.append(f"[phase2-extract] rc={rc}")
    if rc != 0:
        # phase2-extract auto-destroys on failure; no manual destroy needed
        log.append(out[-500:])
        return "retry", iid, "\n".join(log)

    # phase2-launch — subshell-detach lane (auto-destroys on NVDEC_BAD detection)
    rc, out = run_stage(
        [PYBIN, str(LAUNCHER), "phase2-launch", "--instance-id", str(iid),
         "--lane-script", args.lane_script],
        timeout=PHASE2_LAUNCH_TIMEOUT_SECONDS,
    )
    log.append(f"[phase2-launch] rc={rc}")
    if rc == 124:
        log.append(out[-500:])
        log.append(
            "UNKNOWN_REMOTE_STATE: phase2-launch timed out after the lane may "
            "have been detached in tmux. Not retrying blindly; verify or "
            "recover/destroy the instance explicitly."
        )
        return "unknown", iid, "\n".join(log)
    if rc != 0:
        log.append(out[-500:])
        # phase2-launch's NVDEC_BAD check destroys the instance itself
        return "retry", iid, "\n".join(log)

    log.append(f"✓ SUCCESS — instance {iid} running lane {args.label}")
    return "success", iid, "\n".join(log)


def main() -> int:
    _install_signal_handlers()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lane-script", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--max-dph", type=float, default=0.40)
    p.add_argument("--predicted-band", nargs=2, type=float, metavar=("LOW", "HIGH"))
    p.add_argument("--estimated-cost", type=float)
    p.add_argument("--max-retries", type=int, default=3,
                   help="Max attempts before giving up (default 3).")
    p.add_argument("--retry-delay", type=int, default=15,
                   help="Seconds between retries (default 15).")
    p.add_argument(
        "--allow-existing-label-prefix",
        action="store_true",
        help=(
            "Override the duplicate-spend guard for labels that already have "
            "live Vast instances. Use only after manual recovery/destroy."
        ),
    )
    p.add_argument(
        "--override-dispatch-hold",
        action="store_true",
        help=(
            "Override .omx/state/dispatch_holds.json for this launch. Use "
            "only after Grand Council clears the hold and records why."
        ),
    )
    p.add_argument(
        "--override-lane12-retraining-gate",
        action="store_true",
        help=(
            "Override the no-new-retraining-before-Lane12-L2 gate. Use only "
            "after recorded Grand Council approval for this specific launch."
        ),
    )
    args = p.parse_args()

    if not (REPO_ROOT / args.lane_script).exists():
        print(f"FATAL: lane script missing: {args.lane_script}", file=sys.stderr)
        return 2

    lane12_gate = lane12_retraining_gate_violations(
        lane_script=args.lane_script,
        label=args.label,
    )
    if lane12_gate and not args.override_lane12_retraining_gate:
        print(
            "FATAL_LANE12_RETRAINING_GATE: no new retraining dispatch before "
            "Lane 12/Alpha L2 clearance",
            file=sys.stderr,
        )
        for violation in lane12_gate[:12]:
            print(f"  - {violation}", file=sys.stderr)
        if len(lane12_gate) > 12:
            print(f"  - ... {len(lane12_gate) - 12} more", file=sys.stderr)
        print(
            "No Vast instance created. Build-only, harvest, and exact-eval "
            "lanes remain allowed; retraining lanes require "
            ".omx/state/lane12_nerv_l2_clearance.json or a recorded override.",
            file=sys.stderr,
        )
        return 2

    hold = dispatch_hold_for_label(args.label)
    if hold and not args.override_dispatch_hold:
        reason = hold.get("reason") or "dispatch hold is active"
        clearance_violations = hold.get("clearance_violations")
        print(
            "FATAL_DISPATCH_HOLD: "
            f"logical_key={logical_lane_key(args.label)!r} reason={reason}",
            file=sys.stderr,
        )
        if isinstance(clearance_violations, list) and clearance_violations:
            for violation in clearance_violations[:10]:
                print(f"  - {violation}", file=sys.stderr)
            if len(clearance_violations) > 10:
                print(
                    f"  - ... {len(clearance_violations) - 10} more",
                    file=sys.stderr,
                )
        print(
            "No Vast instance created. Clear the hold in "
            ".omx/state/dispatch_holds.json or pass --override-dispatch-hold "
            "only after recorded Grand Council approval.",
            file=sys.stderr,
        )
        return 2

    try:
        with LaunchLock(args.label):
            print(f"=== launch_lane_with_retry: {args.label} (max {args.max_retries} attempts) ===")
            for attempt in range(1, args.max_retries + 1):
                status, iid, log = attempt_dispatch(args, attempt)
                print(log)
                if status == "success":
                    print(f"\n✓ DISPATCHED: instance={iid} label={args.label} attempts={attempt}")
                    return 0
                if status == "unknown":
                    print(
                        f"\n? UNKNOWN_REMOTE_STATE: instance={iid} label={args.label}. "
                        "No duplicate retry launched; inspect the remote logs or recover artifacts.",
                        file=sys.stderr,
                    )
                    return 3
                if attempt < args.max_retries:
                    print(f"  retrying in {args.retry_delay}s...\n")
                    time.sleep(args.retry_delay)
    except RuntimeError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2

    print(f"\n✗ FAILED: {args.max_retries} attempts exhausted for {args.label}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
