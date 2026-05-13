"""Run a 100-epoch Modal smoke BEFORE the full operator-authorize dispatch.

PHASE-B1-PIVOT bug-class anchor (2026-05-12). Two consecutive 2000-epoch
sane_hnerv Modal A100 dispatches crashed rc=1 within 15s and 72s
respectively - burned $0.30 + a harvest slot each. A 100-epoch smoke would
have caught the integration failure for the same $0.30, then the full
canary could have been gated on smoke-green.

This tool is the canonical wrapper for the smoke-before-full pattern.
Catalog #167 (`check_substrate_dispatch_uses_smoke_before_full_pattern`)
refuses any operator-authorize wrapper that fires a "full" canary
(`cost_band.epochs >= 1000`) without routing through this tool first.

Mechanism
---------

Given a recipe name + smoke override args, this tool:

1. **Smoke phase** - invokes ``tools/operator_authorize.py --recipe <name>``
   with ``SMOKE_EPOCHS`` (default 100) instead of ``cost_band.epochs``,
   ``--gpu T4`` (cheaper than A100 for the smoke), and ``--timeout-hours 1``.
2. **Smoke validation** - polls the resulting Modal call_id via
   ``experiments/modal_recover_lane.py`` until the run completes (rc=0)
   OR fails. On failure, prints the failure-class diagnosis and exits
   non-zero (operator must fix BEFORE the $5-15 full dispatch).
3. **Full phase** - only if smoke green: invokes
   ``tools/operator_authorize.py --recipe <name>`` (unchanged, the recipe's
   own ``cost_band.epochs`` and ``cost_band.gpu`` apply).

Cost band
---------

* Smoke: 100 epoch / T4 / sane_hnerv class approximately $0.30 (hand-calibrated).
* Full: 2000 epoch / A100 / sane_hnerv class approximately $5-8 (from the recipe).

The smoke-vs-full bucket separation in the cost-band posterior is honored:
each phase appends its OWN anchor with ``epochs`` matching what was
actually dispatched.

Same-line waiver
----------------

Operator wrappers may carry ``# SMOKE_BEFORE_FULL_OK:<reason>`` if the
trainer has >=3 successful Modal anchors at the target config (cost band
is empirically calibrated; smoke would not surface new info).

Scope of protection (R1 Low #5 clarification, 2026-05-13)
---------------------------------------------------------

This tool's smoke gate CATCHES:

* Infrastructure crashes (rc != 0)
* Archive-missing failures (the canonical inflate.sh "FATAL: archive
  missing" class)
* Scorer-load failures at training-startup time
* Mount/dependency import failures (e.g. missing wheel, missing video,
  wrong CUDA driver)
* Worker source staleness (Catalog #166 ledger landed at startup)
* Any failure mode that surfaces in the FIRST 100 epochs

This tool's smoke gate does NOT catch:

* Architectural divergence where the 100-epoch smoke score is plausible
  in-band but the 2000-epoch full run diverges (e.g. an instability that
  develops slowly, or a regime change as ρ ramps in Lagrangian training)
* Score regressions that only emerge after EMA stabilization
* OOM that only triggers at larger-than-smoke batch sizes
* Late-stage NaN that takes 1000+ epochs to develop

Smoke is NECESSARY but NOT SUFFICIENT. Operators MUST still:

* Read the smoke proxy_score + check it tracks the expected band
* Watch the first full-run epoch milestones for in-band proxy_score
* Cancel the full dispatch if smoke score is implausibly high/low even
  though the run completed rc=0

For architectural-divergence detection at the 2000-epoch scale, see the
"Operator gates must be wired and used" CLAUDE.md non-negotiable and the
Phase 2 trainer-flag-manifest gates (Catalog #151 / #152).

Cross-references
----------------

* Catalog #167 (the STRICT preflight gate)
* CLAUDE.md "Auth eval EVERYWHERE" Pose TTO clause (the proxy-auth-gap
  smoke principle this tool generalizes to substrate dispatch)
* ``feedback_phase_b1_pivot_modal_source_staleness_fix_LANDED_20260512.md``
* ``feedback_fix_wave_1_r1_findings_LANDED_20260513.md`` (the R1 Low #5
  clarification this docstring section addresses)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

_REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(_REPO_ROOT)

from tac.auth_eval_result import parse_auth_eval_score_claim  # noqa: E402

DEFAULT_SMOKE_EPOCHS = 100
DEFAULT_SMOKE_GPU = "T4"
DEFAULT_SMOKE_TIMEOUT_HOURS = 1.0
DEFAULT_SMOKE_POLL_INTERVAL_SECONDS = 30
DEFAULT_SMOKE_MAX_WAIT_SECONDS = 1800  # 30 min
SMOKE_WAIT_MARGIN_SECONDS = 300
# Per CLAUDE.md SIREN audit 2026-05-13 DEFECT #2: previously (0.0, 10.0) was
# trivially wide — it only caught NaN / Inf. The contest's frontier score
# range is ~0.18-0.25; bands below 0.05 are physically implausible (the
# scorer architecture floor is ~0.001 for distortion + ~0.013 for rate);
# bands above 5.0 indicate trainer crash or output-shape mismatch (a normal
# untrained model scores ~1.0-3.0). Recipes can override this default by
# declaring `predicted_band: [lo, hi]` — see _resolve_smoke_band.
SMOKE_PLAUSIBLE_SCORE_BAND = (0.05, 5.0)


def _resolve_recipe_path(recipe_name: str, repo_root: Path) -> Path:
    """Return the path to ``.omx/operator_authorize_recipes/<name>.yaml``."""
    recipes_dir = repo_root / ".omx" / "operator_authorize_recipes"
    candidate = (
        recipes_dir / recipe_name
        if recipe_name.endswith(".yaml")
        else recipes_dir / f"{recipe_name}.yaml"
    )
    if not candidate.is_file():
        raise FileNotFoundError(
            f"recipe not found: {candidate} (recipe={recipe_name!r})"
        )
    return candidate


def _resolve_smoke_band(
    recipe_text: str,
    *,
    default_band: tuple[float, float] = SMOKE_PLAUSIBLE_SCORE_BAND,
) -> tuple[float, float]:
    """Return the score band the smoke validator should enforce.

    Per CLAUDE.md SIREN audit 2026-05-13 DEFECT #8 (predicted_band declared
    in remote driver but never enforced) + DEFECT #2 (global default
    trivially wide): recipes can declare a smoke-specific
    ``smoke_score_band: [lo, hi]``. If absent, they can declare a
    council-approved full-run band via ``predicted_band: [lo, hi]`` (the same
    shape the remote driver writes to provenance.json). When neither is
    present, the global default applies.

    The recipe's band is widened by ~50% on each side because the smoke
    runs at lower epochs (typically 100 vs full's 2000) and may not reach
    the same Pareto operating point — the council prediction is for the
    FULL dispatch, not the smoke. Widening avoids false-negative smoke
    reds when the smoke is healthy but undertrained.
    """
    # Parse only the top-level `predicted_band: [...]` line.
    # We avoid a full YAML import to keep this stdlib-only.
    import re

    m = None
    field_used = ""
    for field in ("smoke_score_band", "predicted_band"):
        m = re.search(
            rf'^{field}\s*:\s*\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\]\s*$',
            recipe_text,
            re.MULTILINE,
        )
        if m:
            field_used = field
            break
    if not m:
        return default_band
    try:
        lo = float(m.group(1))
        hi = float(m.group(2))
    except (TypeError, ValueError):
        return default_band
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi and lo >= 0):
        return default_band
    if field_used == "smoke_score_band":
        return (lo, hi)
    # Widen by 50% on each side; floor at 0.0; cap at 10.0.
    span = hi - lo
    widened_lo = max(0.0, lo - 0.5 * span)
    widened_hi = min(10.0, hi + 0.5 * span)
    return (widened_lo, widened_hi)


def _epoch_env_var_from_recipe(recipe_text: str) -> str | None:
    """Return the env var name the trainer reads for epoch count.

    The convention: each substrate recipe declares `<NAME>_EPOCHS: "<N>"`
    inside its `env_overrides` block. This helper sniffs the first matching
    `*_EPOCHS:` line so the smoke phase can override just the epoch count.
    """
    for line in recipe_text.splitlines():
        stripped = line.strip()
        if stripped.endswith("_EPOCHS:") or "_EPOCHS:" in stripped:
            # Parse "FOO_EPOCHS: \"2000\"" or "FOO_EPOCHS: 2000"
            head = stripped.split(":", 1)[0].strip()
            if head.endswith("_EPOCHS"):
                return head
    return None


def _recipe_requests_smoke_only(recipe_text: str) -> bool:
    """Return True when the recipe explicitly declares `smoke_only: true`."""
    for raw in recipe_text.splitlines():
        stripped = raw.split("#", 1)[0].strip()
        if not stripped.startswith("smoke_only:"):
            continue
        value = stripped.split(":", 1)[1].strip().lower()
        return value in {"1", "on", "true", "yes"}
    return False


def _lane_id_from_recipe(recipe_text: str) -> str:
    for line in recipe_text.splitlines():
        if line.startswith("lane_id:"):
            return line.split(":", 1)[1].strip().strip("\"'")
    return "lane_unknown"


def _expected_auth_artifact_markers(
    recipe_text: str,
    *,
    instance_job_id: str = "",
) -> tuple[str, ...]:
    """Return path markers that identify auth JSONs from this smoke run.

    Modal returns a flat artifact map keyed by repo-relative paths. The worker
    may copy historical ``experiments/results`` files into the writable
    workspace, so accepting any key containing ``auth_eval`` can validate stale
    score evidence. Recipe output-dir env overrides are the stable way to find
    the lane-local auth JSON path without hardcoding one substrate.
    """

    import re

    markers: list[str] = []
    if instance_job_id:
        markers.append(f"results/{instance_job_id}/")
        markers.append(instance_job_id)
    for match in re.finditer(
        r"^\s+[A-Z0-9_]*OUTPUT_DIR\s*:\s*/workspace/pact/([^\n#]+)",
        recipe_text,
        re.MULTILINE,
    ):
        rel = match.group(1).strip().strip("\"'")
        if rel:
            markers.append(rel.rstrip("/") + "/")
    out: list[str] = []
    seen: set[str] = set()
    for marker in markers:
        if marker and marker not in seen:
            seen.add(marker)
            out.append(marker)
    return tuple(out)


def _close_smoke_claim(
    *,
    repo_root: Path,
    recipe_text: str,
    instance_job_id: str,
    operator_handle: str,
    status: str,
    notes: str,
) -> None:
    """Append a terminal row for the smoke claim before same-lane full dispatch."""
    if not instance_job_id:
        print(
            "[smoke-before-full] WARN: cannot close smoke claim; "
            "instance_job_id was not parsed",
            file=sys.stderr,
        )
        return
    cmd = [
        sys.executable,
        "tools/claim_lane_dispatch.py",
        "claim",
        "--lane-id",
        _lane_id_from_recipe(recipe_text),
        "--platform",
        "modal",
        "--instance-job-id",
        instance_job_id,
        "--agent",
        f"{operator_handle}:run_modal_smoke_before_full",
        "--status",
        status,
        "--notes",
        notes,
        "--force",
    ]
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(
            "[smoke-before-full] WARN: failed to append terminal smoke claim "
            f"status={status}: {proc.stderr[-800:]}",
            file=sys.stderr,
        )


def _spawn_smoke_dispatch(
    recipe_path: Path,
    *,
    epoch_env_var: str | None,
    smoke_epochs: int,
    smoke_gpu: str,
    smoke_timeout_hours: float,
    operator_handle: str,
    repo_root: Path,
) -> tuple[str, str]:
    """Invoke the operator_authorize CLI in smoke mode.

    The smoke override is implemented by exporting:
      * MODAL_GPU=<smoke_gpu>      (T4 default; cheaper than A100 for smoke)
      * <epoch_env_var>=<smoke_epochs>  (typically 100)
      * MODAL_TIMEOUT_HOURS=<smoke_timeout_hours>  (1.0 default)
      * SMOKE_LABEL_SUFFIX=__smoke__<epochs>ep    (so cost-band bucket separates)

    The recipe's env_overrides block defaults each var so the override
    binds at process-environment level (highest precedence in the
    trainer's env->default ladder per Catalog #151).
    """
    label = f"smoke_{recipe_path.stem}_{int(time.time())}"
    env = {
        **os.environ,
        "MODAL_GPU": smoke_gpu,
        "MODAL_TIMEOUT_HOURS": str(smoke_timeout_hours),
        "SMOKE_LABEL_SUFFIX": f"__smoke__{smoke_epochs}ep",
    }
    if epoch_env_var:
        env[epoch_env_var] = str(smoke_epochs)
    cmd = [
        sys.executable,
        "tools/operator_authorize.py",
        "--recipe",
        recipe_path.stem,
        "--agent",
        f"{operator_handle}:run_modal_smoke_before_full",
        "--yes",
        "--label-suffix",
        f"__smoke__{smoke_epochs}ep",
        "--timeout-hours-override",
        str(smoke_timeout_hours),
        "--cost-band-epochs-override",
        str(smoke_epochs),
    ]
    print(f"[smoke-before-full] dispatching SMOKE: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print("[smoke-before-full] SMOKE dispatch FAILED:")
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:], file=sys.stderr)
        raise SystemExit(proc.returncode)
    print(proc.stdout[-2000:])
    # Extract the call_id from the dispatcher's output (printed as
    # "✓ DISPATCHED via .spawn() - call_id=fc-...")
    call_id = ""
    for line in proc.stdout.splitlines():
        if "call_id=" in line and "fc-" in line:
            call_id = line.split("call_id=", 1)[1].strip()
            break
    instance_job_id = ""
    for line in proc.stdout.splitlines():
        if "instance_job_id=" in line:
            instance_job_id = line.split("instance_job_id=", 1)[1].strip()
            break
    if not call_id:
        # Fall back to the sentinel file the dispatcher writes.
        for sentinel_dir in (repo_root / "experiments" / "results").glob(
            f"lane_*{label}*_modal"
        ):
            f = sentinel_dir / "modal_call_id.txt"
            if f.is_file():
                call_id = f.read_text().strip()
                break
    if not call_id:
        print(
            "[smoke-before-full] FATAL: could not extract call_id from dispatcher",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if not instance_job_id:
        print(
            "[smoke-before-full] FATAL: could not extract smoke instance_job_id "
            "from dispatcher",
            file=sys.stderr,
        )
        raise SystemExit(2)
    print(f"[smoke-before-full] SMOKE call_id={call_id} instance_job_id={instance_job_id}")
    return call_id, instance_job_id


def _wait_for_smoke_completion(
    call_id: str,
    *,
    repo_root: Path,
    poll_interval_s: int = DEFAULT_SMOKE_POLL_INTERVAL_SECONDS,
    max_wait_s: int = DEFAULT_SMOKE_MAX_WAIT_SECONDS,
) -> dict:
    """Poll Modal until the smoke call completes, returning the parsed result.

    Uses ``modal.functions.FunctionCall.from_id(call_id).get(timeout=2)`` per
    CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" canonical poll pattern.
    """
    import modal  # local import (modal not on every dev path)

    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        try:
            fn_call = modal.functions.FunctionCall.from_id(call_id)
            result = fn_call.get(timeout=2)
            return result if isinstance(result, dict) else {"raw": result}
        except modal.exception.OutputExpiredError as exc:
            print(
                f"[smoke-before-full] SMOKE call {call_id} expired (24h cache TTL)",
                file=sys.stderr,
            )
            raise SystemExit(3) from exc
        except TimeoutError:
            elapsed = max_wait_s - (deadline - time.monotonic())
            print(f"[smoke-before-full] still polling smoke (elapsed={elapsed:.0f}s)")
            time.sleep(poll_interval_s)
        except Exception as exc:
            # Catch-all to avoid silent infinite-loops on Modal API hiccups.
            print(
                f"[smoke-before-full] poll error {type(exc).__name__}: {exc}; "
                f"retrying in {poll_interval_s}s",
                file=sys.stderr,
            )
            time.sleep(poll_interval_s)
    raise TimeoutError(
        f"SMOKE call {call_id} did not complete within {max_wait_s}s"
    )


def _validate_smoke_result(
    result: dict,
    *,
    plausible_band: tuple[float, float] = SMOKE_PLAUSIBLE_SCORE_BAND,
    required_artifact_markers: tuple[str, ...] = (),
) -> tuple[bool, str]:
    """Return ``(green, diagnostic)``.

    Smoke is considered green when:
      * returncode == 0 AND timed_out is False
      * artifacts dict contains an `auth_eval_*.json` entry
      * the auth-eval JSON parses to a finite, in-band score
      * the auth-eval JSON explicitly authorizes a contest-CUDA score claim
    Otherwise red (with diagnostic text).
    """
    rc = result.get("returncode", -1)
    if rc != 0:
        return False, (
            f"SMOKE returncode={rc} (expected 0); "
            f"stderr_tail={result.get('stderr_tail', '')[-400:]!r}; "
            f"stdout_tail={result.get('stdout_tail', '')[-400:]!r}"
        )
    if result.get("timed_out"):
        return False, "SMOKE timed_out=True (training did not finish in budget)"
    artifacts = result.get("artifacts", {})
    all_auth_keys = [
        k for k in artifacts
        if "auth_eval" in k.lower() and k.lower().endswith(".json")
    ]
    auth_keys = [
        k for k in all_auth_keys
        if not required_artifact_markers
        or any(marker in k for marker in required_artifact_markers)
    ]
    if not auth_keys:
        if all_auth_keys and required_artifact_markers:
            return False, (
                "SMOKE artifacts contained auth-eval JSONs, but none were "
                "from the current smoke output path; refusing stale evidence. "
                f"required_markers={required_artifact_markers}; "
                f"auth_keys={sorted(all_auth_keys)[:8]}"
            )
        return False, (
            "SMOKE artifacts missing auth_eval_*.json - did the trainer "
            "reach the auth-eval stage?"
        )
    lo, hi = plausible_band
    diagnostics: list[str] = []
    out_of_band: list[str] = []
    for key in sorted(auth_keys):
        raw = artifacts[key]
        try:
            if isinstance(raw, bytes):
                payload = json.loads(raw.decode())
            elif isinstance(raw, str):
                payload = json.loads(raw)
            else:
                payload = raw
        except json.JSONDecodeError as exc:
            diagnostics.append(f"{key}:json_parse_failed:{exc!r}")
            continue
        if not isinstance(payload, dict):
            diagnostics.append(f"{key}:json_payload_not_object:{type(payload).__name__}")
            continue
        parsed = parse_auth_eval_score_claim(
            payload,
            required_score_axis="contest_cuda",
            require_component_recompute=True,
        )
        if parsed is None:
            diagnostics.append(
                f"{key}:not_contest_cuda_claim("
                f"score_axis={payload.get('score_axis')!r}, "
                f"lane_tag={payload.get('lane_tag')!r}, "
                f"score_claim={payload.get('score_claim')!r}, "
                f"score_claim_valid={payload.get('score_claim_valid')!r}, "
                f"exact_cuda_eval_complete={payload.get('exact_cuda_eval_complete')!r})"
            )
            continue
        score_f = parsed.score
        if lo <= score_f <= hi:
            return True, (
                "SMOKE GREEN: rc=0, "
                f"{parsed.lane_tag or '[contest-CUDA]'} auth_eval score={score_f:.4f} "
                f"in band [{lo}, {hi}] from {key}"
            )
        out_of_band.append(f"{key}:score={score_f}")

    if out_of_band:
        return False, (
            "SMOKE auth_eval contest-CUDA score OUT OF PLAUSIBLE BAND "
            f"[{lo}, {hi}]: {', '.join(out_of_band)}. "
            "Likely math/wiring bug; refusing to fire full canary."
        )
    return False, (
        "SMOKE artifacts did not contain any finite component-coherent "
        "contest-CUDA score claim; diagnostics="
        + "; ".join(diagnostics[:8])
    )


def _spawn_full_dispatch(
    recipe_path: Path,
    *,
    operator_handle: str,
    repo_root: Path,
) -> int:
    """Invoke the operator_authorize CLI in full mode (no env overrides)."""
    cmd = [
        sys.executable,
        "tools/operator_authorize.py",
        "--recipe",
        recipe_path.stem,
        "--agent",
        f"{operator_handle}:run_modal_full_after_smoke_green",
        "--yes",
    ]
    print(f"[smoke-before-full] dispatching FULL: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=repo_root, check=False)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-before-full canonical wrapper. Fire a 100-epoch ~$0.30 "
            "smoke against a Modal recipe, validate auth-eval green, then "
            "fire the full canary. Catalog #167 enforces this pattern."
        ),
    )
    parser.add_argument(
        "--recipe",
        required=True,
        help=(
            "Recipe name under .omx/operator_authorize_recipes/ "
            "(without .yaml suffix)."
        ),
    )
    parser.add_argument(
        "--smoke-epochs",
        type=int,
        default=DEFAULT_SMOKE_EPOCHS,
        help=f"Smoke epoch count (default {DEFAULT_SMOKE_EPOCHS}).",
    )
    parser.add_argument(
        "--smoke-gpu",
        default=DEFAULT_SMOKE_GPU,
        help=f"Smoke GPU class (default {DEFAULT_SMOKE_GPU}).",
    )
    parser.add_argument(
        "--smoke-timeout-hours",
        type=float,
        default=DEFAULT_SMOKE_TIMEOUT_HOURS,
        help=f"Smoke timeout (default {DEFAULT_SMOKE_TIMEOUT_HOURS}h).",
    )
    parser.add_argument(
        "--operator-handle",
        default="claude:phase_b1_pivot",
        help="Operator handle for lane-claim attribution.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Exit after smoke (do not fire full even on green).",
    )
    parser.add_argument(
        "--full-only",
        action="store_true",
        help="Skip smoke entirely (operator override; defeats the gate).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve recipe and print planned smoke/full dispatches without spawning Modal work.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_REPO_ROOT,
        help="Override repo root (test scaffold).",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    recipe_path = _resolve_recipe_path(args.recipe, repo_root)
    recipe_text = recipe_path.read_text()
    epoch_env_var = _epoch_env_var_from_recipe(recipe_text)
    recipe_smoke_only = _recipe_requests_smoke_only(recipe_text)
    smoke_only = args.smoke_only or recipe_smoke_only

    if args.dry_run:
        print("[smoke-before-full] --dry-run; no Modal dispatch")
        print(f"[smoke-before-full] recipe={recipe_path}")
        print(f"[smoke-before-full] epoch_env_var={epoch_env_var or '<none>'}")
        if recipe_smoke_only:
            print("[smoke-before-full] recipe declares smoke_only: true")
        print(
            "[smoke-before-full] would dispatch SMOKE: "
            f"recipe={args.recipe} smoke_epochs={args.smoke_epochs} "
            f"smoke_gpu={args.smoke_gpu} timeout_hours={args.smoke_timeout_hours}"
        )
        if smoke_only:
            print("[smoke-before-full] would stop after SMOKE because smoke-only is set")
        elif args.full_only:
            print("[smoke-before-full] would dispatch FULL only because --full-only is set")
        else:
            print("[smoke-before-full] would dispatch FULL only after SMOKE GREEN")
        return 0

    if recipe_smoke_only and args.full_only:
        print(
            "[smoke-before-full] FATAL: recipe declares smoke_only: true; "
            "--full-only would bypass the recipe scaffold guard.",
            file=sys.stderr,
        )
        return 6

    if args.full_only:
        print(
            "[smoke-before-full] WARNING: --full-only requested; SKIPPING "
            "smoke. Operator accepts the $5-15 risk of integration crash."
        )
        return _spawn_full_dispatch(
            recipe_path,
            operator_handle=args.operator_handle,
            repo_root=repo_root,
        )

    # Smoke phase
    call_id, smoke_instance_job_id = _spawn_smoke_dispatch(
        recipe_path,
        epoch_env_var=epoch_env_var,
        smoke_epochs=args.smoke_epochs,
        smoke_gpu=args.smoke_gpu,
        smoke_timeout_hours=args.smoke_timeout_hours,
        operator_handle=args.operator_handle,
        repo_root=repo_root,
    )
    print("[smoke-before-full] waiting for smoke to complete...")
    try:
        max_wait_s = max(
            DEFAULT_SMOKE_MAX_WAIT_SECONDS,
            int(args.smoke_timeout_hours * 3600) + SMOKE_WAIT_MARGIN_SECONDS,
        )
        result = _wait_for_smoke_completion(
            call_id,
            repo_root=repo_root,
            max_wait_s=max_wait_s,
        )
    except TimeoutError as exc:
        print(f"[smoke-before-full] {exc!s}", file=sys.stderr)
        _close_smoke_claim(
            repo_root=repo_root,
            recipe_text=recipe_text,
            instance_job_id=smoke_instance_job_id,
            operator_handle=args.operator_handle,
            status="failed_modal_smoke_timeout",
            notes=str(exc)[:200],
        )
        return 4

    # Per CLAUDE.md SIREN audit 2026-05-13 DEFECT #2 + #8: resolve the
    # smoke band from recipe predicted_band when present; otherwise use
    # the tightened global default (0.05, 5.0). The widened band accounts
    # for the smoke being undertrained relative to the full canary.
    smoke_band = _resolve_smoke_band(recipe_text)
    print(
        f"[smoke-before-full] enforcing smoke score band {smoke_band} "
        f"(default={SMOKE_PLAUSIBLE_SCORE_BAND}; "
        f"override_from_recipe={smoke_band != SMOKE_PLAUSIBLE_SCORE_BAND})"
    )
    artifact_markers = _expected_auth_artifact_markers(
        recipe_text,
        instance_job_id=smoke_instance_job_id,
    )
    green, diagnostic = _validate_smoke_result(
        result,
        plausible_band=smoke_band,
        required_artifact_markers=artifact_markers,
    )
    print(f"[smoke-before-full] SMOKE verdict: {diagnostic}")
    if not green:
        _close_smoke_claim(
            repo_root=repo_root,
            recipe_text=recipe_text,
            instance_job_id=smoke_instance_job_id,
            operator_handle=args.operator_handle,
            status="failed_modal_smoke_red",
            notes=diagnostic[:200],
        )
        print(
            "[smoke-before-full] FATAL: SMOKE RED - refusing full canary. "
            "Fix the integration first, then re-run.",
            file=sys.stderr,
        )
        return 5

    if smoke_only:
        _close_smoke_claim(
            repo_root=repo_root,
            recipe_text=recipe_text,
            instance_job_id=smoke_instance_job_id,
            operator_handle=args.operator_handle,
            status="completed_modal_smoke_green",
            notes="smoke green; smoke-only requested by CLI or recipe",
        )
        print("[smoke-before-full] smoke-only set; not firing full. Done.")
        return 0

    # Full phase
    _close_smoke_claim(
        repo_root=repo_root,
        recipe_text=recipe_text,
        instance_job_id=smoke_instance_job_id,
        operator_handle=args.operator_handle,
        status="completed_modal_smoke_green",
        notes="smoke green; terminal row closes smoke claim before full same-lane dispatch",
    )
    print(
        "[smoke-before-full] SMOKE GREEN - proceeding with FULL canary "
        f"(recipe={args.recipe})"
    )
    return _spawn_full_dispatch(
        recipe_path,
        operator_handle=args.operator_handle,
        repo_root=repo_root,
    )


__all__ = [
    "DEFAULT_SMOKE_EPOCHS",
    "DEFAULT_SMOKE_GPU",
    "DEFAULT_SMOKE_TIMEOUT_HOURS",
    "SMOKE_PLAUSIBLE_SCORE_BAND",
    "_epoch_env_var_from_recipe",
    "_resolve_recipe_path",
    "_resolve_smoke_band",
    "_validate_smoke_result",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
