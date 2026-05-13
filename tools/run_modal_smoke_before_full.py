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

DEFAULT_SMOKE_EPOCHS = 100
DEFAULT_SMOKE_GPU = "T4"
DEFAULT_SMOKE_TIMEOUT_HOURS = 1.0
DEFAULT_SMOKE_POLL_INTERVAL_SECONDS = 30
DEFAULT_SMOKE_MAX_WAIT_SECONDS = 1800  # 30 min
SMOKE_PLAUSIBLE_SCORE_BAND = (0.0, 10.0)


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


def _spawn_smoke_dispatch(
    recipe_path: Path,
    *,
    epoch_env_var: str | None,
    smoke_epochs: int,
    smoke_gpu: str,
    smoke_timeout_hours: float,
    operator_handle: str,
    repo_root: Path,
) -> str:
    """Invoke the operator_authorize CLI in smoke mode and return the call_id.

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
    print(f"[smoke-before-full] SMOKE call_id={call_id}")
    return call_id


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
) -> tuple[bool, str]:
    """Return ``(green, diagnostic)``.

    Smoke is considered green when:
      * returncode == 0 AND timed_out is False
      * artifacts dict contains an `auth_eval_*.json` entry
      * the auth-eval JSON parses to a finite, in-band score
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
    auth_keys = [
        k for k in artifacts
        if "auth_eval" in k.lower() and k.lower().endswith(".json")
    ]
    if not auth_keys:
        return False, (
            "SMOKE artifacts missing auth_eval_*.json - did the trainer "
            "reach the auth-eval stage?"
        )
    raw = artifacts[auth_keys[0]]
    try:
        if isinstance(raw, bytes):
            payload = json.loads(raw.decode())
        elif isinstance(raw, str):
            payload = json.loads(raw)
        else:
            payload = raw
    except json.JSONDecodeError as exc:
        return False, f"SMOKE auth_eval JSON parse failed: {exc!r}"
    score = payload.get("score") or payload.get("total_score") or payload.get("contest_score")
    if score is None:
        return False, f"SMOKE auth_eval has no recognizable score field: keys={list(payload.keys())}"
    try:
        score_f = float(score)
    except (TypeError, ValueError):
        return False, f"SMOKE auth_eval score not numeric: {score!r}"
    lo, hi = plausible_band
    if not (lo <= score_f <= hi):
        return False, (
            f"SMOKE auth_eval score {score_f} OUT OF PLAUSIBLE BAND "
            f"[{lo}, {hi}] - likely indicates a math/wiring bug, not a "
            "research result; refusing to fire full canary"
        )
    return True, (
        f"SMOKE GREEN: rc=0, auth_eval score={score_f:.4f} in band [{lo}, {hi}]"
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
    call_id = _spawn_smoke_dispatch(
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
        result = _wait_for_smoke_completion(call_id, repo_root=repo_root)
    except TimeoutError as exc:
        print(f"[smoke-before-full] {exc!s}", file=sys.stderr)
        return 4

    green, diagnostic = _validate_smoke_result(result)
    print(f"[smoke-before-full] SMOKE verdict: {diagnostic}")
    if not green:
        print(
            "[smoke-before-full] FATAL: SMOKE RED - refusing full canary. "
            "Fix the integration first, then re-run.",
            file=sys.stderr,
        )
        return 5

    if args.smoke_only:
        print("[smoke-before-full] --smoke-only set; not firing full. Done.")
        return 0

    # Full phase
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
    "_validate_smoke_result",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
