#!/usr/bin/env python3
"""Canonical operator-authorize entry point (FIX-G: T1-C wrapper unification).

Per simplification audit `a2a901c4f43d66a74` (operator approved 2026-05-12),
the 10 ``scripts/operator_authorize_*.sh`` wrappers (~1,497 LOC, ~70%
structurally duplicate) collapse to ONE canonical Python entry point + N
small YAML recipes.

The canonical sequence each wrapper executes is:

    1. Resolve the cost band via ``tac.cost_band_calibration.predict()``
       (with a hand-calibrated fallback when the posterior is cold-start)
    2. Show the operator confirmation banner (cost band, predicted delta, risk,
       envelope status) and read y/N
    3. Run a pre-flight required-input-files validation
       (``tools/validate_dispatch_required_inputs.py``) when the recipe
       declares ``required_input_files``
    4. Acquire a lane claim via ``tools/claim_lane_dispatch.py claim`` per
       CLAUDE.md "CROSS-AGENT DISPATCH COORDINATION"
    5. Invoke the platform-specific dispatcher (Modal, Vast.ai, Lightning,
       Kaggle, Azure, or GHA) wiring the env->CLI flag ladder from the
       recipe's ``env_overrides`` block
    6. Print the harvest / next-step instructions

This is Part 1 of FIX-G. Part 2 lands the 10 YAML recipes (one per existing
wrapper) under ``.omx/operator_authorize_recipes/``. Part 3 lands Catalog
#162 ``check_operator_authorize_canonical_use`` STRICT preflight gate.

Recipe YAML schema (canonical fields)::

    schema_version: 1
    name: <short slug, matches filename without .yaml>
    lane_id: <e.g. lane_t1_balle_128k_endtoend>
    summary: <one-liner shown in the banner>
    platform: modal | vastai | lightning | kaggle | azure | gha | local | none
    gpu: T4 | A10G | A100 | H100 | RTX_4090 | M5_Max | cpu | none
    cost_band:
      epochs: 3000
      all_flags_on: true
      hand_calibrated_fallback_p50_usd: 8.00
      platform_key: modal       # optional; defaults to platform field
      gpu_key: T4               # optional; defaults to gpu field
    predicted_delta: "-0.012 ± 0.007"
    predicted_delta_basis: <citation>
    risk: <one-liner>
    envelope_status: <human banner string>
    remote_driver: scripts/remote_lane_t1_balle_endtoend.sh   # optional
    required_input_files: []     # optional; passed to validate_dispatch_required_inputs.py
    env_overrides: {KEY: VALUE}  # optional; threaded into the dispatch invocation
    dependencies: []             # optional; for documentation only
    timeout_hours: 4.0           # optional; default 4.0 hours
    modal:                       # optional; only when platform == modal
      lane_script: scripts/remote_lane_<id>.sh
      cost_band_trainer: experiments/train_<id>.py
      cost_band_epochs: 3000
      cost_band_batch_size: 16
      cost_band_all_flags_on: true
    vastai: {launcher: scripts/launch_lane_on_vastai.py}  # optional
    kaggle:                      # optional; only when platform == kaggle
      kernel_script: experiments/kaggle_<id>.py
      kernel_metadata_template: ...
    gha:                         # optional; only when platform == gha
      trigger_command: tools/trigger_gha_cpu_eval.py
    notes: |
      Multi-line operator-facing rationale (Amdahl decomposition, etc.)

CLI usage::

    .venv/bin/python tools/operator_authorize.py --recipe <name>
    .venv/bin/python tools/operator_authorize.py --recipe <name> --dry-run
    .venv/bin/python tools/operator_authorize.py --list

Per CLAUDE.md "Operator gates must be wired and used", this script is the
canonical wrapper for every operator-authorize action. The 10 legacy .sh
wrappers become thin shims that delegate to ``--recipe <name>`` for one
release cycle, then can be removed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPES_DIR = REPO_ROOT / ".omx/operator_authorize_recipes"
VENV_PYTHON = REPO_ROOT / ".venv/bin/python"


def _python_bin() -> str:
    """Return the venv python if present, else system python3."""
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable or "python3"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Parse a recipe YAML file.

    Per CLAUDE.md "Tooling - non-negotiable": prefer the venv-managed PyYAML;
    fall back to a minimal hand-parser if PyYAML is unavailable (recipes are
    intentionally simple - flat scalars + a few nested dicts/lists).
    """
    try:
        import yaml  # type: ignore[import-untyped]

        with path.open() as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise SystemExit(
                f"[operator-authorize] FATAL: recipe {path} did not parse to a dict"
            )
        return data
    except ImportError:
        raise SystemExit(
            "[operator-authorize] FATAL: PyYAML not installed. Run "
            "`uv pip install pyyaml` (or `pip install pyyaml`)."
        ) from None


def _resolve_utc_label() -> str:
    """Produce a deterministic UTC timestamp label for instance_job_id."""
    import datetime

    return datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class CostBandPrediction:
    """Mirror of ``tac.cost_band_calibration.predict`` return value."""

    p10_cost_usd: float
    p50_cost_usd: float
    p90_cost_usd: float
    n_anchors: int
    confidence_tag: str
    source: str  # "posterior" or "hand_calibrated_fallback"


def _predict_cost_band(
    platform_key: str,
    gpu_key: str,
    epochs: int,
    all_flags_on: bool,
    hand_calibrated_fallback_p50_usd: float,
) -> CostBandPrediction:
    """Call ``tac.cost_band_calibration.predict`` defensively.

    The posterior at ``.omx/state/cost_band_posterior.jsonl`` is cold-start
    for many buckets; in that case predict() returns a hand_calibrated_fallback
    band that may have p50=0 if no hand-stub exists. We supplement with the
    recipe's own hand_calibrated_fallback_p50_usd in that case.
    """
    code = (
        "from tac.cost_band_calibration import predict\n"
        f"p = predict({platform_key!r}, {gpu_key!r}, {int(epochs)}, "
        f"all_flags_on={bool(all_flags_on)!r})\n"
        "import json\n"
        "print(json.dumps({"
        "'p10': p.p10_cost_usd, 'p50': p.p50_cost_usd, 'p90': p.p90_cost_usd, "
        "'n_anchors': p.n_anchors, 'confidence_tag': p.confidence_tag}))"
    )
    try:
        result = subprocess.run(
            [_python_bin(), "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(REPO_ROOT),
        )
        if result.returncode == 0 and result.stdout.strip():
            payload = json.loads(result.stdout.strip())
            p50 = float(payload["p50"])
            if p50 > 0:
                return CostBandPrediction(
                    p10_cost_usd=float(payload["p10"]),
                    p50_cost_usd=p50,
                    p90_cost_usd=float(payload["p90"]),
                    n_anchors=int(payload["n_anchors"]),
                    confidence_tag=str(payload["confidence_tag"]),
                    source="posterior",
                )
            # p50 == 0 -> cold-start bucket with no hand stub; use recipe fallback.
            return CostBandPrediction(
                p10_cost_usd=hand_calibrated_fallback_p50_usd * 0.5,
                p50_cost_usd=hand_calibrated_fallback_p50_usd,
                p90_cost_usd=hand_calibrated_fallback_p50_usd * 2.0,
                n_anchors=int(payload.get("n_anchors", 0)),
                confidence_tag="hand_calibrated_fallback",
                source="hand_calibrated_fallback",
            )
    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError):
        pass
    # predict() unavailable or errored - emit explicit hand-calibrated fallback.
    return CostBandPrediction(
        p10_cost_usd=hand_calibrated_fallback_p50_usd * 0.5,
        p50_cost_usd=hand_calibrated_fallback_p50_usd,
        p90_cost_usd=hand_calibrated_fallback_p50_usd * 2.0,
        n_anchors=0,
        confidence_tag="hand_calibrated_fallback",
        source="hand_calibrated_fallback",
    )


def _required_input_flag_values_from_recipe(recipe: Recipe) -> list[str]:
    """Return validator ``--flag-value`` args from recipe-required inputs."""

    out: list[str] = []
    raw = recipe.raw.get("required_input_files", []) or []
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        flag = str(entry.get("flag", "")).strip()
        default_path = entry.get("default_path")
        if flag and default_path:
            out.append(f"{flag}={default_path}")
    return out


def _validate_required_input_files(
    trainer_path: str,
    recipe: Recipe,
) -> None:
    """Run ``tools/validate_dispatch_required_inputs.py`` per Catalog #152.

    Fails closed (non-zero exit) if any ``required_input_file=True`` flag
    points to a missing default file. The local 100ms check catches the same
    bug class as the 2026-05-12 Modal A100 dispatch fc-01KREJST89QHFRWJXHAKXD850C
    BEFORE the GPU meter starts.
    """
    validator = REPO_ROOT / "tools/validate_dispatch_required_inputs.py"
    if not validator.exists():
        print(
            "[operator-authorize] WARN: required-input validator not found at "
            f"{validator}; skipping pre-flight",
            file=sys.stderr,
        )
        return
    result = subprocess.run(
        [
            _python_bin(),
            str(validator),
            "--trainer",
            trainer_path,
            *[
                f"--flag-value={flag_value}"
                for flag_value in _required_input_flag_values_from_recipe(recipe)
            ],
        ],
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        raise SystemExit(
            "[operator-authorize] FATAL: required input missing per "
            f"Catalog #152 (validator returned {result.returncode}); aborting "
            "before GPU dispatch"
        )


def _claim_lane(
    lane_id: str,
    platform: str,
    instance_job_id: str,
    agent: str,
    notes: str,
) -> None:
    """Acquire a lane claim per CLAUDE.md CROSS-AGENT DISPATCH COORDINATION."""
    helper = REPO_ROOT / "tools/claim_lane_dispatch.py"
    if not helper.exists():
        raise SystemExit(
            "[operator-authorize] FATAL: lane-claim helper missing at "
            f"{helper}"
        )
    result = subprocess.run(
        [
            _python_bin(),
            str(helper),
            "claim",
            "--lane-id",
            lane_id,
            "--platform",
            platform,
            "--instance-job-id",
            instance_job_id,
            "--agent",
            agent,
            "--status",
            "active_dispatch",
            "--notes",
            notes,
        ],
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        raise SystemExit(
            f"[operator-authorize] FATAL: lane claim failed "
            f"(returncode={result.returncode}); investigate "
            ".omx/state/active_lane_dispatch_claims.md"
        )


def _terminal_claim(
    *,
    lane_id: str,
    platform: str,
    instance_job_id: str,
    agent: str,
    status: str,
    notes: str,
) -> None:
    """Append a terminal claim row after a failed native dispatch."""
    helper = REPO_ROOT / "tools/claim_lane_dispatch.py"
    if not helper.exists():
        print(
            "[operator-authorize] WARN: lane-claim helper missing while trying "
            f"to close failed claim at {helper}",
            file=sys.stderr,
        )
        return
    result = subprocess.run(
        [
            _python_bin(),
            str(helper),
            "claim",
            "--force",
            "--lane-id",
            lane_id,
            "--platform",
            platform,
            "--instance-job-id",
            instance_job_id,
            "--agent",
            agent,
            "--status",
            status,
            "--notes",
            notes,
        ],
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        print(
            "[operator-authorize] WARN: failed to append terminal lane-claim "
            f"row (returncode={result.returncode}); investigate "
            ".omx/state/active_lane_dispatch_claims.md",
            file=sys.stderr,
        )


def _confirm(prompt: str, *, default_yes: bool = False) -> bool:
    """Read a y/N from stdin. Returns True only on explicit yes."""
    suffix = " [Y/n] " if default_yes else " [y/N] "
    try:
        ans = input(prompt + suffix).strip().lower()
    except EOFError:
        return default_yes
    if not ans:
        return default_yes
    return ans in {"y", "yes"}


def _format_cost_band(band: CostBandPrediction) -> str:
    return (
        f"${band.p10_cost_usd:.2f}/${band.p50_cost_usd:.2f}/${band.p90_cost_usd:.2f}"
        f"  (N={band.n_anchors}, {band.confidence_tag})"
    )


def _resolve_env_var(value: Any, default: Any) -> Any:
    """Resolve a recipe value that may be an env-var reference like ${MODAL_GPU:-T4}."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        body = value[2:-1]
        if ":-" in body:
            var, fallback = body.split(":-", 1)
            return os.environ.get(var, fallback)
        return os.environ.get(body, default)
    return value if value is not None else default


@dataclass
class Recipe:
    """In-memory recipe representation after env-var resolution."""

    name: str
    path: Path
    raw: dict[str, Any]

    @property
    def lane_id(self) -> str:
        return str(self.raw.get("lane_id", "lane_unknown"))

    @property
    def platform(self) -> str:
        return str(_resolve_env_var(self.raw.get("platform", "none"), "none")).lower()

    @property
    def summary(self) -> str:
        return str(self.raw.get("summary", ""))

    @property
    def gpu(self) -> str:
        return str(_resolve_env_var(self.raw.get("gpu", "none"), "none"))

    @property
    def predicted_delta(self) -> str:
        return str(self.raw.get("predicted_delta", "<not declared>"))

    @property
    def risk(self) -> str:
        return str(self.raw.get("risk", "<not declared>"))

    @property
    def envelope_status(self) -> str:
        return str(self.raw.get("envelope_status", "<not declared>"))

    @property
    def remote_driver(self) -> str | None:
        v = self.raw.get("remote_driver")
        return str(v) if v else None

    @property
    def dispatch_enabled(self) -> bool:
        """Return whether this recipe is allowed to create provider work."""

        return bool(self.raw.get("dispatch_enabled", True))

    @property
    def dispatch_blockers(self) -> list[str]:
        """Return declared recipe blockers, normalized to strings."""

        raw = self.raw.get("dispatch_blockers", []) or []
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, list):
            return [str(item) for item in raw]
        return [str(raw)]

    @property
    def pre_promotion_blockers(self) -> list[str]:
        """Return declared pre-promotion blockers, normalized to strings."""

        raw = self.raw.get("pre_promotion_blockers", []) or []
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, list):
            return [str(item) for item in raw]
        return [str(raw)]

    @property
    def defer_reason(self) -> str:
        return str(self.raw.get("defer_reason", "")).strip()

    @property
    def timeout_hours(self) -> float:
        return float(self.raw.get("timeout_hours", 4.0))

    @property
    def notes(self) -> str:
        return str(self.raw.get("notes", "")).strip()


def _load_recipe(name: str) -> Recipe:
    """Locate + parse the named recipe."""
    path = RECIPES_DIR / f"{name}.yaml"
    if not path.exists():
        # Allow caller to pass an explicit path.
        explicit = Path(name)
        if explicit.exists() and explicit.suffix in {".yaml", ".yml"}:
            path = explicit.resolve()
        else:
            raise SystemExit(
                f"[operator-authorize] FATAL: recipe '{name}' not found at "
                f"{RECIPES_DIR}/{name}.yaml. List available recipes with "
                f"`{Path(__file__).name} --list`."
            )
    raw = _load_yaml(path)
    return Recipe(name=name, path=path, raw=raw)


def _recipe_dispatch_refusal(recipe: Recipe) -> str | None:
    """Return a fail-closed refusal reason for non-dispatchable recipes.

    Recipes are the operator-facing source of truth for dispatchability. A
    recipe that declares ``dispatch_enabled: false`` or still lists
    ``pre_promotion_blockers`` must not reach confirmation, claim creation, or
    provider dispatch. This is deliberately independent of provider-specific
    preflights so deferred research recipes also fail before cost prediction or
    provider setup can create misleading state.
    """

    if not recipe.dispatch_enabled:
        details: list[str] = ["dispatch_enabled=false"]
        if recipe.dispatch_blockers:
            details.append("dispatch_blockers=" + ", ".join(recipe.dispatch_blockers))
        if recipe.defer_reason:
            details.append("defer_reason=" + recipe.defer_reason.splitlines()[0])
        return "; ".join(details)
    if recipe.pre_promotion_blockers:
        return (
            "pre_promotion_blockers still declared: "
            + ", ".join(recipe.pre_promotion_blockers)
            + ". Clear these in the recipe only after the required exact anchors "
            "or custody artifacts land."
        )
    return None


def _list_recipes() -> int:
    """Print one line per recipe to stdout."""
    if not RECIPES_DIR.exists():
        print(f"(no recipes directory at {RECIPES_DIR})")
        return 0
    recipes = sorted(RECIPES_DIR.glob("*.yaml"))
    if not recipes:
        print(f"(no recipes in {RECIPES_DIR})")
        return 0
    for p in recipes:
        try:
            raw = _load_yaml(p)
        except SystemExit:
            print(f"  {p.stem:50s}  <unparseable>")
            continue
        platform = raw.get("platform", "?")
        summary = raw.get("summary", "")
        print(f"  {p.stem:50s}  [{platform:10s}]  {summary}")
    return 0


def _print_banner(
    recipe: Recipe,
    band: CostBandPrediction,
) -> None:
    """Print the operator confirmation banner."""
    print()
    print(f"=== operator-authorize: {recipe.name} ===")
    print()
    if recipe.summary:
        print(f"  summary:                 {recipe.summary}")
    print(f"  lane_id:                 {recipe.lane_id}")
    print(f"  platform:                {recipe.platform} ({recipe.gpu})")
    print(f"  cost band p10/p50/p90:   {_format_cost_band(band)}")
    print("                           Source: tac.cost_band_calibration.predict()")
    print("                           Posterior: .omx/state/cost_band_posterior.jsonl")
    print(f"                           Confidence: {band.confidence_tag}")
    print(f"  predicted delta:             {recipe.predicted_delta}")
    print(f"  risk:                    {recipe.risk}")
    print(f"  envelope status:         {recipe.envelope_status}")
    if recipe.notes:
        print()
        print("  notes:")
        for line in recipe.notes.splitlines():
            print(f"    {line}")
    print()


def _build_env_overrides(recipe: Recipe, instance_job_id: str) -> str:
    """Build the comma-separated env-override string for modal_train_lane.py."""
    raw_env = recipe.raw.get("env_overrides", {}) or {}
    parts: list[str] = []
    # Always include the instance_job_id binding so the remote driver can
    # correlate the dispatch claim with its INSTANCE_JOB_ID env var.
    if isinstance(raw_env, dict):
        for k, v in raw_env.items():
            resolved = os.environ.get(str(k), _resolve_env_var(v, v))
            # Substitute ${INSTANCE_JOB_ID} sentinel if present.
            if isinstance(resolved, str) and "${INSTANCE_JOB_ID}" in resolved:
                resolved = resolved.replace("${INSTANCE_JOB_ID}", instance_job_id)
            parts.append(f"{k}={resolved}")
    return ",".join(parts)


def _dispatch_modal(
    recipe: Recipe,
    instance_job_id: str,
    env_overrides: str,
    *,
    timeout_hours_override: float | None = None,
    cost_band_epochs_override: int | None = None,
) -> int:
    """Invoke ``.venv/bin/modal run --detach experiments/modal_train_lane.py``."""
    modal_cfg = recipe.raw.get("modal", {}) or {}
    lane_script = modal_cfg.get("lane_script") or recipe.remote_driver
    if not lane_script:
        raise SystemExit(
            "[operator-authorize] FATAL: modal recipe missing 'modal.lane_script' "
            "and 'remote_driver'"
        )
    if not (REPO_ROOT / lane_script).exists():
        raise SystemExit(
            "[operator-authorize] FATAL: canonical remote driver missing at "
            f"{lane_script}"
        )
    gpu = recipe.gpu if recipe.gpu and recipe.gpu != "none" else os.environ.get(
        "MODAL_GPU", "T4"
    )
    timeout_hours = timeout_hours_override or recipe.timeout_hours
    cmd = [
        ".venv/bin/modal",
        "run",
        "--detach",
        "experiments/modal_train_lane.py",
        "--lane-script",
        lane_script,
        "--label",
        instance_job_id,
        "--gpu",
        gpu,
        "--timeout-hours",
        f"{timeout_hours}",
        "--env-overrides",
        env_overrides,
    ]
    # Optional cost-band hooks for the modal_train_lane wrapper.
    if modal_cfg.get("cost_band_trainer"):
        cmd.extend(["--cost-band-trainer", str(modal_cfg["cost_band_trainer"])])
    cost_band_epochs = cost_band_epochs_override or modal_cfg.get("cost_band_epochs")
    if cost_band_epochs:
        cmd.extend(["--cost-band-epochs", str(cost_band_epochs)])
    if modal_cfg.get("cost_band_batch_size"):
        cmd.extend(
            ["--cost-band-batch-size", str(modal_cfg["cost_band_batch_size"])]
        )
    if modal_cfg.get("cost_band_all_flags_on", False):
        cmd.append("--cost-band-all-flags-on")
    print(f"[operator-authorize] dispatching: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def _dispatch_vastai(
    recipe: Recipe,
    instance_job_id: str,
    env_overrides: str,
) -> int:
    """Invoke the Vast.ai canonical launcher."""
    vastai_cfg = recipe.raw.get("vastai", {}) or {}
    launcher = vastai_cfg.get("launcher")
    if not launcher:
        for candidate in (
            "scripts/launch_lane_on_vastai.py",
            "tools/launch_lane_on_vastai.py",
        ):
            if (REPO_ROOT / candidate).exists():
                launcher = candidate
                break
    if not launcher:
        raise SystemExit(
            "[operator-authorize] FATAL: Vast.ai launcher not found; declare "
            "vastai.launcher: <path> in the recipe or install the canonical one"
        )
    lane_script = vastai_cfg.get("lane_script") or recipe.remote_driver
    if not lane_script:
        raise SystemExit(
            "[operator-authorize] FATAL: vastai recipe missing 'vastai.lane_script' "
            "and 'remote_driver'"
        )
    gpu = recipe.gpu if recipe.gpu and recipe.gpu != "none" else "RTX_4090"
    timeout_hours = recipe.timeout_hours
    cmd = [
        _python_bin(),
        launcher,
        "--lane-script",
        lane_script,
        "--label",
        instance_job_id,
        "--gpu",
        gpu,
        "--timeout-hours",
        f"{timeout_hours}",
        "--env-overrides",
        env_overrides,
    ]
    print(f"[operator-authorize] dispatching: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def _dispatch_local(
    recipe: Recipe,
    instance_job_id: str,
    env_overrides: str,
) -> int:
    """Invoke a local remote_driver directly (for in-process dispatch)."""
    lane_script = recipe.remote_driver
    if not lane_script:
        raise SystemExit(
            "[operator-authorize] FATAL: local recipe needs 'remote_driver'"
        )
    if not (REPO_ROOT / lane_script).exists():
        raise SystemExit(
            f"[operator-authorize] FATAL: remote_driver missing at {lane_script}"
        )
    # Build env dict from comma-separated overrides.
    env = dict(os.environ)
    if env_overrides:
        for pair in env_overrides.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                env[k.strip()] = v.strip()
    cmd = ["bash", lane_script]
    print(f"[operator-authorize] dispatching local: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(REPO_ROOT), env=env)


def _dispatch_noop(recipe: Recipe, instance_job_id: str, env_overrides: str) -> int:
    """No-op dispatcher for recipes that handle their own action (e.g. git push).

    These recipes use ``platform: none`` and rely on a custom 'action' block
    that the legacy .sh shim still owns. For now, this just prints a notice;
    the legacy shim continues to perform the actual push.
    """
    print(
        "[operator-authorize] recipe has platform=none; no auto-dispatch. "
        "The legacy shim (if present) handles the operator-side action."
    )
    return 0


def _native_dispatch_preflight(recipe: Recipe) -> None:
    """Validate native provider prerequisites before claiming a lane."""
    platform = recipe.platform
    if platform == "modal":
        modal_cfg = recipe.raw.get("modal", {}) or {}
        lane_script = modal_cfg.get("lane_script") or recipe.remote_driver
        if not lane_script:
            raise SystemExit(
                "[operator-authorize] FATAL: modal recipe missing 'modal.lane_script' "
                "and 'remote_driver'"
            )
        if not (REPO_ROOT / str(lane_script)).exists():
            raise SystemExit(
                "[operator-authorize] FATAL: canonical remote driver missing at "
                f"{lane_script}"
            )
        if not (REPO_ROOT / ".venv/bin/modal").exists():
            raise SystemExit(
                "[operator-authorize] FATAL: Modal CLI missing at .venv/bin/modal; "
                "install/sync dependencies before creating a dispatch claim"
            )
    elif platform in {"vastai", "vast"}:
        vastai_cfg = recipe.raw.get("vastai", {}) or {}
        launcher = vastai_cfg.get("launcher")
        if not launcher:
            for candidate in (
                "scripts/launch_lane_on_vastai.py",
                "tools/launch_lane_on_vastai.py",
            ):
                if (REPO_ROOT / candidate).exists():
                    launcher = candidate
                    break
        if not launcher or not (REPO_ROOT / str(launcher)).exists():
            raise SystemExit(
                "[operator-authorize] FATAL: Vast.ai launcher not found before "
                "claim; declare vastai.launcher or install the canonical launcher"
            )
        lane_script = vastai_cfg.get("lane_script") or recipe.remote_driver
        if not lane_script or not (REPO_ROOT / str(lane_script)).exists():
            raise SystemExit(
                "[operator-authorize] FATAL: Vast.ai lane script missing before "
                f"claim: {lane_script}"
            )
    elif platform == "local":
        lane_script = recipe.remote_driver
        if not lane_script or not (REPO_ROOT / str(lane_script)).exists():
            raise SystemExit(
                "[operator-authorize] FATAL: local remote_driver missing before "
                f"claim: {lane_script}"
            )


def _run_dispatch(
    recipe: Recipe,
    instance_job_id: str,
    *,
    timeout_hours_override: float | None = None,
    cost_band_epochs_override: int | None = None,
) -> int:
    """Route to the appropriate platform dispatcher."""
    env_overrides = _build_env_overrides(recipe, instance_job_id)
    platform = recipe.platform
    if platform == "modal":
        return _dispatch_modal(
            recipe,
            instance_job_id,
            env_overrides,
            timeout_hours_override=timeout_hours_override,
            cost_band_epochs_override=cost_band_epochs_override,
        )
    if platform in {"vastai", "vast"}:
        return _dispatch_vastai(recipe, instance_job_id, env_overrides)
    if platform == "local":
        return _dispatch_local(recipe, instance_job_id, env_overrides)
    if platform in {"none", "kaggle", "gha", "lightning", "azure"}:
        # These platforms have bespoke dispatch flows (gh release create,
        # kaggle kernels push, etc.) - for now, the legacy .sh shim handles
        # them. A future revision will add native handlers.
        return _dispatch_noop(recipe, instance_job_id, env_overrides)
    raise SystemExit(
        f"[operator-authorize] FATAL: unsupported platform '{platform}' in "
        f"recipe '{recipe.name}'"
    )


def _platform_has_native_dispatch(platform: str) -> bool:
    """Return True when this tool will actually start provider/local work."""

    return platform in {"modal", "vastai", "vast", "local"}


def _sanitize_terminal_status(status: str) -> str:
    return "".join(c if c.isalnum() or c in {"_", "-"} else "_" for c in status)[:80]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--recipe",
        help="Recipe name (filename without .yaml under .omx/operator_authorize_recipes/)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available recipes",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the operator confirmation + actual dispatch (print plan only)",
    )
    parser.add_argument(
        "--no-claim",
        action="store_true",
        help="Skip the lane-claim step (for recipes that handle their own coordination)",
    )
    parser.add_argument(
        "--agent",
        default="claude:operator_authorize",
        help="Agent string for the lane-claim ledger",
    )
    parser.add_argument(
        "--label-suffix",
        default="",
        help=(
            "Optional suffix appended to the generated instance_job_id. Used by "
            "smoke-before-full wrappers to keep smoke and full runs distinct."
        ),
    )
    parser.add_argument(
        "--timeout-hours-override",
        type=float,
        default=None,
        help="Override recipe timeout_hours for this dispatch.",
    )
    parser.add_argument(
        "--cost-band-epochs-override",
        type=int,
        default=None,
        help=(
            "Override cost_band.epochs and Modal cost-band metadata for this "
            "dispatch. Used by smoke-before-full wrappers."
        ),
    )
    args = parser.parse_args(argv)

    if args.list:
        return _list_recipes()

    if not args.recipe:
        parser.print_help()
        return 2

    recipe = _load_recipe(args.recipe)
    dispatch_refusal = _recipe_dispatch_refusal(recipe)

    # Cost-band prediction.
    cost_cfg = recipe.raw.get("cost_band", {}) or {}
    platform_key = str(_resolve_env_var(cost_cfg.get("platform_key"), recipe.platform))
    gpu_key = str(_resolve_env_var(cost_cfg.get("gpu_key"), recipe.gpu or "T4"))
    raw_epochs = (
        args.cost_band_epochs_override
        if args.cost_band_epochs_override is not None
        else cost_cfg.get("epochs", 1000)
    )
    try:
        epochs = int(raw_epochs)
    except (TypeError, ValueError):
        if dispatch_refusal:
            epochs = 0
        else:
            raise SystemExit(
                "[operator-authorize] FATAL: dispatchable recipe has "
                f"non-numeric cost_band.epochs={raw_epochs!r}; fix the recipe "
                "before operator authorization"
            ) from None
    all_flags_on = bool(cost_cfg.get("all_flags_on", True))
    fallback_p50 = float(cost_cfg.get("hand_calibrated_fallback_p50_usd", 1.00))
    band = _predict_cost_band(
        platform_key=platform_key,
        gpu_key=gpu_key,
        epochs=epochs,
        all_flags_on=all_flags_on,
        hand_calibrated_fallback_p50_usd=fallback_p50,
    )

    # Print banner.
    _print_banner(recipe, band)

    if args.dry_run:
        if dispatch_refusal:
            print(f"[operator-authorize] --dry-run; would refuse: {dispatch_refusal}")
        print("[operator-authorize] --dry-run; no confirmation prompt, no dispatch")
        return 0
    elif dispatch_refusal:
        raise SystemExit(
            "[operator-authorize] FATAL: recipe is not dispatchable: "
            f"{dispatch_refusal}"
        )

    # Operator confirmation.
    if not _confirm(
        f"operator confirmation: proceed with {recipe.name} dispatch "
        f"(p50approximately${band.p50_cost_usd:.2f}, {band.confidence_tag})?"
    ):
        print("[operator-authorize] aborted - no dispatch fired")
        return 0

    # Required-input-files pre-flight (Catalog #152). Run this after the
    # operator confirms so a missing optional local artifact does not mask the
    # prompt/no-dispatch contract, but still before native provider preflight,
    # lane claim creation, or any GPU-metered work.
    required_files = recipe.raw.get("required_input_files", []) or []
    if required_files:
        # Recipe declares a trainer path under modal.cost_band_trainer or
        # at top-level required_input_files_trainer.
        trainer = (
            recipe.raw.get("modal", {}).get("cost_band_trainer")
            or recipe.raw.get("required_input_files_trainer")
        )
        if trainer:
            _validate_required_input_files(str(trainer), recipe)
        else:
            print(
                "[operator-authorize] WARN: recipe declares required_input_files but "
                "no trainer path - skipping local validation",
                file=sys.stderr,
            )

    native_dispatch = _platform_has_native_dispatch(recipe.platform)
    if native_dispatch:
        _native_dispatch_preflight(recipe)

    # Lane claim. Do not create an active claim for recipe-only/no-op platforms:
    # their legacy shims still own the real action and should own any claim.
    instance_job_id = f"{recipe.name}_{_resolve_utc_label()}{args.label_suffix}"
    claim_created = False
    if not args.no_claim and native_dispatch:
        notes = (
            f"operator-authorized via tools/operator_authorize.py --recipe "
            f"{recipe.name}; expected p50 cost ${band.p50_cost_usd:.2f} "
            f"({band.confidence_tag})"
        )
        _claim_lane(
            lane_id=recipe.lane_id,
            platform=recipe.platform,
            instance_job_id=instance_job_id,
            agent=args.agent,
            notes=notes,
        )
        claim_created = True
    elif not args.no_claim:
        print(
            "[operator-authorize] no native dispatch for platform="
            f"{recipe.platform}; skipping lane claim so no phantom active "
            "dispatch row is created"
        )

    # Dispatch.
    try:
        if (
            args.timeout_hours_override is None
            and args.cost_band_epochs_override is None
        ):
            rc = _run_dispatch(recipe, instance_job_id)
        else:
            rc = _run_dispatch(
                recipe,
                instance_job_id,
                timeout_hours_override=args.timeout_hours_override,
                cost_band_epochs_override=args.cost_band_epochs_override,
            )
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if claim_created:
            _terminal_claim(
                lane_id=recipe.lane_id,
                platform=recipe.platform,
                instance_job_id=instance_job_id,
                agent=args.agent,
                status=_sanitize_terminal_status(f"failed_dispatch_exception_{code}"),
                notes=(
                    "operator-authorize native dispatch raised SystemExit after "
                    f"claim: {exc}"
                ),
            )
        raise
    except Exception as exc:
        if claim_created:
            _terminal_claim(
                lane_id=recipe.lane_id,
                platform=recipe.platform,
                instance_job_id=instance_job_id,
                agent=args.agent,
                status=_sanitize_terminal_status(
                    f"failed_dispatch_exception_{type(exc).__name__}"
                ),
                notes=(
                    "operator-authorize native dispatch raised after claim: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
        raise
    if rc != 0:
        if claim_created:
            _terminal_claim(
                lane_id=recipe.lane_id,
                platform=recipe.platform,
                instance_job_id=instance_job_id,
                agent=args.agent,
                status=_sanitize_terminal_status(f"failed_dispatch_rc_{rc}"),
                notes=(
                    "operator-authorize native dispatch returned non-zero "
                    f"rc={rc}; terminal row closes active claim"
                ),
            )
        print(
            f"[operator-authorize] WARN: dispatch returned rc={rc}; review "
            "the dispatch logs + .omx/state/active_lane_dispatch_claims.md",
            file=sys.stderr,
        )
    else:
        print(
            f"[operator-authorize] complete; lane_id={recipe.lane_id} "
            f"instance_job_id={instance_job_id}"
        )
        print("  review .omx/state/active_lane_dispatch_claims.md")
    return rc


if __name__ == "__main__":
    sys.exit(main())
