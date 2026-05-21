#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
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
    dependencies: []             # optional; local paths are preflighted before native dispatch
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
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPES_DIR = REPO_ROOT / ".omx/operator_authorize_recipes"
VENV_PYTHON = REPO_ROOT / ".venv/bin/python"

# D9 per-class provider routing (council Decision 9 binding verdict).
# Imported lazily inside _resolve_routing_decision so this module loads
# even if tac.cost_band_calibration is unavailable in tests.


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


@dataclass(frozen=True)
class CostBandRequest:
    """Resolved operator-facing cost request for one authorization run."""

    platform_key: str
    gpu_key: str
    epochs: int
    all_flags_on: bool
    fallback_p50_usd: float
    context_label: str
    full_run_platform_key: str
    full_run_gpu_key: str
    full_run_epochs: int
    full_run_fallback_p50_usd: float

    @property
    def is_smoke_scaled(self) -> bool:
        return self.context_label == "smoke_override"


def _as_int_cost_epochs(raw_epochs: Any, *, field_name: str) -> int:
    try:
        epochs = int(raw_epochs)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be numeric; got {raw_epochs!r}") from None
    if epochs < 0:
        raise ValueError(f"{field_name} must be non-negative; got {epochs}")
    return epochs


def _smoke_scaled_fallback_p50(
    *,
    full_run_fallback_p50_usd: float,
    full_run_epochs: int,
    smoke_epochs: int,
) -> float:
    """Scale a full-run hand fallback down to the smoke epoch budget.

    This fallback is only used when ``tac.cost_band_calibration.predict`` has no
    empirical bucket and no hand stub for the smoke (platform, GPU, epochs)
    tuple. It prevents a 100-epoch smoke authorization from inheriting a
    full-run A100 p50 while still preserving the full-run p50 as separate
    reference metadata in the banner/claim notes.
    """

    if full_run_fallback_p50_usd <= 0.0 or full_run_epochs <= 0:
        return 0.0
    return max(0.0, full_run_fallback_p50_usd * (smoke_epochs / full_run_epochs))


def _resolve_cost_band_request(
    recipe: Recipe,
    *,
    cost_band_epochs_override: int | None = None,
    cost_band_gpu_override: str | None = None,
) -> CostBandRequest:
    """Resolve the cost bucket and fallback used for the authorization banner."""

    cost_cfg = recipe.raw.get("cost_band", {}) or {}
    platform_key = str(_resolve_env_var(cost_cfg.get("platform_key"), recipe.platform))
    full_run_gpu_key = str(_resolve_env_var(cost_cfg.get("gpu_key"), recipe.gpu or "T4"))
    try:
        full_run_epochs = _as_int_cost_epochs(
            cost_cfg.get("epochs", 1000),
            field_name="cost_band.epochs",
        )
    except ValueError:
        raise
    all_flags_on = bool(cost_cfg.get("all_flags_on", True))
    full_run_fallback_p50 = float(
        cost_cfg.get("hand_calibrated_fallback_p50_usd", 1.00)
    )

    if cost_band_epochs_override is None:
        return CostBandRequest(
            platform_key=platform_key,
            gpu_key=full_run_gpu_key,
            epochs=full_run_epochs,
            all_flags_on=all_flags_on,
            fallback_p50_usd=full_run_fallback_p50,
            context_label="recipe_full",
            full_run_platform_key=platform_key,
            full_run_gpu_key=full_run_gpu_key,
            full_run_epochs=full_run_epochs,
            full_run_fallback_p50_usd=full_run_fallback_p50,
        )

    smoke_epochs = _as_int_cost_epochs(
        cost_band_epochs_override,
        field_name="--cost-band-epochs-override",
    )
    # Smoke-before-full changes the provider GPU with MODAL_GPU. Recipes that
    # expose top-level `gpu: "${MODAL_GPU:-A100}"` resolve recipe.gpu to the
    # smoke GPU, but an explicit CLI override wins when a caller supplies one.
    smoke_gpu_key = (cost_band_gpu_override or recipe.gpu or full_run_gpu_key).strip()
    if not smoke_gpu_key or smoke_gpu_key.lower() == "none":
        smoke_gpu_key = full_run_gpu_key
    smoke_fallback_p50 = _smoke_scaled_fallback_p50(
        full_run_fallback_p50_usd=full_run_fallback_p50,
        full_run_epochs=full_run_epochs,
        smoke_epochs=smoke_epochs,
    )
    return CostBandRequest(
        platform_key=platform_key,
        gpu_key=smoke_gpu_key,
        epochs=smoke_epochs,
        all_flags_on=all_flags_on,
        fallback_p50_usd=smoke_fallback_p50,
        context_label="smoke_override",
        full_run_platform_key=platform_key,
        full_run_gpu_key=full_run_gpu_key,
        full_run_epochs=full_run_epochs,
        full_run_fallback_p50_usd=full_run_fallback_p50,
    )


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
                    source=(
                        "posterior"
                        if str(payload["confidence_tag"])
                        in {"empirical_posterior", "weak_posterior"}
                        else "hand_calibrated_fallback"
                    ),
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


def _check_predecessor_probe_outcome(recipe_path: Path) -> None:
    """Catalog #313 — refuse dispatch if recipe's substrate has a recent
    blocking probe-disambiguator verdict in the canonical ledger.

    Operator directive 2026-05-16 (PROBE-OUTCOMES-BAKE-IN subagent): *"bake
    in the FULL 4-layer canonical pattern per the Catalog #245 Modal call_id
    ledger exemplar so probe-disambiguator verdicts are queryable across
    sessions and gating dispatch BEFORE we re-run something an existing
    adjudicated probe already settled"*.

    The canonical adjudicated-outcomes ledger lives at
    ``.omx/state/probe_outcomes.jsonl`` via ``tac.probe_outcomes_ledger``.
    Empirical anchor: ATW v2 D4 H(latent|scorer_class) probe (Codex
    ``tools/run_atw_v2_d4_probe_from_a1.py`` 2026-05-16 22:47:41Z) returned
    ``INDEPENDENT`` verdict (MI=0.006385 bits/symbol; 2 orders of magnitude
    below MEANINGFUL_CONDITIONING threshold 0.5). Without this gate, future
    subagent dispatchers could re-fire ATW v2 Phase 2 dispatch despite the
    apparatus having already settled the question — burning paid GPU.

    Per CLAUDE.md "Forbidden premature KILL without research exhaustion":
    a blocking outcome does NOT mean the lane is killed — it means the
    apparatus has already adjudicated this probe within the 30-day
    staleness window. Override via paired-env (Catalog #199 sister rule):
        OPERATOR_AUTHORIZE_PROBE_PREDECESSOR_BYPASS_VERDICT=1
        OPERATOR_AUTHORIZE_PROBE_PREDECESSOR_BYPASS_RATIONALE=<text>

    Bare ``OPERATOR_AUTHORIZE_PROBE_PREDECESSOR_BYPASS_VERDICT=1`` without
    paired rationale raises SystemExit.
    """
    bypass_active = (
        os.environ.get("OPERATOR_AUTHORIZE_PROBE_PREDECESSOR_BYPASS_VERDICT") == "1"
    )
    if bypass_active:
        rationale = os.environ.get(
            "OPERATOR_AUTHORIZE_PROBE_PREDECESSOR_BYPASS_RATIONALE", ""
        ).strip()
        if not rationale:
            raise SystemExit(
                "[operator-authorize] FATAL: "
                "OPERATOR_AUTHORIZE_PROBE_PREDECESSOR_BYPASS_VERDICT=1 "
                "requires paired "
                "OPERATOR_AUTHORIZE_PROBE_PREDECESSOR_BYPASS_RATIONALE=<text> "
                "(per CLAUDE.md paired-env discipline + Catalog #199 sister rule "
                "+ Catalog #313). Refusing to skip predecessor probe-outcome "
                "check without rationale."
            )
        print(
            f"[operator-authorize] [PROBE-PREDECESSOR BYPASS ACTIVE] "
            f"reason={rationale!r}; recipe={recipe_path}",
            file=sys.stderr,
        )
        return

    try:
        from tac.probe_outcomes_ledger import latest_blocking_outcome_by_recipe
    except ImportError:
        print(
            "[operator-authorize] WARN: tac.probe_outcomes_ledger not "
            "importable; skipping Catalog #313 predecessor probe-outcome "
            "check (operator should investigate)",
            file=sys.stderr,
        )
        return

    try:
        view = latest_blocking_outcome_by_recipe(recipe_path)
    except Exception as exc:
        print(
            f"[operator-authorize] WARN: Catalog #313 ledger query failed "
            f"({exc}); skipping predecessor probe-outcome check",
            file=sys.stderr,
        )
        return

    if view is None:
        return

    raise SystemExit(
        f"[operator-authorize] FATAL: probe-disambiguator predecessor verdict "
        f"{view.verdict} blocks dispatch (substrate={view.substrate}; "
        f"probe_id={view.probe_id}; adjudicated_at={view.adjudicated_at_utc}; "
        f"evidence={view.evidence_path}; metric={view.metric_name}="
        f"{view.metric_value} vs threshold={view.threshold}). "
        f"Per Catalog #313 PROBE-OUTCOMES-BAKE-IN: the apparatus has already "
        f"adjudicated this probe within the 30-day staleness window. Either "
        f"(a) address the blocker via sister probe with alternative reducer "
        f"(Catalog #308), (b) have the council ratify fresh-evidence "
        f"override, OR (c) set "
        f"OPERATOR_AUTHORIZE_PROBE_PREDECESSOR_BYPASS_VERDICT=1 + paired "
        f"OPERATOR_AUTHORIZE_PROBE_PREDECESSOR_BYPASS_RATIONALE=<text>."
    )


def _run_local_pre_deploy_check(
    trainer_path: str,
    recipe_name: str,
) -> None:
    """Invoke ``tools/local_pre_deploy_check.py --strict`` per WIRE-AND-INTEGRATE-ALL.

    Local 30-second pre-flight harness that catches bug classes BEFORE paid
    Modal/Vast.ai dispatch (operator directive 2026-05-15: *"shouldn't you run
    python compile locally before deploy to find those bugs"*). Empirical
    anchors: Z3 v2 smoke ``fc-01KRNHEGC9ZE48Y68GGJHP7FXN`` ($2 wasted) + Z4
    smoke ``fc-01KRNHE942JSV7VRGXGR1FJGHQ`` ($2 wasted) both crashed on bugs
    that the harness would have caught.

    Checks (per ``tools/local_pre_deploy_check.py``):
      1. py-compile (syntax errors)
      2. _full_main implemented (no NotImplementedError stub)
      3. archive grammar (canonical ``0.bin`` member name)
      4. auth-eval reachability (gate_auth_eval_call OR auth_eval_renderer)
      5. canonical inflate device (Catalog #205)
      6. deterministic ZIP (Catalog #19)

    Bypass via ``OPERATOR_AUTHORIZE_SKIP_LOCAL_PRE_DEPLOY_CHECK=1`` (rare;
    e.g. during gate development when the harness itself is being modified).
    The bypass requires a paired ``OPERATOR_AUTHORIZE_LOCAL_PRE_DEPLOY_BYPASS_REASON=<text>``
    so the audit trail captures rationale (per CLAUDE.md "Comment-only
    contracts are FORBIDDEN" pattern + Catalog #199 paired-env discipline).
    """
    if os.environ.get("OPERATOR_AUTHORIZE_SKIP_LOCAL_PRE_DEPLOY_CHECK") == "1":
        reason = os.environ.get(
            "OPERATOR_AUTHORIZE_LOCAL_PRE_DEPLOY_BYPASS_REASON", ""
        ).strip()
        if not reason:
            raise SystemExit(
                "[operator-authorize] FATAL: OPERATOR_AUTHORIZE_SKIP_LOCAL_PRE_DEPLOY_CHECK=1 "
                "requires paired OPERATOR_AUTHORIZE_LOCAL_PRE_DEPLOY_BYPASS_REASON=<text> "
                "(per CLAUDE.md paired-env discipline + Catalog #199 sister rule). "
                "Refusing to dispatch without rationale."
            )
        print(
            f"[operator-authorize] [LOCAL-PRE-DEPLOY BYPASS ACTIVE] "
            f"reason={reason!r}; trainer={trainer_path}",
            file=sys.stderr,
        )
        return
    harness = REPO_ROOT / "tools/local_pre_deploy_check.py"
    if not harness.exists():
        print(
            f"[operator-authorize] WARN: local pre-deploy harness not found at "
            f"{harness}; skipping pre-flight (operator should investigate)",
            file=sys.stderr,
        )
        return
    cmd = [
        _python_bin(),
        str(harness),
        "--trainer",
        trainer_path,
        "--recipe",
        recipe_name,
        "--strict",
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise SystemExit(
            "[operator-authorize] FATAL: local pre-deploy harness FAILED "
            f"(exit {result.returncode}); aborting BEFORE GPU dispatch. "
            "Per operator 2026-05-15 directive + WIRE-AND-INTEGRATE-ALL "
            "subagent (lane_wire_and_integrate_all_cross_stack_20260515): "
            "every dispatch must pass tools/local_pre_deploy_check.py BEFORE "
            "spending. Either fix the bug class the harness flagged OR set "
            "OPERATOR_AUTHORIZE_SKIP_LOCAL_PRE_DEPLOY_CHECK=1 with paired "
            "OPERATOR_AUTHORIZE_LOCAL_PRE_DEPLOY_BYPASS_REASON=<text>."
        )


def _run_codex_pre_dispatch_review(
    trainer_path: str,
    recipe_path: str,
    estimated_cost_usd: float,
) -> None:
    """Invoke ``tools/run_codex_review_for_dispatch.py`` per Catalog #271.

    Operator-approved 2026-05-15: wire codex adversarial-review automation
    into the operator-authorize flow BEFORE every paid Modal/Lightning/Vast.ai
    dispatch >$1 estimated cost. The codex review of Z3-G1 caught F1+F2
    BEFORE FULL CUDA dispatched but AFTER $0.59 smoke spend - this gate moves
    the review BEFORE smoke.

    Sister of:
    - Catalog #243 (local pre-deploy harness): same insertion point.
    - Catalog #167 (smoke-before-full): same dispatch-wrapper canonical
      routing META.
    - Catalog #199 (paired-env operator bypass discipline): same paired-env
      pattern.

    Cost gate: only invokes codex when estimated cost > $1. Below threshold,
    the helper returns 'advisory' without burning codex tokens.

    Bypass via paired-env discipline (per CLAUDE.md "Comment-only contracts
    are FORBIDDEN" + Catalog #199 sister rule):
      OPERATOR_AUTHORIZE_SKIP_CODEX_PRE_DISPATCH_REVIEW=1
      OPERATOR_AUTHORIZE_CODEX_PRE_DISPATCH_BYPASS_REASON=<text>

    Bare ``OPERATOR_AUTHORIZE_SKIP_CODEX_PRE_DISPATCH_REVIEW=1`` (without
    paired rationale) raises SystemExit per paired-env discipline.
    """
    if os.environ.get("OPERATOR_AUTHORIZE_SKIP_CODEX_PRE_DISPATCH_REVIEW") == "1":
        reason = os.environ.get(
            "OPERATOR_AUTHORIZE_CODEX_PRE_DISPATCH_BYPASS_REASON", ""
        ).strip()
        if not reason:
            raise SystemExit(
                "[operator-authorize] FATAL: "
                "OPERATOR_AUTHORIZE_SKIP_CODEX_PRE_DISPATCH_REVIEW=1 requires "
                "paired OPERATOR_AUTHORIZE_CODEX_PRE_DISPATCH_BYPASS_REASON="
                "<text> (per CLAUDE.md paired-env discipline + Catalog #199 "
                "+ #271 sister rule). Refusing to skip codex review without "
                "rationale."
            )
        print(
            f"[operator-authorize] [CODEX-PRE-DISPATCH-REVIEW BYPASS ACTIVE] "
            f"reason={reason!r}; trainer={trainer_path}",
            file=sys.stderr,
        )
        return
    helper = REPO_ROOT / "tools/run_codex_review_for_dispatch.py"
    if not helper.exists():
        raise SystemExit(
            "[operator-authorize] FATAL: codex pre-dispatch helper not found "
            f"at {helper}; refusing paid dispatch before review. Restore the "
            "helper or use OPERATOR_AUTHORIZE_SKIP_CODEX_PRE_DISPATCH_REVIEW=1 "
            "with paired OPERATOR_AUTHORIZE_CODEX_PRE_DISPATCH_BYPASS_REASON=<text>."
        )
    cmd = [
        _python_bin(),
        str(helper),
        "--trainer",
        trainer_path,
        "--recipe",
        recipe_path,
        "--estimated-cost-usd",
        f"{estimated_cost_usd:.2f}",
        # Catalog #282 — codex bfa2p1uex F2 fix: paid-dispatch path MUST NOT
        # reuse a cached approve from a stale working tree. Codex companion
        # is invoked with --scope working-tree; the cache key MUST match.
        "--no-cache-for-paid-dispatch",
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode == 0:
        return
    if result.returncode == 13:
        # Paired-env bypass invariant violated inside helper - propagate
        raise SystemExit(
            "[operator-authorize] FATAL: codex pre-dispatch review helper "
            "refused due to paired-env bypass invariant violation (rc=13). "
            "Set OPERATOR_AUTHORIZE_CODEX_REVIEW_BYPASS_VERDICT=1 with paired "
            "OPERATOR_AUTHORIZE_CODEX_REVIEW_BYPASS_RATIONALE=<text> OR fix "
            "the underlying findings."
        )
    if result.returncode == 2:
        # Catalog #281 — codex bfa2p1uex F1 fix: invocation-error means the
        # codex companion timed out / crashed / exited nonzero with no
        # severity-tagged findings. The review did NOT complete; refuse paid
        # dispatch unless paired-env bypass.
        raise SystemExit(
            "[operator-authorize] FATAL: codex pre-dispatch review helper "
            "exited rc=2 (invocation-error per Catalog #281). The codex "
            "companion timed out, crashed, or otherwise failed to produce "
            "a review. Refusing paid dispatch BEFORE GPU spend. Either "
            "diagnose the codex companion failure (timeout / token-refresh / "
            "node missing) OR set OPERATOR_AUTHORIZE_CODEX_REVIEW_BYPASS_"
            "VERDICT=1 with paired OPERATOR_AUTHORIZE_CODEX_REVIEW_BYPASS_"
            "RATIONALE=<text> if intentional."
        )
    raise SystemExit(
        "[operator-authorize] FATAL: codex pre-dispatch adversarial review "
        f"FAILED (exit {result.returncode}); aborting BEFORE GPU dispatch. "
        "Per operator 2026-05-15 directive + Catalog #271 PRE-DISPATCH-CODEX-"
        "REVIEW-AUTOMATION subagent: every paid dispatch (>$1) must pass a "
        "fresh codex adversarial review BEFORE spending. Either fix the bug "
        "class codex flagged OR set "
        "OPERATOR_AUTHORIZE_SKIP_CODEX_PRE_DISPATCH_REVIEW=1 with paired "
        "OPERATOR_AUTHORIZE_CODEX_PRE_DISPATCH_BYPASS_REASON=<text>."
    )


def _run_dispatch_protocol_complete(
    recipe: Recipe,
    trainer_path: str | None,
    *,
    native_dispatch: bool,
) -> None:
    """Require the full dispatch feasibility conjunction before lane claims.

    This is the Boyd-style umbrella over the narrow catalog gates:
    dispatch_protocol_complete =
    AND(tier1_engineering, tier2_hardware_correctness, tier3_substrate_correctness).
    It runs before claim creation/provider setup, so a partial feasibility
    intersection cannot burn GPU or create a phantom active dispatch row.
    """

    if not native_dispatch:
        return
    src = REPO_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from tac.deploy.dispatch_protocol import require_dispatch_protocol_complete

    modal_cfg = recipe.raw.get("modal", {}) or {}
    lane_script = (
        modal_cfg.get("lane_script") if isinstance(modal_cfg, dict) else None
    ) or recipe.remote_driver
    report = require_dispatch_protocol_complete(
        recipe.raw,
        repo_root=REPO_ROOT,
        recipe_path=recipe.path,
        trainer_path=trainer_path,
        remote_driver_path=str(lane_script) if lane_script else None,
        native_dispatch=native_dispatch,
    )
    print(
        "[operator-authorize] dispatch_protocol_complete=PASS "
        f"recipe={report.recipe_name}"
    )


def _normalize_declared_local_path(value: Any) -> str | None:
    """Return a repo-local path declared in a recipe, or ``None`` if non-local.

    Recipe ``dependencies`` often carry human annotations such as
    ``tools/foo.py (Catalog #152)``. The dispatch preflight should validate the
    actual path while ignoring bracketed placeholders, URLs, env refs, and
    provider-absolute paths such as ``/workspace/pact/...``.
    """

    if value is None:
        return None
    text = str(value).strip().strip("'\"")
    if not text:
        return None
    lowered = text.lower()
    if (
        text.startswith("[")
        or text.startswith("$")
        or text.startswith("${")
        or "://" in text
        or lowered in {"none", "null", "n/a", "na"}
    ):
        return None
    if text.startswith("/"):
        try:
            Path(text).resolve().relative_to(REPO_ROOT)
        except (OSError, ValueError):
            return None
        return text
    # Strip common prose annotations while preserving paths with spaces as
    # invalid/missing rather than guessing a different path.
    for marker in (" (", "\t("):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text or None


def _iter_declared_local_path_refs(recipe: Recipe) -> list[tuple[str, str, bool]]:
    """Collect declared local recipe paths that must exist before dispatch.

    The boolean marks entries that must be files rather than directories.
    """

    refs: list[tuple[str, str, bool]] = []

    def add(label: str, value: Any, *, must_be_file: bool = True) -> None:
        normalized = _normalize_declared_local_path(value)
        if normalized:
            refs.append((label, normalized, must_be_file))

    add("remote_driver", recipe.remote_driver)

    modal_cfg = recipe.raw.get("modal", {}) or {}
    if isinstance(modal_cfg, dict):
        add("modal.lane_script", modal_cfg.get("lane_script"))
        add("modal.cost_band_trainer", modal_cfg.get("cost_band_trainer"))

    vastai_cfg = recipe.raw.get("vastai", {}) or {}
    if isinstance(vastai_cfg, dict):
        add("vastai.launcher", vastai_cfg.get("launcher"))
        add("vastai.lane_script", vastai_cfg.get("lane_script"))

    add("required_input_files_trainer", recipe.raw.get("required_input_files_trainer"))

    readiness_gate = recipe.raw.get("readiness_gate")
    if isinstance(readiness_gate, str):
        add("readiness_gate", readiness_gate.split(maxsplit=1)[0])

    for idx, value in enumerate(recipe.raw.get("sentinel_files", []) or []):
        add(f"sentinel_files[{idx}]", value)

    for idx, value in enumerate(recipe.raw.get("dependencies", []) or []):
        add(f"dependencies[{idx}]", value, must_be_file=False)

    # Deduplicate by path while preserving the most specific label list for
    # actionable diagnostics.
    seen: dict[str, tuple[list[str], bool]] = {}
    for label, rel, must_be_file in refs:
        labels, existing_must_be_file = seen.setdefault(rel, ([], must_be_file))
        labels.append(label)
        if must_be_file and not existing_must_be_file:
            seen[rel] = (labels, True)

    return [(", ".join(labels), rel, must_be_file) for rel, (labels, must_be_file) in seen.items()]


def _validate_declared_local_paths(recipe: Recipe) -> None:
    """Fail before claim/provider setup when a dispatchable recipe is incomplete."""

    failures: list[str] = []
    for labels, rel, must_be_file in _iter_declared_local_path_refs(recipe):
        path = Path(rel)
        full = path if path.is_absolute() else REPO_ROOT / path
        if not full.exists():
            failures.append(f"{labels}: missing {rel}")
            continue
        if must_be_file and not full.is_file():
            failures.append(f"{labels}: expected file, found non-file {rel}")

    if failures:
        joined = "; ".join(failures)
        raise SystemExit(
            "[operator-authorize] FATAL: declared local path preflight failed "
            f"for recipe '{recipe.name}' before lane claim/provider setup: {joined}"
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


_SESSION_DIRECTIVE_ENV_VAR = "OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE"
_SESSION_BUDGET_ENV_VAR = "OPERATOR_AUTHORIZE_SESSION_BUDGET_USD"

# Catalog #202 — paired-env-var bypass of `--require-clean-head` for trusted
# sentinel-clean Modal dispatches. See `_whole_tree_clean_check_bypass_active`
# below for the contract; STRICT preflight gate
# `check_catalog_202_bypass_requires_paired_env_attestation` enforces the
# paired-env contract structurally.
_WHOLE_TREE_CLEAN_BYPASS_INTENT_ENV_VAR = (
    "OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK"
)
_WHOLE_TREE_CLEAN_BYPASS_ATTESTATION_ENV_VAR = (
    "OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED"
)
_WHOLE_TREE_CLEAN_BYPASS_AUDIT_JSON_ENV_VAR = (
    "OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_AUDIT_JSON"
)


def _git_dirty_paths() -> set[str]:
    """Return dirty git paths using porcelain output, fail-closed to empty."""

    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return set()
    out: set[str] = set()
    for line in proc.stdout.splitlines():
        if not line:
            continue
        rel = line[3:] if len(line) > 3 else ""
        if " -> " in rel:
            old, new = rel.split(" -> ", 1)
            out.add(old)
            out.add(new)
        elif rel:
            out.add(rel)
    return out


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sentinel_set_sha256(path_to_sha: dict[str, str]) -> str:
    import hashlib

    encoded = json.dumps(
        path_to_sha,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _resolve_catalog202_audit_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _verify_catalog202_sentinel_audit(recipe: Recipe, raw_path: str) -> None:
    """Verify a Catalog #202 sentinel audit matches current effective sentinels."""

    path = _resolve_catalog202_audit_path(raw_path)
    if not path.is_file():
        print(
            "[operator-authorize] FATAL: Catalog #202 sentinel audit JSON "
            f"missing at {path}",
            file=sys.stderr,
        )
        raise SystemExit(12)
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(
            "[operator-authorize] FATAL: Catalog #202 sentinel audit JSON "
            f"could not be parsed at {path}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(12) from exc
    if audit.get("schema") != "catalog202_sentinel_cleanliness_audit_v1":
        print(
            "[operator-authorize] FATAL: Catalog #202 sentinel audit JSON has "
            f"unexpected schema={audit.get('schema')!r}",
            file=sys.stderr,
        )
        raise SystemExit(12)
    if str(audit.get("recipe_name")) != recipe.path.stem:
        print(
            "[operator-authorize] FATAL: Catalog #202 sentinel audit recipe "
            f"mismatch: audit={audit.get('recipe_name')!r} recipe={recipe.path.stem!r}",
            file=sys.stderr,
        )
        raise SystemExit(12)
    if audit.get("missing_sentinel_files") or audit.get(
        "outside_modal_mount_sentinel_files"
    ):
        print(
            "[operator-authorize] FATAL: Catalog #202 sentinel audit has "
            "missing/outside-mount sentinels; rerun or fix the audit before "
            "bypassing --require-clean-head.",
            file=sys.stderr,
        )
        raise SystemExit(12)

    expected_paths = [p for p in _modal_sentinel_files(recipe).split(",") if p]
    audit_paths = list(audit.get("effective_sentinel_files") or [])
    if audit_paths != expected_paths:
        print(
            "[operator-authorize] FATAL: Catalog #202 sentinel audit path set "
            f"mismatch. audit={audit_paths!r} current={expected_paths!r}",
            file=sys.stderr,
        )
        raise SystemExit(12)
    audit_records = {
        str(row.get("path")): str(row.get("sha256"))
        for row in audit.get("sentinel_records") or []
        if isinstance(row, dict) and row.get("path") and row.get("sha256")
    }
    current: dict[str, str] = {}
    mismatches: list[str] = []
    for rel in expected_paths:
        current_path = REPO_ROOT / rel
        if not current_path.is_file():
            mismatches.append(f"{rel}:MISSING_CURRENT")
            continue
        current_sha = _sha256_file(current_path)
        current[rel] = current_sha
        if audit_records.get(rel) != current_sha:
            mismatches.append(
                f"{rel}:audit={audit_records.get(rel)} current={current_sha}"
            )
    if mismatches:
        print(
            "[operator-authorize] FATAL: Catalog #202 sentinel audit hash "
            "mismatch; rerun audit after stabilizing intended sentinel bytes: "
            + "; ".join(mismatches[:5]),
            file=sys.stderr,
        )
        raise SystemExit(12)
    current_set_sha = _sentinel_set_sha256(current)
    if audit.get("sentinel_set_sha256") != current_set_sha:
        print(
            "[operator-authorize] FATAL: Catalog #202 sentinel_set_sha256 "
            f"mismatch audit={audit.get('sentinel_set_sha256')} "
            f"current={current_set_sha}",
            file=sys.stderr,
        )
        raise SystemExit(12)
    print(
        "[operator-authorize] Catalog #202 sentinel audit verified: "
        f"{path} sentinel_set_sha256={current_set_sha}",
        file=sys.stderr,
    )


def _whole_tree_clean_check_bypass_active(recipe: Recipe | None = None) -> bool:
    """Catalog #202 — return True iff the paired-env bypass attestation is set.

    The bypass fires only when BOTH env vars are set to truthy values:
      * ``OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1`` (intent flag)
      * ``OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=1`` (operator
        attestation that the explicit sentinel set is clean)

    If an effective Modal sentinel file is itself dirty in git status, the
    operator must also provide
    ``OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_AUDIT_JSON=<path>``. That artifact
    must match the current effective sentinel paths and SHA-256s. This keeps
    trusted dirty-sentinel snapshots explicit without weakening Catalog #166's
    worker-side hash check.

    When ONLY the intent flag is set without the attestation flag, the helper
    raises ``SystemExit(12)`` (sister to Catalog #199's ``SystemExit(11)``)
    so a partial-config bypass cannot silently take effect. When BOTH are
    set, prints a LOUD ``[OPERATOR-AUTHORIZE BYPASS]`` banner to stderr and
    returns True.

    Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
    + Catalog #199 paired-env-var pattern: the bypass is the immediate fix
    for the recurring "sister-subagent unrelated dirty files block trusted
    sentinel-clean dispatch" failure mode (3 occurrences in the 2026-05-13
    5-smoke-wave session). Catalog #166's worker-side sentinel hash check
    runs INDEPENDENTLY of ``--require-clean-head`` (see
    ``experiments/modal_train_lane.py::main`` lines 944-991) so the worker
    still refuses dispatch if any sentinel byte differs between the local
    snapshot and what Modal mounts. The whole-tree clean check is a SISTER
    safety surface that is overly conservative when the operator has
    independently verified the sentinel set is clean.

    The STRICT preflight gate
    ``check_catalog_202_bypass_requires_paired_env_attestation`` refuses any
    state of this file that drops the paired-env-var contract.
    """
    raw_intent = os.environ.get(_WHOLE_TREE_CLEAN_BYPASS_INTENT_ENV_VAR, "")
    if not raw_intent or raw_intent.strip().lower() in {"", "0", "false", "no"}:
        return False
    raw_attestation = os.environ.get(
        _WHOLE_TREE_CLEAN_BYPASS_ATTESTATION_ENV_VAR, ""
    )
    if (
        not raw_attestation
        or raw_attestation.strip().lower() in {"", "0", "false", "no"}
    ):
        print(
            f"[operator-authorize] FATAL: "
            f"{_WHOLE_TREE_CLEAN_BYPASS_INTENT_ENV_VAR}=1 is set but "
            f"{_WHOLE_TREE_CLEAN_BYPASS_ATTESTATION_ENV_VAR} is missing or "
            "falsy. Catalog #202 requires the paired attestation env var "
            "(set =1) so the operator must explicitly attest that the "
            "explicit sentinel set is clean. Catalog #166 worker-side "
            "hash check still runs regardless.",
            file=sys.stderr,
        )
        raise SystemExit(12)
    if recipe is not None:
        dirty_paths = _git_dirty_paths()
        effective_sentinels = {
            rel for rel in _modal_sentinel_files(recipe).split(",") if rel
        }
        dirty_sentinels = sorted(effective_sentinels & dirty_paths)
        raw_audit = os.environ.get(_WHOLE_TREE_CLEAN_BYPASS_AUDIT_JSON_ENV_VAR, "")
        if dirty_sentinels and not raw_audit:
            print(
                "[operator-authorize] FATAL: Catalog #202 paired env vars are "
                "set, but effective Modal sentinel file(s) are dirty in git "
                f"status: {dirty_sentinels}. Set "
                f"{_WHOLE_TREE_CLEAN_BYPASS_AUDIT_JSON_ENV_VAR}=<audit.json> "
                "after running tools/audit_catalog202_sentinel_cleanliness.py "
                "so the dirty sentinel snapshot is hash-verified.",
                file=sys.stderr,
            )
            raise SystemExit(12)
        if raw_audit:
            _verify_catalog202_sentinel_audit(recipe, raw_audit)

    print(
        "[OPERATOR-AUTHORIZE BYPASS] Catalog #202 ACTIVE: "
        "--require-clean-head DISABLED — operator attests sentinel set is "
        "clean. Catalog #166 worker-side hash check still runs.",
        file=sys.stderr,
    )
    return True


def _session_directive_bypass_active() -> tuple[bool, str | None]:
    """Catalog #199 — return (bypass_active, error_message).

    The bypass fires only when BOTH env vars are set:
      * ``OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1`` (or any truthy)
      * ``OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=<float>`` (parseable as float)

    Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against":
    forcing a paired budget env var prevents bare ``CONFIRMED=1`` from acting
    as a blanket auto-approve. STRICT preflight Catalog #199 refuses any
    caller that toggles the directive flag without declaring the budget.
    """
    raw_confirmed = os.environ.get(_SESSION_DIRECTIVE_ENV_VAR, "")
    if not raw_confirmed or raw_confirmed.strip().lower() in {"", "0", "false", "no"}:
        return False, None
    raw_budget = os.environ.get(_SESSION_BUDGET_ENV_VAR, "")
    if not raw_budget:
        return False, (
            f"{_SESSION_DIRECTIVE_ENV_VAR} is set but {_SESSION_BUDGET_ENV_VAR} "
            "is missing. Catalog #199 requires the paired session-budget env "
            "var (in USD, e.g. 20.0) so the operator must explicitly "
            "acknowledge the session-wide envelope."
        )
    try:
        budget_usd = float(raw_budget)
    except ValueError:
        return False, (
            f"{_SESSION_BUDGET_ENV_VAR}={raw_budget!r} is not a parseable "
            "float (USD). Set e.g. =20.0."
        )
    if budget_usd <= 0.0:
        return False, (
            f"{_SESSION_BUDGET_ENV_VAR}={budget_usd} must be > 0."
        )
    print(
        "[OPERATOR-AUTHORIZE BYPASS ACTIVE] Session-directive bypass + "
        f"${budget_usd:.2f} budget cap; HUMAN APPROVAL ASSUMED BY ENV VAR "
        f"({_SESSION_DIRECTIVE_ENV_VAR}=1 + "
        f"{_SESSION_BUDGET_ENV_VAR}={budget_usd:.2f}).",
        file=sys.stderr,
    )
    return True, None


def _confirm(prompt: str, *, default_yes: bool = False) -> bool:
    """Read a y/N from stdin. Returns True only on explicit yes.

    Catalog #199 bypass: when both ``OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE``
    AND ``OPERATOR_AUTHORIZE_SESSION_BUDGET_USD`` are set, return True without
    prompting (and log a LOUD banner to stderr). This addresses the orchestrator
    failure mode where ``run_modal_smoke_before_full.py`` invokes
    ``operator_authorize.py`` via ``subprocess.run(capture_output=True)`` —
    stdin is closed, ``input()`` raises EOFError, ``_confirm`` returns False,
    dispatch is silently aborted.

    Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against":
    a STRICT preflight gate (Catalog #199) refuses any caller of the bypass
    that does NOT declare the paired budget env var.
    """
    bypass_active, err_msg = _session_directive_bypass_active()
    if bypass_active:
        return True
    if err_msg is not None:
        print(
            f"[operator-authorize] FATAL: {err_msg}",
            file=sys.stderr,
        )
        raise SystemExit(11)
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


def _required_env_var_reference(value: Any) -> str | None:
    """Return the referenced env var for ``${VAR}`` values without fallbacks."""

    if not isinstance(value, str) or not value.startswith("${") or not value.endswith("}"):
        return None
    body = value[2:-1]
    if ":-" in body or not body:
        return None
    return body


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
    def smoke_only(self) -> bool:
        """Return whether this recipe is restricted to the smoke wrapper path."""

        raw = self.raw.get("smoke_only", False)
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, int):
            return bool(raw)
        return str(raw).strip().lower() in {"1", "on", "true", "yes"}

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
    recipe that declares ``dispatch_enabled: false``, still lists
    ``dispatch_blockers``, or still lists ``pre_promotion_blockers`` must not
    reach confirmation, claim creation, or provider dispatch. This is
    deliberately independent of provider-specific preflights so deferred
    research recipes also fail before cost prediction or provider setup can
    create misleading state.
    """

    if not recipe.dispatch_enabled:
        details: list[str] = ["dispatch_enabled=false"]
        if recipe.dispatch_blockers:
            details.append("dispatch_blockers=" + ", ".join(recipe.dispatch_blockers))
        if recipe.defer_reason:
            details.append("defer_reason=" + recipe.defer_reason.splitlines()[0])
        return "; ".join(details)
    if recipe.dispatch_blockers:
        return (
            "dispatch_blockers still declared: "
            + ", ".join(recipe.dispatch_blockers)
            + ". Clear these in the recipe only after the substrate trainer, "
            "runtime, archive grammar, and exact-eval packet path blockers are "
            "actually resolved."
        )
    if recipe.pre_promotion_blockers:
        return (
            "pre_promotion_blockers still declared: "
            + ", ".join(recipe.pre_promotion_blockers)
            + ". Clear these in the recipe only after the required exact anchors "
            "or custody artifacts land."
        )
    return None


def _smoke_only_direct_dispatch_refusal(
    recipe: Recipe,
    *,
    cost_band_epochs_override: int | None,
) -> str | None:
    """Return a refusal reason for direct full dispatch of smoke-only recipes."""

    if not recipe.smoke_only:
        return None
    if cost_band_epochs_override is not None:
        return None
    return (
        "smoke_only=true; direct operator_authorize full dispatch is refused. "
        "Use tools/run_modal_smoke_before_full.py so the wrapper emits only the "
        "implemented smoke anchor, or remove smoke_only only after the full "
        "trainer/runtime path lands."
    )


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
    cost_request: CostBandRequest,
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
    if cost_request.is_smoke_scaled:
        print(
            "  cost context:            smoke override "
            f"{cost_request.platform_key}/{cost_request.gpu_key} x "
            f"{cost_request.epochs} epochs"
        )
        print(
            "                           full-run reference "
            f"{cost_request.full_run_platform_key}/"
            f"{cost_request.full_run_gpu_key} x "
            f"{cost_request.full_run_epochs} epochs, "
            f"fallback p50=${cost_request.full_run_fallback_p50_usd:.2f}"
        )
        print(
            "                           cold-start fallback uses the smoke bucket "
            "or smoke-scaled fallback, never the full-run p50"
        )
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
            required_var = _required_env_var_reference(v)
            if (
                required_var is not None
                and required_var != "INSTANCE_JOB_ID"
                and required_var not in os.environ
            ):
                raise SystemExit(
                    "[operator-authorize] FATAL: recipe env_overrides requires "
                    f"explicit environment variable {required_var} for key {k}; "
                    "no fallback is declared, refusing provider dispatch before "
                    "a phantom default can be launched."
                )
            resolved = os.environ.get(str(k), _resolve_env_var(v, v))
            # Substitute ${INSTANCE_JOB_ID} sentinel if present.
            if isinstance(resolved, str) and "${INSTANCE_JOB_ID}" in resolved:
                resolved = resolved.replace("${INSTANCE_JOB_ID}", instance_job_id)
            parts.append(f"{k}={resolved}")
    return ",".join(parts)


# Catalog #201: sentinel-files MUST be under the canonical Modal mount set.
# Files in `.omx/`, `.ralph/`, `configs/`, `docs/`, `cuda/`, `runtime-rs/`, etc.
# are operator-side and NEVER mounted onto the Modal worker — so they cannot
# satisfy Catalog #166's worker-side hash check and dispatch fails rc=13.
# Keep this list synchronized with mount_manifest.STRUCTURAL_MINIMUM_DIRS.
_MODAL_MOUNT_SET_PREFIXES: tuple[str, ...] = (
    "src/",
    "scripts/",
    "upstream/",
    "submissions/",
    "experiments/",
    "tools/",
    "pyproject.toml",
)


def _path_under_modal_mount_set(rel: str) -> bool:
    """True iff `rel` is mounted by mount_manifest.STRUCTURAL_MINIMUM_DIRS."""
    rel = rel.strip()
    while rel.startswith("./"):
        rel = rel[2:]
    return any(
        rel.startswith(p) if p.endswith("/") else rel == p
        for p in _MODAL_MOUNT_SET_PREFIXES
    )


def _iter_sentinel_path_values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = [item for item in value if isinstance(item, str)]
    else:
        values = [str(value)]
    return [item.strip() for item in values if item.strip()]


def _modal_sentinel_files(recipe: Recipe) -> str:
    """Return comma-separated source-custody sentinels for Modal uploads.

    Per CLAUDE.md SIREN audit 2026-05-13 CRITICAL #2 + Catalog #191
    (``check_modal_dispatch_threads_sentinel_files_per_catalog_166``): the
    auto-discovered set (dispatcher / lane_script / cost_band_trainer /
    trainer) is good defense-in-depth, but substrate-specific modules
    (e.g. ``src/tac/substrates/siren/score_aware_loss.py``) are not
    captured. Recipes can declare their own ``sentinel_files: [...]`` list
    and those paths are appended to the auto-discovered set.

    Catalog #201 (2026-05-13, PR95++ smoke fc-01KRHNMT4SEB794HFPH5GNTHFP):
    sentinels MUST be under the canonical Modal mount set. The recipe YAML
    at ``.omx/operator_authorize_recipes/`` is operator-side metadata that
    ``operator_authorize.py`` reads to make dispatch decisions; the Modal
    worker never reads it. Sentinels outside the mount set ALWAYS produce
    ``MISSING_WORKER`` in Catalog #166's hash ledger and refuse dispatch
    rc=13 before training starts. Drift detection for operator-side files
    should be a separate operator-side check (compare sha at dispatch start
    vs end), not a worker-side Catalog #166 mismatch.
    """

    raw_paths: list[str] = [
        "experiments/modal_train_lane.py",
        "tools/operator_authorize.py",
        "tools/run_modal_smoke_before_full.py",
        "src/tac/deploy/modal/mount_manifest.py",
    ]
    if recipe.remote_driver:
        raw_paths.append(recipe.remote_driver)
    modal_cfg = recipe.raw.get("modal", {}) or {}
    if isinstance(modal_cfg, dict):
        for key in ("lane_script", "cost_band_trainer"):
            value = modal_cfg.get(key)
            if value:
                raw_paths.append(str(value))
    trainer = recipe.raw.get("required_input_files_trainer")
    raw_paths.extend(_iter_sentinel_path_values(trainer))
    # Recipe-declared sentinel_files (Catalog #191): substrate-specific
    # modules that must remain stable across the dispatch window.
    recipe_sentinels = recipe.raw.get("sentinel_files") or []
    if isinstance(recipe_sentinels, list):
        for entry in recipe_sentinels:
            raw_paths.extend(_iter_sentinel_path_values(entry))

    seen: set[str] = set()
    out: list[str] = []
    skipped_outside_mount: list[str] = []
    for rel in raw_paths:
        rel = rel.strip()
        while rel.startswith("./"):
            rel = rel[2:]
        if not rel or rel in seen:
            continue
        if not (REPO_ROOT / rel).is_file():
            continue
        if not _path_under_modal_mount_set(rel):
            # Catalog #201 runtime filter: silently dropping would mask the
            # bug; warn loudly so operators see the misconfigured sentinel.
            skipped_outside_mount.append(rel)
            continue
        seen.add(rel)
        out.append(rel)
    if skipped_outside_mount:
        print(
            "[operator-authorize] Catalog #201 WARN: dropping sentinel(s) "
            f"outside Modal mount set: {skipped_outside_mount}. Move the file "
            "under src/scripts/upstream/submissions/experiments/tools/ or "
            "drop it from `sentinel_files`.",
            file=sys.stderr,
        )
    return ",".join(out)


def _run_modal_dispatch_process(
    cmd: list[str],
) -> subprocess.CompletedProcess[str]:
    """Run the Modal CLI while capturing output for pre-spawn retry checks."""

    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _print_modal_dispatch_output(proc: subprocess.CompletedProcess[str]) -> None:
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        print(
            proc.stderr,
            end="" if proc.stderr.endswith("\n") else "\n",
            file=sys.stderr,
        )


def _load_modal_mount_retry_helpers() -> tuple[Any, Any, Any]:
    """Import reusable Modal upload-race helpers from the deploy layer."""

    src_dir = str(REPO_ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from tac.deploy.modal.mount_manifest import (
        modal_mount_upload_cli_retry_settings,
        modal_output_indicates_mount_upload_race,
        modal_output_indicates_spawned_call,
    )

    return (
        modal_mount_upload_cli_retry_settings,
        modal_output_indicates_mount_upload_race,
        modal_output_indicates_spawned_call,
    )


def _modal_mount_upload_retry_settings() -> tuple[int, float]:
    settings_fn, _race_fn, _spawn_fn = _load_modal_mount_retry_helpers()
    return settings_fn()


def _modal_output_indicates_mount_upload_race(output: str) -> bool:
    _settings_fn, race_fn, _spawn_fn = _load_modal_mount_retry_helpers()
    return bool(race_fn(output))


def _modal_output_indicates_spawned_call(output: str) -> bool:
    _settings_fn, _race_fn, spawn_fn = _load_modal_mount_retry_helpers()
    return bool(spawn_fn(output))


# Catalog #339 (SILENT-NO-SPAWN-STRUCTURAL-EXTINCTION 2026-05-19) sister
# mitigation helpers — extract the call_id from the modal_train_lane.py
# stdout banner and poll the canonical ledger for its appearance. If
# `.spawn()` happened (stdout marker present) but the ledger has no row
# within ~10s, the harvester is blind. _dispatch_modal raises SystemExit
# in that case.
_MODAL_CALL_ID_PATTERN = re.compile(r"\bcall_id=(fc-[A-Z0-9]+)\b")


def _extract_modal_call_id_from_output(output: str) -> str | None:
    """Extract the first ``call_id=fc-...`` literal from Modal stdout/stderr.

    Used by :func:`_dispatch_modal` Catalog #339 sister mitigation to verify
    the canonical ledger has a row for the dispatched call_id within the
    10-second polling window after a successful `.spawn()`.
    """

    match = _MODAL_CALL_ID_PATTERN.search(output)
    if match is None:
        return None
    return match.group(1)


def _poll_ledger_for_dispatched_call(
    call_id: str, *, timeout_seconds: float = 10.0
) -> bool:
    """Poll the canonical Modal call_id ledger for ``call_id`` presence.

    Catalog #339 sister mitigation. Returns ``True`` once a row with
    ``call_id`` is observed (any event_type); ``False`` if the polling
    window elapses.

    Lazy-imports the canonical helper so missing-helper does NOT crash
    the dispatcher (the wrapper-side fail-closed path at Catalog #339
    Layer 2 is the authoritative protection — this is defense in depth).
    """

    src_dir = str(REPO_ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    try:
        from tac.deploy.modal.call_id_ledger import poll_ledger_for_call_id
    except Exception:
        # If the canonical helper is missing or broken, defer to the
        # wrapper's Catalog #339 Layer 2 protection (which exits non-zero
        # on registration failure). Return True to avoid a false-positive
        # SystemExit when the protection is already broken-but-noisy.
        return True
    return bool(poll_ledger_for_call_id(call_id, timeout_seconds=timeout_seconds))


def _dispatch_modal(
    recipe: Recipe,
    instance_job_id: str,
    env_overrides: str,
    *,
    agent: str = "claude:operator_authorize",
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
        "--lane-id",
        recipe.lane_id,
        "--label",
        instance_job_id,
        "--gpu",
        gpu,
        "--timeout-hours",
        f"{timeout_hours}",
        "--env-overrides",
        env_overrides,
        "--agent",
        agent,
    ]
    # Catalog #202: paired-env-var bypass of `--require-clean-head` for
    # trusted sentinel-clean dispatches. The bypass helper raises
    # SystemExit(12) if the intent flag is set without the attestation flag,
    # so a partial-config bypass cannot silently take effect. When the
    # bypass is inactive (default), `--require-clean-head` is appended so the
    # Modal worker fails-closed on any uncommitted edit. Catalog #166's
    # worker-side sentinel hash check (experiments/modal_train_lane.py)
    # runs INDEPENDENTLY of `--require-clean-head` regardless.
    if not _whole_tree_clean_check_bypass_active(recipe):
        cmd.append("--require-clean-head")
    else:
        print(
            "[operator-authorize] Catalog #202: skipping --require-clean-head "
            "per paired env attestation. Catalog #166 worker-side hash check "
            "still runs.",
            file=sys.stderr,
        )
    sentinel_files = _modal_sentinel_files(recipe)
    if sentinel_files:
        cmd.extend(["--sentinel-files", sentinel_files])
    trainer_module_path = (
        recipe.raw.get("required_input_files_trainer")
        or modal_cfg.get("cost_band_trainer")
    )
    if trainer_module_path:
        cmd.extend(["--trainer-module-path", str(trainer_module_path)])
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
    max_attempts, retry_sleep_seconds = _modal_mount_upload_retry_settings()
    last_rc = 1
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            print(
                "[operator-authorize] Catalog #165 Modal mount-upload retry "
                f"{attempt}/{max_attempts} for instance_job_id={instance_job_id}"
            )
        proc = _run_modal_dispatch_process(cmd)
        _print_modal_dispatch_output(proc)
        last_rc = proc.returncode
        combined_output = f"{proc.stdout}\n{proc.stderr}"
        spawned_call = _modal_output_indicates_spawned_call(combined_output)
        if proc.returncode == 0:
            if spawned_call:
                # Catalog #339 (SILENT-NO-SPAWN-STRUCTURAL-EXTINCTION
                # 2026-05-19) — sister mitigation. The stdout-marker check
                # confirms `.spawn()` happened but does NOT prove the
                # canonical ledger row was written. Today's 3 silent-no-spawn
                # incidents would have leaked PAST this check if the wrapper
                # had silently swallowed `register_dispatched_call_id`
                # failure. Poll the canonical ledger for the call_id within
                # 10s. If the row never appears, the harvester is blind —
                # raise SystemExit citing Catalog #339 so the operator gets
                # an actionable diagnostic instead of silent success.
                call_id = _extract_modal_call_id_from_output(combined_output)
                if call_id is not None and not _poll_ledger_for_dispatched_call(
                    call_id, timeout_seconds=10.0
                ):
                    raise SystemExit(
                        f"[operator-authorize] FATAL [Catalog #339]: Modal "
                        f"dispatcher exited rc=0 AND printed spawned-call "
                        f"marker AND call_id={call_id} but the canonical "
                        f"ledger at .omx/state/modal_call_id_ledger.jsonl "
                        f"has no matching row after 10s of polling. The "
                        f"harvester is BLIND to this dispatch per CLAUDE.md "
                        f"'Modal .spawn() HARVEST OR LOSE' non-negotiable. "
                        f"Check .omx/state/modal_call_id_ledger_recovery_tmp/ "
                        f"for last-resort dumps and run "
                        f"`tools/harvest_modal_calls.py --recover-from-tmp`."
                    )
                return 0
            print(
                "[operator-authorize] FATAL: Modal dispatcher exited rc=0 "
                "without a spawned function call marker. Refusing to treat "
                "Modal app initialization / mount creation as a dispatched job; "
                "expected [modal_train_lane] dispatch_completed call_id=fc-... "
                "or an equivalent .spawn() call_id marker.",
                file=sys.stderr,
            )
            return 1
        if spawned_call:
            return proc.returncode
        if not _modal_output_indicates_mount_upload_race(combined_output):
            return proc.returncode
        if attempt >= max_attempts:
            print(
                "[operator-authorize] FATAL: Modal mount set remained unstable "
                f"after {max_attempts} pre-spawn attempt(s); failing closed. "
                "Catalog #165 was not bypassed, and no retry is attempted "
                "after a Modal call_id appears.",
                file=sys.stderr,
            )
            return proc.returncode
        print(
            "[operator-authorize] Catalog #165 detected a pre-spawn Modal "
            "mount upload race; waiting for source mtimes to settle before "
            f"retrying in {retry_sleep_seconds:.1f}s.",
            file=sys.stderr,
        )
        if retry_sleep_seconds > 0:
            time.sleep(retry_sleep_seconds)
    return last_rc



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


# --- Local research-signal dispatchers ----------------------------------------
#
# Operator directive 2026-05-17 verbatim: *"Deploying to local MPS versus modal
# should be super easy to configure, like one arg in a func"* + *"Do everything
# possible you can to accelerate dev velocity and save money using local MPS"*.
#
# Per CLAUDE.md "MPS auth eval is NOISE" non-negotiable + Catalog #1 + Catalog
# #192: local MPS and local CPU dispatches are PERMANENTLY non-authoritative.
# These dispatchers route results through the canonical
# ``tac.optimization.mps_research_signal.append_manifest_row_to_jsonl`` (for
# local_mps) or ``tac.optimization.macos_cpu_advisory_signal.append_manifest_
# row_to_jsonl`` (for local_cpu) helpers so the canonical
# ``[contest-CPU]``/``[contest-CUDA]`` posterior is structurally protected.
#
# The implementation auto-stamps ``evidence_grade="MPS-research-signal"`` /
# ``"macOS-CPU-advisory"`` + ``score_claim=False`` + ``promotion_eligible=False`` +
# ``ready_for_exact_eval_dispatch=False`` so a malformed caller cannot bypass
# the contract. A loud banner notifies the operator that the dispatch is
# NON-AUTHORITATIVE.

# Canonical evidence grade tokens (mirrored from sister helpers; presence
# verified at module import + by STRICT preflight gate Catalog #317).
_LOCAL_MPS_EVIDENCE_GRADE = "MPS-research-signal"
_LOCAL_CPU_EVIDENCE_GRADE = "macOS-CPU-advisory"
_LOCAL_MPS_MANIFEST_JSONL = ".omx/state/mps_research_signal_manifest.jsonl"
_LOCAL_CPU_MANIFEST_JSONL = ".omx/state/macos_cpu_advisory_signal_manifest.jsonl"
_LOCAL_MPS_BANNER = (
    "\n"
    "[LOCAL-MPS RESEARCH-SIGNAL — NON-AUTHORITATIVE]\n"
    "    Per CLAUDE.md 'MPS auth eval is NOISE' non-negotiable + Catalog #1.\n"
    "    Results are PERMANENTLY non-authoritative; manifest writes to\n"
    f"    {_LOCAL_MPS_MANIFEST_JSONL} with evidence_grade={_LOCAL_MPS_EVIDENCE_GRADE!r}.\n"
    "    The canonical [contest-CPU] / [contest-CUDA] posterior is NEVER touched.\n"
    "    Hardware: Apple Silicon Metal (MPS). Cost: $0 (operator machine time).\n"
)
_LOCAL_CPU_BANNER = (
    "\n"
    "[LOCAL-CPU ADVISORY — NON-AUTHORITATIVE]\n"
    "    Per CLAUDE.md 'Submission auth eval — BOTH CPU AND CUDA' non-negotiable + Catalog #192.\n"
    "    macOS-CPU is NEVER 1:1 contest-compliant; promotion requires a paired\n"
    "    [contest-CPU GHA Linux x86_64] anchor. Manifest writes to\n"
    f"    {_LOCAL_CPU_MANIFEST_JSONL} with evidence_grade={_LOCAL_CPU_EVIDENCE_GRADE!r}.\n"
    "    The canonical [contest-CPU] / [contest-CUDA] posterior is NEVER touched.\n"
    "    Hardware: Apple Silicon CPU. Cost: $0 (operator machine time).\n"
)


def _build_local_research_signal_env(
    recipe: Recipe, env_overrides: str, *, force_mps_no_fallback: bool
) -> dict[str, str]:
    """Build env dict for local_mps / local_cpu dispatch.

    For local_mps: forces ``PYTORCH_ENABLE_MPS_FALLBACK=0`` so MPS-unavailable
    ops raise rather than silently fall back to CPU (Catalog #1 sister rule).
    For local_cpu: forces no CUDA (operator machine doesn't have it anyway,
    but be explicit) + no MPS (so torch picks pure CPU).
    """
    env = dict(os.environ)
    if env_overrides:
        for pair in env_overrides.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                env[k.strip()] = v.strip()
    if force_mps_no_fallback:
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    return env


def _dispatch_local_mps(
    recipe: Recipe,
    instance_job_id: str,
    env_overrides: str,
) -> int:
    """Invoke a local Apple-silicon MPS dispatch (NON-AUTHORITATIVE).

    Contract:
      1. Requires ``torch.backends.mps.is_available()`` at runtime; FATAL
         otherwise (no silent CPU fallback per CLAUDE.md Catalog #1).
      2. Routes results through canonical
         ``tac.optimization.mps_research_signal.append_manifest_row_to_jsonl``;
         results MUST land at ``.omx/state/mps_research_signal_manifest.jsonl``
         with ``evidence_grade="MPS-research-signal"`` + ``score_claim=False`` +
         ``promotion_eligible=False`` + ``ready_for_exact_eval_dispatch=False``
         (auto-stamped by the canonical helper; the canonical posterior is
         NEVER touched).
      3. Sets ``PYTORCH_ENABLE_MPS_FALLBACK=0`` so MPS-unavailable ops raise
         rather than silently fall back to CPU (Catalog #1 sister rule).
      4. Emits a loud ``[LOCAL-MPS RESEARCH-SIGNAL — NON-AUTHORITATIVE]``
         banner BEFORE the trainer runs so the operator sees the contract.
      5. Refuses if the recipe declares any of ``score_claim: true`` /
         ``promotion_eligible: true`` / ``ready_for_exact_eval_dispatch: true``.
    """
    # Layer 1: hardware availability (no silent fallback).
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "[operator-authorize] FATAL: local-MPS dispatch requires torch; "
            f"install via uv sync: {exc}"
        ) from exc
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise SystemExit(
            "[operator-authorize] FATAL: local-MPS dispatch requested but "
            "torch.backends.mps.is_available()==False on this host. Per "
            "CLAUDE.md Catalog #1 sister rule, MPS-unavailable hosts cannot "
            "fall back to CPU silently. Run on Apple Silicon, OR use "
            "`--target local-cpu` for the macOS-CPU advisory path, OR "
            "use `--target modal` / `vastai` / `lightning` for the paid "
            "authoritative path."
        )

    # Layer 2: canonical helper imports successfully (defense-in-depth).
    try:
        from tac.optimization.mps_research_signal import (
            EVIDENCE_GRADE as _CANONICAL_EVIDENCE_GRADE,
        )
        from tac.optimization.mps_research_signal import (
            append_manifest_row_to_jsonl,
        )
    except ImportError as exc:
        raise SystemExit(
            "[operator-authorize] FATAL: cannot import canonical "
            f"tac.optimization.mps_research_signal helper: {exc}"
        ) from exc
    assert _CANONICAL_EVIDENCE_GRADE == _LOCAL_MPS_EVIDENCE_GRADE, (
        "evidence_grade drift between operator_authorize.py and canonical helper"
    )

    # Layer 3: recipe must not claim authority.
    if recipe.raw.get("score_claim") is True:
        raise SystemExit(
            "[operator-authorize] FATAL: local-MPS recipe cannot declare "
            "`score_claim: true` — local MPS is permanently non-authoritative "
            "per CLAUDE.md 'MPS auth eval is NOISE'."
        )
    if recipe.raw.get("promotion_eligible") is True:
        raise SystemExit(
            "[operator-authorize] FATAL: local-MPS recipe cannot declare "
            "`promotion_eligible: true` — promotion requires [contest-CUDA] "
            "OR [contest-CPU GHA Linux x86_64] evidence."
        )
    if recipe.raw.get("ready_for_exact_eval_dispatch") is True:
        raise SystemExit(
            "[operator-authorize] FATAL: local-MPS recipe cannot declare "
            "`ready_for_exact_eval_dispatch: true`."
        )

    # Layer 4: loud non-authoritative banner.
    print(_LOCAL_MPS_BANNER, file=sys.stderr, flush=True)

    # Layer 5: env injection + dispatch.
    lane_script = recipe.remote_driver
    if not lane_script:
        raise SystemExit(
            "[operator-authorize] FATAL: local-MPS recipe needs 'remote_driver'"
        )
    if not (REPO_ROOT / lane_script).exists():
        raise SystemExit(
            f"[operator-authorize] FATAL: remote_driver missing at {lane_script}"
        )
    env = _build_local_research_signal_env(
        recipe, env_overrides, force_mps_no_fallback=True
    )
    env["PACT_LOCAL_RESEARCH_SIGNAL_KIND"] = "local_mps"
    env["PACT_LOCAL_INSTANCE_JOB_ID"] = instance_job_id
    cmd = ["bash", lane_script]
    print(
        f"[operator-authorize] dispatching local_mps: {' '.join(cmd)} "
        f"(instance_job_id={instance_job_id})"
    )
    rc = subprocess.call(cmd, cwd=str(REPO_ROOT), env=env)

    # Layer 6: write the dispatch row through the canonical helper. The
    # dispatcher records the dispatch attempt + outcome; the trainer is
    # responsible for writing its own per-sweep rows via the same helper.
    manifest_row: dict[str, Any] = {
        "schema": "mps_research_signal_dispatcher_row.v1",
        "instance_job_id": instance_job_id,
        "lane_id": recipe.lane_id,
        "recipe_name": recipe.name,
        "lane_script": lane_script,
        "platform": "local_mps",
        "subprocess_rc": rc,
        "evidence_grade": _CANONICAL_EVIDENCE_GRADE,
        # The canonical helper auto-stamps these but we also set them here for
        # belt-and-suspenders; both surfaces enforce the same contract.
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    try:
        append_manifest_row_to_jsonl(
            manifest_row,
            output_path=REPO_ROOT / _LOCAL_MPS_MANIFEST_JSONL,
        )
    except Exception as exc:
        raise SystemExit(
            "[operator-authorize] FATAL: failed to write MPS research-signal "
            f"manifest row after subprocess_rc={rc}; refusing success to avoid "
            f"signal loss: {exc}"
        ) from exc
    return rc


def _dispatch_local_cpu(
    recipe: Recipe,
    instance_job_id: str,
    env_overrides: str,
) -> int:
    """Invoke a local macOS-CPU dispatch (NON-AUTHORITATIVE advisory).

    Contract (mirror of :func:`_dispatch_local_mps` for the macOS-CPU advisory
    path per CLAUDE.md Catalog #192):
      1. Routes results through canonical
         ``tac.optimization.macos_cpu_advisory_signal.append_manifest_row_to_jsonl``;
         results land at
         ``.omx/state/macos_cpu_advisory_signal_manifest.jsonl`` with
         ``evidence_grade="macOS-CPU-advisory"`` + ``score_claim=False`` +
         ``promotion_eligible=False`` (the canonical posterior is NEVER touched).
      2. Detects whether the host is macOS ARM64 (warns if not — the
         calibration anchor PR107 is M5 Max specific).
      3. Emits a loud ``[LOCAL-CPU ADVISORY — NON-AUTHORITATIVE]`` banner.
      4. Refuses recipes that declare ``score_claim: true`` etc.
    """
    # Layer 1: canonical helper imports successfully.
    try:
        from tac.optimization.macos_cpu_advisory_signal import (
            EVIDENCE_GRADE as _CANONICAL_EVIDENCE_GRADE,
        )
        from tac.optimization.macos_cpu_advisory_signal import (
            append_manifest_row_to_jsonl,
            is_running_on_macos_arm64,
        )
    except ImportError as exc:
        raise SystemExit(
            "[operator-authorize] FATAL: cannot import canonical "
            f"tac.optimization.macos_cpu_advisory_signal helper: {exc}"
        ) from exc
    assert _CANONICAL_EVIDENCE_GRADE == _LOCAL_CPU_EVIDENCE_GRADE, (
        "evidence_grade drift between operator_authorize.py and canonical helper"
    )

    # Layer 2: recipe must not claim authority.
    if recipe.raw.get("score_claim") is True:
        raise SystemExit(
            "[operator-authorize] FATAL: local-CPU recipe cannot declare "
            "`score_claim: true` — local macOS-CPU is permanently advisory."
        )
    if recipe.raw.get("promotion_eligible") is True:
        raise SystemExit(
            "[operator-authorize] FATAL: local-CPU recipe cannot declare "
            "`promotion_eligible: true` — promotion requires [contest-CPU "
            "GHA Linux x86_64] evidence."
        )
    if recipe.raw.get("ready_for_exact_eval_dispatch") is True:
        raise SystemExit(
            "[operator-authorize] FATAL: local-CPU recipe cannot declare "
            "`ready_for_exact_eval_dispatch: true`."
        )

    # Layer 3: host warning (not fatal — operators may legitimately run
    # local_cpu dispatch on Linux x86_64 too, but the evidence grade then
    # warrants different downstream treatment).
    if not is_running_on_macos_arm64():
        print(
            "[operator-authorize] WARN: local-CPU dispatch not on macOS "
            "ARM64; the PR107 calibration anchor is M5 Max specific. "
            "Results still routed through macos_cpu_advisory_signal "
            "manifest (non-authoritative).",
            file=sys.stderr,
        )

    # Layer 4: loud non-authoritative banner.
    print(_LOCAL_CPU_BANNER, file=sys.stderr, flush=True)

    # Layer 5: env injection + dispatch.
    lane_script = recipe.remote_driver
    if not lane_script:
        raise SystemExit(
            "[operator-authorize] FATAL: local-CPU recipe needs 'remote_driver'"
        )
    if not (REPO_ROOT / lane_script).exists():
        raise SystemExit(
            f"[operator-authorize] FATAL: remote_driver missing at {lane_script}"
        )
    env = _build_local_research_signal_env(
        recipe, env_overrides, force_mps_no_fallback=False
    )
    env["PACT_LOCAL_RESEARCH_SIGNAL_KIND"] = "local_cpu"
    env["PACT_LOCAL_INSTANCE_JOB_ID"] = instance_job_id
    # Force CPU at the trainer surface — explicit per CLAUDE.md "Forbidden
    # device-selection defaults" (no silent fallback).
    env.setdefault("PACT_INFLATE_DEVICE", "cpu")
    cmd = ["bash", lane_script]
    print(
        f"[operator-authorize] dispatching local_cpu: {' '.join(cmd)} "
        f"(instance_job_id={instance_job_id})"
    )
    rc = subprocess.call(cmd, cwd=str(REPO_ROOT), env=env)

    # Layer 6: write the dispatch row through the canonical helper.
    manifest_row: dict[str, Any] = {
        "schema": "macos_cpu_advisory_signal_dispatcher_row.v1",
        "instance_job_id": instance_job_id,
        "lane_id": recipe.lane_id,
        "recipe_name": recipe.name,
        "lane_script": lane_script,
        "platform": "local_cpu",
        "subprocess_rc": rc,
        "evidence_grade": _CANONICAL_EVIDENCE_GRADE,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ranking_only": True,
    }
    try:
        append_manifest_row_to_jsonl(
            manifest_row,
            output_path=REPO_ROOT / _LOCAL_CPU_MANIFEST_JSONL,
        )
    except Exception as exc:
        raise SystemExit(
            "[operator-authorize] FATAL: failed to write macOS-CPU advisory "
            f"manifest row after subprocess_rc={rc}; refusing success to avoid "
            f"signal loss: {exc}"
        ) from exc
    return rc


def _poll_ledger_for_dispatched_hf_jobs_id(
    hf_jobs_id: str, *, timeout_seconds: float = 10.0
) -> bool:
    """Poll the canonical HF Jobs ledger for ``hf_jobs_id`` presence.

    Catalog #339 sister mitigation for HF Jobs (sister of
    :func:`_poll_ledger_for_dispatched_call` for Modal). Returns ``True``
    once a row with ``hf_jobs_id`` is observed (any event_type); ``False``
    if the polling window elapses.

    Lazy-imports the canonical helper so missing-helper does NOT crash the
    dispatcher (the wrapper-side fail-closed path at
    :func:`tac.deploy.hf_jobs.job_id_ledger.register_dispatched_hf_jobs_id_fail_closed`
    is the authoritative protection — this is defense in depth).
    """

    src_dir = str(REPO_ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    try:
        from tac.deploy.hf_jobs.job_id_ledger import poll_ledger_for_hf_jobs_id
    except Exception:
        # If the canonical helper is missing or broken, defer to the
        # wrapper's Catalog #339 sister protection (which exits non-zero
        # on registration failure). Return True to avoid a false-positive
        # SystemExit when the protection is already broken-but-noisy.
        return True
    return bool(poll_ledger_for_hf_jobs_id(hf_jobs_id, timeout_seconds=timeout_seconds))


def _extract_hf_jobs_id_from_dispatch_stdout(stdout: str) -> str | None:
    """Extract ``hf_jobs_id`` from dispatcher stdout.

    ``tools/dispatch_hf_jobs_vision_training.py`` emits pretty-printed JSON on
    success. A last-line parse only sees ``}``, so parse the whole stream first,
    then scan for the last JSON object in case future wrappers prepend logs.
    """

    text = (stdout or "").strip()
    if not text:
        return None

    def _id_from_payload(payload: object) -> str | None:
        if isinstance(payload, dict):
            hf_jobs_id = payload.get("hf_jobs_id")
            if isinstance(hf_jobs_id, str) and hf_jobs_id.strip():
                return hf_jobs_id.strip()
        return None

    try:
        return _id_from_payload(json.loads(text))
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    found: str | None = None
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        hf_jobs_id = _id_from_payload(payload)
        if hf_jobs_id:
            found = hf_jobs_id
    return found


def _find_hf_jobs_id_in_ledger_by_lane_label(
    *, lane_id: str, label: str
) -> str | None:
    """Recover a dispatched HF Jobs id from the canonical ledger by lane+label."""

    try:
        from tac.deploy.hf_jobs.job_id_ledger import query_by_lane
    except ImportError:
        return None

    try:
        rows = query_by_lane(lane_id, strict=True)
    except Exception:
        rows = []
    for row in reversed(rows):
        if row.get("label") != label:
            continue
        hf_jobs_id = row.get("hf_jobs_id")
        if isinstance(hf_jobs_id, str) and hf_jobs_id.strip():
            if hf_jobs_id.startswith("pending:"):
                continue
            return hf_jobs_id.strip()
    return None


def _dispatch_hf_jobs(
    recipe: Recipe,
    instance_job_id: str,
    env_overrides: str,
    *,
    timeout_hours_override: float | None = None,
    cost_band_epochs_override: int | None = None,
) -> int:
    """Invoke ``tools/dispatch_hf_jobs_vision_training.py`` for HF Jobs.

    Slot 8 wire-in (2026-05-19; per operator approval verbatim *"all operator
    routable items approved"*). Sister of :func:`_dispatch_modal` for the
    Hugging Face Jobs platform per Catalog #342 (HF-DATASET-JOBS-IMPL-WAVE
    landing 2026-05-19 commit ``e588d9f65``).

    Contract:

    1. **Pre-dispatch harness** (Catalog #243 + #271 + #313): runs via the
       canonical insertion points BEFORE this function fires; this dispatcher
       inherits the same protections every paid platform gets.
    2. **Resolved arguments**: the recipe must declare ``hf_jobs.script`` +
       ``hf_jobs.hub_dataset_repo`` + ``hf_jobs.hub_model_repo`` so the
       canonical dispatcher CLI is invocable without operator-side guesswork.
    3. **Ledger-poll fail-closed** (Catalog #339 sister): after the
       subprocess returns rc=0, poll
       ``.omx/state/hf_jobs_call_id_ledger.jsonl`` for the canonical
       ``register_dispatched_hf_jobs_id`` row. If the ledger row never
       appears within 10s, raise SystemExit citing Catalog #339 — the
       harvester is BLIND to this dispatch per CLAUDE.md "Modal `.spawn()`
       HARVEST OR LOSE" applied to HF Jobs.
    4. **Paired-env operator bypass** (Catalog #199): the canonical operator
       bypass via ``OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1``
       + ``OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=<USD>`` is honored upstream
       in ``_confirm()``; this dispatcher does NOT re-validate.
    """

    hf_cfg = recipe.raw.get("hf_jobs", {}) or {}
    script = hf_cfg.get("script")
    if not script:
        raise SystemExit(
            "[operator-authorize] FATAL: hf_jobs recipe missing 'hf_jobs.script' "
            "(path to the canonical HF Jobs training script)"
        )
    if not (REPO_ROOT / str(script)).exists():
        raise SystemExit(
            "[operator-authorize] FATAL: HF Jobs training script missing at "
            f"{script}"
        )
    hub_dataset_repo = hf_cfg.get("hub_dataset_repo")
    hub_model_repo = hf_cfg.get("hub_model_repo")
    if not hub_dataset_repo or not hub_model_repo:
        raise SystemExit(
            "[operator-authorize] FATAL: hf_jobs recipe missing "
            "'hf_jobs.hub_dataset_repo' and/or 'hf_jobs.hub_model_repo'"
        )
    flavor = hf_cfg.get("flavor", "t4-small")
    model = hf_cfg.get("model")
    num_epochs = cost_band_epochs_override or hf_cfg.get("num_epochs", 200)
    learning_rate = hf_cfg.get("learning_rate")
    train_batch_size = hf_cfg.get("train_batch_size")
    eval_batch_size = hf_cfg.get("eval_batch_size")
    timeout_seconds_default = hf_cfg.get("timeout_seconds", 14400)
    if timeout_hours_override is not None:
        timeout_seconds = round(float(timeout_hours_override) * 3600.0)
    else:
        timeout_seconds = int(timeout_seconds_default)
    hub_dataset_sha = hf_cfg.get("hub_dataset_sha")
    expected_axis = hf_cfg.get("expected_axis", "cuda")
    upstream_snapshot_sha256 = hf_cfg.get("upstream_snapshot_sha256")

    cmd: list[str] = [
        _python_bin(),
        "tools/dispatch_hf_jobs_vision_training.py",
        "--script",
        str(script),
        "--hub-dataset-repo",
        str(hub_dataset_repo),
        "--hub-model-repo",
        str(hub_model_repo),
        "--flavor",
        str(flavor),
        "--num-epochs",
        str(num_epochs),
        "--timeout-seconds",
        str(timeout_seconds),
        "--lane-id",
        recipe.lane_id,
        "--label",
        instance_job_id,
        "--expected-axis",
        str(expected_axis),
    ]
    if model:
        cmd.extend(["--model", str(model)])
    if learning_rate is not None:
        cmd.extend(["--learning-rate", str(learning_rate)])
    if train_batch_size is not None:
        cmd.extend(["--train-batch-size", str(train_batch_size)])
    if eval_batch_size is not None:
        cmd.extend(["--eval-batch-size", str(eval_batch_size)])
    if hub_dataset_sha:
        cmd.extend(["--hub-dataset-sha", str(hub_dataset_sha)])
    if upstream_snapshot_sha256:
        cmd.extend(["--upstream-snapshot-sha256", str(upstream_snapshot_sha256)])
    recipe_path = recipe.raw.get("_recipe_path")
    if recipe_path:
        cmd.extend(["--recipe", str(recipe_path)])
    extra_script_args = hf_cfg.get("extra_script_args") or []
    for arg in extra_script_args:
        cmd.extend(["--extra-script-arg", str(arg)])

    print(f"[operator-authorize] dispatching hf_jobs: {' '.join(cmd)}")

    # Build env dict from comma-separated overrides (same shape as
    # ``_dispatch_local`` so the wrapper inherits operator-style env passthrough).
    env = dict(os.environ)
    if env_overrides:
        for pair in env_overrides.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                env[k.strip()] = v.strip()

    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)

    if proc.returncode != 0:
        return proc.returncode

    # Catalog #339 sister (HF Jobs) — extract the hf_jobs_id from the
    # canonical dispatcher's stdout payload (pretty-printed JSON on success)
    # and poll the canonical ledger for the row. If stdout is malformed, fall
    # back to the ledger by the unique operator-authorize label. If neither
    # surface yields an id, fail closed: the harvester would be blind.
    hf_jobs_id = _extract_hf_jobs_id_from_dispatch_stdout(proc.stdout or "")
    if not hf_jobs_id:
        hf_jobs_id = _find_hf_jobs_id_in_ledger_by_lane_label(
            lane_id=recipe.lane_id,
            label=instance_job_id,
        )

    if not hf_jobs_id:
        raise SystemExit(
            "[operator-authorize] FATAL [Catalog #339 sister]: HF Jobs "
            "dispatcher exited rc=0 but did not emit a parseable hf_jobs_id "
            "and no canonical ledger row matched "
            f"lane_id={recipe.lane_id} label={instance_job_id}. The harvester "
            "would be blind to this dispatch; refusing silent success."
        )

    if hf_jobs_id and not _poll_ledger_for_dispatched_hf_jobs_id(
        hf_jobs_id, timeout_seconds=10.0
    ):
        raise SystemExit(
            f"[operator-authorize] FATAL [Catalog #339 sister]: HF Jobs "
            f"dispatcher exited rc=0 AND emitted hf_jobs_id={hf_jobs_id} "
            f"but the canonical ledger at "
            f".omx/state/hf_jobs_call_id_ledger.jsonl has no matching row "
            f"after 10s of polling. The harvester is BLIND to this dispatch "
            f"per CLAUDE.md 'Modal .spawn() HARVEST OR LOSE' applied to HF "
            f"Jobs. Check "
            f".omx/state/hf_jobs_call_id_ledger_recovery_tmp/ for last-resort "
            f"dumps; a future `tools/harvest_hf_jobs_calls.py "
            f"--recover-from-tmp` sister CLI will re-attempt the canonical "
            f"append."
        )
    return 0


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
    elif platform in {"local_mps", "local_cpu"}:
        # Local research-signal dispatches need a remote_driver shell script;
        # the dispatcher functions handle hardware-availability + canonical-
        # helper-import gating at dispatch time (not here). Sister 2026-05-17
        # (lane_one_arg_local_mps_vs_modal_dispatch_switch_20260517).
        lane_script = recipe.remote_driver
        if not lane_script or not (REPO_ROOT / str(lane_script)).exists():
            raise SystemExit(
                "[operator-authorize] FATAL: local research-signal "
                "remote_driver missing before claim: "
                f"{lane_script} (platform={platform})"
            )
    elif platform == "hf_jobs":
        # Slot 8 wire-in (Catalog #342 sister of #245). The dispatcher
        # function handles hub-token + script-existence + canonical-helper
        # gating at dispatch time. Here we only validate the canonical
        # dispatcher CLI is present + the recipe declares hf_jobs.script.
        hf_cfg = recipe.raw.get("hf_jobs", {}) or {}
        script = hf_cfg.get("script")
        if not script:
            raise SystemExit(
                "[operator-authorize] FATAL: hf_jobs recipe missing "
                "'hf_jobs.script' before claim"
            )
        if not (REPO_ROOT / str(script)).exists():
            raise SystemExit(
                "[operator-authorize] FATAL: HF Jobs training script missing "
                f"at {script} (platform=hf_jobs)"
            )
        dispatcher = REPO_ROOT / "tools" / "dispatch_hf_jobs_vision_training.py"
        if not dispatcher.exists():
            raise SystemExit(
                "[operator-authorize] FATAL: canonical HF Jobs dispatcher "
                f"missing at {dispatcher} (Catalog #342 sister; slot 7 "
                "should have landed this at commit e588d9f65)"
            )


def _resolve_routing_decision(recipe: Recipe) -> dict[str, Any] | None:
    """Consult the D9 per-class routing API for one recipe.

    Council Decision 9 (binding verdict, omnibus commit 7872c9f4b):
    per-class canonical routing with cost-band-posterior dynamic
    re-routing (Time-Traveler amendment).

    Returns a routing decision dict (from
    :func:`tac.cost_band_calibration.select_provider_for_recipe`) or
    ``None`` if the routing API is unavailable. The caller logs the
    decision; the operator's explicit ``platform:`` choice in the recipe
    YAML still wins (the routing decision flags this as a recipe override
    via its ``rationale`` string). This preserves backward compatibility
    with every existing recipe while making the canonical routing visible
    at dispatch time.

    Operators may opt into auto-routing by setting either:
      - ``platform: auto`` (recipe-side opt-in to canonical Decision 9
        provider — the routing helper resolves the platform AND gpu and
        the dispatch is rerouted)
      - ``dispatch_class: <smoke|full|long_burn|eval|cpu>`` (explicit
        class override; auto-routing applies the canonical for that class)
    """
    try:
        from tac.cost_band_calibration import select_provider_for_recipe

        decision = select_provider_for_recipe(
            recipe.raw,
            consult_posterior=True,
        )
        return decision.as_dict()
    except Exception as exc:  # pragma: no cover - defensive
        print(
            f"[operator-authorize] D9 routing API unavailable; passing through "
            f"recipe.platform={recipe.platform!r} (error: {exc})",
            file=sys.stderr,
        )
        return None


def _maybe_apply_auto_routing(recipe: Recipe) -> Recipe:
    """If the recipe declares ``platform: auto``, resolve to canonical via D9.

    Returns a possibly-mutated Recipe: the in-memory raw dict has
    ``platform`` and ``gpu`` rewritten to the routing decision's resolved
    pair when ``platform: auto`` is the operator's declared intent. The
    routing decision is recorded under ``raw["_d9_routing_decision"]`` for
    forensic audit.

    When the recipe carries an explicit ``platform: <provider>`` (the
    legacy default, e.g. ``platform: modal``), this helper is a no-op:
    the operator's explicit choice wins per CLAUDE.md "Subagent
    coherence-by-default" anti-fragmentation primitive.
    """
    raw_platform = (recipe.raw.get("platform") or "").strip().lower()
    if raw_platform != "auto":
        # Operator explicitly chose this platform; record D9's
        # recommendation for forensics but do NOT mutate.
        decision = _resolve_routing_decision(recipe)
        if decision is not None:
            recipe.raw["_d9_routing_decision"] = decision
            print(
                f"[operator-authorize] D9 routing: class="
                f"{decision['dispatch_class']!r} canonical="
                f"{decision['canonical_provider']}/{decision['canonical_gpu']} "
                f"(operator chose {recipe.platform}/{recipe.gpu}; pass-through)"
            )
        return recipe

    decision = _resolve_routing_decision(recipe)
    if decision is None:
        raise SystemExit(
            "[operator-authorize] FATAL: recipe declared platform=auto but "
            "the D9 routing API is unavailable; cannot resolve canonical "
            "provider/gpu. Either install tac.cost_band_calibration or "
            "declare an explicit platform: in the recipe."
        )
    recipe.raw["platform"] = decision["provider"]
    recipe.raw["gpu"] = decision["gpu"]
    recipe.raw["_d9_routing_decision"] = decision
    print(
        f"[operator-authorize] D9 routing: class={decision['dispatch_class']!r} "
        f"-> {decision['provider']}/{decision['gpu']} "
        f"(canonical={decision['canonical_provider']}/{decision['canonical_gpu']}, "
        f"re_routed={decision['re_routed']})"
    )
    if decision.get("rationale"):
        print(f"[operator-authorize] D9 rationale: {decision['rationale']}")
    return recipe


def _run_dispatch(
    recipe: Recipe,
    instance_job_id: str,
    *,
    agent: str = "claude:operator_authorize",
    timeout_hours_override: float | None = None,
    cost_band_epochs_override: int | None = None,
) -> int:
    """Route to the appropriate platform dispatcher.

    D9 per-class provider routing (council Decision 9, omnibus commit
    7872c9f4b) is consulted before the platform-keyed dispatch fork.
    Recipes declaring ``platform: auto`` are auto-routed to the canonical
    Decision 9 provider/gpu; recipes with an explicit ``platform:`` value
    pass through unchanged but the routing recommendation is logged for
    forensics. See :func:`_maybe_apply_auto_routing`.
    """
    recipe = _maybe_apply_auto_routing(recipe)
    env_overrides = _build_env_overrides(recipe, instance_job_id)
    platform = recipe.platform
    if platform == "modal":
        return _dispatch_modal(
            recipe,
            instance_job_id,
            env_overrides,
            agent=agent,
            timeout_hours_override=timeout_hours_override,
            cost_band_epochs_override=cost_band_epochs_override,
        )
    if platform in {"vastai", "vast"}:
        return _dispatch_vastai(recipe, instance_job_id, env_overrides)
    if platform == "local":
        return _dispatch_local(recipe, instance_job_id, env_overrides)
    if platform == "local_mps":
        return _dispatch_local_mps(recipe, instance_job_id, env_overrides)
    if platform == "local_cpu":
        return _dispatch_local_cpu(recipe, instance_job_id, env_overrides)
    if platform == "hf_jobs":
        # Slot 8 wire-in (2026-05-19; Catalog #342 sister of #245).
        return _dispatch_hf_jobs(
            recipe,
            instance_job_id,
            env_overrides,
            timeout_hours_override=timeout_hours_override,
            cost_band_epochs_override=cost_band_epochs_override,
        )
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

    return platform in {
        "modal",
        "vastai",
        "vast",
        "local",
        "local_mps",
        "local_cpu",
        "hf_jobs",
    }


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
        "--yes",
        action="store_true",
        help=(
            "Non-interactive confirmation for caller-owned workflows that have "
            "already recorded operator approval."
        ),
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
    parser.add_argument(
        "--cost-band-gpu-override",
        default=None,
        help=(
            "Override cost_band.gpu_key for this authorization. Used by "
            "smoke-before-full wrappers when the smoke GPU differs from the "
            "full-run recipe GPU."
        ),
    )
    # The "one-arg toggle" the operator asked for 2026-05-17 verbatim:
    # *"Deploying to local MPS versus modal should be super easy to configure,
    # like one arg in a func"*. CLI > recipe precedence; CLI override is
    # written into recipe.raw["platform"] BEFORE _maybe_apply_auto_routing
    # consumes the recipe, and a sister `cli_target_override` field is
    # recorded for forensic clarity.
    parser.add_argument(
        "--target",
        choices=[
            "auto",
            "modal",
            "vastai",
            "lightning",
            "local",
            "local-mps",
            "local-cpu",
            "hf_jobs",
            "hf-jobs",
            "kaggle",
            "gha",
            "azure",
            "none",
        ],
        default=None,
        help=(
            "One-arg platform override. Wins over recipe `platform:`. "
            "`local-mps` and `local-cpu` route through the canonical "
            "MPS-research-signal / macOS-CPU-advisory manifests "
            "(NON-AUTHORITATIVE per CLAUDE.md 'MPS auth eval is NOISE'). "
            "`hf_jobs` / `hf-jobs` route through the canonical HF Jobs "
            "dispatcher (Catalog #342 sister; slot 8 wire-in 2026-05-19). "
            "Per operator directive 2026-05-17: *'Deploying to local MPS "
            "versus modal should be super easy to configure, like one arg "
            "in a func'*."
        ),
    )
    args = parser.parse_args(argv)

    if args.list:
        return _list_recipes()

    if not args.recipe:
        parser.print_help()
        return 2

    recipe = _load_recipe(args.recipe)
    # Apply --target CLI override BEFORE downstream consumers read recipe.platform.
    # CLI > recipe precedence per operator directive 2026-05-17.
    if args.target is not None:
        # Map dash-form CLI values to underscore-form platform tokens.
        target_to_platform = {
            "local-mps": "local_mps",
            "local-cpu": "local_cpu",
            "hf-jobs": "hf_jobs",
        }
        new_platform = target_to_platform.get(args.target, args.target)
        original_platform = recipe.raw.get("platform")
        recipe.raw["platform"] = new_platform
        recipe.raw["cli_target_override"] = {
            "from": str(original_platform) if original_platform is not None else "<unset>",
            "to": new_platform,
            "via": "operator_authorize.py --target",
        }
        print(
            f"[operator-authorize] --target {args.target}: overriding "
            f"recipe.platform from {original_platform!r} to {new_platform!r}",
            file=sys.stderr,
        )
    dispatch_refusal = _recipe_dispatch_refusal(recipe)
    direct_smoke_refusal = _smoke_only_direct_dispatch_refusal(
        recipe,
        cost_band_epochs_override=args.cost_band_epochs_override,
    )
    if direct_smoke_refusal:
        dispatch_refusal = (
            f"{dispatch_refusal}; {direct_smoke_refusal}"
            if dispatch_refusal
            else direct_smoke_refusal
        )

    try:
        cost_request = _resolve_cost_band_request(
            recipe,
            cost_band_epochs_override=args.cost_band_epochs_override,
            cost_band_gpu_override=args.cost_band_gpu_override,
        )
    except ValueError as exc:
        if dispatch_refusal:
            cost_request = CostBandRequest(
                platform_key=recipe.platform,
                gpu_key=recipe.gpu,
                epochs=0,
                all_flags_on=True,
                fallback_p50_usd=0.0,
                context_label="refused",
                full_run_platform_key=recipe.platform,
                full_run_gpu_key=recipe.gpu,
                full_run_epochs=0,
                full_run_fallback_p50_usd=0.0,
            )
        else:
            raise SystemExit(
                f"[operator-authorize] FATAL: dispatchable recipe has invalid "
                f"cost-band metadata: {exc}; fix the recipe before operator "
                "authorization"
            ) from None
    band = _predict_cost_band(
        platform_key=cost_request.platform_key,
        gpu_key=cost_request.gpu_key,
        epochs=cost_request.epochs,
        all_flags_on=cost_request.all_flags_on,
        hand_calibrated_fallback_p50_usd=cost_request.fallback_p50_usd,
    )

    # Print banner.
    _print_banner(recipe, band, cost_request)

    native_dispatch = _platform_has_native_dispatch(recipe.platform)
    if not dispatch_refusal and native_dispatch:
        _validate_declared_local_paths(recipe)

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
    if args.yes:
        print("[operator-authorize] --yes; using caller-owned operator approval")
    elif not _confirm(
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
    # Recipe declares a trainer path under modal.cost_band_trainer or
    # at top-level required_input_files_trainer. Resolve once for both the
    # required-input validator (Catalog #152) AND the local pre-deploy
    # harness (WIRE-AND-INTEGRATE-ALL 2026-05-15).
    trainer = (
        recipe.raw.get("modal", {}).get("cost_band_trainer")
        or recipe.raw.get("required_input_files_trainer")
    )
    if required_files:
        if trainer:
            _validate_required_input_files(str(trainer), recipe)
        else:
            print(
                "[operator-authorize] WARN: recipe declares required_input_files but "
                "no trainer path - skipping local validation",
                file=sys.stderr,
            )

    # 2026-05-16 Catalog #313 — PROBE-OUTCOMES-BAKE-IN. Refuse dispatch if
    # this recipe's substrate has a recent blocking probe-disambiguator
    # verdict in the canonical adjudicated-outcomes ledger. Empirical anchor:
    # ATW v2 D4 H(latent|scorer_class) probe returned INDEPENDENT verdict
    # 2026-05-16 — without this gate the apparatus could re-fire dispatch on
    # the same architectural surface despite the question already being
    # settled. Paired-env bypass per Catalog #199 sister rule.
    _check_predecessor_probe_outcome(recipe.path)

    # Local pre-deploy check (WIRE-AND-INTEGRATE-ALL 2026-05-15). Catches bug
    # classes (syntax errors, NotImplementedError stubs, non-canonical archive
    # grammar, missing auth-eval invocation, inline cuda-fallback, bare ZIP
    # writes) BEFORE the GPU meter starts. Empirical anchor: Z3 v2 + Z4 smoke
    # crashes 2026-05-15 burned ~$4 on bug classes a 30s local check would
    # have caught. Only runs for native-dispatch platforms (modal/vastai/local)
    # since recipe-only platforms do not actually start trainer code.
    if native_dispatch and trainer:
        _run_dispatch_protocol_complete(
            recipe,
            str(trainer),
            native_dispatch=native_dispatch,
        )
        _run_local_pre_deploy_check(str(trainer), recipe.name)
        # 2026-05-15 Catalog #271 - PRE-DISPATCH-CODEX-REVIEW-AUTOMATION
        # operator-approved: wire codex adversarial-review BEFORE every paid
        # Modal/Lightning/Vast.ai dispatch >$1. Catches bug classes the
        # local pre-deploy harness misses (semantic / cross-file / archive-
        # grammar drift) at the cheapest possible surface (BEFORE smoke).
        # Cost gate: helper internally skips codex when estimated cost
        # <= $1 (advisory verdict, no token burn). Paired-env bypass per
        # Catalog #199 sister rule.
        _run_codex_pre_dispatch_review(
            str(trainer),
            str(recipe.path),
            float(band.p50_cost_usd),
        )
    elif native_dispatch:
        _run_dispatch_protocol_complete(
            recipe,
            None,
            native_dispatch=native_dispatch,
        )

    if native_dispatch:
        _native_dispatch_preflight(recipe)

    # Lane claim. Do not create an active claim for recipe-only/no-op platforms:
    # their legacy shims still own the real action and should own any claim.
    instance_job_id = f"{recipe.name}_{_resolve_utc_label()}{args.label_suffix}"
    claim_created = False
    if not args.no_claim and native_dispatch:
        cost_note = (
            f"expected p50 cost ${band.p50_cost_usd:.2f} "
            f"({band.confidence_tag})"
        )
        if cost_request.is_smoke_scaled:
            cost_note += (
                f"; smoke cost context {cost_request.platform_key}/"
                f"{cost_request.gpu_key} x {cost_request.epochs}ep; "
                f"full-run reference {cost_request.full_run_platform_key}/"
                f"{cost_request.full_run_gpu_key} x "
                f"{cost_request.full_run_epochs}ep fallback p50 "
                f"${cost_request.full_run_fallback_p50_usd:.2f}"
            )
        notes = (
            f"operator-authorized via tools/operator_authorize.py --recipe "
            f"{recipe.name}; {cost_note}"
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
            rc = _run_dispatch(recipe, instance_job_id, agent=args.agent)
        else:
            rc = _run_dispatch(
                recipe,
                instance_job_id,
                agent=args.agent,
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
