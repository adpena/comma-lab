# SPDX-License-Identifier: MIT
"""Canonical dispatch optimization protocol — Tier 1/2/3 umbrella verdict.

Operator directive 2026-05-15 (NON-NEGOTIABLE, HIGHEST EMPHASIS):
*"remember the multiple deployments that have failed over and over because of
missing optimizations? we should investigate and develop a protocol for those
too and enforce best practices and production hardened optimization, extreme
optimization and correctness and performance and scalability"*

Empirical anchors (representative, audit table fully enumerated in
``feedback_canonical_dispatch_optimization_protocol_landed_20260515.md``):

- D1 ``substrate_d1_segnet_margin_polytope_modal_t4_dispatch_..._smoke__50ep``
  2026-05-15T08:26:38Z crashed at ``nvml error (999)`` because the lane's
  remote driver was missing the canonical 3-export Modal/CUDA env block
  (``DALI_DISABLE_NVML`` + ``CUBLAS_WORKSPACE_CONFIG`` +
  ``PYTORCH_CUDA_ALLOC_CONF``). Same NVML 999 had hit on D1 6 times in 24h.
- D4 ``fc-01KRK9RKD3QV4C276Y5KXFMF65`` 2026-05-14 OOM'd at 121s on T4 because
  ``reconstruct_pair`` ran a 600-pair forward pass at full resolution.
- C6 5ep smoke ``fc-01KRKG566Z2F48CVCGF8JFA0S1`` 2026-05-14 returned
  ``auth_eval rc=2`` because the trainer hand-wrote ``--archive-zip`` /
  ``--output-json`` flags that don't exist on the canonical contest
  auth-eval CLI.
- Z3 v2 + Z4 smoke pair 2026-05-15 (``fc-01KRNHEGC9ZE48Y68GGJHP7FXN`` +
  ``fc-01KRNHE942JSV7VRGXGR1FJGHQ``, $2 each wasted) crashed on the
  recipe-vs-trainer-state divergence Z3 v2 / Z4 / Z5 bug class.
- Z3 v2 FULL Modal A100 2026-05-15T11:41:15Z wrote CPU eval results to
  ``contest_auth_eval_cuda.json`` because the trainer hardcoded the
  ``_cuda`` filename suffix while the dispatcher injected
  ``AUTH_EVAL_DEVICE=cpu`` (Catalog #249 phantom-score bug class).
- T1 Balle ``fc-01KR955JSYQAVTTYZA48VAV7WJ`` 2026-05-10 timed out
  rc=124 at 84,608s wall-clock because the trainer lacked autocast fp16 +
  TF32 + torch.compile (the Tier 1 engineering hygiene primitives).

The umbrella protocol is the conjunction
``protocol_complete = AND(tier1_complete, tier2_complete, tier3_complete)``
across every CANONICAL existing strict gate that closes a dispatch-failure
bug class. Empty conjunction = REFUSE dispatch.

This module is the single source of truth a CALLER (operator-authorize,
local pre-deploy harness, STRICT preflight gate) consults. It NEVER
re-implements bug-class detection; it routes through existing canonical
gates per the standing directive *"all possible should be pulled into the
decorator or similar reusable and shareable tools and helpers"*.

Tier classification (per audit ``feedback_canonical_dispatch_optimization_protocol_landed_20260515.md``):

- **Tier 1 — Engineering primitives**: autocast_fp16 / TF32 / torch.compile /
  no_grad-at-eval / GTScorerCache (F3) / canonical-helper routing
  (#172/#178/#179/#180/#228/#226/#164/#205/#218).
- **Tier 2 — Hardware correctness**: NVML 999 + CUDA-alloc + CUBLAS-config +
  cu13-vs-cu124 + smoke-min-gpu (#244/#224/#170/#171/#181/#182/#215/#203).
- **Tier 3 — Substrate correctness**: recipe-vs-trainer-state divergence +
  scorer-loader assignment + auth-eval CLI + canonical inflate device +
  Modal source/head parity + sentinel-files-in-mount-set
  (#240/#222/#223/#226/#205/#166/#201/#249).

Public API:

    verify_dispatch_protocol_complete(
        trainer: str | Path,
        recipe: str | None = None,
        *,
        repo_root: str | Path | None = None,
    ) -> ProtocolVerdict

Returns a typed verdict with per-tier verdicts + overall pass/fail +
machine-readable blockers list (suitable for JSON serialization or
operator review).

Wire-in surfaces (per Catalog #270 STRICT gate):

1. ``tools/operator_authorize.py::_run_local_pre_deploy_check`` already
   wires the local 30s harness; this protocol is invoked by the harness's
   8th check (CHECKS list).
2. ``tools/local_pre_deploy_check.py`` exposes the protocol verdict to
   non-operator-authorize callers.
3. STRICT preflight gate ``check_dispatch_optimization_protocol_complete``
   refuses the repo state if ANY known dispatchable trainer is missing a
   tier acceptance signal.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


# Tier 1 — engineering primitives the trainer MUST declare/use.
# Detection uses presence of canonical token in the trainer source body OR
# a sister opt-out marker in the trainer's first ~25 lines.
_TIER1_TOKENS: dict[str, tuple[str, ...]] = {
    "autocast_fp16": ("--enable-autocast-fp16", "torch.autocast", "autocast(", "AUTOCAST_FP16_WAIVED"),
    "tf32": (
        "torch.backends.cuda.matmul.allow_tf32",
        "torch.backends.cudnn.allow_tf32",
        "trainer_skeleton.device_or_die",  # canonical helper enables TF32
        "TF32_WAIVED",
    ),
    "torch_compile": ("--enable-torch-compile", "torch.compile(", "TORCH_COMPILE_WAIVED"),
    "no_grad_at_eval": (
        "with torch.no_grad():",
        "@torch.no_grad",
        "torch.inference_mode",
        "NO_GRAD_WAIVED",
    ),
    "canonical_scorer_loss": (
        "tac.substrates._shared.score_aware_common",
        "score_pair_components",
        "scorer_loss_terms_btchw",
        "SCORER_PREPROCESS_HANDLED_OK",
    ),
}

# Tier 2 — hardware correctness signals the LANE DRIVER + recipe MUST declare.
# Detection routes through the lane driver script + recipe YAML.
_TIER2_RECIPE_FIELDS: tuple[str, ...] = (
    "min_vram_gb",
    "min_smoke_gpu",
    "video_input_strategy",
    "pyav_decode_strategy",
    "target_modes",
)

_TIER2_LANE_DRIVER_NVML_TOKENS: tuple[str, ...] = (
    "DALI_DISABLE_NVML",
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTORCH_CUDA_ALLOC_CONF",
)

# Tier 3 — substrate correctness signals; the trainer + recipe state must be
# consistent (recipe-vs-trainer-state divergence is the Z3 v2 / Z4 / Z5
# bug class) AND auth-eval / inflate device / scorer loader must use
# canonical helpers.
_TIER3_TRAINER_TOKENS: dict[str, tuple[str, ...]] = {
    "canonical_auth_eval_helper": (
        "gate_auth_eval_call",
        "smoke_auth_eval_gate",
        "AUTH_EVAL_DIRECT_SUBPROCESS_OK",
    ),
    "canonical_inflate_device": (
        "select_inflate_device",
        "INLINE_DEVICE_FORK_OK",
    ),
    "scorer_loader_order_correct": (
        "pose_scorer, seg_scorer = ",
        "posenet, segnet = ",
        "load_default_scorers",
        "load_differentiable_scorers",
        "SCORER_LOADER_ORDER_OK",
    ),
}

# Reversed-order patterns; presence is a hard violation
_TIER3_REVERSED_LOADER_PATTERNS: tuple[str, ...] = (
    r"seg_scorer\s*,\s*pose_scorer\s*=\s*load_differentiable_scorers",
    r"seg_scorer\s*,\s*pose_scorer\s*=\s*load_default_scorers",
    r"segnet\s*,\s*posenet\s*=\s*load_default_scorers",
    r"segnet\s*,\s*posenet\s*=\s*load_differentiable_scorers",
)

# Recipe-vs-trainer-state divergence (Z3 v2 / Z4 / Z5 bug class anchor)
_RECIPE_RESEARCH_ONLY_FLAGS: tuple[str, ...] = (
    "smoke_only",
    "research_only",
    "dispatch_enabled",
)


@dataclass
class TierVerdict:
    tier: str  # "tier1_engineering" | "tier2_hardware" | "tier3_substrate"
    pass_signals: dict[str, bool] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)

    @property
    def overall_pass(self) -> bool:
        # Empty pass_signals is an empty conjunction — treat as PASS
        # (caller can distinguish via len(pass_signals)).
        # But blockers list must be empty.
        return not self.blockers and all(self.pass_signals.values())


@dataclass
class ProtocolVerdict:
    trainer_path: str
    recipe_name: str | None
    repo_root: str
    tier1: TierVerdict
    tier2: TierVerdict
    tier3: TierVerdict
    overall_pass: bool
    blockers: list[str] = field(default_factory=list)
    advisory_notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        return out

    def as_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True)


def _read_text_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _resolve_recipe_path(recipe: str | None, repo_root: Path) -> Path | None:
    if recipe is None:
        return None
    candidates = [
        repo_root / ".omx" / "operator_authorize_recipes" / f"{recipe}.yaml",
        repo_root / ".omx" / "operator_authorize_recipes" / recipe,
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _resolve_lane_driver_path(trainer: Path, repo_root: Path) -> Path | None:
    """Map ``experiments/train_substrate_<id>.py`` → ``scripts/remote_lane_substrate_<id>.sh``.

    Returns the driver path if it exists; otherwise None (not all trainers
    have a dedicated lane driver — some go through ``modal_train_lane.py``
    and a generic recipe).
    """
    name = trainer.name
    m = re.match(r"^train_substrate_(.+)\.py$", name)
    if not m:
        return None
    substrate_id = m.group(1)
    candidate = repo_root / "scripts" / f"remote_lane_substrate_{substrate_id}.sh"
    return candidate if candidate.is_file() else None


def _verify_tier1(trainer_text: str, trainer_path: Path) -> TierVerdict:
    """Tier 1 — engineering primitives.

    Detection: presence of canonical token in trainer body OR sister
    opt-out marker in the file header (~first 25 lines).
    """
    verdict = TierVerdict(tier="tier1_engineering")
    header = "\n".join(trainer_text.splitlines()[:30])
    body = trainer_text
    for signal_name, tokens in _TIER1_TOKENS.items():
        present = False
        for tok in tokens:
            if tok in header or tok in body:
                present = True
                break
        verdict.pass_signals[signal_name] = present
        if not present:
            verdict.blockers.append(
                f"tier1_engineering: trainer {trainer_path.name} missing "
                f"`{signal_name}` (looked for any of: {list(tokens)}); "
                "wire the canonical primitive OR add a same-line waiver "
                "from the canonical waiver vocabulary."
            )
    return verdict


def _verify_tier2(
    trainer_text: str,
    trainer_path: Path,
    recipe_path: Path | None,
    lane_driver_path: Path | None,
    *,
    allow_no_recipe_advisory_mode: bool = False,
    no_recipe_rationale: str | None = None,
) -> TierVerdict:
    """Tier 2 — hardware correctness.

    Detection:
      - recipe YAML must declare min_vram_gb / min_smoke_gpu /
        video_input_strategy / pyav_decode_strategy / target_modes
      - lane driver script (if present) must export DALI_DISABLE_NVML +
        CUBLAS_WORKSPACE_CONFIG + PYTORCH_CUDA_ALLOC_CONF

    F2 fix (codex review bklem3v5j HIGH 2026-05-15) per Catalog #280: when
    ``recipe_path is None`` the prior implementation marked every Tier 2
    recipe-field check ``True`` (vacuous PASS) which made strict callers
    silently approve dispatch with NO recipe-side coverage. Now the branch
    BLOCKS dispatch unless the caller explicitly opted into advisory mode
    via paired ``allow_no_recipe_advisory_mode=True`` AND a non-empty
    ``no_recipe_rationale`` (Catalog #199 paired-env discipline).
    """
    verdict = TierVerdict(tier="tier2_hardware")

    # Recipe-side fields
    if recipe_path is not None:
        recipe_text = _read_text_safe(recipe_path)
        for field_name in _TIER2_RECIPE_FIELDS:
            present = bool(re.search(rf"^\s*{re.escape(field_name)}\s*:", recipe_text, re.M))
            verdict.pass_signals[f"recipe_declares_{field_name}"] = present
            if not present:
                verdict.blockers.append(
                    f"tier2_hardware: recipe {recipe_path.name} missing "
                    f"top-level `{field_name}` field; sister Catalog "
                    "#170/#171/#181/#182/#215 require this for hardware-"
                    "correctness routing. Add the field per the canonical "
                    "vocabulary."
                )
    else:
        # No recipe was resolved. Per F2 fix (Catalog #280) this is BLOCKED
        # unless the caller explicitly opts into advisory mode via paired
        # discipline (Catalog #199 sister): both flag AND rationale required.
        paired_waiver_active = bool(
            allow_no_recipe_advisory_mode
            and no_recipe_rationale
            and no_recipe_rationale.strip()
        )
        if paired_waiver_active:
            # Advisory-mode opt-in: mark signals as advisory-skipped (False
            # in pass_signals so overall_pass remains gated, but no per-
            # field blocker entry — the umbrella adds ONE explicit
            # advisory-mode blocker so --strict callers exit nonzero).
            for field_name in _TIER2_RECIPE_FIELDS:
                verdict.pass_signals[f"recipe_declares_{field_name}"] = False
            verdict.blockers.append(
                "tier2_hardware: recipe unresolved AND advisory-mode "
                f"explicitly opted in (--allow-no-recipe-advisory-mode "
                f"--no-recipe-rationale {no_recipe_rationale!r}); Tier 2 "
                "recipe-field checks SKIPPED. Per F2 (Catalog #280) "
                "+ codex review bklem3v5j recommendation: --strict "
                "remains nonzero whenever recipe-side checks are skipped "
                "so the operator can intentionally route a recipe-less "
                "advisory dispatch BUT cannot accidentally bypass the "
                "umbrella. This row is the explicit acknowledgment."
            )
        else:
            # Default: BLOCK every recipe-side signal so the umbrella
            # refuses dispatch.
            for field_name in _TIER2_RECIPE_FIELDS:
                verdict.pass_signals[f"recipe_declares_{field_name}"] = False
                verdict.blockers.append(
                    f"tier2_hardware: recipe unresolved — cannot verify "
                    f"`{field_name}` declaration (sister Catalog "
                    "#170/#171/#181/#182/#215). Per F2 fix (Catalog #280) "
                    "the prior vacuous-True default has been removed; "
                    "supply an explicit recipe via `--recipe <name>` OR "
                    "opt into advisory mode via paired "
                    "`--allow-no-recipe-advisory-mode "
                    "--no-recipe-rationale <text>` (Catalog #199 sister)."
                )

    # Lane driver side (only check when a driver script exists)
    if lane_driver_path is not None:
        driver_text = _read_text_safe(lane_driver_path)
        for tok in _TIER2_LANE_DRIVER_NVML_TOKENS:
            present = tok in driver_text
            verdict.pass_signals[f"driver_exports_{tok}"] = present
            if not present:
                verdict.blockers.append(
                    f"tier2_hardware: lane driver {lane_driver_path.name} "
                    f"missing `export {tok}=...` (the canonical 3-export "
                    "Modal/CUDA env block per Catalog #244 + commit "
                    "611495f26). Insert the block IMMEDIATELY after "
                    "`set -euo pipefail` per the META layer auto-emit "
                    "pattern in `tac.substrate_registry.driver_generator`."
                )
    else:
        # No lane driver — substrate uses a generic recipe path. Tier 2
        # is then verified at the recipe layer + Modal image layer
        # (Catalog #224). Mark vacuous.
        for tok in _TIER2_LANE_DRIVER_NVML_TOKENS:
            verdict.pass_signals[f"driver_exports_{tok}"] = True

    return verdict


def _verify_tier3(
    trainer_text: str,
    trainer_path: Path,
    recipe_path: Path | None,
    *,
    allow_no_recipe_advisory_mode: bool = False,
    no_recipe_rationale: str | None = None,
) -> TierVerdict:
    """Tier 3 — substrate correctness.

    Detection:
      - recipe-vs-trainer-state divergence (Z3 v2 / Z4 / Z5 bug class)
      - canonical auth-eval helper routing
      - canonical inflate device routing
      - scorer-loader assignment order (refuse reversed pattern)
    """
    verdict = TierVerdict(tier="tier3_substrate")

    # Trainer canonical-helper routing
    for signal_name, tokens in _TIER3_TRAINER_TOKENS.items():
        present = any(tok in trainer_text for tok in tokens)
        verdict.pass_signals[signal_name] = present
        if not present:
            verdict.blockers.append(
                f"tier3_substrate: trainer {trainer_path.name} missing "
                f"`{signal_name}` canonical token (any of: {list(tokens)}); "
                "wire through the canonical helper per "
                "Catalog #205/#222/#226/#249."
            )

    # Reversed scorer-loader pattern is a hard violation regardless
    for pat in _TIER3_REVERSED_LOADER_PATTERNS:
        m = re.search(pat, trainer_text)
        if m:
            verdict.pass_signals["scorer_loader_order_correct"] = False
            verdict.blockers.append(
                f"tier3_substrate: trainer {trainer_path.name} contains "
                f"REVERSED scorer-loader assignment matching pattern "
                f"`{pat}` — canonical API returns (posenet, segnet); per "
                "Catalog #222, reversed assignment crashes inside "
                "`scorer_loss_terms_btchw` with `TypeError: new(): invalid "
                "data type 'str'`. Swap the LHS tuple."
            )

    # Recipe-vs-trainer-state divergence (Z3 v2 / Z4 / Z5)
    if recipe_path is not None:
        recipe_text = _read_text_safe(recipe_path)
        full_main_def = re.search(r"^def\s+_full_main\s*\([^)]*\)[^:]*:", trainer_text, re.M)
        if full_main_def is not None:
            body_start = full_main_def.end()
            body = trainer_text[body_start : body_start + 3000]
            raises_not_impl = bool(re.search(r"\braise\s+NotImplementedError\b", body))
            research_only = bool(
                re.search(r"^\s*smoke_only:\s*true\b", recipe_text, re.M)
                or re.search(r"^\s*research_only:\s*true\b", recipe_text, re.M)
                or re.search(r"^\s*dispatch_enabled:\s*false\b", recipe_text, re.M)
            )
            consistent = (not raises_not_impl) or research_only
            verdict.pass_signals["recipe_vs_trainer_state_consistent"] = consistent
            if not consistent:
                verdict.blockers.append(
                    f"tier3_substrate: recipe-vs-trainer-state divergence "
                    f"— trainer {trainer_path.name} `_full_main` raises "
                    f"NotImplementedError but recipe {recipe_path.name} "
                    "lacks `smoke_only: true` / `research_only: true` / "
                    "`dispatch_enabled: false`. Modal dispatch will burn "
                    "$2-15 reaching trainer + crashing pre-auth-eval "
                    "(Z3 v2 / Z4 / Z5 bug class; Catalog #240)."
                )
        else:
            verdict.pass_signals["recipe_vs_trainer_state_consistent"] = True
    else:
        # F2 fix (codex review bklem3v5j HIGH 2026-05-15) per Catalog #280:
        # an unresolved recipe means we CANNOT verify recipe-vs-trainer-state
        # consistency. Default fail-closed; advisory-mode opt-in requires
        # paired discipline.
        paired_waiver_active = bool(
            allow_no_recipe_advisory_mode
            and no_recipe_rationale
            and no_recipe_rationale.strip()
        )
        verdict.pass_signals["recipe_vs_trainer_state_consistent"] = False
        if paired_waiver_active:
            verdict.blockers.append(
                "tier3_substrate: recipe unresolved AND advisory-mode "
                f"explicitly opted in (--allow-no-recipe-advisory-mode "
                f"--no-recipe-rationale {no_recipe_rationale!r}); recipe-"
                "vs-trainer-state divergence check SKIPPED. Per F2 "
                "(Catalog #280) + codex review bklem3v5j recommendation: "
                "--strict remains nonzero whenever recipe-side checks "
                "are skipped."
            )
        else:
            verdict.blockers.append(
                "tier3_substrate: recipe unresolved — cannot verify "
                "recipe-vs-trainer-state consistency (Z3 v2 / Z4 / Z5 "
                "bug class anchor; Catalog #240). Per F2 fix (Catalog "
                "#280) the prior vacuous-True default has been removed; "
                "supply an explicit recipe via `--recipe <name>` OR opt "
                "into advisory mode via paired "
                "`--allow-no-recipe-advisory-mode "
                "--no-recipe-rationale <text>` (Catalog #199 sister)."
            )

    # Phantom-score directory class (Catalog #249) — refuse hardcoded
    # `_cuda.json` filename in non-canonical-helper substrate trainers.
    if "contest_auth_eval_cuda.json" in trainer_text and "gate_auth_eval_call" not in trainer_text:
        verdict.pass_signals["no_phantom_device_named_output"] = False
        verdict.blockers.append(
            f"tier3_substrate: trainer {trainer_path.name} hardcodes "
            "`contest_auth_eval_cuda.json` filename without routing "
            "through `gate_auth_eval_call` canonical helper — phantom-"
            "score device-named directory class (Catalog #249). The "
            "filename device-token must be derived from the actual "
            "auth-eval device, OR route through the canonical helper "
            "which auto-redirects mismatched filenames."
        )
    else:
        verdict.pass_signals["no_phantom_device_named_output"] = True

    return verdict


def verify_dispatch_protocol_complete(
    trainer: str | Path,
    recipe: str | None = None,
    *,
    repo_root: str | Path | None = None,
    allow_no_recipe_advisory_mode: bool = False,
    no_recipe_rationale: str | None = None,
) -> ProtocolVerdict:
    """Compute the canonical dispatch optimization protocol verdict.

    The umbrella formula is
    ``protocol_complete = AND(tier1_complete, tier2_complete, tier3_complete)``.
    Empty intersection (any tier with non-empty blockers) = REFUSE
    dispatch.

    Args:
        trainer: Absolute or repo-relative path to the substrate trainer
            (``experiments/train_substrate_*.py``).
        recipe: Optional recipe NAME (without ``.yaml``); when supplied,
            Tier 2 recipe-field checks + Tier 3 recipe-vs-trainer-state
            divergence checks fire. When omitted, the Tier 2 + Tier 3
            recipe-side signals are BLOCKED (overall_pass=False) UNLESS
            the caller explicitly opts into advisory mode via paired
            ``allow_no_recipe_advisory_mode=True`` AND a non-empty
            ``no_recipe_rationale`` (Catalog #199 paired-env discipline +
            Catalog #280 F2 fix per codex review bklem3v5j HIGH
            2026-05-15). Even in advisory mode the umbrella verdict
            preserves an explicit advisory-mode blocker so ``--strict``
            callers exit nonzero.
        repo_root: Override repo root (defaults to module-resolved
            ``REPO_ROOT``).
        allow_no_recipe_advisory_mode: Paired-env opt-in to permit
            recipe-less advisory invocation. F2 fix per Catalog #280;
            requires ``no_recipe_rationale`` to be a non-empty string.
        no_recipe_rationale: Operator-supplied rationale string for the
            advisory opt-in. Empty / ``None`` rejects the opt-in.

    Returns:
        :class:`ProtocolVerdict` with per-tier verdicts + overall pass/fail
        + machine-readable blockers list.
    """
    root = Path(repo_root).resolve() if repo_root is not None else REPO_ROOT
    trainer_path = Path(trainer)
    if not trainer_path.is_absolute():
        trainer_path = (root / trainer_path).resolve()

    trainer_text = _read_text_safe(trainer_path)
    if not trainer_text:
        # Trainer not found / unreadable — return a single-blocker verdict.
        v = ProtocolVerdict(
            trainer_path=str(trainer_path),
            recipe_name=recipe,
            repo_root=str(root),
            tier1=TierVerdict(tier="tier1_engineering"),
            tier2=TierVerdict(tier="tier2_hardware"),
            tier3=TierVerdict(tier="tier3_substrate"),
            overall_pass=False,
            blockers=[
                f"protocol: trainer {trainer_path} not found or unreadable; "
                "cannot verify dispatch optimization protocol."
            ],
        )
        return v

    recipe_path = _resolve_recipe_path(recipe, root)
    lane_driver_path = _resolve_lane_driver_path(trainer_path, root)

    tier1 = _verify_tier1(trainer_text, trainer_path)
    tier2 = _verify_tier2(
        trainer_text,
        trainer_path,
        recipe_path,
        lane_driver_path,
        allow_no_recipe_advisory_mode=allow_no_recipe_advisory_mode,
        no_recipe_rationale=no_recipe_rationale,
    )
    tier3 = _verify_tier3(
        trainer_text,
        trainer_path,
        recipe_path,
        allow_no_recipe_advisory_mode=allow_no_recipe_advisory_mode,
        no_recipe_rationale=no_recipe_rationale,
    )

    blockers = list(tier1.blockers) + list(tier2.blockers) + list(tier3.blockers)
    overall = tier1.overall_pass and tier2.overall_pass and tier3.overall_pass

    advisory: list[str] = []
    if recipe_path is None:
        # F2 fix (Catalog #280) — distinguish DEFAULT-FAIL-CLOSED vs
        # OPERATOR-OPTED-IN-ADVISORY-MODE so the advisory log is honest.
        paired_waiver_active = bool(
            allow_no_recipe_advisory_mode
            and no_recipe_rationale
            and no_recipe_rationale.strip()
        )
        if paired_waiver_active:
            advisory.append(
                f"advisory: recipe unresolved AND advisory-mode opted "
                f"in (rationale={no_recipe_rationale!r}). Per Catalog "
                "#280 + codex bklem3v5j F2 fix the umbrella retains "
                "explicit blockers so --strict callers still exit "
                "nonzero — advisory-mode is NOT a pass; it is an "
                "operator-acknowledged skip."
            )
        else:
            advisory.append(
                "advisory: no recipe supplied — Tier 2 recipe-field "
                "checks + Tier 3 recipe-vs-trainer-state checks now "
                "FAIL CLOSED per F2 fix (Catalog #280). Pass a recipe "
                "NAME via --recipe to enable full coverage, OR opt "
                "into advisory mode via paired "
                "`--allow-no-recipe-advisory-mode "
                "--no-recipe-rationale <text>`."
            )
    if lane_driver_path is None:
        advisory.append(
            "advisory: no lane driver script found — Tier 2 NVML/CUDA "
            "env block check skipped at the driver layer (the substrate "
            "may be using a generic Modal recipe path; verify Catalog "
            "#224 Modal image env block separately)."
        )

    return ProtocolVerdict(
        trainer_path=str(trainer_path),
        recipe_name=recipe,
        repo_root=str(root),
        tier1=tier1,
        tier2=tier2,
        tier3=tier3,
        overall_pass=overall,
        blockers=blockers,
        advisory_notes=advisory,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Canonical dispatch optimization protocol — Tier 1/2/3 umbrella verdict."
    )
    parser.add_argument(
        "--trainer",
        required=True,
        type=Path,
        help="Path to experiments/train_substrate_*.py",
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default=None,
        help="Optional recipe NAME (without .yaml) — enables Tier 2/3 recipe-side checks",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 on protocol fail (default: warn-only)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON verdict to stdout instead of human report",
    )
    # F2 fix (codex review bklem3v5j HIGH 2026-05-15) per Catalog #280:
    # paired-env discipline for recipe-less advisory invocations. The
    # paired flag REQUIRES a non-empty rationale; bare `--allow-no-recipe-
    # advisory-mode` without `--no-recipe-rationale <text>` raises
    # SystemExit. Even with both flags supplied, --strict still exits
    # nonzero per codex's explicit recommendation (advisory != pass).
    parser.add_argument(
        "--allow-no-recipe-advisory-mode",
        action="store_true",
        help=(
            "Opt into recipe-less advisory invocation. Per F2 fix "
            "(Catalog #280) recipe-less is otherwise BLOCKED. Requires "
            "paired --no-recipe-rationale <text>. --strict still exits "
            "nonzero in advisory mode (operator-acknowledged skip)."
        ),
    )
    parser.add_argument(
        "--no-recipe-rationale",
        type=str,
        default=None,
        help=(
            "Operator-supplied rationale for recipe-less advisory mode "
            "(Catalog #199 paired-env discipline)."
        ),
    )
    args = parser.parse_args()

    # F2 fix paired-env validator: bare opt-in flag without rationale is
    # SystemExit per Catalog #199 sister discipline + Catalog #280.
    if args.allow_no_recipe_advisory_mode and not (
        args.no_recipe_rationale and args.no_recipe_rationale.strip()
    ):
        print(
            "FATAL: --allow-no-recipe-advisory-mode requires paired "
            "--no-recipe-rationale <text>. Per Catalog #280 F2 fix + "
            "Catalog #199 sister paired-env discipline. Refused.",
            file=sys.stderr,
        )
        return 11

    verdict = verify_dispatch_protocol_complete(
        args.trainer,
        args.recipe,
        allow_no_recipe_advisory_mode=args.allow_no_recipe_advisory_mode,
        no_recipe_rationale=args.no_recipe_rationale,
    )

    if args.json:
        print(verdict.as_json())
    else:
        print(f"[dispatch-optimization-protocol] trainer: {verdict.trainer_path}")
        print(f"[dispatch-optimization-protocol] recipe:  {verdict.recipe_name}")
        print(f"[dispatch-optimization-protocol] mode:    {'STRICT' if args.strict else 'WARN-ONLY'}")
        for tier in (verdict.tier1, verdict.tier2, verdict.tier3):
            mark = "✓" if tier.overall_pass else "✗"
            print(f"  {mark} [{tier.tier}] signals_pass={sum(tier.pass_signals.values())}/{len(tier.pass_signals)} blockers={len(tier.blockers)}")
            for blocker in tier.blockers:
                print(f"    BLOCKER: {blocker}")
        for note in verdict.advisory_notes:
            print(f"  ADVISORY: {note}")
        if verdict.overall_pass:
            print("[dispatch-optimization-protocol] OVERALL: PASS — Tier 1/2/3 all complete; safe to dispatch.")
        else:
            print(f"[dispatch-optimization-protocol] OVERALL: FAIL — {len(verdict.blockers)} blocker(s) prevent dispatch.")

    if not verdict.overall_pass and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
