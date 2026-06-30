# SPDX-License-Identifier: MIT
"""v2_compose.launch_command — emit the FLAG-VALIDATED residual-INR launch command (Step 4 actuator).

PHASE-A's final act: derive the residual-INR training command, VALIDATE every flag against the REAL
trainer argparse (CLAUDE.md "NEVER invent CLI flags" — grep argparse, never invent), prefix the perf
env, and HOLD for operator GO. This module EMITS the command string; it NEVER launches (no GPU here).

Flag-validation discipline (the dogfood of ``witness_autoconfig``): the trainer
``experiments/train_levelset_witness_realized_through_R_mlx.py`` builds its parser INLINE in ``main()``
(no ``build_parser()`` factory), so importing+calling it is unsafe (it trains). We STATICALLY parse the
``add_argument`` lines (the reliable flag-validation source) and assert every emitted flag exists.

HONEST GAP (the spec's S3 seam, confirmed): the existing trainer has NO residual-target-subtraction
flag (no ``--residual-target`` / ``--bulk-subtract`` / residual-only INR sizing). Its closest mechanism
is ``--structured-init`` (+ ``--lane-prior-phi1``), which bakes the bulk SDFs INTO the INR weights as a
TRAIN-TIME init — it accelerates convergence but does NOT shrink the rate (the bulk still ships inside
the trained weights). The TRUE residual-only mode (consume ``residual_target.npz``, subtract the bulk,
size the INR for the residual) is a GPU-SIDE trainer change = NEEDS-WIRING (NOT this tool's job; NOT
emittable via existing flags). The emitted command is the best available existing-flag run (structured-
init + lane-prior + curriculum + per-stage ckpts); the report flags the missing capability explicitly.

Perf env (memory ``[[launch-gate-must-verify-perf-env]]``): prefix ``TAC_MLX_CUSTOM_GROUPED_BACKWARD=1``
(the bit-identical ~17x fast through-R backward; default ON in the adapter, set explicitly per launch-gate
discipline + verify the ``custom_grouped_backward active=true`` log line at launch).
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "LaunchCommand",
    "parse_trainer_flags",
    "build_residual_inr_command",
    "build_residual_only_command",
    "DEFAULT_TRAINER",
    "PERF_ENV",
]

_REPO = Path(__file__).resolve().parents[3]
DEFAULT_TRAINER = "experiments/train_levelset_witness_realized_through_R_mlx.py"
PERF_ENV = {"TAC_MLX_CUSTOM_GROUPED_BACKWARD": "1"}

_ADD_ARG_RE = re.compile(r"""add_argument\(\s*["'](--[A-Za-z0-9][A-Za-z0-9\-]*)["']""")
_BOOL_OPT_RE = re.compile(
    r"""add_argument\(\s*["'](--[A-Za-z0-9][A-Za-z0-9\-]*)["'][^)]*BooleanOptionalAction""",
    re.DOTALL,
)


@dataclass(frozen=True)
class LaunchCommand:
    """A flag-validated launch command held for operator GO (never auto-fired)."""

    command: str                         # the full shell command (perf-env prefix + interpreter + flags)
    flags_validated: dict[str, bool]     # emitted flag -> exists-in-real-argparse
    all_flags_valid: bool
    unknown_flags: tuple[str, ...]
    perf_env: dict[str, str]
    trainer_path: str
    hold_note: str
    missing_capability_note: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_json(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "all_flags_valid": self.all_flags_valid,
            "flags_validated": dict(self.flags_validated),
            "unknown_flags": list(self.unknown_flags),
            "perf_env": dict(self.perf_env),
            "trainer_path": self.trainer_path,
            "hold_for_operator_go": True,
            "auto_launched": False,
            "hold_note": self.hold_note,
            "missing_capability_note": self.missing_capability_note,
            "notes": list(self.notes),
        }


def parse_trainer_flags(trainer_path: str | Path) -> tuple[set[str], set[str]]:
    """STATICALLY parse the trainer's ``add_argument`` lines (the reliable flag contract).

    Returns (all_flag_names, bool_optional_flag_names). ``BooleanOptionalAction`` flags also accept
    a ``--no-<name>`` negation (added to all_flag_names).
    """
    path = Path(trainer_path)
    if not path.is_absolute():
        path = _REPO / path
    src = path.read_text()
    all_flags = set(_ADD_ARG_RE.findall(src))
    bool_opt = set(_BOOL_OPT_RE.findall(src))
    for f in bool_opt:
        all_flags.add("--no-" + f[2:])  # the auto-generated negation
    if not all_flags:
        raise ValueError(f"parse_trainer_flags: no add_argument flags found in {path}")
    return all_flags, bool_opt


def _flag_name(token: str) -> str | None:
    """Return the flag name from a command token (``--foo`` or ``--foo=bar``), else None."""
    if not token.startswith("--"):
        return None
    return token.split("=", 1)[0]


def build_residual_inr_command(
    *,
    out_dir: str,
    gt_cache: str,
    num_pairs: int = 600,
    epochs: int = 1500,
    seed: int = 0,
    hidden_dim: int = 48,
    mod_dim: int = 16,
    ema_decay: float = 0.997,
    resume_from: str | None = None,
    use_structured_init: bool = True,
    use_lane_prior: bool = True,
    use_curriculum: bool = True,
    muon_start_epoch: int | None = None,
    extra_flags: dict[str, Any] | None = None,
    trainer_path: str | Path = DEFAULT_TRAINER,
    interpreter: str = ".venv/bin/python",
    strict: bool = True,
) -> LaunchCommand:
    """Build + flag-validate the residual-INR training command (HOLD for operator GO).

    The residual INR is SIZED SMALL (``hidden_dim``/``mod_dim`` below the full-partition INR's 96/32)
    because it carries only the LEARN residual. It uses the existing ``--structured-init`` (+
    ``--lane-prior-phi1``) init-into-weights mechanism + the CE->tau->l7 curriculum + per-stage
    checkpoints (resumability non-negotiable). EVERY flag is validated against the real argparse.

    Args mirror the residual-run knobs; ``extra_flags`` injects any additional REAL flag (validated).

    Raises:
        ValueError (when ``strict``) if any emitted flag is not in the real trainer argparse.
    """
    all_flags, _bool_opt = parse_trainer_flags(trainer_path)

    # Build the flag list using ONLY real flag names (verified below).
    pairs: list[tuple[str, Any]] = [
        ("--out-dir", out_dir),
        ("--gt-cache", gt_cache),
        ("--num-pairs", num_pairs),
        ("--epochs", epochs),
        ("--seed", seed),
        ("--mlx-device", "gpu"),
        ("--hidden-dim", hidden_dim),
        ("--mod-dim", mod_dim),
        ("--ema-decay", ema_decay),
    ]
    bare: list[str] = ["--stage-checkpoints"]  # per-stage ckpt (resumability non-negotiable; default on)
    if use_curriculum:
        bare.append("--curriculum")
    if use_structured_init:
        bare.append("--structured-init")
    if use_lane_prior:
        bare.append("--lane-prior-phi1")  # requires --structured-init (trainer fail-closes otherwise)
    if resume_from:
        pairs.append(("--resume-from", resume_from))
    if muon_start_epoch is not None:
        pairs.append(("--muon-start-epoch", muon_start_epoch))
    if extra_flags:
        for k, v in extra_flags.items():
            flag = k if k.startswith("--") else "--" + k.replace("_", "-")
            if isinstance(v, bool):
                if v:
                    bare.append(flag)
            else:
                pairs.append((flag, v))

    # --- validate every emitted flag against the real argparse ---
    emitted = [f for f, _ in pairs] + bare
    flags_validated = {f: (f in all_flags) for f in emitted}
    unknown = tuple(f for f, ok in flags_validated.items() if not ok)
    all_valid = not unknown
    if strict and not all_valid:
        raise ValueError(
            "build_residual_inr_command: emitted flags NOT in the real trainer argparse "
            f"(NEVER invent flags): {list(unknown)}. Real flags: {sorted(all_flags)[:12]}..."
        )

    # --- assemble the command (perf-env prefix + interpreter + trainer + flags) ---
    env_prefix = " ".join(f"{k}={v}" for k, v in PERF_ENV.items())
    parts: list[str] = [str(interpreter), str(trainer_path)]
    for f, v in pairs:
        parts += [f, str(v)]
    parts += bare
    command = env_prefix + " " + " ".join(shlex.quote(p) if " " in p else p for p in parts)

    hold_note = (
        "HOLD for operator GO. This tool EMITS the command; it does NOT launch. One GPU, "
        "resumable (--resume-from) + per-stage checkpoints (--stage-checkpoints), EMA-shadow, "
        "single seed. Verify the 'custom_grouped_backward active=true' log line at launch."
    )
    missing_capability_note = (
        "S3 GAP (NEEDS-WIRING, GPU-side): the trainer has NO residual-target-subtraction flag. "
        "This command uses --structured-init (+ --lane-prior-phi1) = init-into-weights, which "
        "accelerates convergence but does NOT by itself shrink the rate (the bulk still ships in "
        "the trained weights). The TRUE residual-only mode (consume residual_target.npz, subtract "
        "the deterministic bulk, size the INR for the residual) is the rate-bearing trainer change "
        "to wire next. The residual_target.npz this pipeline emits is its input."
    )
    notes = (
        f"residual INR sized small (hidden_dim={hidden_dim}, mod_dim={mod_dim}) vs the full-partition "
        "96/32 — the rate-win lever (only valid once the bulk is subtracted, GPU-side).",
        "means != ends: emitting this command moves NOTHING. The pointer (0.19110) moves only when "
        "this run lands + the 4-section archive byte-closes + exact-evals below 0.19110.",
    )
    return LaunchCommand(
        command=command,
        flags_validated=flags_validated,
        all_flags_valid=all_valid,
        unknown_flags=unknown,
        perf_env=dict(PERF_ENV),
        trainer_path=str(trainer_path),
        hold_note=hold_note,
        missing_capability_note=missing_capability_note,
        notes=notes,
    )


def build_residual_only_command(
    *,
    out_dir: str,
    gt_cache: str,
    residual_target_npz: str,
    num_pairs: int = 600,
    epochs: int = 1500,
    seed: int = 0,
    hidden_dim: int = 48,
    mod_dim: int = 16,
    ema_decay: float = 0.997,
    resume_from: str | None = None,
    use_curriculum: bool = True,
    muon_start_epoch: int | None = None,
    extra_flags: dict[str, Any] | None = None,
    trainer_path: str | Path = DEFAULT_TRAINER,
    interpreter: str = ".venv/bin/python",
    strict: bool = True,
) -> LaunchCommand:
    """Build + flag-validate the CORRECTED residual-ONLY INR command (the rate-bearing run).

    Supersedes :func:`build_residual_inr_command` (which used ``--structured-init`` =
    init-into-weights = NO rate shrink). This emits the REAL residual mechanism: ``--residual-mode``
    + ``--residual-target-npz`` so the FIXED deterministic bulk is GENERATED at decode (OUTSIDE the
    counted weights) and COMPOSED before R, and the small INR (hidden 48 / mod 16) trains ONLY on
    the Lane+Movable residual annulus (the byte win). EXPLICITLY does NOT pass ``--structured-init``
    / ``--lane-prior-phi1`` (the trainer fail-closes on residual-mode + structured-init).

    This is the CLEAN BASE (best-known priors, optimal-form-first): curriculum + per-stage ckpts +
    small INR. The theta* lever sweep runs ON this hybrid residual: each arm == this command + ONE
    surgical lever varied (``--lane-thin-weight`` / ``--margin-saliency-weight`` / ``--hardness-*``),
    which weight the COMPOSED-render d_seg (the residual IS the Lane+Movable annulus -> maximally
    relevant). Pass them via ``extra_flags`` (each validated against the real argparse).

    Raises:
        ValueError (when ``strict``) if any emitted flag is not in the real trainer argparse.
    """
    all_flags, _bool_opt = parse_trainer_flags(trainer_path)
    pairs: list[tuple[str, Any]] = [
        ("--out-dir", out_dir),
        ("--gt-cache", gt_cache),
        ("--num-pairs", num_pairs),
        ("--epochs", epochs),
        ("--seed", seed),
        ("--mlx-device", "gpu"),
        ("--hidden-dim", hidden_dim),
        ("--mod-dim", mod_dim),
        ("--ema-decay", ema_decay),
        ("--residual-target-npz", residual_target_npz),
    ]
    bare: list[str] = ["--stage-checkpoints", "--residual-mode"]
    if use_curriculum:
        bare.append("--curriculum")
    if resume_from:
        pairs.append(("--resume-from", resume_from))
    if muon_start_epoch is not None:
        pairs.append(("--muon-start-epoch", muon_start_epoch))
    if extra_flags:
        for k, v in extra_flags.items():
            flag = k if k.startswith("--") else "--" + k.replace("_", "-")
            if isinstance(v, bool):
                if v:
                    bare.append(flag)
            else:
                pairs.append((flag, v))

    emitted = [f for f, _ in pairs] + bare
    flags_validated = {f: (f in all_flags) for f in emitted}
    unknown = tuple(f for f, ok in flags_validated.items() if not ok)
    all_valid = not unknown
    if strict and not all_valid:
        raise ValueError(
            "build_residual_only_command: emitted flags NOT in the real trainer argparse "
            f"(NEVER invent flags): {list(unknown)}. Real flags: {sorted(all_flags)[:12]}...")

    env_prefix = " ".join(f"{k}={v}" for k, v in PERF_ENV.items())
    parts: list[str] = [str(interpreter), str(trainer_path)]
    for f, v in pairs:
        parts += [f, str(v)]
    parts += bare
    command = env_prefix + " " + " ".join(shlex.quote(p) if " " in p else p for p in parts)

    hold_note = (
        "HOLD for operator GO + GPU reallocation. EMITS the command; does NOT launch. One GPU, "
        "resumable (--resume-from) + per-stage checkpoints (--stage-checkpoints), EMA-shadow, "
        "single seed. Verify the 'custom_grouped_backward active=true' log line at launch."
    )
    missing_capability_note = (
        "CORRECTED: this is the REAL residual mechanism (--residual-mode + --residual-target-npz). "
        "The bulk is GENERATED at decode (OUTSIDE the counted weights) and COMPOSED before R, so the "
        "small INR carries ONLY the residual -> the rate shrinks. (build_residual_inr_command's "
        "--structured-init bakes the bulk INTO the weights = NO shrink; that path is SUPERSEDED.)"
    )
    notes = (
        f"residual INR sized small (hidden_dim={hidden_dim}, mod_dim={mod_dim}) vs the full-partition "
        "96/32 -- valid now BECAUSE the bulk is subtracted (composed deterministically). The theta* "
        "lever sweep runs ON this hybrid residual: each arm = this command + one surgical lever varied.",
        "means != ends: emitting this command moves NOTHING. The pointer (0.19110) moves only when "
        "this run lands + the 4-section archive byte-closes + exact-evals below 0.19110.",
    )
    return LaunchCommand(
        command=command,
        flags_validated=flags_validated,
        all_flags_valid=all_valid,
        unknown_flags=unknown,
        perf_env=dict(PERF_ENV),
        trainer_path=str(trainer_path),
        hold_note=hold_note,
        missing_capability_note=missing_capability_note,
        notes=notes,
    )
