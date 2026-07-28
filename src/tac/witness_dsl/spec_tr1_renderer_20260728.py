"""ddm_tb1 — SPEC_tr1 renderer DSL: typed Lever factories + program compile (SoT).

Config for the tr1 trained partition→pixel renderer is DSL-COMPILED, never ad-hoc
argv (CLAUDE.md "The DSL HOLDS every designed lever"). Every SPEC S4.1 lever lands
here as a ``Lever`` factory (the canonical ``tac.witness_dsl.curriculum_dsl.Lever``
dataclass: name + flag ``overrides``); ``TR1RendererProgramV1.compile_trainer_argv``
merges lever overrides (later levers win = theta* composition) into the trainer's
argv, and ``validate()`` FAIL-CLOSES on any flag the trainer's argparse does not
declare (never-invent-flags), by AST-scanning the trainer source's ``add_argument``
calls — no import of MLX needed at validation time.

Trainer: ``experiments/train_tr1_partition_renderer_mlx.py`` (this tree).
Evidence axis: config-generation only; score_claim=False.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.witness_dsl.curriculum_dsl import Lever

TRAINER_RELPATH = "experiments/train_tr1_partition_renderer_mlx.py"


# ---------------------------------------------------------------------------
# Lever factories (SPEC S4.1, amended by the 2026-07-28 recall directive forces).
# ---------------------------------------------------------------------------
def lever_variant(variant: str) -> Lever:
    if variant not in ("plain", "lotto"):
        raise ValueError(f"variant must be plain|lotto, got {variant!r}")
    return Lever(name=f"tr1_variant_{variant}", overrides={"--variant": variant},
                 notes="A2 race arm: plain-conv vs G1-LOTTO supermask under matched counted bytes")


def lever_token_grid(downsample: int = 16, code_width: int = 4) -> Lever:
    if downsample not in (8, 16):
        raise ValueError("grid downsample raced over {8,16} (D=12 excluded: 512/12 "
                         "non-integer lattice — tb1 memo deviation from SPEC S1.2)")
    if code_width not in (2, 4, 6):
        raise ValueError("code width raced over {2,4,6} per SPEC S1.2")
    return Lever(name=f"tr1_token_grid_D{downsample}_c{code_width}",
                 overrides={"--grid-downsample": str(downsample),
                            "--code-width": str(code_width)},
                 notes="grid pitch vs bytes; ERF-bounded (r50~85px); Pareto-raced")


def lever_renderer_capacity(width: int = 24) -> Lever:
    return Lever(name=f"tr1_renderer_w{width}", overrides={"--renderer-width": str(width)},
                 notes="G3 capacity; topology derives from D (conv0 + per-up conv + head)")


def lever_desc_level_roundtrip(quant_levels: int = 16, ste: str = "round") -> Lever:
    if ste not in ("round", "dither"):
        raise ValueError("token STE raced over round|dither (asymmetry force)")
    return Lever(name=f"tr1_token_quant_L{quant_levels}_{ste}",
                 overrides={"--token-quant-levels": str(quant_levels), "--token-ste": ste},
                 notes="S2.1 fd2-wall STE across the description lattice; STE variant RACED")


def lever_token_temporal(mode: str = "shared_base") -> Lever:
    if mode not in ("shared_base", "independent"):
        raise ValueError("token temporal mode is shared_base|independent")
    return Lever(name=f"tr1_token_temporal_{mode}", overrides={"--token-temporal-mode": mode},
                 notes="Einstein d_cov/d_gauge force: identity-xi shared base vs A/B control")


def lever_lotto(seed: int = 118, mask_density_init: float = 0.5) -> Lever:
    return Lever(name=f"tr1_lotto_seed{seed}",
                 overrides={"--lotto-seed": str(seed),
                            "--lotto-mask-density-init": str(mask_density_init)},
                 notes="rule-118: PRNG expansion FREE; seed+density COUNTED in selector ledger")


def lever_seg_physics(form_start: str = "ce", w_seg: float = 100.0,
                      class_weight_lane: float = 1.0, margin_target: float = 1.0) -> Lever:
    return Lever(name=f"tr1_seg_{form_start}",
                 overrides={"--seg-form-start": form_start, "--w-seg": str(w_seg),
                            "--class-weight-lane": str(class_weight_lane),
                            "--margin-target": str(margin_target)},
                 notes="scorer-in-loop seg trunk; pose TERMINAL (#383) — no pose flag exists "
                       "on this trainer by design; margin_hinge = step-native raced form; "
                       "class_weight_lane = sn1 sided-asymmetry lever")


def lever_a1_gate(gate_every: int = 5) -> Lever:
    return Lever(name=f"tr1_a1_gate_every{gate_every}", overrides={"--gate-every": str(gate_every)},
                 notes="A1 (fd2 binding transfer lesson): realized-argmax gate cadence")


def lever_window(epochs: int, max_wall_minutes: float, batch_pairs: int = 8,
                 lr: float = 2e-3) -> Lever:
    return Lever(name=f"tr1_window_ep{epochs}",
                 overrides={"--epochs": str(epochs),
                            "--max-wall-minutes": str(max_wall_minutes),
                            "--batch-pairs": str(batch_pairs), "--lr": str(lr)},
                 notes="bounded governed window; checkpoint-on-exit (P0 resumability)")


# ---------------------------------------------------------------------------
# Program
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TR1RendererProgramV1:
    levers: tuple[Lever, ...]
    num_pairs: int
    out_dir: str
    seed: int = 0
    gt_cache: str | None = None
    resume_from: str | None = None
    full_confirm: bool = False

    def merged_overrides(self) -> dict[str, str]:
        merged: dict[str, str] = {}
        for lever in self.levers:
            merged.update({str(k): str(v) for k, v in lever.overrides.items()})
        return merged

    def compile_trainer_argv(self) -> list[str]:
        argv: list[str] = [TRAINER_RELPATH,
                           "--num-pairs", str(self.num_pairs),
                           "--out-dir", self.out_dir,
                           "--seed", str(self.seed)]
        for k, v in sorted(self.merged_overrides().items()):
            argv.extend([k, v])
        if self.gt_cache:
            argv.extend(["--gt-cache", self.gt_cache])
        if self.resume_from:
            argv.extend(["--resume-from", self.resume_from])
        if self.full_confirm:
            argv.append("--full-confirm")
        self.validate()
        return argv

    def validate(self, trainer_path: Path | None = None) -> None:
        """FAIL-CLOSED never-invent-flags: every emitted flag must exist in the
        trainer's argparse (AST scan of ``add_argument`` string literals)."""
        declared = trainer_declared_flags(trainer_path)
        emitted = set(self.merged_overrides())
        emitted |= {"--num-pairs", "--out-dir", "--seed"}
        if self.gt_cache:
            emitted.add("--gt-cache")
        if self.resume_from:
            emitted.add("--resume-from")
        if self.full_confirm:
            emitted.add("--full-confirm")
        invented = sorted(emitted - declared)
        if invented:
            raise ValueError(
                f"TR1 DSL validate FAIL-CLOSED (never-invent-flags): {invented} not "
                f"declared by {TRAINER_RELPATH} argparse; declared={sorted(declared)}")

    def sealed_ticket(self) -> dict[str, Any]:
        """The sealed DSL ticket for a governed T2 window (committed before launch)."""
        argv = self.compile_trainer_argv()
        payload = {
            "schema": "ddm_tb1_tr1_sealed_ticket.v1",
            "trainer": TRAINER_RELPATH,
            "argv": argv,
            "levers": [{"name": lv.name, "overrides": dict(lv.overrides), "notes": lv.notes}
                       for lv in self.levers],
            "score_claim": False,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        payload["ticket_hash"] = hashlib.sha256(blob).hexdigest()
        return payload


def trainer_declared_flags(trainer_path: Path | None = None) -> set[str]:
    if trainer_path is None:
        # this module may be imported from a linked worktree OR from MAIN — resolve
        # the trainer RELATIVE TO THIS FILE's tree (shared-venv hijack guard).
        trainer_path = Path(__file__).resolve().parents[3] / TRAINER_RELPATH
    tree = ast.parse(trainer_path.read_text(encoding="utf-8"))
    flags: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                        and arg.value.startswith("--"):
                    flags.add(arg.value)
    return flags


def default_t3_long_burn_program(variant: str, out_dir: str, *, epochs: int = 400,
                                 max_wall_minutes: float = 480.0,
                                 gt_cache: str | None = None,
                                 resume_from: str | None = None) -> TR1RendererProgramV1:
    """The T3 sealed long-burn skeleton (READY_TO_FIRE_UNDER_STANDING_GO — fires from
    MAIN only, never from a build arm). Event-driven schedule inside the trainer;
    resumable-from-disk; per-stage EMA-shadow checkpoints; A1 stage-exit gates."""
    levers = [
        lever_variant(variant),
        lever_token_grid(16, 4),
        lever_renderer_capacity(24),
        lever_desc_level_roundtrip(16, "round"),
        lever_token_temporal("shared_base"),
        lever_seg_physics("ce", 100.0),
        lever_a1_gate(10),
        lever_window(epochs, max_wall_minutes, batch_pairs=8, lr=2e-3),
    ]
    if variant == "lotto":
        levers.append(lever_lotto(118, 0.5))
    return TR1RendererProgramV1(levers=tuple(levers), num_pairs=600, out_dir=out_dir,
                                gt_cache=gt_cache, resume_from=resume_from,
                                full_confirm=True)


def default_t1_smoke_program(variant: str, out_dir: str, *, num_pairs: int = 24,
                             epochs: int = 60, max_wall_minutes: float = 75.0,
                             gt_cache: str | None = None) -> TR1RendererProgramV1:
    """The pre-registered T1 smoke config per variant (both arms of the A2 race)."""
    levers = [
        lever_variant(variant),
        lever_token_grid(16, 4),
        lever_renderer_capacity(24),
        lever_desc_level_roundtrip(16, "round"),
        lever_token_temporal("shared_base"),
        lever_seg_physics("ce", 100.0),
        lever_a1_gate(5),
        lever_window(epochs, max_wall_minutes, batch_pairs=8, lr=2e-3),
    ]
    if variant == "lotto":
        levers.append(lever_lotto(118, 0.5))
    return TR1RendererProgramV1(levers=tuple(levers), num_pairs=num_pairs,
                                out_dir=out_dir, gt_cache=gt_cache)
