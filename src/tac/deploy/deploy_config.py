"""Provider-agnostic training configuration for the asymmetric warp renderer.

This is the single source of truth for all experiment hyperparameters.
Modal, Lightning, and Kaggle all import from here — no flag duplication.

Provider-specific concerns (volume mounts, SSH wiring, image builds, kernel
bootstrap) stay in each provider's own module/script. Only the training
command flags live here.

Usage (Python):
    from tac.deploy.deploy_config import BASE_FLAGS, VARIANT_FLAGS, ALL_VARIANTS
    cmd = ["python", script_path] + build_flags(variant="supervised")

Usage (bash — emits shell-safe flags for eval/source):
    python -m tac.deploy.deploy_config --variant supervised
    python -m tac.deploy.deploy_config --variant raft_only
    python -m tac.deploy.deploy_config --variant base
    python -m tac.deploy.deploy_config --list-variants
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Canonical experiment script (relative path from repo root)
# ---------------------------------------------------------------------------
EXPERIMENT_SCRIPT = "experiments/train_renderer_fridrich.py"

# ---------------------------------------------------------------------------
# Base flags — council-approved v4 configuration (Round 19 fixes applied)
#
# Changelog:
#   v2: Lagrangian stability, flow warmup, batch size, 43 bugs
#   v3: 20K epochs, tighter pose_boundary (asym_v3_longer_tight)
#   v4: double PoseNet fwd fix, p1_pose_sup_weight split, ego_flow reuse,
#       logging cache (all 4 Round-19 council issues resolved)
#   v5: Lagrangian R2 caps (rho_max 1000→100, lambda_cap 10000→1000)
# ---------------------------------------------------------------------------
BASE_FLAGS: list[str] = [
    "--pair-mode", "asymmetric",
    "--epochs", "20000",
    "--batch-size", "4",        # T4-safe: scorers+GT+masks+RAFT all loaded
    "--lr", "2e-4",
    "--embed-dim", "6",
    "--base-ch", "36",
    "--mid-ch", "60",
    "--motion-hidden", "32",
    "--max-flow-px", "20.0",
    "--max-residual", "20.0",
    "--seg-boundary", "0.005",
    "--pose-boundary", "0.02",
    "--rho-init", "10.0",
    "--rho-growth", "1.005",
    "--rho-max", "100",         # council R2: 1000 saturated at ep~920; 100 gives task loss room
    "--lambda-cap", "1000",      # council R2: 10000 → 1000 to prevent λ dominating supervision
    "--phase1-end", "0.25",
    "--phase2-end", "0.85",
    "--flow-warmup-epochs", "0",      # no warmup when resuming
    "--residual-ramp-epochs", "0",
    "--tv-weight", "0.1",
    "--flow-weight", "0.0",
    "--rate-weight", "0.01",
    "--target-bytes", "256000",
    "--gate-reg-weight", "0.1",
    "--even-pairs-only",
    "--device", "cuda",
    "--seed", "42",
    "--checkpoint-every", "500",
    "--eval-every", "200",
    "--log-every", "25",
    "--max-hours", "5.5",
    "--phase2-mse-weight", "0.1",
    "--p1-pose-sup-weight", "0.1",    # round 19: separate from p1-pose-weight=0.01
]

# ---------------------------------------------------------------------------
# Variant flag extensions
#
# Path A (supervised): PoseNet supervision (Layer 1) + RAFT flow (Layer 2)
#   Tests whether geometric prior + direct pose supervision accelerates
#   convergence past what the Lagrangian alone achieves.
#
# Path B (raft_only): RAFT flow supervision only
#   Isolates Layer 2 contribution. If Path A wins, this tells us whether
#   supervision or RAFT flow drove the improvement.
#
# base: No supervision layers — pure Lagrangian. Use as baseline/resume source.
#
# Prerequisites on results volume:
#   /results/posenet_targets.bin  (required by supervised)
#   /results/raft_flow.pt         (required by supervised + raft_only)
# ---------------------------------------------------------------------------
VARIANT_FLAGS: dict[str, list[str]] = {
    "base": [],
    "supervised": [
        "--pose-supervision-weight", "0.5",
        "--pose-targets-path", "/results/posenet_targets.bin",
        "--flow-supervision-weight", "0.1",
        "--raft-flow-path", "/results/raft_flow.pt",
    ],
    "raft_only": [
        "--flow-supervision-weight", "0.1",
        "--raft-flow-path", "/results/raft_flow.pt",
    ],
}

ALL_VARIANTS: list[str] = list(VARIANT_FLAGS.keys())


def build_flags(
    variant: str = "base",
    provider_script_path: str | None = None,
    resume_from: str | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    """Build the full training flag list for a given variant.

    Args:
        variant: One of ALL_VARIANTS.
        provider_script_path: If given, prepend ["python", path] to the result.
        resume_from: If given, append ["--resume", resume_from].
        extra: Additional flags appended last.

    Returns:
        Full command as a flat list of strings.
    """
    if variant not in VARIANT_FLAGS:
        raise ValueError(f"Unknown variant {variant!r}. Valid: {ALL_VARIANTS}")

    flags = list(BASE_FLAGS) + list(VARIANT_FLAGS[variant])

    if extra:
        flags += list(extra)

    if resume_from:
        # Guard: raise if --resume already appears in flags (from extra) to prevent silent duplication
        if "--resume" in flags:
            raise ValueError(
                f"build_flags: resume_from={resume_from!r} was provided but '--resume' already "
                f"appears in extra flags. Pass resume via resume_from only, not both."
            )
        flags += ["--resume", resume_from]

    if provider_script_path:
        return ["python", provider_script_path] + flags

    return flags


# ---------------------------------------------------------------------------
# CLI — emit flags for bash consumption
# ---------------------------------------------------------------------------
def _main() -> None:
    import argparse
    import shlex

    parser = argparse.ArgumentParser(
        description="Emit provider-agnostic training flags for a given variant."
    )
    parser.add_argument("--variant", default="base", choices=ALL_VARIANTS)
    parser.add_argument("--script-path", default=None,
                        help="If set, prepend 'python <path>' to output")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--list-variants", action="store_true",
                        help="Print available variants and exit")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON array instead of shell-quoted string")
    args = parser.parse_args()

    if args.list_variants:
        for v in ALL_VARIANTS:
            extra = " (base: no supervision layers)" if v == "base" else ""
            print(f"  {v}{extra}")
        return

    cmd = build_flags(
        variant=args.variant,
        provider_script_path=args.script_path,
        resume_from=args.resume_from,
    )

    if args.json:
        import json
        print(json.dumps(cmd))
    else:
        # Shell-safe for eval "$(python -m tac.deploy.deploy_config ...)"
        print(shlex.join(cmd))


if __name__ == "__main__":
    _main()
