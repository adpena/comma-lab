# SPDX-License-Identifier: MIT
"""Python entry point for canonical remote bootstraps.

DX #10 (2026-04-26): the three remote bootstrap shell scripts
(`scripts/remote_*_bootstrap.sh`) are 240+ lines of bash each. They work,
but they are not composable: callers cannot import them, unit-test them,
or compose stages programmatically. This module is a Python-only entry
point that scaffolds the same workflow with a typed, importable API:

    from tac.deploy.bootstrap import Bootstrap

    bs = Bootstrap.for_profile("shiraz")
    bs.run_train()      # equivalent to remote_train_bootstrap.sh shiraz
    bs.run_pose_tto()   # equivalent to remote_pose_tto_bootstrap.sh shiraz

The bash bootstraps remain the ONLY way to launch a one-shot remote
session (they self-contain tmux + heartbeat + apt deps). This Python
interface is for tooling that needs to compose the workflow — e.g. the
lane watchdog, CI smoke tests, or future self-healing daemons.

THIS IS A SCAFFOLD. The implementation here delegates to the bash
scripts via subprocess. A full multi-day reimplementation would replace
the bash with native Python (apt-via-Popen, tmux-via-libtmux,
heartbeat-via-thread). That work is tracked separately; the scaffold
exists so consumers can write against the API today.

Invoke the CLI:
    python -m tac.deploy.bootstrap <profile> [--mode train|pose_tto|pose_tto_only] [...]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "scripts"

BootstrapMode = Literal["train", "pose_tto", "pose_tto_only"]


@dataclass
class Bootstrap:
    """Scaffold for the canonical bootstrap workflow.

    Public attributes mirror the bash script CLI args. Methods invoke
    the underlying script via subprocess; replace this with native
    Python in a future PR (out of scope for this DX pass).
    """
    profile: str
    output_subdir: str = ""
    workspace: Path = field(default_factory=lambda: Path("/workspace/pact"))

    @classmethod
    def for_profile(cls, profile: str, output_subdir: str | None = None) -> "Bootstrap":
        """Construct with output_subdir defaulting to the bash convention."""
        sub = output_subdir or f"experiments/results/{profile.lower()}"
        return cls(profile=profile.lower(), output_subdir=sub)

    # ── Stage helpers ────────────────────────────────────────────────────

    def _script_path(self, mode: BootstrapMode) -> Path:
        """Map a mode to its bash script path. Centralised so the
        Python re-implementation only has to touch one function."""
        return {
            "train": SCRIPTS / "remote_train_bootstrap.sh",
            "pose_tto": SCRIPTS / "remote_pose_tto_bootstrap.sh",
            "pose_tto_only": SCRIPTS / "remote_pose_tto_only_bootstrap.sh",
        }[mode]

    def run_train(self, *, dry_run: bool = False) -> int:
        """Launch the canonical train bootstrap.

        Equivalent to:
            bash scripts/remote_train_bootstrap.sh <profile> <output_subdir>
        """
        script = self._script_path("train")
        argv = ["bash", str(script), self.profile, self.output_subdir]
        return self._exec(argv, dry_run=dry_run)

    def run_pose_tto(self, checkpoint: str, masks: str,
                     *, dry_run: bool = False) -> int:
        """Launch the canonical full post-training (FP4 + QAT + pose TTO + auth eval).

        Equivalent to:
            bash scripts/remote_pose_tto_bootstrap.sh <ckpt> <masks> <profile> <out>
        """
        script = self._script_path("pose_tto")
        argv = ["bash", str(script), checkpoint, masks,
                self.profile, self.output_subdir]
        return self._exec(argv, dry_run=dry_run)

    def run_pose_tto_only(self, renderer_bin: str, masks: str,
                          *, posetto_noise_std: float = 0.5,
                          steps: int = 1000,
                          dry_run: bool = False) -> int:
        """Launch the canonical pose-TTO-only bootstrap (no QAT, no retrain).

        Equivalent to:
            bash scripts/remote_pose_tto_only_bootstrap.sh <bin> <masks> <out> <noise> <steps>
        """
        script = self._script_path("pose_tto_only")
        argv = ["bash", str(script), renderer_bin, masks, self.output_subdir,
                str(posetto_noise_std), str(steps)]
        return self._exec(argv, dry_run=dry_run)

    # ── internals ────────────────────────────────────────────────────────

    def _exec(self, argv: list[str], *, dry_run: bool) -> int:
        """Run the bash script with our env. dry_run prints + exits 0.

        We intentionally do NOT capture output — the bash scripts already
        write to tmux + tee to a logfile; capturing would buffer GB of
        training output in memory.
        """
        if dry_run:
            print("[bootstrap dry-run]", " ".join(argv))
            return 0
        env = os.environ.copy()
        # The bash scripts expect to be run on the remote. Locally this
        # will fail at the first stage (no /workspace/pact). That's fine
        # for tests — they should pass dry_run=True.
        try:
            return subprocess.call(argv, env=env)
        except FileNotFoundError:
            print(f"[bootstrap] bash or {argv[1]} not on PATH", file=sys.stderr)
            return 127


def main(argv: list[str] | None = None) -> int:
    """CLI: `python -m tac.deploy.bootstrap <profile> --mode <mode> ...`"""
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("profile", help="Profile name (e.g. 'shiraz', 'den')")
    p.add_argument("--mode", choices=["train", "pose_tto", "pose_tto_only"],
                   default="train", help="Which bootstrap stage to run")
    p.add_argument("--output-subdir", default=None,
                   help="Override output dir (default: experiments/results/<profile>)")
    p.add_argument("--checkpoint", default=None,
                   help="Path to checkpoint .pt (pose_tto mode) or .bin (pose_tto_only)")
    p.add_argument("--masks", default=None,
                   help="Path to masks.mkv (required for pose_tto / pose_tto_only)")
    p.add_argument("--posetto-noise-std", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--dry-run", action="store_true",
                   help="Print the bash invocation without running it")
    args = p.parse_args(argv)

    bs = Bootstrap.for_profile(args.profile, args.output_subdir)

    if args.mode == "train":
        return bs.run_train(dry_run=args.dry_run)
    if args.mode == "pose_tto":
        if not (args.checkpoint and args.masks):
            print("--checkpoint and --masks required for pose_tto mode",
                  file=sys.stderr)
            return 2
        return bs.run_pose_tto(args.checkpoint, args.masks, dry_run=args.dry_run)
    # pose_tto_only
    if not (args.checkpoint and args.masks):
        print("--checkpoint and --masks required for pose_tto_only mode",
              file=sys.stderr)
        return 2
    return bs.run_pose_tto_only(args.checkpoint, args.masks,
                                posetto_noise_std=args.posetto_noise_std,
                                steps=args.steps,
                                dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
