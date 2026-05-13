"""Kaggle parallel-sweep harness for the T1 Ballé end-to-end trainer.

This is the kernel script that runs on a Kaggle T4 (free tier). It is the
parallel-dispatch actuator counterpart to the Modal/Vast.ai dispatch path
in ``scripts/operator_authorize_phase1_t1_balle_cheap_config_dispatch.sh`` —
per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" this
file IS the actuator (build BEFORE the ranker; build BEFORE the predictor).

Operating contract:

1. **CUDA required, P100 FATAL.** Kaggle randomly assigns P100 (sm_60,
   unsupported by PyTorch >= 2.5) or T4 (sm_75). The P100 check at the top
   exits with rc=2 BEFORE any model load so the operator wrapper can detect
   the P100 trap and re-push with a new slug. NEVER CPU, NEVER MPS — per
   CLAUDE.md "MPS auth eval is NOISE".

2. **No inline bootstrap.** Per CLAUDE.md "Forbidden re-implementing remote
   bootstrap inline" the tac wheel is consumed from the Kaggle dataset
   mounted at ``/kaggle/input/datasets/adpena/<dataset-slug>/`` rather than
   ``uv pip install``. The dataset is expected to contain a built wheel
   matching the current ``pyproject.toml`` version.

3. **Tier-1 flags from manifest.** Per CLAUDE.md "Deployment version
   checklist" we do NOT hardcode the Tier-1 flag list. We AST-extract the
   ``TIER_1_OPERATOR_REQUIRED_FLAGS`` constant from the trainer source so
   the harness stays in lock-step with the trainer manifest. If the
   trainer adds a new Tier-1 flag, this harness picks it up automatically.

4. **9-hour wall clock budget.** Kaggle's free-tier kernels are capped at
   9h. We default to ``--epochs 1500`` (the dev-loop sweet spot per the
   cost-band council deliberation 2026-05-12) which fits comfortably on T4
   with Tier-1 wins, leaves headroom for auth eval, and produces a
   3000-epoch-half data point that supersedes the ``hand_calibrated_fallback``
   in ``tac.cost_band_calibration.predict('kaggle', 'T4', ...)``.

5. **Proxy auth-eval harvest.** The kernel can run CUDA auth eval to decide
   what deserves exact dispatch, but Kaggle is a proxy/sweep substrate unless
   a future exact target contract wires the full lifecycle, runtime closure,
   claim, harvest, and adjudication path. Metadata emitted from this harness
   must stay ``score_claim=false``, ``promotion_eligible=false``, and
   ``ready_for_exact_eval_dispatch=false``. GHA Linux x86_64 remains the
   ``[contest-CPU]`` axis; exact ``[contest-CUDA]`` requires the canonical
   exact-eval path, not this sweep harness by itself.

6. **Cost-band anchor.** End-of-run appends an anchor to
   ``.omx/state/cost_band_posterior.jsonl`` via
   ``tools/append_cost_band_anchor.py`` with ``--platform kaggle --gpu T4
   --actual-cost-usd 0.0`` (free tier). After 3+ Kaggle anchors land, future
   ``tac.cost_band_calibration.predict('kaggle','T4',...)`` calls return
   ``empirical_posterior``-confidence bands.

Slug variant convention: ``comma-lab-t1-balle-<variant>`` (< 25 chars per
CLAUDE.md "Kaggle API/CLI"). Variants enumerate the hyperparameter axis
the operator wants to sweep (e.g. ``a`` = batch_size 16, ``b`` = batch_size
32, ``c`` = batch_size 64). The variant flag is consumed by the operator
wrapper, NOT by this kernel script.

Forbidden patterns this script honors:
- NO ``uv pip install`` / ``pip install`` / ``curl ... uv`` (use the wheel
  shipped via the Kaggle dataset).
- NO ``--device cpu`` / ``--device mps`` (CUDA required; P100 FATAL).
- NO ``eval_roundtrip=False`` (trainer default is True; we do not touch).
- NO ``/tmp`` paths (Kaggle output goes to ``/kaggle/working/``).
- NO scorer load at inflate time (trainer-side concern; we just dispatch).
- NO hardcoded Tier-1 flag list (AST-extracted).

Usage on Kaggle (executed by ``kaggle kernels push``):

    python experiments/kaggle_t1_balle_sweep.py \\
        --epochs 1500 \\
        --batch-size 32 \\
        --kaggle-dataset-slug adpena/comma-lab-t1-balle-source

Locally (smoke test, exits early with rc=99 if no CUDA):

    python experiments/kaggle_t1_balle_sweep.py --smoke
"""
from __future__ import annotations

import argparse
import ast
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# -----------------------------------------------------------------------------
# P100 FATAL CHECK — must run BEFORE any model/torch heavy import.
# Per CLAUDE.md "Kaggle API/CLI": Kaggle's free-tier GPU assignment is
# random; P100 has sm_60 which PyTorch >= 2.5 does NOT support. Exit early
# with a structured error so the harvester can detect the P100 trap and
# the operator wrapper can re-push.
# -----------------------------------------------------------------------------
P100_FATAL_RC = 2
NO_CUDA_RC = 99


def assert_cuda_t4_or_better(*, _torch_module: Any | None = None) -> dict[str, Any]:
    """Refuse to proceed unless a non-P100 CUDA device is available.

    Returns a dict with ``gpu_name`` / ``major`` / ``minor`` / ``mem_gb`` so
    downstream callers (and tests) can inspect the resolved device.
    Exits with ``P100_FATAL_RC`` on P100; ``NO_CUDA_RC`` on no CUDA.
    """
    if _torch_module is None:
        import torch as _torch_module
    if not _torch_module.cuda.is_available():
        print(
            "[kaggle-t1-balle-sweep] FATAL: torch.cuda.is_available() == False; "
            "Kaggle kernel was not assigned a GPU.",
            file=sys.stderr,
        )
        raise SystemExit(NO_CUDA_RC)
    props = _torch_module.cuda.get_device_properties(0)
    info = {
        "gpu_name": props.name,
        "major": props.major,
        "minor": props.minor,
        "mem_gb": round(props.total_memory / (1024**3), 2),
    }
    # P100 = sm_60. Tesla P100 lists major==6, minor==0. Anything sm_70+ is fine.
    if props.major == 6:
        print(
            f"[kaggle-t1-balle-sweep] FATAL: P100 trap — Kaggle assigned "
            f"{info['gpu_name']} (sm_{props.major}{props.minor}); PyTorch >= 2.5 "
            f"does not support sm_60. Operator wrapper should re-push with a "
            f"new slug suffix.",
            file=sys.stderr,
        )
        print(json.dumps({"gpu_info": info, "trap": "p100"}), file=sys.stderr)
        raise SystemExit(P100_FATAL_RC)
    print(
        f"[kaggle-t1-balle-sweep] GPU OK: {info['gpu_name']} "
        f"(sm_{props.major}{props.minor}, {info['mem_gb']} GB)"
    )
    return info


# -----------------------------------------------------------------------------
# Tier-1 manifest extraction — per CLAUDE.md "Deployment version checklist"
# we never hardcode the Tier-1 flag list. The trainer source is the single
# source of truth; this AST-walk pulls the manifest without importing the
# trainer (avoids pulling its heavy deps at flag-resolution time).
# -----------------------------------------------------------------------------
def extract_tier1_flags(trainer_source_path: Path) -> dict[str, dict[str, Any]]:
    """AST-extract ``TIER_1_OPERATOR_REQUIRED_FLAGS`` from the trainer source.

    We use ``ast.literal_eval`` on the assigned value so this works even when
    the trainer cannot be imported (heavy CUDA/torch deps). Raises ``KeyError``
    if the constant is not found.
    """
    tree = ast.parse(trainer_source_path.read_text(encoding="utf-8"))
    # META-CATALOG-152-FIX 2026-05-12: substrate trainers declare the
    # manifest as `TIER_1: dict[...] = {...}` (AnnAssign), not `TIER_1 = {...}`
    # (Assign). The previous Assign-only walk silently returned KeyError on
    # every substrate trainer in the canvas. Catalog #168 enforces this
    # bug class extinction repo-wide.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            target_iter = list(node.targets)
            value_node = node.value
        elif (isinstance(node, ast.AnnAssign)
              and node.value is not None):
            target_iter = [node.target]
            value_node = node.value
        else:
            continue
        for target in target_iter:
            if isinstance(target, ast.Name) and target.id == "TIER_1_OPERATOR_REQUIRED_FLAGS":
                try:
                    return ast.literal_eval(value_node)
                except (ValueError, SyntaxError):
                    # Fall through; literal_eval rejects tuples-with-names.
                    # In that case extract the keys via AST inspection.
                    keys: dict[str, dict[str, Any]] = {}
                    if isinstance(value_node, ast.Dict):
                        for k_node, v_node in zip(
                            value_node.keys, value_node.values, strict=True
                        ):
                            if isinstance(k_node, ast.Constant) and isinstance(v_node, ast.Dict):
                                entry: dict[str, Any] = {}
                                for ek, ev in zip(
                                    v_node.keys, v_node.values, strict=True
                                ):
                                    if isinstance(ek, ast.Constant) and isinstance(ev, ast.Constant):
                                        entry[ek.value] = ev.value
                                keys[k_node.value] = entry
                    if keys:
                        return keys
    raise KeyError(
        f"TIER_1_OPERATOR_REQUIRED_FLAGS not found in {trainer_source_path}"
    )


def build_trainer_argv(
    *,
    tier1_flags: dict[str, dict[str, Any]],
    epochs: int,
    batch_size: int,
    output_dir: Path,
    video_path: Path,
    auth_eval: bool,
    smoke: bool,
) -> list[str]:
    """Compose the argv that runs the T1 Ballé trainer with all Tier-1 wins on.

    Per CLAUDE.md "NEVER invent CLI flags" — every flag emitted here comes
    from the AST-extracted manifest. We do not hardcode the flag NAMES.
    """
    argv: list[str] = [
        "--output-dir", str(output_dir),
        "--device", "cuda",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--enable-scorer-domain-loss",
        "--video-path", str(video_path),
    ]
    if auth_eval:
        argv.append("--auth-eval")
    if smoke:
        argv.append("--smoke")
    for flag_name, meta in tier1_flags.items():
        default_value = meta.get("default") if isinstance(meta, dict) else None
        if default_value is None:
            # Boolean flag — emit if not already in argv.
            if flag_name not in argv:
                argv.append(flag_name)
        else:
            # Value flag — emit "--flag VALUE".
            if flag_name not in argv:
                argv.extend([flag_name, str(default_value)])
    return argv


# -----------------------------------------------------------------------------
# Path resolution — Kaggle vs local.
# -----------------------------------------------------------------------------
def resolve_repo_root() -> Path:
    """Locate the pact repo root.

    On Kaggle, the wheel is consumed via dataset mount so the runtime
    "repo" is whatever the operator's kernel script provided. We default
    to ``/kaggle/working/`` for outputs and ``/kaggle/input/datasets/<slug>/``
    for the source bundle. Locally we walk up from this file.
    """
    if Path("/kaggle/working").exists():
        # On Kaggle. Source bundle was extracted by the kernel's setup cell.
        return Path("/kaggle/working")
    # Local development path.
    return Path(__file__).resolve().parent.parent


def resolve_trainer_source(repo_root: Path, *, dataset_root: Path | None) -> Path:
    """Find the T1 Ballé trainer source file."""
    candidates = [
        repo_root / "experiments" / "train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py",
    ]
    if dataset_root is not None:
        candidates.append(
            dataset_root / "experiments" / "train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py"
        )
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"T1 Ballé trainer not found at any of: {[str(c) for c in candidates]}"
    )


# -----------------------------------------------------------------------------
# Main kernel entry.
# -----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs",
        type=int,
        default=1500,
        help="Trainer epoch count. 1500 fits Kaggle's 9-hour limit on T4 with Tier-1 wins.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size; trainer default 32 per audit 2026-05-12.",
    )
    parser.add_argument(
        "--kaggle-dataset-slug",
        default=None,
        help=(
            "Kaggle dataset slug (e.g. adpena/comma-lab-t1-balle-source) that "
            "supplies the tac wheel + trainer source. If not set we fall back "
            "to the in-repo working tree (local smoke)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output dir; defaults to /kaggle/working/t1_balle_sweep.",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=None,
        help="Override contest video path; defaults to <repo>/upstream/videos/0.mkv.",
    )
    parser.add_argument(
        "--no-auth-eval",
        action="store_true",
        help="Skip the final auth-eval-on-best stage (smoke / dev-loop only).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="1-epoch smoke test (passes --smoke to the trainer).",
    )
    parser.add_argument(
        "--dispatch-label",
        default=None,
        help="Operator-facing label for cost-band anchor / lane claim audit.",
    )
    parser.add_argument(
        "--actual-cost-usd",
        type=float,
        default=0.0,
        help="Reported cost. Kaggle free tier => 0.0.",
    )
    args = parser.parse_args(argv)

    repo_root = resolve_repo_root()
    dataset_root: Path | None = None
    if args.kaggle_dataset_slug is not None:
        # Kaggle mounts datasets at /kaggle/input/datasets/<slug>/ per CLAUDE.md
        # "Kaggle API/CLI — non-negotiable".
        guess = Path("/kaggle/input/datasets") / args.kaggle_dataset_slug
        if guess.is_dir():
            dataset_root = guess
        else:
            # Fall back to repo root; this is the local-dev path.
            print(
                f"[kaggle-t1-balle-sweep] WARN: dataset mount {guess} not found; "
                f"falling back to repo root {repo_root}.",
                file=sys.stderr,
            )

    output_dir = args.output_dir or (repo_root / "t1_balle_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = args.video_path or (repo_root / "upstream" / "videos" / "0.mkv")

    # -------------------------------------------------------------------------
    # Step 1: P100 trap check. Must run BEFORE any heavy import.
    # -------------------------------------------------------------------------
    gpu_info = assert_cuda_t4_or_better()

    # -------------------------------------------------------------------------
    # Step 2: AST-extract Tier-1 manifest from the trainer source.
    # -------------------------------------------------------------------------
    trainer_source = resolve_trainer_source(repo_root, dataset_root=dataset_root)
    tier1_flags = extract_tier1_flags(trainer_source)
    print(
        f"[kaggle-t1-balle-sweep] extracted {len(tier1_flags)} Tier-1 flags "
        f"from {trainer_source.name}: {sorted(tier1_flags.keys())}"
    )

    # -------------------------------------------------------------------------
    # Step 3: Run the trainer with all Tier-1 wins on.
    # -------------------------------------------------------------------------
    trainer_argv = build_trainer_argv(
        tier1_flags=tier1_flags,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=output_dir,
        video_path=video_path,
        auth_eval=not args.no_auth_eval,
        smoke=args.smoke,
    )
    cmd = [sys.executable, str(trainer_source), *trainer_argv]
    print(f"[kaggle-t1-balle-sweep] launching trainer: {shlex.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, check=False)
    wall_clock_sec = time.time() - t0
    rc = proc.returncode
    print(
        f"[kaggle-t1-balle-sweep] trainer exited rc={rc} "
        f"after wall_clock={wall_clock_sec:.0f}s"
    )

    # -------------------------------------------------------------------------
    # Step 4: Cost-band anchor append. Even on rc!=0 we record the anchor
    # so the posterior reflects real measurement (including failure modes).
    # Per CLAUDE.md "EVERY auth eval must use the EXACT archive that will be
    # submitted" — that gate lives inside the trainer; this is metadata only.
    # -------------------------------------------------------------------------
    label = args.dispatch_label or f"kaggle_t1_balle_{int(t0)}"
    anchor_tool = repo_root / "tools" / "append_cost_band_anchor.py"
    if anchor_tool.is_file():
        anchor_cmd = [
            sys.executable, str(anchor_tool),
            "--dispatch-label", label,
            "--trainer", "experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py",
            "--platform", "kaggle",
            "--gpu", gpu_info["gpu_name"].replace(" ", "_"),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--all-flags-on",
            "--actual-wall-clock-sec", f"{wall_clock_sec:.0f}",
            "--actual-cost-usd", f"{args.actual_cost_usd:.2f}",
            "--notes", f"kaggle_kernel_rc={rc};{gpu_info['gpu_name']}",
        ]
        anchor_proc = subprocess.run(anchor_cmd, check=False, capture_output=True, text=True)
        print(
            f"[kaggle-t1-balle-sweep] cost-band anchor append rc={anchor_proc.returncode}: "
            f"{anchor_proc.stdout.strip()}{anchor_proc.stderr.strip()}"
        )
    else:
        print(
            f"[kaggle-t1-balle-sweep] WARN: cost-band anchor tool not found at "
            f"{anchor_tool}; skipping anchor append.",
            file=sys.stderr,
        )

    # -------------------------------------------------------------------------
    # Step 5: Write a structured kernel summary for the harvester.
    # -------------------------------------------------------------------------
    summary = {
        "dispatch_label": label,
        "trainer": str(trainer_source.name),
        "platform": "kaggle",
        "gpu_info": gpu_info,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "tier1_flags_threaded": sorted(tier1_flags.keys()),
        "wall_clock_sec": wall_clock_sec,
        "trainer_returncode": rc,
        "auth_eval_requested": not args.no_auth_eval,
        "smoke": args.smoke,
        "output_dir": str(output_dir),
    }
    # PCC3 internal-consistency check (CLAUDE.md non-negotiable): when both
    # epochs and wall_clock_sec are present, refuse to write a summary that
    # claims more epochs than physics permits (≥ 50 ms per epoch is a very
    # loose lower bound; a real T4 batched train_renderer epoch is seconds).
    MIN_SEC_PER_EPOCH = 0.05
    if (
        args.epochs > 0
        and wall_clock_sec is not None
        and wall_clock_sec >= 0
        and wall_clock_sec < args.epochs * MIN_SEC_PER_EPOCH
    ):
        raise RuntimeError(
            f"stats internal-consistency violation (PCC3): "
            f"epochs={args.epochs} but wall_clock_sec={wall_clock_sec:.3f} "
            f"< epochs * {MIN_SEC_PER_EPOCH} = "
            f"{args.epochs * MIN_SEC_PER_EPOCH:.3f}. "
            f"Stub-loop suspected; refusing to write summary."
        )
    summary_path = output_dir / "kaggle_kernel_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[kaggle-t1-balle-sweep] wrote summary to {summary_path}")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
