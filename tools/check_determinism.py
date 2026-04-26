"""Determinism preflight for remote training.

Run BEFORE pipeline.py / train_renderer.py to verify:
  - CUDA available
  - CUBLAS_WORKSPACE_CONFIG set (required for bit-exact cuBLAS)
  - Profile has seed + deterministic + eval_roundtrip configured

Usage:
    python tools/check_determinism.py <profile_name>

Exits 0 on pass, 1 on any failure with explicit diagnostic.
Reusable: invoked by scripts/remote_train_bootstrap.sh and any future
deploy path. Centralised so adding new determinism rules updates every
caller automatically.
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: check_determinism.py <profile_name>", file=sys.stderr)
        return 1
    profile_name = sys.argv[1]

    # 1. CUDA availability (CLAUDE.md MPS-NOISE non-negotiable: auth eval
    # must be on CUDA, never MPS).
    import torch
    if not torch.cuda.is_available():
        print(f"FATAL: CUDA not available — torch={torch.__version__}", file=sys.stderr)
        return 1
    print(f"OK torch={torch.__version__} cuda={torch.cuda.is_available()} "
          f"device={torch.cuda.get_device_name(0)}")

    # 2. CUBLAS_WORKSPACE_CONFIG (required BEFORE first cuBLAS call).
    cuda_cfg = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if cuda_cfg not in (":4096:8", ":16:8"):
        print(
            f"FATAL: CUBLAS_WORKSPACE_CONFIG={cuda_cfg!r} — must be ':4096:8' "
            f"for deterministic cuBLAS matmuls. Export BEFORE any torch CUDA "
            f"call, otherwise PyTorch raises at first matmul.",
            file=sys.stderr,
        )
        return 1
    print(f"OK CUBLAS_WORKSPACE_CONFIG={cuda_cfg}")

    # 3. Profile-side determinism contract.
    sys.path.insert(0, "src")
    try:
        from tac.profiles import PROFILES
    except ImportError as e:
        print(f"FATAL: cannot import tac.profiles ({e})", file=sys.stderr)
        return 1
    if profile_name not in PROFILES:
        print(f"FATAL: profile {profile_name!r} not in PROFILES", file=sys.stderr)
        return 1
    prof = PROFILES[profile_name]

    seed = prof.get("seed")
    if seed is None:
        print(f"FATAL: profile {profile_name!r} missing 'seed' key", file=sys.stderr)
        return 1
    print(f"OK profile.seed={seed}")

    deterministic = prof.get("deterministic")
    if deterministic is not True:
        print(
            f"FATAL: profile {profile_name!r} has deterministic={deterministic!r}, "
            f"must be True per CLAUDE.md canonical pipeline standard.",
            file=sys.stderr,
        )
        return 1
    print(f"OK profile.deterministic=True")

    eval_rt = prof.get("eval_roundtrip")
    if eval_rt is not True:
        print(
            f"FATAL: profile {profile_name!r} has eval_roundtrip={eval_rt!r}, "
            f"must be True (CLAUDE.md HIGHEST EMPHASIS — proxy-auth gap "
            f"2-11x without it).",
            file=sys.stderr,
        )
        return 1
    print(f"OK profile.eval_roundtrip=True")

    # 4. Sanity: configure_reproducibility actually applies. Smoke-test by
    # importing it and invoking — catches any regressions in the function
    # itself (e.g. a future PR that disables one of the calls).
    try:
        from tac.experiments.train_renderer import configure_reproducibility
        configure_reproducibility(seed=seed, deterministic=True)
    except Exception as e:
        print(f"FATAL: configure_reproducibility raised: {e}", file=sys.stderr)
        return 1
    # After configure_reproducibility, cudnn must be deterministic + non-bench.
    if torch.backends.cudnn.benchmark is True:
        print(f"FATAL: cudnn.benchmark=True after configure_reproducibility", file=sys.stderr)
        return 1
    if torch.backends.cudnn.deterministic is False:
        print(f"FATAL: cudnn.deterministic=False after configure_reproducibility",
              file=sys.stderr)
        return 1
    print(f"OK cudnn.{{deterministic=True, benchmark=False}}")

    print(f"\nDETERMINISM_OK profile={profile_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
