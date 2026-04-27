#!/usr/bin/env python3
"""DEPRECATED — FP4 quality is now measured through the canonical auth-eval path.

Codex R5-2 Finding #2 (2026-04-27): this smoke-test script imported three
functions that never landed in their target modules — `extract_masks_from_video`
(`tac.camera`), `posenet_forward_pair` (`tac.scorer`), and `segnet_disagreement`
(`tac.scorer`). It crashed with ImportError on every run since fc38d60a
introduced it. With the dead-resolver scanner now strict, the import would block
preflight as well.

Use the canonical FP4 quality path instead:

    1. Train + auto-FP4-export (CUDA):
         .venv/bin/python -m tac.experiments.train_renderer \\
             --profile <profile> --tag <tag> \\
             --auth-eval-on-best \\
             --auth-eval-masks .../masks.mkv \\
             --auth-eval-poses .../optimized_poses.bin

    2. Standalone FP4 auth eval against an existing checkpoint (CUDA):
         .venv/bin/python experiments/auth_eval_renderer.py \\
             --checkpoint renderer.bin \\
             --upstream-dir .../upstream \\
             --device cuda \\
             --archive-size-bytes <bytes> \\
             --output-dir <dir> \\
             --poses .../optimized_poses.bin

Both routes go through the contest scoring formula on the EXACT submission
archive bytes (CLAUDE.md non-negotiable). The previous proxy-style scoring in
this file would have been [advisory only] at best and a "wrong baseline" trap
at worst (cf. memory: feedback_proxy_auth_math_useless).
"""
from __future__ import annotations

import sys


def main() -> int:
    print(__doc__, file=sys.stderr)
    print(
        "ERROR: experiments/test_fp4_quality.py is deprecated. "
        "See the docstring above for the canonical replacement.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
