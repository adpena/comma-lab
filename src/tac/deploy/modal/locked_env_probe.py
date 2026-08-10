# SPDX-License-Identifier: MIT
"""Prove a Modal image's LOCKED upstream venv before spending a paid eval on it.

WHY THIS EXISTS. On 2026-08-10 a real n600 T4 CUDA row came back
(S = 0.170536856816211 @ 188,636 B, -0.001604 vs the PR130 bar) and was REFUSED by
``contest_auth_eval.py``'s upstream-lock parity gate: the image hand-pinned torch 2.5.1
while ``upstream/uv.lock`` resolves 2.9.0+cu128. The cure is to BUILD ``upstream/.venv``
from the lock inside the image (see ``UPSTREAM_UV_GROUP_CUDA`` /
``UPSTREAM_UV_GROUP_CPU`` in the dispatchers) and evaluate through it.

That cure is a NEW image layer. Firing a paid GPU eval on an unbuilt layer risks
spending the envelope to discover a ``uv sync`` failure. So this module is the CHEAP
PROOF: attached to a CPU function on the SAME image object the paid path uses, it forces
the identical build and then verifies three things that a paid row depends on:

1. the locked interpreter EXECS (the 2026-08-10 failure was a macOS Mach-O binary copied
   into a Linux image -- it ``exists()`` but raises ``OSError(8, Exec format error)``);
2. it imports the four packages the parity gate compares, and reports their versions;
3. it can import what ``upstream/evaluate.py`` imports under the real PYTHONPATH,
   including the group-specific DALI on the CUDA axis. An import failure there is
   exactly what would kill a paid row mid-flight.

Deliberately NOT here: any assertion on specific version literals. The parity gate in
``contest_auth_eval.py`` is the single authority on which versions must match; a second
hardcoded copy would be a drifting authority (constants-are-poison).

This is a BUILD/EXEC proof, never a score. It runs on CPU, invokes no scorer, and emits
``score_claim=False``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

# Reported by the parity gate in experiments/contest_auth_eval.py. Imported by name so
# the probe reports exactly the packages the gate compares -- not a re-typed list.
PARITY_PACKAGES: tuple[str, ...] = ("torch", "torchvision", "timm", "numpy")

_VERSION_PROBE = (
    "import sys, json, torch, torchvision, timm, numpy;"
    "print(json.dumps({"
    "'python': sys.version.split()[0],"
    "'executable': sys.executable,"
    "'torch': torch.__version__,"
    "'torchvision': torchvision.__version__,"
    "'timm': timm.__version__,"
    "'numpy': numpy.__version__}))"
)


def _run(argv: list[str], timeout: int) -> tuple[int, str, str]:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


def _last_json(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def probe_locked_upstream_env(
    upstream_dir: str,
    *,
    expect_dali: bool,
    timeout: int = 600,
) -> dict[str, Any]:
    """Verify the locked upstream venv in THIS container. Returns a typed receipt.

    ``upstream_dir`` is the remote upstream root (the dispatcher's
    ``REMOTE_REPO / "upstream"``). ``expect_dali`` is True on the CUDA axis, whose
    dependency group carries ``nvidia-dali-cuda120``; the CPU group does not.

    ``ok`` is True only when the locked interpreter execs, reports all parity packages,
    and can import upstream's own modules. It is a PRECONDITION for spending, never
    evidence about a score.
    """

    venv_python = f"{upstream_dir}/.venv/bin/python"
    receipt: dict[str, Any] = {
        "schema": "modal_locked_upstream_env_probe.v1",
        "score_claim": False,
        "scorer_invoked": False,
        "upstream_dir": upstream_dir,
        "venv_python": venv_python,
        "parity_packages": list(PARITY_PACKAGES),
        "expect_dali": expect_dali,
    }

    rc, out, err = _run([venv_python, "-c", _VERSION_PROBE], timeout)
    receipt["locked_rc"] = rc
    receipt["locked_env"] = _last_json(out)
    if rc != 0:
        receipt["locked_stderr_tail"] = err[-2000:]

    # The image interpreter, for the contrast the 2026-08-10 refusal was about. Recorded
    # as context; the parity gate -- not this probe -- decides whether it matters.
    rc_img, out_img, _ = _run([sys.executable, "-c", _VERSION_PROBE], timeout)
    receipt["image_rc"] = rc_img
    receipt["image_env"] = _last_json(out_img)

    dali_stmt = "import nvidia.dali as _d; print('dali', _d.__version__);" if expect_dali else ""
    evaluate_probe = (
        f"import sys; sys.path.insert(0, {upstream_dir!r});"
        "import frame_utils, modules;"
        f"{dali_stmt}"
        "print('evaluate-imports-ok')"
    )
    rc_ev, out_ev, err_ev = _run([venv_python, "-c", evaluate_probe], timeout)
    receipt["evaluate_import_rc"] = rc_ev
    receipt["evaluate_import_stdout"] = out_ev.strip()[-400:]
    if rc_ev != 0:
        receipt["evaluate_import_stderr_tail"] = err_ev[-2000:]

    locked = receipt["locked_env"] or {}
    receipt["missing_parity_packages"] = [p for p in PARITY_PACKAGES if not locked.get(p)]
    receipt["ok"] = bool(
        rc == 0 and rc_ev == 0 and not receipt["missing_parity_packages"]
    )
    return receipt


__all__ = ["PARITY_PACKAGES", "probe_locked_upstream_env"]
