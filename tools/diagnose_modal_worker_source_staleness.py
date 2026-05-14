# SPDX-License-Identifier: MIT
"""Diagnose Modal worker source-staleness vs local HEAD.

PHASE-B1-PIVOT bug-class anchor (2026-05-12):

Two consecutive Modal A100 dispatches of `experiments/train_substrate_sane_hnerv.py`
crashed rc=1 (`fc-01KREXK209TRX7ED5ZRVXHY1VT` 14.77s + `fc-01KREXXSKGTDCF61QXQNBF6SX3`
72.03s). The 72-sec traceback hit
`src/tac/substrates/sane_hnerv/score_aware_loss.py:129 unsqueeze(1)` — a bug
that does NOT exist at HEAD (commit `6048d690` removed it). The canary subagent
hypothesized "Modal worker mounted stale source" but the actual root cause was
**chronological**: the dispatch was fired at 20:26:47Z, the fix landed at
20:44:00Z (17 minutes LATER). The local source on disk at dispatch time WAS
still broken; the worker faithfully ran what it was given.

The structural problem this tool surfaces:

1. ``modal_train_lane.py::main`` reads ``git rev-parse HEAD`` and threads the
   SHA into env-vars on the worker, but does NOT serialize the SHA into
   ``modal_metadata.json``. Post-mortem cannot distinguish "Modal worker
   mounted stale code" from "operator dispatched before fix landed".

2. There is no spawn-time verification that the worker's PYTHONPATH-resolved
   ``tac`` package source matches the local HEAD SHA. The fix to that surface
   lives in ``modal_train_lane.py`` itself (Step 1 of this landing); this tool
   provides a SECOND, independent probe-channel for operators to spot-check
   the mount discipline at any time, without dispatching a real lane.

USAGE
-----

Spawn a tiny Modal A100 probe (~$0.05) that:

* Imports ``tac`` from the worker's PYTHONPATH and prints
  ``tac.__file__``.
* Reads ``/workspace/pact/.git`` HEAD SHA (if ``git`` is on PATH; falls back
  to the env-var ``T1_MOUNTED_CODE_GIT_HEAD`` injected by
  ``modal_train_lane.py``).
* SHA-256 hashes a curated set of source files (the FIX-H sentinel files +
  ``score_aware_loss.py``) on the worker.
* Returns the hashes via the FunctionCall result cache.

Local side compares worker-side hashes to ``git show HEAD:<path> | sha256``
and reports ``H1`` (image cache), ``H2`` (eval-timing), ``H3`` (deploy stale),
``H4`` (manifest gap), ``H5`` (PYTHONPATH ordering), or ``HOK`` (parity).

The probe is designed to be cheap (~10 sec wall-clock on A100, no training,
no scorer load) and to fail-loud if any of the 5 hypotheses is observed in
the field. It is the runtime sister of the static Modal source-parity gate
that should be wired into preflight before any automated Modal fan-out.

TYPICAL OUTPUT (parity case)
----------------------------

::

    [diagnose-modal-staleness] spawning A100 probe (~$0.05 expected)…
    [diagnose-modal-staleness] worker call_id=fc-…
    [diagnose-modal-staleness] worker tac.__file__=/workspace/pact/src/tac/__init__.py
    [diagnose-modal-staleness] worker HEAD=6048d690… (env), git=6048d690… (rev-parse)
    [diagnose-modal-staleness] hash parity:
      score_aware_loss.py worker=8a9b… local=8a9b… ✓
      common.py           worker=…     local=…     ✓
      preflight.py        worker=…     local=…     ✓
    [diagnose-modal-staleness] verdict=HOK (worker source matches local HEAD)
    [diagnose-modal-staleness] cost: $0.04 (8.2s @ A100 $0.000019/s)

DIVERGENCE OUTPUT
-----------------

::

    [diagnose-modal-staleness] hash parity:
      score_aware_loss.py worker=DEAD… local=8a9b… ✗
    [diagnose-modal-staleness] verdict=H4 (mount-manifest gap; required source
      file missing or stale on worker mount)

Notes
-----

* Cost: A100 $1.10/h hand-calibrated → ~$0.05 for a 3-min probe (cold-start +
  10s probe + 1-min teardown). Worth the spend before firing a $5-15 full
  canary.
* This tool does NOT run inside ``preflight_all()``; it's an operator-facing
  diagnostic. A static no-fan-out-without-source-parity gate remains the
  corresponding hardening target.
* Dependency parity with ``experiments/modal_train_lane.py``: same image,
  same mount manifest builder.

Cross-references
----------------

* Modal source-parity dispatch hardening target
* CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" + "Remote code parity"
* ``feedback_phase_b1_nv7_fix_canary_pair_LANDED_20260512.md``
* ``feedback_modal_mount_manifest_consolidation_landed_20260512.md``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

_REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(_REPO_ROOT)

# Sentinel files we hash on both worker and local HEAD. These are the files
# whose stale state caused the WWW4 + 20:26 dispatches to crash. Adding new
# files here is acceptable when a future canary surfaces a new sentinel
# location; the existing entries should NEVER be removed (they represent
# proven prior bug-class anchors).
SENTINEL_FILES: tuple[str, ...] = (
    "src/tac/substrates/sane_hnerv/score_aware_loss.py",
    "src/tac/substrates/score_aware_common.py",
    "src/tac/preflight.py",
    "src/tac/__init__.py",
    "experiments/modal_train_lane.py",
    "src/tac/deploy/modal/mount_manifest.py",
)


def _local_head_sha(repo_root: Path) -> str:
    """Return the local git HEAD SHA, or ``unknown`` if git unavailable."""
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    sha = proc.stdout.strip()
    return sha if proc.returncode == 0 and sha else "unknown"


def _local_file_sha256(repo_root: Path, rel: str) -> str:
    """Return sha256 of the LOCAL on-disk file (not the HEAD blob)."""
    p = repo_root / rel
    if not p.is_file():
        return "MISSING"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _local_head_blob_sha256(repo_root: Path, rel: str) -> str:
    """Return sha256 of the file as committed at HEAD (independent of working
    tree edits). This is the gold standard for comparison: when worker hashes
    match HEAD blob hashes, the worker mounted code matching HEAD."""
    proc = subprocess.run(
        ["git", "show", f"HEAD:{rel}"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return "MISSING"
    return hashlib.sha256(proc.stdout).hexdigest()


def _classify_divergence(
    *,
    worker_head_env: str,
    worker_head_git: str,
    local_head: str,
    worker_hashes: dict,
    local_head_blob_hashes: dict,
    local_disk_hashes: dict,
) -> tuple[str, str]:
    """Return ``(verdict_code, verdict_text)`` from the 5-hypothesis taxonomy.

    Verdicts:
      * HOK — worker matches local HEAD (mount discipline correct)
      * H1  — image cache: worker SHAs reproducible only if image was rebuilt
              with newer source than current mount.
      * H2  — eval-timing: worker tac module path resolves to a snapshot
              older than the env-injected SHA suggests.
      * H3  — deploy stale: env-injected SHA differs from worker git SHA
              (operator dispatched a stale snapshot).
      * H4  — manifest gap: at least one SENTINEL_FILE is MISSING on worker
              (mount didn't include it).
      * H5  — PYTHONPATH ordering: worker tac.__file__ is NOT under
              /workspace/pact (resolved to a different package install).
    """

    # H4: any sentinel file missing on worker
    missing = [k for k, v in worker_hashes.items() if v == "MISSING"]
    if missing:
        return "H4", (
            f"mount-manifest gap; sentinel files missing on worker: "
            f"{', '.join(missing)}"
        )

    # H5: tac.__file__ not under /workspace/pact
    tac_file = worker_hashes.get("__tac_file__", "")
    if tac_file and not tac_file.startswith("/workspace/pact/"):
        return "H5", (
            f"PYTHONPATH ordering: worker tac.__file__={tac_file} is NOT "
            f"under /workspace/pact (resolved to a different install)"
        )

    # H3: env-injected SHA differs from git rev-parse on worker
    if (
        worker_head_env != "unknown"
        and worker_head_git != "unknown"
        and worker_head_env != worker_head_git
    ):
        return "H3", (
            f"deploy stale; env={worker_head_env[:12]} != "
            f"git={worker_head_git[:12]} on worker"
        )

    # H2 / H1: hash divergence
    diverged = []
    for rel, worker_sha in worker_hashes.items():
        if rel.startswith("__"):
            continue
        head_sha = local_head_blob_hashes.get(rel, "MISSING")
        if worker_sha != head_sha and head_sha != "MISSING":
            diverged.append((rel, worker_sha, head_sha))
    if diverged:
        # Distinguish H1 (image cache: worker hash matches an older HEAD blob)
        # from H2 (eval-timing: worker hash matches local disk but not HEAD).
        for rel, worker_sha, head_sha in diverged:
            disk_sha = local_disk_hashes.get(rel, "MISSING")
            if worker_sha == disk_sha:
                return "H2", (
                    f"eval-timing: worker {rel} matches local disk but NOT "
                    f"HEAD blob (worker={worker_sha[:12]}, "
                    f"disk={disk_sha[:12]}, head={head_sha[:12]})"
                )
        return "H1", (
            f"image cache; worker hashes differ from HEAD blobs for "
            f"{len(diverged)} files (oldest divergence: {diverged[0][0]})"
        )

    return "HOK", "worker source matches local HEAD"


def _build_remote_probe_payload() -> str:
    """Return the Python source the Modal worker executes."""
    sentinel_repr = repr(list(SENTINEL_FILES))
    return f"""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import tac

result = {{}}
result["__tac_file__"] = str(Path(tac.__file__).resolve())
result["__tac_path__"] = list(getattr(tac, "__path__", []))
result["__head_env__"] = os.environ.get("T1_MOUNTED_CODE_GIT_HEAD", "unknown")

try:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd="/workspace/pact",
        capture_output=True,
        text=True,
        check=False,
    )
    result["__head_git__"] = proc.stdout.strip() if proc.returncode == 0 else "unknown"
except Exception as exc:
    result["__head_git__"] = f"err:{{exc!r}}"

# Hash sentinel files from the worker's mount.
sentinels = {sentinel_repr}
for rel in sentinels:
    p = Path("/workspace/pact") / rel
    if not p.is_file():
        result[rel] = "MISSING"
        continue
    try:
        result[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError as exc:
        result[rel] = f"err:{{exc!r}}"

print("[probe] OK", flush=True)
"""


def _spawn_probe(*, gpu: str = "A100", quiet: bool = False) -> dict:
    """Spawn the diagnostic probe on Modal and return the parsed result.

    Lazy-imports modal so this tool can be used in --offline-only mode
    (verdict computed against a local fixture).
    """
    import modal  # local import (modal not on every dev path)

    from tac.deploy.modal.mount_manifest import build_training_image

    app = modal.App("comma-diagnose-source-staleness")
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .pip_install("torch==2.5.1", extra_index_url="https://pypi.nvidia.com")
    )
    image = build_training_image(
        image.env({"PYTHONPATH": "/workspace/pact/src:/workspace/pact/upstream:/workspace/pact"}),
        trainer_module_path=None,
    )

    @app.function(image=image, gpu=gpu, timeout=600)
    def probe(payload: str) -> dict:
        # We exec the payload in a fresh namespace so its `result` dict is
        # visible to us.
        ns: dict = {}
        try:
            exec(payload, ns)
        except Exception as exc:
            return {"__exec_error__": f"{type(exc).__name__}: {exc}"}
        out = ns.get("result", {})
        if not isinstance(out, dict):
            return {"__exec_error__": f"result is {type(out).__name__}, not dict"}
        return out

    payload = _build_remote_probe_payload()
    if not quiet:
        print(f"[diagnose-modal-staleness] spawning {gpu} probe (~$0.05 expected)…")
    t0 = time.monotonic()
    with app.run():
        out = probe.remote(payload)
    elapsed = time.monotonic() - t0
    if not quiet:
        print(f"[diagnose-modal-staleness] probe completed in {elapsed:.1f}s")
    out["__elapsed_seconds__"] = elapsed
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Spawn a tiny Modal probe that hashes source files on the worker "
            "and compares against local HEAD. Diagnoses Modal worker "
            "source-staleness via the H1-H5 taxonomy."
        ),
    )
    parser.add_argument(
        "--gpu",
        default="A100",
        help="Modal GPU class for the probe (T4 cheapest; A100 matches the "
        "real dispatch image; default A100).",
    )
    parser.add_argument(
        "--offline-fixture",
        type=Path,
        default=None,
        help=(
            "Skip the Modal spawn; load worker hashes from this JSON fixture "
            "instead. Used by tests + offline post-mortem of harvested "
            "metadata."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Write the structured diagnosis report (worker hashes + local "
            "head + verdict) to this path."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_REPO_ROOT,
        help="Override repo root (test scaffold).",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()

    if args.offline_fixture is not None:
        if not args.offline_fixture.is_file():
            print(
                f"FATAL: --offline-fixture {args.offline_fixture} not found",
                file=sys.stderr,
            )
            return 2
        worker = json.loads(args.offline_fixture.read_text())
    else:
        worker = _spawn_probe(gpu=args.gpu, quiet=args.quiet)

    if "__exec_error__" in worker:
        print(
            f"FATAL: worker probe execution failed: {worker['__exec_error__']}",
            file=sys.stderr,
        )
        return 3

    local_head = _local_head_sha(repo_root)
    head_blob = {
        rel: _local_head_blob_sha256(repo_root, rel)
        for rel in SENTINEL_FILES
    }
    disk = {
        rel: _local_file_sha256(repo_root, rel)
        for rel in SENTINEL_FILES
    }

    verdict_code, verdict_text = _classify_divergence(
        worker_head_env=worker.get("__head_env__", "unknown"),
        worker_head_git=worker.get("__head_git__", "unknown"),
        local_head=local_head,
        worker_hashes=worker,
        local_head_blob_hashes=head_blob,
        local_disk_hashes=disk,
    )

    report = {
        "schema": "modal_worker_source_staleness_diagnosis_v1",
        "local_head": local_head,
        "worker": worker,
        "local_head_blob_sha256": head_blob,
        "local_disk_sha256": disk,
        "verdict_code": verdict_code,
        "verdict_text": verdict_text,
    }

    if not args.quiet:
        print(f"[diagnose-modal-staleness] verdict={verdict_code} ({verdict_text})")
        print(
            f"[diagnose-modal-staleness] local HEAD={local_head[:12]}; "
            f"worker env-HEAD={worker.get('__head_env__', 'unknown')[:12]}; "
            f"worker git-HEAD={worker.get('__head_git__', 'unknown')[:12]}"
        )
        for rel in SENTINEL_FILES:
            w = worker.get(rel, "MISSING")
            h = head_blob.get(rel, "MISSING")
            mark = "✓" if w == h else "✗"
            short_w = w[:12] if w not in ("MISSING",) else w
            short_h = h[:12] if h not in ("MISSING",) else h
            print(f"  {mark} {rel}  worker={short_w}  head={short_h}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if not args.quiet:
            print(f"[diagnose-modal-staleness] report → {args.output_json}")

    return 0 if verdict_code == "HOK" else 4


__all__ = [
    "SENTINEL_FILES",
    "_build_remote_probe_payload",
    "_classify_divergence",
    "_local_file_sha256",
    "_local_head_blob_sha256",
    "_local_head_sha",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
