#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""B2 exact-eval BRIDGE: HiNeRV backend-only archive -> contest score.

This is the handoff that turns a HiNeRV BACKEND-ONLY ``archive.zip`` (the
sidecar-stripped double-win packet emitted by
``tools/build_hi_nerv_backend_only_archive.py`` /
``tac.substrates.hi_nerv.archive_candidate``) into an EXACT contest score so a
B1 pilot archive can be arbitrated against the public frontier the moment it
lands.

Pipeline (identical to the contest ``upstream/evaluate.sh`` flow, reused via the
canonical, fully-hardened ``experiments/contest_auth_eval.py``):

    archive.zip (member ``x`` or ``0.bin``)
      -> build the canonical HiNeRV ``inflate.sh`` + ``inflate.py`` runtime dir
         (vendored, numpy/torch/brotli-portable, NO scorer imports)
      -> inflate.sh <archive_dir> <inflated_dir> <file_list>   (per-video .raw)
      -> upstream/evaluate.py --device {cpu,cuda}               (d_seg/d_pose/rate)
      -> parse exact final score
      -> emit hi_nerv_backend_only_exact_eval.v1 JSON

This tool does NOT reimplement ``evaluate.py`` or the inflate decode. It REUSES
``experiments/contest_auth_eval.py`` (the canonical archive.zip -> inflate.sh ->
evaluate.py -> score pipeline with full ZIP integrity, runtime-tree custody, and
device->evidence-grade tagging) and the canonical HiNeRV runtime emitter
(``tac.substrates._shared.pact_nerv_full_main.write_contest_runtime`` semantics,
adapted for the ``hi_nerv`` package + member-``x`` parser).

=== AUTHORITATIVE vs ADVISORY (NON-NEGOTIABLE, per CLAUDE.md) ===

* ``--device cpu`` on LOCAL macOS is ``[macOS-CPU advisory]`` — NEVER a contest
  score, NEVER a frontier claim. The downstream ``contest_auth_eval.py``
  evidence contract stamps this automatically (score_claim=false,
  promotable=false). Its ONLY purpose is to prove the e2e pipeline RUNS and to
  measure inflate/evaluate wall-clock vs the 30-minute contest budget.
* The AUTHORITATIVE B2 result requires BOTH axes on 1:1 contest-compliant
  hardware: ``[contest-CPU]`` on Linux x86_64 (the public leaderboard axis) AND
  ``[contest-CUDA]`` on an NVIDIA T4/equivalent. Those are paid + deferred; this
  tool emits the exact dual-axis recipe (see ``--print-dual-axis-recipe`` and
  the ``dual_axis_authoritative_recipe`` block in the emitted JSON) but does NOT
  fire them.
* A 1-pair smoke archive (num_pairs < 600) CANNOT pass the contest's 600-sample
  assertion. For such archives the bridge runs the INFLATE stage end-to-end
  (the PR106 dep-closure-bug-class half) and reports
  ``pipeline_inflate_ok_evaluate_requires_600_pairs`` honestly — the
  PIPELINE-WORKS claim is real; there is no score claim.

Usage (advisory-local validation on a real backend-only archive):

    .venv/bin/python tools/run_hi_nerv_backend_only_b2_exact_eval.py \\
        --archive /Volumes/VertigoDataTier/pact/hinerv_backend_only_candidate_20260608/archive_backend_only.zip \\
        --replay-row /Volumes/VertigoDataTier/pact/hinerv_backend_only_candidate_20260608/hi_nerv_backend_only_exact_replay.json \\
        --device cpu \\
        --work-root /Volumes/VertigoDataTier/pact/b2_bridge_work \\
        --out-row /Volumes/VertigoDataTier/pact/b2_bridge_work/hi_nerv_backend_only_exact_eval.json

Authoritative dual-axis recipe (paid; deferred until a B1 600-pair archive
lands + operator approval):

    .venv/bin/python tools/run_hi_nerv_backend_only_b2_exact_eval.py \\
        --archive <B1_600pair_archive.zip> --print-dual-axis-recipe
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

# Imported AFTER ensure_repo_imports so `tac` resolves under direct script
# execution, not only under an editable-install interpreter.
from tac.process_group_kill import run_in_process_group  # noqa: E402

CONTEST_ARCHIVE_RATE_DENOM = 37_545_489
CONTEST_NUM_EVAL_SAMPLES = 600
CONTEST_BUDGET_SECONDS = 1800  # 30 min on T4 (upstream/README budget)
SCHEMA = "hi_nerv_backend_only_exact_eval.v1"
DEFAULT_VIDEO_NAMES = "upstream/public_test_video_names.txt"
# The canonical x/0.bin-accepting HiNeRV inflate.py emitted into the runtime dir.
# Mirrors tac.substrates._shared.pact_nerv_full_main.write_contest_runtime but
# pinned to the hi_nerv package + member-``x`` parser (matches the canonical
# hi_nerv_export runtime so a backend-only archive whose member is ``x`` decodes).
_HI_NERV_INFLATE_PY = '''#!/usr/bin/env python
"""hi_nerv contest-compliant inflate runtime (B2 bridge emitted).

Reads archive_dir/x or archive_dir/0.bin via the packaged substrate parser, then
for each video in file_list writes contest raw bytes under output_dir/*.raw.
No scorer-network imports (strict-scorer-rule contract). numpy/torch/brotli
portable (no MLX dep) per the MLX-FIRST standing directive.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'src'))
from tac.substrates.hi_nerv.inflate import inflate_one_video


def _read_archive_bytes(archive_dir: Path) -> bytes:
    present = [p for p in (archive_dir / 'x', archive_dir / '0.bin') if p.is_file()]
    if len(present) != 1:
        names = ', '.join(p.name for p in present) or 'none'
        raise FileNotFoundError(
            "expected exactly one payload member named 'x' or '0.bin'; "
            f"found {names}"
        )
    return present[0].read_bytes()


def main() -> int:
    if len(sys.argv) != 4:
        print('usage: inflate.py <archive_dir> <output_dir> <file_list>',
              file=sys.stderr)
        return 2
    archive_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])
    archive_bytes = _read_archive_bytes(archive_dir)
    for line in file_list_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        rel = Path(line).with_suffix('.raw')
        if rel.is_absolute() or any(part in {'', '..'} for part in rel.parts):
            raise ValueError(f'unsafe file_list entry: {line!r}')
        inflate_one_video(archive_bytes, output_dir / rel, device='cpu')
    return 0


if __name__ == '__main__':
    sys.exit(main())
'''

_HI_NERV_INFLATE_SH = (
    "#!/usr/bin/env bash\n"
    "# hi_nerv backend-only contest-compliant inflate (B2 bridge emitted)\n"
    "# Contract: $1=archive_dir $2=output_dir $3=file_list\n"
    "set -euo pipefail\n"
    'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
    'DATA_DIR="$1"\n'
    'OUTPUT_DIR="$2"\n'
    'FILE_LIST="$3"\n'
    'mkdir -p "$OUTPUT_DIR"\n'
    'export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"\n'
    'exec "${PYTHON:-python3}" "$HERE/inflate.py" "$DATA_DIR" "$OUTPUT_DIR" "$FILE_LIST"\n'
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _utc_compact() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _archive_member_names(archive: Path) -> list[str]:
    with zipfile.ZipFile(archive) as zf:
        return [n for n in zf.namelist() if not n.endswith("/")]


def _payload_member_bytes(archive: Path) -> tuple[str, bytes]:
    with zipfile.ZipFile(archive) as zf:
        members = [n for n in zf.namelist() if not n.endswith("/")]
        pick = [n for n in members if n in {"x", "0.bin"}] or members
        if len(pick) != 1:
            raise SystemExit(
                f"expected exactly one payload member (x/0.bin); got {members}"
            )
        return pick[0], zf.read(pick[0])


def _probe_num_pairs(archive: Path) -> int | None:
    """Return the HiNeRV archive's pair count, or None if it cannot be parsed.

    Used ONLY to classify a 1-pair smoke archive vs a full 600-pair archive
    BEFORE running the full evaluate stage, so the bridge can honestly report
    that a smoke archive validates the inflate half but cannot satisfy the
    contest's 600-sample assertion. Decode authority is the inflate stage; this
    is a cheap pre-check.
    """
    try:
        _member, payload = _payload_member_bytes(archive)
        from tac.substrates.hi_nerv.inflate import build_model_from_archive

        _arc, cfg, _model = build_model_from_archive(payload, device="cpu")
        return int(cfg.num_pairs)
    except Exception:  # pre-check only; never gates the real run
        return None


def _emit_runtime_dir(runtime_dir: Path) -> dict[str, Any]:
    """Emit the canonical x/0.bin-accepting HiNeRV inflate runtime (self-contained).

    Vendors the hi_nerv substrate package + the shared inflate_runtime helpers
    into ``<runtime_dir>/src/tac/...`` so the inflate path is hermetic (Catalog
    #295 self-containment; no PYTHONPATH shim depending on the dev repo). The
    inflate.py reads member ``x`` OR ``0.bin`` (the canonical hi_nerv_export
    contract), so a backend-only archive whose member is ``x`` decodes.

    NO scorer-network imports are vendored (strict-scorer-rule). numpy/torch/
    brotli portable (no MLX dep).
    """
    runtime_dir.mkdir(parents=True, exist_ok=True)
    # Vendored package tree.
    src_tac = runtime_dir / "src" / "tac"
    hi_nerv_pkg = src_tac / "substrates" / "hi_nerv"
    shared_pkg = src_tac / "substrates" / "_shared"
    hi_nerv_pkg.mkdir(parents=True, exist_ok=True)
    shared_pkg.mkdir(parents=True, exist_ok=True)
    for init in (
        src_tac / "__init__.py",
        src_tac / "substrates" / "__init__.py",
        hi_nerv_pkg / "__init__.py",
        shared_pkg / "__init__.py",
    ):
        init.write_text("", encoding="utf-8")

    repo_hi_nerv = REPO_ROOT / "src" / "tac" / "substrates" / "hi_nerv"
    repo_shared = REPO_ROOT / "src" / "tac" / "substrates" / "_shared"
    # The decode-only modules the inflate path imports (NO scorer modules).
    hi_nerv_modules = (
        "architecture.py",
        "archive.py",
        "inflate.py",
        "target_region_actions.py",
    )
    shared_modules = (
        "inflate_runtime.py",
        "decoder_state_codec.py",
        "int_stream_codec.py",
    )
    vendored: list[str] = []
    for name in hi_nerv_modules:
        src = repo_hi_nerv / name
        if src.is_file():
            shutil.copy2(src, hi_nerv_pkg / name)
            vendored.append(f"hi_nerv/{name}")
    for name in shared_modules:
        src = repo_shared / name
        if src.is_file():
            shutil.copy2(src, shared_pkg / name)
            vendored.append(f"_shared/{name}")

    inflate_py = runtime_dir / "inflate.py"
    inflate_sh = runtime_dir / "inflate.sh"
    inflate_py.write_text(_HI_NERV_INFLATE_PY, encoding="utf-8")
    inflate_sh.write_text(_HI_NERV_INFLATE_SH, encoding="utf-8")
    inflate_sh.chmod(0o755)
    return {
        "runtime_dir": runtime_dir.as_posix(),
        "inflate_sh": inflate_sh.as_posix(),
        "inflate_py": inflate_py.as_posix(),
        "vendored_modules": sorted(vendored),
        "scorer_imports_vendored": False,
        "member_parser_accepts": ["x", "0.bin"],
    }


def _device_axis_caveat(device: str) -> dict[str, Any]:
    """Describe the device's evidence axis BEFORE the run (mirrors the contest_auth_eval contract)."""
    is_darwin = platform.system() == "Darwin"
    is_linux_x86 = platform.system() == "Linux" and platform.machine().lower() in {
        "x86_64",
        "amd64",
    }
    if device == "cpu" and is_darwin:
        return {
            "axis_tag": "[macOS-CPU advisory]",
            "authoritative": False,
            "score_claim": False,
            "promotable": False,
            "frontier_claim": False,
            "rationale": (
                "Local macOS CPU is NEVER authoritative. The authoritative CPU "
                "axis is Linux x86_64 ([contest-CPU], the public leaderboard "
                "axis). This run only proves the pipeline runs + measures runtime."
            ),
        }
    if device == "cpu" and is_linux_x86:
        return {
            "axis_tag": "[contest-CPU]",
            "authoritative": True,
            "score_claim": True,
            "promotable": False,
            "frontier_claim": False,
            "rationale": "Linux x86_64 CPU = public leaderboard axis (contest-CPU).",
        }
    if device == "cuda":
        return {
            "axis_tag": "[contest-CUDA-if-1to1-hardware]",
            "authoritative": True,
            "score_claim": True,
            "promotable": False,
            "frontier_claim": False,
            "rationale": (
                "CUDA on NVIDIA T4/equivalent (Linux x86_64) = contest-CUDA "
                "promotion axis. Verify the GPU is 1:1 contest-compliant."
            ),
        }
    return {
        "axis_tag": "[diagnostic]",
        "authoritative": False,
        "score_claim": False,
        "promotable": False,
        "frontier_claim": False,
        "rationale": f"device={device} is diagnostic-only.",
    }


def _dual_axis_authoritative_recipe(
    archive: Path,
    *,
    inflate_sh: str,
    video_names_file: Path,
    out_row: Path | None,
) -> dict[str, Any]:
    """The EXACT paired Linux-x86_64-CPU + T4-CUDA recipe (NOT fired here).

    Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-
    COMPLIANT HARDWARE". Both axes run on the SAME archive bytes so CPU/CUDA
    drift is measured, not inferred. Fire only on a real 600-pair B1 archive
    after operator approval.
    """
    arc = archive.as_posix()
    vnf = video_names_file.as_posix()
    cpu_json = "<linux_cpu_workdir>/hi_nerv_backend_only_exact_eval_contest_cpu.json"
    cuda_json = "<t4_cuda_workdir>/hi_nerv_backend_only_exact_eval_contest_cuda.json"
    base = (
        ".venv/bin/python -u experiments/contest_auth_eval.py "
        f"--archive {arc} --inflate-sh {inflate_sh} "
        f"--video-names-file {vnf} --upstream-dir upstream"
    )
    return {
        "note": (
            "AUTHORITATIVE dual-axis B2 result. Run BOTH on the SAME archive "
            "bytes. Paid + deferred until a real 600-pair B1 archive lands + "
            "operator approval. Local macOS-CPU is advisory only and is NOT a "
            "substitute for either axis."
        ),
        "axis_1_contest_cpu_linux_x86_64": {
            "where": "Modal CPU container / Vast.ai CPU / Lightning CPU Studio (Linux x86_64; ~$0.06/hr)",
            "command": (
                f"{base} --device cpu --json-out {cpu_json} "
                "--work-dir <linux_cpu_workdir>"
            ),
            "expected_axis_tag": "[contest-CPU]",
            "note": "Public leaderboard axis. ~60-120 min for 600 samples on contest-CI-class CPU.",
        },
        "axis_2_contest_cuda_t4": {
            "where": "Modal T4 / Vast.ai 4090 / equivalent NVIDIA (Linux x86_64; ~$0.30-0.60)",
            "command": (
                f"{base} --device cuda --json-out {cuda_json} "
                "--work-dir <t4_cuda_workdir> --expected-runtime-tree-sha256 <sha-from-cpu-run>"
            ),
            "expected_axis_tag": "[contest-CUDA]",
            "note": "Promotion/ranking axis. The contest bot scores CUDA on T4.",
        },
        "step_3_compliance_gate": {
            "where": "before any judge-facing/public submission",
            "command": (
                ".venv/bin/python scripts/pre_submission_compliance_check.py "
                f"--contest-final --strict --archive {arc} "
                "--expected-archive-sha256 <sha256> "
                "--expected-archive-size-bytes <bytes> "
                "--auth-eval-json <contest_cuda_or_cpu_json>"
            ),
            "note": (
                "Canonical upload-surface gate (Operator gates must be wired and "
                "used). Validates archive identity, inflate.sh, ZIP safety, "
                "auth-eval identity, runtime custody, dispatch-claim linkage."
            ),
        },
        "lane_claim": (
            "tools/claim_lane_dispatch.py claim --lane-id <lane> --platform modal "
            "--agent <agent> --instance-job-id <job> --status dispatched_dual_axis"
        ),
        "out_row_hint": (out_row.as_posix() if out_row is not None else None),
    }


def _run_contest_auth_eval(
    *,
    archive: Path,
    inflate_sh: Path,
    device: str,
    video_names_file: Path,
    work_dir: Path,
    json_out: Path,
    inflate_timeout: int,
    evaluate_timeout: int,
) -> tuple[int, str, str]:
    """Invoke the canonical contest_auth_eval.py (archive.zip -> inflate.sh -> evaluate.py -> score)."""
    cmd = [
        ".venv/bin/python",
        "-u",
        "experiments/contest_auth_eval.py",
        "--archive",
        archive.as_posix(),
        "--inflate-sh",
        inflate_sh.as_posix(),
        "--upstream-dir",
        "upstream",
        "--video-names-file",
        video_names_file.as_posix(),
        "--device",
        device,
        "--work-dir",
        work_dir.as_posix(),
        "--json-out",
        json_out.as_posix(),
        "--keep-work-dir",
    ]
    print(f"[b2-bridge] running canonical pipeline: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT.as_posix(),
        capture_output=True,
        text=True,
        timeout=inflate_timeout + evaluate_timeout + 300,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _inflate_only_validation(
    *,
    archive: Path,
    inflate_sh: Path,
    video_names_file: Path,
    inflated_dir: Path,
    timeout: int,
    inflate_python: str | None = None,
) -> dict[str, Any]:
    """Run ONLY the real inflate.sh (the PR106 dep-closure-bug-class half).

    For a 1-pair smoke archive, evaluate.py cannot satisfy the 600-sample
    assertion, so the canonical full pipeline aborts at the evaluate stage. This
    function exercises the inflate HALF end-to-end on the REAL archive bytes
    through the REAL contest inflate.sh contract (extract -> inflate.sh
    archive_dir output_dir file_list -> .raw), proving the dep closure
    (torch/brotli/numpy) + decode + raw-write path works. NO score claim.
    """
    archive_dir = inflated_dir.parent / "extracted"
    if archive_dir.exists():
        shutil.rmtree(archive_dir, ignore_errors=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        # Validate member names before extraction (no path escapes / absolute).
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            p = Path(name)
            if p.is_absolute() or any(part in {"", ".."} for part in p.parts):
                raise SystemExit(f"unsafe archive member: {name!r}")
        zf.extractall(archive_dir)  # RAW_EXTRACTALL_OK: per-member absolute/dotdot validation loop directly above
    if inflated_dir.exists():
        shutil.rmtree(inflated_dir, ignore_errors=True)
    inflated_dir.mkdir(parents=True, exist_ok=True)

    # The inflate.sh contract honours ${PYTHON:-python3}. The contest runtime
    # dep-closure must provide brotli + torch + numpy; on a clean machine bare
    # `python3` may lack them (the PR106 ModuleNotFoundError bug class). For
    # local advisory validation we point PYTHON at a venv that satisfies the
    # closure (the contest-faithful equivalent of a self-contained runtime venv).
    import os

    env = {**os.environ}
    if inflate_python:
        env["PYTHON"] = inflate_python
    t0 = time.monotonic()
    proc = run_in_process_group(
        ["bash", inflate_sh.as_posix(), archive_dir.as_posix(), inflated_dir.as_posix(), video_names_file.as_posix()],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    elapsed = time.monotonic() - t0
    bases = [
        line.strip()
        for line in video_names_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw_outputs: list[dict[str, Any]] = []
    bytes_per_frame = 874 * 1164 * 3
    for base in bases:
        raw_path = inflated_dir / f"{Path(base).with_suffix('.raw')}"
        if not raw_path.is_file():
            # inflate_runtime writes <base>.raw (full name incl. .mkv -> .mkv.raw
            # is wrong; .with_suffix replaces .mkv with .raw). Try the literal too.
            alt = inflated_dir / f"{base}.raw"
            raw_path = alt if alt.is_file() else raw_path
        info: dict[str, Any] = {"base": base, "raw_path": raw_path.as_posix()}
        if raw_path.is_file():
            sz = raw_path.stat().st_size
            info["raw_bytes"] = sz
            info["raw_frames"] = sz // bytes_per_frame if sz % bytes_per_frame == 0 else None
            info["raw_bytes_frame_aligned"] = (sz % bytes_per_frame == 0)
            info["raw_sha256_first16"] = _sha256(raw_path)[:16]
        else:
            info["raw_bytes"] = None
        raw_outputs.append(info)
    return {
        "inflate_returncode": proc.returncode,
        "inflate_elapsed_seconds": round(elapsed, 3),
        "inflate_ok": proc.returncode == 0 and all(o.get("raw_bytes") for o in raw_outputs),
        "raw_outputs": raw_outputs,
        "bytes_per_frame": bytes_per_frame,
        "inflate_stdout_tail": proc.stdout[-2048:],
        "inflate_stderr_tail": proc.stderr[-2048:],
    }


def _certify_and_clean(inflated_root: Path, *, keep: bool) -> dict[str, Any]:
    """Disk hygiene: certify rebuildable inflated frames, then delete (per CLAUDE.md).

    Inflated .raw frames are deterministically rebuildable from the archive via
    inflate.sh, so they are certified rebuildable scratch and deleted by default
    (the certify-or-block rule: the archive sha + inflate.sh ARE the rebuild
    record). ``--keep-work-dir`` preserves them for debugging.
    """
    total_bytes = 0
    n_files = 0
    if inflated_root.exists():
        for p in inflated_root.rglob("*"):
            if p.is_file():
                total_bytes += p.stat().st_size
                n_files += 1
    record = {
        "inflated_root": inflated_root.as_posix(),
        "inflated_bytes": total_bytes,
        "inflated_files": n_files,
        "rebuildable": True,
        "rebuild_record": "deterministic via inflate.sh on the archive bytes (sha recorded)",
    }
    if not keep and inflated_root.exists():
        shutil.rmtree(inflated_root, ignore_errors=True)
        record["cleaned"] = True
    else:
        record["cleaned"] = False
    return record


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--archive", help="HiNeRV backend-only archive.zip (member x or 0.bin)")
    ap.add_argument(
        "--replay-row",
        default=None,
        help="Optional hi_nerv_backend_only_exact_replay.v1 JSON from "
        "build_hi_nerv_backend_only_archive.py (carries region-win proxy + byte deltas).",
    )
    ap.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Eval device. LOCAL macOS cpu => [macOS-CPU advisory] (default). "
        "Authoritative requires Linux-x86_64 cpu (contest-CPU) AND T4 cuda.",
    )
    ap.add_argument(
        "--video-names-file",
        default=DEFAULT_VIDEO_NAMES,
        help=f"Contest video names list (default: {DEFAULT_VIDEO_NAMES}).",
    )
    ap.add_argument(
        "--work-root",
        default=None,
        help="Scratch root for runtime + extract + inflated frames. Prefer SSD "
        "(/Volumes/VertigoDataTier/pact/...) or .omx/tmp; NEVER /tmp. Default: .omx/tmp.",
    )
    ap.add_argument(
        "--out-row",
        default=None,
        help="Durable hi_nerv_backend_only_exact_eval.v1 JSON output path.",
    )
    ap.add_argument("--inflate-timeout", type=int, default=CONTEST_BUDGET_SECONDS)
    ap.add_argument("--evaluate-timeout", type=int, default=CONTEST_BUDGET_SECONDS)
    ap.add_argument(
        "--inflate-python",
        default=".venv/bin/python",
        help="Python the inflate.sh subprocess uses (sets ${PYTHON}). Must "
        "satisfy the contest dep-closure (brotli+torch+numpy). Default "
        ".venv/bin/python is the contest-faithful local equivalent of a "
        "self-contained runtime venv. The full-pipeline path always runs the "
        "canonical contest_auth_eval.py under the repo venv regardless.",
    )
    ap.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep the inflated frames + work dir (debugging). Default deletes "
        "the rebuildable inflated frames after eval (disk hygiene).",
    )
    ap.add_argument(
        "--print-dual-axis-recipe",
        action="store_true",
        help="Print the AUTHORITATIVE paid dual-axis (Linux-CPU + T4-CUDA) "
        "recipe and exit WITHOUT running anything.",
    )
    args = ap.parse_args()

    video_names_file = (REPO_ROOT / args.video_names_file).resolve() if not Path(
        args.video_names_file
    ).is_absolute() else Path(args.video_names_file)

    # --print-dual-axis-recipe is a no-run informational mode.
    if args.print_dual_axis_recipe:
        archive = Path(args.archive).expanduser().resolve() if args.archive else Path("<B1_600pair_archive.zip>")
        recipe = _dual_axis_authoritative_recipe(
            archive,
            inflate_sh="<runtime_dir>/inflate.sh (emitted by this tool's --work-root run)",
            video_names_file=video_names_file,
            out_row=(Path(args.out_row).resolve() if args.out_row else None),
        )
        print(json.dumps({"schema": SCHEMA, "mode": "dual_axis_recipe_only", "dual_axis_authoritative_recipe": recipe}, indent=2))
        return 0

    if not args.archive:
        raise SystemExit("--archive is required (unless --print-dual-axis-recipe)")
    archive = Path(args.archive).expanduser().resolve()
    if not archive.is_file():
        raise SystemExit(f"--archive not found: {archive}")
    if not video_names_file.is_file():
        raise SystemExit(f"--video-names-file not found: {video_names_file}")

    # Scratch root: SSD/.omx, never the system /tmp tree (CLAUDE.md disk-hygiene).
    # Reject ONLY the system ephemeral temp roots (/tmp, /private/tmp, /var/tmp);
    # the canonical repo-local ``.omx/tmp`` scratch is explicitly allowed even
    # though its path contains the substring "/tmp/".
    if args.work_root:
        work_root = Path(args.work_root).expanduser().resolve()
    else:
        work_root = (REPO_ROOT / ".omx" / "tmp" / f"b2_bridge_{_utc_compact()}").resolve()
    _wr = work_root.as_posix()
    _system_tmp_roots = ("/tmp", "/private/tmp", "/var/tmp")
    if any(_wr == r or _wr.startswith(r + "/") for r in _system_tmp_roots):
        raise SystemExit(
            "refusing system /tmp work root (CLAUDE.md disk-hygiene). "
            "Use SSD (/Volumes/VertigoDataTier/pact/...) or repo .omx/tmp."
        )
    work_root.mkdir(parents=True, exist_ok=True)

    replay_row: dict[str, Any] | None = None
    if args.replay_row:
        rp = Path(args.replay_row).expanduser().resolve()
        if rp.is_file():
            replay_row = json.loads(rp.read_text(encoding="utf-8"))

    archive_sha256 = _sha256(archive)
    archive_size_bytes = archive.stat().st_size
    member_names = _archive_member_names(archive)
    num_pairs = _probe_num_pairs(archive)
    axis = _device_axis_caveat(args.device)

    # Emit the canonical x/0.bin-accepting HiNeRV runtime dir.
    runtime_dir = work_root / "hi_nerv_runtime"
    runtime_info = _emit_runtime_dir(runtime_dir)
    inflate_sh = Path(runtime_info["inflate_sh"])

    out_row_path = (
        Path(args.out_row).expanduser().resolve()
        if args.out_row
        else work_root / "hi_nerv_backend_only_exact_eval.json"
    )

    row: dict[str, Any] = {
        "schema": SCHEMA,
        "family": "hinerv",
        "generated_at_utc": _utc_now(),
        "tool": "tools/run_hi_nerv_backend_only_b2_exact_eval.py",
        "archive": archive.as_posix(),
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_size_bytes,
        "archive_member_names": member_names,
        "archive_num_pairs": num_pairs,
        "device": args.device,
        "axis_tag": axis["axis_tag"],
        "contest_budget_seconds": CONTEST_BUDGET_SECONDS,
        "runtime": runtime_info,
        "video_names_file": video_names_file.as_posix(),
        # False-authority / axis markers (Catalog #341 / #323).
        "authority": "planning_control_false_authority"
        if not axis["authoritative"]
        else "contest_axis_authoritative_pending_compliance_gate",
        "score_claim": False,  # set True only when a 600-pair contest axis score lands
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "frontier_claim": False,
        "human_visual_fidelity_objective": False,
        "device_axis_caveat": axis,
    }
    if replay_row is not None:
        # Carry forward the build-tool's EXACT byte deltas + region-win proxy.
        row["backend_only_replay_row"] = {
            "schema": replay_row.get("schema"),
            "input_archive_sha256": replay_row.get("input_archive_sha256"),
            "backend_only_archive_sha256": replay_row.get("backend_only_archive_sha256"),
            "zip_bytes_delta": replay_row.get("zip_bytes_delta"),
            "exact_rate_delta_score": replay_row.get("exact_rate_delta_score"),
            "region_win_original": replay_row.get("region_win_original"),
            "region_win_backend_only": replay_row.get("region_win_backend_only"),
            "region_win_delta": replay_row.get("region_win_delta"),
            "estimated_seg_delta_score_advisory": replay_row.get(
                "estimated_seg_delta_score_advisory"
            ),
            "verdict": replay_row.get("verdict"),
        }
        # Custody cross-check: the replay row's backend_only sha must match.
        bo_sha = replay_row.get("backend_only_archive_sha256")
        row["replay_row_sha_matches_archive"] = bo_sha == archive_sha256
        if bo_sha is not None and bo_sha != archive_sha256:
            row["replay_row_sha_mismatch_blocker"] = (
                "replay_row backend_only_archive_sha256 != supplied --archive sha256; "
                "the proxy may not correspond to these exact bytes"
            )

    # Always include the authoritative dual-axis recipe so the JSON is turnkey.
    row["dual_axis_authoritative_recipe"] = _dual_axis_authoritative_recipe(
        archive,
        inflate_sh=inflate_sh.as_posix(),
        video_names_file=video_names_file,
        out_row=out_row_path,
    )

    is_full_contest = num_pairs == CONTEST_NUM_EVAL_SAMPLES
    inflated_root = work_root / "inflated"

    if is_full_contest:
        # Full 600-pair archive: run the canonical end-to-end pipeline
        # (archive.zip -> inflate.sh -> evaluate.py -> score) via contest_auth_eval.py.
        cae_json = work_root / "contest_auth_eval.json"
        cae_work = work_root / "cae_work"
        rc, stdout, stderr = _run_contest_auth_eval(
            archive=archive,
            inflate_sh=inflate_sh,
            device=args.device,
            video_names_file=video_names_file,
            work_dir=cae_work,
            json_out=cae_json,
            inflate_timeout=args.inflate_timeout,
            evaluate_timeout=args.evaluate_timeout,
        )
        row["pipeline_mode"] = "full_contest_auth_eval"
        row["contest_auth_eval_returncode"] = rc
        row["contest_auth_eval_stdout_tail"] = stdout[-3072:]
        if rc != 0:
            row["contest_auth_eval_stderr_tail"] = stderr[-3072:]
        if cae_json.is_file():
            cae = json.loads(cae_json.read_text(encoding="utf-8"))
            # Lift the exact score + axis fields from the canonical result.
            for k in (
                "canonical_score",
                "final_score",
                "avg_posenet_dist",
                "avg_segnet_dist",
                "rate_unscaled",
                "score_seg_contribution",
                "score_pose_contribution",
                "score_rate_contribution",
                "n_samples",
                "evidence_grade",
                "lane_tag",
                "score_axis",
                "evidence_semantics",
                "inflate_elapsed_seconds",
                "evaluate_elapsed_seconds",
                "contest_auth_eval_elapsed_seconds",
                "score_claim",
                "score_claim_valid",
                "promotion_eligible",
                "rank_or_kill_eligible",
            ):
                if k in cae:
                    row[f"eval_{k}" if k in {"score_claim", "promotion_eligible", "rank_or_kill_eligible"} else k] = cae[k]
            # runtime-tree custody
            prov = cae.get("provenance", {})
            row["runtime_tree_sha256"] = prov.get("inflate_runtime_tree_sha256") or prov.get(
                "runtime_tree_sha256"
            )
            row["platform_system"] = prov.get("platform_system")
            row["platform_machine"] = prov.get("platform_machine")
            # 30-min budget check on the inflate stage (contest budget is inflate-side).
            infl = cae.get("inflate_elapsed_seconds")
            row["inflate_within_30min_budget"] = (
                None if infl is None else bool(infl <= CONTEST_BUDGET_SECONDS)
            )
            row["pipeline_works"] = True
            # A real contest-axis score only counts when the evidence contract says so.
            row["score_claim"] = bool(cae.get("score_claim")) and bool(
                cae.get("n_samples") == CONTEST_NUM_EVAL_SAMPLES
            )
            row["verdict"] = (
                "exact_score_landed"
                if row["score_claim"]
                else "evaluated_advisory_or_diagnostic_axis"
            )
        else:
            row["pipeline_works"] = False
            row["verdict"] = "contest_auth_eval_failed_no_json"
            row["blocker"] = (
                "contest_auth_eval.py produced no JSON; inspect stderr_tail "
                "(common: PR106 dep-closure — missing brotli/torch in runtime)."
            )
    else:
        # Smoke archive (num_pairs != 600): cannot satisfy the 600-sample
        # contest assertion. Validate the INFLATE half end-to-end on the REAL
        # archive bytes (proves dep closure + decode + raw-write).
        row["pipeline_mode"] = "inflate_only_validation_smoke_archive"
        inflate_python = args.inflate_python
        if inflate_python and not Path(inflate_python).is_absolute():
            # Join to repo root WITHOUT resolving symlinks: `.venv/bin/python`
            # is a symlink to the bare uv interpreter whose site-packages LACK
            # torch; following it would re-trigger the dep-closure failure. Keep
            # the venv launcher path so its site-packages (torch/brotli) load.
            cand = REPO_ROOT / inflate_python
            if cand.exists():
                inflate_python = cand.as_posix()
        row["inflate_python"] = inflate_python
        row["dep_closure_contract"] = (
            "inflate.sh honours ${PYTHON:-python3}; runtime must provide "
            "brotli+torch+numpy. Bare homebrew python3 lacks brotli (PR106 bug "
            "class). On the contest machine the self-contained runtime venv "
            "provides the closure; here PYTHON is pointed at a venv that does."
        )
        infl = _inflate_only_validation(
            archive=archive,
            inflate_sh=inflate_sh,
            video_names_file=video_names_file,
            inflated_dir=inflated_root,
            timeout=args.inflate_timeout,
            inflate_python=inflate_python,
        )
        row.update(
            {
                "inflate_returncode": infl["inflate_returncode"],
                "inflate_elapsed_seconds": infl["inflate_elapsed_seconds"],
                "inflate_ok": infl["inflate_ok"],
                "raw_outputs": infl["raw_outputs"],
                "bytes_per_frame": infl["bytes_per_frame"],
                "inflate_stdout_tail": infl["inflate_stdout_tail"],
                "inflate_stderr_tail": infl["inflate_stderr_tail"],
            }
        )
        row["inflate_within_30min_budget"] = bool(
            infl["inflate_elapsed_seconds"] <= CONTEST_BUDGET_SECONDS
        )
        row["pipeline_works"] = bool(infl["inflate_ok"])
        row["score_claim"] = False
        row["verdict"] = (
            "pipeline_inflate_ok_evaluate_requires_600_pairs"
            if infl["inflate_ok"]
            else "inflate_failed_dep_closure_or_decode"
        )
        row["smoke_note"] = (
            f"archive has num_pairs={num_pairs}; contest requires "
            f"{CONTEST_NUM_EVAL_SAMPLES}. The inflate stage ran end-to-end on "
            "REAL bytes via the contest inflate.sh contract (dep closure + decode "
            "+ raw-write proven). NO score claim — a frontier score needs a "
            "600-pair archive on a contest-compliant axis."
        )

    # Disk hygiene: certify + clean the rebuildable inflated frames.
    row["disk_hygiene"] = _certify_and_clean(inflated_root, keep=args.keep_work_dir)
    if is_full_contest and not args.keep_work_dir:
        # contest_auth_eval kept its work dir (--keep-work-dir); clean its inflated frames too.
        cae_inflated = work_root / "cae_work" / "inflated"
        row["disk_hygiene_cae"] = _certify_and_clean(cae_inflated, keep=False)

    out_row_path.parent.mkdir(parents=True, exist_ok=True)
    out_row_path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(row, indent=2, sort_keys=True))
    print(f"\n[b2-bridge] wrote {out_row_path}", flush=True)
    print(f"[b2-bridge] verdict={row['verdict']} axis={row['axis_tag']} "
          f"pipeline_works={row.get('pipeline_works')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
