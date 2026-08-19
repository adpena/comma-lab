# SPDX-License-Identifier: MIT
"""#906 — the ONE qualifying Modal job: DALI-vs-AV GT-cache diff on ONE host.

WHY (operator binding 2026-08-09 compute split): Modal/CUDA is for what physically
CANNOT run on the Mac, and short, and it must buy a DURABLE ASSET rather than a
service.  DALI/nvdec decode requires CUDA and has no local equivalent.  This job
is that case and only that case.

WHAT IT RESOLVES: ``upstream/evaluate.py:33-42`` selects the dataset BY DEVICE --
CUDA -> ``DaliVideoDataset`` (nvdec), else ``AVVideoDataset`` (PyAV).  The contest
leaderboard runs CUDA, so the AUTHORITY's GT labels come from DALI while every
local GT cache we own comes from AV.  ``frame_utils.yuv420_to_rgb`` is *intended*
to match nvdec, but that is an intent, not proven bit-identity.

WHY IT IS NOW A PREREQUISITE (MEASURED 2026-08-09, afa34a0860 / 38e08900c3):
``tools/measure_chroma_siting_argmax_sensitivity.py`` measured the frozen SegNet
argmax sensitivity to the chroma-siting convention (centered, which upstream uses,
vs left/co-sited, which H.264/HEVC and therefore nvdec use) at **2.2790696885850695e-4
pooled over n=120 stratified pairs = 79.66% of PR130's ENTIRE d_seg (2.8609e-4)**,
with 120/120 pairs disagreeing and a byte-identity positive control.  So IF the two
decode paths differ by a siting convention, the label delta is ~80% of the bar's
whole seg term.  This job measures whether they actually differ.

HOW (no reimplementation): it runs PR130's OWN
``code/build_gt_cache_official.py`` -- twice, in ONE container:
  1. ``--dataset av``   -> av_cache.pt
  2. ``--dataset dali --reference-cache av_cache.pt``
Their builder's own ``--reference-cache`` branch emits ``reference_seg_disagreement``
(argmax mismatch fraction, directly comparable to 2.86e-4) plus per-dim pose MSE.
Running BOTH on the SAME host is the point: the diff then isolates the DECODER,
not decoder+platform.  Local macOS-AV vs this Modal-AV is a separate free third
leg, computed afterwards on the returned artifact.

BORROWED SUBSTRATE (NO-FAKE #7 honesty half): the builder script is PR130's, run
unmodified from the read-only intake clone at
``/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo``.  It is
MOUNTED, never edited.  The measurement design, the mount/return plumbing, and the
sensitivity result that motivates it are ours.

DURABLE ASSET: the DALI cache (the authority-side GT labels) comes back xz-compressed
and lands on the SSD tier for every subsequent LOCAL training run.  Nothing about this
job trains anything; all training stays on Metal per the compute split.

AXIS: the returned disagreement is an exact property of the two decode paths under
the frozen scorer on contest CUDA.  It is NOT a contest score and no score claim is
made here.
"""
from __future__ import annotations

import hashlib
import json
import lzma
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

APP_NAME = "comma-dali-av-gt-diff"
REMOTE_REPO = Path("/workspace/pact")
REMOTE_UPSTREAM = REMOTE_REPO / "upstream"
REMOTE_BUILDER = REMOTE_REPO / "pr130_build_gt_cache_official.py"
REMOTE_WORK = Path("/workspace/gtdiff")

PR130_REPO = Path("/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo")
PR130_BUILDER = PR130_REPO / "code" / "build_gt_cache_official.py"

app = modal.App(APP_NAME, include_source=False)

# Catalog #244 canonical contest-CUDA env block. These literals MUST equal
# tac.deploy.modal.runtime.{DALI_DISABLE_NVML_VALUE, PYTORCH_CUDA_ALLOC_CONF_VALUE,
# CUBLAS_WORKSPACE_CONFIG_VALUE} (src/tac/deploy/modal/runtime.py:52-62). They are
# literals here rather than an import because this module is itself uploaded and
# imported INSIDE the container, where `tac` is not mounted -- so `main()` below
# fail-closes on a LOCAL parity assertion against the canonical module before any
# dispatch fires. Single source of truth preserved; remote import kept safe.
#
# WHY THIS EXISTS (measured 2026-08-09, run dali_av_gt_diff_20260809T012723Z):
# omitting DALI_DISABLE_NVML killed the DALI leg at 12.7s with
#   `nvml error (999): A nvml internal driver error occurred`
# inside `nvidia.dali.fn.experimental.inputs.video` -- the exact operator and
# exact error CLAUDE.md records as the D1 incident anchor (6 occurrences in 24h
# before Catalog #244 closed it for lane DRIVERS). This Modal app is a surface
# that gate does not scan, so the cure had to be applied by recall, and was not.
DALI_DISABLE_NVML_VALUE = "1"
PYTORCH_CUDA_ALLOC_CONF_VALUE = "expandable_segments:True"
CUBLAS_WORKSPACE_CONFIG_VALUE = ":4096:8"
# CWD-INDEPENDENT LOCAL MOUNTS. add_local_dir/add_local_file resolve RELATIVE
# PATHS against the process CWD and do NOT validate at image-construction time --
# they fail only when `modal run` uploads. So a bare "upstream/videos" silently
# constructs fine and dies at dispatch under any launcher that sets a different
# cwd (measured 2026-08-09 run r3, launched with --cwd experiments: died on
# FileNotFoundError for experiments/upstream/public_test_video_names.txt).
# Anchoring to __file__ makes the mount independent of who launches us, and the
# assert below turns a silent construction into a loud LOCAL refusal.
REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_UPSTREAM = REPO_ROOT / "upstream"

_LOCAL_MOUNT_SOURCES = (
    LOCAL_UPSTREAM / "videos",
    LOCAL_UPSTREAM / "models",
    LOCAL_UPSTREAM / "frame_utils.py",
    LOCAL_UPSTREAM / "modules.py",
    LOCAL_UPSTREAM / "public_test_video_names.txt",
    PR130_BUILDER,
)
# LOCAL-ONLY. This module is ALSO imported inside the container (Modal hydrates
# the function by importing it), where these host paths cannot exist by
# construction -- an unguarded module-scope check is a guaranteed false positive
# that kills the run AFTER the mounts uploaded fine (measured 2026-08-09 run r4).
# modal.is_local() returns False only inside a running Modal Function.
if modal.is_local():
    _missing_local = [str(p) for p in _LOCAL_MOUNT_SOURCES if not p.exists()]
    if _missing_local:  # loud at IMPORT, not silent until the Modal upload
        raise RuntimeError(
            f"#906 local mount source(s) absent (SSD unmounted? wrong checkout?): {_missing_local}"
        )

# Provenance: the contest authority tier is the 600-sample eval (upstream
# evaluate.py over public_test_video_names.txt), and our own frozen GT cache is
# gt_n600.npz -- 600 pairs. Used ONLY as a coverage floor: this job must measure
# the whole population, never a prefix.
EXPECTED_PAIRS = 600

CONTEST_CUDA_ENV = {
    "DALI_DISABLE_NVML": DALI_DISABLE_NVML_VALUE,
    "PYTORCH_CUDA_ALLOC_CONF": PYTORCH_CUDA_ALLOC_CONF_VALUE,
    "CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE_CONFIG_VALUE,
}

# Image mirrors the PROVEN experiments/modal_auth_eval.py base (r9m dispatched
# through it successfully) -- same torch pin, same nvidia-dali-cuda120 from the
# NVIDIA index.  Only the mounts differ: this job needs upstream/ and the builder.
gt_diff_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ca-certificates", "curl", "ffmpeg", "libglib2.0-0", "libgl1", "xz-utils")
    .pip_install(
        "torch==2.5.1",
        "torchvision",
        "safetensors",
        "einops",  # REQUIRED: upstream/modules.py:2 imports it. Do NOT drop.
        "segmentation-models-pytorch",
        "av",
        "nvidia-dali-cuda120==1.52.0",
        "timm",
        "numpy<2.0",
        "Pillow",
        extra_index_url="https://pypi.nvidia.com",
    )
    # Catalog #244 -- DALI/nvdec refuses without DALI_DISABLE_NVML.
    # MUST precede every add_local_* call: `.env()` is a BUILD STEP, and Modal
    # refuses build steps after local-file adds ("An image tried to run a build
    # step after using image.add_local_* to include local files") unless every
    # such add uses copy=True. Measured 2026-08-09 run r2, which died here in
    # 28s at $0. Order is load-bearing; do not move this below the mounts.
    .env(CONTEST_CUDA_ENV)
    # PRECISE MOUNT, derived from the builder's actual reads (verified at source):
    #   root/public_test_video_names.txt · root/videos/ · sys.path.insert(root) then
    #   `from frame_utils import ...` + `from modules import ...` (modules.py:17-18
    #   resolves models/{segnet,posenet}.safetensors relative to its own HERE).
    # Mounting all of upstream/ would upload 938 MB -- 812 MB of which is .venv (566 MB)
    # and .git (158 MB) that the builder never touches. These five entries are ~126 MB.
    .add_local_dir(  # MODAL_MANUAL_MOUNT_OK:the scored video, the builder's only data input
        str(LOCAL_UPSTREAM / "videos"),
        remote_path=str(REMOTE_UPSTREAM / "videos"),
        copy=True,
    )
    .add_local_dir(  # MODAL_MANUAL_MOUNT_OK:frozen scorer weights, resolved by modules.py:17-18
        str(LOCAL_UPSTREAM / "models"),
        remote_path=str(REMOTE_UPSTREAM / "models"),
        copy=True,
    )
    .add_local_file(str(LOCAL_UPSTREAM / "frame_utils.py"), remote_path=str(REMOTE_UPSTREAM / "frame_utils.py"))
    .add_local_file(str(LOCAL_UPSTREAM / "modules.py"), remote_path=str(REMOTE_UPSTREAM / "modules.py"))
    .add_local_file(
        str(LOCAL_UPSTREAM / "public_test_video_names.txt"),
        remote_path=str(REMOTE_UPSTREAM / "public_test_video_names.txt"),
    )
    .add_local_file(  # MODAL_MANUAL_MOUNT_OK:PR130 builder, mounted read-only, never edited
        str(PR130_BUILDER),
        remote_path=str(REMOTE_BUILDER),
    )
    .add_local_python_source("modal_dali_av_gt_cache_diff")  # MODAL_ENTRYPOINT_SELF_MOUNT_OK:include_source=False
)


def _sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


@app.function(image=gt_diff_image, gpu="T4", timeout=3600)
def build_and_diff() -> dict:
    """Run PR130's builder at --dataset av then --dataset dali --reference-cache av."""
    import torch

    REMOTE_WORK.mkdir(parents=True, exist_ok=True)
    av_cache = REMOTE_WORK / "gt_cache_av.pt"  # GT_LINEAGE_OK: this is the #906 PRODUCER that builds BOTH caches in order to diff them -- naming the AV lineage is the job, not an undeclared objective
    dali_cache = REMOTE_WORK / "gt_cache_dali.pt"
    av_report = REMOTE_WORK / "report_av.json"
    dali_report = REMOTE_WORK / "report_dali_vs_av.json"

    env_probe = {
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if not env_probe["cuda_available"]:
        raise RuntimeError("#906 job requires CUDA (that is the entire reason it is on Modal)")

    # Fail fast and loudly on a mount typo, rather than inside the builder's import.
    required = [
        REMOTE_UPSTREAM / "public_test_video_names.txt",
        REMOTE_UPSTREAM / "videos" / "0.mkv",
        REMOTE_UPSTREAM / "frame_utils.py",
        REMOTE_UPSTREAM / "modules.py",
        REMOTE_UPSTREAM / "models" / "segnet.safetensors",
        REMOTE_UPSTREAM / "models" / "posenet.safetensors",
        REMOTE_BUILDER,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"#906 mount incomplete: {missing}")

    def _pack(path: Path) -> dict:
        """xz a built cache so a leg's product survives a LATER leg's failure."""
        raw = path.read_bytes()
        return {
            "xz": lzma.compress(raw, preset=6),
            "sha256": _sha256_bytes(raw),
            "bytes": len(raw),
        }

    def _run(argv: list[str], label: str) -> dict:
        started = time.time()
        proc = subprocess.run(
            [sys.executable, str(REMOTE_BUILDER), *argv],
            capture_output=True,
            text=True,
            cwd=str(REMOTE_WORK),
            env={**os.environ, **CONTEST_CUDA_ENV},  # Catalog #244, explicit
        )
        return {
            "label": label,
            "returncode": proc.returncode,
            "elapsed_seconds": round(time.time() - started, 1),
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }

    leg_av = _run(
        [
            "--challenge-root", str(REMOTE_UPSTREAM),
            "--dataset", "av",
            "--out", str(av_cache),
            "--report", str(av_report),
        ],
        "av",
    )
    if leg_av["returncode"] != 0:
        return {"status": "AV_LEG_FAILED", "env": env_probe, "legs": [leg_av]}

    leg_dali = _run(
        [
            "--challenge-root", str(REMOTE_UPSTREAM),
            "--dataset", "dali",
            "--reference-cache", str(av_cache),
            "--out", str(dali_cache),
            "--report", str(dali_report),
        ],
        "dali_vs_av",
    )
    # HARVEST OR LOSE: the AV leg's cache is a durable asset in its own right
    # (it is the free third leg vs our local macOS-AV cache). The first run of
    # this job returned early on DALI failure and threw away a successful
    # 100.4s AV build. Never again -- ship whatever legs succeeded.
    av_packed = _pack(av_cache)
    av_report_obj = json.loads(av_report.read_text())
    partial = {
        "env": env_probe,
        "legs": [leg_av, leg_dali],
        "av_report": av_report_obj,
        "av_pairs": av_report_obj.get("pairs"),
        "av_cache_xz": av_packed["xz"],
        "av_cache_sha256": av_packed["sha256"],
        "av_cache_bytes": av_packed["bytes"],
    }
    if leg_dali["returncode"] != 0:
        return {"status": "DALI_LEG_FAILED_AV_HARVESTED", **partial}

    dali_report_obj = json.loads(dali_report.read_text())

    # COVERAGE ASSERTION (report the DENOMINATOR, never a bare rate).
    # The builder's `!=` compare broadcasts, so a SILENTLY TRUNCATED pair of
    # caches (both 16 pairs, say) yields a clean-looking disagreement rate over
    # a 16-pair PREFIX. Measured law (m88/m96): a prefix of this population is a
    # DIFFERENT population -- video order is temporally correlated, and the bias
    # SIGN INVERTS by axis (pose prefixes 2.5-4.2x HARDER, seg ~0.96x easier).
    # A prefix answer here would be actively misleading, not merely weaker.
    # NOTE: this REPORTS, it does not raise. Both caches are already built and
    # are durable assets; raising here would destroy them on the way out --
    # the exact harvest-or-lose failure that cost run 1 its AV leg.
    av_pairs = av_report_obj.get("pairs")
    dali_pairs = dali_report_obj.get("pairs")
    coverage_ok = (av_pairs == dali_pairs == EXPECTED_PAIRS)

    dali_packed = _pack(dali_cache)
    return {
        "status": "OK" if coverage_ok else "OK_BUT_COVERAGE_SUSPECT",
        "coverage": {
            "av_pairs": av_pairs,
            "dali_pairs": dali_pairs,
            "expected_pairs": EXPECTED_PAIRS,
            "ok": coverage_ok,
        },
        **partial,
        # THE HEADLINE lives in dali_report_obj["reference_seg_disagreement"] --
        # PR130's own field, argmax mismatch fraction, comparable to 2.8609e-4.
        #
        # BINDING POSITIVE CONTROL: `reference_pose_mse`. The two legs differ ONLY
        # in decoder (dataset_device gates DALI-vs-AV; the scorer is CUDA in both).
        # If BOTH reference_seg_disagreement == 0.0 AND reference_pose_mse == 0.0,
        # that is INDISTINGUISHABLE from "the decoder never actually varied" (a
        # vacuous pass -- silent-instrument genus). Only a NONZERO pose delta
        # licenses reading a zero seg delta as "the decoders agree on the argmax."
        "dali_vs_av_report": dali_report_obj,
        "dali_pairs": dali_pairs,
        "positive_control_pose_mse": dali_report_obj.get("reference_pose_mse"),
        "dali_cache_xz": dali_packed["xz"],
        "dali_cache_sha256": dali_packed["sha256"],
        "dali_cache_bytes": dali_packed["bytes"],
    }


@app.local_entrypoint()
def main(out_dir: str = "") -> None:
    """Fire the remote job and land its artifacts durably (SSD tier for the caches)."""
    # FAIL-CLOSED LOCAL PARITY: our literals must equal the canonical Catalog #244
    # values. Runs on the Mac (where `tac` imports); refuses BEFORE any spend.
    from tac.deploy.modal import runtime as _canon

    _expected = {
        "DALI_DISABLE_NVML": _canon.DALI_DISABLE_NVML_VALUE,
        "PYTORCH_CUDA_ALLOC_CONF": _canon.PYTORCH_CUDA_ALLOC_CONF_VALUE,
        "CUBLAS_WORKSPACE_CONFIG": _canon.CUBLAS_WORKSPACE_CONFIG_VALUE,
    }
    if CONTEST_CUDA_ENV != _expected:
        raise AssertionError(
            "Catalog #244 env drift vs tac.deploy.modal.runtime: "
            f"local={CONTEST_CUDA_ENV} canonical={_expected}"
        )

    target = Path(out_dir) if out_dir else Path(
        "/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809"
    )
    # CERTIFY-OR-BLOCK: if the SSD tier is UNMOUNTED, /Volumes/VertigoDataTier does
    # not exist and mkdir(parents=True) would silently CREATE it on the boot disk --
    # landing the caches on the wrong tier with no error. Refuse instead. Checked
    # BEFORE .remote() so a storage fault costs $0 rather than a completed job.
    for mount in (Path("/Volumes/VertigoDataTier"),):
        if str(target).startswith(str(mount)) and not mount.is_dir():
            raise RuntimeError(
                f"SSD tier {mount} is not mounted; refusing to write cache artifacts "
                "to the boot disk. Mount it or pass --out-dir."
            )
    target.mkdir(parents=True, exist_ok=True)

    result = build_and_diff.remote()
    status = result.get("status")

    summary = {k: v for k, v in result.items() if not k.endswith("_xz")}
    (target / "result_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Land EVERY cache the run actually produced, whatever the status.
    for name in ("dali", "av"):
        blob = result.get(f"{name}_cache_xz")
        if blob is not None:
            path = target / f"gt_cache_{name}.pt.xz"
            path.write_bytes(blob)
            print(json.dumps({
                "wrote": str(path),
                "xz_bytes": len(blob),
                "raw_bytes": result[f"{name}_cache_bytes"],
                "raw_sha256": result[f"{name}_cache_sha256"],
            }), flush=True)
    if status in ("OK", "OK_BUT_COVERAGE_SUSPECT"):
        report = result["dali_vs_av_report"]
        headline = report.get("reference_seg_disagreement")
        pose_mse = report.get("reference_pose_mse")
        print(json.dumps({
            "status": status,
            "HEADLINE_reference_seg_disagreement": headline,
            "coverage": result.get("coverage"),
            "pairs_DENOMINATOR": report.get("pairs"),
            "reference_pose_mse": pose_mse,
            # POSITIVE CONTROL verdict. A 0.0/0.0 pair proves nothing about the
            # decoders -- it is what a NON-VARYING decoder would also produce.
            "positive_control": (
                "VACUOUS_BOTH_ZERO_decoder_may_not_have_varied"
                if (headline == 0.0 and pose_mse == 0.0)
                else "LIVE_decoder_varied"
            ),
            "reference_pose_max_abs": report.get("reference_pose_max_abs"),
            "vs_pr130_d_seg_2p8609e-4_ratio": (
                (headline / 2.8609e-4) if isinstance(headline, (int, float)) else None
            ),
            "vs_chroma_siting_sensitivity_2p2791e-4_ratio": (
                (headline / 2.2790696885850695e-4) if isinstance(headline, (int, float)) else None
            ),
        }, indent=2), flush=True)
    else:
        # Partial or failed: the legs table names exactly which one died and why.
        print(json.dumps(summary, indent=2), flush=True)
    print(json.dumps({"summary_written": str(target / "result_summary.json")}), flush=True)
