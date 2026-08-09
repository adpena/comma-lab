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
    # PRECISE MOUNT, derived from the builder's actual reads (verified at source):
    #   root/public_test_video_names.txt · root/videos/ · sys.path.insert(root) then
    #   `from frame_utils import ...` + `from modules import ...` (modules.py:17-18
    #   resolves models/{segnet,posenet}.safetensors relative to its own HERE).
    # Mounting all of upstream/ would upload 938 MB -- 812 MB of which is .venv (566 MB)
    # and .git (158 MB) that the builder never touches. These five entries are ~126 MB.
    .add_local_dir(  # MODAL_MANUAL_MOUNT_OK:the scored video, the builder's only data input
        "upstream/videos",
        remote_path=str(REMOTE_UPSTREAM / "videos"),
        copy=True,
    )
    .add_local_dir(  # MODAL_MANUAL_MOUNT_OK:frozen scorer weights, resolved by modules.py:17-18
        "upstream/models",
        remote_path=str(REMOTE_UPSTREAM / "models"),
        copy=True,
    )
    .add_local_file("upstream/frame_utils.py", remote_path=str(REMOTE_UPSTREAM / "frame_utils.py"))
    .add_local_file("upstream/modules.py", remote_path=str(REMOTE_UPSTREAM / "modules.py"))
    .add_local_file(
        "upstream/public_test_video_names.txt",
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
    av_cache = REMOTE_WORK / "gt_cache_av.pt"
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

    def _run(argv: list[str], label: str) -> dict:
        started = time.time()
        proc = subprocess.run(
            [sys.executable, str(REMOTE_BUILDER), *argv],
            capture_output=True,
            text=True,
            cwd=str(REMOTE_WORK),
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
    if leg_dali["returncode"] != 0:
        return {"status": "DALI_LEG_FAILED", "env": env_probe, "legs": [leg_av, leg_dali]}

    av_report_obj = json.loads(av_report.read_text())
    dali_report_obj = json.loads(dali_report.read_text())

    dali_bytes = dali_cache.read_bytes()
    av_bytes = av_cache.read_bytes()
    return {
        "status": "OK",
        "env": env_probe,
        "legs": [leg_av, leg_dali],
        "av_report": av_report_obj,
        # THE HEADLINE lives in dali_report_obj["reference_seg_disagreement"] --
        # PR130's own field, argmax mismatch fraction, comparable to 2.8609e-4.
        "dali_vs_av_report": dali_report_obj,
        "dali_cache_xz": lzma.compress(dali_bytes, preset=6),
        "dali_cache_sha256": _sha256_bytes(dali_bytes),
        "dali_cache_bytes": len(dali_bytes),
        "av_cache_xz": lzma.compress(av_bytes, preset=6),
        "av_cache_sha256": _sha256_bytes(av_bytes),
        "av_cache_bytes": len(av_bytes),
    }


@app.local_entrypoint()
def main(out_dir: str = "") -> None:
    """Fire the remote job and land its artifacts durably (SSD tier for the caches)."""
    target = Path(out_dir) if out_dir else Path(
        "/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809"
    )
    target.mkdir(parents=True, exist_ok=True)

    result = build_and_diff.remote()
    status = result.get("status")

    summary = {k: v for k, v in result.items() if not k.endswith("_xz")}
    (target / "result_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if status == "OK":
        for name in ("dali", "av"):
            blob = result[f"{name}_cache_xz"]
            path = target / f"gt_cache_{name}.pt.xz"
            path.write_bytes(blob)
            print(json.dumps({
                "wrote": str(path),
                "xz_bytes": len(blob),
                "raw_bytes": result[f"{name}_cache_bytes"],
                "raw_sha256": result[f"{name}_cache_sha256"],
            }), flush=True)
        headline = result["dali_vs_av_report"].get("reference_seg_disagreement")
        print(json.dumps({
            "HEADLINE_reference_seg_disagreement": headline,
            "reference_pose_mse": result["dali_vs_av_report"].get("reference_pose_mse"),
            "reference_pose_max_abs": result["dali_vs_av_report"].get("reference_pose_max_abs"),
            "vs_pr130_d_seg_2p8609e-4_ratio": (
                (headline / 2.8609e-4) if isinstance(headline, (int, float)) else None
            ),
            "vs_chroma_siting_sensitivity_2p2791e-4_ratio": (
                (headline / 2.2790696885850695e-4) if isinstance(headline, (int, float)) else None
            ),
        }, indent=2), flush=True)
    else:
        print(json.dumps(summary, indent=2), flush=True)
    print(json.dumps({"summary_written": str(target / "result_summary.json")}), flush=True)
