# SPDX-License-Identifier: MIT
"""Modal CPU dispatch: exact-score-gated latent click-polish SEARCH + SAME-CONTAINER
exact eval on our PR110-lineage frontier archive (task #399 phase-2).

HONEST NAMING (NO-FAKE #6): a search/polish, not a solver. Borrowed-substrate
accounting (NO-FAKE #7): mechanism = PR128 click-polish [external unverified];
substrate = OURS. DEFENSIVE BANK, not innovation.

SAME-CONTAINER DETERMINISM MANDATE: torch bicubic LSBs differ across CPU
microarchitectures, so click SELECTION and the authoritative ``upstream/evaluate.py``
row run in ONE container on ONE microarch (recorded in the result row). GT targets
are built in-container via the upstream AVVideoDataset (yuv420_to_rgb — the
evaluator's own GT decode) + the frozen scorers.

Resumable (P0): the accepted-clicks ledger + banked candidate live on a Modal
Volume, committed after every accepted round; a re-dispatch with the same run-id
resumes from the ledger.

Usage (local):
  # n8 validation (proves image + GT + search + evaluate.py row + custody):
  .venv/bin/modal run experiments/modal_click_polish_cpu.py \
      --n-pairs 8 --max-rounds 1 --sweep-deltas 1,-1 --run-id n8_validation --detach
  # bounded n600 (after n8 green):
  .venv/bin/modal run experiments/modal_click_polish_cpu.py \
      --n-pairs 600 --max-rounds 2 --sweep-deltas 1,-1 \
      --wall-clock-cap-s 16200 --run-id n600_r1 --detach
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "clickpolish-pr110-cpu"
VOLUME_NAME = "clickpolish-pr110-vol"
REMOTE_REPO = Path("/workspace/pact")
REMOTE_VOL = Path("/vol")
SUBMISSION_REL = "experiments/results/pr110_payload_entropy_recode_20260610/submission_dir"
LANE_ID = "lane_clickpolish_pr110_frontier_20260710"
EXPECTED_RAW_BYTES = 1200 * 874 * 1164 * 3  # 3,662,409,600 (n600 full inflate)
FRONTIER_SHA256 = "b46897267ded1e73a581dad57143f6c1cd181b515479d4efce40e4536d50e73e"

app = modal.App(APP_NAME, include_source=False)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Image: proven experiments/modal_auth_eval_cpu.py base (linux/x86_64 debian_slim,
# CPU torch wheel) with constriction PINNED (coded bytes depend on the exact coder).
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ca-certificates", "curl", "ffmpeg", "libglib2.0-0", "libgl1", "unzip")
    .pip_install(
        "torch==2.5.1",
        "torchvision",
        "safetensors",
        "einops",
        "segmentation-models-pytorch",
        "av",
        "tqdm",
        "timm",
        "numpy<2.0",
        "Pillow",
        "constriction==0.4.2",
        "brotli>=1.0",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
)

eval_image = (
    base_image
    .env({"PYTHONPATH": f"{REMOTE_REPO}:{REMOTE_REPO}/upstream"})
    .add_local_dir(  # MODAL_MANUAL_MOUNT_OK:narrow bespoke click-polish dispatcher
        "upstream",
        remote_path=str(REMOTE_REPO / "upstream"),
        copy=True,
        ignore=["**/__pycache__/**", "**/.git/**", "**/*.pyc"],
    )
    .add_local_dir(  # MODAL_MANUAL_MOUNT_OK:narrow bespoke click-polish dispatcher
        SUBMISSION_REL,
        remote_path=str(REMOTE_REPO / SUBMISSION_REL),
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_python_source("modal_click_polish_cpu", "tac")  # MODAL_ENTRYPOINT_SELF_MOUNT_OK:include_source=False requires explicit re-add
)


def _microarch() -> dict:
    import platform

    info = {"machine": platform.machine(), "platform": platform.platform()}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("flags"):
                flags = set(line.split(":", 1)[1].split())
                info["isa"] = {k: (k in flags) for k in ("avx2", "avx512f", "fma")}
                break
    except OSError:
        pass
    return info


def _build_gt_targets(n_pairs: int, scorer) -> tuple:
    """Decode GT pairs via upstream AVVideoDataset (yuv420_to_rgb — the evaluator's
    decode) and extract lstars (SegNet argmax) + gt_poses (PoseNet pose[:6]) with
    the frozen scorers. Batches of 16, pair order == evaluator order."""
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, str(REMOTE_REPO / "upstream"))
    from frame_utils import AVVideoDataset  # type: ignore[import-not-found]

    names = [
        line.strip()
        for line in (REMOTE_REPO / "upstream/public_test_video_names.txt")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    ds = AVVideoDataset(
        names,
        data_dir=REMOTE_REPO / "upstream/videos",
        batch_size=16,
        device=torch.device("cpu"),
        num_threads=2,
        seed=1234,
    )
    ds.prepare_data()
    lstars_parts, pose_parts = [], []
    collected = 0
    with torch.inference_mode():
        for _path, _idx, batch in ds:  # (B, 2, H, W, 3) uint8
            take = min(batch.shape[0], n_pairs - collected)
            b = batch[:take].float().permute(0, 1, 4, 2, 3)  # b t c h w
            seg = scorer.segnet(scorer.segnet.preprocess_input(b)).argmax(dim=1)
            pose = scorer.posenet(scorer.posenet.preprocess_input(b))["pose"][:, :6]
            lstars_parts.append(seg.cpu().numpy().astype(np.int64))
            pose_parts.append(pose.cpu().numpy().astype(np.float64))
            collected += take
            if collected >= n_pairs:
                break
    lstars = np.concatenate(lstars_parts, axis=0)[:n_pairs]
    gt_poses = np.concatenate(pose_parts, axis=0)[:n_pairs]
    return lstars, gt_poses


def _run_exact_eval(candidate_archive: bytes, out_dir: Path) -> dict:
    """SAME-CONTAINER authoritative row: inflate.sh + upstream/evaluate.py --device
    cpu on the exact candidate bytes. Fails closed on raw-size/custody violations."""
    import hashlib
    import shutil
    import subprocess
    import zipfile

    sub_src = REMOTE_REPO / SUBMISSION_REL
    eval_dir = out_dir / "eval_submission"
    if eval_dir.exists():
        shutil.rmtree(eval_dir)
    eval_dir.mkdir(parents=True)
    # runtime tree = OUR frontier inflate surfaces, verbatim. encoder/ is REQUIRED:
    # src/fec10_hybrid_decoder.py re-exports the FECa selector decoder from
    # ../encoder/build_pr101_frame_exploit_selector_packet_fec10_hybrid.py
    # (single-source-of-truth pattern) — omitting it broke the first n8 validation.
    shutil.copy2(sub_src / "inflate.py", eval_dir / "inflate.py")
    shutil.copy2(sub_src / "inflate.sh", eval_dir / "inflate.sh")
    shutil.copytree(sub_src / "src", eval_dir / "src",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(sub_src / "encoder", eval_dir / "encoder",
                    ignore=shutil.ignore_patterns("__pycache__"))
    # atomic archive write
    tmp = eval_dir / "archive.zip.tmp"
    tmp.write_bytes(candidate_archive)
    tmp.rename(eval_dir / "archive.zip")

    # storage preflight (fail closed): raw + slack
    free = shutil.disk_usage(str(eval_dir)).free
    need = EXPECTED_RAW_BYTES + 2 * len(candidate_archive) + (1 << 30)
    if free < need:
        raise RuntimeError(f"storage preflight FAILED: free={free} < need={need}")

    archive_dir = eval_dir / "archive"
    archive_dir.mkdir()
    with zipfile.ZipFile(eval_dir / "archive.zip") as z:
        z.extract("x", archive_dir)

    names_file = REMOTE_REPO / "upstream/public_test_video_names.txt"
    inflated = eval_dir / "inflated"
    subprocess.run(
        ["bash", str(eval_dir / "inflate.sh"), str(archive_dir), str(inflated), str(names_file)],
        check=True,
        cwd=str(eval_dir),
    )
    raw_path = inflated / "0.raw"
    raw_bytes = raw_path.stat().st_size
    if raw_bytes != EXPECTED_RAW_BYTES:
        raise RuntimeError(
            f"raw-output custody FAILED: {raw_bytes} != {EXPECTED_RAW_BYTES} "
            "(evaluator truncation = NO-FAKE failure)"
        )

    report = eval_dir / "report_cpu.txt"
    subprocess.run(
        [
            "python", str(REMOTE_REPO / "upstream/evaluate.py"),
            "--submission-dir", str(eval_dir),
            "--uncompressed-dir", str(REMOTE_REPO / "upstream/videos"),
            "--video-names-file", str(names_file),
            "--device", "cpu",
            "--batch-size", "16",
            "--num-threads", "2",
            "--seed", "1234",
            "--report", str(report),
        ],
        check=True,
        cwd=str(REMOTE_REPO / "upstream"),
    )
    report_text = report.read_text()
    # free the 3.4 GiB raw once scored (candidate + report retained)
    shutil.rmtree(inflated, ignore_errors=True)
    return {
        "report_text": report_text,
        "raw_bytes": raw_bytes,
        "archive_bytes": len(candidate_archive),
        "archive_sha256": hashlib.sha256(candidate_archive).hexdigest(),
    }


@app.function(
    image=eval_image,
    cpu=8.0,
    memory=16 * 1024,
    timeout=30_600,  # 8.5 h hard cap (worst-case cost ≈ $4.3 at ~$0.5/hr list)
    volumes={str(REMOTE_VOL): vol},
)
def click_polish_run(
    n_pairs: int = 8,
    max_rounds: int = 1,
    sweep_deltas: str = "1,-1",
    wall_clock_cap_s: float = 0.0,
    run_id: str = "n8_validation",
    run_eval: bool = True,
) -> dict:
    import hashlib
    import os
    import time

    import torch

    t_start = time.time()
    os.chdir(str(REMOTE_REPO))
    torch.set_num_threads(8)
    torch.manual_seed(1234)

    from tac import click_polish as cp

    out_dir = REMOTE_VOL / "clickpolish" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {
        "run_id": run_id,
        "n_pairs": n_pairs,
        "max_rounds": max_rounds,
        "sweep_deltas": sweep_deltas,
        "microarch": _microarch(),
        "torch": torch.__version__,
        "axis_note": "selection+eval SAME container (Linux x86_64, CPU-torch)",
    }

    # 1) parse packet + verify round-trip IN-CONTAINER (constriction/microarch guard)
    pkt = cp.FrozenPacket.parse(str(REMOTE_REPO / SUBMISSION_REL / "archive.zip"),
                                str(REMOTE_REPO / SUBMISSION_REL))
    rt = pkt.verify_roundtrip()
    result["roundtrip"] = {k: rt[k] for k in
                           ("archive_byte_exact", "archive_bytes", "archive_sha256")}
    if not rt["archive_byte_exact"] or rt["archive_sha256"] != FRONTIER_SHA256:
        result["status"] = "FAILED_roundtrip_in_container"
        return result

    # 2) scorers + in-container GT targets (evaluator-aligned decode)
    t = time.time()
    scorer = cp.Scorer(upstream_dir=str(REMOTE_REPO / "upstream"), device="cpu")
    lstars, gt_poses = _build_gt_targets(n_pairs, scorer)
    result["gt_build_s"] = round(time.time() - t, 1)
    result["gt_shapes"] = [list(lstars.shape), list(gt_poses.shape)]

    # 3) the search (resumable ledger on the Volume; committed per accepted round)
    renderer = cp.Renderer(pkt, device="cpu")
    deltas = tuple(int(x) for x in sweep_deltas.split(","))
    search = cp.ClickPolishSearch(
        packet=pkt, renderer=renderer, scorer=scorer,
        gt_lstars=lstars, gt_poses=gt_poses,
        out_dir=str(out_dir),
        axis_tag="[contest-CPU candidate-selection in-container]",
        max_rounds=max_rounds, sweep_deltas=deltas,
        wall_clock_cap_s=wall_clock_cap_s,
    )
    _orig_append = search._append_ledger

    def _append_and_commit(row):
        _orig_append(row)
        vol.commit()  # durability: resume survives preemption

    search._append_ledger = _append_and_commit
    t = time.time()
    search_result = search.run()
    result["search_s"] = round(time.time() - t, 1)
    search_result.pop("d_seg_per_pair", None)
    search_result.pop("d_pose_per_pair", None)
    result["search"] = search_result

    candidate = pkt.repack_archive_bytes(search.Q)
    result["candidate_sha256"] = hashlib.sha256(candidate).hexdigest()
    result["candidate_bytes"] = len(candidate)
    (out_dir / "candidate_archive.zip").write_bytes(candidate)  # PAYLOAD_WRITE_ORDER_OK:the result binds the completed archive's exact bytes and digest
    vol.commit()

    # 4) SAME-CONTAINER exact eval (the authoritative [contest-CPU] row)
    if run_eval:
        t = time.time()
        eval_out = _run_exact_eval(candidate, Path("/root/clickpolish_eval"))
        result["eval_s"] = round(time.time() - t, 1)
        result["exact_eval"] = eval_out
        (out_dir / "report_cpu.txt").write_text(eval_out["report_text"])

    result["ledger_text"] = (out_dir / "accepted_clicks_ledger.jsonl").read_text() \
        if (out_dir / "accepted_clicks_ledger.jsonl").exists() else ""
    result["candidate_archive_b"] = candidate  # returned for local custody
    result["total_s"] = round(time.time() - t_start, 1)
    result["status"] = "OK"
    (out_dir / "result_meta.json").write_text(
        json.dumps({k: v for k, v in result.items() if k != "candidate_archive_b"},
                   indent=2, default=str)
    )
    vol.commit()
    return result


@app.local_entrypoint()
def main(
    n_pairs: int = 8,
    max_rounds: int = 1,
    sweep_deltas: str = "1,-1",
    wall_clock_cap_s: float = 0.0,
    run_id: str = "n8_validation",
    run_eval: bool = True,
):
    """Spawn the run, record the call_id (HARVEST OR LOSE), print harvest command."""
    # #513 SINGLE-FLIGHT pre-spawn guard (operator binding 2026-07-15): refuse
    # when ANY live Modal work exists (call-id ledger / claims file / live
    # `modal app list`); operator-override escape via env
    # TAC_MODAL_SINGLE_FLIGHT_FORCE_RATIONALE (quote it in the claim notes).
    from tac.deploy.modal.single_flight import assert_modal_single_flight
    assert_modal_single_flight(label=run_id, lane_id=LANE_ID)
    fc = click_polish_run.spawn(
        n_pairs=n_pairs, max_rounds=max_rounds, sweep_deltas=sweep_deltas,
        wall_clock_cap_s=wall_clock_cap_s, run_id=run_id, run_eval=run_eval,
    )
    call_id = fc.object_id
    print(f"SPAWNED call_id={call_id} run_id={run_id} n_pairs={n_pairs}")

    from tac.deploy.modal.call_id_ledger import register_dispatched_call_id

    register_dispatched_call_id(
        call_id=call_id,
        lane_id=LANE_ID,
        label=f"clickpolish_{run_id}",
        gpu="cpu",
        expected_axis="cpu",
        expected_cost_usd=1.0 if n_pairs <= 16 else 4.0,
        max_seconds=30_600,
        agent="claude",
        subagent_id="clickpolish-build",
    )
    out_dir = Path("experiments/results/clickpolish_pr110_20260710")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "call_id": call_id,
        "run_id": run_id,
        "n_pairs": n_pairs,
        "max_rounds": max_rounds,
        "sweep_deltas": sweep_deltas,
        "wall_clock_cap_s": wall_clock_cap_s,
        "lane_id": LANE_ID,
        "app": APP_NAME,
        "volume": VOLUME_NAME,
    }
    (out_dir / f"modal_metadata_{run_id}.json").write_text(json.dumps(meta, indent=2))
    print(
        "HARVEST: .venv/bin/python -c \"import modal; "
        f"r=modal.FunctionCall.from_id('{call_id}').get(timeout=30); print(r['status'])\""
    )
