# SPDX-License-Identifier: MIT
"""Modal CPU (Linux x86_64) dispatcher for the OWED #288 n600 OT head-offset gate.

Runs ``experiments/probe_laguerre_logit_offset_sweep.py`` at ALL 600 pairs on the
mod32cap ep650 EMA-BEST snapshot (the READ-ONLY checkpoint #288's subset gate used;
sha 6dd28a6e...), realized-through-R on the frozen CPU SegNet. The 3-arm gate
(no_offset / menon / ot_newton) at n600 confirms-or-overturns the #288 subset
verdict (n48/n24: no_offset 0.00273 < menon 0.00293 < ot_newton 0.00487) at the ONLY
admissible evidence scale (CLAUDE.md "allergic to non-n600-scale").

Why Modal CPU: #288 could not reach n600 locally because a background task dies at
~5 min; the caching pass alone is ~50 min at n600. Modal runs the function
server-side (immune to the local lifetime cap), on Linux x86_64 (the contest-CPU
substrate family). NO GPU, NO training, #205 untouched. The witness ckpt is loaded
READ-ONLY; the offset is folded byte-free into a COPY of ``out_sdf.bias`` inside the
probe — the snapshot on disk is never mutated.

Slim GT cache: the offset probe consumes ONLY ``gt.lstars`` (it renders frame1 from
the witness, not from GT), so we upload a 2.1 MB lstars-only cache (real
frozen-SegNet argmax, byte-identical to the canonical gt_n600 lstars; content
sha bf1d0e5c...) instead of the 4.8 GB full cache.

AUTHORITY: realized-through-R, frozen CPU SegNet. [contest-CPU advisory] class
measurement (NOT a score claim; promotion_eligible=false) until byte-closed exact
eval. NO FAKE: real ckpt, real frozen-SegNet argmax GT, real scorer.

Dispatch (detached; returns a call_id in seconds, remote runs ~60-70 min):
    PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach \
        experiments/modal_ot_offset_n600_gate.py

Harvest by call_id after completion, through the CANONICAL harvester. It mirrors
the terminal outcome into ``.omx/state/modal_call_id_ledger.jsonl`` (Catalog
#330); a hand-rolled ``FunctionCall.get`` poll retrieves the payload but leaves
the call_id stuck at ``dispatched`` forever, so it is not an admissible harvest::

    .venv/bin/python tools/harvest_modal_calls.py --from-ledger \
        --call-id <call_id> --execute
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

APP_NAME = "comma-ot-offset-n600"
REMOTE_REPO = Path("/workspace/pact")
REMOTE_PYTHONPATH = (
    f"{REMOTE_REPO / 'src'}:{REMOTE_REPO / 'upstream'}:"
    f"{REMOTE_REPO / 'experiments'}:{REMOTE_REPO}"
)
RESULT_VOLUME_NAME = "comma-ot-offset-n600-results"
RESULT_VOLUME_ROOT = Path("/results_vol")

# Local artifacts (validated locally: n6 reproduced the #288 subset ordering;
# full n2 executed with mlx BLOCKED => numpy compute path is x86-safe).
CKPT_LOCAL = (
    "experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z/"
    "levelset_witness_ema_BEST.npz"
)
SLIM_CACHE_LOCAL = (
    "experiments/results/ot_offset_n600_modal_20260709/gt_n600_lstars_slim.npz"
)
CKPT_REMOTE = str(REMOTE_REPO / "ckpt_mod32cap_ep650_ema_BEST.npz")
SLIM_CACHE_REMOTE = str(REMOTE_REPO / "gt_n600_lstars_slim.npz")

app = modal.App(APP_NAME, include_source=False)
result_vol = modal.Volume.from_name(RESULT_VOLUME_NAME, create_if_missing=True)


def _ignore_heavy(path: str) -> bool:
    """Mount the experiments *.py the probe imports, exclude the 17 GB results tree
    and other generated/large dirs so the image build stays fast + deterministic."""
    p = str(path)
    parts = set(Path(p).parts)
    if "results" in parts or "precomputed" in parts or "precomputed_local" in parts:
        return True
    if ".git" in parts or "__pycache__" in parts:
        return True
    return False


def _ignore_src(path: str) -> bool:
    """Mount src/ (so ``tac`` resolves at the repo-relative path
    ``/workspace/pact/src/tac`` — seg_core.REPO_ROOT = parents[3] then points at
    ``/workspace/pact/upstream/models``, not ``/upstream``). Exclude caches/tests
    to keep the build fast."""
    parts = set(Path(str(path)).parts)
    if "__pycache__" in parts or "tests" in parts or ".git" in parts:
        return True
    if str(path).endswith(".pyc"):
        return True
    return False


# CPU-only image: torch CPU wheel + the frozen-scorer stack (segmentation-models,
# timm, einops, safetensors). NO nvidia-dali (CUDA-only), NO av/ffmpeg (this probe
# does not decode video — it consumes cached lstars + renders synthetic frames).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libglib2.0-0", "libgl1")
    .pip_install(
        "torch==2.5.1",
        "torchvision",
        "safetensors",
        "einops",
        "segmentation-models-pytorch",
        "timm",
        "scipy",
        "numpy<2.0",
        "Pillow",
        "pydantic>=2.0",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .env({"PYTHONPATH": REMOTE_PYTHONPATH})
    .add_local_dir(  # MODAL_MANUAL_MOUNT_OK:CPU-only probe image (not the training stack); src at the repo-relative path so seg_core.REPO_ROOT resolves
        "src", remote_path=str(REMOTE_REPO / "src"), copy=True, ignore=_ignore_src,
    )
    .add_local_dir(  # MODAL_MANUAL_MOUNT_OK:frozen upstream scorer files the probe consumes (modules.py, frame_utils.py, models/*.safetensors)
        "upstream", remote_path=str(REMOTE_REPO / "upstream"), copy=True,
    )
    .add_local_dir(  # MODAL_MANUAL_MOUNT_OK:probe + train_witness base modules only, results/ excluded via _ignore_heavy
        "experiments", remote_path=str(REMOTE_REPO / "experiments"), copy=True,
        ignore=_ignore_heavy,
    )
    .add_local_file(CKPT_LOCAL, remote_path=CKPT_REMOTE, copy=True)  # MODAL_MANUAL_MOUNT_OK:pinned witness checkpoint, the probe's model input
    .add_local_file(SLIM_CACHE_LOCAL, remote_path=SLIM_CACHE_REMOTE, copy=True)  # MODAL_MANUAL_MOUNT_OK:pinned slim lstars cache, the probe's data input
)


@app.function(
    image=image,
    cpu=2.0,  # workload is single-threaded numpy caching + torch SegNet batch; 8
    # cores idled at ~$2.4 projected (> $2 cap). cpu=2 + 16 GiB keeps this < $1.
    memory=16384,  # peak RSS ~7-8 GiB (phis 2.4 + texs 1.4 + lstars 0.9 + torch)
    timeout=10800,  # 3 h ceiling (expect ~60-70 min); crash => cheap re-run
    volumes={str(RESULT_VOLUME_ROOT): result_vol},
)
def run_ot_offset_n600(num_pairs: int = 600, so_iters: int = 3, ot_tau: float = 1.0) -> dict:
    import platform
    import time

    import torch

    t0 = time.time()
    # Fail closed on mis-spec: this MUST be CPU on Linux x86_64 (contest-CPU family).
    assert not torch.cuda.is_available(), "GPU detected — this is a CPU-only gate"
    machine = platform.machine()
    assert machine in ("x86_64", "AMD64"), f"expected x86_64, got {machine}"

    for p in (
        str(REMOTE_REPO / "experiments"),
        str(REMOTE_REPO / "src"),
        str(REMOTE_REPO / "upstream"),
    ):
        if p not in sys.path:
            sys.path.insert(0, p)

    RESULT_VOLUME_ROOT.mkdir(parents=True, exist_ok=True)
    out_json = str(RESULT_VOLUME_ROOT / "ot_offset_n600_gate.json")

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "probe_mod", str(REMOTE_REPO / "experiments" / "probe_laguerre_logit_offset_sweep.py")
    )
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

    sys.argv = [
        "probe_laguerre_logit_offset_sweep.py",
        "--ckpt", CKPT_REMOTE,
        "--gt-cache", SLIM_CACHE_REMOTE,
        "--num-pairs", str(num_pairs),
        "--so-iters", str(so_iters),
        "--ot-tau", str(ot_tau),
        "--out-json", out_json,
        # default grid=-0.4,-0.2,0.0,0.2,0.4 focus=0,1,3 (faithful to #288)
    ]
    probe.main()

    result = json.loads(Path(out_json).read_text())
    result["_modal"] = {
        "machine": machine,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "elapsed_s": round(time.time() - t0, 1),
        "num_pairs": num_pairs,
        "ot_tau": ot_tau,
        "ckpt_remote": CKPT_REMOTE,
    }
    Path(out_json).write_text(json.dumps(result, indent=2))
    result_vol.commit()
    return result


@app.local_entrypoint()
def main(num_pairs: int = 600):
    """Spawn detached, save the call_id (HARVEST-OR-LOSE), exit in seconds."""
    from tac.deploy.modal.auth_eval import function_call_id

    # #513 SINGLE-FLIGHT pre-spawn guard (operator binding 2026-07-15): refuse
    # when ANY live Modal work exists (call-id ledger / claims file / live
    # `modal app list`); operator-override escape via env
    # TAC_MODAL_SINGLE_FLIGHT_FORCE_RATIONALE (quote it in the claim notes).
    from tac.deploy.modal.single_flight import assert_modal_single_flight
    assert_modal_single_flight(label="ot_offset_n600", lane_id="")
    fn_call = run_ot_offset_n600.spawn(num_pairs=num_pairs)
    call_id = function_call_id(fn_call)
    sentinel = Path("experiments/results/ot_offset_n600_modal_20260709")
    sentinel.mkdir(parents=True, exist_ok=True)
    (sentinel / "modal_call_id.txt").write_text(call_id + "\n")
    print(f"DISPATCHED via .spawn() call_id={call_id}")
    print(f"call_id saved: {sentinel / 'modal_call_id.txt'}")
    # Catalog #330: point the operator at the CANONICAL harvester, which mirrors
    # the terminal outcome into the call-id ledger. The raw
    # ``FunctionCall.get`` poll this used to print harvests the payload but
    # leaves the ledger row stuck at ``dispatched`` — printing it taught the very
    # bug class the gate exists to refuse.
    print(
        "Harvest: .venv/bin/python tools/harvest_modal_calls.py "
        f"--from-ledger --call-id {call_id} --execute"
    )
