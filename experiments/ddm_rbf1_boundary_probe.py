# SPDX-License-Identifier: MIT
"""Restartable move-44 render-seam falsifier, n600 only, macOS-CPU advisory.

The shipped archive parser and SemanticTokenRenderer produce each master. The
unchanged even frames come from the hash-pinned full public parse-back. This
is a render-seam probe, not a new full cold public decode or a contest row.
Every native master, camera pair, argmax and pose output is retained by chunk.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
sys.dont_write_bytecode = True
from experiments.ddm_rbf1_boundary_treatments import edges, treat

ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_rbf1/retained")
RECEIVER = Path("/Volumes/VertigoDataTier/pact/ddm_rlc5_cure_on_move43/candidate_runtime")
RAW = Path("/Volumes/VertigoDataTier/pact/ddm_rlc5_cure_on_move43/public_rlc4/output/0.raw")
FIELD = Path("/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/retained/fields/subset6.u8")
GT = Path("/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/gt_cache_dali.pt")
PINS = {
    "archive": "04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e",
    "raw": "2b762eba4a20a315c104f8447d6ea0e604f73c3d8b8b69b3fc63b0fc792d59fc",
    "field": "a92e7d902a4498961217f02c2b90d3fb9025901ba6d047201ff3bf297fa2f7a8",
    "gt": "a91d98252fe377c51ff7f3380c2fc9d30d84093fc54ee89e5e5f5102e6354994",
}
MODES = ("baseline", "guided", "ssaa", "sdf", "composition")
N, H, W, CH, CW = 600, 384, 512, 874, 1164
CHUNK = 5
AXIS = "[macOS-CPU advisory]"


def fact(path: Path) -> dict:
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_array(path: Path, array: np.ndarray) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        prior = np.load(path, mmap_mode="r", allow_pickle=False)
        if prior.dtype != array.dtype or prior.shape != array.shape or not np.array_equal(prior, array):
            raise ValueError(f"refusing to overwrite different retained bytes: {path}")
        return fact(path)
    # Interrupted payload writes remain under unique names for custody. They
    # never replace or delete another partial payload on a restart.
    temporary = path.with_suffix(path.suffix + f".partial.{time.time_ns()}")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return fact(path)


def verify_receipt(path: Path, binding: dict) -> dict | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value["binding"] != binding:
        raise ValueError(f"resume source/config changed: {path}")
    for record in value["artifacts"].values():
        if fact(Path(record["path"])) != record:
            raise ValueError(f"retained checkpoint content changed: {record['path']}")
    return value


def prepare() -> dict:
    ROOT.mkdir(parents=True, exist_ok=True)
    sources = {"archive": RECEIVER / "archive.zip", "raw": RAW, "field": FIELD, "gt": GT}
    records = {key: fact(path) for key, path in sources.items()}
    for key, record in records.items():
        if record["sha256"] != PINS[key]:
            raise ValueError(f"source identity mismatch: {key}")
    if records["raw"]["bytes"] != N * 2 * CH * CW * 3 or records["field"]["bytes"] != N * H * W:
        raise ValueError("source is not n600")
    from tac.gt_lineage import AUTHORITY_LINEAGE, assert_gt_lineage
    assert_gt_lineage(GT, required=AUTHORITY_LINEAGE, instrument="ddm_rbf1_boundary_probe")
    copied = ROOT / "receiver_readonly"
    source_files = [p for p in sorted(RECEIVER.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    manifest = []
    for path in source_files:
        dest = copied / path.relative_to(RECEIVER)
        record = fact(path)
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
        if fact(dest)["sha256"] != record["sha256"]:
            raise ValueError(f"read-only receiver copy drift: {dest}")
        manifest.append({"relative_path": str(path.relative_to(RECEIVER)), "source": record, "copy": fact(dest)})
    identity = {
        "inputs": records,
        "source_files": manifest,
        "probe": fact(Path(__file__)),
        "treatment": fact(Path(__file__).with_name("ddm_rbf1_boundary_treatments.py")),
        "torch": torch.__version__, "numpy": np.__version__,
        "host": platform.platform(), "seed": 0, "threads": 4, "chunk": CHUNK,
        "modes": list(MODES), "axis": AXIS, "score_claim": False,
    }
    prior = ROOT / "INPUTS.json"
    if prior.exists() and json.loads(prior.read_text()) != identity:
        raise ValueError("input binding changed; use a separately reviewed new generation")
    atomic_json(prior, identity)
    return identity


def load_model():
    """Exactly the semantic setup in the shipped f26 inflate path, riders first."""
    copied = ROOT / "receiver_readonly"
    sys.path.insert(0, str(copied))
    f26 = importlib.import_module("runtime.f26_inflate")
    renderer = f26._load_renderer(copied / "cpr1")
    parts = f26.read_residual_archive(copied / "archive.zip")
    model = renderer.SemanticTokenRenderer(96)
    if parts.semantic_blob.startswith(f26.SM1_SEMANTIC_MAGIC):
        parts = dataclasses.replace(parts, semantic_blob=f26.restore_sm1_semantic(parts.semantic_blob, model.state_dict()))
    if parts.semantic_blob.startswith(f26.RC1_SEMANTIC_MAGIC):
        parts = dataclasses.replace(parts, semantic_blob=f26.restore_rc1_semantic(parts.semantic_blob, model.state_dict()))
    state = renderer.unpack_variant_semantic_or_none(parts.semantic_blob, model.state_dict())
    if state is None:
        state = {r.schema.name: torch.from_numpy(np.ascontiguousarray(r.values, dtype=np.float32)) for r in f26.decode_wans1(parts.semantic_blob)}
    model.load_state_dict(state, strict=True)
    return model.eval()


def render(binding: dict) -> None:
    model = load_model()
    tokens = np.memmap(FIELD, mode="r", dtype=np.uint8, shape=(N, H, W))
    public_raw = np.memmap(RAW, mode="r", dtype=np.uint8, shape=(N, 2, CH, CW, 3))
    all_rows = []
    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        directory = ROOT / "chunks" / f"{start:03d}_{end:03d}"
        receipt = directory / "RENDER.json"
        existing = verify_receipt(receipt, binding)
        if existing is not None:
            all_rows.append(existing)
            continue
        # Reserve includes every unfinished candidate and native master; never
        # purge a prior payload to create space for a new candidate.
        remaining = N - start
        required = 40 * 1024**3 + remaining * (len(MODES) * 2 * CH * CW * 3 + H * W * 3 * 4)
        if shutil.disk_usage(ROOT).free < required:
            raise RuntimeError(f"certify-or-block storage refusal: need {required} free bytes")
        native = []
        camera = {name: [] for name in MODES}
        seconds = dict.fromkeys(MODES, 0.0)
        changes = dict.fromkeys(MODES, 0)
        source_diff = 0
        for pair in range(start, end):
            tic = time.perf_counter()
            with torch.inference_mode():
                out = model(torch.from_numpy(np.array(tokens[pair:pair + 1], dtype=np.int64)), torch.tensor([pair]))
                master = torch.nn.functional.interpolate(out, size=(CH, CW), mode="bilinear", align_corners=False).clamp(0, 255).round()
            base = master[0].to(torch.uint8).permute(1, 2, 0).numpy()
            seconds["baseline"] += time.perf_counter() - tic
            native.append(out[0].numpy())
            source_diff += int(np.count_nonzero(base != public_raw[pair, 1]))
            for name in MODES:
                tic = time.perf_counter()
                candidate = base if name == "baseline" else treat(base, tokens[pair], name)
                if name != "baseline":
                    seconds[name] += time.perf_counter() - tic
                changes[name] += int(np.count_nonzero(candidate != base))
                camera[name].append(np.stack((np.array(public_raw[pair, 0]), candidate)))
        artifacts = {"native": save_array(directory / "native_f32.npy", np.stack(native))}
        for name in MODES:
            artifacts[name] = save_array(directory / f"{name}_camera_u8.npy", np.stack(camera[name]))
        row = {"binding": binding, "start": start, "end": end, "artifacts": artifacts,
               "render_seconds": seconds, "changed_camera_channels": changes,
               "baseline_vs_retained_public_changed_channels": source_diff}
        atomic_json(receipt, row)
        all_rows.append(row)
        print(json.dumps({"stage": "render", "completed_pairs": end, "source_changed_channels": source_diff, "seconds": seconds}), flush=True)
    atomic_json(ROOT / "RENDER_COMPLETE.json", {"axis": AXIS, "score_claim": False, "n": N,
        "chunks": [fact(ROOT / "chunks" / f"{r['start']:03d}_{r['end']:03d}" / "RENDER.json") for r in all_rows],
        "baseline_vs_retained_public_changed_channels": sum(r["baseline_vs_retained_public_changed_channels"] for r in all_rows),
        "render_seconds": {m: sum(r["render_seconds"][m] for r in all_rows) for m in MODES},
        "timing_scope": "semantic render+resize for baseline; additive post-render CPU treatment only for candidates; not full decode or T4 timing"})


def score(binding: dict, slot_receipt: Path) -> None:
    """One sequential n600 CPU scorer job. Explicit fleet slot receipt required."""
    from scipy.ndimage import distance_transform_cdt, maximum_filter
    slot = json.loads(slot_receipt.read_text())
    if slot.get("owner") != "ddm_rbf1" or slot.get("status") != "assigned":
        raise ValueError("fleet scorer slot is not assigned to ddm_rbf1")
    if not (ROOT / "RENDER_COMPLETE.json").is_file():
        raise ValueError("all n600 renders must complete before scoring")
    sys.path.insert(0, str(REPO / "upstream"))
    modules = importlib.import_module("modules")
    net = modules.DistortionNet().eval()
    net.load_state_dicts(modules.posenet_sd_path, modules.segnet_sd_path, torch.device("cpu"))
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    gt = torch.load(GT, map_location="cpu", weights_only=False)
    labels = gt["seg"].numpy().astype(np.uint8)
    poses = gt["pose"].numpy().astype(np.float64)
    if labels.shape != (N, H, W) or poses.shape != (N, 6):
        raise ValueError("GT geometry mismatch")
    tokens = np.memmap(FIELD, mode="r", dtype=np.uint8, shape=(N, H, W))
    score_binding = {"inputs": binding, "modules": fact(REPO / "upstream/modules.py"),
                     "frame_utils": fact(REPO / "upstream/frame_utils.py"),
                     "segnet": fact(modules.segnet_sd_path), "posenet": fact(modules.posenet_sd_path)}
    totals = {m: {"seg_errors": 0, "pose_sse": 0.0, "benefit": 0, "harm": 0, "wash": 0, "seconds": 0.0} for m in MODES}
    residual_total = {"errors": 0, "token_correct": 0, "token_edge_distance_le1": 0, "gt_class_in_predicted_3x3": 0}
    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        directory = ROOT / "chunks" / f"{start:03d}_{end:03d}"
        render_receipt = verify_receipt(directory / "RENDER.json", binding)
        if render_receipt is None:
            raise ValueError("missing render checkpoint")
        receipt = directory / "SCORE.json"
        row = verify_receipt(receipt, score_binding)
        if row is None:
            predictions, pose_outputs, times = {}, {}, {}
            artifacts = {}
            for mode in MODES:
                camera = np.load(directory / f"{mode}_camera_u8.npy", mmap_mode="r", allow_pickle=False)
                predictions[mode], pose_outputs[mode] = [], []
                tic = time.perf_counter()
                for local in range(end - start):
                    with torch.inference_mode():
                        p, s = net(torch.from_numpy(np.array(camera[local:local + 1])))
                    predictions[mode].append(s.argmax(dim=1)[0].numpy().astype(np.uint8))
                    pose_outputs[mode].append(p["pose"][0, :6].numpy())
                times[mode] = time.perf_counter() - tic
                predictions[mode] = np.stack(predictions[mode])
                pose_outputs[mode] = np.stack(pose_outputs[mode])
                artifacts[mode + "_argmax"] = save_array(directory / f"{mode}_argmax.npy", predictions[mode])
                artifacts[mode + "_pose"] = save_array(directory / f"{mode}_pose.npy", pose_outputs[mode])
            target = labels[start:end]
            baseline = predictions["baseline"]
            stats = {}
            for mode in MODES:
                predicted = predictions[mode]
                changed = predicted != baseline
                stats[mode] = {
                    "seg_errors": int(np.count_nonzero(predicted != target)),
                    "pose_sse": float(np.square(pose_outputs[mode].astype(np.float64) - poses[start:end]).sum()),
                    "benefit": int(np.count_nonzero(changed & (baseline != target) & (predicted == target))),
                    "harm": int(np.count_nonzero(changed & (baseline == target) & (predicted != target))),
                    "wash": int(np.count_nonzero(changed & (baseline != target) & (predicted != target))),
                    "seconds": times[mode],
                }
            residual = dict.fromkeys(residual_total, 0)
            error_rows = []
            for local, pair in enumerate(range(start, end)):
                wrong = baseline[local] != target[local]
                yy, xx = np.nonzero(wrong)
                token_correct = tokens[pair][wrong] == target[local][wrong]
                boundary = edges(tokens[pair])
                distances = distance_transform_cdt(~boundary, metric="chessboard") if boundary.any() else np.full((H, W), max(H, W))
                nearby = np.zeros((H, W), dtype=bool)
                for label in np.unique(target[local][wrong]):
                    nearby |= (target[local] == label) & maximum_filter(baseline[local] == label, size=3, mode="nearest")
                residual["errors"] += len(yy)
                residual["token_correct"] += int(token_correct.sum())
                residual["token_edge_distance_le1"] += int((distances[wrong] <= 1).sum())
                residual["gt_class_in_predicted_3x3"] += int(nearby[wrong].sum())
                error_rows.append(np.column_stack((np.full(len(yy), pair), yy, xx, target[local][wrong], baseline[local][wrong], tokens[pair][wrong], token_correct, distances[wrong], nearby[wrong])).astype(np.int32))
            artifacts["residual_cells"] = save_array(directory / "residual_cells_i32.npy", np.concatenate(error_rows))
            row = {"binding": score_binding, "start": start, "end": end, "artifacts": artifacts, "stats": stats, "residual": residual,
                   "residual_columns": ["pair", "y", "x", "gt", "predicted", "token", "token_correct", "chebyshev_distance_to_token_edge", "gt_class_present_in_predicted_3x3"]}
            atomic_json(receipt, row)
        for mode in MODES:
            for key in totals[mode]:
                totals[mode][key] += row["stats"][mode][key]
        for key in residual_total:
            residual_total[key] += row["residual"][key]
        print(json.dumps({"stage": "score", "completed_pairs": end, "errors": {m: totals[m]["seg_errors"] for m in MODES}}), flush=True)
    for values in totals.values():
        values["d_seg"] = values["seg_errors"] / (N * H * W)
        values["d_pose"] = values["pose_sse"] / (N * 6)
        values["S"] = 100 * values["d_seg"] + math.sqrt(10 * values["d_pose"]) + 25 * 180406 / 37545489
    for values in totals.values():
        values["delta_S"] = values["S"] - totals["baseline"]["S"]
    atomic_json(ROOT / "RESULT.json", {"axis": AXIS, "score_claim": False, "n": N, "field_positions": N * H * W,
        "modes": totals, "residual": residual_total, "slot_receipt": fact(slot_receipt),
        "composition_scope": "guided then ssaa then sdf; no claim of optimality over unmeasured subsets",
        "render": json.loads((ROOT / "RENDER_COMPLETE.json").read_text())})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "render", "score"))
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--slot-receipt", type=Path)
    args = parser.parse_args()
    if args.resume_from.resolve() != ROOT.resolve():
        raise ValueError("resume root must be the chartered retained SSD directory")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    binding = prepare()
    atomic_json(ROOT / f"COMMAND_{args.stage}.json", {"argv": sys.argv, "cwd": str(Path.cwd()),
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "axis": AXIS,
        "score_claim": False, "retention": "all per-candidate renders and stage checkpoints kept; incomplete chunks preserved; no deletion"})
    if args.stage == "render":
        render(binding)
    elif args.stage == "score":
        if args.slot_receipt is None:
            parser.error("score requires --slot-receipt with explicit fleet ownership")
        score(binding, args.slot_receipt)
    print(json.dumps({"completed_stage": args.stage, "root": str(ROOT)}), flush=True)


if __name__ == "__main__":
    main()
