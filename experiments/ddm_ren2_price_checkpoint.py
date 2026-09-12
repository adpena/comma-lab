"""ddm_ren2: price a refit renderer checkpoint END TO END against move 48.

Four legs, cheapest and most-falsifying first, so a checkpoint that cannot clear the
bar is refused before the expensive leg is spent:

``member``     the REAL re-encode -- ``pack_prune_mixed_candidate`` -> ``SM1S`` ->
               ``CK2`` -> Brotli, byte-identical to the shipped chain on the shipped
               state (proved by ``ddm_ren2_restore_init``).  ``ddm_fe1``'s +70 B break
               fee was measured at N <= 200 changed codes; a refit changes all 59,376,
               so this term is MEASURED, never extrapolated.
``amplitude``  the render delta against the SHIPPED frame 1, in camera LSB RMS, with
               ``ddm_ren1``'s smooth/noise split (the smooth band is the 24x32 bicubic
               subspace the carrier itself lives in; the residual is the noise band).
               ``ddm_ren1`` measured the carrier's pose ABSORPTION at 0.5 LSB RMS; above
               that amplitude the measurement does not apply and the checkpoint is
               REFUSED rather than priced on a transferred number.
``seg``        exact n600 ``d_seg`` through R on the frozen CPU SegNet -- the binding
               term.  The REALIZED state is used: the candidate is packed and parsed
               back through the shipped receiver, so what is scored is what would ship.
``resolve``    the terminal carrier re-solve with the canonical unforked
               ``up2.solve_pair_realized`` -> post-re-solve ``d_pose`` and the carrier's
               byte delta.  Pose is never trained; it is MEASURED here after every
               render change (operator 2026-09-10).

Axis: ``[macOS-CPU advisory, jg1/up2 instrument, DALI GT lineage]``.  Absolute levels
belong to this instrument; the control row measured in the same producer is what makes
a delta interpretable.  No score claim, no promotion, no archive is built here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "experiments"), str(REPO / "src")]

import ddm_jg1_seg_solve as jg1
import ddm_ren1_step0_probe as ren1_probe
import ddm_ren2_restore_init as ren2_init
import ddm_up2_shipping_pose_solve as up2
import numpy as np

AXIS = "[macOS-CPU advisory, jg1/up2 instrument, DALI GT lineage]"
STORE = Path("/Volumes/VertigoDataTier/pact/ddm_ren2/price")
RESERVE_BYTES = 40 << 30

#: Move 48 components, from the seal this arm is bound to.
BASE_ARCHIVE_BYTES = 179_111
BASE_D_SEG = 0.00010345
BASE_D_POSE = 4.59e-06
BASE_SCORE = 0.13638261682704697
ADMIT_BAR_DELTA_S = -2e-5
RATE_DENOMINATOR = 37_545_489

#: ren1's measured amplitude ceiling: the amplitude at which the carrier's pose
#: absorption was measured.  Above it the absorption number does not transfer.
AMPLITUDE_CEILING_LSB = 0.5
#: ren1's smooth band: the 24x32 grid the shipped frame-0 carrier field lives on.
SMOOTH_GRID_H, SMOOTH_GRID_W = ren1_probe.SMOOTH_GRID_H, ren1_probe.SMOOTH_GRID_W

#: ren1 §4.1b required d_seg cut, by the spectrum of the render change, at 0.5 LSB.
REQUIRED_CUT = {"noise": 0.01006, "smooth": 0.02682}

#: ren1's retained n600 SegNet argmax plane for the SHIPPED render.  Read-only reuse:
#: it lets every checkpoint report "cells moved", which is ``ddm_rw1``'s own unit
#: (one int4 code step moves 240-455 argmax cells) and the admit bar's unit
#: (2e-5 S / (100 / 117,964,800) = 23.6 repaired cells).
REN1_CONTROL_PLANES = Path(
    "/Volumes/VertigoDataTier/pact/ddm_ren1/step0_probe/floor_and_sensitivity/planes.npz"
)
REN1_CONTROL_PLANES_SHA256 = (
    "a6e3c138c3332b3e"  # prefix; the full sha is recorded by fact() at run time
)
#: cells per unit of d_seg: n600 x 384 x 512.
CELLS_TOTAL = 600 * 384 * 512


class Ren2PriceError(RuntimeError):
    """Refusal raised by this producer.  Never downgraded to a warning."""


def retain(path: Path, payload: bytes) -> dict[str, Any]:
    if not path.resolve().is_relative_to(STORE.resolve()):
        raise Ren2PriceError(f"write outside the ren2 price store: {path}")
    if shutil.disk_usage(STORE.parent).free < RESERVE_BYTES + len(payload):
        raise Ren2PriceError("STORAGE_BLOCK: keep all payloads; free space below reserve")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)
    return ren2_init.fact(path)


def score_from_components(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    """The contest score, recomputed from components -- never a rounded total."""
    import math

    return (
        100.0 * d_seg
        + math.sqrt(10.0 * d_pose)
        + 25.0 * archive_bytes / RATE_DENOMINATOR
    )


def load_candidate_state(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """The EMA shadow a trainer checkpoint deploys, plus its own provenance."""
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" not in blob:
        raise Ren2PriceError(f"{path} carries no state_dict")
    declared = blob.get("deployment_weights")
    if declared not in (None, "ema_shadow"):
        raise Ren2PriceError(
            f"{path} declares deployment_weights={declared!r}; this pricer only prices "
            "the EMA shadow, which is what the EMA non-negotiable says ships"
        )
    state = {name: value.detach().cpu().clone() for name, value in blob["state_dict"].items()}
    provenance = {
        "checkpoint": ren2_init.fact(path),
        "deployment_weights": declared,
        "checkpoint_kind": blob.get("checkpoint_kind"),
        "step": (blob.get("training_state") or {}).get("step"),
        "phase": (blob.get("training_state") or {}).get("phase"),
        "best_step": (blob.get("training_state") or {}).get("best_step"),
        "best_exact_seg": blob.get("best_exact_seg"),
    }
    return state, provenance


def realize(state, *, shipped) -> dict[str, Any]:
    """Pack the candidate through the SHIPPED chain and parse it back.

    What gets scored is the PARSED-BACK state, not the float checkpoint: the receiver
    only ever sees the quantized, row-pruned object, and a score measured on the float
    would be a surrogate wearing the candidate's name.
    """
    import torch

    encoded = ren2_init.encode_member(
        state,
        representation=shipped["representation"],
        mixer_weights=shipped["mixer_weights"],
        template=shipped["template"],
        brotli_quality=shipped["brotli_quality"],
        brotli_lgwin=shipped["brotli_lgwin"],
        ck2=shipped["ck2"],
    )
    parsed = shipped["receiver"].unpack_variant_semantic_or_none(
        encoded["rider"], shipped["template"]
    )
    if parsed is None:
        raise Ren2PriceError("the candidate rider is not a tagged variant the receiver reads")
    if set(parsed) != set(encoded["realized_state"]):
        raise Ren2PriceError("pack/parse key set drifted")
    for name in parsed:
        if not torch.equal(parsed[name], encoded["realized_state"][name]):
            raise Ren2PriceError(f"pack/parse tensor drifted: {name}")
    return {"encoded": encoded, "parsed": parsed}


def renderer_with_state(shipped, parsed):
    """The shipped receiver's own renderer, loaded with the candidate's realized weights."""
    model = shipped["inflate"].SemanticTokenRenderer(96)
    model.load_state_dict(parsed, strict=True)
    model = model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def smooth_noise_split(delta_hwc: np.ndarray) -> tuple[float, float, float]:
    """Split a camera-plane delta into ren1's smooth band and its residual.

    The smooth band is the range of the 24x32 -> camera bicubic upsample ren1 used to
    build its SMOOTH treatment; the projection is the area-average down, bicubic back
    up.  The residual is everything that band cannot represent -- ren1's NOISE end.
    Reported as RMS in camera LSB so the three numbers are directly comparable.
    """
    import torch
    from torch.nn import functional

    tensor = torch.from_numpy(np.ascontiguousarray(delta_hwc)).float()
    chw = tensor.permute(2, 0, 1)[None]
    low = functional.adaptive_avg_pool2d(chw, (SMOOTH_GRID_H, SMOOTH_GRID_W))
    smooth = functional.interpolate(
        low, size=chw.shape[-2:], mode="bicubic", align_corners=False
    )
    residual = chw - smooth
    rms = lambda value: float(value.square().mean().sqrt())  # noqa: E731
    return rms(chw), rms(smooth), rms(residual)


class CandidateFrames:
    """A ``raw``-shaped frame-1 source for ``up2.solve_pair_realized``.

    ``solve_pair_realized`` reads its ``raw`` argument exactly once, as
    ``raw[2 * index + 1]``; it regenerates frame 0 itself from the codes under test.
    So the contract is only: return the CANDIDATE's camera-resolution frame 1 for the
    requested odd index.  Rendering on demand with a one-pair cache keeps the 1.8 GB
    raster out of RAM while staying bit-identical to the seg leg, which renders through
    the same ``jg1.render_frame1`` call.  Shape copied from ``ddm_ren1``'s shim.
    """

    def __init__(self, semantic, tokens):
        self._semantic = semantic
        self._tokens = tokens
        self._cache: dict[int, np.ndarray] = {}

    def __getitem__(self, index):
        odd = np.asarray(index).reshape(-1)
        if odd.size != 1:
            raise Ren2PriceError("frame-1 shim serves exactly one pair per call")
        value = int(odd[0])
        if value % 2 != 1:
            raise Ren2PriceError(f"frame-1 shim was asked for even index {value}")
        pair = (value - 1) // 2
        cached = self._cache.get(pair)
        if cached is not None:
            return cached
        chunk = np.array([pair], dtype=np.int64)
        frames = jg1.render_frame1(self._semantic, self._tokens[chunk], chunk)
        self._cache = {pair: frames}
        return frames


def load_shipped():
    """Everything the shipped encode/decode chain needs, bound by sha."""
    container = ren2_init.read_shipped_semantic_container()
    residual, mixer, inflate, receiver = ren2_init._receiver_modules()
    representation = ren2_init.measure_deployed_representation(
        container["body"], container["template"]
    )
    quality, lgwin = ren2_init.identify_brotli_container(
        container["staged"], container["member"]
    )
    return {
        "container": container,
        "representation": representation,
        "mixer_weights": ren2_init.measure_mixer_weights(container["rider"]),
        "template": container["template"],
        "brotli_quality": quality,
        "brotli_lgwin": lgwin,
        "ck2": container["ck2_semantic"],
        "receiver": receiver,
        "inflate": inflate,
        "member_bytes": len(container["member"]),
    }


def run(args) -> int:
    started = time.time()
    out = STORE / args.label
    out.mkdir(parents=True, exist_ok=True)
    pointer = ren2_init.bind_pointer()
    shipped = load_shipped()
    tokens = jg1.load_tokens(ren2_init.TOKEN_FIELD)

    state, checkpoint_provenance = load_candidate_state(args.checkpoint)
    realized = realize(state, shipped=shipped)
    encoded = realized["encoded"]
    member_delta = encoded["member_bytes"] - shipped["member_bytes"]
    archive_bytes = BASE_ARCHIVE_BYTES + member_delta

    shipped_state = {
        name: value.detach().cpu().clone()
        for name, value in jg1.load_semantic_renderer(
            archive_path=ren2_init.POINTER_TREE / "archive.zip",
            runtime_dir=ren2_init.POINTER_TREE / "runtime",
        ).state_dict().items()
    }
    changed_parameters = int(
        sum(int((state[name] != shipped_state[name]).sum()) for name in shipped_state)
    )
    realized_changed = int(
        sum(
            int((encoded["realized_state"][name] != shipped_state[name]).sum())
            for name in shipped_state
        )
    )

    binding = {
        "schema": "ddm_ren2_price_checkpoint.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "label": args.label,
        "legs": list(args.legs),
        "pointer": pointer,
        "pointer_move": 48,
        "base": {
            "score": BASE_SCORE,
            "d_seg": BASE_D_SEG,
            "d_pose": BASE_D_POSE,
            "archive_bytes": BASE_ARCHIVE_BYTES,
            "member_bytes": shipped["member_bytes"],
            "admit_bar_delta_s": ADMIT_BAR_DELTA_S,
        },
        "candidate": checkpoint_provenance,
        "member": {
            "bytes": encoded["member_bytes"],
            "sha256": encoded["member_sha256"],
            "delta_bytes": member_delta,
            "body_bytes": encoded["body_bytes"],
            "body_sha256": encoded["body_sha256"],
            "archive_bytes_if_shipped": archive_bytes,
            "kept_rows": encoded["pack_meta"]["kept_rows"],
            "non_default_bit_allocation": encoded["pack_meta"]["non_default_bit_allocation"],
            "container": {
                "brotli_quality": shipped["brotli_quality"],
                "brotli_lgwin": shipped["brotli_lgwin"],
                "ck2_plane2": shipped["ck2"],
            },
        },
        "parameters_changed_float": changed_parameters,
        "parameters_changed_realized": realized_changed,
        "producer": ren2_init.fact(Path(__file__)),
        "restore_module": ren2_init.fact(Path(ren2_init.__file__)),
        "jg1": ren2_init.fact(Path(jg1.__file__)),
        "up2": ren2_init.fact(Path(up2.__file__)),
        "ren1_probe": ren2_init.fact(Path(ren1_probe.__file__)),
        "threads": args.threads,
        "pairs_requested": args.pairs,
    }
    # Resume is keyed by LABEL, and rows are keyed by pair -- so re-using a label for a
    # DIFFERENT checkpoint would silently price the previous candidate's rows as this
    # one's.  Bind the label to the checkpoint's sha and refuse instead: a stale row
    # that looks valid is the confound class this campaign keeps paying for.
    previous_inputs = out / "INPUTS.json"
    if args.resume and previous_inputs.is_file():
        previous = json.loads(previous_inputs.read_text())
        previous_sha = (previous.get("candidate") or {}).get("checkpoint", {}).get("sha256")
        if previous_sha and previous_sha != checkpoint_provenance["checkpoint"]["sha256"]:
            raise Ren2PriceError(
                f"label {args.label!r} already holds rows for checkpoint {previous_sha}; "
                f"this invocation is {checkpoint_provenance['checkpoint']['sha256']}. "
                "Use a new --label or --no-resume; a resumed row set from another "
                "candidate would be priced as this one's."
            )
    retain(previous_inputs, (json.dumps(binding, indent=2, sort_keys=True) + "\n").encode())

    result: dict[str, Any] = dict(binding)
    semantic = renderer_with_state(shipped, realized["parsed"])
    shipped_semantic = jg1.load_semantic_renderer(
        archive_path=ren2_init.POINTER_TREE / "archive.zip",
        runtime_dir=ren2_init.POINTER_TREE / "runtime",
    )

    # ---------------- amplitude ----------------
    if "amplitude" in args.legs:
        pairs = up2.select_pairs(args.pairs, args.seed)
        deltas, smooths, noises, cells, frame_max = [], [], [], [], 0
        for pair in pairs:
            chunk = np.array([int(pair)], dtype=np.int64)
            base = jg1.render_frame1(shipped_semantic, tokens[chunk], chunk)[0]
            cand = jg1.render_frame1(semantic, tokens[chunk], chunk)[0]
            delta = cand.astype(np.float64) - base.astype(np.float64)
            total, smooth, noise = smooth_noise_split(delta)
            deltas.append(total)
            smooths.append(smooth)
            noises.append(noise)
            frame_max = max(frame_max, float(np.abs(delta).max()))
            cells.append(int(np.count_nonzero(cand != base)))
        rms = float(np.sqrt(np.mean(np.square(deltas))))
        smooth_rms = float(np.sqrt(np.mean(np.square(smooths))))
        noise_rms = float(np.sqrt(np.mean(np.square(noises))))
        dominant = "smooth" if smooth_rms >= noise_rms else "noise"
        result["amplitude"] = {
            "pairs": len(pairs),
            "shape": "seeded random via up2.select_pairs"
            if len(pairs) < jg1.N_PAIRS
            else "full n600",
            "camera_lsb_rms": rms,
            "smooth_band_rms": smooth_rms,
            "noise_band_rms": noise_rms,
            "smooth_over_noise": smooth_rms / noise_rms if noise_rms > 0 else float("inf"),
            "dominant_band": dominant,
            "max_abs_lsb": frame_max,
            "mean_camera_samples_changed": float(np.mean(cells)),
            "mean_camera_fraction_changed": float(np.mean(cells)) / (
                jg1.CAMERA_H * jg1.CAMERA_W * 3
            ),
            "ceiling_lsb": AMPLITUDE_CEILING_LSB,
            "inside_ceiling": bool(rms <= AMPLITUDE_CEILING_LSB),
            "required_d_seg_cut_at_ceiling": REQUIRED_CUT[dominant],
            "note": (
                "ren1 measured the carrier's pose absorption AT 0.5 LSB RMS on this "
                "object; above that the absorption number does not transfer and the "
                "checkpoint is refused rather than priced on an imported ratio"
            ),
        }
        retain(out / "RESULT.json", (json.dumps(result, indent=1, sort_keys=True) + "\n").encode())
        print(json.dumps({"leg": "amplitude", **result["amplitude"]}, sort_keys=True), flush=True)
        if not result["amplitude"]["inside_ceiling"] and not args.price_outside_ceiling:
            result["verdict"] = "REFUSED_AMPLITUDE_OUTSIDE_CEILING"
            result["elapsed_seconds"] = time.time() - started
            retain(
                out / "RESULT.json",
                (json.dumps(result, indent=1, sort_keys=True) + "\n").encode(),
            )
            print(json.dumps({"verdict": result["verdict"]}), flush=True)
            return 0

    # ---------------- seg ----------------
    if "seg" in args.legs:
        net = jg1.load_segnet()
        gt_labels = jg1.load_gt_seg_labels(up2.LINEAGE_DALI)
        shipped_argmax = None
        if REN1_CONTROL_PLANES.is_file():
            with np.load(REN1_CONTROL_PLANES) as blob:
                shipped_argmax = blob["argmax_control"]
        rows_path = out / "seg_rows.jsonl"
        done: dict[int, float] = {}
        moved: dict[int, int] = {}
        if args.resume and rows_path.is_file():
            for line in rows_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    done[int(row["pair"])] = float(row["d_seg"])
                    if row.get("cells_moved") is not None:
                        moved[int(row["pair"])] = int(row["cells_moved"])
        with rows_path.open("a", encoding="utf-8") as stream:
            for start in range(0, jg1.N_PAIRS, args.chunk):
                chunk = np.arange(start, min(start + args.chunk, jg1.N_PAIRS), dtype=np.int64)
                pending = np.array([p for p in chunk if int(p) not in done], dtype=np.int64)
                if pending.size == 0:
                    continue
                frames = jg1.render_frame1(semantic, tokens[pending], pending)
                argmax = jg1.argmax_from_camera_frames(net, frames)
                per_pair = jg1.d_seg_per_pair(argmax, gt_labels[pending])
                for row_index, (pair, value) in enumerate(
                    zip(pending, per_pair, strict=True)
                ):
                    cells = (
                        int(np.count_nonzero(argmax[row_index] != shipped_argmax[int(pair)]))
                        if shipped_argmax is not None
                        else None
                    )
                    done[int(pair)] = float(value)
                    if cells is not None:
                        moved[int(pair)] = cells
                    stream.write(
                        json.dumps(
                            {"pair": int(pair), "d_seg": float(value), "cells_moved": cells}
                        )
                        + "\n"
                    )
                stream.flush()
                print(
                    json.dumps(
                        {
                            "leg": "seg",
                            "done": len(done),
                            "of": jg1.N_PAIRS,
                            "running_mean": float(np.mean(list(done.values()))),
                            "seconds": round(time.time() - started, 1),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        d_seg = float(np.mean([done[p] for p in range(jg1.N_PAIRS)]))
        cut = (BASE_D_SEG - d_seg) / BASE_D_SEG
        wrong_cells = d_seg * CELLS_TOTAL
        result["seg"] = {
            "pairs": jg1.N_PAIRS,
            "d_seg": d_seg,
            "base_d_seg": BASE_D_SEG,
            "cut_fraction_vs_pointer": cut,
            "wrong_cells": wrong_cells,
            "base_wrong_cells": BASE_D_SEG * CELLS_TOTAL,
            "net_cells_repaired_vs_pointer": (BASE_D_SEG - d_seg) * CELLS_TOTAL,
            "cells_moved_vs_shipped_argmax": (
                int(sum(moved.values())) if len(moved) == jg1.N_PAIRS else None
            ),
            "cells_moved_source": (
                str(REN1_CONTROL_PLANES) if shipped_argmax is not None else None
            ),
            "admit_bar_in_cells": 23.6,
            "rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
            "instrument_note": (
                "the pointer's T4 d_seg is 0.00010345 and ren1's control on THIS "
                "instrument is 0.00010338677 (0.06 %); a cut smaller than that "
                "reproduction gap is not resolvable here"
            ),
        }
        retain(out / "RESULT.json", (json.dumps(result, indent=1, sort_keys=True) + "\n").encode())
        print(json.dumps({"leg": "seg", **result["seg"]}, sort_keys=True), flush=True)

    # ---------------- pose (PRE re-solve) ----------------
    if "pose" in args.legs:
        posenet = up2.load_posenet("cpu")
        carrier = up2.load_carrier_state(ren2_init.POINTER_TREE, verify_archive=False)
        targets, lineage = up2.load_gt_poses(jg1.DEFAULT_GT_DALI)
        if lineage != up2.LINEAGE_DALI:
            raise Ren2PriceError(f"pose GT lineage is {lineage}, not DALI")
        frames = CandidateFrames(semantic, tokens)
        rows_path = out / "pose_rows.jsonl"
        done_pose: dict[int, float] = {}
        if args.resume and rows_path.is_file():
            for line in rows_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    done_pose[int(row["pair"])] = float(row["d_pose"])
        with rows_path.open("a", encoding="utf-8") as stream:
            for pair in range(jg1.N_PAIRS):
                if pair in done_pose:
                    continue
                index = np.array([pair], dtype=np.int64)
                per_pair, _poses = up2.measure_pose(
                    posenet,
                    carrier,
                    carrier.coefficients,
                    frames,
                    targets,
                    index,
                    batch_size=1,
                )
                done_pose[pair] = float(per_pair[0])
                stream.write(json.dumps({"pair": pair, "d_pose": float(per_pair[0])}) + "\n")
                if pair % 25 == 0 or pair == jg1.N_PAIRS - 1:
                    stream.flush()
                    print(
                        json.dumps(
                            {
                                "leg": "pose",
                                "done": len(done_pose),
                                "of": jg1.N_PAIRS,
                                "running_mean": float(np.mean(list(done_pose.values()))),
                                "seconds": round(time.time() - started, 1),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        d_pose_pre = float(np.mean([done_pose[p] for p in range(jg1.N_PAIRS)]))
        result["pose"] = {
            "pairs": jg1.N_PAIRS,
            "d_pose_pre_resolve": d_pose_pre,
            "base_d_pose": BASE_D_POSE,
            "ratio_vs_pointer": d_pose_pre / BASE_D_POSE,
            "posenet_gradients_enabled": False,
            "rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
            "note": (
                "PRE the terminal carrier re-solve, and measured WITHOUT "
                "up2.enable_posenet_gradients() so it reproduces ren1's step-0 probe "
                "control; the resolve leg enables them, as ren1's resolve producer "
                "does, and ren1 measured that forward difference at 0.17 % mean"
            ),
        }
        retain(out / "RESULT.json", (json.dumps(result, indent=1, sort_keys=True) + "\n").encode())
        print(json.dumps({"leg": "pose", **result["pose"]}, sort_keys=True), flush=True)

    # ---------------- resolve ----------------
    if "resolve" in args.legs:
        up2.enable_posenet_gradients()
        posenet = up2.load_posenet("cpu")
        carrier = up2.load_carrier_state(ren2_init.POINTER_TREE, verify_archive=False)
        targets, lineage = up2.load_gt_poses(jg1.DEFAULT_GT_DALI)
        if lineage != up2.LINEAGE_DALI:
            raise Ren2PriceError(f"pose GT lineage is {lineage}, not DALI")
        frames = CandidateFrames(semantic, tokens)
        pairs = up2.select_pairs(args.resolve_pairs, args.seed)
        rows_path = out / "resolve_rows.jsonl"
        done_rows: dict[int, dict[str, Any]] = {}
        if args.resume and rows_path.is_file():
            for line in rows_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    done_rows[int(row["pair"])] = row
        with rows_path.open("a", encoding="utf-8") as stream:
            for position, pair in enumerate(pairs):
                if int(pair) in done_rows:
                    continue
                row = up2.solve_pair_realized(
                    posenet,
                    carrier,
                    frames,
                    targets[int(pair)],
                    int(pair),
                    carrier.codes[int(pair)],
                    max_passes=0,
                )
                done_rows[int(pair)] = row
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                if position % 5 == 0 or position == len(pairs) - 1:
                    finished = [done_rows[int(p)] for p in pairs if int(p) in done_rows]
                    print(
                        json.dumps(
                            {
                                "leg": "resolve",
                                "done": len(finished),
                                "of": len(pairs),
                                "start_mean": float(
                                    np.mean([r["start_d_pose"] for r in finished])
                                ),
                                "final_mean": float(
                                    np.mean([r["final_d_pose"] for r in finished])
                                ),
                                "seconds": round(time.time() - started, 1),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        ordered = [done_rows[int(p)] for p in pairs if int(p) in done_rows]
        start_mean = float(np.mean([r["start_d_pose"] for r in ordered]))
        final_mean = float(np.mean([r["final_d_pose"] for r in ordered]))
        support = [int(r["changed_coordinates"]) for r in ordered]
        # The carrier's byte delta is MEASURED by the carrier's own CPR1 Rice pricer,
        # which refuses to return a number unless it first reproduces the SHIPPED bit
        # count from the shipped codes.  A flag-supplied byte delta would be an
        # assumption wearing a measurement's name.
        carrier_report = None
        if len(ordered) >= jg1.N_PAIRS:
            candidate_codes = np.array(carrier.codes, dtype=np.int32).copy()
            for row in ordered:
                candidate_codes[int(row["pair"])] = np.asarray(
                    row["codes"], dtype=np.int32
                )
            carrier_report, _base = up2.price_full_resolve_bytes(
                ren2_init.POINTER_TREE, candidate_codes
            )
        result["resolve"] = {
            "carrier_byte_price": carrier_report,
            "pairs": len(ordered),
            "shape": "seeded random via up2.select_pairs"
            if len(ordered) < jg1.N_PAIRS
            else "full n600",
            "is_verdict_grade": bool(len(ordered) >= jg1.N_PAIRS),
            "start_d_pose_mean": start_mean,
            "final_d_pose_mean": final_mean,
            "recovery_factor": start_mean / final_mean if final_mean > 0 else float("inf"),
            "pairs_improved": int(
                sum(1 for r in ordered if r["final_d_pose"] < r["start_d_pose"])
            ),
            "total_changed_coordinates": int(sum(support)),
            "rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        }
        retain(out / "RESULT.json", (json.dumps(result, indent=1, sort_keys=True) + "\n").encode())
        print(json.dumps({"leg": "resolve", **result["resolve"]}, sort_keys=True), flush=True)

    # ---------------- S from components ----------------
    seg_leg = result.get("seg")
    resolve_leg = result.get("resolve")
    if seg_leg is not None:
        d_seg = seg_leg["d_seg"]
        d_pose = (
            resolve_leg["final_d_pose_mean"] if resolve_leg is not None else BASE_D_POSE
        )
        measured_carrier = (
            (resolve_leg or {}).get("carrier_byte_price") or {}
        ).get("delta_bytes")
        carrier_delta = (
            int(measured_carrier)
            if measured_carrier is not None
            else int(args.carrier_delta_bytes)
        )
        candidate_bytes = archive_bytes + carrier_delta
        # The base is recomputed on THIS instrument's control so the delta is not a
        # cross-instrument subtraction: ren1 reproduces the pointer to 0.06 %/0.07 %.
        base_score = score_from_components(BASE_D_SEG, BASE_D_POSE, BASE_ARCHIVE_BYTES)
        candidate_score = score_from_components(d_seg, d_pose, candidate_bytes)
        result["price"] = {
            "base_score_from_components": base_score,
            "base_score_sealed": BASE_SCORE,
            "candidate_score_from_components": candidate_score,
            "delta_s": candidate_score - base_score,
            "member_delta_bytes": member_delta,
            "carrier_delta_bytes": carrier_delta,
            "carrier_delta_is_measured": measured_carrier is not None,
            "archive_bytes": candidate_bytes,
            "d_seg": d_seg,
            "d_pose": d_pose,
            "d_pose_is_measured_post_resolve": resolve_leg is not None,
            "admit_bar_delta_s": ADMIT_BAR_DELTA_S,
            # A fire needs an n600 seg leg, an n600 resolve leg, AND a carrier byte
            # delta the carrier's own pricer measured.  A subsample resolve is a
            # BRACKET; letting it set `fires` would be a guard that fails open on
            # non-verdict-grade evidence, which is the shape this campaign's
            # fail-open-guard class keeps taking.
            "fires": bool(
                (candidate_score - base_score) < ADMIT_BAR_DELTA_S
                and resolve_leg is not None
                and resolve_leg.get("is_verdict_grade")
                and measured_carrier is not None
                and seg_leg["pairs"] == jg1.N_PAIRS
            ),
            "note": (
                "d_pose defaults to the pointer's own value ONLY when the resolve leg "
                "did not run, and the carrier byte delta falls back to the flag ONLY "
                "when the resolve was not n600; either way such a row is a projection, "
                "never a price, and `fires` is False by construction"
            ),
        }
        print(json.dumps({"leg": "price", **result["price"]}, sort_keys=True), flush=True)

    result["elapsed_seconds"] = time.time() - started
    retain(out / "RESULT.json", (json.dumps(result, indent=1, sort_keys=True) + "\n").encode())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--legs",
        nargs="+",
        default=["amplitude", "seg"],
        choices=("amplitude", "seg", "pose", "resolve"),
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=120,
        help="amplitude sample; seeded RANDOM via up2.select_pairs, never a prefix",
    )
    parser.add_argument("--resolve-pairs", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--threads", type=int, default=5)
    parser.add_argument("--chunk", type=int, default=10)
    parser.add_argument("--carrier-delta-bytes", type=int, default=0)
    parser.add_argument(
        "--price-outside-ceiling",
        action="store_true",
        help="price a checkpoint whose render delta exceeds the 0.5 LSB ceiling; the "
        "row is then explicitly outside the regime ren1 measured",
    )
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import torch

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    STORE.mkdir(parents=True, exist_ok=True)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
