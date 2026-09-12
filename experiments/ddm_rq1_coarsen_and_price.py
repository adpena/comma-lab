"""ddm_rq1 Stage A: COARSEN the shipped renderer's realization and price the trade.

``ddm_ren2`` measured that a gradient refit of the deployed renderer at a fixed coded
field loses at every amplitude -- the coded field is the renderer's pre-image, so the
search has nowhere to go.  This producer inverts the order and asks the RATE question
first: the semantic member is 29,862 B of the 179,111 B archive, and its realization is
chosen by a per-tensor DEPTH TABLE and a ``keep_percent`` that both live IN the packed
bytes.  Coarsening them changes only the member's VALUES -- never the receiver, which
reads both fields out of the SM3R header it is handed.  No training happens here: the
SAME float weights are re-realized on a coarser grid.

Each variant is priced on two legs, cheapest and most falsifying first:

``member``   the REAL encode -- ``pack_prune_mixed_candidate`` -> ``SM1S`` -> ``CK2`` ->
             Brotli(q10, lgwin16), the container ``ddm_ren2`` identified by BYTE
             IDENTITY against the shipped stream.  The control row re-encodes the
             shipped state and must reproduce the shipped member byte-for-byte
             (29,862 B, sha ``786950a5...``); the producer STOPS if it does not, because
             a member delta from a look-alike encoder is not a price.
``seg``      exact n600 ``d_seg`` through R on the frozen CPU SegNet with the DALI GT
             lineage (``ddm_ren1``'s instrument, reached through ``ddm_jg1``), at the
             FIXED coded field.  The REALIZED state is scored: the candidate is packed
             and parsed back by the shipped receiver first, so what is measured is what
             would ship.  Reported with the moved/damaged/repaired cell decomposition
             against ``ddm_ren1``'s retained shipped argmax plane, split by GT class.

The trade is then explicit in ONE unit: rate bought (member bytes x 6.658589531221714e-7
S/B) against seg debt incurred (cells x 100/117,964,800 S).  A variant only earns a
Stage B -- where the token field is re-pre-distorted against the coarsened renderer and
asked to re-absorb the debt -- if the debt does not exceed the rate gain even at a 100 %
repair fraction.

Axis: ``[macOS-CPU advisory, jg1 instrument, DALI GT lineage, byte-exact encoder]``.
No score claim, no promotion, no archive is built here, no scorer weight is trained.
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
import ddm_ren2_price_checkpoint as ren2_price
import ddm_ren2_restore_init as ren2_init
import ddm_up2_shipping_pose_solve as up2
import numpy as np

AXIS = "[macOS-CPU advisory, jg1 instrument, DALI GT lineage, byte-exact encoder]"
STORE = Path("/Volumes/VertigoDataTier/pact/ddm_rq1")
RESERVE_BYTES = 40 << 30

#: Move 48 components, from the seal this arm is bound to (same pins as ``ddm_ren2``).
BASE_ARCHIVE_BYTES = ren2_price.BASE_ARCHIVE_BYTES
BASE_D_SEG = ren2_price.BASE_D_SEG
BASE_D_POSE = ren2_price.BASE_D_POSE
BASE_SCORE = ren2_price.BASE_SCORE
ADMIT_BAR_DELTA_S = ren2_price.ADMIT_BAR_DELTA_S
RATE_DENOMINATOR = ren2_price.RATE_DENOMINATOR

#: The exchange rate, re-derived at move 48 rather than imported as a constant.
BYTE_TO_S = 25.0 / RATE_DENOMINATOR
CELLS_TOTAL = ren2_price.CELLS_TOTAL
CELL_TO_S = 100.0 / CELLS_TOTAL

#: ``ddm_ren2``'s restored init: the shipped float weights plus the deployed
#: representation, proved byte-exact at all three encode stages.  READ ONLY.
RESTORED_INIT = Path("/Volumes/VertigoDataTier/pact/ddm_ren2/init/init_restored.pt")
RESTORED_INIT_SHA256 = (
    "ba51f59f3afd017baa2ac36a73493c8754d407fb3b734d8d712c609b4fd103b2"
)

#: The shipped member, pinned by BYTES so a drifted control refuses instead of
#: re-baselining the price on whatever the encoder happens to produce today.
SHIPPED_MEMBER_BYTES = 29_862
SHIPPED_MEMBER_SHA256_PREFIX = "786950a5"

#: ``ddm_ren1``'s retained n600 SegNet argmax plane for the SHIPPED render.
REN1_CONTROL_PLANES = ren2_price.REN1_CONTROL_PLANES

#: The SHIPPED renderer's n600 ``d_seg`` on THIS instrument, reproduced independently by
#: ``ddm_ren1`` and ``ddm_ren2`` to eleven figures.  The pointer's own T4 row is
#: 0.00010345, 0.06 % away; a variant's debt is quoted against BOTH so no cross-instrument
#: subtraction is hidden.  The control label re-measures this and must reproduce it.
INSTRUMENT_CONTROL_D_SEG = 0.00010338677300347222

#: The fourteen tensors the shipped table realizes at 4 bits, in packed order.  Read
#: off the shipped allocation at run time; this list is the declared VARIANT vocabulary
#: and is cross-checked against the measured table so a drift refuses.
HIGH_BIT_ORDER = (
    "token_embed.weight",
    "coord_mix.weight",
    "blocks.0.dw.weight",
    "blocks.0.pw.weight",
    "blocks.1.dw.weight",
    "blocks.1.pw.weight",
    "blocks.1.film.weight",
    "blocks.2.dw.weight",
    "blocks.2.pw.weight",
    "blocks.2.film.weight",
    "blocks.3.dw.weight",
    "blocks.3.pw.weight",
    "blocks.3.film.weight",
    "head.weight",
)

#: ``ddm_rw1``'s most sensitive tensors, named in the charter as variant (c).
RW1_SENSITIVE = ("head.weight", "blocks.3.pw.weight", "blocks.3.dw.weight")

#: Variants (f) and (g) are NOT in the charter's (a)-(e) list.  They are added for one
#: reason: the STOP rule's verdict is only family-scoped along the SIZE axis if the
#: cells-per-byte-bought ratio is measured over a range of rung sizes rather than
#: extrapolated from the charter's three.  (f) is the single largest 4-bit tensor and
#: (g) the single smallest, so the charter's rungs plus these two span roughly a 40x
#: range in coarsened values.  Declared as a SCOPE addition, not a mechanism change.
#:
#: (h) and (i) are the same axis read in the OTHER direction: the shipped object already
#: realizes `frame_embed.weight` and `blocks.0.film.weight` at THREE bits, so promoting
#: them to four SPENDS bytes to buy argmax cells.  `ddm_ren2` named this as `ddm_ntb2`'s
#: still-open prize and priced it from a DERIVED number; these rungs measure it on this
#: object, on the same instrument, in the same units as the coarsening rungs.  A
#: refinement rung is the only member-side action in this family that could clear the
#: bar, so it is measured rather than argued.


class Rq1Error(RuntimeError):
    """Refusal raised by this producer.  Never downgraded to a warning."""


def retain(path: Path, payload: bytes) -> dict[str, Any]:
    """Persist a payload inside this arm's store, refusing to fill the tier."""
    if not path.resolve().is_relative_to(STORE.resolve()):
        raise Rq1Error(f"write outside the rq1 store: {path}")
    if shutil.disk_usage(STORE.parent).free < RESERVE_BYTES + len(payload):
        raise Rq1Error("STORAGE_BLOCK: keep all payloads; free space below reserve")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)
    return ren2_init.fact(path)


def score_from_components(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    return ren2_price.score_from_components(d_seg, d_pose, archive_bytes)


def load_float_state() -> tuple[dict[str, Any], dict[str, Any]]:
    """``ddm_ren2``'s restored init, bound by sha.  These weights never change here."""
    import torch

    if ren2_init.sha256_file(RESTORED_INIT) != RESTORED_INIT_SHA256:
        raise Rq1Error(f"the restored init drifted: {RESTORED_INIT}")
    blob = torch.load(RESTORED_INIT, map_location="cpu", weights_only=False)
    state = {
        name: value.detach().cpu().clone() for name, value in blob["state_dict"].items()
    }
    provenance = {
        "init": ren2_init.fact(RESTORED_INIT),
        "deployed_representation": blob.get("deployed_representation"),
        "schema": blob.get("schema"),
    }
    return state, provenance


def variant_representation(
    shipped_representation: dict[str, Any], variant: str
) -> dict[str, Any]:
    """The variant's realization: the SHIPPED representation with the rung applied.

    Only ``bit_allocation`` and ``keep_percent`` move.  Both are fields the receiver
    reads out of the SM3R header, so a variant is a change of the member's VALUES and
    never of the receiver's code.
    """
    allocation = dict(shipped_representation["bit_allocation"])
    high = [name for name, bits in allocation.items() if bits == 4]
    if sorted(high) != sorted(HIGH_BIT_ORDER):
        raise Rq1Error(
            f"the shipped 4-bit set is {sorted(high)}; the declared variant vocabulary "
            f"is {sorted(HIGH_BIT_ORDER)} -- refusing rather than coarsening a set this "
            "producer has not been reviewed against"
        )
    keep_percent = int(shipped_representation["keep_percent"])
    if variant == "control":
        names: tuple[str, ...] = ()
    elif variant == "a_all14_q3":
        names = HIGH_BIT_ORDER
    elif variant == "b_largest8_q3":
        names = tuple(
            sorted(high, key=lambda name: -TENSOR_NUMEL[name])[:8]
        )
    elif variant == "c_head_block3_q3":
        names = RW1_SENSITIVE
    elif variant == "d_keep_percent_deeper":
        # Declared and measured in `film_prune_floor`: keep_percent is already at the
        # format's floor, so this rung is not expressible.  The caller refuses.
        names = ()
        keep_percent = keep_percent - 1
    elif variant == "e_scale_refit_control":
        names = ()
    elif variant == "f_largest_single_q3":
        names = (max(high, key=lambda name: TENSOR_NUMEL[name]),)
    elif variant == "g_smallest_single_q3":
        names = (min(high, key=lambda name: TENSOR_NUMEL[name]),)
    elif variant in ("h_both_low_bit_q4", "i_block0_film_q4"):
        low = sorted(name for name, bits in allocation.items() if bits == 3)
        if low != ["blocks.0.film.weight", "frame_embed.weight"]:
            raise Rq1Error(
                f"the shipped 3-bit set is {low}; the refinement rungs are reviewed only "
                "against ['blocks.0.film.weight', 'frame_embed.weight']"
            )
        promote = low if variant == "h_both_low_bit_q4" else ["blocks.0.film.weight"]
        for name in promote:
            allocation[name] = 4
        names = ()
    else:
        raise Rq1Error(f"unknown variant {variant!r}")
    for name in names:
        allocation[name] = 3
    representation = dict(shipped_representation)
    representation["bit_allocation"] = allocation
    representation["keep_percent"] = keep_percent
    return representation


#: Element counts of the quantized tensors, filled from the template at run time so the
#: "largest eight" selection is measured rather than hand-ranked.
TENSOR_NUMEL: dict[str, int] = {}


def fill_tensor_numel(template) -> None:
    TENSOR_NUMEL.clear()
    for name, value in template.items():
        TENSOR_NUMEL[name] = int(value.numel())


def measure_film_prune_floor(shipped) -> dict[str, Any]:
    """Enumerate every admissible ``keep_percent`` and report the realized row count.

    The charter's variant (d) asks for "keep_percent pruning one notch deeper on the
    FiLM rows".  Whether such a notch EXISTS is a property of the shipped format, not
    an opinion, so it is measured here from the packer's own rule rather than argued:
    ``keep = max(1, round(rows * keep_percent / 100))`` with ``keep_percent`` refused
    outside ``[1, 99]``.
    """
    import ddm_sm3_semantic_representation as sm3

    template = shipped["template"]
    rows_by_name = {
        name: int(template[name].shape[0]) for name in sorted(sm3.PRUNE_NAMES)
    }
    shipped_keep = int(shipped["representation"]["keep_percent"])
    ladder = []
    for keep_percent in range(1, 100):
        realized = {
            name: max(1, round(rows * keep_percent / 100.0))
            for name, rows in rows_by_name.items()
        }
        ladder.append({"keep_percent": keep_percent, "kept_rows": realized})
    shipped_rows = ladder[shipped_keep - 1]["kept_rows"]
    strictly_deeper = [
        row
        for row in ladder
        if all(row["kept_rows"][name] < shipped_rows[name] for name in rows_by_name)
    ]
    return {
        "rows_per_film_tensor": rows_by_name,
        "shipped_keep_percent": shipped_keep,
        "shipped_kept_rows": shipped_rows,
        "admissible_range": [1, 99],
        "packer_rule": "keep = max(1, round(rows * keep_percent / 100))",
        "packer_rule_source": "experiments/ddm_sm3_semantic_representation.py::pack_prune_mixed_candidate",
        "strictly_deeper_keep_percents": [row["keep_percent"] for row in strictly_deeper],
        "expressible": bool(strictly_deeper),
        "payload_bytes_at_shipped_depth": {
            name: rows * 0 + (shipped_rows[name] * int(template[name].shape[1]) * 4) // 8
            for name, rows in rows_by_name.items()
        },
        "note": (
            "each pruned FiLM tensor stores only its kept rows, so the whole family is "
            "a handful of bytes; even an expressible deeper notch could not buy a "
            "material rate gain"
        ),
    }


def encode_variant(state, shipped, representation) -> dict[str, Any]:
    """Real encode + parse-back through the shipped receiver.  Never an extrapolation."""
    import torch

    encoded = ren2_init.encode_member(
        state,
        representation=representation,
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
        raise Rq1Error("the variant rider is not a tagged variant the receiver reads")
    if set(parsed) != set(encoded["realized_state"]):
        raise Rq1Error("pack/parse key set drifted")
    for name in parsed:
        if not torch.equal(parsed[name], encoded["realized_state"][name]):
            raise Rq1Error(f"pack/parse tensor drifted: {name}")
    return {"encoded": encoded, "parsed": parsed}


def seg_leg(
    *,
    semantic,
    tokens,
    out: Path,
    resume: bool,
    chunk: int,
    started: float,
) -> dict[str, Any]:
    """Exact n600 ``d_seg`` through R, with the moved/damaged/repaired decomposition."""
    net = jg1.load_segnet()
    gt_labels = jg1.load_gt_seg_labels(up2.LINEAGE_DALI)
    if not REN1_CONTROL_PLANES.is_file():
        raise Rq1Error(
            "ddm_ren1's retained shipped argmax plane is absent; the cells-moved "
            "decomposition would be unmeasurable and this producer refuses to report "
            "a seg row without it"
        )
    with np.load(REN1_CONTROL_PLANES) as blob:
        shipped_argmax = blob["argmax_control"]
    rows_path = out / "seg_rows.jsonl"
    done: dict[int, dict[str, Any]] = {}
    if resume and rows_path.is_file():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[int(row["pair"])] = row
    with rows_path.open("a", encoding="utf-8") as stream:
        for start in range(0, jg1.N_PAIRS, chunk):
            span = np.arange(start, min(start + chunk, jg1.N_PAIRS), dtype=np.int64)
            pending = np.array([p for p in span if int(p) not in done], dtype=np.int64)
            if pending.size == 0:
                continue
            frames = jg1.render_frame1(semantic, tokens[pending], pending)
            argmax = jg1.argmax_from_camera_frames(net, frames)
            per_pair = jg1.d_seg_per_pair(argmax, gt_labels[pending])
            for index, pair in enumerate(pending):
                gt = gt_labels[int(pair)]
                base = shipped_argmax[int(pair)]
                cand = argmax[index]
                moved = cand != base
                base_wrong = base != gt
                cand_wrong = cand != gt
                damaged = moved & ~base_wrong & cand_wrong
                repaired = moved & base_wrong & ~cand_wrong
                classes = np.bincount(
                    gt[damaged].astype(np.int64), minlength=5
                ).tolist()
                row = {
                    "pair": int(pair),
                    "d_seg": float(per_pair[index]),
                    "cells_moved": int(moved.sum()),
                    "cells_damaged": int(damaged.sum()),
                    "cells_repaired": int(repaired.sum()),
                    "damaged_by_gt_class": classes,
                }
                done[int(pair)] = row
                stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(
                json.dumps(
                    {
                        "leg": "seg",
                        "done": len(done),
                        "of": jg1.N_PAIRS,
                        "running_mean": float(
                            np.mean([r["d_seg"] for r in done.values()])
                        ),
                        "seconds": round(time.time() - started, 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    ordered = [done[p] for p in range(jg1.N_PAIRS)]
    d_seg = float(np.mean([r["d_seg"] for r in ordered]))
    class_split = np.zeros(5, dtype=np.int64)
    for row in ordered:
        class_split += np.asarray(row["damaged_by_gt_class"], dtype=np.int64)
    return {
        "pairs": jg1.N_PAIRS,
        "d_seg": d_seg,
        "wrong_cells": d_seg * CELLS_TOTAL,
        "cells_moved": int(sum(r["cells_moved"] for r in ordered)),
        "cells_damaged": int(sum(r["cells_damaged"] for r in ordered)),
        "cells_repaired": int(sum(r["cells_repaired"] for r in ordered)),
        "damaged_by_gt_class": class_split.tolist(),
        "class_names": ["road", "lane", "undrivable", "movable", "mycar"],
        "cells_moved_source": str(REN1_CONTROL_PLANES),
        "rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
    }


def run(args) -> int:
    started = time.time()
    out = STORE / "stageA" / args.label
    out.mkdir(parents=True, exist_ok=True)
    pointer = ren2_init.bind_pointer()
    shipped = ren2_price.load_shipped()
    fill_tensor_numel(shipped["template"])
    state, init_provenance = load_float_state()

    if args.variant == "d_keep_percent_deeper":
        floor = measure_film_prune_floor(shipped)
        retain(
            out / "FILM_PRUNE_FLOOR.json",
            (json.dumps(floor, indent=2, sort_keys=True) + "\n").encode(),
        )
        print(json.dumps({"leg": "film_prune_floor", **floor}, sort_keys=True), flush=True)
        raise Rq1Error(
            "variant (d) is NOT EXPRESSIBLE in the shipped format -- MEASURED, see "
            f"{out / 'FILM_PRUNE_FLOOR.json'}: the shipped keep_percent is already the "
            "format's minimum and no admissible value realizes fewer kept rows. "
            "Deepening the FiLM prune is a RECEIVER change, which this arm may not make."
        )
    representation = variant_representation(shipped["representation"], args.variant)

    # The byte-identity gate runs on EVERY invocation, not only on the control label:
    # a variant's member delta is only a price if the same encoder reproduces the
    # shipped member exactly on the shipped realization.
    control_encode = encode_variant(
        state, shipped, shipped["representation"]
    )["encoded"]
    realized = encode_variant(state, shipped, representation)
    encoded = realized["encoded"]
    member_delta = encoded["member_bytes"] - shipped["member_bytes"]
    archive_bytes = BASE_ARCHIVE_BYTES + member_delta

    if (
        control_encode["member_bytes"] != SHIPPED_MEMBER_BYTES
        or not control_encode["member_sha256"].startswith(SHIPPED_MEMBER_SHA256_PREFIX)
        or control_encode["member"] != shipped["container"]["member"]
    ):
        raise Rq1Error(
            "STOP: the control encode did not reproduce the shipped member "
            f"({control_encode['member_bytes']} B, sha "
            f"{control_encode['member_sha256'][:8]}); expected {SHIPPED_MEMBER_BYTES} B "
            f"sha {SHIPPED_MEMBER_SHA256_PREFIX}. Without byte identity the member "
            "pricer is a look-alike and every variant's delta is unpriced."
        )

    binding: dict[str, Any] = {
        "schema": "ddm_rq1_coarsen_and_price.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "label": args.label,
        "variant": args.variant,
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
            "byte_to_s": BYTE_TO_S,
            "cell_to_s": CELL_TO_S,
        },
        "init": init_provenance,
        "control_member_identity": {
            "bytes": control_encode["member_bytes"],
            "sha256": control_encode["member_sha256"],
            "byte_identical_to_shipped": True,
        },
        "realization": {
            "keep_percent": representation["keep_percent"],
            "bit_allocation": representation["bit_allocation"],
            "coarsened_names": sorted(
                name
                for name, bits in representation["bit_allocation"].items()
                if bits != shipped["representation"]["bit_allocation"][name]
            ),
        },
        "member": {
            "bytes": encoded["member_bytes"],
            "sha256": encoded["member_sha256"],
            "delta_bytes": member_delta,
            "body_bytes": encoded["body_bytes"],
            "body_sha256": encoded["body_sha256"],
            "rider_bytes": encoded["rider_bytes"],
            "archive_bytes_if_shipped": archive_bytes,
            "rate_gain_s": -member_delta * BYTE_TO_S,
            "rate_gain_in_admit_bars": -member_delta * BYTE_TO_S / -ADMIT_BAR_DELTA_S,
            "debt_budget_cells_at_full_repair": -member_delta * BYTE_TO_S / CELL_TO_S,
            "kept_rows": encoded["pack_meta"]["kept_rows"],
            "container": {
                "brotli_quality": shipped["brotli_quality"],
                "brotli_lgwin": shipped["brotli_lgwin"],
                "ck2_plane2": shipped["ck2"],
            },
        },
        "producer": ren2_init.fact(Path(__file__)),
        "restore_module": ren2_init.fact(Path(ren2_init.__file__)),
        "price_module": ren2_init.fact(Path(ren2_price.__file__)),
        "jg1": ren2_init.fact(Path(jg1.__file__)),
        "threads": args.threads,
    }

    previous_inputs = out / "INPUTS.json"
    if args.resume and previous_inputs.is_file():
        previous = json.loads(previous_inputs.read_text())
        previous_member = (previous.get("member") or {}).get("sha256")
        if previous_member and previous_member != encoded["member_sha256"]:
            raise Rq1Error(
                f"label {args.label!r} already holds rows for member {previous_member}; "
                f"this invocation is {encoded['member_sha256']}. Use a new --label or "
                "--no-resume; resumed rows from another realization would be priced as "
                "this one's."
            )
    retain(previous_inputs, (json.dumps(binding, indent=2, sort_keys=True) + "\n").encode())
    # ALWAYS KEEP THE PAYLOAD: this producer holds the real encoded member in memory and
    # would otherwise persist only its length and sha, which is the measure-and-discard
    # class CLAUDE.md forbids at the typing moment.  All three encode stages are kept, so
    # a successor can price, parse back, or splice this realization without re-encoding.
    binding["retained_payloads"] = {
        "body": retain(out / "body.sm3r", encoded["body"]),
        "rider": retain(out / "rider.sm1s", encoded["rider"]),
        "member": retain(out / "member.rx1m", encoded["member"]),
    }
    retain(previous_inputs, (json.dumps(binding, indent=2, sort_keys=True) + "\n").encode())
    result: dict[str, Any] = dict(binding)
    retain(out / "RESULT.json", (json.dumps(result, indent=1, sort_keys=True) + "\n").encode())
    print(json.dumps({"leg": "member", **result["member"]}, sort_keys=True), flush=True)

    if "seg" in args.legs:
        tokens = jg1.load_tokens(ren2_init.TOKEN_FIELD)
        semantic = ren2_price.renderer_with_state(shipped, realized["parsed"])
        seg = seg_leg(
            semantic=semantic,
            tokens=tokens,
            out=out,
            resume=args.resume,
            chunk=args.chunk,
            started=started,
        )
        debt_cells = (seg["d_seg"] - INSTRUMENT_CONTROL_D_SEG) * CELLS_TOTAL
        seg["instrument_control_d_seg"] = INSTRUMENT_CONTROL_D_SEG
        seg["pointer_t4_d_seg"] = BASE_D_SEG
        seg["net_cells_vs_instrument_control"] = -debt_cells
        seg["debt_s"] = (seg["d_seg"] - INSTRUMENT_CONTROL_D_SEG) * 100.0
        seg["debt_s_vs_pointer_t4"] = (seg["d_seg"] - BASE_D_SEG) * 100.0
        seg["cells_damaged_per_byte_bought"] = (
            seg["cells_damaged"] / -member_delta if member_delta < 0 else None
        )
        seg["cells_per_byte_the_exchange_allows"] = CELL_TO_S and BYTE_TO_S / CELL_TO_S
        # The whole-score projection, recomputed from components.  Pose is HELD at the
        # pointer's value: a render change moves pose too and the terminal carrier
        # re-solve is not run on a row its own cheapest binding term has refused, so this
        # is an OPTIMISTIC projection on the pose axis and is labelled as one.
        seg["projected_score_zero_repair_pose_held"] = score_from_components(
            seg["d_seg"], BASE_D_POSE, archive_bytes
        )
        seg["projected_delta_s_vs_move48_pose_held"] = (
            seg["projected_score_zero_repair_pose_held"] - BASE_SCORE
        )
        seg["projection_note"] = (
            "pose HELD at the move-48 value, so the projection is optimistic on the "
            "pose axis; a realization change moves the render and therefore the pose, "
            "and no terminal carrier re-solve was spent on a refused row"
        )
        seg["net_delta_s_zero_repair"] = seg["debt_s"] + member_delta * BYTE_TO_S
        seg["net_delta_s_full_repair"] = member_delta * BYTE_TO_S
        seg["repair_fraction_to_break_even"] = (
            1.0 + (member_delta * BYTE_TO_S) / seg["debt_s"]
            if seg["debt_s"] > 0
            else None
        )
        seg["repair_fraction_to_clear_bar"] = (
            1.0 + (member_delta * BYTE_TO_S - ADMIT_BAR_DELTA_S) / seg["debt_s"]
            if seg["debt_s"] > 0
            else None
        )
        seg["instrument_note"] = (
            "ddm_ren1's control on THIS instrument is d_seg 0.00010338677 against the "
            "pointer's T4 0.00010345 (0.06 %); a delta smaller than that reproduction "
            "gap is not resolvable here"
        )
        result["seg"] = seg
        retain(
            out / "RESULT.json",
            (json.dumps(result, indent=1, sort_keys=True) + "\n").encode(),
        )
        print(json.dumps({"leg": "seg", **seg}, sort_keys=True), flush=True)

    result["elapsed_seconds"] = time.time() - started
    retain(out / "RESULT.json", (json.dumps(result, indent=1, sort_keys=True) + "\n").encode())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--variant",
        required=True,
        choices=(
            "control",
            "a_all14_q3",
            "b_largest8_q3",
            "c_head_block3_q3",
            "d_keep_percent_deeper",
            "e_scale_refit_control",
            "f_largest_single_q3",
            "g_smallest_single_q3",
            "h_both_low_bit_q4",
            "i_block0_film_q4",
        ),
    )
    parser.add_argument("--legs", nargs="+", default=["member"], choices=("member", "seg"))
    parser.add_argument("--chunk", type=int, default=10)
    parser.add_argument("--threads", type=int, default=5)
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
