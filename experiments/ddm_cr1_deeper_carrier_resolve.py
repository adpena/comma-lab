"""ddm_cr1 -- the DEEPER per-pair carrier re-solve on all 600 pairs of move 49.

pc3 named this "ITEM 4, unowned"; ``ddm_cb1`` reproduced it on move 49 as its keep-12
CONTROL and measured **0.99679** on a seeded-random n48 at **zero** archive bytes.  That
control is this arm's reference form, reused verbatim rather than re-written:

* solver: ``ddm_jg5.refine_pair`` -- alternating damped Gauss-Newton on the int12
  coefficient lattice and a +-2 polish, stopping on the pair's own projected-gain
  materiality floor or on a physical refusal.  ``outer_rounds=40``,
  ``max_gn_iterations=400``, ``dd_threshold = jg5.materiality_dd_threshold(base_mean)``
  -- cb1's ``resolve`` defaults, read off its launch manifests.
* instrument: ``pp1.open_live_raw`` / ``pp1.build_pose_instrument`` on MOVE 49's own
  cold decode, with the pointer sha asserted before every stage (cb1 section 3's
  instrument correction -- ``sj1.load_pose_instrument(None)`` defaults to the **cl2**
  parse-back, two bodies behind, and reads pair 0 2,878x high).

**The one declared delta from cb1's control, and it is deliberate.**  cb1 warm-started
from the least-squares projection of the shipped field onto the candidate span, because
its candidate spans were DIFFERENT bases and a projection is the only common start.  This
arm's basis IS the shipped basis, so the charter's start is the **shipped coefficient
codes themselves**.  That start is strictly better than the re-projection (which
re-quantises and can land a lattice step away) and it makes ``refine_pair`` monotone
from the base: ``best`` only ever decreases, so per pair ``final <= start`` by
construction and no pair can be made worse.

Acceptance is on the RESOLVED pose under pp1's MEASURED tolerance: a pair is accepted
only when ``start - final`` exceeds the batch-1 vs batch-8 reproduction band
(**2.586e-09** absolute, pp1 section 10, 16 pairs), so no accepted improvement is
smaller than the measurement order can move the authoritative batch-8 n600 read.

Bytes are EXACT: the solver moves int12 coefficient CODES, which the receiver's Rice /
DX2 chain re-encodes, so "zero bytes" is a claim about the CONTAINER and is measured by
building the real ``archive.zip`` through ``ddm_up3.build_archive`` with the twins
control (repacking move 49's own codes must return ``73e41a66...`` at 179,153 B) passing
on the same call.

Modes
-----
``base``      reproduce move 49's n600 pose base, the twins control, and the shipped codes.
``resolve``   the sharded n600 deeper re-solve from the shipped codes.
``assemble``  merge shards, apply the acceptance gate, re-measure n600 at batch 8, and
              build the exact candidate archive.

No Modal, no fire, no packet.  Every pose number is
``[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI-lineage GT, n600]``; bytes are EXACT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

import ddm_br1_pose_basis_reorientation as br1
import ddm_cb1_carrier_basis_refit as cb1
import ddm_jg5_pose_resolve_on_edited_renders as jg5
import ddm_pp1_pose_actuation as pp1
import ddm_sj1_multipass_token_predistortion as sj1
import ddm_up2_shipping_pose_solve as up2

N_PAIRS = up2.N_PAIRS_TOTAL
CARRIER_DIM = up2.CARRIER_DIM

POINTER = sj1.LIVE_POINTER
POINTER_TREE = POINTER.tree
POINTER_ARCHIVE_SHA256 = POINTER.archive_sha256
POINTER_ARCHIVE_BYTES = POINTER.archive_bytes
POINTER_SCORE_T4 = POINTER.score_t4
POINTER_D_SEG_T4 = POINTER.d_seg_t4
POINTER_D_POSE_T4 = POINTER.d_pose_t4

#: pp1's own n600 base on move 49 -- REPRODUCED here, never inherited.
PP1_BASE_VECTOR = Path("/Volumes/APDataStore/pact/ddm_pp1/base/pose_base_move49.npy")
PP1_BASE_MEAN = 4.543568679770593e-06
#: pp1 section 10, MEASURED: batch-1 vs batch-8 moves the per-pair pose vector by at most
#: 2.586e-09 ABSOLUTE over 16 pairs.
POSE_BATCH_BAND_ABS = 2.586e-09
#: the BASE reproduction gate is 10x the observed maximum (cb1's C1).
POSE_BASE_GATE_ABS = 10.0 * POSE_BATCH_BAND_ABS
#: the per-pair ACCEPTANCE tolerance is the observed maximum itself: an accepted gain
#: must be bigger than the thing that could have produced it by re-ordering a batch.
POSE_ACCEPT_TOL_ABS = POSE_BATCH_BAND_ABS

RATE_PER_BYTE = 25.0 / 37_545_489.0
ADMIT_BAR = -2e-05

#: cb1's ``resolve`` defaults, read off ``stage_res_12_*/launch_manifest.json``.
OUTER_ROUNDS = 40
MAX_GN_ITERATIONS = 400

WORK = Path(os.environ.get("CR1_WORK", "/Volumes/VertigoDataTier/pact/ddm_cr1"))


class Cr1Error(RuntimeError):
    """A ddm_cr1 precondition failed. Fail closed; never approximate."""


# ---------------------------------------------------------------------------
# housekeeping
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_threads(threads: int) -> None:
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = str(threads)
    import torch

    torch.set_num_threads(threads)


def assert_pointer() -> dict[str, Any]:
    """Refuse every stage that is not standing on move 49's own shipped bytes."""
    observed = sha256_file(POINTER.archive)
    if observed != POINTER_ARCHIVE_SHA256:
        raise Cr1Error(
            f"pointer tree {POINTER_TREE} has archive sha {observed}, not move 49's "
            f"{POINTER_ARCHIVE_SHA256}"
        )
    size = POINTER.archive.stat().st_size
    if size != POINTER_ARCHIVE_BYTES:
        raise Cr1Error(f"pointer archive is {size} B, not {POINTER_ARCHIVE_BYTES}")
    up2.verify_gt_lineage(axis="contest_cuda", declared_lineage=up2.LINEAGE_DALI)
    return {
        "pointer_tree": str(POINTER_TREE),
        "pointer_archive_sha256": observed,
        "pointer_archive_bytes": size,
        "pointer_score_t4": POINTER_SCORE_T4,
        "pointer_d_seg_t4": POINTER_D_SEG_T4,
        "pointer_d_pose_t4": POINTER_D_POSE_T4,
        "gt_lineage": up2.LINEAGE_DALI,
        "solver": "ddm_jg5.refine_pair (cb1 keep-12 control configuration)",
        "outer_rounds": OUTER_ROUNDS,
        "max_gn_iterations": MAX_GN_ITERATIONS,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable))


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON-serialisable: {type(value)!r}")


def composed_score(d_seg: float, d_pose: float, archive_bytes: float) -> float:
    return 100.0 * d_seg + jg5.pose_leg(d_pose) + 25.0 * archive_bytes / 37_545_489.0


# ---------------------------------------------------------------------------
# stage: base
# ---------------------------------------------------------------------------


def cmd_base(args) -> int:
    """Reproduce move 49's n600 pose base and repack its carrier byte-identically."""
    set_threads(args.threads)
    receipts = assert_pointer()
    import ddm_up3_carrier_splice as splice

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    started = time.time()
    raw = pp1.open_live_raw()
    inst = pp1.build_pose_instrument(raw)
    codes = np.asarray(inst.state.codes, dtype=np.int32)
    if codes.shape != (N_PAIRS, CARRIER_DIM):
        raise Cr1Error(f"shipped coefficient codes have shape {codes.shape}")
    per_pair = cb1.n600_pose(inst, codes, batch_size=args.batch_size)
    elapsed = time.time() - started

    reference = np.load(PP1_BASE_VECTOR)
    abs_gap = np.abs(per_pair - reference)
    if abs_gap.max() > POSE_BASE_GATE_ABS:
        raise Cr1Error(
            f"C1 base reproduction max |gap| {abs_gap.max():.3e} exceeds pp1's measured "
            f"band gate {POSE_BASE_GATE_ABS:.3e}"
        )

    body = splice.parse_shipped_body(POINTER_TREE, verify_sha=False)
    body_codes = np.asarray(body.codes, dtype=np.int32)
    if not np.array_equal(body_codes, codes):
        raise Cr1Error(
            "C3 FAILED: the carrier codes the instrument renders from are not the codes "
            "the archive stores"
        )
    rebuilt = splice.build_archive(
        body, body_codes, runtime_dir=POINTER_TREE, container_search=True, verify=True
    )
    if rebuilt["archive_sha256"] != POINTER_ARCHIVE_SHA256:
        raise Cr1Error(
            f"C2 TWINS FAILED: repacking move 49's own carrier gives "
            f"{rebuilt['archive_sha256']} ({rebuilt['archive_size']} B)"
        )

    np.save(out / "pose_base_move49.npy", per_pair)
    np.save(out / "coefficient_codes_shipped.npy", codes)

    report = {
        "schema": "ddm_cr1_base.v1",
        "receipts": receipts,
        "pairs": int(N_PAIRS),
        "d_pose_mean": float(per_pair.mean()),
        "d_pose_median": float(np.median(per_pair)),
        "d_pose_max": float(per_pair.max()),
        "pose_leg": jg5.pose_leg(float(per_pair.mean())),
        "C1_base_reproduction": {
            "pp1_reference_mean": PP1_BASE_MEAN,
            "max_abs_gap": float(abs_gap.max()),
            "gate_abs": POSE_BASE_GATE_ABS,
            "band_source": "ddm_pp1 section 10 (batch-1 vs batch-8, 16 pairs)",
            "passed": True,
        },
        "C2_twins": {
            "rebuilt_sha256": rebuilt["archive_sha256"],
            "rebuilt_bytes": int(rebuilt["archive_size"]),
            "byte_identical": True,
        },
        "C3_codes_identity": {
            "instrument_codes_equal_archive_codes": True,
            "codes_sha256": hashlib.sha256(
                np.ascontiguousarray(codes).tobytes()
            ).hexdigest(),
        },
        "instrument_ratio_vs_t4_print": float(per_pair.mean()) / POINTER_D_POSE_T4,
        "materiality_dd_threshold": jg5.materiality_dd_threshold(float(per_pair.mean())),
        "elapsed_seconds": elapsed,
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI-lineage GT, n600]",
        "score_claim": False,
        "promotable": False,
    }
    write_json(out / "BASE.json", report)
    print(json.dumps({k: report[k] for k in ("d_pose_mean", "pose_leg", "elapsed_seconds")}))
    return 0


# ---------------------------------------------------------------------------
# stage: resolve
# ---------------------------------------------------------------------------


def cmd_resolve(args) -> int:
    """The deeper per-pair re-solve, from the SHIPPED codes, on the shipped basis."""
    set_threads(args.threads)
    receipts = assert_pointer()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    base_vector = np.load(args.base_pose)
    shipped = np.load(args.shipped_codes).astype(np.int32)
    raw = pp1.open_live_raw()
    inst = pp1.build_pose_instrument(raw)
    if args.start_codes:
        shipped = np.load(args.start_codes).astype(np.int32)

    pairs = np.arange(N_PAIRS, dtype=np.int64)
    if args.pairs_file:
        pairs = np.asarray(json.loads(Path(args.pairs_file).read_text()), dtype=np.int64)
    shard = pairs[args.shard_index :: args.shard_count]
    if args.overlay_dir:
        #: the overlay is restricted to exactly the pairs being solved, so no pair outside
        #: the composed subset can be read through a render the candidate does not carry.
        inst = _overlay_instrument(inst, Path(args.overlay_dir), keep=pairs.tolist())

    dd_threshold = jg5.materiality_dd_threshold(float(base_vector.mean()))
    rows: list[dict[str, Any]] = []
    started = time.time()
    rows_path = out / f"rows_{args.shard_index:02d}_of_{args.shard_count:02d}.jsonl"
    #: resumable from disk (P0): a shard that died mid-way re-reads its own rows and
    #: continues.  The rows are the checkpoint -- each is flushed as it lands -- so there
    #: is no separate state to keep consistent with them.
    if rows_path.is_file():
        for line in rows_path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    done = {int(r["pair"]) for r in rows}
    with open(rows_path, "a") as handle:
        for pair in (int(p) for p in shard):
            if pair in done:
                continue
            row = jg5.refine_pair(
                inst,
                pair,
                shipped[pair],
                dd_threshold=dd_threshold,
                outer_rounds=OUTER_ROUNDS,
                max_gn_iterations=MAX_GN_ITERATIONS,
            )
            row["pair"] = pair
            row["base_d_pose"] = float(base_vector[pair])
            row["shipped_codes"] = shipped[pair].tolist()
            #: monotone by construction; asserted so a solver change cannot slip past.
            if row["final_d_pose"] > row["start_d_pose"]:
                raise Cr1Error(
                    f"pair {pair}: refine_pair returned a WORSE value "
                    f"{row['final_d_pose']:.6e} > {row['start_d_pose']:.6e}"
                )
            rows.append(row)
            handle.write(json.dumps(row, default=_jsonable) + "\n")
            handle.flush()
            print(
                f"pair {pair}: base {base_vector[pair]:.6e} start {row['start_d_pose']:.6e} "
                f"final {row['final_d_pose']:.6e} dc {row['changed_coordinates']} "
                f"({row.get('stop_reason')}) {time.time() - started:.0f}s",
                flush=True,
            )

    starts = np.array([r["start_d_pose"] for r in rows], dtype=np.float64)
    finals = np.array([r["final_d_pose"] for r in rows], dtype=np.float64)
    gains = starts - finals
    write_json(
        out / f"RESOLVE_{args.shard_index:02d}_of_{args.shard_count:02d}.json",
        {
            "schema": "ddm_cr1_resolve_shard.v1",
            "receipts": receipts,
            "warm_start": "SHIPPED coefficient codes (charter); cb1 used the LS projection",
            "overlay_dir": args.overlay_dir,
            "pairs_solved": len(rows),
            "pairs_improved_any": int((gains > 0.0).sum()),
            "pairs_improved_above_band": int((gains > POSE_ACCEPT_TOL_ABS).sum()),
            "gain_sum": float(gains.sum()),
            "subset_start_mean": float(starts.mean()),
            "subset_final_mean": float(finals.mean()),
            "dd_threshold": dd_threshold,
            "elapsed_seconds": time.time() - started,
            "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI GT]",
            "score_claim": False,
        },
    )
    print(
        json.dumps(
            {
                "pairs": len(rows),
                "improved_any": int((gains > 0.0).sum()),
                "improved_above_band": int((gains > POSE_ACCEPT_TOL_ABS).sum()),
                "gain_sum": float(gains.sum()),
            }
        )
    )
    return 0


def _overlay_instrument(inst, overlay_dir: Path, keep: Sequence[int] | None = None):
    """br1's instrument with pass 8's re-rendered ODD frames in place of the shipped ones.

    ``keep`` RESTRICTS the substitution to the named pairs.  Pass 8's overlay was rendered
    from its FULL 96-pair edit field, but the composed object carries only the ADMITTED
    subset, so every pair outside ``keep`` must fall through to move 49's own decode.
    Without the restriction the re-solve would silently run against 85 pairs of renders the
    candidate does not contain -- the stale-partner failure, wearing a correct-looking
    manifest.  The admitted planes were MEASURED identical to the full-edit planes on all
    11 kept pairs, which is what makes reusing the big overlay legitimate at all.
    """
    import ddm_sj1_joint_admission as joint

    overlay = joint.open_overlay(Path(pp1.LIVE_RAW), overlay_dir)
    if keep is not None:
        wanted = {int(p) for p in keep}
        missing = sorted(wanted - set(overlay.index))
        if missing:
            raise Cr1Error(f"overlay {overlay_dir} does not carry pairs {missing}")
        overlay.index = {p: s for p, s in overlay.index.items() if p in wanted}
    return br1.Instrument(
        inst.state, overlay, inst.targets, inst.posenet, inst.blow, inst.gram, inst.bmat
    )


# ---------------------------------------------------------------------------
# stage: assemble
# ---------------------------------------------------------------------------


def merge_rows(rows_dirs: Sequence[Path]) -> dict[int, dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for rows_dir in rows_dirs:
        for path in sorted(Path(rows_dir).glob("rows_*.jsonl")):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                merged[int(row["pair"])] = row
    return merged


def cmd_assemble(args) -> int:
    """Gate the per-pair gains, re-measure n600 at batch 8, and price the exact bytes."""
    set_threads(args.threads)
    receipts = assert_pointer()
    import ddm_up3_carrier_splice as splice

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    base_vector = np.load(args.base_pose)
    shipped = np.load(args.shipped_codes).astype(np.int32)
    merged = merge_rows([Path(d) for d in args.rows_dir])
    missing = [p for p in range(N_PAIRS) if p not in merged]
    if missing and not args.allow_partial:
        raise Cr1Error(f"{len(missing)} pairs missing from the shards: {missing[:8]}")

    candidate = shipped.copy()
    accepted: list[int] = []
    gains: list[float] = []
    #: a pair whose FINAL codes equal its START codes cannot have improved: the same
    #: codes through the same receiver are the same render.  A positive "gain" there is
    #: the batch-order artifact ``evaluate_codes`` carries (``best`` is first read at
    #: batch 1 and later blocks are read at batch 10-32), so it is REPORTED as the
    #: measured artifact and never accepted.  Without this the acceptance count reads
    #: high while the archive is byte-identical.
    phantom: list[dict[str, Any]] = []
    per_pair_gain = np.zeros(N_PAIRS, dtype=np.float64)
    for pair, row in sorted(merged.items()):
        gain = float(row["start_d_pose"]) - float(row["final_d_pose"])
        per_pair_gain[pair] = gain
        moved = not np.array_equal(
            np.asarray(row["codes"], dtype=np.int32), shipped[pair]
        )
        if gain > POSE_ACCEPT_TOL_ABS and moved:
            candidate[pair] = np.asarray(row["codes"], dtype=np.int32)
            accepted.append(pair)
            gains.append(gain)
        elif gain > 0.0 and not moved:
            phantom.append({"pair": pair, "gain": gain})

    changed_pairs = int((candidate != shipped).any(axis=1).sum())
    changed_coords = int((candidate != shipped).sum())

    raw = pp1.open_live_raw()
    inst = pp1.build_pose_instrument(raw)
    started = time.time()
    resolved = cb1.n600_pose(inst, candidate, batch_size=args.batch_size)
    measure_seconds = time.time() - started

    #: identity control: the UNCHANGED pairs must reproduce the base exactly, batch 8
    #: against batch 8, so any movement there is the measurement and not the re-solve.
    unchanged = np.array(
        [p for p in range(N_PAIRS) if not (candidate[p] != shipped[p]).any()],
        dtype=np.int64,
    )
    unchanged_gap = (
        float(np.abs(resolved[unchanged] - base_vector[unchanged]).max())
        if len(unchanged)
        else 0.0
    )
    #: the AUTHORITATIVE per-pair check.  ``refine_pair`` seeds ``best`` at batch 1 and
    #: then reads candidate blocks at batch 10-32, so an in-loop "gain" can be the batch
    #: order.  Here every accepted pair is re-read at the SAME batch 8 as its own base,
    #: which is the only paired comparison the n600 mean is actually made of.
    accepted_verified = [
        {
            "pair": int(pair),
            "base_d_pose": float(base_vector[pair]),
            "resolved_d_pose": float(resolved[pair]),
            "batch8_gain": float(base_vector[pair] - resolved[pair]),
            "in_loop_gain": float(per_pair_gain[pair]),
        }
        for pair in accepted
    ]

    body = splice.parse_shipped_body(POINTER_TREE, verify_sha=False)
    identity = splice.build_archive(
        body,
        np.asarray(body.codes, dtype=np.int32),
        runtime_dir=POINTER_TREE,
        container_search=True,
        verify=True,
    )
    if identity["archive_sha256"] != POINTER_ARCHIVE_SHA256:
        raise Cr1Error("TWINS FAILED: the identity repack does not reproduce move 49")
    built = splice.build_archive(
        body, candidate, runtime_dir=POINTER_TREE, container_search=True, verify=True
    )
    archive_path = out / "candidate_archive.zip"
    archive_path.write_bytes(built["archive_bytes"])
    proof = jg5.section_identity(
        built["archive_bytes"], body.archive_bytes, runtime=POINTER_TREE
    )
    if not proof["frame1_sections_all_identical"]:
        raise Cr1Error(f"a frame-1 section moved ({proof}); the seg leg cannot be carried")

    archive_bytes = int(built["archive_size"])
    delta_bytes = archive_bytes - POINTER_ARCHIVE_BYTES
    base_mean = float(base_vector.mean())
    resolved_mean = float(resolved.mean())

    #: LIKE-FOR-LIKE: this instrument's own base against this instrument's own resolved
    #: value, which is the only pose difference that transfers to a T4 fire.
    like_for_like = (
        jg5.pose_leg(resolved_mean) - jg5.pose_leg(base_mean) + delta_bytes * RATE_PER_BYTE
    )
    #: NOMINAL: the convention pass 8's admission used -- the local pose against the T4
    #: PRINT 4.55e-06.  It carries the print's rounding as a phantom credit and is
    #: reported only for continuity with that ledger.
    nominal = (
        composed_score(POINTER_D_SEG_T4, resolved_mean, archive_bytes) - POINTER_SCORE_T4
    )

    order = np.argsort(per_pair_gain)[::-1]
    report = {
        "schema": "ddm_cr1_assemble.v1",
        "receipts": receipts,
        "acceptance": {
            "tolerance_abs": POSE_ACCEPT_TOL_ABS,
            "tolerance_source": "ddm_pp1 section 10, batch-1 vs batch-8 max |gap|, 16 pairs",
            "pairs_solved": len(merged),
            "pairs_improved_any": int((per_pair_gain > 0.0).sum()),
            "pairs_accepted": len(accepted),
            "accepted_pairs": accepted,
            "gain_sum_accepted": float(sum(gains)),
            "gain_sum_all": float(per_pair_gain.sum()),
            "identical_codes_with_positive_gain": phantom,
            "identical_codes_gain_max": (
                max((p["gain"] for p in phantom), default=0.0)
            ),
            "top10_gains": [
                {"pair": int(p), "gain": float(per_pair_gain[p])} for p in order[:10]
            ],
        },
        "codes": {
            "changed_pairs": changed_pairs,
            "changed_coordinates": changed_coords,
            "candidate_sha256": hashlib.sha256(
                np.ascontiguousarray(candidate).tobytes()
            ).hexdigest(),
        },
        "pose": {
            "base_mean_n600": base_mean,
            "resolved_mean_n600": resolved_mean,
            "delta": resolved_mean - base_mean,
            "fraction_of_base": (resolved_mean - base_mean) / base_mean,
            "base_leg": jg5.pose_leg(base_mean),
            "resolved_leg": jg5.pose_leg(resolved_mean),
            "dS_pose_like_for_like": jg5.pose_leg(resolved_mean) - jg5.pose_leg(base_mean),
            "batch_size": args.batch_size,
            "measure_seconds": measure_seconds,
            "unchanged_pairs": len(unchanged),
            "unchanged_max_abs_gap_vs_base": unchanged_gap,
            "unchanged_gap_within_band": bool(unchanged_gap <= POSE_BATCH_BAND_ABS),
            "accepted_pairs_verified_at_batch8": accepted_verified,
        },
        "rate": {
            "identity_control_sha256": identity["archive_sha256"],
            "candidate_sha256": built["archive_sha256"],
            "candidate_bytes": archive_bytes,
            "delta_bytes": delta_bytes,
            "dS_rate": delta_bytes * RATE_PER_BYTE,
            "archive_path": str(archive_path),
        },
        "frame1_section_identity": proof,
        "net": {
            "like_for_like_dS": like_for_like,
            "like_for_like_bars": like_for_like / ADMIT_BAR,
            "nominal_vs_pointer_dS": nominal,
            "nominal_bars": nominal / ADMIT_BAR,
            "admit_bar": ADMIT_BAR,
            "clears_like_for_like": bool(like_for_like <= ADMIT_BAR),
        },
        "axis": (
            "seg CARRIED from move 49 (frame-1 sections byte-identical); pose "
            "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI GT, n600]; bytes EXACT"
        ),
        "score_claim": False,
        "promotable": False,
    }
    np.save(out / "coefficient_codes_candidate.npy", candidate)
    np.save(out / "pose_resolved_n600.npy", resolved)
    np.save(out / "per_pair_gain.npy", per_pair_gain)
    write_json(out / "ASSEMBLE.json", report)
    print(
        json.dumps(
            {
                "pairs_accepted": len(accepted),
                "delta_bytes": delta_bytes,
                "resolved_mean": resolved_mean,
                "like_for_like_dS": like_for_like,
                "nominal_dS": nominal,
            }
        )
    )
    return 0


# ---------------------------------------------------------------------------
# stage: verdict
# ---------------------------------------------------------------------------


def cmd_verdict(args) -> int:
    """The three-leg table: re-solve ALONE, pass 8's subset ALONE, and the composition.

    The composition is by OBJECT CHANGE, never by adding legs.  The two moves are
    MEASURED disjoint -- pass 8's admitted pairs and the one pair this re-solve moves
    share no pair -- so the composed object's d_pose is its own mean over the same 600
    pairs, recomposed here from the two measured per-pair changes rather than from a sum
    of score deltas.  The realized fraction of the sum is then a RESULT, not an
    assumption, and it is quoted with the reason it lands where it does.

    Two accountings are reported for every row and neither is silently preferred:

    * **like-for-like** -- this instrument's own move-49 null control as the base.  It is
      the only difference that transfers to a T4 fire, because both numbers come off the
      same frozen CPU-torch PoseNet.
    * **nominal** -- the pointer's banked T4 row as the base, which is the convention
      pass 8's admission ledger used.  It carries a **-4.769e-06 phantom** that is purely
      the gap between this instrument's base (4.5435687e-06) and the T4 PRINT (4.55e-06),
      and that phantom is the same in every row, so it inflates every candidate equally.
    """
    receipts = assert_pointer()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    assemble = json.loads(Path(args.assemble).read_text())
    admission = json.loads(Path(args.admission).read_text())

    base_mean = float(assemble["pose"]["base_mean_n600"])
    null_seg = float(admission["reference_drop_everything"]["d_seg_t4"])
    null_bytes = float(admission["reference_drop_everything"]["archive_bytes_modelled"])
    if float(admission["reference_drop_everything"]["d_pose"]) != base_mean:
        raise Cr1Error(
            "pass 8's null control and this arm's base disagree; the two ledgers are "
            "not standing on the same instrument"
        )

    resolve_mean = float(assemble["pose"]["resolved_mean_n600"])
    resolve_bytes = float(assemble["rate"]["candidate_bytes"])
    subset_mean = float(admission["best"]["d_pose"])
    subset_seg = float(admission["best"]["d_seg_t4"])
    subset_bytes = float(admission["best"]["archive_bytes_modelled"])

    accepted = {int(p) for p in assemble["acceptance"]["accepted_pairs"]}
    kept = {int(p) for p in json.loads(Path(args.kept_pairs).read_text())}
    overlap = sorted(accepted & kept)
    if overlap:
        raise Cr1Error(
            f"the two moves share pairs {overlap}; the composed d_pose cannot be "
            "recomposed from two independent per-pair changes"
        )
    composed_mean = subset_mean - (base_mean - resolve_mean)

    null = composed_score(null_seg, base_mean, null_bytes)
    rows = {
        "null_control": (null_seg, base_mean, null_bytes),
        "A_resolve_alone": (null_seg, resolve_mean, resolve_bytes),
        "B_pass8_subset_alone": (subset_seg, subset_mean, subset_bytes),
        "C_composed": (subset_seg, composed_mean, subset_bytes),
    }
    table = {}
    for name, (seg, pose, byts) in rows.items():
        score = composed_score(seg, pose, byts)
        table[name] = {
            "d_seg_t4": seg,
            "d_pose": pose,
            "archive_bytes": byts,
            "S": score,
            "like_for_like_dS": score - null,
            "like_for_like_bars": (score - null) / ADMIT_BAR,
            "nominal_dS": score - POINTER_SCORE_T4,
            "nominal_bars": (score - POINTER_SCORE_T4) / ADMIT_BAR,
        }
    total = table["A_resolve_alone"]["like_for_like_dS"] + (
        table["B_pass8_subset_alone"]["like_for_like_dS"]
    )
    report = {
        "schema": "ddm_cr1_verdict.v1",
        "receipts": receipts,
        "table": table,
        "sum_of_the_two_alone_dS": total,
        "realized_fraction_of_the_sum": table["C_composed"]["like_for_like_dS"] / total,
        "disjoint_pairs": {
            "resolve_accepted": sorted(accepted),
            "pass8_kept": sorted(kept),
            "overlap": overlap,
        },
        "clears_admit_bar_like_for_like": bool(
            table["C_composed"]["like_for_like_dS"] <= ADMIT_BAR
        ),
        "clears_admit_bar_nominal": bool(
            table["C_composed"]["nominal_dS"] <= ADMIT_BAR
        ),
        "admit_bar": ADMIT_BAR,
        "rate_leg_caveat": (
            "rows B and C carry pass 8's SUBSET byte count, which is a sum of its "
            "MEASURED per-frame RLC1 bit ledgers and is therefore a RANKING, not a "
            "charge; sj1 measured the same estimator under-charge a 370-pair subset by "
            "19.6 B (+0.0108%). Row A's bytes are EXACT (a real archive, twinned)."
        ),
        "axis": (
            "seg from pass 8's jg1 instrument carried to T4 by its same-instrument "
            "ratio; pose [macOS-CPU advisory, frozen CPU-torch PoseNet, DALI GT, n600]; "
            "row A bytes EXACT, rows B/C bytes MODELLED by pass 8's bit ledger"
        ),
        "score_claim": False,
        "promotable": False,
    }
    write_json(out / "VERDICT.json", report)
    print(
        json.dumps(
            {
                "composed_like_for_like_bars": table["C_composed"]["like_for_like_bars"],
                "composed_nominal_bars": table["C_composed"]["nominal_bars"],
                "realized_fraction": report["realized_fraction_of_the_sum"],
            }
        )
    )
    return 0


# ---------------------------------------------------------------------------
# stage: twins
# ---------------------------------------------------------------------------


def cmd_twins(args) -> int:
    """TWIN the candidate carrier: compile it twice and demand identical bytes.

    "Zero bytes" is a claim about the CONTAINER, and a container whose size depends on a
    brotli search is entitled to be non-deterministic.  The identity control (move 49's
    own codes back to move 49's own sha) and the double compile are the two halves of
    the proof; neither alone is enough.
    """
    set_threads(args.threads)
    receipts = assert_pointer()
    import ddm_up3_carrier_splice as splice

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    codes = np.load(args.codes).astype(np.int32)

    def double_compile(payload: np.ndarray) -> dict[str, Any]:
        """``splice.control_determinism``'s body, with the sha pin this body needs.

        ``control_determinism`` re-parses through ``parse_shipped_body``'s DEFAULT
        ``verify_sha=True``, which checks up3's own generation's pointer
        (``50e56145...``), not move 49's.  The pin is not wrong -- it is a different
        body's -- so this arm asserts move 49 itself in ``assert_pointer`` above and
        parses with the pin OFF, rather than editing a sister arm's module.
        """
        out = []
        for _ in range(2):
            body = splice.parse_shipped_body(POINTER_TREE, verify_sha=False)
            out.append(
                splice.build_archive(
                    body,
                    payload,
                    runtime_dir=POINTER_TREE,
                    container_search=True,
                    verify=True,
                )
            )
        return {
            "control": "double_compile_determinism",
            "first_sha256": out[0]["archive_sha256"],
            "second_sha256": out[1]["archive_sha256"],
            "bytes": int(out[0]["archive_size"]),
            "identical": out[0]["archive_bytes"] == out[1]["archive_bytes"],
        }

    shipped = np.asarray(
        splice.parse_shipped_body(POINTER_TREE, verify_sha=False).codes, dtype=np.int32
    )
    identity = double_compile(shipped)
    candidate = double_compile(codes)
    report = {
        "schema": "ddm_cr1_twins.v1",
        "receipts": receipts,
        "identity_double_compile": identity,
        "identity_reproduces_move49": bool(
            identity["first_sha256"] == POINTER_ARCHIVE_SHA256
        ),
        "candidate_double_compile": candidate,
        "candidate_sha256": candidate["first_sha256"],
        "codes_path": str(args.codes),
        "score_claim": False,
    }
    write_json(out / "TWINS.json", report)
    if not (identity["identical"] and candidate["identical"]):
        raise Cr1Error(f"TWIN FAILED: {report}")
    if not report["identity_reproduces_move49"]:
        raise Cr1Error("IDENTITY FAILED: move 49's own codes do not rebuild move 49")
    print(json.dumps({k: report[k] for k in ("candidate_sha256", "identity_reproduces_move49")}))
    return 0


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    base = sub.add_parser("base", help="reproduce move 49's n600 pose base + twins")
    base.add_argument("--out-dir", default=str(WORK / "base"))
    base.add_argument("--batch-size", type=int, default=8)
    base.add_argument("--threads", type=int, default=6)
    base.set_defaults(func=cmd_base)

    resolve = sub.add_parser("resolve", help="the deeper n600 re-solve from shipped codes")
    resolve.add_argument("--base-pose", type=Path, required=True)
    resolve.add_argument("--shipped-codes", type=Path, required=True)
    resolve.add_argument("--start-codes", type=Path, default=None,
                         help="override the start (composition: the re-solved codes)")
    resolve.add_argument("--overlay-dir", default=None,
                         help="pass 8's re-rendered ODD frames, for the composed pairs")
    resolve.add_argument("--pairs-file", type=Path, default=None,
                         help="JSON list of pairs; default is all 600")
    resolve.add_argument("--out-dir", required=True)
    resolve.add_argument("--shard-index", type=int, default=0)
    resolve.add_argument("--shard-count", type=int, default=1)
    resolve.add_argument("--threads", type=int, default=2)
    resolve.set_defaults(func=cmd_resolve)

    assemble = sub.add_parser("assemble", help="gate, re-measure n600, price exact bytes")
    assemble.add_argument("--base-pose", type=Path, required=True)
    assemble.add_argument("--shipped-codes", type=Path, required=True)
    assemble.add_argument("--rows-dir", action="append", required=True)
    assemble.add_argument("--out-dir", required=True)
    assemble.add_argument("--batch-size", type=int, default=8)
    assemble.add_argument("--threads", type=int, default=6)
    assemble.add_argument("--allow-partial", action="store_true")
    assemble.set_defaults(func=cmd_assemble)

    verdict = sub.add_parser("verdict", help="the three-leg alone/subset/composed table")
    verdict.add_argument("--assemble", type=Path, required=True)
    verdict.add_argument("--admission", type=Path, required=True)
    verdict.add_argument("--kept-pairs", type=Path, required=True)
    verdict.add_argument("--out-dir", required=True)
    verdict.set_defaults(func=cmd_verdict)

    twins = sub.add_parser("twins", help="double-compile the candidate carrier")
    twins.add_argument("--codes", type=Path, required=True)
    twins.add_argument("--out-dir", required=True)
    twins.add_argument("--threads", type=int, default=4)
    twins.set_defaults(func=cmd_twins)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
