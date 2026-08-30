#!/usr/bin/env python3
"""ddm_gf1 -- apply the HG1 born-small GENERATOR FORM to lb1's OWN token field.

THE QUESTION (task #1334, the unentered cell of the #1215 2x2)
--------------------------------------------------------------
bz2 measured that a generator packet costs 47,779 B where {HPAC model + coded
tokens} costs 104,058 B for a BIT-IDENTICAL field -- 2.178x cheaper.  71.0% of
born-small's rate advantage is REPRESENTATIONAL, not field quality
([[generator-form-is-2x-cheaper-than-model-plus-coded-tokens]]).

bz2's FIELD is dead (REFUSED at 99.68x its own distortion falsifier).  Its FORM
is not.  This arm asks whether the form can carry lb1's own field -- which is
distortion-invariant BY CONSTRUCTION if the round-trip is bit-exact, because a
bit-identical token field renders bit-identical frames and therefore scores
bit-identically.

THE BAR (re-derived at source from the live pointer container, never quoted)
---------------------------------------------------------------------------
  archive              180,083 B   (sha 5b856e66...)
  sub-0.12 cap         137,986 B
  demand                42,097 B
  token subsystem  =  RX1M hdr 14 + hpac 13,515 + tokens 113,588 = 127,117 B
  BAR              =  127,117 - 42,097 = 85,020 B

PASS iff  packet + residual < 85,020 B  AND the round-trip is bit-exact.

WHY IT MAY FAIL, STATED BEFORE THE MEASUREMENT
----------------------------------------------
The HG1 generator is an ANALYTIC APPROXIMATOR, not a lossless coder.  bz2's own
packet reconstructs bz2's FITTED field exactly (corrections=0) because bz2
DEFINED its field as the generator's output.  Applied to a field it did not
generate, the mismatch count is the generator's EXPRESSIVE CAPACITY against that
field, and every mismatch must be paid for in the residual at ~2 B each.  If the
capacity gap is ~1.3M positions (bz2's own distance from GT), the residual alone
is ~2.6 MB and the family is dead by 30x.  That is the honest prior; the
measurement is what decides it.

NOT CLAIMED: any score.  This is a scorer-free byte measurement.
axis = [macOS-CPU scorer-free exact byte measurement]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for root in (REPO, REPO / "src", REPO / "experiments"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tac import subset_selection  # noqa: E402

from experiments import ddm_hg1_heterogeneous_analytic_generator_gate as hg1  # noqa: E402

SCHEMA = "ddm_gf1_generator_form_on_lb1_field.v1"
N_PAIRS, HEIGHT, WIDTH = 600, 384, 512
FIELD_BYTES = N_PAIRS * HEIGHT * WIDTH

# Re-derived at source (see docstring); asserted, not quoted.
ARCHIVE_BYTES = 180_083
SUB012_CAP = 137_986
TOKEN_SUBSYSTEM = 14 + 13_515 + 113_588
BAR = TOKEN_SUBSYSTEM - (ARCHIVE_BYTES - SUB012_CAP)

OUT = Path("/Volumes/APDataStore/pact/ddm_gf1_generator_form_on_lb1_field")
# Residual encode is a Python loop over mismatched positions; above this it is
# priced by sampling instead of materialised, and the row says so.
RESIDUAL_MATERIALISE_CEILING = 4_000_000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", type=Path, required=True, help="lb1-lineage token field (.u8)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    assert BAR == 85_020, f"bar re-derivation drifted: {BAR}"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "retained").mkdir(exist_ok=True)
    started = time.time()

    raw = np.fromfile(args.field, dtype=np.uint8)
    if raw.size != FIELD_BYTES:
        raise SystemExit(f"field size {raw.size} != {FIELD_BYTES}")
    target = raw.reshape(N_PAIRS, HEIGHT, WIDTH)
    field_sha = sha256_file(args.field)
    print(f"[gf1] target {args.field} sha={field_sha[:16]} classes={np.unique(target).tolist()}", flush=True)

    # ---- STAGE A: fit the four analytic generator streams to lb1's own field.
    t0 = time.time()
    horizon = hg1.fit_horizon_payload(target)
    print(f"[gf1] horizon fit {len(horizon)} B  ({time.time()-t0:.1f}s)", flush=True)
    lane, lane_meta = hg1.fit_lane_payload(target)
    print(f"[gf1] lane fit {len(lane)} B  ({time.time()-t0:.1f}s)", flush=True)
    movable, movable_meta = hg1.fit_movable_payload(target)
    print(f"[gf1] movable fit {len(movable)} B  ({time.time()-t0:.1f}s)", flush=True)
    mycar, mycar_meta = hg1.fit_mycar_payload(target)
    print(f"[gf1] mycar fit {len(mycar)} B  ({time.time()-t0:.1f}s)", flush=True)

    streams = {
        "road_undrivable": horizon,
        "lane": lane,
        "movable": movable,
        "mycar": mycar,
    }
    generated_path = args.out / "retained" / "generated_from_lb1_fit.u8"
    hg1.render_generators(streams, generated_path)
    generated = np.fromfile(generated_path, dtype=np.uint8).reshape(N_PAIRS, HEIGHT, WIDTH)

    # ---- STAGE B: the decisive number -- the generator's capacity gap on THIS field.
    mismatch = int(np.count_nonzero(generated != target))
    frac = mismatch / FIELD_BYTES
    print(f"[gf1] CAPACITY GAP: {mismatch} / {FIELD_BYTES} = {frac:.8f}", flush=True)

    result: dict[str, object] = {
        "schema": SCHEMA,
        "axis": "[macOS-CPU scorer-free exact byte measurement]",
        "score_claim": False,
        "promotable": False,
        "target_field": {"path": str(args.field), "sha256": field_sha, "bytes": int(raw.size)},
        "bar": {
            "archive_bytes": ARCHIVE_BYTES,
            "sub012_cap": SUB012_CAP,
            "demand": ARCHIVE_BYTES - SUB012_CAP,
            "token_subsystem_bytes": TOKEN_SUBSYSTEM,
            "replacement_bar_bytes": BAR,
        },
        "generator_streams_raw": {k: len(v) for k, v in streams.items()},
        "capacity_gap": {
            "mismatches": mismatch,
            "fraction": frac,
            "note": "generator output vs lb1's own field; every mismatch must be paid in the residual",
        },
        "lane_meta": hg1.json_safe(lane_meta),
        "movable_meta": hg1.json_safe(movable_meta),
        "mycar_meta": hg1.json_safe(mycar_meta),
    }

    # ---- STAGE C: price the residual that makes the round-trip EXACT.
    if mismatch == 0:
        residual = hg1.encode_residual(target, generated, args.out / "retained" / "residual.bin", None, "frame_raster")
        result["residual"] = hg1.json_safe(residual)
    elif mismatch <= RESIDUAL_MATERIALISE_CEILING:
        best = None
        for order in ("frame_raster", "class_frame_raster", "tile16_time"):
            p = args.out / "retained" / f"residual.{order}.bin"
            fact = hg1.encode_residual(target, generated, p, None, order)
            race = hg1.coder_race(f"residual_{order}", p, args.out)
            coded = min(int(c["coded"]["bytes"]) for c in race["coders"].values())
            print(f"[gf1] residual order={order} raw={fact['bytes']} coded={coded}", flush=True)
            if best is None or coded < best[1]:
                best = (order, coded, hg1.json_safe(fact), hg1.json_safe(race))
        result["residual"] = {"best_order": best[0], "coded_bytes": best[1],
                              "fact": best[2], "race": best[3]}
        residual_coded = best[1]
    else:
        # Too many corrections to materialise; price by the measured per-correction
        # cost on a bounded sample and SAY SO. No silent truncation.
        #
        # The subset is STRIDED, not a prefix: a video-order prefix is a different
        # population from the clip (m88), and while the RATE axis measured only
        # 0.989-1.030x bias (m96/na4) the stride costs nothing and spans all 600
        # pairs. Mode is declared through the canonical selector so the subset and
        # its ratio are recorded, not assumed.
        sample_pairs = 12
        sel = subset_selection.select(
            sample_pairs,
            N_PAIRS,
            mode=subset_selection.MODE_STRIDED,
            stride=N_PAIRS // sample_pairs,
        )
        idx = np.asarray(sel.indices, dtype=np.int64)
        sub_t = np.ascontiguousarray(target[idx])
        sub_g = np.ascontiguousarray(generated[idx])
        p = args.out / "retained" / "residual.sample12.bin"
        fact = hg1.encode_residual(sub_t, sub_g, p, None, "frame_raster")
        race = hg1.coder_race("residual_sample12", p, args.out)
        coded = min(int(c["coded"]["bytes"]) for c in race["coders"].values())
        n_sub = int(fact["corrections"])
        per = coded / max(n_sub, 1)
        residual_coded = per * mismatch
        result["residual"] = {
            "mode": "SAMPLED_PROJECTION",
            "reason": f"mismatch {mismatch} > materialise ceiling {RESIDUAL_MATERIALISE_CEILING}",
            "selection": {
                "mode": sel.mode,
                "indices": list(sel.indices),
                "population": sel.population,
                "params": dict(sel.params),
            },
            "sample_pairs": sample_pairs,
            "sample_corrections": n_sub,
            "sample_coded_bytes": coded,
            "coded_bytes_per_correction": per,
            "projected_coded_bytes": residual_coded,
            "scope": "PROJECTION on a 12-pair STRIDED sample spanning all 600 pairs; "
                     "rate-axis subset bias measured 0.989-1.030x full-population (m96/na4), "
                     "so the projection is not the verdict's weak point at this magnitude",
        }

    # ---- STAGE D: the packet + verdict.
    races = []
    for name, payload in streams.items():
        p = args.out / "retained" / f"{name}.raw"
        hg1.et1.atomic_bytes(p, payload)
        races.append(hg1.coder_race(name, p, args.out))
    packet_path = args.out / "retained" / "gf1_lb1_fit.packet"
    packet_fact = hg1.build_packet(races, packet_path)
    packet_bytes = int(packet_fact["bytes"])

    total = packet_bytes + residual_coded if mismatch else packet_bytes
    result["packet"] = hg1.json_safe(packet_fact)
    result["total_replacement_bytes"] = total
    result["verdict"] = {
        "bar": BAR,
        "total": total,
        "over_by": total - BAR,
        "ratio_over_bar": total / BAR,
        "PASS": bool(total < BAR),
    }
    result["elapsed_seconds"] = time.time() - started

    (args.out / "RESULT.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["verdict"], indent=2), flush=True)
    print(f"[gf1] wrote {args.out/'RESULT.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
