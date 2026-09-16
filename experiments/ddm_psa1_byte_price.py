#!/usr/bin/env python3
"""Scorer-free prices of minimum nonzero pair-selective renderer sections.

Research grammar only: no receiver/renderer implementation or score claim.
The shipped SM1S walker prices appended descriptors without changing its source.
Every raw, coded, repeat, and research archive payload is retained before pricing.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "experiments"), str(REPO), str(REPO / "src")]
STORE = Path("/Volumes/APDataStore/pact/ddm_psa1")
SOURCE = Path("/Volumes/APDataStore/pact/ddm_pd3/candidate/candidate_runtime")
RUNTIME = STORE / "price_runtime"
PIN = "ae59c5109597968409c4b581337b8e23b5b3c4262fd3710c315cc83913b3b20e"
GT = Path("/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/gt_cache_dali.pt")
ARGMAX = Path("/Volumes/APDataStore/pact/ddm_pd3/seg_final/argmax_n600.npy")
AXIS = "[macOS-CPU advisory / scorer-free exact bytes; inherited DALI seg labels]"
HEADER = struct.Struct("<4sBBBBHH")
FORMS = {"A": 1, "B3": 2, "C": 3, "D": 4, "B5_control": 5}


def fact(path: Path) -> dict:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest)


def load(path: Path):
    return json.loads(path.read_text())


def storage() -> dict:
    total = 0
    for path in STORE.rglob("*"):
        if any(p.startswith("._") or p.endswith(".pending") for p in path.parts):
            continue
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            continue  # sibling atomic rename; no sibling payload is owned by this arm
    return dict(retained_bytes=total, free_bytes=shutil.disk_usage(STORE.parent).free,
                limit_bytes=1 << 30, reserve_bytes=8 << 30)


def capacity(additional: int = 0) -> dict:
    status = storage()
    if status["retained_bytes"] + additional + (2 << 20) > status["limit_bytes"]:
        raise ValueError("RETENTION_LIMIT: preserve bytes and block")
    if status["free_bytes"] < status["reserve_bytes"] + additional + (2 << 20):
        raise ValueError("STORAGE_RESERVE: preserve bytes and block")
    return status


def retain(path: Path, blob: bytes) -> dict:
    if not path.resolve().is_relative_to(STORE.resolve()):
        raise ValueError("write outside owned store")
    if path.exists():
        if path.read_bytes() != blob:
            raise ValueError(f"immutable resume conflict: {path}")
        return fact(path)
    capacity(len(blob))
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    if pending.exists() and pending.read_bytes() != blob:
        raise ValueError(f"incomplete pending bytes require custody: {pending}")
    with pending.open("wb") as handle:
        handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())
    pending.replace(path)
    return fact(path)


def save(path: Path, value) -> dict:
    return retain(path, (json.dumps(value, sort_keys=True, indent=2) + "\n").encode())


def prepare() -> None:
    import numpy as np
    import torch

    print(json.dumps({"storage_before_heavy": capacity(16 << 20)}), flush=True)
    STORE.mkdir(parents=True, exist_ok=True)
    inputs = []
    for path in sorted(SOURCE.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and not path.name.startswith("._"):
            origin = fact(path)
            copied = retain(RUNTIME / path.relative_to(SOURCE), path.read_bytes())
            if copied["sha256"] != origin["sha256"]:
                raise ValueError("runtime copy differs")
            inputs.append({"source": origin, "copy": copied})
    if fact(RUNTIME / "archive.zip")["sha256"] != PIN:
        raise ValueError("wrong move52 archive")
    refs = {
        "sample": Path("/Volumes/APDataStore/pact/ddm_jrd1/SAMPLE.json"),
        "pool": Path("/Volumes/APDataStore/pact/ddm_pd4/sheets/priced_rows.jsonl"),
        "pd4_binding": Path("/Volumes/APDataStore/pact/ddm_pd4/BIND_RECEIPT_pre.json"),
        "pd5_binding": Path("/Volumes/APDataStore/pact/ddm_pd5/BIND_RECEIPT.json"),
        "seg_receipt": ARGMAX.parent / "STEP0_RESULT.json",
        "jrd1_bytes": Path("/Volumes/APDataStore/pact/ddm_jrd1/renderer/RESULT.json"),
        "jrx2_actions": REPO / ".omx/research/ddm_jrx2_20260916/RENDERER_BYTE_PROPOSALS.json",
    }
    sources = {}
    for key, path in refs.items():
        sources[key] = fact(path)
        retain(STORE / "inputs" / (key + path.suffix), path.read_bytes())
    if fact(ARGMAX)["sha256"] != load(refs["pd5_binding"])["pd4_receipts"]["argmax_sha256"]:
        raise ValueError("move52 argmax hash drift")
    gt_fact = fact(GT)
    if gt_fact["sha256"] != "a91d98252fe377c51ff7f3380c2fc9d30d84093fc54ee89e5e5f5102e6354994":
        raise ValueError("retained DALI target drift")
    gt = torch.load(GT, map_location="cpu", weights_only=False, mmap=True)["seg"].numpy()
    ours = np.load(ARGMAX, mmap_mode="r")
    if gt.shape != (600, 384, 512) or ours.shape != gt.shape:
        raise ValueError("label geometry mismatch")
    counts = [int(np.count_nonzero(ours[p] != gt[p])) for p in range(600)]
    if sum(counts) != 12147:
        raise ValueError("retained residual does not reproduce pd4/pd5 debt")
    sample = load(refs["sample"])
    k24 = sample["pairs"]
    pool = sorted({json.loads(row)["pair"] for row in refs["pool"].read_text().splitlines()})
    if (sample["seed"] != 20260916 or sample["K"] != 24 or
            sample["base_archive_sha256"] != PIN or len(set(k24)) != 24 or
            any(p < 0 or p >= 600 for p in k24) or len(pool) != 156 or
            sum(p in pool for p in k24) != 12):
        raise ValueError("inherited K24/pool contract drift")
    top12 = sorted(pool, key=lambda p: (-counts[p], p))[:12]
    overlap = sorted(set(k24) & set(top12))
    # Retain literal charter roster AND a distinct-pair completion; do not inflate k/36.
    supplement = sorted(set(pool) - set(k24), key=lambda p: (-counts[p], p))[:12]
    pairs = k24 + supplement
    if len(set(pairs)) != 36:
        raise ValueError("not 36 distinct pairs")
    records = []
    for p in pairs:
        locations = np.argwhere(ours[p] != gt[p]).astype("<u2")
        stream = io.BytesIO()
        np.save(stream, locations, allow_pickle=False)
        loc = retain(STORE / "inputs" / f"residual_{p:03d}.npy", stream.getvalue())
        records.append(dict(pair=p, cells=counts[p], debt_S=100 * counts[p] / (600 * 384 * 512),
                            debt_bytes=counts[p] * 37545489 / (25 * 600 * 384 * 512 / 100),
                            residual_locations=loc, center=locations[0].tolist() if len(locations) else None))
    save(STORE / "INPUTS.json", dict(runtime=inputs, sources=sources, gt=gt_fact,
                                    argmax=fact(ARGMAX), axis=AXIS, score_claim=False))
    save(STORE / "SAMPLE.json", dict(seed=20260916, K24=k24, top12_literal=top12,
                                    overlap=overlap, heavy12=supplement, pairs=pairs,
                                    population_counts=counts, records=records,
                                    selection="jrd1 K24 plus 12 heaviest pool pairs excluding K24; literal roster retained",
                                    median_population_debt_bytes=float(np.median(counts) * 1.273108215332031)))
    print(json.dumps({"prepared": 36, "top12_overlap": overlap, "residual_total": sum(counts)}), flush=True)


def section():
    import ddm_jrx2_renderer as renderer
    import ddm_ren2_restore_init as ren2

    inputs = load(STORE / "INPUTS.json")
    for key, expected in inputs["sources"].items():
        copied = STORE / "inputs" / (key + Path(expected["path"]).suffix)
        if fact(copied)["sha256"] != expected["sha256"]:
            raise ValueError("copied source receipt drift")
    for row in inputs["runtime"]:
        if fact(Path(row["copy"]["path"])) != row["copy"]:
            raise ValueError("copied runtime drift")
    if fact(RUNTIME / "archive.zip")["sha256"] != PIN:
        raise ValueError("full move52 archive pin mismatch")
    current = load(REPO / ".omx/state/canonical_frontier_pointer.json")
    if current["our_local_frontier_contest_cuda"]["archive_sha256"] != PIN:
        raise ValueError("pointer moved; no silent rebase")
    renderer.c.STORE = STORE  # only section() reads this binding; no predecessor writes
    container, runs, rc1, residual, mixer, receiver = renderer.section()
    for module in (rc1, residual, mixer, receiver):
        if not Path(module.__file__).resolve().is_relative_to(RUNTIME):
            raise ValueError(f"module was not loaded from copied runtime: {module.__file__}")
    weights = ren2.measure_mixer_weights(container["rider"])
    metadata, groups = mixer.split_body(container["body"], container["template"])
    _, descriptors = mixer.plan_metadata(metadata, container["template"])
    return container, runs, rc1, residual, mixer, receiver, weights, metadata, groups, descriptors


def control() -> None:
    print(json.dumps({"storage_before_heavy": capacity(8 << 20)}), flush=True)
    container, runs, rc1, residual, mixer, receiver, weights, metadata, groups, descriptors = section()
    if (STORE / "CONTROL.json").exists():
        for repeat in range(2):
            if (STORE / "control" / f"repeat{repeat}" / "rider.sm1s").read_bytes() != container["rider"]:
                raise ValueError("retained control drift")
        print(json.dumps({"resumed_completed_control": str(STORE / "CONTROL.json")}), flush=True)
        return
    started = time.monotonic()
    for repeat in range(2):
        rider, payload, meta = mixer.encode(container["body"], container["template"], weights)
        for name, blob in (("rider.sm1s", rider), ("range.bin", payload), ("metadata.bin", meta)):
            retain(STORE / "control" / f"repeat{repeat}" / name, blob)
        if rider != container["rider"] or receiver.restore_sm1_semantic(rider, container["template"]) != container["body"]:
            raise ValueError("shipped SM1S null replay differs")
    save(STORE / "CONTROL.json", dict(axis=AXIS, score_claim=False, null_rider_identity=True,
         shapes={name: list(run["shape"]) for name, run in runs.items()},
         groups=len(groups), codes=sum(len(g) for g in groups),
         elapsed_seconds=time.monotonic() - started, source=fact(Path(__file__))))
    print(json.dumps(load(STORE / "CONTROL.json")), flush=True)


def instance(form: str, row: dict, runs: dict, rc1, mixer) -> dict:
    """A real, minimum-nonzero command; efficacy and optimized values are unmeasured."""
    import numpy as np

    actions = load(STORE / "inputs/jrx2_actions.json")["rows"]
    existing = {r["pair"]: r["action"] for r in actions}
    head = runs["head.weight"]
    width = head["shape"][1]
    if head["shape"] != (3, 96, 3, 3):
        raise ValueError("unexpected renderer head")
    pair = row["pair"]
    if pair in existing:
        action = existing[pair]
        if action["tensor"] != "head.weight" or head["codes"][action["flat_index"]] != action["old"]:
            raise ValueError("jrx2 head proposal source mismatch")
        location, sign = action["flat_index"], action["new"] - action["old"]
        provenance = "retained jrx2 per-pair nearest-int4 direction"
    else:
        # Deterministic cheapest nonzero instance in the REAL shipped tensor, not a trained direction.
        eligible = np.flatnonzero(head["codes"] != 0)
        location = int(eligible[pair % len(eligible)])
        sign = -int(np.sign(head["codes"][location]))
        provenance = "pair-indexed existing nonzero head code toward zero; unoptimized price instance"
    channel, col = divmod(location, 864)
    input_channel = col // 9
    scales = np.asarray(head["scales"], dtype="<f2")
    vectors, scale_groups, mask_runs = [], [], []
    if form == "A":
        u, v = np.zeros(3, dtype=np.int32), np.zeros(864, dtype=np.int32)
        u[channel], v[col] = sign, 1
        vectors = [u, v]
        scale_groups = [scales, np.asarray([1], dtype="<f2")]
        if np.count_nonzero(np.outer(u, v)) != 1:
            raise ValueError("rank1 command did not produce exactly one nonzero weight delta")
    elif form in ("B3", "B5_control", "D"):
        n = 5 if form == "B5_control" else 3
        vector = np.zeros(n, dtype=np.int32)
        vector[channel] = sign
        vectors = [vector]
        scale_groups = [scales if n == 3 else np.resize(scales, 5).astype("<f2")]
        if form == "D":
            if row["center"] is None:
                raise ValueError("no residual locus: do not manufacture a spatial actuator")
            y, x = row["center"]
            for yy in range(max(0, y - 1), min(384, y + 2)):
                xx, end = max(0, x - 1), min(512, x + 2)
                mask_runs.append([yy * 512 + xx, end - xx])
    elif form == "C":
        scale_delta, shift = np.zeros(width, dtype=np.int32), np.zeros(width, dtype=np.int32)
        shift[input_channel] = sign
        vectors = [scale_delta, shift]
        scale_groups = [np.asarray([1 / 128], dtype="<f2"), np.asarray([scales[channel]], dtype="<f2")]
    else:
        raise ValueError("unknown form")
    if not any(np.any(v) for v in vectors):
        raise ValueError("zero actuator is not a price instance")
    # All section-derived metadata, including pair key, scales and mask, is counted.
    meta = struct.pack("<HB", pair, len(vectors))
    packed, descs = [], []
    for n, (vector, scale) in enumerate(zip(vectors, scale_groups, strict=True)):
        cols = 1 if len(scale) == len(vector) else len(vector)
        meta += struct.pack("<HHH", len(vector), cols, len(scale)) + scale.tobytes()
        packed.append(rc1.pack_signed_codes(vector, 4))
        descs.append(dict(name=f"psa.{form}.{pair}.{n}", shape=(len(vector) // cols, cols),
                          count=len(vector), bits=4, cols=cols, scales=mixer.scale_buckets(scale.tobytes()),
                          embedding=False, kind=6, block=4))
    meta += struct.pack("<B", len(mask_runs))
    for start, length in mask_runs:
        meta += struct.pack("<IH", start, length)
    return dict(form=form, pair=pair, metadata=meta, packed=b"".join(packed),
                groups=vectors, descriptors=descs, mask_runs=mask_runs,
                direction_source=provenance, flat_index=location, sign=sign)


def serialize(form: str, instances: list[dict], codec: int) -> bytes:
    header = HEADER.pack(b"PSA1", 1, FORMS[form], codec, 4, 96, len(instances))
    return header + b"".join(i["metadata"] + (i["packed"] if codec == 0 else b"") for i in instances)


def parse_section(blob: bytes, rc1, mixer):
    """Research-only exact inverse; rejects truncated, duplicate, invalid or trailing bytes."""
    import numpy as np

    if len(blob) < HEADER.size:
        raise ValueError("truncated header")
    magic, version, form, codec, layer, width, count = HEADER.unpack_from(blob)
    if (magic, version, layer, width) != (b"PSA1", 1, 4, 96) or form not in FORMS.values() or codec not in (0, 1):
        raise ValueError("unsupported section")
    cursor, ids, descs, groups = HEADER.size, [], [], []

    def read(size):
        nonlocal cursor
        value = blob[cursor:cursor + size]
        if len(value) != size:
            raise ValueError("truncated section")
        cursor += size
        return value

    for _ in range(count):
        pair, nvec = struct.unpack("<HB", read(3))
        if pair >= 600 or pair in ids or nvec not in (1, 2):
            raise ValueError("invalid pair/vector index")
        ids.append(pair)
        local = []
        for n in range(nvec):
            size, cols, ns = struct.unpack("<HHH", read(6))
            scale_blob = read(2 * ns)
            if not size or not cols or size % cols or ns != size // cols:
                raise ValueError("invalid vector geometry")
            # Reject zero/NaN scales as well as invalid SM1 bucket representations.
            scale = np.frombuffer(scale_blob, dtype="<f2")
            if not np.all(np.isfinite(scale) & (scale > 0)):
                raise ValueError("invalid scales")
            name = next(k for k, v in FORMS.items() if v == form)
            desc = dict(name=f"psa.{name}.{pair}.{n}", shape=(size // cols, cols),
                        count=size, bits=4, cols=cols, scales=mixer.scale_buckets(scale_blob),
                        embedding=False, kind=6, block=4)
            local.append(desc)
        nruns = read(1)[0]
        sizes = {1: [3, 864], 2: [3], 3: [96, 96], 4: [3], 5: [5]}
        if [d["count"] for d in local] != sizes[form] or (form == 4) != (nruns > 0):
            raise ValueError("actuator form/shape/mask mismatch")
        if nruns > 3:
            raise ValueError("radius1 mask exceeds three rows")
        last = -1
        for _ in range(nruns):
            start, length = struct.unpack("<IH", read(6))
            if not length or start <= last or start + length > 384 * 512:
                raise ValueError("invalid spatial run")
            last = start + length - 1
        descs.extend(local)
        if codec == 0:
            for desc in local:
                packed = read((desc["count"] + 1) // 2)
                values = np.asarray(rc1.unpack_signed_codes(packed, desc["count"], 4), dtype=np.int32)
                if rc1.pack_signed_codes(values, 4) != packed:
                    raise ValueError("noncanonical packed padding")
                groups.append(values)
    if cursor != len(blob):
        raise ValueError("trailing section bytes")
    return ids, descs, groups


def verify_sm1x(rider: bytes, container: dict, rc1, mixer, expected: list, ids: list[int]) -> None:
    """Read every descriptor/value from the final counted research rider before decode."""
    import numpy as np

    magic, version, count, reserved, length = mixer.HEADER.unpack_from(rider)
    if (magic, version, count, reserved) != (b"SM1X", 1, 24, 0) or length < 4:
        raise ValueError("bad research rider header")
    end = len(rider) - length - 24
    meta_start, cursor = mixer.HEADER.size, 10
    if end < meta_start + 10:
        raise ValueError("research rider length overrun")

    def read(kind, size):
        nonlocal cursor
        if kind == "codes":
            return b""
        value = rider[meta_start + cursor:meta_start + cursor + size]
        if len(value) != size or meta_start + cursor + size > end:
            raise ValueError("truncated original model metadata")
        cursor += size
        return value

    rc1.walk_sm3r(read, container["template"], tuple(rider[meta_start + 4:meta_start + 8]))
    metadata = rider[meta_start:meta_start + cursor]
    plan, original_desc = mixer.plan_metadata(metadata, container["template"])
    aux = rider[meta_start + cursor:end]
    read_ids, extra_desc, _ = parse_section(aux, rc1, mixer)
    if read_ids != ids or aux[6] != 1:
        raise ValueError("research descriptor index differs")
    weights = np.frombuffer(rider[end:end + 24], dtype=np.int8)
    _, decoded, _, _ = mixer.walk(original_desc + extra_desc, weights, payload=rider[end + 24:])
    if any(not np.array_equal(a, b) for a, b in zip(decoded, expected, strict=True)):
        raise ValueError("retained research ZIP decoded different symbols")
    parts, g = [metadata[:10]], 0
    for item in plan:
        if item["kind"] == "codes":
            parts.append(rc1.pack_signed_codes(decoded[g], item["bits"]))
            g += 1
        else:
            parts.append(item["blob"])
    if b"".join(parts) != container["body"]:
        raise ValueError("original model changed in appended research archive")


def price(form: str) -> None:
    import brotli
    import numpy as np
    import ddm_jg2_tail_reencode as jg2
    from ddm_up3_carrier_splice import _ck2_interleave_planes

    print(json.dumps({"form": form, "storage_before_heavy": capacity(150 << 20)}), flush=True)
    container, runs, rc1, residual, mixer, receiver, weights, metadata, source, descriptors = section()
    base_payload = (STORE / "control/repeat0/range.bin").read_bytes()
    base_sections = jg2.split_member(jg2.read_archive_member(RUNTIME / "archive.zip"))
    base_member = jg2.join_member(base_sections)
    shape = load(STORE / "inputs/jrd1_bytes.json")["container"]
    records = load(STORE / "SAMPLE.json")["records"]
    producer = retain(STORE / "prices" / form / "producer.py", Path(__file__).read_bytes())
    save(STORE / "prices" / form / "BINDING.json", dict(producer=producer,
         helpers=[fact(Path(m.__file__)) for m in (jg2, rc1, mixer, residual, receiver)],
         input_receipt=fact(STORE / "INPUTS.json"), sample=fact(STORE / "SAMPLE.json"),
         method="full shipped walk; no cached/patched encoder state", axis=AXIS))
    all_instances = [instance(form, row, runs, rc1, mixer) for row in records]
    jobs = [("empty", [], None)] + [(f"pair_{r['pair']:03d}", [i], r) for r, i in zip(records, all_instances, strict=True)]
    jobs.append(("joint36", all_instances, None))
    results = []
    for label, selected, debt in jobs:
        dest = STORE / "prices" / form / label
        result_path = dest / "RESULT.json"
        if result_path.exists():
            prior = load(result_path)
            for expected in prior["artifacts"]:
                if fact(Path(expected["path"])) != expected:
                    raise ValueError("completed price checkpoint drift")
            results.append(prior)
            print(json.dumps({"form": form, "resumed": label}), flush=True)
            continue
        print(json.dumps({"form": form, "job": label, "storage_before_heavy": capacity(3 << 20)}), flush=True)
        artifacts = []
        raw = serialize(form, selected, 0)
        aux = serialize(form, selected, 1)
        artifacts += [retain(dest / "section.raw", raw), retain(dest / "section.metadata", aux)]
        ids, _, raw_groups = parse_section(raw, rc1, mixer)
        expected = [g for i in selected for g in i["groups"]]
        if ids != [i["pair"] for i in selected] or any(not np.array_equal(a, b) for a, b in zip(raw_groups, expected, strict=True)):
            raise ValueError("standalone raw parse-back differs")
        _, extra_desc, _ = parse_section(aux, rc1, mixer)
        br_twins, sm_twins, stream_twins = [], [], []
        for repeat in range(2):
            rd = dest / f"repeat{repeat}"
            br = brotli.compress(raw, quality=11)
            artifacts.append(retain(rd / "section.q11.br", br))
            if brotli.decompress(br) != raw:
                raise ValueError("Brotli section parse-back differs")
            # Explicit new counted trailer, never passed to a public renderer or scorer.
            trailer = br + struct.pack("<4sI", b"PSAX", len(br))
            standalone_member = base_member + trailer
            artifacts.append(retain(rd / "standalone.member.p", standalone_member))
            path = rd / "standalone.research.zip"
            if not path.exists():
                jg2.pack_archive(standalone_member, path)
            if jg2.read_archive_member(path) != standalone_member:
                raise ValueError("research archive pack mismatch")
            recovered = jg2.read_archive_member(path)
            footer_magic, br_length = struct.unpack("<4sI", recovered[-8:])
            if (footer_magic != b"PSAX" or recovered[:-8 - br_length] != base_member or
                    brotli.decompress(recovered[-8 - br_length:-8]) != raw):
                raise ValueError("retained standalone research ZIP parse-back differs")
            br_twins.append(fact(path)); artifacts.append(br_twins[-1])
            # The EXACT shipped walker, with a new counted descriptor section.
            payload, encoded, _, _ = mixer.walk(descriptors + extra_desc, weights, source=source + expected)
            artifacts.append(retain(rd / "extended.range.bin", payload))
            if not selected and payload != base_payload:
                raise ValueError("empty extension changed the original range stream")
            if any(not np.array_equal(a, b) for a, b in zip(encoded, source + expected, strict=True)):
                raise ValueError("appended encoder changed symbols")
            # SM1X is deliberately a new research magic; unchanged receiver rejects it.
            rider = mixer.HEADER.pack(b"SM1X", 1, 24, 0, len(payload)) + metadata + aux + weights.tobytes() + payload
            artifacts.append(retain(rd / "rider.sm1x", rider))
            stream_twins.append(fact(rd / "extended.range.bin"))
            staged = _ck2_interleave_planes(rider) if container["ck2_semantic"] else rider
            artifacts.append(retain(rd / "staged.bin", staged))
            member = brotli.compress(staged, quality=shape["quality"], lgwin=shape["lgwin"])
            artifacts.append(retain(rd / "semantic.br", member))
            changed = dict(base_sections)
            header = list(jg2.RX1_HEADER.unpack(changed["header"]))
            header[6] = len(member)
            changed["header"], changed["semantic"] = jg2.RX1_HEADER.pack(*header), member
            whole = jg2.join_member(changed)
            artifacts.append(retain(rd / "sm1x.member.p", whole))
            path = rd / "sm1x.research.zip"
            if not path.exists():
                jg2.pack_archive(whole, path)
            if jg2.read_archive_member(path) != whole:
                raise ValueError("SM1X research archive mismatch")
            if repeat == 0:
                recovered_parts = jg2.split_member(jg2.read_archive_member(path))
                for key in base_sections:
                    if key not in ("header", "semantic") and recovered_parts[key] != base_sections[key]:
                        raise ValueError("unrelated research archive section changed")
                recovered = brotli.decompress(recovered_parts["semantic"])
                if container["ck2_semantic"]:
                    recovered = residual._ck2_uninterleave_planes(recovered)
                verify_sm1x(recovered, container, rc1, mixer, source + expected, ids)
            sm_twins.append(fact(path)); artifacts.append(sm_twins[-1])
        for twins in (br_twins, sm_twins, stream_twins):
            if twins[0]["sha256"] != twins[1]["sha256"]:
                raise ValueError("twin nondeterminism")
        standalone = br_twins[0]["bytes"] - 179332
        appended = len(aux) + stream_twins[0]["bytes"] - len(base_payload)
        result = dict(form=form, label=label, pair=debt["pair"] if debt else None,
            axis=AXIS, score_claim=False, research_only=True, receiver_implemented=False,
            standalone_q11_bytes=standalone, q11_section_bytes=standalone - 8,
            sm1_append_bytes=appended, sm1_archive_delta_bytes=sm_twins[0]["bytes"] - 179332,
            raw_section_bytes=len(raw), metadata_bytes=len(aux), header_bytes=HEADER.size,
            pair_index_bytes=2 * len(selected), standalone_footer_bytes=8,
            debt=debt, standalone_pays=bool(debt and standalone < debt["debt_bytes"]),
            sm1_append_pays=bool(debt and appended < debt["debt_bytes"]),
            directions=[dict(pair=i["pair"], source=i["direction_source"], flat_index=i["flat_index"], sign=i["sign"], mask_runs=i["mask_runs"]) for i in selected],
            artifacts=artifacts, twins_identical=True, exact_code_parseback=True)
        save(result_path, result)
        results.append(result)
        print(json.dumps({"form": form, "completed": label, "q11": standalone, "sm1": appended}), flush=True)
    save(STORE / "prices" / form / "RESULTS.json", results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "control", "price"))
    parser.add_argument("--form", choices=tuple(FORMS))
    parser.add_argument("--resume-from", type=Path, required=True)
    args = parser.parse_args()
    if args.resume_from.resolve() != STORE.resolve():
        raise ValueError("wrong resume store")
    import numpy as np
    import torch
    np.random.seed(20260916)
    torch.manual_seed(20260916)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.stage == "price":
        if args.form is None:
            raise ValueError("price requires --form")
        price(args.form)
    else:
        {"prepare": prepare, "control": control}[args.stage]()


if __name__ == "__main__":
    main()
