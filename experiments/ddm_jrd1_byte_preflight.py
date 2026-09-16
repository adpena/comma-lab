#!/usr/bin/env python3
"""Scorer-free jrd1 preparation, NOT the joint exchange measurement.

Keep the real move-52 bytes, draw the preregistered pairs, and exercise the
current SM1S weight coder on held-scale int4 steps. No gradient, renderer
forward, scorer, pose solve, training, or exchange verdict is implemented here.
The common contract queues those operations until MAIN assigns the scorer slot.
Every materialized coded payload, including container trials, is retained.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "experiments")]
STORE = Path("/Volumes/APDataStore/pact/ddm_jrd1")
ROOT = STORE / "rlc1_price"
RUNTIME = ROOT / "runtime_copy"
SEED = 20260916
ARCHIVE_SHA = "ae59c5109597968409c4b581337b8e23b5b3c4262fd3710c315cc83913b3b20e"
POOL = Path("/Volumes/APDataStore/pact/ddm_pd4/sheets/priced_rows.jsonl")
LIMIT = 3 << 30
RESERVE = 8 << 30
AXIS = "[macOS-CPU advisory / scorer-free exact byte measurement]"


def fact(path: Path) -> dict:
    with path.open("rb") as handle:
        sha = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha}


def storage() -> dict:
    files = [p for p in STORE.rglob("*") if p.is_file()]
    return {"retained_bytes": sum(p.stat().st_size for p in files),
            "free_bytes": shutil.disk_usage(STORE).free,
            "limit_bytes": LIMIT, "reserve_bytes": RESERVE}


def check_capacity(additional_bytes: int) -> None:
    status = storage()
    if status["retained_bytes"] + additional_bytes + (2 << 20) > LIMIT:
        raise ValueError("RETENTION_LIMIT: keep existing bytes; do not launch another stage")
    if status["free_bytes"] < RESERVE + additional_bytes + (2 << 20):
        raise ValueError("STORAGE_BLOCK: preserve bytes; reserve would be crossed")


def retain(path: Path, payload: bytes) -> dict:
    if not path.resolve().is_relative_to(STORE.resolve()):
        raise ValueError(f"write outside owned store: {path}")
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"refuse to replace retained payload: {path}")
        return fact(path)
    check_capacity(len(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    # On a interrupted write, preserve and validate the complete pending payload.
    temporary = path.with_suffix(path.suffix + ".pending")
    if temporary.exists() and temporary.read_bytes() != payload:
        raise ValueError(f"incomplete prior write needs custody review: {temporary}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return fact(path)


def record(path: Path, data: dict) -> dict:
    return retain(path, (json.dumps(data, indent=2, sort_keys=True) + "\n").encode())


def guard() -> dict:
    inputs = json.loads((ROOT / "INPUTS.json").read_text())
    current = json.loads((REPO / ".omx/state/canonical_frontier_pointer.json").read_text())
    if current["our_local_frontier_contest_cuda"]["archive_sha256"] != ARCHIVE_SHA:
        raise ValueError("POINTER_MOVED: do not silently rebase the experiment")
    if fact(RUNTIME / "archive.zip")["sha256"] != ARCHIVE_SHA:
        raise ValueError("runtime copy archive pin mismatch")
    for rel, expected in inputs["runtime_sources"].items():
        observed = fact(RUNTIME / rel)
        if any(observed[k] != expected[k] for k in ("bytes", "sha256")):
            raise ValueError(f"runtime copy drift: {rel}")
    return inputs


def sample() -> None:
    import numpy as np

    inputs = guard()
    frozen = STORE / "inputs/pd4_priced_rows.jsonl"
    retain(frozen, POOL.read_bytes())
    rows = [json.loads(line) for line in frozen.read_text().splitlines() if line.strip()]
    pool = sorted({int(row["pair"]) for row in rows})
    if len(pool) != 156 or any(p < 0 or p >= 600 for p in pool):
        raise ValueError("pd4 pool geometry changed; expected 156 of 600 pairs")
    remainder = sorted(set(range(600)) - set(pool))
    rng = np.random.Generator(np.random.PCG64(SEED))
    selected = [int(x) for x in rng.choice(pool, 12, replace=False)]
    other = [int(x) for x in rng.choice(remainder, 12, replace=False)]
    pairs = selected + other
    field_fact = inputs["fields"]["control"]["u8"]
    if fact(Path(field_fact["path"])) != field_fact:
        raise ValueError("control field custody mismatch")
    field = np.memmap(field_fact["path"], dtype=np.uint8, mode="r", shape=(600, 384, 512))
    stream = io.BytesIO()
    np.save(stream, np.asarray(field[pairs]), allow_pickle=False)
    subset = retain(STORE / "inputs/selected_planes.npy", stream.getvalue())
    receipt = dict(schema="ddm_jrd1_pair_sample.v1", seed=SEED, rng="numpy.PCG64",
                   numpy_version=np.__version__, pool_source=fact(frozen),
                   pool_definition="156 distinct pairs in pd4 sheets/priced_rows.jsonl; pd5 charter confirms this pool",
                   pool=pool, remainder=remainder, pool_n=len(pool), remainder_n=len(remainder),
                   selected_pool=selected, selected_remainder=other, pairs=pairs, K=24,
                   selection="12+12 seeded draws without replacement; no outcome selection",
                   interpretation="Untouched means outside pd4's priced pool, not never edited in lineage.",
                   selected_planes=subset, base_archive_sha256=ARCHIVE_SHA,
                   score_claim=False, banked_negative=False,
                   scope="K24 charter diagnostic only; n>=32 contract required before banking a sampled negative")
    record(STORE / "SAMPLE.json", receipt)
    print(json.dumps(receipt), flush=True)


def renderer() -> None:
    """Measure only the byte instrument on six seeded, non-gradient grid actions."""
    import brotli
    import ddm_jg2_tail_reencode as jg2
    import ddm_ren2_restore_init as ren2
    import numpy as np
    import torch
    from ddm_up3_carrier_splice import _ck2_interleave_planes

    guard()
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    ren2.POINTER_TREE = RUNTIME  # process-local binding; no source/tree mutation
    container = ren2.read_shipped_semantic_container()
    residual, mixer, _inflate, receiver = ren2._receiver_modules()
    rc1 = importlib.import_module("rc1_adaptive_model_sections")
    body, template = container["body"], container["template"]
    out = STORE / "renderer"
    source_facts = {name: retain(out / "source" / (name + ".bin"), container[name])
                    for name in ("body", "rider", "staged", "member")}
    weights = ren2.measure_mixer_weights(container["rider"])
    if container["sz1_split"]:
        raise ValueError("unsupported semantic split: refuse a mechanism substitution")
    shape = None
    # Retain ALL trial bytes, unlike a scalar-only container search.
    for quality, lgwin in ren2.BROTLI_GRID:
        candidate = brotli.compress(container["staged"], quality=quality, lgwin=lgwin)
        retain(out / "container_controls" / f"q{quality}_w{lgwin}.br", candidate)
        if candidate == container["member"]:
            shape = (quality, lgwin)
            break
    if shape is None:
        raise ValueError("no byte-identical semantic container control")

    cursor, offsets = 10, []

    def read(_kind: str, count: int) -> bytes:
        nonlocal cursor
        offsets.append(cursor)
        result = body[cursor:cursor + count]
        cursor += count
        return result

    plan = rc1.walk_sm3r(read, template, tuple(body[4:8]))
    if cursor != len(body):
        raise ValueError("receiver walk did not consume the complete SM3R body")
    runs, index = {}, 1
    for name, value in template.items():
        if value.ndim < 2:
            index += 1
            continue
        pruned = name in rc1.ROW_PRUNE_NAMES
        if pruned:
            index += 1
        index += 1  # scales stay byte-identical
        item, offset = plan[index], offsets[index]
        if item["kind"] != "codes":
            raise ValueError("receiver code-run walk mismatch")
        runs[name] = (item, offset, pruned)
        index += 1

    sections = jg2.split_member(jg2.read_archive_member(RUNTIME / "archive.zip"))
    base_parts = residual.read_residual_archive(RUNTIME / "archive.zip")
    rng = np.random.Generator(np.random.PCG64(SEED))
    actions = [("control", body, None)]
    for name in ("head.weight", "blocks.3.dw.weight", "blocks.3.pw.weight"):
        item, offset, pruned = runs[name]
        if pruned or int(item["bits"]) != 4:
            raise ValueError(f"not an unpruned int4 actuator: {name}")
        codes = np.asarray(rc1.unpack_signed_codes(body[offset:offset + item["length"]],
                                                item["count"], item["bits"]), dtype=np.int32)
        eligible = np.flatnonzero((codes > -8) & (codes < 7))
        location = int(rng.choice(eligible))
        for delta in (-1, 1):
            changed = codes.copy()
            changed[location] += delta
            encoded = rc1.pack_signed_codes(changed, 4)
            label = name.replace(".", "_") + ("_minus" if delta < 0 else "_plus")
            retain(out / label / "code_run.bin", encoded)
            result = body[:offset] + encoded + body[offset + item["length"]:]
            retain(out / label / "action_body.sm3r", result)
            if len(result) != len(body) or np.count_nonzero(changed != codes) != 1:
                raise ValueError("single-grid-action identity failed")
            actions.append((label, result, dict(tensor=name, flat_index=location,
                                                old=int(codes[location]), new=int(changed[location]),
                                                changed_codes=1, selection="seeded instrumentation; NOT a gradient")))
    results = []
    for label, new_body, action in actions:
        twins = []
        for repeat in range(2):
            dest = out / label / f"repeat{repeat}"
            retain(dest / "body.sm3r", new_body)
            rider, payload, metadata = mixer.encode(new_body, template, weights)
            retain(dest / "rider.sm1s", rider)
            retain(dest / "range_payload.bin", payload)
            retain(dest / "metadata.bin", metadata)
            if receiver.restore_sm1_semantic(rider, template) != new_body:
                raise ValueError("SM1S parse-back did not reproduce the exact grid action")
            staged = _ck2_interleave_planes(rider) if container["ck2_semantic"] else rider
            retain(dest / "staged.bin", staged)
            member = brotli.compress(staged, quality=shape[0], lgwin=shape[1])
            retain(dest / "member.br", member)
            changed_sections = dict(sections)
            header = list(jg2.RX1_HEADER.unpack(sections["header"]))
            header[6] = len(member)
            changed_sections["header"] = jg2.RX1_HEADER.pack(*header)
            changed_sections["semantic"] = member
            whole = jg2.join_member(changed_sections)
            retain(dest / "member.p", whole)
            archive = dest / "archive.zip"
            if not archive.exists():
                check_capacity(len(whole) + 1024)
                jg2.pack_archive(whole, archive)
            if jg2.read_archive_member(archive) != whole:
                raise ValueError("archive resume or pack mismatch")
            parsed = residual.read_residual_archive(archive)
            if bytes(parsed.semantic_blob) != rider:
                raise ValueError("archive semantic parse-back mismatch")
            for attr in ("hpac_blob", "carrier_blob", "token_stream", "tc1_weights", "residual_payload"):
                if getattr(parsed, attr) != getattr(base_parts, attr):
                    raise ValueError(f"unrelated component changed: {attr}")
            observed = fact(archive)
            if label == "control" and observed["sha256"] != ARCHIVE_SHA:
                raise ValueError("semantic null rebuild failed archive identity")
            twins.append(observed)
        if twins[0]["sha256"] != twins[1]["sha256"]:
            raise ValueError("semantic encode twins differ")
        delta = twins[0]["bytes"] - 179332
        result = dict(label=label, action=action, archives=twins,
                      delta_archive_bytes=delta, delta_coded_bits=8 * delta,
                      rate_delta_S=delta * 25 / 37545489,
                      seg_credit=None, resolved_pose=None, exchange_S_per_B=None)
        record(out / label / "RESULT.json", result)  # per-action durable resume boundary
        results.append(result)
    receipt = dict(schema="ddm_jrd1_renderer_byte_instrument.v1", axis=AXIS,
                   score_claim=False, seed=SEED, source_facts=source_facts,
                   producer=fact(Path(__file__)),
                   helpers=[fact(Path(m.__file__)) for m in (jg2, ren2, mixer, rc1, receiver)],
                   numpy_version=np.__version__, torch_version=torch.__version__,
                   container=dict(quality=shape[0], lgwin=shape[1], ck2=container["ck2_semantic"]),
                   results=results, checkpoint_policy="immutable per-action results; repeat same command to resume",
                   scope="instrument checks only; these random actions are NOT charter renderer-row gradients",
                   no_exchange_verdict=True)
    record(out / "RESULT.json", receipt)
    print(json.dumps(receipt), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("sample", "renderer"))
    parser.add_argument("--resume-from", type=Path, required=True,
                        help="Owned store; immutable outputs and per-action receipts are validated on resume")
    args = parser.parse_args()
    if args.resume_from.resolve() != STORE.resolve():
        raise ValueError("resume store must be the charter's owned APDataStore directory")
    actual_nice = os.getpriority(os.PRIO_PROCESS, 0)
    if actual_nice != 0:
        raise ValueError(f"NICENESS_REFUSED: actual {actual_nice}, charter requires 0")
    print(json.dumps(dict(stage=args.stage, verified_child_nice=actual_nice, storage=storage())), flush=True)
    {"sample": sample, "renderer": renderer}[args.stage]()


if __name__ == "__main__":
    main()
