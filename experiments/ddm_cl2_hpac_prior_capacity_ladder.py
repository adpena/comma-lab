#!/usr/bin/env python3
"""ddm_cl2 -- exact price of one HPAC prior-capacity rung on the SHIPPED fs2 mixer.

WHAT THIS PRICES.  A rung is one terminal epoch-60 QAT checkpoint of
``tools/train_ddm_cl1_hpac_capacity.py`` (profile ``cl2_shipped_ladder`` on Metal,
or the JF1 profile on CPU for the free CPU-law control row), trained at one
``--rate-lambda`` from the shipped epoch-634 EMA warm start on the CURRENT token
field.  The field is held bit-identical, so d_seg / d_pose are held by
construction; the only two numbers that move are the packed model bytes and the
RC64 token-stream bytes.  cl1's break-even for adjacent rungs is

    delta(stream bytes) / delta(packed model bytes) < -1 .

HOW IT PRICES (the shipped path, nothing re-implemented).

  1. ``pack``   -- ``ddm_rx2_mc36_identity_race._pack_terminal_ihs1`` (the exact
                   IHS1 packer from the PR130 intake, idempotency + decode
                   determinism proved) then the Brotli q0..q11 race, exactly as
                   ``ddm_jf1_joint_field_model_refit._pack_model`` does.
  2. ``stage``  -- a copy of the fs2 fire tree whose archive carries the NEW hpac
                   section (header length rewritten) and everything else shipped.
  3. ``encode`` -- ``ddm_jg2_tail_reencode.encode_tail`` over the exact field:
                   ``decode_production_tokens`` line for line with the decode
                   replaced by an encode of the known symbols, so the emitted
                   stream walks the receiver's own trajectory (model, group order,
                   fixed table, FreeCorrector = the fx1 mixer, RC64).  Run TWICE
                   (fresh corrector state each time); the two streams must be
                   byte-identical.
  4. ``decode`` -- a receiver copy parses the candidate archive and decodes the
                   token field with the shipped ``decode_production_tokens``; the
                   decoded field must equal the target byte for byte.  Wall-clock
                   is recorded next to the shipped archive's own decode (``control``).

Every payload (raw IHS1, every Brotli representation, both streams, the candidate
archive, the receiver-copy runtime, the decoded field) is retained with sha256 +
bytes.  No scorer runs here.  Axis of every number:
``[macOS-CPU advisory / scorer-free EXACT byte measurement]``; ``score_claim=false``.

Usage::

  python experiments/ddm_cl2_hpac_prior_capacity_ladder.py control
  python experiments/ddm_cl2_hpac_prior_capacity_ladder.py price --rung lambda_1p0 \
      --checkpoint <qat_stage_end_epoch_0060.pt> --expected-profile cl2_shipped_ladder
  python experiments/ddm_cl2_hpac_prior_capacity_ladder.py report
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_jg2_tail_reencode as jg2
from experiments import ddm_rx2_mc36_identity_race as rx2

STORE = Path("/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder")
FS2_RUNTIME = Path("/Volumes/VertigoDataTier/pact/ddm_fs2_carrier_resolve/fire_runtime_D_alternation")
FS2_ARCHIVE = FS2_RUNTIME / "archive.zip"
FS2_ARCHIVE_BYTES = 180_023
FS2_ARCHIVE_SHA256 = "a8f3a3791499b2b62ee4d16bc67f15f819f454dc9b88e3cce04fe50a30427bb6"
FS2_STREAM_BYTES = 113_411
FS2_STREAM_SHA256 = "5601d6fd792c60c176e7cb7478e6033c4ed9a7e87404582340ed3f50ed60cfe3"
SHIPPED_MODEL_BYTES = 13_515
SHIPPED_JOINT_BYTES = FS2_STREAM_BYTES + SHIPPED_MODEL_BYTES  # 126,926
FIELD = STORE / "inputs/tokens_null.u8"
FIELD_SHA256 = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
N, H, W = 600, 384, 512
FIELD_BYTES = N * H * W
# Charter arithmetic (DERIVED from the fs2 pointer; re-derive at every pointer move).
DEMAND_BYTES = 41_817.8
RATE_CORNER_ARCHIVE_BYTES = 138_205.2
S_PER_BYTE = 25.0 / 37_545_489
AXIS = "[macOS-CPU advisory / scorer-free EXACT byte measurement]"
CHECKPOINT_SCHEMA = "ddm_cl1_hpac_capacity_checkpoint.v2"
RUNG_LAMBDA = {
    "lambda_1p0": 1.0,
    "lambda_0p5": 0.5,
    "lambda_0p25": 0.25,
    "cpu_control_jf2_null": 1.0,
}


class Cl2Error(RuntimeError):
    pass


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def persist_bytes(path: Path, payload: bytes, *, label: str) -> dict[str, Any]:
    """ALWAYS KEEP THE PAYLOAD: persist the bytes, then describe them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    try:
        tmp.write_bytes(payload)
        os.replace(tmp, path)
    except OSError as exc:
        raise Cl2Error(f"{label}: could not persist {path}: {exc}") from exc
    return file_fact(path)


def progress(record: dict[str, Any]) -> None:
    record.setdefault("utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print(json.dumps(record, sort_keys=True), flush=True)


def verify_inputs() -> dict[str, Any]:
    if not FS2_ARCHIVE.is_file():
        raise Cl2Error(f"fs2 fire tree archive is absent: {FS2_ARCHIVE}")
    archive = file_fact(FS2_ARCHIVE)
    if archive["bytes"] != FS2_ARCHIVE_BYTES or archive["sha256"] != FS2_ARCHIVE_SHA256:
        raise Cl2Error("fs2 archive custody failed (bytes or sha differ from the pointer)")
    if not FIELD.is_file():
        raise Cl2Error(f"token field is absent: {FIELD}")
    field = file_fact(FIELD)
    if field["bytes"] != FIELD_BYTES or field["sha256"] != FIELD_SHA256:
        raise Cl2Error("token field custody failed")
    return {"fs2_archive": archive, "field": field}


def copy_runtime_tree(destination: Path) -> None:
    if destination.exists():
        return
    shutil.copytree(
        FS2_RUNTIME,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "._*"),
    )


def replace_hpac_section(member: bytes, compressed_hpac: bytes) -> bytes:
    sections = jg2.split_member(member)
    magic, version, a, b, reserved, _hpac, semantic, carrier = jg2.RX1_HEADER.unpack(sections["header"])
    if len(compressed_hpac) > 65_535:
        raise Cl2Error("packed HPAC section exceeds the uint16 header field")
    header = jg2.RX1_HEADER.pack(magic, version, a, b, reserved, len(compressed_hpac), semantic, carrier)
    return header + compressed_hpac + sections["semantic"] + sections["carrier"] + sections["tail"]


def patch_inflate_pins(runtime_root: Path, archive_sha256: str, archive_bytes: int) -> dict[str, Any]:
    """Patch BOTH pins in the receiver copy's inflate.py (jf2 #1237: a half-updated pin)."""
    path = runtime_root / "inflate.py"
    text = path.read_text(encoding="utf-8")
    old_sha = f'ARCHIVE_SHA256 = "{FS2_ARCHIVE_SHA256}"'
    old_bytes = f"ARCHIVE_BYTES = {FS2_ARCHIVE_BYTES:_}"
    new_sha = f'ARCHIVE_SHA256 = "{archive_sha256}"'
    new_bytes = f"ARCHIVE_BYTES = {archive_bytes:_}"
    if text.count(old_sha) != 1 or text.count(old_bytes) != 1:
        if text.count(new_sha) == 1 and text.count(new_bytes) == 1:
            return file_fact(path)
        raise Cl2Error("receiver copy inflate.py pins are absent or ambiguous")
    text = text.replace(old_sha, new_sha).replace(old_bytes, new_bytes)
    path.write_text(text, encoding="utf-8")
    if text.count(new_sha) != 1 or text.count(new_bytes) != 1:
        raise Cl2Error("inflate.py pin patch did not land exactly once")
    return file_fact(path)


def load_checkpoint_facts(checkpoint: Path, expected_profile: str) -> dict[str, Any]:
    import torch

    from tools import train_ddm_cl1_hpac_capacity as trainer

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("run_identity", {}).get("training_config", {})
    if (
        payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("epoch") != 60
        or payload.get("phase") != "discrete_qat"
        or payload.get("deployment_weights") != "ema_shadow"
        or config.get("profile") != expected_profile
    ):
        raise Cl2Error(
            f"checkpoint is not a terminal epoch-60 EMA QAT checkpoint of profile {expected_profile}: "
            f"schema={payload.get('schema')} epoch={payload.get('epoch')} phase={payload.get('phase')} "
            f"profile={config.get('profile')}"
        )
    if trainer._causal_state_sha256(payload) != payload.get("causal_state_sha256"):
        raise Cl2Error("checkpoint causal-state hash does not verify")
    return {
        "checkpoint": file_fact(checkpoint),
        "profile": config.get("profile"),
        "device": config.get("device"),
        "rate_lambda": config.get("rate_lambda"),
        "epochs": config.get("epochs"),
        "seed": config.get("seed"),
        "init_sha256": payload.get("run_identity", {}).get("init_sha256"),
        "cache_sha256": payload.get("run_identity", {}).get("cache_sha256"),
        "causal_state_sha256": payload.get("causal_state_sha256"),
        "best_by_surrogate": {
            key: payload.get("best", {}).get(key)
            for key in ("epoch", "estimated_model_bytes", "estimated_token_bytes", "estimated_joint_bytes")
        },
    }


# --------------------------------------------------------------------------------------
# stage: pack
# --------------------------------------------------------------------------------------


def pack_model(checkpoint: Path, retained: Path) -> dict[str, Any]:
    packed = rx2._pack_terminal_ihs1(checkpoint, retained / "model")
    raw = Path(packed["raw"]["path"]).read_bytes()
    representations: list[dict[str, Any]] = []
    for quality in range(12):
        payload = rx2.rx1._brotli(raw, quality, "brotli")
        if rx2.rx1._brotli_restore(payload, "brotli") != raw:
            raise Cl2Error(f"Brotli q{quality} model parse-back failed")
        fact = persist_bytes(retained / "model" / f"hpac.ihs1.br.q{quality}", payload, label="model representation")
        representations.append({"quality": quality, "payload": fact})
    winner = min(representations, key=lambda row: (row["payload"]["bytes"], row["quality"]))
    return {"pack": packed, "raw_bytes": len(raw), "representations": representations, "winner": winner}


# --------------------------------------------------------------------------------------
# stage: encode (twice) + candidate + decode
# --------------------------------------------------------------------------------------


def decode_with_receiver(env: dict[str, Any], archive: Path) -> tuple[np.ndarray, dict[str, Any], float]:
    import torch

    previous = os.environ.get("CPR1_RC64_LIBRARY")
    os.environ["CPR1_RC64_LIBRARY"] = str(env["library"])
    try:
        parts = env["residual"].read_residual_archive(archive)
        started = time.perf_counter()
        decoded_tensor, report = env["residual"].decode_production_tokens(
            parts, env["renderer"], env["renderer_dir"], torch.device("cpu")
        )
        elapsed = time.perf_counter() - started
    finally:
        if previous is None:
            os.environ.pop("CPR1_RC64_LIBRARY", None)
        else:
            os.environ["CPR1_RC64_LIBRARY"] = previous
    return decoded_tensor.numpy(), report, elapsed


def stage_control(args: argparse.Namespace) -> dict[str, Any]:
    """Decode the SHIPPED archive through the same receiver copy: identity + wall-clock."""
    facts = verify_inputs()
    root = STORE / "control"
    result_path = root / "CONTROL_RESULT.json"
    if result_path.is_file() and not args.force:
        raise Cl2Error(f"control already recorded at {result_path}; pass --force to redo")
    retained = root / "retained"
    staged = retained / "shipped_runtime"
    copy_runtime_tree(staged)
    if sha256_file(staged / "archive.zip") != FS2_ARCHIVE_SHA256:
        raise Cl2Error("shipped runtime copy carries foreign archive bytes")
    atomic_json(
        root / "CONTROL_START.json", {"schema": "ddm_cl2_control_start.v1", "inputs": facts, "staged": str(staged)}
    )
    env = jg2._prepare(SimpleNamespace(store=str(root), runtime_root=str(staged)), "cl2_control")
    shipped_stream = env["parts"].token_stream
    if len(shipped_stream) != FS2_STREAM_BYTES or hashlib.sha256(shipped_stream).hexdigest() != FS2_STREAM_SHA256:
        raise Cl2Error("shipped token stream custody failed")
    target = jg2.load_tokens(FIELD)
    decoded, report, elapsed = decode_with_receiver(env, staged / "archive.zip")
    identity = bool(np.array_equal(decoded, np.asarray(target)))
    decoded_fact = persist_bytes(retained / "decoded_tokens.u8", decoded.tobytes(order="C"), label="decoded field")
    result = {
        "schema": "ddm_cl2_control.v1",
        "axis": AXIS,
        "score_claim": False,
        "inputs": facts,
        "rc64_build": env["build"],
        "shipped_model_section_bytes": len(jg2.split_member(jg2.read_archive_member(FS2_ARCHIVE))["hpac"]),
        "shipped_stream": {"bytes": len(shipped_stream), "sha256": hashlib.sha256(shipped_stream).hexdigest()},
        "decoded": decoded_fact,
        "decoded_identity": identity,
        "decode_report": report,
        "decode_wall_clock_seconds": elapsed,
    }
    atomic_json(result_path, result)
    if not identity:
        raise Cl2Error("CONTROL FAILED: the shipped archive did not decode to the retained field")
    return result


def stage_price(args: argparse.Namespace) -> dict[str, Any]:
    facts = verify_inputs()
    rung = args.rung
    if rung not in RUNG_LAMBDA:
        raise Cl2Error(f"unknown rung {rung}; admitted: {sorted(RUNG_LAMBDA)}")
    root = STORE / "rungs" / rung
    result_path = root / "RUNG_RESULT.json"
    if result_path.is_file() and not args.force:
        raise Cl2Error(f"rung already priced at {result_path}; pass --force to redo")
    retained = root / "retained"
    retained.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    ck = load_checkpoint_facts(checkpoint, args.expected_profile)
    if ck["rate_lambda"] != RUNG_LAMBDA[rung]:
        raise Cl2Error(f"checkpoint rate_lambda {ck['rate_lambda']} does not match rung {rung}")
    if ck["cache_sha256"] is None or ck["init_sha256"] is None:
        raise Cl2Error("checkpoint carries no input custody")
    atomic_json(
        root / "RUNG_START.json",
        {"schema": "ddm_cl2_rung_start.v1", "rung": rung, "inputs": facts, "checkpoint": ck},
    )
    retained_checkpoint = retained / "terminal_checkpoint.pt"
    if not retained_checkpoint.is_file():
        shutil.copyfile(checkpoint, retained_checkpoint)
    if sha256_file(retained_checkpoint) != ck["checkpoint"]["sha256"]:
        raise Cl2Error("retained checkpoint copy drifted")

    # 1. pack -------------------------------------------------------------------------
    model = pack_model(retained_checkpoint, retained)
    winner_fact = model["winner"]["payload"]
    compressed_hpac = Path(winner_fact["path"]).read_bytes()
    progress({"stage": "pack", "rung": rung, "raw_bytes": model["raw_bytes"], "packed_bytes": winner_fact["bytes"]})

    # 2. stage the receiver copy with the new model section -----------------------------
    staged = retained / "model_only_runtime"
    copy_runtime_tree(staged)
    base_member = jg2.read_archive_member(FS2_ARCHIVE)
    model_member = replace_hpac_section(base_member, compressed_hpac)
    model_archive = retained / "model_only_runtime_archive.zip"
    jg2.pack_archive(model_member, model_archive)
    shutil.copyfile(model_archive, staged / "archive.zip")
    env = jg2._prepare(SimpleNamespace(store=str(root), runtime_root=str(staged)), f"cl2_{rung}")
    if env["sections"]["hpac"] != compressed_hpac:
        raise Cl2Error("staged archive does not carry the packed model section")
    target = jg2.load_tokens(FIELD)

    # 3. encode twice ---------------------------------------------------------------------
    encodes: list[dict[str, Any]] = []
    for pass_index, tag in enumerate((f"cl2_{rung}", f"cl2_{rung}_repeat")):
        if pass_index == 1 and args.skip_repeat:
            break
        encoded = jg2.encode_tail(
            residual=env["residual"],
            renderer=env["renderer"],
            renderer_dir=env["renderer_dir"],
            parts=env["parts"],
            target=target,
            library=env["library"],
            route_b=env["route_b"],
            work=env["work"],
            tag=tag,
            frames=N,
            checkpoint_every=20,
            resume=True,
        )
        encodes.append(encoded)
        progress(
            {
                "stage": "encode",
                "rung": rung,
                "pass": pass_index,
                "stream_bytes": encoded["stream"]["bytes"],
                "code_bytes_ideal": encoded["code_bytes_ideal"],
                "elapsed_seconds": encoded["elapsed_seconds"],
            }
        )
    stream_path = Path(encodes[0]["stream"]["path"])
    emitted = stream_path.read_bytes()
    two_encodes_identical = len(encodes) == 2 and Path(encodes[1]["stream"]["path"]).read_bytes() == emitted
    stream_fact = persist_bytes(retained / "token_stream.rc64.bin", emitted, label="rung stream")

    # 4. candidate archive + receiver copy + decode --------------------------------------
    sections = dict(env["sections"])
    sections["tail"] = sections["residual_compact"] + emitted
    candidate = retained / "candidate_archive.zip"
    jg2.pack_archive(jg2.join_member(sections), candidate)
    candidate_fact = file_fact(candidate)
    receiver_copy = retained / "receiver_copy_runtime"
    copy_runtime_tree(receiver_copy)
    shutil.copyfile(candidate, receiver_copy / "archive.zip")
    pin_fact = patch_inflate_pins(receiver_copy, candidate_fact["sha256"], candidate_fact["bytes"])
    decoded, decode_report, decode_seconds = decode_with_receiver(env, receiver_copy / "archive.zip")
    identity = bool(np.array_equal(decoded, np.asarray(target)))
    decoded_fact = persist_bytes(retained / "decoded_tokens.u8", decoded.tobytes(order="C"), label="decoded field")

    model_bytes = int(winner_fact["bytes"])
    stream_bytes = int(stream_fact["bytes"])
    joint = model_bytes + stream_bytes
    control_path = STORE / "control" / "CONTROL_RESULT.json"
    control_decode = None
    if control_path.is_file():
        control_decode = json.loads(control_path.read_text(encoding="utf-8")).get("decode_wall_clock_seconds")
    result = {
        "schema": "ddm_cl2_rung_price.v1",
        "axis": AXIS,
        "score_claim": False,
        "rung": rung,
        "rate_lambda": RUNG_LAMBDA[rung],
        "inputs": facts,
        "checkpoint": ck,
        "retained_checkpoint": file_fact(retained_checkpoint),
        "rc64_build": env["build"],
        "model": model,
        "model_raw_ihs1_bytes": model["raw_bytes"],
        "model_packed_bytes": model_bytes,
        "model_delta_vs_shipped": model_bytes - SHIPPED_MODEL_BYTES,
        "model_only_runtime_archive": file_fact(model_archive),
        "encodes": encodes,
        "stream": stream_fact,
        "stream_bytes": stream_bytes,
        "stream_delta_vs_shipped": stream_bytes - FS2_STREAM_BYTES,
        "two_encodes_identical": two_encodes_identical,
        "joint_bytes": joint,
        "joint_delta_vs_shipped_126926": joint - SHIPPED_JOINT_BYTES,
        "fraction_of_demand_41818": (SHIPPED_JOINT_BYTES - joint) / DEMAND_BYTES,
        "candidate_archive": candidate_fact,
        "candidate_archive_delta_vs_fs2_180023": candidate_fact["bytes"] - FS2_ARCHIVE_BYTES,
        "candidate_rate_only_delta_S": (candidate_fact["bytes"] - FS2_ARCHIVE_BYTES) * S_PER_BYTE,
        "receiver_copy_runtime": {"path": str(receiver_copy), "inflate_pins": pin_fact},
        "decoded": decoded_fact,
        "decoded_identity": identity,
        "decode_report": decode_report,
        "decode_wall_clock_seconds": decode_seconds,
        "shipped_decode_wall_clock_seconds": control_decode,
        "decode_wall_clock_delta_seconds": (decode_seconds - control_decode) if control_decode else None,
        "distortion": "HELD by construction (field bit-identical); NOT re-measured",
    }
    atomic_json(result_path, result)
    if not identity:
        raise Cl2Error(f"{rung}: candidate did not decode to the target field; row inadmissible")
    if len(encodes) == 2 and not two_encodes_identical:
        raise Cl2Error(f"{rung}: two encodes differ; row inadmissible")
    return result


# --------------------------------------------------------------------------------------
# stage: report (slopes + decision rule)
# --------------------------------------------------------------------------------------


def adjacent_slope(left: str, left_row: dict[str, Any], right: str, right_row: dict[str, Any]) -> dict[str, Any]:
    """cl1's break-even between two adjacent rungs: delta(stream)/delta(model) < -1 pays.

    A rung that grows the model by ``d_model`` bytes must cut the stream by MORE than
    ``d_model`` bytes; equivalently the joint must fall.  When the model does not grow
    (``d_model <= 0``) there is no capacity to repay, so ``pays`` is simply ``joint fell``.
    """
    d_model = int(right_row["model_packed_bytes"]) - int(left_row["model_packed_bytes"])
    d_stream = int(right_row["stream_bytes"]) - int(left_row["stream_bytes"])
    slope = (d_stream / d_model) if d_model else None
    pays = (slope < -1.0) if (d_model > 0 and slope is not None) else (d_model + d_stream < 0)
    return {
        "from": left,
        "to": right,
        "delta_model_bytes": d_model,
        "delta_stream_bytes": d_stream,
        "slope_stream_per_model": slope,
        "pays": bool(pays),
        "delta_joint_bytes": d_model + d_stream,
    }


def stage_report(_args: argparse.Namespace) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for rung in RUNG_LAMBDA:
        path = STORE / "rungs" / rung / "RUNG_RESULT.json"
        if path.is_file():
            rows[rung] = json.loads(path.read_text(encoding="utf-8"))
    ladder = [rung for rung in ("lambda_1p0", "lambda_0p5", "lambda_0p25") if rung in rows]
    slopes: list[dict[str, Any]] = [
        adjacent_slope(left, rows[left], right, rows[right]) for left, right in itertools.pairwise(ladder)
    ]
    control = rows.get("lambda_1p0")
    control_gap = None if control is None else control["joint_delta_vs_shipped_126926"]
    best = min(rows.values(), key=lambda row: row["joint_bytes"]) if rows else None
    decision = "NO_ROWS"
    if control is not None:
        if control_gap > 500:
            decision = "INSTRUMENT-REFUSED"
        elif best is not None and best["candidate_archive"]["bytes"] < FS2_ARCHIVE_BYTES and best["decoded_identity"]:
            decision = (
                "FIRE-ORDER"
                if best["candidate_archive"]["bytes"] <= RATE_CORNER_ARCHIVE_BYTES
                else "READY-FOR-T4-CANDIDATE"
            )
        else:
            decision = "REFUSED"
    table = {
        rung: {
            "rate_lambda": row["rate_lambda"],
            "model_packed_bytes": row["model_packed_bytes"],
            "stream_bytes": row["stream_bytes"],
            "joint_bytes": row["joint_bytes"],
            "joint_delta_vs_shipped_126926": row["joint_delta_vs_shipped_126926"],
            "fraction_of_demand_41818": row["fraction_of_demand_41818"],
            "candidate_archive_bytes": row["candidate_archive"]["bytes"],
            "decoded_identity": row["decoded_identity"],
            "two_encodes_identical": row["two_encodes_identical"],
            "decode_wall_clock_seconds": row["decode_wall_clock_seconds"],
        }
        for rung, row in rows.items()
    }
    report = {
        "schema": "ddm_cl2_ladder_report.v1",
        "axis": AXIS,
        "score_claim": False,
        "shipped": {
            "model_bytes": SHIPPED_MODEL_BYTES,
            "stream_bytes": FS2_STREAM_BYTES,
            "joint_bytes": SHIPPED_JOINT_BYTES,
        },
        "control_reproduction_gap_joint_bytes": control_gap,
        "control_tolerance_bytes": 500,
        "rows": table,
        "adjacent_slopes": slopes,
        "best_rung": None if best is None else best["rung"],
        "decision": decision,
        "demand_bytes": DEMAND_BYTES,
        "rate_corner_archive_bytes": RATE_CORNER_ARCHIVE_BYTES,
    }
    atomic_json(STORE / "LADDER_REPORT.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    control = sub.add_parser(
        "control", help="decode the shipped fs2 archive through the receiver copy (identity + wall-clock)"
    )
    control.add_argument("--force", action="store_true")
    price = sub.add_parser("price", help="exact price of one rung through the shipped pack + mixer + RC64 path")
    price.add_argument("--rung", required=True, choices=sorted(RUNG_LAMBDA))
    price.add_argument("--checkpoint", required=True, type=Path)
    price.add_argument(
        "--expected-profile", default="cl2_shipped_ladder", choices=("cl2_shipped_ladder", "jf1_joint_refit")
    )
    price.add_argument(
        "--skip-repeat", action="store_true", help="skip the second encode (the row is then NOT admissible)"
    )
    price.add_argument("--force", action="store_true")
    sub.add_parser("report", help="ladder table, adjacent slopes and the pre-registered decision")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "control":
        stage_control(args)
    elif args.stage == "price":
        stage_price(args)
    else:
        stage_report(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
