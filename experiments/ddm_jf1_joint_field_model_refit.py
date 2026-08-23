#!/usr/bin/env python3
"""JF1 joint token-field/model refit on the retained LD1 ladder.

The trainer remains the owned HPAC reference implementation.  This runner owns
the byte-close half: input custody, exact IHS1 packing, a complete Brotli
representation race, native-RC64 re-encoding through the DX2 receiver model,
and a decoded-token identity proof.  It never invokes a scorer and never writes
either SSD tier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_jg2_tail_reencode as jg2
from experiments import ddm_rx2_mc36_identity_race as rx2
from tools import train_ddm_cl1_hpac_capacity as trainer

STORE = REPO / ".omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit"
DX2_RUNTIME = Path("/Volumes/APDataStore/pact/ddm_dx2/r7/candidate_runtime_dx2")
DX2_ARCHIVE = DX2_RUNTIME / "archive.zip"
DX2_ARCHIVE_BYTES = 180_368
DX2_ARCHIVE_SHA256 = "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674"
BASE_FIELD = Path(
    "/Volumes/VertigoDataTier/pact/ddm_bl1_per_position_bit_allocation/"
    "measurement_v1/retained/fields/decoded_tokens_instrumented.u8"
)
BASE_FIELD_SHA256 = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
SOURCE_CHECKPOINT = Path(
    "/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/"
    "full_e480b_e960/checkpoints/full_mps_e960.checkpoints/periodic/epoch_0634.pt"
)
SOURCE_CHECKPOINT_SHA256 = "5007beae7af7789758092f12f49096e13692e2e59850c85eb4642cd6fad147ec"
GT_DALI = Path("/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/retained/inputs/gt_argmax_n600.npy")
GT_DALI_SHA256 = "91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248"
LD1_ROOT = (
    REPO
    / ".omx/tmp/arm_receipts_local/ddm_ld1_lane_lossy_drop_exchange/"
    "measurement_v1"
)
LD1_FIXED_STREAM_BYTES = {
    "k002500": 113_973,
    "k005000": 114_056,
    "k010000": 114_601,
    "k020000": 115_305,
    "k040000": 114_375,
    "k060000": 113_798,
}
FIELD_SHA256 = {
    "null": BASE_FIELD_SHA256,
    "k002500": "c45979acb7a87bdae41fe23d67c9efd10661d5320e5e0c84f9d863a743b3831e",
    "k005000": "6c210dd19eefb2b67dad5c5f93ee8008a625b8aea50e685553ee5335f179f000",
    "k010000": "297cee64f3e1438b985f9b242d6405ad5521b5cf320865390bc0ca105fe8351d",
    "k020000": "7251367a078796a12c2302d726d2d5b1941c9d35d5755745fc664f29de0344fb",
    "k040000": "03ce7bd8a8498ea2a1fc61a0191d0c9eeab3e5ff729e7d522dc07f64add08093",
    "k060000": "15018481bd8007dd9099d1b67d5e8014283465d062a34ba3f06b3450758b5878",
}
N, H, W = 600, 384, 512
FIELD_BYTES = N * H * W
SHIPPED_STREAM_BYTES = 113_777
SHIPPED_STREAM_SHA256 = "e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5"
SHIPPED_MODEL_BYTES = 13_515
SHIPPED_COMBINED_BYTES = SHIPPED_STREAM_BYTES + SHIPPED_MODEL_BYTES
S_PER_BYTE = 25.0 / 37_545_489
AXIS = "[macOS-CPU advisory / scorer-free exact model-pack and RC64 measurement]"


class JF1Error(RuntimeError):
    """A JF1 custody, reference-form, or identity gate refused."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))


def atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def copy_exact(source: Path, destination: Path, expected_sha256: str) -> dict[str, Any]:
    if source.stat().st_size < 1 or sha256_file(source) != expected_sha256:
        raise JF1Error(f"source pin failed: {source}")
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    if sha256_file(destination) != expected_sha256:
        raise JF1Error(f"retained copy changed bytes: {destination}")
    return file_record(destination)


def source_field(tag: str) -> Path:
    if tag == "null":
        return BASE_FIELD
    return (
        LD1_ROOT
        / "retained/fields"
        / f"tokens_lane2road_topcost_{tag}.u8"
    )


def retained_field(tag: str) -> Path:
    return STORE / "inputs/fields" / f"tokens_{tag}.u8"


def cache_path(tag: str) -> Path:
    return STORE / "inputs/caches" / f"tokens_{tag}.pt"


def training_checkpoint(tag: str, fitting_epoch: int) -> Path:
    root = STORE / "training" / tag / "model.checkpoints"
    if fitting_epoch == 60:
        return root / "qat_stage_end_epoch_0060.pt"
    return root / "periodic" / f"epoch_{fitting_epoch:04d}.pt"


def measurement_root(tag: str, fitting_epoch: int) -> Path:
    if fitting_epoch == 60:
        return STORE / "rows" / tag
    return STORE / f"scope_e{fitting_epoch:04d}" / "rows" / tag


def _verify_dx2() -> dict[str, Any]:
    if DX2_ARCHIVE.stat().st_size != DX2_ARCHIVE_BYTES or sha256_file(DX2_ARCHIVE) != DX2_ARCHIVE_SHA256:
        raise JF1Error("DX2 archive drifted")
    residual, _renderer, _code_dir = jg2.load_runtime(DX2_RUNTIME)
    parts = residual.read_residual_archive(DX2_ARCHIVE)
    stream_path = STORE / "inputs/shipped/dx2_token_stream.rc64.bin"
    atomic_bytes(stream_path, parts.token_stream)
    if len(parts.token_stream) != SHIPPED_STREAM_BYTES or sha256_file(stream_path) != SHIPPED_STREAM_SHA256:
        raise JF1Error("DX2 RC64 stream drifted")
    if len(parts.hpac_blob) != 17_952 or hashlib.sha256(parts.hpac_blob).hexdigest() != "e8c0cfd73d3275adeff2897ea83efa9d045855c43fb3bb66ac037e5c84f2e6dd":
        raise JF1Error("DX2 decompressed IHS1 object drifted")
    return {
        "archive": file_record(DX2_ARCHIVE),
        "stream": file_record(stream_path),
        "ihs1_raw_bytes": len(parts.hpac_blob),
        "ihs1_raw_sha256": hashlib.sha256(parts.hpac_blob).hexdigest(),
    }


def _verify_source_checkpoint() -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(SOURCE_CHECKPOINT) != SOURCE_CHECKPOINT_SHA256:
        raise JF1Error("epoch-634 checkpoint drifted")
    checkpoint = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema") != trainer.CHECKPOINT_SCHEMA
        or checkpoint.get("epoch") != 634
        or checkpoint.get("phase") != "discrete_qat"
        or checkpoint.get("deployment_weights") != "ema_shadow"
        or checkpoint.get("run_identity", {}).get("training_config", {}).get("profile") != "rx2_mc36"
    ):
        raise JF1Error("epoch-634 checkpoint is not the shipped RX2 EMA-QAT source")
    if trainer._causal_state_sha256(checkpoint) != checkpoint.get("causal_state_sha256"):
        raise JF1Error("epoch-634 checkpoint causal-state hash failed")
    retained = STORE / "inputs/shipped/epoch_0634.pt"
    checkpoint_record = copy_exact(SOURCE_CHECKPOINT, retained, SOURCE_CHECKPOINT_SHA256)
    init_path = STORE / "inputs/shipped/epoch_0634_ema_init.pt"
    if not init_path.exists():
        atomic_torch(
            init_path,
            {
                "state_dict": checkpoint["state_dict"],
                "deployment_weights": "ema_shadow",
                "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            },
        )
    init = torch.load(init_path, map_location="cpu", weights_only=False)
    if init.get("source_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256:
        raise JF1Error("warm-start source lineage is absent or changed")
    return checkpoint_record, file_record(init_path)


def prepare() -> dict[str, Any]:
    STORE.mkdir(parents=True, exist_ok=True)
    dx2 = _verify_dx2()
    checkpoint, init = _verify_source_checkpoint()
    gt_retained = STORE / "inputs/gt/gt_argmax_n600.dali.npy"
    gt_record = copy_exact(GT_DALI, gt_retained, GT_DALI_SHA256)
    fields: dict[str, Any] = {}
    for tag, expected in FIELD_SHA256.items():
        copied = copy_exact(source_field(tag), retained_field(tag), expected)
        cache = cache_path(tag)
        if not cache.exists():
            field = np.memmap(retained_field(tag), dtype=np.uint8, mode="r", shape=(N, H, W))
            atomic_torch(
                cache,
                {
                    "seg": torch.from_numpy(np.asarray(field).copy()),
                    "spatial_token_sha256": expected,
                    "source_field": copied,
                },
            )
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        observed = trainer._verify_jf1_cache_payload(payload, expected)
        fields[tag] = {
            "source": file_record(source_field(tag)),
            "retained": copied,
            "cache": file_record(cache),
            "cache_content_sha256": observed,
        }
    result = {
        "schema": "ddm_jf1_prepare.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": False,
        "storage_tier": str(STORE),
        "storage_policy": "explicit local opt-in; both SSD tiers read-only for this arm",
        "dx2": dx2,
        "source_checkpoint": checkpoint,
        "warm_start": init,
        "gt_dali_source": file_record(GT_DALI),
        "gt_dali_retained": gt_record,
        "fields": fields,
        "reference_form": {
            "profile": "jf1_joint_refit",
            "same_as_rx2_config": trainer.JF1_PREREGISTERED_CONFIG == trainer.RX2_PREREGISTERED_CONFIG,
            "architecture": "IntegerHPAC C64 patch64 delta2 frame_dim8 raw target",
            "member_count_changed": False,
            "coder_family_changed": False,
            "fitting_budget": "60 epochs, same full RX2 two-phase schedule; warm-started from shipped epoch-634 EMA",
        },
    }
    atomic_json(STORE / "PREPARE_RESULT.json", result)
    return result


def _replace_hpac(member: bytes, compressed_hpac: bytes) -> bytes:
    sections = jg2.split_member(member)
    magic, version, a, b, reserved, _hpac, semantic, carrier = jg2.RX1_HEADER.unpack(sections["header"])
    if len(compressed_hpac) > 65_535:
        raise JF1Error("refit HPAC physical section exceeds uint16 header capacity")
    header = jg2.RX1_HEADER.pack(
        magic,
        version,
        a,
        b,
        reserved,
        len(compressed_hpac),
        semantic,
        carrier,
    )
    return header + compressed_hpac + sections["semantic"] + sections["carrier"] + sections["tail"]


def _pack_model(
    tag: str,
    checkpoint_path: Path,
    retained: Path,
    fitting_epoch: int,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_phase = "discrete_qat" if fitting_epoch > 30 else "continuous"
    if (
        checkpoint.get("schema") != trainer.CHECKPOINT_SCHEMA
        or checkpoint.get("epoch") != fitting_epoch
        or checkpoint.get("phase") != expected_phase
        or checkpoint.get("deployment_weights") != "ema_shadow"
        or checkpoint.get("run_identity", {}).get("training_config", {}).get("profile")
        != "jf1_joint_refit"
    ):
        raise JF1Error(
            f"{tag}: checkpoint is not JF1 EMA {expected_phase} epoch {fitting_epoch}"
        )
    if trainer._causal_state_sha256(checkpoint) != checkpoint.get("causal_state_sha256"):
        raise JF1Error(f"{tag}: terminal checkpoint causal-state hash failed")
    packed = rx2._pack_terminal_ihs1(checkpoint_path, retained / "model")
    raw_path = Path(packed["raw"]["path"])
    raw = raw_path.read_bytes()
    representations: list[dict[str, Any]] = []
    for quality in range(12):
        payload = rx2.rx1._brotli(raw, quality, "brotli")
        path = retained / "model" / f"hpac.ihs1.br.q{quality}"
        atomic_bytes(path, payload)
        if rx2.rx1._brotli_restore(payload, "brotli") != raw:
            raise JF1Error(f"{tag}: Brotli q{quality} model parse-back failed")
        representations.append({"quality": quality, "payload": file_record(path)})
    winner = min(representations, key=lambda row: (row["payload"]["bytes"], row["quality"]))
    return {"pack": packed, "representations": representations, "winner": winner}


def _prepare_staged_runtime(tag: str, compressed_hpac: bytes, retained: Path) -> tuple[Path, dict[str, Any]]:
    base_member = jg2.read_archive_member(DX2_ARCHIVE)
    model_member = _replace_hpac(base_member, compressed_hpac)
    model_archive = retained / "model_only_runtime_archive.zip"
    jg2.pack_archive(model_member, model_archive)
    staged = retained / "model_only_runtime"
    if not staged.exists():
        shutil.copytree(DX2_RUNTIME, staged)
    destination = staged / "archive.zip"
    if destination.exists() and sha256_file(destination) not in {
        DX2_ARCHIVE_SHA256,
        sha256_file(model_archive),
    }:
        raise JF1Error(f"{tag}: staged runtime archive has foreign bytes")
    if sha256_file(destination) != sha256_file(model_archive):
        shutil.copyfile(model_archive, destination)
    return staged, file_record(model_archive)


def _patch_runtime_archive_pin(runtime_root: Path, archive_sha256: str) -> dict[str, Any]:
    path = runtime_root / "inflate.py"
    text = path.read_text(encoding="utf-8")
    if archive_sha256 not in text:
        if text.count(DX2_ARCHIVE_SHA256) != 1:
            raise JF1Error("candidate runtime archive pin is absent or ambiguous")
        text = text.replace(DX2_ARCHIVE_SHA256, archive_sha256)
        atomic_bytes(path, text.encode("utf-8"))
    if path.read_text(encoding="utf-8").count(archive_sha256) != 1:
        raise JF1Error("candidate runtime archive pin patch did not land exactly once")
    return file_record(path)


def _retain_decoded_masks(decoded: np.ndarray, retained: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for class_id in range(5):
        payload = np.packbits(decoded == class_id, bitorder="little").tobytes()
        path = retained / "decoded_token_class_masks" / f"class_{class_id}.n600.packbits"
        atomic_bytes(path, payload)
        records.append({"class_id": class_id, **file_record(path)})
    return records


def measure(tag: str, fitting_epoch: int) -> dict[str, Any]:
    if tag not in FIELD_SHA256:
        raise JF1Error(f"unknown tag {tag}")
    if fitting_epoch < 2 or fitting_epoch > 60 or fitting_epoch % 2:
        raise JF1Error("--fitting-epoch must be an even evaluation checkpoint in 2..60")
    prepare_receipt = STORE / "PREPARE_RESULT.json"
    if not prepare_receipt.is_file():
        raise JF1Error("prepare stage is absent")
    root = measurement_root(tag, fitting_epoch)
    retained = root / "retained"
    result_path = root / "MEASURE_RESULT.json"
    if result_path.is_file():
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        for key in ("candidate_archive", "refit_stream", "decoded_tokens"):
            record = prior[key]
            path = Path(record["path"])
            if not path.is_file() or file_record(path) != record:
                raise JF1Error(f"{tag}: retained result artifact drifted: {key}")
        return prior
    checkpoint = training_checkpoint(tag, fitting_epoch)
    if not checkpoint.is_file():
        raise JF1Error(f"{tag}: terminal training checkpoint is absent: {checkpoint}")
    model = _pack_model(tag, checkpoint, retained, fitting_epoch)
    model_payload = Path(model["winner"]["payload"]["path"]).read_bytes()
    staged, model_archive = _prepare_staged_runtime(tag, model_payload, retained)
    args = SimpleNamespace(store=str(root), runtime_root=str(staged))
    env = jg2._prepare(args, f"refit_e{fitting_epoch:04d}_{tag}")
    target = jg2.load_tokens(retained_field(tag))
    encoded = jg2.encode_tail(
        residual=env["residual"],
        renderer=env["renderer"],
        renderer_dir=env["renderer_dir"],
        parts=env["parts"],
        target=target,
        library=env["library"],
        route_b=env["route_b"],
        work=env["work"],
        tag=f"refit_e{fitting_epoch:04d}_{tag}",
        frames=N,
        checkpoint_every=20,
        resume=True,
    )
    stream_path = Path(encoded["stream"]["path"])
    emitted = stream_path.read_bytes()
    sections = env["sections"]
    sections["tail"] = sections["residual_compact"] + emitted
    candidate = retained / "candidate_archive.zip"
    jg2.pack_archive(jg2.join_member(sections), candidate)
    candidate_runtime = retained / "candidate_runtime"
    if not candidate_runtime.exists():
        shutil.copytree(staged, candidate_runtime)
    shutil.copyfile(candidate, candidate_runtime / "archive.zip")
    candidate_runtime_pin = _patch_runtime_archive_pin(candidate_runtime, sha256_file(candidate))
    previous_decoder_library = os.environ.get("CPR1_RC64_LIBRARY")
    os.environ["CPR1_RC64_LIBRARY"] = str(env["library"])
    try:
        candidate_parts = env["residual"].read_residual_archive(candidate)
        decoded_tensor, decode_report = env["residual"].decode_production_tokens(
            candidate_parts,
            env["renderer"],
            env["renderer_dir"],
            torch.device("cpu"),
        )
    finally:
        if previous_decoder_library is None:
            os.environ.pop("CPR1_RC64_LIBRARY", None)
        else:
            os.environ["CPR1_RC64_LIBRARY"] = previous_decoder_library
    decoded = decoded_tensor.numpy()
    decoded_path = retained / "decoded_tokens.u8"
    atomic_bytes(decoded_path, decoded.tobytes(order="C"))
    identity = bool(np.array_equal(decoded, np.asarray(target)))
    masks = _retain_decoded_masks(decoded, retained)
    if not identity or sha256_file(decoded_path) != FIELD_SHA256[tag]:
        raise JF1Error(f"{tag}: refit stream did not decode to its target field")
    model_bytes = int(model["winner"]["payload"]["bytes"])
    stream_bytes = stream_path.stat().st_size
    combined = model_bytes + stream_bytes
    fixed_stream = SHIPPED_STREAM_BYTES if tag == "null" else LD1_FIXED_STREAM_BYTES[tag]
    result = {
        "schema": "ddm_jf1_measure.v1",
        "complete": True,
        "tag": tag,
        "fitting_epoch": fitting_epoch,
        "fitting_budget_scope": (
            "FULL_REFERENCE_60_EPOCHS"
            if fitting_epoch == 60
            else f"SCOPE_REDUCTION_EPOCH_{fitting_epoch}_OF_60"
        ),
        "axis": AXIS,
        "score_claim": False,
        "scorer_ran": False,
        "gt_lineage_for_pending_scorer": {
            "name": "contest-authority DALI/NVDEC n600 argmax",
            **file_record(STORE / "inputs/gt/gt_argmax_n600.dali.npy"),
        },
        "field": file_record(retained_field(tag)),
        "terminal_checkpoint": file_record(checkpoint),
        "model": model,
        "model_physical_bytes": model_bytes,
        "model_delta_vs_shipped": model_bytes - SHIPPED_MODEL_BYTES,
        "model_only_runtime_archive": model_archive,
        "refit_stream": file_record(stream_path),
        "refit_stream_bytes": stream_bytes,
        "fixed_model_stream_bytes": fixed_stream,
        "refit_contribution_bytes_refit_minus_fixed": stream_bytes - fixed_stream,
        "bytes_recovered_by_refit_fixed_minus_refit": fixed_stream - stream_bytes,
        "combined_stream_plus_model_bytes": combined,
        "combined_delta_vs_127292": combined - SHIPPED_COMBINED_BYTES,
        "candidate_archive": file_record(candidate),
        "candidate_runtime": {
            "path": str(candidate_runtime),
            "archive": file_record(candidate_runtime / "archive.zip"),
            "inflate_archive_pin": candidate_runtime_pin,
            "role": "measurement-only runtime for the queued local advisory scorer; not a shipping candidate",
        },
        "candidate_archive_delta_vs_dx2": candidate.stat().st_size - DX2_ARCHIVE_BYTES,
        "delta_S_rate_only": (candidate.stat().st_size - DX2_ARCHIVE_BYTES) * S_PER_BYTE,
        "encoded": encoded,
        "decoded_tokens": file_record(decoded_path),
        "decoded_token_identity": identity,
        "decoded_token_class_masks": masks,
        "decode_report": decode_report,
        "positive_control_stream_deficit_bytes": (
            stream_bytes - SHIPPED_STREAM_BYTES if tag == "null" else None
        ),
        "d_seg_per_class": None,
        "d_pose": None,
        "net_delta_S": None,
        "measurement_boundary": (
            "real n600 model/stream/archive bytes and token receiver identity measured; "
            "SegNet/PoseNet components queued because JF1 does not own the scorer lane"
        ),
    }
    atomic_json(result_path, result)
    return result


def finalize(fitting_epoch: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for tag in FIELD_SHA256:
        path = measurement_root(tag, fitting_epoch) / "MEASURE_RESULT.json"
        if path.is_file():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    rung_rows = [row for row in rows if row["tag"] != "null"]
    if not any(row["tag"] == "null" for row in rows) or len(rung_rows) < 4:
        raise JF1Error("finalize requires the null and at least four diagonal rows")
    best = min(rung_rows, key=lambda row: row["combined_stream_plus_model_bytes"])
    null = next(row for row in rows if row["tag"] == "null")
    byte_verdict = {
        "schema": "ddm_jf1_byte_diagonal.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": False,
        "fitting_epoch": fitting_epoch,
        "fitting_budget_scope": (
            "FULL_REFERENCE_60_EPOCHS"
            if fitting_epoch == 60
            else f"SCOPE_REDUCTION_EPOCH_{fitting_epoch}_OF_60"
        ),
        "rows": rows,
        "positive_control_stream_deficit_bytes": null["positive_control_stream_deficit_bytes"],
        "positive_control_passed": null["positive_control_stream_deficit_bytes"] <= 0,
        "best_tag_by_combined_bytes": best["tag"],
        "best_combined_bytes": best["combined_stream_plus_model_bytes"],
        "best_combined_delta_vs_127292": best["combined_delta_vs_127292"],
        "prior_refit_prediction_over_2000_recovered": any(
            row["bytes_recovered_by_refit_fixed_minus_refit"] > 2_000 for row in rung_rows
        ),
        "prior_total_prediction_below_127292": any(
            row["combined_stream_plus_model_bytes"] < SHIPPED_COMBINED_BYTES for row in rung_rows
        ),
        "all_diagonal_combined_byte_deltas_positive": all(
            row["combined_delta_vs_127292"] > 0 for row in rung_rows
        ),
        "scorer_status": "QUEUED-WITH-A-FIRE-ORDER",
        "verdict_scope": "BYTE-LEG ONLY; full joint score verdict withheld pending exclusive n600 scorer",
    }
    output = (
        STORE / "BYTE_DIAGONAL.json"
        if fitting_epoch == 60
        else STORE / f"BYTE_DIAGONAL_SCOPE_E{fitting_epoch:04d}.json"
    )
    atomic_json(output, byte_verdict)
    return byte_verdict


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("stage", choices=("prepare", "measure", "finalize"))
    value.add_argument("--tag", choices=tuple(FIELD_SHA256))
    value.add_argument("--fitting-epoch", type=int, default=60)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.stage == "prepare":
        result = prepare()
    elif args.stage == "measure":
        if args.tag is None:
            raise SystemExit("measure requires --tag")
        result = measure(args.tag, args.fitting_epoch)
    else:
        result = finalize(args.fitting_epoch)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
