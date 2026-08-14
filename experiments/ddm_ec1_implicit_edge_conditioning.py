#!/usr/bin/env python3
"""Receiver-derived latent edge-conditioning consumer for the CP135 vehicle.

``analyze`` performs a full-n600 scorer-free selectivity race on retained
contest-CUDA argmax fields.  It persists every derived context/prediction
field and every fitted table.  ``package`` puts a separately trained adapter
into the counted archive and patches the actual CP135 receiver to consume it.
This arm does not train on or dispatch to Modal.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.ddm_ec1_runtime import ec1_latent_conditioner as runtime

OUTPUT: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/edge_conditioned/ddm_ec1_20260814"
)
JS1C_ROOT: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/contest_cuda/ddm_js1c_20260814"
)
STAGE0: Final = JS1C_ROOT / "STAGE0_RESULT.json"
FIELDS: Final = JS1C_ROOT / "retained/fields"
GT_FIELD: Final = FIELDS / "gt_argmax_n600.npy"
BASE_FIELD: Final = FIELDS / "cp135_base_argmax_n600.npy"
TOKENS: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/candidates/cp135_base/retained/decoded_tokens_n600.npy"
)
BASE_RUNTIME: Final = Path(
    "/Volumes/APDataStore/pact/submittable_custody_mirror_20260811/cp135_packet/adapted_runtime"
)
BASE_ARCHIVE: Final = BASE_RUNTIME / "archive.zip"
RUNTIME_TEMPLATE: Final = REPO / "experiments/ddm_ec1_runtime/ec1_latent_conditioner.py"

STAGE0_SHA256: Final = "472fc816f6656ec0cdd37bd475598e8e9683260dc97adeb4163ead5ae90b3e67"
GT_SHA256: Final = "91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248"
BASE_FIELD_SHA256: Final = "7648ad42e9f21942f86e81b97cabf46b710af747bba0909f7837ef3891232727"
BASE_ARCHIVE_SHA256: Final = "6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6"
JS8_MEMO_SHA256: Final = "f9486d646ba7c70a509fb1a85b6bc84ffb09d5989edbaa045b9553cc16b162e5"
N: Final = 600
H: Final = 384
W: Final = 512
PIXELS: Final = N * H * W
BASE_FLIPS: Final = 34_970
BASE_BYTES: Final = 186_252
BASE_D_POSE: Final = 6.885642960696714e-6
BREAK_EVEN_FLIPS_PER_BYTE: Final = 0.785
CHUNK: Final = 24
SEED: Final = 20_260_814
HIDDEN: Final = 4
MAX_DELTA: Final = 0.25
REQUIRED_FREE_BYTES: Final = 1_500_000_000
AXIS: Final = "[contest-CUDA T4 retained-field analysis, n600, scorer-free]"


class EC1Error(RuntimeError):
    """A custody, retention, receiver, or analysis invariant failed."""


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_npy(path: Path, value: np.ndarray) -> None:
    stream = io.BytesIO()
    np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
    atomic_bytes(path, stream.getvalue())


def retain_exact(path: Path, value: bytes) -> None:
    if path.is_file():
        if path.read_bytes() != value:
            raise EC1Error(f"retained payload differs; preserve it and use a fresh output: {path}")
        return
    atomic_bytes(path, value)


def require_file(path: Path, *, digest: str | None = None, size: int | None = None) -> None:
    if not path.is_file():
        raise EC1Error(f"missing custody input: {path}")
    if size is not None and path.stat().st_size != size:
        raise EC1Error(f"custody size differs: {path}")
    if digest is not None and sha256_file(path) != digest:
        raise EC1Error(f"custody SHA-256 differs: {path}")


def storage_preflight(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output)
    row = {
        "schema": "ddm_ec1_storage_preflight.v1",
        "path": str(output.resolve()),
        "free_bytes": usage.free,
        "required_free_bytes": REQUIRED_FREE_BYTES,
        "pass": usage.free >= REQUIRED_FREE_BYTES,
        "retention_plan": "context and cross-fit prediction fields plus all fitted/model payloads remain on SSD",
        "cleanup_policy": "certify-or-block; no produced payload is deleted or moved",
    }
    atomic_json(output / "preflight/STORAGE_PREFLIGHT.json", row)
    if not row["pass"]:
        raise EC1Error("SSD storage preflight failed")
    return row


def custody_preflight(output: Path) -> dict[str, Any]:
    require_file(STAGE0, digest=STAGE0_SHA256)
    require_file(GT_FIELD, digest=GT_SHA256, size=117_964_928)
    require_file(BASE_FIELD, digest=BASE_FIELD_SHA256, size=117_964_928)
    require_file(TOKENS, size=117_964_928)
    require_file(BASE_ARCHIVE, digest=BASE_ARCHIVE_SHA256, size=BASE_BYTES)
    require_file(RUNTIME_TEMPLATE)
    stage = json.loads(STAGE0.read_text())
    if (
        int(stage["comparison"]["base_flips"]) != BASE_FLIPS
        or stage["selection_mode"] != "full population, no sampling, all 600 non-overlapping pairs"
    ):
        raise EC1Error("JS1C matched-field instrument differs")
    fields = {
        "gt": np.load(GT_FIELD, mmap_mode="r", allow_pickle=False),
        "base": np.load(BASE_FIELD, mmap_mode="r", allow_pickle=False),
        "tokens": np.load(TOKENS, mmap_mode="r", allow_pickle=False),
    }
    if any(value.shape != (N, H, W) or value.dtype != np.uint8 for value in fields.values()):
        raise EC1Error("field or decoded-token geometry differs")
    errors = materialize_base_error(fields["gt"], fields["base"], output)
    if int(np.count_nonzero(errors)) != BASE_FLIPS:
        raise EC1Error("matched base flip denominator differs")
    row = {
        "schema": "ddm_ec1_custody.v1",
        "axis": AXIS,
        "stage0_result": file_record(STAGE0),
        "gt_field": file_record(GT_FIELD),
        "base_field": file_record(BASE_FIELD),
        "decoded_semantic_tokens": file_record(TOKENS),
        "base_error_field": file_record(output / "inputs/retained/base_error_n600.bool.npy"),
        "base_archive": file_record(BASE_ARCHIVE),
        "runtime_template": file_record(RUNTIME_TEMPLATE),
        "js8_memo_sha256_pin": JS8_MEMO_SHA256,
        "base_flips": BASE_FLIPS,
        "base_d_pose_recalled_not_remeasured": BASE_D_POSE,
        "selection_mode": "full n600; cross-fit by pair parity, never a prefix",
    }
    atomic_json(output / "inputs/CUSTODY.json", row)
    return row


def _neighbors(tokens: np.ndarray) -> tuple[np.ndarray, ...]:
    left = np.empty_like(tokens)
    right = np.empty_like(tokens)
    up = np.empty_like(tokens)
    down = np.empty_like(tokens)
    left[:, :, 0] = tokens[:, :, 0]
    left[:, :, 1:] = tokens[:, :, :-1]
    right[:, :, -1] = tokens[:, :, -1]
    right[:, :, :-1] = tokens[:, :, 1:]
    up[:, 0, :] = tokens[:, 0, :]
    up[:, 1:, :] = tokens[:, :-1, :]
    down[:, -1, :] = tokens[:, -1, :]
    down[:, :-1, :] = tokens[:, 1:, :]
    return left, right, up, down


def context_codes(tokens: np.ndarray, family: str) -> np.ndarray:
    center = np.asarray(tokens, dtype=np.uint16)
    if family == "class_only":
        return center.astype(np.uint16, copy=True)
    neighbors = _neighbors(np.asarray(tokens))
    if family == "undirected":
        mask = np.zeros(center.shape, dtype=np.uint16)
        for neighbor in neighbors:
            edge = neighbor != tokens
            mask |= np.where(edge, np.left_shift(np.uint16(1), neighbor.astype(np.uint16)), 0)
        return center + np.uint16(5) * mask
    if family == "oriented":
        value = center.copy()
        stride = np.uint16(5)
        for neighbor in neighbors:
            value += stride * neighbor.astype(np.uint16)
            stride = np.uint16(int(stride) * 5)
        return value
    raise ValueError(f"unknown family: {family}")


def bucket_count(family: str) -> int:
    return {"class_only": 5, "undirected": 160, "oriented": 3_125}[family]


def _open_retained_field(path: Path, shape: tuple[int, ...], dtype: np.dtype[Any]) -> np.memmap:
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial.npy")
    if partial.exists():
        raise EC1Error(f"uncertified partial retained field requires review: {partial}")
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.lib.format.open_memmap(partial, mode="w+", dtype=dtype, shape=shape)
    value.flush()
    return value


def _finish_retained_field(value: np.memmap, destination: Path) -> None:
    source = Path(value.filename)
    value.flush()
    del value
    os.replace(source, destination)


def materialize_base_error(gt: np.ndarray, base: np.ndarray, output: Path) -> np.ndarray:
    path = output / "inputs/retained/base_error_n600.bool.npy"
    if path.is_file():
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.shape != (N, H, W) or value.dtype != np.bool_:
            raise EC1Error("retained base-error field differs")
        return value
    value = _open_retained_field(path, (N, H, W), np.dtype(np.bool_))
    for start in range(0, N, CHUNK):
        stop = min(N, start + CHUNK)
        value[start:stop] = np.asarray(base[start:stop]) != np.asarray(gt[start:stop])
        value.flush()
    _finish_retained_field(value, path)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def materialize_context(
    tokens: np.ndarray,
    output: Path,
    family: str,
) -> np.ndarray:
    path = output / f"context_race/{family}/retained/context_codes_n600.uint16.npy"
    if path.is_file():
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.shape != (N, H, W) or value.dtype != np.uint16:
            raise EC1Error(f"retained {family} context differs")
        return value
    value = _open_retained_field(path, (N, H, W), np.dtype(np.uint16))
    for start in range(0, N, CHUNK):
        stop = min(N, start + CHUNK)
        value[start:stop] = context_codes(np.asarray(tokens[start:stop]), family)
        value.flush()
    _finish_retained_field(value, path)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _fit_lut(codes: np.ndarray, errors: np.ndarray, parity: int, buckets: int) -> np.ndarray:
    total = np.zeros(buckets, dtype=np.int64)
    positive = np.zeros(buckets, dtype=np.int64)
    for start in range(parity, N, 2 * CHUNK):
        indices = np.arange(start, min(N, start + 2 * CHUNK), 2)
        code = np.asarray(codes[indices]).reshape(-1)
        target = np.asarray(errors[indices]).reshape(-1)
        total += np.bincount(code, minlength=buckets)
        positive += np.bincount(code, weights=target, minlength=buckets).astype(np.int64)
    return ((positive + 0.5) / (total + 1.0)).astype(np.float32)


def _prediction_metrics(prediction: np.ndarray, errors: np.ndarray) -> dict[str, Any]:
    histogram_total = np.zeros(65_536, dtype=np.int64)
    histogram_positive = np.zeros(65_536, dtype=np.int64)
    brier_sum = 0.0
    log_sum = 0.0
    for start in range(0, N, CHUNK):
        stop = min(N, start + CHUNK)
        code = np.asarray(prediction[start:stop]).reshape(-1)
        target = np.asarray(errors[start:stop]).reshape(-1)
        histogram_total += np.bincount(code, minlength=65_536)
        histogram_positive += np.bincount(code, weights=target, minlength=65_536).astype(np.int64)
        probability = np.clip(code.astype(np.float64) / 65_535.0, 1e-9, 1.0 - 1e-9)
        brier_sum += float(np.square(probability - target).sum())
        log_sum -= float((target * np.log(probability) + (1 - target) * np.log1p(-probability)).sum())
    positives = int(histogram_positive.sum())
    negatives = int(histogram_total.sum() - positives)
    cumulative_negative = 0
    auc_numerator = 0.0
    for count, positive in zip(histogram_total, histogram_positive, strict=True):
        negative = int(count - positive)
        auc_numerator += float(positive) * (cumulative_negative + 0.5 * negative)
        cumulative_negative += negative
    remaining = BASE_FLIPS
    selected_positive = 0.0
    for count, positive in zip(histogram_total[::-1], histogram_positive[::-1], strict=True):
        if remaining <= 0:
            break
        take = min(remaining, int(count))
        if count:
            selected_positive += float(positive) * take / int(count)
        remaining -= take
    return {
        "denominator_pixels": PIXELS,
        "positive_errors": positives,
        "prevalence": positives / PIXELS,
        "crossfit_auroc": auc_numerator / (positives * negatives),
        "crossfit_brier": brier_sum / PIXELS,
        "crossfit_log_loss": log_sum / PIXELS,
        "top_k": BASE_FLIPS,
        "expected_errors_in_top_k_with_fractional_ties": selected_positive,
        "top_k_precision": selected_positive / BASE_FLIPS,
        "error_recall_at_top_k": selected_positive / positives,
        "lift_over_prevalence_at_top_k": (selected_positive / BASE_FLIPS) / (positives / PIXELS),
    }


def fit_family(tokens: np.ndarray, errors: np.ndarray, output: Path, family: str) -> dict[str, Any]:
    root = output / "context_race" / family
    result_path = root / "RESULT.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        for record in result["payloads"].values():
            require_file(Path(record["path"]), digest=record["sha256"], size=record["bytes"])
        if "crossfit_lut_raw" not in result["payloads"]:
            lut_path = Path(result["payloads"]["crossfit_lut"]["path"])
            lut = np.load(lut_path, allow_pickle=False)
            raw_path = root / "retained/crossfit_lut.float32.raw"
            raw = np.ascontiguousarray(lut).tobytes(order="C")
            if raw_path.is_file():
                if raw_path.read_bytes() != raw:
                    raise EC1Error(f"{family} retained LUT raw repair differs")
            else:
                atomic_bytes(raw_path, raw)
            coded_path = Path(result["payloads"]["crossfit_lut_coded"]["path"])
            if brotli.decompress(coded_path.read_bytes()) != raw:
                raise EC1Error(f"{family} recovered LUT raw bytes differ from retained Brotli payload")
            result["payloads"]["crossfit_lut_raw"] = file_record(raw_path)
            repair = {
                "schema": "ddm_ec1_retention_repair.v1",
                "family": family,
                "disposition": "REPAIRED",
                "reason": (
                    "The first analysis retained the NPY and compressed LUT but omitted the exact raw bytes "
                    "fed to Brotli. They were deterministically recovered from the retained float32 NPY; no scorer reran."
                ),
                "source": file_record(lut_path),
                "recovered_raw": file_record(raw_path),
            }
            atomic_json(root / "RETENTION_REPAIR.json", repair)
            result["retention_repair"] = file_record(root / "RETENTION_REPAIR.json")
        return result
    codes = materialize_context(tokens, output, family)
    buckets = bucket_count(family)
    even_lut = _fit_lut(codes, errors, 0, buckets)
    odd_lut = _fit_lut(codes, errors, 1, buckets)
    lut = np.stack((even_lut, odd_lut))
    lut_path = root / "retained/crossfit_lut.float32.npy"
    atomic_npy(lut_path, lut)
    lut_raw = lut.tobytes(order="C")
    raw_path = root / "retained/crossfit_lut.float32.raw"
    atomic_bytes(raw_path, lut_raw)
    lut_coded = brotli.compress(lut_raw, quality=11)
    coded_path = root / "retained/crossfit_lut.float32.br"
    repeat_path = root / "retained/crossfit_lut.repeat.float32.br"
    atomic_bytes(coded_path, lut_coded)
    atomic_bytes(repeat_path, brotli.compress(lut.tobytes(order="C"), quality=11))
    if coded_path.read_bytes() != repeat_path.read_bytes() or brotli.decompress(lut_coded) != lut_raw:
        raise EC1Error(f"{family} LUT coder failed deterministic parse-back")
    prediction_path = root / "retained/crossfit_error_probability_n600.uint16.npy"
    prediction = _open_retained_field(prediction_path, (N, H, W), np.dtype(np.uint16))
    for start in range(0, N, CHUNK):
        stop = min(N, start + CHUNK)
        code = np.asarray(codes[start:stop])
        for offset, pair in enumerate(range(start, stop)):
            trained_on = 1 - (pair % 2)
            prediction[offset + start] = np.rint(lut[trained_on][code[offset]] * 65_535.0).astype(np.uint16)
        prediction.flush()
    _finish_retained_field(prediction, prediction_path)
    retained_prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    metrics = _prediction_metrics(retained_prediction, errors)
    payloads = {
        "context_codes": file_record(root / "retained/context_codes_n600.uint16.npy"),
        "crossfit_lut": file_record(lut_path),
        "crossfit_lut_raw": file_record(raw_path),
        "crossfit_lut_coded": file_record(coded_path),
        "crossfit_lut_repeat": file_record(repeat_path),
        "crossfit_prediction": file_record(prediction_path),
    }
    row = {
        "schema": "ddm_ec1_context_family_result.v1",
        "family": family,
        "axis": AXIS,
        "fit_policy": "two-fold cross-fit by pair parity; each pixel predicted only by the opposite parity fit",
        "receiver_computable": True,
        "explicit_mask_or_sidecar": False,
        "buckets": buckets,
        "metrics": metrics,
        "payloads": payloads,
        "score_claim": False,
    }
    atomic_json(result_path, row)
    return row


def build_model(torch: Any, family: str, *, identity: bool) -> Any:
    torch.manual_seed(SEED + runtime.FAMILIES.index(family))
    model = runtime.LatentEdgeConditioner(HIDDEN, MAX_DELTA, family)
    torch.nn.init.kaiming_uniform_(model.context.weight, a=math.sqrt(5))
    torch.nn.init.zeros_(model.context.bias)
    torch.nn.init.dirac_(model.depthwise.weight)
    torch.nn.init.zeros_(model.depthwise.bias)
    if identity:
        torch.nn.init.zeros_(model.head.weight)
        torch.nn.init.zeros_(model.head.bias)
    else:
        torch.nn.init.kaiming_uniform_(model.head.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(model.head.bias)
    return model.eval()


def serialize_module(model: Any, output: Path, label: str) -> dict[str, Any]:
    metadata = []
    chunks = []
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().numpy()
        if name.endswith("weight"):
            scale = max(float(np.max(np.abs(array))) / 127.0, 1e-8)
            stored = np.clip(np.rint(array / scale), -127, 127).astype(np.int8)
            dtype = "int8"
        else:
            scale = None
            stored = np.asarray(array, dtype="<f2")
            dtype = "float16"
        metadata.append(
            {
                "name": name,
                "shape": list(array.shape),
                "dtype": dtype,
                "scale": scale,
                "bytes": stored.nbytes,
            }
        )
        chunks.append(stored.tobytes(order="C"))
    header = json.dumps(
        {
            "schema": runtime.SCHEMA,
            "family": model.family,
            "hidden": model.hidden,
            "max_delta": model.max_delta,
            "tensors": metadata,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    raw = runtime.MAGIC + len(header).to_bytes(4, "little") + header + b"".join(chunks)
    coded = brotli.compress(raw, quality=11)
    root = output / "design/retained" / model.family
    raw_path = root / f"{label}.raw"
    coded_path = root / f"{label}.br"
    repeat_path = root / f"{label}.repeat.br"
    retain_exact(raw_path, raw)
    retain_exact(coded_path, coded)
    retain_exact(repeat_path, brotli.compress(raw, quality=11))
    parsed_header, parsed = runtime.parse_module(coded)
    if coded_path.read_bytes() != repeat_path.read_bytes() or set(parsed) != set(model.state_dict()):
        raise EC1Error("latent adapter deterministic parse-back failed")
    decoded_records = {}
    for name, value in sorted(parsed.items()):
        path = root / f"{label}.decoded_state" / f"{name}.float32.npy"
        if path.is_file():
            retained = np.load(path, allow_pickle=False)
            if not np.array_equal(retained, value):
                raise EC1Error(f"retained decoded adapter tensor differs: {path}")
        else:
            atomic_npy(path, value)
        decoded_records[name] = file_record(path)
    return {
        "label": label,
        "family": model.family,
        "parameter_count": sum(value.numel() for value in model.state_dict().values()),
        "header": parsed_header,
        "raw": file_record(raw_path),
        "coded": file_record(coded_path),
        "repeat": file_record(repeat_path),
        "decoded_state": decoded_records,
        "parseback_exact_quantized_state": True,
    }


def price_design_archive(export: dict[str, Any], output: Path, label: str) -> dict[str, Any]:
    module = Path(export["coded"]["path"]).read_bytes()
    archive = deterministic_archive(BASE_ARCHIVE, module)
    repeat = deterministic_archive(BASE_ARCHIVE, module)
    root = output / "design/retained" / str(export["family"])
    archive_path = root / f"{label}.archive.zip"
    repeat_path = root / f"{label}.archive.repeat.zip"
    retain_exact(archive_path, archive)
    retain_exact(repeat_path, repeat)
    if archive != repeat:
        raise EC1Error("design-price archive is not deterministic")
    return {
        "classification": "DESIGN_PRICE_CONTROL",
        "is_candidate": False,
        "archive": file_record(archive_path),
        "archive_repeat": file_record(repeat_path),
        "archive_delta_bytes_vs_cp135": len(archive) - BASE_BYTES,
        "module_member_bytes": len(module),
    }


def design_ladder(output: Path) -> list[dict[str, Any]]:
    torch = __import__("torch")
    rows = []
    for family in runtime.FAMILIES:
        identity = serialize_module(build_model(torch, family, identity=True), output, "identity")
        identity["archive_price_control"] = price_design_archive(identity, output, "identity")
        nonzero = serialize_module(
            build_model(torch, family, identity=False), output, "seeded_nonzero_capacity_reference"
        )
        nonzero["archive_price_control"] = price_design_archive(
            nonzero, output, "seeded_nonzero_capacity_reference"
        )
        rows.append(
            {
                "family": family,
                "identity": identity,
                "nonzero_capacity_price_reference": nonzero,
                "capacity_reference_is_candidate": False,
            }
        )
    atomic_json(output / "design/CAPACITY_LADDER.json", {"schema": "ddm_ec1_capacity_ladder.v1", "rows": rows})
    return rows


def actual_receiver_probe(output: Path, ladder: list[dict[str, Any]], family: str) -> dict[str, Any]:
    """Prove identity/non-identity inside the exact decoded CP135 renderer."""
    result_path = output / "receiver_probe/RESULT.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        if result.get("family") != family:
            raise EC1Error("retained receiver probe family differs")
        for record in result["payloads"].values():
            require_file(Path(record["path"]), digest=record["sha256"], size=record["bytes"])
        return result
    from experiments import ddm_js2b_edge_conditioning_relative_gauge as js2b

    torch = __import__("torch")
    modules = js2b.load_modules()
    parts = modules.residual.read_residual_archive(BASE_ARCHIVE)
    records = modules.renderer_codec.decode_wans1(parts.semantic_blob)
    semantic = modules.renderer_runtime.SemanticTokenRenderer(96).eval()
    semantic.load_state_dict(
        {
            record.schema.name: torch.from_numpy(np.ascontiguousarray(record.values, dtype=np.float32))
            for record in records
        },
        strict=True,
    )
    pair = int(np.random.default_rng(SEED).integers(0, N))
    token_field = np.load(TOKENS, mmap_mode="r", allow_pickle=False)
    tokens = torch.from_numpy(np.asarray(token_field[pair : pair + 1]).astype(np.int64, copy=True))
    indices = torch.tensor([pair], dtype=torch.long)
    design = next(row for row in ladder if row["family"] == family)
    identity_blob = Path(design["identity"]["coded"]["path"]).read_bytes()
    nonzero_blob = Path(design["nonzero_capacity_price_reference"]["coded"]["path"]).read_bytes()
    with torch.inference_mode():
        base = semantic(tokens, indices).cpu().numpy().astype(np.float32)
        identity = runtime.conditioned_semantic_forward(semantic, tokens, indices, identity_blob).cpu().numpy().astype(np.float32)
        nonzero = runtime.conditioned_semantic_forward(semantic, tokens, indices, nonzero_blob).cpu().numpy().astype(np.float32)
    root = output / "receiver_probe/retained"
    atomic_npy(root / "base_pre_r.float32.npy", base)
    atomic_npy(root / "identity_pre_r.float32.npy", identity)
    atomic_npy(root / "seeded_nonzero_pre_r.float32.npy", nonzero)
    delta = nonzero - base
    atomic_npy(root / "seeded_nonzero_delta.float32.npy", delta)
    if not np.array_equal(base, identity):
        raise EC1Error("zero EC1 adapter is not identity on the exact CP135 renderer")
    changed = int(np.count_nonzero(delta))
    if changed == 0:
        raise EC1Error("nonzero EC1 adapter is inert on the exact CP135 renderer")
    payloads = {
        "base_pre_r": file_record(root / "base_pre_r.float32.npy"),
        "identity_pre_r": file_record(root / "identity_pre_r.float32.npy"),
        "seeded_nonzero_pre_r": file_record(root / "seeded_nonzero_pre_r.float32.npy"),
        "seeded_nonzero_delta": file_record(root / "seeded_nonzero_delta.float32.npy"),
    }
    result = {
        "schema": "ddm_ec1_actual_receiver_probe.v1",
        "axis": "[macOS-CPU exact CP135 renderer mechanism surface, no scorer]",
        "pair": pair,
        "family": family,
        "base_vs_identity_bit_exact": True,
        "base_vs_seeded_nonzero_changed_values": changed,
        "base_vs_seeded_nonzero_max_abs_pre_r_delta": float(np.max(np.abs(delta))),
        "seeded_nonzero_is_candidate": False,
        "payloads": payloads,
        "score_claim": False,
    }
    atomic_json(result_path, result)
    return result


def write_fire_order(
    output: Path,
    selected: str,
    module_bytes: int,
    archive_delta_bytes: int,
) -> dict[str, Any]:
    trained_module = output / "main_cuda/stages/selected/retained/ec1_latent.int8.br"
    package_root = output / "main_cuda/packaged_candidate"
    row = {
        "schema": "ddm_ec1_main_cuda_fire_order.v1",
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN true-CUDA training and exact-row owner",
        "consumer_store": str((output / "main_cuda").resolve()),
        "fire_trigger": (
            "MAIN owns the sole full-n600 scorer lane; no other full-n600 scorer job is active; "
            "the trainer injects the adapter before CP135 TokenBlocks, uses the pinned JS1C GT/base fields, "
            "saves distinct live+EMA stage checkpoints, and retains every field/model/archive"
        ),
        "selected_context_family": selected,
        "capacity_reference_module_bytes": module_bytes,
        "capacity_reference_exact_archive_delta_bytes": archive_delta_bytes,
        "break_even_flips_per_byte": BREAK_EVEN_FLIPS_PER_BYTE,
        "minimum_realized_reduction_for_capacity_reference": math.ceil(
            BREAK_EVEN_FLIPS_PER_BYTE * archive_delta_bytes
        ),
        "training_requirements": {
            "axis": "contest-CUDA T4 true scorer-in-loop; local CPU is advisory only",
            "base_flips": BASE_FLIPS,
            "base_archive_bytes": BASE_BYTES,
            "base_d_pose": BASE_D_POSE,
            "same_receiver_object": True,
            "equal_parameter_controls": list(runtime.FAMILIES),
            "resume_from_disk": True,
            "stage_checkpoints": "distinct live and EMA files at every stage plus periodic intra-stage saves",
            "payload_retention": "all model, field, camera, scorer-input, archive, and repeat-archive payloads",
        },
        "true_cuda_trainer_implemented_by_producer": False,
        "true_cuda_trainer_blocker": (
            "this arm did not own CUDA or a scorer slot; MAIN must land the resumable scorer-in-loop trainer "
            "before the package command becomes fireable"
        ),
        "package_command_after_training": [
            ".venv/bin/python",
            "experiments/ddm_ec1_implicit_edge_conditioning.py",
            "package",
            "--module",
            str(trained_module.resolve()),
            "--output",
            str(package_root.resolve()),
            "--classification",
            "candidate_proposal",
        ],
        "exact_measurement_after_package": (
            "MAIN adapts the proven re1t/js1b one-archive candidate-only worker to the packaged archive/runtime "
            "SHAs, then measures the retained full-n600 T4 argmax field against the same 34,970-flip base"
        ),
        "producer_dispatched_modal": False,
        "t4_measurement_sealed_now": False,
        "t4_measurement_blocker": "no trained module or candidate archive exists yet; sealing archive SHAs now would be fake",
    }
    atomic_json(output / "MAIN_CUDA_FIRE_ORDER.json", row)
    return row


def analyze(output: Path) -> dict[str, Any]:
    output = output.resolve()
    storage = storage_preflight(output)
    custody = custody_preflight(output)
    gt = np.load(GT_FIELD, mmap_mode="r", allow_pickle=False)
    base = np.load(BASE_FIELD, mmap_mode="r", allow_pickle=False)
    tokens = np.load(TOKENS, mmap_mode="r", allow_pickle=False)
    errors = materialize_base_error(gt, base, output)
    rows = [fit_family(tokens, errors, output, family) for family in runtime.FAMILIES]
    ladder = design_ladder(output)
    selected_row = max(
        rows,
        key=lambda row: (
            float(row["metrics"]["crossfit_auroc"]),
            float(row["metrics"]["top_k_precision"]),
            -int(row["buckets"]),
        ),
    )
    selected = str(selected_row["family"])
    selected_design = next(row for row in ladder if row["family"] == selected)
    module_bytes = int(selected_design["nonzero_capacity_price_reference"]["coded"]["bytes"])
    archive_delta_bytes = int(
        selected_design["nonzero_capacity_price_reference"]["archive_price_control"][
            "archive_delta_bytes_vs_cp135"
        ]
    )
    receiver_probe = actual_receiver_probe(output, ladder, selected)
    fire_order = write_fire_order(output, selected, module_bytes, archive_delta_bytes)
    result = {
        "schema": "ddm_ec1_analysis_result.v1",
        "status": "DESIGN_COMPLETE_T4_TRAINING_QUEUED",
        "axis": AXIS,
        "score_claim": False,
        "pointer_moved": False,
        "selection_mode": "full n600 retained matched fields; two-fold pair-parity cross-fit",
        "base": {"flips": BASE_FLIPS, "archive_bytes": BASE_BYTES, "d_pose": BASE_D_POSE},
        "context_race": rows,
        "selected_family_for_first_true_cuda_training": selected,
        "selection_reason": "best cross-fit error-ranking AUROC; not a realized-flip or score claim",
        "capacity_ladder": ladder,
        "actual_receiver_mechanism_probe": receiver_probe,
        "capacity_reference": {
            "module_bytes": module_bytes,
            "exact_archive_delta_bytes": archive_delta_bytes,
            "break_even_flips_per_byte": BREAK_EVEN_FLIPS_PER_BYTE,
            "minimum_realized_reduction": math.ceil(
                BREAK_EVEN_FLIPS_PER_BYTE * archive_delta_bytes
            ),
            "reference_is_not_trained_or_a_candidate": True,
        },
        "measured": (
            "receiver-computable context selectivity against all 117,964,800 matched T4 GT/base pixels; "
            "exact retained payload sizes and deterministic parse-back for three equal-width adapter grammars"
        ),
        "not_measured": (
            "no conditioned render, realized candidate flips, PoseNet delta, candidate archive, contest score, "
            "contest-CPU row, or exact contest-CUDA candidate row"
        ),
        "boundaries": {
            "context_selectivity_is_admission_authority": False,
            "local_scorer_run": False,
            "full_n600_scorer_slot_used": False,
            "modal_or_metal_used": False,
            "explicit_edge_mask_or_sidecar": False,
            "verdict_scope": "DESIGN/INSTANCE screen on pinned CP135 tokens and JS1C matched fields; no family verdict",
        },
        "storage_preflight": storage,
        "custody": custody,
        "follow_on": fire_order,
    }
    atomic_json(output / "FINAL_RESULT.json", result)
    return result


def deterministic_archive(base_archive: Path, module: bytes) -> bytes:
    with zipfile.ZipFile(base_archive) as archive:
        if archive.namelist() != ["p"]:
            raise EC1Error("CP135 base archive grammar differs")
        payload = archive.read("p")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, value in (("p", payload), ("ec1_latent.br", module)):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, value)
    return stream.getvalue()


def replace_once(path: Path, old: str, new: str) -> None:
    value = path.read_text()
    if value.count(old) != 1:
        raise EC1Error(f"runtime adapter expected one source match in {path}: {old[:80]!r}")
    atomic_bytes(path, value.replace(old, new).encode())


def adapt_runtime(archive_record: dict[str, Any], output: Path) -> Path:
    destination = output / "adapted_runtime"
    if destination.exists():
        raise EC1Error("adapted runtime destination exists; preserve it and choose a fresh output")
    staging = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    shutil.copytree(
        BASE_RUNTIME,
        staging,
        ignore=shutil.ignore_patterns("archive.zip", "__pycache__", "*.pyc", "._*", ".DS_Store"),
    )
    shutil.copy2(RUNTIME_TEMPLATE, staging / "runtime/ec1_latent_conditioner.py")
    atomic_bytes(staging / "archive.zip", Path(archive_record["path"]).read_bytes())
    residual = staging / "runtime/residual_archive.py"
    replace_once(
        residual,
        'if archive.namelist() != ["p"]:\n            raise ResidualArchiveError("archive must contain exactly member p")',
        'if archive.namelist() != ["p", "ec1_latent.br"]:\n            raise ResidualArchiveError("EC1 archive must contain p plus counted latent adapter")',
    )
    f26 = staging / "runtime/f26_inflate.py"
    replace_once(f26, "import time\n", "import time\nimport zipfile\n")
    replace_once(
        f26,
        "    parts = read_residual_archive(archive_path)\n",
        '    with zipfile.ZipFile(archive_path) as archive:\n'
        '        if archive.namelist() != ["p", "ec1_latent.br"]:\n'
        '            raise InflationError("EC1 archive members differ")\n'
        '        ec1_blob = archive.read("ec1_latent.br")\n'
        "    parts = read_residual_archive(archive_path)\n",
    )
    replace_once(
        f26,
        "    renderer.render_video(semantic, basis, coefficients, tokens, destination, device)\n",
        "    renderer.render_video(semantic, basis, coefficients, tokens, destination, device, ec1_blob=ec1_blob)\n",
    )
    renderer = staging / "cpr1/inflate.py"
    replace_once(
        renderer,
        "from torch.nn import functional\n",
        "from torch.nn import functional\nfrom runtime.ec1_latent_conditioner import conditioned_semantic_forward\n",
    )
    replace_once(
        renderer,
        "def render_video(semantic, basis, coefficients, tokens, destination: Path, device):\n",
        "def render_video(semantic, basis, coefficients, tokens, destination: Path, device, *, ec1_blob):\n",
    )
    replace_once(
        renderer,
        "                semantic(tokens[start:end].long().to(device), indices),\n",
        "                conditioned_semantic_forward(semantic, tokens[start:end].long().to(device), indices, ec1_blob),\n",
    )
    outer = staging / "inflate.py"
    replace_once(
        outer,
        f'ARCHIVE_SHA256 = "{BASE_ARCHIVE_SHA256}"',
        f'ARCHIVE_SHA256 = "{archive_record["sha256"]}"',
    )
    replace_once(outer, f"ARCHIVE_BYTES = {BASE_BYTES:_}", f"ARCHIVE_BYTES = {archive_record['bytes']:_}")
    replace_once(
        outer,
        '        if archive.namelist() != ["p"]:\n            raise ValueError("archive.zip must contain exactly the payload file p")',
        '        if archive.namelist() != ["p", "ec1_latent.br"]:\n            raise ValueError("archive.zip must contain p plus the counted EC1 latent adapter")',
    )
    os.replace(staging, destination)
    return destination


def package(module_path: Path, output: Path, classification: str) -> dict[str, Any]:
    if classification not in {"candidate_proposal", "identity_control"}:
        raise ValueError("package classification differs")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    require_file(module_path)
    module = module_path.read_bytes()
    header, decoded = runtime.parse_module(module)
    head_is_zero = not np.any(decoded["head.weight"]) and not np.any(decoded["head.bias"])
    if classification == "identity_control" and not head_is_zero:
        raise EC1Error("identity-control classification requires a zero adapter head")
    if classification == "candidate_proposal" and head_is_zero:
        raise EC1Error("candidate-proposal classification refuses an identity adapter")
    archive = deterministic_archive(BASE_ARCHIVE, module)
    repeat = deterministic_archive(BASE_ARCHIVE, module)
    archive_path = output / "retained/archive.zip"
    repeat_path = output / "retained/archive.repeat.zip"
    module_copy = output / "retained/ec1_latent.br"
    module_raw = output / "retained/ec1_latent.raw"
    atomic_bytes(module_copy, module)
    atomic_bytes(module_raw, brotli.decompress(module))
    decoded_records = {}
    for name, value in sorted(decoded.items()):
        path = output / "retained/decoded_state" / f"{name}.float32.npy"
        atomic_npy(path, value)
        decoded_records[name] = file_record(path)
    atomic_bytes(archive_path, archive)
    atomic_bytes(repeat_path, repeat)
    if archive != repeat:
        raise EC1Error("candidate archive is not deterministic")
    archive_record = file_record(archive_path)
    adapted = adapt_runtime(archive_record, output)
    row = {
        "schema": "ddm_ec1_packaged_module.v1",
        "status": "PACKAGED_NOT_MEASURED" if classification == "candidate_proposal" else "IDENTITY_CONTROL",
        "classification": classification,
        "is_candidate": classification == "candidate_proposal",
        "module": file_record(module_copy),
        "module_raw": file_record(module_raw),
        "decoded_module_state": decoded_records,
        "module_header": header,
        "archive": archive_record,
        "archive_repeat": file_record(repeat_path),
        "archive_delta_bytes": int(archive_record["bytes"]) - BASE_BYTES,
        "adapted_runtime": str(adapted.resolve()),
        "receiver_consumes_counted_module": True,
        "receiver_features_are_decoded_token_only": True,
        "exact_row_measured": False,
        "score_claim": False,
    }
    atomic_json(output / "PACKAGE_RESULT.json", row)
    return row


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--output", type=Path, default=OUTPUT)
    package_parser = sub.add_parser("package")
    package_parser.add_argument("--module", type=Path, required=True)
    package_parser.add_argument("--output", type=Path, required=True)
    package_parser.add_argument(
        "--classification", choices=("candidate_proposal", "identity_control"), required=True
    )
    return value


def main() -> None:
    args = parser().parse_args()
    result = (
        analyze(args.output)
        if args.command == "analyze"
        else package(args.module, args.output, args.classification)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
