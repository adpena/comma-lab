#!/usr/bin/env python3
"""Select safe renderer payload bytes for official component-response plans.

The companion perturbation-plan builder can mutate any archive byte. This
script narrows that freedom for ASYM renderers: it parses the renderer binary
container, selects deterministic bytes inside quantized weight payloads, and
optionally verifies every epsilon variant still decodes with the canonical
renderer loader.

Output is a ``perturbation_basis_v1``-compatible JSON file whose ``atoms`` can
be passed to ``experiments/build_component_response_perturbation_plan.py``.
No scorer is run and no score claim is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


FORMAT = "renderer_blob_perturbation_basis_v1"
PRODUCER = "experiments/select_renderer_blob_perturbation_basis.py"
PLAN_BASIS_FORMAT = "perturbation_basis_v1"
DEFAULT_EPSILONS = (-1.0, 0.0, 1.0)
SCORE_EPS = 1e-12


class RendererBasisSelectionError(ValueError):
    """Raised when a safe perturbation basis cannot be selected."""


@dataclass(frozen=True)
class BlobRegion:
    layer_index: int
    layer_name: str
    blob_kind: str
    bits: int
    shape: tuple[int, ...]
    blob_start: int
    blob_len: int
    payload_bytes: int
    output_channels: int
    fan_in: int
    per_channel_payload_bytes: int
    transposed: bool
    is_linear: bool
    is_embedding: bool


@dataclass(frozen=True)
class SelectedAtom:
    atom_id: str
    member: str
    offset: int
    delta_per_epsilon: int
    layer_index: int
    layer_name: str
    blob_kind: str
    bits: int
    shape: tuple[int, ...]
    channel_index: int
    byte_index_within_channel_payload: int
    original_byte: int
    selection_rank: int
    payload_bytes: int
    margin_to_byte_range: int
    sensitivity_score: float | None = None
    selection_source: str = "payload_size"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _safe_member_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise RendererBasisSelectionError("archive member name must be non-empty")
    if "\\" in name or "\x00" in name:
        raise RendererBasisSelectionError(f"unsafe archive member path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RendererBasisSelectionError(f"unsafe archive member path: {name!r}")
    return path.as_posix()


def _read_renderer_from_archive(archive: Path) -> tuple[bytes, dict[str, Any]]:
    if not archive.is_file():
        raise RendererBasisSelectionError(f"archive not found: {archive}")
    renderer: bytes | None = None
    manifest: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(archive, "r") as zf:
            seen: set[str] = set()
            for info in zf.infolist():
                if info.is_dir():
                    raise RendererBasisSelectionError(
                        f"directory entries are not allowed: {info.filename!r}"
                    )
                name = _safe_member_name(info.filename)
                if name in seen:
                    raise RendererBasisSelectionError(
                        f"duplicate archive member rejected: {name!r}"
                    )
                seen.add(name)
                data = zf.read(info)
                manifest.append(
                    {
                        "name": name,
                        "raw_bytes": int(info.file_size),
                        "compressed_bytes": int(info.compress_size),
                        "crc32": f"{info.CRC:08x}",
                        "sha256": _sha256_bytes(data),
                    }
                )
                if name == "renderer.bin":
                    renderer = data
    except zipfile.BadZipFile as exc:
        raise RendererBasisSelectionError(f"not a valid zip archive: {archive}") from exc
    if renderer is None:
        raise RendererBasisSelectionError("archive missing renderer.bin")
    if renderer[:4] != b"ASYM":
        raise RendererBasisSelectionError(
            f"renderer.bin magic {renderer[:4]!r}; only ASYM is supported"
        )
    return renderer, {"members": manifest}


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RendererBasisSelectionError(f"{field} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise RendererBasisSelectionError(f"{field} must be finite")
    return out


def _require_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RendererBasisSelectionError(f"{field} must be a positive integer")
    return int(value)


def _read_u32(data: bytes, offset: int, *, field: str) -> int:
    if offset + 4 > len(data):
        raise RendererBasisSelectionError(f"{field}: truncated uint32 at offset {offset}")
    return int.from_bytes(data[offset : offset + 4], byteorder="little")


def _parse_shape(raw: Any, *, field: str) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        raise RendererBasisSelectionError(f"{field} must be a non-empty shape list")
    shape: list[int] = []
    for index, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RendererBasisSelectionError(f"{field}[{index}] must be a positive integer")
        shape.append(int(value))
    return tuple(shape)


def _product(values: tuple[int, ...]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def parse_asym_payload_regions(renderer: bytes) -> tuple[dict[str, Any], list[BlobRegion]]:
    """Return ASYM header and non-bias weight-payload regions."""

    if renderer[:4] != b"ASYM":
        raise RendererBasisSelectionError(
            f"renderer.bin magic {renderer[:4]!r}; expected b'ASYM'"
        )
    header_len = _read_u32(renderer, 4, field="header_len")
    header_start = 8
    header_end = header_start + header_len
    if header_len <= 0 or header_end > len(renderer):
        raise RendererBasisSelectionError(
            f"invalid ASYM header length {header_len} for {len(renderer)} byte renderer"
        )
    try:
        header = json.loads(renderer[header_start:header_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RendererBasisSelectionError("invalid ASYM JSON header") from exc
    if not isinstance(header, dict):
        raise RendererBasisSelectionError("ASYM header must be a JSON object")
    if header.get("version") != 2:
        raise RendererBasisSelectionError(
            f"unsupported ASYM version {header.get('version')!r}; expected 2"
        )
    layers = header.get("layers")
    if not isinstance(layers, list) or not layers:
        raise RendererBasisSelectionError("ASYM header.layers must be a non-empty list")

    offset = header_end
    regions: list[BlobRegion] = []
    for layer_index, raw_layer in enumerate(layers):
        if not isinstance(raw_layer, Mapping):
            raise RendererBasisSelectionError(f"layers[{layer_index}] must be an object")
        layer_name = raw_layer.get("name")
        if not isinstance(layer_name, str) or not layer_name:
            raise RendererBasisSelectionError(f"layers[{layer_index}].name must be non-empty")
        bits = _require_positive_int(raw_layer.get("bits"), field=f"layers[{layer_index}].bits")
        bits = max(bits, 2)
        shape = _parse_shape(raw_layer.get("shape"), field=f"layers[{layer_index}].shape")
        is_embedding = bool(raw_layer.get("is_embedding", False))
        transposed = bool(raw_layer.get("transposed", False))
        is_linear = bool(raw_layer.get("is_linear", False))

        blob_len = _read_u32(renderer, offset, field=f"{layer_name}.weight_blob_len")
        offset += 4
        blob_start = offset
        blob_end = blob_start + blob_len
        if blob_end > len(renderer):
            raise RendererBasisSelectionError(
                f"{layer_name}: weight blob overruns renderer ({blob_end}>{len(renderer)})"
            )
        offset = blob_end

        if is_embedding:
            numel = _product(shape)
            per_channel_payload_bytes = (numel * bits + 7) // 8
            payload_bytes = max(0, blob_len - 2)
            if payload_bytes > 0:
                regions.append(
                    BlobRegion(
                        layer_index=layer_index,
                        layer_name=layer_name,
                        blob_kind="embedding_weight",
                        bits=bits,
                        shape=shape,
                        blob_start=blob_start,
                        blob_len=blob_len,
                        payload_bytes=payload_bytes,
                        output_channels=1,
                        fan_in=numel,
                        per_channel_payload_bytes=per_channel_payload_bytes,
                        transposed=False,
                        is_linear=False,
                        is_embedding=True,
                    )
                )
            continue

        if len(shape) < 2:
            raise RendererBasisSelectionError(
                f"{layer_name}: non-embedding weight shape must have rank >= 2"
            )
        output_channels = shape[1] if transposed else shape[0]
        fan_shape = (shape[0], *shape[2:]) if transposed else tuple(shape[1:])
        fan_in = _product(fan_shape)
        per_channel_payload_bytes = (fan_in * bits + 7) // 8
        expected_min = output_channels * (2 + per_channel_payload_bytes)
        if blob_len < expected_min:
            raise RendererBasisSelectionError(
                f"{layer_name}: weight blob {blob_len} bytes shorter than expected "
                f"{expected_min} bytes"
            )
        payload_bytes = output_channels * per_channel_payload_bytes
        if payload_bytes > 0:
            regions.append(
                BlobRegion(
                    layer_index=layer_index,
                    layer_name=layer_name,
                    blob_kind="weight",
                    bits=bits,
                    shape=shape,
                    blob_start=blob_start,
                    blob_len=blob_len,
                    payload_bytes=payload_bytes,
                    output_channels=output_channels,
                    fan_in=fan_in,
                    per_channel_payload_bytes=per_channel_payload_bytes,
                    transposed=transposed,
                    is_linear=is_linear,
                    is_embedding=False,
                )
            )

        bias_len = _read_u32(renderer, offset, field=f"{layer_name}.bias_blob_len")
        offset += 4
        bias_end = offset + bias_len
        if bias_end > len(renderer):
            raise RendererBasisSelectionError(
                f"{layer_name}: bias blob overruns renderer ({bias_end}>{len(renderer)})"
            )
        offset = bias_end

    if offset != len(renderer):
        raise RendererBasisSelectionError(
            f"trailing ASYM payload bytes after layer parse: {len(renderer) - offset}"
        )
    return header, regions


def _sanitize_atom_token(value: str) -> str:
    out = []
    for char in value:
        out.append(char if char.isalnum() else "_")
    token = "".join(out).strip("_")
    return token[:80] or "layer"


def _nearest_payload_index(preferred: int, payload_bytes: int) -> list[int]:
    order = [preferred]
    for distance in range(1, max(preferred + 1, payload_bytes - preferred)):
        left = preferred - distance
        right = preferred + distance
        if left >= 0:
            order.append(left)
        if right < payload_bytes:
            order.append(right)
    return order


def _select_offset_for_region(
    renderer: bytes,
    region: BlobRegion,
    *,
    max_abs_byte_delta: int,
    channel_index: int | None = None,
) -> tuple[int, int, int, int] | None:
    if region.output_channels <= 0 or region.per_channel_payload_bytes <= 0:
        return None
    if channel_index is None:
        preferred_channel = region.output_channels // 2
        channel_order = _nearest_payload_index(preferred_channel, region.output_channels)
    else:
        if channel_index < 0 or channel_index >= region.output_channels:
            return None
        channel_order = [int(channel_index)]
    preferred_byte = region.per_channel_payload_bytes // 2
    byte_order = _nearest_payload_index(preferred_byte, region.per_channel_payload_bytes)
    stride = 2 + region.per_channel_payload_bytes
    for channel_index in channel_order:
        channel_payload_start = region.blob_start + channel_index * stride + 2
        for byte_index in byte_order:
            offset = channel_payload_start + byte_index
            if offset >= region.blob_start + region.blob_len:
                continue
            original = renderer[offset]
            margin = min(original, 255 - original)
            if margin >= max_abs_byte_delta:
                return offset, channel_index, byte_index, margin
    return None


def _region_sensitivity_scores(
    sensitivity_scores: Mapping[str, list[float]] | None,
    region: BlobRegion,
) -> list[float] | None:
    if sensitivity_scores is None:
        return None
    for key in (f"{region.layer_name}.weight", region.layer_name):
        raw = sensitivity_scores.get(key)
        if raw is not None:
            values = [float(value) for value in raw]
            break
    else:
        return None
    if len(values) != region.output_channels:
        raise RendererBasisSelectionError(
            f"sensitivity map for {region.layer_name}.weight has {len(values)} "
            f"channel(s), expected {region.output_channels}"
        )
    for index, value in enumerate(values):
        if not math.isfinite(value) or value < 0:
            raise RendererBasisSelectionError(
                f"sensitivity map for {region.layer_name}.weight channel {index} "
                "must be finite and non-negative"
            )
    return values


def _load_sensitivity_scores(path: Path) -> tuple[dict[str, list[float]], dict[str, Any]]:
    repo_src = Path(__file__).resolve().parents[1] / "src"
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))
    from tac.sensitivity_map import load_sensitivity_map

    sensitivities, metadata = load_sensitivity_map(path)
    scores = {
        key: [float(v) for v in value.detach().cpu().float().reshape(-1).tolist()]
        for key, value in sensitivities.items()
    }
    return scores, dict(metadata)


def select_basis_atoms(
    renderer: bytes,
    *,
    max_atoms: int,
    epsilons: list[float],
    delta_per_epsilon: int = 1,
    include_embeddings: bool = False,
    include_transposed: bool = False,
    sensitivity_scores: Mapping[str, list[float]] | None = None,
    selection_mode: str = "payload-desc",
) -> tuple[dict[str, Any], list[SelectedAtom]]:
    if max_atoms <= 0:
        raise RendererBasisSelectionError("max_atoms must be positive")
    if delta_per_epsilon == 0:
        raise RendererBasisSelectionError("delta_per_epsilon must be nonzero")
    if not epsilons or not any(abs(eps) > SCORE_EPS for eps in epsilons):
        raise RendererBasisSelectionError("epsilon ladder needs at least one nonzero value")
    max_abs_epsilon = max(abs(float(eps)) for eps in epsilons)
    max_abs_byte_delta = int(math.ceil(max_abs_epsilon * abs(delta_per_epsilon)))
    if max_abs_byte_delta <= 0:
        raise RendererBasisSelectionError("epsilon ladder produces no integer byte movement")
    header, regions = parse_asym_payload_regions(renderer)
    candidates = [
        region
        for region in regions
        if include_embeddings or not region.is_embedding
        if include_transposed or not region.transposed
    ]
    candidates.sort(
        key=lambda region: (
            -region.payload_bytes,
            region.layer_index,
            region.layer_name,
            region.blob_kind,
        )
    )

    if selection_mode not in {"payload-desc", "sensitivity-desc", "sensitivity-asc"}:
        raise RendererBasisSelectionError(
            "selection_mode must be one of: payload-desc, sensitivity-desc, sensitivity-asc"
        )
    if sensitivity_scores is None and selection_mode != "payload-desc":
        raise RendererBasisSelectionError(
            f"selection_mode={selection_mode!r} requires --sensitivity-map"
        )
    if sensitivity_scores is not None and selection_mode == "payload-desc":
        raise RendererBasisSelectionError(
            "--sensitivity-map requires --selection-mode sensitivity-desc or sensitivity-asc"
        )

    atoms: list[SelectedAtom] = []
    used_offsets: set[int] = set()
    if sensitivity_scores is not None:
        scored_candidates: list[tuple[float, int, BlobRegion, int]] = []
        for region_order, region in enumerate(candidates):
            scores = _region_sensitivity_scores(sensitivity_scores, region)
            if scores is None:
                continue
            for channel_index, score in enumerate(scores):
                scored_candidates.append((float(score), region_order, region, channel_index))
        if not scored_candidates:
            raise RendererBasisSelectionError(
                "sensitivity map contains no entries compatible with selectable ASYM regions"
            )
        reverse = selection_mode == "sensitivity-desc"
        if not reverse:
            scored_candidates.sort(
                key=lambda item: (
                    item[0],
                    item[1],
                    item[2].layer_index,
                    item[2].layer_name,
                    item[3],
                )
            )
        else:
            scored_candidates.sort(
                key=lambda item: (
                    -item[0],
                    item[1],
                    item[2].layer_index,
                    item[2].layer_name,
                    item[3],
                )
            )
        for score, _region_order, region, channel_index in scored_candidates:
            selected = _select_offset_for_region(
                renderer,
                region,
                max_abs_byte_delta=max_abs_byte_delta,
                channel_index=channel_index,
            )
            if selected is None:
                continue
            offset, selected_channel_index, byte_index, margin = selected
            if offset in used_offsets:
                continue
            used_offsets.add(offset)
            atom_id = (
                "asym_sens_"
                f"{len(atoms):04d}_"
                f"l{region.layer_index:03d}_"
                f"{_sanitize_atom_token(region.layer_name)}_"
                f"ch{selected_channel_index}_b{byte_index}"
            )
            atoms.append(
                SelectedAtom(
                    atom_id=atom_id,
                    member="renderer.bin",
                    offset=offset,
                    delta_per_epsilon=delta_per_epsilon,
                    layer_index=region.layer_index,
                    layer_name=region.layer_name,
                    blob_kind=region.blob_kind,
                    bits=region.bits,
                    shape=region.shape,
                    channel_index=selected_channel_index,
                    byte_index_within_channel_payload=byte_index,
                    original_byte=int(renderer[offset]),
                    selection_rank=len(atoms),
                    payload_bytes=region.payload_bytes,
                    margin_to_byte_range=margin,
                    sensitivity_score=float(score),
                    selection_source=selection_mode,
                )
            )
            if len(atoms) >= max_atoms:
                break
        if not atoms:
            raise RendererBasisSelectionError(
                "no sensitivity-ranked payload bytes found with enough byte-range margin"
            )
        return header, atoms

    for region in candidates:
        selected = _select_offset_for_region(
            renderer,
            region,
            max_abs_byte_delta=max_abs_byte_delta,
        )
        if selected is None:
            continue
        offset, channel_index, byte_index, margin = selected
        if offset in used_offsets:
            continue
        used_offsets.add(offset)
        atom_id = (
            "asym_payload_"
            f"{len(atoms):04d}_"
            f"l{region.layer_index:03d}_"
            f"{_sanitize_atom_token(region.layer_name)}_"
            f"ch{channel_index}_b{byte_index}"
        )
        atoms.append(
            SelectedAtom(
                atom_id=atom_id,
                member="renderer.bin",
                offset=offset,
                delta_per_epsilon=delta_per_epsilon,
                layer_index=region.layer_index,
                layer_name=region.layer_name,
                blob_kind=region.blob_kind,
                bits=region.bits,
                shape=region.shape,
                channel_index=channel_index,
                byte_index_within_channel_payload=byte_index,
                original_byte=int(renderer[offset]),
                selection_rank=len(atoms),
                payload_bytes=region.payload_bytes,
                margin_to_byte_range=margin,
                selection_source="payload-desc",
            )
        )
        if len(atoms) >= max_atoms:
            break
    if not atoms:
        raise RendererBasisSelectionError(
            "no safe payload bytes found with enough byte-range margin"
        )
    return header, atoms


def _apply_atoms(renderer: bytes, atoms: list[SelectedAtom], epsilon: float) -> bytes:
    out = bytearray(renderer)
    for atom in atoms:
        delta = epsilon * atom.delta_per_epsilon
        rounded = round(delta)
        if abs(delta - rounded) > 1e-9:
            raise RendererBasisSelectionError(
                f"epsilon {epsilon!r} does not produce integer byte delta"
            )
        offset = atom.offset
        value = out[offset] + int(rounded)
        if value < 0 or value > 255:
            raise RendererBasisSelectionError(
                f"epsilon {epsilon!r} moves atom {atom.atom_id} outside byte range"
            )
        out[offset] = value
    return bytes(out)


def _decode_verify(
    renderer: bytes,
    atoms: list[SelectedAtom],
    *,
    epsilons: list[float],
) -> dict[str, Any]:
    from tac.renderer_export import load_asymmetric_checkpoint

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="renderer_basis_decode_") as tmp:
        tmp_dir = Path(tmp)
        for epsilon in epsilons:
            mutated = _apply_atoms(renderer, atoms, epsilon)
            path = tmp_dir / f"renderer_eps_{epsilon:+.12g}.bin"
            path.write_bytes(mutated)
            load_asymmetric_checkpoint(path, device="cpu")
            results.append(
                {
                    "epsilon": float(epsilon),
                    "renderer_bytes": len(mutated),
                    "renderer_sha256": _sha256_bytes(mutated),
                    "decode_ok": True,
                }
            )
    return {
        "implemented": True,
        "device": "cpu",
        "loader": "tac.renderer_export.load_asymmetric_checkpoint",
        "points": results,
    }


def build_renderer_blob_perturbation_basis(
    *,
    archive: Path,
    output_json: Path,
    epsilons: list[float],
    max_atoms: int,
    delta_per_epsilon: int = 1,
    include_embeddings: bool = False,
    include_transposed: bool = False,
    decode_verify: bool = True,
    sensitivity_map: Path | None = None,
    selection_mode: str = "payload-desc",
) -> dict[str, Any]:
    archive = archive.resolve()
    output_json = output_json.resolve()
    epsilon_values = sorted({round(_finite_float(eps, field="epsilon"), 15) for eps in epsilons})
    renderer, archive_manifest = _read_renderer_from_archive(archive)
    sensitivity_scores: dict[str, list[float]] | None = None
    sensitivity_map_meta: dict[str, Any] | None = None
    if sensitivity_map is not None:
        sensitivity_map = sensitivity_map.resolve()
        sensitivity_scores, sensitivity_metadata = _load_sensitivity_scores(sensitivity_map)
        sensitivity_map_meta = {
            "path": str(sensitivity_map),
            "bytes": int(sensitivity_map.stat().st_size),
            "sha256": _sha256_file(sensitivity_map),
            "metadata": sensitivity_metadata,
        }
    elif selection_mode != "payload-desc":
        raise RendererBasisSelectionError(
            f"selection_mode={selection_mode!r} requires --sensitivity-map"
        )
    header, atoms = select_basis_atoms(
        renderer,
        max_atoms=max_atoms,
        epsilons=epsilon_values,
        delta_per_epsilon=delta_per_epsilon,
        include_embeddings=include_embeddings,
        include_transposed=include_transposed,
        sensitivity_scores=sensitivity_scores,
        selection_mode=selection_mode,
    )
    verification = (
        _decode_verify(renderer, atoms, epsilons=epsilon_values)
        if decode_verify
        else {"implemented": False, "reason": "disabled_by_cli"}
    )
    atom_payload = [
        {
            "atom_id": atom.atom_id,
            "member": atom.member,
            "offset": atom.offset,
            "delta_per_epsilon": atom.delta_per_epsilon,
            "layer_index": atom.layer_index,
            "layer_name": atom.layer_name,
            "blob_kind": atom.blob_kind,
            "bits": atom.bits,
            "shape": list(atom.shape),
            "channel_index": atom.channel_index,
            "byte_index_within_channel_payload": atom.byte_index_within_channel_payload,
            "original_byte": atom.original_byte,
            "selection_rank": atom.selection_rank,
            "payload_bytes": atom.payload_bytes,
            "margin_to_byte_range": atom.margin_to_byte_range,
            "sensitivity_score": atom.sensitivity_score,
            "selection_source": atom.selection_source,
        }
        for atom in atoms
    ]
    if sensitivity_map_meta is None:
        selection_rule = "largest_non_embedding_weight_layers_median_payload_byte_with_byte_range_margin"
    else:
        selection_rule = (
            "sensitivity_ranked_non_embedding_weight_channels_median_payload_byte_"
            "with_byte_range_margin"
        )
    payload = {
        "schema_version": 1,
        "format": PLAN_BASIS_FORMAT,
        "extended_format": FORMAT,
        "producer": PRODUCER,
        "basis_kind": "asym_renderer_quantized_weight_payload_byte_additive",
        "selection_rule": selection_rule,
        "selection_mode": selection_mode,
        "sensitivity_map": sensitivity_map_meta,
        "epsilon_units": "signed_integer_step",
        "epsilon_ladder": epsilon_values,
        "source_archive": {
            "path": str(archive),
            "bytes": int(archive.stat().st_size),
            "sha256": _sha256_file(archive),
        },
        "source_members": archive_manifest["members"],
        "renderer": {
            "member": "renderer.bin",
            "magic": "ASYM",
            "bytes": len(renderer),
            "sha256": _sha256_bytes(renderer),
            "header_sha256": _sha256_bytes(
                json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ),
            "version": header.get("version"),
            "layer_count": len(header.get("layers", [])),
        },
        "mutation_policy": {
            "allowed_mutation_members": ["renderer.bin"],
            "forbidden_regions": [
                "zip metadata",
                "renderer magic",
                "renderer header length",
                "renderer JSON header",
                "blob length fields",
                "per-channel scale fields",
                "bias blobs",
            ],
            "byte_delta_formula": "new_byte = old_byte + round(epsilon * delta_per_epsilon)",
            "byte_range": [0, 255],
            "out_of_range": "fail_closed",
            "include_embeddings": bool(include_embeddings),
            "include_transposed": bool(include_transposed),
        },
        "decode_verification": verification,
        "atoms": atom_payload,
        "atom_count": len(atom_payload),
        "canonical_response_eval_path": "archive.zip -> inflate.sh -> upstream/evaluate.py",
        "auth_eval_required": "cuda",
        "score_claim": "none",
    }
    _write_json(output_json, payload)
    return {
        "format": "renderer_blob_perturbation_basis_summary_v1",
        "basis": {
            "path": str(output_json),
            "bytes": int(output_json.stat().st_size),
            "sha256": _sha256_file(output_json),
        },
        "atom_count": len(atom_payload),
        "epsilon_ladder": epsilon_values,
        "decode_verification": verification,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--epsilon", action="append", type=float, default=None)
    parser.add_argument("--max-atoms", type=int, default=64)
    parser.add_argument("--delta-per-epsilon", type=int, default=1)
    parser.add_argument(
        "--sensitivity-map",
        type=Path,
        default=None,
        help=(
            "Optional tac_score_sensitivity_map_v1 .pt file. When set, "
            "--selection-mode must be sensitivity-desc or sensitivity-asc."
        ),
    )
    parser.add_argument(
        "--selection-mode",
        choices=("payload-desc", "sensitivity-desc", "sensitivity-asc"),
        default="payload-desc",
        help=(
            "Deterministic atom ordering rule. payload-desc preserves the "
            "legacy largest-layer selection; sensitivity modes rank individual "
            "channels by the supplied map."
        ),
    )
    parser.add_argument("--include-embeddings", action="store_true")
    parser.add_argument(
        "--include-transposed",
        action="store_true",
        help=(
            "Allow ConvTranspose/transposed ASYM payload bytes. Disabled by "
            "default because current promotion sensitivity maps cover "
            "Conv2d/Linear parameters and would not contain transposed keys."
        ),
    )
    parser.add_argument("--skip-decode-verify", action="store_true")
    args = parser.parse_args(argv)
    if args.max_atoms <= 0:
        parser.error("--max-atoms must be positive")
    if args.delta_per_epsilon == 0:
        parser.error("--delta-per-epsilon must be nonzero")
    epsilons = args.epsilon if args.epsilon is not None else list(DEFAULT_EPSILONS)
    args.epsilons = [_finite_float(eps, field="epsilon") for eps in epsilons]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = build_renderer_blob_perturbation_basis(
            archive=args.archive,
            output_json=args.output_json,
            epsilons=args.epsilons,
            max_atoms=args.max_atoms,
            delta_per_epsilon=args.delta_per_epsilon,
            include_embeddings=args.include_embeddings,
            include_transposed=args.include_transposed,
            decode_verify=not args.skip_decode_verify,
            sensitivity_map=args.sensitivity_map,
            selection_mode=args.selection_mode,
        )
    except RendererBasisSelectionError as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
