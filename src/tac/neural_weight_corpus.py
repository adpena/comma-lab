"""Deterministic corpus manifests for Lane J-NWC codec training."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import torch


MANIFEST_SCHEMA_VERSION = 1
DEFAULT_STATE_DICT_KEYS = ("model_state_dict", "state_dict", "model")
TENSOR_INCLUDE_REASON = "selected_weight_tensor"


class CorpusManifestError(RuntimeError):
    """Raised when a corpus manifest cannot be trusted or replayed."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _path_key(path: Path, corpus_dir: Path | None) -> str:
    resolved = path.resolve(strict=False)
    if corpus_dir is None:
        return resolved.as_posix()
    try:
        return resolved.relative_to(corpus_dir.resolve(strict=False)).as_posix()
    except ValueError:
        return resolved.as_posix()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _manifest_safe_relative_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise CorpusManifestError("manifest selected file missing relative_path")
    if "\\" in value:
        raise CorpusManifestError(f"manifest unsafe relative_path: {value!r}")
    parsed = PurePosixPath(value)
    if parsed.is_absolute():
        raise CorpusManifestError(f"manifest unsafe relative_path: {value!r}")
    parts = parsed.parts
    if not parts:
        raise CorpusManifestError("manifest selected file missing relative_path")
    for part in parts:
        if part in ("", ".", "..") or part.startswith(".") or part == "__MACOSX":
            raise CorpusManifestError(f"manifest unsafe relative_path: {value!r}")
    return Path(*parts)


def _extract_state_dict(obj: Any) -> tuple[Mapping[str, Any] | None, str | None]:
    if not isinstance(obj, Mapping):
        return None, None
    for key in DEFAULT_STATE_DICT_KEYS:
        value = obj.get(key)
        if isinstance(value, Mapping):
            return value, key
    return obj, None


def _empty_file_entry(
    path: Path,
    *,
    corpus_dir: Path | None,
    size_bytes: int,
    sha256: str | None,
    selected: bool,
    inclusion_reason: str | None,
    exclusion_reason: str | None,
) -> dict[str, Any]:
    resolved = path.resolve(strict=False)
    return {
        "path": resolved.as_posix(),
        "relative_path": _path_key(path, corpus_dir),
        "size_bytes": int(size_bytes),
        "sha256": sha256,
        "selected": bool(selected),
        "inclusion_reason": inclusion_reason,
        "exclusion_reason": exclusion_reason,
        "state_dict_key": None,
        "selected_tensor_count": 0,
        "excluded_tensor_count": 0,
        "selected_block_count": 0,
        "cap_reached": False,
        "tensors": [],
    }


def _tensor_to_blocks(tensor: torch.Tensor, block_size: int) -> torch.Tensor:
    flat = tensor.detach().reshape(-1).float()
    n_blocks = flat.numel() // block_size
    if n_blocks == 0:
        return torch.zeros(0, block_size, dtype=torch.float32)
    blocks = flat[: n_blocks * block_size].reshape(n_blocks, block_size)
    scales = blocks.abs().amax(dim=1).clamp(min=1e-8)
    return blocks / scales.unsqueeze(1)


def _classify_tensor(name: str, value: Any, block_size: int) -> str | None:
    if not isinstance(value, torch.Tensor):
        return "non_tensor_value"
    if not torch.is_floating_point(value):
        return "non_floating_tensor"
    lower_name = name.lower()
    if value.dim() == 1 and (
        lower_name.endswith(".bias") or lower_name == "bias" or "bias" in lower_name
    ):
        return "bias_1d_small"
    if value.numel() < block_size:
        return "tensor_too_small"
    if value.dim() == 1 and value.numel() < 2048:
        return "bias_1d_small"
    return None


def _tensor_entry(
    *,
    name: str,
    value: Any,
    selected: bool,
    inclusion_reason: str | None,
    exclusion_reason: str | None,
    block_count: int,
    used_block_count: int,
    order_index: int | None,
    corpus_block_start: int | None,
    corpus_block_end: int | None,
) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        dtype = str(value.dtype)
        shape = [int(s) for s in value.shape]
        numel = int(value.numel())
        dim = int(value.dim())
    else:
        dtype = type(value).__name__
        shape = None
        numel = None
        dim = None
    return {
        "name": name,
        "dtype": dtype,
        "shape": shape,
        "dim": dim,
        "numel": numel,
        "selected": bool(selected),
        "inclusion_reason": inclusion_reason,
        "exclusion_reason": exclusion_reason,
        "block_count": int(block_count),
        "used_block_count": int(used_block_count),
        "order_index": order_index,
        "corpus_block_start": corpus_block_start,
        "corpus_block_end": corpus_block_end,
    }


def discover_checkpoint_paths(corpus_dir: Path) -> list[Path]:
    if not corpus_dir.is_dir():
        raise CorpusManifestError(f"corpus dir does not exist: {corpus_dir}")
    root = corpus_dir.resolve(strict=False)
    return sorted(corpus_dir.rglob("*.pt"), key=lambda p: _path_key(p, root))


def build_corpus_manifest(
    checkpoint_paths: Sequence[Path | str],
    *,
    block_size: int,
    max_files: int | None = None,
    max_blocks_per_ckpt: int = 50_000,
    min_checkpoint_bytes: int = 0,
    corpus_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Build a machine-readable, deterministic J-NWC corpus manifest.

    The manifest records every discovered checkpoint, exact file size/SHA,
    selected tensors, skipped tensor reasons, block ordering, and file/block
    caps. No wall-clock timestamps are included so repeated generation over
    unchanged files is byte-stable under canonical JSON serialization.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if max_files is not None and max_files <= 0:
        raise ValueError("max_files must be positive when set")
    if max_blocks_per_ckpt <= 0:
        raise ValueError("max_blocks_per_ckpt must be positive")
    if min_checkpoint_bytes < 0:
        raise ValueError("min_checkpoint_bytes must be non-negative")

    root = Path(corpus_dir).resolve(strict=False) if corpus_dir is not None else None
    paths = [Path(p) for p in checkpoint_paths]
    paths = sorted(paths, key=lambda p: _path_key(p, root))

    files: list[dict[str, Any]] = []
    eligible_seen = 0
    selected_file_count = 0
    selected_tensor_count = 0
    excluded_tensor_count = 0
    selected_blocks_total = 0
    corpus_order_index = 0

    for path in paths:
        resolved = path.resolve(strict=False)
        try:
            size_bytes = resolved.stat().st_size
        except OSError as e:
            files.append(
                _empty_file_entry(
                    resolved,
                    corpus_dir=root,
                    size_bytes=0,
                    sha256=None,
                    selected=False,
                    inclusion_reason=None,
                    exclusion_reason=f"stat_error:{type(e).__name__}",
                )
            )
            continue

        if root is not None and not _is_relative_to(resolved, root):
            files.append(
                _empty_file_entry(
                    resolved,
                    corpus_dir=root,
                    size_bytes=size_bytes,
                    sha256=sha256_file(resolved),
                    selected=False,
                    inclusion_reason=None,
                    exclusion_reason="outside_corpus_dir",
                )
            )
            continue

        digest = sha256_file(resolved)
        if root is not None:
            relative_path = _path_key(resolved, root)
            try:
                _manifest_safe_relative_path(relative_path)
            except CorpusManifestError:
                files.append(
                    _empty_file_entry(
                        resolved,
                        corpus_dir=root,
                        size_bytes=size_bytes,
                        sha256=digest,
                        selected=False,
                        inclusion_reason=None,
                        exclusion_reason="unsafe_relative_path",
                    )
                )
                continue

        if size_bytes < min_checkpoint_bytes:
            files.append(
                _empty_file_entry(
                    resolved,
                    corpus_dir=root,
                    size_bytes=size_bytes,
                    sha256=digest,
                    selected=False,
                    inclusion_reason=None,
                    exclusion_reason="checkpoint_too_small",
                )
            )
            continue

        if max_files is not None and eligible_seen >= max_files:
            files.append(
                _empty_file_entry(
                    resolved,
                    corpus_dir=root,
                    size_bytes=size_bytes,
                    sha256=digest,
                    selected=False,
                    inclusion_reason=None,
                    exclusion_reason="max_files_cap",
                )
            )
            continue
        eligible_seen += 1

        file_entry = _empty_file_entry(
            resolved,
            corpus_dir=root,
            size_bytes=size_bytes,
            sha256=digest,
            selected=False,
            inclusion_reason=None,
            exclusion_reason=None,
        )
        try:
            obj = torch.load(str(resolved), map_location="cpu", weights_only=False)
        except Exception as e:  # pragma: no cover - exercised by corrupt files
            file_entry["exclusion_reason"] = f"checkpoint_load_error:{type(e).__name__}"
            files.append(file_entry)
            continue

        sd, state_dict_key = _extract_state_dict(obj)
        file_entry["state_dict_key"] = state_dict_key
        if sd is None:
            file_entry["exclusion_reason"] = "state_dict_not_found"
            files.append(file_entry)
            continue

        ckpt_blocks = 0
        tensors: list[dict[str, Any]] = []
        for raw_name, value in sorted(sd.items(), key=lambda item: str(item[0])):
            name = str(raw_name)
            reason = _classify_tensor(name, value, block_size)
            block_count = (
                int(value.numel() // block_size)
                if isinstance(value, torch.Tensor) and torch.is_floating_point(value)
                else 0
            )
            if reason is not None:
                tensors.append(
                    _tensor_entry(
                        name=name,
                        value=value,
                        selected=False,
                        inclusion_reason=None,
                        exclusion_reason=reason,
                        block_count=block_count,
                        used_block_count=0,
                        order_index=None,
                        corpus_block_start=None,
                        corpus_block_end=None,
                    )
                )
                excluded_tensor_count += 1
                continue

            remaining = max_blocks_per_ckpt - ckpt_blocks
            if remaining <= 0:
                tensors.append(
                    _tensor_entry(
                        name=name,
                        value=value,
                        selected=False,
                        inclusion_reason=None,
                        exclusion_reason="max_blocks_per_checkpoint_cap",
                        block_count=block_count,
                        used_block_count=0,
                        order_index=None,
                        corpus_block_start=None,
                        corpus_block_end=None,
                    )
                )
                excluded_tensor_count += 1
                file_entry["cap_reached"] = True
                continue

            used = min(block_count, remaining)
            start = selected_blocks_total
            end = start + used
            tensors.append(
                _tensor_entry(
                    name=name,
                    value=value,
                    selected=True,
                    inclusion_reason=TENSOR_INCLUDE_REASON,
                    exclusion_reason=None,
                    block_count=block_count,
                    used_block_count=used,
                    order_index=corpus_order_index,
                    corpus_block_start=start,
                    corpus_block_end=end,
                )
            )
            corpus_order_index += 1
            selected_tensor_count += 1
            selected_blocks_total += used
            ckpt_blocks += used
            if used < block_count or ckpt_blocks >= max_blocks_per_ckpt:
                file_entry["cap_reached"] = True

        file_entry["tensors"] = tensors
        file_entry["selected_tensor_count"] = sum(1 for t in tensors if t["selected"])
        file_entry["excluded_tensor_count"] = sum(1 for t in tensors if not t["selected"])
        file_entry["selected_block_count"] = ckpt_blocks
        if ckpt_blocks > 0:
            file_entry["selected"] = True
            file_entry["inclusion_reason"] = "selected_checkpoint"
            selected_file_count += 1
        else:
            file_entry["selected"] = False
            file_entry["exclusion_reason"] = "no_usable_tensors"
        files.append(file_entry)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "format": "tac.neural_weight_corpus.v1",
        "selection": {
            "block_size": int(block_size),
            "max_files": None if max_files is None else int(max_files),
            "max_blocks_per_checkpoint": int(max_blocks_per_ckpt),
            "min_checkpoint_bytes": int(min_checkpoint_bytes),
            "corpus_dir": None if root is None else root.as_posix(),
            "discovery_order": "stable_path_sort",
            "tensor_order": "stable_name_sort",
        },
        "totals": {
            "discovered_files": len(files),
            "selected_files": selected_file_count,
            "excluded_files": len(files) - selected_file_count,
            "selected_tensors": selected_tensor_count,
            "excluded_tensors": excluded_tensor_count,
            "selected_blocks": selected_blocks_total,
        },
        "files": files,
    }
    validate_manifest_has_corpus(manifest)
    return manifest


def build_corpus_manifest_from_dir(
    corpus_dir: Path | str,
    *,
    block_size: int,
    max_files: int | None,
    max_blocks_per_ckpt: int = 50_000,
    min_checkpoint_bytes: int = 1024,
) -> dict[str, Any]:
    root = Path(corpus_dir)
    paths = discover_checkpoint_paths(root)
    return build_corpus_manifest(
        paths,
        block_size=block_size,
        max_files=max_files,
        max_blocks_per_ckpt=max_blocks_per_ckpt,
        min_checkpoint_bytes=min_checkpoint_bytes,
        corpus_dir=root,
    )


def validate_manifest_has_corpus(manifest: Mapping[str, Any]) -> None:
    totals = manifest.get("totals")
    if not isinstance(totals, Mapping):
        raise CorpusManifestError("manifest missing totals")
    selected_blocks = int(totals.get("selected_blocks", 0))
    selected_files = int(totals.get("selected_files", 0))
    if selected_files <= 0 or selected_blocks <= 0:
        raise CorpusManifestError(
            "manifest has empty J-NWC corpus "
            f"(selected_files={selected_files}, selected_blocks={selected_blocks})"
        )


def canonical_manifest_json(manifest: Mapping[str, Any]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def write_corpus_manifest(manifest: Mapping[str, Any], path: Path | str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(canonical_manifest_json(manifest))


def load_corpus_manifest(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    try:
        manifest = json.loads(p.read_text())
    except Exception as e:
        raise CorpusManifestError(f"could not load corpus manifest {p}: {e}") from e
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise CorpusManifestError(
            f"unsupported corpus manifest schema_version={manifest.get('schema_version')}"
        )
    validate_manifest_has_corpus(manifest)
    return manifest


def _manifest_checkpoint_path(
    file_entry: Mapping[str, Any],
    *,
    replay_root: Path | None,
) -> Path:
    if replay_root is not None:
        relative = _manifest_safe_relative_path(file_entry.get("relative_path"))
        root = replay_root.resolve(strict=False)
        candidate = root / relative
        if not candidate.is_file():
            raise CorpusManifestError(f"manifest checkpoint missing under replay_root: {candidate}")
        resolved = candidate.resolve(strict=True)
        if not _is_relative_to(resolved, root):
            raise CorpusManifestError(f"manifest relative_path escapes replay_root: {relative}")
        return resolved
    return Path(str(file_entry["path"]))


def build_corpus_from_manifest(
    manifest_or_path: Mapping[str, Any] | Path | str,
    *,
    replay_root: Path | str | None = None,
) -> torch.Tensor:
    """Replay a manifest and return the exact corpus it describes."""
    if isinstance(manifest_or_path, (str, Path)):
        manifest = load_corpus_manifest(manifest_or_path)
    else:
        manifest = dict(manifest_or_path)
        if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise CorpusManifestError(
                f"unsupported corpus manifest schema_version={manifest.get('schema_version')}"
            )
        validate_manifest_has_corpus(manifest)
    replay_root_path = Path(replay_root).resolve(strict=False) if replay_root is not None else None

    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise CorpusManifestError("manifest missing selection")
    block_size = int(selection.get("block_size", 0))
    if block_size <= 0:
        raise CorpusManifestError(f"invalid block_size in manifest: {block_size}")

    out_blocks: list[torch.Tensor] = []
    expected_next = 0
    for file_entry in manifest.get("files", []):
        if not file_entry.get("selected"):
            continue
        path = _manifest_checkpoint_path(file_entry, replay_root=replay_root_path)
        if not path.is_file():
            raise CorpusManifestError(f"manifest checkpoint missing: {path}")
        size_bytes = path.stat().st_size
        if int(file_entry["size_bytes"]) != size_bytes:
            raise CorpusManifestError(
                f"manifest size mismatch for {path}: "
                f"manifest={file_entry['size_bytes']} actual={size_bytes}"
            )
        digest = sha256_file(path)
        if file_entry.get("sha256") != digest:
            raise CorpusManifestError(f"manifest sha256 mismatch for {path}")

        obj = torch.load(str(path), map_location="cpu", weights_only=False)
        sd, _ = _extract_state_dict(obj)
        if sd is None:
            raise CorpusManifestError(f"manifest selected file has no state dict: {path}")

        for tensor_entry in file_entry.get("tensors", []):
            if not tensor_entry.get("selected"):
                continue
            name = str(tensor_entry["name"])
            if name not in sd:
                raise CorpusManifestError(f"manifest tensor {name!r} missing in {path}")
            value = sd[name]
            reason = _classify_tensor(name, value, block_size)
            if reason is not None:
                raise CorpusManifestError(
                    f"manifest tensor {name!r} in {path} is no longer selectable: {reason}"
                )
            shape = [int(s) for s in value.shape]
            if shape != list(tensor_entry["shape"]):
                raise CorpusManifestError(
                    f"manifest shape mismatch for {path}:{name}: "
                    f"manifest={tensor_entry['shape']} actual={shape}"
                )
            if str(value.dtype) != str(tensor_entry["dtype"]):
                raise CorpusManifestError(
                    f"manifest dtype mismatch for {path}:{name}: "
                    f"manifest={tensor_entry['dtype']} actual={value.dtype}"
                )
            block_count = int(value.numel() // block_size)
            if block_count != int(tensor_entry["block_count"]):
                raise CorpusManifestError(
                    f"manifest block_count mismatch for {path}:{name}: "
                    f"manifest={tensor_entry['block_count']} actual={block_count}"
                )
            used = int(tensor_entry["used_block_count"])
            if used <= 0 or used > block_count:
                raise CorpusManifestError(
                    f"manifest used_block_count invalid for {path}:{name}: {used}"
                )
            start = int(tensor_entry["corpus_block_start"])
            end = int(tensor_entry["corpus_block_end"])
            if start != expected_next or end != start + used:
                raise CorpusManifestError(
                    f"manifest block ordering gap at {path}:{name}: "
                    f"start={start} end={end} expected_start={expected_next}"
                )
            blocks = _tensor_to_blocks(value, block_size)[:used]
            out_blocks.append(blocks)
            expected_next = end

    expected_total = int(manifest["totals"]["selected_blocks"])
    if expected_next != expected_total:
        raise CorpusManifestError(
            f"manifest total block mismatch: replayed={expected_next} "
            f"manifest={expected_total}"
        )
    if not out_blocks:
        raise CorpusManifestError("manifest replay produced no blocks")
    return torch.cat(out_blocks, dim=0)


def build_corpus_from_checkpoints(
    checkpoint_paths: Sequence[Path | str],
    block_size: int,
    *,
    max_blocks_per_ckpt: int = 50_000,
    max_files: int | None = None,
    min_checkpoint_bytes: int = 0,
) -> torch.Tensor:
    """Backward-compatible corpus builder with deterministic manifest semantics."""
    manifest = build_corpus_manifest(
        checkpoint_paths,
        block_size=block_size,
        max_files=max_files,
        max_blocks_per_ckpt=max_blocks_per_ckpt,
        min_checkpoint_bytes=min_checkpoint_bytes,
    )
    return build_corpus_from_manifest(manifest)
