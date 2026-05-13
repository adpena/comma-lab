from __future__ import annotations

import lzma
import multiprocessing
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path

from .arithmetic import FRAME_BOS_TOKEN, SEGMENT_EOT_TOKEN, flatten_tokens_for_gpt_arithmetic
from .contracts import LosslessCompressionResult
from .data import TokenRecord, build_token_records, load_commavq_dataset
from .evaluate import compression_rate, evaluate_local_submission_contract
from .frequency_coder import decode_uint16_prev_symbol_stream, encode_uint16_prev_symbol_stream
from .global_prev_symbol import (
    encode_corpus_global_prev_symbol_position_major,
)
from .profiles import PROFILES
from .submission import build_submission_zip

_FIXED_ZPAQ_MTIME = 1704067200  # 2024-01-01T00:00:00Z
_ZPAQ_SUFFIX = ".zpaq"
_PREV_SYMBOL_SUFFIX = ".tpc"
_ZPAQ_PRESET_TO_METHOD = {
    "fast": "1",
    "normal": "3",
    "better": "4",
    "max": "5",
}
_LOCAL_ONLY_ZPAQ_REASON = "zpaq is local-only unless a self-contained runtime is bundled in the submission payload"
_DEFAULT_ZSTD_MAX_TRAINING_SAMPLES = 1024


def _require_zstd_backend():
    try:
        import zstandard as zstd

        class _Backend:
            @staticmethod
            def train_dictionary(samples, *, dict_size: int) -> bytes:
                dictionary = zstd.train_dictionary(dict_size, list(samples))
                return dictionary.as_bytes()

            @staticmethod
            def compress(data: bytes, *, dictionary: bytes) -> bytes:
                dict_data = zstd.ZstdCompressionDict(dictionary)
                return zstd.ZstdCompressor(dict_data=dict_data).compress(data)

            @staticmethod
            def decompress(data: bytes, *, dictionary: bytes) -> bytes:
                dict_data = zstd.ZstdCompressionDict(dictionary)
                return zstd.ZstdDecompressor(dict_data=dict_data).decompress(data)

        return _Backend()
    except ImportError:
        binary = shutil.which("zstd")
        if binary is None:
            raise RuntimeError("zstandard backend is unavailable; install zstandard to use zstd_dict experiments")

    class _CliBackend:
        @staticmethod
        def train_dictionary(samples, *, dict_size: int) -> bytes:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                dict_path = root / "dict.zstd"
                sample_paths: list[str] = []
                for index, sample in enumerate(samples):
                    sample_path = root / f"sample_{index}.bin"
                    sample_path.write_bytes(sample)
                    sample_paths.append(str(sample_path))
                subprocess.run(
                    [binary, "--train", *sample_paths, "--maxdict", str(dict_size), "-o", str(dict_path)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return dict_path.read_bytes()

        @staticmethod
        def compress(data: bytes, *, dictionary: bytes) -> bytes:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                dict_path = root / "dict.zstd"
                source_path = root / "input.bin"
                output_path = root / "output.zst"
                dict_path.write_bytes(dictionary)
                source_path.write_bytes(data)
                subprocess.run(
                    [binary, "-q", "-f", "-D", str(dict_path), str(source_path), "-o", str(output_path)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return output_path.read_bytes()

        @staticmethod
        def decompress(data: bytes, *, dictionary: bytes) -> bytes:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                dict_path = root / "dict.zstd"
                source_path = root / "input.zst"
                output_path = root / "output.bin"
                dict_path.write_bytes(dictionary)
                source_path.write_bytes(data)
                subprocess.run(
                    [binary, "-q", "-f", "-d", str(source_path), "-D", str(dict_path), "-o", str(output_path)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return output_path.read_bytes()

    return _CliBackend()


def _cap_samples(samples: list[bytes], *, max_samples: int | None) -> list[bytes]:
    if max_samples is None or len(samples) <= max_samples:
        return samples
    if max_samples <= 0:
        raise ValueError("max_training_samples must be positive")
    if max_samples == 1:
        return [samples[0]]
    last_index = len(samples) - 1
    return [samples[(index * last_index) // (max_samples - 1)] for index in range(max_samples)]


def _profile_config(profile: str) -> dict[str, object]:
    try:
        config = PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"unknown lossless profile: {profile}") from exc
    method = str(config.get("method", ""))
    if method not in {"lzma", "zpaq", "prev_symbol_position_major", "global_prev_symbol_position_major"}:
        raise ValueError(f"unsupported lossless method for real codec path: {method}")
    return config


def _profile_method(profile: str) -> str:
    return str(_profile_config(profile).get("method"))


def _require_zpaq_binary() -> str:
    binary = shutil.which("zpaq")
    if binary is None:
        raise RuntimeError("zpaq binary is unavailable; install zpaq or use lzma_baseline")
    return binary


def _zpaq_method_arg(config: dict[str, object]) -> str:
    preset = str(config.get("preset", "max")).strip().lower()
    if preset.isdigit():
        return preset
    return _ZPAQ_PRESET_TO_METHOD.get(preset, "5")


def _run_external(cmd: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _write_deterministic_stage_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    os.utime(path, (_FIXED_ZPAQ_MTIME, _FIXED_ZPAQ_MTIME))


def _zpaq_output_path(root: Path, file_name: str) -> Path:
    return root / f"{file_name}{_ZPAQ_SUFFIX}"


def _decode_target_name(archive_name: str) -> str:
    if archive_name.endswith(_PREV_SYMBOL_SUFFIX):
        return archive_name[: -len(_PREV_SYMBOL_SUFFIX)]
    if archive_name.endswith(_ZPAQ_SUFFIX):
        return archive_name[: -len(_ZPAQ_SUFFIX)]
    return archive_name


def _prev_symbol_output_path(root: Path, file_name: str) -> Path:
    return root / f"{file_name}{_PREV_SYMBOL_SUFFIX}"


def _flatten_prev_symbol_position_major(tokens) -> bytes:
    import numpy as np

    flattened = flatten_tokens_for_gpt_arithmetic(tokens, layout="position_major").astype(np.uint16)
    return flattened.tobytes()


def _inflate_prev_symbol_position_major(payload: bytes):
    import numpy as np

    flat = np.frombuffer(payload, dtype=np.uint16)
    if flat.size == 0 or int(flat[-1]) != SEGMENT_EOT_TOKEN:
        raise ValueError("position-major payload must end with segment EOT")
    body = flat[:-1]
    positions = 128
    if body.size % positions != 0:
        raise ValueError("position-major payload body must be divisible by 128")
    stream_len = body.size // positions
    if stream_len <= 1:
        raise ValueError("position-major payload must contain at least one frame")
    streams = body.reshape(positions, stream_len)
    if not np.all(streams[:, 0] == FRAME_BOS_TOKEN):
        raise ValueError("position-major payload is missing BOS headers")
    frames = streams[:, 1:].T
    return frames.reshape(frames.shape[0], 8, 16).astype(np.int16, copy=False)


def _runtime_metadata(*, method: str, runtime_bundle_path: str | Path | None = None) -> dict[str, object]:
    if method != "zpaq":
        return {
            "local_only": False,
            "challenge_valid": True,
            "runtime_bundle_included": False,
            "runtime_bundle_relpath": None,
            "challenge_validity_reason": None,
        }

    if runtime_bundle_path is None:
        return {
            "local_only": True,
            "challenge_valid": False,
            "runtime_bundle_included": False,
            "runtime_bundle_relpath": None,
            "challenge_validity_reason": _LOCAL_ONLY_ZPAQ_REASON,
        }

    source = Path(runtime_bundle_path)
    if not source.is_file():
        raise ValueError(f"runtime_bundle_path must point to an existing file: {source}")
    return {
        "local_only": False,
        "challenge_valid": True,
        "runtime_bundle_included": True,
        "runtime_bundle_relpath": f"runtime/{source.name}",
        "challenge_validity_reason": None,
    }


def _bundle_runtime_if_needed(
    *,
    payload_dir: Path,
    runtime_metadata: Mapping[str, object],
    runtime_bundle_path: str | Path | None,
) -> None:
    if not runtime_metadata.get("runtime_bundle_included"):
        return
    if runtime_bundle_path is None:
        raise ValueError("runtime_bundle_path is required when runtime_bundle_included is true")
    relpath = runtime_metadata.get("runtime_bundle_relpath")
    if not isinstance(relpath, str) or not relpath:
        raise ValueError("runtime_bundle_relpath must be a non-empty string")
    target = payload_dir / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(runtime_bundle_path, target)


def _extract_single_file_from_dir(extracted_dir: Path, preferred_name: str | None = None) -> Path:
    candidates = sorted(path for path in extracted_dir.rglob("*") if path.is_file())
    if not candidates:
        raise FileNotFoundError(f"zpaq extraction produced no files in {extracted_dir}")
    if preferred_name is not None:
        for candidate in candidates:
            if candidate.name == preferred_name:
                return candidate
    return candidates[0]


def _compress_with_zpaq(*, source_name: str, payload: bytes, output_path: Path, config: dict[str, object]) -> int:
    binary = _require_zpaq_binary()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    with tempfile.TemporaryDirectory() as tmpdir:
        stage_root = Path(tmpdir)
        staged_source = stage_root / source_name
        _write_deterministic_stage_file(staged_source, payload)
        # Prepend "./" to prevent argument injection from filenames starting
        # with "-", which zpaq would interpret as option flags.
        safe_name = "./" + source_name
        _run_external(
            [binary, "add", str(output_path), safe_name, "-method", _zpaq_method_arg(config)],
            cwd=stage_root,
        )
    return output_path.stat().st_size


def _decompress_with_zpaq(*, archive_path: Path, output_path: Path) -> Path:
    binary = _require_zpaq_binary()
    with tempfile.TemporaryDirectory() as tmpdir:
        extracted_dir = Path(tmpdir) / "extract"
        extracted_dir.mkdir(parents=True, exist_ok=True)
        _run_external(
            [binary, "extract", str(archive_path), "-to", str(extracted_dir)],
            cwd=extracted_dir,
        )
        extracted = _extract_single_file_from_dir(extracted_dir, preferred_name=output_path.name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(extracted.read_bytes())
    return output_path


def zstd_dict_roundtrip_file(
    *,
    source_path: str | Path,
    compressed_path: str | Path,
    restored_path: str | Path,
    dict_size: int = 8192,
    sample_payloads: list[bytes] | None = None,
    sample_block_bytes: int | None = None,
    max_training_samples: int | None = _DEFAULT_ZSTD_MAX_TRAINING_SAMPLES,
) -> dict[str, object]:
    backend = _require_zstd_backend()
    source = Path(source_path)
    compressed = Path(compressed_path)
    restored = Path(restored_path)

    payload = source.read_bytes()
    samples = list(sample_payloads) if sample_payloads is not None else [payload]
    if sample_block_bytes is not None:
        if sample_block_bytes <= 0:
            raise ValueError("sample_block_bytes must be positive")
        blocked_samples: list[bytes] = []
        for sample in samples:
            blocked_samples.extend(sample[offset : offset + sample_block_bytes] for offset in range(0, len(sample), sample_block_bytes))
        samples = [sample for sample in blocked_samples if sample]
    samples = _cap_samples(samples, max_samples=max_training_samples)
    total_sample_bytes = sum(len(sample) for sample in samples)
    if len(samples) < 2 or total_sample_bytes <= dict_size:
        raise ValueError(
            "sample corpus is too small for zstd dictionary training: "
            f"samples={len(samples)} total_sample_bytes={total_sample_bytes} dict_size={dict_size}"
        )
    dict_bytes = backend.train_dictionary(samples, dict_size=dict_size)
    compressed_bytes = backend.compress(payload, dictionary=dict_bytes)
    restored_bytes = backend.decompress(compressed_bytes, dictionary=dict_bytes)

    compressed.parent.mkdir(parents=True, exist_ok=True)
    compressed.write_bytes(compressed_bytes)
    restored.parent.mkdir(parents=True, exist_ok=True)
    restored.write_bytes(restored_bytes)

    return {
        "method": "zstd_dict",
        "source_path": str(source),
        "compressed_path": str(compressed),
        "restored_path": str(restored),
        "dictionary_bytes": len(dict_bytes),
        "sample_count": len(samples),
        "archive_bytes": compressed.stat().st_size,
        "original_bytes": len(payload),
        "compression_rate": compression_rate(compressed.stat().st_size, len(payload)),
    }


def benchmark_zstd_dict_file(
    *,
    source_path: str | Path,
    compressed_path: str | Path,
    restored_path: str | Path,
    sample_paths: list[str | Path] | None = None,
    dict_size: int = 8192,
    sample_block_bytes: int | None = None,
    max_training_samples: int | None = _DEFAULT_ZSTD_MAX_TRAINING_SAMPLES,
) -> dict[str, object]:
    source = Path(source_path)
    samples = [Path(path).read_bytes() for path in sample_paths] if sample_paths else None
    payload = zstd_dict_roundtrip_file(
        source_path=source,
        compressed_path=compressed_path,
        restored_path=restored_path,
        dict_size=dict_size,
        sample_payloads=samples,
        sample_block_bytes=sample_block_bytes,
        max_training_samples=max_training_samples,
    )
    payload["command"] = "lossless_zstd_dict_benchmark"
    return payload


def benchmark_zstd_dict_directory(
    *,
    source_root: str | Path,
    compressed_root: str | Path,
    restored_root: str | Path,
    sample_paths: list[str | Path] | None = None,
    dict_size: int = 8192,
    sample_block_bytes: int | None = None,
    max_training_samples: int | None = _DEFAULT_ZSTD_MAX_TRAINING_SAMPLES,
) -> dict[str, object]:
    source_dir = Path(source_root)
    compressed_dir = Path(compressed_root)
    restored_dir = Path(restored_root)
    files = sorted(path for path in source_dir.rglob("*") if path.is_file())
    sample_paths = list(sample_paths or [])

    archive_bytes = 0
    original_bytes = 0
    dictionary_bytes = 0
    for source_path in files:
        relative = source_path.relative_to(source_dir)
        result = zstd_dict_roundtrip_file(
            source_path=source_path,
            compressed_path=compressed_dir / f"{relative.as_posix()}.zst",
            restored_path=restored_dir / relative,
            dict_size=dict_size,
            sample_payloads=[Path(path).read_bytes() for path in sample_paths] if sample_paths else None,
            sample_block_bytes=sample_block_bytes,
            max_training_samples=max_training_samples,
        )
        archive_bytes += int(result["archive_bytes"])
        original_bytes += int(result["original_bytes"])
        dictionary_bytes = max(dictionary_bytes, int(result["dictionary_bytes"]))

    return {
        "command": "lossless_zstd_dict_directory_benchmark",
        "method": "zstd_dict",
        "source_root": str(source_dir),
        "compressed_root": str(compressed_dir),
        "restored_root": str(restored_dir),
        "dictionary_bytes": dictionary_bytes,
        "sample_count": len(sample_paths),
        "file_count": len(files),
        "archive_bytes": archive_bytes,
        "original_bytes": original_bytes,
        "compression_rate": compression_rate(archive_bytes, original_bytes) if original_bytes else 0.0,
    }


def benchmark_zstd_dict_chunked_file(
    *,
    source_path: str | Path,
    compressed_root: str | Path,
    restored_root: str | Path,
    block_bytes: int,
    sample_paths: list[str | Path] | None = None,
    dict_size: int = 8192,
    sample_block_bytes: int | None = None,
    max_training_samples: int | None = _DEFAULT_ZSTD_MAX_TRAINING_SAMPLES,
) -> dict[str, object]:
    source = Path(source_path)
    if block_bytes <= 0:
        raise ValueError("block_bytes must be positive")

    with tempfile.TemporaryDirectory() as tmpdir:
        chunk_root = Path(tmpdir) / "chunks"
        chunk_root.mkdir(parents=True, exist_ok=True)
        payload = source.read_bytes()
        for index, offset in enumerate(range(0, len(payload), block_bytes)):
            (chunk_root / f"{index:06d}.bin").write_bytes(payload[offset : offset + block_bytes])

        result = benchmark_zstd_dict_directory(
            source_root=chunk_root,
            compressed_root=compressed_root,
            restored_root=restored_root,
            sample_paths=sample_paths,
            dict_size=dict_size,
            sample_block_bytes=sample_block_bytes,
            max_training_samples=max_training_samples,
        )
        restored_bytes = bytearray()
        for chunk in sorted(path for path in Path(restored_root).rglob("*") if path.is_file()):
            restored_bytes.extend(chunk.read_bytes())

    return {
        "command": "lossless_zstd_dict_chunked_benchmark",
        "method": "zstd_dict",
        "source_path": str(source),
        "compressed_root": str(Path(compressed_root)),
        "restored_root": str(Path(restored_root)),
        "dictionary_bytes": result["dictionary_bytes"],
        "sample_count": result["sample_count"],
        "file_count": result["file_count"],
        "block_bytes": block_bytes,
        "archive_bytes": result["archive_bytes"],
        "original_bytes": result["original_bytes"],
        "compression_rate": result["compression_rate"],
        "exact_match": bytes(restored_bytes) == payload,
    }


def compress_lossless_file(
    *, profile: str, input_path: str | Path, output_path: str | Path
) -> LosslessCompressionResult:
    import numpy as np

    config = _profile_config(profile)
    method = str(config["method"])
    source = Path(input_path)
    target = Path(output_path)
    payload = source.read_bytes()

    if method == "lzma":
        compressed = lzma.compress(payload, preset=int(config.get("level", 6)))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(compressed)
    elif method == "zpaq":
        _compress_with_zpaq(source_name=source.name, payload=payload, output_path=target, config=config)
    else:
        if len(payload) % 2 != 0:
            raise ValueError("prev_symbol_position_major requires an even-byte uint16 token stream")
        tokens = np.frombuffer(payload, dtype=np.uint16)
        encoded = encode_uint16_prev_symbol_stream(tokens)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(encoded.encoded_bytes)

    archive_bytes = target.stat().st_size
    return LosslessCompressionResult(
        profile=profile,
        archive_path=str(target),
        archive_bytes=archive_bytes,
        original_bytes=len(payload),
        compression_rate=compression_rate(archive_bytes, len(payload)),
        method=method,
    )


def decompress_lossless_file(*, profile: str, archive_path: str | Path, output_path: str | Path) -> Path:
    import numpy as np

    config = _profile_config(profile)
    method = str(config["method"])
    source = Path(archive_path)
    target = Path(output_path)

    if method == "lzma":
        payload = lzma.decompress(source.read_bytes())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return target

    if method == "zpaq":
        return _decompress_with_zpaq(archive_path=source, output_path=target)

    tokens = decode_uint16_prev_symbol_stream(source.read_bytes())
    target.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(tokens, dtype=np.uint16).tofile(target)
    return target


def _encode_commavq_tokens(tokens) -> bytes:
    import numpy as np

    array = np.asarray(tokens).astype(np.int16)
    return array.reshape(-1, 128).T.ravel().tobytes()


def _decode_commavq_tokens(payload: bytes):
    import numpy as np

    return np.frombuffer(payload, dtype=np.int16).reshape(128, -1).T.reshape(-1, 8, 16)


def _effective_num_proc(num_proc: int | None) -> int | None:
    if num_proc is not None:
        return num_proc
    return multiprocessing.cpu_count() or 1


def _dataset_train_split(dataset) -> object:
    if not isinstance(dataset, Mapping):
        raise ValueError("dataset_loader must return a mapping of splits to datasets")
    if "train" not in dataset:
        raise ValueError("dataset_loader must provide a 'train' split")
    return dataset["train"]


def _compress_record_to_payload(example: Mapping[str, object], *, profile: str, payload_dir: str) -> dict[str, object]:
    import numpy as np

    meta = example.get("json")
    if not isinstance(meta, Mapping):
        raise ValueError("dataset example is missing json metadata")
    file_name = meta.get("file_name")
    if not isinstance(file_name, str) or not file_name.strip():
        raise ValueError("dataset example is missing json.file_name")
    if "token.npy" not in example:
        raise ValueError(f"dataset example {file_name!r} is missing token.npy")

    output_root = Path(payload_dir)
    method = _profile_method(profile)
    config = _profile_config(profile)

    if method == "lzma":
        payload = _encode_commavq_tokens(example["token.npy"])
        compressed = lzma.compress(payload, preset=int(config.get("level", 6)))
        target = output_root / file_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(compressed)
        archive_bytes = target.stat().st_size
    elif method == "zpaq":
        payload = _encode_commavq_tokens(example["token.npy"])
        target = _zpaq_output_path(output_root, file_name)
        archive_bytes = _compress_with_zpaq(
            source_name=file_name,
            payload=payload,
            output_path=target,
            config=config,
        )
    else:
        payload = _flatten_prev_symbol_position_major(example["token.npy"])
        encoded = encode_uint16_prev_symbol_stream(np.frombuffer(payload, dtype=np.uint16))
        target = _prev_symbol_output_path(output_root, file_name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(encoded.encoded_bytes)
        archive_bytes = target.stat().st_size
    return {
        "archive_bytes": archive_bytes,
        "file_name": file_name,
    }


def _sum_archive_bytes(mapped_output: object) -> int:
    if isinstance(mapped_output, Mapping):
        if "archive_bytes" in mapped_output:
            values = mapped_output["archive_bytes"]
        elif "train" in mapped_output:
            values = mapped_output["train"]
        else:
            raise ValueError("dataset map output did not include archive_bytes")
    elif hasattr(mapped_output, "__getitem__"):
        try:
            values = mapped_output["archive_bytes"]
        except Exception:
            values = mapped_output
    else:
        values = mapped_output

    if isinstance(values, Mapping):
        if "archive_bytes" not in values:
            raise ValueError("dataset map output did not include archive_bytes")
        values = values["archive_bytes"]

    if isinstance(values, Iterable) and not isinstance(values, (str, bytes, bytearray)):
        total = 0
        for item in values:
            if isinstance(item, Mapping):
                total += int(item["archive_bytes"])
            else:
                total += int(item)
        return total

    raise ValueError("dataset map output did not include iterable archive byte metadata")


def _compress_dataset_split(*, profile: str, split_dataset, payload_dir: Path, num_proc: int | None) -> tuple[int, int]:
    payload_dir.mkdir(parents=True, exist_ok=True)
    effective_num_proc = _effective_num_proc(num_proc)

    if hasattr(split_dataset, "map"):
        mapped = split_dataset.map(
            _compress_record_to_payload,
            desc="compress_example",
            num_proc=effective_num_proc,
            load_from_cache_file=False,
            fn_kwargs={
                "profile": profile,
                "payload_dir": str(payload_dir),
            },
        )
        total_rows = getattr(split_dataset, "num_rows", None)
        if total_rows is None:
            try:
                total_rows = len(split_dataset)
            except TypeError:
                total_rows = None
        return _sum_archive_bytes(mapped), int(total_rows) if total_rows is not None else 0

    total_bytes = 0
    total_rows = 0
    for example in split_dataset:
        output = _compress_record_to_payload(
            example=example,
            profile=profile,
            payload_dir=str(payload_dir),
        )
        total_bytes += int(output["archive_bytes"])
        total_rows += 1
    return total_bytes, total_rows


def compress_token_records(*, profile: str, records: list[TokenRecord], output_dir: str | Path) -> int:
    config = _profile_config(profile)
    method = str(config["method"])
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    total_bytes = 0

    for record in sorted(records, key=lambda item: item.file_name):
        payload = _encode_commavq_tokens(record.tokens)
        if method == "lzma":
            compressed = lzma.compress(payload, preset=int(config.get("level", 6)))
            target = root / record.file_name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(compressed)
        else:
            target = _zpaq_output_path(root, record.file_name)
            total_bytes += _compress_with_zpaq(
                source_name=record.file_name,
                payload=payload,
                output_path=target,
                config=config,
            )
            continue
        total_bytes += target.stat().st_size
    return total_bytes


def decompress_token_records(*, profile: str, compressed_dir: str | Path, output_dir: str | Path) -> None:
    import numpy as np

    config = _profile_config(profile)
    method = str(config["method"])
    source_root = Path(compressed_dir)
    target_root = Path(output_dir)
    target_root.mkdir(parents=True, exist_ok=True)

    for compressed in sorted(
        path for path in source_root.rglob("*") if path.is_file() and path.name != "decompress.py"
    ):
        if method == "lzma":
            relative_name = compressed.relative_to(source_root).as_posix()
            tokens = _decode_commavq_tokens(lzma.decompress(compressed.read_bytes()))
            target = target_root / relative_name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as handle:
                np.save(handle, tokens)
            continue

        relative_name = compressed.relative_to(source_root).as_posix()
        decoded_name = _decode_target_name(relative_name)
        target = target_root / decoded_name
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            extracted_dir = Path(tmpdir) / "extract"
            extracted_dir.mkdir(parents=True, exist_ok=True)
            _run_external(
                [_require_zpaq_binary(), "extract", str(compressed), "-to", str(extracted_dir)],
                cwd=extracted_dir,
            )
            extracted = _extract_single_file_from_dir(extracted_dir, preferred_name=Path(decoded_name).name)
            # Read inside the with block — temp dir is deleted on exit
            try:
                np.load(extracted, allow_pickle=False)
            except Exception:
                tokens = _decode_commavq_tokens(extracted.read_bytes())
                with target.open("wb") as handle:
                    np.save(handle, tokens)
            else:
                target.write_bytes(extracted.read_bytes())


def render_lzma_decompress_script() -> str:
    return """#!/usr/bin/env python3
import os
import lzma
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
output_dir = Path(os.environ.get("OUTPUT_DIR", HERE / "compression_challenge_submission_decompressed"))

def decompress_bytes(x: bytes):
    return np.frombuffer(lzma.decompress(x), dtype=np.int16).reshape(128, -1).T.reshape(-1, 8, 16)

def main():
    output_dir.mkdir(parents=True, exist_ok=True)
    for payload in sorted(path for path in HERE.iterdir() if path.is_file() and path.name != "decompress.py"):
        tokens = decompress_bytes(payload.read_bytes())
        with (output_dir / payload.name).open("wb") as handle:
            np.save(handle, tokens)

if __name__ == "__main__":
    main()
"""


def render_zpaq_decompress_script(*, runtime_bundle_relpath: str | None = None) -> str:
    zpaq_bin_expr = "str((HERE / %r).resolve())" % (runtime_bundle_relpath,) if runtime_bundle_relpath else '"zpaq"'
    return f"""#!/usr/bin/env python3
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
output_dir = Path(os.environ.get("OUTPUT_DIR", HERE / "compression_challenge_submission_decompressed"))
ZPAQ_BIN = {zpaq_bin_expr}

def decode_bytes(payload: bytes):
    return np.frombuffer(payload, dtype=np.int16).reshape(128, -1).T.reshape(-1, 8, 16)

def main():
    output_dir.mkdir(parents=True, exist_ok=True)
    for payload in sorted(path for path in HERE.rglob("*") if path.is_file() and path.name.endswith(".zpaq")):
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_dir = Path(tmpdir) / "extract"
            extract_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run([ZPAQ_BIN, "extract", str(payload), "-to", str(extract_dir)], check=True)
            candidates = sorted(path for path in extract_dir.rglob("*") if path.is_file())
            if not candidates:
                raise FileNotFoundError(f"no extracted file for {{payload.name}}")
            raw = candidates[0]
            rel = payload.relative_to(HERE).as_posix()[:-5]
            target = output_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                np.load(raw, allow_pickle=False)
            except Exception:
                with target.open("wb") as handle:
                    np.save(handle, decode_bytes(raw.read_bytes()))
            else:
                target.write_bytes(raw.read_bytes())

if __name__ == "__main__":
    main()
"""


def render_prev_symbol_position_major_runtime_module() -> str:
    return """from __future__ import annotations

import heapq

PREV_SYMBOL_STREAM_MAGIC = b"TPC1"
STREAM_MAGIC = b"TFC1"
UINT16_MAX = 0xFFFF
FRAME_BOS_TOKEN = 1024
SEGMENT_EOT_TOKEN = 1025
POSITIONS = 128


def _decode_varint(data: bytes, offset: int, *, label: str):
    value = 0
    shift = 0
    cursor = offset
    while True:
        if cursor >= len(data):
            raise ValueError(f"truncated {label}")
        byte = data[cursor]
        cursor += 1
        if shift == 63 and (byte & 0x7F) > 1:
            raise ValueError(f"{label} exceeds supported bounds")
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            if cursor - offset != _varint_size(value):
                raise ValueError(f"non-canonical {label}")
            return value, cursor
        shift += 7
        if shift > 63:
            raise ValueError(f"{label} exceeds supported bounds")


def _varint_size(value: int):
    size = 1
    remaining = value >> 7
    while remaining:
        size += 1
        remaining >>= 7
    return size


def _build_code_lengths(frequencies: dict[int, int]):
    if not frequencies:
        return {}
    if len(frequencies) == 1:
        symbol = next(iter(frequencies))
        return {symbol: 0}
    nodes = {}
    heap = []
    next_node_id = 0
    for symbol, count in sorted(frequencies.items()):
        nodes[next_node_id] = (None, None, symbol)
        heapq.heappush(heap, (count, symbol, next_node_id))
        next_node_id += 1
    while len(heap) > 1:
        left_count, left_min_symbol, left_id = heapq.heappop(heap)
        right_count, right_min_symbol, right_id = heapq.heappop(heap)
        parent_id = next_node_id
        next_node_id += 1
        nodes[parent_id] = (left_id, right_id, None)
        heapq.heappush(heap, (left_count + right_count, min(left_min_symbol, right_min_symbol), parent_id))
    lengths = {}
    stack = [(heap[0][2], 0)]
    while stack:
        node_id, depth = stack.pop()
        left_id, right_id, symbol = nodes[node_id]
        if symbol is not None:
            lengths[symbol] = depth
            continue
        stack.append((right_id, depth + 1))
        stack.append((left_id, depth + 1))
    return lengths


def _build_canonical_codebook(lengths: dict[int, int]):
    encode_table = {}
    positive_lengths = sorted((length, symbol) for symbol, length in lengths.items() if length > 0)
    if not positive_lengths:
        return encode_table, {}, {}, {}, (), 0
    code = 0
    previous_length = 0
    ordered_symbols = []
    first_code_by_length = {}
    first_index_by_length = {}
    count_by_length = {}
    for index, (length, symbol) in enumerate(positive_lengths):
        code <<= length - previous_length
        if length not in first_code_by_length:
            first_code_by_length[length] = code
            first_index_by_length[length] = index
        count_by_length[length] = count_by_length.get(length, 0) + 1
        encode_table[symbol] = (code, length)
        ordered_symbols.append(symbol)
        code += 1
        previous_length = length
    return encode_table, first_code_by_length, first_index_by_length, count_by_length, tuple(ordered_symbols), max(count_by_length)


def _validate_padding(payload: bytes, *, bits_consumed: int):
    if bits_consumed == len(payload) * 8:
        return
    byte_index = bits_consumed // 8
    bit_offset = bits_consumed % 8
    if bit_offset:
        trailing_mask = (1 << (8 - bit_offset)) - 1
        if payload[byte_index] & trailing_mask:
            raise ValueError("non-zero trailing padding bits")
        byte_index += 1
    if byte_index != len(payload):
        raise ValueError("trailing payload bytes")


def _decode_payload(payload: bytes, *, token_count: int, first_code_by_length, first_index_by_length, count_by_length, ordered_symbols, max_code_bits):
    restored = [0] * token_count
    produced = 0
    code = 0
    width = 0
    bits_consumed = 0
    for byte in payload:
        for shift in range(7, -1, -1):
            code = (code << 1) | ((byte >> shift) & 1)
            width += 1
            bits_consumed += 1
            first_code = first_code_by_length.get(width)
            if first_code is None:
                if width > max_code_bits:
                    raise ValueError("payload contains an invalid prefix code")
                continue
            code_offset = code - first_code
            count = count_by_length[width]
            if 0 <= code_offset < count:
                symbol_index = first_index_by_length[width] + code_offset
                restored[produced] = ordered_symbols[symbol_index]
                produced += 1
                code = 0
                width = 0
                if produced == token_count:
                    _validate_padding(payload, bits_consumed=bits_consumed)
                    return restored
    raise ValueError("truncated payload")


def decode_uint16_frequency_stream(encoded: bytes):
    if not encoded.startswith(STREAM_MAGIC):
        raise ValueError("invalid frequency stream header")
    cursor = len(STREAM_MAGIC)
    token_count, cursor = _decode_varint(encoded, cursor, label="token count")
    unique_symbols, cursor = _decode_varint(encoded, cursor, label="unique symbol count")
    payload_size, cursor = _decode_varint(encoded, cursor, label="payload size")
    frequencies = {}
    total = 0
    previous_symbol = 0
    for index in range(unique_symbols):
        delta, cursor = _decode_varint(encoded, cursor, label="frequency header")
        count, cursor = _decode_varint(encoded, cursor, label="frequency header")
        symbol = delta if index == 0 else previous_symbol + delta
        frequencies[symbol] = count
        total += count
        previous_symbol = symbol
    if total != token_count:
        raise ValueError("frequency table does not sum to token count")
    payload = encoded[cursor : cursor + payload_size]
    lengths = _build_code_lengths(frequencies)
    if len(frequencies) == 1:
        only_symbol = next(iter(frequencies))
        return [only_symbol] * token_count
    _, first_code_by_length, first_index_by_length, count_by_length, ordered_symbols, max_code_bits = _build_canonical_codebook(lengths)
    return _decode_payload(
        payload,
        token_count=token_count,
        first_code_by_length=first_code_by_length,
        first_index_by_length=first_index_by_length,
        count_by_length=count_by_length,
        ordered_symbols=ordered_symbols,
        max_code_bits=max_code_bits,
    )


def decode_uint16_prev_symbol_stream(encoded: bytes):
    if not encoded.startswith(PREV_SYMBOL_STREAM_MAGIC):
        raise ValueError("invalid prev-symbol stream header")
    cursor = len(PREV_SYMBOL_STREAM_MAGIC)
    token_count, cursor = _decode_varint(encoded, cursor, label="token count")
    if token_count == 0:
        context_count, cursor = _decode_varint(encoded, cursor, label="context count")
        if context_count != 0 or cursor != len(encoded):
            raise ValueError("invalid empty prev-symbol stream")
        return []
    first_symbol, cursor = _decode_varint(encoded, cursor, label="first symbol")
    context_count, cursor = _decode_varint(encoded, cursor, label="context count")
    context_sizes = {}
    previous_symbol = 0
    for index in range(context_count):
        delta, cursor = _decode_varint(encoded, cursor, label="context header")
        payload_size, cursor = _decode_varint(encoded, cursor, label="context header")
        symbol = delta if index == 0 else previous_symbol + delta
        context_sizes[symbol] = payload_size
        previous_symbol = symbol
    payload_cursor = cursor
    decoded_contexts = {}
    for symbol, payload_size in context_sizes.items():
        end = payload_cursor + payload_size
        decoded_contexts[symbol] = decode_uint16_frequency_stream(encoded[payload_cursor:end])
        payload_cursor = end
    restored = [first_symbol]
    context_offsets = {symbol: 0 for symbol in decoded_contexts}
    for _ in range(1, token_count):
        previous = restored[-1]
        values = decoded_contexts[previous]
        offset = context_offsets[previous]
        restored.append(values[offset])
        context_offsets[previous] = offset + 1
    return restored


def inflate_prev_symbol_position_major(encoded: bytes):
    import numpy as np

    flat = np.asarray(decode_uint16_prev_symbol_stream(encoded), dtype=np.uint16)
    if flat.size == 0 or int(flat[-1]) != SEGMENT_EOT_TOKEN:
        raise ValueError("position-major payload must end with segment EOT")
    body = flat[:-1]
    if body.size % POSITIONS != 0:
        raise ValueError("position-major payload body must be divisible by 128")
    stream_len = body.size // POSITIONS
    if stream_len <= 1:
        raise ValueError("position-major payload must contain at least one frame")
    streams = body.reshape(POSITIONS, stream_len)
    if not np.all(streams[:, 0] == FRAME_BOS_TOKEN):
        raise ValueError("position-major payload is missing BOS headers")
    frames = streams[:, 1:].T
    return frames.reshape(frames.shape[0], 8, 16).astype(np.int16, copy=False)
"""


def render_prev_symbol_position_major_decompress_script() -> str:
    return """#!/usr/bin/env python3
import os
from pathlib import Path

import numpy as np

from _lossless_prev_symbol_runtime import inflate_prev_symbol_position_major

HERE = Path(__file__).resolve().parent
output_dir = Path(os.environ.get("OUTPUT_DIR", HERE / "compression_challenge_submission_decompressed"))

def main():
    output_dir.mkdir(parents=True, exist_ok=True)
    for payload in sorted(path for path in HERE.rglob("*") if path.is_file() and path.name.endswith(".tpc")):
        tokens = inflate_prev_symbol_position_major(payload.read_bytes())
        rel = payload.relative_to(HERE).as_posix()[:-4]
        target = output_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            np.save(handle, tokens)

if __name__ == "__main__":
    main()
"""


def render_global_prev_symbol_position_major_runtime_module() -> str:
    return """from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import heapq

PREV_SYMBOL_STREAM_MAGIC = b"TPC1"
STREAM_MAGIC = b"TFC1"
UINT16_MAX = 0xFFFF


def _decode_varint(data: bytes, offset: int, *, label: str):
    value = 0
    shift = 0
    cursor = offset
    while True:
        if cursor >= len(data):
            raise ValueError(f"truncated {label}")
        byte = data[cursor]
        cursor += 1
        if shift == 63 and (byte & 0x7F) > 1:
            raise ValueError(f"{label} exceeds supported bounds")
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            if cursor - offset != _varint_size(value):
                raise ValueError(f"non-canonical {label}")
            return value, cursor
        shift += 7
        if shift > 63:
            raise ValueError(f"{label} exceeds supported bounds")


def _varint_size(value: int):
    size = 1
    remaining = value >> 7
    while remaining:
        size += 1
        remaining >>= 7
    return size


def _build_code_lengths(frequencies: dict[int, int]):
    if not frequencies:
        return {}
    if len(frequencies) == 1:
        symbol = next(iter(frequencies))
        return {symbol: 0}
    nodes = {}
    heap = []
    next_node_id = 0
    for symbol, count in sorted(frequencies.items()):
        nodes[next_node_id] = (None, None, symbol)
        heapq.heappush(heap, (count, symbol, next_node_id))
        next_node_id += 1
    while len(heap) > 1:
        left_count, left_min_symbol, left_id = heapq.heappop(heap)
        right_count, right_min_symbol, right_id = heapq.heappop(heap)
        parent_id = next_node_id
        next_node_id += 1
        nodes[parent_id] = (left_id, right_id, None)
        heapq.heappush(heap, (left_count + right_count, min(left_min_symbol, right_min_symbol), parent_id))
    lengths = {}
    stack = [(heap[0][2], 0)]
    while stack:
        node_id, depth = stack.pop()
        left_id, right_id, symbol = nodes[node_id]
        if symbol is not None:
            lengths[symbol] = depth
            continue
        stack.append((right_id, depth + 1))
        stack.append((left_id, depth + 1))
    return lengths


def _build_canonical_codebook(lengths: dict[int, int]):
    positive_lengths = sorted((length, symbol) for symbol, length in lengths.items() if length > 0)
    if not positive_lengths:
        return {}, {}, {}, (), 0
    code = 0
    previous_length = 0
    first_code_by_length = {}
    first_index_by_length = {}
    count_by_length = {}
    ordered_symbols = []
    for index, (length, symbol) in enumerate(positive_lengths):
        code <<= length - previous_length
        if length not in first_code_by_length:
            first_code_by_length[length] = code
            first_index_by_length[length] = index
        count_by_length[length] = count_by_length.get(length, 0) + 1
        ordered_symbols.append(symbol)
        code += 1
        previous_length = length
    return first_code_by_length, first_index_by_length, count_by_length, tuple(ordered_symbols), max(count_by_length)


def _validate_padding(payload: bytes, *, bits_consumed: int):
    if bits_consumed == len(payload) * 8:
        return
    byte_index = bits_consumed // 8
    bit_offset = bits_consumed % 8
    if bit_offset:
        trailing_mask = (1 << (8 - bit_offset)) - 1
        if payload[byte_index] & trailing_mask:
            raise ValueError("non-zero trailing padding bits")
        byte_index += 1
    if byte_index != len(payload):
        raise ValueError("trailing payload bytes")


def decode_uint16_frequency_stream(encoded: bytes):
    if not encoded.startswith(STREAM_MAGIC):
        raise ValueError("invalid frequency stream header")
    cursor = len(STREAM_MAGIC)
    token_count, cursor = _decode_varint(encoded, cursor, label="token count")
    unique_symbols, cursor = _decode_varint(encoded, cursor, label="unique symbol count")
    payload_size, cursor = _decode_varint(encoded, cursor, label="payload size")
    frequencies = {}
    total = 0
    previous_symbol = 0
    for index in range(unique_symbols):
        delta, cursor = _decode_varint(encoded, cursor, label="frequency header")
        count, cursor = _decode_varint(encoded, cursor, label="frequency header")
        symbol = delta if index == 0 else previous_symbol + delta
        frequencies[symbol] = count
        total += count
        previous_symbol = symbol
    if total != token_count:
        raise ValueError("frequency table does not sum to token count")
    payload = encoded[cursor : cursor + payload_size]
    if len(frequencies) == 1:
        return [next(iter(frequencies))] * token_count
    lengths = _build_code_lengths(frequencies)
    first_code_by_length, first_index_by_length, count_by_length, ordered_symbols, max_code_bits = _build_canonical_codebook(lengths)
    restored = [0] * token_count
    produced = 0
    code = 0
    width = 0
    bits_consumed = 0
    for byte in payload:
        for shift in range(7, -1, -1):
            code = (code << 1) | ((byte >> shift) & 1)
            width += 1
            bits_consumed += 1
            first_code = first_code_by_length.get(width)
            if first_code is None:
                if width > max_code_bits:
                    raise ValueError("payload contains an invalid prefix code")
                continue
            code_offset = code - first_code
            count = count_by_length[width]
            if 0 <= code_offset < count:
                restored[produced] = ordered_symbols[first_index_by_length[width] + code_offset]
                produced += 1
                code = 0
                width = 0
                if produced == token_count:
                    _validate_padding(payload, bits_consumed=bits_consumed)
                    return restored
    raise ValueError("truncated payload")


def decode_uint16_prev_symbol_stream(encoded: bytes):
    if not encoded.startswith(PREV_SYMBOL_STREAM_MAGIC):
        raise ValueError("invalid prev-symbol stream header")
    cursor = len(PREV_SYMBOL_STREAM_MAGIC)
    token_count, cursor = _decode_varint(encoded, cursor, label="token count")
    if token_count == 0:
        context_count, cursor = _decode_varint(encoded, cursor, label="context count")
        if context_count != 0 or cursor != len(encoded):
            raise ValueError("invalid empty prev-symbol stream")
        return []
    first_symbol, cursor = _decode_varint(encoded, cursor, label="first symbol")
    context_count, cursor = _decode_varint(encoded, cursor, label="context count")
    context_sizes = {}
    previous_symbol = 0
    for index in range(context_count):
        delta, cursor = _decode_varint(encoded, cursor, label="context header")
        payload_size, cursor = _decode_varint(encoded, cursor, label="context header")
        symbol = delta if index == 0 else previous_symbol + delta
        context_sizes[symbol] = payload_size
        previous_symbol = symbol
    payload_cursor = cursor
    decoded_contexts = {}
    for symbol, payload_size in context_sizes.items():
        end = payload_cursor + payload_size
        decoded_contexts[symbol] = decode_uint16_frequency_stream(encoded[payload_cursor:end])
        payload_cursor = end
    restored = [first_symbol]
    context_offsets = {symbol: 0 for symbol in decoded_contexts}
    for _ in range(1, token_count):
        previous = restored[-1]
        values = decoded_contexts[previous]
        offset = context_offsets[previous]
        restored.append(values[offset])
        context_offsets[previous] = offset + 1
    return restored


def decode_corpus_global_prev_symbol_position_major(*, encoded_dir: str | Path, output_dir: str | Path):
    root = Path(encoded_dir)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    decoded_chunks = {}
    chunk_offsets = {}

    for record in manifest["records"]:
        chunk_index = int(record["chunk_index"])
        if chunk_index not in decoded_chunks:
            decoded_chunks[chunk_index] = np.asarray(
                decode_uint16_prev_symbol_stream((root / f"chunk_{chunk_index:03d}.tpc").read_bytes()),
                dtype=np.uint16,
            )
            chunk_offsets[chunk_index] = 0
        decoded = decoded_chunks[chunk_index]
        offset = chunk_offsets[chunk_index]
        token_count = int(record["token_count"])
        payload = decoded[offset : offset + token_count]
        chunk_offsets[chunk_index] = offset + token_count
        body = payload[:-1].reshape(128, -1)
        frames = body[:, 1:].T.reshape(-1, 8, 16).astype(np.int16)
        with (target / str(record["file_name"])).open("wb") as handle:
            np.save(handle, frames)
"""


def render_global_prev_symbol_position_major_decompress_script() -> str:
    return """#!/usr/bin/env python3
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path.cwd()))
ENCODED_DIR = HERE if (HERE / "manifest.json").exists() else Path.cwd()

from _lossless_global_prev_symbol_runtime import decode_corpus_global_prev_symbol_position_major

output_dir = Path(os.environ.get("OUTPUT_DIR", HERE / "compression_challenge_submission_decompressed"))

def main():
    decode_corpus_global_prev_symbol_position_major(encoded_dir=ENCODED_DIR, output_dir=output_dir)

if __name__ == "__main__":
    main()
"""


def _build_baseline_submission(
    *,
    profile: str,
    split="challenge",
    work_dir: str | Path,
    dataset_loader=None,
    num_proc: int | None = None,
    runtime_bundle_path: str | Path | None = None,
) -> dict[str, object]:
    root = Path(work_dir)
    payload_dir = root / "payload"
    payload_dir.mkdir(parents=True, exist_ok=True)
    effective_num_proc = _effective_num_proc(num_proc)
    dataset = load_commavq_dataset(split=split, dataset_loader=dataset_loader, num_proc=effective_num_proc)
    decompress_path = root / "decompress.py"
    method = _profile_method(profile)
    runtime_metadata = _runtime_metadata(method=method, runtime_bundle_path=runtime_bundle_path)
    split_dataset = _dataset_train_split(dataset)
    if method == "global_prev_symbol_position_major":
        records = build_token_records(split_dataset)
        corpus_result = encode_corpus_global_prev_symbol_position_major(
            records=records,
            output_dir=payload_dir,
            chunk_count=int(_profile_config(profile).get("chunk_count", 1)),
        )
        payload_bytes = sum(path.stat().st_size for path in payload_dir.glob("chunk_*.tpc"))
        record_count = corpus_result["record_count"]
        (payload_dir / "_lossless_global_prev_symbol_runtime.py").write_text(
            render_global_prev_symbol_position_major_runtime_module()
        )
        decompress_path.write_text(render_global_prev_symbol_position_major_decompress_script())
    else:
        payload_bytes, record_count = _compress_dataset_split(
            profile=profile,
            split_dataset=split_dataset,
            payload_dir=payload_dir,
            num_proc=effective_num_proc,
        )
        _bundle_runtime_if_needed(
            payload_dir=payload_dir,
            runtime_metadata=runtime_metadata,
            runtime_bundle_path=runtime_bundle_path,
        )
        if method == "lzma":
            decompress_path.write_text(render_lzma_decompress_script())
        elif method == "zpaq":
            decompress_path.write_text(
                render_zpaq_decompress_script(
                    runtime_bundle_relpath=runtime_metadata.get("runtime_bundle_relpath"),
                )
            )
        else:
            (payload_dir / "_lossless_prev_symbol_runtime.py").write_text(
                render_prev_symbol_position_major_runtime_module()
            )
            decompress_path.write_text(render_prev_symbol_position_major_decompress_script())
    archive_path = root / f"{profile}_submission.zip"
    build_submission_zip(payload_dir=payload_dir, decompress_path=decompress_path, output_path=archive_path)
    archive_bytes = archive_path.stat().st_size
    result = {
        "profile": profile,
        "method": method,
        "archive_path": str(archive_path),
        "archive_bytes": archive_bytes,
        "payload_bytes": payload_bytes,
        "record_count": record_count,
        "payload_dir": str(payload_dir),
    }
    result.update(runtime_metadata)
    return result


def build_lzma_baseline_submission(
    *,
    profile: str = "lzma_baseline",
    split="challenge",
    work_dir: str | Path,
    dataset_loader=None,
    num_proc: int | None = None,
) -> dict[str, object]:
    return _build_baseline_submission(
        profile=profile,
        split=split,
        work_dir=work_dir,
        dataset_loader=dataset_loader,
        num_proc=num_proc,
    )


def build_zpaq_baseline_submission(
    *,
    profile: str = "zpaq_baseline",
    split="challenge",
    work_dir: str | Path,
    dataset_loader=None,
    num_proc: int | None = None,
    runtime_bundle_path: str | Path | None = None,
) -> dict[str, object]:
    return _build_baseline_submission(
        profile=profile,
        split=split,
        work_dir=work_dir,
        dataset_loader=dataset_loader,
        num_proc=num_proc,
        runtime_bundle_path=runtime_bundle_path,
    )


def evaluate_lossless_baseline_submission(
    *,
    profile: str,
    split="challenge",
    work_dir: str | Path,
    dataset_loader=None,
    num_proc: int | None = None,
    runtime_bundle_path: str | Path | None = None,
) -> dict[str, object]:
    return _evaluate_baseline_submission(
        profile=profile,
        split=split,
        work_dir=work_dir,
        dataset_loader=dataset_loader,
        num_proc=num_proc,
        runtime_bundle_path=runtime_bundle_path,
    )


def _evaluate_baseline_submission(
    *,
    profile: str,
    split="challenge",
    work_dir: str | Path,
    dataset_loader=None,
    num_proc: int | None = None,
    runtime_bundle_path: str | Path | None = None,
) -> dict[str, object]:
    root = Path(work_dir)
    baseline = _build_baseline_submission(
        profile=profile,
        split=split,
        work_dir=root,
        dataset_loader=dataset_loader,
        num_proc=num_proc,
        runtime_bundle_path=runtime_bundle_path,
    )
    compression, verification, decompressed_dir, _cleanup_fn = evaluate_local_submission_contract(
        profile=profile,
        archive_path=baseline["archive_path"],
        method=str(baseline["method"]),
        split=split,
        work_dir=root / "contract_eval",
        dataset_loader=dataset_loader,
        num_proc=num_proc,
    )
    return {
        "command": "lossless_baseline_evaluate",
        "baseline": baseline,
        "compression": compression.__dict__,
        "verification": verification.__dict__,
        "decompressed_root": str(decompressed_dir),
        "local_only": baseline["local_only"],
        "challenge_valid": baseline["challenge_valid"],
        "runtime_bundle_included": baseline["runtime_bundle_included"],
        "runtime_bundle_relpath": baseline["runtime_bundle_relpath"],
        "challenge_validity_reason": baseline["challenge_validity_reason"],
    }


def evaluate_lzma_baseline_submission(
    *,
    split="challenge",
    work_dir: str | Path,
    dataset_loader=None,
    num_proc: int | None = None,
) -> dict[str, object]:
    return _evaluate_baseline_submission(
        profile="lzma_baseline",
        split=split,
        work_dir=work_dir,
        dataset_loader=dataset_loader,
        num_proc=num_proc,
    )


def evaluate_zpaq_baseline_submission(
    *,
    split="challenge",
    work_dir: str | Path,
    dataset_loader=None,
    num_proc: int | None = None,
    runtime_bundle_path: str | Path | None = None,
) -> dict[str, object]:
    return _evaluate_baseline_submission(
        profile="zpaq_baseline",
        split=split,
        work_dir=work_dir,
        dataset_loader=dataset_loader,
        num_proc=num_proc,
        runtime_bundle_path=runtime_bundle_path,
    )


def lzma_roundtrip_file(
    *, source_path: str | Path, compressed_path: str | Path, restored_path: str | Path
) -> dict[str, object]:
    compression = compress_lossless_file(
        profile="lzma_baseline",
        input_path=source_path,
        output_path=compressed_path,
    )
    restored = decompress_lossless_file(
        profile="lzma_baseline",
        archive_path=compressed_path,
        output_path=restored_path,
    )
    return {
        "profile": compression.profile,
        "method": compression.method,
        "archive_path": compression.archive_path,
        "archive_bytes": compression.archive_bytes,
        "original_bytes": compression.original_bytes,
        "restored_path": str(restored),
    }


def zpaq_roundtrip_file(
    *, source_path: str | Path, compressed_path: str | Path, restored_path: str | Path
) -> dict[str, object]:
    compression = compress_lossless_file(
        profile="zpaq_baseline",
        input_path=source_path,
        output_path=compressed_path,
    )
    restored = decompress_lossless_file(
        profile="zpaq_baseline",
        archive_path=compressed_path,
        output_path=restored_path,
    )
    return {
        "profile": compression.profile,
        "method": compression.method,
        "archive_path": compression.archive_path,
        "archive_bytes": compression.archive_bytes,
        "original_bytes": compression.original_bytes,
        "restored_path": str(restored),
    }
