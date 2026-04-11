from __future__ import annotations

import lzma
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import multiprocessing
from collections.abc import Iterable, Mapping

from .contracts import LosslessCompressionResult
from .data import TokenRecord, load_commavq_dataset, load_commavq_reference_records
from .evaluate import compression_rate, evaluate_local_submission_contract
from .profiles import PROFILES
from .submission import build_submission_zip


_FIXED_ZPAQ_MTIME = 1704067200  # 2024-01-01T00:00:00Z
_ZPAQ_SUFFIX = ".zpaq"
_ZPAQ_PRESET_TO_METHOD = {
    "fast": "1",
    "normal": "3",
    "better": "4",
    "max": "5",
}
_LOCAL_ONLY_ZPAQ_REASON = (
    "zpaq is local-only unless a self-contained runtime is bundled in the submission payload"
)


def _profile_config(profile: str) -> dict[str, object]:
    try:
        config = PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"unknown lossless profile: {profile}") from exc
    method = str(config.get("method", ""))
    if method not in {"lzma", "zpaq"}:
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
    if archive_name.endswith(_ZPAQ_SUFFIX):
        return archive_name[: -len(_ZPAQ_SUFFIX)]
    return archive_name


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
        _run_external(
            [binary, "add", str(output_path), source_name, "-method", _zpaq_method_arg(config)],
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


def compress_lossless_file(*, profile: str, input_path: str | Path, output_path: str | Path) -> LosslessCompressionResult:
    config = _profile_config(profile)
    method = str(config["method"])
    source = Path(input_path)
    target = Path(output_path)
    payload = source.read_bytes()

    if method == "lzma":
        compressed = lzma.compress(payload, preset=int(config.get("level", 6)))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(compressed)
    else:
        _compress_with_zpaq(source_name=source.name, payload=payload, output_path=target, config=config)

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
    config = _profile_config(profile)
    method = str(config["method"])
    source = Path(archive_path)
    target = Path(output_path)

    if method == "lzma":
        payload = lzma.decompress(source.read_bytes())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return target

    return _decompress_with_zpaq(archive_path=source, output_path=target)


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
    meta = example.get("json")
    if not isinstance(meta, Mapping):
        raise ValueError("dataset example is missing json metadata")
    file_name = meta.get("file_name")
    if not isinstance(file_name, str) or not file_name.strip():
        raise ValueError("dataset example is missing json.file_name")
    if "token.npy" not in example:
        raise ValueError(f"dataset example {file_name!r} is missing token.npy")

    payload = _encode_commavq_tokens(example["token.npy"])
    output_root = Path(payload_dir)
    method = _profile_method(profile)
    config = _profile_config(profile)

    if method == "lzma":
        compressed = lzma.compress(payload, preset=int(config.get("level", 6)))
        target = output_root / file_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(compressed)
        archive_bytes = target.stat().st_size
    else:
        target = _zpaq_output_path(output_root, file_name)
        archive_bytes = _compress_with_zpaq(
            source_name=file_name,
            payload=payload,
            output_path=target,
            config=config,
        )
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

    for compressed in sorted(path for path in source_root.rglob("*") if path.is_file() and path.name != "decompress.py"):
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
        with tempfile.TemporaryDirectory() as tmpdir:
            extracted_dir = Path(tmpdir) / "extract"
            extracted_dir.mkdir(parents=True, exist_ok=True)
            _run_external(
                [_require_zpaq_binary(), "extract", str(compressed), "-to", str(extracted_dir)],
                cwd=extracted_dir,
            )
            extracted = _extract_single_file_from_dir(extracted_dir, preferred_name=Path(decoded_name).name)
        target = target_root / decoded_name
        target.parent.mkdir(parents=True, exist_ok=True)
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
    payload_bytes, record_count = _compress_dataset_split(
        profile=profile,
        split_dataset=_dataset_train_split(dataset),
        payload_dir=payload_dir,
        num_proc=effective_num_proc,
    )
    decompress_path = root / "decompress.py"
    method = _profile_method(profile)
    runtime_metadata = _runtime_metadata(method=method, runtime_bundle_path=runtime_bundle_path)
    _bundle_runtime_if_needed(
        payload_dir=payload_dir,
        runtime_metadata=runtime_metadata,
        runtime_bundle_path=runtime_bundle_path,
    )
    decompress_path.write_text(
        render_lzma_decompress_script()
        if method == "lzma"
        else render_zpaq_decompress_script(
            runtime_bundle_relpath=runtime_metadata.get("runtime_bundle_relpath"),
        )
    )
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
    compression, verification, decompressed_dir = evaluate_local_submission_contract(
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


def lzma_roundtrip_file(*, source_path: str | Path, compressed_path: str | Path, restored_path: str | Path) -> dict[str, object]:
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


def zpaq_roundtrip_file(*, source_path: str | Path, compressed_path: str | Path, restored_path: str | Path) -> dict[str, object]:
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
