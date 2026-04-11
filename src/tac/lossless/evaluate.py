from __future__ import annotations

import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path

from .contracts import LosslessCompressionResult, LosslessVerificationResult
from .data import (
    TokenRecord,
    TokenSource,
    build_token_records,
    commavq_original_bytes,
    load_commavq_dataset,
    load_token_file,
    token_bytes,
)


def verify_exact_tokens(original_tokens: TokenSource, decompressed_tokens: TokenSource) -> LosslessVerificationResult:
    import numpy as np

    original = token_bytes(original_tokens)
    decoded = token_bytes(decompressed_tokens)

    # Fast path via numpy for large byte sequences
    min_len = min(len(original), len(decoded))
    if min_len > 0:
        orig_arr = np.frombuffer(original[:min_len], dtype=np.uint8)
        dec_arr = np.frombuffer(decoded[:min_len], dtype=np.uint8)
        mismatch_count = int(np.count_nonzero(orig_arr != dec_arr))
    else:
        mismatch_count = 0
    mismatch_count += abs(len(original) - len(decoded))
    return LosslessVerificationResult(
        exact_match=mismatch_count == 0,
        checked_items=len(original),
        mismatch_count=mismatch_count,
    )


def verify_exact_token_files(
    reference_records: list[TokenRecord],
    decompressed_root: str | Path,
) -> LosslessVerificationResult:
    import numpy as np

    root = Path(decompressed_root)
    mismatch_count = 0
    for record in reference_records:
        candidate = root / record.file_name
        if not candidate.is_file():
            mismatch_count += 1
            continue
        actual = load_token_file(candidate)
        if not np.array_equal(actual, record.tokens):
            mismatch_count += 1
    return LosslessVerificationResult(
        exact_match=mismatch_count == 0,
        checked_items=len(reference_records),
        mismatch_count=mismatch_count,
    )


def compression_rate(archive_bytes: int, original_bytes: int) -> float:
    if archive_bytes <= 0:
        raise ValueError("archive_bytes must be positive")
    if original_bytes <= 0:
        raise ValueError("original_bytes must be positive")
    return original_bytes / archive_bytes


def _resolve_archive_bytes(archive_path: str | Path, archive_bytes: int | None) -> int:
    if archive_bytes is not None:
        if archive_bytes <= 0:
            raise ValueError("archive_bytes must be positive")
        return archive_bytes
    return Path(archive_path).stat().st_size


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


def _compare_dataset_example(example: Mapping[str, object], *, decompressed_root: str) -> dict[str, int]:
    import numpy as np

    meta = example.get("json")
    if not isinstance(meta, Mapping):
        raise ValueError("dataset example is missing json metadata")
    file_name = meta.get("file_name")
    if not isinstance(file_name, str) or not file_name.strip():
        raise ValueError("dataset example is missing json.file_name")
    if "token.npy" not in example:
        raise ValueError(f"dataset example {file_name!r} is missing token.npy")

    candidate = Path(decompressed_root) / file_name
    if not candidate.is_file():
        return {"mismatch_count": 1}

    actual = load_token_file(candidate)
    expected = example["token.npy"]
    return {"mismatch_count": 0 if np.array_equal(actual, expected) else 1}


def _summarize_compare_rows(rows: object) -> tuple[int, int]:
    if isinstance(rows, Mapping):
        if "mismatch_count" in rows:
            values = rows["mismatch_count"]
        else:
            raise ValueError("dataset compare output did not include mismatch_count")
    elif hasattr(rows, "__getitem__"):
        try:
            values = rows["mismatch_count"]
        except Exception:
            values = rows
    else:
        values = rows

    if not isinstance(values, Iterable) or isinstance(values, (str, bytes, bytearray)):
        raise ValueError("dataset compare output did not include iterable mismatch metadata")

    checked_items = 0
    mismatch_count = 0
    for item in values:
        checked_items += 1
        if isinstance(item, Mapping):
            mismatch_count += int(item["mismatch_count"])
        else:
            mismatch_count += int(item)
    return checked_items, mismatch_count


def evaluate_lossless_archive(
    *,
    profile: str,
    archive_path: str | Path,
    method: str,
    original_tokens: TokenSource | None = None,
    decompressed_tokens: TokenSource | None = None,
    archive_bytes: int | None = None,
    reference_records: list[TokenRecord] | None = None,
    decompressed_root: str | Path | None = None,
    original_bytes: int | None = None,
) -> tuple[LosslessCompressionResult, LosslessVerificationResult]:
    resolved_archive_bytes = _resolve_archive_bytes(archive_path, archive_bytes)

    if reference_records is not None or decompressed_root is not None:
        if reference_records is None or decompressed_root is None:
            raise ValueError("reference_records and decompressed_root must be provided together")
        verification = verify_exact_token_files(reference_records, decompressed_root)
        resolved_original_bytes = (
            original_bytes if original_bytes is not None else commavq_original_bytes(len(reference_records))
        )
    else:
        if original_tokens is None or decompressed_tokens is None:
            raise ValueError("either token inputs or file-based reference inputs must be provided")
        original = token_bytes(original_tokens)
        verification = verify_exact_tokens(original, decompressed_tokens)
        resolved_original_bytes = original_bytes if original_bytes is not None else len(original)

    if not verification.exact_match:
        raise ValueError(
            f"Lossless evaluation requires an exact round-trip; found {verification.mismatch_count} mismatched items"
        )

    result = LosslessCompressionResult(
        profile=profile,
        archive_path=str(archive_path),
        archive_bytes=resolved_archive_bytes,
        original_bytes=resolved_original_bytes,
        compression_rate=compression_rate(resolved_archive_bytes, resolved_original_bytes),
        method=method,
        record_count=verification.checked_items,
        checked_items=verification.checked_items,
    )
    return result, verification


def evaluate_commavq_dataset_archive(
    *,
    profile: str,
    archive_path: str | Path,
    decompressed_root: str | Path,
    method: str,
    split="challenge",
    dataset_loader=None,
    dataset_name: str = "commaai/commavq",
    num_proc: int | None = None,
    archive_bytes: int | None = None,
    original_bytes: int | None = None,
) -> tuple[LosslessCompressionResult, LosslessVerificationResult]:
    dataset = load_commavq_dataset(
        split=split,
        dataset_loader=dataset_loader,
        dataset_name=dataset_name,
        num_proc=_effective_num_proc(num_proc),
    )
    train_split = _dataset_train_split(dataset)

    if hasattr(train_split, "map"):
        mapped = train_split.map(
            _compare_dataset_example,
            desc="compare",
            num_proc=_effective_num_proc(num_proc),
            load_from_cache_file=False,
            fn_kwargs={"decompressed_root": str(decompressed_root)},
        )
        checked_items, mismatch_count = _summarize_compare_rows(mapped)
        verification = LosslessVerificationResult(
            exact_match=mismatch_count == 0,
            checked_items=checked_items,
            mismatch_count=mismatch_count,
        )
        if not verification.exact_match:
            raise ValueError(
                f"Lossless evaluation requires an exact round-trip; found {verification.mismatch_count} mismatched items"
            )
        resolved_archive_bytes = _resolve_archive_bytes(archive_path, archive_bytes)
        resolved_original_bytes = (
            original_bytes if original_bytes is not None else commavq_original_bytes(checked_items)
        )
        result = LosslessCompressionResult(
            profile=profile,
            archive_path=str(archive_path),
            archive_bytes=resolved_archive_bytes,
            original_bytes=resolved_original_bytes,
            compression_rate=compression_rate(resolved_archive_bytes, resolved_original_bytes),
            method=method,
            record_count=verification.checked_items,
            checked_items=verification.checked_items,
        )
        return result, verification

    reference_records = build_token_records(train_split)
    return evaluate_lossless_archive(
        profile=profile,
        archive_path=archive_path,
        method=method,
        archive_bytes=archive_bytes,
        original_bytes=original_bytes,
        reference_records=reference_records,
        decompressed_root=decompressed_root,
    )


def evaluate_local_submission_contract(
    *,
    profile: str,
    archive_path: str | Path,
    method: str,
    split="challenge",
    work_dir: str | Path | None = None,
    dataset_loader=None,
    dataset_name: str = "commaai/commavq",
    num_proc: int | None = None,
    python_executable: str | None = None,
) -> tuple[LosslessCompressionResult, LosslessVerificationResult, Path, "typing.Callable[[], None] | None"]:
    """Evaluate a submission archive against the commavq dataset.

    Returns (result, verification, decompressed_root, cleanup_fn).
    When *work_dir* is ``None`` the caller **must** invoke *cleanup_fn()*
    after it is done with *decompressed_root* to remove the temporary
    directory.  When *work_dir* is provided, *cleanup_fn* is ``None``.
    """
    import typing

    archive = Path(archive_path).resolve()
    interpreter = python_executable or sys.executable

    if work_dir is None:
        tmpdir = tempfile.TemporaryDirectory(prefix="tac-lossless-eval-")
        root = Path(tmpdir.name)
    else:
        tmpdir = None
        root = Path(work_dir).resolve()
        if root.exists():
            # Trust boundary: work_dir is caller-supplied. shutil.rmtree will
            # recursively delete whatever path is given. Callers must ensure
            # work_dir points to a disposable scratch directory, not a
            # sensitive location.
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)

    try:
        unpacked_root = root / "submission"
        decompressed_root = root / "decompressed"
        unpacked_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "r") as zf:
            # Mitigate zip-slip (path traversal) for Python < 3.12 which does
            # not enforce this by default in extractall.
            for member in zf.namelist():
                if ".." in member or member.startswith("/"):
                    raise ValueError(f"Suspicious zip member rejected (path traversal): {member}")
            zf.extractall(unpacked_root)

        decompress_script = unpacked_root / "decompress.py"
        if not decompress_script.is_file():
            raise FileNotFoundError(f"submission archive does not contain decompress.py: {archive}")

        env = {
            **os.environ,
            "OUTPUT_DIR": str(decompressed_root),
        }
        subprocess.run(
            [interpreter, str(decompress_script)],
            cwd=str(unpacked_root),
            env=env,
            check=True,
        )

        compression, verification = evaluate_commavq_dataset_archive(
            profile=profile,
            archive_path=archive,
            decompressed_root=decompressed_root,
            method=method,
            split=split,
            dataset_loader=dataset_loader,
            dataset_name=dataset_name,
            num_proc=num_proc,
        )
        promoted_result = LosslessCompressionResult(
            profile=compression.profile,
            archive_path=compression.archive_path,
            archive_bytes=compression.archive_bytes,
            original_bytes=compression.original_bytes,
            compression_rate=compression.compression_rate,
            method=compression.method,
            payload_bytes=compression.payload_bytes,
            record_count=compression.record_count,
            checked_items=compression.checked_items,
            split=tuple(split) if isinstance(split, (list, tuple)) else (str(split),),
            evidence_root=str(root),
        )
        cleanup_fn = tmpdir.cleanup if tmpdir is not None else None
        return promoted_result, verification, decompressed_root, cleanup_fn
    except BaseException:
        if tmpdir is not None:
            tmpdir.cleanup()
        raise
