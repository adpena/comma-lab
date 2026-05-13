#!/usr/bin/env python3
"""Materialize a byte-closed PR106 + SIREN residual sidecar candidate.

See ``submissions/pr106_siren_residual_sidecar/inflate.py`` for the wire format.
SIREN's coordinate-MLP with sinusoidal activations is canonically encoded
here as a SPARSE FREQUENCY-DOMAIN coefficient set. Each coefficient is a
tuple: (frame_idx u16, k_row i16, k_col i16, channel u8, real i8, imag i8) = 9B.

This is the smallest-byte SIREN-compatible residual representation: the
inflate runtime places each coef into the 2D-FFT spectrum then inverse-FFTs
per frame per channel.

Default mode='empty' emits empty residual (scaffold-readiness archive).
"""
from __future__ import annotations

import argparse
import heapq
import struct
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.residual_basis.pr106_materializer_helpers import (  # noqa: E402
    DEFAULT_PR106_ARCHIVE,
    MaterializerError,
    materialize_family_archive,
    repack_dense_as_sparse,
    run_no_op_detector_byte_mutation,
    sha256_file,
)
from tac.residual_basis.pr106_sidecar_packing import (  # noqa: E402
    PR106_RESIDUAL_FORMAT_IDS,
    sparse_family_name,
)

PER_COEF_BYTES = 9  # u16 + i16 + i16 + u8 + i8 + i8
CAMERA_H, CAMERA_W = 874, 1164
RGB_CHANNELS = 3


def build_siren_residual_blob(
    *, n_coefs: int, n_frames: int, mode: str, default_scale: float = 0.0, max_k: int = 32
) -> bytes:
    """Build a sparse-FFT-coefs blob. Each coef is 9B; header = 4B scale + 2B count = 6B.

    mode='zero': all-zero coefs (identity bolt-on; scale also zero).
    mode='probe': low-frequency probe coefs at (k_row=0, k_col in [1..n_coefs], frame 0, channel 0, real=1, imag=0).
    """
    if mode not in ("zero", "probe"):
        raise MaterializerError(f"unknown residual mode: {mode!r}")
    if n_coefs < 0:
        raise MaterializerError(f"n_coefs must be >= 0; got {n_coefs}")
    if n_coefs > 0xFFFF:
        raise MaterializerError(f"n_coefs={n_coefs} exceeds 16-bit cap 65535")
    header = struct.pack("<fH", default_scale, n_coefs)
    coefs: list[bytes] = []
    for i in range(n_coefs):
        frame_idx = i % max(n_frames, 1)
        k_row = 0
        k_col = (i % max_k) + 1
        channel = 0
        if mode == "zero":
            real_q, imag_q = 0, 0
        else:  # probe
            real_q, imag_q = 1, 0
        coefs.append(
            struct.pack("<HhhBbb", frame_idx, k_row, k_col, channel, real_q, imag_q)
        )
    return header + b"".join(coefs)


def _signed_fft_indices(size: int) -> tuple[int, ...]:
    return tuple(i if i <= size // 2 else i - size for i in range(size))


def _encode_siren_coefficients(
    selected: list[tuple[float, int, int, int, int, complex]],
    *,
    n_coefs: int,
) -> bytes:
    """Encode the first ``n_coefs`` selected FFT coefficients in dense SIREN wire format."""

    if n_coefs <= 0:
        return b""
    chosen = selected[:n_coefs]
    max_component = max(
        max(abs(float(value.real)), abs(float(value.imag)))
        for *_prefix, value in chosen
    )
    if max_component == 0.0:
        return b""
    scale = max_component / 127.0
    records: list[bytes] = []
    for _mag, frame_idx, channel, k_row, k_col, value in chosen:
        real_q = round(float(value.real) / scale)
        imag_q = round(float(value.imag) / scale)
        real_q = max(-127, min(127, real_q))
        imag_q = max(-127, min(127, imag_q))
        if real_q == 0 and imag_q == 0:
            continue
        records.append(
            struct.pack(
                "<HhhBbb",
                int(frame_idx),
                int(k_row),
                int(k_col),
                int(channel),
                int(real_q),
                int(imag_q),
            )
        )
    if not records:
        return b""
    if len(records) > 0xFFFF:
        raise MaterializerError(f"SIREN coefficient count {len(records)} exceeds 65535")
    return struct.pack("<fH", float(scale), len(records)) + b"".join(records)


def _build_l2_encoded_siren_residual_blob(
    *,
    decoded_raw_path: Path,
    gt_raw_path: Path,
    n_frames: int,
    byte_budget: int,
    max_k: int,
    sparse_wire: bool,
    saliency_map_path: Path | None = None,
    saliency_power: float = 1.0,
    saliency_floor: float = 0.0,
) -> tuple[bytes, dict[str, object]]:
    """Select top low-frequency FFT residual coefficients from decoded/GT raw frames.

    This is the deterministic SIREN bridge from scaffold to score-moving bytes:
    it fits the residual ``gt - decoded`` in the frequency basis consumed by
    ``submissions/pr106_siren_residual_sidecar/inflate.py``. If a saliency
    map is provided, the residual is weighted before the FFT so the byte budget
    goes to scorer-relevant atoms instead of raw L2 energy. It is still a proxy
    encoder, not a score claim; exact CPU/CUDA eval decides promotion.
    """

    import numpy as np

    frame_bytes = CAMERA_H * CAMERA_W * RGB_CHANNELS
    decoded_total = decoded_raw_path.stat().st_size
    gt_total = gt_raw_path.stat().st_size
    if decoded_total % frame_bytes != 0:
        raise MaterializerError(
            f"decoded raw file {decoded_raw_path} size {decoded_total} "
            f"not divisible by frame_bytes {frame_bytes}"
        )
    if gt_total % frame_bytes != 0:
        raise MaterializerError(
            f"gt raw file {gt_raw_path} size {gt_total} "
            f"not divisible by frame_bytes {frame_bytes}"
        )
    n_decoded = decoded_total // frame_bytes
    n_gt = gt_total // frame_bytes
    if n_frames > 0:
        if n_decoded < n_frames or n_gt < n_frames:
            raise MaterializerError(
                f"explicit --n-frames={n_frames} exceeds available raw frames "
                f"(decoded={n_decoded}, gt={n_gt}); refusing silent truncation"
            )
        n_to_use = n_frames
    else:
        if n_decoded != n_gt:
            raise MaterializerError(
                f"decoded/gt raw frame counts differ (decoded={n_decoded}, gt={n_gt}); "
                f"pass explicit --n-frames to choose a prefix"
            )
        n_to_use = n_decoded
    if n_to_use <= 0:
        raise MaterializerError("no frames to encode")
    if byte_budget < 6 + PER_COEF_BYTES:
        raise MaterializerError(
            f"--byte-budget must fit at least one SIREN coef "
            f"({6 + PER_COEF_BYTES}B dense floor); got {byte_budget}"
        )
    if max_k < 0:
        raise MaterializerError(f"--max-k must be >= 0; got {max_k}")
    if saliency_power <= 0.0:
        raise MaterializerError(f"--saliency-power must be > 0; got {saliency_power}")
    if not (0.0 <= saliency_floor <= 1.0):
        raise MaterializerError(
            f"--saliency-floor must be in [0, 1]; got {saliency_floor}"
        )

    max_dense_coefs = min(0xFFFF, (byte_budget - 6) // PER_COEF_BYTES)
    if max_dense_coefs <= 0:
        raise MaterializerError(f"--byte-budget too small for SIREN dense payload: {byte_budget}")

    signed_rows = _signed_fft_indices(CAMERA_H)
    signed_cols = _signed_fft_indices(CAMERA_W)
    row_indices = [idx for idx, signed in enumerate(signed_rows) if abs(signed) <= max_k]
    col_indices = [idx for idx, signed in enumerate(signed_cols) if abs(signed) <= max_k]
    if not row_indices or not col_indices:
        raise MaterializerError("empty SIREN frequency window")
    rr, cc = np.meshgrid(
        np.array(row_indices, dtype=np.int64),
        np.array(col_indices, dtype=np.int64),
        indexing="ij",
    )
    flat_rows = rr.ravel()
    flat_cols = cc.ravel()

    decoded_mm = np.memmap(
        decoded_raw_path,
        dtype=np.uint8,
        mode="r",
        shape=(n_decoded, CAMERA_H, CAMERA_W, RGB_CHANNELS),
    )
    gt_mm = np.memmap(
        gt_raw_path,
        dtype=np.uint8,
        mode="r",
        shape=(n_gt, CAMERA_H, CAMERA_W, RGB_CHANNELS),
    )
    saliency_mm = None
    saliency_shape: tuple[int, ...] = ()
    saliency_dtype = ""
    saliency_sha = ""
    if saliency_map_path is not None:
        if not saliency_map_path.is_file():
            raise MaterializerError(f"saliency map missing: {saliency_map_path}")
        saliency_mm = np.load(saliency_map_path, mmap_mode="r")
        saliency_shape = tuple(int(v) for v in saliency_mm.shape)
        saliency_dtype = str(saliency_mm.dtype)
        saliency_sha = sha256_file(saliency_map_path)
        if saliency_mm.ndim not in (3, 4):
            raise MaterializerError(
                f"saliency map must have shape (T,H,W), (T,H,W,1), "
                f"or (T,H,W,3); got {saliency_shape}"
            )
        if saliency_shape[0] < n_to_use:
            raise MaterializerError(
                f"saliency map has only {saliency_shape[0]} frames; "
                f"need {n_to_use}"
            )
        if saliency_shape[1:3] != (CAMERA_H, CAMERA_W):
            raise MaterializerError(
                f"saliency spatial shape {saliency_shape[1:3]} != "
                f"{(CAMERA_H, CAMERA_W)}"
            )
        if saliency_mm.ndim == 4 and saliency_shape[3] not in (1, RGB_CHANNELS):
            raise MaterializerError(
                f"saliency channel dimension must be 1 or {RGB_CHANNELS}; "
                f"got {saliency_shape[3]}"
            )

    def _saliency_weights(frame_idx: int) -> np.ndarray | None:
        if saliency_mm is None:
            return None
        sal = np.asarray(saliency_mm[frame_idx], dtype=np.float64)
        if sal.ndim == 3 and sal.shape[2] == 1:
            sal = sal[..., 0]
        if not np.isfinite(sal).all():
            raise MaterializerError(
                f"saliency map contains non-finite values at frame {frame_idx}"
            )
        sal = np.maximum(sal, 0.0)
        max_sal = float(sal.max()) if sal.size else 0.0
        if max_sal <= 0.0:
            return np.zeros_like(sal, dtype=np.float64)
        weights = (sal / max_sal) ** saliency_power
        if saliency_floor:
            weights = saliency_floor + (1.0 - saliency_floor) * weights
        return weights.astype(np.float64, copy=False)

    # Atoms, not lone complex bins. Because inflate uses real(ifft2(...)),
    # non-self-conjugate real residual modes must be emitted as conjugate
    # pairs; otherwise the reconstructed amplitude is cut in half.
    atom_heap_cap = max_dense_coefs
    self_heap: list[
        tuple[float, int, int, int, int, int, int, int, int, float, float]
    ] = []
    pair_heap: list[
        tuple[
            float,
            int,
            int,
            int,
            int,
            int,
            int,
            int,
            int,
            float,
            float,
            int,
            int,
            float,
            float,
        ]
    ] = []

    def _push_self_atom(
        *,
        energy: float,
        frame_idx: int,
        channel: int,
        k_row: int,
        k_col: int,
        value: complex,
    ) -> None:
        key = (
            energy,
            -frame_idx,
            -channel,
            -k_row,
            -k_col,
            frame_idx,
            channel,
            k_row,
            k_col,
            float(value.real),
            float(value.imag),
        )
        if len(self_heap) < atom_heap_cap:
            heapq.heappush(self_heap, key)
        elif key > self_heap[0]:
            heapq.heapreplace(self_heap, key)

    def _push_pair_atom(
        *,
        energy: float,
        frame_idx: int,
        channel: int,
        k_row: int,
        k_col: int,
        value: complex,
        conj_k_row: int,
        conj_k_col: int,
        conj_value: complex,
    ) -> None:
        key = (
            energy,
            -frame_idx,
            -channel,
            -k_row,
            -k_col,
            frame_idx,
            channel,
            k_row,
            k_col,
            float(value.real),
            float(value.imag),
            conj_k_row,
            conj_k_col,
            float(conj_value.real),
            float(conj_value.imag),
        )
        pair_cap = max(atom_heap_cap // 2, 1)
        if len(pair_heap) < pair_cap:
            heapq.heappush(pair_heap, key)
        elif key > pair_heap[0]:
            heapq.heapreplace(pair_heap, key)

    for frame_idx in range(n_to_use):
        residual = (
            gt_mm[frame_idx].astype(np.float64)
            - decoded_mm[frame_idx].astype(np.float64)
        )
        weights = _saliency_weights(frame_idx)
        if weights is not None:
            if weights.ndim == 2:
                residual = residual * weights[..., None]
            else:
                residual = residual * weights
        spectrum = np.fft.fft2(residual, axes=(0, 1))
        values = spectrum[flat_rows, flat_cols, :]
        magnitudes = np.abs(values).ravel()
        frame_local_k = min(max_dense_coefs * 2, magnitudes.size)
        if frame_local_k < magnitudes.size:
            local = np.argpartition(magnitudes, -frame_local_k)[-frame_local_k:]
        else:
            local = np.arange(magnitudes.size)
        for idx in local:
            magnitude = float(magnitudes[idx])
            if magnitude == 0.0:
                continue
            bin_idx, channel = divmod(int(idx), RGB_CHANNELS)
            row = int(flat_rows[bin_idx])
            col = int(flat_cols[bin_idx])
            k_row = int(signed_rows[row])
            k_col = int(signed_cols[col])
            value = values[bin_idx, channel]
            conj_row = (-row) % CAMERA_H
            conj_col = (-col) % CAMERA_W
            is_self_conjugate = row == conj_row and col == conj_col
            if is_self_conjugate:
                _push_self_atom(
                    energy=magnitude * magnitude,
                    frame_idx=frame_idx,
                    channel=channel,
                    k_row=k_row,
                    k_col=k_col,
                    value=value,
                )
                continue
            if (row, col) > (conj_row, conj_col):
                continue
            conj_value = spectrum[conj_row, conj_col, channel]
            pair_energy = magnitude * magnitude + float(abs(conj_value) ** 2)
            _push_pair_atom(
                energy=pair_energy,
                frame_idx=frame_idx,
                channel=channel,
                k_row=k_row,
                k_col=k_col,
                value=value,
                conj_k_row=int(signed_rows[conj_row]),
                conj_k_col=int(signed_cols[conj_col]),
                conj_value=conj_value,
            )

    self_atoms = [
        (energy, frame_idx, channel, k_row, k_col, complex(real, imag))
        for (
            energy,
            _neg_frame,
            _neg_channel,
            _neg_k_row,
            _neg_k_col,
            frame_idx,
            channel,
            k_row,
            k_col,
            real,
            imag,
        ) in self_heap
    ]
    self_atoms.sort(key=lambda item: (-item[0], item[1], item[2], item[3], item[4]))
    pair_atoms = [
        (
            energy,
            frame_idx,
            channel,
            k_row,
            k_col,
            complex(real, imag),
            conj_k_row,
            conj_k_col,
            complex(conj_real, conj_imag),
        )
        for (
            energy,
            _neg_frame,
            _neg_channel,
            _neg_k_row,
            _neg_k_col,
            frame_idx,
            channel,
            k_row,
            k_col,
            real,
            imag,
            conj_k_row,
            conj_k_col,
            conj_real,
            conj_imag,
        ) in pair_heap
    ]
    pair_atoms.sort(key=lambda item: (-item[0], item[1], item[2], item[3], item[4]))

    self_prefix = [0.0]
    for atom in self_atoms:
        self_prefix.append(self_prefix[-1] + atom[0])
    pair_prefix = [0.0]
    for atom in pair_atoms:
        pair_prefix.append(pair_prefix[-1] + atom[0])
    best_self_count = 0
    best_pair_count = 0
    best_energy = -1.0
    max_self_count = min(len(self_atoms), max_dense_coefs)
    for self_count in range(max_self_count + 1):
        pair_count = min(len(pair_atoms), (max_dense_coefs - self_count) // 2)
        energy = self_prefix[self_count] + pair_prefix[pair_count]
        record_count = self_count + 2 * pair_count
        best_record_count = best_self_count + 2 * best_pair_count
        if energy > best_energy or (
            energy == best_energy and record_count < best_record_count
        ):
            best_energy = energy
            best_self_count = self_count
            best_pair_count = pair_count

    record_groups: list[
        tuple[float, int, int, int, int, list[tuple[float, int, int, int, int, complex]]]
    ] = []
    for atom in self_atoms[:best_self_count]:
        energy, frame_idx, channel, k_row, k_col, _value = atom
        record_groups.append((energy, frame_idx, channel, k_row, k_col, [atom]))
    for atom in pair_atoms[:best_pair_count]:
        energy, frame_idx, channel, k_row, k_col, value, conj_k_row, conj_k_col, conj_value = atom
        records = [
            (energy, frame_idx, channel, k_row, k_col, value),
            (energy, frame_idx, channel, conj_k_row, conj_k_col, conj_value),
        ]
        records.sort(key=lambda item: (item[1], item[2], item[3], item[4]))
        record_groups.append((energy, frame_idx, channel, k_row, k_col, records))
    record_groups.sort(key=lambda item: (-item[0], item[1], item[2], item[3], item[4]))
    selected: list[tuple[float, int, int, int, int, complex]] = []
    valid_prefix_counts: list[int] = []
    for *_key, records in record_groups:
        selected.extend(records)
        valid_prefix_counts.append(len(selected))
    dense = _encode_siren_coefficients(selected, n_coefs=len(selected))
    if sparse_wire and dense:
        best = b""
        valid_counts = [0, *valid_prefix_counts]
        lo, hi = 0, len(valid_counts) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            n_prefix = valid_counts[mid]
            dense_mid = _encode_siren_coefficients(selected, n_coefs=n_prefix)
            sparse_mid = (
                repack_dense_as_sparse(
                    family="siren",
                    dense_residual_bytes=dense_mid,
                    n_frames=n_to_use,
                )
                if dense_mid
                else b""
            )
            if len(sparse_mid) <= byte_budget:
                best = sparse_mid
                lo = mid + 1
            else:
                hi = mid - 1
        residual_bytes = best
        if selected and not residual_bytes:
            raise MaterializerError(
                f"SIREN sparse residual could not fit any runtime-consumed "
                f"coefficient into --byte-budget {byte_budget}; refusing "
                f"empty no-op L2 artifact"
            )
    else:
        residual_bytes = dense
    if len(residual_bytes) > byte_budget:
        raise MaterializerError(
            f"SIREN residual bytes {len(residual_bytes)} exceed --byte-budget {byte_budget}"
        )
    diag = {
        "decoded_raw_path": str(decoded_raw_path),
        "decoded_raw_sha256": sha256_file(decoded_raw_path),
        "decoded_raw_bytes": float(decoded_total),
        "decoded_raw_frame_count": float(n_decoded),
        "gt_raw_path": str(gt_raw_path),
        "gt_raw_sha256": sha256_file(gt_raw_path),
        "gt_raw_bytes": float(gt_total),
        "gt_raw_frame_count": float(n_gt),
        "n_frames_encoded": float(n_to_use),
        "candidate_frequency_bins": float(flat_rows.size),
        "candidate_frequency_components": float(flat_rows.size * RGB_CHANNELS),
        "selected_coefficients": float(len(selected)),
        "selected_self_conjugate_atoms": float(best_self_count),
        "selected_conjugate_pair_atoms": float(best_pair_count),
        "selected_complete_atoms": float(len(record_groups)),
        "selected_atom_energy": float(max(best_energy, 0.0)),
        "emitted_residual_bytes": float(len(residual_bytes)),
        "byte_budget": float(byte_budget),
        "max_k": float(max_k),
        "sparse_wire": float(1.0 if sparse_wire else 0.0),
        "saliency_weighted": bool(saliency_map_path is not None),
        "saliency_map_path": str(saliency_map_path) if saliency_map_path else "",
        "saliency_map_sha256": saliency_sha,
        "saliency_map_shape": list(saliency_shape),
        "saliency_map_dtype": saliency_dtype,
        "saliency_power": float(saliency_power),
        "saliency_floor": float(saliency_floor),
    }
    return residual_bytes, diag


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize PR106 + SIREN residual sidecar candidate"
    )
    parser.add_argument("--pr106-archive", type=Path, default=DEFAULT_PR106_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-frames", type=int, default=1200)
    parser.add_argument(
        "--n-coefs",
        type=int,
        default=0,
        help="Number of sparse FFT coefs (0 = empty residual scaffold mode).",
    )
    parser.add_argument(
        "--residual-mode",
        choices=("empty", "zero", "probe", "l2_encoded"),
        default="empty",
        help=(
            "empty/zero/probe = L1 scaffold modes; l2_encoded = deterministic "
            "top-FFT residual fitting from --decoded-raw/--gt-raw under "
            "--byte-budget. No score claim is emitted."
        ),
    )
    parser.add_argument("--default-scale", type=float, default=0.0)
    parser.add_argument("--max-k", type=int, default=32)
    parser.add_argument(
        "--decoded-raw",
        type=Path,
        default=None,
        help="(l2_encoded only) Path to (N,874,1164,3) uint8 decoded raw frames",
    )
    parser.add_argument(
        "--gt-raw",
        type=Path,
        default=None,
        help="(l2_encoded only) Path to (N,874,1164,3) uint8 ground-truth raw frames",
    )
    parser.add_argument(
        "--byte-budget",
        type=int,
        default=0,
        help="(l2_encoded only) Residual byte budget; must fit at least one coefficient.",
    )
    parser.add_argument(
        "--saliency-map-npy",
        type=Path,
        default=None,
        help=(
            "(l2_encoded only) Optional scorer/surrogate saliency map .npy "
            "with shape (T,874,1164), (T,874,1164,1), or (T,874,1164,3). "
            "Residuals are weighted before FFT so selected SIREN atoms target "
            "score-relevant pixels. This is proxy guidance only; no score "
            "claim is emitted."
        ),
    )
    parser.add_argument(
        "--saliency-power",
        type=float,
        default=1.0,
        help="(l2_encoded only) Exponent applied to normalized saliency weights.",
    )
    parser.add_argument(
        "--saliency-floor",
        type=float,
        default=0.0,
        help=(
            "(l2_encoded only) Floor blended into normalized saliency weights. "
            "0.0 permits hard masking; 1.0 disables saliency weighting."
        ),
    )
    parser.add_argument(
        "--decoded-axis",
        choices=("contest_cpu", "contest_cuda", "macos_cpu_advisory", "synthetic_test"),
        default="contest_cpu",
        help=(
            "(l2_encoded only) Evidence axis used to produce --decoded-raw. "
            "CPU/CUDA axes are never interchangeable."
        ),
    )
    parser.add_argument(
        "--decoded-inflate-device",
        choices=("cpu", "cuda", "synthetic"),
        default="cpu",
        help="(l2_encoded only) Device used by the runtime that produced --decoded-raw.",
    )
    parser.add_argument(
        "--decoded-runtime-sha256",
        default="",
        help="(l2_encoded only) SHA-256 of the inflate runtime file that produced --decoded-raw.",
    )
    parser.add_argument(
        "--decoded-runtime-tree-sha256",
        default="",
        help="(l2_encoded only) Runtime-tree SHA-256 for the decoded raw producer.",
    )
    parser.add_argument("--skip-no-op-smoke", action="store_true")
    parser.add_argument(
        "--encoding",
        choices=("dense", "sparse"),
        default="dense",
        help="Wire-format encoding: dense (0x13) or sparse PacketIR (0x23).",
    )
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        print(f"ERROR: --output-dir must be empty or not exist: {args.output_dir}", file=sys.stderr)
        return 2
    if args.residual_mode == "empty":
        residual_bytes = b""
        encoder_diagnostics: dict[str, object] = {}
        sparse_encoded_by_l2 = False
    elif args.residual_mode == "l2_encoded":
        if args.decoded_raw is None or args.gt_raw is None:
            print(
                "ERROR: --decoded-raw and --gt-raw are required for --residual-mode l2_encoded",
                file=sys.stderr,
            )
            return 2
        if args.byte_budget <= 0:
            print("ERROR: --byte-budget > 0 required for --residual-mode l2_encoded", file=sys.stderr)
            return 2
        if args.decoded_axis != "synthetic_test" and not (
            args.decoded_runtime_sha256 or args.decoded_runtime_tree_sha256
        ):
            print(
                "ERROR: --decoded-runtime-sha256 or --decoded-runtime-tree-sha256 "
                "is required for non-synthetic l2_encoded raw custody",
                file=sys.stderr,
            )
            return 2
        try:
            residual_bytes, encoder_diagnostics = _build_l2_encoded_siren_residual_blob(
                decoded_raw_path=args.decoded_raw,
                gt_raw_path=args.gt_raw,
                n_frames=args.n_frames,
                byte_budget=args.byte_budget,
                max_k=args.max_k,
                sparse_wire=args.encoding == "sparse",
                saliency_map_path=args.saliency_map_npy,
                saliency_power=args.saliency_power,
                saliency_floor=args.saliency_floor,
            )
            encoder_diagnostics.update(
                {
                    "decoded_axis": args.decoded_axis,
                    "decoded_inflate_device": args.decoded_inflate_device,
                    "decoded_runtime_sha256": args.decoded_runtime_sha256,
                    "decoded_runtime_tree_sha256": args.decoded_runtime_tree_sha256,
                    "score_claim_eligible": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                    "cpu_cuda_axes_interchangeable": False,
                }
            )
            sparse_encoded_by_l2 = args.encoding == "sparse"
        except MaterializerError as exc:
            print(f"ERROR: l2_encoded SIREN encoder failed: {exc}", file=sys.stderr)
            return 2
    else:
        if args.n_coefs <= 0:
            print(
                "ERROR: --n-coefs > 0 required when --residual-mode != empty",
                file=sys.stderr,
            )
            return 2
        residual_bytes = build_siren_residual_blob(
            n_coefs=args.n_coefs,
            n_frames=args.n_frames,
            mode=args.residual_mode,
            default_scale=args.default_scale,
            max_k=args.max_k,
        )
        encoder_diagnostics = {}
        sparse_encoded_by_l2 = False
    is_sparse = args.encoding == "sparse"
    if is_sparse and residual_bytes and not sparse_encoded_by_l2:
        try:
            residual_bytes = repack_dense_as_sparse(
                family="siren",
                dense_residual_bytes=residual_bytes,
                n_frames=args.n_frames,
            )
        except MaterializerError as exc:
            print(f"ERROR: sparse repack failed: {exc}", file=sys.stderr)
            return 2
    family = sparse_family_name("siren") if is_sparse else "siren"
    archive_zip, manifest_path, manifest, build = materialize_family_archive(
        family=family,
        pr106_archive=args.pr106_archive,
        residual_bytes=residual_bytes,
        output_dir=args.output_dir,
        extra={
            "residual_mode": args.residual_mode,
            "encoding": args.encoding,
            "n_frames": args.n_frames,
            "n_coefs": args.n_coefs,
            "per_coef_bytes": PER_COEF_BYTES,
            "default_scale": args.default_scale,
            "max_k": args.max_k,
            "l2_encoder_diagnostics": encoder_diagnostics,
            "rationale": (
                "Sitzmann et al. 2020 SIREN sinusoidal coord-MLP encoded as "
                "sparse 2D FFT coefs; inverse-FFT in inflate."
            ),
        },
    )
    if not args.skip_no_op_smoke and residual_bytes:
        smoke = run_no_op_detector_byte_mutation(
            archive_bytes=build.archive_bytes,
            expected_format_id=PR106_RESIDUAL_FORMAT_IDS[family],
        )
        print(f"[no_op_detector_byte_mutation_smoke] {smoke}", file=sys.stderr)
    print(f"materialized archive: {archive_zip}")
    print(f"manifest:             {manifest_path}")
    print(f"archive sha256:       {manifest.archive_sha256}")
    print(f"archive size bytes:   {manifest.archive_bytes_size}")
    print(f"residual bytes:       {manifest.residual_bytes_size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
