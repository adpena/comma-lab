#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prove dense HFV1 and sparse foveation sidecars produce identical frames.

Sparse HFV sidecars replace a dense per-frame HFV1 foveation table with a
compact equivalent. This tool avoids writing multi-GB raw outputs by replaying
the PR101/FEC6 inflate path in batches and comparing the dense-HFV1 and sparse
postprocessed frame tensors before bytes would be written.

It does not run the contest scorer and does not claim a score.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

DEFAULT_DENSE_ARCHIVE = Path(
    "experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/"
    "archive_seed_top16_component_hardpairs/archive.zip"
)
DEFAULT_SPARSE_ARCHIVE = Path(
    "experiments/results/hfv2_sparse_sidecar_candidate_20260521T070412Z/archive.zip"
)
DEFAULT_SPARSE_SUBMISSION_DIR = Path(
    "experiments/results/hfv2_sparse_sidecar_candidate_20260521T070412Z/"
    "submission_dir_hfv2"
)
OUTPUT_BASENAME = "sparse_foveation_inflate_parity"


@dataclass(frozen=True)
class Hfv2InflateParityProof:
    schema: str
    generated_at_utc: str
    dense_archive: str
    dense_archive_bytes: int
    dense_archive_sha256: str
    sparse_archive: str
    sparse_archive_bytes: int
    sparse_archive_sha256: str
    sparse_submission_dir: str
    sparse_inflate_py_sha256: str
    device: str
    pair_count_total: int
    pair_indices_checked: list[int]
    frame_count_checked: int
    batch_pairs: int
    x_payload_sha256: str
    x_payload_bytes: int
    dense_sidecar_name: str
    sparse_sidecar_name: str
    dense_sidecar_sha256: str
    sparse_sidecar_sha256: str
    dense_output_sha256: str
    sparse_output_sha256: str
    output_sha256_match: bool
    tensor_equal: bool
    max_abs_diff: float
    mismatched_pair_batches: list[list[int]]
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _import_sparse_inflate(submission_dir: Path) -> Any:
    inflate_py = submission_dir / "inflate.py"
    if not inflate_py.is_file():
        raise FileNotFoundError(f"inflate.py not found: {inflate_py}")
    spec = importlib.util.spec_from_file_location("hfv2_sparse_inflate_runtime", inflate_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load inflate module from {inflate_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_archive(path: Path, target: Path) -> dict[str, bytes]:
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(target)
        return {info.filename: archive.read(info.filename) for info in archive.infolist()}


def _embedded_sidecar_from_wrapped_x(raw: bytes) -> tuple[str, bytes] | None:
    if len(raw) < 10:
        raise ValueError("wrapped x payload truncated before header")
    if raw[:4] != b"FP11":
        raise ValueError(f"wrapped x magic mismatch: {raw[:4]!r}")
    pos = 4
    (source_len,) = struct.unpack_from("<I", raw, pos)
    pos += 4 + int(source_len)
    if pos > len(raw):
        raise ValueError("wrapped x source payload truncated")
    if pos + 2 > len(raw):
        raise ValueError("wrapped x selector length missing")
    (selector_len,) = struct.unpack_from("<H", raw, pos)
    pos += 2 + int(selector_len)
    if pos > len(raw):
        raise ValueError("wrapped x selector payload truncated")
    if pos == len(raw):
        return None
    trailer = raw[pos:]
    if trailer.startswith(b"HFV2"):
        return "embedded_foveation_params.hfv2", trailer
    if trailer.startswith(b"HFV3"):
        return "embedded_foveation_params.hfv3", trailer
    if trailer.startswith(b"HFV4"):
        return "embedded_foveation_params.hfv4", trailer
    raise ValueError(f"unsupported embedded sidecar trailer magic: {trailer[:4]!r}")


def _parse_pair_indices(value: str, *, n_pairs: int) -> list[int]:
    indices: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"bad descending pair range: {token!r}")
            indices.update(range(start, end + 1))
        else:
            indices.add(int(token))
    bad = [index for index in indices if index < 0 or index >= n_pairs]
    if bad:
        raise ValueError(f"pair indices outside [0,{n_pairs}): {bad[:10]}")
    return sorted(indices)


def _chunks(indices: list[int], size: int) -> list[list[int]]:
    return [indices[start : start + size] for start in range(0, len(indices), size)]


def _select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("requested cuda but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("requested mps but torch.backends.mps.is_available() is false")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_parity_proof(
    *,
    dense_archive: Path,
    sparse_archive: Path,
    sparse_submission_dir: Path,
    pair_indices_arg: str | None,
    pair_cap: int | None,
    batch_pairs: int,
    device_arg: str,
) -> Hfv2InflateParityProof:
    if batch_pairs <= 0:
        raise ValueError("--batch-pairs must be positive")
    device = _select_device(device_arg)
    module = _import_sparse_inflate(sparse_submission_dir)
    with tempfile.TemporaryDirectory(prefix="hfv2_parity_") as tmp_s:
        tmp = Path(tmp_s)
        dense_dir = tmp / "dense"
        sparse_dir = tmp / "sparse"
        dense_members = _extract_archive(dense_archive, dense_dir)
        sparse_members = _extract_archive(sparse_archive, sparse_dir)
        dense_x = dense_members.get("x")
        sparse_x = sparse_members.get("x")
        if dense_x is None or sparse_x is None:
            raise ValueError("both archives must contain member x")
        if dense_x != sparse_x:
            embedded = _embedded_sidecar_from_wrapped_x(sparse_x)
            if embedded is None or not sparse_x.startswith(dense_x):
                raise ValueError("dense and sparse x payloads differ outside embedded sidecar trailer")
        dense_sidecar = dense_members.get("foveation_params.bin")
        sparse_sidecar_name = "foveation_params.hfv2"
        sparse_sidecar = sparse_members.get(sparse_sidecar_name)
        if sparse_sidecar is None:
            embedded = _embedded_sidecar_from_wrapped_x(sparse_x)
            if embedded is not None:
                sparse_sidecar_name, sparse_sidecar = embedded
        if dense_sidecar is None:
            raise ValueError("dense archive missing foveation_params.bin")
        if sparse_sidecar is None:
            raise ValueError("sparse archive missing external or embedded foveation sidecar")

        source_payload, selector_kind, selector_codes, selector_specs = (
            module.parse_pr101_frame_selector_archive(dense_x)
        )
        decoder_sd, latents, meta = module.parse_archive(source_payload)
        dense_params = module.load_foveation_sidecar(dense_dir / "x")
        sparse_params = module.load_foveation_sidecar(sparse_dir / "x")
        n_pairs = int(meta["n_pairs"])
        if len(selector_codes) != n_pairs:
            raise ValueError(f"selector has {len(selector_codes)} pairs; expected {n_pairs}")
        if pair_indices_arg:
            pair_indices = _parse_pair_indices(pair_indices_arg, n_pairs=n_pairs)
        else:
            pair_indices = list(range(n_pairs))
        if pair_cap is not None:
            if pair_cap < 0:
                raise ValueError("--pair-cap must be nonnegative")
            pair_indices = pair_indices[:pair_cap]

        decoder = module.HNeRVDecoder(
            latent_dim=meta["latent_dim"],
            base_channels=meta["base_channels"],
            eval_size=tuple(meta["eval_size"]),
        ).to(device)
        decoder.load_state_dict(decoder_sd)
        decoder.eval()
        latents = latents.to(device)
        eval_h, eval_w = meta["eval_size"]
        dense_hash = hashlib.sha256()
        sparse_hash = hashlib.sha256()
        max_abs_diff = 0.0
        mismatched_batches: list[list[int]] = []

        with torch.inference_mode():
            for pair_batch in _chunks(pair_indices, batch_pairs):
                index_tensor = torch.tensor(pair_batch, dtype=torch.long, device=device)
                decoded = decoder(latents.index_select(0, index_tensor))
                batch = len(pair_batch)
                flat = decoded.reshape(batch * 2, 3, eval_h, eval_w)
                up = F.interpolate(
                    flat,
                    size=(module.CAMERA_H, module.CAMERA_W),
                    mode="bicubic",
                    align_corners=False,
                )
                up = up.reshape(batch, 2, 3, module.CAMERA_H, module.CAMERA_W)
                up[:, 0, 0].sub_(1.0)
                up[:, 0, 2].sub_(1.0)
                up[:, 1, 1].sub_(1.0)
                rounded = up.reshape(batch * 2, 3, module.CAMERA_H, module.CAMERA_W).clamp(
                    0,
                    255,
                ).round()
                selected = rounded.clone()
                for local_pair, pair_index in enumerate(pair_batch):
                    start = local_pair * 2
                    selected[start : start + 2] = module.apply_pr101_selector_to_frames(
                        selected[start : start + 2],
                        selector_kind,
                        selector_codes,
                        selector_specs,
                        pair_start=pair_index,
                    )

                dense_out_parts: list[torch.Tensor] = []
                sparse_out_parts: list[torch.Tensor] = []
                for local_pair, pair_index in enumerate(pair_batch):
                    start = local_pair * 2
                    frame_start = int(pair_index) * 2
                    dense_out_parts.append(
                        module.apply_hfv1_to_rounded_frames(
                            selected[start : start + 2].clone(),
                            dense_params,
                            frame_start=frame_start,
                        )
                    )
                    sparse_out_parts.append(
                        module.apply_hfv1_to_rounded_frames(
                            selected[start : start + 2].clone(),
                            sparse_params,
                            frame_start=frame_start,
                        )
                    )
                dense_out = torch.cat(dense_out_parts, dim=0)
                sparse_out = torch.cat(sparse_out_parts, dim=0)
                diff = (dense_out - sparse_out).abs()
                if diff.numel():
                    max_abs_diff = max(max_abs_diff, float(diff.max().item()))
                if not torch.equal(dense_out, sparse_out):
                    mismatched_batches.append(pair_batch)
                dense_bytes = dense_out.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy().tobytes()
                sparse_bytes = sparse_out.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy().tobytes()
                dense_hash.update(dense_bytes)
                sparse_hash.update(sparse_bytes)

    dense_output_sha = dense_hash.hexdigest()
    sparse_output_sha = sparse_hash.hexdigest()
    return Hfv2InflateParityProof(
        schema="sparse_foveation_inflate_parity_proof_v2",
        generated_at_utc=_utc_iso(),
        dense_archive=_repo_rel(dense_archive),
        dense_archive_bytes=dense_archive.stat().st_size,
        dense_archive_sha256=_sha256_file(dense_archive),
        sparse_archive=_repo_rel(sparse_archive),
        sparse_archive_bytes=sparse_archive.stat().st_size,
        sparse_archive_sha256=_sha256_file(sparse_archive),
        sparse_submission_dir=_repo_rel(sparse_submission_dir),
        sparse_inflate_py_sha256=_sha256_file(sparse_submission_dir / "inflate.py"),
        device=str(device),
        pair_count_total=n_pairs,
        pair_indices_checked=pair_indices,
        frame_count_checked=len(pair_indices) * 2,
        batch_pairs=batch_pairs,
        x_payload_sha256=_sha256_bytes(dense_x),
        x_payload_bytes=len(dense_x),
        dense_sidecar_name="foveation_params.bin",
        sparse_sidecar_name=sparse_sidecar_name,
        dense_sidecar_sha256=_sha256_bytes(dense_sidecar),
        sparse_sidecar_sha256=_sha256_bytes(sparse_sidecar),
        dense_output_sha256=dense_output_sha,
        sparse_output_sha256=sparse_output_sha,
        output_sha256_match=dense_output_sha == sparse_output_sha,
        tensor_equal=not mismatched_batches,
        max_abs_diff=max_abs_diff,
        mismatched_pair_batches=mismatched_batches,
    )


def render_markdown(proof: Hfv2InflateParityProof) -> str:
    checked = "all" if len(proof.pair_indices_checked) == proof.pair_count_total else str(len(proof.pair_indices_checked))
    return "\n".join(
        [
            "# Sparse Foveation Inflate Parity Proof",
            "",
            f"- Generated UTC: {proof.generated_at_utc}",
            f"- Dense archive: `{proof.dense_archive}`",
            f"- Sparse archive: `{proof.sparse_archive}`",
            f"- Sparse runtime: `{proof.sparse_submission_dir}`",
            f"- Device: `{proof.device}`",
            f"- Pair coverage: {checked} / {proof.pair_count_total}",
            f"- Frame count checked: {proof.frame_count_checked}",
            f"- X payload bytes: {proof.x_payload_bytes}",
            f"- X payload SHA-256: `{proof.x_payload_sha256}`",
            f"- Dense output SHA-256: `{proof.dense_output_sha256}`",
            f"- Sparse output SHA-256: `{proof.sparse_output_sha256}`",
            f"- Output SHA-256 match: {str(proof.output_sha256_match).lower()}",
            f"- Tensor equal: {str(proof.tensor_equal).lower()}",
            f"- Max abs diff: {proof.max_abs_diff}",
            f"- Mismatched batches: {len(proof.mismatched_pair_batches)}",
            "- Score claim: false",
            "- Promotion eligible: false",
            "",
        ]
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-archive", type=Path, default=DEFAULT_DENSE_ARCHIVE)
    parser.add_argument("--sparse-archive", type=Path, default=DEFAULT_SPARSE_ARCHIVE)
    parser.add_argument("--sparse-submission-dir", type=Path, default=DEFAULT_SPARSE_SUBMISSION_DIR)
    parser.add_argument("--pair-indices", default=None, help="Comma/range list, e.g. 0,64,79-80")
    parser.add_argument("--pair-cap", type=int, default=None)
    parser.add_argument("--batch-pairs", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results") / f"hfv2_sparse_inflate_parity_{_utc_stamp()}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    proof = build_parity_proof(
        dense_archive=args.dense_archive,
        sparse_archive=args.sparse_archive,
        sparse_submission_dir=args.sparse_submission_dir,
        pair_indices_arg=args.pair_indices,
        pair_cap=args.pair_cap,
        batch_pairs=args.batch_pairs,
        device_arg=args.device,
    )
    payload = json.dumps(proof.to_dict(), indent=2, sort_keys=True) + "\n"
    (args.output_dir / f"{OUTPUT_BASENAME}.json").write_text(payload, encoding="utf-8")
    (args.output_dir / f"{OUTPUT_BASENAME}.md").write_text(
        render_markdown(proof),
        encoding="utf-8",
    )
    sys.stdout.write(payload)
    return 0 if proof.output_sha256_match and proof.tensor_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
