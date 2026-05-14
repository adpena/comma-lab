#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Bounded PR86 HPAC replay/parity diagnostic.

This is a local, non-promotable probe for the public PR86 archive intake. It
does not run contest eval, does not dispatch remote work, and intentionally
decodes only a small prefix unless the caller opts into more.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PR86_DIR = REPO_ROOT / "experiments/results/public_pr86_intake_20260504_codex"
DEFAULT_ARCHIVE = DEFAULT_PR86_DIR / "archive.zip"
DEFAULT_JSON_OUT = DEFAULT_PR86_DIR / "pr86_hpac_replay_parity_diagnostic.json"

SEGNET_IN_H = 384
SEGNET_IN_W = 512
NUM_CLASSES = 5


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return type(node).__name__


class _CallFinder(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def visit_Call(self, node: ast.Call) -> None:
        name = ""
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        self.calls.append(
            {
                "name": name,
                "lineno": getattr(node, "lineno", None),
                "args": [_unparse(arg) for arg in node.args],
            }
        )
        self.generic_visit(node)


def find_calls(path: Path, name: str) -> list[dict[str, Any]]:
    tree = ast.parse(_read_text(path), filename=str(path))
    finder = _CallFinder()
    finder.visit(tree)
    return [row for row in finder.calls if row["name"] == name]


def analyze_token_semantics(pr86_dir: Path) -> dict[str, Any]:
    training_hpac = pr86_dir / "training/hpac.py"
    training_archive = pr86_dir / "training/archive.py"
    inflate_py = pr86_dir / "inflate.py"
    hpac_text = _read_text(training_hpac)
    archive_text = _read_text(training_archive)
    inflate_text = _read_text(inflate_py)

    write_tokens_calls = find_calls(training_archive, "write_tokens")
    encode_calls = find_calls(training_archive, "encode_frame")
    write_tokens_arg = None
    if write_tokens_calls and len(write_tokens_calls[0]["args"]) >= 2:
        write_tokens_arg = write_tokens_calls[0]["args"][1]

    training_uses_residual_targets = (
        "compute_residuals(gt_tokens)" in hpac_text
        and "res = gt_residuals" in hpac_text
        and "res[i] = (tok[i] - tok[i-1]) mod" in hpac_text
    )
    archive_computes_raw_gt = "gt = compute_gt_tokens()" in archive_text
    archive_encodes_raw_gt = write_tokens_arg == "gt"
    archive_mentions_residuals = "compute_residuals" in archive_text or "residual" in archive_text
    inflate_reconstructs_residuals = (
        "compute_residuals" in inflate_text
        or "decoded_prev" in inflate_text
        and "% NUM_CLASSES" in inflate_text
        and "+" in inflate_text.partition("decompress_tokens_hpac")[2]
    )

    if archive_computes_raw_gt and archive_encodes_raw_gt and not inflate_reconstructs_residuals:
        submitted_encoding = "raw_tokens"
    elif training_uses_residual_targets and inflate_reconstructs_residuals:
        submitted_encoding = "residual_tokens"
    else:
        submitted_encoding = "ambiguous"

    return {
        "training_hpac_path": str(training_hpac.relative_to(REPO_ROOT)),
        "training_archive_path": str(training_archive.relative_to(REPO_ROOT)),
        "inflate_path": str(inflate_py.relative_to(REPO_ROOT)),
        "training_objective": "residual_tokens" if training_uses_residual_targets else "unknown",
        "training_residual_definition_present": training_uses_residual_targets,
        "archive_compute_gt_tokens_call_present": archive_computes_raw_gt,
        "archive_write_tokens_second_arg": write_tokens_arg,
        "archive_encode_frame_calls": encode_calls,
        "archive_mentions_residuals": archive_mentions_residuals,
        "inflate_reconstructs_residuals": inflate_reconstructs_residuals,
        "submitted_archive_token_encoding": submitted_encoding,
        "interpretation": (
            "PR86 training optimizes residual-token targets, but the submitted "
            "archive builder passes raw SegNet token maps into encode_frame and "
            "inflate.py returns decoded tokens directly. Treat submitted "
            "tokens.bin as raw-token coded unless a separate archive proves "
            "residual reconstruction."
        ),
    }


def analyze_inflate_device_contract(pr86_dir: Path) -> dict[str, Any]:
    inflate_py = pr86_dir / "inflate.py"
    text = _read_text(inflate_py)
    calls = find_calls(inflate_py, "decompress_tokens_hpac")
    hpac_call_args = calls[-1]["args"] if calls else []
    device_arg = hpac_call_args[8] if len(hpac_call_args) > 8 else None
    comment_claims_cpu = "Force HPAC decode onto CPU" in text
    main_selects_cuda = '"cuda" if torch.cuda.is_available() else "cpu"' in text
    passes_main_device = device_arg == "device"
    return {
        "inflate_path": str(inflate_py.relative_to(REPO_ROOT)),
        "comment_claims_hpac_cpu": comment_claims_cpu,
        "main_device_expression_selects_cuda_when_available": main_selects_cuda,
        "decompress_tokens_hpac_device_arg": device_arg,
        "submitted_code_passes_main_device_to_hpac": passes_main_device,
        "comment_code_mismatch": comment_claims_cpu and passes_main_device,
        "interpretation": (
            "Submitted inflate.py comments say HPAC is forced to CPU, but the "
            "call passes the main runtime device. On CUDA hosts, HPAC decode "
            "would run on CUDA unless the runtime is patched or CUDA is hidden."
            if comment_claims_cpu and passes_main_device
            else "No CPU/CUDA comment mismatch detected."
        ),
    }


def analyze_probability_contract(pr86_dir: Path) -> dict[str, Any]:
    training_archive = pr86_dir / "training/archive.py"
    inflate_py = pr86_dir / "inflate.py"
    readme = pr86_dir / "training/README.md"
    archive_text = _read_text(training_archive)
    inflate_text = _read_text(inflate_py)
    readme_text = _read_text(readme) if readme.is_file() else ""
    source_text = archive_text + "\n" + inflate_text
    return {
        "archive_path": str(training_archive.relative_to(REPO_ROOT)),
        "inflate_path": str(inflate_py.relative_to(REPO_ROOT)),
        "readme_path": str(readme.relative_to(REPO_ROOT)) if readme.is_file() else None,
        "categorical_perfect_false_in_archive_code": "perfect=False" in source_text,
        "probability_clip_eps": "1e-7" if "1e-7" in source_text else None,
        "explicit_16384_grid_in_archive_code": "16384" in source_text,
        "readme_mentions_16384_grid": "1/16384" in readme_text or "16384" in readme_text,
        "interpretation": (
            "Submitted encode/decode loops pass clipped, renormalized float64 "
            "probabilities to constriction Categorical(..., perfect=False). "
            "The README mentions a 1/16384 grid, but the submitted archive and "
            "inflate code do not implement an explicit 16384-grid quantizer."
        ),
    }


def archive_custody(archive: Path) -> dict[str, Any]:
    members: list[dict[str, Any]] = []
    names: list[str] = []
    with zipfile.ZipFile(archive, "r") as zf:
        for info in zf.infolist():
            data = zf.read(info.filename)
            names.append(info.filename)
            members.append(
                {
                    "name": info.filename,
                    "file_size": info.file_size,
                    "compress_size": info.compress_size,
                    "crc32_hex": f"{info.CRC:08x}",
                    "sha256": sha256_bytes(data),
                }
            )
    duplicates = sorted({name for name in names if names.count(name) > 1})
    return {
        "path": str(archive.relative_to(REPO_ROOT)),
        "exists": archive.is_file(),
        "size_bytes": archive.stat().st_size,
        "sha256": sha256_path(archive),
        "member_count": len(members),
        "duplicate_member_names": duplicates,
        "members": members,
    }


def constriction_queue_self_test() -> dict[str, Any]:
    version = package_version("constriction")
    if version is None:
        return {
            "available": False,
            "version": None,
            "status": "missing_dependency",
        }
    try:
        import constriction

        probabilities = np.array([0.04, 0.16, 0.50, 0.20, 0.10], dtype=np.float64)
        symbols = [int((i * 7 + 3) % 5) for i in range(257)]
        encoder = constriction.stream.queue.RangeEncoder()
        for sym in symbols:
            model = constriction.stream.model.Categorical(
                probabilities=probabilities, perfect=False
            )
            encoder.encode(int(sym), model)
        words = encoder.get_compressed()
        decoder = constriction.stream.queue.RangeDecoder(words)
        decoded: list[int] = []
        for _ in symbols:
            model = constriction.stream.model.Categorical(
                probabilities=probabilities, perfect=False
            )
            decoded.append(int(decoder.decode(model)))
        same_order_ok = decoded == symbols

        reversed_words_result: dict[str, Any]
        try:
            rev_decoder = constriction.stream.queue.RangeDecoder(words[::-1].copy())
            rev_decoded: list[int] = []
            for _ in symbols:
                model = constriction.stream.model.Categorical(
                    probabilities=probabilities, perfect=False
                )
                rev_decoded.append(int(rev_decoder.decode(model)))
            reversed_words_result = {
                "status": "decoded",
                "decoded_symbols_prefix": rev_decoded[:16],
                "decoded_symbols_sha256": sha256_bytes(
                    np.array(rev_decoded, dtype=np.uint8).tobytes()
                ),
                "matches_original": rev_decoded == symbols,
            }
        except Exception as exc:  # pragma: no cover - exact exception is version-specific.
            reversed_words_result = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

        return {
            "available": True,
            "version": version,
            "queue_api": "constriction.stream.queue.RangeEncoder/RangeDecoder",
            "model_api": "constriction.stream.model.Categorical(..., perfect=False)",
            "compressed_dtype": str(words.dtype),
            "compressed_word_count": int(words.size),
            "same_order_roundtrip_ok": same_order_ok,
            "symbol_count": len(symbols),
            "encoded_symbols_prefix": symbols[:16],
            "decoded_symbols_prefix": decoded[:16],
            "encoded_symbols_sha256": sha256_bytes(np.array(symbols, dtype=np.uint8).tobytes()),
            "decoded_symbols_sha256": sha256_bytes(np.array(decoded, dtype=np.uint8).tobytes()),
            "reversed_words_probe": reversed_words_result,
            "status": "passed" if same_order_ok else "failed",
        }
    except Exception as exc:
        return {
            "available": True,
            "version": version,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _load_module(module_path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _extract_required_members(archive: Path, dst: Path, names: list[str]) -> None:
    with zipfile.ZipFile(archive, "r") as zf:
        for name in names:
            target = dst / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(name))


def _extract_available_hpac_members(archive: Path, dst: Path) -> None:
    with zipfile.ZipFile(archive, "r") as zf:
        names = set(zf.namelist())
        for name in ("tokens.bin", "meta.pt"):
            if name not in names:
                raise KeyError(f"archive missing required PR86 member {name!r}")
        hpac_names = [name for name in ("hpac.pt.ppmd", "hpac.pt.gz", "hpac.pt") if name in names]
        if not hpac_names:
            raise KeyError("archive missing required HPAC member hpac.pt.ppmd/hpac.pt.gz/hpac.pt")
        for name in ["tokens.bin", "meta.pt", hpac_names[0]]:
            target = dst / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(name))


def _full_group_masks(P: int, delta: int, H: int, W: int, device: str) -> list[torch.Tensor | None]:
    nrp, ncp = H // P, W // P
    rs = torch.arange(P, device=device).view(P, 1).expand(P, P)
    cs = torch.arange(P, device=device).view(1, P).expand(P, P)
    s_grid = cs + delta * rs
    n_groups = int((1 + delta) * P - delta)
    masks: list[torch.Tensor | None] = []
    for s in range(n_groups):
        mask_p = s_grid == s
        if not bool(mask_p.any()):
            masks.append(None)
            continue
        full = (
            mask_p.unsqueeze(0)
            .unsqueeze(0)
            .expand(nrp, ncp, P, P)
            .permute(0, 2, 1, 3)
            .reshape(nrp * P, ncp * P)
        )
        masks.append(full)
    return masks


def _load_hpac_generator(
    *,
    inflate: Any,
    tmpdir: Path,
    meta: dict[str, Any],
    device: str,
    num_classes: int = NUM_CLASSES,
) -> tuple[Any, dict[str, Any], Path]:
    hpac_path = next(
        (tmpdir / name for name in ("hpac.pt.ppmd", "hpac.pt.gz", "hpac.pt") if (tmpdir / name).exists()),
        None,
    )
    if hpac_path is None:
        raise FileNotFoundError("no HPAC model member extracted")

    if hpac_path.name.endswith(".ppmd"):
        decoded_hpac = inflate.pyppmd.decompress(
            hpac_path.read_bytes(),
            max_order=int(meta.get("ppmd_max_order", 4)),
            mem_size=16 << 20,
        )
        packed_sd = torch.load(io.BytesIO(decoded_hpac), map_location="cpu", weights_only=False)
    elif hpac_path.name.endswith(".gz"):
        import gzip

        with gzip.open(hpac_path, "rb") as f:
            packed_sd = torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=False)
    else:
        packed_sd = torch.load(hpac_path, map_location="cpu", weights_only=False)

    sd = inflate._reconstruct_hpac_state_dict(packed_sd, device)
    P = int(meta.get("P", 32))
    delta = int(meta.get("delta", 2))
    ch = int(meta.get("ch", 64))
    use_spm = bool(meta.get("use_spm", False))
    hpac_d_film = int(meta.get("hpac_d_film", 32))
    total_frames = int(meta["N"])
    model_pairs = int(sd["frame_embed.weight"].shape[0]) if "frame_embed.weight" in sd else total_frames
    gen = inflate.HPACMini(
        num_pairs=model_pairs,
        num_classes=num_classes,
        P=P,
        delta=delta,
        ch=ch,
        d_film=hpac_d_film,
        use_spm=use_spm,
    ).to(device).eval()
    incompatible = gen.load_state_dict(sd, strict=False)
    state_load = {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
    return gen, state_load, hpac_path


def decode_pr86_prefix(
    *,
    archive: Path,
    pr86_dir: Path,
    device: str,
    max_frames: int,
    max_groups: int | None,
) -> dict[str, Any]:
    if device == "cuda" and not torch.cuda.is_available():
        return {
            "device": device,
            "status": "skipped",
            "skip_reason": "torch.cuda.is_available() is false",
        }

    started = time.time()
    inflate = _load_module(pr86_dir / "inflate.py", f"pr86_inflate_replay_{device}")
    with tempfile.TemporaryDirectory(prefix="pr86_hpac_probe_") as tmp:
        tmpdir = Path(tmp)
        _extract_available_hpac_members(archive, tmpdir)
        meta = torch.load(tmpdir / "meta.pt", map_location="cpu", weights_only=False)
        P = int(meta.get("P", 32))
        delta = int(meta.get("delta", 2))
        ch = int(meta.get("ch", 64))
        use_spm = bool(meta.get("use_spm", False))
        hpac_d_film = int(meta.get("hpac_d_film", 32))
        total_frames = int(meta["N"])
        n_groups = int((1 + delta) * P - delta)
        groups_to_decode = n_groups if max_groups is None else min(max_groups, n_groups)
        frames_to_decode = max(0, min(max_frames, total_frames))
        if groups_to_decode < n_groups and frames_to_decode > 1:
            frames_to_decode = 1

        gen, state_load, _hpac_path = _load_hpac_generator(
            inflate=inflate,
            tmpdir=tmpdir,
            meta=meta,
            device=device,
        )
        model_pairs = int(getattr(gen, "num_pairs", total_frames))

        blob = (tmpdir / "tokens.bin").read_bytes()
        decoder = inflate.constriction.stream.queue.RangeDecoder(
            np.frombuffer(blob, dtype=np.uint32)
        )
        masks = _full_group_masks(P, delta, SEGNET_IN_H, SEGNET_IN_W, device)
        decoded_prev = torch.zeros(
            (1, SEGNET_IN_H, SEGNET_IN_W), dtype=torch.long, device=device
        )
        decoded_stream = bytearray()
        frame_reports: list[dict[str, Any]] = []

        try:
            with torch.no_grad():
                for frame_idx in range(frames_to_decode):
                    idx = torch.tensor([frame_idx], dtype=torch.long, device=device)
                    cur = torch.zeros_like(decoded_prev)
                    group_reports: list[dict[str, Any]] = []
                    for group_idx in range(groups_to_decode):
                        mask = masks[group_idx]
                        if mask is None:
                            group_reports.append(
                                {
                                    "group": group_idx,
                                    "positions": 0,
                                    "status": "empty",
                                }
                            )
                            continue
                        logits = gen(cur, idx, decoded_prev)
                        probs = F.softmax(logits.float(), dim=1)
                        probs_at_group = probs[0][:, mask].permute(1, 0).contiguous()
                        probs_np = probs_at_group.detach().cpu().numpy().astype(np.float64)
                        probs_np = np.clip(probs_np, 1e-7, 1.0)
                        probs_np = probs_np / probs_np.sum(axis=1, keepdims=True)
                        decoded = np.empty(probs_np.shape[0], dtype=np.int64)
                        for i in range(probs_np.shape[0]):
                            model = inflate.constriction.stream.model.Categorical(
                                probabilities=probs_np[i], perfect=False
                            )
                            decoded[i] = int(decoder.decode(model))
                        cur[0, mask] = torch.from_numpy(decoded).to(device)
                        decoded_u8 = decoded.astype(np.uint8)
                        decoded_stream.extend(decoded_u8.tobytes())
                        hist = np.bincount(decoded_u8, minlength=NUM_CLASSES).astype(int)
                        group_reports.append(
                            {
                                "group": group_idx,
                                "positions": int(decoded_u8.size),
                                "histogram": hist.tolist(),
                                "first_symbols": decoded_u8[:16].astype(int).tolist(),
                                "status": "decoded",
                            }
                        )
                    if groups_to_decode == n_groups:
                        decoded_prev = cur.clone()
                    frame_reports.append(
                        {
                            "frame": frame_idx,
                            "groups_decoded": groups_to_decode,
                            "full_frame_decoded": groups_to_decode == n_groups,
                            "groups": group_reports,
                        }
                    )
        except Exception as exc:
            return {
                "device": device,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "frames_requested": max_frames,
                "frames_decoded_before_error": len(frame_reports),
                "groups_requested": max_groups,
                "groups_per_frame_total": n_groups,
                "elapsed_sec": round(time.time() - started, 3),
            }

        return {
            "device": device,
            "status": "passed",
            "scope": "bounded_prefix_decode",
            "frames_requested": max_frames,
            "frames_decoded": frames_to_decode,
            "groups_requested": max_groups,
            "groups_decoded_per_frame": groups_to_decode,
            "groups_per_frame_total": n_groups,
            "full_frame_decode": groups_to_decode == n_groups,
            "full_archive_decode": frames_to_decode == total_frames and groups_to_decode == n_groups,
            "meta": {
                "N": total_frames,
                "P": P,
                "delta": delta,
                "ch": ch,
                "hpac_d_film": hpac_d_film,
                "use_spm": use_spm,
                "ppmd_max_order": int(meta.get("ppmd_max_order", 4)),
                "tokens_bpp": float(meta.get("tokens_bpp", 0.0)),
            },
            "hpac_model_pairs": model_pairs,
            "state_load": state_load,
            "decoded_symbol_count": len(decoded_stream),
            "decoded_prefix_sha256": sha256_bytes(bytes(decoded_stream)),
            "frames": frame_reports,
            "elapsed_sec": round(time.time() - started, 3),
        }


def decode_reencode_pr86_tokens(
    *,
    archive: Path,
    pr86_dir: Path,
    device: str,
    max_frames: int | None = None,
    max_groups: int | None = None,
    height: int = SEGNET_IN_H,
    width: int = SEGNET_IN_W,
) -> dict[str, Any]:
    """Decode PR86 tokens and re-encode them through the same HPAC contract.

    Only a full archive pass can satisfy the byte-exact gate. Bounded calls are
    useful for tests and local diagnostics but remain non-promotable.
    """
    if device == "cuda" and not torch.cuda.is_available():
        return {
            "device": device,
            "status": "skipped",
            "gate": "pr86_full_decode_reencode",
            "skip_reason": "torch.cuda.is_available() is false",
            "score_claim": False,
            "planning_only": True,
        }

    started = time.time()
    source_blob = b""
    decoded_symbol_count = 0
    frame_idx = 0
    group_idx = 0
    symbol_idx = 0
    try:
        inflate = _load_module(pr86_dir / "inflate.py", f"pr86_inflate_reencode_{device}")
        with tempfile.TemporaryDirectory(prefix="pr86_hpac_reencode_") as tmp:
            tmpdir = Path(tmp)
            _extract_available_hpac_members(archive, tmpdir)
            meta = torch.load(tmpdir / "meta.pt", map_location="cpu", weights_only=False)
            P = int(meta.get("P", 32))
            delta = int(meta.get("delta", 2))
            ch = int(meta.get("ch", 64))
            use_spm = bool(meta.get("use_spm", False))
            hpac_d_film = int(meta.get("hpac_d_film", 32))
            total_frames = int(meta["N"])
            n_groups = int((1 + delta) * P - delta)
            frames_to_process = total_frames if max_frames is None else min(max_frames, total_frames)
            groups_to_process = n_groups if max_groups is None else min(max_groups, n_groups)
            if groups_to_process < n_groups and frames_to_process > 1:
                frames_to_process = 1
            full_stream = frames_to_process == total_frames and groups_to_process == n_groups
            scope = "full_decode_reencode" if full_stream else "bounded_decode_reencode"

            source_blob = (tmpdir / "tokens.bin").read_bytes()
            if len(source_blob) % 4 != 0:
                return {
                    "device": device,
                    "status": "failed_closed",
                    "gate": "pr86_full_decode_reencode",
                    "failure_class": "invalid_tokens_bin_word_alignment",
                    "error": "tokens.bin length is not divisible by 4 bytes for uint32 queue words",
                    "source_tokens_bytes": len(source_blob),
                    "source_tokens_sha256": sha256_bytes(source_blob),
                    "byte_exact_reencode": False,
                    "score_claim": False,
                    "planning_only": True,
                    "elapsed_sec": round(time.time() - started, 3),
                }

            gen, state_load, hpac_path = _load_hpac_generator(
                inflate=inflate,
                tmpdir=tmpdir,
                meta=meta,
                device=device,
            )
            model_pairs = int(getattr(gen, "num_pairs", total_frames))
            decoder = inflate.constriction.stream.queue.RangeDecoder(
                np.frombuffer(source_blob, dtype=np.uint32)
            )
            encoder = inflate.constriction.stream.queue.RangeEncoder()
            masks = _full_group_masks(P, delta, height, width, device)
            decoded_prev = torch.zeros((1, height, width), dtype=torch.long, device=device)
            decoded_symbol_hasher = hashlib.sha256()
            frame_reports: list[dict[str, Any]] = []

            with torch.no_grad():
                for frame_idx in range(frames_to_process):
                    idx = torch.tensor([frame_idx], dtype=torch.long, device=device)
                    cur = torch.zeros_like(decoded_prev)
                    frame_symbol_count = 0
                    for group_idx in range(groups_to_process):
                        mask = masks[group_idx]
                        if mask is None:
                            continue
                        logits = gen(cur, idx, decoded_prev)
                        probs = F.softmax(logits.float(), dim=1)
                        probs_at_group = probs[0][:, mask].permute(1, 0).contiguous()
                        probs_np = probs_at_group.detach().cpu().numpy().astype(np.float64)
                        probs_np = np.clip(probs_np, 1e-7, 1.0)
                        probs_np = probs_np / probs_np.sum(axis=1, keepdims=True)
                        decoded = np.empty(probs_np.shape[0], dtype=np.int64)
                        for symbol_idx in range(probs_np.shape[0]):
                            model = inflate.constriction.stream.model.Categorical(
                                probabilities=probs_np[symbol_idx],
                                perfect=False,
                            )
                            sym = int(decoder.decode(model))
                            encoder.encode(sym, model)
                            decoded[symbol_idx] = sym
                        decoded_u8 = decoded.astype(np.uint8)
                        decoded_symbol_hasher.update(decoded_u8.tobytes())
                        decoded_symbol_count += int(decoded_u8.size)
                        frame_symbol_count += int(decoded_u8.size)
                        cur[0, mask] = torch.from_numpy(decoded).to(device)
                    if groups_to_process == n_groups:
                        decoded_prev = cur.clone()
                    if frame_idx < 3 or frame_idx == frames_to_process - 1:
                        frame_reports.append(
                            {
                                "frame": frame_idx,
                                "groups_processed": groups_to_process,
                                "symbols": frame_symbol_count,
                            }
                        )

            reencoded_blob = encoder.get_compressed().tobytes()
            byte_exact = full_stream and reencoded_blob == source_blob
            maybe_exhausted: bool | None
            try:
                maybe_exhausted = bool(decoder.maybe_exhausted())
            except Exception:
                maybe_exhausted = None
            status = "passed" if byte_exact else ("bounded_passed" if not full_stream else "failed_closed")
            result = {
                "device": device,
                "status": status,
                "gate": "pr86_full_decode_reencode",
                "scope": scope,
                "score_claim": False,
                "planning_only": True,
                "promotable": False,
                "full_stream": full_stream,
                "byte_exact_reencode": byte_exact,
                "failure_class": None if byte_exact else ("bounded_probe_not_full_stream" if not full_stream else "byte_mismatch"),
                "source_tokens_bytes": len(source_blob),
                "source_tokens_sha256": sha256_bytes(source_blob),
                "source_tokens_word_count": len(source_blob) // 4,
                "reencoded_tokens_bytes": len(reencoded_blob),
                "reencoded_tokens_sha256": sha256_bytes(reencoded_blob),
                "reencoded_word_count": len(reencoded_blob) // 4,
                "decoded_symbol_count": decoded_symbol_count,
                "decoded_symbols_sha256": decoded_symbol_hasher.hexdigest(),
                "decoder_maybe_exhausted": maybe_exhausted,
                "frames_requested": max_frames,
                "frames_processed": frames_to_process,
                "groups_requested": max_groups,
                "groups_processed_per_frame": groups_to_process,
                "groups_per_frame_total": n_groups,
                "meta": {
                    "N": total_frames,
                    "P": P,
                    "delta": delta,
                    "ch": ch,
                    "hpac_d_film": hpac_d_film,
                    "use_spm": use_spm,
                    "ppmd_max_order": int(meta.get("ppmd_max_order", 4)),
                    "height": height,
                    "width": width,
                },
                "contract": {
                    "queue_api": "constriction.stream.queue.RangeEncoder/RangeDecoder",
                    "model_api": "constriction.stream.model.Categorical(..., perfect=False)",
                    "probability_clip_eps": "1e-7",
                    "hpac_member": hpac_path.name,
                    "hpac_model_pairs": model_pairs,
                    "state_load": state_load,
                },
                "frame_reports": frame_reports,
                "elapsed_sec": round(time.time() - started, 3),
            }
            if full_stream and not byte_exact:
                result["mismatch"] = {
                    "same_length": len(reencoded_blob) == len(source_blob),
                    "source_prefix_hex": source_blob[:32].hex(),
                    "reencoded_prefix_hex": reencoded_blob[:32].hex(),
                    "source_suffix_hex": source_blob[-32:].hex(),
                    "reencoded_suffix_hex": reencoded_blob[-32:].hex(),
                }
            return result
    except Exception as exc:
        return {
            "device": device,
            "status": "failed_closed",
            "gate": "pr86_full_decode_reencode",
            "scope": "full_decode_reencode" if max_frames is None and max_groups is None else "bounded_decode_reencode",
            "score_claim": False,
            "planning_only": True,
            "promotable": False,
            "byte_exact_reencode": False,
            "failure_class": "decode_or_reencode_error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "source_tokens_bytes": len(source_blob) if source_blob else None,
            "source_tokens_sha256": sha256_bytes(source_blob) if source_blob else None,
            "decoded_symbol_count": decoded_symbol_count,
            "failed_at": {
                "frame": frame_idx,
                "group": group_idx,
                "symbol_in_group": symbol_idx,
            },
            "elapsed_sec": round(time.time() - started, 3),
        }


def required_transfer_gates(
    *,
    token_semantics: dict[str, Any],
    device_contract: dict[str, Any],
    decode_probes: list[dict[str, Any]],
    queue_contract: dict[str, Any],
    decode_reencode_gate: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    any_decode_passed = any(row.get("status") == "passed" for row in decode_probes)
    cuda_probe = next((row for row in decode_probes if row.get("device") == "cuda"), None)
    reencode_status = "not_run"
    if decode_reencode_gate is not None:
        if decode_reencode_gate.get("status") == "passed" and decode_reencode_gate.get("byte_exact_reencode") is True:
            reencode_status = "passed"
        elif decode_reencode_gate.get("status") == "bounded_passed":
            reencode_status = "bounded_only_not_full_stream"
        elif decode_reencode_gate.get("status") == "skipped":
            reencode_status = "skipped"
        else:
            reencode_status = "failed_closed"
    return [
        {
            "gate": "pr86_archive_custody",
            "required_before_pr85_transfer": True,
            "status": "required",
            "requirement": (
                "Record archive.zip SHA-256, exact member names, member sizes, "
                "member SHA-256s, and meta.pt HPAC config."
            ),
        },
        {
            "gate": "constriction_queue_contract",
            "required_before_pr85_transfer": True,
            "status": "prefix_passed" if queue_contract.get("status") == "passed" else "blocked",
            "requirement": (
                "Pin constriction and pyppmd versions and use "
                "stream.queue.RangeEncoder/RangeDecoder over uint32 words with "
                "Categorical(..., perfect=False)."
            ),
        },
        {
            "gate": "token_semantics",
            "required_before_pr85_transfer": True,
            "status": (
                "raw_token_contract_detected"
                if token_semantics["submitted_archive_token_encoding"] == "raw_tokens"
                else "blocked_ambiguous"
            ),
            "requirement": (
                "PR85 transfer must encode/decode the same token semantics as "
                "the submitted PR86 archive. Current diagnostic classifies "
                "submitted PR86 tokens.bin as raw tokens, not residual tokens."
            ),
        },
        {
            "gate": "pr86_own_stream_decode",
            "required_before_pr85_transfer": True,
            "status": "bounded_prefix_passed" if any_decode_passed else "blocked",
            "requirement": (
                "Before transfer, PR86 tokens.bin must decode with PR86 hpac.pt.ppmd "
                "under the pinned runtime. Bounded prefix pass is diagnostic only; "
                "full-frame/full-archive decode is still required for promotion."
            ),
        },
        {
            "gate": "cpu_cuda_hpac_contract",
            "required_before_pr85_transfer": True,
            "status": (
                "cuda_untested_or_unavailable"
                if cuda_probe is None or cuda_probe.get("status") == "skipped"
                else cuda_probe.get("status", "unknown")
            ),
            "requirement": (
                "Resolve the submitted inflate.py CPU/CUDA mismatch: either force "
                "HPAC decode to CPU and prove replay parity, or prove CPU and CUDA "
                "HPAC token decode equality under the exact constriction/torch stack."
            ),
            "device_contract_note": device_contract["interpretation"],
        },
        {
            "gate": "byte_exact_reencode",
            "required_before_pr85_transfer": True,
            "status": reencode_status,
            "requirement": (
                "Decode PR86 full tokens, re-encode with the same HPAC model and "
                "queue coder, and require byte-identical tokens.bin before using "
                "the coder as a transfer template."
            ),
            "failure_class": (
                decode_reencode_gate.get("failure_class")
                if decode_reencode_gate is not None
                else None
            ),
        },
        {
            "gate": "pr85_transfer_parity",
            "required_before_pr85_transfer": True,
            "status": "not_run",
            "requirement": (
                "For PR85, prove decoded mask-token shape/range, raw-token parity "
                "against the PR85 baseline token source, archive byte closure, and "
                "runtime output parity before any exact-eval dispatch."
            ),
        },
    ]


def build_diagnostic(
    *,
    pr86_dir: Path,
    archive: Path,
    devices: list[str],
    max_frames: int,
    max_groups: int | None,
    skip_decode: bool,
    run_decode_reencode: bool = False,
    decode_reencode_device: str = "cpu",
    decode_reencode_max_frames: int | None = None,
    decode_reencode_max_groups: int | None = None,
) -> dict[str, Any]:
    token_semantics = analyze_token_semantics(pr86_dir)
    device_contract = analyze_inflate_device_contract(pr86_dir)
    probability_contract = analyze_probability_contract(pr86_dir)
    queue_contract = constriction_queue_self_test()
    custody = archive_custody(archive)
    decode_probes: list[dict[str, Any]] = []
    if skip_decode:
        decode_probes.append({"status": "skipped", "skip_reason": "--skip-decode"})
    else:
        for device in devices:
            decode_probes.append(
                decode_pr86_prefix(
                    archive=archive,
                    pr86_dir=pr86_dir,
                    device=device,
                    max_frames=max_frames,
                    max_groups=max_groups,
                )
            )

    decode_reencode_gate: dict[str, Any] | None = None
    if run_decode_reencode:
        decode_reencode_gate = decode_reencode_pr86_tokens(
            archive=archive,
            pr86_dir=pr86_dir,
            device=decode_reencode_device,
            max_frames=decode_reencode_max_frames,
            max_groups=decode_reencode_max_groups,
        )

    gates = required_transfer_gates(
        token_semantics=token_semantics,
        device_contract=device_contract,
        decode_probes=decode_probes,
        queue_contract=queue_contract,
        decode_reencode_gate=decode_reencode_gate,
    )
    own_stream_status = "blocked"
    if any(row.get("status") == "passed" for row in decode_probes):
        own_stream_status = "bounded_prefix_decodes"
    elif any(row.get("status") == "failed" for row in decode_probes):
        own_stream_status = "decode_failed"

    return {
        "schema_version": 1,
        "tool": "pr86_hpac_replay_parity",
        "recorded_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "score_claim": False,
        "dispatch_performed": False,
        "planning_only": True,
        "promotable": False,
        "evidence_grade": (
            "planning_only_local_decode_reencode_gate"
            if run_decode_reencode
            else "planning_only_local_bounded_decode"
        ),
        "archive": custody,
        "dependencies": {
            "python": sys.version.split()[0],
            "torch": package_version("torch"),
            "numpy": package_version("numpy"),
            "constriction": package_version("constriction"),
            "pyppmd": package_version("pyppmd"),
        },
        "token_semantics": token_semantics,
        "inflate_device_contract": device_contract,
        "probability_model_contract": probability_contract,
        "constriction_queue_contract": queue_contract,
        "decode_probes": decode_probes,
        "full_decode_reencode_gate": decode_reencode_gate
        if decode_reencode_gate is not None
        else {
            "status": "not_run",
            "gate": "pr86_full_decode_reencode",
            "score_claim": False,
            "planning_only": True,
            "requirement": "pass --run-decode-reencode to execute the full byte-exact token parity gate",
        },
        "required_gates_before_hpac_transfer_to_pr85": gates,
        "conclusions": {
            "submitted_archive_token_encoding": token_semantics[
                "submitted_archive_token_encoding"
            ],
            "training_objective": token_semantics["training_objective"],
            "own_stream_decode_status": own_stream_status,
            "full_decode_reencode_status": (
                decode_reencode_gate.get("status")
                if decode_reencode_gate is not None
                else "not_run"
            ),
            "cpu_cuda_contract_status": (
                "mismatch_needs_resolution"
                if device_contract["comment_code_mismatch"]
                else "no_mismatch_detected"
            ),
            "pr85_transfer_status": (
                "blocked_until_full PR86 own-stream decode/reencode parity, "
                "CPU/CUDA contract closure, and PR85 raw-token transfer parity "
                "are proven"
            ),
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr86-dir", type=Path, default=DEFAULT_PR86_DIR)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument(
        "--devices",
        default="cpu",
        help="Comma-separated devices to probe. CUDA is skipped if unavailable.",
    )
    parser.add_argument("--max-frames", type=int, default=1)
    parser.add_argument(
        "--max-groups",
        type=int,
        default=4,
        help="Groups per frame to decode; omit via --full-frame-prefix for all groups.",
    )
    parser.add_argument(
        "--full-frame-prefix",
        action="store_true",
        help="Decode all HPAC groups for each requested frame.",
    )
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument(
        "--run-decode-reencode",
        action="store_true",
        help=(
            "Run the PR86 decode->encode parity gate. With no decode-reencode "
            "limits this attempts the full tokens.bin byte-exact gate."
        ),
    )
    parser.add_argument(
        "--decode-reencode-device",
        default="cpu",
        help="Device for the decode->encode parity gate.",
    )
    parser.add_argument(
        "--decode-reencode-max-frames",
        type=int,
        default=None,
        help="Diagnostic limit for decode->encode; unset means full PR86 frame count.",
    )
    parser.add_argument(
        "--decode-reencode-max-groups",
        type=int,
        default=None,
        help="Diagnostic limit for decode->encode; unset means full HPAC group count.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    pr86_dir = args.pr86_dir.resolve()
    archive = args.archive.resolve()
    devices = [part.strip() for part in args.devices.split(",") if part.strip()]
    max_groups = None if args.full_frame_prefix else args.max_groups
    payload = build_diagnostic(
        pr86_dir=pr86_dir,
        archive=archive,
        devices=devices,
        max_frames=args.max_frames,
        max_groups=max_groups,
        skip_decode=args.skip_decode,
        run_decode_reencode=args.run_decode_reencode,
        decode_reencode_device=args.decode_reencode_device,
        decode_reencode_max_frames=args.decode_reencode_max_frames,
        decode_reencode_max_groups=args.decode_reencode_max_groups,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
