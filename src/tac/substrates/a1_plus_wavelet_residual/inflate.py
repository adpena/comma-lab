# SPDX-License-Identifier: MIT
"""Contest-compliant inflate runtime for A1 + wavelet residual composition.

Reads ``archive_dir/x`` (or 0.bin / archive.zip), splits it into
(a1_bytes, wavelet_sidecar_bytes), runs A1's existing decoder over the 600
pair indices, then applies the wavelet residual ONLY at selected pair
indices.  The residual is a single-level DB4 IDWT reconstruction at the
foveal patch (LL = 0 — A1 carries the approximation; only detail bands).

NO scorer code is imported per CLAUDE.md strict-scorer-rule.  Per Catalog
#146 the inflate.py is ≤ 200 LOC of substantive code (substrate-engineering
exemption per HNeRV parity lesson L4).

Wire format reference: ``tac.substrates.a1_plus_wavelet_residual.archive``.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path
from typing import Any


def _vendored_a1_root(here: Path) -> Path:
    """Resolve the directory containing the A1 codec.py + model.py vendor."""
    candidate = here / "src" / "a1"
    if candidate.is_dir():
        return candidate
    candidate = here / "src"
    if (candidate / "codec.py").is_file() and (candidate / "model.py").is_file():
        return candidate
    for parent in (here, *here.parents):
        candidate = parent / "src"
        if (candidate / "codec.py").is_file() and (candidate / "model.py").is_file():
            return candidate
        if (parent / "codec.py").is_file() and (parent / "model.py").is_file():
            return parent
    raise FileNotFoundError(
        f"A1 vendored modules not found near {here}; expected src/a1/ or "
        "an ancestor src/codec.py + src/model.py"
    )


def inflate_one(src_bin: Path, dst_raw: Path) -> int:
    """Inflate one A1+wavelet composition archive into a .raw frame stream.

    Returns the number of frames written.
    """
    import torch
    import torch.nn.functional as F

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(_vendored_a1_root(here)))

    from codec import (  # type: ignore[import-not-found]
        LATENT_BLOB_LEN,
        apply_latent_sidecar,
        decode_decoder_compact,
        decode_latents_compact,
    )
    from model import HNeRVDecoder  # type: ignore[import-not-found]

    from tac.substrates.a1_plus_wavelet_residual.architecture import (
        A1_BASE_CHANNELS,
        A1_CAMERA_H,
        A1_CAMERA_W,
        A1_EVAL_H,
        A1_EVAL_W,
        A1_LATENT_DIM,
        A1_N_PAIRS,
        _db4_idwt_single_level,
    )
    from tac.substrates.a1_plus_wavelet_residual.archive import (
        WAVELET_SIDECAR_MAGIC,
        decode_wavelet_sidecar,
        split_composition_archive,
    )

    archive_bytes = src_bin.read_bytes()
    a1_bytes, wav_bytes = split_composition_archive(archive_bytes)

    # A1 parse (mirrors track4_sg_a1_t178000_20260509 submission_dir/inflate.py).
    if len(a1_bytes) < 4:
        raise ValueError("a1 base section too short")
    section_total = struct.unpack_from("<I", a1_bytes, 0)[0]
    decoder_blob = a1_bytes[4:section_total]
    latent_blob = a1_bytes[section_total : section_total + LATENT_BLOB_LEN]
    a1_sidecar_blob = a1_bytes[section_total + LATENT_BLOB_LEN :]
    decoder_sd = decode_decoder_compact(decoder_blob)
    latents = apply_latent_sidecar(decode_latents_compact(latent_blob), a1_sidecar_blob)

    # Wavelet parse (or no-op if sidecar absent).
    wav_indices: tuple[int, ...] = ()
    wav_coeffs = None
    wav_meta: dict[str, Any] = {}
    if wav_bytes and wav_bytes[:4] == WAVELET_SIDECAR_MAGIC:
        wav_indices, wav_coeffs, wav_meta = decode_wavelet_sidecar(wav_bytes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = HNeRVDecoder(
        latent_dim=A1_LATENT_DIM,
        base_channels=A1_BASE_CHANNELS,
        eval_size=(A1_EVAL_H, A1_EVAL_W),
    ).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()

    pair_to_slot = {int(p): i for i, p in enumerate(wav_indices)}
    fov_h = int(wav_meta.get("foveal_h", 0))
    fov_w = int(wav_meta.get("foveal_w", 0))
    rank = int(wav_meta.get("coeff_rank", 0))
    # Foveal patch at CAMERA-native resolution (after IDWT)
    full_h = 2 * fov_h
    full_w = 2 * fov_w
    fov_top = max(0, A1_CAMERA_H // 2 - full_h // 2)
    fov_left = max(0, A1_CAMERA_W // 2 - full_w // 2)

    latents = latents.to(device)
    if wav_coeffs is not None:
        wav_coeffs = wav_coeffs.to(device)
    n = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(0, A1_N_PAIRS, 16):
            j = min(i + 16, A1_N_PAIRS)
            batch = j - i
            decoded = decoder(latents[i:j])
            flat = decoded.reshape(batch * 2, 3, A1_EVAL_H, A1_EVAL_W)
            up = F.interpolate(
                flat,
                size=(A1_CAMERA_H, A1_CAMERA_W),
                mode="bicubic",
                align_corners=False,
            )
            up = up.reshape(batch, 2, 3, A1_CAMERA_H, A1_CAMERA_W)
            # A1's canonical bias correction (verbatim from A1 inflate.py).
            up[:, 0, 0].sub_(1.0)
            up[:, 0, 2].sub_(1.0)
            up[:, 1, 1].sub_(1.0)
            # Wavelet composition overlay at the foveal patch (selected pairs only).
            if wav_coeffs is not None and rank > 0 and fov_h > 0 and fov_w > 0:
                for b in range(batch):
                    pair_id = i + b
                    if pair_id not in pair_to_slot:
                        continue
                    slot = pair_to_slot[pair_id]
                    for fi in (0, 1):
                        # Build LL=0, LH/HL/HH from per-pair coefficients
                        ll = torch.zeros(3, fov_h, fov_w, device=device)
                        bands = []
                        for band_idx in (0, 1, 2):  # LH, HL, HH
                            uv = wav_coeffs[slot, band_idx, fi]  # (3, rank, fh+fw)
                            u = uv[..., :fov_h]  # (3, rank, fh)
                            v = uv[..., fov_h:]  # (3, rank, fw)
                            bands.append(torch.einsum("ckp,ckw->cpw", u, v))
                        resid = _db4_idwt_single_level(ll, bands[0], bands[1], bands[2])
                        # Add to camera-native foveal patch
                        up[
                            b,
                            fi,
                            :,
                            fov_top : fov_top + full_h,
                            fov_left : fov_left + full_w,
                        ] += resid
            frames = (
                up.reshape(batch * 2, 3, A1_CAMERA_H, A1_CAMERA_W)
                .clamp(0, 255)
                .permute(0, 2, 3, 1)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            fout.write(frames.tobytes())
            n += batch * 2
    return n


def main_cli() -> int:
    if len(sys.argv) < 4:
        print(
            "Usage: inflate.py <archive_dir> <output_dir> <file_list>",
            file=sys.stderr,
        )
        return 2
    archive_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    file_list = Path(sys.argv[3])
    output_dir.mkdir(parents=True, exist_ok=True)
    src_x = archive_dir / "x"
    for line in file_list.read_text().splitlines():
        base = line.split(".")[0]
        if not base:
            continue
        src = src_x if src_x.is_file() else (archive_dir / f"{base}.bin")
        dst = output_dir / f"{base}.raw"
        inflate_one(src, dst)
        print(f"inflated {base}")
    return 0

if __name__ == "__main__":
    sys.exit(main_cli())
