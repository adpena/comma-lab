"""Sidecar decode + apply for H3 inflate.

Wire format (single blob, xz-compressed):
  header b"BPGD" | u16 n_pairs | per-pair record:
    pi (delta or 0xFF + u16)
    flags u8 (bit0=x2, bit1=cmaes, bit2=pattern, bit3=pose, bit4=warp)
    [if x2/cmaes/pattern: u8 n then n * 3-byte packed (x:9 y:9 c:3 [pid:3])]
    [if pose: target_dims * i8]
    [if warp: i8 qx, i8 qy]

Decoder applies (in order):
  - x2 patches: masks[pi, y:y+2, x:x+2] = c
  - pattern patches: masks[pi, y:y+ph, x:x+pw] = c (PATTERN_SIZES below)
  - cmaes patches: masks[pi, y, x] = c
  - pose deltas: pose[pi, dim] += int8 * pose_scale[dim]
  - frame-1 warps (post-generation): grid_sample translate by (qx/qscale, qy/qscale)
"""
import lzma
import struct

import numpy as np
import torch
import torch.nn.functional as F


PATTERN_SIZES = {0: (1, 1), 1: (3, 3), 2: (1, 4), 3: (4, 1), 4: (2, 2)}
POSE_SCALE = (0.001, 0.005, 0.005, 0.001, 0.001, 0.005)
DEFAULT_TARGET_DIMS = (1, 2, 5)
WARP_QSCALE = 10.0  # must match qscale used at encode (v2_unified_warp_only uses 10)
MASK_H, MASK_W = 384, 512  # generator input mask shape (after our codec decodes back)


def decode_grouped_bitpack(raw, target_dims=DEFAULT_TARGET_DIMS):
    if raw[:4] != b"BPGD":
        raise ValueError(f"bad sidecar header: {raw[:4]!r}")
    pos = 4
    n_pairs = struct.unpack_from("<H", raw, pos)[0]; pos += 2
    out = {"x2": {}, "cmaes": {}, "pattern": {}, "pose": {}, "warp": {}}
    prev = 0
    for idx in range(n_pairs):
        if idx == 0:
            pi = struct.unpack_from("<H", raw, pos)[0]; pos += 2
        else:
            d = raw[pos]; pos += 1
            if d == 0xFF:
                pi = struct.unpack_from("<H", raw, pos)[0]; pos += 2
            else:
                pi = prev + d
        prev = pi
        flags = raw[pos]; pos += 1
        if flags & 1:
            n = raw[pos]; pos += 1
            ps = []
            for _ in range(n):
                v = int.from_bytes(raw[pos:pos+3], "little"); pos += 3
                ps.append((v & 0x1FF, (v >> 9) & 0x1FF, (v >> 18) & 0x7))
            out["x2"][pi] = ps
        if flags & 2:
            n = raw[pos]; pos += 1
            ps = []
            for _ in range(n):
                v = int.from_bytes(raw[pos:pos+3], "little"); pos += 3
                ps.append((v & 0x1FF, (v >> 9) & 0x1FF, (v >> 18) & 0x7))
            out["cmaes"][pi] = ps
        if flags & 4:
            n = raw[pos]; pos += 1
            ps = []
            for _ in range(n):
                v = int.from_bytes(raw[pos:pos+3], "little"); pos += 3
                ps.append((v & 0x1FF, (v >> 9) & 0x1FF, (v >> 21) & 0x7, (v >> 18) & 0x7))
            out["pattern"][pi] = ps
        if flags & 8:
            vals = []
            for _ in target_dims:
                vals.append(struct.unpack_from("<b", raw, pos)[0]); pos += 1
            out["pose"][pi] = tuple(vals)
        if flags & 16:
            qx, qy = struct.unpack_from("<bb", raw, pos); pos += 2
            out["warp"][pi] = (qx, qy)
    if pos != len(raw):
        raise ValueError(f"sidecar trailing bytes: pos={pos} len={len(raw)}")
    return out


def decode_sidecar_blob(blob, target_dims=DEFAULT_TARGET_DIMS):
    """xz-decompress then bit-unpack."""
    raw = lzma.decompress(blob, format=lzma.FORMAT_XZ)
    return decode_grouped_bitpack(raw, target_dims=target_dims)


def apply_mask_edits(masks, sidecar):
    """In-place edit masks tensor (n_pairs, H, W) long. x2 then pattern then cmaes."""
    H, W = masks.shape[1], masks.shape[2]
    for pi, ps in sidecar["x2"].items():
        for x, y, c in ps:
            x2 = min(x + 2, W); y2 = min(y + 2, H)
            masks[pi, y:y2, x:x2] = c
    for pi, ps in sidecar["pattern"].items():
        for x, y, p_id, c in ps:
            ph, pw = PATTERN_SIZES[int(p_id)]
            masks[pi, y:min(y + ph, H), x:min(x + pw, W)] = c
    for pi, ps in sidecar["cmaes"].items():
        for x, y, c in ps:
            masks[pi, y, x] = c


def apply_pose_deltas(poses, sidecar, target_dims=DEFAULT_TARGET_DIMS):
    """In-place add int8 * scale to selected pose dims."""
    for pi, vals in sidecar["pose"].items():
        for dim, v in zip(target_dims, vals):
            poses[pi, dim] = poses[pi, dim] + float(v) * POSE_SCALE[dim]


def _base_grid(h, w, device):
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    x = (xx + 0.5) * (2.0 / w) - 1.0
    y = (yy + 0.5) * (2.0 / h) - 1.0
    return torch.stack([x, y], dim=-1)


def apply_warps_to_f1_inplace(out_arr, sidecar, qscale=WARP_QSCALE, device=None, batch_size=8):
    """out_arr: ndarray (n_pairs, 2, H, W, 3) uint8 — overwrites out_arr[pi, 0] for warped pairs.

    qx, qy are int8 in pixel units of the GENERATOR-internal frame. For the final
    upsampled output (H, W), we scale by sx=W/MODEL_W, sy=H/MODEL_H — but since
    the warp was searched in OUT_W/OUT_H pixel units already, sx=sy=1 here when
    qscale is in OUT pixels. v2_warp_probe uses sx=OUT_W/MODEL_W=1164/512≈2.27,
    sy=OUT_H/MODEL_H=874/384≈2.28 so we replicate that.
    """
    active = [(int(pi), int(qx), int(qy)) for pi, (qx, qy) in sidecar["warp"].items()
              if (qx or qy)]
    if not active:
        return
    n_pairs, _, H, W, _ = out_arr.shape
    MODEL_H, MODEL_W = 384, 512  # generator output before upsample
    sx = W / MODEL_W
    sy = H / MODEL_H
    grid = _base_grid(H, W, device)
    with torch.inference_mode():
        for start in range(0, len(active), batch_size):
            chunk = active[start:start + batch_size]
            pair_ids = [c[0] for c in chunk]
            params = [(c[1] / qscale * sx, c[2] / qscale * sy) for c in chunk]
            # f1 frames pulled into a torch tensor (B, H, W, 3) -> (B, 3, H, W)
            np_batch = out_arr[pair_ids, 0]  # (B, H, W, 3)
            batch = torch.from_numpy(np_batch).to(device).float().permute(0, 3, 1, 2)
            b = batch.shape[0]
            offsets = torch.tensor(params, device=device, dtype=torch.float32)
            g = grid.unsqueeze(0).expand(b, -1, -1, -1).clone()
            g[..., 0] -= offsets[:, 0].view(b, 1, 1) * (2.0 / W)
            g[..., 1] -= offsets[:, 1].view(b, 1, 1) * (2.0 / H)
            warped = F.grid_sample(
                batch, g, mode="bilinear", padding_mode="reflection", align_corners=False
            )
            warped_np = warped.clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for k, pi in enumerate(pair_ids):
                out_arr[pi, 0] = warped_np[k]
