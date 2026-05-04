"""Latent-correction sidecar for hnerv_repack_latent.

Supports PR99's original brotli'd two-byte-per-pair wire format plus local
lossless candidate formats used for no-dispatch byte screens.
"""
import struct
import numpy as np

DELTA_SCALE = 0.01


def _unpack_bits(data, count, width):
    acc = 0
    bits = 0
    idx = 0
    out = []
    mask = (1 << width) - 1
    for _ in range(count):
        while bits < width:
            if idx >= len(data):
                raise RuntimeError("bitstream underflow")
            acc |= data[idx] << bits
            idx += 1
            bits += 8
        out.append(acc & mask)
        acc >>= width
        bits -= width
    return out


def decode_corrections(blob):
    import brotli
    if blob.startswith(b"LC2S"):
        n, dim_len = struct.unpack_from("<HH", blob, 4)
        dim_b = blob[8:8 + dim_len]
        delta_b = blob[8 + dim_len:]
        dim = np.frombuffer(brotli.decompress(dim_b), dtype=np.uint8).copy()
        delta_q = np.frombuffer(brotli.decompress(delta_b), dtype=np.uint8).view(np.int8).copy()
        if len(dim) != n or len(delta_q) != n:
            raise RuntimeError("LC2S length mismatch")
        return dim, delta_q
    if blob.startswith(b"LCBP"):
        raw = brotli.decompress(blob[4:])
        n, bitmap_len, dim_bits_len = struct.unpack_from("<HHH", raw, 0)
        cursor = 6
        bitmap = raw[cursor:cursor + bitmap_len]
        cursor += bitmap_len
        dim_bits = raw[cursor:cursor + dim_bits_len]
        cursor += dim_bits_len
        delta_q = np.frombuffer(raw[cursor:cursor + n], dtype=np.uint8).view(np.int8).copy()
        corrected = np.unpackbits(np.frombuffer(bitmap, dtype=np.uint8), bitorder="little")[:n].astype(bool)
        dims = _unpack_bits(dim_bits, int(corrected.sum()), 5)
        dim = np.full(n, 255, dtype=np.uint8)
        dim[corrected] = np.array(dims, dtype=np.uint8)
        if len(delta_q) != n:
            raise RuntimeError("LCBP delta length mismatch")
        return dim, delta_q
    if blob.startswith(b"LCSP"):
        raw = brotli.decompress(blob[4:])
        n, k = struct.unpack_from("<HH", raw, 0)
        cursor = 4
        idx = np.frombuffer(raw[cursor:cursor + 2 * k], dtype=np.uint16)
        cursor += 2 * k
        vals_dim = np.frombuffer(raw[cursor:cursor + k], dtype=np.uint8)
        cursor += k
        vals_delta = np.frombuffer(raw[cursor:cursor + k], dtype=np.uint8).view(np.int8)
        dim = np.full(n, 255, dtype=np.uint8)
        delta_q = np.zeros(n, dtype=np.int8)
        dim[idx] = vals_dim
        delta_q[idx] = vals_delta
        return dim, delta_q
    raw = brotli.decompress(blob)
    n = struct.unpack_from("<H", raw, 0)[0]
    arr = np.frombuffer(raw[2:2 + 2*n], dtype=np.uint8).reshape(n, 2)
    dim = arr[:, 0]
    delta_q = arr[:, 1].view(np.int8)
    return dim, delta_q


def encode_corrections(out_dim, out_delta_q):
    import brotli
    n = len(out_dim)
    assert len(out_delta_q) == n
    dim_packed = np.where(out_delta_q == 0, 255, out_dim).astype(np.uint8)
    payload = struct.pack("<H", n) + np.stack([dim_packed, out_delta_q.astype(np.int8).view(np.uint8)], axis=1).tobytes()
    return brotli.compress(payload, quality=11)


def apply_corrections(latents_tensor, dim_arr, delta_q_arr, scale=DELTA_SCALE):
    n = latents_tensor.shape[0]
    for p in range(n):
        d = int(dim_arr[p])
        if d == 255:
            continue
        latents_tensor[p, d] = latents_tensor[p, d] + float(delta_q_arr[p]) * scale
    return latents_tensor
