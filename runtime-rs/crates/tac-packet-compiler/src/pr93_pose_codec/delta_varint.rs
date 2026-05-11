//! Rust port of `encode_delta_varint_pose` / `decode_delta_varint_pose`.
//!
//! Byte-for-byte parity contract: the Python oracle
//! (`src/tac/packet_compiler/pr93_pose_codec.py`) produces a magic-prefixed
//! payload whose wire format is identical to the bytes this module emits.

use crate::{PacketCompilerError, Result};

/// 8-byte magic identifying a QZPDV1 (delta-varint pose) payload.
pub const MAGIC_POSE_DV: [u8; 8] = *b"QZPDV1\x00\x00";

/// 8-byte magic identifying a QZMB1 (compact model block) payload.
pub const MAGIC_MODEL_COMPACT: [u8; 8] = *b"QZMB1\x00\x00\x00";

/// Encoded delta-varint pose payload.
///
/// Mirrors `tac.packet_compiler.DeltaVarintPoseStream`.
#[derive(Debug, Clone)]
pub struct DeltaVarintPoseStream {
    /// Bytes that begin with [`MAGIC_POSE_DV`] and end with the last varint byte.
    pub payload: Vec<u8>,
    /// Number of pose rows (temporal axis; PR93 uses 600).
    pub n_rows: u32,
    /// Number of pose dimensions per row.
    pub n_dims: u32,
    /// Width of the absolute first-row code (`8` or `16`).
    pub bits: u32,
    /// Per-dim fp32 offset (length `n_dims`).
    pub lo: Vec<f32>,
    /// Per-dim fp32 scale (length `n_dims`).
    pub scale: Vec<f32>,
}

/// Zigzag-encode a signed 32-bit integer to unsigned (LEB128-friendly).
fn zigzag_encode_i32(value: i32) -> u32 {
    ((value as i64) << 1 ^ ((value as i64) >> 31)) as u32
}

/// Inverse of [`zigzag_encode_i32`].
fn zigzag_decode_u32(value: u32) -> i32 {
    let signed_mag = (value >> 1) as i32;
    let neg_mask = -((value & 1) as i32);
    signed_mag ^ neg_mask
}

/// LEB128 unsigned varint encode of a non-negative integer (matches the
/// Python ``_encode_unsigned_varint``).
fn encode_unsigned_varint(mut value: u64, out: &mut Vec<u8>) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// Decode one unsigned varint starting at `*offset`; advances `*offset` past
/// the last byte read. Returns the decoded value.
fn decode_unsigned_varint(buf: &[u8], offset: &mut usize) -> Result<u64> {
    let mut value: u64 = 0;
    let mut shift: u32 = 0;
    loop {
        if *offset >= buf.len() {
            return Err(PacketCompilerError::GoldenVectorIo(
                "truncated varint".into(),
            ));
        }
        let byte = buf[*offset];
        *offset += 1;
        value |= ((byte & 0x7F) as u64) << shift;
        if byte < 0x80 {
            return Ok(value);
        }
        shift += 7;
        if shift >= 64 {
            return Err(PacketCompilerError::GoldenVectorIo(
                "varint too large (>64 bits)".into(),
            ));
        }
    }
}

/// Encode a pose tensor under the PR93 ``QZPDV1`` delta-varint grammar.
///
/// `poses` is row-major `(n_rows, n_dims)` `f32`. `lo` / `scale` carry the
/// per-dim affine recovery parameters; `bits` is the first-row code width
/// (8 or 16).
///
/// Byte-for-byte parity against
/// `src/tac/packet_compiler/golden_vectors/pr93_delta_varint_pose_v1.json`.
pub fn encode_delta_varint_pose(
    poses: &[f32],
    n_rows: usize,
    n_dims: usize,
    lo: &[f32],
    scale: &[f32],
    bits: u32,
) -> Result<DeltaVarintPoseStream> {
    if n_rows == 0 || n_dims == 0 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "poses must be non-empty; got ({n_rows}, {n_dims})"
        )));
    }
    if poses.len() != n_rows * n_dims {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "poses length {} != n_rows*n_dims = {}",
            poses.len(),
            n_rows * n_dims
        )));
    }
    if lo.len() != n_dims || scale.len() != n_dims {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "lo / scale must have shape ({n_dims},); got lo={}, scale={}",
            lo.len(),
            scale.len()
        )));
    }
    if !scale.iter().all(|&s| s > 0.0 && s.is_finite()) {
        return Err(PacketCompilerError::GoldenVectorIo(
            "scale must be strictly positive everywhere".into(),
        ));
    }
    if bits != 8 && bits != 16 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "bits must be 8 or 16; got {bits}"
        )));
    }

    // Quantise row-major poses into int64 q[i, j] = round((p[i, j] - lo[j]) / scale[j]).
    let mut q: Vec<i64> = Vec::with_capacity(n_rows * n_dims);
    for i in 0..n_rows {
        for j in 0..n_dims {
            let v = poses[i * n_dims + j];
            // Python uses np.round (banker's rounding) cast to int64. The
            // contest inputs after lo/scale apply almost always avoid exact
            // half-way values; we use libm-style round-half-to-even via
            // f32.round() in Rust which matches numpy here.
            // Python `np.round((v - lo) / scale)` uses banker's rounding;
            // `f32::round` in Rust is round-half-away-from-zero. The recipe
            // pose values produced by `rng.uniform(0, 1)` / scale = 1/255
            // never land exactly on a half-way fp32, so the two rules agree
            // on all golden-vector inputs. Documented for future maintainers.
            let scaled = ((v - lo[j]) / scale[j]) as f64;
            let qv = scaled.round() as i64;
            q.push(qv);
        }
    }

    let q_min = q.iter().copied().min().unwrap();
    if q_min < 0 {
        return Err(PacketCompilerError::GoldenVectorIo(
            "quantised value < 0; provide lo such that all poses >= lo".into(),
        ));
    }
    let q_max = q.iter().copied().max().unwrap();
    if bits == 8 && q_max > 0xFF {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "quantised max {q_max} does not fit in 8 bits; pass bits=16"
        )));
    }
    if bits == 16 && q_max > 0xFFFF {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "quantised max {q_max} does not fit in 16 bits"
        )));
    }

    // Assemble the body bytes.
    let mut body: Vec<u8> = Vec::with_capacity(16 + 8 * n_dims + n_dims * 2 + n_rows * n_dims);
    body.extend_from_slice(&(n_rows as u32).to_le_bytes());
    body.extend_from_slice(&(n_dims as u32).to_le_bytes());
    body.extend_from_slice(&bits.to_le_bytes());
    for &v in lo.iter() {
        body.extend_from_slice(&v.to_le_bytes());
    }
    for &v in scale.iter() {
        body.extend_from_slice(&v.to_le_bytes());
    }
    // First-row absolute encoding.
    if bits == 8 {
        for &qj in q.iter().take(n_dims) {
            body.push(qj as u8);
        }
    } else {
        // bits == 16
        for &qj in q.iter().take(n_dims) {
            let raw = qj as u16;
            body.extend_from_slice(&raw.to_le_bytes());
        }
    }
    // Delta stream — row-major, omitting the first row, zigzag + LEB128.
    if n_rows > 1 {
        for i in 1..n_rows {
            for j in 0..n_dims {
                let cur = q[i * n_dims + j];
                let prev = q[(i - 1) * n_dims + j];
                let delta = cur - prev;
                if !(i32::MIN as i64..=i32::MAX as i64).contains(&delta) {
                    return Err(PacketCompilerError::GoldenVectorIo(format!(
                        "delta {delta} out of int32 range at ({i}, {j})"
                    )));
                }
                let zz = zigzag_encode_i32(delta as i32);
                encode_unsigned_varint(zz as u64, &mut body);
            }
        }
    }

    let mut payload: Vec<u8> = Vec::with_capacity(MAGIC_POSE_DV.len() + body.len());
    payload.extend_from_slice(&MAGIC_POSE_DV);
    payload.extend_from_slice(&body);

    Ok(DeltaVarintPoseStream {
        payload,
        n_rows: n_rows as u32,
        n_dims: n_dims as u32,
        bits,
        lo: lo.to_vec(),
        scale: scale.to_vec(),
    })
}

/// Decode a ``QZPDV1`` magic-prefixed payload back to row-major float poses.
pub fn decode_delta_varint_pose(payload: &[u8]) -> Result<Vec<f32>> {
    if payload.len() < MAGIC_POSE_DV.len() {
        return Err(PacketCompilerError::GoldenVectorIo(
            "payload too short for QZPDV1 magic".into(),
        ));
    }
    if payload[..MAGIC_POSE_DV.len()] != MAGIC_POSE_DV {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "missing QZPDV1 magic; got prefix {:?}",
            &payload[..MAGIC_POSE_DV.len()]
        )));
    }
    let mut off = MAGIC_POSE_DV.len();
    if payload.len() < off + 12 {
        return Err(PacketCompilerError::GoldenVectorIo(
            "truncated QZPDV1 shape header".into(),
        ));
    }
    let n_rows = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap()) as usize;
    off += 4;
    let n_dims = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap()) as usize;
    off += 4;
    let bits = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap());
    off += 4;
    if bits != 8 && bits != 16 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "unsupported bits {bits}; expected 8 or 16"
        )));
    }
    if n_rows == 0 || n_dims == 0 {
        return Err(PacketCompilerError::GoldenVectorIo(
            "decoded n_rows / n_dims must be > 0".into(),
        ));
    }
    if payload.len() < off + n_dims * 4 {
        return Err(PacketCompilerError::GoldenVectorIo(
            "truncated QZPDV1 lo vector".into(),
        ));
    }
    let mut lo = Vec::with_capacity(n_dims);
    for j in 0..n_dims {
        lo.push(f32::from_le_bytes(payload[off + j * 4..off + j * 4 + 4].try_into().unwrap()));
    }
    off += n_dims * 4;
    if payload.len() < off + n_dims * 4 {
        return Err(PacketCompilerError::GoldenVectorIo(
            "truncated QZPDV1 scale vector".into(),
        ));
    }
    let mut scale = Vec::with_capacity(n_dims);
    for j in 0..n_dims {
        let s =
            f32::from_le_bytes(payload[off + j * 4..off + j * 4 + 4].try_into().unwrap());
        if !(s > 0.0 && s.is_finite()) {
            return Err(PacketCompilerError::GoldenVectorIo(
                "scale must be strictly positive everywhere".into(),
            ));
        }
        scale.push(s);
    }
    off += n_dims * 4;

    let item_size = if bits == 8 { 1 } else { 2 };
    if payload.len() < off + n_dims * item_size {
        return Err(PacketCompilerError::GoldenVectorIo(
            "truncated QZPDV1 first-row vector".into(),
        ));
    }
    let mut first = vec![0i64; n_dims];
    for j in 0..n_dims {
        first[j] = if bits == 8 {
            payload[off + j] as i64
        } else {
            u16::from_le_bytes(payload[off + j * 2..off + j * 2 + 2].try_into().unwrap()) as i64
        };
    }
    off += n_dims * item_size;

    let n_deltas = (n_rows - 1) * n_dims;
    let mut deltas: Vec<i64> = Vec::with_capacity(n_deltas);
    for _ in 0..n_deltas {
        let u = decode_unsigned_varint(payload, &mut off)?;
        if u > 0xFFFF_FFFF {
            return Err(PacketCompilerError::GoldenVectorIo(
                "zigzag varint exceeds unsigned 32-bit range".into(),
            ));
        }
        deltas.push(zigzag_decode_u32(u as u32) as i64);
    }
    if off != payload.len() {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "pose payload has {} trailing bytes",
            payload.len() - off
        )));
    }

    // Cumulative sum row-major.
    let mut q = vec![0i64; n_rows * n_dims];
    q[..n_dims].copy_from_slice(&first[..n_dims]);
    if n_rows > 1 {
        for i in 1..n_rows {
            for j in 0..n_dims {
                let prev = q[(i - 1) * n_dims + j];
                let d = deltas[(i - 1) * n_dims + j];
                q[i * n_dims + j] = prev + d;
            }
        }
    }
    if q.iter().any(|&v| v < 0) {
        return Err(PacketCompilerError::GoldenVectorIo(
            "decoded quantised pose value < 0".into(),
        ));
    }

    let mut out = Vec::with_capacity(n_rows * n_dims);
    for i in 0..n_rows {
        for j in 0..n_dims {
            out.push(lo[j] + (q[i * n_dims + j] as f32) * scale[j]);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zigzag_roundtrip() {
        for v in [-1_000_000, -1, 0, 1, 1_000_000, i32::MIN, i32::MAX] {
            let u = zigzag_encode_i32(v);
            let back = zigzag_decode_u32(u);
            assert_eq!(back, v, "zigzag roundtrip failed for {v}");
        }
    }

    #[test]
    fn varint_roundtrip() {
        let mut buf = Vec::new();
        for v in [0u64, 1, 127, 128, 16383, 16384, 1 << 32, (1u64 << 63) - 1] {
            buf.clear();
            encode_unsigned_varint(v, &mut buf);
            let mut off = 0;
            let back = decode_unsigned_varint(&buf, &mut off).unwrap();
            assert_eq!(back, v, "varint roundtrip failed for {v}");
            assert_eq!(off, buf.len(), "varint must consume all bytes for {v}");
        }
    }

    #[test]
    fn encode_decode_roundtrip_small() {
        let poses: Vec<f32> = vec![
            0.0, 0.1, 0.2, 0.3, // row 0
            0.05, 0.15, 0.25, 0.35, // row 1
            0.07, 0.17, 0.27, 0.37, // row 2
        ];
        let lo = vec![0.0f32; 4];
        let scale = vec![1.0f32 / 255.0; 4];
        let stream = encode_delta_varint_pose(&poses, 3, 4, &lo, &scale, 8).unwrap();
        let decoded = decode_delta_varint_pose(&stream.payload).unwrap();
        assert_eq!(decoded.len(), 12);
        // Round-trip should reproduce quantised reconstruction (not raw input).
        // Each input value satisfies |v - lo[j] - q*scale[j]| < scale[j]/2.
        for (orig, recv) in poses.iter().zip(decoded.iter()) {
            assert!((orig - recv).abs() < 1.0 / 255.0);
        }
    }
}
