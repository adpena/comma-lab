//! RLE-of-zeros sparse representation — Rust port.
//!
//! Byte-for-byte parity contract: the Python oracle
//! (`tac.packet_compiler.sparse_packet_ir::encode_rle_of_zeros`,
//! `serialize_rle_of_zeros`) emits the wire format below. The Python
//! `__post_init__` invariant requires strictly-increasing indices and
//! non-zero values; the Rust port honours the same contract.
//!
//! # Wire format
//!
//! ```text
//! magic(4)             = b"SRL1"
//! total_length         u32 LE  (length of the dense array represented)
//! n_nonzero            u32 LE
//! nonzero_dtype_code   u8  (0 = int8, 1 = int16, 2 = int32)
//! nonzero_indices      n_nonzero * u32 LE
//! nonzero_values       n_nonzero * signed itemsize (LE)
//! ```
//!
//! At 99.4% sparsity the encoder compresses 1024 bytes of dense int8 to
//! 333 bytes (per the committed `sparse_rle_of_zeros_v1.json` golden
//! vector — 64 nonzero entries → 13-byte header + 256 indices + 64 values).

use crate::{PacketCompilerError, Result};

/// 4-byte magic prefix for SRL1 sparse RLE-of-zeros payloads.
pub const SPARSE_RLE_MAGIC: [u8; 4] = *b"SRL1";

/// Sparse representation of a dense array via non-zero (index, value) pairs.
///
/// Mirrors `tac.packet_compiler.RleOfZerosStream`. Strictly-increasing
/// indices and non-zero values are required (the Python `__post_init__`
/// enforces both); the constructors here re-check.
#[derive(Debug, Clone)]
pub struct RleOfZerosStream {
    /// `uint32` indices into the dense layout (C-order flat). Strictly
    /// increasing; no duplicates.
    pub nonzero_indices: Vec<u32>,
    /// Non-zero values aligned with `nonzero_indices`. Stored as
    /// little-endian signed bytes; width is recorded by
    /// [`RleOfZerosStream::nonzero_value_itemsize`].
    pub nonzero_values: Vec<u8>,
    /// Width of one value in bytes (1 = int8, 2 = int16, 4 = int32).
    pub nonzero_value_itemsize: u8,
    /// Length of the dense array the stream represents.
    pub total_length: u32,
}

impl RleOfZerosStream {
    /// Validate the stream invariants.
    fn validate(&self) -> Result<()> {
        let n = self.nonzero_indices.len();
        let expected_value_bytes = n * (self.nonzero_value_itemsize as usize);
        if self.nonzero_values.len() != expected_value_bytes {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "RLE-of-zeros nonzero_values len {} != n_nonzero ({n}) * itemsize ({}) = {expected_value_bytes}",
                self.nonzero_values.len(),
                self.nonzero_value_itemsize
            )));
        }
        if !matches!(self.nonzero_value_itemsize, 1 | 2 | 4) {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "RLE-of-zeros unsupported nonzero_value_itemsize {}; expected 1/2/4",
                self.nonzero_value_itemsize
            )));
        }
        if n > 0 {
            // Strictly increasing.
            for w in self.nonzero_indices.windows(2) {
                if w[1] <= w[0] {
                    return Err(PacketCompilerError::GoldenVectorIo(
                        "RLE-of-zeros nonzero_indices must be strictly increasing".into(),
                    ));
                }
            }
            let last = *self.nonzero_indices.last().expect("non-empty checked");
            if last >= self.total_length {
                return Err(PacketCompilerError::GoldenVectorIo(format!(
                    "RLE-of-zeros nonzero_indices contains index >= total_length ({last} >= {})",
                    self.total_length
                )));
            }
            // Non-zero values: value bytes for each entry must not all be zero.
            let itemsize = self.nonzero_value_itemsize as usize;
            for i in 0..n {
                let off = i * itemsize;
                if self.nonzero_values[off..off + itemsize]
                    .iter()
                    .all(|&b| b == 0)
                {
                    return Err(PacketCompilerError::GoldenVectorIo(format!(
                        "RLE-of-zeros nonzero_values[{i}] is zero (the contract is non-zero only)"
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Map a dtype code (0/1/2) to its byte width.
fn dtype_code_to_itemsize(code: u8) -> Result<u8> {
    match code {
        0 => Ok(1),
        1 => Ok(2),
        2 => Ok(4),
        _ => Err(PacketCompilerError::GoldenVectorIo(format!(
            "unknown nonzero dtype code {code}"
        ))),
    }
}

/// Map a byte width (1/2/4) to its dtype code.
fn itemsize_to_dtype_code(itemsize: u8) -> Result<u8> {
    match itemsize {
        1 => Ok(0),
        2 => Ok(1),
        4 => Ok(2),
        _ => Err(PacketCompilerError::GoldenVectorIo(format!(
            "unsupported nonzero itemsize {itemsize}"
        ))),
    }
}

/// Pick the smallest signed dtype that contains every value.
///
/// Mirrors the Python oracle's auto-dtype selection:
///   v_min ∈ [-128, 127] AND v_max ∈ [-128, 127] → int8
///   v_min ∈ [-32768, 32767] AND v_max ∈ [-32768, 32767] → int16
///   else → int32
fn pick_dtype_for_int_values(values: &[i32]) -> u8 {
    if values.is_empty() {
        return 1; // matches Python default for empty input (int8)
    }
    let v_min = values.iter().copied().min().expect("non-empty");
    let v_max = values.iter().copied().max().expect("non-empty");
    if (-128..=127).contains(&v_min) && (-128..=127).contains(&v_max) {
        1
    } else if (-32768..=32767).contains(&v_min) && (-32768..=32767).contains(&v_max) {
        2
    } else {
        4
    }
}

/// Build an [`RleOfZerosStream`] from a dense int8 array.
///
/// Mirrors `tac.packet_compiler.encode_rle_of_zeros`. Auto-selects int8
/// dtype since the input is already int8 (the Python oracle's auto-dtype
/// would resolve to the same width).
///
/// # Errors
///
/// * [`PacketCompilerError::GoldenVectorIo`] if the dense input is longer
///   than `u32::MAX` (the wire `total_length` field is a `u32`).
pub fn encode_rle_of_zeros(dense: &[i8]) -> Result<RleOfZerosStream> {
    if dense.len() > u32::MAX as usize {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "RLE-of-zeros dense length {} exceeds u32::MAX",
            dense.len()
        )));
    }
    let mut indices: Vec<u32> = Vec::new();
    let mut values_i32: Vec<i32> = Vec::new();
    for (i, &v) in dense.iter().enumerate() {
        if v != 0 {
            indices.push(i as u32);
            values_i32.push(v as i32);
        }
    }
    let itemsize = pick_dtype_for_int_values(&values_i32);
    let mut value_bytes: Vec<u8> = Vec::with_capacity(values_i32.len() * (itemsize as usize));
    for &v in &values_i32 {
        match itemsize {
            1 => value_bytes.push((v as i8) as u8),
            2 => value_bytes.extend_from_slice(&(v as i16).to_le_bytes()),
            4 => value_bytes.extend_from_slice(&v.to_le_bytes()),
            _ => unreachable!("pick_dtype_for_int_values returns 1/2/4"),
        }
    }
    let stream = RleOfZerosStream {
        nonzero_indices: indices,
        nonzero_values: value_bytes,
        nonzero_value_itemsize: itemsize,
        total_length: dense.len() as u32,
    };
    stream.validate()?;
    Ok(stream)
}

/// Inverse of [`encode_rle_of_zeros`] for the int8 specialisation.
///
/// Returns the dense int8 representation. For wider dtypes the caller
/// should bit-cast `nonzero_values` directly.
pub fn decode_rle_of_zeros(stream: &RleOfZerosStream) -> Result<Vec<i8>> {
    stream.validate()?;
    if stream.nonzero_value_itemsize != 1 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "decode_rle_of_zeros only supports int8 nonzero values; got itemsize {}",
            stream.nonzero_value_itemsize
        )));
    }
    let mut out = vec![0i8; stream.total_length as usize];
    for (slot, &idx) in stream.nonzero_indices.iter().enumerate() {
        out[idx as usize] = stream.nonzero_values[slot] as i8;
    }
    Ok(out)
}

/// Serialise an [`RleOfZerosStream`] to self-delimiting wire bytes.
///
/// Mirrors `tac.packet_compiler.serialize_rle_of_zeros`. Header layout:
/// `[4s magic][u32 LE total_length][u32 LE n_nonzero][u8 dtype_code]`.
pub fn serialize_rle_of_zeros(stream: &RleOfZerosStream) -> Result<Vec<u8>> {
    stream.validate()?;
    let n = stream.nonzero_indices.len();
    let dtype_code = itemsize_to_dtype_code(stream.nonzero_value_itemsize)?;
    let body_len = 4 + 4 + 4 + 1 + n * 4 + stream.nonzero_values.len();
    let mut out = Vec::with_capacity(body_len);
    out.extend_from_slice(&SPARSE_RLE_MAGIC);
    out.extend_from_slice(&stream.total_length.to_le_bytes());
    out.extend_from_slice(&(n as u32).to_le_bytes());
    out.push(dtype_code);
    for &idx in &stream.nonzero_indices {
        out.extend_from_slice(&idx.to_le_bytes());
    }
    out.extend_from_slice(&stream.nonzero_values);
    Ok(out)
}

/// Inverse of [`serialize_rle_of_zeros`]. Hard errors on truncation,
/// magic-byte mismatch, or trailing bytes.
pub fn deserialize_rle_of_zeros(blob: &[u8]) -> Result<RleOfZerosStream> {
    if blob.len() < 13 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "RLE blob too short for header: {} < 13",
            blob.len()
        )));
    }
    if blob[0..4] != SPARSE_RLE_MAGIC {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "RLE magic mismatch: got {:?} expected {:?}",
            &blob[0..4],
            SPARSE_RLE_MAGIC
        )));
    }
    let total_length = u32::from_le_bytes([blob[4], blob[5], blob[6], blob[7]]);
    let n_nonzero = u32::from_le_bytes([blob[8], blob[9], blob[10], blob[11]]) as usize;
    let dtype_code = blob[12];
    let itemsize = dtype_code_to_itemsize(dtype_code)?;
    let mut pos = 13;
    let idx_bytes = n_nonzero * 4;
    let val_bytes = n_nonzero * (itemsize as usize);
    if pos + idx_bytes + val_bytes > blob.len() {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "RLE blob truncated: need {} bytes, have {}",
            pos + idx_bytes + val_bytes,
            blob.len()
        )));
    }
    let mut indices = Vec::with_capacity(n_nonzero);
    for _ in 0..n_nonzero {
        indices.push(u32::from_le_bytes([
            blob[pos],
            blob[pos + 1],
            blob[pos + 2],
            blob[pos + 3],
        ]));
        pos += 4;
    }
    let values = blob[pos..pos + val_bytes].to_vec();
    pos += val_bytes;
    if pos != blob.len() {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "RLE blob has trailing bytes: pos={pos} total={}",
            blob.len()
        )));
    }
    let stream = RleOfZerosStream {
        nonzero_indices: indices,
        nonzero_values: values,
        nonzero_value_itemsize: itemsize,
        total_length,
    };
    stream.validate()?;
    Ok(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_zero_input_emits_empty_stream() {
        let dense = vec![0i8; 8];
        let s = encode_rle_of_zeros(&dense).unwrap();
        assert_eq!(s.total_length, 8);
        assert!(s.nonzero_indices.is_empty());
        assert!(s.nonzero_values.is_empty());
        let blob = serialize_rle_of_zeros(&s).unwrap();
        // 13-byte header only.
        assert_eq!(blob.len(), 13);
        assert_eq!(&blob[0..4], &SPARSE_RLE_MAGIC);
    }

    #[test]
    fn encode_decode_roundtrip_smoke() {
        let mut dense = vec![0i8; 32];
        dense[3] = 5;
        dense[10] = -7;
        dense[31] = 1;
        let s = encode_rle_of_zeros(&dense).unwrap();
        assert_eq!(s.nonzero_indices, vec![3, 10, 31]);
        assert_eq!(s.nonzero_values, vec![5u8, (-7i8) as u8, 1u8]);
        let blob = serialize_rle_of_zeros(&s).unwrap();
        let s2 = deserialize_rle_of_zeros(&blob).unwrap();
        let back = decode_rle_of_zeros(&s2).unwrap();
        assert_eq!(back, dense);
    }

    #[test]
    fn auto_dtype_picks_int16_when_value_overflows_int8() {
        // Manually build a stream with an int16 value (200, doesn't fit in i8).
        let stream = RleOfZerosStream {
            nonzero_indices: vec![0, 1],
            nonzero_values: {
                let mut v = Vec::new();
                v.extend_from_slice(&200i16.to_le_bytes());
                v.extend_from_slice(&(-300i16).to_le_bytes());
                v
            },
            nonzero_value_itemsize: 2,
            total_length: 2,
        };
        let blob = serialize_rle_of_zeros(&stream).unwrap();
        assert_eq!(blob[12], 1); // dtype_code = 1 (int16)
        let back = deserialize_rle_of_zeros(&blob).unwrap();
        assert_eq!(back.nonzero_value_itemsize, 2);
    }

    #[test]
    fn deserialize_rejects_bad_magic() {
        let bad = [b'X', b'X', b'X', b'X', 0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert!(deserialize_rle_of_zeros(&bad).is_err());
    }

    #[test]
    fn deserialize_rejects_truncated() {
        let dense = vec![0i8; 4];
        let s = encode_rle_of_zeros(&dense).unwrap();
        let blob = serialize_rle_of_zeros(&s).unwrap();
        assert!(deserialize_rle_of_zeros(&blob[..blob.len() - 1]).is_err());
    }

    #[test]
    fn deserialize_rejects_trailing_bytes() {
        let dense = vec![0i8; 4];
        let s = encode_rle_of_zeros(&dense).unwrap();
        let mut blob = serialize_rle_of_zeros(&s).unwrap();
        blob.push(0xFF);
        assert!(deserialize_rle_of_zeros(&blob).is_err());
    }

    #[test]
    fn validate_rejects_non_strictly_increasing_indices() {
        let s = RleOfZerosStream {
            nonzero_indices: vec![0, 0, 1],
            nonzero_values: vec![1, 2, 3],
            nonzero_value_itemsize: 1,
            total_length: 4,
        };
        assert!(serialize_rle_of_zeros(&s).is_err());
    }

    #[test]
    fn validate_rejects_zero_value_entry() {
        let s = RleOfZerosStream {
            nonzero_indices: vec![0],
            nonzero_values: vec![0],
            nonzero_value_itemsize: 1,
            total_length: 1,
        };
        assert!(serialize_rle_of_zeros(&s).is_err());
    }
}
