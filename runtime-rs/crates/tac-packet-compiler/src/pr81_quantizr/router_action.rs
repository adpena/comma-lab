//! Rust port of PR81's ROUTER_ACTION small-integer LSB-first bit-stream
//! packing.
//!
//! Byte-for-byte parity contract: the Python oracle
//! (`src/tac/packet_compiler/pr81_quantizr.py::encode_router_actions`)
//! produces a packed byte stream whose layout is reproduced here exactly.
//!
//! # Algorithm (matches Python oracle)
//!
//! 1. Walk the action array left-to-right.
//! 2. Maintain a `u64` bit-accumulator + bit count.
//! 3. For each action, OR its `bits`-bit code into the accumulator at the
//!    current bit position (LSB-first), advance the bit count by `bits`.
//! 4. While the bit count is ≥ 8, emit the low byte and shift the
//!    accumulator down 8 bits.
//! 5. After the loop, if the bit count is > 0 emit one more byte (the
//!    final partial byte; high bits are zero-padded).
//!
//! Worst-case accumulator width: when `accbits` is ≥ 8 we emit; the
//! maximum residual after emit is `8 - 1 = 7` bits; the next push adds at
//! most 8 bits → max 15 bits; safely fits in `u64`.

use crate::{PacketCompilerError, Result};

/// Pack small-integer per-frame actions as an LSB-first bit stream.
///
/// Mirrors `tac.packet_compiler.encode_router_actions`. Each action must
/// fit in `bits` bits (i.e. satisfy `0 <= action < (1 << bits)`).
///
/// Output length is `ceil(actions.len() * bits / 8)` bytes. The final byte
/// is zero-padded on the high side if the total bit count is not a
/// multiple of 8.
pub fn encode_router_actions(actions: &[u8], bits: u32) -> Result<Vec<u8>> {
    if !(1..=8).contains(&bits) {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "bits must be in [1, 8]; got {bits}"
        )));
    }
    let mask: u8 = ((1u32 << bits) - 1) as u8;
    for (i, &a) in actions.iter().enumerate() {
        if a > mask {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "action values out of range [0, {}); actions[{}] = {}",
                1u32 << bits,
                i,
                a
            )));
        }
    }
    let total_bits: usize = actions.len() * (bits as usize);
    let out_len: usize = total_bits.div_ceil(8);
    let mut out: Vec<u8> = Vec::with_capacity(out_len);
    let mut acc: u64 = 0;
    let mut accbits: u32 = 0;
    for &a in actions {
        acc |= ((a as u64) & (mask as u64)) << accbits;
        accbits += bits;
        while accbits >= 8 {
            out.push((acc & 0xFF) as u8);
            acc >>= 8;
            accbits -= 8;
        }
    }
    if accbits > 0 {
        out.push((acc & 0xFF) as u8);
    }
    debug_assert_eq!(
        out.len(),
        out_len,
        "packed length mismatch: wrote {} bytes, expected {}",
        out.len(),
        out_len
    );
    Ok(out)
}

/// Decode an LSB-first packed action stream back to the action array.
///
/// Mirrors `tac.packet_compiler.decode_router_actions`. The caller MUST
/// supply the original `count` and `bits` (the wire format is
/// metadata-free at this layer; a wrapping grammar — e.g. PR81's RSA1 —
/// carries them).
pub fn decode_router_actions(payload: &[u8], count: usize, bits: u32) -> Result<Vec<u8>> {
    if !(1..=8).contains(&bits) {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "bits must be in [1, 8]; got {bits}"
        )));
    }
    let total_bits = count.checked_mul(bits as usize).ok_or_else(|| {
        PacketCompilerError::GoldenVectorIo(format!(
            "count*bits overflows usize: count={count} bits={bits}"
        ))
    })?;
    if total_bits > 8 * payload.len() {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "count={count} bits={bits} requires {total_bits} bits but payload has only {} bits",
            8 * payload.len()
        )));
    }
    let mask: u64 = (1u64 << bits) - 1;
    let mut out: Vec<u8> = Vec::with_capacity(count);
    let mut acc: u64 = 0;
    let mut accbits: u32 = 0;
    let mut j: usize = 0;
    for &byte in payload {
        acc |= (byte as u64) << accbits;
        accbits += 8;
        while accbits >= bits && j < count {
            out.push((acc & mask) as u8);
            acc >>= bits;
            accbits -= bits;
            j += 1;
        }
        if j >= count {
            break;
        }
    }
    debug_assert_eq!(j, count, "decoded {j} actions, expected {count}",);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_3bit_600() {
        let actions: Vec<u8> = (0..600).map(|i| (i % 8) as u8).collect();
        let packed = encode_router_actions(&actions, 3).unwrap();
        assert_eq!(packed.len(), 225); // ceil(600*3/8) = 225
        let recovered = decode_router_actions(&packed, 600, 3).unwrap();
        assert_eq!(recovered, actions);
    }

    #[test]
    fn roundtrip_4bit_1200() {
        let actions: Vec<u8> = (0..1200).map(|i| (i % 16) as u8).collect();
        let packed = encode_router_actions(&actions, 4).unwrap();
        assert_eq!(packed.len(), 600); // 1200*4/8 = 600
        let recovered = decode_router_actions(&packed, 1200, 4).unwrap();
        assert_eq!(recovered, actions);
    }

    #[test]
    fn empty_input_produces_empty_output() {
        let packed = encode_router_actions(&[], 3).unwrap();
        assert!(packed.is_empty());
        let recovered = decode_router_actions(&[], 0, 3).unwrap();
        assert!(recovered.is_empty());
    }

    #[test]
    fn rejects_oversized_action() {
        let actions = [0u8, 1, 2, 8]; // 8 doesn't fit in 3 bits
        assert!(encode_router_actions(&actions, 3).is_err());
    }

    #[test]
    fn rejects_invalid_bits() {
        assert!(encode_router_actions(&[0], 0).is_err());
        assert!(encode_router_actions(&[0], 9).is_err());
    }

    #[test]
    fn known_single_action_3bit() {
        // Single action with value 5 (binary 101) at bits=3.
        // Output: low byte = 0b00000101 = 0x05, then a trailing 0 because
        // 3 bits doesn't fill a byte — wait, actually a single 3-bit value
        // produces one partial byte. Let's check:
        // accbits = 3 after push, < 8, loop exit, emit final partial → 1 byte.
        let packed = encode_router_actions(&[5], 3).unwrap();
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0], 0x05);
        let recovered = decode_router_actions(&packed, 1, 3).unwrap();
        assert_eq!(recovered, vec![5]);
    }

    #[test]
    fn known_two_actions_3bit() {
        // Two 3-bit actions = 6 bits total, still 1 byte (partial).
        // a=5 (101), b=3 (011) → acc = (101) | (011 << 3) = 011_101 = 0x1D.
        let packed = encode_router_actions(&[5, 3], 3).unwrap();
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0], 0x1D);
    }
}
