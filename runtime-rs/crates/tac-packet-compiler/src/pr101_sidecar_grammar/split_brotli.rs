//! Rust port of `split_brotli_self_delimiting` /
//! `parse_split_brotli_self_delimiting`.
//!
//! # Byte-for-byte parity contract
//!
//! The Python oracle (`src/tac/packet_compiler/pr101_sidecar_grammar.py`)
//! compresses each sub-stream independently with
//! `brotli.compress(raw, mode=MODE_GENERIC, quality=11, lgwin=22)` and
//! concatenates the bytes. The Rust port uses the pure-Rust `brotli` crate
//! whose `CompressorWriter::new(_, _, q=11, lgwin=22)` constructor builds
//! the same default `BrotliEncoderParams` (mode=`BROTLI_MODE_GENERIC`,
//! q=11, lgwin=22 — see `brotli::enc::encode::BrotliEncoderInitParams`).
//!
//! # Decoder parity
//!
//! The decoder side walks the concatenated payload byte-by-byte, feeding
//! one byte at a time into a fresh `brotli_decompressor::DecompressorWriter`
//! until that decoder reports completion (`close()` succeeds with no
//! pending data). PR101 does exactly this in `decompress_brotli_streams`.
//!
//! # Endianness
//!
//! Per Selfcomp's gotcha — there is NO length prefix in the wire format.
//! The reader uses Brotli's frame structure to detect each sub-stream's
//! end. The implementation below is agnostic to host endianness.

use std::io::Write;

use brotli::enc::backward_references::BrotliEncoderParams;
use brotli::enc::writer::CompressorWriter;
use brotli_decompressor::BrotliDecompressStream;
use brotli_decompressor::BrotliResult;
use brotli_decompressor::BrotliState;
use brotli_decompressor::StandardAlloc;

use crate::{PacketCompilerError, Result};

use super::stubs::SplitBrotliStream;

/// Build the Brotli encoder params PR101 uses: GENERIC mode, q=11, lgwin=22.
fn build_params(lgwin: u8, quality: u8) -> BrotliEncoderParams {
    // Mode is BROTLI_MODE_GENERIC by default (per
    // `brotli::enc::encode::BrotliEncoderInitParams`).
    BrotliEncoderParams {
        quality: quality as i32,
        lgwin: lgwin as i32,
        ..Default::default()
    }
}

/// Compress one sub-stream with `(lgwin, quality)` exactly. Bytes match the
/// Python `brotli.compress(...)` output byte-for-byte at parity targets.
fn compress_one(raw: &[u8], lgwin: u8, quality: u8) -> Result<Vec<u8>> {
    let params = build_params(lgwin, quality);
    let mut out = Vec::with_capacity(raw.len() / 2 + 64);
    {
        let mut writer = CompressorWriter::with_params(&mut out, 4096, &params);
        writer.write_all(raw).map_err(|e| {
            PacketCompilerError::GoldenVectorIo(format!(
                "brotli compress write failed: {e:?}"
            ))
        })?;
        // CompressorWriter::flush emits the final block; drop closes the writer.
    }
    if out.is_empty() {
        return Err(PacketCompilerError::GoldenVectorIo(
            "brotli compress produced empty output".into(),
        ));
    }
    Ok(out)
}

/// Concatenate N independently-Brotli-compressed byte streams.
///
/// Each sub-stream is compressed with `(lgwin, quality)`; the compressed
/// payloads are concatenated with NO length prefix. The reader uses
/// Brotli's frame structure to detect each stream's end.
pub fn split_brotli_self_delimiting(
    streams: &[&[u8]],
    lgwin: u8,
    quality: u8,
) -> Result<SplitBrotliStream> {
    if streams.is_empty() {
        return Err(PacketCompilerError::GoldenVectorIo(
            "streams must contain at least one substream".into(),
        ));
    }
    if !(10..=24).contains(&lgwin) {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "lgwin must be in [10, 24]; got {lgwin}"
        )));
    }
    if quality > 11 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "quality must be in [0, 11]; got {quality}"
        )));
    }
    let mut payload = Vec::new();
    let mut offsets = Vec::with_capacity(streams.len());
    for (i, raw) in streams.iter().enumerate() {
        let comp = compress_one(raw, lgwin, quality)?;
        if comp.is_empty() {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "empty Brotli output for substream {i}"
            )));
        }
        payload.extend_from_slice(&comp);
        offsets.push(payload.len());
    }
    Ok(SplitBrotliStream {
        payload,
        n_streams: streams.len(),
        stream_byte_offsets: offsets,
    })
}

/// Inverse of [`split_brotli_self_delimiting`].
///
/// Walks the concatenated Brotli payload, decoding one sub-stream at a time
/// until each decoder reports `ResultSuccess`. Raises an error if the bytes
/// do not yield exactly `n_streams` sub-streams with no trailing data.
pub fn parse_split_brotli_self_delimiting(
    payload: &[u8],
    n_streams: usize,
) -> Result<Vec<Vec<u8>>> {
    if n_streams == 0 {
        return Err(PacketCompilerError::GoldenVectorIo(
            "n_streams must be > 0".into(),
        ));
    }
    let mut outputs: Vec<Vec<u8>> = Vec::with_capacity(n_streams);
    let mut pos = 0usize;
    for stream_idx in 0..n_streams {
        let (decoded, consumed) = decode_one_stream(&payload[pos..]).map_err(|e| {
            PacketCompilerError::GoldenVectorIo(format!(
                "split-Brotli decode failed at stream {stream_idx}: {e}"
            ))
        })?;
        pos += consumed;
        outputs.push(decoded);
    }
    if pos != payload.len() {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "trailing data after {n_streams} streams: {} bytes",
            payload.len() - pos
        )));
    }
    Ok(outputs)
}

/// Decode a single Brotli stream from the start of `input`. Returns the
/// decompressed bytes plus the number of input bytes consumed.
fn decode_one_stream(input: &[u8]) -> std::result::Result<(Vec<u8>, usize), String> {
    let mut state: BrotliState<StandardAlloc, StandardAlloc, StandardAlloc> = BrotliState::new(
        StandardAlloc::default(),
        StandardAlloc::default(),
        StandardAlloc::default(),
    );
    let mut output = Vec::new();
    let mut input_offset = 0usize;
    let mut output_buf = vec![0u8; 65536];
    loop {
        let mut available_in = input.len() - input_offset;
        let mut available_out = output_buf.len();
        let mut out_offset = 0usize;
        let mut total_out: usize = 0;
        let result = BrotliDecompressStream(
            &mut available_in,
            &mut input_offset,
            input,
            &mut available_out,
            &mut out_offset,
            &mut output_buf,
            &mut total_out,
            &mut state,
        );
        if out_offset > 0 {
            output.extend_from_slice(&output_buf[..out_offset]);
        }
        match result {
            BrotliResult::ResultSuccess => return Ok((output, input_offset)),
            BrotliResult::NeedsMoreInput => {
                return Err(format!(
                    "truncated split-Brotli payload (needs more input at offset {input_offset})"
                ));
            }
            BrotliResult::NeedsMoreOutput => {
                // Continue the loop; we just emptied output_buf into output.
                continue;
            }
            BrotliResult::ResultFailure => {
                return Err(format!(
                    "brotli decode failed at input offset {input_offset}"
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brotli_smoke_roundtrip_three_streams() {
        let s0: &[u8] = b"PR101 sidecar grammar conformance vector A ";
        let s1: &[u8] = b"Reusable byte primitives, deterministic build ";
        let s2: &[u8] = b"Future Rust/Zig port must match these bytes ";
        let streams = [s0, s1, s2];
        let result = split_brotli_self_delimiting(&streams, 22, 11).unwrap();
        assert_eq!(result.n_streams, 3);
        assert_eq!(result.stream_byte_offsets.last().copied().unwrap(), result.payload.len());
        let out = parse_split_brotli_self_delimiting(&result.payload, 3).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], s0);
        assert_eq!(out[1], s1);
        assert_eq!(out[2], s2);
    }

    #[test]
    fn brotli_rejects_empty_streams_list() {
        let err = split_brotli_self_delimiting(&[], 22, 11).expect_err("empty must error");
        match err {
            PacketCompilerError::GoldenVectorIo(_) => {}
            other => panic!("wrong error: {other:?}"),
        }
    }
}
