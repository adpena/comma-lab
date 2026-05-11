//! PR97 H3 wire-format grammar — Rust port.
//!
//! Mirrors the public API of
//! [`tac.packet_compiler.pr97_h3_grammar`](
//! file:../../../../../src/tac/packet_compiler/pr97_h3_grammar.py).
//!
//! Two REUSABLE wire-format primitives from the PR97 public submission:
//!
//! 1. **Length-prefixed multi-section payload grammar**
//!    ([`encode_length_prefixed_sections`] /
//!    [`decode_length_prefixed_sections`]) — packs N byte sections into a
//!    single blob via `[u32 LE len_i][bytes_i]` pairs. Has NO global
//!    section-count prefix; the consumer must know how many sections to
//!    read (PR97 reads at least 3 — mask/pose/model — and tolerates an
//!    optional trailing 4th sidecar section).
//!
//! 2. **Tile-band multi-stream wire format**
//!    ([`encode_tile_band_streams`] /
//!    [`decode_tile_band_streams`]) — packs N per-tile streams into
//!    `[u32 LE n_chunks][u32 LE size_i][bytes_i]...`. PR97 uses this for 4
//!    horizontal bands × per-band W-splits = 22 streams. The 2D tile
//!    reassembly is PR97-specific and stays out of this primitive.
//!
//! Byte-for-byte parity target SHA-256s live in
//! `src/tac/packet_compiler/golden_vectors/pr97_h3_length_prefixed_sections_v1.json`
//! and
//! `src/tac/packet_compiler/golden_vectors/pr97_h3_tile_band_streams_v1.json`.
//!
//! # Selfcomp gotcha (length prefix endianness)
//!
//! Per Selfcomp's note on the parent PR93 / PR101 grammars, BOTH wire
//! formats use **little-endian** u32 length prefixes (NOT big-endian).
//! This matches the wider `tac.packet_compiler` framing convention (e.g.
//! the RMC1 composite + PR101 split-Brotli). The Python oracle uses
//! `struct.pack("<I", ...)` exclusively.

pub mod length_prefixed_sections;
pub mod tile_band_streams;

pub use length_prefixed_sections::{
    decode_length_prefixed_sections, encode_length_prefixed_sections,
    LengthPrefixedSectionPayload,
};
pub use tile_band_streams::{
    decode_tile_band_streams, encode_tile_band_streams, TileBandStreamPayload,
};
