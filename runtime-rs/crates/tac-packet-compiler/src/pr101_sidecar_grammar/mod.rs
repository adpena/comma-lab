//! PR101 sidecar grammar — Rust native port with one remaining decoder scaffold.
//!
//! Mirrors the public API of
//! [`tac.packet_compiler.pr101_sidecar_grammar`](
//! file:../../../../../src/tac/packet_compiler/pr101_sidecar_grammar.py).
//!
//! Three primitives are exposed:
//!
//! 1. **Ranked Huffman/no-op sidecar** — encoder implemented; decoder still
//!    scaffold-only. Co-lex combination rank for no-op positions + canonical
//!    Huffman code with rank-encoded length vector.
//! 2. **Centered-delta uint8 latents** — implemented. Column-major uint8 block under raw
//!    LZMA1 (filter-only, no XZ container; dict 4 KiB, lc=3, lp=0, pb=0).
//! 3. **Self-delimiting split Brotli** — implemented. Concatenated Brotli streams; reader
//!    feeds byte-by-byte until each frame closes.
//!
//! Each primitive has a paired golden vector under
//! `src/tac/packet_compiler/golden_vectors/`:
//!
//! - `ranked_no_op_sidecar_v1.json`
//! - `centered_delta_uint8_v1.json`
//! - `split_brotli_self_delim_v1.json`
//!
//! Implementation work MUST reproduce the pinned SHA-256s exactly.

pub mod centered_delta_uint8;
pub mod ranked_no_op_sidecar;
pub mod split_brotli;
pub mod stubs;

pub use stubs::{
    decode_centered_delta_uint8, decode_ranked_no_op_sidecar, encode_centered_delta_uint8,
    encode_ranked_no_op_sidecar, parse_split_brotli_self_delimiting, split_brotli_self_delimiting,
    CenteredDeltaUint8Stream, RankedSidecarSchema, SplitBrotliStream,
};
