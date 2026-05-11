//! PR91 ``hpac_coder_hybrid`` grammar primitives — Rust port.
//!
//! Mirrors the public API of
//! [`tac.packet_compiler.pr91_hpac_grammar`](
//! file:../../../../../src/tac/packet_compiler/pr91_hpac_grammar.py).
//!
//! Two primitives are exposed:
//!
//! 1. **Universal constriction arithmetic coder wrapper**
//!    ([`encode_categorical_stream`] / [`decode_categorical_stream`]) —
//!    implemented; byte-for-byte parity against
//!    `pr91_arithmetic_coder_constriction_v1.json`.
//!
//! 2. **PR91 QM0 / QH0 magic grammar** — magic constants exposed; the
//!    pack / unpack helpers are scaffold for now.

pub mod arithmetic_coder_constriction;

pub use arithmetic_coder_constriction::{
    decode_categorical_stream, encode_categorical_stream, MAGIC_QH0, MAGIC_QM0,
};
