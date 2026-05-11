//! `tac-packet-compiler-wasm` — WebAssembly target for the canonical
//! `tac-packet-compiler` primitives.
//!
//! # Status — RESEARCH / OSS-DEMO ONLY
//!
//! - `score_claim=false`
//! - `promotion_eligible=false`
//! - `ready_for_exact_eval_dispatch=false`
//!
//! Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1
//! CONTEST-COMPLIANT HARDWARE", any score produced via a WASM-runtime
//! path is `[advisory only]`. The contest CI runs native (x86_64 Linux);
//! a WASM-compiled encoder is a useful OSS browser demo but **not** a
//! contest-axis substrate.
//!
//! # Why a separate crate
//!
//! The sister crate `tac-packet-compiler` ships native Rust artefacts
//! (libstd, liblzma C bindings, criterion benches). Trying to compile
//! that crate directly to `wasm32-unknown-unknown` fails because liblzma
//! pulls in C objects.
//!
//! This crate is a **thin export layer** that:
//!
//! 1. Re-exports only the PURE-RUST primitives (no liblzma, no
//!    constriction-C-bindings — fortunately constriction is pure Rust).
//! 2. Uses `wasm-bindgen` to expose them as JS-callable functions.
//! 3. Adds a `console_error_panic_hook` so browser console gets proper
//!    panic messages.
//!
//! # What's exposed
//!
//! - [`extract_hi_bytes_wasm`] — SIMD-portable hi-byte extraction.
//! - [`scan_nonzero_indices_wasm`] — RLE-of-zeros nonzero scan.
//! - [`centered_delta_uint8_wasm`] — centered-delta column-major emit.
//! - [`encode_tacp_container_wasm`] / [`decode_tacp_container_wasm`] —
//!   custom binary container (research-only).
//! - [`encode_latent_hi_wasm`] — PR103 hi-byte arithmetic coder.
//!
//! # Byte-parity
//!
//! Every WASM-exported function is a thin wrapper around the sister
//! `tac-packet-compiler` function it delegates to. There is no
//! re-implementation, no divergent control flow. The `wasm-bindgen-test`
//! suite verifies WASM output equals native output on the canonical
//! golden vectors.
//!
//! # Building
//!
//! ```bash
//! wasm-pack build --target web crates/tac-packet-compiler-wasm
//! # → pkg/tac_packet_compiler_wasm_bg.wasm + pkg/tac_packet_compiler_wasm.js
//! ```
//!
//! Then drop the `pkg/` directory next to `crates/tac-packet-compiler-wasm/web/index.html`
//! and serve via `python -m http.server` (or any static server).

#![warn(missing_docs)]
#![warn(unsafe_code)]

use wasm_bindgen::prelude::*;

/// Initialise the WASM panic hook for browser console diagnostics.
///
/// Call once from JS before invoking any export.
#[wasm_bindgen(start)]
pub fn wasm_start() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Extract hi-bytes of a `Uint16Array` as a `Uint32Array` of usize symbols.
///
/// Mirrors [`tac_packet_compiler::simd::hi_byte::extract_hi_bytes_u16_to_usize`].
///
/// Each element of the returned `Uint32Array` is in `[0, 256)`. The active
/// SIMD back-end on the WASM target is currently `Portable` (WASM SIMD-128
/// support is a future extension), but the public contract is identical.
#[wasm_bindgen(js_name = extractHiBytes)]
pub fn extract_hi_bytes_wasm(latents: &[u16]) -> Vec<u32> {
    tac_packet_compiler::simd::hi_byte::extract_hi_bytes_u16_to_usize(latents)
        .into_iter()
        .map(|x| x as u32)
        .collect()
}

/// Scan an `Int8Array` for nonzero positions; return paired `(indices, values)` arrays.
///
/// Mirrors [`tac_packet_compiler::simd::rle_of_zeros::scan_nonzero_indices_i8`].
///
/// Returns a `Vec<JsValue>` `[indices_u32, values_i8]` so JS can destructure
/// to two typed arrays.
#[wasm_bindgen(js_name = scanNonzeroIndices)]
pub fn scan_nonzero_indices_wasm(dense: &[i8]) -> Box<[JsValue]> {
    let (indices, values) =
        tac_packet_compiler::simd::rle_of_zeros::scan_nonzero_indices_i8(dense);
    let indices_js: js_sys::Uint32Array = js_sys::Uint32Array::from(indices.as_slice());
    let values_js: js_sys::Int8Array = js_sys::Int8Array::from(values.as_slice());
    vec![JsValue::from(indices_js), JsValue::from(values_js)].into_boxed_slice()
}

/// Compute the centered-delta column-major block from a row-major q matrix.
///
/// Mirrors [`tac_packet_compiler::simd::centered_delta::centered_delta_uint8_column_major`].
#[wasm_bindgen(js_name = centeredDeltaUint8)]
pub fn centered_delta_uint8_wasm(q: &[u8], n_pairs: usize, n_dims: usize) -> Vec<u8> {
    tac_packet_compiler::simd::centered_delta::centered_delta_uint8_column_major(
        q, n_pairs, n_dims,
    )
}

/// Encode an array of `(name, body)` records as a TACP container.
///
/// `names` is a `Vec<String>` (zero-copy from JS via `Box<[JsValue]>`).
/// `bodies` is `Box<[Uint8Array]>` (each body is a JS Uint8Array).
///
/// Returns the serialised TACP bytes or a JS error string on failure.
#[wasm_bindgen(js_name = encodeTacpContainer)]
pub fn encode_tacp_container_wasm(
    names: Box<[JsValue]>,
    bodies: Box<[js_sys::Uint8Array]>,
) -> Result<Vec<u8>, JsValue> {
    if names.len() != bodies.len() {
        return Err(JsValue::from_str(
            "names and bodies must be the same length",
        ));
    }
    let mut records = Vec::with_capacity(names.len());
    for (i, (name_js, body_js)) in names.iter().zip(bodies.iter()).enumerate() {
        let name = name_js
            .as_string()
            .ok_or_else(|| JsValue::from_str(&format!("name[{i}] is not a string")))?;
        let body = body_js.to_vec();
        records.push(tac_packet_compiler::custom_binary_container::TacpRecord { name, body });
    }
    tac_packet_compiler::custom_binary_container::encode_container(&records)
        .map_err(|e| JsValue::from_str(&format!("{e}")))
}

/// Decode a TACP container blob back into `(names, bodies)` arrays.
#[wasm_bindgen(js_name = decodeTacpContainer)]
pub fn decode_tacp_container_wasm(blob: &[u8]) -> Result<Box<[JsValue]>, JsValue> {
    let records = tac_packet_compiler::custom_binary_container::decode_container(blob)
        .map_err(|e| JsValue::from_str(&format!("{e}")))?;
    let names = js_sys::Array::new();
    let bodies = js_sys::Array::new();
    for r in records {
        names.push(&JsValue::from_str(&r.name));
        bodies.push(&JsValue::from(js_sys::Uint8Array::from(r.body.as_slice())));
    }
    Ok(vec![JsValue::from(names), JsValue::from(bodies)].into_boxed_slice())
}

/// Encode PR103 latent-hi arithmetic-coded payload (constriction big-endian bytes).
///
/// Mirrors [`tac_packet_compiler::pr103_arithmetic_coding::encode_latent_hi_arithmetic`]
/// — byte-identical to the Python oracle's `latent_hi_arithmetic_v1` golden
/// vector.
#[wasm_bindgen(js_name = encodeLatentHi)]
pub fn encode_latent_hi_wasm(latents: &[u16], histogram: &[f64]) -> Result<Vec<u8>, JsValue> {
    tac_packet_compiler::pr103_arithmetic_coding::latent_hi::encode_latent_hi_arithmetic(
        latents, histogram,
    )
    .map_err(|e| JsValue::from_str(&format!("{e}")))
}

#[wasm_bindgen]
extern "C" {}

// Minimal js_sys dep — only the types we use are pulled in.
#[allow(unused_imports)]
mod _deps {
    pub use js_sys;
}
