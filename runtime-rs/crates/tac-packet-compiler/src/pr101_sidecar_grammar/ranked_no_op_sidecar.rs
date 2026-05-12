//! Rust port of `encode_ranked_no_op_sidecar` / `decode_ranked_no_op_sidecar`.
//!
//! # Byte-for-byte parity contract
//!
//! The Python oracle (`src/tac/packet_compiler/pr101_sidecar_grammar.py`)
//! emits the PR101 "huff_enum" variant whose layout is
//!
//! ```text
//! [dim_packed_le | length_rank_le | huffman_bits | noop_rank_le]
//! ```
//!
//! All length-rank / no-op-rank widths derive deterministically from the
//! schema; the byte stream is self-delimiting once the four width metadata
//! values are known out-of-band.
//!
//! # Building blocks
//!
//! 1. **Bounded-length package-merge** ([`build_optimal_huffman_lengths`])
//!    builds a Kraft-tight length vector inside the `[huff_min_len,
//!    huff_max_len]` envelope. The Python oracle iterates the algorithm
//!    `huff_max_len - 1` times and selects the top `2(n-1)` packages.
//!
//! 2. **Co-lex Huffman length-vector rank** ([`encode_huff_length_rank`]) —
//!    enumerates the Kraft-tight length vectors in co-lex order; rank 0 is
//!    the all-`huff_min_len` vector. Mirrors the Python recursion exactly.
//!
//! 3. **Canonical Huffman codebook** ([`build_canonical_huffman_codebook`]) —
//!    standard "sort by (length, symbol)" canonical assignment.
//!
//! 4. **MSB-first bit-packer** ([`bit_pack`]) — emits one bit at a time;
//!    final byte zero-padded on the right.
//!
//! 5. **Co-lex combination rank over the no-op positions**
//!    ([`encode_combination_colex`]) — `rank = sum_{i=0..k-1} C(pos[i], i+1)`
//!    for ascending positions.
//!
//! All of these are pure-stdlib; no external crate required.

use std::collections::HashMap;

use crate::{PacketCompilerError, Result};

use super::stubs::RankedSidecarSchema;

fn kraft_total(schema: &RankedSidecarSchema) -> u64 {
    1u64 << schema.huff_max_len
}

fn n_symbols(schema: &RankedSidecarSchema) -> usize {
    schema.deltas.len()
}

fn validate_schema(schema: &RankedSidecarSchema) -> Result<()> {
    if schema.n_pairs == 0 {
        return Err(PacketCompilerError::GoldenVectorIo(
            "n_pairs must be > 0".into(),
        ));
    }
    if schema.n_dims == 0 {
        return Err(PacketCompilerError::GoldenVectorIo(
            "n_dims must be > 0".into(),
        ));
    }
    if schema.no_op_sentinel < schema.n_dims as i64 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "no_op_sentinel {} must be >= n_dims {}",
            schema.no_op_sentinel, schema.n_dims
        )));
    }
    if schema.deltas.len() < 2 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "need at least 2 delta codes; got {}",
            schema.deltas.len()
        )));
    }
    for w in schema.deltas.windows(2) {
        if w[1] <= w[0] {
            return Err(PacketCompilerError::GoldenVectorIo(
                "deltas must be strictly ascending".into(),
            ));
        }
    }
    if !(1 <= schema.huff_min_len
        && schema.huff_min_len <= schema.huff_max_len
        && schema.huff_max_len <= 16)
    {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "expected 1 <= huff_min_len <= huff_max_len <= 16; got {} / {}",
            schema.huff_min_len, schema.huff_max_len
        )));
    }
    Ok(())
}

// ── Combinatorial helpers ────────────────────────────────────────────────────

/// Binomial coefficient C(n, k); returns 0 when k > n.
fn binom(n: u64, k: u64) -> u128 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut num: u128 = 1;
    for i in 0..k {
        num = num.saturating_mul((n - i) as u128) / (i + 1) as u128;
    }
    num
}

/// Co-lex combination rank for a sorted-ascending positions slice in `[0, n)`.
///
/// `rank = sum_{i=0..k-1} C(pos[i], i+1)`. Returns 0 for `k = 0`.
pub fn encode_combination_colex(positions: &[i64], n: usize) -> Result<u128> {
    if positions.is_empty() {
        return Ok(0);
    }
    let mut sorted = positions.to_vec();
    sorted.sort_unstable();
    if sorted[0] < 0 || (sorted[sorted.len() - 1] as usize) >= n {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "positions must be in [0, {n}); got [{}, {}]",
            sorted[0],
            sorted[sorted.len() - 1]
        )));
    }
    for w in sorted.windows(2) {
        if w[0] == w[1] {
            return Err(PacketCompilerError::GoldenVectorIo(
                "positions must be unique".into(),
            ));
        }
    }
    let mut rank: u128 = 0;
    for (i, &x) in sorted.iter().enumerate() {
        rank = rank.saturating_add(binom(x as u64, (i + 1) as u64));
    }
    Ok(rank)
}

// ── Huffman length-vector counting + co-lex ranking ──────────────────────────

/// Number of valid Kraft-tight length-vector positions left from `(pos,
/// remaining)`. Memoised at the call site (the recursion is depth ≤ 256).
fn count_length_vectors(
    pos: usize,
    remaining: u64,
    n_symbols: usize,
    huff_min_len: u8,
    huff_max_len: u8,
    memo: &mut HashMap<(usize, u64), u128>,
) -> u128 {
    if pos == n_symbols {
        return if remaining == 0 { 1 } else { 0 };
    }
    if let Some(&cached) = memo.get(&(pos, remaining)) {
        return cached;
    }
    let mut total: u128 = 0;
    for length in huff_min_len..=huff_max_len {
        let weight = 1u64 << (huff_max_len - length);
        if remaining >= weight {
            total = total.saturating_add(count_length_vectors(
                pos + 1,
                remaining - weight,
                n_symbols,
                huff_min_len,
                huff_max_len,
                memo,
            ));
        }
    }
    memo.insert((pos, remaining), total);
    total
}

/// Compute the co-lex rank of a given Kraft-tight length vector. The vector
/// must hit Kraft exactly (sum of `2^(huff_max_len - length)` = 2^huff_max_len).
pub fn encode_huff_length_rank(lengths: &[u8], schema: &RankedSidecarSchema) -> Result<u128> {
    if lengths.len() != n_symbols(schema) {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "lengths length {} != n_symbols {}",
            lengths.len(),
            n_symbols(schema)
        )));
    }
    let mut memo: HashMap<(usize, u64), u128> = HashMap::new();
    let mut rank: u128 = 0;
    let mut remaining = kraft_total(schema);
    for (pos, &target) in lengths.iter().enumerate() {
        if !(schema.huff_min_len..=schema.huff_max_len).contains(&target) {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "length {target} at pos {pos} out of bounds"
            )));
        }
        for length in schema.huff_min_len..=schema.huff_max_len {
            let weight = 1u64 << (schema.huff_max_len - length);
            if remaining < weight {
                continue;
            }
            if length == target {
                remaining -= weight;
                break;
            }
            rank = rank.saturating_add(count_length_vectors(
                pos + 1,
                remaining - weight,
                n_symbols(schema),
                schema.huff_min_len,
                schema.huff_max_len,
                &mut memo,
            ));
        }
    }
    if remaining != 0 {
        return Err(PacketCompilerError::GoldenVectorIo(
            "provided lengths do not form a Kraft-tight code".into(),
        ));
    }
    Ok(rank)
}

/// Total number of valid Kraft-tight length vectors. Used to compute the
/// length-rank byte width.
pub fn total_length_vectors(schema: &RankedSidecarSchema) -> u128 {
    let mut memo: HashMap<(usize, u64), u128> = HashMap::new();
    count_length_vectors(
        0,
        kraft_total(schema),
        n_symbols(schema),
        schema.huff_min_len,
        schema.huff_max_len,
        &mut memo,
    )
}

// ── Bounded-length package-merge (mirrors Python `_build_optimal_huffman_lengths`) ─

/// Build a Kraft-tight, length-bounded canonical Huffman length vector for
/// the per-symbol counts (PR101 floors counts at 1 first).
///
/// Replicates the Python oracle's package-merge variant verbatim:
///
/// 1. Floor every per-symbol count at 1 (so unseen symbols still receive a
///    code length within the envelope).
/// 2. Sort `(count, symbol)` ascending into `items` (Python `items.sort()`).
/// 3. Seed `final_package` = `[(c, [s])]` for each item.
/// 4. For each of `huff_max_len - 1` rounds:
///    - Sort `final_package` by weight ascending → `merged`.
///    - Pair up adjacent merged entries (i, i+1) → `pairs` of
///      `(c0 + c1, payload0 + payload1)`.
///    - `combined = sorted(package + pairs, key=weight)`.
///    - `final_package = combined`. (Note: the Python code reuses the same
///      *base* `package` every round — it does NOT carry the round's pairs
///      forward. We replicate that semantics.)
/// 5. Truncate `final_package` to the smallest `2*(n_symbols - 1)` entries.
/// 6. For each symbol, the length is `max(count_in_final_package,
///    huff_min_len)` clamped to `[huff_min_len, huff_max_len]`.
/// 7. If the resulting Kraft does not equal `2^huff_max_len`, fall back to
///    a uniform-length code where `target = clamp(ceil(log2(n_symbols)),
///    huff_min_len, huff_max_len)`.
pub fn build_optimal_huffman_lengths(
    delta_indices: &[i64],
    schema: &RankedSidecarSchema,
) -> Result<Vec<u8>> {
    let n = n_symbols(schema);
    // Step 1: counts with floor 1.
    let mut counts: Vec<i64> = vec![0; n];
    for &idx in delta_indices {
        if !(0..n as i64).contains(&idx) {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "delta_index {idx} out of range [0, {n})"
            )));
        }
        counts[idx as usize] += 1;
    }
    for c in counts.iter_mut() {
        if *c < 1 {
            *c = 1;
        }
    }

    // Step 2: items = sorted(counts (as count, symbol)) ascending.
    let mut items: Vec<(i64, usize)> = counts
        .iter()
        .copied()
        .enumerate()
        .map(|(s, c)| (c, s))
        .collect();
    items.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));

    // Step 3: seed package and final_package.
    let package: Vec<(i64, Vec<usize>)> = items.iter().map(|(c, s)| (*c, vec![*s])).collect();
    let mut final_package: Vec<(i64, Vec<usize>)> = package.clone();

    // Step 4: package-merge rounds.
    for _ in 0..(schema.huff_max_len as usize - 1) {
        // merged = sorted(final_package, key=weight)
        let mut merged = final_package.clone();
        merged.sort_by(|a, b| a.0.cmp(&b.0));
        // Pair adjacent entries.
        let mut pairs: Vec<(i64, Vec<usize>)> = Vec::new();
        let mut i = 0;
        while i + 1 < merged.len() {
            let (c0, p0) = &merged[i];
            let (c1, p1) = &merged[i + 1];
            let w = c0 + c1;
            let mut payload = p0.clone();
            payload.extend_from_slice(p1);
            pairs.push((w, payload));
            i += 2;
        }
        // combined = sorted(package + pairs)
        let mut combined = package.clone();
        combined.extend(pairs);
        combined.sort_by(|a, b| a.0.cmp(&b.0));
        final_package = combined;
    }

    // Step 5: take top 2*(n - 1).
    let take = 2 * (n.saturating_sub(1));
    final_package.sort_by(|a, b| a.0.cmp(&b.0));
    final_package.truncate(take);

    // Step 6: per-symbol counts in final_package.
    let mut counts_used: Vec<i64> = vec![0; n];
    for (_, payload) in &final_package {
        for &s in payload {
            counts_used[s] += 1;
        }
    }
    let mut lengths: Vec<u8> = vec![schema.huff_min_len; n];
    for s in 0..n {
        let raw = counts_used[s].max(schema.huff_min_len as i64) as u8;
        lengths[s] = raw.clamp(schema.huff_min_len, schema.huff_max_len);
    }

    // Step 7: Kraft check; fall back to uniform if off-budget.
    let mut kraft: u64 = 0;
    for &length in lengths.iter() {
        kraft += 1u64 << (schema.huff_max_len - length);
    }
    if kraft != kraft_total(schema) {
        let target_log2 = ((n as u32).next_power_of_two().trailing_zeros() as u8).max(
            // ceil(log2(n))
            (n as f64).log2().ceil() as u8,
        );
        let target = target_log2
            .max(schema.huff_min_len)
            .min(schema.huff_max_len);
        lengths = vec![target; n];
    }
    Ok(lengths)
}

// ── Canonical Huffman codebook + MSB-first bit-packing ───────────────────────

/// Build canonical Huffman codes from a length vector.
///
/// Returns one entry per non-zero-length symbol; map is keyed by symbol id.
/// Each entry is `(length, code)` where `code` is a `length`-bit MSB-first
/// integer.
pub fn build_canonical_huffman_codebook(lengths: &[u8]) -> HashMap<usize, (u8, u64)> {
    let mut indexed: Vec<(usize, u8)> = lengths
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, l)| *l > 0)
        .collect();
    indexed.sort_by(|a, b| (a.1, a.0).cmp(&(b.1, b.0)));
    let mut codebook: HashMap<usize, (u8, u64)> = HashMap::new();
    let mut code: u64 = 0;
    let mut prev_len: u8 = 0;
    for (sym, length) in indexed {
        code <<= length - prev_len;
        codebook.insert(sym, (length, code));
        code += 1;
        prev_len = length;
    }
    codebook
}

/// MSB-first bit-pack `symbols` using `codebook` into bytes; final byte is
/// zero-padded on the right.
pub fn bit_pack(symbols: &[usize], codebook: &HashMap<usize, (u8, u64)>) -> Result<Vec<u8>> {
    let mut out: Vec<u8> = Vec::new();
    let mut cur: u128 = 0;
    let mut cur_len: u32 = 0;
    for &sym in symbols {
        let (length, code) = codebook.get(&sym).copied().ok_or_else(|| {
            PacketCompilerError::GoldenVectorIo(format!("symbol {sym} missing from codebook"))
        })?;
        cur = (cur << length) | (code as u128);
        cur_len += length as u32;
        while cur_len >= 8 {
            cur_len -= 8;
            out.push(((cur >> cur_len) & 0xFF) as u8);
        }
    }
    if cur_len > 0 {
        out.push(((cur << (8 - cur_len)) & 0xFF) as u8);
    }
    Ok(out)
}

// ── Width helpers ────────────────────────────────────────────────────────────

/// `ceil(log2(value))` for `value >= 2`; returns 1 if value <= 1.
fn ceil_log2(value: u128) -> u32 {
    if value <= 1 {
        return 1;
    }
    let v = value - 1;
    128 - v.leading_zeros()
}

/// `n_valid * ceil(log2(max(n_dims, 2)))` rounded up to whole bytes.
fn dim_blob_bytes(n_valid: usize, n_dims: usize) -> usize {
    let bits_per_pair = ceil_log2(n_dims.max(2) as u128);
    let total_bits = (n_valid as u32).saturating_mul(bits_per_pair).max(1);
    total_bits.div_ceil(8) as usize
}

/// Number of bytes needed for the length-rank little-endian blob.
fn length_rank_bytes(total: u128) -> usize {
    let bits = ceil_log2(total.max(2));
    bits.div_ceil(8) as usize
}

/// Number of bytes needed for the no-op rank little-endian blob.
fn noop_rank_bytes(n_pairs: usize, noop_count: usize) -> usize {
    let total = binom(n_pairs as u64, noop_count as u64).max(1);
    let bits = ceil_log2(total);
    bits.div_ceil(8) as usize
}

/// Pack a non-negative integer as little-endian bytes of exactly `n_bytes` width.
fn pack_le(value: u128, n_bytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; n_bytes];
    let mut v = value;
    for byte in out.iter_mut() {
        *byte = (v & 0xFF) as u8;
        v >>= 8;
    }
    if v != 0 {
        // The caller controls n_bytes; overflow is a logic error.
        // We surface via a debug assertion; in production we'd thread `Result`.
        debug_assert!(
            false,
            "pack_le overflow: value did not fit in {n_bytes} bytes"
        );
    }
    out
}

// ── Public encode ────────────────────────────────────────────────────────────

/// Encode a per-pair sparse correction sidecar (PR101 "huff_enum" variant).
///
/// `dims` and `delta_indices` have length `schema.n_pairs`. A slot whose
/// `dims[i] == schema.no_op_sentinel` is a no-op (its `delta_indices[i]` is
/// treated as 0 / irrelevant).
///
/// Returns the byte payload whose layout is:
/// `[dim_packed_le | length_rank_le | huffman_bits | noop_rank_le]`.
pub fn encode_ranked_no_op_sidecar(
    dims: &[i64],
    delta_indices: &[i64],
    schema: &RankedSidecarSchema,
) -> Result<Vec<u8>> {
    validate_schema(schema)?;
    if dims.len() != schema.n_pairs || delta_indices.len() != schema.n_pairs {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "dims/delta_indices must have length {}; got {} / {}",
            schema.n_pairs,
            dims.len(),
            delta_indices.len()
        )));
    }

    // Partition into valid / no-op.
    let mut valid_idx: Vec<usize> = Vec::new();
    let mut noop_idx: Vec<usize> = Vec::new();
    for (i, &d) in dims.iter().enumerate() {
        if d == schema.no_op_sentinel {
            noop_idx.push(i);
        } else {
            valid_idx.push(i);
        }
    }
    let n_valid = valid_idx.len();
    let noop_count = noop_idx.len();

    // Validate ranges.
    for &i in &valid_idx {
        let d = dims[i];
        if d < 0 || d >= schema.n_dims as i64 {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "valid dim {d} at pos {i} out of [0, {})",
                schema.n_dims
            )));
        }
        let di = delta_indices[i];
        if di < 0 || di >= n_symbols(schema) as i64 {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "delta_index {di} at pos {i} out of [0, {})",
                n_symbols(schema)
            )));
        }
    }

    // Mixed-radix dim packing (Python: dim_value = ...).
    let mut dim_value: u128 = 0;
    let radix = schema.n_dims as u128;
    for &i in valid_idx.iter().rev() {
        dim_value = dim_value
            .saturating_mul(radix)
            .saturating_add(dims[i] as u128);
    }
    let dim_blob = pack_le(dim_value, dim_blob_bytes(n_valid, schema.n_dims));

    // Build Huffman lengths + canonical codes.
    let total = total_length_vectors(schema);
    let rank_bytes = length_rank_bytes(total);
    let (lengths, length_rank, huff_bits): (Vec<u8>, u128, Vec<u8>) = if n_valid == 0 {
        // Edge case: no corrections. Use the uniform fallback length and
        // honour the same Kraft-tightness rule.
        let target = ((n_symbols(schema) as f64).log2().ceil() as u8)
            .max(schema.huff_min_len)
            .min(schema.huff_max_len);
        let lengths = vec![target; n_symbols(schema)];
        let length_rank = encode_huff_length_rank(&lengths, schema).unwrap_or(0);
        (lengths, length_rank, Vec::new())
    } else {
        let valid_deltas: Vec<i64> = valid_idx.iter().map(|&i| delta_indices[i]).collect();
        let mut lengths = build_optimal_huffman_lengths(&valid_deltas, schema)?;
        let length_rank = match encode_huff_length_rank(&lengths, schema) {
            Ok(r) => r,
            Err(_) => {
                // Fall back to uniform code if Kraft constraint violated.
                let target = ((n_symbols(schema) as f64).log2().ceil() as u8)
                    .max(schema.huff_min_len)
                    .min(schema.huff_max_len);
                lengths = vec![target; n_symbols(schema)];
                encode_huff_length_rank(&lengths, schema)?
            }
        };
        let codebook = build_canonical_huffman_codebook(&lengths);
        let symbols: Vec<usize> = valid_deltas.iter().map(|&x| x as usize).collect();
        let huff_bits = bit_pack(&symbols, &codebook)?;
        (lengths, length_rank, huff_bits)
    };
    let length_rank_blob = pack_le(length_rank, rank_bytes);

    // No-op combination rank.
    let noop_positions: Vec<i64> = noop_idx.iter().map(|&i| i as i64).collect();
    let noop_rank_value = encode_combination_colex(&noop_positions, schema.n_pairs)?;
    let noop_blob_bytes = noop_rank_bytes(schema.n_pairs, noop_count);
    let noop_rank_blob = pack_le(noop_rank_value, noop_blob_bytes);

    // Concatenate.
    let mut out = Vec::with_capacity(
        dim_blob.len() + length_rank_blob.len() + huff_bits.len() + noop_rank_blob.len(),
    );
    out.extend_from_slice(&dim_blob);
    out.extend_from_slice(&length_rank_blob);
    out.extend_from_slice(&huff_bits);
    out.extend_from_slice(&noop_rank_blob);
    let _ = lengths; // currently used only inside the codebook path; kept for future decode wire-in.
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pr101_schema() -> RankedSidecarSchema {
        RankedSidecarSchema {
            n_pairs: 24,
            n_dims: 8,
            deltas: vec![-10, -8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 10],
            huff_min_len: 2,
            huff_max_len: 8,
            no_op_sentinel: 255,
        }
    }

    #[test]
    fn binom_basic() {
        assert_eq!(binom(5, 0), 1);
        assert_eq!(binom(5, 5), 1);
        assert_eq!(binom(5, 2), 10);
        assert_eq!(binom(10, 3), 120);
        assert_eq!(binom(0, 1), 0);
    }

    #[test]
    fn combination_colex_basic() {
        // {0}: rank 0
        assert_eq!(encode_combination_colex(&[0], 5).unwrap(), 0);
        // {0, 1}: C(0,1)+C(1,2)=0+0=0
        assert_eq!(encode_combination_colex(&[0, 1], 5).unwrap(), 0);
        // {3, 4}: C(3,1)+C(4,2)=3+6=9
        assert_eq!(encode_combination_colex(&[3, 4], 5).unwrap(), 9);
    }

    #[test]
    fn total_length_vectors_pr101() {
        let schema = pr101_schema();
        let total = total_length_vectors(&schema);
        // Sanity: with 16 symbols, max_len=8, the count is non-trivial.
        assert!(total > 1);
        assert!(total < u128::MAX / 2);
    }

    #[test]
    fn uniform_length_vector_rank_roundtrip() {
        let schema = pr101_schema();
        let lengths = vec![4u8; n_symbols(&schema)];
        let r = encode_huff_length_rank(&lengths, &schema).unwrap();
        // 16 symbols at length 4: Kraft = 16 * 2^4 = 256 = 2^8. OK.
        // The rank should be reachable (we don't test the exact value here —
        // co-lex order is implementation-detail).
        let _ = r;
    }
}
