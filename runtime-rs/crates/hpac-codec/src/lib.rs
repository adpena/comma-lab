//! Native HPAC/HPM1 entropy-coding primitives.
//!
//! This crate is intentionally small: Python/PyTorch still owns HPAC neural
//! probability-row generation. The Rust side owns deterministic 5-symbol range
//! coding once symbols and probability rows or fixed CDFs have been materialized.

use constriction::stream::{
    model::DefaultContiguousCategoricalEntropyModel,
    queue::{DefaultRangeDecoder, DefaultRangeEncoder},
    Decode, Encode,
};

pub const HPAC_SYMBOLS: usize = 5;
pub const DEFAULT_PROB_EPS_F64: f64 = 1e-7;
pub const DEFAULT_PROB_EPS_F32: f32 = 1e-7;
pub const CONSTRICTION_PRECISION: u32 = 24;
pub const FIXED_POINT_TOTAL: u32 = 1u32 << CONSTRICTION_PRECISION;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HpacCodecError {
    LengthMismatch { symbols: usize, models: usize },
    SymbolOutOfRange { symbol: u8 },
    InvalidProbabilityRow,
    InvalidCdf(String),
    InvalidLeBytes { bytes: usize },
    Constriction(String),
}

impl std::fmt::Display for HpacCodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LengthMismatch { symbols, models } => {
                write!(f, "symbol/model length mismatch: {symbols} != {models}")
            }
            Self::SymbolOutOfRange { symbol } => {
                write!(f, "HPAC symbol out of range 0..4: {symbol}")
            }
            Self::InvalidProbabilityRow => write!(f, "invalid HPAC probability row"),
            Self::InvalidCdf(message) => write!(f, "invalid HPAC fixed CDF: {message}"),
            Self::InvalidLeBytes { bytes } => {
                write!(f, "compressed byte stream is not uint32-aligned: {bytes}")
            }
            Self::Constriction(message) => write!(f, "constriction error: {message}"),
        }
    }
}

impl std::error::Error for HpacCodecError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CategoricalQuantization {
    SourceFloat64PerfectFalse,
    SourceFloat64PerfectTrue,
    SourceFloat32PerfectFalse,
    SourceFloat32PerfectTrue,
}

impl CategoricalQuantization {
    pub fn probability_dtype(self) -> &'static str {
        match self {
            Self::SourceFloat64PerfectFalse | Self::SourceFloat64PerfectTrue => "float64",
            Self::SourceFloat32PerfectFalse | Self::SourceFloat32PerfectTrue => "float32",
        }
    }

    pub fn perfect(self) -> bool {
        matches!(
            self,
            Self::SourceFloat64PerfectTrue | Self::SourceFloat32PerfectTrue
        )
    }
}

/// Normalize a PR86/PR91 source-contract float64 row.
///
/// This mirrors `tac.pr86_hpac_codec._normalize_probability_row` for the
/// `source_float64_*` variants: cast/keep float64, clip to `[prob_eps, 1]`,
/// divide by the row sum, and preserve left-to-right summation semantics.
pub fn normalize_probability_row_f64(
    row: [f64; HPAC_SYMBOLS],
    prob_eps: f64,
) -> Result<[f64; HPAC_SYMBOLS], HpacCodecError> {
    if !prob_eps.is_finite() || prob_eps <= 0.0 {
        return Err(HpacCodecError::InvalidProbabilityRow);
    }
    let mut out = [0.0f64; HPAC_SYMBOLS];
    for (dst, src) in out.iter_mut().zip(row) {
        if !src.is_finite() {
            return Err(HpacCodecError::InvalidProbabilityRow);
        }
        *dst = src.clamp(prob_eps, 1.0);
    }
    let sum = out.iter().copied().sum::<f64>();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(HpacCodecError::InvalidProbabilityRow);
    }
    for value in &mut out {
        *value /= sum;
    }
    Ok(out)
}

/// Normalize a PR86/PR91 source-contract float32 row.
///
/// The returned row stays `f32` so constriction receives the same dtype family
/// as the Python off-contract `source_float32_*` probes.
pub fn normalize_probability_row_f32(
    row: [f32; HPAC_SYMBOLS],
    prob_eps: f32,
) -> Result<[f32; HPAC_SYMBOLS], HpacCodecError> {
    if !prob_eps.is_finite() || prob_eps <= 0.0 {
        return Err(HpacCodecError::InvalidProbabilityRow);
    }
    let mut out = [0.0f32; HPAC_SYMBOLS];
    for (dst, src) in out.iter_mut().zip(row) {
        if !src.is_finite() {
            return Err(HpacCodecError::InvalidProbabilityRow);
        }
        *dst = src.clamp(prob_eps, 1.0);
    }
    let sum = out.iter().copied().sum::<f32>();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(HpacCodecError::InvalidProbabilityRow);
    }
    for value in &mut out {
        *value /= sum;
    }
    Ok(out)
}

pub fn encode_f64_probability_rows(
    symbols: &[u8],
    probability_rows: &[[f64; HPAC_SYMBOLS]],
) -> Result<Vec<u32>, HpacCodecError> {
    encode_f64_probability_rows_with_options(symbols, probability_rows, DEFAULT_PROB_EPS_F64, false)
}

pub fn encode_f64_probability_rows_with_options(
    symbols: &[u8],
    probability_rows: &[[f64; HPAC_SYMBOLS]],
    prob_eps: f64,
    perfect: bool,
) -> Result<Vec<u32>, HpacCodecError> {
    require_equal_lengths(symbols, probability_rows.len())?;
    let mut encoder = DefaultRangeEncoder::new();
    for (&symbol, row) in symbols.iter().zip(probability_rows) {
        require_symbol(symbol)?;
        let normalized = normalize_probability_row_f64(*row, prob_eps)?;
        let model = build_f64_model(&normalized, perfect)?;
        encoder
            .encode_symbol(symbol as usize, &model)
            .map_err(|err| HpacCodecError::Constriction(format!("{err:?}")))?;
    }
    encoder
        .into_compressed()
        .map_err(|err| HpacCodecError::Constriction(format!("{err:?}")))
}

pub fn decode_f64_probability_rows(
    words: &[u32],
    probability_rows: &[[f64; HPAC_SYMBOLS]],
) -> Result<Vec<u8>, HpacCodecError> {
    decode_f64_probability_rows_with_options(words, probability_rows, DEFAULT_PROB_EPS_F64, false)
}

pub fn decode_f64_probability_rows_with_options(
    words: &[u32],
    probability_rows: &[[f64; HPAC_SYMBOLS]],
    prob_eps: f64,
    perfect: bool,
) -> Result<Vec<u8>, HpacCodecError> {
    let mut decoder = DefaultRangeDecoder::from_compressed(words.to_vec())
        .map_err(|err| HpacCodecError::Constriction(format!("{err:?}")))?;
    let mut out = Vec::with_capacity(probability_rows.len());
    for row in probability_rows {
        let normalized = normalize_probability_row_f64(*row, prob_eps)?;
        let model = build_f64_model(&normalized, perfect)?;
        let symbol: usize = decoder
            .decode_symbol(&model)
            .map_err(|err| HpacCodecError::Constriction(format!("{err:?}")))?;
        if symbol >= HPAC_SYMBOLS {
            return Err(HpacCodecError::SymbolOutOfRange {
                symbol: symbol as u8,
            });
        }
        out.push(symbol as u8);
    }
    Ok(out)
}

pub fn encode_f32_probability_rows_with_options(
    symbols: &[u8],
    probability_rows: &[[f32; HPAC_SYMBOLS]],
    prob_eps: f32,
    perfect: bool,
) -> Result<Vec<u32>, HpacCodecError> {
    require_equal_lengths(symbols, probability_rows.len())?;
    let mut encoder = DefaultRangeEncoder::new();
    for (&symbol, row) in symbols.iter().zip(probability_rows) {
        require_symbol(symbol)?;
        let normalized = normalize_probability_row_f32(*row, prob_eps)?;
        let model = build_f32_model(&normalized, perfect)?;
        encoder
            .encode_symbol(symbol as usize, &model)
            .map_err(|err| HpacCodecError::Constriction(format!("{err:?}")))?;
    }
    encoder
        .into_compressed()
        .map_err(|err| HpacCodecError::Constriction(format!("{err:?}")))
}

pub fn decode_f32_probability_rows_with_options(
    words: &[u32],
    probability_rows: &[[f32; HPAC_SYMBOLS]],
    prob_eps: f32,
    perfect: bool,
) -> Result<Vec<u8>, HpacCodecError> {
    let mut decoder = DefaultRangeDecoder::from_compressed(words.to_vec())
        .map_err(|err| HpacCodecError::Constriction(format!("{err:?}")))?;
    let mut out = Vec::with_capacity(probability_rows.len());
    for row in probability_rows {
        let normalized = normalize_probability_row_f32(*row, prob_eps)?;
        let model = build_f32_model(&normalized, perfect)?;
        let symbol: usize = decoder
            .decode_symbol(&model)
            .map_err(|err| HpacCodecError::Constriction(format!("{err:?}")))?;
        if symbol >= HPAC_SYMBOLS {
            return Err(HpacCodecError::SymbolOutOfRange {
                symbol: symbol as u8,
            });
        }
        out.push(symbol as u8);
    }
    Ok(out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cdf5 {
    cumulative: [u32; HPAC_SYMBOLS + 1],
}

impl Cdf5 {
    pub fn new(cumulative: [u32; HPAC_SYMBOLS + 1]) -> Result<Self, HpacCodecError> {
        if cumulative[0] != 0 {
            return Err(HpacCodecError::InvalidCdf(
                "first cumulative entry must be zero".to_string(),
            ));
        }
        if cumulative[HPAC_SYMBOLS] != FIXED_POINT_TOTAL {
            return Err(HpacCodecError::InvalidCdf(format!(
                "last cumulative entry must be {FIXED_POINT_TOTAL}"
            )));
        }
        for pair in cumulative.windows(2) {
            if pair[0] >= pair[1] {
                return Err(HpacCodecError::InvalidCdf(
                    "cumulative entries must be strictly increasing".to_string(),
                ));
            }
        }
        Ok(Self { cumulative })
    }

    pub fn from_nonzero_counts(counts: [u32; HPAC_SYMBOLS]) -> Result<Self, HpacCodecError> {
        if counts.iter().any(|&value| value == 0) {
            return Err(HpacCodecError::InvalidCdf(
                "all adaptive counts must be nonzero".to_string(),
            ));
        }
        let sum = counts
            .iter()
            .try_fold(0u128, |acc, &value| acc.checked_add(value as u128))
            .ok_or_else(|| HpacCodecError::InvalidCdf("count sum overflow".to_string()))?;
        if sum == 0 {
            return Err(HpacCodecError::InvalidCdf(
                "count sum must be positive".to_string(),
            ));
        }

        let free = (FIXED_POINT_TOTAL - HPAC_SYMBOLS as u32) as u128;
        let mut weights = [1u32; HPAC_SYMBOLS];
        let mut remainders = [(0u128, 0usize); HPAC_SYMBOLS];
        let mut used = HPAC_SYMBOLS as u32;
        for (idx, &count) in counts.iter().enumerate() {
            let scaled = (count as u128) * free;
            let extra = (scaled / sum) as u32;
            weights[idx] = weights[idx]
                .checked_add(extra)
                .ok_or_else(|| HpacCodecError::InvalidCdf("weight overflow".to_string()))?;
            used = used
                .checked_add(extra)
                .ok_or_else(|| HpacCodecError::InvalidCdf("weight sum overflow".to_string()))?;
            remainders[idx] = (scaled % sum, idx);
        }
        let mut remaining = FIXED_POINT_TOTAL
            .checked_sub(used)
            .ok_or_else(|| HpacCodecError::InvalidCdf("weights exceed total".to_string()))?;
        remainders.sort_by(|left, right| right.cmp(left));
        for &(_, idx) in remainders.iter().cycle().take(remaining as usize) {
            weights[idx] += 1;
            remaining -= 1;
            if remaining == 0 {
                break;
            }
        }
        Self::from_weights(weights)
    }

    pub fn from_weights(weights: [u32; HPAC_SYMBOLS]) -> Result<Self, HpacCodecError> {
        if weights.iter().any(|&value| value == 0) {
            return Err(HpacCodecError::InvalidCdf(
                "all fixed-point weights must be nonzero".to_string(),
            ));
        }
        let mut cumulative = [0u32; HPAC_SYMBOLS + 1];
        for (idx, &weight) in weights.iter().enumerate() {
            cumulative[idx + 1] = cumulative[idx]
                .checked_add(weight)
                .ok_or_else(|| HpacCodecError::InvalidCdf("CDF overflow".to_string()))?;
        }
        Self::new(cumulative)
    }

    pub fn uniform() -> Self {
        // 2^24 is not divisible by five; assign the extra unit to the final
        // symbol deterministically.
        Self::from_weights([3_355_443, 3_355_443, 3_355_443, 3_355_443, 3_355_444])
            .expect("uniform CDF is valid")
    }

    pub fn cumulative(self) -> [u32; HPAC_SYMBOLS + 1] {
        self.cumulative
    }

    pub fn weights(self) -> [u32; HPAC_SYMBOLS] {
        let mut out = [0u32; HPAC_SYMBOLS];
        for (idx, value) in out.iter_mut().enumerate() {
            *value = self.cumulative[idx + 1] - self.cumulative[idx];
        }
        out
    }

    fn model(self) -> Result<DefaultContiguousCategoricalEntropyModel, HpacCodecError> {
        let weights = self.weights();
        DefaultContiguousCategoricalEntropyModel::from_nonzero_fixed_point_probabilities(
            weights.iter().copied(),
            false,
        )
        .map_err(|()| HpacCodecError::InvalidCdf("constriction rejected weights".to_string()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptiveCdf5 {
    counts: [u32; HPAC_SYMBOLS],
    update_delta: u32,
    rescale_threshold: u32,
}

impl AdaptiveCdf5 {
    pub fn new(
        counts: [u32; HPAC_SYMBOLS],
        update_delta: u32,
        rescale_threshold: u32,
    ) -> Result<Self, HpacCodecError> {
        if update_delta == 0 {
            return Err(HpacCodecError::InvalidCdf(
                "adaptive update delta must be nonzero".to_string(),
            ));
        }
        Cdf5::from_nonzero_counts(counts)?;
        Ok(Self {
            counts,
            update_delta,
            rescale_threshold: rescale_threshold.max(HPAC_SYMBOLS as u32),
        })
    }

    pub fn unit() -> Self {
        Self::new([1; HPAC_SYMBOLS], 1, 1 << 20).expect("unit adaptive model is valid")
    }

    pub fn cdf(&self) -> Result<Cdf5, HpacCodecError> {
        Cdf5::from_nonzero_counts(self.counts)
    }

    pub fn observe(&mut self, symbol: u8) -> Result<(), HpacCodecError> {
        require_symbol(symbol)?;
        let idx = symbol as usize;
        self.counts[idx] = self.counts[idx]
            .checked_add(self.update_delta)
            .ok_or_else(|| HpacCodecError::InvalidCdf("adaptive count overflow".to_string()))?;
        let total = self.counts.iter().map(|&value| value as u64).sum::<u64>();
        if total >= self.rescale_threshold as u64 {
            for value in &mut self.counts {
                *value = ((*value + 1) >> 1).max(1);
            }
        }
        Ok(())
    }
}

pub fn encode_with_cdfs(symbols: &[u8], cdfs: &[Cdf5]) -> Result<Vec<u32>, HpacCodecError> {
    require_equal_lengths(symbols, cdfs.len())?;
    let mut encoder = DefaultRangeEncoder::new();
    for (&symbol, cdf) in symbols.iter().zip(cdfs) {
        require_symbol(symbol)?;
        let model = cdf.model()?;
        encoder
            .encode_symbol(symbol as usize, &model)
            .map_err(|err| HpacCodecError::Constriction(format!("{err:?}")))?;
    }
    encoder
        .into_compressed()
        .map_err(|err| HpacCodecError::Constriction(format!("{err:?}")))
}

pub fn decode_with_cdfs(words: &[u32], cdfs: &[Cdf5]) -> Result<Vec<u8>, HpacCodecError> {
    let mut decoder = DefaultRangeDecoder::from_compressed(words.to_vec())
        .map_err(|err| HpacCodecError::Constriction(format!("{err:?}")))?;
    let mut out = Vec::with_capacity(cdfs.len());
    for cdf in cdfs {
        let model = cdf.model()?;
        let symbol: usize = decoder
            .decode_symbol(&model)
            .map_err(|err| HpacCodecError::Constriction(format!("{err:?}")))?;
        if symbol >= HPAC_SYMBOLS {
            return Err(HpacCodecError::SymbolOutOfRange {
                symbol: symbol as u8,
            });
        }
        out.push(symbol as u8);
    }
    Ok(out)
}

pub fn encode_adaptive_symbols(symbols: &[u8]) -> Result<Vec<u32>, HpacCodecError> {
    let mut adaptive = AdaptiveCdf5::unit();
    let mut cdfs = Vec::with_capacity(symbols.len());
    for &symbol in symbols {
        require_symbol(symbol)?;
        cdfs.push(adaptive.cdf()?);
        adaptive.observe(symbol)?;
    }
    encode_with_cdfs(symbols, &cdfs)
}

pub fn decode_adaptive_symbols(
    words: &[u32],
    symbol_count: usize,
) -> Result<Vec<u8>, HpacCodecError> {
    let mut decoder = DefaultRangeDecoder::from_compressed(words.to_vec())
        .map_err(|err| HpacCodecError::Constriction(format!("{err:?}")))?;
    let mut adaptive = AdaptiveCdf5::unit();
    let mut out = Vec::with_capacity(symbol_count);
    for _ in 0..symbol_count {
        let model = adaptive.cdf()?.model()?;
        let symbol: usize = decoder
            .decode_symbol(&model)
            .map_err(|err| HpacCodecError::Constriction(format!("{err:?}")))?;
        if symbol >= HPAC_SYMBOLS {
            return Err(HpacCodecError::SymbolOutOfRange {
                symbol: symbol as u8,
            });
        }
        let symbol = symbol as u8;
        out.push(symbol);
        adaptive.observe(symbol)?;
    }
    Ok(out)
}

pub fn words_to_le_bytes(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for word in words {
        out.extend_from_slice(&word.to_le_bytes());
    }
    out
}

pub fn words_from_le_bytes(bytes: &[u8]) -> Result<Vec<u32>, HpacCodecError> {
    if bytes.len() % 4 != 0 {
        return Err(HpacCodecError::InvalidLeBytes { bytes: bytes.len() });
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("chunk size checked")))
        .collect())
}

fn build_f64_model(
    row: &[f64; HPAC_SYMBOLS],
    perfect: bool,
) -> Result<DefaultContiguousCategoricalEntropyModel, HpacCodecError> {
    if perfect {
        DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_perfect(row)
    } else {
        DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast(row, None)
    }
    .map_err(|()| HpacCodecError::InvalidProbabilityRow)
}

fn build_f32_model(
    row: &[f32; HPAC_SYMBOLS],
    perfect: bool,
) -> Result<DefaultContiguousCategoricalEntropyModel, HpacCodecError> {
    if perfect {
        DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_perfect(row)
    } else {
        DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast(row, None)
    }
    .map_err(|()| HpacCodecError::InvalidProbabilityRow)
}

fn require_equal_lengths(symbols: &[u8], models: usize) -> Result<(), HpacCodecError> {
    if symbols.len() != models {
        return Err(HpacCodecError::LengthMismatch {
            symbols: symbols.len(),
            models,
        });
    }
    Ok(())
}

fn require_symbol(symbol: u8) -> Result<(), HpacCodecError> {
    if symbol as usize >= HPAC_SYMBOLS {
        return Err(HpacCodecError::SymbolOutOfRange { symbol });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PYTHON_F64_ROWS: [[f64; HPAC_SYMBOLS]; 3] = [
        [0.30, 0.10, 0.10, 0.30, 0.20],
        [0.10, 0.40, 0.20, 0.10, 0.20],
        [0.40, 0.20, 0.10, 0.20, 0.10],
    ];
    const PYTHON_SYMBOLS: [u8; 3] = [0, 4, 1];
    const PYTHON_CONSTRICTION_042_F64_FAST_WORDS: [u32; 1] = [0x4395_8018];
    const PYTHON_ADAPTIVE_SYMBOLS: [u8; 32] = [
        0, 1, 4, 4, 2, 3, 3, 3, 1, 0, 2, 4, 0, 0, 1, 2, 3, 4, 4, 4, 2, 1, 0, 3, 2, 2, 1, 4, 0, 3,
        3, 1,
    ];
    const PYTHON_CONSTRICTION_042_ADAPTIVE_F64_FAST_WORDS: [u32; 3] =
        [0x1974_03aa, 0x40f0_50c1, 0xe60e_4cac];

    #[test]
    fn f64_fast_matches_python_constriction_042_fixture() {
        let words = encode_f64_probability_rows(&PYTHON_SYMBOLS, &PYTHON_F64_ROWS)
            .expect("fixture encodes");
        assert_eq!(words, PYTHON_CONSTRICTION_042_F64_FAST_WORDS);
        assert_eq!(words_to_le_bytes(&words), [0x18, 0x80, 0x95, 0x43]);

        let decoded =
            decode_f64_probability_rows(&words, &PYTHON_F64_ROWS).expect("fixture decodes");
        assert_eq!(decoded, PYTHON_SYMBOLS);
    }

    #[test]
    fn adaptive_f64_fast_matches_python_constriction_042_queue_words() {
        let rows = adaptive_f64_rows_from_unit_counts(&PYTHON_ADAPTIVE_SYMBOLS);
        let words = encode_f64_probability_rows(&PYTHON_ADAPTIVE_SYMBOLS, &rows)
            .expect("adaptive fixture encodes");
        assert_eq!(words, PYTHON_CONSTRICTION_042_ADAPTIVE_F64_FAST_WORDS);
        assert_eq!(
            words_to_le_bytes(&words),
            [0xaa, 0x03, 0x74, 0x19, 0xc1, 0x50, 0xf0, 0x40, 0xac, 0x4c, 0x0e, 0xe6]
        );

        let decoded = decode_f64_probability_rows(&words, &rows).expect("adaptive fixture decodes");
        assert_eq!(decoded, PYTHON_ADAPTIVE_SYMBOLS);
    }

    #[test]
    fn fixed_cdf_stream_roundtrips_five_symbols() {
        let cdfs = [
            Cdf5::uniform(),
            Cdf5::from_weights([4_000_000, 2_000_000, 3_000_000, 3_000_000, 4_777_216]).unwrap(),
            Cdf5::from_nonzero_counts([2, 3, 5, 7, 11]).unwrap(),
            Cdf5::from_nonzero_counts([13, 8, 5, 3, 2]).unwrap(),
            Cdf5::from_nonzero_counts([1, 1, 8, 1, 1]).unwrap(),
        ];
        let symbols = [4, 0, 2, 3, 2];
        let words = encode_with_cdfs(&symbols, &cdfs).expect("fixed CDF encode");
        assert!(!words.is_empty());
        let decoded = decode_with_cdfs(&words, &cdfs).expect("fixed CDF decode");
        assert_eq!(decoded, symbols);
    }

    #[test]
    fn adaptive_five_symbol_stream_roundtrips() {
        let symbols = [0, 1, 4, 4, 2, 3, 3, 3, 1, 0, 2, 4, 0, 0, 1, 2, 3, 4, 4, 4];
        let words = encode_adaptive_symbols(&symbols).expect("adaptive encode");
        assert!(!words.is_empty());
        let decoded = decode_adaptive_symbols(&words, symbols.len()).expect("adaptive decode");
        assert_eq!(decoded, symbols);
    }

    #[test]
    fn byte_word_helpers_are_little_endian_and_fail_closed() {
        let words = [0x4395_8018, 0x0102_0304];
        let bytes = words_to_le_bytes(&words);
        assert_eq!(bytes, [0x18, 0x80, 0x95, 0x43, 0x04, 0x03, 0x02, 0x01]);
        assert_eq!(words_from_le_bytes(&bytes).unwrap(), words);
        assert!(matches!(
            words_from_le_bytes(&bytes[..7]),
            Err(HpacCodecError::InvalidLeBytes { bytes: 7 })
        ));
    }

    #[test]
    fn rejects_out_of_range_symbols_and_bad_cdfs() {
        assert!(matches!(
            encode_f64_probability_rows(&[5], &[[0.2; HPAC_SYMBOLS]]),
            Err(HpacCodecError::SymbolOutOfRange { symbol: 5 })
        ));
        assert!(Cdf5::new([0, 1, 1, 3, 4, FIXED_POINT_TOTAL]).is_err());
        assert!(Cdf5::from_weights([1, 1, 1, 1, 1]).is_err());
    }

    fn adaptive_f64_rows_from_unit_counts(symbols: &[u8]) -> Vec<[f64; HPAC_SYMBOLS]> {
        let mut counts = [1.0f64; HPAC_SYMBOLS];
        let mut rows = Vec::with_capacity(symbols.len());
        for &symbol in symbols {
            require_symbol(symbol).expect("test symbols stay in range");
            let sum = counts.iter().copied().sum::<f64>();
            let mut row = [0.0f64; HPAC_SYMBOLS];
            for (dst, count) in row.iter_mut().zip(counts) {
                *dst = count / sum;
            }
            rows.push(row);
            counts[symbol as usize] += 1.0;
        }
        rows
    }
}
