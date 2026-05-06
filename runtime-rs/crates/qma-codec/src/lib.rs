use std::collections::HashMap;

const QMA9_MAGIC: &[u8; 4] = b"QMA9";
const QMA9_HEADER_BYTES: usize = 20;
const QMA9_SENTINEL: u8 = 5;
const QMA9_CLASS_SYMBOLS: u8 = 5;
const QMA9_CONTEXTS: u32 = 6u32.pow(9);
const QMA9_SCALE_TOTAL: u32 = 65_535;
const QMA9_UPDATE_DELTA: u32 = 20;

const ARITH_TOP: u32 = 0xffff_ffff;
const ARITH_HALF: u32 = 0x8000_0000;
const ARITH_FIRST_QTR: u32 = 0x4000_0000;
const ARITH_THIRD_QTR: u32 = 0xc000_0000;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Fail-closed errors for QMA-family wire parsing and decode.
pub enum QmaError {
    PayloadTooShort { len: usize },
    BadMagic { found: [u8; 4] },
    BadDimensions,
    BitstreamOverrun { declared: usize, available: usize },
    DecodedSizeOverflow,
    CompactBundleTooShort { len: usize },
    CompactBundleBadLengths,
    CompactBundleTruncated { declared: usize, available: usize },
    PrefixOutOfRange { requested: usize, available: usize },
    ContextSymbolOutOfRange { symbol: u8 },
    ArithmeticSymbolOutOfRange,
    InvalidDecodedClass { symbol: u8 },
}

impl std::fmt::Display for QmaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PayloadTooShort { len } => {
                write!(f, "QMA9 payload is shorter than its 20-byte header: {len}")
            }
            Self::BadMagic { found } => write!(f, "expected QMA9 magic, got {found:?}"),
            Self::BadDimensions => write!(f, "QMA9 dimensions must be positive"),
            Self::BitstreamOverrun {
                declared,
                available,
            } => {
                write!(
                    f,
                    "QMA9 bitstream declares {declared} bytes but payload has {available} bytes"
                )
            }
            Self::DecodedSizeOverflow => write!(f, "QMA9 decoded mask size overflow"),
            Self::CompactBundleTooShort { len } => {
                write!(
                    f,
                    "compact mask bundle is shorter than its 24-byte v5 header: {len}"
                )
            }
            Self::CompactBundleBadLengths => {
                write!(f, "compact mask bundle has invalid segment lengths")
            }
            Self::CompactBundleTruncated {
                declared,
                available,
            } => {
                write!(
                    f,
                    "compact mask bundle declares {declared} bytes but payload has {available} bytes"
                )
            }
            Self::PrefixOutOfRange {
                requested,
                available,
            } => {
                write!(
                    f,
                    "QMA9 prefix frame count {requested} exceeds payload frames {available}"
                )
            }
            Self::ContextSymbolOutOfRange { symbol } => {
                write!(f, "QMA9 context symbol out of range: {symbol}")
            }
            Self::ArithmeticSymbolOutOfRange => {
                write!(f, "QMA9 arithmetic decode symbol out of range")
            }
            Self::InvalidDecodedClass { symbol } => {
                write!(f, "decoded invalid QMA9 class symbol: {symbol}")
            }
        }
    }
}

impl std::error::Error for QmaError {}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Parsed QMA9 semantic mask header.
pub struct Qma9Header {
    pub frame_count: u32,
    pub width: u32,
    pub height: u32,
    pub bitstream_bytes: u32,
    pub header_bytes: usize,
    pub packed_bytes: usize,
    pub decoded_mask_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Decoded QMA9 mask bytes in storage order.
///
/// Storage order is frame-major with each frame laid out as
/// `header.width x header.height`, matching the public QMA9 arithmetic stream
/// before any replay-runtime transpose.
pub struct Qma9DecodedMask {
    pub header: Qma9Header,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Parsed PR92/PR91-style compact bundle v5 micro header.
///
/// The wire header stores eight 24-bit little-endian lengths. The PR92 runtime
/// then consumes fixed 223-byte `bias`, fixed 273-byte `region`, and a
/// nonempty `randmulti` tail after those eight variable segments.
pub struct CompactBundleV5MicroHeader {
    pub header_bytes: usize,
    pub mask_bytes: usize,
    pub model_bytes: usize,
    pub pose_bytes: usize,
    pub post_bytes: usize,
    pub shift_bytes: usize,
    pub frac_bytes: usize,
    pub frac2_bytes: usize,
    pub frac3_bytes: usize,
    pub bias_bytes: usize,
    pub region_bytes: usize,
    pub randmulti_bytes: usize,
    pub packed_bytes: usize,
}

/// Parse a complete QMA9 payload header and require that declared bitstream
/// bytes are present in `payload`.
pub fn parse_qma9_header(payload: &[u8]) -> Result<Qma9Header, QmaError> {
    parse_qma9_header_impl(payload, true)
}

/// Parse only the fixed 20-byte QMA9 header.
///
/// This is for custody/profile checks where only member header bytes are
/// available. Use `parse_qma9_header` or the decode functions before consuming
/// score-affecting payloads.
pub fn parse_qma9_header_prefix(payload: &[u8]) -> Result<Qma9Header, QmaError> {
    parse_qma9_header_impl(payload, false)
}

/// Parse the 24-byte compact bundle v5 micro header used by PR92 intake.
///
/// This only establishes archive anatomy and the mask slice boundary. It does
/// not inflate model/post streams or make any score claim.
pub fn parse_compact_bundle_v5_micro_header(
    payload: &[u8],
) -> Result<CompactBundleV5MicroHeader, QmaError> {
    const HEADER_BYTES: usize = 24;
    const FIXED_BIAS_BYTES: usize = 223;
    const FIXED_REGION_BYTES: usize = 273;

    if payload.len() < HEADER_BYTES {
        return Err(QmaError::CompactBundleTooShort { len: payload.len() });
    }

    let mask_bytes = read_u24_le(payload, 0);
    let model_bytes = read_u24_le(payload, 3);
    let pose_bytes = read_u24_le(payload, 6);
    let post_bytes = read_u24_le(payload, 9);
    let shift_bytes = read_u24_le(payload, 12);
    let frac_bytes = read_u24_le(payload, 15);
    let frac2_bytes = read_u24_le(payload, 18);
    let frac3_bytes = read_u24_le(payload, 21);

    if mask_bytes <= 1000
        || model_bytes <= 1000
        || pose_bytes <= 100
        || post_bytes == 0
        || post_bytes >= 10_000
        || shift_bytes == 0
        || shift_bytes >= 10_000
        || frac_bytes == 0
        || frac_bytes >= 10_000
        || frac2_bytes == 0
        || frac2_bytes >= 10_000
        || frac3_bytes == 0
        || frac3_bytes >= 10_000
    {
        return Err(QmaError::CompactBundleBadLengths);
    }

    let fixed_prefix = [
        HEADER_BYTES,
        mask_bytes,
        model_bytes,
        pose_bytes,
        post_bytes,
        shift_bytes,
        frac_bytes,
        frac2_bytes,
        frac3_bytes,
        FIXED_BIAS_BYTES,
        FIXED_REGION_BYTES,
    ]
    .into_iter()
    .try_fold(0usize, |acc, value| acc.checked_add(value))
    .ok_or(QmaError::DecodedSizeOverflow)?;
    if fixed_prefix >= payload.len() {
        return Err(QmaError::CompactBundleTruncated {
            declared: fixed_prefix + 1,
            available: payload.len(),
        });
    }

    Ok(CompactBundleV5MicroHeader {
        header_bytes: HEADER_BYTES,
        mask_bytes,
        model_bytes,
        pose_bytes,
        post_bytes,
        shift_bytes,
        frac_bytes,
        frac2_bytes,
        frac3_bytes,
        bias_bytes: FIXED_BIAS_BYTES,
        region_bytes: FIXED_REGION_BYTES,
        randmulti_bytes: payload.len() - fixed_prefix,
        packed_bytes: payload.len(),
    })
}

/// Return the QMA-family mask slice from a v5 micro compact bundle.
pub fn compact_bundle_v5_micro_mask_slice(payload: &[u8]) -> Result<&[u8], QmaError> {
    let header = parse_compact_bundle_v5_micro_header(payload)?;
    Ok(&payload[header.header_bytes..header.header_bytes + header.mask_bytes])
}

fn read_u24_le(payload: &[u8], offset: usize) -> usize {
    usize::from(payload[offset])
        | (usize::from(payload[offset + 1]) << 8)
        | (usize::from(payload[offset + 2]) << 16)
}

fn parse_qma9_header_impl(
    payload: &[u8],
    require_full_payload: bool,
) -> Result<Qma9Header, QmaError> {
    if payload.len() < QMA9_HEADER_BYTES {
        return Err(QmaError::PayloadTooShort { len: payload.len() });
    }
    let magic: [u8; 4] = payload[0..4].try_into().expect("slice length checked");
    if &magic != QMA9_MAGIC {
        return Err(QmaError::BadMagic { found: magic });
    }
    let frame_count = u32::from_le_bytes(payload[4..8].try_into().expect("slice length checked"));
    let width = u32::from_le_bytes(payload[8..12].try_into().expect("slice length checked"));
    let height = u32::from_le_bytes(payload[12..16].try_into().expect("slice length checked"));
    let bitstream_bytes =
        u32::from_le_bytes(payload[16..20].try_into().expect("slice length checked"));
    if frame_count == 0 || width == 0 || height == 0 {
        return Err(QmaError::BadDimensions);
    }
    let bitstream_len = bitstream_bytes as usize;
    let packed_bytes = QMA9_HEADER_BYTES
        .checked_add(bitstream_len)
        .ok_or(QmaError::DecodedSizeOverflow)?;
    if require_full_payload && packed_bytes > payload.len() {
        return Err(QmaError::BitstreamOverrun {
            declared: bitstream_len,
            available: payload.len(),
        });
    }
    let decoded_mask_bytes = (frame_count as usize)
        .checked_mul(width as usize)
        .and_then(|v| v.checked_mul(height as usize))
        .ok_or(QmaError::DecodedSizeOverflow)?;
    Ok(Qma9Header {
        frame_count,
        width,
        height,
        bitstream_bytes,
        header_bytes: QMA9_HEADER_BYTES,
        packed_bytes,
        decoded_mask_bytes,
    })
}

/// Decode every frame from a complete QMA9 payload.
pub fn decode_qma9_mask(payload: &[u8]) -> Result<Qma9DecodedMask, QmaError> {
    let header = parse_qma9_header(payload)?;
    let data = decode_qma9_pixels(payload, &header, header.frame_count as usize)?;
    Ok(Qma9DecodedMask { header, data })
}

/// Decode the first `frame_count` complete frames from a complete QMA9 payload.
pub fn decode_qma9_prefix_frames(
    payload: &[u8],
    frame_count: usize,
) -> Result<Qma9DecodedMask, QmaError> {
    let header = parse_qma9_header(payload)?;
    if frame_count == 0 || frame_count > header.frame_count as usize {
        return Err(QmaError::PrefixOutOfRange {
            requested: frame_count,
            available: header.frame_count as usize,
        });
    }
    let data = decode_qma9_pixels(payload, &header, frame_count)?;
    Ok(Qma9DecodedMask { header, data })
}

fn decode_qma9_pixels(
    payload: &[u8],
    header: &Qma9Header,
    frame_count: usize,
) -> Result<Vec<u8>, QmaError> {
    let bitstream = &payload[QMA9_HEADER_BYTES..header.packed_bytes];
    let mut decoder = ArithmeticDecoder::new(bitstream);
    let mut model = AdaptiveModel9Binary::new();
    let rows = header.width as usize;
    let cols = header.height as usize;
    let frame_size = rows
        .checked_mul(cols)
        .ok_or(QmaError::DecodedSizeOverflow)?;
    let mut out = vec![
        0u8;
        frame_count
            .checked_mul(frame_size)
            .ok_or(QmaError::DecodedSizeOverflow)?
    ];

    for t in 0..frame_count {
        for y in 0..rows {
            let base = t * frame_size + y * cols;
            for x in 0..cols {
                let n = neighbours(&out, frame_size, t, y, rows, cols, x);
                let ctx_id = qma9_context_id(
                    n.prev,
                    n.left,
                    n.up,
                    n.up_left,
                    n.up_right,
                    n.prev_right,
                    n.prev_down,
                    n.up2,
                    n.left2,
                )?;
                let ctx = model.context(ctx_id);

                let bit = decode_symbol(&mut decoder, &ctx.up_freq)?;
                update_adaptive(&mut ctx.up_freq, bit);
                let cls = if bit != 0 {
                    n.up
                } else {
                    let bit = decode_symbol(&mut decoder, &ctx.left_freq)?;
                    update_adaptive(&mut ctx.left_freq, bit);
                    if bit != 0 {
                        n.left
                    } else {
                        let bit = decode_symbol(&mut decoder, &ctx.prev_freq)?;
                        update_adaptive(&mut ctx.prev_freq, bit);
                        if bit != 0 {
                            n.prev
                        } else {
                            let cls = decode_symbol(&mut decoder, &ctx.class_freq)?;
                            update_adaptive(&mut ctx.class_freq, cls);
                            cls as u8
                        }
                    }
                };

                if cls >= QMA9_CLASS_SYMBOLS {
                    return Err(QmaError::InvalidDecodedClass { symbol: cls });
                }
                out[base + x] = cls;
            }
        }
    }

    Ok(out)
}

/// Return the base-6 adaptive context id used by the QMA9 mask stream.
#[allow(clippy::too_many_arguments)]
pub fn qma9_context_id(
    prev: u8,
    left: u8,
    up: u8,
    up_left: u8,
    up_right: u8,
    prev_right: u8,
    prev_down: u8,
    up2: u8,
    left2: u8,
) -> Result<u32, QmaError> {
    let mut ctx = prev as u32;
    for symbol in [
        left, up, up_left, up_right, prev_right, prev_down, up2, left2,
    ] {
        if symbol > QMA9_SENTINEL {
            return Err(QmaError::ContextSymbolOutOfRange { symbol });
        }
        ctx = ctx * 6 + symbol as u32;
    }
    Ok(ctx)
}

#[derive(Debug, Clone, Copy)]
struct Neighbours {
    prev: u8,
    left: u8,
    up: u8,
    up_left: u8,
    up_right: u8,
    prev_right: u8,
    prev_down: u8,
    up2: u8,
    left2: u8,
}

fn neighbours(
    data: &[u8],
    frame_size: usize,
    t: usize,
    y: usize,
    rows: usize,
    cols: usize,
    x: usize,
) -> Neighbours {
    let base = t * frame_size + y * cols;
    let prev_base = if t == 0 {
        0
    } else {
        (t - 1) * frame_size + y * cols
    };
    Neighbours {
        prev: if t == 0 {
            QMA9_SENTINEL
        } else {
            data[prev_base + x]
        },
        left: if x == 0 {
            QMA9_SENTINEL
        } else {
            data[base + x - 1]
        },
        up: if y == 0 {
            QMA9_SENTINEL
        } else {
            data[base - cols + x]
        },
        up_left: if y == 0 || x == 0 {
            QMA9_SENTINEL
        } else {
            data[base - cols + x - 1]
        },
        up_right: if y == 0 || x + 1 >= cols {
            QMA9_SENTINEL
        } else {
            data[base - cols + x + 1]
        },
        prev_right: if t == 0 || x + 1 >= cols {
            QMA9_SENTINEL
        } else {
            data[prev_base + x + 1]
        },
        prev_down: if t == 0 || y + 1 >= rows {
            QMA9_SENTINEL
        } else {
            data[(t - 1) * frame_size + (y + 1) * cols + x]
        },
        up2: if y < 2 {
            QMA9_SENTINEL
        } else {
            data[base - 2 * cols + x]
        },
        left2: if x < 2 {
            QMA9_SENTINEL
        } else {
            data[base + x - 2]
        },
    }
}

#[derive(Debug)]
struct ModelContext {
    prev_freq: [u32; 2],
    left_freq: [u32; 2],
    up_freq: [u32; 2],
    class_freq: [u32; 5],
}

#[derive(Debug, Default)]
struct AdaptiveModel9Binary {
    contexts: HashMap<u32, ModelContext>,
}

impl AdaptiveModel9Binary {
    fn new() -> Self {
        Self {
            contexts: HashMap::new(),
        }
    }

    fn context(&mut self, ctx_id: u32) -> &mut ModelContext {
        self.contexts.entry(ctx_id).or_insert_with(|| {
            let (prev, left, up) = decode_context_predictors(ctx_id);
            let mut up_freq = [1, 3];
            let mut left_freq = [1, 4];
            let mut prev_freq = [1, 3];
            let mut class_freq = [1, 1, 1, 1, 1];

            if up == QMA9_SENTINEL {
                up_freq = [60_000, 1];
            }
            if left == QMA9_SENTINEL || left == up {
                left_freq = [60_000, 1];
            }
            if prev == QMA9_SENTINEL || prev == up || prev == left {
                prev_freq = [60_000, 1];
            }
            for cls in 0..QMA9_CLASS_SYMBOLS {
                if cls != up && cls != left && cls != prev {
                    class_freq[cls as usize] = 3;
                }
            }

            ModelContext {
                prev_freq,
                left_freq,
                up_freq,
                class_freq,
            }
        })
    }
}

fn decode_context_predictors(ctx_id: u32) -> (u8, u8, u8) {
    debug_assert!(ctx_id < QMA9_CONTEXTS);
    let mut value = ctx_id;
    let _left2 = (value % 6) as u8;
    value /= 6;
    let _up2 = (value % 6) as u8;
    value /= 6;
    let _prev_down = (value % 6) as u8;
    value /= 6;
    let _prev_right = (value % 6) as u8;
    value /= 6;
    let _up_right = (value % 6) as u8;
    value /= 6;
    let _up_left = (value % 6) as u8;
    value /= 6;
    let up = (value % 6) as u8;
    value /= 6;
    let left = (value % 6) as u8;
    value /= 6;
    let prev = (value % 6) as u8;
    (prev, left, up)
}

fn update_adaptive(freq: &mut [u32], symbol: usize) {
    if freq.iter().sum::<u32>() >= QMA9_SCALE_TOTAL {
        for value in freq.iter_mut() {
            *value = ((*value + 1) >> 1).max(1);
        }
    }
    freq[symbol] = (freq[symbol] + QMA9_UPDATE_DELTA).min(u16::MAX as u32);
}

fn decode_symbol(decoder: &mut ArithmeticDecoder<'_>, freq: &[u32]) -> Result<usize, QmaError> {
    let total = freq.iter().sum::<u32>();
    let value = decoder.scaled(total);
    let mut cumulative = 0u32;
    for (symbol, weight) in freq.iter().copied().enumerate() {
        let next = cumulative + weight;
        if value < next {
            decoder.update(cumulative, next, total);
            return Ok(symbol);
        }
        cumulative = next;
    }
    Err(QmaError::ArithmeticSymbolOutOfRange)
}

#[derive(Debug)]
struct BitReader<'a> {
    bytes: &'a [u8],
    pos: usize,
    left: u8,
    cur: u8,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            pos: 0,
            left: 0,
            cur: 0,
        }
    }

    fn bit(&mut self) -> u32 {
        if self.left == 0 {
            self.cur = if self.pos < self.bytes.len() {
                let byte = self.bytes[self.pos];
                self.pos += 1;
                byte
            } else {
                0
            };
            self.left = 8;
        }
        let bit = (self.cur >> 7) & 1;
        self.cur <<= 1;
        self.left -= 1;
        bit as u32
    }
}

#[derive(Debug)]
struct ArithmeticDecoder<'a> {
    low: u32,
    high: u32,
    value: u32,
    reader: BitReader<'a>,
}

impl<'a> ArithmeticDecoder<'a> {
    fn new(data: &'a [u8]) -> Self {
        let mut reader = BitReader::new(data);
        let mut value = 0u32;
        for _ in 0..32 {
            value = (value << 1) | reader.bit();
        }
        Self {
            low: 0,
            high: ARITH_TOP,
            value,
            reader,
        }
    }

    fn scaled(&self, total: u32) -> u32 {
        let range = self.high as u64 - self.low as u64 + 1;
        ((((self.value as u64 - self.low as u64 + 1) * total as u64) - 1) / range) as u32
    }

    fn update(&mut self, cum_low: u32, cum_high: u32, total: u32) {
        let range = self.high as u64 - self.low as u64 + 1;
        let old_low = self.low;
        self.high = (old_low as u64 + (range * cum_high as u64) / total as u64 - 1) as u32;
        self.low = (old_low as u64 + (range * cum_low as u64) / total as u64) as u32;
        loop {
            if self.high < ARITH_HALF {
                // no-op
            } else if self.low >= ARITH_HALF {
                self.value -= ARITH_HALF;
                self.low -= ARITH_HALF;
                self.high -= ARITH_HALF;
            } else if self.low >= ARITH_FIRST_QTR && self.high < ARITH_THIRD_QTR {
                self.value -= ARITH_FIRST_QTR;
                self.low -= ARITH_FIRST_QTR;
                self.high -= ARITH_FIRST_QTR;
            } else {
                break;
            }
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
            self.value = (self.value << 1) | self.reader.bit();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PYTHON_QMA9_FIXTURE: &[u8] = &[
        0x51, 0x4d, 0x41, 0x39, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00,
        0x00, 0x11, 0x00, 0x00, 0x00, 0x03, 0x84, 0xe7, 0x13, 0xbf, 0x51, 0x70, 0xa3, 0xd4, 0x3b,
        0x40, 0xd3, 0x52, 0xfd, 0x5d, 0x1e, 0x00,
    ];
    const PYTHON_QMA9_RAW: &[u8] = &[
        0, 2, 4, 1, 1, 3, 0, 2, 2, 4, 1, 3, 1, 3, 0, 2, 2, 4, 1, 3, 3, 0, 2, 4,
    ];
    const PR85_QMA9_HEADER: &[u8] = &[
        0x51, 0x4d, 0x41, 0x39, 0x58, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x80, 0x01, 0x00,
        0x00, 0x0f, 0x6d, 0x02, 0x00,
    ];
    const PR92_COMPACT_V5_HEADER: &[u8] = &[
        0x23, 0x6d, 0x02, 0xf2, 0xde, 0x00, 0xcf, 0x05, 0x00, 0x78, 0x05, 0x00, 0xe2, 0x00, 0x00,
        0x6a, 0x00, 0x00, 0x95, 0x00, 0x00, 0x9a, 0x00, 0x00,
    ];
    const PR92_QMA9_HEADER: &[u8] = &[
        0x51, 0x4d, 0x41, 0x39, 0x58, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x80, 0x01, 0x00,
        0x00, 0x0f, 0x6d, 0x02, 0x00,
    ];

    #[test]
    fn parses_python_qma9_header() {
        let header = parse_qma9_header(PYTHON_QMA9_FIXTURE).expect("valid fixture");
        assert_eq!(header.frame_count, 2);
        assert_eq!(header.width, 3);
        assert_eq!(header.height, 4);
        assert_eq!(header.bitstream_bytes, 17);
        assert_eq!(header.packed_bytes, 37);
        assert_eq!(header.decoded_mask_bytes, 24);
    }

    #[test]
    fn decodes_python_qma9_fixture_deterministically() {
        let decoded = decode_qma9_mask(PYTHON_QMA9_FIXTURE).expect("fixture decodes");
        assert_eq!(decoded.data, PYTHON_QMA9_RAW);

        let prefix = decode_qma9_prefix_frames(PYTHON_QMA9_FIXTURE, 1).expect("prefix decodes");
        assert_eq!(prefix.data, &PYTHON_QMA9_RAW[..12]);
    }

    #[test]
    fn fails_closed_on_bad_magic_and_overrun() {
        let mut bad_magic = PYTHON_QMA9_FIXTURE.to_vec();
        bad_magic[0..4].copy_from_slice(b"NOPE");
        assert!(matches!(
            parse_qma9_header(&bad_magic),
            Err(QmaError::BadMagic { .. })
        ));

        let mut overrun = PYTHON_QMA9_FIXTURE[..20].to_vec();
        overrun[16..20].copy_from_slice(&18u32.to_le_bytes());
        assert!(matches!(
            parse_qma9_header(&overrun),
            Err(QmaError::BitstreamOverrun { .. })
        ));
    }

    #[test]
    fn parses_known_pr85_qma9_segment_header() {
        let header = parse_qma9_header_prefix(PR85_QMA9_HEADER).expect("PR85 header parses");
        assert_eq!(header.frame_count, 600);
        assert_eq!(header.width, 512);
        assert_eq!(header.height, 384);
        assert_eq!(header.bitstream_bytes, 158_991);
        assert_eq!(header.packed_bytes, 159_011);
        assert_eq!(header.decoded_mask_bytes, 117_964_800);
    }

    #[test]
    fn parses_pr92_compact_bundle_v5_micro_header_and_mask_boundary() {
        let mut payload = Vec::new();
        payload.extend_from_slice(PR92_COMPACT_V5_HEADER);
        payload.extend_from_slice(PR92_QMA9_HEADER);
        payload.resize(24 + 159_011, 0);
        payload.resize(24 + 159_011 + 57_074, 0xa1);
        payload.resize(24 + 159_011 + 57_074 + 1_487, 0xa2);
        payload.resize(24 + 159_011 + 57_074 + 1_487 + 1_400, 0xa3);
        payload.resize(24 + 159_011 + 57_074 + 1_487 + 1_400 + 226, 0xa4);
        payload.resize(24 + 159_011 + 57_074 + 1_487 + 1_400 + 226 + 106, 0xa5);
        payload.resize(
            24 + 159_011 + 57_074 + 1_487 + 1_400 + 226 + 106 + 149,
            0xa6,
        );
        payload.resize(
            24 + 159_011 + 57_074 + 1_487 + 1_400 + 226 + 106 + 149 + 154,
            0xa7,
        );
        payload.resize(
            24 + 159_011 + 57_074 + 1_487 + 1_400 + 226 + 106 + 149 + 154 + 223,
            0xa8,
        );
        payload.resize(
            24 + 159_011 + 57_074 + 1_487 + 1_400 + 226 + 106 + 149 + 154 + 223 + 273,
            0xa9,
        );
        payload.resize(235_952, 0xaa);

        let header = parse_compact_bundle_v5_micro_header(&payload).expect("PR92 header parses");
        assert_eq!(header.header_bytes, 24);
        assert_eq!(header.mask_bytes, 159_011);
        assert_eq!(header.model_bytes, 57_074);
        assert_eq!(header.pose_bytes, 1_487);
        assert_eq!(header.post_bytes, 1_400);
        assert_eq!(header.shift_bytes, 226);
        assert_eq!(header.frac_bytes, 106);
        assert_eq!(header.frac2_bytes, 149);
        assert_eq!(header.frac3_bytes, 154);
        assert_eq!(header.bias_bytes, 223);
        assert_eq!(header.region_bytes, 273);
        assert_eq!(header.randmulti_bytes, 15_825);
        assert_eq!(header.packed_bytes, 235_952);

        let mask = compact_bundle_v5_micro_mask_slice(&payload).expect("mask slice");
        assert_eq!(mask.len(), 159_011);
        let qma9 = parse_qma9_header(mask).expect("PR92 QMA9 mask header parses");
        assert_eq!(qma9.frame_count, 600);
        assert_eq!(qma9.width, 512);
        assert_eq!(qma9.height, 384);
        assert_eq!(qma9.bitstream_bytes, 158_991);
        assert_eq!(qma9.packed_bytes, mask.len());
    }

    #[test]
    fn compact_bundle_v5_micro_fails_closed_on_truncation() {
        assert!(matches!(
            parse_compact_bundle_v5_micro_header(&PR92_COMPACT_V5_HEADER[..23]),
            Err(QmaError::CompactBundleTooShort { len: 23 })
        ));
        assert!(matches!(
            parse_compact_bundle_v5_micro_header(PR92_COMPACT_V5_HEADER),
            Err(QmaError::CompactBundleTruncated { .. })
        ));
    }
}
