use std::collections::HashMap;
use std::io::Read;

const STBM1BR_MAGIC: &[u8; 8] = b"STBM1BR\0";
const N_CLASSES: usize = 5;
const N_SYM: usize = N_CLASSES - 1;

const FEAT_DIAG_TLTL: u8 = 0;
const FEAT_LEFT_LEFT: u8 = 1;
const FEAT_TOP_TOP_TOP: u8 = 2;
const FEAT_PREV_PREV_PREV: u8 = 3;
const FEAT_DIAG_TRTR: u8 = 4;
const FEAT_PREV_LEFT: u8 = 5;
const FEAT_PREV_RIGHT: u8 = 6;
const FEAT_PREV_TOP: u8 = 7;
const FEAT_PREV_BOTTOM: u8 = 8;
const FEAT_PREV2_LEFT: u8 = 9;
const FEAT_PREV2_RIGHT: u8 = 10;
const FEAT_PREV_BOTTOM_RIGHT: u8 = 11;
const FEAT_PREV_BOTTOM_LEFT: u8 = 12;
const FEAT_PREV_TOP_RIGHT: u8 = 13;
const FEAT_PREV_BOTTOM2: u8 = 14;
const FEAT_PREV_RIGHT2: u8 = 15;
const FEAT_X_BIN5: u8 = 16;
const FEAT_Y_BIN5: u8 = 17;
const FEAT_X_BIN5_SHIFT: u8 = 20;
const FEAT_PEEL_DIST42: u8 = 30;
const FEAT_PEEL_BOUND5: u8 = 31;
const FEAT_PEEL_SLOPE5: u8 = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StbmError {
    BadMagic { found: Vec<u8> },
    Truncated(&'static str),
    Unsupported(&'static str),
    Invalid(String),
    Decompress(String),
}

impl std::fmt::Display for StbmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadMagic { found } => write!(f, "bad STBM1BR magic: {found:?}"),
            Self::Truncated(what) => write!(f, "truncated {what}"),
            Self::Unsupported(what) => write!(f, "unsupported STBM1BR subformat: {what}"),
            Self::Invalid(msg) => write!(f, "invalid STBM1BR payload: {msg}"),
            Self::Decompress(msg) => write!(f, "STBM1BR decompression failed: {msg}"),
        }
    }
}

impl std::error::Error for StbmError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StbmMetadata {
    pub segment_bytes: usize,
    pub brotli_body_bytes: usize,
    pub qtbm_blob_bytes: usize,
    pub qtbm_magic: [u8; 6],
    pub n_pairs: usize,
    pub height: usize,
    pub width: usize,
    pub precision: u8,
    pub top_bins: u8,
    pub boundary_xbins: u8,
    pub shift_dy: i8,
    pub shift_dx: i8,
    pub residual_order: Option<[u8; N_SYM]>,
    pub top_payload_bytes: usize,
    pub road_payload_bytes: usize,
    pub spatial_table_bytes: usize,
    pub m5_table_bytes: usize,
    pub sparse_feature_ids: Vec<u8>,
    pub sparse_table_bytes: usize,
    pub residual_bitstream_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedStbmMask {
    pub metadata: StbmMetadata,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone)]
struct QTbBlob {
    metadata: StbmMetadata,
    top_payload: Vec<u8>,
    road_payload: Vec<u8>,
    spatial_table: Vec<u8>,
    m5_table: Vec<u8>,
    sparse_feature_ids: Vec<u8>,
    sparse_table: Vec<u8>,
    residual_bitstream: Vec<u8>,
}

#[derive(Debug, Clone)]
struct TopBand {
    bins: usize,
    bounds: Vec<u16>,
}

impl TopBand {
    fn contains(&self, frame: usize, y: usize, x: usize, width: usize) -> bool {
        let bin = (x * self.bins) / width;
        y < self.bounds[frame * self.bins + bin] as usize
    }
}

#[derive(Debug, Clone)]
struct BoundaryMask {
    mask: Vec<u8>,
    feature_bounds: Vec<u16>,
}

impl BoundaryMask {
    fn contains(&self, frame_size: usize, frame: usize, local_idx: usize) -> bool {
        self.mask[frame * frame_size + local_idx] != 0
    }

    fn feature_bound(&self, width: usize, frame: usize, x: usize) -> i32 {
        self.feature_bounds[frame * width + x] as i32
    }
}

#[derive(Debug, Clone)]
struct SparseRows {
    idx: Vec<usize>,
    rows: Vec<[u32; N_SYM]>,
}

pub fn parse_stbm1br_metadata(segment: &[u8]) -> Result<StbmMetadata, StbmError> {
    let body = strip_stbm_header(segment)?;
    let blob = brotli_decompress(body)?;
    let mut qtb = parse_qtbm_blob(&blob)?;
    qtb.metadata.segment_bytes = segment.len();
    qtb.metadata.brotli_body_bytes = body.len();
    Ok(qtb.metadata)
}

pub fn decode_stbm1br_segment(segment: &[u8]) -> Result<DecodedStbmMask, StbmError> {
    let body = strip_stbm_header(segment)?;
    let blob = brotli_decompress(body)?;
    let mut decoded = decode_qtbm_blob(&blob)?;
    decoded.metadata.segment_bytes = segment.len();
    decoded.metadata.brotli_body_bytes = body.len();
    Ok(decoded)
}

pub fn decode_stbm1br_segment_expected(
    segment: &[u8],
    expected_frames: usize,
    expected_height: usize,
    expected_width: usize,
) -> Result<DecodedStbmMask, StbmError> {
    let decoded = decode_stbm1br_segment(segment)?;
    if decoded.metadata.n_pairs != expected_frames
        || decoded.metadata.height != expected_height
        || decoded.metadata.width != expected_width
    {
        return Err(StbmError::Invalid(format!(
            "decoded shape ({}, {}, {}) != expected ({expected_frames}, {expected_height}, {expected_width})",
            decoded.metadata.n_pairs, decoded.metadata.height, decoded.metadata.width
        )));
    }
    Ok(decoded)
}

pub fn metadata_json(meta: &StbmMetadata) -> String {
    let qtbm_magic = String::from_utf8_lossy(&meta.qtbm_magic).replace('\0', "\\u0000");
    let residual_order = match meta.residual_order {
        Some(order) => format!("[{},{},{},{}]", order[0], order[1], order[2], order[3]),
        None => "null".to_string(),
    };
    let sparse_features = meta
        .sparse_feature_ids
        .iter()
        .map(u8::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!(
        concat!(
            "{{",
            "\"segment_bytes\":{},",
            "\"brotli_body_bytes\":{},",
            "\"qtbm_blob_bytes\":{},",
            "\"qtbm_magic\":\"{}\",",
            "\"n_pairs\":{},",
            "\"height\":{},",
            "\"width\":{},",
            "\"precision\":{},",
            "\"top_bins\":{},",
            "\"boundary_xbins\":{},",
            "\"shift_dy\":{},",
            "\"shift_dx\":{},",
            "\"residual_order\":{},",
            "\"top_payload_bytes\":{},",
            "\"road_payload_bytes\":{},",
            "\"spatial_table_bytes\":{},",
            "\"m5_table_bytes\":{},",
            "\"sparse_feature_ids\":[{}],",
            "\"sparse_table_bytes\":{},",
            "\"residual_bitstream_bytes\":{}",
            "}}"
        ),
        meta.segment_bytes,
        meta.brotli_body_bytes,
        meta.qtbm_blob_bytes,
        qtbm_magic,
        meta.n_pairs,
        meta.height,
        meta.width,
        meta.precision,
        meta.top_bins,
        meta.boundary_xbins,
        meta.shift_dy,
        meta.shift_dx,
        residual_order,
        meta.top_payload_bytes,
        meta.road_payload_bytes,
        meta.spatial_table_bytes,
        meta.m5_table_bytes,
        sparse_features,
        meta.sparse_table_bytes,
        meta.residual_bitstream_bytes,
    )
}

fn strip_stbm_header(segment: &[u8]) -> Result<&[u8], StbmError> {
    if segment.len() < STBM1BR_MAGIC.len() || &segment[..STBM1BR_MAGIC.len()] != STBM1BR_MAGIC {
        return Err(StbmError::BadMagic {
            found: segment[..segment.len().min(STBM1BR_MAGIC.len())].to_vec(),
        });
    }
    let body = &segment[STBM1BR_MAGIC.len()..];
    if body.is_empty() {
        return Err(StbmError::Invalid("empty Brotli body".to_string()));
    }
    Ok(body)
}

fn brotli_decompress(data: &[u8]) -> Result<Vec<u8>, StbmError> {
    let mut out = Vec::new();
    let mut decoder = brotli::Decompressor::new(data, 4096);
    decoder
        .read_to_end(&mut out)
        .map_err(|err| StbmError::Decompress(format!("brotli: {err}")))?;
    Ok(out)
}

fn bzip2_decompress(data: &[u8]) -> Result<Vec<u8>, StbmError> {
    let mut out = Vec::new();
    let mut decoder = bzip2::read::BzDecoder::new(data);
    decoder
        .read_to_end(&mut out)
        .map_err(|err| StbmError::Decompress(format!("bzip2: {err}")))?;
    Ok(out)
}

fn deflate_decompress(data: &[u8]) -> Result<Vec<u8>, StbmError> {
    let mut out = Vec::new();
    let mut decoder = flate2::read::DeflateDecoder::new(data);
    decoder
        .read_to_end(&mut out)
        .map_err(|err| StbmError::Decompress(format!("raw-deflate: {err}")))?;
    Ok(out)
}

fn read_u16_le(buf: &[u8], pos: &mut usize, what: &'static str) -> Result<u16, StbmError> {
    if *pos + 2 > buf.len() {
        return Err(StbmError::Truncated(what));
    }
    let value = u16::from_le_bytes([buf[*pos], buf[*pos + 1]]);
    *pos += 2;
    Ok(value)
}

fn read_u32_le(buf: &[u8], pos: &mut usize, what: &'static str) -> Result<u32, StbmError> {
    if *pos + 4 > buf.len() {
        return Err(StbmError::Truncated(what));
    }
    let value = u32::from_le_bytes([buf[*pos], buf[*pos + 1], buf[*pos + 2], buf[*pos + 3]]);
    *pos += 4;
    Ok(value)
}

fn read_i8(buf: &[u8], pos: &mut usize, what: &'static str) -> Result<i8, StbmError> {
    if *pos >= buf.len() {
        return Err(StbmError::Truncated(what));
    }
    let value = buf[*pos] as i8;
    *pos += 1;
    Ok(value)
}

fn read_u8(buf: &[u8], pos: &mut usize, what: &'static str) -> Result<u8, StbmError> {
    if *pos >= buf.len() {
        return Err(StbmError::Truncated(what));
    }
    let value = buf[*pos];
    *pos += 1;
    Ok(value)
}

fn take<'a>(
    buf: &'a [u8],
    pos: &mut usize,
    len: usize,
    what: &'static str,
) -> Result<&'a [u8], StbmError> {
    if *pos + len > buf.len() {
        return Err(StbmError::Truncated(what));
    }
    let out = &buf[*pos..*pos + len];
    *pos += len;
    Ok(out)
}

fn parse_qtbm_blob(blob: &[u8]) -> Result<QTbBlob, StbmError> {
    let mut pos = 0usize;
    let magic = if blob.starts_with(b"QTBM5\0") {
        *b"QTBM5\0"
    } else {
        return Err(StbmError::Unsupported("only QTBM5 is implemented in Rust"));
    };
    pos += magic.len();
    let n_pairs = read_u16_le(blob, &mut pos, "QTBM dimensions")? as usize;
    let height = read_u16_le(blob, &mut pos, "QTBM dimensions")? as usize;
    let width = read_u16_le(blob, &mut pos, "QTBM dimensions")? as usize;
    let precision = read_u8(blob, &mut pos, "QTBM precision")?;
    let top_bins = read_u8(blob, &mut pos, "QTBM top bins")?;
    let boundary_xbins = read_u8(blob, &mut pos, "QTBM boundary bins")?;
    let shift_dy = read_i8(blob, &mut pos, "QTBM shift dy")?;
    let shift_dx = read_i8(blob, &mut pos, "QTBM shift dx")?;
    if n_pairs == 0 || height == 0 || width == 0 {
        return Err(StbmError::Invalid(
            "QTBM dimensions must be positive".to_string(),
        ));
    }
    if precision == 0 || precision > 24 {
        return Err(StbmError::Invalid(format!(
            "unsupported precision {precision}"
        )));
    }
    let residual_raw = take(blob, &mut pos, N_SYM, "QTBM residual order")?;
    let residual_order = [
        residual_raw[0],
        residual_raw[1],
        residual_raw[2],
        residual_raw[3],
    ];
    let mut sorted = residual_order;
    sorted.sort_unstable();
    if sorted != [0, 1, 2, 3] {
        return Err(StbmError::Invalid(format!(
            "invalid residual order: {residual_order:?}"
        )));
    }
    let top_len = read_u32_le(blob, &mut pos, "QTBM top payload length")? as usize;
    let road_len = read_u32_le(blob, &mut pos, "QTBM road payload length")? as usize;
    let top_payload = take(blob, &mut pos, top_len, "QTBM top payload")?.to_vec();
    let road_payload = take(blob, &mut pos, road_len, "QTBM road payload")?.to_vec();
    let spatial_len = read_u16_le(blob, &mut pos, "QTBM spatial table length")? as usize;
    let m5_len = read_u16_le(blob, &mut pos, "QTBM M5 table length")? as usize;
    let spatial_table = take(blob, &mut pos, spatial_len, "QTBM spatial table")?.to_vec();
    let m5_table = take(blob, &mut pos, m5_len, "QTBM M5 table")?.to_vec();
    let n_feats = read_u8(blob, &mut pos, "QTBM sparse feature count")? as usize;
    let sparse_feature_ids = take(blob, &mut pos, n_feats, "QTBM sparse feature ids")?.to_vec();
    let _threshold_q8 = read_u16_le(blob, &mut pos, "QTBM sparse threshold")?;
    let sparse_len = read_u32_le(blob, &mut pos, "QTBM sparse table length")? as usize;
    let sparse_table = take(blob, &mut pos, sparse_len, "QTBM sparse table")?.to_vec();
    let bitstream_len = read_u32_le(blob, &mut pos, "QTBM residual bitstream length")? as usize;
    let residual_bitstream =
        take(blob, &mut pos, bitstream_len, "QTBM residual bitstream")?.to_vec();
    if pos != blob.len() {
        return Err(StbmError::Invalid(
            "QTBM blob has trailing bytes".to_string(),
        ));
    }
    let metadata = StbmMetadata {
        segment_bytes: 0,
        brotli_body_bytes: 0,
        qtbm_blob_bytes: blob.len(),
        qtbm_magic: magic,
        n_pairs,
        height,
        width,
        precision,
        top_bins,
        boundary_xbins,
        shift_dy,
        shift_dx,
        residual_order: Some(residual_order),
        top_payload_bytes: top_payload.len(),
        road_payload_bytes: road_payload.len(),
        spatial_table_bytes: spatial_table.len(),
        m5_table_bytes: m5_table.len(),
        sparse_feature_ids: sparse_feature_ids.clone(),
        sparse_table_bytes: sparse_table.len(),
        residual_bitstream_bytes: residual_bitstream.len(),
    };
    Ok(QTbBlob {
        metadata,
        top_payload,
        road_payload,
        spatial_table,
        m5_table,
        sparse_feature_ids,
        sparse_table,
        residual_bitstream,
    })
}

fn decode_qtbm_blob(blob: &[u8]) -> Result<DecodedStbmMask, StbmError> {
    let qtb = parse_qtbm_blob(blob)?;
    let n_pairs = qtb.metadata.n_pairs;
    let height = qtb.metadata.height;
    let width = qtb.metadata.width;
    let frame_size = height
        .checked_mul(width)
        .ok_or_else(|| StbmError::Invalid("frame size overflow".to_string()))?;
    let total_size = n_pairs
        .checked_mul(frame_size)
        .ok_or_else(|| StbmError::Invalid("decoded size overflow".to_string()))?;
    let total = 1u32
        .checked_shl(qtb.metadata.precision as u32)
        .ok_or_else(|| StbmError::Invalid("precision overflow".to_string()))?;
    if total <= N_SYM as u32 - 1 {
        return Err(StbmError::Invalid(
            "precision total is too small".to_string(),
        ));
    }
    let inv = qtb.metadata.residual_order.unwrap_or([0, 1, 2, 3]);
    let top = decode_topband_payload(&qtb.top_payload, n_pairs, height, width)?;
    let road = decode_boundary_mask_payload(&qtb.road_payload, n_pairs, height, width)?;

    let spatial = unpack_sparse_big_plain_colsfirst(&qtb.spatial_table, qtb.metadata.precision)?;
    let m5 = unpack_sparse_big_plain(&qtb.m5_table, qtb.metadata.precision)?;
    let fired = unpack_sparse_big_plain_colsfirst(&qtb.sparse_table, qtb.metadata.precision)?;
    let spatial_cdf = build_cdf_table(5usize.pow(4), &spatial, total, "spatial")?;
    let m5_cdf = build_cdf_table(5usize.pow(5), &m5, total, "m5")?;
    let fired_cdf = build_cdf_rows(&fired.rows, total, "fired")?;
    let fired_slots = fired
        .idx
        .iter()
        .copied()
        .enumerate()
        .map(|(slot, idx)| (idx, slot))
        .collect::<HashMap<usize, usize>>();

    let mut out = vec![0u8; total_size];
    let mut decoder = ArithmeticDecoder::new(&qtb.residual_bitstream);
    for frame in 0..n_pairs {
        decode_frame_topband(
            &mut decoder,
            &mut out,
            frame,
            &top,
            &road,
            &spatial_cdf,
            &m5_cdf,
            &fired_cdf,
            &fired_slots,
            &qtb.sparse_feature_ids,
            total,
            height,
            width,
            qtb.metadata.shift_dy,
            qtb.metadata.shift_dx,
            &inv,
        )?;
    }
    Ok(DecodedStbmMask {
        metadata: qtb.metadata,
        data: out,
    })
}

fn decode_topband_payload(
    payload: &[u8],
    n_pairs: usize,
    _height: usize,
    width: usize,
) -> Result<TopBand, StbmError> {
    if !payload.starts_with(b"QTBZ\0") {
        return Err(StbmError::Unsupported(
            "only QTBZ top-band payloads are implemented",
        ));
    }
    let mut pos = 5usize;
    let n2 = read_u16_le(payload, &mut pos, "QTBZ dimensions")? as usize;
    let w2 = read_u16_le(payload, &mut pos, "QTBZ dimensions")? as usize;
    let bins = read_u16_le(payload, &mut pos, "QTBZ bins")? as usize;
    let bounds_len = read_u32_le(payload, &mut pos, "QTBZ bounds length")? as usize;
    if n2 != n_pairs || w2 != width || bins == 0 {
        return Err(StbmError::Invalid(format!(
            "QTBZ dimensions ({n2}, {w2}, {bins}) do not match ({n_pairs}, {width}, >0)"
        )));
    }
    let compressed = take(payload, &mut pos, bounds_len, "QTBZ bounds stream")?;
    if pos != payload.len() {
        return Err(StbmError::Invalid(
            "QTBZ payload has trailing bytes".to_string(),
        ));
    }
    let raw = bzip2_decompress(compressed)?;
    let expected = n_pairs
        .checked_mul(bins)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| StbmError::Invalid("QTBZ bounds size overflow".to_string()))?;
    if raw.len() != expected {
        return Err(StbmError::Invalid(format!(
            "QTBZ bounds bytes {} != expected {expected}",
            raw.len()
        )));
    }
    let mut bounds = Vec::with_capacity(n_pairs * bins);
    for chunk in raw.chunks_exact(2) {
        bounds.push(u16::from_le_bytes([chunk[0], chunk[1]]));
    }
    Ok(TopBand { bins, bounds })
}

fn decode_boundary_mask_payload(
    payload: &[u8],
    n_pairs: usize,
    height: usize,
    width: usize,
) -> Result<BoundaryMask, StbmError> {
    if !payload.starts_with(b"QBD2\0") {
        return Err(StbmError::Unsupported(
            "only QBD2 boundary payloads are implemented",
        ));
    }
    let mut pos = 5usize;
    let bins = read_u8(payload, &mut pos, "QBD2 bins")? as usize;
    let dx_nsym = read_u8(payload, &mut pos, "QBD2 dx symbol count")? as usize;
    let dx_offset = read_u8(payload, &mut pos, "QBD2 dx offset")? as i32;
    let first_len = read_u32_le(payload, &mut pos, "QBD2 first length")? as usize;
    let dx_len = read_u32_le(payload, &mut pos, "QBD2 dx length")? as usize;
    let err_len = read_u32_le(payload, &mut pos, "QBD2 exception length")? as usize;
    let err_count = read_u32_le(payload, &mut pos, "QBD2 exception count")? as usize;
    if bins == 0 || dx_nsym == 0 {
        return Err(StbmError::Invalid(
            "QBD2 bins and symbol count must be positive".to_string(),
        ));
    }
    let first_raw = bzip2_decompress(take(payload, &mut pos, first_len, "QBD2 first stream")?)?;
    if first_raw.len() != n_pairs * 2 {
        return Err(StbmError::Invalid(format!(
            "QBD2 first stream bytes {} != expected {}",
            first_raw.len(),
            n_pairs * 2
        )));
    }
    let mut first = Vec::with_capacity(n_pairs);
    for chunk in first_raw.chunks_exact(2) {
        first.push(u16::from_le_bytes([chunk[0], chunk[1]]) as i32);
    }
    let freq_bytes = bins
        .checked_mul(dx_nsym)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| StbmError::Invalid("QBD2 frequency table overflow".to_string()))?;
    let freq_raw = take(payload, &mut pos, freq_bytes, "QBD2 frequency table")?;
    let mut cdfs = vec![vec![0u32; dx_nsym + 1]; bins];
    for b in 0..bins {
        let mut acc = 0u32;
        for s in 0..dx_nsym {
            let o = (b * dx_nsym + s) * 2;
            acc = acc
                .checked_add(u16::from_le_bytes([freq_raw[o], freq_raw[o + 1]]) as u32)
                .ok_or_else(|| StbmError::Invalid("QBD2 frequency sum overflow".to_string()))?;
            cdfs[b][s + 1] = acc;
        }
    }
    let total = cdfs[0][dx_nsym];
    if total == 0 {
        return Err(StbmError::Invalid(
            "QBD2 frequency total is zero".to_string(),
        ));
    }
    if cdfs.iter().any(|row| row[dx_nsym] != total) {
        return Err(StbmError::Invalid(
            "QBD2 frequency totals differ by bin".to_string(),
        ));
    }
    let bitstream = take(payload, &mut pos, dx_len, "QBD2 arithmetic dx stream")?;
    let err_compressed = take(payload, &mut pos, err_len, "QBD2 exception stream")?;
    if pos != payload.len() {
        return Err(StbmError::Invalid(
            "QBD2 payload has trailing bytes".to_string(),
        ));
    }
    let err_raw = brotli_decompress(err_compressed)?;
    let err_idx = leb128_cumsum_indices(&err_raw, 0, err_count)?;
    let frame_size = height
        .checked_mul(width)
        .ok_or_else(|| StbmError::Invalid("QBD2 frame size overflow".to_string()))?;
    let total_size = n_pairs
        .checked_mul(frame_size)
        .ok_or_else(|| StbmError::Invalid("QBD2 output size overflow".to_string()))?;
    if let Some(&last) = err_idx.last() {
        if last >= total_size {
            return Err(StbmError::Invalid(format!(
                "QBD2 exception index {last} out of range {total_size}"
            )));
        }
    }

    let mut decoder = ArithmeticDecoder::new(bitstream);
    let mut bounds = vec![0i32; n_pairs * width];
    for frame in 0..n_pairs {
        let mut x_bound = first[frame];
        bounds[frame * width] = x_bound;
        for x in 0..width - 1 {
            let row = &cdfs[(x * bins) / (width - 1)];
            let target = decoder.decode_target(total);
            let sym = cdf_symbol(row, target, dx_nsym)?;
            let dx = sym as i32 - dx_offset;
            x_bound += dx;
            bounds[frame * width + x + 1] = x_bound;
            decoder.advance(row[sym], row[sym + 1], total);
        }
    }

    let mut mask = vec![0u8; total_size];
    for frame in 0..n_pairs {
        let frame_base = frame * frame_size;
        for y in 0..height {
            let row_base = frame_base + y * width;
            for x in 0..width {
                let bound = bounds[frame * width + x];
                mask[row_base + x] = u8::from(y as i32 >= bound);
            }
        }
    }
    for idx in err_idx {
        mask[idx] ^= 1;
    }

    let mut feature_bounds = vec![height as u16; n_pairs * width];
    for frame in 0..n_pairs {
        let frame_base = frame * frame_size;
        for x in 0..width {
            let mut bound = height;
            if mask[frame_base + (height - 1) * width + x] != 0 {
                let mut y = height - 1;
                loop {
                    if mask[frame_base + y * width + x] == 0 {
                        bound = y + 1;
                        break;
                    }
                    if y == 0 {
                        bound = 0;
                        break;
                    }
                    y -= 1;
                }
            }
            feature_bounds[frame * width + x] = bound as u16;
        }
    }
    Ok(BoundaryMask {
        mask,
        feature_bounds,
    })
}

fn unpack_sparse_big_plain(raw: &[u8], precision: u8) -> Result<SparseRows, StbmError> {
    let mut pos = 0usize;
    let n_fired = read_u32_le(raw, &mut pos, "sparse fired count")? as usize;
    if n_fired == 0 {
        if pos != raw.len() {
            return Err(StbmError::Invalid(
                "empty sparse table has trailing bytes".to_string(),
            ));
        }
        return Ok(SparseRows {
            idx: Vec::new(),
            rows: Vec::new(),
        });
    }
    let deltas = leb128_decode(raw, &mut pos, n_fired)?;
    let table_bytes = n_fired
        .checked_mul(N_SYM - 1)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| StbmError::Invalid("sparse row table size overflow".to_string()))?;
    if pos + table_bytes != raw.len() {
        return Err(StbmError::Invalid(
            "sparse row table length mismatch".to_string(),
        ));
    }
    let partial = &raw[pos..pos + table_bytes];
    let rows = rows_from_partials_row_major(partial, n_fired, precision)?;
    let idx = cumsum_deltas_to_indices(&deltas)?;
    Ok(SparseRows { idx, rows })
}

fn unpack_sparse_big_plain_colsfirst(raw: &[u8], precision: u8) -> Result<SparseRows, StbmError> {
    let mut pos = 0usize;
    let n_fired = read_u32_le(raw, &mut pos, "cols-first sparse fired count")? as usize;
    if n_fired == 0 {
        if pos != raw.len() {
            return Err(StbmError::Invalid(
                "empty cols-first sparse table has trailing bytes".to_string(),
            ));
        }
        return Ok(SparseRows {
            idx: Vec::new(),
            rows: Vec::new(),
        });
    }
    let table_bytes = n_fired
        .checked_mul(N_SYM - 1)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| StbmError::Invalid("cols-first sparse table size overflow".to_string()))?;
    let partial = take(
        raw,
        &mut pos,
        table_bytes,
        "cols-first sparse frequency table",
    )?;
    let deltas = leb128_decode(raw, &mut pos, n_fired)?;
    if pos != raw.len() {
        return Err(StbmError::Invalid(
            "cols-first sparse table has trailing bytes".to_string(),
        ));
    }
    let rows = rows_from_partials_cols_first(partial, n_fired, precision)?;
    let idx = cumsum_deltas_to_indices(&deltas)?;
    Ok(SparseRows { idx, rows })
}

#[allow(dead_code)]
fn unpack_sparse_big(compressed: &[u8], precision: u8) -> Result<SparseRows, StbmError> {
    let raw = deflate_decompress(compressed)?;
    unpack_sparse_big_plain(&raw, precision)
}

fn rows_from_partials_row_major(
    partial: &[u8],
    n_fired: usize,
    precision: u8,
) -> Result<Vec<[u32; N_SYM]>, StbmError> {
    let total = 1u32
        .checked_shl(precision as u32)
        .ok_or_else(|| StbmError::Invalid("precision overflow".to_string()))?;
    let mut rows = vec![[0u32; N_SYM]; n_fired];
    for (i, row) in rows.iter_mut().enumerate() {
        let mut sum = 0u32;
        for (sym, cell) in row.iter_mut().enumerate().take(N_SYM - 1) {
            let o = (i * (N_SYM - 1) + sym) * 2;
            let value = u16::from_le_bytes([partial[o], partial[o + 1]]) as u32;
            *cell = value;
            sum = sum
                .checked_add(value)
                .ok_or_else(|| StbmError::Invalid("sparse row sum overflow".to_string()))?;
        }
        if sum > total {
            return Err(StbmError::Invalid(
                "sparse row exceeds precision total".to_string(),
            ));
        }
        row[N_SYM - 1] = total - sum;
    }
    Ok(rows)
}

fn rows_from_partials_cols_first(
    partial: &[u8],
    n_fired: usize,
    precision: u8,
) -> Result<Vec<[u32; N_SYM]>, StbmError> {
    let total = 1u32
        .checked_shl(precision as u32)
        .ok_or_else(|| StbmError::Invalid("precision overflow".to_string()))?;
    let mut rows = vec![[0u32; N_SYM]; n_fired];
    for sym in 0..N_SYM - 1 {
        for (i, row) in rows.iter_mut().enumerate() {
            let o = (sym * n_fired + i) * 2;
            row[sym] = u16::from_le_bytes([partial[o], partial[o + 1]]) as u32;
        }
    }
    for row in &mut rows {
        let sum: u32 = row[..N_SYM - 1].iter().sum();
        if sum > total {
            return Err(StbmError::Invalid(
                "cols-first sparse row exceeds precision total".to_string(),
            ));
        }
        row[N_SYM - 1] = total - sum;
    }
    Ok(rows)
}

fn build_cdf_table(
    row_count: usize,
    sparse: &SparseRows,
    total: u32,
    label: &'static str,
) -> Result<Vec<[u32; N_SYM + 1]>, StbmError> {
    if sparse.idx.len() != sparse.rows.len() {
        return Err(StbmError::Invalid(format!(
            "{label} sparse row/index count mismatch"
        )));
    }
    let mut cdf = vec![[0u32, 1, 2, 3, total]; row_count];
    for (idx, row) in sparse.idx.iter().copied().zip(sparse.rows.iter()) {
        if idx >= row_count {
            return Err(StbmError::Invalid(format!(
                "{label} sparse index {idx} out of range {row_count}"
            )));
        }
        let mut acc = 0u32;
        let mut dst = [0u32; N_SYM + 1];
        for (sym, freq) in row.iter().enumerate() {
            acc = acc
                .checked_add(*freq)
                .ok_or_else(|| StbmError::Invalid(format!("{label} CDF overflow")))?;
            dst[sym + 1] = acc;
        }
        if acc != total {
            return Err(StbmError::Invalid(format!(
                "{label} sparse row total {acc} != precision total {total}"
            )));
        }
        cdf[idx] = dst;
    }
    Ok(cdf)
}

fn build_cdf_rows(
    rows: &[[u32; N_SYM]],
    total: u32,
    label: &'static str,
) -> Result<Vec<[u32; N_SYM + 1]>, StbmError> {
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let mut acc = 0u32;
        let mut dst = [0u32; N_SYM + 1];
        for (sym, freq) in row.iter().enumerate() {
            acc = acc
                .checked_add(*freq)
                .ok_or_else(|| StbmError::Invalid(format!("{label} CDF overflow")))?;
            dst[sym + 1] = acc;
        }
        if acc != total {
            return Err(StbmError::Invalid(format!(
                "{label} row total {acc} != precision total {total}"
            )));
        }
        out.push(dst);
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn decode_frame_topband(
    decoder: &mut ArithmeticDecoder<'_>,
    out: &mut [u8],
    frame: usize,
    top: &TopBand,
    road: &BoundaryMask,
    spatial_cdf: &[[u32; N_SYM + 1]],
    m5_cdf: &[[u32; N_SYM + 1]],
    fired_cdf: &[[u32; N_SYM + 1]],
    fired_slots: &HashMap<usize, usize>,
    feat_ids: &[u8],
    total: u32,
    height: usize,
    width: usize,
    shift_dy: i8,
    shift_dx: i8,
    inv: &[u8; N_SYM],
) -> Result<(), StbmError> {
    let frame_size = height * width;
    let frame_base = frame * frame_size;
    for y in 0..height {
        for x in 0..width {
            let local = y * width + x;
            let dst = frame_base + local;
            if top.contains(frame, y, x, width) {
                out[dst] = 2;
                continue;
            }
            if road.contains(frame_size, frame, local) {
                out[dst] = 4;
                continue;
            }
            let top_v = if y > 0 {
                out[frame_base + (y - 1) * width + x]
            } else {
                0
            };
            let left_v = if x > 0 { out[dst - 1] } else { 0 };
            let tl_v = if y > 0 && x > 0 {
                out[frame_base + (y - 1) * width + x - 1]
            } else {
                0
            };
            let tr_v = if y > 0 && x + 1 < width {
                out[frame_base + (y - 1) * width + x + 1]
            } else {
                0
            };
            let cdf = if frame == 0 {
                let ctx = (((top_v as usize * 5 + left_v as usize) * 5 + tl_v as usize) * 5)
                    + tr_v as usize;
                &spatial_cdf[ctx]
            } else {
                let prev_base = (frame - 1) * frame_size;
                let sy = y as isize + shift_dy as isize;
                let sx = x as isize + shift_dx as isize;
                let (prev_v, pp_v) =
                    if sy >= 0 && sx >= 0 && sy < height as isize && sx < width as isize {
                        let shifted = sy as usize * width + sx as usize;
                        let prev = out[prev_base + shifted];
                        let pp = if frame >= 2 {
                            out[(frame - 2) * frame_size + shifted]
                        } else {
                            0
                        };
                        (prev, pp)
                    } else {
                        (0, 0)
                    };
                let tt_v = if y > 1 {
                    out[frame_base + (y - 2) * width + x]
                } else {
                    0
                };
                let m5_ctx = m5_ctx(top_v, left_v, tl_v, tr_v, prev_v);
                let mut cdf = &m5_cdf[m5_ctx];
                if frame >= 2 && !fired_cdf.is_empty() {
                    let mut m12_ctx = ((m5_ctx * 5 + pp_v as usize) * 5) + tt_v as usize;
                    for &fid in feat_ids {
                        let fv =
                            feature_value(out, road, frame, y, x, fid, height, width, frame_size)?;
                        m12_ctx = m12_ctx
                            .checked_mul(5)
                            .and_then(|v| v.checked_add(fv as usize))
                            .ok_or_else(|| {
                                StbmError::Invalid("sparse feature context overflow".to_string())
                            })?;
                    }
                    if let Some(&slot) = fired_slots.get(&m12_ctx) {
                        cdf = &fired_cdf[slot];
                    }
                }
                cdf
            };
            let target = decoder.decode_target(total);
            let sym = cdf_symbol(cdf, target, N_SYM)?;
            out[dst] = inv[sym];
            decoder.advance(cdf[sym], cdf[sym + 1], total);
        }
    }
    Ok(())
}

fn m5_ctx(top_v: u8, left_v: u8, tl_v: u8, tr_v: u8, prev_v: u8) -> usize {
    ((((top_v as usize * 5 + left_v as usize) * 5 + tl_v as usize) * 5 + tr_v as usize) * 5)
        + prev_v as usize
}

#[allow(clippy::too_many_arguments)]
fn feature_value(
    out: &[u8],
    road: &BoundaryMask,
    frame: usize,
    y: usize,
    x: usize,
    fid: u8,
    height: usize,
    width: usize,
    frame_size: usize,
) -> Result<u8, StbmError> {
    let frame_base = frame * frame_size;
    let prev_base = (frame - 1) * frame_size;
    let pp_base = (frame - 2) * frame_size;
    let ppp_base = if frame >= 3 {
        Some((frame - 3) * frame_size)
    } else {
        None
    };
    let at_cur = |yy: usize, xx: usize| out[frame_base + yy * width + xx];
    let at_prev = |yy: usize, xx: usize| out[prev_base + yy * width + xx];
    let at_pp = |yy: usize, xx: usize| out[pp_base + yy * width + xx];
    let value = match fid {
        FEAT_DIAG_TLTL => {
            if y >= 2 && x >= 2 {
                at_cur(y - 2, x - 2)
            } else {
                0
            }
        }
        FEAT_LEFT_LEFT => {
            if x >= 2 {
                at_cur(y, x - 2)
            } else {
                0
            }
        }
        FEAT_TOP_TOP_TOP => {
            if y >= 3 {
                at_cur(y - 3, x)
            } else {
                0
            }
        }
        FEAT_PREV_PREV_PREV => ppp_base.map_or(0, |base| out[base + y * width + x]),
        FEAT_DIAG_TRTR => {
            if y >= 2 && x + 2 < width {
                at_cur(y - 2, x + 2)
            } else {
                0
            }
        }
        FEAT_PREV_LEFT => {
            if x >= 1 {
                at_prev(y, x - 1)
            } else {
                0
            }
        }
        FEAT_PREV_RIGHT => {
            if x + 1 < width {
                at_prev(y, x + 1)
            } else {
                0
            }
        }
        FEAT_PREV_TOP => {
            if y >= 1 {
                at_prev(y - 1, x)
            } else {
                0
            }
        }
        FEAT_PREV_BOTTOM => {
            if y + 1 < height {
                at_prev(y + 1, x)
            } else {
                0
            }
        }
        FEAT_PREV2_LEFT => {
            if x >= 1 {
                at_pp(y, x - 1)
            } else {
                0
            }
        }
        FEAT_PREV2_RIGHT => {
            if x + 1 < width {
                at_pp(y, x + 1)
            } else {
                0
            }
        }
        FEAT_PREV_BOTTOM_RIGHT => {
            if y + 1 < height && x + 1 < width {
                at_prev(y + 1, x + 1)
            } else {
                0
            }
        }
        FEAT_PREV_BOTTOM_LEFT => {
            if y + 1 < height && x >= 1 {
                at_prev(y + 1, x - 1)
            } else {
                0
            }
        }
        FEAT_PREV_TOP_RIGHT => {
            if y >= 1 && x + 1 < width {
                at_prev(y - 1, x + 1)
            } else {
                0
            }
        }
        FEAT_PREV_BOTTOM2 => {
            if y + 2 < height {
                at_prev(y + 2, x)
            } else {
                0
            }
        }
        FEAT_PREV_RIGHT2 => {
            if x + 2 < width {
                at_prev(y, x + 2)
            } else {
                0
            }
        }
        FEAT_X_BIN5 => ((x * 5) / width) as u8,
        FEAT_Y_BIN5 => ((y * 5) / height) as u8,
        FEAT_X_BIN5_SHIFT => (((x + width / 10) * 5) / width).min(4) as u8,
        FEAT_PEEL_DIST42 => {
            let dist = road.feature_bound(width, frame, x) - y as i32;
            if dist <= 0 {
                0
            } else {
                (((dist - 1) / 42) + 1).min(4) as u8
            }
        }
        FEAT_PEEL_BOUND5 => {
            ((road.feature_bound(width, frame, x) as usize * 5) / height).min(4) as u8
        }
        FEAT_PEEL_SLOPE5 => {
            let bound = road.feature_bound(width, frame, x);
            let prev_bound = if x >= 1 {
                road.feature_bound(width, frame, x - 1)
            } else {
                bound
            };
            ((bound - prev_bound).min(2) + 2).max(0) as u8
        }
        _ => {
            return Err(StbmError::Unsupported("unknown sparse feature id"));
        }
    };
    Ok(value)
}

fn cdf_symbol(cdf: &[u32], target: u32, n_sym: usize) -> Result<usize, StbmError> {
    let mut sym = 0usize;
    while sym + 1 < cdf.len() && cdf[sym + 1] <= target {
        sym += 1;
        if sym >= n_sym {
            return Err(StbmError::Invalid(
                "arithmetic decoded symbol out of range".to_string(),
            ));
        }
    }
    Ok(sym)
}

fn leb128_decode(buf: &[u8], pos: &mut usize, count: usize) -> Result<Vec<u64>, StbmError> {
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let mut result = 0u64;
        let mut shift = 0u32;
        loop {
            if *pos >= buf.len() {
                return Err(StbmError::Truncated("LEB128 delta stream"));
            }
            let byte = buf[*pos];
            *pos += 1;
            result |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift > 63 {
                return Err(StbmError::Invalid("overlong LEB128 delta".to_string()));
            }
        }
        out.push(result);
    }
    Ok(out)
}

fn leb128_cumsum_indices(buf: &[u8], pos: usize, count: usize) -> Result<Vec<usize>, StbmError> {
    let mut cursor = pos;
    let deltas = leb128_decode(buf, &mut cursor, count)?;
    if cursor != buf.len() {
        return Err(StbmError::Invalid("trailing LEB128 bytes".to_string()));
    }
    cumsum_deltas_to_indices(&deltas)
}

fn cumsum_deltas_to_indices(deltas: &[u64]) -> Result<Vec<usize>, StbmError> {
    let mut idx = Vec::with_capacity(deltas.len());
    let mut acc = 0u64;
    for &delta in deltas {
        acc = acc
            .checked_add(delta)
            .ok_or_else(|| StbmError::Invalid("LEB128 cumulative index overflow".to_string()))?;
        if acc == 0 {
            return Err(StbmError::Invalid(
                "LEB128 cumulative index underflow".to_string(),
            ));
        }
        idx.push((acc - 1) as usize);
    }
    Ok(idx)
}

#[derive(Debug, Clone)]
struct ArithmeticDecoder<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
    low: u32,
    high: u32,
    code: u32,
}

impl<'a> ArithmeticDecoder<'a> {
    const TOP: u32 = 0xffff_ffff;
    const HALF: u32 = 0x8000_0000;
    const QUARTER: u32 = 0x4000_0000;
    const THREE_QUARTER: u32 = 0xc000_0000;

    fn new(data: &'a [u8]) -> Self {
        let mut dec = Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
            low: 0,
            high: Self::TOP,
            code: 0,
        };
        for _ in 0..32 {
            dec.code = (dec.code << 1) | dec.read_bit();
        }
        dec
    }

    fn read_bit(&mut self) -> u32 {
        if self.byte_pos >= self.data.len() {
            return 0;
        }
        let byte = self.data[self.byte_pos];
        let bit = ((byte >> (7 - self.bit_pos)) & 1) as u32;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        bit
    }

    fn decode_target(&self, total: u32) -> u32 {
        let range = self.high as u64 - self.low as u64 + 1;
        (((self.code as u64 - self.low as u64 + 1) * total as u64 - 1) / range) as u32
    }

    fn advance(&mut self, cum_low: u32, cum_high: u32, total: u32) {
        let range = self.high as u64 - self.low as u64 + 1;
        let low = self.low as u64;
        self.high = (low + (range * cum_high as u64) / total as u64 - 1) as u32;
        self.low = (low + (range * cum_low as u64) / total as u64) as u32;
        loop {
            if self.high < Self::HALF {
            } else if self.low >= Self::HALF {
                self.low -= Self::HALF;
                self.high -= Self::HALF;
                self.code -= Self::HALF;
            } else if self.low >= Self::QUARTER && self.high < Self::THREE_QUARTER {
                self.low -= Self::QUARTER;
                self.high -= Self::QUARTER;
                self.code -= Self::QUARTER;
            } else {
                break;
            }
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
            self.code = (self.code << 1) | self.read_bit();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_magic() {
        let err = parse_stbm1br_metadata(b"QMA9not-stbm").unwrap_err();
        assert!(matches!(err, StbmError::BadMagic { .. }));
    }
}
