use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const MAGIC: &[u8; 8] = b"RC2LDPC1";
const FRAME_SYMBOLS: usize = 384 * 512;
const GROUPS_PER_FRAME: usize = (1 + 2) * 64 - 2;
const NUM_CLASSES: usize = 5;
const ARITH_TOTAL: u32 = 32768;
const HALF: u64 = 0x8000_0000;
const QUARTER: u64 = 0x4000_0000;
const THREE_QUARTERS: u64 = 0xc000_0000;
const FULL: u64 = 0xffff_ffff;

#[derive(Clone, Copy, Debug)]
struct Config {
    alpha_milli: u16,
    degree: u8,
    iterations: u8,
    group_offset: u64,
}

#[derive(Debug)]
struct NpyArray {
    data: Vec<u8>,
    shape: Vec<usize>,
    descr: String,
}

#[derive(Debug)]
struct Graph {
    edge_var: Vec<usize>,
    edge_check: Vec<usize>,
    var_edges: Vec<Vec<usize>>,
    check_edges: Vec<Vec<usize>>,
}

#[derive(Debug)]
struct ChunkPacket {
    symbol_count: usize,
    group_count: usize,
    config: Config,
    flag_bits: usize,
    hit_bits: usize,
    miss_bits: usize,
    flags: Vec<u8>,
    hit_payload: Vec<u8>,
    miss_payload: Vec<u8>,
}

#[derive(Default, Debug)]
struct EncodeStats {
    groups: usize,
    fallback_groups: usize,
    miss_symbols: usize,
    syndrome_bits_attempted: usize,
    selected_hit_bits: usize,
    bp_groups: usize,
    bp_iterations: usize,
}

struct BitWriter {
    bytes: Vec<u8>,
    current: u8,
    used: u8,
    bits: usize,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            current: 0,
            used: 0,
            bits: 0,
        }
    }

    fn write(&mut self, bit: bool) {
        self.current |= u8::from(bit) << (7 - self.used);
        self.used += 1;
        self.bits += 1;
        if self.used == 8 {
            self.bytes.push(self.current);
            self.current = 0;
            self.used = 0;
        }
    }

    fn extend(&mut self, bits: &[bool]) {
        for &bit in bits {
            self.write(bit);
        }
    }

    fn finish(mut self) -> (Vec<u8>, usize) {
        if self.used != 0 {
            self.bytes.push(self.current);
        }
        (self.bytes, self.bits)
    }
}

struct BitReader<'a> {
    bytes: &'a [u8],
    position: usize,
    limit: usize,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8], limit: usize) -> Self {
        Self {
            bytes,
            position: 0,
            limit,
        }
    }

    fn read(&mut self) -> Result<bool, String> {
        if self.position >= self.limit {
            return Err("bitstream exhausted".to_string());
        }
        let byte = self.bytes[self.position / 8];
        let bit = ((byte >> (7 - (self.position % 8))) & 1) != 0;
        self.position += 1;
        Ok(bit)
    }

    fn read_many(&mut self, count: usize) -> Result<Vec<bool>, String> {
        (0..count).map(|_| self.read()).collect()
    }
}

struct ArithmeticEncoder {
    low: u64,
    high: u64,
    pending: usize,
    bits: BitWriter,
}

impl ArithmeticEncoder {
    fn new() -> Self {
        Self {
            low: 0,
            high: FULL,
            pending: 0,
            bits: BitWriter::new(),
        }
    }

    fn emit_with_pending(&mut self, bit: bool) {
        self.bits.write(bit);
        for _ in 0..self.pending {
            self.bits.write(!bit);
        }
        self.pending = 0;
    }

    fn encode(&mut self, symbol: usize, cumulative: &[u32; 5]) {
        let total = u64::from(cumulative[4]);
        let range = self.high - self.low + 1;
        self.high = self.low + range * u64::from(cumulative[symbol + 1]) / total - 1;
        self.low += range * u64::from(cumulative[symbol]) / total;
        loop {
            if self.high < HALF {
                self.emit_with_pending(false);
            } else if self.low >= HALF {
                self.emit_with_pending(true);
                self.low -= HALF;
                self.high -= HALF;
            } else if self.low >= QUARTER && self.high < THREE_QUARTERS {
                self.pending += 1;
                self.low -= QUARTER;
                self.high -= QUARTER;
            } else {
                break;
            }
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
            self.low &= FULL;
            self.high &= FULL;
        }
    }

    fn finish(mut self) -> (Vec<u8>, usize) {
        self.pending += 1;
        if self.low < QUARTER {
            self.emit_with_pending(false);
        } else {
            self.emit_with_pending(true);
        }
        self.bits.finish()
    }
}

struct ArithmeticDecoder<'a> {
    low: u64,
    high: u64,
    value: u64,
    bits: BitReader<'a>,
}

impl<'a> ArithmeticDecoder<'a> {
    fn new(bytes: &'a [u8], bit_count: usize) -> Self {
        let mut bits = BitReader::new(bytes, bit_count);
        let mut value = 0u64;
        for _ in 0..32 {
            value = (value << 1) | u64::from(bits.read().unwrap_or(false));
        }
        Self {
            low: 0,
            high: FULL,
            value,
            bits,
        }
    }

    fn decode(&mut self, cumulative: &[u32; 5]) -> Result<usize, String> {
        let total = u64::from(cumulative[4]);
        let range = self.high - self.low + 1;
        let scaled = ((self.value - self.low + 1) * total - 1) / range;
        let symbol = (0..4)
            .find(|&index| scaled < u64::from(cumulative[index + 1]))
            .ok_or_else(|| "arithmetic symbol lookup failed".to_string())?;
        self.high = self.low + range * u64::from(cumulative[symbol + 1]) / total - 1;
        self.low += range * u64::from(cumulative[symbol]) / total;
        loop {
            if self.high < HALF {
            } else if self.low >= HALF {
                self.value -= HALF;
                self.low -= HALF;
                self.high -= HALF;
            } else if self.low >= QUARTER && self.high < THREE_QUARTERS {
                self.value -= QUARTER;
                self.low -= QUARTER;
                self.high -= QUARTER;
            } else {
                break;
            }
            self.low = (self.low << 1) & FULL;
            self.high = ((self.high << 1) | 1) & FULL;
            self.value = ((self.value << 1) & FULL) | u64::from(self.bits.read().unwrap_or(false));
        }
        Ok(symbol)
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn build_graph(n: usize, m: usize, degree: usize, seed: u64) -> Graph {
    let mut graph = Graph {
        edge_var: Vec::new(),
        edge_check: Vec::new(),
        var_edges: vec![Vec::new(); n],
        check_edges: vec![Vec::new(); m],
    };
    if m == 0 {
        return graph;
    }
    let actual_degree = degree.min(m);
    for variable in 0..n {
        let mut selected = Vec::with_capacity(actual_degree);
        let mut attempt = 0u64;
        while selected.len() < actual_degree {
            let mixed = splitmix64(
                seed ^ (variable as u64).wrapping_mul(0xd6e8_feb8_6659_fd93)
                    ^ attempt.wrapping_mul(0xa076_1d64_78bd_642f),
            );
            let check = (mixed % m as u64) as usize;
            if !selected.contains(&check) {
                selected.push(check);
            }
            attempt += 1;
        }
        for check in selected {
            let edge = graph.edge_var.len();
            graph.edge_var.push(variable);
            graph.edge_check.push(check);
            graph.var_edges[variable].push(edge);
            graph.check_edges[check].push(edge);
        }
    }
    graph
}

fn syndrome(graph: &Graph, bits: &[bool]) -> Vec<bool> {
    graph
        .check_edges
        .iter()
        .map(|edges| {
            edges
                .iter()
                .fold(false, |value, &edge| value ^ bits[graph.edge_var[edge]])
        })
        .collect()
}

fn syndrome_matches(graph: &Graph, bits: &[bool], expected: &[bool]) -> bool {
    graph
        .check_edges
        .iter()
        .zip(expected)
        .all(|(edges, &target)| {
            edges
                .iter()
                .fold(false, |value, &edge| value ^ bits[graph.edge_var[edge]])
                == target
        })
}

fn bp_decode(
    graph: &Graph,
    priors: &[f64],
    target_syndrome: &[bool],
    max_iterations: usize,
) -> (Vec<bool>, usize, bool) {
    let mut q = graph
        .edge_var
        .iter()
        .map(|&variable| priors[variable])
        .collect::<Vec<_>>();
    let mut r = vec![0.0f64; q.len()];
    let mut hard = priors.iter().map(|&value| value < 0.0).collect::<Vec<_>>();
    if syndrome_matches(graph, &hard, target_syndrome) {
        return (hard, 0, true);
    }
    for iteration in 1..=max_iterations {
        for (check, edges) in graph.check_edges.iter().enumerate() {
            for &edge in edges {
                let mut sign = if target_syndrome[check] { -1.0 } else { 1.0 };
                let mut minimum = 20.0f64;
                for &other in edges {
                    if other == edge {
                        continue;
                    }
                    sign *= if q[other] < 0.0 { -1.0 } else { 1.0 };
                    minimum = minimum.min(q[other].abs());
                }
                r[edge] = 0.8 * sign * minimum;
            }
        }
        for (variable, edges) in graph.var_edges.iter().enumerate() {
            let posterior = priors[variable] + edges.iter().map(|&edge| r[edge]).sum::<f64>();
            hard[variable] = posterior < 0.0;
            for &edge in edges {
                q[edge] = (posterior - r[edge]).clamp(-20.0, 20.0);
            }
        }
        if syndrome_matches(graph, &hard, target_syndrome) {
            return (hard, iteration, true);
        }
    }
    (hard, max_iterations, false)
}

fn group_sizes() -> Vec<usize> {
    let mut local = vec![0usize; GROUPS_PER_FRAME];
    for row in 0..64 {
        for column in 0..64 {
            local[column + 2 * row] += 1;
        }
    }
    local.into_iter().map(|count| count * 48).collect()
}

fn probabilities(codes: &[i16; 5]) -> ([f64; 5], usize) {
    let maximum = *codes.iter().max().expect("five logits") as f64 / 8.0;
    let mut weights = [0.0f64; 5];
    let mut total = 0.0;
    for index in 0..5 {
        weights[index] = (codes[index] as f64 / 8.0 - maximum).exp();
        total += weights[index];
    }
    for value in &mut weights {
        *value /= total;
    }
    // Match the PR130 NumPy/Torch argmax contract: retain the first class on
    // an equal maximum.  This is both the receiver convention and the lower-
    // miss rule on the retained token object.
    let mut top = 0usize;
    for index in 1..5 {
        if weights[index] > weights[top] {
            top = index;
        }
    }
    (weights, top)
}

fn binary_entropy(probability: f64) -> f64 {
    let p = probability.clamp(1e-15, 1.0 - 1e-15);
    -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
}

fn miss_cumulative(probabilities: &[f64; 5], top: usize) -> ([usize; 4], [u32; 5]) {
    let mut alternatives = [0usize; 4];
    let mut position = 0;
    for class in 0..5 {
        if class != top {
            alternatives[position] = class;
            position += 1;
        }
    }
    let total_probability = alternatives
        .iter()
        .map(|&class| probabilities[class])
        .sum::<f64>();
    let budget = ARITH_TOTAL - 4;
    let mut frequencies = [1u32; 4];
    let mut fractions = [0.0f64; 4];
    let mut allocated = 4u32;
    for index in 0..4 {
        let exact = probabilities[alternatives[index]] / total_probability * f64::from(budget);
        let floor = exact.floor() as u32;
        frequencies[index] += floor;
        fractions[index] = exact - f64::from(floor);
        allocated += floor;
    }
    while allocated < ARITH_TOTAL {
        let index = (0..4)
            .max_by(|&a, &b| fractions[a].total_cmp(&fractions[b]))
            .unwrap();
        frequencies[index] += 1;
        fractions[index] = -1.0;
        allocated += 1;
    }
    while allocated > ARITH_TOTAL {
        let index = (0..4)
            .filter(|&candidate| frequencies[candidate] > 1)
            .max_by_key(|&candidate| frequencies[candidate])
            .expect("at least one reducible arithmetic frequency");
        frequencies[index] -= 1;
        allocated -= 1;
    }
    let mut cumulative = [0u32; 5];
    for index in 0..4 {
        cumulative[index + 1] = cumulative[index] + frequencies[index];
    }
    (alternatives, cumulative)
}

fn parse_npy(path: &Path) -> Result<NpyArray, String> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(format!("{} is not an NPY file", path.display()));
    }
    let major = bytes[6];
    let (header_len, header_start) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10usize)
    } else {
        if bytes.len() < 12 {
            return Err("truncated NPY v2 header".to_string());
        }
        (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12usize,
        )
    };
    let header_end = header_start + header_len;
    if header_end > bytes.len() {
        return Err("NPY header exceeds file".to_string());
    }
    let header = std::str::from_utf8(&bytes[header_start..header_end])
        .map_err(|error| format!("NPY header UTF-8: {error}"))?;
    let descr = if header.contains("'|u1'") {
        "|u1"
    } else if header.contains("'<i2'") {
        "<i2"
    } else {
        return Err(format!("unsupported NPY dtype in {header:?}"));
    };
    let shape_start = header
        .find("'shape': (")
        .ok_or_else(|| "NPY shape missing".to_string())?
        + 10;
    let shape_end = header[shape_start..]
        .find(')')
        .ok_or_else(|| "NPY shape terminator missing".to_string())?
        + shape_start;
    let shape = header[shape_start..shape_end]
        .split(',')
        .filter_map(|part| part.trim().parse::<usize>().ok())
        .collect::<Vec<_>>();
    Ok(NpyArray {
        data: bytes[header_end..].to_vec(),
        shape,
        descr: descr.to_string(),
    })
}

fn load_inputs(symbols_path: &Path, codes_path: &Path) -> Result<(Vec<u8>, Vec<[i16; 5]>), String> {
    let symbols = parse_npy(symbols_path)?;
    let codes = parse_npy(codes_path)?;
    if symbols.descr != "|u1" || symbols.shape.len() != 1 {
        return Err("symbols NPY must be one-dimensional uint8".to_string());
    }
    if codes.descr != "<i2" || codes.shape != vec![symbols.shape[0], NUM_CLASSES] {
        return Err("codes NPY must have shape (symbols, 5) and dtype int16".to_string());
    }
    if codes.data.len() != symbols.data.len() * NUM_CLASSES * 2 {
        return Err("codes NPY byte count mismatch".to_string());
    }
    let mut parsed_codes = Vec::with_capacity(symbols.data.len());
    for row in codes.data.chunks_exact(10) {
        let mut values = [0i16; 5];
        for class in 0..5 {
            values[class] = i16::from_le_bytes([row[class * 2], row[class * 2 + 1]]);
        }
        parsed_codes.push(values);
    }
    Ok((symbols.data, parsed_codes))
}

fn atomic_write(path: &Path, payload: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| "output path has no parent".to_string())?;
    fs::create_dir_all(parent).map_err(|error| format!("mkdir {}: {error}", parent.display()))?;
    let temporary = path.with_file_name(format!(
        ".{}.{}.tmp",
        path.file_name().unwrap_or_default().to_string_lossy(),
        std::process::id()
    ));
    fs::write(&temporary, payload)
        .map_err(|error| format!("write {}: {error}", temporary.display()))?;
    fs::rename(&temporary, path).map_err(|error| format!("rename {}: {error}", path.display()))
}

fn u32_bytes(value: usize) -> Result<[u8; 4], String> {
    Ok(u32::try_from(value)
        .map_err(|_| "value does not fit u32".to_string())?
        .to_le_bytes())
}

fn serialize_packet(packet: &ChunkPacket) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    output.extend_from_slice(MAGIC);
    output.extend_from_slice(&u32_bytes(packet.symbol_count)?);
    output.extend_from_slice(&u32_bytes(packet.group_count)?);
    output.extend_from_slice(&packet.config.alpha_milli.to_le_bytes());
    output.push(packet.config.degree);
    output.push(packet.config.iterations);
    output.extend_from_slice(&packet.config.group_offset.to_le_bytes());
    output.extend_from_slice(&(packet.flag_bits as u64).to_le_bytes());
    output.extend_from_slice(&(packet.hit_bits as u64).to_le_bytes());
    output.extend_from_slice(&(packet.miss_bits as u64).to_le_bytes());
    output.extend_from_slice(&u32_bytes(packet.flags.len())?);
    output.extend_from_slice(&u32_bytes(packet.hit_payload.len())?);
    output.extend_from_slice(&u32_bytes(packet.miss_payload.len())?);
    output.extend_from_slice(&packet.flags);
    output.extend_from_slice(&packet.hit_payload);
    output.extend_from_slice(&packet.miss_payload);
    Ok(output)
}

fn read_u32(data: &[u8], cursor: &mut usize) -> Result<u32, String> {
    let end = *cursor + 4;
    let bytes: [u8; 4] = data
        .get(*cursor..end)
        .ok_or_else(|| "packet truncated".to_string())?
        .try_into()
        .unwrap();
    *cursor = end;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(data: &[u8], cursor: &mut usize) -> Result<u64, String> {
    let end = *cursor + 8;
    let bytes: [u8; 8] = data
        .get(*cursor..end)
        .ok_or_else(|| "packet truncated".to_string())?
        .try_into()
        .unwrap();
    *cursor = end;
    Ok(u64::from_le_bytes(bytes))
}

fn parse_packet(data: &[u8]) -> Result<ChunkPacket, String> {
    if data.len() < 8 || &data[..8] != MAGIC {
        return Err("LDPC packet magic mismatch".to_string());
    }
    let mut cursor = 8usize;
    let symbol_count = read_u32(data, &mut cursor)? as usize;
    let group_count = read_u32(data, &mut cursor)? as usize;
    let alpha_bytes: [u8; 2] = data
        .get(cursor..cursor + 2)
        .ok_or_else(|| "packet truncated".to_string())?
        .try_into()
        .unwrap();
    let alpha_milli = u16::from_le_bytes(alpha_bytes);
    cursor += 2;
    let degree = *data
        .get(cursor)
        .ok_or_else(|| "packet truncated".to_string())?;
    let iterations = *data
        .get(cursor + 1)
        .ok_or_else(|| "packet truncated".to_string())?;
    cursor += 2;
    let group_offset = read_u64(data, &mut cursor)?;
    let flag_bits = read_u64(data, &mut cursor)? as usize;
    let hit_bits = read_u64(data, &mut cursor)? as usize;
    let miss_bits = read_u64(data, &mut cursor)? as usize;
    let flag_bytes = read_u32(data, &mut cursor)? as usize;
    let hit_bytes = read_u32(data, &mut cursor)? as usize;
    let miss_bytes = read_u32(data, &mut cursor)? as usize;
    let flags = data
        .get(cursor..cursor + flag_bytes)
        .ok_or_else(|| "flags truncated".to_string())?
        .to_vec();
    cursor += flag_bytes;
    let hit_payload = data
        .get(cursor..cursor + hit_bytes)
        .ok_or_else(|| "hit payload truncated".to_string())?
        .to_vec();
    cursor += hit_bytes;
    let miss_payload = data
        .get(cursor..cursor + miss_bytes)
        .ok_or_else(|| "miss payload truncated".to_string())?
        .to_vec();
    cursor += miss_bytes;
    if cursor != data.len() {
        return Err("packet has trailing bytes".to_string());
    }
    Ok(ChunkPacket {
        symbol_count,
        group_count,
        config: Config {
            alpha_milli,
            degree,
            iterations,
            group_offset,
        },
        flag_bits,
        hit_bits,
        miss_bits,
        flags,
        hit_payload,
        miss_payload,
    })
}

fn encode_chunk(
    symbols: &[u8],
    codes: &[[i16; 5]],
    config: Config,
) -> Result<(ChunkPacket, Vec<u8>, EncodeStats), String> {
    if symbols.len() != codes.len() || !symbols.len().is_multiple_of(FRAME_SYMBOLS) {
        return Err("chunk is not an integral number of PR130 frames".to_string());
    }
    let sizes = group_sizes();
    let mut flags = BitWriter::new();
    let mut selected_hits = BitWriter::new();
    let mut attempted = BitWriter::new();
    let mut miss_encoder = ArithmeticEncoder::new();
    let mut stats = EncodeStats::default();
    let mut offset = 0usize;
    for group_index in 0..symbols.len() / FRAME_SYMBOLS * GROUPS_PER_FRAME {
        let n = sizes[group_index % GROUPS_PER_FRAME];
        let source = &symbols[offset..offset + n];
        let context = &codes[offset..offset + n];
        let mut actual_hits = Vec::with_capacity(n);
        let mut priors = Vec::with_capacity(n);
        let mut entropy = 0.0f64;
        let mut probability_rows = Vec::with_capacity(n);
        let mut tops = Vec::with_capacity(n);
        for index in 0..n {
            let (probability, top) = probabilities(&context[index]);
            let hit = usize::from(source[index]) == top;
            let p_hit = probability[top].clamp(1e-15, 1.0 - 1e-15);
            actual_hits.push(hit);
            priors.push(((1.0 - p_hit) / p_hit).ln().clamp(-20.0, 20.0));
            entropy += binary_entropy(p_hit);
            probability_rows.push(probability);
            tops.push(top);
        }
        let m = ((entropy * f64::from(config.alpha_milli) / 1000.0).ceil() as usize).min(n);
        let seed = splitmix64(config.group_offset + group_index as u64);
        let graph = build_graph(n, m, usize::from(config.degree), seed);
        let check_bits = syndrome(&graph, &actual_hits);
        attempted.extend(&check_bits);
        stats.syndrome_bits_attempted += check_bits.len();
        let (decoded, iterations, syndrome_ok) =
            bp_decode(&graph, &priors, &check_bits, usize::from(config.iterations));
        let exact = syndrome_ok && decoded == actual_hits;
        flags.write(!exact);
        if exact {
            selected_hits.extend(&check_bits);
            stats.bp_groups += usize::from(iterations > 0);
            stats.bp_iterations += iterations;
        } else {
            selected_hits.extend(&actual_hits);
            stats.fallback_groups += 1;
        }
        stats.selected_hit_bits += if exact { m } else { n };
        for index in 0..n {
            if !actual_hits[index] {
                let (alternatives, cumulative) =
                    miss_cumulative(&probability_rows[index], tops[index]);
                let symbol = alternatives
                    .iter()
                    .position(|&class| class == usize::from(source[index]))
                    .ok_or_else(|| "miss symbol equals neither alternative".to_string())?;
                miss_encoder.encode(symbol, &cumulative);
                stats.miss_symbols += 1;
            }
        }
        stats.groups += 1;
        offset += n;
    }
    if offset != symbols.len() {
        return Err("group geometry did not consume the chunk".to_string());
    }
    let (flags_bytes, flag_bits) = flags.finish();
    let (hit_payload, hit_bits) = selected_hits.finish();
    let (attempted_payload, _) = attempted.finish();
    let (miss_payload, miss_bits) = miss_encoder.finish();
    let packet = ChunkPacket {
        symbol_count: symbols.len(),
        group_count: stats.groups,
        config,
        flag_bits,
        hit_bits,
        miss_bits,
        flags: flags_bytes,
        hit_payload,
        miss_payload,
    };
    Ok((packet, attempted_payload, stats))
}

fn decode_chunk(codes: &[[i16; 5]], packet: &ChunkPacket) -> Result<Vec<u8>, String> {
    if codes.len() != packet.symbol_count || !packet.symbol_count.is_multiple_of(FRAME_SYMBOLS) {
        return Err("decoder input shape does not match packet".to_string());
    }
    let sizes = group_sizes();
    let mut flags = BitReader::new(&packet.flags, packet.flag_bits);
    let mut hits = BitReader::new(&packet.hit_payload, packet.hit_bits);
    let mut miss_decoder = ArithmeticDecoder::new(&packet.miss_payload, packet.miss_bits);
    let mut output = Vec::with_capacity(packet.symbol_count);
    let mut offset = 0usize;
    for group_index in 0..packet.group_count {
        let n = sizes[group_index % GROUPS_PER_FRAME];
        let context = &codes[offset..offset + n];
        let mut priors = Vec::with_capacity(n);
        let mut entropy = 0.0f64;
        let mut probability_rows = Vec::with_capacity(n);
        let mut tops = Vec::with_capacity(n);
        for code in context {
            let (probability, top) = probabilities(code);
            let p_hit = probability[top].clamp(1e-15, 1.0 - 1e-15);
            priors.push(((1.0 - p_hit) / p_hit).ln().clamp(-20.0, 20.0));
            entropy += binary_entropy(p_hit);
            probability_rows.push(probability);
            tops.push(top);
        }
        let m = ((entropy * f64::from(packet.config.alpha_milli) / 1000.0).ceil() as usize).min(n);
        let fallback = flags.read()?;
        let decoded_hits = if fallback {
            hits.read_many(n)?
        } else {
            let target = hits.read_many(m)?;
            let seed = splitmix64(packet.config.group_offset + group_index as u64);
            let graph = build_graph(n, m, usize::from(packet.config.degree), seed);
            let (decoded, _, syndrome_ok) = bp_decode(
                &graph,
                &priors,
                &target,
                usize::from(packet.config.iterations),
            );
            if !syndrome_ok {
                return Err(format!("BP failed for non-fallback group {group_index}"));
            }
            decoded
        };
        for index in 0..n {
            if decoded_hits[index] {
                output.push(tops[index] as u8);
            } else {
                let (alternatives, cumulative) =
                    miss_cumulative(&probability_rows[index], tops[index]);
                output.push(alternatives[miss_decoder.decode(&cumulative)?] as u8);
            }
        }
        offset += n;
    }
    if output.len() != packet.symbol_count
        || flags.position != packet.flag_bits
        || hits.position != packet.hit_bits
    {
        return Err("decoder did not consume the declared symbol/bit counts".to_string());
    }
    Ok(output)
}

fn json_stats(stats: &EncodeStats, packet_bytes: usize, elapsed: f64) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"axis\": \"[macOS-CPU advisory, scorer-free]\",\n",
            "  \"score_claim\": false,\n",
            "  \"groups\": {},\n",
            "  \"fallback_groups\": {},\n",
            "  \"miss_symbols\": {},\n",
            "  \"syndrome_bits_attempted\": {},\n",
            "  \"selected_hit_bits\": {},\n",
            "  \"bp_groups\": {},\n",
            "  \"bp_iterations\": {},\n",
            "  \"packet_bytes\": {},\n",
            "  \"encode_seconds\": {:.9}\n",
            "}}\n"
        ),
        stats.groups,
        stats.fallback_groups,
        stats.miss_symbols,
        stats.syndrome_bits_attempted,
        stats.selected_hit_bits,
        stats.bp_groups,
        stats.bp_iterations,
        packet_bytes,
        elapsed,
    )
}

fn parse_value<T: std::str::FromStr>(args: &[String], name: &str) -> Result<T, String> {
    let index = args
        .iter()
        .position(|value| value == name)
        .ok_or_else(|| format!("missing {name}"))?;
    args.get(index + 1)
        .ok_or_else(|| format!("missing value for {name}"))?
        .parse::<T>()
        .map_err(|_| format!("invalid value for {name}"))
}

fn parse_path(args: &[String], name: &str) -> Result<PathBuf, String> {
    parse_value::<String>(args, name).map(PathBuf::from)
}

fn run_encode(args: &[String]) -> Result<(), String> {
    let symbols_path = parse_path(args, "--symbols")?;
    let codes_path = parse_path(args, "--codes")?;
    let output = parse_path(args, "--output")?;
    let config = Config {
        alpha_milli: parse_value(args, "--alpha-milli")?,
        degree: parse_value(args, "--degree")?,
        iterations: parse_value(args, "--iterations")?,
        group_offset: parse_value(args, "--group-offset")?,
    };
    let (symbols, codes) = load_inputs(&symbols_path, &codes_path)?;
    let started = Instant::now();
    let (packet, attempted, stats) = encode_chunk(&symbols, &codes, config)?;
    let packet_bytes = serialize_packet(&packet)?;
    atomic_write(&output.join("chunk_packet.bin"), &packet_bytes)?;
    atomic_write(&output.join("attempted_syndromes.bin"), &attempted)?;
    atomic_write(
        &output.join("encode_metrics.json"),
        json_stats(&stats, packet_bytes.len(), started.elapsed().as_secs_f64()).as_bytes(),
    )?;
    println!(
        "{}",
        json_stats(&stats, packet_bytes.len(), started.elapsed().as_secs_f64())
    );
    Ok(())
}

fn run_decode(args: &[String]) -> Result<(), String> {
    let symbols_path = parse_path(args, "--symbols")?;
    let codes_path = parse_path(args, "--codes")?;
    let packet_path = parse_path(args, "--packet")?;
    let output = parse_path(args, "--output")?;
    let (symbols, codes) = load_inputs(&symbols_path, &codes_path)?;
    let packet_bytes =
        fs::read(&packet_path).map_err(|error| format!("{}: {error}", packet_path.display()))?;
    let packet = parse_packet(&packet_bytes)?;
    let started = Instant::now();
    let decoded = decode_chunk(&codes, &packet)?;
    atomic_write(&output, &decoded)?;
    if decoded != symbols {
        let mismatch = decoded
            .iter()
            .zip(&symbols)
            .position(|(left, right)| left != right)
            .unwrap_or(decoded.len().min(symbols.len()));
        return Err(format!("decoded symbols differ at index {mismatch}"));
    }
    let metrics = format!(
        "{{\"axis\":\"[macOS-CPU advisory, scorer-free]\",\"score_claim\":false,\"exact_decode\":true,\"symbols\":{},\"decode_seconds\":{:.9}}}",
        decoded.len(),
        started.elapsed().as_secs_f64()
    );
    let metrics_path = output.with_file_name("decode_metrics.json");
    atomic_write(&metrics_path, format!("{metrics}\n").as_bytes())?;
    println!("{metrics}");
    Ok(())
}

fn usage() -> &'static str {
    "usage:\n  ddm-rc2-ldpc-native encode --symbols S.npy --codes C.npy --output DIR --alpha-milli N --degree N --iterations N --group-offset N\n  ddm-rc2-ldpc-native decode --symbols S.npy --codes C.npy --packet P.bin --output decoded.bin"
}

fn main() {
    let args = env::args().collect::<Vec<_>>();
    let result = match args.get(1).map(String::as_str) {
        Some("encode") => run_encode(&args[2..]),
        Some("decode") => run_decode(&args[2..]),
        _ => Err(usage().to_string()),
    };
    if let Err(error) = result {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
