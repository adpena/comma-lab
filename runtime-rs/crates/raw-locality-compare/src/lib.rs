use memmap2::{Mmap, MmapOptions};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct CompareInput {
    pub parent_raw: PathBuf,
    pub global_mutated_raw: PathBuf,
    pub selective_raw: PathBuf,
    pub selected_frame_indices: Vec<i64>,
    pub frame_count: usize,
    pub frame_bytes: Option<usize>,
    pub rel_path: String,
    pub sample_limit: usize,
}

#[derive(Debug)]
pub enum CompareError {
    InvalidInput(String),
    Io(io::Error),
}

impl std::fmt::Display for CompareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompareError::InvalidInput(message) => write!(f, "{message}"),
            CompareError::Io(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for CompareError {}

impl From<io::Error> for CompareError {
    fn from(err: io::Error) -> Self {
        CompareError::Io(err)
    }
}

pub fn compare_raw_triplet(input: &CompareInput) -> Result<Value, CompareError> {
    if input.frame_count == 0 {
        return Err(CompareError::InvalidInput(
            "frame_count must be positive".to_string(),
        ));
    }

    for path in [
        &input.parent_raw,
        &input.global_mutated_raw,
        &input.selective_raw,
    ] {
        if !path.is_file() {
            return Ok(json!({
                "raw_path": input.rel_path,
                "raw_byte_sizes_match": false,
                "frame_bytes": input.frame_bytes,
                "frame_count": input.frame_count,
                "selected_frame_indices": input.selected_frame_indices,
                "hashes": {},
                "mismatch_counts": {
                    "missing_raw_file_count": 1,
                    "raw_size_mismatch_count": 1,
                    "selected_frame_mismatch_count": 0,
                    "unselected_frame_mismatch_count": 0,
                },
                "mismatch_samples": [],
                "blockers": [format!("missing_raw_file:{}:{}", input.rel_path, path.display())],
                "compare_engine": compare_engine_record(),
            }));
        }
    }

    let sizes = raw_sizes(input)?;
    let raw_size = sizes["parent"];
    let mut mismatch_counts = mismatch_counts(0, 0, 0, 0);
    let mut blockers: Vec<String> = Vec::new();

    if sizes.values().collect::<BTreeSet<&u64>>().len() != 1 {
        mismatch_counts.insert("raw_size_mismatch_count".to_string(), 1);
        blockers.push(format!("raw_size_mismatch:{}", input.rel_path));
        return Ok(json!({
            "raw_path": input.rel_path,
            "raw_byte_sizes_match": false,
            "frame_bytes": input.frame_bytes,
            "frame_count": input.frame_count,
            "selected_frame_indices": input.selected_frame_indices,
            "hashes": {"raw_files": raw_file_hashes(input)?},
            "raw_bytes": sizes,
            "mismatch_counts": mismatch_counts,
            "mismatch_samples": [],
            "blockers": blockers,
            "compare_engine": compare_engine_record(),
        }));
    }

    let frame_bytes = match input.frame_bytes {
        Some(value) => value,
        None => {
            if raw_size % input.frame_count as u64 != 0 {
                mismatch_counts.insert("raw_size_mismatch_count".to_string(), 1);
                blockers.push(format!(
                    "raw_size_not_divisible_by_frame_count:{}",
                    input.rel_path
                ));
                return Ok(json!({
                    "raw_path": input.rel_path,
                    "raw_byte_sizes_match": false,
                    "frame_bytes": null,
                    "frame_count": input.frame_count,
                    "selected_frame_indices": input.selected_frame_indices,
                    "hashes": {"raw_files": raw_file_hashes(input)?},
                    "raw_bytes": sizes,
                    "mismatch_counts": mismatch_counts,
                    "mismatch_samples": [],
                    "blockers": blockers,
                    "compare_engine": compare_engine_record(),
                }));
            }
            (raw_size / input.frame_count as u64) as usize
        }
    };
    if frame_bytes == 0 {
        return Err(CompareError::InvalidInput(
            "frame_bytes must be positive".to_string(),
        ));
    }

    let expected_size = frame_bytes
        .checked_mul(input.frame_count)
        .ok_or_else(|| CompareError::InvalidInput("raw frame geometry overflow".to_string()))?;
    if raw_size != expected_size as u64 {
        mismatch_counts.insert("raw_size_mismatch_count".to_string(), 1);
        blockers.push(format!(
            "raw_size_does_not_match_frame_geometry:{}",
            input.rel_path
        ));
        return Ok(json!({
            "raw_path": input.rel_path,
            "raw_byte_sizes_match": false,
            "frame_bytes": frame_bytes,
            "frame_count": input.frame_count,
            "selected_frame_indices": input.selected_frame_indices,
            "hashes": {"raw_files": raw_file_hashes(input)?},
            "raw_bytes": sizes,
            "mismatch_counts": mismatch_counts,
            "mismatch_samples": [],
            "blockers": blockers,
            "compare_engine": compare_engine_record(),
        }));
    }

    let invalid_frames: Vec<i64> = input
        .selected_frame_indices
        .iter()
        .copied()
        .filter(|frame| *frame < 0 || *frame >= input.frame_count as i64)
        .collect();
    if !invalid_frames.is_empty() {
        blockers.push(format!(
            "selected_frame_index_out_of_range:{invalid_frames:?}"
        ));
        return Ok(json!({
            "raw_path": input.rel_path,
            "raw_byte_sizes_match": true,
            "frame_bytes": frame_bytes,
            "frame_count": input.frame_count,
            "selected_frame_indices": input.selected_frame_indices,
            "hashes": {"raw_files": raw_file_hashes(input)?},
            "raw_bytes": sizes,
            "mismatch_counts": mismatch_counts,
            "mismatch_samples": [],
            "blockers": blockers,
            "compare_engine": compare_engine_record(),
        }));
    }

    let parent = mmap_file(&input.parent_raw)?;
    let global_mutated = mmap_file(&input.global_mutated_raw)?;
    let selective = mmap_file(&input.selective_raw)?;
    let selected_set: BTreeSet<usize> = input
        .selected_frame_indices
        .iter()
        .map(|frame| *frame as usize)
        .collect();

    let mut raw_hashes = RegionHashers::default();
    let mut selected_hashes = RegionHashers::default();
    let mut unselected_hashes = RegionHashers::default();
    let mut selected_samples: Vec<Value> = Vec::new();
    let mut unselected_samples: Vec<Value> = Vec::new();
    let mut selected_mismatches = 0_u64;
    let mut unselected_mismatches = 0_u64;

    let mut frame_index = 0_usize;
    while frame_index < input.frame_count {
        let range_selected = selected_set.contains(&frame_index);
        let start_frame = frame_index;
        frame_index += 1;
        while frame_index < input.frame_count
            && selected_set.contains(&frame_index) == range_selected
        {
            frame_index += 1;
        }
        let end_frame = frame_index;
        let start = start_frame * frame_bytes;
        let end = end_frame * frame_bytes;
        let parent_chunk = &parent[start..end];
        let global_chunk = &global_mutated[start..end];
        let selective_chunk = &selective[start..end];

        raw_hashes.update(parent_chunk, global_chunk, selective_chunk);
        if range_selected {
            selected_hashes.update(parent_chunk, global_chunk, selective_chunk);
            if selective_chunk != global_chunk {
                for mismatched_frame in mismatched_frames(
                    start_frame,
                    end_frame,
                    frame_bytes,
                    global_chunk,
                    selective_chunk,
                ) {
                    selected_mismatches += 1;
                    if selected_samples.len() < input.sample_limit {
                        let offset = (mismatched_frame - start_frame) * frame_bytes;
                        selected_samples.push(frame_sample(
                            mismatched_frame,
                            "global_mutated",
                            &global_chunk[offset..offset + frame_bytes],
                            &selective_chunk[offset..offset + frame_bytes],
                        ));
                    }
                }
            }
        } else {
            unselected_hashes.update(parent_chunk, global_chunk, selective_chunk);
            if selective_chunk != parent_chunk {
                for mismatched_frame in mismatched_frames(
                    start_frame,
                    end_frame,
                    frame_bytes,
                    parent_chunk,
                    selective_chunk,
                ) {
                    unselected_mismatches += 1;
                    if unselected_samples.len() < input.sample_limit {
                        let offset = (mismatched_frame - start_frame) * frame_bytes;
                        unselected_samples.push(frame_sample(
                            mismatched_frame,
                            "parent",
                            &parent_chunk[offset..offset + frame_bytes],
                            &selective_chunk[offset..offset + frame_bytes],
                        ));
                    }
                }
            }
        }
    }

    if selected_mismatches != 0 {
        blockers.push(format!(
            "selected_frame_locality_mismatch:{}",
            input.rel_path
        ));
    }
    if unselected_mismatches != 0 {
        blockers.push(format!(
            "unselected_frame_parent_regression:{}",
            input.rel_path
        ));
    }
    mismatch_counts.insert(
        "selected_frame_mismatch_count".to_string(),
        selected_mismatches,
    );
    mismatch_counts.insert(
        "unselected_frame_mismatch_count".to_string(),
        unselected_mismatches,
    );

    let mut mismatch_samples = selected_samples;
    mismatch_samples.extend(unselected_samples);

    Ok(json!({
        "raw_path": input.rel_path,
        "raw_byte_sizes_match": true,
        "frame_bytes": frame_bytes,
        "frame_count": input.frame_count,
        "selected_frame_indices": input.selected_frame_indices,
        "compared_selected_frame_count": selected_set.len(),
        "compared_unselected_frame_count": input.frame_count - selected_set.len(),
        "hashes": {
            "raw_files": raw_hashes.finish(),
            "selected_frames": selected_hashes.finish(),
            "unselected_frames": unselected_hashes.finish(),
        },
        "raw_bytes": sizes,
        "mismatch_counts": mismatch_counts,
        "mismatch_samples": mismatch_samples,
        "blockers": blockers,
        "compare_engine": compare_engine_record(),
    }))
}

#[derive(Default)]
struct RegionHashers {
    parent: Sha256Context,
    global_mutated: Sha256Context,
    selective: Sha256Context,
}

impl RegionHashers {
    fn update(&mut self, parent: &[u8], global_mutated: &[u8], selective: &[u8]) {
        self.parent.update(parent);
        self.global_mutated.update(global_mutated);
        self.selective.update(selective);
    }

    fn finish(self) -> BTreeMap<&'static str, String> {
        BTreeMap::from([
            ("global_mutated", hex_digest_finalize(self.global_mutated)),
            ("parent", hex_digest_finalize(self.parent)),
            ("selective", hex_digest_finalize(self.selective)),
        ])
    }
}

fn compare_engine_record() -> Value {
    json!({
        "name": "rust",
        "schema": "raw_locality_compare_engine.v1",
        "crate": "raw-locality-compare",
    })
}

fn mmap_file(path: &Path) -> Result<Mmap, CompareError> {
    let file = File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    Ok(mmap)
}

fn raw_sizes(input: &CompareInput) -> Result<BTreeMap<&'static str, u64>, CompareError> {
    Ok(BTreeMap::from([
        (
            "global_mutated",
            fs::metadata(&input.global_mutated_raw)?.len(),
        ),
        ("parent", fs::metadata(&input.parent_raw)?.len()),
        ("selective", fs::metadata(&input.selective_raw)?.len()),
    ]))
}

fn raw_file_hashes(input: &CompareInput) -> Result<BTreeMap<&'static str, String>, CompareError> {
    Ok(BTreeMap::from([
        ("global_mutated", hash_file(&input.global_mutated_raw)?),
        ("parent", hash_file(&input.parent_raw)?),
        ("selective", hash_file(&input.selective_raw)?),
    ]))
}

fn hash_file(path: &Path) -> Result<String, CompareError> {
    let mmap = mmap_file(path)?;
    Ok(hex_digest(&mmap))
}

fn mismatched_frames<'a>(
    start_frame: usize,
    end_frame: usize,
    frame_bytes: usize,
    expected: &'a [u8],
    actual: &'a [u8],
) -> impl Iterator<Item = usize> + 'a {
    (start_frame..end_frame).filter(move |frame_index| {
        let offset = (*frame_index - start_frame) * frame_bytes;
        expected[offset..offset + frame_bytes] != actual[offset..offset + frame_bytes]
    })
}

fn frame_sample(frame_index: usize, expected_role: &str, expected: &[u8], actual: &[u8]) -> Value {
    json!({
        "frame_index": frame_index,
        "expected_role": expected_role,
        "expected_sha256": hex_digest(expected),
        "actual_selective_sha256": hex_digest(actual),
    })
}

fn mismatch_counts(
    missing: u64,
    raw_size: u64,
    selected: u64,
    unselected: u64,
) -> BTreeMap<String, u64> {
    BTreeMap::from([
        ("missing_raw_file_count".to_string(), missing),
        ("raw_size_mismatch_count".to_string(), raw_size),
        ("selected_frame_mismatch_count".to_string(), selected),
        ("unselected_frame_mismatch_count".to_string(), unselected),
    ])
}

fn hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn hex_digest_finalize(digest: Sha256Context) -> String {
    digest.finish()
}

#[cfg(target_os = "macos")]
mod sha256_backend {
    use super::hex_bytes;
    use std::ffi::c_void;
    use std::mem::MaybeUninit;
    use std::os::raw::c_int;

    type CcLong = u32;

    #[repr(C)]
    struct CcSha256Ctx {
        count: [CcLong; 2],
        hash: [CcLong; 8],
        wbuf: [CcLong; 16],
    }

    #[link(name = "System")]
    unsafe extern "C" {
        fn CC_SHA256_Init(c: *mut CcSha256Ctx) -> c_int;
        fn CC_SHA256_Update(c: *mut CcSha256Ctx, data: *const c_void, len: CcLong) -> c_int;
        fn CC_SHA256_Final(md: *mut u8, c: *mut CcSha256Ctx) -> c_int;
    }

    pub struct Sha256Context {
        ctx: CcSha256Ctx,
    }

    impl Default for Sha256Context {
        fn default() -> Self {
            let mut ctx = MaybeUninit::<CcSha256Ctx>::uninit();
            let ok = unsafe { CC_SHA256_Init(ctx.as_mut_ptr()) };
            assert!(ok == 1, "CC_SHA256_Init failed");
            Self {
                ctx: unsafe { ctx.assume_init() },
            }
        }
    }

    impl Sha256Context {
        pub fn update(&mut self, mut data: &[u8]) {
            while !data.is_empty() {
                let take = data.len().min(CcLong::MAX as usize);
                let (chunk, rest) = data.split_at(take);
                let ok = unsafe {
                    CC_SHA256_Update(
                        &mut self.ctx,
                        chunk.as_ptr().cast::<c_void>(),
                        chunk.len() as CcLong,
                    )
                };
                assert!(ok == 1, "CC_SHA256_Update failed");
                data = rest;
            }
        }

        pub fn finish(mut self) -> String {
            let mut out = [0_u8; 32];
            let ok = unsafe { CC_SHA256_Final(out.as_mut_ptr(), &mut self.ctx) };
            assert!(ok == 1, "CC_SHA256_Final failed");
            hex_bytes(&out)
        }
    }

    pub fn hex_digest(data: &[u8]) -> String {
        let mut digest = Sha256Context::default();
        digest.update(data);
        digest.finish()
    }
}

#[cfg(not(target_os = "macos"))]
mod sha256_backend {
    use super::hex_bytes;
    use ring::digest::{Context, SHA256};

    #[derive(Default)]
    pub struct Sha256Context {
        context: Option<Context>,
    }

    impl Sha256Context {
        pub fn update(&mut self, data: &[u8]) {
            self.context
                .get_or_insert_with(|| Context::new(&SHA256))
                .update(data);
        }

        pub fn finish(self) -> String {
            match self.context {
                Some(context) => hex_bytes(context.finish().as_ref()),
                None => hex_digest(&[]),
            }
        }
    }

    pub fn hex_digest(data: &[u8]) -> String {
        hex_bytes(ring::digest::digest(&SHA256, data).as_ref())
    }
}

use sha256_backend::{hex_digest, Sha256Context};

#[cfg(test)]
mod tests {
    use super::{compare_raw_triplet, CompareInput};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "raw_locality_compare_{name}_{}_{}",
            std::process::id(),
            nanos
        ));
        fs::create_dir_all(&path).expect("mkdir");
        path
    }

    fn write_raw(path: &PathBuf, frames: &[&[u8]]) {
        let mut raw = Vec::new();
        for frame in frames {
            raw.extend_from_slice(frame);
        }
        fs::write(path, raw).expect("write raw");
    }

    #[test]
    fn accepts_selective_locality() {
        let root = temp_dir("accepts");
        let parent = root.join("parent.raw");
        let global = root.join("global.raw");
        let selective = root.join("selective.raw");
        write_raw(&parent, &[b"p0", b"p1", b"p2", b"p3", b"p4", b"p5"]);
        write_raw(&global, &[b"p0", b"p1", b"g2", b"g3", b"p4", b"p5"]);
        write_raw(&selective, &[b"p0", b"p1", b"g2", b"g3", b"p4", b"p5"]);

        let report = compare_raw_triplet(&CompareInput {
            parent_raw: parent,
            global_mutated_raw: global,
            selective_raw: selective,
            selected_frame_indices: vec![2, 3],
            frame_count: 6,
            frame_bytes: Some(2),
            rel_path: "0.raw".to_string(),
            sample_limit: 8,
        })
        .expect("compare");

        assert_eq!(report["blockers"], serde_json::json!([]));
        assert_eq!(
            report["mismatch_counts"]["selected_frame_mismatch_count"],
            0
        );
        assert_eq!(
            report["mismatch_counts"]["unselected_frame_mismatch_count"],
            0
        );
        assert_eq!(report["compare_engine"]["name"], "rust");
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn reports_selected_and_unselected_mismatches() {
        let root = temp_dir("mismatch");
        let parent = root.join("parent.raw");
        let global = root.join("global.raw");
        let selective = root.join("selective.raw");
        write_raw(&parent, &[b"p0", b"p1", b"p2", b"p3", b"p4", b"p5"]);
        write_raw(&global, &[b"p0", b"p1", b"p2", b"g3", b"p4", b"p5"]);
        write_raw(&selective, &[b"p0", b"p1", b"p2", b"xx", b"yy", b"p5"]);

        let report = compare_raw_triplet(&CompareInput {
            parent_raw: parent,
            global_mutated_raw: global,
            selective_raw: selective,
            selected_frame_indices: vec![3],
            frame_count: 6,
            frame_bytes: Some(2),
            rel_path: "0.raw".to_string(),
            sample_limit: 8,
        })
        .expect("compare");

        assert_eq!(
            report["mismatch_counts"]["selected_frame_mismatch_count"],
            1
        );
        assert_eq!(
            report["mismatch_counts"]["unselected_frame_mismatch_count"],
            1
        );
        assert_eq!(
            report["blockers"],
            serde_json::json!([
                "selected_frame_locality_mismatch:0.raw",
                "unselected_frame_parent_regression:0.raw"
            ])
        );
        fs::remove_dir_all(root).ok();
    }
}
