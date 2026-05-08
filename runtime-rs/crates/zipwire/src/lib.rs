use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::fs;
use std::io;
use std::path::Path;

const LOCAL_FILE_HEADER_SIG: u32 = 0x0403_4b50;
const CENTRAL_DIRECTORY_HEADER_SIG: u32 = 0x0201_4b50;
const END_OF_CENTRAL_DIRECTORY_SIG: u32 = 0x0605_4b50;
const ZIP_STORED: u16 = 0;
const ZIP_DEFLATED: u16 = 8;
const FLAG_ENCRYPTED: u16 = 1 << 0;
const FLAG_DATA_DESCRIPTOR: u16 = 1 << 3;
const FLAG_UTF8_NAME: u16 = 1 << 11;
const FORBIDDEN_ARCHIVE_MEMBER_NAMES: [&str; 4] =
    ["__MACOSX", ".DS_Store", "Thumbs.db", "desktop.ini"];

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ArchiveInspect {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
    pub member_count: usize,
    pub duplicate_member_names: Vec<String>,
    pub members: Vec<MemberInspect>,
    pub blockers: Vec<String>,
    pub zip_strict: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct MemberInspect {
    pub name: String,
    pub local_header_name: String,
    pub local_central_name_match: bool,
    pub header_offset: u64,
    pub payload_offset: Option<u64>,
    pub compress_type: u16,
    pub compressed_bytes: u64,
    pub uncompressed_bytes: u64,
    pub crc32: String,
    pub flag_bits: u16,
    pub blockers: Vec<String>,
    pub local_header: Option<LocalHeaderInspect>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct LocalHeaderInspect {
    pub flag_bits: u16,
    pub compress_type: u16,
    pub crc32: String,
    pub compressed_bytes: u64,
    pub uncompressed_bytes: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RewriteSingleProof {
    pub mode: String,
    pub mutation_requested: bool,
    pub byte_identical: bool,
    pub input: RewriteArchiveProof,
    pub output: RewriteArchiveProof,
    pub member: RewriteMemberProof,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RewriteArchiveProof {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RewriteMemberProof {
    pub name: String,
    pub compress_type: u16,
    pub header_offset: u64,
    pub payload_offset: u64,
    pub compressed_bytes: u64,
    pub uncompressed_bytes: u64,
    pub crc32: String,
    pub payload_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewriteSingleOutput {
    pub output_bytes: Vec<u8>,
    pub proof: RewriteSingleProof,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewriteSingleError {
    blockers: Vec<String>,
}

impl RewriteSingleError {
    pub fn blockers(&self) -> &[String] {
        &self.blockers
    }
}

impl fmt::Display for RewriteSingleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "rewrite-single rejected archive: ")?;
        for (index, blocker) in self.blockers.iter().enumerate() {
            if index != 0 {
                write!(f, "; ")?;
            }
            write!(f, "{blocker}")?;
        }
        Ok(())
    }
}

impl Error for RewriteSingleError {}

#[derive(Debug, Clone, Copy)]
struct Eocd {
    total_entries: u16,
    central_directory_size: u32,
    central_directory_offset: u32,
    comment_len: u16,
}

#[derive(Debug, Clone)]
struct CentralDirectoryEntry {
    flags: u16,
    compress_type: u16,
    crc32: u32,
    compressed_bytes: u32,
    uncompressed_bytes: u32,
    header_offset: u32,
    central_extra_bytes: u16,
    central_comment_bytes: u16,
    name: String,
}

/// Inspect a ZIP archive file and emit deterministic packet-compiler metadata.
pub fn inspect_zip_path(path: &Path) -> io::Result<ArchiveInspect> {
    let raw = fs::read(path)?;
    Ok(inspect_zip_bytes(&path.to_string_lossy(), &raw))
}

/// Inspect ZIP bytes without extracting or decompressing any member payloads.
///
/// This is a native golden-vector surface for the Python submission packet
/// compiler. It intentionally fails closed through blockers when the archive
/// uses unsupported features, unsafe names, or central/local name divergence.
pub fn inspect_zip_bytes(path: &str, raw_zip: &[u8]) -> ArchiveInspect {
    let mut blockers = Vec::new();
    let mut members = Vec::new();
    let mut duplicate_names = BTreeSet::new();

    match parse_central_directory(raw_zip) {
        Ok(entries) => {
            let mut seen = BTreeSet::new();
            for entry in entries {
                if !seen.insert(entry.name.clone()) {
                    duplicate_names.insert(entry.name.clone());
                }
                let member = inspect_member(raw_zip, entry);
                blockers.extend(
                    member
                        .blockers
                        .iter()
                        .map(|blocker| format!("{}:{blocker}", member.name)),
                );
                members.push(member);
            }
        }
        Err(err) => blockers.push(format!("bad_zip:{err}")),
    }

    for name in &duplicate_names {
        blockers.push(format!("duplicate_archive_member:{name}"));
    }
    blockers.sort();
    blockers.dedup();

    ArchiveInspect {
        path: path.to_string(),
        bytes: raw_zip.len() as u64,
        sha256: sha256_hex(raw_zip),
        member_count: members.len(),
        duplicate_member_names: duplicate_names.into_iter().collect(),
        members,
        zip_strict: blockers.is_empty(),
        blockers,
    }
}

/// Rewrite a simple stored single-member ZIP archive in identity mode.
///
/// This intentionally narrow native surface validates strict central/local
/// metadata, rejects unsupported ZIP features, and returns byte-identical
/// output bytes plus a deterministic proof. It does not decompress payloads and
/// does not support mutation.
pub fn rewrite_single_identity_bytes(
    input_path: &str,
    raw_zip: &[u8],
    output_path: &str,
) -> Result<RewriteSingleOutput, RewriteSingleError> {
    let member = validate_rewrite_single_identity(input_path, raw_zip)?;
    let archive_sha256 = sha256_hex(raw_zip);
    let archive_bytes = raw_zip.len() as u64;
    Ok(RewriteSingleOutput {
        output_bytes: raw_zip.to_vec(),
        proof: RewriteSingleProof {
            mode: "identity".to_string(),
            mutation_requested: false,
            byte_identical: true,
            input: RewriteArchiveProof {
                path: input_path.to_string(),
                bytes: archive_bytes,
                sha256: archive_sha256.clone(),
            },
            output: RewriteArchiveProof {
                path: output_path.to_string(),
                bytes: archive_bytes,
                sha256: archive_sha256,
            },
            member,
        },
    })
}

/// Rewrite a simple stored single-member ZIP archive to a path in identity mode.
///
/// Invalid archives are rejected before the output path is written.
pub fn rewrite_single_identity_path(
    input_path: &Path,
    output_path: &Path,
) -> io::Result<RewriteSingleProof> {
    let raw = fs::read(input_path)?;
    let rewrite = rewrite_single_identity_bytes(
        &input_path.to_string_lossy(),
        &raw,
        &output_path.to_string_lossy(),
    )
    .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    fs::write(output_path, &rewrite.output_bytes)?;
    Ok(rewrite.proof)
}

fn validate_rewrite_single_identity(
    input_path: &str,
    raw_zip: &[u8],
) -> Result<RewriteMemberProof, RewriteSingleError> {
    let inspect = inspect_zip_bytes(input_path, raw_zip);
    let mut blockers = inspect.blockers.clone();

    if inspect.member_count != 1 {
        blockers.push(format!(
            "rewrite_single_requires_exactly_one_member:{}",
            inspect.member_count
        ));
    }

    let eocd_offset = match find_eocd(raw_zip) {
        Ok(offset) => Some(offset),
        Err(err) => {
            blockers.push(format!("bad_zip:{err}"));
            None
        }
    };
    let eocd = match eocd_offset {
        Some(offset) => match parse_eocd(raw_zip, offset) {
            Ok(eocd) => Some(eocd),
            Err(err) => {
                blockers.push(format!("bad_zip:{err}"));
                None
            }
        },
        None => None,
    };
    let entries = match parse_central_directory(raw_zip) {
        Ok(entries) => entries,
        Err(err) => {
            blockers.push(format!("bad_zip:{err}"));
            Vec::new()
        }
    };

    let mut proof = None;
    if let (Some(eocd), [entry]) = (eocd, entries.as_slice()) {
        if eocd.comment_len != 0 {
            blockers.push(format!(
                "rewrite_single_eocd_comment_not_supported:{}",
                eocd.comment_len
            ));
        }
        if entry.header_offset != 0 {
            blockers.push(format!(
                "rewrite_single_requires_local_header_at_start:{}",
                entry.header_offset
            ));
        }
        if entry.central_extra_bytes != 0 {
            blockers.push(format!(
                "rewrite_single_central_extra_not_supported:{}",
                entry.central_extra_bytes
            ));
        }
        if entry.central_comment_bytes != 0 {
            blockers.push(format!(
                "rewrite_single_central_comment_not_supported:{}",
                entry.central_comment_bytes
            ));
        }
        if entry.flags & !(FLAG_UTF8_NAME | FLAG_ENCRYPTED) != 0 {
            blockers.push(format!(
                "rewrite_single_unsupported_flag_bits:0x{:04x}",
                entry.flags & !(FLAG_UTF8_NAME | FLAG_ENCRYPTED)
            ));
        }
        if entry.compress_type != ZIP_STORED {
            blockers.push(format!(
                "rewrite_single_requires_stored_method:{}",
                entry.compress_type
            ));
        }
        if entry.compressed_bytes != entry.uncompressed_bytes {
            blockers.push(format!(
                "rewrite_single_stored_size_mismatch:{}!={}",
                entry.compressed_bytes, entry.uncompressed_bytes
            ));
        }

        match parse_local_header(raw_zip, entry.header_offset as usize) {
            Ok(local) => {
                if local.extra_bytes != 0 {
                    blockers.push(format!(
                        "rewrite_single_local_extra_not_supported:{}",
                        local.extra_bytes
                    ));
                }
                let payload_end = local
                    .data_offset
                    .saturating_add(entry.compressed_bytes as usize);
                let central_directory_offset = eocd.central_directory_offset as usize;
                let central_directory_end =
                    central_directory_offset.saturating_add(eocd.central_directory_size as usize);
                if payload_end > raw_zip.len() {
                    blockers.push(format!(
                        "rewrite_single_payload_out_of_range:{}>{}",
                        payload_end,
                        raw_zip.len()
                    ));
                }
                if payload_end != central_directory_offset {
                    blockers.push(format!(
                        "rewrite_single_requires_central_directory_after_payload:{}!={}",
                        payload_end, central_directory_offset
                    ));
                }
                if let Some(offset) = eocd_offset {
                    if central_directory_end != offset {
                        blockers.push(format!(
                            "rewrite_single_central_directory_eocd_gap:{}!={}",
                            central_directory_end, offset
                        ));
                    }
                }

                if payload_end <= raw_zip.len() {
                    proof = Some(RewriteMemberProof {
                        name: entry.name.clone(),
                        compress_type: entry.compress_type,
                        header_offset: entry.header_offset as u64,
                        payload_offset: local.data_offset as u64,
                        compressed_bytes: entry.compressed_bytes as u64,
                        uncompressed_bytes: entry.uncompressed_bytes as u64,
                        crc32: format!("{:08x}", entry.crc32),
                        payload_sha256: sha256_hex(&raw_zip[local.data_offset..payload_end]),
                    });
                }
            }
            Err(err) => blockers.push(format!("local_header_error:{err}")),
        }
    }

    blockers.sort();
    blockers.dedup();
    if blockers.is_empty() {
        Ok(proof.expect("single-member proof produced when blockers are empty"))
    } else {
        Err(RewriteSingleError { blockers })
    }
}

fn inspect_member(raw_zip: &[u8], entry: CentralDirectoryEntry) -> MemberInspect {
    let mut row_blockers = member_safety_blockers(&entry.name);
    if entry.name.ends_with('/') {
        row_blockers.push("directory_member_not_supported_for_contest_packet".to_string());
    }
    if entry.flags & FLAG_ENCRYPTED != 0 {
        row_blockers.push("encrypted_member".to_string());
    }
    if entry.flags & FLAG_DATA_DESCRIPTOR != 0 {
        row_blockers.push("data_descriptor_member_not_supported".to_string());
    }
    if !matches!(entry.compress_type, ZIP_STORED | ZIP_DEFLATED) {
        row_blockers.push(format!("unsupported_zip_method:{}", entry.compress_type));
    }

    let mut local_header = None;
    let mut local_header_name = String::new();
    let mut payload_offset = None;
    match parse_local_header(raw_zip, entry.header_offset as usize) {
        Ok(header) => {
            local_header_name = header.name.clone();
            payload_offset = Some(header.data_offset as u64);
            let local_summary = LocalHeaderInspect {
                flag_bits: header.flags,
                compress_type: header.compress_type,
                crc32: format!("{:08x}", header.crc32),
                compressed_bytes: header.compressed_bytes as u64,
                uncompressed_bytes: header.uncompressed_bytes as u64,
            };
            if header.flags != entry.flags {
                row_blockers.push(format!(
                    "local_central_flag_bits_mismatch:{}!={}",
                    header.flags, entry.flags
                ));
            }
            if header.compress_type != entry.compress_type {
                row_blockers.push(format!(
                    "local_central_compress_type_mismatch:{}!={}",
                    header.compress_type, entry.compress_type
                ));
            }
            if header.crc32 != entry.crc32 {
                row_blockers.push(format!(
                    "local_central_crc32_mismatch:{:08x}!={:08x}",
                    header.crc32, entry.crc32
                ));
            }
            if header.compressed_bytes != entry.compressed_bytes {
                row_blockers.push(format!(
                    "local_central_compressed_size_mismatch:{}!={}",
                    header.compressed_bytes, entry.compressed_bytes
                ));
            }
            if header.uncompressed_bytes != entry.uncompressed_bytes {
                row_blockers.push(format!(
                    "local_central_uncompressed_size_mismatch:{}!={}",
                    header.uncompressed_bytes, entry.uncompressed_bytes
                ));
            }
            local_header = Some(local_summary);
        }
        Err(err) => row_blockers.push(format!("local_header_error:{err}")),
    }
    let local_central_name_match = !local_header_name.is_empty() && local_header_name == entry.name;
    if !local_header_name.is_empty() && !local_central_name_match {
        row_blockers.push("local_central_name_mismatch".to_string());
    }
    row_blockers.sort();
    row_blockers.dedup();

    MemberInspect {
        name: entry.name,
        local_header_name,
        local_central_name_match,
        header_offset: entry.header_offset as u64,
        payload_offset,
        compress_type: entry.compress_type,
        compressed_bytes: entry.compressed_bytes as u64,
        uncompressed_bytes: entry.uncompressed_bytes as u64,
        crc32: format!("{:08x}", entry.crc32),
        flag_bits: entry.flags,
        blockers: row_blockers,
        local_header,
    }
}

fn parse_central_directory(raw_zip: &[u8]) -> Result<Vec<CentralDirectoryEntry>, String> {
    let eocd_offset = find_eocd(raw_zip)?;
    let eocd = parse_eocd(raw_zip, eocd_offset)?;
    let cd_start = eocd.central_directory_offset as usize;
    let cd_size = eocd.central_directory_size as usize;
    let cd_end = cd_start
        .checked_add(cd_size)
        .ok_or_else(|| "central directory size overflow".to_string())?;
    if cd_end > raw_zip.len() {
        return Err(format!(
            "central directory out of range: offset {cd_start}, size {cd_size}"
        ));
    }

    let mut cursor = cd_start;
    let mut entries = Vec::with_capacity(eocd.total_entries as usize);
    for _ in 0..eocd.total_entries {
        if cursor + 46 > raw_zip.len() {
            return Err(format!(
                "central directory header truncated at offset {cursor}"
            ));
        }
        let sig = read_u32(raw_zip, cursor);
        if sig != CENTRAL_DIRECTORY_HEADER_SIG {
            return Err(format!(
                "bad central directory signature at offset {cursor}"
            ));
        }
        let version_made_by = read_u16(raw_zip, cursor + 4);
        let version_needed = read_u16(raw_zip, cursor + 6);
        let flags = read_u16(raw_zip, cursor + 8);
        let compress_type = read_u16(raw_zip, cursor + 10);
        let crc32 = read_u32(raw_zip, cursor + 16);
        let compressed_bytes = read_u32(raw_zip, cursor + 20);
        let uncompressed_bytes = read_u32(raw_zip, cursor + 24);
        let name_len = read_u16(raw_zip, cursor + 28) as usize;
        let extra_len = read_u16(raw_zip, cursor + 30) as usize;
        let comment_len = read_u16(raw_zip, cursor + 32) as usize;
        let disk_start = read_u16(raw_zip, cursor + 34);
        let header_offset = read_u32(raw_zip, cursor + 42);
        if uses_zip64_or_multidisk(
            version_made_by,
            version_needed,
            compressed_bytes,
            uncompressed_bytes,
            header_offset,
            disk_start,
        ) {
            return Err("zip64 or multidisk archive is not supported".to_string());
        }

        let name_start = cursor + 46;
        let name_end = name_start
            .checked_add(name_len)
            .ok_or_else(|| "central directory name length overflow".to_string())?;
        let entry_end = name_end
            .checked_add(extra_len)
            .and_then(|offset| offset.checked_add(comment_len))
            .ok_or_else(|| "central directory entry length overflow".to_string())?;
        if entry_end > raw_zip.len() {
            return Err(format!(
                "central directory entry truncated at offset {cursor}"
            ));
        }
        let name = decode_zip_name(&raw_zip[name_start..name_end], flags).map_err(|err| {
            format!("central directory name decode error at offset {cursor}: {err}")
        })?;

        entries.push(CentralDirectoryEntry {
            flags,
            compress_type,
            crc32,
            compressed_bytes,
            uncompressed_bytes,
            header_offset,
            central_extra_bytes: extra_len as u16,
            central_comment_bytes: comment_len as u16,
            name,
        });
        cursor = entry_end;
    }
    if cursor != cd_end {
        return Err(format!(
            "central directory length mismatch: parsed end {cursor}, expected {cd_end}"
        ));
    }
    Ok(entries)
}

fn find_eocd(raw_zip: &[u8]) -> Result<usize, String> {
    if raw_zip.len() < 22 {
        return Err("end of central directory missing".to_string());
    }
    let min = raw_zip.len().saturating_sub(22 + u16::MAX as usize);
    for offset in (min..=raw_zip.len() - 22).rev() {
        if read_u32(raw_zip, offset) != END_OF_CENTRAL_DIRECTORY_SIG {
            continue;
        }
        let comment_len = read_u16(raw_zip, offset + 20) as usize;
        if offset + 22 + comment_len == raw_zip.len() {
            return Ok(offset);
        }
    }
    Err("end of central directory missing".to_string())
}

fn parse_eocd(raw_zip: &[u8], offset: usize) -> Result<Eocd, String> {
    let disk_number = read_u16(raw_zip, offset + 4);
    let cd_disk_number = read_u16(raw_zip, offset + 6);
    let disk_entries = read_u16(raw_zip, offset + 8);
    let total_entries = read_u16(raw_zip, offset + 10);
    let central_directory_size = read_u32(raw_zip, offset + 12);
    let central_directory_offset = read_u32(raw_zip, offset + 16);
    if disk_number != 0 || cd_disk_number != 0 || disk_entries != total_entries {
        return Err("multidisk archive is not supported".to_string());
    }
    if total_entries == u16::MAX
        || central_directory_size == u32::MAX
        || central_directory_offset == u32::MAX
    {
        return Err("zip64 archive is not supported".to_string());
    }
    Ok(Eocd {
        total_entries,
        central_directory_size,
        central_directory_offset,
        comment_len: read_u16(raw_zip, offset + 20),
    })
}

#[derive(Debug, Clone)]
struct LocalHeader {
    flags: u16,
    compress_type: u16,
    crc32: u32,
    compressed_bytes: u32,
    uncompressed_bytes: u32,
    extra_bytes: u16,
    data_offset: usize,
    name: String,
}

fn parse_local_header(raw_zip: &[u8], offset: usize) -> Result<LocalHeader, String> {
    if offset + 30 > raw_zip.len() {
        return Err(format!("local header offset out of range: {offset}"));
    }
    let sig = read_u32(raw_zip, offset);
    if sig != LOCAL_FILE_HEADER_SIG {
        return Err(format!("bad local header signature at offset {offset}"));
    }
    let flags = read_u16(raw_zip, offset + 6);
    let compress_type = read_u16(raw_zip, offset + 8);
    let crc32 = read_u32(raw_zip, offset + 14);
    let compressed_bytes = read_u32(raw_zip, offset + 18);
    let uncompressed_bytes = read_u32(raw_zip, offset + 22);
    let name_len = read_u16(raw_zip, offset + 26) as usize;
    let extra_len = read_u16(raw_zip, offset + 28) as usize;
    let start = offset + 30;
    let end = start
        .checked_add(name_len)
        .ok_or_else(|| format!("local header name length overflow at offset {offset}"))?;
    let after_extra = end
        .checked_add(extra_len)
        .ok_or_else(|| format!("local header extra length overflow at offset {offset}"))?;
    if end > raw_zip.len() || after_extra > raw_zip.len() {
        return Err(format!(
            "local header name/extra out of range at offset {offset}"
        ));
    }
    let name = decode_zip_name(&raw_zip[start..end], flags)
        .map_err(|err| format!("local header name decode error at offset {offset}: {err}"))?;
    Ok(LocalHeader {
        flags,
        compress_type,
        crc32,
        compressed_bytes,
        uncompressed_bytes,
        extra_bytes: extra_len as u16,
        data_offset: after_extra,
        name,
    })
}

fn uses_zip64_or_multidisk(
    version_made_by: u16,
    version_needed: u16,
    compressed_bytes: u32,
    uncompressed_bytes: u32,
    header_offset: u32,
    disk_start: u16,
) -> bool {
    disk_start != 0
        || (version_made_by & 0x00ff) >= 45
        || version_needed >= 45
        || compressed_bytes == u32::MAX
        || uncompressed_bytes == u32::MAX
        || header_offset == u32::MAX
}

fn decode_zip_name(raw: &[u8], flags: u16) -> Result<String, String> {
    if flags & FLAG_UTF8_NAME != 0 {
        return std::str::from_utf8(raw)
            .map(str::to_owned)
            .map_err(|err| format!("invalid utf-8 member name: {err}"));
    }
    if raw.iter().all(u8::is_ascii) {
        return String::from_utf8(raw.to_vec())
            .map_err(|err| format!("invalid ascii member name: {err}"));
    }
    Err("cp437 non-ascii member names are not supported".to_string())
}

fn member_safety_blockers(name: &str) -> Vec<String> {
    match validate_archive_member_name(name) {
        Ok(()) => Vec::new(),
        Err(err) => vec![format!("unsafe_member_name:{err}")],
    }
}

fn validate_archive_member_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("archive member name is empty".to_string());
    }
    if name.contains('\0') {
        return Err(format!("archive member contains NUL byte: {name:?}"));
    }
    if name.contains('\\') {
        return Err(format!("archive member uses backslashes: {name:?}"));
    }
    if name.starts_with('/') {
        return Err(format!("archive member path is absolute: {name:?}"));
    }

    let normalized = name.strip_suffix('/').unwrap_or(name);
    let parts = normalized.split('/').collect::<Vec<_>>();
    if parts.is_empty()
        || parts
            .iter()
            .any(|part| part.is_empty() || *part == "." || *part == "..")
    {
        return Err(format!("zip-slip archive member path: {name:?}"));
    }
    if parts
        .iter()
        .any(|part| FORBIDDEN_ARCHIVE_MEMBER_NAMES.contains(part) || part.starts_with('.'))
    {
        return Err(format!("hidden/system archive member: {name:?}"));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(
        bytes[offset..offset + 2]
            .try_into()
            .expect("checked u16 read"),
    )
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("checked u32 read"),
    )
}
