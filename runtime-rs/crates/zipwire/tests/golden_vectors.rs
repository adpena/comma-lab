use std::fs;
use std::process::Command;
use zipwire::{inspect_zip_bytes, rewrite_single_identity_bytes, rewrite_single_identity_path};

const STORED_SINGLE_MEMBER_ZIP: &[u8] = &[
    0x50, 0x4b, 0x03, 0x04, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x21, 0x00, 0x74, 0xb6,
    0x28, 0x64, 0x0d, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x78, 0x70,
    0x61, 0x79, 0x6c, 0x6f, 0x61, 0x64, 0x2d, 0x62, 0x79, 0x74, 0x65, 0x73, 0x50, 0x4b, 0x01, 0x02,
    0x14, 0x03, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x21, 0x00, 0x74, 0xb6, 0x28, 0x64,
    0x0d, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0xa4, 0x81, 0x00, 0x00, 0x00, 0x00, 0x78, 0x50, 0x4b, 0x05, 0x06, 0x00,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x2f, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x00, 0x00, 0x00,
    0x00,
];

const DEFLATED_SINGLE_MEMBER_ZIP: &[u8] = &[
    0x50, 0x4b, 0x03, 0x04, 0x14, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x21, 0x00, 0x74, 0xb6,
    0x28, 0x64, 0x0f, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x78, 0x2b,
    0x48, 0xac, 0xcc, 0xc9, 0x4f, 0x4c, 0xd1, 0x4d, 0xaa, 0x2c, 0x49, 0x2d, 0x06, 0x00, 0x50, 0x4b,
    0x01, 0x02, 0x14, 0x03, 0x14, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x21, 0x00, 0x74, 0xb6,
    0x28, 0x64, 0x0f, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xa4, 0x81, 0x00, 0x00, 0x00, 0x00, 0x78, 0x50, 0x4b, 0x05,
    0x06, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x2f, 0x00, 0x00, 0x00, 0x2e, 0x00, 0x00,
    0x00, 0x00, 0x00,
];

#[test]
fn inspects_stored_single_member_python_oracle_core_fields() {
    let inspect = inspect_zip_bytes("fixture/stored.zip", STORED_SINGLE_MEMBER_ZIP);

    assert!(inspect.zip_strict);
    assert_eq!(inspect.bytes, 113);
    assert_eq!(
        inspect.sha256,
        "7cae837c71aa1abbc55b52dcdb51487a847725bb97cb507d5761ac23c344bf86"
    );
    assert_eq!(inspect.member_count, 1);
    assert!(inspect.duplicate_member_names.is_empty());
    assert!(inspect.blockers.is_empty());

    let member = &inspect.members[0];
    assert_eq!(member.name, "x");
    assert_eq!(member.local_header_name, "x");
    assert!(member.local_central_name_match);
    assert_eq!(member.header_offset, 0);
    assert_eq!(member.payload_offset, Some(31));
    assert_eq!(member.compress_type, 0);
    assert_eq!(member.crc32, "6428b674");
    assert_eq!(member.compressed_bytes, 13);
    assert_eq!(member.uncompressed_bytes, 13);
    let local = member.local_header.as_ref().expect("local header");
    assert_eq!(local.flag_bits, 0);
    assert_eq!(local.compress_type, 0);
    assert_eq!(local.crc32, "6428b674");
    assert_eq!(local.compressed_bytes, 13);
    assert_eq!(local.uncompressed_bytes, 13);
    assert!(member.blockers.is_empty());
}

#[test]
fn accepts_deflated_metadata_without_payload_inflate() {
    let inspect = inspect_zip_bytes("fixture/deflated.zip", DEFLATED_SINGLE_MEMBER_ZIP);

    assert!(inspect.zip_strict);
    assert_eq!(inspect.bytes, 115);
    assert_eq!(
        inspect.sha256,
        "daab5ca1ab1d25158079e5e9d1f7d00b2b3d912ce96fdf55cd5b9320b909d810"
    );
    let member = &inspect.members[0];
    assert_eq!(member.compress_type, 8);
    assert_eq!(member.payload_offset, Some(31));
    assert_eq!(member.compressed_bytes, 15);
    assert_eq!(member.uncompressed_bytes, 13);
    assert_eq!(member.crc32, "6428b674");
    assert_eq!(
        member.local_header.as_ref().expect("local").compress_type,
        8
    );
    assert!(member.blockers.is_empty());
}

#[test]
fn rewrite_single_identity_is_byte_identical_and_proven() {
    let rewrite = rewrite_single_identity_bytes(
        "fixture/stored.zip",
        STORED_SINGLE_MEMBER_ZIP,
        "fixture/out.zip",
    )
    .expect("rewrite-single identity");

    assert_eq!(rewrite.output_bytes, STORED_SINGLE_MEMBER_ZIP);
    assert_eq!(rewrite.proof.mode, "identity");
    assert!(!rewrite.proof.mutation_requested);
    assert!(rewrite.proof.byte_identical);
    assert_eq!(rewrite.proof.input.path, "fixture/stored.zip");
    assert_eq!(rewrite.proof.output.path, "fixture/out.zip");
    assert_eq!(rewrite.proof.input.bytes, 113);
    assert_eq!(rewrite.proof.output.bytes, 113);
    assert_eq!(
        rewrite.proof.input.sha256,
        "7cae837c71aa1abbc55b52dcdb51487a847725bb97cb507d5761ac23c344bf86"
    );
    assert_eq!(rewrite.proof.output.sha256, rewrite.proof.input.sha256);
    assert_eq!(rewrite.proof.member.name, "x");
    assert_eq!(rewrite.proof.member.compress_type, 0);
    assert_eq!(rewrite.proof.member.header_offset, 0);
    assert_eq!(rewrite.proof.member.payload_offset, 31);
    assert_eq!(rewrite.proof.member.compressed_bytes, 13);
    assert_eq!(rewrite.proof.member.uncompressed_bytes, 13);
    assert_eq!(rewrite.proof.member.crc32, "6428b674");
    assert_eq!(
        rewrite.proof.member.payload_sha256,
        "808b59664b6adb9274e3bbd0766e7aec9659786c22fdb825c49ca7fda1c6236e"
    );
}

#[test]
fn rewrite_single_identity_is_deterministic_across_repeated_runs() {
    let first = rewrite_single_identity_bytes(
        "fixture/stored.zip",
        STORED_SINGLE_MEMBER_ZIP,
        "fixture/out.zip",
    )
    .expect("first rewrite");
    let second = rewrite_single_identity_bytes(
        "fixture/stored.zip",
        STORED_SINGLE_MEMBER_ZIP,
        "fixture/out.zip",
    )
    .expect("second rewrite");

    assert_eq!(first.output_bytes, second.output_bytes);
    assert_eq!(first.proof, second.proof);
}

#[test]
fn rewrite_single_fails_closed_on_multi_member_archives() {
    let raw = multi_stored_members_fixture();

    let err = rewrite_single_identity_bytes("fixture/multi.zip", &raw, "fixture/out.zip")
        .expect_err("multi-member rewrite must fail");

    assert_eq!(
        err.blockers(),
        ["rewrite_single_requires_exactly_one_member:2"]
    );
}

#[test]
fn rewrite_single_fails_closed_on_unsupported_methods() {
    let err = rewrite_single_identity_bytes(
        "fixture/deflated.zip",
        DEFLATED_SINGLE_MEMBER_ZIP,
        "fixture/out.zip",
    )
    .expect_err("deflated rewrite must fail");

    assert_eq!(
        err.blockers(),
        [
            "rewrite_single_requires_stored_method:8",
            "rewrite_single_stored_size_mismatch:15!=13"
        ]
    );
}

#[test]
fn rewrite_single_fails_closed_on_encrypted_members() {
    let mut raw = STORED_SINGLE_MEMBER_ZIP.to_vec();
    raw[6] = 1;
    raw[52] = 1;

    let err = rewrite_single_identity_bytes("fixture/encrypted.zip", &raw, "fixture/out.zip")
        .expect_err("encrypted rewrite must fail");

    assert_eq!(err.blockers(), ["x:encrypted_member"]);
}

#[test]
fn rewrite_single_path_rejection_does_not_write_output() {
    let dir = std::env::temp_dir().join(format!("zipwire-rewrite-reject-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("test dir");
    let archive = dir.join("input.zip");
    let rewritten = dir.join("output.zip");
    fs::write(&archive, DEFLATED_SINGLE_MEMBER_ZIP).expect("archive write");

    let err = rewrite_single_identity_path(&archive, &rewritten)
        .expect_err("deflated path rewrite must fail");

    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(!rewritten.exists());

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn cli_rewrite_single_writes_identity_output_and_json_proof() {
    let dir = std::env::temp_dir().join(format!("zipwire-rewrite-cli-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("test dir");
    let archive = dir.join("input.zip");
    let rewritten = dir.join("output.zip");
    fs::write(&archive, STORED_SINGLE_MEMBER_ZIP).expect("archive write");

    let output = Command::new(env!("CARGO_BIN_EXE_zipwire"))
        .arg("rewrite-single")
        .arg(&archive)
        .arg(&rewritten)
        .output()
        .expect("run zipwire rewrite-single");

    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        fs::read(&rewritten).expect("rewritten archive"),
        STORED_SINGLE_MEMBER_ZIP
    );
    let proof: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("rewrite proof JSON");
    assert_eq!(proof["mode"], "identity");
    assert_eq!(proof["mutation_requested"], false);
    assert_eq!(proof["byte_identical"], true);
    assert_eq!(proof["input"]["bytes"], 113);
    assert_eq!(proof["output"]["bytes"], 113);
    assert_eq!(proof["member"]["name"], "x");
    assert_eq!(
        proof["member"]["payload_sha256"],
        "808b59664b6adb9274e3bbd0766e7aec9659786c22fdb825c49ca7fda1c6236e"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn mismatched_local_and_central_names_block_strict_zip() {
    let mut raw = STORED_SINGLE_MEMBER_ZIP.to_vec();
    raw[30] = b'y';

    let inspect = inspect_zip_bytes("fixture/mismatch.zip", &raw);

    assert!(!inspect.zip_strict);
    assert_eq!(inspect.members[0].name, "x");
    assert_eq!(inspect.members[0].local_header_name, "y");
    assert_eq!(inspect.members[0].blockers, ["local_central_name_mismatch"]);
    assert_eq!(inspect.blockers, ["x:local_central_name_mismatch"]);
}

#[test]
fn local_central_crc_and_size_mismatches_block_strict_zip() {
    let mut raw = STORED_SINGLE_MEMBER_ZIP.to_vec();
    // Corrupt local-header CRC, compressed size, and uncompressed size only.
    raw[14] ^= 0x01;
    raw[18] = 14;
    raw[22] = 12;

    let inspect = inspect_zip_bytes("fixture/local-central-metadata.zip", &raw);

    assert!(!inspect.zip_strict);
    assert_eq!(
        inspect.members[0].blockers,
        [
            "local_central_compressed_size_mismatch:14!=13",
            "local_central_crc32_mismatch:6428b675!=6428b674",
            "local_central_uncompressed_size_mismatch:12!=13"
        ]
    );
    assert_eq!(
        inspect.blockers,
        [
            "x:local_central_compressed_size_mismatch:14!=13",
            "x:local_central_crc32_mismatch:6428b675!=6428b674",
            "x:local_central_uncompressed_size_mismatch:12!=13"
        ]
    );
}

#[test]
fn duplicate_names_are_reported_as_top_level_blockers() {
    let raw = duplicate_stored_members_fixture();

    let inspect = inspect_zip_bytes("fixture/duplicate.zip", &raw);

    assert!(!inspect.zip_strict);
    assert_eq!(inspect.member_count, 2);
    assert_eq!(inspect.duplicate_member_names, ["x"]);
    assert_eq!(inspect.blockers, ["duplicate_archive_member:x"]);
}

#[test]
fn unsafe_and_unsupported_members_fail_closed() {
    let mut raw = STORED_SINGLE_MEMBER_ZIP.to_vec();
    raw[30] = b'.';
    raw[90] = b'.';
    raw[54] = 99;

    let inspect = inspect_zip_bytes("fixture/unsafe.zip", &raw);

    assert!(!inspect.zip_strict);
    let member = &inspect.members[0];
    assert_eq!(member.name, ".");
    assert_eq!(member.local_header_name, ".");
    assert_eq!(
        member.blockers,
        [
            "local_central_compress_type_mismatch:0!=99",
            "unsafe_member_name:zip-slip archive member path: \".\"",
            "unsupported_zip_method:99"
        ]
    );
    assert_eq!(
        inspect.blockers,
        [
            ".:local_central_compress_type_mismatch:0!=99",
            ".:unsafe_member_name:zip-slip archive member path: \".\"",
            ".:unsupported_zip_method:99"
        ]
    );
}

#[test]
fn encrypted_members_fail_closed() {
    let mut raw = STORED_SINGLE_MEMBER_ZIP.to_vec();
    raw[6] = 1;
    raw[52] = 1;

    let inspect = inspect_zip_bytes("fixture/encrypted.zip", &raw);

    assert!(!inspect.zip_strict);
    assert_eq!(inspect.members[0].flag_bits, 1);
    assert_eq!(inspect.members[0].blockers, ["encrypted_member"]);
    assert_eq!(inspect.blockers, ["x:encrypted_member"]);
}

#[test]
fn data_descriptor_members_fail_closed() {
    let mut raw = STORED_SINGLE_MEMBER_ZIP.to_vec();
    raw[6] |= 0x08;
    raw[52] |= 0x08;

    let inspect = inspect_zip_bytes("fixture/descriptor.zip", &raw);

    assert!(!inspect.zip_strict);
    assert_eq!(inspect.members[0].flag_bits, 0x08);
    assert_eq!(
        inspect.members[0].blockers,
        ["data_descriptor_member_not_supported"]
    );
    assert_eq!(inspect.blockers, ["x:data_descriptor_member_not_supported"]);
}

#[test]
fn cli_emits_json_and_exits_nonzero_when_blocked() {
    let dir = std::env::temp_dir().join(format!("zipwire-cli-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("test dir");
    let archive = dir.join("blocked.zip");
    let mut raw = STORED_SINGLE_MEMBER_ZIP.to_vec();
    raw[30] = b'y';
    fs::write(&archive, raw).expect("archive write");

    let output = Command::new(env!("CARGO_BIN_EXE_zipwire"))
        .arg(&archive)
        .output()
        .expect("run zipwire");

    assert!(!output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("\"local_central_name_mismatch\""));

    let _ = fs::remove_dir_all(&dir);
}

fn duplicate_stored_members_fixture() -> Vec<u8> {
    let mut out = Vec::new();
    local_stored_file(&mut out, b"x", b"a", 0xe8b7_be43);
    local_stored_file(&mut out, b"x", b"b", 0x71be_eff9);
    let central_offset = out.len() as u32;
    central_stored_file(&mut out, b"x", 0xe8b7_be43, 1, 0);
    central_stored_file(&mut out, b"x", 0x71be_eff9, 1, 32);
    let central_size = out.len() as u32 - central_offset;
    end_of_central_directory(&mut out, 2, central_size, central_offset);
    out
}

fn multi_stored_members_fixture() -> Vec<u8> {
    let mut out = Vec::new();
    local_stored_file(&mut out, b"x", b"a", 0xe8b7_be43);
    local_stored_file(&mut out, b"y", b"b", 0x71be_eff9);
    let central_offset = out.len() as u32;
    central_stored_file(&mut out, b"x", 0xe8b7_be43, 1, 0);
    central_stored_file(&mut out, b"y", 0x71be_eff9, 1, 32);
    let central_size = out.len() as u32 - central_offset;
    end_of_central_directory(&mut out, 2, central_size, central_offset);
    out
}

fn local_stored_file(out: &mut Vec<u8>, name: &[u8], payload: &[u8], crc32: u32) {
    out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
    out.extend_from_slice(&20u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0x0021u16.to_le_bytes());
    out.extend_from_slice(&crc32.to_le_bytes());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&(name.len() as u16).to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(name);
    out.extend_from_slice(payload);
}

fn central_stored_file(out: &mut Vec<u8>, name: &[u8], crc32: u32, size: u32, header_offset: u32) {
    out.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
    out.extend_from_slice(&0x0314u16.to_le_bytes());
    out.extend_from_slice(&20u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0x0021u16.to_le_bytes());
    out.extend_from_slice(&crc32.to_le_bytes());
    out.extend_from_slice(&size.to_le_bytes());
    out.extend_from_slice(&size.to_le_bytes());
    out.extend_from_slice(&(name.len() as u16).to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(0o100644u32 << 16).to_le_bytes());
    out.extend_from_slice(&header_offset.to_le_bytes());
    out.extend_from_slice(name);
}

fn end_of_central_directory(
    out: &mut Vec<u8>,
    entries: u16,
    central_size: u32,
    central_offset: u32,
) {
    out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&entries.to_le_bytes());
    out.extend_from_slice(&entries.to_le_bytes());
    out.extend_from_slice(&central_size.to_le_bytes());
    out.extend_from_slice(&central_offset.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
}
