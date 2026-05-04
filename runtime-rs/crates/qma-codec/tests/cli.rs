use std::fs;
use std::process::Command;

const PYTHON_QMA9_FIXTURE: &[u8] = &[
    0x51, 0x4d, 0x41, 0x39, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
    0x11, 0x00, 0x00, 0x00, 0x03, 0x84, 0xe7, 0x13, 0xbf, 0x51, 0x70, 0xa3, 0xd4, 0x3b, 0x40, 0xd3,
    0x52, 0xfd, 0x5d, 0x1e, 0x00,
];
const PYTHON_QMA9_RAW: &[u8] = &[
    0, 2, 4, 1, 1, 3, 0, 2, 2, 4, 1, 3, 1, 3, 0, 2, 2, 4, 1, 3, 3, 0, 2, 4,
];

#[test]
fn cli_decodes_python_contract_fixture_and_prefix() {
    let dir = std::env::temp_dir().join(format!("qma-codec-cli-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("test dir");
    let input = dir.join("fixture.qma9");
    let output = dir.join("decoded.raw");
    let metadata = dir.join("decoded.json");
    fs::write(&input, PYTHON_QMA9_FIXTURE).expect("fixture write");

    let status = Command::new(env!("CARGO_BIN_EXE_qma-codec"))
        .arg("decode")
        .arg(&input)
        .arg(&output)
        .arg("--expected-frames")
        .arg("2")
        .arg("--expected-width")
        .arg("3")
        .arg("--expected-height")
        .arg("4")
        .arg("--metadata-json")
        .arg(&metadata)
        .status()
        .expect("run qma-codec decode");
    assert!(status.success());
    assert_eq!(fs::read(&output).expect("decoded raw"), PYTHON_QMA9_RAW);
    let metadata_text = fs::read_to_string(&metadata).expect("metadata json");
    assert!(metadata_text.contains("\"written_mask_bytes\":24"));

    let prefix = dir.join("prefix.raw");
    let status = Command::new(env!("CARGO_BIN_EXE_qma-codec"))
        .arg("decode")
        .arg(&input)
        .arg(&prefix)
        .arg("--prefix-frames")
        .arg("1")
        .arg("--expected-frames")
        .arg("1")
        .arg("--expected-width")
        .arg("3")
        .arg("--expected-height")
        .arg("4")
        .status()
        .expect("run qma-codec prefix decode");
    assert!(status.success());
    assert_eq!(
        fs::read(&prefix).expect("prefix raw"),
        &PYTHON_QMA9_RAW[..12]
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn cli_rejects_partial_shape_checks() {
    let dir = std::env::temp_dir().join(format!("qma-codec-cli-bad-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("test dir");
    let input = dir.join("fixture.qma9");
    let output = dir.join("decoded.raw");
    fs::write(&input, PYTHON_QMA9_FIXTURE).expect("fixture write");

    let output_status = Command::new(env!("CARGO_BIN_EXE_qma-codec"))
        .arg("decode")
        .arg(&input)
        .arg(&output)
        .arg("--expected-frames")
        .arg("2")
        .output()
        .expect("run qma-codec bad shape");
    assert!(!output_status.status.success());
    let stderr = String::from_utf8_lossy(&output_status.stderr);
    assert!(stderr.contains("shape checking requires all"));

    let _ = fs::remove_dir_all(&dir);
}
