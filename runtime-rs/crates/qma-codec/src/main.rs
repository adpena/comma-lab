use std::fs;
use std::io::Write;
use std::path::PathBuf;

use qma_codec::{decode_qma9_mask, decode_qma9_prefix_frames, parse_qma9_header};

fn main() {
    if let Err(err) = run() {
        eprintln!("qma-codec: {err}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match args.first().map(String::as_str) {
        Some("decode") => decode_cmd(&args[1..]),
        Some("metadata") => metadata_cmd(&args[1..]),
        _ => {
            eprintln!(
                "usage:\n  qma-codec decode <input.qma9> <output.raw> [--prefix-frames N] [--expected-frames N --expected-width W --expected-height H] [--metadata-json PATH]\n  qma-codec metadata <input.qma9>"
            );
            std::process::exit(2);
        }
    }
}

fn decode_cmd(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.len() < 2 {
        return Err("decode requires <input> <output>".into());
    }
    let input = PathBuf::from(&args[0]);
    let output = PathBuf::from(&args[1]);
    let mut prefix_frames = None;
    let mut expected_frames = None;
    let mut expected_width = None;
    let mut expected_height = None;
    let mut metadata_json_path = None::<PathBuf>;

    let mut i = 2usize;
    while i < args.len() {
        match args[i].as_str() {
            "--prefix-frames" => {
                prefix_frames = Some(parse_value(args, i + 1, "--prefix-frames")?);
                i += 2;
            }
            "--expected-frames" => {
                expected_frames = Some(parse_value(args, i + 1, "--expected-frames")?);
                i += 2;
            }
            "--expected-width" => {
                expected_width = Some(parse_value(args, i + 1, "--expected-width")?);
                i += 2;
            }
            "--expected-height" => {
                expected_height = Some(parse_value(args, i + 1, "--expected-height")?);
                i += 2;
            }
            "--metadata-json" => {
                metadata_json_path = Some(PathBuf::from(value(args, i + 1, "--metadata-json")?));
                i += 2;
            }
            flag => return Err(format!("unknown decode flag: {flag}").into()),
        }
    }

    let payload = fs::read(&input)?;
    let decoded = match prefix_frames {
        Some(frames) => decode_qma9_prefix_frames(&payload, frames)?,
        None => decode_qma9_mask(&payload)?,
    };
    check_shape(
        written_frames(&decoded)?,
        decoded.header.width as usize,
        decoded.header.height as usize,
        expected_frames,
        expected_width,
        expected_height,
    )?;

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::File::create(&output)?;
    file.write_all(&decoded.data)?;
    file.flush()?;

    if let Some(path) = metadata_json_path {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, metadata_json(&decoded))?;
    }

    eprintln!(
        "qma-codec: decoded {} frame(s) from {}x{} QMA9 to {} bytes",
        written_frames(&decoded)?,
        decoded.header.width,
        decoded.header.height,
        decoded.data.len()
    );
    Ok(())
}

fn metadata_cmd(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.len() != 1 {
        return Err("metadata requires <input>".into());
    }
    let payload = fs::read(&args[0])?;
    let header = parse_qma9_header(&payload)?;
    println!(
        "{{\"magic\":\"QMA9\",\"frame_count\":{},\"width\":{},\"height\":{},\"bitstream_bytes\":{},\"header_bytes\":{},\"packed_bytes\":{},\"decoded_mask_bytes\":{}}}",
        header.frame_count,
        header.width,
        header.height,
        header.bitstream_bytes,
        header.header_bytes,
        header.packed_bytes,
        header.decoded_mask_bytes
    );
    Ok(())
}

fn metadata_json(decoded: &qma_codec::Qma9DecodedMask) -> String {
    format!(
        "{{\"magic\":\"QMA9\",\"frame_count\":{},\"width\":{},\"height\":{},\"bitstream_bytes\":{},\"header_bytes\":{},\"packed_bytes\":{},\"decoded_mask_bytes\":{},\"written_mask_bytes\":{}}}",
        decoded.header.frame_count,
        decoded.header.width,
        decoded.header.height,
        decoded.header.bitstream_bytes,
        decoded.header.header_bytes,
        decoded.header.packed_bytes,
        decoded.header.decoded_mask_bytes,
        decoded.data.len()
    )
}

fn check_shape(
    frames: usize,
    width: usize,
    height: usize,
    expected_frames: Option<usize>,
    expected_width: Option<usize>,
    expected_height: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    match (expected_frames, expected_width, expected_height) {
        (None, None, None) => Ok(()),
        (Some(ef), Some(ew), Some(eh)) => {
            if (frames, width, height) == (ef, ew, eh) {
                Ok(())
            } else {
                Err(format!(
                    "decoded shape ({frames}, {width}, {height}) != expected ({ef}, {ew}, {eh})"
                )
                .into())
            }
        }
        _ => Err("shape checking requires all of --expected-frames, --expected-width, and --expected-height".into()),
    }
}

fn written_frames(
    decoded: &qma_codec::Qma9DecodedMask,
) -> Result<usize, Box<dyn std::error::Error>> {
    let frame_size = decoded.header.width as usize * decoded.header.height as usize;
    if frame_size == 0 || decoded.data.len() % frame_size != 0 {
        return Err("decoded QMA9 output does not contain complete frames".into());
    }
    Ok(decoded.data.len() / frame_size)
}

fn parse_value(
    args: &[String],
    index: usize,
    flag: &str,
) -> Result<usize, Box<dyn std::error::Error>> {
    Ok(value(args, index, flag)?.parse::<usize>()?)
}

fn value<'a>(
    args: &'a [String],
    index: usize,
    flag: &str,
) -> Result<&'a str, Box<dyn std::error::Error>> {
    args.get(index)
        .map(String::as_str)
        .ok_or_else(|| format!("{flag} requires a value").into())
}
