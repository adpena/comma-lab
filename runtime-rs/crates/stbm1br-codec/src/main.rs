use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use stbm1br_codec::{
    decode_stbm1br_segment, decode_stbm1br_segment_expected, metadata_json, parse_stbm1br_metadata,
};

fn main() {
    if let Err(err) = run() {
        eprintln!("stbm1br-codec: {err}");
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
                "usage:\n  stbm1br-codec decode <input.stbm1br> <output.raw> [--expected-frames N --expected-height H --expected-width W] [--metadata-json PATH]\n  stbm1br-codec metadata <input.stbm1br>"
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
    let mut expected_frames = None;
    let mut expected_height = None;
    let mut expected_width = None;
    let mut metadata_json_path = None::<PathBuf>;
    let mut i = 2usize;
    while i < args.len() {
        match args[i].as_str() {
            "--expected-frames" => {
                expected_frames = Some(parse_value(args, i + 1, "--expected-frames")?);
                i += 2;
            }
            "--expected-height" => {
                expected_height = Some(parse_value(args, i + 1, "--expected-height")?);
                i += 2;
            }
            "--expected-width" => {
                expected_width = Some(parse_value(args, i + 1, "--expected-width")?);
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
    let started = Instant::now();
    let decoded = match (expected_frames, expected_height, expected_width) {
        (Some(frames), Some(height), Some(width)) => {
            decode_stbm1br_segment_expected(&payload, frames, height, width)?
        }
        (None, None, None) => decode_stbm1br_segment(&payload)?,
        _ => {
            return Err(
                "shape checking requires all of --expected-frames, --expected-height, and --expected-width"
                    .into(),
            );
        }
    };
    let elapsed = started.elapsed();
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
        fs::write(path, metadata_json(&decoded.metadata))?;
    }
    eprintln!(
        "stbm1br-codec: decoded {} frames {}x{} to {} bytes in {:.3}s",
        decoded.metadata.n_pairs,
        decoded.metadata.height,
        decoded.metadata.width,
        decoded.data.len(),
        elapsed.as_secs_f64()
    );
    Ok(())
}

fn metadata_cmd(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.len() != 1 {
        return Err("metadata requires <input>".into());
    }
    let payload = fs::read(&args[0])?;
    let meta = parse_stbm1br_metadata(&payload)?;
    println!("{}", metadata_json(&meta));
    Ok(())
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
