use raw_locality_compare::{compare_raw_triplet, CompareInput};
use std::env;
use std::ffi::OsString;
use std::io::{self, Write};
use std::path::PathBuf;

fn main() {
    let input = match parse_args(env::args_os().skip(1).collect()) {
        Ok(input) => input,
        Err(message) => {
            eprintln!("{message}");
            print_usage();
            std::process::exit(2);
        }
    };

    let report = match compare_raw_triplet(&input) {
        Ok(report) => report,
        Err(err) => {
            eprintln!("raw-locality-compare failed: {err}");
            std::process::exit(2);
        }
    };
    match serde_json::to_string_pretty(&report) {
        Ok(text) => {
            let mut stdout = io::stdout().lock();
            if let Err(err) = writeln!(stdout, "{text}") {
                if err.kind() == io::ErrorKind::BrokenPipe {
                    std::process::exit(0);
                }
                eprintln!("failed to write report JSON: {err}");
                std::process::exit(1);
            }
        }
        Err(err) => {
            eprintln!("failed to serialize report JSON: {err}");
            std::process::exit(1);
        }
    }
}

fn parse_args(args: Vec<OsString>) -> Result<CompareInput, String> {
    let mut parent_raw: Option<PathBuf> = None;
    let mut global_mutated_raw: Option<PathBuf> = None;
    let mut selective_raw: Option<PathBuf> = None;
    let mut selected_frame_indices: Option<Vec<i64>> = None;
    let mut frame_count: Option<usize> = None;
    let mut frame_bytes: Option<usize> = None;
    let mut rel_path = "0.raw".to_string();
    let mut sample_limit = 8_usize;

    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        let flag = arg
            .to_str()
            .ok_or_else(|| "arguments must be valid UTF-8".to_string())?;
        match flag {
            "--parent" => parent_raw = Some(next_path(&mut iter, flag)?),
            "--global-mutated" => global_mutated_raw = Some(next_path(&mut iter, flag)?),
            "--selective" => selective_raw = Some(next_path(&mut iter, flag)?),
            "--selected-frame-indices" | "--selected-frames" => {
                selected_frame_indices = Some(parse_index_list(&next_string(&mut iter, flag)?)?)
            }
            "--frame-count" => {
                frame_count = Some(parse_usize(&next_string(&mut iter, flag)?, flag)?)
            }
            "--frame-bytes" => {
                frame_bytes = Some(parse_usize(&next_string(&mut iter, flag)?, flag)?)
            }
            "--rel-path" | "--raw-path" => rel_path = next_string(&mut iter, flag)?,
            "--sample-limit" => sample_limit = parse_usize(&next_string(&mut iter, flag)?, flag)?,
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {flag}")),
        }
    }

    Ok(CompareInput {
        parent_raw: parent_raw.ok_or_else(|| "--parent is required".to_string())?,
        global_mutated_raw: global_mutated_raw
            .ok_or_else(|| "--global-mutated is required".to_string())?,
        selective_raw: selective_raw.ok_or_else(|| "--selective is required".to_string())?,
        selected_frame_indices: selected_frame_indices
            .ok_or_else(|| "--selected-frame-indices is required".to_string())?,
        frame_count: frame_count.ok_or_else(|| "--frame-count is required".to_string())?,
        frame_bytes,
        rel_path,
        sample_limit,
    })
}

fn next_path<I>(iter: &mut I, flag: &str) -> Result<PathBuf, String>
where
    I: Iterator<Item = OsString>,
{
    iter.next()
        .map(PathBuf::from)
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn next_string<I>(iter: &mut I, flag: &str) -> Result<String, String>
where
    I: Iterator<Item = OsString>,
{
    iter.next()
        .ok_or_else(|| format!("{flag} requires a value"))?
        .into_string()
        .map_err(|_| format!("{flag} value must be valid UTF-8"))
}

fn parse_usize(value: &str, flag: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|err| format!("{flag} must be a non-negative integer: {err}"))
}

fn parse_index_list(value: &str) -> Result<Vec<i64>, String> {
    let mut indices = Vec::new();
    for part in value
        .split(|ch: char| ch == ',' || ch.is_ascii_whitespace())
        .filter(|part| !part.is_empty())
    {
        indices.push(
            part.parse::<i64>()
                .map_err(|err| format!("bad selected frame index {part:?}: {err}"))?,
        );
    }
    Ok(indices)
}

fn print_usage() {
    eprintln!(
        "usage: raw-locality-compare --parent parent.raw --global-mutated global.raw \\\n\
         --selective selective.raw --selected-frame-indices 2,3 --frame-count 1200 \\\n\
         [--frame-bytes BYTES] [--rel-path 0.raw] [--sample-limit 8]"
    );
}
