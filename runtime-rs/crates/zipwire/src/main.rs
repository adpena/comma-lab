use std::env;
use std::io::{self, Write};
use std::path::PathBuf;
use zipwire::inspect_zip_path;

fn main() {
    let mut args = env::args_os().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: zipwire <archive.zip>");
        std::process::exit(2);
    };
    if args.next().is_some() {
        eprintln!("usage: zipwire <archive.zip>");
        std::process::exit(2);
    }

    let inspect = match inspect_zip_path(&PathBuf::from(path)) {
        Ok(inspect) => inspect,
        Err(err) => {
            eprintln!("failed to read archive: {err}");
            std::process::exit(1);
        }
    };

    match serde_json::to_string_pretty(&inspect) {
        Ok(text) => {
            let mut stdout = io::stdout().lock();
            if let Err(err) = writeln!(stdout, "{text}") {
                if err.kind() == io::ErrorKind::BrokenPipe {
                    std::process::exit(0);
                }
                eprintln!("failed to write inspect JSON: {err}");
                std::process::exit(1);
            }
        }
        Err(err) => {
            eprintln!("failed to serialize inspect JSON: {err}");
            std::process::exit(1);
        }
    }

    if !inspect.zip_strict {
        std::process::exit(1);
    }
}
