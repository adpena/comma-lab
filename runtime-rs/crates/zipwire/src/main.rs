use std::env;
use std::ffi::OsStr;
use std::io::{self, Write};
use std::path::PathBuf;
use zipwire::{inspect_zip_path, rewrite_single_identity_path};

fn main() {
    let mut args = env::args_os().skip(1);
    let Some(path) = args.next() else {
        print_usage_and_exit();
    };

    if path == OsStr::new("rewrite-single") {
        let Some(input) = args.next() else {
            print_usage_and_exit();
        };
        let Some(output) = args.next() else {
            print_usage_and_exit();
        };
        if args.next().is_some() {
            print_usage_and_exit();
        }
        let proof =
            match rewrite_single_identity_path(&PathBuf::from(input), &PathBuf::from(output)) {
                Ok(proof) => proof,
                Err(err) => {
                    eprintln!("rewrite-single failed: {err}");
                    std::process::exit(1);
                }
            };
        write_json_or_exit(&proof, "rewrite-single proof JSON");
        return;
    }

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

    write_json_or_exit(&inspect, "inspect JSON");

    if !inspect.zip_strict {
        std::process::exit(1);
    }
}

fn write_json_or_exit<T: serde::Serialize>(value: &T, label: &str) {
    match serde_json::to_string_pretty(value) {
        Ok(text) => {
            let mut stdout = io::stdout().lock();
            if let Err(err) = writeln!(stdout, "{text}") {
                if err.kind() == io::ErrorKind::BrokenPipe {
                    std::process::exit(0);
                }
                eprintln!("failed to write {label}: {err}");
                std::process::exit(1);
            }
        }
        Err(err) => {
            eprintln!("failed to serialize {label}: {err}");
            std::process::exit(1);
        }
    }
}

fn print_usage_and_exit() -> ! {
    eprintln!("usage: zipwire <archive.zip>");
    eprintln!("       zipwire rewrite-single <input.zip> <output.zip>");
    std::process::exit(2);
}
