use python_ast_indexer::index_python_file;
use rayon::prelude::*;
use std::env;
use std::ffi::OsString;
use std::io::{self, Write};
use std::path::PathBuf;

fn main() {
    let args: Vec<OsString> = env::args_os().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: python-ast-indexer <file.py>");
        eprintln!("       python-ast-indexer --batch <file.py>...");
        std::process::exit(2);
    }

    let (payload, parse_ok) = if args[0] == "--batch" {
        if args.len() == 1 {
            eprintln!("usage: python-ast-indexer --batch <file.py>...");
            std::process::exit(2);
        }
        let indices = args[1..]
            .par_iter()
            .map(|path| index_python_file(&PathBuf::from(path)))
            .collect::<Vec<_>>();
        let parse_ok = indices.iter().all(|index| index.parse_ok);
        (serde_json::to_string_pretty(&indices), parse_ok)
    } else {
        if args.len() != 1 {
            eprintln!("usage: python-ast-indexer <file.py>");
            eprintln!("       python-ast-indexer --batch <file.py>...");
            std::process::exit(2);
        }
        let index = index_python_file(&PathBuf::from(&args[0]));
        let parse_ok = index.parse_ok;
        (serde_json::to_string_pretty(&index), parse_ok)
    };

    match payload {
        Ok(text) => {
            let mut stdout = io::stdout().lock();
            if let Err(err) = writeln!(stdout, "{text}") {
                if err.kind() == io::ErrorKind::BrokenPipe {
                    std::process::exit(0);
                }
                eprintln!("failed to write index: {err}");
                std::process::exit(1);
            }
        }
        Err(err) => {
            eprintln!("failed to serialize index: {err}");
            std::process::exit(1);
        }
    }
    if !parse_ok {
        std::process::exit(1);
    }
}
