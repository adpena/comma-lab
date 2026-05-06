use python_ast_indexer::index_python_file;
use std::env;
use std::io::{self, Write};
use std::path::PathBuf;

fn main() {
    let mut args = env::args_os().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: python-ast-indexer <file.py>");
        std::process::exit(2);
    };
    if args.next().is_some() {
        eprintln!("usage: python-ast-indexer <file.py>");
        std::process::exit(2);
    }

    let index = index_python_file(&PathBuf::from(path));
    match serde_json::to_string_pretty(&index) {
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
    if !index.parse_ok {
        std::process::exit(1);
    }
}
