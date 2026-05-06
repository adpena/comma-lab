use rustpython_parser::{ast, Parse};
use serde::Serialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct FunctionIndex {
    pub name: String,
    pub lineno: u32,
    pub args: Vec<String>,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct ClassIndex {
    pub name: String,
    pub lineno: u32,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct ImportIndex {
    pub module: Option<String>,
    pub names: Vec<String>,
    pub lineno: u32,
    pub level: u32,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct PythonAstIndex {
    pub path: String,
    pub parse_ok: bool,
    pub error: Option<String>,
    pub functions: Vec<FunctionIndex>,
    pub classes: Vec<ClassIndex>,
    pub imports: Vec<ImportIndex>,
    pub top_level_names: Vec<String>,
}

pub fn index_python_file(path: &Path) -> PythonAstIndex {
    let source_path = path.to_string_lossy().to_string();
    match fs::read_to_string(path) {
        Ok(source) => index_python_source(&source_path, &source),
        Err(err) => PythonAstIndex {
            path: source_path,
            parse_ok: false,
            error: Some(format!("read error: {err}")),
            functions: vec![],
            classes: vec![],
            imports: vec![],
            top_level_names: vec![],
        },
    }
}

pub fn index_python_source(path: &str, source: &str) -> PythonAstIndex {
    match ast::Suite::parse(source, path) {
        Ok(suite) => {
            let mut index = PythonAstIndex {
                path: path.to_string(),
                parse_ok: true,
                error: None,
                functions: vec![],
                classes: vec![],
                imports: vec![],
                top_level_names: vec![],
            };
            collect_top_level(&mut index, &suite, source);
            index.top_level_names.sort();
            index.top_level_names.dedup();
            index
        }
        Err(err) => PythonAstIndex {
            path: path.to_string(),
            parse_ok: false,
            error: Some(err.to_string()),
            functions: vec![],
            classes: vec![],
            imports: vec![],
            top_level_names: vec![],
        },
    }
}

fn collect_top_level(index: &mut PythonAstIndex, suite: &[ast::Stmt], source: &str) {
    for stmt in suite {
        match stmt {
            ast::Stmt::FunctionDef(node) => {
                let name = node.name.to_string();
                index.top_level_names.push(name.clone());
                index.functions.push(FunctionIndex {
                    name,
                    lineno: lineno(source, node.range.start().to_usize()),
                    args: argument_names(&node.args),
                });
            }
            ast::Stmt::AsyncFunctionDef(node) => {
                let name = node.name.to_string();
                index.top_level_names.push(name.clone());
                index.functions.push(FunctionIndex {
                    name,
                    lineno: lineno(source, node.range.start().to_usize()),
                    args: argument_names(&node.args),
                });
            }
            ast::Stmt::ClassDef(node) => {
                let name = node.name.to_string();
                index.top_level_names.push(name.clone());
                index.classes.push(ClassIndex {
                    name,
                    lineno: lineno(source, node.range.start().to_usize()),
                });
            }
            ast::Stmt::Import(node) => {
                let names = node.names.iter().map(alias_name).collect::<Vec<_>>();
                index.top_level_names.extend(names.iter().cloned());
                index.imports.push(ImportIndex {
                    module: None,
                    names,
                    lineno: lineno(source, node.range.start().to_usize()),
                    level: 0,
                });
            }
            ast::Stmt::ImportFrom(node) => {
                let names = node.names.iter().map(alias_name).collect::<Vec<_>>();
                index.top_level_names.extend(names.iter().cloned());
                index.imports.push(ImportIndex {
                    module: node.module.as_ref().map(ToString::to_string),
                    names,
                    lineno: lineno(source, node.range.start().to_usize()),
                    level: node.level.as_ref().map(|value| value.to_u32()).unwrap_or(0),
                });
            }
            ast::Stmt::Assign(node) => {
                for target in &node.targets {
                    if let Some(name) = assigned_name(target) {
                        index.top_level_names.push(name);
                    }
                }
            }
            ast::Stmt::AnnAssign(node) => {
                if let Some(name) = assigned_name(&node.target) {
                    index.top_level_names.push(name);
                }
            }
            _ => {}
        }
    }
}

fn argument_names(args: &ast::Arguments) -> Vec<String> {
    let mut out = Vec::new();
    for arg in &args.posonlyargs {
        out.push(arg.def.arg.to_string());
    }
    for arg in &args.args {
        out.push(arg.def.arg.to_string());
    }
    if let Some(arg) = &args.vararg {
        out.push(arg.arg.to_string());
    }
    for arg in &args.kwonlyargs {
        out.push(arg.def.arg.to_string());
    }
    if let Some(arg) = &args.kwarg {
        out.push(arg.arg.to_string());
    }
    out
}

fn alias_name(alias: &ast::Alias) -> String {
    alias.asname.as_ref().unwrap_or(&alias.name).to_string()
}

fn assigned_name(expr: &ast::Expr) -> Option<String> {
    match expr {
        ast::Expr::Name(node) => Some(node.id.to_string()),
        _ => None,
    }
}

fn lineno(source: &str, byte_offset: usize) -> u32 {
    let end = byte_offset.min(source.len());
    source.as_bytes()[..end]
        .iter()
        .filter(|byte| **byte == b'\n')
        .count() as u32
        + 1
}

#[cfg(test)]
mod tests {
    use super::index_python_source;

    #[test]
    fn indexes_top_level_python_contract() {
        let source = r#"
import json as json_lib
from tac.preflight import check_public_release_hygiene as hygiene

VALUE = 3
typed_value: int = 4

class Example:
    pass

def f(a, /, b, *args, c=1, **kwargs):
    return a + b

async def g(x):
    return x
"#;

        let index = index_python_source("sample.py", source);

        assert!(index.parse_ok);
        assert_eq!(index.functions[0].name, "f");
        assert_eq!(index.functions[0].args, ["a", "b", "args", "c", "kwargs"]);
        assert_eq!(index.functions[1].name, "g");
        assert_eq!(index.classes[0].name, "Example");
        assert_eq!(index.imports[0].names, ["json_lib"]);
        assert_eq!(index.imports[1].module.as_deref(), Some("tac.preflight"));
        assert_eq!(index.imports[1].names, ["hygiene"]);
        assert!(index.top_level_names.contains(&"VALUE".to_string()));
        assert!(index.top_level_names.contains(&"typed_value".to_string()));
    }

    #[test]
    fn reports_parse_errors_without_panicking() {
        let index = index_python_source("bad.py", "def nope(:\n");

        assert!(!index.parse_ok);
        assert!(index.error.is_some());
        assert!(index.functions.is_empty());
    }
}
