use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

const DEFINITION: i32 = 1;

#[derive(Default)]
struct ScipIndex {
    documents: Vec<Document>,
}

#[derive(Default)]
struct Document {
    path: String,
    occurrences: Vec<Occurrence>,
    symbols: Vec<SymbolInfo>,
}

#[derive(Default)]
struct Occurrence {
    range: Vec<i32>,
    symbol: String,
    roles: i32,
}

#[derive(Default)]
struct SymbolInfo {
    symbol: String,
    display: String,
    signature: Option<String>,
}

struct SymbolMatch<'a> {
    document: &'a Document,
    info: &'a SymbolInfo,
}

enum Wire<'a> {
    Var(u64),
    Bytes(&'a [u8]),
    Fixed,
}

struct Proto<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> Proto<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }

    fn next(&mut self) -> Result<Option<(u32, Wire<'a>)>, String> {
        if self.cursor == self.bytes.len() {
            return Ok(None);
        }
        let key = self.varint()?;
        let field = u32::try_from(key >> 3).map_err(|_| "SCIP field number overflow")?;
        if field == 0 {
            return Err("SCIP contains field zero".to_owned());
        }
        let wire = match key & 7 {
            0 => Wire::Var(self.varint()?),
            1 => {
                self.take(8)?;
                Wire::Fixed
            }
            2 => {
                let size = usize::try_from(self.varint()?)
                    .map_err(|_| "SCIP field length overflow".to_owned())?;
                Wire::Bytes(self.take(size)?)
            }
            5 => {
                self.take(4)?;
                Wire::Fixed
            }
            wire => return Err(format!("unsupported SCIP wire type {wire}")),
        };
        Ok(Some((field, wire)))
    }

    fn varint(&mut self) -> Result<u64, String> {
        let mut value = 0_u64;
        for shift in (0..70).step_by(7) {
            let byte = *self.bytes.get(self.cursor).ok_or("truncated SCIP varint")?;
            self.cursor += 1;
            if shift == 63 && byte > 1 {
                return Err("SCIP varint overflow".to_owned());
            }
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err("SCIP varint overflow".to_owned())
    }

    fn take(&mut self, size: usize) -> Result<&'a [u8], String> {
        let end = self
            .cursor
            .checked_add(size)
            .ok_or("SCIP field length overflow")?;
        let bytes = self
            .bytes
            .get(self.cursor..end)
            .ok_or("truncated SCIP field")?;
        self.cursor = end;
        Ok(bytes)
    }
}

pub fn index(root: &Path, force: bool) -> Result<(), String> {
    let path = cache_path(root);
    if !force && fresh(root, &path)? {
        let value = load_index(&path)?;
        print_stats(&path, &value);
        return Ok(());
    }
    let parent = path.parent().ok_or("SCIP index has no parent directory")?;
    fs::create_dir_all(parent).map_err(|error| format!("create SCIP cache: {error}"))?;
    let temporary = path.with_extension("scip.tmp");
    let output = Command::new("rust-analyzer")
        .current_dir(root)
        .args(["scip", ".", "--output"])
        .arg(&temporary)
        .output()
        .map_err(|error| format!("start rust-analyzer SCIP export: {error}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let detail = stderr.lines().rev().take(8).collect::<Vec<_>>();
        return Err(format!(
            "rust-analyzer SCIP export exited with {}: {}",
            output.status,
            detail.into_iter().rev().collect::<Vec<_>>().join("\n")
        ));
    }
    let value = load_index(&temporary)?;
    fs::rename(&temporary, &path).map_err(|error| format!("publish SCIP index: {error}"))?;
    print_stats(&path, &value);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let warnings = stderr
        .lines()
        .filter(|line| {
            line.contains(" ERROR ") || line.trim_start().starts_with("Duplicate symbol:")
        })
        .count();
    if warnings > 0 {
        eprintln!("indexer warnings={warnings} (rust-analyzer omitted or duplicated some symbols)");
    }
    Ok(())
}

pub fn symbol(root: &Path, query: &str, references: bool, limit: usize) -> Result<(), String> {
    let path = cache_path(root);
    if !path.is_file() {
        return Err("Rust SCIP index is absent; run `./blckbaux radar index`".to_owned());
    }
    if !fresh(root, &path)? {
        return Err("Rust SCIP index is stale; run `./blckbaux radar index`".to_owned());
    }
    let value = load_index(&path)?;
    let symbols = select_symbols(&value, query);
    if symbols.is_empty() {
        return Err(format!(
            "no compiler-resolved Rust symbol matches `{query}`"
        ));
    }
    let mut remaining = limit;
    for (position, symbol) in symbols.iter().enumerate() {
        if position > 0 {
            println!();
        }
        print_symbol(&value, symbol, references, &mut remaining);
        if remaining == 0 {
            break;
        }
    }
    Ok(())
}

pub fn doctor(root: &Path) -> Result<(), String> {
    let path = cache_path(root);
    report(
        "lexical",
        root.join("tools/ripwire").is_file(),
        "Ripwire AST search",
    );
    report("rust", rust_scip(), "rust-analyzer SCIP");
    report(
        "rust-index",
        path.is_file() && fresh(root, &path)?,
        path.to_string_lossy().as_ref(),
    );
    report(
        "lean",
        has_command("lake", "--version"),
        "Lean kernel and Lake",
    );
    report(
        "theorem",
        root.join("research/lean/.lake/packages/LeanSearchClient")
            .is_dir(),
        "LeanSearch and Loogle clients",
    );
    report("metal", metal_ready(), "Apple Metal compiler");
    report("cuda", has_command("nvcc", "--version"), "NVIDIA compiler");
    report(
        "codeql",
        has_command("codeql", "version"),
        "optional batch data-flow analysis",
    );
    report(
        "glean",
        has_command("glean", "--version"),
        "optional cross-language fact store",
    );
    Ok(())
}

fn cache_path(root: &Path) -> PathBuf {
    root.join(".cache/blckbaux/radar/rust.scip")
}

fn fresh(root: &Path, index: &Path) -> Result<bool, String> {
    let Ok(index_time) = fs::metadata(index).and_then(|metadata| metadata.modified()) else {
        return Ok(false);
    };
    for path in [
        root.join("Cargo.toml"),
        root.join("rust"),
        root.join("backends/cuda"),
    ] {
        if newer(&path, index_time)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn newer(path: &Path, cutoff: SystemTime) -> Result<bool, String> {
    let metadata =
        fs::metadata(path).map_err(|error| format!("inspect {}: {error}", path.display()))?;
    if metadata.is_file() {
        let relevant = path.file_name().is_some_and(|name| name == "Cargo.toml")
            || path.extension().is_some_and(|extension| extension == "rs");
        return Ok(relevant && metadata.modified().is_ok_and(|time| time > cutoff));
    }
    for entry in fs::read_dir(path).map_err(|error| format!("read {}: {error}", path.display()))? {
        let entry = entry.map_err(|error| format!("read {}: {error}", path.display()))?;
        let child = entry.path();
        if child.file_name().is_some_and(|name| {
            matches!(
                name.to_str(),
                Some(".cache" | ".cargo" | ".git" | ".jj" | "buck-out" | "target" | "vendor")
            )
        }) {
            continue;
        }
        if newer(&child, cutoff)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn load_index(path: &Path) -> Result<ScipIndex, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    parse_index(&bytes).map_err(|error| format!("decode {}: {error}", path.display()))
}

fn parse_index(bytes: &[u8]) -> Result<ScipIndex, String> {
    let mut index = ScipIndex::default();
    let mut proto = Proto::new(bytes);
    while let Some((field, wire)) = proto.next()? {
        if let (2, Wire::Bytes(bytes)) = (field, wire) {
            index.documents.push(parse_doc(bytes)?);
        }
    }
    Ok(index)
}

fn parse_doc(bytes: &[u8]) -> Result<Document, String> {
    let mut document = Document::default();
    let mut proto = Proto::new(bytes);
    while let Some((field, wire)) = proto.next()? {
        match (field, wire) {
            (1, Wire::Bytes(bytes)) => document.path = parse_text(bytes)?,
            (2, Wire::Bytes(bytes)) => document.occurrences.push(parse_occurrence(bytes)?),
            (3, Wire::Bytes(bytes)) => document.symbols.push(parse_info(bytes)?),
            _ => {}
        }
    }
    Ok(document)
}

fn parse_occurrence(bytes: &[u8]) -> Result<Occurrence, String> {
    let mut occurrence = Occurrence::default();
    let mut proto = Proto::new(bytes);
    while let Some((field, wire)) = proto.next()? {
        match (field, wire) {
            (1, Wire::Bytes(bytes)) => {
                let mut packed = Proto::new(bytes);
                while packed.cursor < packed.bytes.len() {
                    occurrence.range.push(packed.varint()? as i32);
                }
            }
            (1, Wire::Var(value)) => occurrence.range.push(value as i32),
            (2, Wire::Bytes(bytes)) => occurrence.symbol = parse_text(bytes)?,
            (3, Wire::Var(value)) => occurrence.roles = value as i32,
            _ => {}
        }
    }
    Ok(occurrence)
}

fn parse_info(bytes: &[u8]) -> Result<SymbolInfo, String> {
    let mut info = SymbolInfo::default();
    let mut proto = Proto::new(bytes);
    while let Some((field, wire)) = proto.next()? {
        match (field, wire) {
            (1, Wire::Bytes(bytes)) => info.symbol = parse_text(bytes)?,
            (6, Wire::Bytes(bytes)) => info.display = parse_text(bytes)?,
            (7, Wire::Bytes(bytes)) => info.signature = parse_signature(bytes)?,
            _ => {}
        }
    }
    Ok(info)
}

fn parse_signature(bytes: &[u8]) -> Result<Option<String>, String> {
    let mut proto = Proto::new(bytes);
    while let Some((field, wire)) = proto.next()? {
        if let (5, Wire::Bytes(bytes)) = (field, wire) {
            return parse_text(bytes).map(Some);
        }
    }
    Ok(None)
}

fn parse_text(bytes: &[u8]) -> Result<String, String> {
    std::str::from_utf8(bytes)
        .map(str::to_owned)
        .map_err(|error| format!("SCIP contains invalid UTF-8: {error}"))
}

fn print_stats(path: &Path, index: &ScipIndex) {
    let definitions = index
        .documents
        .iter()
        .flat_map(|document| &document.occurrences)
        .filter(|occurrence| occurrence.roles & DEFINITION != 0)
        .count();
    let occurrences = index
        .documents
        .iter()
        .map(|document| document.occurrences.len())
        .sum::<usize>();
    println!(
        "SCIP {} documents={} definitions={} occurrences={}",
        path.display(),
        index.documents.len(),
        definitions,
        occurrences
    );
}

fn select_symbols<'a>(index: &'a ScipIndex, query: &str) -> Vec<SymbolMatch<'a>> {
    let query = query.to_ascii_lowercase();
    let mut matches = BTreeMap::<String, (u8, SymbolMatch<'a>)>::new();
    for document in &index.documents {
        let definitions = document
            .occurrences
            .iter()
            .filter(|occurrence| occurrence.roles & DEFINITION != 0)
            .map(|occurrence| occurrence.symbol.as_str())
            .collect::<BTreeSet<_>>();
        for info in &document.symbols {
            if !definitions.contains(info.symbol.as_str()) {
                continue;
            }
            let display = info.display.to_ascii_lowercase();
            let symbol = info.symbol.to_ascii_lowercase();
            let score = if display == query {
                0
            } else if symbol == query {
                1
            } else if display.contains(&query) {
                2
            } else if symbol.contains(&query) {
                3
            } else {
                continue;
            };
            let key = if local_symbol(&info.symbol) {
                format!("{}\0{}", document.path, info.symbol)
            } else {
                info.symbol.clone()
            };
            matches
                .entry(key)
                .or_insert((score, SymbolMatch { document, info }));
        }
    }
    let Some(best) = matches.values().map(|(score, _)| *score).min() else {
        return Vec::new();
    };
    matches
        .into_values()
        .filter(|(score, _)| *score == best)
        .map(|(_, info)| info)
        .collect()
}

fn print_symbol(index: &ScipIndex, found: &SymbolMatch<'_>, references: bool, limit: &mut usize) {
    let info = found.info;
    println!("SYMBOL {}", info.display);
    if local_symbol(&info.symbol) {
        println!("canonical: {} @ {}", info.symbol, found.document.path);
    } else {
        println!("canonical: {}", info.symbol);
    }
    if let Some(signature) = info.signature.as_ref() {
        println!("signature: {}", signature.replace('\n', " "));
    }
    let mut definitions = Vec::new();
    let mut refs = Vec::new();
    let documents: &[Document] = if local_symbol(&info.symbol) {
        std::slice::from_ref(found.document)
    } else {
        &index.documents
    };
    for document in documents {
        for occurrence in &document.occurrences {
            if occurrence.symbol != info.symbol {
                continue;
            }
            let hit = format_hit(&document.path, occurrence);
            if occurrence.roles & DEFINITION != 0 {
                definitions.push(hit);
            } else {
                refs.push(hit);
            }
        }
    }
    definitions.sort();
    refs.sort();
    for hit in definitions {
        if *limit == 0 {
            break;
        }
        println!("definition: {hit}");
        *limit -= 1;
    }
    println!("references: {}", refs.len());
    if references {
        for hit in refs {
            if *limit == 0 {
                break;
            }
            println!("reference: {hit}");
            *limit -= 1;
        }
    }
}

fn local_symbol(symbol: &str) -> bool {
    symbol.starts_with("local ")
}

fn format_hit(path: &str, occurrence: &Occurrence) -> String {
    let (line, column) = match occurrence.range.as_slice() {
        [line, column, _] | [line, column, _, _] => (*line + 1, *column + 1),
        _ => (0, 0),
    };
    format!("{path}:{line}:{column}")
}

fn rust_scip() -> bool {
    Command::new("rust-analyzer")
        .arg("--help")
        .output()
        .is_ok_and(|output| {
            output.status.success() && String::from_utf8_lossy(&output.stdout).contains("scip")
        })
}

fn has_command(program: &str, argument: &str) -> bool {
    Command::new(program)
        .arg(argument)
        .output()
        .is_ok_and(|output| output.status.success())
}

fn metal_ready() -> bool {
    Command::new("xcrun")
        .args(["-sdk", "macosx", "--find", "metal"])
        .output()
        .is_ok_and(|output| output.status.success())
}

fn report(name: &str, ready: bool, detail: &str) {
    let state = if ready { "ready" } else { "absent" };
    println!("{name:<12} {state:<6} {detail}");
}

#[cfg(test)]
mod tests {
    use super::{DEFINITION, Occurrence, format_hit, parse_index};

    fn varint(mut value: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            bytes.push(byte);
            if value == 0 {
                return bytes;
            }
        }
    }

    fn bytes(field: u8, value: &[u8]) -> Vec<u8> {
        let mut output = varint(u64::from(field) << 3 | 2);
        output.extend(varint(value.len() as u64));
        output.extend(value);
        output
    }

    #[test]
    fn location() {
        let occurrence = Occurrence {
            range: vec![4, 2, 9],
            ..Occurrence::default()
        };
        assert_eq!(format_hit("source.rs", &occurrence), "source.rs:5:3");
    }

    #[test]
    fn decode_index() {
        let mut occurrence = bytes(1, &[4, 2, 9]);
        occurrence.extend(bytes(2, b"rust test symbol"));
        occurrence.extend([24, 1]);

        let signature = bytes(5, b"fn test()");
        let mut info = bytes(1, b"rust test symbol");
        info.extend(bytes(6, b"test"));
        info.extend(bytes(7, &signature));

        let mut document = bytes(1, b"source.rs");
        document.extend(bytes(2, &occurrence));
        document.extend(bytes(3, &info));
        let index = parse_index(&bytes(2, &document)).expect("valid synthetic SCIP");

        assert_eq!(index.documents[0].path, "source.rs");
        assert_eq!(index.documents[0].occurrences[0].roles, DEFINITION);
        assert_eq!(index.documents[0].symbols[0].display, "test");
        assert_eq!(
            index.documents[0].symbols[0].signature.as_deref(),
            Some("fn test()")
        );
    }
}
