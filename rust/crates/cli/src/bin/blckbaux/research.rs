use clap::Subcommand;
use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};

use crate::config::BuildSystem;

mod agy;

const RUNTIME_TARGET: &str = "//rust/crates/runner:blckbaux-runner";
const AENEAS_REV: &str = "e62dd83d133d9043e7d3628096838b7430f36565";
const CHARON_REV: &str = "de8736f7c995ac54dc490345cef9ea5143c06c62";
const ACTION_ROOTS: [&str; 3] = [
    "crate::reliability::radius::accept_bits",
    "crate::reliability::radius::accept_decision",
    "crate::reliability::radius::decide_action",
];

#[derive(Subcommand)]
pub enum ResearchAction {
    /// Check all Lean proofs and the typed experiment contract.
    #[command(
        long_about = "Check all Lean proofs and the typed experiment contract.\n\nThe default reuses Lake's local build cache. Use --via-build to route the Lean build through the selected Buck2 or Bazel target. Both paths currently use the host's pinned Lake toolchain."
    )]
    Check {
        /// Route the Lean build through Buck2 or Bazel instead of local Lake.
        #[arg(long = "via-build", visible_alias = "isolated", alias = "clean")]
        via_build: bool,
    },
    /// Recheck accepted proofs with every kernel shipped by Lean (requires bubblewrap).
    Audit,
    /// Show the checked research frontier.
    #[command(visible_alias = "st")]
    Status,
    /// List formalized construction topics.
    #[command(
        visible_alias = "ls",
        long_about = "List formalized construction topics.\n\nAn optional query performs a case-insensitive substring filter. Use `show TOPIC` to read one complete construction record."
    )]
    List {
        /// Keep only topics containing this text.
        query: Option<String>,
    },
    /// Show a construction or Lean declaration.
    #[command(
        alias = "inspect",
        long_about = "Show a construction or Lean declaration.\n\nA construction topic is read immediately from the research catalogue. A fully qualified Lean declaration is elaborated in the pinned BLCKBAUX environment and reports its type, axioms, home module, and minimal import."
    )]
    Show {
        /// Exact construction topic or fully qualified Lean declaration.
        name: String,
        /// Print the complete proof or definition body.
        #[arg(short, long)]
        body: bool,
        /// Load all of Mathlib instead of the BLCKBAUX import closure.
        #[arg(long, visible_alias = "all-mathlib")]
        mathlib: bool,
    },
    /// Search for an exact proof of a Lean proposition.
    #[command(
        alias = "suggest",
        long_about = "Search for an exact proof of a Lean proposition.\n\nThis runs Lean's indexed `exact?` search. The result is a candidate, not an accepted theorem; place it in a declaration and run `./blckbaux research check` for kernel replay. Pass '-' to read the proposition from standard input."
    )]
    Prove {
        /// Closed proposition, or '-' to read it from standard input.
        proposition: String,
        /// Load all of Mathlib instead of the BLCKBAUX import closure.
        #[arg(long, visible_alias = "all-mathlib")]
        mathlib: bool,
    },
    /// Search Mathlib by mathematical meaning or Lean type structure.
    #[command(
        long_about = "Search Mathlib by mathematical meaning or Lean type structure.\n\nThe default sends a natural-language query through Mathlib's LeanSearch client. --structural sends Lean syntax through its Loogle client. Search results are discovery evidence; inspect a result with `show`, then kernel-check any theorem that uses it."
    )]
    Search {
        /// Natural-language question or Loogle structural pattern.
        query: String,
        /// Interpret the query as a Loogle type pattern.
        #[arg(long)]
        structural: bool,
    },
    /// Start a persistent Lean proof session.
    Repl,
    /// Extract the production optimizer decision law from Rust into Lean.
    #[command(
        long_about = "Extract the production optimizer decision law from Rust into Lean.\n\nCharon reads the selected Rust call graph and emits LLBC; Aeneas translates that LLBC into the checked Lean model under BlckbauxResearch/Generated. The locally cached tools must match the repository's pinned commits. Use --check to reject drift without modifying tracked files."
    )]
    Extract {
        /// Verify that the checked-in Lean translation matches the current Rust source.
        #[arg(long)]
        check: bool,
    },
    /// Import an Anti-Gravity conversation as compact research context.
    #[command(
        long_about = "Import an Anti-Gravity conversation from its local SQLite store.\n\nBy default this prints the newest context summary followed by accepted user and assistant turns after that summary. Use --full for the complete human transcript and --tools to include tool calls. The database is opened read-only and step payloads are decoded in-process."
    )]
    Import {
        /// Anti-Gravity conversation UUID.
        conversation: String,
        /// Print the complete human transcript instead of resuming from the newest summary.
        #[arg(long)]
        full: bool,
        /// Include tool names and arguments; tool output remains omitted.
        #[arg(long)]
        tools: bool,
        /// Write the import to this file instead of standard output.
        #[arg(short, long)]
        output: Option<std::path::PathBuf>,
    },
    /// Compatibility spelling for the old catalogue command.
    #[command(hide = true)]
    Catalogue { topic: Option<String> },
}

pub fn run(
    root: &std::path::Path,
    action: ResearchAction,
    build_system: BuildSystem,
) -> Result<(), String> {
    match action {
        ResearchAction::Check { via_build } => check(root, build_system, via_build),
        ResearchAction::Audit => {
            require_sandbox()?;
            check(root, build_system, false)?;
            let mut command = Command::new("lake");
            command
                .current_dir(root.join("research/lean"))
                .args(["check", "--paranoid"]);
            crate::execute(command)?;
            println!("blckbaux: Lean proofs independently rechecked");
            Ok(())
        }
        ResearchAction::Status => catalogue(root, Some("frontier")),
        ResearchAction::List { query } => list(root, query.as_deref()),
        ResearchAction::Show {
            name,
            body,
            mathlib,
        } => show(root, &name, body, mathlib),
        ResearchAction::Prove {
            proposition,
            mathlib,
        } => prove(root, &proposition, mathlib),
        ResearchAction::Search { query, structural } => search(root, &query, structural),
        ResearchAction::Repl => {
            let mut command = Command::new("lake");
            command
                .current_dir(root.join("research/lean"))
                .args(["exe", "repl"]);
            crate::execute(command)
        }
        ResearchAction::Extract { check } => extract(root, check),
        ResearchAction::Import {
            conversation,
            full,
            tools,
            output,
        } => agy::import(&conversation, full, tools, output.as_deref()),
        ResearchAction::Catalogue { topic } => catalogue(root, topic.as_deref()),
    }
}

fn extract(root: &Path, check: bool) -> Result<(), String> {
    let tools = root.join(".cache/blckbaux/tools/aeneas");
    let aeneas = tools.join("bin/aeneas");
    let charon_root = tools.join("charon");
    let charon = charon_root.join("bin/charon");
    require_tool(&aeneas, "Aeneas", AENEAS_REV)?;
    require_tool(&charon, "Charon", CHARON_REV)?;
    require_revision(&tools, "Aeneas", AENEAS_REV)?;
    require_revision(&charon_root, "Charon", CHARON_REV)?;

    let cache = root.join(".cache/blckbaux/aeneas");
    fs::create_dir_all(&cache)
        .map_err(|error| format!("cannot create {}: {error}", cache.display()))?;
    let llbc = cache.join("optimizer.llbc");
    let mut lower = Command::new(&charon);
    lower
        .current_dir(root.join("rust/crates/semantics"))
        .args(["cargo", "--preset=aeneas"]);
    for action in ACTION_ROOTS {
        lower.args(["--start-from", action]);
    }
    lower
        .arg("--dest-file")
        .arg(&llbc)
        .args(["--", "--no-default-features"]);
    crate::execute(lower)?;

    let lean_root = root.join("research/lean");
    let check_root = cache.join(format!("check-{}", std::process::id()));
    let destination = if check {
        if check_root.exists() {
            fs::remove_dir_all(&check_root)
                .map_err(|error| format!("cannot clear {}: {error}", check_root.display()))?;
        }
        fs::create_dir_all(&check_root)
            .map_err(|error| format!("cannot create {}: {error}", check_root.display()))?;
        &check_root
    } else {
        sync_aeneas(&lean_root)?;
        &lean_root
    };
    let mut translate = Command::new(&aeneas);
    translate
        .current_dir(root)
        .args([
            "-backend",
            "lean",
            "-namespace",
            "BlckbauxResearch.Generated.Optimizer",
            "-dest",
        ])
        .arg(destination)
        .args([
            "-subdir",
            "BlckbauxResearch/Generated/Optimizer",
            "-split-files",
            "-no-progress-bar",
        ])
        .arg(&llbc);
    let translated = crate::execute(translate);
    if check {
        let checked = translated.and_then(|()| compare_translation(&lean_root, &check_root));
        let cleaned = fs::remove_dir_all(&check_root)
            .map_err(|error| format!("cannot clear {}: {error}", check_root.display()));
        checked?;
        cleaned?;
        println!("blckbaux: generated optimizer semantics are current");
        return Ok(());
    }
    translated?;
    println!(
        "blckbaux: extracted optimizer decisions with Aeneas {} and Charon {}",
        short_revision(AENEAS_REV),
        short_revision(CHARON_REV)
    );
    Ok(())
}

fn compare_translation(lean_root: &Path, check_root: &Path) -> Result<(), String> {
    let relative = Path::new("BlckbauxResearch/Generated/Optimizer");
    for file in ["Types.lean", "Funs.lean"] {
        let tracked = lean_root.join(relative).join(file);
        let generated = check_root.join(relative).join(file);
        let tracked_bytes = fs::read(&tracked)
            .map_err(|error| format!("cannot read {}: {error}", tracked.display()))?;
        let generated_bytes = fs::read(&generated)
            .map_err(|error| format!("cannot read {}: {error}", generated.display()))?;
        if tracked_bytes != generated_bytes {
            return Err(format!(
                "{} is stale; run `./blckbaux research extract`",
                tracked.display()
            ));
        }
    }
    Ok(())
}

fn sync_aeneas(lean_root: &Path) -> Result<(), String> {
    let manifest = lean_root.join("lake-manifest.json");
    let current = fs::read_to_string(&manifest).unwrap_or_default();
    if current.contains("\"name\": \"aeneas\"") && current.contains(AENEAS_REV) {
        return Ok(());
    }
    let mut update = Command::new("lake");
    update
        .current_dir(lean_root)
        .env("MATHLIB_NO_CACHE_ON_UPDATE", "1")
        .args(["update", "aeneas"]);
    crate::execute(update)
}

fn require_tool(path: &Path, name: &str, revision: &str) -> Result<(), String> {
    if path.is_file() {
        Ok(())
    } else {
        Err(format!(
            "{name} {} is not built at {}; install the pinned research toolchain before extraction",
            short_revision(revision),
            path.display()
        ))
    }
}

fn require_revision(path: &Path, name: &str, expected: &str) -> Result<(), String> {
    let output = Command::new("git")
        .current_dir(path)
        .args(["rev-parse", "HEAD"])
        .output()
        .map_err(|error| format!("cannot inspect cached {name}: {error}"))?;
    let actual = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if output.status.success() && actual == expected {
        Ok(())
    } else {
        Err(format!(
            "cached {name} is {actual}, but BLCKBAUX requires {expected}"
        ))
    }
}

fn short_revision(revision: &str) -> &str {
    &revision[..7]
}

fn require_sandbox() -> Result<(), String> {
    let sandbox = std::env::var_os("COMPARATOR_BWRAP").unwrap_or_else(|| "bwrap".into());
    let available = Command::new(&sandbox)
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success());
    if available {
        Ok(())
    } else {
        Err("Lean's sandboxed audit requires bubblewrap; run `./blckbaux research audit` on a Linux host with bwrap installed".to_owned())
    }
}

fn check(root: &Path, build_system: BuildSystem, via_build: bool) -> Result<(), String> {
    lint_names(root)?;
    let target = if via_build {
        let artifact = build_system.build_artifact(root, "//research/lean:check")?;
        fs::read_to_string(&artifact.path).map_err(|error| {
            format!(
                "cannot read checked Lean artifact {}: {error}",
                artifact.path.display()
            )
        })?
    } else {
        ensure_cache(root)?;
        let mut build = Command::new("lake");
        build.current_dir(root.join("research/lean")).args([
            "--quiet",
            "--log-level=error",
            "build",
            "BlckbauxResearch",
        ]);
        crate::execute(build)?;
        let output = Command::new("lake")
            .current_dir(root.join("research/lean"))
            .args(["env", "lean", "--run", "Main.lean"])
            .output()
            .map_err(|error| format!("cannot run checked Lean target: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "checked Lean target executable failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        String::from_utf8(output.stdout)
            .map_err(|error| format!("Lean target output is not UTF-8: {error}"))?
    };
    check_contract(root, build_system, &target)?;
    println!("blckbaux: Lean optimizer formalization checked");
    Ok(())
}

fn ensure_cache(root: &Path) -> Result<(), String> {
    let lean_root = root.join("research/lean");
    let mathlib =
        lean_root.join(".lake/packages/mathlib/.lake/build/lib/lean/Mathlib/Data/Real/Basic.olean");
    if mathlib.is_file() {
        return Ok(());
    }
    let mut cache = Command::new("lake");
    cache.current_dir(lean_root).args(["exe", "cache", "get"]);
    crate::execute(cache)
}

fn check_contract(root: &Path, build_system: BuildSystem, target: &str) -> Result<(), String> {
    let value = |key: &str| -> Result<&str, String> {
        target
            .lines()
            .find_map(|line| line.strip_prefix(key))
            .map(str::trim)
            .ok_or_else(|| format!("Lean target output lacks `{key}`"))
    };
    let number = |key: &str| -> Result<u64, String> {
        value(key)?
            .parse::<u64>()
            .map_err(|error| format!("invalid Lean target value for `{key}`: {error}"))
    };

    let args = vec![
        "research-contract".to_owned(),
        "performance/configs/hnet-1m-causal-1000.toml".to_owned(),
        "--parameters".to_owned(),
        number("target.parameters: ")?.to_string(),
        "--context-tokens".to_owned(),
        number("target.context_tokens: ")?.to_string(),
        "--candidates".to_owned(),
        number("target.candidates: ")?.to_string(),
        "--neighbors".to_owned(),
        number("target.neighbors: ")?.to_string(),
        "--target-ms".to_owned(),
        number("target.transaction_target_ms: ")?.to_string(),
        "--ceiling-ms".to_owned(),
        number("target.transaction_ceiling_ms: ")?.to_string(),
        "--dense-flops".to_owned(),
        number("target.dense_flops: ")?.to_string(),
        "--objective".to_owned(),
        value("target.objective: ")?.to_owned(),
        "--sampling".to_owned(),
        value("target.sampling: ")?.to_owned(),
    ];
    build_system.run(root, RUNTIME_TARGET, &args)
}

fn show(root: &Path, name: &str, body: bool, mathlib: bool) -> Result<(), String> {
    let contents = read_catalogue(root)?;
    if let Some(card) = find_card(&contents, name) {
        if body || mathlib {
            return Err("--body and --mathlib apply only to Lean declarations".to_owned());
        }
        println!("CONSTRUCTION {card}");
        return Ok(());
    }
    if !name.contains('.') {
        return Err(format!(
            "unknown construction `{}`; run `./blckbaux research list`",
            name.trim()
        ));
    }
    inspect(root, name, body, mathlib)
}

fn inspect(root: &Path, declaration: &str, body: bool, all_mathlib: bool) -> Result<(), String> {
    let declaration = declaration.trim();
    if !declaration_ok(declaration) {
        return Err(format!(
            "invalid Lean declaration `{declaration}`; use a fully qualified name"
        ));
    }
    let body = if body {
        format!("#print {declaration}\n")
    } else {
        String::new()
    };
    let imports = if all_mathlib {
        "import Mathlib\nimport BlckbauxResearch"
    } else {
        "import BlckbauxResearch"
    };
    let input = format!(
        "{imports}\n\n\
         #check @{declaration}\n\
         {body}\
         #print axioms {declaration}\n\
         #find_home {declaration}\n\
         #min_imports in\n\
         #check @{declaration}\n"
    );
    lean_stdin(root, &input)
}

fn prove(root: &Path, proposition: &str, all_mathlib: bool) -> Result<(), String> {
    let mut standard_input = String::new();
    let proposition = if proposition == "-" {
        io::stdin()
            .read_to_string(&mut standard_input)
            .map_err(|error| format!("could not read proposition: {error}"))?;
        standard_input.trim()
    } else {
        proposition.trim()
    };
    if proposition.is_empty() || proposition.contains('\n') || proposition.contains('\r') {
        return Err("Lean proposition must be one nonempty line".to_owned());
    }
    let imports = if all_mathlib {
        "import Mathlib\nimport BlckbauxResearch"
    } else {
        "import BlckbauxResearch"
    };
    let input = format!("{imports}\n\nexample : {proposition} := by\n  exact?\n");
    lean_stdin(root, &input)
}

fn search(root: &Path, query: &str, structural: bool) -> Result<(), String> {
    let query = query.trim();
    if query.is_empty() || query.contains('\n') || query.contains('\r') {
        return Err("theorem query must be one nonempty line".to_owned());
    }
    let command = if structural {
        format!("#loogle {query}")
    } else {
        let mut sentence = query.to_owned();
        if !sentence.ends_with(['.', '?']) {
            sentence.push('?');
        }
        format!("#leansearch {}", lean_string(&sentence))
    };
    lean_stdin(root, &format!("import Mathlib\n\n{command}\n"))
}

fn lean_string(value: &str) -> String {
    let mut output = String::with_capacity(value.len() + 2);
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            character => output.push(character),
        }
    }
    output.push('"');
    output
}

fn lean_stdin(root: &Path, input: &str) -> Result<(), String> {
    let mut child = Command::new("lake")
        .current_dir(root.join("research/lean"))
        .args(["env", "lean", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|error| format!("could not start Lean: {error}"))?;
    child
        .stdin
        .take()
        .ok_or_else(|| "could not open Lean input".to_owned())?
        .write_all(input.as_bytes())
        .map_err(|error| format!("could not send input to Lean: {error}"))?;
    let status = child
        .wait()
        .map_err(|error| format!("could not wait for Lean: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("Lean feedback failed with {status}"))
    }
}

fn declaration_ok(declaration: &str) -> bool {
    !declaration.is_empty()
        && !declaration.starts_with('.')
        && !declaration.ends_with('.')
        && !declaration.contains("..")
        && declaration.split('.').all(|part| {
            part.chars()
                .next()
                .is_some_and(|character| character.is_ascii_alphabetic() || character == '_')
                && part.chars().all(|character| {
                    character.is_ascii_alphanumeric() || character == '_' || character == '\''
                })
        })
}

fn catalogue(root: &Path, topic: Option<&str>) -> Result<(), String> {
    let contents = read_catalogue(root)?;
    let Some(topic) = topic else {
        println!("{contents}");
        return Ok(());
    };
    if let Some(card) = find_card(&contents, topic) {
        println!("CONSTRUCTION {card}");
        return Ok(());
    }
    let topics = contents
        .split("\nCONSTRUCTION ")
        .skip(1)
        .filter_map(|card| card.lines().next())
        .collect::<Vec<_>>()
        .join(", ");
    Err(format!(
        "unknown construction `{topic}`; available topics: {topics}"
    ))
}

fn list(root: &Path, query: Option<&str>) -> Result<(), String> {
    let contents = read_catalogue(root)?;
    let query = query.map(str::trim).filter(|query| !query.is_empty());
    let matches = topics(&contents)
        .filter(|topic| {
            query.is_none_or(|query| {
                topic
                    .to_ascii_lowercase()
                    .contains(&query.to_ascii_lowercase())
            })
        })
        .collect::<Vec<_>>();
    if matches.is_empty() {
        return Err(format!(
            "no construction topic matches `{}`; run `./blckbaux research list`",
            query.unwrap_or("")
        ));
    }
    for topic in matches {
        println!("{topic}");
    }
    Ok(())
}

fn read_catalogue(root: &Path) -> Result<String, String> {
    fs::read_to_string(root.join("docs/mathlib-constructions.txt"))
        .map_err(|error| format!("could not read construction catalogue: {error}"))
}

fn topics(contents: &str) -> impl Iterator<Item = &str> {
    contents
        .split("\nCONSTRUCTION ")
        .skip(1)
        .filter_map(|card| card.lines().next())
}

fn find_card<'a>(contents: &'a str, topic: &str) -> Option<&'a str> {
    contents
        .split("\nCONSTRUCTION ")
        .skip(1)
        .find(|card| card.lines().next().is_some_and(|name| name == topic.trim()))
}

#[cfg(test)]
mod tests {
    use super::{declaration_ok, find_card, lean_string, topics};

    #[test]
    fn exact_topic() {
        let text = "Heading\nCONSTRUCTION parameter\nfirst\nCONSTRUCTION parameters\nsecond\n";
        assert_eq!(find_card(text, "parameter"), Some("parameter\nfirst"));
        assert_eq!(
            find_card(text, " parameters "),
            Some("parameters\nsecond\n")
        );
        assert!(find_card(text, "param").is_none());
        assert!(find_card(text, "missing").is_none());
    }

    #[test]
    fn topic_names() {
        let text = "Heading\nCONSTRUCTION parameter\nfirst\nCONSTRUCTION regret\nsecond\n";
        assert_eq!(topics(text).collect::<Vec<_>>(), ["parameter", "regret"]);
    }

    #[test]
    fn declaration_names() {
        assert!(declaration_ok(
            "BlckbauxResearch.Optimization.Rate.expected"
        ));
        assert!(declaration_ok("Metric.mem_closedBall'"));
        assert!(!declaration_ok(""));
        assert!(!declaration_ok("Rate.expected\n#eval 1"));
        assert!(!declaration_ok("Rate..expected"));
        assert!(!declaration_ok("Rate.expected x"));
    }

    #[test]
    fn query_string() {
        assert_eq!(
            lean_string("say \"hello\" \\"),
            "\"say \\\"hello\\\" \\\\\""
        );
    }
}

fn lint_names(root: &Path) -> Result<(), String> {
    let source = root.join("research/lean");
    let mut issues = Vec::new();
    scan_path(&source, root, &mut issues)?;
    if issues.is_empty() {
        return Ok(());
    }
    Err(format!(
        "Lean naming policy failed (two words, one underscore, 24 characters):\n{}",
        issues.join("\n")
    ))
}

fn scan_path(path: &Path, root: &Path, issues: &mut Vec<String>) -> Result<(), String> {
    for entry in fs::read_dir(path).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|name| name == ".lake") {
                continue;
            }
            scan_path(&path, root, issues)?;
            continue;
        }
        if path.extension().is_none_or(|extension| extension != "lean") {
            continue;
        }
        let relative = path.strip_prefix(root).unwrap_or(&path).display();
        if let Some(stem) = path.file_stem().and_then(|name| name.to_str()) {
            if !name_ok(stem) {
                issues.push(format!("{relative}:1: file `{stem}`"));
            }
        }
        let text = fs::read_to_string(&path).map_err(|error| error.to_string())?;
        for (line_index, line) in text.lines().enumerate() {
            let mut words = line.split_whitespace();
            let mut word = words.next().unwrap_or("");
            while matches!(word, "private" | "protected" | "nonrec" | "noncomputable") {
                word = words.next().unwrap_or("");
            }
            let kind = match word {
                "abbrev" | "class" | "def" | "inductive" | "lemma" | "namespace" | "opaque"
                | "structure" | "theorem" => word,
                _ => continue,
            };
            let Some(raw_name) = words.next() else {
                continue;
            };
            let name = raw_name.trim_end_matches(|character: char| {
                !character.is_ascii_alphanumeric() && character != '_' && character != '.'
            });
            if name.is_empty() || !name.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) {
                continue;
            }
            if name.split('.').any(|segment| !name_ok(segment)) {
                issues.push(format!("{relative}:{}: {kind} `{name}`", line_index + 1));
            }
        }
    }
    Ok(())
}

fn name_ok(name: &str) -> bool {
    name.len() <= 24 && name.matches('_').count() <= 1 && word_count(name) <= 2
}

fn word_count(name: &str) -> usize {
    let mut normalized = name.to_owned();
    for atom in ["BFloat16", "BF16", "BLCKBAUX", "HNet"] {
        normalized = normalized.replace(atom, "x");
    }
    normalized
        .split('_')
        .map(|part| {
            let bytes = part.as_bytes();
            if bytes.is_empty() {
                return 0;
            }
            1 + (1..bytes.len())
                .filter(|&index| {
                    let previous = bytes[index - 1];
                    let current = bytes[index];
                    let next = bytes.get(index + 1).copied();
                    (previous.is_ascii_lowercase() && current.is_ascii_uppercase())
                        || (previous.is_ascii_alphabetic() && current.is_ascii_digit())
                        || (previous.is_ascii_digit() && current.is_ascii_alphabetic())
                        || (previous.is_ascii_uppercase()
                            && current.is_ascii_uppercase()
                            && next.is_some_and(|value| value.is_ascii_lowercase()))
                })
                .count()
        })
        .sum()
}
