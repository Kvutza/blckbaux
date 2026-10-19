use std::path::{Path, PathBuf};

use crate::config::BuildSystem;

const RUNTIME_TARGET: &str = "//rust/crates/runner:blckbaux-runner";

pub(crate) fn run(
    root: &Path,
    path: &Path,
    prepare: bool,
    rounds: Option<u32>,
    engine: BuildSystem,
) -> Result<(), String> {
    blckbaux_tooling::terminal::stderr(
        blckbaux_tooling::terminal::Tag::Orchestration,
        "runtime · load experiment engine",
    );
    let mut args = vec!["tune".to_owned(), path.display().to_string()];
    if prepare {
        args.push("--prepare".to_owned());
    }
    if let Some(rounds) = rounds {
        args.extend(["--rounds".to_owned(), rounds.to_string()]);
    }
    engine.run(root, RUNTIME_TARGET, &args)
}

pub(crate) fn resolve(
    root: &Path,
    path: &Path,
    rounds: Option<u32>,
    engine: BuildSystem,
) -> Result<PathBuf, String> {
    let mut args = vec!["resolve".to_owned(), path.display().to_string()];
    if let Some(rounds) = rounds {
        args.extend(["--rounds".to_owned(), rounds.to_string()]);
    }
    let output = engine.run_output(root, RUNTIME_TARGET, &args)?;
    output
        .lines()
        .last()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(PathBuf::from)
        .ok_or("runtime helper did not return a resolved configuration".into())
}
