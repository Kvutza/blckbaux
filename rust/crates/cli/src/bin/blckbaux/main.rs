use clap::{Parser, Subcommand};
use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

mod config;
mod cuda_track;
mod experiment;
mod radar;
mod research;

#[derive(Parser)]
#[command(
    name = "blckbaux",
    bin_name = "./blckbaux",
    version = env!("CARGO_PKG_VERSION"),
    about = "Build and run BLCKBAUX experiments",
    arg_required_else_help = true
)]
struct Cli {
    #[command(subcommand)]
    action: Action,
}

#[derive(Subcommand)]
enum Action {
    /// Build native BLCKBAUX targets.
    Build {
        /// Run the full test suite for the selected build.
        #[arg(long)]
        tests: bool,
        /// Build Python wheels for every supported interpreter.
        #[arg(long)]
        distribution: bool,
        /// Directory for distribution artifacts.
        #[arg(long, requires = "distribution")]
        out: Option<PathBuf>,
    },
    /// Run the repository verification suite.
    Test {
        /// Limit verification to test targets affected by working-copy changes.
        #[arg(long)]
        affected: bool,
    },
    /// Format repository source.
    Fmt {
        #[arg(long)]
        check: bool,
    },
    /// Check repository naming policy.
    Check {
        /// Files or directories to check.
        paths: Vec<PathBuf>,
        /// Output findings as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Run a pretraining experiment.
    Tune {
        config: PathBuf,
        #[arg(long, conflicts_with = "rounds")]
        prepare: bool,
        #[arg(long, conflicts_with = "prepare")]
        rounds: Option<u32>,
    },
    /// Run CUDA-Oxide locally or on a configured NVIDIA runner.
    #[command(hide = true)]
    Cuda {
        #[command(subcommand)]
        command: cuda_track::Action,
    },
    /// Inspect source reachability and performance evidence.
    Radar {
        #[command(subcommand)]
        command: radar::RadarAction,
    },
    /// Work on the checked optimizer theory.
    Research {
        #[command(subcommand)]
        command: research::ResearchAction,
    },
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            blckbaux_tooling::terminal::stderr(
                blckbaux_tooling::terminal::Tag::Error,
                &format!("blckbaux: {error}"),
            );
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let root = env::var_os("BUILD_WORKSPACE_DIRECTORY")
        .map(PathBuf::from)
        .map_or_else(|| env::current_dir().map_err(|error| error.to_string()), Ok)?;
    env::set_current_dir(&root).map_err(|error| error.to_string())?;
    let settings = config::Config::from_env();
    match Cli::parse().action {
        Action::Build {
            tests,
            distribution,
            out,
        } => build(&root, tests, distribution, out),
        Action::Test { affected } => {
            dev_action(&root, "test", &[if affected { "affected" } else { "all" }])
        }
        Action::Fmt { check } => dev_action(&root, "fmt", &[if check { "check" } else { "write" }]),
        Action::Check { paths, json } => blckbaux_tooling::names::cli(&root, &paths, json),
        Action::Tune {
            config,
            prepare,
            rounds,
        } => experiment::run(&root, &config, prepare, rounds, settings.build_system),
        Action::Cuda { command } => cuda_track::run(&root, command),
        Action::Radar { command } => radar::run(&root, command),
        Action::Research { command } => research::run(&root, command, settings.build_system),
    }
}

pub(crate) use blckbaux_tooling::execute;

fn dev_action(root: &Path, action: &str, args: &[&str]) -> Result<(), String> {
    let mut command = Command::new("sh");
    command
        .current_dir(root)
        .args(["tools/dev-actions", action])
        .args(args);
    execute(command)
}

fn build(root: &Path, tests: bool, distribution: bool, out: Option<PathBuf>) -> Result<(), String> {
    let verification = if tests { "full" } else { "smoke" };
    if distribution {
        let out = out.unwrap_or_else(|| PathBuf::from("dist"));
        dev_action(
            root,
            "build",
            &["distribution", verification, &out.to_string_lossy()],
        )
    } else {
        dev_action(root, "build", &["native", verification])
    }
}
