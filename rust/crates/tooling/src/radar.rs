use clap::{Subcommand, ValueEnum};
use std::path::{Path, PathBuf};

#[derive(Debug, PartialEq, Eq, Subcommand)]
pub enum RadarAction {
    /// Build the compiler-resolved Rust SCIP index.
    Index {
        /// Rebuild even when the cached index is current.
        #[arg(long)]
        force: bool,
    },
    /// Find a compiler-resolved Rust symbol and its exact source locations.
    Symbol {
        /// Exact or partial symbol name.
        name: String,
        /// Include reference locations as well as definitions.
        #[arg(long)]
        refs: bool,
        /// Maximum number of source locations to print.
        #[arg(long, default_value_t = 40)]
        limit: usize,
    },
    /// Report the availability of every semantic backend.
    Doctor,
    /// Search for code patterns with enclosing AST symbols and complexity.
    Find {
        /// Pattern or literal text to search.
        pattern: String,
        /// Treat pattern as regular expression instead of literal string.
        #[arg(long, short = 'e')]
        regex: bool,
        /// Source context lines around matches.
        #[arg(long, short = 'C', default_value_t = 0)]
        context: usize,
    },
    /// Transitive blast radius and reachability of a symbol across the call graph.
    Impact {
        /// Symbol identifier (e.g. "path/to/file.rs:fn_name" or "fn_name").
        symbol: String,
    },
    /// Direct and transitive callers of a symbol.
    Callers {
        /// Symbol identifier.
        symbol: String,
    },
    /// Direct and transitive callees called by a symbol.
    Callees {
        /// Symbol identifier.
        symbol: String,
    },
    /// Inspect and gate BO plus generation performance evidence.
    #[command(alias = "perf")]
    Performance {
        #[command(subcommand)]
        command: PerformanceAction,
    },
    /// Token-budgeted context bundle for AI agents with Mermaid flowchart.
    #[command(hide = true)]
    Task {
        /// Task description in natural language.
        query: String,
        /// Include Mermaid architecture flowchart diagram.
        #[arg(long, default_value_t = true)]
        graph: bool,
    },
    /// Start the Model Context Protocol (MCP) server.
    #[command(hide = true)]
    Mcp,
}

pub fn run(
    root: &Path,
    action: RadarAction,
    performance: impl FnOnce(&Path, PerformanceAction) -> Result<(), String>,
) -> Result<(), String> {
    match action {
        RadarAction::Index { force } => crate::radar_scip::index(root, force),
        RadarAction::Symbol { name, refs, limit } => {
            crate::radar_scip::symbol(root, &name, refs, limit)
        }
        RadarAction::Doctor => crate::radar_scip::doctor(root),
        RadarAction::Find {
            pattern,
            regex,
            context,
        } => run_find(root, &pattern, regex, context),
        RadarAction::Impact { symbol } => run_arg(root, "--impact", &symbol),
        RadarAction::Callers { symbol } => run_arg(root, "--callers", &symbol),
        RadarAction::Callees { symbol } => run_arg(root, "--callees", &symbol),
        RadarAction::Performance { command } => performance(root, command),
        RadarAction::Task { query, graph } => run_task(root, &query, graph),
        RadarAction::Mcp => run_tool(root, &["--mcp"]),
    }
}

#[derive(Debug, PartialEq, Eq, Subcommand)]
pub enum PerformanceAction {
    /// Show a compact decision report or the complete evidence document.
    Show {
        /// Experiment whose exact workload is being certified.
        #[arg(default_value = "performance/configs/hnet-1m-causal-1000.toml")]
        config: PathBuf,
        /// Accelerator lowering used by the experiment.
        #[arg(long, value_enum, default_value_t = PerformanceBackend::Metal)]
        backend: PerformanceBackend,
        /// Completed run directory supplying measured evidence.
        #[arg(long)]
        run: Option<PathBuf>,
        /// Print the complete machine-readable evidence document.
        #[arg(long)]
        json: bool,
    },
    /// Write a new measured baseline after a completed matching-source run.
    Record {
        #[arg(default_value = "performance/configs/hnet-1m-causal-1000.toml")]
        config: PathBuf,
        #[arg(long, value_enum, default_value_t = PerformanceBackend::Metal)]
        backend: PerformanceBackend,
        /// Completed run directory produced by the current hot-path source.
        #[arg(long)]
        run: PathBuf,
        /// Baseline artifact to create or replace.
        #[arg(long, default_value = "performance/hnet-1m-metal.json")]
        output: PathBuf,
    },
    /// Reject hot-path regressions against a measured baseline.
    Gate {
        #[arg(default_value = "performance/configs/hnet-1m-causal-1000.toml")]
        config: PathBuf,
        #[arg(long, value_enum, default_value_t = PerformanceBackend::Metal)]
        backend: PerformanceBackend,
        /// Fresh completed run. Required only after the hot path changes.
        #[arg(long)]
        run: Option<PathBuf>,
        #[arg(long, default_value = "performance/hnet-1m-metal.json")]
        baseline: PathBuf,
        /// Allowed p95 increase for noisy device measurements, as a percentage.
        #[arg(long, default_value_t = 5)]
        tolerance_percent: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum PerformanceBackend {
    Metal,
    CudaOxide,
}

fn run_find(root: &Path, pattern: &str, regex: bool, context: usize) -> Result<(), String> {
    let flag = if regex { "--regex" } else { "--grep" };
    let grep_arg = format!("{flag}={pattern}");
    let mut args = vec![".", &grep_arg];
    let ctx_arg = format!("--grep-context={context}");
    if context > 0 {
        args.push(&ctx_arg);
    }
    run_tool(root, &args)
}

fn run_arg(root: &Path, flag: &str, value: &str) -> Result<(), String> {
    let arg = format!("{flag}={value}");
    run_tool(root, &[".", &arg])
}

fn run_task(root: &Path, query: &str, graph: bool) -> Result<(), String> {
    let task_arg = format!("--pack-task={query}");
    let mut args = vec![".", &task_arg];
    if graph {
        args.push("--with-graph");
    }
    run_tool(root, &args)
}

fn run_tool(root: &Path, args: &[&str]) -> Result<(), String> {
    super::dev_exec(root, "radar", args)
}

impl PerformanceBackend {
    pub fn name(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::CudaOxide => "cuda-oxide",
        }
    }
}
