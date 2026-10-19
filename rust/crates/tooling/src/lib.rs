pub mod build;
pub mod names;
pub mod radar;
pub mod radar_scip;
pub mod terminal;

use std::path::Path;
use std::process::Command;

pub fn dev_exec(root: &Path, action: &str, args: &[&str]) -> Result<(), String> {
    let mut command = Command::new("sh");
    command
        .current_dir(root)
        .args(["tools/dev-actions", action])
        .args(args);
    execute(command)
}

pub fn execute(mut command: Command) -> Result<(), String> {
    let status = command.status().map_err(|error| error.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("command exited with {status}"))
    }
}
