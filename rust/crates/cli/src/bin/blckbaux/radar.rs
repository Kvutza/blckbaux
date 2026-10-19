mod performance;

pub use blckbaux_tooling::radar::{PerformanceAction, PerformanceBackend, RadarAction};

pub fn run(root: &std::path::Path, action: RadarAction) -> Result<(), String> {
    blckbaux_tooling::radar::run(root, action, performance::run)
}
