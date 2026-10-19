//! Backend-neutral optimizer semantics for BLCKBAUX.
//!
//! Provides modular optimizer components:
//! - `core`: prediction values and shared optimizer vocabulary
//! - `reliability`: neighborhood estimation and reliability-controlled radius

pub mod calibration;
pub mod core;
pub mod reliability;
mod response;
mod reuse;
pub use reuse::SeedReuse;
pub mod settings;
pub use response::response_rms;
pub use settings::SurrogateSpec;

// Canonical exports directly at crate root
pub use reliability::{
    CANDIDATES, FitState, INITIAL_OBSERVATIONS, MAX_HISTORY, NeighborValueEstimator, PoolShape,
    RadiusControl, RadiusController, ReliabilityAction, ReliabilityDecision, ReliabilityEvidence,
    ReliabilityPolicy, ReliabilityTelemetry, ValuePrediction, accept_bits, accept_candidate,
    accept_decision, candidate_seed, decide_action, splitmix64,
};
