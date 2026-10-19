//! Neighbor estimation, candidate streams, acceptance, and radius control.

mod control;
mod gaussian;
mod neighbor;
mod radius;

pub const CANDIDATES: usize = 4;
pub const INITIAL_OBSERVATIONS: usize = 32;
pub use crate::neighbor::MAX_HISTORY;

pub use crate::core::ValuePrediction;
pub use control::{RadiusControl, ReliabilityEvidence, ReliabilityPolicy, ReliabilityTelemetry};
pub use gaussian::{PoolShape, candidate_seed, splitmix64};
pub use neighbor::{FitState, NeighborValueEstimator};
pub use radius::{
    RadiusController, ReliabilityAction, ReliabilityDecision, accept_bits, accept_candidate,
    accept_decision, decide_action,
};

#[cfg(test)]
mod tests;
